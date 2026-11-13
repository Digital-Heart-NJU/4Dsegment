"""Subject-level base detection and read-only common-plane previews."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import manifold3d
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_open_volmesh as pipeline


def cones(direction=(0, 0, 1), scale=1.):
    return {name: pv.Cone(center=(x, 0, 0), direction=direction,
                          height=height * scale, radius=radius * scale,
                          resolution=20, capping=True).triangulate().clean()
            for name, x, height, radius in (("LVendo", -5, 20, 4), ("RVendo", 5, 18, 3))}


def synthetic_shell(cavities):
    outer = pv.Cube(bounds=(-15, 15, -12, 12, -15, 15)).triangulate().clean()
    outer.cell_data["sourceSurface"] = np.where(outer.cell_centers().points[:, 0] < 0, 2, 4)
    outer.cell_data["elemTag"] = (outer.cell_data["sourceSurface"] == 2).astype(np.int32)
    parts = [outer]
    for name, source_id, tag in (("LVendo", 1, 1), ("RVendo", 3, 0)):
        inner = cavities[name].compute_normals(consistent_normals=True, auto_orient_normals=True,
                                               split_vertices=False)
        faces = inner.faces.reshape(-1, 4).copy()
        faces[:, 1:] = faces[:, 1:][:, ::-1]
        inner.faces = faces.ravel()
        inner.cell_data["sourceSurface"] = np.full(inner.n_cells, source_id)
        inner.cell_data["elemTag"] = np.full(inner.n_cells, tag)
        parts.append(inner)
    return pipeline._join_triangle_surfaces(parts)


class DetectionTests(unittest.TestCase):
    def test_exact_area_preserves_concavity(self):
        outside = manifold3d.Manifold.cube((10, 10, 4), center=True)
        notch = manifold3d.Manifold.cube((8, 6, 6), center=True).translate((3, 0, 0))
        mesh = (outside - notch).to_mesh64()
        triangles = np.asarray(mesh.tri_verts, dtype=np.int64)
        surface = pv.PolyData(np.asarray(mesh.vert_properties)[:, :3].copy(),
            np.column_stack((np.full(len(triangles), 3), triangles)).ravel())
        sample = pipeline._section_sampler(surface, np.zeros(3), np.array([0., 0., 1.]))
        area, loops = sample(0)
        self.assertAlmostEqual(area, 64)
        self.assertEqual(loops, 1)

    def test_rotated_subject_has_one_base_axis_across_frames(self):
        direction = np.array([.8, -.4, .45])
        direction /= np.linalg.norm(direction)
        _, axis, profiles, info = pipeline.detect_subject_base(
            {"ED": cones(direction), "ES": cones(direction, .9)}, samples=64)
        self.assertGreater(axis @ -direction, .99)
        self.assertEqual(info["frames_used"], ["ED", "ES"])
        self.assertGreater(info["confidence_score"], .12)
        for profile in profiles.values():
            self.assertTrue(np.all(np.diff(profile["levels"]) > 0))

    def test_ambiguous_shape_requires_manual_axis_and_side(self):
        sphere = pv.Sphere(theta_resolution=32, phi_resolution=32).triangulate()
        cavities = {"ED": {"LVendo": sphere, "RVendo": sphere.translate((3, 0, 0))}}
        with self.assertRaisesRegex(RuntimeError, "set --base-axis"):
            pipeline.detect_subject_base(cavities, samples=64)
        with self.assertRaisesRegex(RuntimeError, "set --base-side"):
            pipeline.detect_subject_base(cavities, base_axis="z", samples=64)
        _, axis, _, info = pipeline.detect_subject_base(cavities, "z", "high", 64)
        np.testing.assert_array_equal(axis, (0, 0, 1))
        self.assertEqual(info["base_side_override"], "high")

    def test_conflicting_frames_do_not_silently_pick_a_base(self):
        with self.assertRaisesRegex(RuntimeError, "Ambiguous base/apex decision"):
            pipeline.detect_subject_base({"ED": cones(), "ES": cones((0, 0, -1))}, samples=64)

    def test_basal_cap_consensus_recovers_common_interval_without_global_z(self):
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_euler("xyz", (28, 17, -13), degrees=True)
        cavities = {"ED": cones(), "ES": cones(scale=.9)}
        for parts in cavities.values():
            for name, mesh in parts.items():
                points = np.asarray(mesh.points, dtype=float).copy()
                points[:, 0] += .32 * points[:, 2] + (-11 if name == "LVendo" else 11)
                mesh.points = rotation.apply(points)
        _, axis, _, info = pipeline.detect_subject_base(cavities, samples=80)
        refinement = info["axis_refinement"]
        self.assertIsNotNone(refinement)
        self.assertGreater(axis @ rotation.apply((0, 0, -1)), .99)
        self.assertTrue(any(low >= high for low, high in refinement["initial_basal_intervals_mm"].values()))
        self.assertTrue(all(low < high for low, high in refinement["refined_basal_intervals_mm"].values()))

    def test_disconnected_retained_volume_is_rejected(self):
        cavities = cones()
        with contextlib.redirect_stdout(io.StringIO()):
            grid, _ = pipeline.tetrahedralize_lvrv_surface(synthetic_shell(cavities))
        island = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]), np.array([pv.CellType.TETRA]),
            np.array([[30., 0., 5.], [31., 0., 5.], [30., 1., 5.], [30., 0., 6.]]))
        disconnected = pv.merge([grid, island], merge_points=False)
        with self.assertRaisesRegex(RuntimeError, "connected myocardial"):
            pipeline._basal_candidate_plane(disconnected, cavities, np.zeros(3), np.array([0., 0., -1.]), samples=8)


class Stage7Tests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.subject = Path(directory.name) / "subject_test"
        self.root = self.subject / "volmesh_pipline"
        self.output = self.root / "step_07_base_detection"
        self.inputs = []
        for frame, scale in (("ED", 1.), ("ES", .9)):
            surface = synthetic_shell(cones(scale=scale))
            with contextlib.redirect_stdout(io.StringIO()):
                grid, _ = pipeline.tetrahedralize_lvrv_surface(surface)
            boundary_path = self.root / "step_05_LVRV_surf_clean" / f"subject_test_{frame}_LVRV_surf.vtk"
            volume_path = self.root / "step_06_volmesh_generate" / f"subject_test_{frame}_volmesh_notag.vtk"
            for path, mesh in ((boundary_path, surface), (volume_path, grid)):
                path.parent.mkdir(parents=True, exist_ok=True)
                mesh.save(path)
                self.inputs.append(path)

    def run_step(self, frames=("ED", "ES"), overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.step_07_base_detection(self.subject, frames, overwrite, samples=64)

    def test_previews_share_direction_and_do_not_change_source_geometry(self):
        before = {path: pipeline.checksum(path) for path in self.inputs}
        results = self.run_step()
        self.assertEqual(len(results), 2)
        for record in results:
            cut = record["cut"]
            self.assertEqual(cut["predicted_face_connected_components"], 1)
            self.assertGreater(cut["LV_opening_area_mm2"], 0)
            self.assertGreater(cut["RV_opening_area_mm2"], 0)
            self.assertGreater(cut["crossing_tetrahedra"], 0)
            self.assertFalse(record["volume_cut_performed"])
            plane = pv.read(record["outputs"]["base_plane.vtk"])
            axis = np.asarray(cut["plane_normal_toward_base"])
            origin = np.asarray(cut["plane_origin_mm"])
            np.testing.assert_allclose((plane.points - origin) @ axis, 0, atol=1e-7)
            self.assertEqual(pv.read(record["outputs"]["apex_point.vtk"]).n_points, 1)
            self.assertEqual(len(Path(record["outputs"]["area_profiles.csv"]).read_text().splitlines()), 65)
        np.testing.assert_array_equal(results[0]["cut"]["plane_normal_toward_base"],
                                      results[1]["cut"]["plane_normal_toward_base"])
        self.assertNotEqual(results[0]["cut"]["axial_cut_mm"], results[1]["cut"]["axial_cut_mm"])
        self.assertEqual(before, {path: pipeline.checksum(path) for path in self.inputs})
        self.assertFalse((self.root / "step_08_LV_RV_openbase").exists())
        summary = json.loads((self.output / "subject_test_base_detection_summary.json").read_text())
        self.assertTrue(summary["preview_only"])

    def test_single_requested_frame_still_uses_both_available_surfaces(self):
        self.inputs[-1].unlink()
        result = self.run_step(("ED",))[0]
        self.assertEqual(result["subject_detection"]["frames_used"], ["ED", "ES"])
        self.assertFalse(list(self.output.glob("*_ES_*")))

    def test_failed_candidate_preserves_previous_preview(self):
        self.output.mkdir(parents=True)
        path = self.output / "subject_test_ED_base_plane.vtk"
        path.write_bytes(b"previous preview")
        with patch.object(pipeline, "_basal_candidate_plane", side_effect=RuntimeError("invalid candidate")):
            with self.assertRaisesRegex(RuntimeError, "invalid candidate"):
                self.run_step(overwrite=True)
        self.assertEqual(path.read_bytes(), b"previous preview")
        self.assertFalse(list(self.output.glob(".step07_base_*")))

    def test_cli_starts_at_step07_with_manual_overrides(self):
        with patch.object(pipeline, "step_06_volmesh_generate") as previous, \
                patch.object(pipeline, "step_07_base_detection") as current, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--subject-dir", str(self.subject), "--frame", "ED", "--start-step", "7",
                           "--stop-after", "7", "--base-axis", "z", "--base-side", "low", "--base-samples", "64"])
        previous.assert_not_called()
        current.assert_called_once_with(self.subject, ["ED"], False, "z", "low", 64, .30)


if __name__ == "__main__":
    unittest.main()
