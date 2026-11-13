"""Step 04 assembles a separated outer envelope and both complete inner surfaces."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv
import vtk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_open_volmesh as pipeline


def cube(radius, center):
    surface = pv.Cube(bounds=(-radius, radius, -radius, radius, -radius, radius)).triangulate().clean()
    surface.points = np.asarray(surface.points, dtype=float) + np.asarray(center)
    return surface


def inputs():
    return {"LVendo": cube(1, (-2, 0, 0)), "LVepi": cube(5, (-2, 0, 0)),
            "RV": cube(1, (2, 0, 0)), "RVepi": cube(5, (2, 0, 0))}


class BooleanTests(unittest.TestCase):
    def test_both_blood_pools_remain_empty_with_inward_cavity_normals(self):
        surfaces = inputs()
        original = {name: (mesh.points.copy(), mesh.faces.copy()) for name, mesh in surfaces.items()}
        parts, info = pipeline.extract_lvrv_surface(surfaces)
        result = parts["whole"]
        self.assertEqual(info["blood_pool_cavities"], 2)
        self.assertEqual(info["outer_boundary_components"], 1)
        self.assertAlmostEqual(info["myocardial_volume_mm3"], 1384)
        self.assertEqual(info["boundary_component_signed_volumes_mm3"], [-8, -8, 1400])
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(result)
        self.assertGreater(distance.EvaluateFunction((-2, 0, 0)), 0)
        self.assertGreater(distance.EvaluateFunction((2, 0, 0)), 0)
        self.assertLess(distance.EvaluateFunction((0, 0, 0)), 0)
        for key in ("free_edges", "nonmanifold_edges", "degenerate_triangles",
                    "inconsistent_orientation_edges", "self_intersection_pairs"):
            self.assertEqual(info["validation"][key], 0)
        self.assertEqual(set(result.cell_data["elemTag"]), {0, 1})
        self.assertEqual(set(result.cell_data["sourceSurface"]), {1, 2, 3, 4})
        self.assertEqual(len(np.unique(result.points, axis=0)), result.n_points)
        self.assertEqual(info["outer_adjustment"]["iterations"], 0)
        self.assertFalse(info["endocardial_boolean_subtraction"])
        inner = parts["inner"]
        np.testing.assert_array_equal(inner.points[:surfaces["LVendo"].n_points], surfaces["LVendo"].points)
        np.testing.assert_array_equal(inner.points[surfaces["LVendo"].n_points:], surfaces["RV"].points)
        self.assertEqual(inner.n_cells, surfaces["LVendo"].n_cells + surfaces["RV"].n_cells)
        for name, mesh in surfaces.items():
            np.testing.assert_array_equal(mesh.points, original[name][0])
            np.testing.assert_array_equal(mesh.faces, original[name][1])

    def test_disconnected_epicardia_are_rejected(self):
        surfaces = inputs()
        for name in ("RV", "RVepi"):
            surfaces[name].points += (20, 0, 0)
        with self.assertRaisesRegex(RuntimeError, "one connected external surface"):
            pipeline.extract_lvrv_surface(surfaces)

    def test_contact_moves_only_outer_envelope_and_preserves_complete_endo(self):
        surfaces = inputs()
        surfaces["LVendo"].points += (0, 0, 4)
        parts, info = pipeline.extract_lvrv_surface(surfaces)
        result = parts["whole"]
        self.assertEqual(info["blood_pool_cavities"], 2)
        self.assertGreater(info["outer_adjustment"]["iterations"], 0)
        self.assertGreater(info["outer_adjustment"]["maximum_displacement_mm"], 0)
        self.assertLess(info["outer_adjustment"]["maximum_displacement_mm"], 1)
        self.assertGreaterEqual(info["validation"]["minimum_outer_inner_distance_mm"],
                                info["outer_adjustment"]["minimum_clearance_requested_mm"])
        self.assertEqual(info["boundary_component_signed_volumes_mm3"][:2], [-8, -8])
        np.testing.assert_array_equal(parts["inner"].points[:surfaces["LVendo"].n_points], surfaces["LVendo"].points)
        self.assertEqual(parts["inner"].n_cells, 24)
        self.assertEqual(info["validation"]["free_edges"], 0)
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(result)
        self.assertGreater(distance.EvaluateFunction((-2, 0, 4)), 0)

    def test_overlapping_blood_pools_are_rejected(self):
        surfaces = inputs()
        surfaces["RV"] = cube(1, (-1, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "blood pools overlap"):
            pipeline.extract_lvrv_surface(surfaces)

    def test_small_positive_clearance_does_not_trigger_inflation(self):
        surfaces = inputs()
        surfaces["LVendo"].points += (0, 0, 3.99)
        parts, info = pipeline.extract_lvrv_surface(surfaces)
        self.assertEqual(info["outer_adjustment"]["iterations"], 0)
        self.assertEqual(info["outer_adjustment"]["maximum_displacement_mm"], 0)
        self.assertAlmostEqual(info["validation"]["minimum_outer_inner_distance_mm"], .01)
        self.assertEqual(info["blood_pool_cavities"], 2)

    def test_endo_outside_its_epi_is_rejected(self):
        surfaces = inputs()
        surfaces["LVendo"].points += (-10, 0, 0)
        with self.assertRaisesRegex(RuntimeError, "LVendo is not completely enclosed"):
            pipeline.extract_lvrv_surface(surfaces)


class Stage4Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "subject_test"
        self.root = self.subject / "volmesh_pipline"
        self.output_dir = self.root / "step_04_extract_LVRV_surf"
        (self.root / "step_02_surf_clean").mkdir(parents=True)
        (self.root / "step_03_move_RVepi").mkdir()
        self.sources = []
        for frame in ("ED", "ES"):
            for name, mesh in inputs().items():
                folder, suffix = (("step_03_move_RVepi", f"{name}_moved") if name.endswith("epi")
                                  else ("step_02_surf_clean", name))
                path = self.root / folder / f"subject_test_{frame}_{suffix}.vtk"
                mesh.save(path)
                self.sources.append(path)
            # Step 04 must not fall back to either unmoved epicardium.
            for name in ("LVepi", "RVepi"):
                (self.root / "step_02_surf_clean" / f"subject_test_{frame}_{name}.vtk").write_bytes(b"not an input")

    def run_step(self, frames=("ED",), overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.step_04_extract_LVRV_surf(self.subject, frames, overwrite)

    def test_saved_surface_tags_two_frames_and_fixed_inputs(self):
        before = {path: pipeline.checksum(path) for path in self.sources}
        for info in self.run_step(("ED", "ES")):
            path = Path(info["output_path"])
            self.assertEqual(path.name, f"subject_test_{info['frame']}_LVRV_surf.vtk")
            self.assertIn("ASCII", path.read_text().splitlines()[:3])
            saved = pv.read(path)
            self.assertTrue(np.all(saved.celltypes == pv.CellType.TRIANGLE))
            self.assertEqual(saved.points.dtype, np.float64)
            np.testing.assert_array_equal(saved.cell_data["elemTag"],
                                          np.isin(saved.cell_data["sourceSurface"], (1, 2)).astype(int))
            self.assertEqual(info["saved_validation"]["free_edges"], 0)
            self.assertEqual(info["saved_validation"]["self_intersection_pairs"], 0)
            self.assertEqual(info["extraction"]["blood_pool_cavities"], 2)
            self.assertEqual(set(info["outputs"]), {"outer", "inner", "whole"})
            for role in ("outer", "inner"):
                self.assertTrue(Path(info["outputs"][role]["path"]).is_file())
            self.assertEqual(pv.read(info["outputs"]["inner"]["path"]).n_cells, 24)
        self.assertEqual(before, {path: pipeline.checksum(path) for path in self.sources})
        self.assertFalse((self.root / "step_05_LVRV_surf_clean").exists())

    def test_missing_moved_epi_does_not_fall_back_or_publish_partial_batch(self):
        (self.root / "step_03_move_RVepi" / "subject_test_ES_RVepi_moved.vtk").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "RVepi_moved"):
            self.run_step(("ED", "ES"))
        self.assertFalse(self.output_dir.exists())

    def test_missing_moved_lv_epi_does_not_fall_back(self):
        (self.root / "step_03_move_RVepi" / "subject_test_ED_LVepi_moved.vtk").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "LVepi_moved"):
            self.run_step()
        self.assertFalse(self.output_dir.exists())

    def test_failed_validation_leaves_previous_output_unchanged(self):
        self.output_dir.mkdir()
        path = self.output_dir / "subject_test_ED_LVRV_surf.vtk"
        path.write_bytes(b"previous valid output")
        with patch.object(pipeline, "save_clean_surface", side_effect=RuntimeError("invalid output")):
            with self.assertRaisesRegex(RuntimeError, "invalid output"):
                self.run_step(overwrite=True)
        self.assertEqual(path.read_bytes(), b"previous valid output")
        self.assertFalse(list(self.output_dir.glob(".step04_extract_*")))

    def test_existing_outputs_require_opt_in_and_are_backed_up(self):
        self.run_step()
        with self.assertRaises(FileExistsError):
            self.run_step()
        path = self.output_dir / "subject_test_ED_LVRV_surf.vtk"
        previous = path.read_bytes()
        info = self.run_step(overwrite=True)[0]
        self.assertEqual((Path(info["previous_output_backup"]) / path.name).read_bytes(), previous)
        summary = json.loads((self.output_dir / "subject_test_ED_step04_extract_LVRV_surf_summary.json").read_text())
        self.assertEqual(summary["output_sha256"], pipeline.checksum(path))

    def test_cli_runs_only_step04(self):
        with patch.object(pipeline, "step_01_load_surf") as first, \
                patch.object(pipeline, "step_02_surf_clean") as second, \
                patch.object(pipeline, "step_03_move_RVepi") as third, \
                patch.object(pipeline, "step_04_extract_LVRV_surf") as fourth, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--subject-dir", str(self.subject), "--frame", "ES", "--start-step", "4", "--stop-after", "4"])
        first.assert_not_called()
        second.assert_not_called()
        third.assert_not_called()
        fourth.assert_called_once_with(self.subject, ["ES"], False, 3.0)


if __name__ == "__main__":
    unittest.main()
