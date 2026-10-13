"""Checks for MeshFix repair, saved-file routing, and source preservation."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_open_volmesh_stable as pipeline
from model_vol_septum.clean_step02_surfaces import clean_surface_meshfix, surface_quality
from model_vol_septum.mesh_intersections import confirmed_self_intersections


def intersecting_surface():
    return pv.Cube().triangulate().merge(
        pv.Cube(center=(.4, .3, .2)).triangulate(), merge_points=False,
    )


class SurfaceCleaningTests(unittest.TestCase):
    def test_nearby_disjoint_triangles_are_not_reported_as_intersections(self):
        points = np.array([
            [-9.375, 22.08193, -21.], [-10.625, 23.12462, -19.], [-9.375, 23.125, -19.56493],
            [-10.62384, 23.125, -19.], [-10.625, 23.125, -18.99949], [-9.375, 24.375, -17.78251],
        ])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        self.assertEqual(len(confirmed_self_intersections(points, faces)), 0)
        crossing = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                             [.5, .5, -1.], [.5, .5, 1.], [1., .5, 0.]])
        self.assertEqual(len(confirmed_self_intersections(crossing, faces)), 1)

    def test_meshfix_removes_intersections_without_mutating_source(self):
        source = intersecting_surface()
        points, faces = source.points.copy(), source.faces.copy()
        repaired, info = clean_surface_meshfix(source, 1)
        self.assertGreater(info["before"]["self_intersection_pairs"], 0)
        self.assertEqual(info["after"]["self_intersection_pairs"], 0)
        self.assertEqual(info["after"]["free_edges"], 0)
        self.assertTrue(np.all(repaired.cell_data["elemTag"] == 1))
        np.testing.assert_array_equal(source.points, points)
        np.testing.assert_array_equal(source.faces, faces)

    def test_stage_reads_step01_and_keeps_filenames_tags_and_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir, output_dir = root / "step_01_load_surf", root / "step_02_surf_clean"
            source_dir.mkdir()
            paths = {}
            for name in ("LVendo", "LVepi", "RV", "RVepi"):
                path = source_dir / f"subject_test_ED_{name}.vtk"
                intersecting_surface().save(path)
                paths[name] = str(path)
            hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths.values()}
            surfaces, outputs, summary = pipeline.save_step2_cleaned_surfaces(
                output_dir, "subject_test", "ED", paths,
            )
            for name, key, tag in (("LVendo", "lv_endo", 1), ("LVepi", "lv_epi", 1),
                                   ("RV", "rv_endo", 0), ("RVepi", "rv_epi", 0)):
                self.assertEqual(Path(outputs[name]).name, Path(paths[name]).name)
                saved = pv.read(outputs[name])
                self.assertEqual(saved.points.dtype, np.float64)
                np.testing.assert_array_equal(saved.points, surfaces[key].points)
                self.assertTrue(np.all(saved.cell_data["elemTag"] == tag))
                self.assertEqual(surface_quality(surfaces[key])["self_intersection_pairs"], 0)
                self.assertEqual(summary["surfaces"][name]["input_path"], paths[name])
            for path, before in hashes.items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), before)
            self.assertEqual(Path(outputs["summary"]).name, "subject_test_ED_step02_surf_clean_summary.json")

    def test_failed_saved_validation_keeps_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "subject_test_ED_LVendo.vtk"
            pv.Cube().triangulate().save(source)
            out = root / "step_02_surf_clean"
            out.mkdir()
            existing = out / source.name
            existing.write_text("previous validated result")
            bad_result = intersecting_surface()
            bad_result.cell_data["elemTag"] = np.ones(bad_result.n_cells, dtype=np.int32)
            with patch("model_vol_septum.clean_step02_surfaces.clean_surface_meshfix", return_value=(bad_result, {})):
                with self.assertRaisesRegex(RuntimeError, "self_intersection_pairs"):
                    pipeline.save_step2_cleaned_surfaces(out, "subject_test", "ED", {"LVendo": str(source)})
            self.assertEqual(existing.read_text(), "previous validated result")


if __name__ == "__main__":
    unittest.main()
