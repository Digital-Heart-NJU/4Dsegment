"""Step 05 cleans only the combined VTK and retains the two cavity boundaries."""
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


def shell():
    parts = []
    for radius, center, source_id in ((7, (0, 0, 0), 2), (1, (-2, 0, 0), 1), (1, (2, 0, 0), 3)):
        part = pv.Cube(bounds=(-radius, radius, -radius, radius, -radius, radius)).triangulate().clean()
        part.points = np.asarray(part.points, dtype=float) + np.asarray(center)
        part = part.compute_normals(consistent_normals=True, auto_orient_normals=True, split_vertices=False)
        if source_id != 2:
            faces = part.faces.reshape(-1, 4).copy()
            faces[:, 1:] = faces[:, 1:][:, ::-1]
            part.faces = faces.ravel()
        labels = (np.where(part.cell_centers().points[:, 0] > 0, 4, 2) if source_id == 2
                  else np.full(part.n_cells, source_id))
        part.cell_data["sourceSurface"] = labels.astype(np.int32)
        part.cell_data["elemTag"] = np.isin(labels, (1, 2)).astype(np.int32)
        parts.append(part)
    return pipeline._join_triangle_surfaces(parts)


class SurfaceCleanupTests(unittest.TestCase):
    def test_cleanup_retains_cavities_orientation_and_surface_tags(self):
        source = shell()
        points, faces = source.points.copy(), source.faces.copy()
        with patch.object(pipeline, "move_closed_epi", side_effect=AssertionError("unexpected displacement")):
            cleaned, info = pipeline.clean_lvrv_surface(source)
        self.assertEqual(info["after"]["blood_pool_cavities"], 2)
        self.assertGreaterEqual(info["after"]["minimum_outer_inner_distance_mm"], 3)
        self.assertEqual(set(cleaned.cell_data["sourceSurface"]), {1, 2, 3, 4})
        np.testing.assert_array_equal(cleaned.cell_data["elemTag"],
                                      np.isin(cleaned.cell_data["sourceSurface"], (1, 2)).astype(int))
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(cleaned)
        self.assertGreater(implicit.EvaluateFunction((-2, 0, 0)), 0)
        self.assertGreater(implicit.EvaluateFunction((2, 0, 0)), 0)
        self.assertLess(implicit.EvaluateFunction((0, 0, 0)), 0)
        np.testing.assert_array_equal(source.points, points)
        np.testing.assert_array_equal(source.faces, faces)

    def test_hole_in_inner_boundary_is_repaired_without_losing_cavity(self):
        source = shell()
        keep = np.ones(source.n_cells, dtype=bool)
        keep[np.flatnonzero(source.cell_data["sourceSurface"] == 1)[0]] = False
        opened = source.extract_cells(keep).extract_surface()
        cleaned, info = pipeline.clean_lvrv_surface(opened)
        self.assertGreater(info["before"]["free_edges"], 0)
        self.assertEqual(info["after"]["free_edges"], 0)
        self.assertEqual(info["after"]["blood_pool_cavities"], 2)
        self.assertEqual(cleaned.n_cells, 36)

    def test_missing_provenance_is_rejected_instead_of_guessing_cavities(self):
        source = shell()
        del source.cell_data["sourceSurface"]
        with self.assertRaisesRegex(ValueError, "sourceSurface"):
            pipeline.clean_lvrv_surface(source)


class Stage5Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "subject_test"
        self.root = self.subject / "volmesh_pipline"
        self.input_dir = self.root / "step_04_extract_LVRV_surf"
        self.output_dir = self.root / "step_05_LVRV_surf_clean"
        self.input_dir.mkdir(parents=True)
        for frame in ("ED", "ES"):
            shell().save(self.input_dir / f"subject_test_{frame}_LVRV_surf.vtk")

    def run_step(self, frames=("ED",), overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.step_05_LVRV_surf_clean(self.subject, frames, overwrite)

    def test_only_combined_input_is_needed_and_one_vtk_is_written_per_frame(self):
        hashes = {path: pipeline.checksum(path) for path in self.input_dir.iterdir()}
        for info in self.run_step(("ED", "ES")):
            path = Path(info["output_path"])
            self.assertEqual(path.name, f"subject_test_{info['frame']}_LVRV_surf.vtk")
            self.assertIn("ASCII", path.read_text().splitlines()[:3])
            saved = pv.read(path)
            self.assertTrue(np.all(saved.celltypes == pv.CellType.TRIANGLE))
            self.assertEqual(saved.points.dtype, np.float64)
            validation = info["saved_validation"]
            self.assertEqual(validation["blood_pool_cavities"], 2)
            for key in ("free_edges", "nonmanifold_edges", "degenerate_triangles", "self_intersection_pairs"):
                self.assertEqual(validation[key], 0)
        self.assertEqual(len(list(self.output_dir.glob("*.vtk"))), 2)
        self.assertEqual(hashes, {path: pipeline.checksum(path) for path in self.input_dir.iterdir()})
        self.assertFalse((self.root / "step_06_volmesh_generate").exists())

    def test_missing_input_does_not_publish_partial_batch(self):
        (self.input_dir / "subject_test_ES_LVRV_surf.vtk").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_step(("ED", "ES"))
        self.assertFalse(self.output_dir.exists())

    def test_failed_validation_preserves_previous_output(self):
        self.output_dir.mkdir()
        previous = self.output_dir / "subject_test_ED_LVRV_surf.vtk"
        previous.write_bytes(b"previous valid output")
        with patch.object(pipeline, "save_clean_surface", side_effect=RuntimeError("invalid saved output")):
            with self.assertRaisesRegex(RuntimeError, "invalid saved output"):
                self.run_step(overwrite=True)
        self.assertEqual(previous.read_bytes(), b"previous valid output")
        self.assertFalse(list(self.output_dir.glob(".step05_clean_*")))

    def test_overwrite_backs_up_existing_results(self):
        self.run_step()
        with self.assertRaises(FileExistsError):
            self.run_step()
        path = self.output_dir / "subject_test_ED_LVRV_surf.vtk"
        previous = path.read_bytes()
        info = self.run_step(overwrite=True)[0]
        self.assertEqual((Path(info["previous_output_backup"]) / path.name).read_bytes(), previous)
        record = json.loads((self.output_dir / "subject_test_ED_step05_LVRV_surf_clean_summary.json").read_text())
        self.assertEqual(record["output_sha256"], pipeline.checksum(path))

    def test_cli_runs_step05_without_running_upstream_steps(self):
        with patch.object(pipeline, "step_01_load_surf") as first, \
                patch.object(pipeline, "step_02_surf_clean") as second, \
                patch.object(pipeline, "step_03_move_RVepi") as third, \
                patch.object(pipeline, "step_04_extract_LVRV_surf") as fourth, \
                patch.object(pipeline, "step_05_LVRV_surf_clean") as fifth, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--subject-dir", str(self.subject), "--frame", "ED", "--start-step", "5", "--stop-after", "5"])
        for operation in (first, second, third, fourth):
            operation.assert_not_called()
        fifth.assert_called_once_with(self.subject, ["ED"], False, 3.0)


if __name__ == "__main__":
    unittest.main()
