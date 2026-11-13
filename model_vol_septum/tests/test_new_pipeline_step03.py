"""Closed LVepi/RVepi movement has no basal-ring requirements or remeshing."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_open_volmesh as pipeline
from model_vol_septum.generate_open_volmesh import move_closed_epi, validate_closed_epi


def cube(radius, center=(0, 0, 0)):
    mesh = pv.Cube(bounds=(-radius, radius, -radius, radius, -radius, radius)).triangulate().clean()
    mesh.points = np.asarray(mesh.points, dtype=np.float64) + np.asarray(center)
    return mesh


class ClosedMovementTests(unittest.TestCase):
    def test_already_clear_surface_is_unchanged(self):
        epi, endo = cube(6), cube(1)
        moved, info = move_closed_epi(epi, endo)
        np.testing.assert_array_equal(moved.points, epi.points)
        np.testing.assert_array_equal(moved.faces, epi.faces)
        self.assertEqual(info["iterations"], 0)
        self.assertGreaterEqual(info["validation"]["minimum_surface_distance_mm"], 3)

    def test_triangle_interiors_not_just_vertices_control_clearance(self):
        epi, endo = cube(3.5), cube(1)
        old_points, old_faces = endo.points.copy(), endo.faces.copy()
        moved, info = move_closed_epi(epi, endo, max_iterations=60)
        self.assertTrue(np.all(moved["initialSignedDistanceToRVendo"] > 3))
        self.assertGreater(info["iterations"], 0)
        self.assertGreaterEqual(info["validation"]["minimum_surface_distance_mm"], 3)
        np.testing.assert_array_equal(moved.faces, epi.faces)
        np.testing.assert_array_equal(endo.points, old_points)
        np.testing.assert_array_equal(endo.faces, old_faces)
        self.assertTrue(np.all(moved["outwardDisplacement"] >= -1e-9))
        self.assertEqual(moved.n_open_edges, 0)

    def test_inside_surface_expands_outward_to_contain_endo(self):
        epi, endo = cube(1), cube(5)
        moved, info = move_closed_epi(epi, endo, max_iterations=100)
        self.assertEqual(info["initial_vertices_inside_RVendo"], epi.n_points)
        checked = validate_closed_epi(moved, endo, source=epi)
        self.assertEqual(checked["RVendo_vertices_outside_RVepi"], 0)
        self.assertGreaterEqual(checked["minimum_surface_distance_mm"], 3)

    def test_disjoint_non_enclosing_surface_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "does not enclose"):
            validate_closed_epi(cube(2, center=(20, 0, 0)), cube(1))

    def test_step02_point_and_cell_normals_are_not_confused(self):
        epi = cube(2).compute_normals(point_normals=True, cell_normals=True,
            consistent_normals=True, auto_orient_normals=True, split_vertices=False)
        self.assertNotEqual(epi.n_points, epi.n_cells)
        moved, info = move_closed_epi(epi, cube(1), max_iterations=100)
        self.assertEqual(moved.n_points, epi.n_points)
        self.assertGreaterEqual(info["validation"]["minimum_surface_distance_mm"], 3)

    def test_lv_uses_same_clearance_rule_with_lv_labels(self):
        epi, endo = cube(3.5), cube(1)
        before = endo.points.copy()
        moved, info = move_closed_epi(epi, endo, chamber="LV", max_iterations=100)
        self.assertTrue(np.all(moved.cell_data["elemTag"] == 1))
        self.assertIn("signedDistanceToLVendo", moved.point_data)
        self.assertNotIn("signedDistanceToRVendo", moved.point_data)
        self.assertTrue(np.all(moved.point_data["insideLVendo"] == 0))
        self.assertEqual(info["validation"]["LVendo_vertices_outside_LVepi"], 0)
        self.assertGreaterEqual(info["validation"]["minimum_surface_distance_mm"], 3)
        np.testing.assert_array_equal(moved.faces, epi.faces)
        np.testing.assert_array_equal(endo.points, before)


class Stage3Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "subject_test"
        self.input_dir = self.subject / "volmesh_pipline/step_02_surf_clean"
        self.output_dir = self.subject / "volmesh_pipline/step_03_move_RVepi"
        self.input_dir.mkdir(parents=True)
        for frame in ("ED", "ES"):
            for name in pipeline.SURFACES:
                mesh = cube(2 if name.endswith("epi") else 1)
                mesh.save(self.input_dir / f"subject_test_{frame}_{name}.vtk")

    def test_stage_saves_ascii_diagnostics_and_preserves_fixed_files(self):
        before = {path.name: pipeline.checksum(path) for path in self.input_dir.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            info = pipeline.step_03_move_RVepi(self.subject, ("ED",), max_iterations=100)[0]
        saved = pv.read(info["output_path"])
        self.assertEqual(Path(info["output_path"]).name, "subject_test_ED_RVepi_moved.vtk")
        self.assertIn("ASCII", Path(info["output_path"]).read_text().splitlines()[:3])
        self.assertTrue(np.all(saved.cell_data["elemTag"] == 0))
        self.assertIn("displacement", saved.point_data)
        self.assertIn("signedDistanceToRVendo", saved.point_data)
        self.assertTrue(np.all(saved["insideRVendo"] == 0))
        self.assertGreaterEqual(info["saved_validation"]["minimum_surface_distance_mm"], 3)
        self.assertEqual(set(info["outputs"]), {"LVepi", "RVepi"})
        lv = pv.read(info["outputs"]["LVepi"]["output_path"])
        self.assertTrue(np.all(lv.cell_data["elemTag"] == 1))
        self.assertEqual(Path(info["outputs"]["LVepi"]["output_path"]).name, "subject_test_ED_LVepi_moved.vtk")
        self.assertGreaterEqual(info["outputs"]["LVepi"]["saved_validation"]["minimum_surface_distance_mm"], 3)
        self.assertIn("signedDistanceToLVendo", lv.point_data)
        self.assertEqual(info["fixed_surfaces"], ["LVendo", "RV"])
        self.assertEqual(before, {path.name: pipeline.checksum(path) for path in self.input_dir.iterdir()})
        self.assertFalse((self.output_dir.parent / "step_04_extract_LVRV_surf").exists())

    def test_failed_movement_preserves_previous_file_and_records_failure(self):
        self.output_dir.mkdir()
        previous = self.output_dir / "subject_test_ED_RVepi_moved.vtk"
        previous.write_bytes(b"previous valid output")
        with patch.object(pipeline, "move_closed_epi", side_effect=RuntimeError("failed constraints")):
            with self.assertRaisesRegex(RuntimeError, "failed constraints"):
                pipeline.step_03_move_RVepi(self.subject, ("ED",), overwrite=True)
        self.assertEqual(previous.read_bytes(), b"previous valid output")
        failure = next((self.output_dir / "failed_attempts").glob("*.json"))
        self.assertFalse(json.loads(failure.read_text())["invalid_mesh_published"])

    def test_lv_failure_does_not_publish_partial_rv_result(self):
        original = pipeline.move_closed_epi

        def move_or_fail(*args, **kwargs):
            if kwargs["chamber"] == "LV":
                raise RuntimeError("LV failed constraints")
            return original(*args, **kwargs)

        with patch.object(pipeline, "move_closed_epi", side_effect=move_or_fail):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "LV failed constraints"):
                    pipeline.step_03_move_RVepi(self.subject, ("ED",))
        self.assertFalse(list(self.output_dir.glob("*.vtk")))

    def test_two_frames_save_both_chambers_and_back_up_previous_results(self):
        with contextlib.redirect_stdout(io.StringIO()):
            records = pipeline.step_03_move_RVepi(self.subject, ("ED", "ES"))
        self.assertEqual(len(list(self.output_dir.glob("*.vtk"))), 4)
        before = {path.name: path.read_bytes() for path in self.output_dir.glob("*.vtk")}
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = pipeline.step_03_move_RVepi(self.subject, ("ED", "ES"), overwrite=True)
        backup = Path(repeated[0]["previous_output_backup"])
        for name, contents in before.items():
            self.assertEqual((backup / name).read_bytes(), contents)
        for info in records:
            for chamber in ("LV", "RV"):
                self.assertEqual(Path(info["outputs"][f"{chamber}epi"]["output_path"]).name,
                                 f"subject_test_{info['frame']}_{chamber}epi_moved.vtk")

    def test_already_clear_lv_stays_fixed_while_rv_moves(self):
        source = self.input_dir / "subject_test_ED_LVepi.vtk"
        cube(6).save(source)
        with contextlib.redirect_stdout(io.StringIO()):
            info = pipeline.step_03_move_RVepi(self.subject, ("ED",))[0]
        self.assertEqual(info["outputs"]["LVepi"]["movement"]["iterations"], 0)
        self.assertGreater(info["outputs"]["RVepi"]["movement"]["iterations"], 0)
        np.testing.assert_array_equal(pv.read(source).points,
                                      pv.read(info["outputs"]["LVepi"]["output_path"]).points)

    def test_cli_runs_only_step03_when_requested(self):
        with patch.object(pipeline, "step_01_load_surf") as first, \
                patch.object(pipeline, "step_02_surf_clean") as second, \
                patch.object(pipeline, "step_03_move_RVepi") as third, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--subject-dir", str(self.subject), "--frame", "ED", "--start-step", "3", "--stop-after", "3"])
        first.assert_not_called()
        second.assert_not_called()
        third.assert_called_once_with(self.subject, ["ED"], False, 3.0, 600)


if __name__ == "__main__":
    unittest.main()
