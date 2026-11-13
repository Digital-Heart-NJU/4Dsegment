"""Step 06 preserves the PLC, excludes blood pools and tags one shared volume."""
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
from model_vol_septum.generate_step09_volume import signed_volumes
from model_vol_septum.tests.test_new_pipeline_step05 import shell


def reference():
    return pv.Cube(bounds=(-7, .137, -7, 7, -7, 7)).triangulate().clean()


def grid_from_blocks(blocks):
    points = np.asarray(blocks, dtype=float).reshape(-1, 3)
    cells = np.arange(len(points)).reshape(-1, 4)
    reverse = signed_volumes(points, cells) < 0
    cells[reverse, :2] = cells[reverse, 1::-1]
    return pv.UnstructuredGrid(np.column_stack((np.full(len(cells), 4), cells)).ravel(),
                               np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8), points)


class MeshingTests(unittest.TestCase):
    def test_one_connected_myocardium_has_exact_boundary_and_empty_blood_pools(self):
        surface = shell()
        points, faces = surface.points.copy(), surface.faces.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            grid, info = pipeline.tetrahedralize_lvrv_surface(surface)
        validation = info["volume_validation"]
        self.assertEqual(validation["component_count"], 1)
        self.assertTrue(validation["boundary_matches_step05"])
        self.assertEqual(validation["nonpositive_tetrahedra"], 0)
        self.assertAlmostEqual(validation["total_tetrahedron_volume_mm3"], 2728)
        self.assertEqual(len(grid.cell_data), 0)
        centers = grid.cell_centers().points
        for x in (-2, 2):
            self.assertFalse(np.any(np.all(np.abs(centers - (x, 0, 0)) < 1, axis=1)))
        np.testing.assert_array_equal(surface.points, points)
        np.testing.assert_array_equal(surface.faces, faces)

    def test_centroid_not_vertex_majority_determines_tag(self):
        three_inside = [[.6, 0, 0], [.7, .2, 0], [.7, 0, .2], [5, 0, 0]]
        centroid_only = [[2, 2, 2], [2, -2, -2], [-2, 2, -2], [-2, -2, 2]]
        interior = [[0, 0, 0], [.2, 0, 0], [0, .2, 0], [0, 0, .2]]
        exterior = np.asarray(interior) + (5, 0, 0)
        source = grid_from_blocks([three_inside, centroid_only, interior, exterior])
        ref = pv.Cube(bounds=(-1, 1, -1, 1, -1, 1)).triangulate().clean()
        tagged, info = pipeline.tag_lvrv_volume(source, ref)
        np.testing.assert_array_equal(tagged.cell_data["elemTag"], [0, 1, 1, 0])
        np.testing.assert_array_equal(tagged.cell_data["LVepi_inside_vertex_count"], [3, 0, 4, 0])
        np.testing.assert_array_equal(tagged.cell_data["LVepi_mixed_vertices"], [1, 0, 0, 0])
        np.testing.assert_array_equal(tagged.points, source.points)
        np.testing.assert_array_equal(tagged.cells, source.cells)
        self.assertFalse(info["tag_island_relabeling_performed"])

    def test_ascii_writer_preserves_double_coordinates_exactly(self):
        x = .12345678901234566
        grid = grid_from_blocks([[[x, 0, 0], [x + 1, 0, 0], [x, 1 / 3, 0], [x, 0, 2 / 7]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.vtk"
            for tagged in (False, True):
                if tagged:
                    grid.cell_data["elemTag"] = np.ones(grid.n_cells, dtype=np.int32)
                pipeline.write_tetrahedral_vtk(grid, path)
                saved = pv.read(path)
                self.assertIn("ASCII", path.read_text().splitlines()[:3])
                np.testing.assert_array_equal(saved.points, grid.points)
                np.testing.assert_array_equal(saved.cells, grid.cells)
                self.assertEqual("elemTag" in saved.cell_data, tagged)

    def test_open_boundary_is_rejected_before_tetgen(self):
        source = shell()
        opened = source.extract_cells(np.arange(1, source.n_cells)).extract_surface()
        with patch("tetgen.TetGen") as mesher:
            with self.assertRaisesRegex(RuntimeError, "free_edges"):
                pipeline.tetrahedralize_lvrv_surface(opened)
        mesher.assert_not_called()


class Stage6Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "subject_test"
        self.root = self.subject / "volmesh_pipline"
        self.output_dir = self.root / "step_06_volmesh_generate"
        (self.root / "step_05_LVRV_surf_clean").mkdir(parents=True)
        (self.root / "step_03_move_RVepi").mkdir()
        self.sources = []
        for frame in ("ED", "ES"):
            boundary = self.root / "step_05_LVRV_surf_clean" / f"subject_test_{frame}_LVRV_surf.vtk"
            ref = self.root / "step_03_move_RVepi" / f"subject_test_{frame}_LVepi_moved.vtk"
            shell().save(boundary)
            reference().save(ref)
            self.sources.extend((boundary, ref))

    def run_step(self, frames=("ED",), overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.step_06_volmesh_generate(self.subject, frames, overwrite)

    def test_two_frames_save_shared_geometry_and_requested_tags(self):
        before = {path: pipeline.checksum(path) for path in self.sources}
        for info in self.run_step(("ED", "ES")):
            raw_path = Path(info["outputs"]["untagged"]["path"])
            tagged_path = Path(info["outputs"]["tagged"]["path"])
            self.assertEqual(raw_path.name, f"subject_test_{info['frame']}_volmesh_notag.vtk")
            self.assertEqual(tagged_path.name, f"subject_test_{info['frame']}_volmesh.vtk")
            raw, tagged = pv.read(raw_path), pv.read(tagged_path)
            self.assertEqual(len(raw.cell_data), 0)
            np.testing.assert_array_equal(raw.points, tagged.points)
            np.testing.assert_array_equal(raw.cells, tagged.cells)
            self.assertEqual(set(tagged.cell_data["elemTag"]), {0, 1})
            expected = pv.PolyData(raw.cell_centers().points).select_enclosed_points(reference(), tolerance=0)["SelectedPoints"]
            np.testing.assert_array_equal(tagged.cell_data["elemTag"], expected)
            self.assertEqual(info["saved_volume_validation"]["component_count"], 1)
            self.assertTrue(info["saved_volume_validation"]["boundary_matches_step05"])
            self.assertEqual(Path(info["LVepi_reference_path"]).parent.name, "step_03_move_RVepi")
        self.assertEqual(before, {path: pipeline.checksum(path) for path in self.sources})
        self.assertEqual(len(list(self.output_dir.glob("*.vtk"))), 4)
        self.assertFalse((self.root / "step_07_base_detection").exists())

    def test_missing_moved_reference_does_not_fall_back_or_publish_partial_batch(self):
        (self.root / "step_03_move_RVepi" / "subject_test_ES_LVepi_moved.vtk").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "LVepi_moved"):
            self.run_step(("ED", "ES"))
        self.assertFalse(self.output_dir.exists())

    def test_tagging_failure_preserves_previous_outputs(self):
        self.output_dir.mkdir()
        previous = [self.output_dir / f"subject_test_ED_{suffix}.vtk" for suffix in ("volmesh_notag", "volmesh")]
        for path in previous:
            path.write_bytes(b"previous valid output")
        with patch.object(pipeline, "tag_lvrv_volume", side_effect=RuntimeError("invalid tags")):
            with self.assertRaisesRegex(RuntimeError, "invalid tags"):
                self.run_step(overwrite=True)
        for path in previous:
            self.assertEqual(path.read_bytes(), b"previous valid output")
        self.assertFalse(list(self.output_dir.glob(".step06_volume_*")))

    def test_overwrite_backs_up_both_volume_files(self):
        self.run_step()
        with self.assertRaises(FileExistsError):
            self.run_step()
        previous = {path.name: path.read_bytes() for path in self.output_dir.glob("*.vtk")}
        info = self.run_step(overwrite=True)[0]
        for name, contents in previous.items():
            self.assertEqual((Path(info["previous_output_backup"]) / name).read_bytes(), contents)
        summary = json.loads((self.output_dir / "subject_test_ED_step06_volmesh_generate_summary.json").read_text())
        for output in summary["outputs"].values():
            self.assertEqual(output["sha256"], pipeline.checksum(Path(output["path"])))

    def test_cli_runs_only_step06(self):
        with patch.object(pipeline, "step_01_load_surf") as first, \
                patch.object(pipeline, "step_02_surf_clean") as second, \
                patch.object(pipeline, "step_03_move_RVepi") as third, \
                patch.object(pipeline, "step_04_extract_LVRV_surf") as fourth, \
                patch.object(pipeline, "step_05_LVRV_surf_clean") as fifth, \
                patch.object(pipeline, "step_06_volmesh_generate") as sixth, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--subject-dir", str(self.subject), "--frame", "ED", "--start-step", "6", "--stop-after", "6"])
        for operation in (first, second, third, fourth, fifth):
            operation.assert_not_called()
        sixth.assert_called_once_with(self.subject, ["ED"], False, 1.2, 5.0, 0.0)


if __name__ == "__main__":
    unittest.main()
