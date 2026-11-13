"""Surface-based basal opening preview, TetGen meshing and spatial tag transfer."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv

from test_new_pipeline_step07 import cones, pipeline, synthetic_shell


def tagged_grid(points, cells, tags):
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 4)
    grid = pv.UnstructuredGrid(np.column_stack((np.full(len(cells), 4), cells)).ravel(),
        np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8), np.asarray(points, dtype=float))
    grid.cell_data["elemTag"] = np.asarray(tags, dtype=np.int32)
    return grid


class SurfacePreviewTests(unittest.TestCase):
    def test_spatial_tags_are_transferred_without_parent_child_claim(self):
        original = tagged_grid([[0,0,0], [1,0,0], [0,1,0], [0,0,1],
                                [3,0,0], [4,0,0], [3,1,0], [3,0,1]],
                               [[0,1,2,3], [4,5,6,7]], [17,42])
        new = original.copy(deep=True)
        new.clear_data()
        before_points, before_cells = new.points.copy(), new.cells.copy()
        result = pipeline.label_remeshed_volume(new, original)
        np.testing.assert_array_equal(result.cell_data["elemTag"], [17,42])
        np.testing.assert_array_equal(result.cell_data["referenceCellId"], [0,1])
        np.testing.assert_array_equal(result.points, before_points)
        np.testing.assert_array_equal(result.cells, before_cells)
        self.assertNotIn("parentCellId", result.cell_data)

    def test_nearest_cell_is_used_when_new_center_is_outside_old_mesh(self):
        original = tagged_grid([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], [[0,1,2,3]], [9])
        new = original.translate((3,0,0), inplace=False)
        result = pipeline.label_remeshed_volume(new, original)
        np.testing.assert_array_equal(result.cell_data["elemTag"], [9])
        np.testing.assert_array_equal(result.cell_data["referenceCellId"], [0])

    def test_shell_extends_basal_patches_without_mutating_inputs(self):
        source = synthetic_shell(cones())
        parts = pipeline._lvrv_surface_components(source)
        before = {key: mesh.points.copy() for key, mesh in parts.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            shell, removed, artifacts, info = pipeline.build_open_myocardial_shell(parts, (0,0,-1))
        self.assertEqual(shell.n_open_edges, 0)
        self.assertTrue(shell.is_all_triangles)
        self.assertTrue(removed.is_all_triangles)
        self.assertEqual(len(artifacts), 8)
        for key, mesh in parts.items():
            np.testing.assert_array_equal(mesh.points, before[key])
        for name in ("LV", "RV"):
            delta = np.array((0,0,-info[name]["displacement_mm"]))
            np.testing.assert_allclose(artifacts[f"{name}_basal_moved"].points,
                artifacts[f"{name}_basal_patch"].points+delta)
            self.assertGreater(artifacts[f"{name}_rim"].n_lines, 0)


class Stage8Tests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.subject = Path(directory.name) / "subject_test"
        self.root = self.subject / "volmesh_pipline"
        self.output = self.root / "step_08_LV_RV_openbase"
        surface = synthetic_shell(cones())
        with contextlib.redirect_stdout(io.StringIO()):
            grid, _ = pipeline.tetrahedralize_lvrv_surface(surface)
        for suffix, mesh in (("volmesh_notag", grid), ("volmesh", grid.copy(deep=True))):
            if suffix == "volmesh":
                mesh.cell_data["elemTag"] = (mesh.cell_centers().points[:, 0] < 0).astype(np.int32)
            path = self.root / "step_06_volmesh_generate" / f"subject_test_ED_{suffix}.vtk"
            path.parent.mkdir(parents=True, exist_ok=True)
            pipeline.write_tetrahedral_vtk(mesh, path)
        path = self.root / "step_05_LVRV_surf_clean" / "subject_test_ED_LVRV_surf.vtk"
        path.parent.mkdir()
        surface.save(path)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.step_07_base_detection(self.subject, ("ED",), samples=64,
                                           base_axis="z", base_side="low")

    def run_step(self, overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.step_08_LV_RV_openbase(self.subject, ("ED",), overwrite)[0]

    def test_only_requested_vtks_are_written_with_new_mesh_and_spatial_tags(self):
        with patch.object(pipeline, "tag_lvrv_volume", side_effect=AssertionError("No retagging")):
            result = self.run_step()
        files = list(self.output.iterdir())
        self.assertEqual(len(files), 10)
        self.assertTrue(all(path.suffix == ".vtk" for path in files))
        for chamber in ("LV", "RV"):
            for suffix in ("endo_basal_patch", "endo_basal_patch_displaced", "mapped_endo_rim", "opening_rim"):
                self.assertTrue((self.output / f"subject_test_ED_{chamber}_{suffix}.vtk").is_file())
        original = pv.read(self.root / "step_06_volmesh_generate" / "subject_test_ED_volmesh.vtk")
        kept, removed = pv.read(result["outputs"]["volume"]), pv.read(result["outputs"]["removed"])
        np.testing.assert_array_equal(kept.cell_data["elemTag"],
            original.cell_data["elemTag"][kept.cell_data["referenceCellId"]])
        self.assertEqual(len(np.unique(kept.cells.reshape(-1, 5)[:, 1:])), kept.n_points)
        self.assertTrue(np.all(kept.celltypes == pv.CellType.TETRA))
        self.assertTrue(removed.is_all_triangles)
        self.assertTrue(result["new_tetrahedralization_performed"])
        self.assertNotIn("parentCellId", kept.cell_data)

    def test_previous_extras_are_archived(self):
        self.output.mkdir()
        old = self.output / "subject_test_ED_step08_openbase_summary.json"
        old.write_bytes(b"previous unused output")
        self.run_step(overwrite=True)
        self.assertFalse(old.exists())
        archive = list((self.output / "previous_outputs").glob("*/" + old.name))
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0].read_bytes(), b"previous unused output")

    def test_missing_tagged_input_does_not_fall_back_to_retagging(self):
        path = self.root / "step_06_volmesh_generate" / "subject_test_ED_volmesh.vtk"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "volmesh.vtk"):
            self.run_step()

    def test_cli_only_runs_step08(self):
        with patch.object(pipeline, "step_07_base_detection") as earlier, \
                patch.object(pipeline, "step_08_LV_RV_openbase") as current, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--subject-dir", str(self.subject), "--frame", "ED", "--start-step", "8",
                           "--stop-after", "8"])
        earlier.assert_not_called()
        current.assert_called_once_with(self.subject, ["ED"], False, .30, 1.2, 5., 0.)


if __name__ == "__main__":
    unittest.main()
