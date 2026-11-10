"""The new Step 02 repairs copies, validates saved files and stops safely."""
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


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "subject_test"
        self.input_dir = self.subject / "volmesh_pipline/step_01_load_surf"
        self.output_dir = self.subject / "volmesh_pipline/step_02_surf_clean"
        self.input_dir.mkdir(parents=True)
        for frame in ("ED", "ES"):
            for name in pipeline.SURFACES:
                pv.Cube().triangulate().save(self.input_dir / f"subject_test_{frame}_{name}.vtk")

    def run_step(self, frames=("ED",), overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.step_02_surf_clean(self.subject, frames, overwrite)

    def test_clean_outputs_names_tags_validation_and_immutable_inputs(self):
        before = {path.name: pipeline.checksum(path) for path in self.input_dir.glob("*.vtk")}
        results = self.run_step(("ED", "ES"))
        for info in results:
            self.assertEqual(info["step"], "step_02_surf_clean")
            for name, record in info["surfaces"].items():
                path = Path(record["output_path"])
                self.assertEqual(path.name, f"subject_test_{info['frame']}_{name}.vtk")
                mesh = pv.read(path)
                self.assertEqual(mesh.points.dtype, np.float64)
                self.assertTrue(np.all(mesh.celltypes == pv.CellType.TRIANGLE))
                self.assertTrue(np.all(mesh["elemTag"] == (1 if name.startswith("LV") else 0)))
                self.assertIn("ASCII", path.read_text().splitlines()[:3])
                for key in ("free_edges", "nonmanifold_edges", "degenerate_triangles",
                            "inconsistent_orientation_edges", "self_intersection_pairs"):
                    self.assertEqual(record["saved_validation"][key], 0)
        self.assertEqual(before, {path.name: pipeline.checksum(path) for path in self.input_dir.glob("*.vtk")})
        self.assertFalse((self.output_dir.parent / "step_03_move_RVepi").exists())

    def test_holes_are_filled_only_in_step02(self):
        cube = pv.Cube().triangulate()
        opened = cube.extract_cells(np.flatnonzero(cube.cell_centers().points[:, 2] < .49)).extract_surface()
        source = self.input_dir / "subject_test_ED_RV.vtk"
        opened.save(source)
        info = self.run_step()[0]["surfaces"]["RV"]
        self.assertGreater(info["cleaning"]["before"]["free_edges"], 0)
        self.assertEqual(info["saved_validation"]["free_edges"], 0)
        self.assertGreater(pv.read(source).n_open_edges, 0)

    def test_missing_input_does_not_publish_other_frames(self):
        (self.input_dir / "subject_test_ES_RV.vtk").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_step(("ED", "ES"))
        self.assertFalse(self.output_dir.exists())

    def test_failed_validation_does_not_replace_previous_outputs(self):
        self.output_dir.mkdir()
        previous = self.output_dir / "subject_test_ED_LVendo.vtk"
        previous.write_bytes(b"previous validated surface")
        with patch.object(pipeline, "save_clean_surface", side_effect=RuntimeError("invalid saved mesh")):
            with self.assertRaisesRegex(RuntimeError, "invalid saved mesh"):
                self.run_step(overwrite=True)
        self.assertEqual(previous.read_bytes(), b"previous validated surface")

    def test_existing_outputs_are_backed_up_before_replacement(self):
        self.run_step()
        with self.assertRaises(FileExistsError):
            self.run_step()
        path = self.output_dir / "subject_test_ED_LVepi.vtk"
        previous = path.read_bytes()
        info = self.run_step(overwrite=True)[0]
        backup = Path(info["previous_output_backup"])
        self.assertEqual((backup / path.name).read_bytes(), previous)
        summary = json.loads((self.output_dir / "subject_test_ED_step02_surf_clean_summary.json").read_text())
        self.assertEqual(summary["previous_output_backup"], str(backup))

    def test_cli_can_run_step02_without_reloading_step01(self):
        with patch.object(pipeline, "step_01_load_surf") as first, \
                patch.object(pipeline, "step_02_surf_clean") as second, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--subject-dir", str(self.subject), "--frame", "ES", "--start-step", "2", "--stop-after", "2"])
        first.assert_not_called()
        second.assert_called_once_with(self.subject, ["ES"], False)


if __name__ == "__main__":
    unittest.main()
