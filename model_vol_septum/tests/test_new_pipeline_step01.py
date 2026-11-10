"""The new Step 01 only copies surfaces and preserves previous outputs."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_open_volmesh as pipeline


class LoadSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "subject_test"
        self.input_dir = self.subject / "vtks"
        self.input_dir.mkdir(parents=True)
        self.output_dir = self.subject / "volmesh_pipline/step_01_load_surf"
        for frame in ("ED", "ES"):
            for name in pipeline.SURFACES:
                mesh = pv.Cube().triangulate()
                mesh.point_data["originalPointId"] = np.arange(mesh.n_points)
                mesh.save(self.input_dir / f"{name}_{frame}.vtk")

    def run_step(self, frames=("ED", "ES"), overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.step_01_load_surf(self.subject, frames, overwrite)

    def test_eight_copies_are_byte_identical_and_no_later_stage_created(self):
        results = self.run_step()
        self.assertEqual([result["frame"] for result in results], ["ED", "ES"])
        for frame in ("ED", "ES"):
            for name in pipeline.SURFACES:
                source = self.input_dir / f"{name}_{frame}.vtk"
                saved = self.output_dir / f"subject_test_{frame}_{name}.vtk"
                self.assertEqual(source.read_bytes(), saved.read_bytes())
                np.testing.assert_array_equal(pv.read(source).points, pv.read(saved).points)
                np.testing.assert_array_equal(pv.read(source).faces, pv.read(saved).faces)
        self.assertEqual([path.name for path in self.output_dir.parent.iterdir()], ["step_01_load_surf"])

    def test_one_frame_only(self):
        self.run_step(("ES",))
        self.assertFalse(list(self.output_dir.glob("subject_test_ED*")))
        self.assertEqual(len(list(self.output_dir.glob("subject_test_ES_*.vtk"))), 4)

    def test_missing_es_input_does_not_publish_partial_ed_outputs(self):
        (self.input_dir / "RVepi_ES.vtk").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_step()
        self.assertFalse(self.output_dir.exists())

    def test_existing_outputs_require_opt_in_and_are_backed_up(self):
        self.run_step()
        previous = self.output_dir / "subject_test_ED_LVendo.vtk"
        previous.write_bytes(b"previous pipeline output")
        with self.assertRaises(FileExistsError):
            self.run_step()
        results = self.run_step(overwrite=True)
        backup = Path(results[0]["previous_output_backup"])
        self.assertEqual((backup / previous.name).read_bytes(), b"previous pipeline output")
        self.assertEqual(previous.read_bytes(), (self.input_dir / "LVendo_ED.vtk").read_bytes())

    def test_open_surface_is_retained_without_hole_filling(self):
        plane = pv.Plane().triangulate()
        source = self.input_dir / "RV_ED.vtk"
        plane.save(source)
        self.run_step(("ED",))
        output = self.output_dir / "subject_test_ED_RV.vtk"
        self.assertEqual(source.read_bytes(), output.read_bytes())
        self.assertGreater(pv.read(output).n_open_edges, 0)
        info = json.loads((self.output_dir / "subject_test_ED_step01_load_surf_summary.json").read_text())
        self.assertFalse(info["repair_performed"])

    def test_volume_input_is_rejected_without_modifying_it(self):
        source = self.input_dir / "RV_ED.vtk"
        pv.Cube().triangulate().delaunay_3d().save(source)
        digest = pipeline.checksum(source)
        with self.assertRaises(ValueError):
            self.run_step(("ED",))
        self.assertEqual(digest, pipeline.checksum(source))
        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
