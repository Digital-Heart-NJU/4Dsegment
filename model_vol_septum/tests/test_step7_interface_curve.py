"""Step 07 epicardial interface-curve exports preserve their source surfaces."""
import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_open_volmesh_stable as pipeline


class Step7InterfaceCurveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.args = Namespace(surface_clean_tolerance=1e-7, lv_tag=1, rv_tag=0)

    def save_inputs(self, lv_epi, rv_epi):
        paths = [self.root / "LVepi.vtk", self.root / "RVepi.vtk"]
        for path, mesh in zip(paths, (lv_epi, rv_epi)):
            mesh.save(path, binary=False)
        return paths

    def export(self, inputs, frame="ED"):
        return pipeline.save_step7_rvepi_lvepi_interface_curve(
            self.root, "subject_test", frame, *inputs, self.args,
        )

    def test_crossing_surfaces_save_ascii_lines_without_modifying_inputs(self):
        inputs = self.save_inputs(pv.Sphere(radius=1), pv.Sphere(center=(0.6, 0, 0), radius=1))
        checksums = [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]
        output, info = self.export(inputs)
        curve = pv.read(output)
        self.assertEqual(Path(output).name, "subject_test_ED_septum.vtk")
        self.assertIn("ASCII", Path(output).read_text().splitlines()[:3])
        self.assertIsInstance(curve, pv.PolyData)
        self.assertEqual(curve.points.dtype, np.float64)
        self.assertGreater(curve.n_lines, 0)
        self.assertEqual(curve.n_cells, curve.n_lines)
        self.assertEqual(curve.n_faces_strict, 0)
        self.assertEqual(info["status"], "intersection_found")
        self.assertGreater(info["length_mm"], 0)
        self.assertEqual(info["connected_components"], 1)
        for source in inputs:
            distances = curve.compute_implicit_distance(pv.read(source))["implicit_distance"]
            self.assertLess(np.max(np.abs(distances)), 1e-6)
        self.assertEqual(checksums, [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs])

    def test_open_intersection_is_not_artificially_closed(self):
        inputs = self.save_inputs(
            pv.Plane(i_size=2, j_size=2, i_resolution=1, j_resolution=1).triangulate(),
            pv.Plane(center=(0.2, 0.15, 0), direction=(1, 0, 0),
                     i_size=2, j_size=2, i_resolution=1, j_resolution=1).triangulate(),
        )
        output, info = self.export(inputs, frame="ES")
        curve = pv.read(output)
        self.assertEqual(Path(output).name, "subject_test_ES_septum.vtk")
        self.assertGreater(curve.n_lines, 0)
        edges = curve.lines.reshape(-1, 3)
        self.assertTrue(np.all(edges[:, 0] == 2))
        _, degrees = np.unique(edges[:, 1:], return_counts=True)
        self.assertEqual(np.count_nonzero(degrees == 1), 2)
        self.assertEqual(info["connected_components"], 1)

    def test_disjoint_surfaces_save_empty_vtk_without_fabricating_a_seam(self):
        inputs = self.save_inputs(pv.Sphere(), pv.Sphere(center=(5, 0, 0)))
        output, info = self.export(inputs)
        curve = pv.read(output)
        self.assertEqual(curve.n_points, 0)
        self.assertEqual(curve.n_cells, 0)
        self.assertEqual(info["status"], "no_intersection")
        self.assertEqual(info["connected_components"], 0)
        self.assertEqual(info["length_mm"], 0)

    def test_filter_failure_preserves_previous_output(self):
        inputs = self.save_inputs(pv.Sphere(), pv.Sphere(center=(0.6, 0, 0)))
        output = self.root / "subject_test_ED_septum.vtk"
        previous = b"previous validated curve"
        output.write_bytes(previous)
        with patch.object(pipeline.vtk, "vtkIntersectionPolyDataFilter") as factory:
            factory.return_value.GetStatus.return_value = 0
            with self.assertRaisesRegex(RuntimeError, "extraction failed"):
                self.export(inputs)
        self.assertEqual(output.read_bytes(), previous)

    def test_point_cloud_is_rejected_as_a_surface_input(self):
        inputs = self.save_inputs(pv.PolyData(pv.Sphere().points), pv.Sphere())
        with self.assertRaisesRegex(ValueError, "Invalid Step 07"):
            self.export(inputs)
        self.assertFalse((self.root / "subject_test_ED_septum.vtk").exists())

    def test_step7_saver_includes_curve_in_paths_and_summary(self):
        lv_epi = pv.Sphere(radius=1)
        rv_epi = pv.Sphere(center=(0.6, 0, 0), radius=1)
        paths, info = pipeline.save_step7_reconstructed_annuli(
            self.root, "subject_test", "ED", lv_epi, pv.Sphere(radius=0.5), pv.Disc(),
            rv_epi, pv.Sphere(center=(0.6, 0, 0), radius=0.5), pv.Disc(), {}, {}, self.args,
        )
        self.assertTrue(Path(paths["septum"]).is_file())
        self.assertEqual(len(info["surfaces"]), 6)
        self.assertEqual(info["RVepi_LVepi_interface_curve"]["status"], "intersection_found")
        summary = json.loads(Path(paths["summary"]).read_text())
        self.assertEqual(summary["paths"]["septum"], paths["septum"])
        self.assertEqual(summary["RVepi_LVepi_interface_curve"]["path"], paths["septum"])


if __name__ == "__main__":
    unittest.main()
