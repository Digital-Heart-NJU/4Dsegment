"""Direct pipeline input resolution and PHO surface classification."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_UVC_fiber as driver
from model_vol_septum.pho_boundaries import SURFACE_CLASSES, build_pho_boundaries, load_surface


class PipelineInputTests(unittest.TestCase):
    def test_default_and_explicit_copied_folder_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary) / "subject_11"
            with patch.object(driver, "INPUT_PREFIX", None):
                paths = driver.pipeline_frame_paths(subject, "ED")
                self.assertEqual(paths["volume"].name, "subject_11_ED_volmesh.vtk")
                self.assertEqual(paths["RVepi"].name, "subject_11_ED_RV_openbase_epi_moved.vtk")
                self.assertNotIn("RVseptum", paths)
                paths["volume"].parent.mkdir(parents=True)
                paths["volume"].touch()
                self.assertEqual(driver.frame_names(subject), ["ED"])
            with patch.object(driver, "INPUT_PREFIX", "source_subject"):
                self.assertEqual(driver.pipeline_frame_paths(subject, "ES")["volume"].name,
                                 "source_subject_ES_volmesh.vtk")

    def test_no_silent_fallback_to_another_subject_prefix(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(driver, "INPUT_PREFIX", None):
            subject = Path(temporary) / "subject_111"
            paths = driver.pipeline_frame_paths(subject, "ED")
            paths["volume"].parent.mkdir(parents=True)
            (paths["volume"].parent / "subject_11_ED_volmesh.vtk").touch()
            with self.assertRaises(RuntimeError):
                driver.frame_names(subject)

    def test_full_frame_dispatch_preserves_dependency_order_and_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "subject_11/UVC_fiber_Outputs/ED"
            output.mkdir(parents=True)
            calls = []
            from contextlib import ExitStack
            initial = Path.cwd()
            with ExitStack() as stack:
                stack.enter_context(patch.object(driver, "require_frame_paths"))
                for name in ("coords_z", "landmarks", "phi", "pho", "biv", "fiber"):
                    stack.enter_context(patch.object(
                        driver, "run_frame_" + name,
                        side_effect=lambda *a, stage=name, **kw: calls.append(stage),
                    ))
                driver.run_frame_all("subject_11", "ED", temporary)
            self.assertEqual(calls, ["coords_z", "landmarks", "phi", "pho", "biv", "fiber"])
            self.assertEqual(initial, Path.cwd())


class PHOBoundaryTests(unittest.TestCase):
    def fixture(self, root):
        shell = pv.Cube().triangulate()
        triangles = shell.points[shell.faces.reshape(-1, 4)[:, 1:]]
        planes = ((0, .5), (1, .5), (0, -.5), (1, -.5), (2, .5), (2, -.5))
        paths = {}
        for name, (axis, value) in zip(SURFACE_CLASSES, planes):
            mask = np.all(np.isclose(triangles[:, :, axis], value), axis=1)
            part = shell.extract_cells(np.flatnonzero(mask)).extract_surface()
            paths[name] = root / (name + ".vtk")
            part.save(paths[name], binary=False)
        paths["epi"] = root / "whole_shell.vtk"
        shell.save(paths["epi"], binary=False)
        return shell.delaunay_3d(), paths

    def test_classification_disjoint_conditions_shared_rim_and_immutable_volume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume, paths = self.fixture(root)
            points, cells = volume.points.copy(), volume.cells.copy()
            zero, one, details = build_pho_boundaries(volume, volume.points, paths, root / "work", 1.5)
            self.assertFalse(len(np.intersect1d(zero, one)))
            self.assertGreater(len(zero), 0)
            self.assertGreater(len(one), 0)
            self.assertGreater(details["shared_rim_node_count"], 0)
            self.assertEqual(details["unresolved_faces"], 0)
            self.assertTrue(all(count == 2 for count in details["face_counts"].values()))
            np.testing.assert_array_equal(volume.points, points)
            np.testing.assert_array_equal(volume.cells, cells)
            classified = pv.read(root / "work/combined_shell_classified.vtk")
            self.assertEqual(set(classified["surfaceClass"]), set(range(1, 7)))
            self.assertEqual(set(classified["PHO_BC"]), {0, 1})

    def test_unmatched_shell_stops_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume, paths = self.fixture(root)
            shell = pv.read(paths["epi"])
            shell.translate((10, 0, 0)).save(paths["epi"])
            with self.assertRaisesRegex(RuntimeError, "unresolved faces"):
                build_pho_boundaries(volume, volume.points, paths, root / "work", 1.5)
            info = json.loads((root / "work/boundary_classification_summary.json").read_text())
            self.assertGreater(info["unresolved_faces"], 0)

    def test_curve_cannot_replace_a_triangle_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "septum.vtk"
            pv.Line().save(path)
            with self.assertRaisesRegex(ValueError, "not a curve/cloud"):
                load_surface(path)


if __name__ == "__main__":
    unittest.main()
