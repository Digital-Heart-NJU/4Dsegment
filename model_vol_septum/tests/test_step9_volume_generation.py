"""Step 09 volume output, boundary preservation, and failure handling."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import manifold3d
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_open_volmesh_stable as pipeline
from model_vol_septum import generate_step09_volume as volume


class Step9VolumeTests(unittest.TestCase):
    def test_cavity_boundary_and_untagged_saved_volume(self):
        outer = manifold3d.Manifold.cube((4, 4, 4), center=True)
        opening = manifold3d.Manifold.cube((2, 2, 4), center=True).translate((0, 0, 1))
        cup = (outer - opening).to_mesh64()
        faces = np.column_stack((np.full(len(cup.tri_verts), 3), np.asarray(cup.tri_verts, dtype=np.int64))).ravel()
        surface = pv.PolyData(np.asarray(cup.vert_properties)[:, :3], faces)
        surface.cell_data["elemTag"] = np.ones(surface.n_cells, dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "subject_test_ED_LVRV_surf.vtk"
            surface.save(source)
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(sys, "argv", ["pipeline"]):
                args = pipeline.parse_args()
            cleaned, cleaning = pipeline.save_step8_cleaned_lvrv_surface(
                root / "step_08_LVRV_surf_clean", source, "subject_test", "ED")
            paths, summary = pipeline.save_step9_volume_mesh(
                root / "step_09_volmesh_generate", Path(cleaned["LVRV_surf"]), "subject_test", "ED", args)
            self.assertEqual(summary["input_path"], cleaned["LVRV_surf"])
            self.assertEqual(summary["input_stage"], "step_08_LVRV_surf_clean")
            self.assertTrue(cleaning["step07_input_unchanged"])
            self.assertEqual(Path(paths["volmesh_notag"]).name, "subject_test_ED_volmesh_notag.vtk")
            self.assertTrue(Path(paths["summary"]).is_file())
            saved = pv.read(paths["volmesh_notag"])
            self.assertTrue(np.all(saved.celltypes == pv.CellType.TETRA))
            self.assertEqual(list(saved.cell_data), [])
            self.assertEqual(list(saved.point_data), [])
            result = summary["saved_volume_validation"]
            self.assertEqual(result["nonpositive_tetrahedra"], 0)
            self.assertEqual(result["component_count"], 1)
            self.assertTrue(result["boundary_matches_step08"])
            self.assertAlmostEqual(result["total_tetrahedron_volume_mm3"], 52., places=9)
            centers = saved.cell_centers().points
            cavity = (np.abs(centers[:, 0]) < 1) & (np.abs(centers[:, 1]) < 1) & (centers[:, 2] > -1)
            self.assertFalse(np.any(cavity))
            self.assertEqual(checksum, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_open_input_does_not_run_tetgen_or_overwrite_previous_output(self):
        surface = pv.Cube().triangulate()
        surface = surface.extract_cells(np.arange(surface.n_cells - 1)).extract_surface()
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "open.vtk", Path(directory) / "previous.vtk"
            surface.save(source)
            destination.write_bytes(b"previous successful output")
            with patch.object(volume.tetgen, "TetGen") as mesher:
                with self.assertRaisesRegex(RuntimeError, "free_edges"):
                    volume.generate_volume(source, destination)
            mesher.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"previous successful output")

    def test_reversed_tetrahedron_is_reoriented_before_saving(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        cells = np.array([[1, 0, 2, 3]])
        grid = pv.UnstructuredGrid(np.r_[4, cells.ravel()], np.array([pv.CellType.TETRA]), points)
        source_mesh = grid.extract_surface().compute_normals(consistent_normals=True, auto_orient_normals=True)
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "input.vtk", Path(directory) / "output.vtk"
            source_mesh.save(source)
            with patch.object(volume.tetgen, "TetGen") as mesher:
                mesher.return_value.tetrahedralize.return_value = (points, cells)
                summary = volume.generate_volume(source, destination)
            self.assertEqual(summary["negative_tetrahedra_reoriented"], 1)
            self.assertGreater(summary["saved_volume_validation"]["minimum_tetrahedron_volume_mm3"], 0)


if __name__ == "__main__":
    unittest.main()
