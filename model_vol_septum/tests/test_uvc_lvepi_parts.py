"""Extract the PHO junction using complete shared volume faces."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum.pho_boundaries import _tag_interface_faces


class LVepiInterfaceTests(unittest.TestCase):
    def test_only_complete_shared_triangles_are_interface_faces(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [0., 0., -1.], [0., 0., 1.], [-1., 0., 0.],
                           [0., -1., 0.], [0., 0., 2.]])
        cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 5, 6, 7]])
        packed = np.column_stack((np.full(len(cells), 4), cells))
        volume = pv.UnstructuredGrid(packed.ravel(), np.full(3, pv.CellType.TETRA), points)
        volume.cell_data["elemTag"] = np.array([1, 0, 0])
        before_points, before_cells = volume.points.copy(), volume.cells.copy()
        faces, tag1, tag0 = _tag_interface_faces(volume)
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
        np.testing.assert_array_equal(tag1, [0])
        np.testing.assert_array_equal(tag0, [1])
        np.testing.assert_array_equal(volume.points, before_points)
        np.testing.assert_array_equal(volume.cells, before_cells)

    def test_missing_tag1_is_rejected(self):
        volume = pv.Cube().triangulate().delaunay_3d()
        volume.cell_data["elemTag"] = np.zeros(volume.n_cells, dtype=np.int32)
        with self.assertRaisesRegex(RuntimeError, "requires tag 1"):
            _tag_interface_faces(volume)


if __name__ == "__main__":
    unittest.main()
