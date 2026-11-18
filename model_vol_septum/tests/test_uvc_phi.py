"""PHI tag-interface extraction and area-based curve selection."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum.compute_UVC_fiber import _curve_enclosed_area, extract_tag_interface_curve


class TagInterfaceTests(unittest.TestCase):
    def volume(self, root, inner_radius):
        sheet = pv.Disc(inner=inner_radius, outer=2., r_res=1, c_res=32).triangulate().clean(tolerance=0.)
        triangles = sheet.faces.reshape(-1, 4)[:, 1:]
        points = np.vstack((sheet.points, [[0., 0., 1.], [0., 0., -1.]]))
        tetrahedra = np.vstack((
            np.column_stack((triangles, np.full(len(triangles), sheet.n_points))),
            np.column_stack((triangles, np.full(len(triangles), sheet.n_points + 1))),
        ))
        volume = pv.UnstructuredGrid(
            np.column_stack((np.full(len(tetrahedra), 4), tetrahedra)).ravel(),
            np.full(len(tetrahedra), pv.CellType.TETRA), points,
        )
        volume.cell_data["elemTag"] = np.repeat([0, 1], len(triangles))
        path = root / "final_volmesh.vtk"
        volume.save(path)
        return path, volume

    def test_two_curves_select_larger_enclosed_area_and_keep_mesh_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, source = self.volume(root, .75)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            selected_path, info = extract_tag_interface_curve(path, root / "work")
            self.assertEqual(info["curve_count"], 2)
            self.assertEqual(info["selection"], "largest_enclosed_area")
            areas = [candidate["enclosed_area_mm2"] for candidate in info["candidates"]]
            self.assertAlmostEqual(info["selected_enclosed_area_mm2"], max(areas))
            self.assertGreater(max(areas), 5 * min(areas))
            curve = pv.read(selected_path)
            ids = np.asarray(curve.point_data["volumeNodeId"], dtype=int)
            np.testing.assert_array_equal(curve.points, source.points[ids])
            self.assertTrue(np.all(np.linalg.norm(curve.points[:, :2], axis=1) > 1.9))
            interface = pv.read(info["interface_surface"])
            self.assertTrue(np.all(source.cell_data["elemTag"][interface["tag0CellId"]] == 0))
            self.assertTrue(np.all(source.cell_data["elemTag"][interface["tag1CellId"]] == 1))
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_single_curve_is_used_directly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.volume(root, 0.)
            selected, info = extract_tag_interface_curve(path, root / "work")
            self.assertEqual(info["curve_count"], 1)
            self.assertEqual(info["selection"], "only_interface_curve")
            self.assertTrue(Path(selected).is_file())

    def test_four_or_more_curves_select_largest_and_preserve_volume(self):
        for component_count in (2, 3):
            with self.subTest(curve_count=2 * component_count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, template = self.volume(root, .75)
                source = template.copy()
                for index in range(1, component_count):
                    component = template.copy()
                    component.points = component.points * (index + 1) + [20. * index, 0., 0.]
                    source = source.merge(component, merge_points=False)
                source.save(path)
                before = hashlib.sha256(path.read_bytes()).hexdigest()

                selected, info = extract_tag_interface_curve(path, root / "work")

                self.assertEqual(info["curve_count"], 2 * component_count)
                self.assertEqual(info["selection"], "largest_enclosed_area")
                areas = [candidate["enclosed_area_mm2"] for candidate in info["candidates"]]
                self.assertAlmostEqual(info["selected_enclosed_area_mm2"], max(areas))
                curve = pv.read(selected)
                ids = np.asarray(curve.point_data["volumeNodeId"], dtype=int)
                np.testing.assert_array_equal(curve.points, source.points[ids])
                radii = np.linalg.norm(curve.points[:, :2] - [20. * (component_count - 1), 0.], axis=1)
                np.testing.assert_allclose(radii, 2. * component_count, rtol=1e-6)
                self.assertTrue(all(Path(candidate["path"]).is_file() for candidate in info["candidates"]))
                saved = json.loads((root / "work/tag_interface_curve_summary.json").read_text())
                self.assertEqual(saved["selection"], "largest_enclosed_area")
                self.assertEqual(saved["curve_count"], 2 * component_count)
                self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_vertex_contact_is_not_a_tag_interface(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            grid = pv.UnstructuredGrid([4, 0, 1, 2, 3, 4, 0, 4, 5, 6],
                                       np.full(2, pv.CellType.TETRA), points)
            grid.cell_data["elemTag"] = [0, 1]
            path = root / "mesh.vtk"
            grid.save(path)
            with self.assertRaisesRegex(RuntimeError, "complete triangular faces"):
                extract_tag_interface_curve(path, root / "work")


class CurveAreaTests(unittest.TestCase):
    def test_area_is_rotation_invariant_and_not_perimeter(self):
        square = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]])
        narrow = np.array([[0., 0., 0.], [10., 0., 0.], [10., .1, 0.], [0., .1, 0.]])
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        areas = [_curve_enclosed_area(pv.lines_from_points(points, close=True))["enclosed_area_mm2"]
                 for points in (square, narrow, square @ rotation + [10., -3., 8.])]
        np.testing.assert_allclose(areas, [4., 1., 4.], atol=1e-12)

    def test_open_curve_is_not_closed_artificially(self):
        curve = pv.lines_from_points(np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]))
        with self.assertRaisesRegex(RuntimeError, "open boundary"):
            _curve_enclosed_area(curve)


if __name__ == "__main__":
    unittest.main()
