"""Analytical geometry checks for the mesh Dice utility (standard-library unittest)."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

SPEC = importlib.util.spec_from_file_location(
    "mesh_dice", Path(__file__).resolve().parents[1] / "volmesh_dice_compare" / "calculate_volmesh_dice.py"
)
dice = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dice)


def box(lengths=(2, 2, 2), origin=(0, 0, 0), tag=0, dimensions=(2, 2, 2)):
    spacing = np.asarray(lengths) / (np.asarray(dimensions) - 1)
    mesh = pv.ImageData(dimensions=dimensions, spacing=spacing, origin=origin).cast_to_unstructured_grid().triangulate()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, tag, dtype=np.int64)
    return mesh


class TestVolumeDice(unittest.TestCase):
    def sample_pair(self, source, reference, spacing=0.5):
        low, shape = dice.grid_definition(source, reference, spacing)
        a = dice.sample_volume(dice.encode_volume(source, {0: 1}), low, shape, spacing)
        b = dice.sample_volume(dice.encode_volume(reference, {0: 1}), low, shape, spacing)
        return a, b, low, shape

    def test_zero_tag_is_tissue_and_shifted_boxes_have_half_dice(self):
        a, b, _, _ = self.sample_pair(box(), box(origin=(1, 0, 0)))
        rows = dice.dice_rows(a, b, {0: 1}, 0.5, "test")
        self.assertEqual(rows[0]["dice"], 0.5)
        self.assertEqual(rows[1]["dice"], 0.5)
        self.assertEqual(rows[0]["source_volume_mm3"], 8)

    def test_tetrahedral_interior_not_bounding_box_or_nearest_vertex(self):
        mesh = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]), np.array([pv.CellType.TETRA]),
                                   np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]))
        mesh["elemTag"] = np.array([0])
        sampled = dice.sample_volume(dice.encode_volume(mesh, {0: 1}), np.zeros(3), np.array([4, 4, 4]), 0.5)
        self.assertEqual(sampled[0], 1)  # (.25,.25,.25), inside the tetrahedron
        self.assertEqual(sampled[-1], 0)  # (1.75,1.75,1.75), inside bbox but outside tetrahedron

    def test_shared_basal_clip_removes_extension_and_conserves_volume(self):
        source, reference = box((2, 2, 3)), box()
        plane = dice.normalized_plane([0, 0, 1, 2])
        kept, removed = dice.clip_volume(source, [plane])
        self.assertAlmostEqual(dice.volume_stats(kept)["volume_mm3"], 8)
        self.assertAlmostEqual(dice.volume_stats(removed)["volume_mm3"], 4)
        a, b, low, shape = self.sample_pair(source, reference)
        self.assertAlmostEqual(dice.dice_rows(a, b, {0: 1}, 0.5, "full")[0]["dice"], 0.8)
        region = dice.region_mask(low, shape, 0.5, [plane])
        a[~region], b[~region] = 0, 0
        self.assertEqual(dice.dice_rows(a, b, {0: 1}, 0.5, "trimmed")[0]["dice"], 1)

    def test_lateral_shape_disagreement_remains_after_basal_clip(self):
        source, reference = box((3, 2, 3)), box()
        a, b, low, shape = self.sample_pair(source, reference)
        region = dice.region_mask(low, shape, 0.5, [dice.normalized_plane([0, 0, 1, 2])])
        a[~region], b[~region] = 0, 0
        self.assertAlmostEqual(dice.dice_rows(a, b, {0: 1}, 0.5, "trimmed")[0]["dice"], 0.8)

    def test_empty_regions_have_no_artificial_perfect_score(self):
        rows = dice.dice_rows(np.zeros(10), np.zeros(10), {0: 1}, 1., "test")
        self.assertIsNone(rows[0]["dice"])

    def test_plane_normalization_and_direction(self):
        plane = dice.normalized_plane([0, 0, 2, 4])
        self.assertEqual(plane["normal"], [0, 0, 1])
        self.assertEqual(plane["offset_mm"], 2)
        with self.assertRaises(ValueError):
            dice.normalized_plane([0, 0, 0, 1])

    def test_ambiguous_flat_surfaces_require_explicit_anatomical_plane(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            dice.detect_basal_planes(box(dimensions=(8, 8, 8)))

    def test_auto_detection_recovers_known_basal_plane(self):
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ring = np.column_stack([2 * np.cos(theta), 2 * np.sin(theta), np.zeros(len(theta))])
        points = np.vstack([[0, 0, 0], ring, [0, 0, -3]])
        cells = np.array([[4, 0, i + 1, (i + 1) % len(ring) + 1, len(points) - 1]
                          for i in range(len(ring))]).ravel()
        cone = pv.UnstructuredGrid(cells, np.full(len(ring), pv.CellType.TETRA), points)
        cone["elemTag"] = np.zeros(cone.n_cells, dtype=np.int64)
        planes = dice.detect_basal_planes(cone)
        self.assertEqual(len(planes), 1)
        np.testing.assert_allclose(planes[0]["normal"], [0, 0, 1], atol=1e-6)
        self.assertAlmostEqual(planes[0]["offset_mm"], 0, places=6)

    def test_rigid_registration_recovers_known_transform_with_extra_outliers(self):
        rng = np.random.default_rng(2)
        directions = rng.normal(size=(1800, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        target = directions * [20, 13, 8]
        normals = directions / [20, 13, 8]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        labels = (target[:, 0] > 0).astype(int)
        forward = np.eye(4)
        forward[:3, :3] = Rotation.from_rotvec([.23, -.12, .1]).as_matrix()
        forward[:3, 3] = [12, -5, 3]
        moved = dice.transform_points(target, forward)
        moved_normals = normals @ forward[:3, :3].T
        extra = rng.normal(size=(100, 3)) + [45, 45, 45]
        source_data = (np.vstack([moved, extra]), np.vstack([moved_normals, np.tile([0, 0, 1], (100, 1))]),
                       np.concatenate([labels, np.zeros(100, dtype=int)]))
        initial = np.eye(4)
        initial[:3, 3] = -forward[:3, 3]
        recovered = dice.icp(source_data, (target, normals, labels), initial, 120, .85)
        self.assertLess(np.max(np.linalg.norm(dice.transform_points(moved, recovered) - target, axis=1)), .01)
        self.assertAlmostEqual(np.linalg.det(recovered[:3, :3]), 1, places=8)

    def test_scaling_and_reflection_are_rejected(self):
        for value in [1.1, -1]:
            matrix = np.eye(4)
            matrix[0, 0] = value
            with self.assertRaises(ValueError):
                dice.validate_rigid(matrix)


class TestLandmarkRegistration(unittest.TestCase):
    def test_valve_centroid_excludes_unused_global_points(self):
        points = np.array([[0., 0, 0], [4, 0, 0], [4, 1, 0], [0, 1, 0], [900, 800, 700]])
        valve = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 0, 2, 3]))
        center, info = dice.valve_centroid(valve)
        np.testing.assert_allclose(center, [2, .5, 0])
        self.assertEqual(info["referenced_vertices"], 4)
        self.assertEqual(info["stored_points"], 5)

    def test_rim_centroid_is_weighted_by_length_and_excludes_unused_points(self):
        points = np.array([[0., 0, 0], [4, 0, 0], [4, 1, 0], [900, 800, 700]])
        rim = pv.PolyData(points, lines=np.array([3, 0, 1, 2]))
        center, info = dice.rim_centroid(rim)
        np.testing.assert_allclose(center, [2.4, .1, 0])
        self.assertEqual(info["referenced_vertices"], 3)
        self.assertEqual(info["length_mm"], 5)

    def test_apex_uses_only_endo_vertices_and_actual_valve_triangles(self):
        valve = pv.PolyData(np.array([[-10., -10, 0], [10, -10, 0], [0, 10, 0], [999, 999, 999]]),
                            np.array([3, 0, 1, 2]))
        endo = pv.PolyData(np.array([[0., 0, 5], [0, 0, 2], [1, 0, 2], [1000, 1000, 1000]]),
                           np.array([3, 0, 1, 2]))
        apex, info = dice.endo_apex(endo, valve)
        np.testing.assert_allclose(apex, [0, 0, 5])
        self.assertAlmostEqual(info["distance_to_valve_surface_mm"], 5)
        np.testing.assert_allclose(info["closest_valve_surface_point_mm"], [0, 0, 0], atol=1e-12)
        self.assertEqual(info["original_point_id"], 0)

    def test_anchored_fit_recovers_known_rigid_transform(self):
        moving = np.array([[2., -3, 4], [3, 2, 10], [-3, 4, 7]])
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_rotvec([.4, -.7, .2]).as_matrix()
        expected[:3, 3] = [17, -42, 31]
        fixed = dice.transform_points(moving, expected)
        actual, report = dice.register_anchored_landmarks(moving, fixed)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertLess(report["errors_after_mm"]["lv_endo_apex"], 1e-12)
        self.assertFalse(report["icp_refinement"])

    def test_mismatched_valves_never_move_apex_or_scale_the_mesh(self):
        moving = np.array([[0., 0, 0], [4, 0, 0], [0, 3, 0]])
        fixed = np.array([[100., 200, 300], [104, 200, 300], [100, 206, 300]])
        matrix, report = dice.register_anchored_landmarks(moving, fixed)
        np.testing.assert_allclose(dice.transform_points(moving[:1], matrix)[0], fixed[0], atol=1e-12)
        np.testing.assert_allclose(matrix[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(report["errors_after_mm"]["rv_tricuspid_point"], 3)
        self.assertAlmostEqual(report["valve_squared_error_sum_mm2"], 9)
        mesh = box()
        transformed = mesh.copy()
        transformed.points = dice.transform_points(mesh.points, matrix)
        self.assertAlmostEqual(dice.volume_stats(mesh)["volume_mm3"], dice.volume_stats(transformed)["volume_mm3"])
        np.testing.assert_array_equal(mesh.cells, transformed.cells)

    def test_collinear_landmarks_are_rejected(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaisesRegex(ValueError, "non-collinear"):
            dice.register_anchored_landmarks(points, points)

    def test_landmark_association_keeps_geometric_centroid_definition(self):
        mesh = box()
        mesh.cell_data["original_cell_id"] = np.arange(mesh.n_cells)
        points = np.array([[1., 1, 1], [0, 0, 0], [2, 2, 2]])
        association = dice.associate_landmarks_with_volume(points, mesh, Path("volume.vtk"))
        self.assertFalse(association["coordinates_snapped"])
        np.testing.assert_allclose(association["landmarks"][0]["coordinate_mm"], [1, 1, 1])
        self.assertAlmostEqual(association["landmarks"][0]["distance_to_surface_mm"], 1)


if __name__ == "__main__":
    unittest.main()
