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


if __name__ == "__main__":
    unittest.main()
