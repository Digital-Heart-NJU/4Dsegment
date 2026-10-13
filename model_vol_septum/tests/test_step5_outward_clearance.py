"""Regression checks for exterior-only movement and protected RVepi openings."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rvepi_outward_clearance import FixedEndo, _edges, face_clearance, move_rvepi, triangles
import generate_open_volmesh_stable as pipeline


def open_cube(radius):
    mesh = pv.Cube(bounds=(-radius, radius, -radius, radius, -radius, radius)).triangulate()
    keep = mesh.cell_centers().points[:, 2] < radius - 0.001
    return mesh.extract_cells(keep).extract_surface().triangulate().clean()


def ring_indices(mesh):
    faces = triangles(mesh)
    edges = np.sort(np.vstack((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacent = {}
    for a, b in edges[counts == 1]:
        adjacent.setdefault(int(a), []).append(int(b))
        adjacent.setdefault(int(b), []).append(int(a))
    order = [min(adjacent)]
    last = None
    while True:
        nxt = next(v for v in adjacent[order[-1]] if v != last)
        if nxt == order[0]:
            return np.asarray(order, dtype=np.int64)
        last = order[-1]
        order.append(nxt)


class Step5OutwardTests(unittest.TestCase):
    def test_signed_distance_uses_triangles_and_detects_inside(self):
        fixed = FixedEndo(open_cube(5))
        distance = fixed.values(np.array([[0., 0., -8.], [0., 0., 0.]]))
        np.testing.assert_allclose(distance, [3., -5.], atol=1e-5)

    def test_triangle_interior_is_checked_when_vertices_are_far_away(self):
        fixed = FixedEndo(open_cube(1))
        points = np.array([[-6., -6., 2.], [6., -6., 2.], [0., 6., 2.]])
        self.assertTrue(np.all(fixed.values(points) > 3))
        bad, _, _ = face_clearance(points, np.array([[0, 1, 2]]), fixed, 3)
        np.testing.assert_array_equal(bad, [0])

    def test_inside_epi_expands_and_preserves_opening_connectivity(self):
        epi, endo = open_cube(4), open_cube(5)
        original_endo_points = endo.points.copy()
        original_endo_faces = endo.faces.copy()
        ring = ring_indices(epi)
        result, info = move_rvepi(epi, endo, ring, 3., .05, 2., 15, .20)
        new_ring = np.asarray(info.pop("_opening_ring_ids"))
        edges = {tuple(e) for e in _edges(triangles(result))}
        for a, b in zip(new_ring, np.roll(new_ring, -1)):
            self.assertIn(tuple(sorted((a, b))), edges)
        self.assertEqual(len(new_ring), len(ring))
        self.assertEqual(result.n_open_edges, epi.n_open_edges)
        self.assertEqual(info["final_vertices_inside_RVendo"], 0)
        self.assertGreaterEqual(info["minimum_certified_surface_distance_mm"], 3.)
        self.assertEqual(info["final_RVepi_self_intersection_pairs"], 0)
        np.testing.assert_array_equal(endo.points, original_endo_points)
        np.testing.assert_array_equal(endo.faces, original_endo_faces)

    def test_active_step5_saves_triangle_distances_and_final_displacements(self):
        epi, endo = open_cube(4), open_cube(5)
        ring = ring_indices(epi)
        loop = pipeline.base.make_polyline(epi.points[ring])
        with patch.object(sys, "argv", ["pipeline", "--step6-clearance-method", "triangle"]):
            args = pipeline.parse_args()
        moved, moved_mask, opening, band, final_loop, info = (
            pipeline.enforce_rvepi_distance_adaptive_opening_ring(epi, endo, loop, args)
        )
        self.assertGreaterEqual(info["minimum_certified_surface_distance_mm"], 3.)
        np.testing.assert_array_equal(triangles(moved), triangles(epi))
        np.testing.assert_allclose(moved.point_data["step6Displacement"], moved.points - epi.points)
        self.assertTrue(np.all(moved_mask))
        self.assertEqual(final_loop.n_points, len(ring))
        with tempfile.TemporaryDirectory() as directory:
            paths, summary = pipeline.save_step6_moved_rvepi(
                Path(directory), "test", "ED", moved, endo, moved_mask, opening, band, info, args,
            )
            saved = pv.read(paths["RV_openbase_epi_moved"])
            self.assertIn("step6Displacement", saved.point_data)
            np.testing.assert_allclose(saved.point_data["step6Displacement"], saved.points - epi.points, atol=1e-5)
            exact = FixedEndo(endo).exact_distances(saved.points)
            np.testing.assert_allclose(saved.point_data["distanceToRVendo"], exact, atol=1e-5)
            self.assertEqual(np.count_nonzero(saved.point_data["belowRequested3mm"]), 0)
            self.assertGreaterEqual(summary["surface"]["saved_surface_validation"]["minimum_certified_surface_distance_mm"], 3.)


if __name__ == "__main__":
    unittest.main()
