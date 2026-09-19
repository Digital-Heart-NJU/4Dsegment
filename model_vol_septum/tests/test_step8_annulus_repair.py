"""Focused regression tests for direct Step 8 annulus boundary repair."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_open_volmesh_stable as mesh


def triangle_surface(points, faces=None):
    points = np.asarray(points, dtype=float)
    if faces is None:
        faces = np.arange(len(points)).reshape(-1, 3)
    return pv.PolyData(points, mesh.base.faces_from_triangles(np.asarray(faces)))


def ring(radius, shift=0.0):
    return np.array([
        [-radius + shift, -radius, 0.0], [radius + shift, -radius, 0.0],
        [radius + shift, radius, 0.0], [-radius + shift, radius, 0.0],
    ])


def wall(radius, shift=0.0):
    top = ring(radius, shift)
    points = np.vstack((top, top - [0.0, 0.0, 2.0]))
    faces = []
    for first in range(4):
        second = (first + 1) % 4
        faces.extend(([first, first + 4, second + 4], [first, second + 4, second]))
    return triangle_surface(points, faces)


def direct_annulus(shift=0.0):
    return mesh.base.zipper_annulus(
        mesh.base.make_polyline(ring(3.0, shift)),
        mesh.base.make_polyline(ring(1.0, shift)),
    )


class Step8AnnulusTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            surface_clean_tolerance=1e-7,
            step8_boundary_repair_rounds=12,
            step8_boundary_max_displacement_edge_fraction=0.5,
            step7_interface_ring_distance_tolerance=1e-5,
            rv_min_distance_to_rvepi=3.0,
        )
        self.profile = SimpleNamespace(base_side="high_z")

    def test_shared_edge_contact_is_allowed(self):
        annulus = triangle_surface([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        other = triangle_surface([[0, 0, 0], [1, 0, 0], [0, -1, 1]])
        self.assertFalse(any(mesh.step8_annulus_collisions(annulus, {"epi": other}).values()))

    def test_nonmanifold_overlap_at_shared_edge_is_rejected(self):
        annulus = triangle_surface([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        other = triangle_surface([[0, 0, 0], [1, 0, 0], [0, 0.5, 0]])
        self.assertEqual(len(mesh.step8_annulus_collisions(annulus, {"epi": other})["epi"]), 1)

    def test_apply_moves_mapped_epi_and_endo_boundary_points_only(self):
        epi, endo, annulus = wall(3), wall(1), direct_annulus()
        kind, source_id = mesh.step8_direct_boundary_map(annulus, epi, endo, 1e-7)
        epi_annulus_id = int(np.flatnonzero(kind == 0)[0])
        endo_annulus_id = int(np.flatnonzero(kind == 1)[0])
        epi_snapshot = epi.copy(deep=True)
        endo_snapshot = endo.copy(deep=True)
        trial_epi, trial_endo, trial_annulus = mesh.step8_apply_boundary_displacements(
            annulus, epi, endo, kind, source_id,
            {epi_annulus_id: np.array([0.0, 0.0, 0.2]), endo_annulus_id: np.array([0.0, 0.0, 0.1])},
        )
        self.assertAlmostEqual(trial_epi.points[source_id[epi_annulus_id], 2], 0.2)
        np.testing.assert_array_equal(trial_epi.faces, epi_snapshot.faces)
        self.assertAlmostEqual(trial_endo.points[source_id[endo_annulus_id], 2], 0.1)
        np.testing.assert_array_equal(trial_endo.faces, endo_snapshot.faces)
        self.assertAlmostEqual(trial_annulus.points[epi_annulus_id, 2], 0.2)
        self.assertAlmostEqual(trial_annulus.points[endo_annulus_id, 2], 0.1)

        epi_interior = sorted(set(range(epi.n_points)) - set(source_id[kind == 0]))
        endo_interior = sorted(set(range(endo.n_points)) - set(source_id[kind == 1]))
        np.testing.assert_array_equal(trial_epi.points[epi_interior], epi_snapshot.points[epi_interior])
        np.testing.assert_array_equal(trial_endo.points[endo_interior], endo_snapshot.points[endo_interior])

    def test_epi_crossing_produces_movable_epi_constraint(self):
        epi, endo, annulus = wall(3), wall(1), direct_annulus()
        kind, _ = mesh.step8_direct_boundary_map(annulus, epi, endo, 1e-7)
        epi_ids = np.flatnonzero(kind == 0)
        crossing = {
            "surface": "epi", "surface_face_id": 7,
            "fixed_boundary_edge": annulus.points[epi_ids[:2]].tolist(),
            "surface_triangle": [[-1, -3, -1], [1, -3, 1], [0, -2, 0]],
        }
        constraints, blocked = mesh.step8_collision_constraints(
            annulus, {"epi": epi, "endo": endo}, [crossing], {}, kind,
        )
        self.assertFalse(blocked)
        self.assertTrue(np.all(kind[constraints[0]["annulus_vertex_ids"]] == 0))

    def test_endo_crossing_produces_endo_only_constraint(self):
        epi, endo, annulus = wall(3), wall(1), direct_annulus()
        kind, _ = mesh.step8_direct_boundary_map(annulus, epi, endo, 1e-7)
        endo_ids = np.flatnonzero(kind == 1)
        crossing = {
            "surface": "epi", "surface_face_id": 3,
            "fixed_boundary_edge": annulus.points[endo_ids[:2]].tolist(),
            "surface_triangle": [[-1, -1, -1], [1, -1, 1], [0, 0, 0]],
        }
        constraints, blocked = mesh.step8_collision_constraints(
            annulus, {"epi": epi, "endo": endo}, [crossing], {}, kind,
        )
        self.assertFalse(blocked)
        self.assertTrue(np.all(kind[constraints[0]["annulus_vertex_ids"]] == 1))

    def test_valid_direct_annulus_and_surfaces_are_unchanged(self):
        epi, endo, annulus = wall(3), wall(1), direct_annulus()
        epi_snapshot, endo_snapshot, annulus_snapshot = [part.copy(deep=True) for part in (epi, endo, annulus)]
        out_epi, out_endo, out_annulus, info = mesh.repair_step8_direct_annulus_boundaries(
            "LV", epi, endo, annulus, {}, self.profile, self.args,
        )
        np.testing.assert_array_equal(out_epi.points, epi_snapshot.points)
        np.testing.assert_array_equal(out_endo.points, endo_snapshot.points)
        np.testing.assert_array_equal(out_annulus.points, annulus_snapshot.points)
        self.assertEqual(info["moved_epi_boundary_point_count"], 0)
        self.assertEqual(info["moved_endo_boundary_point_count"], 0)
        self.assertFalse(info["epi_interior_points_moved"])

    def test_degenerate_faces_are_rejected(self):
        original = direct_annulus()
        broken = original.copy(deep=True)
        broken.points[1] = broken.points[0]
        with self.assertRaises(RuntimeError):
            mesh.step8_validate_annulus_topology(broken, original)

    def test_failed_serialization_validation_preserves_previous_outputs(self):
        surface = direct_annulus()
        surfaces = {name: surface for name in (
            "LVepi", "LVendo", "RVepi", "RVendo", "LV_openbase_annulus", "RV_openbase_annulus",
        )}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_ED_LVepi_openbase_surf.vtk"
            path.write_text("previous accepted output")
            with patch.object(mesh, "validate_step8_saved_outputs", side_effect=RuntimeError("reject")):
                with self.assertRaises(RuntimeError):
                    mesh.save_step8_openbase_surface_outputs(Path(directory), "test", "ED", surfaces, {})
            self.assertEqual(path.read_text(), "previous accepted output")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_valid_surfaces_are_saved_with_chamber_tags(self):
        surfaces = {}
        for chamber, shift, tag in (("LV", 0.0, 1), ("RV", 10.0, 0)):
            surfaces[chamber + "epi"] = wall(3, shift)
            surfaces[chamber + "endo"] = wall(1, shift)
            surfaces[chamber + "_openbase_annulus"] = direct_annulus(shift)
            for key in (chamber + "epi", chamber + "endo", chamber + "_openbase_annulus"):
                surfaces[key].cell_data["elemTag"] = np.full(surfaces[key].n_cells, tag)
        with tempfile.TemporaryDirectory() as directory:
            paths, summary = mesh.save_step8_openbase_surface_outputs(Path(directory), "test", "ED", surfaces, {})
            self.assertTrue(summary["serialized_geometry_and_collisions_validated"])
            for key, path in paths.items():
                if key == "stage08_summary":
                    continue
                saved = pv.read(path)
                self.assertTrue(np.all(saved.celltypes == 5))
                self.assertTrue(np.all(saved.cell_data["elemTag"] == (1 if key.startswith("LV") else 0)))


if __name__ == "__main__":
    unittest.main()
