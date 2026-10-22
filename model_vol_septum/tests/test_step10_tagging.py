"""Step 10 classification rules and preservation of the Step 09 mesh."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_open_volmesh_stable as pipeline
from model_vol_septum.generate_step09_volume import signed_volumes
from model_vol_septum.tag_step10_volume import cleanup_tag_islands, save_tagged_volume, shared_face_neighbors


def mesh_from_tetrahedra(blocks):
    points = np.asarray(blocks, dtype=float).reshape(-1, 3)
    cells = np.arange(len(points)).reshape(-1, 4)
    negative = signed_volumes(points, cells) < 0
    cells[negative, :2] = cells[negative, 1::-1]
    return pv.UnstructuredGrid(np.column_stack((np.full(len(cells), 4), cells)).ravel(),
                               np.full(len(cells), pv.CellType.TETRA), points)


INSIDE = np.array([[0., 0., 0.], [.2, 0., 0.], [0., .2, 0.], [0., 0., .2]])


def tetrahedron_chain(count):
    # Consecutive tetrahedra share one complete face; others at most an edge.
    x = np.arange(count + 3, dtype=float) / (count + 3)
    points = np.column_stack((x, x ** 2, x ** 3))
    cells = np.arange(count)[:, None] + np.arange(4)
    return points, cells


class TagIslandCleanupTests(unittest.TestCase):
    def test_sequential_nested_islands_converge_without_swapping(self):
        points, cells = tetrahedron_chain(7)
        tags = np.array([1, 1, 0, 1, 0, 0, 0])
        result, info = cleanup_tag_islands(points, cells, tags)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(tags, [1, 1, 0, 1, 0, 0, 0])
        self.assertEqual(info["relabeled_components"], 1)
        self.assertGreater(info["changes"][0]["supporting_face_area_mm2"], 0)
        again, repeated = cleanup_tag_islands(points, cells, result)
        np.testing.assert_array_equal(again, result)
        self.assertEqual(repeated["relabeled_cells"], 0)

    def test_large_groups_and_disabled_cleanup_preserved(self):
        points, cells = tetrahedron_chain(7)
        tags = np.array([1, 1, 0, 1, 0, 0, 0])
        for options in ({"min_cells": 1, "min_fraction": 0}, {"enabled": False}):
            with self.subTest(options=options):
                result, info = cleanup_tag_islands(points, cells, tags, **options)
                np.testing.assert_array_equal(result, tags)
                self.assertEqual(len(info["unresolved_components"]), 2)

    def test_vertex_and_edge_contacts_do_not_join_components(self):
        points, cells = tetrahedron_chain(7)
        a, b, _ = shared_face_neighbors(cells)
        self.assertEqual({tuple(sorted(pair)) for pair in zip(a, b)}, {(i, i + 1) for i in range(6)})
        disconnected = np.array([[0, 1, 2, 3], [1, 4, 5, 6], [5, 6, 7, 8]])
        a, b, _ = shared_face_neighbors(disconnected)
        self.assertEqual(len(a), 0)
        result, info = cleanup_tag_islands(points, disconnected, np.array([1, 1, 0]))
        np.testing.assert_array_equal(result, [1, 1, 0])
        self.assertEqual(info["unresolved_components"][0]["reason"],
                         "no_shared_face_with_opposite_main_region")

    def test_cleanup_saved_flags_and_initial_comparison(self):
        points, cells = tetrahedron_chain(7)
        negative = signed_volumes(points, cells) < 0
        cells[negative, :2] = cells[negative, 1::-1]
        source = pv.UnstructuredGrid(np.column_stack((np.full(len(cells), 4), cells)).ravel(),
                                     np.full(len(cells), pv.CellType.TETRA), points)
        tags = np.array([1, 1, 0, 1, 0, 0, 0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, reference, final = root / "input.vtk", root / "LV.vtk", root / "final.vtk"
            source.save(source_path)
            pv.Cube().triangulate().save(reference)
            info = save_tagged_volume(source_path, reference, final, lambda *args: (tags, {}))
            saved, before = pv.read(final), pv.read(info["before_cleanup_path"])
            np.testing.assert_array_equal(before["elemTag"], tags)
            np.testing.assert_array_equal(saved["tagChanged"], [0, 0, 1, 0, 0, 0, 0])
            for mesh in (saved, before):
                np.testing.assert_array_equal(mesh.points, source.points)
                np.testing.assert_array_equal(mesh.cells, source.cells)
            self.assertEqual(info["regions"]["LV"]["component_count"], 1)
            self.assertEqual(info["regions"]["RV"]["component_count"], 1)

    def test_repeated_merges_and_fraction_threshold(self):
        points, cells = tetrahedron_chain(11)
        tags = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0])
        result, info = cleanup_tag_islands(points, cells, tags, min_cells=0, min_fraction=.5)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(info["relabeled_components"], 2)
        self.assertEqual(info["unresolved_components"], [])


class Step10TaggingTests(unittest.TestCase):
    def setUp(self):
        with patch.object(sys, "argv", ["pipeline"]):
            self.args = pipeline.parse_args()

    def test_saved_labels_geometry_and_reported_islands(self):
        source = mesh_from_tetrahedra([INSIDE, INSIDE + [3, 0, 0], INSIDE + [.5, .5, .5]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path, reference = root / "untagged.vtk", root / "LVepi.vtk"
            source.save(mesh_path)
            pv.Cube(bounds=(-1, 1, -1, 1, -1, 1)).triangulate().save(reference)
            hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (mesh_path, reference)}
            paths, info = pipeline.save_step10_final_mesh(root / "step_10_final_mesh", mesh_path,
                                                          reference, "subject_test", "ED", self.args)
            self.assertEqual(Path(paths["volmesh"]).name, "subject_test_ED_volmesh.vtk")
            saved = pv.read(paths["volmesh"])
            np.testing.assert_array_equal(saved.points, source.points)
            np.testing.assert_array_equal(saved.cells, source.cells)
            np.testing.assert_array_equal(saved["elemTag"], [1, 0, 1])
            self.assertEqual(saved["elemTag"].dtype, np.int32)
            self.assertEqual(info["regions"]["LV"]["component_count"], 2)
            self.assertEqual(len(info["regions"]["LV"]["cells_outside_largest_component"]), 1)
            self.assertFalse(info["classification"]["island_relabeling_applied"])
            self.assertAlmostEqual(sum(r["volume_mm3"] for r in info["regions"].values()), info["total_volume_mm3"])
            for path, digest in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_centroid_or_three_vertices_rule(self):
        # Three interior vertices with an exterior centroid; then the converse.
        three_inside = [[.6, 0, 0], [.7, .2, 0], [.7, 0, .2], [5, 0, 0]]
        centroid_only = [[2, 2, 2], [2, -2, -2], [-2, 2, -2], [-2, -2, 2]]
        grid = mesh_from_tetrahedra([three_inside, centroid_only, INSIDE + [8, 0, 0]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference = root / "input.vtk", root / "LVepi.vtk"
            grid.save(source)
            pv.Cube(bounds=(-1, 1, -1, 1, -1, 1)).triangulate().save(reference)
            paths, info = pipeline.save_step10_final_mesh(root / "final", source, reference, "test", "ES", self.args)
            np.testing.assert_array_equal(pv.read(paths["volmesh"])["elemTag"], [1, 1, 0])
            self.assertEqual(info["classification"]["centroid_vertex_disagreement_cell_ids"], [0, 1])

    def test_missing_region_rejected_without_overwriting_final_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference = root / "input.vtk", root / "LVepi.vtk"
            mesh_from_tetrahedra([INSIDE]).save(source)
            pv.Cube(bounds=(-1, 1, -1, 1, -1, 1)).triangulate().save(reference)
            final = root / "final"
            final.mkdir()
            old = final / "test_ED_volmesh.vtk"
            old.write_bytes(b"previous validated mesh")
            with self.assertRaisesRegex(RuntimeError, "both LV and RV"):
                pipeline.save_step10_final_mesh(final, source, reference, "test", "ED", self.args)
            self.assertEqual(old.read_bytes(), b"previous validated mesh")
            summary = json.loads((final / "test_ED_step10_final_mesh_summary.json").read_text())
            self.assertEqual(summary["status"], "failed")


if __name__ == "__main__":
    unittest.main()
