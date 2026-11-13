"""Ordered tag-region growth, protected main regions and immutable geometry."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv

from test_new_pipeline_step08 import tagged_grid, pipeline


UNIT = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))


def growth_graph(tags=(0, 1, 1, 0), volumes=(100000., 1., 200000., 2.),
                 edges=((0, 1), (0, 2), (2, 3)), fraction=1e-3):
    pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return pipeline._grow_tag_components(np.asarray(tags, dtype=np.int32), np.asarray(volumes),
        pairs[:, 0], pairs[:, 1], np.ones(len(pairs)), fraction)


def surface_island_mesh(contact="face"):
    big = UNIT*100
    if contact == "face":
        extra = np.array(((0., 0., -.001),))
        small = (0, 2, 1, 4)
    elif contact == "edge":
        extra = np.array(((0., -.1, 0.), (0., 0., -.1)))
        small = (0, 1, 4, 5)
    else:
        extra = np.array(((-.1, 0., 0.), (0., -.1, 0.), (0., 0., -.1)))
        small = (0, 4, 5, 6)
    offset = len(big)+len(extra)
    points = np.vstack((big, extra, big+(1000, 0, 0)))
    return tagged_grid(points, ((0, 1, 2, 3), small, tuple(range(offset, offset+4))), (0, 1, 1))


class GrowthTests(unittest.TestCase):
    def test_tag1_grows_first_then_tag0_and_main_regions_are_protected(self):
        tags, phases, report = growth_graph()
        np.testing.assert_array_equal(tags, (0, 0, 1, 1))
        np.testing.assert_array_equal(phases, (0, 2, 0, 1))
        self.assertEqual([move["pass"] for move in report["changes"]], [1, 2])
        self.assertEqual([move["to_tag"] for move in report["changes"]], [1, 0])
        self.assertEqual(report["regions_after"]["0"]["components"], 1)
        self.assertEqual(report["regions_after"]["1"]["components"], 1)
        self.assertEqual(report["island_volume_limit_mm3"], {"0": 100., "1": 200.})
        self.assertTrue(report["initial_main_regions_preserved"])

    def test_nested_islands_merge_without_reversing_a_previous_change(self):
        tags, phases, report = growth_graph(tags=(0, 1, 0, 1), volumes=(100000, 1, 2, 200000),
            edges=((0, 1), (1, 2), (2, 3)))
        np.testing.assert_array_equal(tags, (0, 1, 1, 1))
        np.testing.assert_array_equal(phases, (0, 0, 1, 0))
        self.assertEqual(report["absorbed_islands"], 1)
        self.assertEqual(report["final_component_count"], 2)

    def test_size_is_based_on_volume_not_cell_count(self):
        tags, _, report = growth_graph(tags=(0, 1, 1, 1, 1),
            volumes=(100000, 200000, 2, 2, 2), edges=((0, 1), (0, 2), (2, 3), (3, 4)))
        np.testing.assert_array_equal(tags, (0, 1, 0, 0, 0))
        self.assertEqual(report["changes"][0]["cells"], 3)
        self.assertEqual(report["changes"][0]["volume_mm3"], 6.)

    def test_equal_or_larger_than_volume_limit_is_retained(self):
        tags, phases, report = growth_graph(volumes=(100000, 200, 200000, 2))
        np.testing.assert_array_equal(tags, (0, 1, 1, 1))
        self.assertEqual(phases[1], 0)
        self.assertEqual(report["unresolved_components"][0]["reason"], "at_or_above_volume_limit")
        self.assertEqual(report["island_volume_limit_mm3"]["1"], 200.)

    def test_zero_fraction_disables_relabeling(self):
        tags, phases, report = growth_graph(fraction=0)
        np.testing.assert_array_equal(tags, (0, 1, 1, 0))
        self.assertFalse(phases.any())
        self.assertEqual(report["changed_cells"], 0)

    def test_volume_limits_do_not_expand_after_growth(self):
        _, _, report = growth_graph()
        self.assertEqual(report["island_volume_limit_mm3"], {"0": 100., "1": 200.})
        self.assertEqual(report["regions_after"]["0"]["largest_component_volume_mm3"], 100001.)

    def test_attached_patch_in_main_region_is_not_eroded(self):
        tags, phases, report = growth_graph(tags=(0, 1, 1), volumes=(100000, 200000, 1),
                                           edges=((0, 2), (2, 1)))
        np.testing.assert_array_equal(tags, (0, 1, 1))
        self.assertFalse(phases.any())
        self.assertEqual(report["changed_cells"], 0)

    def test_uniform_scaling_keeps_growth_decisions(self):
        expected, _, _ = growth_graph()
        actual, _, _ = growth_graph(volumes=np.array((100000, 1, 200000, 2))*123.)
        np.testing.assert_array_equal(actual, expected)

    def test_no_shared_face_with_main_region_is_reported(self):
        tags, _, report = growth_graph(edges=((0, 2),))
        np.testing.assert_array_equal(tags, (0, 1, 1, 0))
        self.assertEqual(len(report["unresolved_components"]), 2)
        self.assertTrue(all(item["reason"] == "no_shared_face_with_opposite_main_region"
                            for item in report["unresolved_components"]))

    def test_invalid_fraction_or_missing_chamber_is_rejected(self):
        for value in (-1, np.nan, np.inf, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fraction"):
                growth_graph(fraction=value)
        with self.assertRaisesRegex(ValueError, "both"):
            growth_graph(tags=(0, 0, 0, 0))


class MeshGrowthTests(unittest.TestCase):
    def correct(self, mesh):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.correct_openbase_tags(mesh)

    def test_complete_face_contact_allows_growth_without_changing_geometry(self):
        mesh = surface_island_mesh()
        mesh.cell_data["referenceCellId"] = np.array((3, 5, 7), dtype=np.int32)
        before = mesh.copy(deep=True)
        result, report = self.correct(mesh)
        np.testing.assert_array_equal(result.cell_data["elemTag"], (0, 0, 1))
        np.testing.assert_array_equal(result.cell_data["elemTag_step08"], (0, 1, 1))
        np.testing.assert_array_equal(result.cell_data["tagChanged"], (0, 1, 0))
        np.testing.assert_array_equal(result.cell_data["growthPass"], (0, 2, 0))
        np.testing.assert_array_equal(result.cell_data["referenceCellId"], (3, 5, 7))
        pipeline._assert_identical_geometry(before, result)
        pipeline._assert_identical_geometry(before, mesh)
        self.assertEqual(report["LV_to_RV"], 1)
        self.assertFalse(report["step06_overlap_voting_performed"])

    def test_default_limit_absorbs_island_between_old_and_new_limits(self):
        mesh = surface_island_mesh()
        mesh.points[4, 2] = -.02
        result, report = self.correct(mesh)
        np.testing.assert_array_equal(result.cell_data["elemTag"], (0, 0, 1))
        self.assertAlmostEqual(report["island_volume_percent"], .1)
        old_limit_result, _ = pipeline.correct_openbase_tags(mesh, island_fraction=1e-4)
        np.testing.assert_array_equal(old_limit_result.cell_data["elemTag"], (0, 1, 1))

    def test_vertex_and_edge_contact_cannot_join_regions(self):
        for contact in ("vertex", "edge"):
            with self.subTest(contact=contact):
                mesh = surface_island_mesh(contact)
                result, report = self.correct(mesh)
                np.testing.assert_array_equal(result.cell_data["elemTag"], mesh.cell_data["elemTag"])
                self.assertEqual(report["unresolved_components"][0]["reason"],
                                 "no_shared_face_with_opposite_main_region")

    def test_negative_tetrahedron_orientation_is_preserved(self):
        mesh = surface_island_mesh()
        cells = mesh.cells.reshape(-1, 5).copy()
        cells[:, [2, 3]] = cells[:, [3, 2]]
        reverse = pv.UnstructuredGrid(cells.ravel(), mesh.celltypes, mesh.points)
        reverse.cell_data["elemTag"] = mesh.cell_data["elemTag"].copy()
        result, _ = self.correct(reverse)
        pipeline._assert_identical_geometry(reverse, result)
        np.testing.assert_array_equal(result.cell_data["elemTag"], (0, 0, 1))

    def test_already_processed_input_is_rejected(self):
        mesh = surface_island_mesh()
        result, _ = self.correct(mesh)
        with self.assertRaisesRegex(ValueError, "original Step 08"):
            self.correct(result)


class Stage9Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name)/"subject_test"
        self.root = self.subject/"volmesh_pipline"
        self.output = self.root/"step_09_final_mesh"
        target = surface_island_mesh()
        target.point_data["test_scalar"] = np.arange(target.n_points, dtype=np.float64)+.12345678901234567
        target.point_data["test_vector"] = np.arange(target.n_points*3, dtype=np.float32).reshape(-1, 3)
        target.cell_data["test_int64"] = np.arange(3, dtype=np.int64)+2**40
        target.cell_data["test_float"] = np.array((.12345678901234567, .2, .3))
        target.field_data["test_field"] = np.array((123,), dtype=np.int32)
        self.inputs = []
        for frame in ("ED", "ES"):
            path = self.root/"step_08_LV_RV_openbase"/f"subject_test_{frame}_openbase_volmesh.vtk"
            path.parent.mkdir(parents=True, exist_ok=True)
            pipeline._write_step09_mesh(target, path)
            self.inputs.append(path)

    def run_step(self, overwrite=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.step_09_final_mesh(self.subject, ("ED", "ES"), overwrite)

    def test_both_frames_are_saved_with_lossless_geometry_and_all_arrays(self):
        hashes = {path: pipeline.checksum(path) for path in self.inputs}
        self.run_step()
        self.assertFalse((self.root/"step_06_volmesh_generate").exists())
        self.assertEqual(len(list(self.output.glob("*.vtk"))), 2)
        for frame in ("ED", "ES"):
            source = pv.read(self.root/"step_08_LV_RV_openbase"/f"subject_test_{frame}_openbase_volmesh.vtk")
            saved = pv.read(self.output/f"subject_test_{frame}_final_volmesh.vtk")
            pipeline._assert_identical_geometry(source, saved)
            for attribute in ("cell_data", "point_data", "field_data"):
                for key, values in getattr(source, attribute).items():
                    if key != "elemTag":
                        np.testing.assert_array_equal(values, getattr(saved, attribute)[key])
            np.testing.assert_array_equal(saved.cell_data["elemTag"], (0, 0, 1))
            np.testing.assert_array_equal(saved.cell_data["growthPass"], (0, 2, 0))
            self.assertEqual(saved.active_scalars_name, "elemTag")
            report = json.loads((self.output/f"subject_test_{frame}_step09_tag_correction_summary.json").read_text())
            self.assertTrue(report["saved_geometry_matches_step08_exactly"])
            self.assertAlmostEqual(report["island_volume_percent"], .1)
        self.assertEqual(hashes, {path: pipeline.checksum(path) for path in self.inputs})

    def test_existing_outputs_are_archived_only_when_requested(self):
        self.run_step()
        old = self.output/"subject_test_ED_final_volmesh.vtk"
        digest = pipeline.checksum(old)
        with self.assertRaises(FileExistsError):
            self.run_step()
        self.run_step(overwrite=True)
        archives = list((self.output/"previous_outputs").glob("*/"+old.name))
        self.assertEqual(len(archives), 1)
        self.assertEqual(pipeline.checksum(archives[0]), digest)

    def test_step09_cli_does_not_run_step08(self):
        with patch.object(pipeline, "step_08_LV_RV_openbase") as earlier, \
                patch.object(pipeline, "step_09_final_mesh") as current, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--subject-dir", str(self.subject), "--frame", "ED", "--start-step", "9", "--stop-after", "9"])
        earlier.assert_not_called()
        current.assert_called_once_with(self.subject, ["ED"], False, .1)


if __name__ == "__main__":
    unittest.main()
