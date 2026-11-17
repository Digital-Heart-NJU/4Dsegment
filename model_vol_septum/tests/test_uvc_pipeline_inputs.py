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
from model_vol_septum.compute_UVC_fiber import _opening_rim_loop
from model_vol_septum.pho_boundaries import SURFACE_CLASSES, build_pho_boundaries, load_surface


class PipelineInputTests(unittest.TestCase):
    def test_default_and_explicit_copied_folder_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary) / "subject_11"
            with patch.object(driver, "INPUT_PREFIX", None):
                paths = driver.pipeline_frame_paths(subject, "ED")
                self.assertEqual(paths["volume"], subject / "volmesh_pipline/step_09_final_mesh/subject_11_ED_final_volmesh.vtk")
                self.assertEqual(paths["LVannulus"], subject / "volmesh_pipline/step_08_LV_RV_openbase/subject_11_ED_LV_opening_rim.vtk")
                self.assertEqual(paths["RVannulus"].name, "subject_11_ED_RV_opening_rim.vtk")
                self.assertEqual(paths["LVendo"], subject / "volmesh_pipline/step_02_surf_clean/subject_11_ED_LVendo.vtk")
                self.assertEqual(paths["LVepi"], subject / "volmesh_pipline/step_03_move_RVepi/subject_11_ED_LVepi_moved.vtk")
                self.assertEqual(paths["RVendo"].name, "subject_11_ED_RV.vtk")
                self.assertEqual(paths["epi"].name, "subject_11_ED_LVRV_outer.vtk")
                self.assertNotIn("RVseptum", paths)
                self.assertNotIn("interface", paths)
                paths["volume"].parent.mkdir(parents=True)
                paths["volume"].touch()
                self.assertEqual(driver.frame_names(subject), ["ED"])
            with patch.object(driver, "INPUT_PREFIX", "source_subject"):
                self.assertEqual(driver.pipeline_frame_paths(subject, "ES")["volume"].name,
                                 "source_subject_ES_final_volmesh.vtk")
                self.assertEqual(driver.pipeline_frame_paths(subject, "ES")["LVannulus"].name,
                                 "source_subject_ES_LV_opening_rim.vtk")

    def test_no_silent_fallback_to_another_subject_prefix(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(driver, "INPUT_PREFIX", None):
            subject = Path(temporary) / "subject_111"
            paths = driver.pipeline_frame_paths(subject, "ED")
            paths["volume"].parent.mkdir(parents=True)
            (paths["volume"].parent / "subject_11_ED_final_volmesh.vtk").touch()
            with self.assertRaises(RuntimeError):
                driver.frame_names(subject)

    def test_saved_outer_surface_filename_and_explicit_name_precedence(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(driver, "INPUT_PREFIX", None):
            subject = Path(temporary) / "subject_11"
            directory = subject / "volmesh_pipline/step_04_extract_LVRV_surf"
            directory.mkdir(parents=True)
            saved = directory / "subject_11_ED_LVRV_outer_surf.vtk"
            saved.touch()
            self.assertEqual(driver.pipeline_frame_paths(subject, "ED")["epi"], saved)
            requested = directory / "subject_11_ED_LVRV_outer.vtk"
            requested.touch()
            self.assertEqual(driver.pipeline_frame_paths(subject, "ED")["epi"], requested)

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


class OpeningRimTests(unittest.TestCase):
    def test_complete_ring_is_used_without_z_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opening_rim.vtk"
            points = np.array([[0., 0., 0.], [1., 0., 3.], [1., 1., 2.], [0., 1., -1.]])
            pv.PolyData(points, lines=np.array([5, 0, 1, 2, 3, 0])).save(path)
            result, info = _opening_rim_loop(str(path))
            self.assertEqual({tuple(p) for p in result}, {tuple(p) for p in points})
            self.assertEqual(info["selection"], "single_closed_opening_rim_used_directly")
            self.assertEqual(info["segment_count"], 4)

    def test_open_or_multiple_rings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opening_rim.vtk"
            points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            for lines, coordinates in (
                ([3, 0, 1, 2], points),
                ([4, 0, 1, 2, 0, 4, 3, 4, 5, 3], np.vstack((points, points + [0., 0., 2.]))),
            ):
                with self.subTest(lines=lines):
                    pv.PolyData(coordinates, lines=np.asarray(lines)).save(path)
                    with self.assertRaisesRegex(RuntimeError, "one connected closed ring"):
                        _opening_rim_loop(str(path))


class PHOBoundaryTests(unittest.TestCase):
    def fixture(self, root):
        shell = pv.Cube().triangulate().subdivide(2)
        triangles = shell.points[shell.faces.reshape(-1, 4)[:, 1:]]
        top = np.all(np.isclose(triangles[:, :, 2], .5), axis=1)
        bottom = np.all(np.isclose(triangles[:, :, 2], -.5), axis=1)
        paths = {}
        for name, mask in (("epi", ~(top | bottom)), ("LVendo", top), ("RVendo", bottom)):
            part = shell.extract_cells(np.flatnonzero(mask)).extract_surface()
            paths[name] = root / (name + ".vtk")
            part.save(paths[name])
        paths["LVepi"] = root / "LVepi_moved.vtk"
        shell.save(paths["LVepi"])
        for name, height in (("LV", .5), ("RV", -.5)):
            points = np.array([[-.5, -.5, height], [.5, -.5, height],
                               [.5, .5, height], [-.5, .5, height]])
            paths[name + "annulus"] = root / (name + "_opening_rim.vtk")
            pv.lines_from_points(points, close=True).save(paths[name + "annulus"])
        volume = shell.delaunay_3d()
        volume.cell_data["elemTag"] = np.ones(volume.n_cells, dtype=np.int32)
        return volume, paths

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
            self.assertTrue(all(details["face_counts"][name] > 0 for name in ("epi", "LVendo", "RVendo")))
            self.assertTrue(details["internal_LVepi_constraint"])
            np.testing.assert_array_equal(volume.points, points)
            np.testing.assert_array_equal(volume.cells, cells)
            classified = pv.read(root / "work/volume_boundary_classified.vtk")
            self.assertEqual(set(classified["surfaceClass"]), {1, 2, 3})
            self.assertEqual(set(classified.point_data["PHO_BC"]), {0, 1})
            epi = pv.read(root / "work/epi_mapped.vtk")
            lvendo = pv.read(root / "work/LVendo_mapped.vtk")
            lvendo_ids = set(lvendo["volumeNodeId"])
            self.assertTrue(np.all(lvendo.point_data["PHO_BC"] == 0))
            self.assertTrue(lvendo_ids.issubset(set(zero)))
            self.assertFalse(lvendo_ids.intersection(one))
            self.assertTrue((set(epi["volumeNodeId"]) - lvendo_ids).issubset(set(one)))
            self.assertGreater(details["LVendo_priority_removed_one_node_count"], 0)

    def test_mapped_lvepi_excludes_junction_and_interface_supplies_internal_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, paths = self.fixture(root)
            volume = pv.ImageData(dimensions=(5, 5, 5), spacing=(.25, .25, .25),
                                  origin=(-.5, -.5, -.5)).cast_to_unstructured_grid().triangulate()
            volume.cell_data["elemTag"] = (volume.cell_centers().points[:, 0] < 0).astype(np.int32)
            pv.Cube(bounds=(-.5, 0., -.5, .5, -.5, .5)).triangulate().subdivide(2).save(paths["LVepi"])
            rim_points = np.array([[-.5, -.5, .5], [0., -.5, .5], [0., .5, .5], [-.5, .5, .5]])
            pv.lines_from_points(rim_points, close=True).save(paths["LVannulus"])
            before_points, before_cells = volume.points.copy(), volume.cells.copy()
            zero, one, details = build_pho_boundaries(volume, volume.points, paths, root / "work", .25)
            mapped = pv.read(root / "work/LVepi_mapped_nodes.vtk")
            ids = np.asarray(mapped["volumeNodeId"], dtype=int)
            np.testing.assert_array_equal(mapped.points, volume.points[ids])
            point_bc = np.asarray(mapped.point_data["PHO_BC"])
            self.assertTrue(set(ids[point_bc == 1]).issubset(set(one)))
            self.assertTrue(set(ids[point_bc == 0]).issubset(set(zero)))
            lvendo_ids = np.asarray(pv.read(root / "work/LVendo_zero_nodes.vtk")["volumeNodeId"], dtype=int)
            self.assertTrue(set(lvendo_ids).issubset(set(zero)))
            self.assertFalse(len(np.intersect1d(lvendo_ids, one)))
            center = int(np.flatnonzero(np.all(volume.points == [0., 0., 0.], axis=1))[0])
            self.assertNotIn(center, ids)
            self.assertIn(center, one)
            self.assertNotIn(center, zero)
            self.assertGreater(details["LVepi_mapping"]["internal_node_count"], 0)
            self.assertGreater(details["LVepi_additional_one_node_count"], 0)
            self.assertGreater(details["LVepi_zero_boundary_override_count"], 0)
            opened = pv.read(root / "work/LVepi_openbase.vtk")
            triangles = opened.points[opened.faces.reshape(-1, 4)[:, 1:]]
            self.assertFalse(np.any(np.all(np.isclose(triangles[:, :, 2], .5), axis=1)))
            bottom = np.all(np.isclose(triangles[:, :, 2], -.5), axis=1)
            self.assertTrue(np.any(bottom))
            self.assertGreater(opened.n_open_edges, 0)
            self.assertTrue(details["LVepi_mapping"]["uses_tag1_surface_selection"])
            self.assertFalse(details["LVepi_mapping"]["source_surface_cut_performed"])
            mapped_surface = pv.read(root / "work/LVepi_mapped.vtk")
            junction = pv.read(root / "work/LVepi_interface.vtk")
            mapped_faces = {tuple(row) for row in mapped_surface.cell_data["volumeFaceNodeIds"]}
            junction_faces = {tuple(row) for row in junction.cell_data["volumeFaceNodeIds"]}
            self.assertFalse(mapped_faces & junction_faces)
            combined_faces = {tuple(row) for row in opened.cell_data["volumeFaceNodeIds"]}
            self.assertEqual(combined_faces, mapped_faces | junction_faces)
            self.assertEqual(opened.n_cells, mapped_surface.n_cells + junction.n_cells)
            np.testing.assert_array_equal(mapped_surface.points, volume.points[mapped_surface["volumeNodeId"]])
            np.testing.assert_array_equal(junction.points, volume.points[junction["volumeNodeId"]])
            self.assertTrue(np.all(volume["elemTag"][mapped_surface["tag1CellId"]] == 1))
            self.assertTrue(np.all(mapped_surface["tag0CellId"] == -1))
            cells = volume.cells_dict[pv.CellType.TETRA]
            for face, owner in zip(mapped_surface["volumeFaceNodeIds"], mapped_surface["tag1CellId"]):
                self.assertTrue(set(face) <= set(cells[owner]))
            for face, left, right in zip(junction["volumeFaceNodeIds"], junction["tag0CellId"], junction["tag1CellId"]):
                self.assertTrue(set(face) <= set(cells[left]))
                self.assertTrue(set(face) <= set(cells[right]))
            self.assertTrue(np.all(volume["elemTag"][junction["tag1CellId"]] == 1))
            self.assertTrue(np.all(volume["elemTag"][junction["tag0CellId"]] == 0))
            self.assertTrue(np.allclose(junction.points[:, 0], 0))
            junction_ids = np.asarray(junction["volumeNodeId"])
            self.assertIn(center, junction_ids)
            self.assertTrue(set(junction_ids) - set(lvendo_ids) <= set(one))
            self.assertTrue(np.any(junction.point_data["PHO_BC"] == 0))
            self.assertEqual(set(opened.cell_data["boundaryPart"]), {1, 2})
            self.assertFalse(len(np.intersect1d(zero, one)))
            np.testing.assert_array_equal(volume.points, before_points)
            np.testing.assert_array_equal(volume.cells, before_cells)

    def test_distant_step03_lvepi_fails_mapping_instead_of_dropping_faces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume, paths = self.fixture(root)
            pv.read(paths["LVepi"]).translate((20, 0, 0)).save(paths["LVepi"])
            with self.assertRaisesRegex(RuntimeError, "mapping tolerance"):
                build_pho_boundaries(volume, volume.points, paths, root / "work", 1.5)
            self.assertTrue((root / "work/LVepi_unmapped_faces.vtk").is_file())

    def test_step03_lvepi_is_required_as_the_source_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume, paths = self.fixture(root)
            paths["LVepi"] = root / "does_not_exist.vtk"
            with self.assertRaises(FileNotFoundError):
                build_pho_boundaries(volume, volume.points, paths, root / "work", 1.5)

    def test_basal_patch_requires_a_complete_opening_rim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume, paths = self.fixture(root)
            endo = pv.read(paths["LVendo"])
            centers = endo.cell_centers().points
            middle = (np.abs(centers[:, 0]) < .26) & (np.abs(centers[:, 1]) < .26)
            endo.extract_cells(np.flatnonzero(middle)).extract_surface().save(paths["LVendo"])
            _, _, details = build_pho_boundaries(volume, volume.points, paths, root / "work", 1.5)
            self.assertGreater(details["face_counts"]["LVannulus"], 0)
            self.assertEqual(details["basal_components"][0]["chamber"], "LV")
            self.assertTrue(details["basal_components"][0]["complete_opening_rim_covered"])
            opened = pv.read(root / "work/LVepi_openbase.vtk")
            self.assertGreater(opened.n_open_edges, 0)
            surface = pv.read(root / "work/LVepi_mapped.vtk")
            faces = surface.points[surface.faces.reshape(-1, 4)[:, 1:]]
            self.assertFalse(np.any(np.all(np.isclose(faces[:, :, 2], .5), axis=1)))

    def test_unmatched_shell_stops_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume, paths = self.fixture(root)
            shell = pv.read(paths["epi"])
            shell.translate((10, 0, 0)).save(paths["epi"])
            with self.assertRaisesRegex(RuntimeError, "did not match any volume faces for epi"):
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
