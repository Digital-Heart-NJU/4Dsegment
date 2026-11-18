"""Plane clipping, GPU preview, deferred calculation and Dice checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pyvista as pv


SPEC = importlib.util.spec_from_file_location(
    "coords_dice", Path(__file__).resolve().parents[1] / "volmesh_dice_compare/interactive_coords_z_dice.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def box():
    mesh = pv.ImageData(dimensions=(3, 3, 3), spacing=(1, 1, 1)).cast_to_unstructured_grid().triangulate()
    mesh.cell_data["elemTag"] = (mesh.cell_centers().points[:, 0] >= 1).astype(np.int32)
    mesh.points = np.asarray(mesh.points, dtype=float)
    return mesh


def args_for(root, *extra):
    root = Path(root)
    return module.parse_args(["--template", str(root / "template.vtk"),
                              "--aligned", str(root / "aligned.vtk"),
                              "--output-dir", str(root / "results"), "--spacing", "1", ".5", *extra])


def plane(x=1.5, side="negative"):
    return module.validate_plane({"origin_mm": [x, 1, 1], "normal": [1, 0, 0], "keep_side": side})


class TestPlaneDice(unittest.TestCase):
    def test_plane_validation_and_normalization(self):
        for normal in ([0, 0, 0], [float("nan"), 0, 1]):
            with self.assertRaises(ValueError):
                module.validate_plane(dict(plane(), normal=normal))
        with self.assertRaises(ValueError):
            module.validate_plane(dict(plane(), keep_side="invalid"))
        np.testing.assert_allclose(module.validate_plane(dict(plane(), normal=[3, 0, 0]))["normal"], [1, 0, 0])

    def test_plane_cuts_tetrahedra_conserves_tags_and_flips_side(self):
        mesh = box()
        selected, removed = module.clip_template(mesh, plane())
        stats = module.volume_stats(selected)
        self.assertAlmostEqual(stats["volume_mm3"], 6.)
        self.assertAlmostEqual(stats["tag_volumes_mm3"]["0"], 4.)
        self.assertAlmostEqual(stats["tag_volumes_mm3"]["1"], 2.)
        module.check_partition(module.volume_stats(mesh), stats, module.volume_stats(removed))
        flipped, _ = module.clip_template(mesh, plane(side="positive"))
        self.assertAlmostEqual(module.volume_stats(flipped)["volume_mm3"], 2.)
        full, empty = module.clip_template(mesh, plane(3))
        self.assertEqual(empty.n_cells, 0)
        np.testing.assert_array_equal(full.cells, mesh.cells)
        empty, full = module.clip_template(mesh, plane(-1))
        self.assertEqual(empty.n_cells, 0)
        self.assertAlmostEqual(module.volume_stats(full)["volume_mm3"], 8.)

    def test_oblique_plane_cuts_cube_into_equal_volumes(self):
        mesh = box()
        spec = module.validate_plane({"origin_mm": [1, 1, 1], "normal": [1, 2, 3]})
        selected, removed = module.clip_template(mesh, spec)
        self.assertAlmostEqual(module.volume_stats(selected)["volume_mm3"], 4.)
        module.check_partition(module.volume_stats(mesh), module.volume_stats(selected), module.volume_stats(removed))

    def test_cut_template_against_complete_aligned_volume_and_zero_tag_is_tissue(self):
        mesh = box()
        selected, _ = module.clip_template(mesh, plane())
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            args.spacing = [.5, .25]
            rows, grids = module.calculate_results(selected, mesh, args)
        for spacing in (.5, .25):
            scores = {row["region"]: row for row in rows if row["spacing_mm"] == spacing}
            self.assertAlmostEqual(scores["all_tissue"]["dice"], 12 / 14)
            self.assertAlmostEqual(scores["tag_0"]["dice"], 1.)
            self.assertAlmostEqual(scores["tag_1"]["dice"], 2 / 3)
            self.assertAlmostEqual(scores["all_tissue"]["aligned_volume_mm3"], 8.)
            self.assertAlmostEqual(scores["all_tissue"]["template_volume_mm3"], 6.)
        self.assertTrue(module.convergence_report(rows, .005)["within_tolerance"])

    def test_nonoverlap_and_empty_tag_are_not_confused_with_background(self):
        template, aligned = box(), box()
        aligned.points += [10., 0, 0]
        template.cell_data["elemTag"][:] = 0
        aligned.cell_data["elemTag"][:] = 0
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = module.calculate_results(template, aligned, args_for(directory))
        for row in rows:
            if row["region"] == "tag_1":
                self.assertIsNone(row["dice"])
            else:
                self.assertEqual(row["dice"], 0.)

    def test_grid_limit_checked_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            args.max_grid_points = 1
            with self.assertRaisesRegex(ValueError, "grid points"):
                module.calculate_results(box(), box(), args)

    def test_worker_saves_reproducible_results_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            mesh = box()
            mesh.save(args.template)
            mesh.save(args.aligned)
            before = args.template.read_bytes(), args.aligned.read_bytes()
            spec = plane()
            provenance = {"paths": [args.template, args.aligned],
                          "hashes": module.hash_files([args.template, args.aligned]),
                          "template_stats": module.volume_stats(mesh),
                          "aligned_stats": module.volume_stats(mesh)}
            args.output_dir.mkdir()
            result = module.run_calculation(mesh, mesh, args, provenance, spec, args.output_dir)
            self.assertTrue(result["inputs_unchanged"])
            self.assertEqual(before, (args.template.read_bytes(), args.aligned.read_bytes()))
            reloaded = pv.read(args.output_dir / "template_selected.vtu")
            self.assertAlmostEqual(module.volume_stats(reloaded)["volume_mm3"], 6.)
            saved = json.loads((args.output_dir / "results.json").read_text())
            self.assertEqual(saved["selection"]["plane"], spec)
            restore_args = args_for(root, "--selection-in", str(args.output_dir / "selection.json"))
            self.assertEqual(module.input_plane(restore_args, mesh), spec)
            self.assertEqual(len(saved["scores"]), 6)
            self.assertNotEqual(module.available_directory(args.output_dir), args.output_dir)

    def test_rendered_preview_keeps_the_same_side_as_volume_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            mesh = box()
            app = module.SelectionWindow(mesh, mesh, args, {}, off_screen=True)
            try:
                app.set_plane(plane())
                app.aligned_actor.visibility = False
                app.plane_widget.Off()
                for actor in (app.selected_actor, app.removed_actor):
                    actor.prop.opacity = 1
                    actor.prop.lighting = False
                app.plotter.view_xy()
                app.plotter.reset_camera()
                image = app.plotter.screenshot(return_img=True)
                def pixel_at(x):
                    coordinate = module.vtk.vtkCoordinate()
                    coordinate.SetCoordinateSystemToWorld()
                    coordinate.SetValue(x, 1, 2)
                    u, v = coordinate.GetComputedDisplayValue(app.plotter.renderer)
                    return image[image.shape[0] - 1 - v, u, :3].astype(int)
                blue, red = pixel_at(.5), pixel_at(1.75)
                self.assertGreater(blue[2], blue[0])
                self.assertGreater(red[0], red[2])
            finally:
                app.executor.shutdown(wait=True)
                app.plotter.close()

    def test_preview_never_clips_volume_then_confirmation_clips_once(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            mesh = box()
            mesh.save(args.template)
            mesh.save(args.aligned)
            provenance = {"paths": [args.template, args.aligned],
                          "hashes": module.hash_files([args.template, args.aligned])}
            app = None
            try:
                with mock.patch.object(module, "clip_template", side_effect=AssertionError("preview clipped volume")), \
                     mock.patch.object(module, "volume_stats", side_effect=AssertionError("preview computed volumes")):
                    app = module.SelectionWindow(mesh, mesh, args, provenance, off_screen=True)
                    app.plotter.show(interactive=False, auto_close=False)
                    surface = app.template_surface
                    app.set_plane(plane())
                    app._offset_changed(app._offset() + .1)
                    app._release()
                    app.plane_widget.SetOrigin(1.25, 1, 1)
                    app.plane_widget.InvokeEvent("InteractionEvent")
                    app.plane_widget.InvokeEvent("EndInteractionEvent")
                    self.assertAlmostEqual(app.plane["origin_mm"][0], 1.25)
                    app._buttons[0][0].InvokeEvent("StateChangedEvent")
                    self.assertEqual(app.plane["keep_side"], "positive")
                    app.undo()
                    self.assertEqual(app.plane["keep_side"], "negative")
                    app.orient(2)
                    app._buttons[1][0].InvokeEvent("StateChangedEvent")
                    self.assertEqual(app.plane, app.initial_plane)
                    app.set_plane(plane())
                    self.assertIs(app.template_surface, surface)
                    self.assertIsNone(app.result)
                    self.assertIsNone(app.output)
                with mock.patch.object(module, "clip_template", wraps=module.clip_template) as clip:
                    app._buttons[2][0].InvokeEvent("StateChangedEvent")
                    self.assertTrue(app.busy)
                    deadline = time.monotonic() + 20
                    while app.busy and time.monotonic() < deadline:
                        app.plotter.iren.interactor.InvokeEvent("TimerEvent")
                        app.plotter.update(stime=10)
                        time.sleep(.01)
                    self.assertFalse(app.busy, app.message)
                    self.assertIsNotNone(app.result, app.message)
                    self.assertEqual(clip.call_count, 1)
                self.assertEqual(app.result["selection"]["plane"], plane())
                self.assertTrue((app.output / "selection_preview.png").exists())
                app.orient(1)
                self.assertNotEqual(app.selection_signature(), app.result_signature)
                np.testing.assert_array_equal(mesh.points, box().points)
            finally:
                if app is not None:
                    app.executor.shutdown(wait=True)
                    app.plotter.close()


if __name__ == "__main__":
    unittest.main()
