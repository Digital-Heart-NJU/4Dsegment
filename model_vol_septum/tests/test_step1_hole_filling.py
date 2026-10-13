"""Regression tests for input hole filling and checked base-detection inputs."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_open_volmesh_stable as pipeline


def closed_surface():
    return pv.Cube().triangulate().clean().compute_normals(
        consistent_normals=True, auto_orient_normals=True, split_vertices=False,
    )


def open_surface():
    mesh = closed_surface()
    keep = mesh.cell_centers().points[:, 2] < 0.49
    return mesh.extract_cells(keep).extract_surface().triangulate()


class Step1HoleFillingTests(unittest.TestCase):
    def test_closed_surface_is_loaded_without_geometry_changes(self):
        source = closed_surface()
        old_points, old_faces = source.points.copy(), source.faces.copy()
        result, info = pipeline.check_and_fill_loaded_surface(source, Path("LVendo_ED.vtk"), 1)
        self.assertIs(result, source)
        self.assertEqual(info["status"], "already_closed_direct_load")
        self.assertEqual(info["added_triangles"], 0)
        np.testing.assert_array_equal(result.points, old_points)
        np.testing.assert_array_equal(result.faces, old_faces)

    def test_hole_caps_preserve_vertices_faces_and_chamber_tag(self):
        source = open_surface()
        old_points, old_faces = source.points.copy(), source.faces.copy()
        result, info = pipeline.check_and_fill_loaded_surface(source, Path("RV_ED.vtk"), 0)
        self.assertEqual(info["before"]["free_edges"], 4)
        self.assertEqual(info["after"]["free_edges"], 0)
        self.assertEqual(info["after"]["inconsistent_orientation_edges"], 0)
        self.assertEqual(info["added_triangles"], 2)
        np.testing.assert_array_equal(result.points, old_points)
        np.testing.assert_array_equal(result.faces[:len(old_faces)], old_faces)
        self.assertTrue(np.all(result.cell_data["elemTag"] == 0))

    def test_four_inputs_and_saved_outputs_preserve_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = (("LVendo", "lv_endo", 1), ("LVepi", "lv_epi", 1),
                     ("RV", "rv_endo", 0), ("RVepi", "rv_epi", 0))
            for prefix, _, _ in specs:
                mesh = closed_surface() if prefix == "LVendo" else open_surface()
                mesh.save(root / f"{prefix}_ED.vtk")
            input_paths = pipeline.input_paths_for_frame(root, "ED")
            before_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in input_paths.items()}
            surfaces, checks = pipeline.load_checked_surfaces(root, "ED")
            output_dir = root / "step_01_load_surf"
            output_dir.mkdir()
            paths, summary = pipeline.save_step1_loaded_surfaces(
                output_dir, "subject_test", "ED", surfaces, input_paths, checks,
            )
            self.assertTrue(summary["input_hole_filling_enabled"])
            for prefix, name, tag in specs:
                saved = pv.read(paths[prefix]).extract_surface().triangulate()
                self.assertEqual(pipeline.topology(saved)["free_edges"], 0)
                self.assertTrue(np.all(saved.cell_data["elemTag"] == tag))
                self.assertEqual(hashlib.sha256(input_paths[name].read_bytes()).hexdigest(),
                                 before_hashes[name])
                self.assertIn("input_hole_check", summary["surfaces"][prefix])

    def test_detector_uses_checked_surfaces_without_rereading_raw_meshes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checked = closed_surface()
            paths = [root / f"{name}_{frame}.vtk"
                     for frame in ("ED", "ES") for name in ("LVendo", "RV")]
            for path in paths:
                checked.save(path)
            seen = []

            def checked_loader(path):
                seen.append(path)
                return checked

            args = SimpleNamespace(
                subject_base_side="auto", subject_base_frames="ED,ES", frame="ED",
                samples=180, slab_fraction=0.035, end_band_fraction=0.12, stable_fraction=0.45,
            )
            with patch.object(pipeline.base, "load_surface", side_effect=AssertionError("raw reread")), \
                    patch.object(pipeline.base, "compute_area_profile",
                                 return_value=SimpleNamespace(low_end_area=1.0, high_end_area=3.0)):
                side, info = pipeline.openbase.detect_subject_base_side(
                    root, args, surface_loader=checked_loader,
                )
            self.assertEqual(seen, paths)
            self.assertEqual(side, "high_z")
            self.assertEqual(info["high_votes"], 4)

    def test_unrepairable_open_surface_reports_input_path(self):
        source = open_surface()
        invalid = pv.PolyData(source.points, np.concatenate((source.faces, source.faces[:4])))
        with self.assertRaisesRegex(RuntimeError, "RV_ED.vtk: input hole filling failed.*nonmanifold"):
            pipeline.check_and_fill_loaded_surface(invalid, Path("RV_ED.vtk"), 0)

    def test_active_pipeline_uses_repaired_inputs_for_steps_1_2_and_3(self):
        with tempfile.TemporaryDirectory() as directory:
            subject_dir = Path(directory) / "subject_test"
            vtk_dir = subject_dir / "vtks"
            vtk_dir.mkdir(parents=True)
            for frame in ("ED", "ES"):
                for prefix in ("LVendo", "LVepi", "RV", "RVepi"):
                    open_surface().save(vtk_dir / f"{prefix}_{frame}.vtk")
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in vtk_dir.glob("*.vtk")}
            with patch.object(sys, "argv", ["pipeline", "--subject-dir", str(subject_dir)]):
                args = pipeline.parse_args()

            checked_profiles = []

            def profile(surface, **kwargs):
                self.assertEqual(pipeline.topology(surface)["free_edges"], 0)
                checked_profiles.append(surface)
                return SimpleNamespace(low_end_area=1.0, high_end_area=3.0)

            class ReachedStep3(Exception):
                pass

            def build_shell(name, endo, epi, *args):
                self.assertEqual(name, "LV")
                self.assertEqual(pipeline.topology(endo)["free_edges"], 0)
                self.assertEqual(pipeline.topology(epi)["free_edges"], 0)
                self.assertIs(endo, checked_profiles[0])
                raise ReachedStep3

            with patch.object(pipeline.base, "compute_area_profile", side_effect=profile), \
                    patch.object(pipeline.base, "load_surface", wraps=pipeline.base.load_surface) as loader, \
                    patch.object(pipeline, "save_step2_base_detection", return_value=({}, {})) as save_step2, \
                    patch.object(pipeline, "build_ventricle_shell_for_pre_step5", side_effect=build_shell):
                with self.assertRaises(ReachedStep3):
                    pipeline.process_frame_new_pipeline(args, "ED")
            # Four ED meshes loaded once, plus two ES endocardial meshes for base detection.
            self.assertEqual(loader.call_count, 6)
            self.assertEqual(len(checked_profiles), 6)
            detection = save_step2.call_args.args[3]
            self.assertEqual(len(detection["additional_frame_input_hole_checks"]), 2)
            for info in detection["additional_frame_input_hole_checks"].values():
                self.assertEqual(info["status"], "filled")
                self.assertEqual(info["after"]["free_edges"], 0)
            step1 = subject_dir / "volmesh_pipline" / "step_01_load_surf"
            summary = json.loads((step1 / "subject_test_ED_step01_load_surf_summary.json").read_text())
            for record in summary["surfaces"].values():
                saved = pv.read(record["path"]).extract_surface().triangulate()
                self.assertEqual(pipeline.topology(saved)["free_edges"], 0)
                self.assertEqual(record["input_hole_check"]["status"], "filled")
            for path, before in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
