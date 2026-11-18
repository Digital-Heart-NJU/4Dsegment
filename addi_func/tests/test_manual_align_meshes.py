"""Rigid/mirrored pose, export round-trip and controller checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyvista as pv


SPEC = importlib.util.spec_from_file_location(
    "manual_align", Path(__file__).resolve().parents[1] / "volmesh_dice_compare" / "manual_align_meshes.py")
manual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manual)


def tetra():
    mesh = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]),
                               np.array([pv.CellType.TETRA]),
                               np.array([[0., 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]]))
    mesh.cell_data["elemTag"] = np.array([0], dtype=np.int32)
    mesh.cell_data["counts"] = np.array([3], dtype=np.int64)
    mesh.point_data["temperature"] = np.arange(4, dtype=float)
    return mesh


class TestManualAlignment(unittest.TestCase):
    def test_rotation_keeps_pivot_and_moves_x_to_y(self):
        pivot = np.array([5., 2, -3])
        matrix = manual.rotation_step(2, 90, pivot)
        np.testing.assert_allclose(manual.transform_points(pivot, matrix), pivot, atol=1e-12)
        np.testing.assert_allclose(manual.transform_points(pivot + [1, 0, 0], matrix),
                                   pivot + [0, 1, 0], atol=1e-12)
        manual.validate_transform(matrix)

    def test_scale_shear_and_nonfinite_rejected_but_reflection_allowed(self):
        for value in (2., 0., float("nan")):
            matrix = np.eye(4)
            matrix[0, 0] = value
            with self.assertRaises(ValueError):
                manual.validate_transform(matrix)
        shear = np.eye(4)
        shear[0, 1] = .2
        with self.assertRaises(ValueError):
            manual.validate_transform(shear)
        manual.validate_transform(manual.mirror_step(0, [0, 0, 0]))

    def test_mirror_all_axes_preserves_center_distances_and_is_reversible(self):
        pivot = np.array([5., 2, -3])
        point = pivot + [1., 2, 4]
        for axis in range(3):
            with self.subTest(axis=axis):
                matrix = manual.mirror_step(axis, pivot)
                expected = point.copy()
                expected[axis] = 2 * pivot[axis] - point[axis]
                np.testing.assert_allclose(manual.transform_points(point, matrix), expected)
                np.testing.assert_allclose(manual.transform_points(pivot, matrix), pivot)
                np.testing.assert_allclose(matrix @ matrix, np.eye(4))
                self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), -1)
                self.assertAlmostEqual(np.linalg.norm(expected - pivot), np.linalg.norm(point - pivot))

    def test_mirrored_export_preserves_orientation_labels_and_transforms_directions(self):
        mesh = tetra()
        mesh.cell_data["fiber"] = np.array([[1., 2, 3]])
        mesh.cell_data.active_vectors_name = "fiber"
        mesh.cell_data["tensor"] = np.array([[1., 2, 3, 2, 5, 6, 3, 6, 9]])
        mesh.cell_data.VTKObject.SetActiveTensors("tensor")
        mesh.point_data["Normals"] = np.tile([1., 0, 0], (mesh.n_points, 1))
        mesh.point_data.active_normals_name = "Normals"
        pose = manual.rotation_step(2, 32, mesh.center)
        pivot = manual.transform_points(np.array(mesh.center), pose)
        pose = manual.mirror_step(0, pivot) @ pose
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = manual.save_alignment(mesh, pose, root / "fixed.vtk", root / "moving.vtk", root)
            result = pv.read(root / "subject_11_volmesh_aligned.vtk")
            np.testing.assert_array_equal(np.sort(result.cells.reshape(-1, 5)[:, 1:], axis=1),
                                          np.sort(mesh.cells.reshape(-1, 5)[:, 1:], axis=1))
            np.testing.assert_allclose(manual.signed_tetra_volumes(result), manual.signed_tetra_volumes(mesh))
            np.testing.assert_allclose(result.cell_data["fiber"], np.array([[1., 2, 3]]) @ pose[:3, :3].T)
            np.testing.assert_allclose(result.point_data["Normals"], mesh.point_data["Normals"] @ pose[:3, :3].T)
            np.testing.assert_allclose(result.cell_data["tensor"].reshape(3, 3),
                                       pose[:3, :3] @ mesh.cell_data["tensor"].reshape(3, 3) @ pose[:3, :3].T)
            np.testing.assert_array_equal(result.cell_data["elemTag"], mesh.cell_data["elemTag"])
            np.testing.assert_array_equal(manual.load_transform(root / "transform.json"), pose)
            self.assertTrue(report["has_reflection"])
            self.assertEqual(report["tetra_local_vertex_swap"], [0, 1])
            self.assertAlmostEqual(report["linear_determinant"], -1)
            # Two flips return the exact original local cell ordering.
            restored_pose = manual.mirror_step(0, pivot) @ pose
            restored = manual.aligned_volume(mesh, restored_pose)
            np.testing.assert_array_equal(restored.cells, mesh.cells)

    def test_full_export_preserves_labels_and_rotates_active_directions(self):
        mesh = tetra()
        mesh.cell_data["fiber"] = np.array([[1., 0, 0]])
        mesh.cell_data.active_vectors_name = "fiber"
        original = mesh.copy(deep=True)
        matrix = manual.rotation_step(2, 90, [0, 0, 0])
        matrix[:3, 3] = [12, -7, 5]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moving, fixed = root / "moving.vtk", root / "fixed.vtk"
            mesh.save(moving)
            mesh.save(fixed)
            before = moving.read_bytes(), fixed.read_bytes()
            report = manual.save_alignment(mesh, matrix, fixed, moving, root / "output")
            result = pv.read(root / "output/subject_11_volmesh_aligned.vtk")
            np.testing.assert_allclose(result.cell_data["fiber"], [[0, 1, 0]], atol=1e-12)
            np.testing.assert_array_equal(result.cell_data["elemTag"], [0])
            np.testing.assert_array_equal(result.points, manual.transform_points(mesh.points, matrix))
            np.testing.assert_array_equal(manual.load_transform(root / "output/transform.json"), matrix)
            self.assertAlmostEqual(report["absolute_volume"], 4.)
            self.assertEqual(before, (moving.read_bytes(), fixed.read_bytes()))
            np.testing.assert_array_equal(mesh.points, original.points)
            with self.assertRaises(FileExistsError):
                manual.save_alignment(mesh, matrix, fixed, moving, root / "output")

    def test_export_cannot_replace_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "input mesh"):
                manual.save_alignment(tetra(), np.eye(4), root / "fixed.vtk",
                                       root / "subject_11_volmesh_aligned.vtk", root, overwrite=True)

    def test_controller_undo_reset_and_mouse_follow_keyboard(self):
        with tempfile.TemporaryDirectory() as directory:
            args = manual.parse_args(["--output-dir", directory])
            moving = tetra()
            fixed = tetra()
            fixed.points += [20., 30, -10]
            original_fixed = fixed.points.copy()
            app = manual.ManualAligner(fixed, moving, args, off_screen=True)
            try:
                app.plotter.show(interactive=False, auto_close=False)
                interactor = app.plotter.iren.interactor
                # Real key+character events must not also trigger VTK's
                # wireframe, fly-to or camera-reset character bindings.
                camera = np.asarray(list(app.plotter.camera_position), dtype=float)
                for key in ("w", "s", "f", "r"):
                    interactor.SetKeySym(key)
                    interactor.SetKeyCode(key)
                    interactor.InvokeEvent("KeyPressEvent")
                    interactor.InvokeEvent("CharEvent")
                self.assertEqual(app.moving_actor.prop.GetRepresentation(), 2)
                np.testing.assert_allclose(list(app.plotter.camera_position), camera)
                app.toggle_fine()
                app.translate(0, 1)
                translated = app.matrix.copy()
                for axis, key in enumerate(("5", "6", "7")):
                    center = app.pivot.copy()
                    interactor.SetKeySym(key)
                    interactor.SetKeyCode(key)
                    interactor.InvokeEvent("KeyPressEvent")
                    interactor.InvokeEvent("CharEvent")
                    self.assertLess(np.linalg.det(app.matrix[:3, :3]), 0)
                    np.testing.assert_allclose(app.pivot, center)
                    np.testing.assert_allclose(app.widget._cached_matrix, app.matrix)
                    app.rotate(axis, 1)
                    self.assertLess(np.linalg.det(app.matrix[:3, :3]), 0)
                    app.undo()
                    app.undo()
                    np.testing.assert_allclose(app.matrix, translated)
                app.rotate(2, 1)
                app.undo()
                np.testing.assert_array_equal(app.matrix, translated)
                # The next drag must start from this external keyboard/undo pose.
                np.testing.assert_array_equal(app.widget._cached_matrix, translated)
                app.center()
                np.testing.assert_allclose(app.pivot, fixed.center)
                np.testing.assert_allclose(app.widget.origin, fixed.center)
                mouse_pose = app.matrix.copy()
                mouse_pose[1, 3] += 4
                app.moving_actor.user_matrix = mouse_pose
                app._mouse_release(mouse_pose)
                app.undo()
                np.testing.assert_allclose(app.pivot, fixed.center)
                app.toggle_handles()
                self.assertIsNone(app.widget)
                app.translate(2, 1)
                app.toggle_handles()
                np.testing.assert_array_equal(app.widget._cached_matrix, app.matrix)
                app._commit(np.eye(4))
                np.testing.assert_array_equal(app.matrix, np.eye(4))
                np.testing.assert_array_equal(fixed.points, original_fixed)
                np.testing.assert_array_equal(app.fixed_actor.GetPosition(), [0, 0, 0])
                app.plotter.screenshot(str(Path(directory) / "preview.png"))
            finally:
                app.plotter.close()


if __name__ == "__main__":
    unittest.main()
