#!/usr/bin/env python3
"""Manually align a moving tetrahedral volume to a fixed volume with PyVista.

Run from addi_func:
    conda run --no-capture-output -n cme-dt-env python \
        volmesh_dice_compare/manual_align_meshes.py

Only boundary surfaces are rendered. Saving transforms the original full volume,
never the display surface. Coordinates are assumed to be millimetres by default.

The window shows the fixed template in blue and moving object in orange.

Drag colored arrows to translate; arcs to rotate.
Press 5/6/7 to mirror along world X/Y/Z through the moving object's center.
Press c to center the object on the template.
Press f for fine adjustments.
Press u to undo; r to reset.
Press k to save; q to close.

"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pyvista as pv


CASE_DIR = Path(__file__).resolve().parent / "5"


def validate_transform(matrix):
    """Accept distance-preserving transforms, including mirror reflections."""
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(abs(np.linalg.det(matrix[:3, :3])), 1, atol=1e-7, rtol=0)):
        raise ValueError("Expected a finite 4x4 transform with rotation, translation or reflection only.")
    return matrix.copy()


def transform_points(points, matrix):
    return np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]


def read_volume(path):
    mesh = pv.read(path)
    if (not isinstance(mesh, pv.UnstructuredGrid) or not mesh.n_cells
            or not mesh.n_points or np.any(mesh.celltypes != pv.CellType.TETRA)):
        raise ValueError(f"{path}: expected a nonempty tetrahedral unstructured grid.")
    if not np.isfinite(mesh.points).all():
        raise ValueError(f"{path}: mesh coordinates contain NaN or infinity.")
    return mesh


def load_transform(path):
    data = json.loads(Path(path).read_text())
    return validate_transform(data["source_to_reference"] if isinstance(data, dict) else data)


def rotation_step(axis, degrees, pivot):
    """World-axis rotation around the current moving-volume center."""
    angle = np.deg2rad(degrees)
    first, second = (axis + 1) % 3, (axis + 2) % 3
    matrix = np.eye(4)
    matrix[first, first] = matrix[second, second] = np.cos(angle)
    matrix[first, second] = -np.sin(angle)
    matrix[second, first] = np.sin(angle)
    matrix[:3, 3] = np.asarray(pivot) - matrix[:3, :3] @ pivot
    return matrix


def mirror_step(axis, pivot):
    """Reflect across the world-axis-normal plane through the moving center.

    For example, X reflection maps x to 2*pivot_x-x and leaves y/z unchanged.
    """
    if axis not in (0, 1, 2):
        raise ValueError("Mirror axis must be 0 (X), 1 (Y) or 2 (Z).")
    matrix = np.eye(4)
    matrix[axis, axis] = -1
    matrix[axis, 3] = 2 * np.asarray(pivot)[axis]
    return matrix


def aligned_volume(mesh, matrix):
    matrix = validate_transform(matrix)
    aligned = mesh.copy(deep=True)
    aligned.points = transform_points(mesh.points, matrix)
    if np.linalg.det(matrix[:3, :3]) < 0:
        # A reflection reverses signed volume. Swap two local vertex slots in
        # every tetrahedron to retain its original orientation, cell ID and
        # vertex membership. Global point order and cell-data order are kept.
        cells = np.asarray(mesh.cells).reshape(-1, 5).copy()
        cells[:, [1, 2]] = cells[:, [2, 1]]
        aligned.cells = cells.ravel()
    # Transform arrays explicitly marked by VTK as directions; leave scalar labels
    # and arbitrary three-component data alone. Translation never affects them.
    rotation = matrix[:3, :3]
    for original, result in ((mesh.point_data, aligned.point_data),
                             (mesh.cell_data, aligned.cell_data)):
        for name in {original.active_vectors_name, original.active_normals_name} - {None}:
            result[name] = np.asarray(original[name]) @ rotation.T
        active_tensor = original.VTKObject.GetTensors()
        name = active_tensor.GetName() if active_tensor is not None else None
        if name is not None:
            tensors = np.asarray(original[name]).reshape(-1, 3, 3)
            result[name] = np.einsum("ij,njk,lk->nil", rotation, tensors, rotation).reshape(-1, 9)
    return aligned


def signed_tetra_volumes(mesh):
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    tetra = np.asarray(mesh.points, dtype=float)[cells]
    return np.einsum("ij,ij->i", np.cross(tetra[:, 1] - tetra[:, 0],
                                         tetra[:, 2] - tetra[:, 0]),
                     tetra[:, 3] - tetra[:, 0]) / 6


def verify_export(original, aligned, matrix):
    matrix = validate_transform(matrix)
    expected = aligned_volume(original, matrix)
    if (original.n_points != aligned.n_points or original.n_cells != aligned.n_cells
            or not np.array_equal(expected.cells, aligned.cells)
            or not np.array_equal(original.celltypes, aligned.celltypes)):
        raise ValueError("Export changed mesh topology or expected tetrahedral orientation.")
    if not np.allclose(aligned.points, transform_points(original.points, matrix), atol=1e-9, rtol=0):
        raise ValueError("Saved coordinates do not match the displayed transform.")
    for before, after in ((expected.point_data, aligned.point_data),
                           (expected.cell_data, aligned.cell_data),
                           (expected.field_data, aligned.field_data)):
        if set(before.keys()) != set(after.keys()):
            raise ValueError("Export changed mesh data-array names.")
        for name in before:
            numeric = np.asarray(before[name]).dtype.kind in "fc"
            if not np.array_equal(before[name], after[name], equal_nan=numeric):
                raise ValueError(f"Export changed array {name!r} unexpectedly.")
    native_volumes, saved_volumes = signed_tetra_volumes(original), signed_tetra_volumes(aligned)
    if not np.allclose(native_volumes, saved_volumes, atol=1e-9, rtol=1e-8):
        raise ValueError("Export did not preserve tetrahedral signed volumes.")
    return {"points": aligned.n_points, "tetrahedra": aligned.n_cells,
            "absolute_volume": float(np.abs(saved_volumes).sum()),
            "max_signed_cell_volume_error": float(np.max(np.abs(native_volumes - saved_volumes))),
            "linear_determinant": float(np.linalg.det(matrix[:3, :3])),
            "has_reflection": bool(np.linalg.det(matrix[:3, :3]) < 0),
            "tetra_local_vertex_swap": [0, 1] if np.linalg.det(matrix[:3, :3]) < 0 else None}


def save_alignment(mesh, matrix, fixed_path, moving_path, output_dir, *,
                   screenshot=None, overwrite=False, units="mm"):
    """Stage and read back the full volume before committing output files."""
    matrix = validate_transform(matrix)
    output_dir = Path(output_dir).resolve()
    names = ("subject_11_volmesh_aligned.vtk", "transform.json", "alignment_preview.png")
    protected = {Path(fixed_path).resolve(), Path(moving_path).resolve()}
    for name in names:
        target = output_dir / name
        if target.resolve() in protected:
            raise ValueError("An output path would overwrite an input mesh.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {target}; use another output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alignment-", dir=output_dir) as staging:
        stage = Path(staging)
        aligned_volume(mesh, matrix).save(stage / names[0], binary=True)
        report = verify_export(mesh, pv.read(stage / names[0]), matrix)
        metadata = {"source": str(Path(moving_path).resolve()),
                    "reference": str(Path(fixed_path).resolve()),
                    "source_to_reference": matrix.tolist(),
                    "convention": "p_fixed = A @ p_original_moving + t (column vectors); A is orthogonal",
                    "coordinate_units": units,
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "registration": {"method": "manual_rigid_with_reflection" if report["has_reflection"] else "manual_rigid",
                                     "has_reflection": report["has_reflection"]},
                    "validation": report}
        (stage / names[1]).write_text(json.dumps(metadata, indent=2) + "\n")
        if screenshot is not None:
            screenshot(stage / names[2])
        for name in names:
            if (stage / name).exists():
                (stage / name).replace(output_dir / name)
    return report


def available_output_dir(requested):
    """A new session does not replace files saved by an earlier session."""
    requested = Path(requested).resolve()
    candidate, index = requested, 1
    while candidate.exists() and any(candidate.iterdir()):
        candidate = requested.with_name(f"{requested.name}_{index:03d}")
        index += 1
    return candidate


class ManualAligner:
    def __init__(self, fixed, moving, args, *, off_screen=False):
        self.fixed, self.moving, self.args = fixed, moving, args
        self.original_center = np.asarray(moving.center, dtype=float)
        self.matrix = load_transform(args.transform_in) if args.transform_in else np.eye(4)
        self.history = [self.matrix.copy()]
        self.saved_matrix = None
        self.fine = False
        self.has_saved = False
        self.widget = None
        self.output_dir = available_output_dir(args.output_dir)
        self.message = "Ready. Drag an arrow to translate; drag an arc to rotate."
        self.plotter = pv.Plotter(window_size=(1400, 950), off_screen=off_screen,
                                  title="Manual mesh alignment: translation, rotation and mirror")
        self.plotter.set_background("#18212d")
        self.plotter.add_axes(viewport=(.85, .80, 1., 1.))
        print("Extracting boundary surfaces for display...", flush=True)
        self.fixed_actor = self.plotter.add_mesh(
            fixed.extract_surface(), color="#68baff", opacity=.3,
            scalars=None, show_scalar_bar=False, pickable=False, name="fixed")
        self.moving_actor = self.plotter.add_mesh(
            moving.extract_surface(), color="#ffac48", opacity=.8,
            scalars=None, show_scalar_bar=False, pickable=False, name="moving")
        self.moving_actor.user_matrix = self.matrix
        self.plotter.add_text(
            "FIXED TEMPLATE (blue)    MOVING OBJECT (orange)\n"
            "Mouse: drag colored arrows/arcs to move object; drag background to orbit camera\n"
            "Translate: Left/Right = X    Down/Up = Y    PageDown/PageUp = Z\n"
            "Rotate +/- about world X/Y/Z:  a/d    w/s    z/x\n"
            "Mirror: 5 = X    6 = Y    7 = Z (world axes, through moving center)\n"
            "f: coarse/fine    u: undo    r: original pose    c: center on template\n"
            "1/2/3: X/Y/Z view    4: isometric    v: fit view\n"
            "t/o: toggle template/object    g: handles    p: parallel/perspective\n"
            "k: SAVE full mesh + transform + image    q: close (does not save)",
            position="upper_left", font_size=11, color="white", name="help")
        self._add_keys()
        self._add_opacity_sliders()
        self.handles_visible = True
        self._rebuild_widget()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self._update_status()

    @property
    def pivot(self):
        return transform_points(self.original_center, self.matrix)

    def _rebuild_widget(self):
        # Recreate after external matrix changes so PyVista's internal drag-start
        # cache matches keyboard, undo and reset operations.
        if self.widget is not None:
            self.widget.remove()
            self.widget = None
        if self.handles_visible:
            self.widget = self.plotter.add_affine_transform_widget(
                self.moving_actor, origin=tuple(self.pivot), scale=.23,
                release_callback=self._mouse_release)

    def _mouse_release(self, matrix):
        try:
            self._commit(matrix, rebuild=False)
            if self.widget is not None:
                self.widget.origin = tuple(self.pivot)
        except ValueError as exc:
            self.moving_actor.user_matrix = self.matrix
            self._rebuild_widget()
            self.message = f"Movement rejected: {exc}"
            self._update_status()

    def _commit(self, matrix, *, rebuild=True):
        matrix = validate_transform(matrix)
        if not np.allclose(matrix, self.matrix, atol=1e-12, rtol=0):
            self.matrix = matrix
            self.history.append(matrix.copy())
            self.history = self.history[-201:]
            self.message = "Pose adjusted. Press k to save."
        self.moving_actor.user_matrix = self.matrix
        if rebuild:
            self._rebuild_widget()
        self._update_status()

    def _update_status(self):
        factor = .1 if self.fine else 1
        saved = self.saved_matrix is not None and np.allclose(
            self.matrix, self.saved_matrix, atol=1e-12, rtol=0)
        lines = [f"{'SAVED' if saved else 'UNSAVED'} | {'FINE' if self.fine else 'COARSE'} | "
                 f"translation {self.args.translation_step * factor:g} {self.args.units} | "
                 f"rotation {self.args.rotation_step * factor:g} deg",
                 "Reflection: " + ("YES (mirrored)" if np.linalg.det(self.matrix[:3, :3]) < 0 else "NO"),
                 "Moving center: " + np.array2string(self.pivot, precision=3),
                 "Original moving -> fixed transform:"]
        lines.extend("  " + " ".join(f"{value:10.4f}" for value in row) for row in self.matrix)
        lines.extend([self.message, f"Output: {self.output_dir}"])
        self.plotter.add_text("\n".join(lines), position="lower_left", font_size=10,
                              color="white", name="status")
        self.plotter.render()

    def translate(self, axis, sign):
        delta = np.eye(4)
        delta[axis, 3] = sign * self.args.translation_step * (.1 if self.fine else 1)
        self._commit(delta @ self.matrix)

    def rotate(self, axis, sign):
        degrees = sign * self.args.rotation_step * (.1 if self.fine else 1)
        self._commit(rotation_step(axis, degrees, self.pivot) @ self.matrix)

    def mirror(self, axis):
        self._commit(mirror_step(axis, self.pivot) @ self.matrix)
        self.message = f"Mirrored along world {'XYZ'[axis]} through moving center. Press k to save."
        self._update_status()

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.matrix = self.history[-1].copy()
            self.moving_actor.user_matrix = self.matrix
            self._rebuild_widget()
        self.message = "Undo complete." if len(self.history) > 1 else "At first retained pose."
        self._update_status()

    def center(self):
        delta = np.eye(4)
        delta[:3, 3] = np.asarray(self.fixed.center) - self.pivot
        self._commit(delta @ self.matrix)
        self.plotter.reset_camera()

    def toggle_fine(self):
        self.fine = not self.fine
        self._update_status()

    def toggle_handles(self):
        self.handles_visible = not self.handles_visible
        self._rebuild_widget()
        self.plotter.render()

    def toggle_visibility(self, actor):
        actor.visibility = not actor.visibility
        self.plotter.render()

    def toggle_projection(self):
        self.plotter.camera.parallel_projection = not self.plotter.camera.parallel_projection
        self.plotter.render()

    def save(self):
        try:
            self.message = "Saving and verifying the complete volume mesh..."
            self._update_status()
            report = save_alignment(
                self.moving, self.matrix, self.args.fixed, self.args.moving, self.output_dir,
                screenshot=lambda path: self.plotter.screenshot(str(path)),
                overwrite=self.has_saved, units=self.args.units)
            self.has_saved = True
            self.saved_matrix = self.matrix.copy()
            self.message = f"Saved and verified {report['tetrahedra']:,} tetrahedra."
            print(f"{self.message}\nOutput: {self.output_dir}", flush=True)
        except Exception as exc:
            self.message = f"SAVE FAILED: {exc}"
            print(self.message, flush=True)
        self._update_status()

    def _add_keys(self):
        callbacks = {
            "Left": lambda: self.translate(0, -1), "Right": lambda: self.translate(0, 1),
            "Down": lambda: self.translate(1, -1), "Up": lambda: self.translate(1, 1),
            "Next": lambda: self.translate(2, -1), "Prior": lambda: self.translate(2, 1),
            "a": lambda: self.rotate(0, -1), "d": lambda: self.rotate(0, 1),
            "w": lambda: self.rotate(1, -1), "s": lambda: self.rotate(1, 1),
            "z": lambda: self.rotate(2, -1), "x": lambda: self.rotate(2, 1),
            "f": self.toggle_fine, "u": self.undo, "r": lambda: self._commit(np.eye(4)),
            "c": self.center, "k": self.save, "g": self.toggle_handles,
            "t": lambda: self.toggle_visibility(self.fixed_actor),
            "o": lambda: self.toggle_visibility(self.moving_actor),
            "p": self.toggle_projection, "v": self.plotter.reset_camera,
            "1": self.plotter.view_yz, "2": self.plotter.view_xz,
            "3": self.plotter.view_xy, "4": self.plotter.view_isometric,
            "5": lambda: self.mirror(0), "6": lambda: self.mirror(1),
            "7": lambda: self.mirror(2),
        }
        # Replace VTK/PyVista defaults (e.g. s=wireframe, w=surface, r=camera reset).
        for key, callback in callbacks.items():
            self.plotter.clear_events_for_key(key, raise_on_missing=False)
            self.plotter.add_key_event(key, callback)
        self._bound_keys = set(callbacks)
        # vtkInteractorStyle also handles CharEvent independently of PyVista's
        # KeyPressEvent callbacks (w=wireframe, r=camera reset, f=fly-to).
        # Abort only our assigned character events; retain q/Escape handling.
        interactor = self.plotter.iren.interactor
        tag = interactor.AddObserver("CharEvent", self._char_event, 1.)
        self._char_command = interactor.GetCommand(tag)

    def _char_event(self, interactor, _event):
        self._char_command.SetAbortFlag(interactor.GetKeySym() in self._bound_keys)

    def _add_opacity_sliders(self):
        for actor, title, y in ((self.fixed_actor, "Template opacity", .52),
                                 (self.moving_actor, "Object opacity", .32)):
            def change(value, actor=actor):
                actor.prop.opacity = value
            self.plotter.add_slider_widget(
                change, [0, 1], value=actor.prop.opacity, title=title,
                pointa=(.76, y), pointb=(.96, y), color="white", fmt="%.2f",
                title_height=.018, slider_width=.025, interaction_event="always")

    def show(self):
        print(f"Output directory: {self.output_dir}\nPress k in the window to save; q to close.", flush=True)
        self.plotter.show()
        if self.saved_matrix is None or not np.allclose(self.matrix, self.saved_matrix, atol=1e-12, rtol=0):
            print("Window closed with an unsaved pose. Only explicit saves are written.", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed", type=Path, default=CASE_DIR / "subject0081_mesh_all.vtk")
    parser.add_argument("--moving", type=Path, default=CASE_DIR / "subject_11_volmesh.vtk")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "manual_alignment")
    parser.add_argument("--transform-in", type=Path, help="Resume a saved 4x4 transform JSON")
    parser.add_argument("--translation-step", type=float, default=1., help="Coarse translation increment (default: 1)")
    parser.add_argument("--rotation-step", type=float, default=1., help="Coarse rotation increment in degrees (default: 1)")
    parser.add_argument("--units", default="mm", help="Coordinate-unit label only; does not rescale inputs")
    parser.add_argument("--preview-only", type=Path, metavar="PNG", help="Render one off-screen preview and exit")
    args = parser.parse_args(argv)
    for value in (args.translation_step, args.rotation_step):
        if not np.isfinite(value) or value <= 0:
            parser.error("Translation and rotation steps must be positive finite numbers.")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"Fixed:  {args.fixed}\nMoving: {args.moving}", flush=True)
    fixed, moving = read_volume(args.fixed), read_volume(args.moving)
    print(f"Loaded {fixed.n_cells:,} fixed and {moving.n_cells:,} moving tetrahedra.", flush=True)
    app = ManualAligner(fixed, moving, args, off_screen=args.preview_only is not None)
    if args.preview_only:
        if args.preview_only.resolve() in {args.fixed.resolve(), args.moving.resolve()}:
            raise ValueError("Preview path cannot overwrite an input mesh.")
        args.preview_only.parent.mkdir(parents=True, exist_ok=True)
        app.plotter.screenshot(str(args.preview_only))
        app.plotter.close()
        print(f"Preview: {args.preview_only}", flush=True)
    else:
        app.show()


if __name__ == "__main__":
    main()
