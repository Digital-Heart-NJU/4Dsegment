#!/usr/bin/env python3
"""Choose a movable cutting plane with a lightweight preview, then measure Dice.

Only subject0081 is clipped. The saved aligned subject 11 volume is used whole,
without registration. Both original files are read only. Coordinates are mm.

    conda run --no-capture-output -n cme-dt-env python \
        volmesh_dice_compare/interactive_coords_z_dice.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue

import numpy as np
import pyvista as pv
import vtk

CASE_DIR = Path(__file__).resolve().parent / "5"
TAG_CODES = {0: 1, 1: 2}  # zero is tissue in the inputs, background in voxel masks


def hash_files(paths):
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def encode_volume(mesh):
    probe = mesh.copy(deep=False)
    tags = np.asarray(mesh.cell_data["elemTag"])
    codes = np.zeros(mesh.n_cells, dtype=np.uint8)
    for tag, code in TAG_CODES.items():
        codes[tags == tag] = code
    if np.any(codes == 0):
        raise ValueError("Unexpected tissue tag while encoding the volume.")
    probe.clear_data()
    probe.cell_data["dice_code"] = codes
    return probe


def grid_definition(template, aligned, spacing):
    bounds = np.array([template.bounds, aligned.bounds]).reshape(2, 3, 2)
    low = np.floor(bounds[:, :, 0].min(axis=0) / spacing) * spacing - spacing
    high = np.ceil(bounds[:, :, 1].max(axis=0) / spacing) * spacing + spacing
    return low, np.ceil((high - low) / spacing).astype(int)


def sample_volume(mesh, low, shape, spacing):
    centers = pv.ImageData(dimensions=tuple(shape), spacing=(spacing,) * 3,
                           origin=low + spacing / 2)
    sampled = centers.sample(mesh, tolerance=1e-8, locator="static_cell", mark_blank=False,
                             pass_point_data=False, pass_cell_data=False, pass_field_data=False)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    codes = np.asarray(sampled.point_data["dice_code"], dtype=np.uint8).copy()
    codes[~valid] = 0
    return codes


def score_masks(template, aligned, spacing):
    rows = []
    for region, code in (("all_tissue", None), ("tag_0", 1), ("tag_1", 2)):
        a = template != 0 if code is None else template == code
        b = aligned != 0 if code is None else aligned == code
        na, nb, intersection = int(a.sum()), int(b.sum()), int(np.count_nonzero(a & b))
        rows.append({"spacing_mm": float(spacing), "region": region,
                     "dice": 2 * intersection / (na + nb) if na + nb else None,
                     "template_volume_mm3": na * spacing ** 3,
                     "aligned_volume_mm3": nb * spacing ** 3,
                     "intersection_volume_mm3": intersection * spacing ** 3,
                     "template_voxels": na, "aligned_voxels": nb,
                     "intersection_voxels": intersection})
    return rows


def read_mesh(path):
    mesh = pv.read(path)
    if (not isinstance(mesh, pv.UnstructuredGrid) or not mesh.n_cells
            or np.any(mesh.celltypes != pv.CellType.TETRA)
            or not np.isfinite(mesh.points).all()):
        raise ValueError(f"{path}: expected a finite, nonempty tetrahedral volume.")
    if "elemTag" not in mesh.cell_data:
        raise ValueError(f"{path}: missing cell array elemTag.")
    tags = np.asarray(mesh.cell_data["elemTag"])
    if tags.shape != (mesh.n_cells,) or not np.isin(tags, [0, 1]).all():
        raise ValueError(f"{path}: elemTag must contain only numeric tags 0 and 1.")
    return mesh


def load_inputs(args):
    paths = list(dict.fromkeys([args.template.resolve(), args.aligned.resolve()]))
    if args.selection_in:
        paths.append(args.selection_in.resolve())
    hashes = hash_files(paths)
    template, aligned = read_mesh(args.template), read_mesh(args.aligned)
    tags = np.asarray(template.cell_data["elemTag"]).copy()
    template.clear_data()
    template.points = np.asarray(template.points, dtype=np.float64)
    template.cell_data["elemTag"] = tags
    template.cell_data["original_cell_id"] = np.arange(template.n_cells, dtype=np.int64)
    return template, aligned, {"paths": paths, "hashes": hashes}


def compact_tetrahedra(mesh):
    """Split clipped wedges into tetrahedra and discard unreferenced points."""
    if not mesh.n_cells:
        return pv.UnstructuredGrid()
    tetra = mesh.triangulate()
    tetra = tetra.extract_cells(np.arange(tetra.n_cells))
    if np.any(tetra.celltypes != pv.CellType.TETRA):
        raise ValueError("Clipping produced unexpected non-tetrahedral cells.")
    return tetra


def validate_plane(spec):
    origin = np.asarray(spec["origin_mm"], dtype=float)
    normal = np.asarray(spec["normal"], dtype=float)
    side = spec.get("keep_side", "negative")
    if (origin.shape != (3,) or normal.shape != (3,) or not np.isfinite(origin).all()
            or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 1e-12
            or side not in ("negative", "positive")):
        raise ValueError("Plane needs a finite origin, nonzero normal, and negative/positive keep side.")
    return {"origin_mm": origin.tolist(), "normal": (normal / np.linalg.norm(normal)).tolist(),
            "keep_side": side}


def input_plane(args, template):
    spec = {"origin_mm": list(template.center), "normal": [0, 0, 1], "keep_side": "negative"}
    if args.selection_in:
        data = json.loads(args.selection_in.read_text())
        data = data.get("selection", data)
        if data.get("method") != "manual_cutting_plane":
            raise ValueError("--selection-in must contain a saved manual cutting-plane selection.")
        spec = data["plane"]
    for argument, key in ((args.origin, "origin_mm"), (args.normal, "normal"), (args.keep_side, "keep_side")):
        if argument is not None:
            spec = dict(spec, **{key: argument})
    return validate_plane(spec)


def clip_template(template, spec):
    """Full-resolution volume clipping, called only after confirmation."""
    spec = validate_plane(spec)
    distances = (np.asarray(template.points, dtype=float) - spec["origin_mm"]) @ spec["normal"]
    if spec["keep_side"] == "positive":
        distances = -distances
    if distances.max() <= 0:
        return template, pv.UnstructuredGrid()
    if distances.min() >= 0:
        return pv.UnstructuredGrid(), template
    selected, removed = template.clip(normal=spec["normal"], origin=spec["origin_mm"],
                                      invert=spec["keep_side"] == "negative", return_clipped=True)
    return compact_tetrahedra(selected), compact_tetrahedra(removed)


def preview_surface(mesh, target_faces):
    """Build display-only geometry once; never use its triangles for Dice."""
    surface = mesh.extract_surface().triangulate()
    surface.clear_data()
    if surface.n_cells > target_faces:
        surface = surface.decimate_pro(1 - target_faces / surface.n_cells,
                                        preserve_topology=True, boundary_vertex_deletion=False)
    return surface


def volume_stats(mesh):
    if not mesh.n_cells:
        return {"points": 0, "tetrahedra": 0, "volume_mm3": 0.,
                "tag_volumes_mm3": {"0": 0., "1": 0.}}
    volumes = np.abs(np.asarray(mesh.compute_cell_sizes(length=False, area=False)["Volume"]))
    if not np.isfinite(volumes).all():
        raise ValueError("Nonfinite tetrahedral volume.")
    tags = np.asarray(mesh.cell_data["elemTag"])
    return {"points": mesh.n_points, "tetrahedra": mesh.n_cells,
            "volume_mm3": float(volumes.sum()),
            "tag_volumes_mm3": {str(tag): float(volumes[tags == tag].sum()) for tag in TAG_CODES}}


def check_partition(native_stats, selected_stats, removed_stats):
    for label in (None, "0", "1"):
        def get(stats):
            return stats["volume_mm3"] if label is None else stats["tag_volumes_mm3"][label]
        if not np.isclose(get(native_stats), get(selected_stats) + get(removed_stats), rtol=1e-6, atol=1e-6):
            raise ValueError(f"Selected + removed volume does not conserve template volume (tag {label}).")


def convergence_report(rows, tolerance):
    spacings = sorted({row["spacing_mm"] for row in rows})
    if len(spacings) < 2:
        return {"checked": False, "reason": "Only one spacing requested."}
    fine, coarse = spacings[:2]
    changes = {}
    for region in ("all_tissue", "tag_0", "tag_1"):
        scores = {row["spacing_mm"]: row["dice"] for row in rows if row["region"] == region}
        changes[region] = abs(scores[fine] - scores[coarse]) if scores[fine] is not None and scores[coarse] is not None else None
    defined = [value for value in changes.values() if value is not None]
    return {"checked": True, "spacings_mm": [coarse, fine], "tolerance": tolerance,
            "absolute_dice_changes": changes,
            "within_tolerance": all(value <= tolerance for value in defined) if defined else None,
            "note": "Resolution sensitivity check, not a proof of exact volume overlap."}


def calculate_results(selected, aligned, args, progress=lambda message: None):
    """Compare clipped template to the entire aligned mesh on union grids."""
    grids = []
    for spacing in sorted(set(args.spacing), reverse=True):
        low, shape = grid_definition(selected, aligned, spacing)
        count = int(np.prod(shape, dtype=np.int64))
        if count > args.max_grid_points:
            raise ValueError(f"Spacing {spacing:g} mm requires {count:,} grid points; limit is {args.max_grid_points:,}.")
        grids.append((spacing, low, shape, count))
    encoded_template = encode_volume(selected)
    encoded_aligned = encode_volume(aligned)
    rows, grid_metadata = [], []
    for spacing, low, shape, count in grids:
        progress(f"Sampling selected template at {spacing:g} mm ({count:,} voxel centers)...")
        template_mask = sample_volume(encoded_template, low, shape, spacing)
        progress(f"Sampling complete aligned subject 11 at {spacing:g} mm...")
        aligned_mask = sample_volume(encoded_aligned, low, shape, spacing)
        rows.extend(score_masks(template_mask, aligned_mask, spacing))
        grid_metadata.append({"spacing_mm": spacing, "grid_low_corner_mm": low.tolist(),
                              "dimensions": shape.tolist(), "voxel_centers": count})
        del template_mask, aligned_mask
    return rows, grid_metadata


def available_directory(path):
    path = Path(path).resolve()
    candidate, index = path, 1
    while candidate.exists() and any(candidate.iterdir()):
        candidate = path.with_name(f"{path.name}_{index:03d}")
        index += 1
    return candidate


def run_calculation(template, aligned, args, provenance, plane, output, progress=lambda message: None):
    plane = validate_plane(plane)
    progress("Clipping original tetrahedral template at the confirmed plane...")
    selected, removed = clip_template(template, plane)
    progress("Checking full-resolution retained and removed volumes...")
    provenance["template_stats"] = volume_stats(template)
    provenance["aligned_stats"] = volume_stats(aligned)
    selected_stats, removed_stats = volume_stats(selected), volume_stats(removed)
    if selected_stats["volume_mm3"] <= 0:
        raise ValueError("The retained template has no positive volume. Move the plane or flip the kept side.")
    check_partition(provenance["template_stats"], selected_stats, removed_stats)
    rows, grids = calculate_results(selected, aligned, args, progress)
    progress("Saving selected volume and checking original input files...")
    selected.save(output / "template_selected.vtu", binary=True)
    if removed.n_cells:
        removed.save(output / "template_removed.vtu", binary=True)
    unchanged = hash_files(provenance["paths"]) == provenance["hashes"]
    if not unchanged:
        raise RuntimeError("An input file changed during this session; results were not accepted.")
    selection = {"method": "manual_cutting_plane", "plane": plane,
                 "template": str(args.template.resolve()), "aligned": str(args.aligned.resolve()),
                 "clipping": "Only the template is clipped by the confirmed plane; crossing tetrahedra are split.",
                 "keep_rule": "dot(normal, point-origin) " + ("<= 0" if plane["keep_side"] == "negative" else ">= 0"),
                 "preview": "Simplified exterior with renderer clipping; original tetrahedra used for Dice.",
                 "aligned_volume": "Entire saved aligned mesh; no further transform or clipping.",
                 "selected_template": selected_stats, "removed_template": removed_stats}
    result = {"selection": selection, "scores": rows, "grids": grids,
              "aligned_full_volume": provenance["aligned_stats"],
              "template_original_volume": provenance["template_stats"],
              "convergence": convergence_report(rows, args.convergence_tolerance),
              "voxel_label_codes": {"background": 0, "tag_0": 1, "tag_1": 2},
              "overall_definition": "Dice of the union of tags 0 and 1, not the mean of tag-wise Dice.",
              "sampling": "Actual tetrahedral containment at shared isotropic voxel centers covering both volumes.",
              "inputs_unchanged": unchanged, "input_sha256": provenance["hashes"],
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    with (output / "dice_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # results.json is the completion marker; failure leaves no completed report.
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    progress(f"Complete. Results: {output}")
    return result


def button_texture(label, color):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (350, 52), color)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    draw.text((175, 26), label, fill="white", font=font, anchor="mm")
    texture = pv.ImageData(dimensions=(350, 52, 1))
    texture.point_data["RGB"] = np.asarray(image)[::-1].reshape(-1, 3).copy()
    return texture


class SelectionWindow:
    def __init__(self, template, aligned, args, provenance, *, off_screen=False):
        self.template, self.aligned, self.args, self.provenance = template, aligned, args, provenance
        self.plane = input_plane(args, template)
        self.initial_plane = dict(self.plane)
        self.history = [dict(self.plane)]
        self.center = np.asarray(template.center)
        self.ready = self.busy = False
        self.plane_visible = True
        self.result = self.result_signature = self.future = self.output = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.messages = queue.Queue()
        self.message = "Move/tilt the plane, inspect BLUE retained tissue, then confirm."
        self.plotter = pv.Plotter(window_size=(1450, 980), off_screen=off_screen,
                                  title="Fast cutting-plane preview and volume Dice")
        self.plotter.set_background("#18212d")
        self.plotter.add_axes(viewport=(.86, .81, 1., .99))
        print("Preparing simplified display surfaces once...", flush=True)
        self.template_surface = preview_surface(template, args.preview_faces)
        self.aligned_surface = preview_surface(aligned, args.preview_faces)
        self.selected_actor = self.plotter.add_mesh(self.template_surface, color="#68baff", opacity=.75,
                                                     show_scalar_bar=False, pickable=False, name="selected")
        self.removed_actor = self.plotter.add_mesh(self.template_surface, color="#ff5364", opacity=.20,
                                                    show_scalar_bar=False, pickable=False, name="removed")
        self.aligned_actor = self.plotter.add_mesh(self.aligned_surface, color="#ffac48", opacity=.45,
                                                    show_scalar_bar=False, pickable=False, name="aligned")
        # VTK mapper planes retain their positive halfspace. Update these same
        # plane objects while interacting; do not run a CPU clip or rebuild meshes.
        self.keep_clip, self.remove_clip = vtk.vtkPlane(), vtk.vtkPlane()
        self.selected_actor.mapper.AddClippingPlane(self.keep_clip)
        self.removed_actor.mapper.AddClippingPlane(self.remove_clip)
        self.plotter.add_text(
            "BLUE: retained template    RED: removed template    ORANGE: entire subject 11\n"
            "Move the plane with its center or the position slider; drag its arrow to tilt.\n"
            "f: flip kept side    x/y/z: orient plane    u: undo    r: reset    c: calculate\n"
            "g: show/hide plane    t: show/hide removed    o: show/hide subject 11\n"
            "1/2/3/4: camera views    v: fit    q: close\n"
            "Preview uses simplified exterior surfaces; full tetrahedra are clipped only after confirmation.",
            position="upper_left", font_size=11, color="white", name="help")
        self.plane_widget = self.plotter.add_plane_widget(
            self._plane_changed, normal=self.plane["normal"], origin=self.plane["origin_mm"],
            bounds=template.bounds, factor=1.1, color="#fff065", outline_translation=False,
            origin_translation=True, normal_rotation=True, implicit=True, test_callback=False,
            interaction_event="always", outline_opacity=0)
        self.plane_widget.GetPlaneProperty().SetOpacity(.08)
        self.plane_widget.GetPlaneProperty().SetColor(1., .94, .40)
        self.plane_widget.SetDrawPlane(True)
        self.plane_widget.AddObserver("EndInteractionEvent", self._release)
        if hasattr(self.plane_widget, "SetOutsideBounds"):
            self.plane_widget.SetOutsideBounds(True)
        diagonal = np.linalg.norm(np.array(template.bounds)[1::2] - np.array(template.bounds)[::2])
        self.offset_slider = self.plotter.add_slider_widget(
            self._offset_changed, [-diagonal, diagonal], value=self._offset(), title="Plane position (mm)",
            pointa=(.74, .65), pointb=(.97, .65), color="white", fmt="%.2f", title_height=.019,
            interaction_event="always")
        self.offset_slider.AddObserver("EndInteractionEvent", self._release)
        for actor, title, y in ((self.selected_actor, "Template opacity", .48),
                                 (self.aligned_actor, "Subject 11 opacity", .32)):
            def change(value, actor=actor):
                actor.prop.opacity = value
            self.plotter.add_slider_widget(change, [0, 1], value=actor.prop.opacity, title=title,
                                            pointa=(.75, y), pointb=(.96, y), color="white", title_height=.018)
        self._buttons = []
        self._add_button("Flip Kept Side", (.73, .22), self.flip, "#a04b26")
        self._add_button("Reset Plane", (.73, .15), self.reset, "#526178")
        self._add_button("Confirm & Calculate Dice", (.73, .08), self.confirm, "#23855c")
        self.plotter.iren.add_observer("ConfigureEvent", self._place_buttons)
        callbacks = {"f": self.flip, "u": self.undo, "r": self.reset, "g": self.toggle_plane,
                     "c": self.confirm, "x": lambda: self.orient(0), "y": lambda: self.orient(1),
                     "z": lambda: self.orient(2), "1": self.plotter.view_yz, "2": self.plotter.view_xz,
                     "3": self.plotter.view_xy, "4": self.plotter.view_isometric,
                     "v": self.plotter.reset_camera, "t": lambda: self.toggle_actor(self.removed_actor),
                     "o": lambda: self.toggle_actor(self.aligned_actor)}
        for key, callback in callbacks.items():
            self.plotter.clear_events_for_key(key, raise_on_missing=False)
            self.plotter.add_key_event(key, callback)
        self._bound_keys = set(callbacks)
        tag = self.plotter.iren.interactor.AddObserver("CharEvent", self._char_event, 1.)
        self._char_command = self.plotter.iren.interactor.GetCommand(tag)
        self.plotter.iren.add_observer("TimerEvent", self._poll)
        self.timer = None
        self.ready = True
        self.update_preview()
        self.plotter.view_isometric()
        self.plotter.reset_camera()

    def _add_button(self, label, position, callback, color):
        rep = vtk.vtkTexturedButtonRepresentation2D()
        rep.SetNumberOfStates(2)
        for state in (0, 1):
            rep.SetButtonTexture(state, button_texture(label, color))
        rep.SetPlaceFactor(1)
        widget = vtk.vtkButtonWidget()
        widget.SetInteractor(self.plotter.iren.interactor)
        widget.SetCurrentRenderer(self.plotter.renderer)
        widget.SetRepresentation(rep)
        def clicked(widget, event):
            widget.GetRepresentation().SetState(0)
            callback()
        widget.AddObserver(vtk.vtkCommand.StateChangedEvent, clicked)
        self._buttons.append((widget, position))
        self.plotter.button_widgets.append(widget)
        self._place_buttons()
        widget.On()

    def _place_buttons(self, *_):
        width, height = self.plotter.window_size
        for widget, (x, y) in self._buttons:
            left, bottom = x * width, y * height
            widget.GetRepresentation().PlaceWidget([left, left + .24 * width, bottom, bottom + .05 * height, 0, 0])

    def _char_event(self, interactor, _event):
        self._char_command.SetAbortFlag(interactor.GetKeySym() in self._bound_keys)

    def selection_signature(self):
        return json.dumps(self.plane, sort_keys=True)

    def _offset(self):
        return float((np.asarray(self.plane["origin_mm"]) - self.center) @ self.plane["normal"])

    def _record(self):
        if self.history[-1] != self.plane:
            self.history.append(dict(self.plane))
            self.history = self.history[-201:]

    def _release(self, *_):
        if self.ready and not self.busy:
            self._record()
            self.plane_widget.SetDrawPlane(True)
            self.plotter.render()

    def _plane_changed(self, normal, origin):
        if self.ready and not self.busy:
            self.plane = validate_plane(dict(self.plane, normal=normal, origin_mm=origin))
            self.update_preview()

    def _offset_changed(self, value):
        if not self.ready:
            return
        if self.busy:
            self.offset_slider.GetRepresentation().SetValue(self._offset())
            return
        origin = np.asarray(self.plane["origin_mm"]) + np.asarray(self.plane["normal"]) * (value - self._offset())
        self.plane = validate_plane(dict(self.plane, origin_mm=origin))
        self.plane_widget.SetOrigin(origin)
        self.update_preview()

    def set_plane(self, spec, *, record=True):
        if self.busy:
            return
        self.plane = validate_plane(spec)
        self.plane_widget.SetNormal(self.plane["normal"])
        self.plane_widget.SetOrigin(self.plane["origin_mm"])
        if record:
            self._record()
        self.update_preview()

    def flip(self):
        self.set_plane(dict(self.plane, keep_side="positive" if self.plane["keep_side"] == "negative" else "negative"))

    def orient(self, axis):
        self.set_plane(dict(self.plane, normal=np.eye(3)[axis]))

    def reset(self):
        self.set_plane(self.initial_plane)

    def undo(self):
        if not self.busy and len(self.history) > 1:
            self.history.pop()
            self.set_plane(self.history[-1], record=False)

    def toggle_plane(self):
        if not self.busy:
            self.plane_visible = not self.plane_visible
            self.plane_widget.SetEnabled(self.plane_visible)
            self.plotter.render()

    def toggle_actor(self, actor):
        actor.visibility = not actor.visibility
        self.plotter.render()

    def update_preview(self):
        if not self.ready:
            return
        normal = np.asarray(self.plane["normal"])
        keep_normal = normal if self.plane["keep_side"] == "positive" else -normal
        self.keep_clip.SetOrigin(self.plane["origin_mm"])
        self.keep_clip.SetNormal(keep_normal)
        self.remove_clip.SetOrigin(self.plane["origin_mm"])
        self.remove_clip.SetNormal(-keep_normal)
        self.offset_slider.GetRepresentation().SetValue(self._offset())
        self._status()

    def _status(self):
        lines = ["Keep BLUE template tissue; discard RED. Subject 11 stays complete.",
                 "Plane origin: " + np.array2string(np.asarray(self.plane["origin_mm"]), precision=2)
                 + "   normal: " + np.array2string(np.asarray(self.plane["normal"]), precision=3),
                 f"Kept side: {self.plane['keep_side']} | Preview: {self.template_surface.n_cells:,} template triangles"]
        if self.result is not None:
            lines.append("CURRENT RESULT" if self.selection_signature() == self.result_signature else "PREVIOUS RESULT - plane changed; confirm again")
            for spacing in sorted({r["spacing_mm"] for r in self.result["scores"]}, reverse=True):
                scores = {r["region"]: r["dice"] for r in self.result["scores"] if r["spacing_mm"] == spacing}
                def number(value):
                    return "undefined" if value is None else f"{value:.6f}"
                lines.append(f"{spacing:g} mm: tag 0={number(scores['tag_0'])}  tag 1={number(scores['tag_1'])}  overall={number(scores['all_tissue'])}")
            check = self.result["convergence"]
            if check["checked"]:
                lines.append("Resolution check: " + ("within tolerance" if check["within_tolerance"] else "inspect resolution sensitivity"))
        lines.append(self.message)
        if self.output:
            lines.append(f"Output: {self.output}")
        self.plotter.add_text("\n".join(lines), position="lower_left", font_size=10,
                              color="white", name="status", render=False)
        self.plotter.render()

    def _progress(self, message):
        print(message, flush=True)
        self.messages.put(message)

    def confirm(self, *, synchronous=False):
        if self.busy or not self.ready:
            return
        self._record()
        self.busy = True
        self.confirmed_plane = dict(self.plane)
        self.confirmed_signature = self.selection_signature()
        try:
            self.output = available_directory(self.args.output_dir)
            self.output.mkdir(parents=True, exist_ok=True)
            self.message = "Plane confirmed. Clipping the full-resolution template and calculating Dice..."
            self._status()
            self.plotter.screenshot(str(self.output / "selection_preview.png"))
            self.plane_widget.Off()
            arguments = (self.template, self.aligned, self.args, self.provenance,
                         self.confirmed_plane, self.output, self._progress)
            if synchronous:
                self.result = run_calculation(*arguments)
                self.result_signature = self.confirmed_signature
                self.message = "Calculation complete. Full-resolution results saved."
                self.busy = False
                self.plane_widget.SetEnabled(self.plane_visible)
                self._status()
            else:
                if self.timer is None:
                    self.timer = self.plotter.iren.create_timer(250)
                self.future = self.executor.submit(run_calculation, *arguments)
        except Exception as exc:
            self.busy = False
            self.plane_widget.SetEnabled(self.plane_visible)
            self.message = f"CALCULATION FAILED: {exc}"
            self._status()
            if synchronous:
                raise
            print(self.message, flush=True)

    def _poll(self, *_):
        changed = False
        while not self.messages.empty():
            self.message = self.messages.get_nowait()
            changed = True
        if self.future is not None and self.future.done():
            try:
                self.result = self.future.result()
                self.result_signature = self.confirmed_signature
                self.message = "Calculation complete. Full-resolution results saved."
            except Exception as exc:
                self.message = f"CALCULATION FAILED: {exc}"
                print(self.message, flush=True)
            self.future = None
            self.busy = False
            self.plane_widget.SetEnabled(self.plane_visible)
            changed = True
        if changed:
            self._status()

    def show(self):
        print("Adjust the plane and Flip Kept Side if needed. Only Confirm & Calculate Dice processes the volume.", flush=True)
        try:
            self.plotter.show()
        finally:
            self.executor.shutdown(wait=True)
            if self.future is not None:
                self.future.result()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, default=CASE_DIR / "subject0081_mesh_all.vtk")
    parser.add_argument("--aligned", type=Path, default=CASE_DIR / "manual_alignment_002/subject_11_volmesh_aligned.vtk")
    parser.add_argument("--normal", nargs=3, type=float, help="Initial plane normal NX NY NZ; default world Z")
    parser.add_argument("--origin", nargs=3, type=float, help="Initial plane point X Y Z in mm; default template center")
    parser.add_argument("--keep-side", choices=["negative", "positive"], help="Keep dot(normal,point-origin) <=0 or >=0; default negative")
    parser.add_argument("--selection-in", type=Path, help="Restore plane from saved selection.json or results.json")
    parser.add_argument("--preview-faces", type=int, default=30000, help="Display triangle target per mesh; never affects Dice (default 30000)")
    parser.add_argument("--spacing", type=float, nargs="+", default=[1., .5])
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "dice_manual_plane")
    parser.add_argument("--max-grid-points", type=int, default=80000000)
    parser.add_argument("--convergence-tolerance", type=float, default=.005)
    parser.add_argument("--preview-only", type=Path, metavar="PNG", help="Write an off-screen preview, without calculating Dice")
    parser.add_argument("--headless", action="store_true", help="Confirm --normal and --origin, or --selection-in, and calculate off-screen")
    args = parser.parse_args(argv)
    if args.normal is not None and (not np.isfinite(args.normal).all() or np.linalg.norm(args.normal) <= 1e-12):
        parser.error("--normal needs three finite values and nonzero length.")
    if args.origin is not None and not np.isfinite(args.origin).all():
        parser.error("--origin must be finite.")
    if args.preview_faces < 100:
        parser.error("--preview-faces must be at least 100.")
    if args.headless and ((not args.selection_in and (args.normal is None or args.origin is None)) or args.preview_only):
        parser.error("--headless requires --normal plus --origin, or --selection-in; do not combine with --preview-only.")
    if (not all(np.isfinite(s) and s > 0 for s in args.spacing) or args.max_grid_points < 1
            or not np.isfinite(args.convergence_tolerance) or args.convergence_tolerance < 0):
        parser.error("Spacings and grid limit must be positive; convergence tolerance must be finite and nonnegative.")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"Template: {args.template}\nFull aligned subject 11: {args.aligned}\nSelection: movable cutting plane (fast display-only preview)", flush=True)
    template, aligned, provenance = load_inputs(args)
    window = SelectionWindow(template, aligned, args, provenance,
                             off_screen=args.headless or args.preview_only is not None)
    if args.headless or args.preview_only:
        try:
            if args.preview_only:
                if args.preview_only.resolve() in provenance["paths"]:
                    raise ValueError("Preview cannot overwrite an input file.")
                args.preview_only.parent.mkdir(parents=True, exist_ok=True)
                window.plotter.screenshot(str(args.preview_only))
                print(f"Preview only: {args.preview_only}", flush=True)
            else:
                window.confirm(synchronous=True)
                if window.result is None:
                    raise ValueError(window.message)
                for row in window.result["scores"]:
                    print(f"{row['spacing_mm']:g} mm  {row['region']}: {row['dice']}", flush=True)
        finally:
            window.executor.shutdown(wait=True)
            window.plotter.close()
    else:
        window.show()


if __name__ == "__main__":
    main()
