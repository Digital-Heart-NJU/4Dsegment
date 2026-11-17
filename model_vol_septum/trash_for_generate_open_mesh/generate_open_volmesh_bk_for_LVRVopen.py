#!/usr/bin/env python3
"""New volumetric pipeline, implemented through Step 08.

Step 01: copy original surfaces without changing their geometry or arrays.
Step 02: MeshFix repair, closed-surface validation and LV/RV surface labels.
Step 03: move closed LVepi/RVepi outward, keeping both endocardia fixed and clearance >=3 mm.
Step 04: extract the outer envelope and append both complete inner surfaces.
Step 05: MeshFix-clean the combined surface, retaining both blood-pool cavities.
Step 06: TetGen mesh of the connected myocardium, then LVepi-based LV/RV tags.
Step 07: detect the subject base and save common cutting-plane previews.
Step 08: clip the connected myocardium at the saved plane and retag it.

Example:
    python model_vol_septum/generate_open_volmesh.py \
        --subject-dir data/subject_11 --frames ED,ES

Input:  SUBJECT/vtks/{LVendo,LVepi,RV,RVepi}_{ED,ES}.vtk
Output: SUBJECT/volmesh_pipline/step_01_load_surf/SUBJECT_FRAME_SURFACE.vtk
        SUBJECT/volmesh_pipline/step_02_surf_clean/SUBJECT_FRAME_SURFACE.vtk
        SUBJECT/volmesh_pipline/step_03_move_RVepi/SUBJECT_FRAME_RVepi_moved.vtk
        SUBJECT/volmesh_pipline/step_03_move_RVepi/SUBJECT_FRAME_LVepi_moved.vtk
        SUBJECT/volmesh_pipline/step_04_extract_LVRV_surf/SUBJECT_FRAME_LVRV_surf.vtk
        SUBJECT/volmesh_pipline/step_05_LVRV_surf_clean/SUBJECT_FRAME_LVRV_surf.vtk
        SUBJECT/volmesh_pipline/step_06_volmesh_generate/SUBJECT_FRAME_volmesh_notag.vtk
        SUBJECT/volmesh_pipline/step_06_volmesh_generate/SUBJECT_FRAME_volmesh.vtk
        SUBJECT/volmesh_pipline/step_07_base_detection/SUBJECT_FRAME_base_plane.vtk
        SUBJECT/volmesh_pipline/step_08_LV_RV_openbase/SUBJECT_FRAME_openbase_volmesh.vtk
Step 04 also saves LVRV_outer_surf.vtk and LVRV_inner_surf.vtk with the same prefix.
Both endocardial surfaces are retained in full; only outer contact regions may
move outward by a mesh-scaled numerical clearance. No basal opening is made.
RV is the RV endocardium. Copies retain the original VTK format and arrays.
Cleaned surfaces use ASCII VTK triangles with LV elemTag=1 and RV elemTag=0.
Use --start-step 2 to clean existing Step 01 files without copying them again.
Use --start-step 3 to move both epicardia using existing Step 02 surfaces.
Use --start-step 4 to extract the combined surface using existing Steps 02/03.
Use --start-step 5 to clean only the existing Step 04 LVRV_surf.vtk files.
Use --start-step 6 to mesh existing Step 05 surfaces; Step 03 LVepi_moved defines LV.
Step 06 assigns elemTag=1 when a tetrahedron center is inside LVepi_moved, else 0.
Use --start-step 7 to preview base detection on existing Step 05/06 outputs.
Use --start-step 8 to cut using the saved Step 07 planes without recomputing them.
Use --stop-after 7 to preview planes without cutting the volume.
Use --overwrite to back up and replace existing outputs of the selected steps.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv


SURFACES = {"LVendo": "LV endocardium", "LVepi": "LV epicardium",
            "RV": "RV endocardium", "RVepi": "RV epicardium"}


def parse_frames(value):
    frames = list(dict.fromkeys(part.strip().upper() for part in value.split(",") if part.strip()))
    if not frames or any(frame not in ("ED", "ES") for frame in frames):
        raise argparse.ArgumentTypeError("Use ED, ES, or ED,ES.")
    return frames


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_surface(path):
    """Check the input type without repairing or changing the loaded mesh."""
    mesh = pv.read(path)
    if isinstance(mesh, pv.PolyData):
        triangles_only = mesh.is_all_triangles
    elif isinstance(mesh, pv.UnstructuredGrid):
        triangles_only = np.all(mesh.celltypes == pv.CellType.TRIANGLE)
    else:
        triangles_only = False
    if not mesh.n_points or not mesh.n_cells or not triangles_only:
        raise ValueError(f"Expected a nonempty triangle surface: {path}")
    if not np.isfinite(mesh.points).all():
        raise ValueError(f"Surface contains non-finite coordinates: {path}")
    return mesh


def step_01_load_surf(subject_dir, frames=("ED", "ES"), overwrite=False):
    subject_dir = Path(subject_dir).resolve()
    frames = parse_frames(",".join(frames))
    subject = subject_dir.name
    input_dir = subject_dir / "vtks"
    output_dir = subject_dir / "volmesh_pipline" / "step_01_load_surf"
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Step 01 output directory must differ from vtks.")

    inputs = {(frame, name): input_dir / f"{name}_{frame}.vtk"
              for frame in frames for name in SURFACES}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input surfaces: " + ", ".join(missing))
    outputs = {key: output_dir / f"{subject}_{key[0]}_{key[1]}.vtk" for key in inputs}
    summaries = {frame: output_dir / f"{subject}_{frame}_step01_load_surf_summary.json" for frame in frames}
    existing = [path for path in (*outputs.values(), *summaries.values()) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 01 outputs already exist. Use --overwrite to back them up and replace them.")
    backup = (output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
              if existing else None)

    # Validate every requested input before publishing any frame's outputs.
    records = {}
    for key, source in inputs.items():
        digest = checksum(source)
        mesh = load_surface(source)
        records[key] = {
            "role": SURFACES[key[1]], "input_path": str(source), "output_path": str(outputs[key]),
            "points": int(mesh.n_points), "triangles": int(mesh.n_cells),
            "bounds": [float(value) for value in mesh.bounds],
            "point_data": list(mesh.point_data), "cell_data": list(mesh.cell_data),
            "sha256": digest,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step01_load_", dir=output_dir) as temporary:
        staged = []
        for key, source in inputs.items():
            destination = outputs[key]
            copy = Path(temporary) / destination.name
            shutil.copy2(source, copy)
            if checksum(copy) != records[key]["sha256"] or checksum(source) != records[key]["sha256"]:
                raise RuntimeError(f"Input changed while copying: {source}")
            staged.append((copy, destination))
        for frame in frames:
            summary = {
                "step": "step_01_load_surf", "subject": subject, "frame": frame,
                "input_dir": str(input_dir), "output_dir": str(output_dir),
                "surfaces": {name: records[frame, name] for name in SURFACES},
                "byte_identical_copies": True, "input_files_modified": False,
                "repair_performed": False, "tagging_performed": False,
                "stopped_after": "step_01_load_surf",
                "previous_output_backup": str(backup) if backup else None,
            }
            copy = Path(temporary) / summaries[frame].name
            copy.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
            staged.append((copy, summaries[frame]))
            results.append(summary)
        if backup:
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
            print(f"Previous Step 01 outputs backed up to: {backup}", flush=True)
        for copy, destination in staged:
            copy.replace(destination)

    for key, record in records.items():
        print(f"{subject} {key[0]} {key[1]}: {record['points']} points, "
              f"{record['triangles']} triangles -> {outputs[key]}", flush=True)
    print("Step 01 complete. Original input surfaces were copied without repair.", flush=True)
    return results


def save_clean_surface(surface, path, tag=None):
    """Write ASCII triangles and recheck the serialized geometry and labels."""
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface

    grid = pv.UnstructuredGrid(
        surface.faces, np.full(surface.n_cells, pv.CellType.TRIANGLE, dtype=np.uint8),
        np.asarray(surface.points, dtype=np.float64),
    )
    for name, values in surface.point_data.items():
        grid.point_data[name] = values
    for name, values in surface.cell_data.items():
        grid.cell_data[name] = values
    grid.save(path, binary=False)
    saved = load_surface(path)
    if (not np.array_equal(saved.cells, surface.faces)
            or not np.allclose(saved.points, surface.points, rtol=0.0, atol=1e-9)):
        raise RuntimeError(f"Saved cleaning result changed its geometry or connectivity: {path}")
    tags = np.asarray(saved.cell_data.get("elemTag", []))
    expected = surface.cell_data["elemTag"] if tag is None else np.full(surface.n_cells, tag)
    if len(tags) != saved.n_cells or not np.array_equal(tags, expected):
        raise RuntimeError(f"Saved cleaning result has incorrect elemTag values: {path}")
    return validate_clean_surface(pv.PolyData(saved.points, saved.cells))


def step_02_surf_clean(subject_dir, frames=("ED", "ES"), overwrite=False):
    """Repair saved Step 01 surfaces without changing any upstream files."""
    from model_vol_septum.clean_step02_surfaces import clean_surface_meshfix

    subject_dir = Path(subject_dir).resolve()
    frames = parse_frames(",".join(frames))
    subject = subject_dir.name
    root = subject_dir / "volmesh_pipline"
    input_dir, output_dir = root / "step_01_load_surf", root / "step_02_surf_clean"
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Step 02 output directory must differ from Step 01.")
    inputs = {(frame, name): input_dir / f"{subject}_{frame}_{name}.vtk"
              for frame in frames for name in SURFACES}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 01 surfaces; run Step 01 first: " + ", ".join(missing))
    outputs = {key: output_dir / path.name for key, path in inputs.items()}
    summaries = {frame: output_dir / f"{subject}_{frame}_step02_surf_clean_summary.json" for frame in frames}
    existing = [path for path in (*outputs.values(), *summaries.values()) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 02 outputs already exist. Use --overwrite to back them up and replace them.")
    backup = (output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
              if existing else None)
    hashes = {key: checksum(path) for key, path in inputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    records, results = {}, []
    with TemporaryDirectory(prefix=".step02_clean_", dir=output_dir) as temporary:
        staged = []
        for key, source in inputs.items():
            frame, name = key
            tag = 1 if name.startswith("LV") else 0
            print(f"{subject} {frame}: cleaning {name}", flush=True)
            surface = load_surface(source).extract_surface().triangulate()
            repaired, info = clean_surface_meshfix(surface, tag)
            copy = Path(temporary) / outputs[key].name
            saved_validation = save_clean_surface(repaired, copy, tag)
            records[key] = {
                "role": SURFACES[name], "input_path": str(source), "output_path": str(outputs[key]),
                "input_sha256": hashes[key], "output_sha256": checksum(copy),
                "cleaning": info, "saved_validation": saved_validation,
                "vtk_format": "ASCII_UNSTRUCTURED_GRID_TRIANGLES_DOUBLE",
            }
            staged.append((copy, outputs[key]))
        for key, source in inputs.items():
            if checksum(source) != hashes[key]:
                raise RuntimeError(f"Step 01 input changed during cleaning: {source}")
        for frame in frames:
            summary = {
                "step": "step_02_surf_clean", "subject": subject, "frame": frame,
                "input_dir": str(input_dir), "output_dir": str(output_dir),
                "surfaces": {name: records[frame, name] for name in SURFACES},
                "input_files_modified": False, "repair_performed": True,
                "surface_tags": {"LV": 1, "RV": 0},
                "stopped_after": "step_02_surf_clean",
                "previous_output_backup": str(backup) if backup else None,
            }
            copy = Path(temporary) / summaries[frame].name
            copy.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
            staged.append((copy, summaries[frame]))
            results.append(summary)
        if backup:
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
            print(f"Previous Step 02 outputs backed up to: {backup}", flush=True)
        for copy, destination in staged:
            copy.replace(destination)
    print(f"Step 02 complete: validated {len(inputs)} surfaces in {output_dir}", flush=True)
    return results


def _surface_object(surface):
    import fcl

    model = fcl.BVHModel()
    model.beginModel()
    model.addSubModel(np.asarray(surface.points, dtype=np.float64),
                      surface.faces.reshape(-1, 4)[:, 1:].astype(np.int32))
    model.endModel()
    return fcl.CollisionObject(model)


def _surface_gap(surface, fixed_object):
    import fcl

    result = fcl.DistanceResult()
    distance = float(fcl.distance(_surface_object(surface), fixed_object,
        fcl.DistanceRequest(enable_nearest_points=True), result))
    if not np.isfinite(distance):
        raise RuntimeError("Non-finite surface-to-surface distance.")
    return distance, result.nearest_points[0]


def validate_closed_epi(epi, endo, minimum=3.0, source=None, chamber="RV"):
    """Validate the closed moved surface without changing either input."""
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface

    epi_name, endo_name = ((f"{chamber}epi", f"{chamber}endo") if chamber else ("outer", "inner"))
    quality = validate_clean_surface(epi)
    validate_clean_surface(endo)
    distance, _ = _surface_gap(epi, _surface_object(endo))
    if distance < minimum:
        raise RuntimeError(f"{epi_name}/{endo_name} surface distance is {distance:.9g} mm, below {minimum:g} mm.")
    outside = pv.PolyData(epi.points).select_enclosed_points(endo, tolerance=0.0)["SelectedPoints"] == 0
    inside = pv.PolyData(endo.points).select_enclosed_points(epi, tolerance=0.0)["SelectedPoints"] != 0
    if not np.all(outside):
        raise RuntimeError(f"Moved {epi_name} contains vertices inside {endo_name}.")
    if not np.all(inside):
        raise RuntimeError(f"Moved {epi_name} does not enclose the complete {endo_name}.")
    info = {"minimum_surface_distance_mm": distance, "self_intersection_pairs": 0,
            f"{epi_name}_vertices_inside_{endo_name}": 0, f"{endo_name}_vertices_outside_{epi_name}": 0,
            "free_edges": quality["free_edges"], "distance_method": "FCL_triangle_triangle"}
    if source is not None:
        if epi.points.shape != source.points.shape or not np.array_equal(epi.faces, source.faces):
            raise RuntimeError(f"{epi_name} point numbering or triangle connectivity changed.")
        reference = source.compute_normals(point_normals=True, cell_normals=False,
            consistent_normals=True, auto_orient_normals=True, split_vertices=False).point_data["Normals"]
        outward = np.einsum("ij,ij->i", epi.points - source.points, reference)
        if np.any(outward < -1e-9):
            raise RuntimeError(f"{epi_name} displacement contains inward movement.")
        info.update({"connectivity_preserved": True, "inward_displacement_vertices": 0,
                     "maximum_displacement_mm": float(np.linalg.norm(epi.points - source.points, axis=1).max())})
    return info


def move_closed_epi(epi, endo, minimum=3.0, max_iterations=600,
                    clearance_padding=0.1, log_label=None, chamber="RV"):
    """Displace a closed epicardium outward; keep its endocardium fixed."""
    import vtk
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import factorized
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface
    from model_vol_septum.mesh_intersections import confirmed_self_intersections

    if not np.isfinite(minimum) or minimum <= 0 or max_iterations < 1:
        raise ValueError("Epicardial clearance and iteration count must be positive.")
    epi_name, endo_name = ((f"{chamber}epi", f"{chamber}endo") if chamber else ("outer", "inner"))
    log_label = log_label or f"Step03 {epi_name}"
    validate_clean_surface(epi)
    validate_clean_surface(endo)
    original, faces = np.asarray(epi.points, dtype=float).copy(), epi.faces.reshape(-1, 4)[:, 1:]
    reference = np.asarray(epi.compute_normals(point_normals=True, cell_normals=False,
        consistent_normals=True, auto_orient_normals=True, split_vertices=False).point_data["Normals"], dtype=float)
    edges = np.unique(np.sort(np.vstack((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(original[edges[:, 0]] - original[edges[:, 1]], axis=1)
    scale = float(np.median(lengths))
    weights = np.minimum((scale / np.maximum(lengths, scale * 1e-5)) ** 2, 1e4)
    a, b = edges[:, 0], edges[:, 1]
    laplace = coo_matrix((np.r_[weights, weights, -weights, -weights],
        (np.r_[a, b, a, b], np.r_[a, b, b, a])), shape=(len(original), len(original))).tocsc()
    smooth = factorized(eye(len(original), format="csc") + 2 * laplace)
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(endo)

    def signed(points):
        return np.fromiter((implicit.EvaluateFunction(point) for point in points), float, count=len(points))

    fixed_object, result = _surface_object(endo), epi.copy(deep=True)
    result.points = original.copy()
    initial = signed(original)
    target, max_step = minimum + clearance_padding, min(1.0, max(0.5 * scale, 1e-3))
    for iteration in range(max_iterations + 1):
        distances = signed(result.points)
        gap, witness = _surface_gap(result, fixed_object)
        if np.all(distances >= minimum) and gap >= minimum:
            validation = validate_closed_epi(result, endo, minimum, source=epi, chamber=chamber)
            break
        if iteration == max_iterations:
            raise RuntimeError(f"{epi_name} movement did not converge; minimum vertex distance={distances.min():.6g} mm.")
        raw = np.maximum(target - distances, 0)
        xyz = result.points[faces]
        probes = np.stack((xyz.mean(axis=1), (xyz[:, 0] + xyz[:, 1]) * .5,
                           (xyz[:, 1] + xyz[:, 2]) * .5, (xyz[:, 2] + xyz[:, 0]) * .5), axis=1)
        deficits = np.maximum(target - signed(probes.reshape(-1, 3)).reshape(-1, 4).min(axis=1), 0)
        np.maximum.at(raw, faces.ravel(), np.repeat(deficits, 3))
        if gap < minimum and np.all(distances >= minimum):
            ids = faces[int(result.find_closest_cell(witness))]
            raw[ids] = np.maximum(raw[ids], target - gap)
        normals = np.asarray(result.compute_normals(point_normals=True, cell_normals=False,
            consistent_normals=True, auto_orient_normals=True, split_vertices=False).point_data["Normals"], dtype=float)
        directions = smooth(normals)
        for _ in range(4):
            for outward in (reference, normals):
                directions -= np.minimum(np.einsum("ij,ij->i", directions, outward), 0)[:, None] * outward
        directions /= np.maximum(np.linalg.norm(directions, axis=1), 1e-12)[:, None]
        delta = directions * np.clip(smooth(raw), 0, max_step)[:, None]
        cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        rejected = None
        for half in range(18):
            step = delta * 0.5 ** half
            candidate = result.points + step
            new = candidate[faces]
            new_cross = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
            folded = np.flatnonzero((np.linalg.norm(new_cross, axis=1) <= 1e-12)
                | (np.einsum("ij,ij->i", cross, new_cross) <= 0))
            if len(folded):
                rejected = f"folded triangle IDs {folded[:5].tolist()}"
                continue
            intersections = confirmed_self_intersections(candidate, faces)
            if not len(intersections):
                result.points = candidate
                break
            rejected = f"self-intersection triangle pairs {intersections[:5].tolist()}"
        else:
            raise RuntimeError(f"Outward {epi_name} displacement stopped at iteration {iteration + 1}: "
                f"{rejected}; no invalid result saved.")
        if np.max(np.linalg.norm(step, axis=1)) < 1e-9:
            raise RuntimeError(f"{epi_name} movement stalled before meeting the requested clearance.")
        if iteration % 5 == 0:
            print(f"[{log_label}] iteration={iteration + 1} minimum_signed_distance={distances.min():.6g} mm "
                  f"maximum_step={np.linalg.norm(step, axis=1).max():.6g} mm", flush=True)
    result = result.compute_normals(point_normals=True, cell_normals=True,
        consistent_normals=False, auto_orient_normals=False, split_vertices=False)
    result.cell_data["elemTag"] = np.full(result.n_cells, int(chamber == "LV"), dtype=np.int32)
    result.point_data["displacement"] = result.points - original
    result.point_data["outwardDisplacement"] = np.einsum("ij,ij->i", result.points - original, reference)
    result.point_data[f"initialSignedDistanceTo{endo_name}"] = initial
    result.point_data[f"signedDistanceTo{endo_name}"] = distances
    result.point_data[f"inside{endo_name}"] = (distances < 0).astype(np.int32)
    return result, {"method": f"outward_{epi_name}_normal_displacement", "iterations": iteration,
                    "minimum_requested_distance_mm": minimum,
                    f"initial_vertices_inside_{endo_name}": int(np.count_nonzero(initial < 0)),
                    "validation": validation}


def step_03_move_RVepi(subject_dir, frames=("ED", "ES"), overwrite=False,
                       minimum=3.0, max_iterations=600):
    """Move LVepi and RVepi independently; retain the existing Step 03 folder name."""
    subject_dir = Path(subject_dir).resolve()
    frames = parse_frames(",".join(frames))
    subject = subject_dir.name
    root = subject_dir / "volmesh_pipline"
    input_dir, output_dir = root / "step_02_surf_clean", root / "step_03_move_RVepi"
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Step 03 output directory must differ from Step 02.")
    inputs = {(frame, name): input_dir / f"{subject}_{frame}_{name}.vtk"
              for frame in frames for name in SURFACES}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 02 surfaces: " + ", ".join(missing))
    outputs = {(frame, chamber): output_dir / f"{subject}_{frame}_{chamber}epi_moved.vtk"
               for frame in frames for chamber in ("LV", "RV")}
    summaries = {frame: output_dir / f"{subject}_{frame}_step03_move_RVepi_summary.json" for frame in frames}
    existing = [path for path in (*outputs.values(), *summaries.values()) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 03 outputs already exist. Use --overwrite to back them up and replace them.")
    backup = (output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
              if existing else None)
    hashes = {key: checksum(path) for key, path in inputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step03_move_", dir=output_dir) as temporary:
        staged = []
        for frame in frames:
            try:
                surfaces = {}
                for name in SURFACES:
                    mesh = load_surface(inputs[frame, name])
                    if isinstance(mesh, pv.UnstructuredGrid):
                        surface = pv.PolyData(mesh.points, mesh.cells)
                        surface.point_data.update(dict(mesh.point_data))
                        surface.cell_data.update(dict(mesh.cell_data))
                    else:
                        surface = mesh.copy(deep=True)
                    surfaces[name] = surface
                records = {}
                for chamber, endo_name, tag in (("RV", "RV", 0), ("LV", "LVendo", 1)):
                    epi_name = f"{chamber}epi"
                    print(f"{subject} {frame}: move closed {epi_name} outward from fixed {chamber}endo", flush=True)
                    moved, info = move_closed_epi(surfaces[epi_name], surfaces[endo_name],
                        minimum=minimum, max_iterations=max_iterations, chamber=chamber)
                    copy = Path(temporary) / outputs[frame, chamber].name
                    save_clean_surface(moved, copy, tag)
                    saved = load_surface(copy)
                    verification = validate_closed_epi(pv.PolyData(saved.points, saved.cells),
                        surfaces[endo_name], minimum, source=surfaces[epi_name], chamber=chamber)
                    records[epi_name] = {"output_path": str(outputs[frame, chamber]),
                        "output_sha256": checksum(copy), "movement": info,
                        "saved_validation": verification, "elemTag": tag}
                    staged.append((copy, outputs[frame, chamber]))
                for key, source in inputs.items():
                    if checksum(source) != hashes[key]:
                        raise RuntimeError(f"Step 02 input changed during movement: {source}")
                summary = {
                    "step": "step_03_move_RVepi", "subject": subject, "frame": frame,
                    "inputs": {name: str(inputs[frame, name]) for name in SURFACES},
                    "input_sha256": {name: hashes[frame, name] for name in SURFACES},
                    "outputs": records,
                    "output_path": records["RVepi"]["output_path"],
                    "output_sha256": records["RVepi"]["output_sha256"],
                    "movement": records["RVepi"]["movement"],
                    "saved_validation": records["RVepi"]["saved_validation"],
                    "fixed_surfaces": ["LVendo", "RV"],
                    "input_files_modified": False, "LV_intersection_handling": False,
                    "stopped_after": "step_03_move_RVepi",
                    "previous_output_backup": str(backup) if backup else None,
                }
                summary_copy = Path(temporary) / summaries[frame].name
                summary_copy.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
                staged.append((summary_copy, summaries[frame]))
                results.append(summary)
            except Exception as error:
                failures = output_dir / "failed_attempts"
                failures.mkdir(exist_ok=True)
                failure = failures / f"{subject}_{frame}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
                failure.write_text(json.dumps({"subject": subject, "frame": frame,
                    "error": str(error), "invalid_mesh_published": False}, indent=2) + "\n", encoding="ascii")
                raise
        if backup:
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
            print(f"Previous Step 03 outputs backed up to: {backup}", flush=True)
        for copy, destination in staged:
            copy.replace(destination)
    print(f"Step 03 complete: validated {2 * len(frames)} moved LVepi/RVepi surfaces in {output_dir}", flush=True)
    return results


def _join_triangle_surfaces(parts):
    """Append triangles without welding or renumbering the points within a part."""
    offsets = np.cumsum([0] + [part.n_points for part in parts])
    triangles = np.vstack([part.faces.reshape(-1, 4)[:, 1:] + offset
                           for part, offset in zip(parts, offsets[:-1])])
    result = pv.PolyData(np.vstack([part.points for part in parts]),
                        np.column_stack((np.full(len(triangles), 3), triangles)).ravel())
    for name in ("elemTag", "sourceSurface"):
        if all(name in part.cell_data for part in parts):
            result.cell_data[name] = np.concatenate([part.cell_data[name] for part in parts])
    return result.compute_normals(point_normals=True, cell_normals=True,
        consistent_normals=False, auto_orient_normals=False, split_vertices=False)


def validate_lvrv_shell(parts, minimum):
    """Require strict nesting and three disjoint, correctly oriented boundaries."""
    import manifold3d
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface

    outer, inner, whole = (parts[name] for name in ("outer", "inner", "whole"))
    quality = validate_clean_surface(whole)
    distance, _ = _surface_gap(outer, _surface_object(inner))
    if distance < minimum:
        raise RuntimeError(f"Outer/inner surface clearance {distance:.9g} mm is below {minimum:.9g} mm.")
    enclosed = pv.PolyData(inner.points).select_enclosed_points(outer, tolerance=0.0)["SelectedPoints"]
    if not np.all(enclosed):
        raise RuntimeError("The outer envelope does not enclose both complete inner surfaces.")
    mesh = manifold3d.Mesh64(np.asarray(whole.points, dtype=np.float64),
                            np.asarray(whole.faces.reshape(-1, 4)[:, 1:], dtype=np.uint64))
    solid = manifold3d.Manifold(mesh)
    if solid.status() != manifold3d.Error.NoError or solid.is_empty():
        raise RuntimeError(f"The assembled myocardial shell is invalid: {solid.status()}")
    volumes = sorted(float(part.volume()) for part in solid.decompose())
    if len(volumes) != 3 or not (volumes[0] < 0 and volumes[1] < 0 < volumes[2]):
        raise RuntimeError(f"Expected one outer boundary and two enclosed blood pools: {volumes}")
    return {**quality, "minimum_outer_inner_distance_mm": distance,
            "inner_vertices_outside_outer": 0, "outer_boundary_components": 1,
            "blood_pool_cavities": 2, "boundary_component_signed_volumes_mm3": volumes,
            "myocardial_volume_mm3": float(solid.volume())}


def extract_lvrv_surface(surfaces):
    """Return outer, complete inner, and combined myocardial triangle surfaces."""
    import manifold3d
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface

    solids, oriented, limits = {}, {}, [0]
    for name in SURFACES:
        surface = surfaces[name]
        validate_clean_surface(surface)
        surface = surface.compute_normals(point_normals=False, cell_normals=True,
            consistent_normals=True, auto_orient_normals=True, split_vertices=False)
        mesh = manifold3d.Mesh64(
            np.asarray(surface.points, dtype=np.float64),
            np.asarray(surface.faces.reshape(-1, 4)[:, 1:], dtype=np.uint64),
            face_id=np.arange(limits[-1], limits[-1] + surface.n_cells, dtype=np.uint64),
        )
        solid = manifold3d.Manifold(mesh)
        if (solid.status() != manifold3d.Error.NoError or solid.is_empty()
                or solid.volume() <= 0 or len(solid.decompose()) != 1):
            raise RuntimeError(f"{name} must define one valid closed solid: {solid.status()}")
        solids[name] = solid
        oriented[name] = surface
        limits.append(limits[-1] + surface.n_cells)

    for endo, epi in (("LVendo", "LVepi"), ("RV", "RVepi")):
        if not (solids[endo] - solids[epi]).is_empty():
            raise RuntimeError(f"{endo} is not completely enclosed by {epi}.")
    endo_gap, _ = _surface_gap(oriented["LVendo"], _surface_object(oriented["RV"]))
    if not (solids["LVendo"] ^ solids["RV"]).is_empty() or endo_gap <= 0:
        raise RuntimeError("LV and RV blood pools overlap or touch; two separate cavities cannot be preserved.")
    union = solids["LVepi"] + solids["RVepi"]
    if union.status() != manifold3d.Error.NoError:
        raise RuntimeError(f"Epicardial Boolean union failed: {union.status()}")
    components = union.decompose()
    exterior = [part for part in components if part.volume() > 0]
    if len(exterior) != 1:
        raise RuntimeError("LVepi and RVepi do not form one connected external surface.")
    mesh = exterior[0].to_mesh64()
    points = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3].copy()
    triangles = np.asarray(mesh.tri_verts, dtype=np.int64)
    face_ids = np.asarray(mesh.face_id, dtype=np.int64)
    if (len(face_ids) != len(triangles) or np.any(face_ids < 0)
            or np.any(face_ids >= limits[-1])):
        raise RuntimeError("Boolean output lost its source-surface face IDs.")
    owners = np.searchsorted(limits[1:], face_ids, side="right") + 1
    if not np.all(np.isin(owners, (2, 4))):
        raise RuntimeError("The outer envelope contains non-epicardial source faces.")
    cells = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    outer = pv.PolyData(points, cells)
    outer.cell_data["sourceSurface"] = owners.astype(np.int32)
    outer.cell_data["elemTag"] = (owners == 2).astype(np.int32)
    endos = [oriented[name] for name in ("LVendo", "RV")]
    inner_outward = _join_triangle_surfaces(endos)
    initial_gap, _ = _surface_gap(outer, _surface_object(inner_outward))
    edges = outer.extract_all_edges()
    edge_ids = edges.lines.reshape(-1, 3)[:, 1:]
    edge_lengths = np.linalg.norm(edges.points[edge_ids[:, 0]] - edges.points[edge_ids[:, 1]], axis=1)
    contact_tolerance = max(1e-6 * outer.length, 1e-6)
    clearance = max(0.01 * float(np.median(edge_lengths)), contact_tolerance)
    original_outer = outer.copy(deep=True)
    # Reuse the outward-only closed-surface solver, with a small numerical
    # separation instead of the 3 mm RV wall constraint. Inner geometry is fixed.
    if initial_gap < contact_tolerance:
        outer, movement = move_closed_epi(outer, inner_outward, minimum=clearance,
            max_iterations=120, clearance_padding=clearance, log_label="Step04 outer envelope", chamber=None)
    else:
        movement = {"iterations": 0}
        clearance = contact_tolerance
    outer.clear_data()
    for name in ("sourceSurface", "elemTag"):
        outer.cell_data[name] = original_outer.cell_data[name].copy()
    displacement = outer.points - original_outer.points
    outer.point_data["outerDisplacement"] = displacement
    outer.point_data["outerDisplacementMagnitude"] = np.linalg.norm(displacement, axis=1)
    inner_parts = []
    for name, tag, source_id in (("LVendo", 1, 1), ("RV", 0, 3)):
        surface = oriented[name]
        faces = surface.faces.reshape(-1, 4).copy()
        faces[:, 1:] = faces[:, 1:][:, ::-1]
        inner_part = pv.PolyData(np.asarray(surfaces[name].points, dtype=float).copy(), faces.ravel())
        inner_part.cell_data["elemTag"] = np.full(surface.n_cells, tag, dtype=np.int32)
        inner_part.cell_data["sourceSurface"] = np.full(surface.n_cells, source_id, dtype=np.int32)
        inner_parts.append(inner_part)
    inner = _join_triangle_surfaces(inner_parts)
    whole = _join_triangle_surfaces([outer, inner])
    parts = {"outer": outer, "inner": inner, "whole": whole}
    verification = validate_lvrv_shell(parts, clearance)
    return parts, {
        "method": "epicardial_outer_envelope_plus_complete_endocardial_boundaries",
        "endocardial_boolean_subtraction": False,
        "source_surface_ids": {name: index for index, name in enumerate(SURFACES, 1)},
        "surface_tag_rule": "LV source faces=1; RV source faces=0; not volume-mesh labels",
        "discarded_internal_epicardial_components": len(components) - 1,
        "outer_boundary_components": 1, "blood_pool_cavities": 2,
        "boundary_component_signed_volumes_mm3": verification["boundary_component_signed_volumes_mm3"],
        "myocardial_volume_mm3": verification["myocardial_volume_mm3"],
        "outer_adjustment": {"minimum_clearance_requested_mm": clearance,
            "initial_minimum_clearance_mm": initial_gap, "iterations": movement["iterations"],
            "contact_tolerance_mm": contact_tolerance,
            "maximum_displacement_mm": float(np.linalg.norm(displacement, axis=1).max()),
            "moved_vertices": int(np.count_nonzero(np.linalg.norm(displacement, axis=1) > 1e-9)),
            "outward_only": True},
        "inner_surfaces": {name: {"points": int(surfaces[name].n_points),
            "triangles": int(surfaces[name].n_cells), "all_points_and_faces_retained": True}
            for name in ("LVendo", "RV")},
        "elemTag_counts": {str(tag): int(np.count_nonzero(whole.cell_data["elemTag"] == tag))
                           for tag in (0, 1)},
        "validation": verification, "point_cloud_reconstruction": False,
        "inner_outer_points_welded": False,
    }


def step_04_extract_LVRV_surf(subject_dir, frames=("ED", "ES"), overwrite=False, minimum=3.0):
    """Extract a closed combined triangle surface from saved Steps 02 and 03."""
    subject_dir = Path(subject_dir).resolve()
    subject, frames = subject_dir.name, parse_frames(",".join(frames))
    root = subject_dir / "volmesh_pipline"
    output_dir = root / "step_04_extract_LVRV_surf"
    inputs = {(frame, name):
        (root / "step_03_move_RVepi" / f"{subject}_{frame}_{name}_moved.vtk" if name.endswith("epi")
         else root / "step_02_surf_clean" / f"{subject}_{frame}_{name}.vtk")
        for frame in frames for name in SURFACES}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 02/03 surfaces: " + ", ".join(missing))
    suffixes = {"outer": "LVRV_outer_surf", "inner": "LVRV_inner_surf", "whole": "LVRV_surf"}
    outputs = {(frame, role): output_dir / f"{subject}_{frame}_{suffix}.vtk"
               for frame in frames for role, suffix in suffixes.items()}
    summaries = {frame: output_dir / f"{subject}_{frame}_step04_extract_LVRV_surf_summary.json" for frame in frames}
    existing = [path for path in (*outputs.values(), *summaries.values()) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 04 outputs already exist. Use --overwrite to back them up and replace them.")
    backup = (output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
              if existing else None)
    hashes = {key: checksum(path) for key, path in inputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step04_extract_", dir=output_dir) as temporary:
        staged = []
        for frame in frames:
            surfaces = {}
            for name in SURFACES:
                mesh = load_surface(inputs[frame, name])
                surfaces[name] = (pv.PolyData(mesh.points, mesh.cells)
                                  if isinstance(mesh, pv.UnstructuredGrid) else mesh.copy(deep=True))
            print(f"{subject} {frame}: extract the whole LV/RV myocardial surface", flush=True)
            epi_validation = {chamber: validate_closed_epi(surfaces[f"{chamber}epi"],
                surfaces[endo_name], minimum, chamber=chamber)
                for chamber, endo_name in (("LV", "LVendo"), ("RV", "RV"))}
            parts, info = extract_lvrv_surface(surfaces)
            saved_parts, records = {}, {}
            for role, surface in parts.items():
                copy = Path(temporary) / outputs[frame, role].name
                quality = save_clean_surface(surface, copy)
                saved = load_surface(copy)
                if not np.array_equal(saved.cell_data["sourceSurface"], surface.cell_data["sourceSurface"]):
                    raise RuntimeError(f"Saved {role} surface lost its source labels.")
                saved_parts[role] = pv.PolyData(saved.points, saved.cells)
                records[role] = {"path": str(outputs[frame, role]), "sha256": checksum(copy),
                                 "validation": quality}
                staged.append((copy, outputs[frame, role]))
            verification = validate_lvrv_shell(saved_parts, info["outer_adjustment"]["minimum_clearance_requested_mm"])
            point_offset, face_offset = 0, 0
            saved_inner = saved_parts["inner"]
            for name in ("LVendo", "RV"):
                source = surfaces[name]
                points = saved_inner.points[point_offset:point_offset + source.n_points]
                faces = saved_inner.faces.reshape(-1, 4)[face_offset:face_offset + source.n_cells, 1:] - point_offset
                if (not np.allclose(points, source.points, rtol=0, atol=1e-9)
                        or not np.array_equal(np.sort(faces, axis=1), np.sort(source.faces.reshape(-1, 4)[:, 1:], axis=1))):
                    raise RuntimeError(f"Saved inner surface changed {name} geometry or dropped its triangles.")
                point_offset += source.n_points
                face_offset += source.n_cells
            summary = {
                "step": "step_04_extract_LVRV_surf", "subject": subject, "frame": frame,
                "inputs": {name: str(inputs[frame, name]) for name in SURFACES},
                "input_sha256": {name: hashes[frame, name] for name in SURFACES},
                "output_path": records["whole"]["path"], "output_sha256": records["whole"]["sha256"],
                "outputs": records,
                "input_epi_clearance_validation": epi_validation,
                "extraction": info, "saved_validation": verification,
                "input_files_modified": False, "basal_opening_performed": False,
                "stopped_after": "step_04_extract_LVRV_surf",
                "previous_output_backup": str(backup) if backup else None,
            }
            summary_copy = Path(temporary) / summaries[frame].name
            summary_copy.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
            staged.append((summary_copy, summaries[frame]))
            results.append(summary)
        for key, source in inputs.items():
            if checksum(source) != hashes[key]:
                raise RuntimeError(f"Step 02/03 input changed during extraction: {source}")
        if backup:
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
        for copy, destination in staged:
            copy.replace(destination)
    print(f"Step 04 complete: validated {len(frames)} combined surfaces in {output_dir}", flush=True)
    return results


def _lvrv_surface_components(surface):
    """Split the single Step 04 file using its retained source-surface labels."""
    labels = np.asarray(surface.cell_data.get("sourceSurface", []))
    if len(labels) != surface.n_cells or not np.all(np.isin(labels, (1, 2, 3, 4))):
        raise ValueError("LVRV_surf.vtk must contain Step 04 sourceSurface labels 1, 2, 3, 4.")
    components = {}
    for name, ids in (("outer", (2, 4)), ("LVendo", (1,)), ("RVendo", (3,))):
        mask = np.isin(labels, ids)
        if not np.any(mask):
            raise ValueError(f"LVRV_surf.vtk is missing its {name} surface.")
        components[name] = surface.extract_cells(mask).extract_surface().triangulate()
    return components


def clean_lvrv_surface(surface, minimum=3.0):
    """Clean the three boundaries separately, then restore cavity-facing normals."""
    from model_vol_septum.clean_step02_surfaces import clean_surface_meshfix, surface_quality

    before = surface_quality(surface)
    components = _lvrv_surface_components(surface)
    cleaned, records = {}, {}
    for name, source in components.items():
        repaired, info = clean_surface_meshfix(source, None)
        # MeshFix may replace triangles. Transfer surface provenance, not final
        # volumetric LV/RV labels, from the nearest original triangle.
        nearest = source.find_closest_cell(repaired.cell_centers().points)
        labels = np.asarray(source.cell_data["sourceSurface"])[nearest].astype(np.int32)
        repaired.cell_data["sourceSurface"] = labels
        repaired.cell_data["elemTag"] = np.isin(labels, (1, 2)).astype(np.int32)
        if name != "outer":
            faces = repaired.faces.reshape(-1, 4).copy()
            faces[:, 1:] = faces[:, 1:][:, ::-1]
            repaired.faces = faces.ravel()
        cleaned[name], records[name] = repaired, info
    inner = _join_triangle_surfaces([cleaned["LVendo"], cleaned["RVendo"]])
    whole = _join_triangle_surfaces([cleaned["outer"], inner])
    validation = validate_lvrv_shell({"outer": cleaned["outer"], "inner": inner, "whole": whole}, minimum)
    return whole, {
        "method": "componentwise_MeshFix_cleanup_of_combined_LVRV_surface",
        "before": before, "components": records, "after": validation,
        "surface_label_transfer": "nearest_original_triangle; provenance only, not volume tags",
        "minimum_wall_clearance_mm": minimum,
        "additional_displacement_performed": False, "basal_opening_performed": False,
        "cleanup_may_change_geometry_and_connectivity": True,
    }


def step_05_LVRV_surf_clean(subject_dir, frames=("ED", "ES"), overwrite=False, minimum=3.0):
    """Read only each Step 04 LVRV_surf.vtk and save one cleaned VTK per frame."""
    subject_dir = Path(subject_dir).resolve()
    subject, frames = subject_dir.name, parse_frames(",".join(frames))
    root = subject_dir / "volmesh_pipline"
    input_dir, output_dir = root / "step_04_extract_LVRV_surf", root / "step_05_LVRV_surf_clean"
    inputs = {frame: input_dir / f"{subject}_{frame}_LVRV_surf.vtk" for frame in frames}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 04 combined surfaces: " + ", ".join(missing))
    outputs = {frame: output_dir / path.name for frame, path in inputs.items()}
    summaries = {frame: output_dir / f"{subject}_{frame}_step05_LVRV_surf_clean_summary.json" for frame in frames}
    existing = [path for path in (*outputs.values(), *summaries.values()) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 05 outputs already exist. Use --overwrite to back them up and replace them.")
    backup = (output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
              if existing else None)
    hashes = {frame: checksum(path) for frame, path in inputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step05_clean_", dir=output_dir) as temporary:
        staged = []
        for frame, source in inputs.items():
            mesh = load_surface(source)
            surface = (pv.PolyData(mesh.points, mesh.cells) if isinstance(mesh, pv.UnstructuredGrid)
                       else mesh.copy(deep=True))
            surface.cell_data.update(dict(mesh.cell_data))
            print(f"{subject} {frame}: clean {source.name}", flush=True)
            cleaned, info = clean_lvrv_surface(surface, minimum)
            copy = Path(temporary) / outputs[frame].name
            save_clean_surface(cleaned, copy)
            saved = load_surface(copy)
            saved_surface = pv.PolyData(saved.points, saved.cells)
            saved_surface.cell_data.update(dict(saved.cell_data))
            if not np.array_equal(saved.cell_data["sourceSurface"], cleaned.cell_data["sourceSurface"]):
                raise RuntimeError("Saved cleaned surface lost its source labels.")
            parts = _lvrv_surface_components(saved_surface)
            inner = _join_triangle_surfaces([parts["LVendo"], parts["RVendo"]])
            verification = validate_lvrv_shell({"outer": parts["outer"], "inner": inner,
                                                "whole": saved_surface}, minimum)
            summary = {
                "step": "step_05_LVRV_surf_clean", "subject": subject, "frame": frame,
                "input_path": str(source), "input_sha256": hashes[frame],
                "output_path": str(outputs[frame]), "output_sha256": checksum(copy),
                "cleaning": info, "saved_validation": verification,
                "input_files_modified": False, "stopped_after": "step_05_LVRV_surf_clean",
                "previous_output_backup": str(backup) if backup else None,
            }
            summary_copy = Path(temporary) / summaries[frame].name
            summary_copy.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
            staged.extend(((copy, outputs[frame]), (summary_copy, summaries[frame])))
            results.append(summary)
        for frame, source in inputs.items():
            if checksum(source) != hashes[frame]:
                raise RuntimeError(f"Step 04 input changed during cleaning: {source}")
        if backup:
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
        for copy, destination in staged:
            copy.replace(destination)
    print(f"Step 05 complete: validated {len(frames)} cleaned surfaces in {output_dir}", flush=True)
    return results


def tetrahedralize_lvrv_surface(surface, minratio=1.2, mindihedral=5.0, max_volume=0.0):
    """Mesh one PLC; exclude the constrained LV/RV blood-pool regions."""
    import tetgen
    from model_vol_septum.generate_step09_volume import signed_volumes, validate_volume

    if (not np.isfinite([minratio, mindihedral, max_volume]).all() or minratio <= 0
            or not 0 <= mindihedral < 180 or max_volume < 0):
        raise ValueError("Invalid TetGen quality or maximum-volume settings.")
    parts = _lvrv_surface_components(surface)
    inner = _join_triangle_surfaces([parts["LVendo"], parts["RVendo"]])
    input_validation = validate_lvrv_shell({"outer": parts["outer"], "inner": inner,
                                           "whole": surface}, max(surface.length, 1.0) * 1e-10)
    settings = dict(plc=True, quality=True, nobisect=True, order=1, regionattrib=False,
                    minratio=float(minratio), mindihedral=float(mindihedral), quiet=True)
    if max_volume > 0:
        settings.update(fixedvolume=True, maxvolume=float(max_volume))
    print(f"[Step06 TetGen] {surface.n_points} surface points, {surface.n_cells} constrained triangles", flush=True)
    result = tetgen.TetGen(surface).tetrahedralize(**settings)
    points = np.asarray(result[0], dtype=float)
    cells = np.asarray(result[1], dtype=np.int64)
    if (cells.ndim != 2 or cells.shape[1] != 4 or not len(cells)
            or not np.isfinite(points).all() or cells.min() < 0 or cells.max() >= len(points)):
        raise RuntimeError("TetGen did not return valid linear tetrahedra.")
    centers = pv.PolyData(points[cells].mean(axis=1))
    blood_pool = np.zeros(len(cells), dtype=bool)
    cavity_counts = {}
    # TetGen 0.6.4 exposes no hole-seed API. Its PLC still constrains every
    # endocardial facet, so cavity tetrahedra can be removed without cutting tets.
    for name in ("LVendo", "RVendo"):
        inside = np.asarray(centers.select_enclosed_points(parts[name], tolerance=0.0)["SelectedPoints"], dtype=bool)
        cavity_counts[name] = int(np.count_nonzero(inside))
        blood_pool |= inside
    initial_count = len(cells)
    cells = cells[~blood_pool].copy()
    if not len(cells):
        raise RuntimeError("Cavity exclusion removed every tetrahedron.")
    reverse = signed_volumes(points, cells) < 0
    cells[reverse, :2] = cells[reverse, 1::-1]
    used, remap = np.unique(cells, return_inverse=True)
    points, cells = points[used], remap.reshape(-1, 4)
    grid = pv.UnstructuredGrid(np.column_stack((np.full(len(cells), 4), cells)).ravel(),
                               np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8), points)
    validation = validate_volume(grid, surface)
    validation["boundary_matches_step05"] = validation.pop("boundary_matches_step08")
    if validation["component_count"] != 1:
        raise RuntimeError(f"The myocardial volume has {validation['component_count']} face-connected components.")
    print(f"[Step06 TetGen] kept {grid.n_cells} myocardial tetrahedra; excluded {int(blood_pool.sum())} blood-pool tetrahedra", flush=True)
    return grid, {"tetgen_version": tetgen.__version__, "settings": settings,
                  "input_validation": input_validation, "volume_validation": validation,
                  "tetgen_tetrahedra_before_cavity_exclusion": int(initial_count),
                  "removed_blood_pool_tetrahedra": int(blood_pool.sum()),
                  "blood_pool_tetrahedra": cavity_counts,
                  "negative_tetrahedra_reoriented": int(np.count_nonzero(reverse)),
                  "surface_repair_performed": False, "boundary_splitting_permitted": False}


def tag_lvrv_volume(grid, lv_epi):
    """Assign whole-cell tags by centroid containment; do not alter tetrahedra."""
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface
    from model_vol_septum.tag_step10_volume import shared_face_neighbors, component_labels

    reference_validation = validate_clean_surface(lv_epi)
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    inside = np.asarray(pv.PolyData(grid.points[cells].mean(axis=1)).select_enclosed_points(
        lv_epi, tolerance=0.0)["SelectedPoints"], dtype=bool)
    tags = inside.astype(np.int32)
    if set(np.unique(tags)) != {0, 1}:
        raise RuntimeError("LVepi classification must produce both LV=1 and RV=0 tetrahedra.")
    vertex_inside = np.asarray(pv.PolyData(grid.points).select_enclosed_points(
        lv_epi, tolerance=0.0)["SelectedPoints"], dtype=bool)
    counts = vertex_inside[cells].sum(axis=1).astype(np.int32)
    mixed = ((counts > 0) & (counts < 4)).astype(np.int32)
    tagged = grid.copy(deep=True)
    tagged.cell_data["elemTag"] = tags
    tagged.cell_data["LVepi_inside_vertex_count"] = counts
    tagged.cell_data["LVepi_mixed_vertices"] = mixed
    tagged.set_active_scalars("elemTag", preference="cell")
    a, b, _ = shared_face_neighbors(cells)
    regions = component_labels(tags, a, b)
    return tagged, {
        "method": "tetrahedron_centroid_inside_closed_step03_LVepi_moved",
        "tag_rule": "centroid inside LVepi_moved: LV=1; otherwise RV=0; LV wins overlap",
        "reference_validation": reference_validation,
        "regions": {name: {"elemTag": tag, "tetrahedra": int(np.count_nonzero(tags == tag)),
                           "face_connected_components": int(len(np.unique(regions[tags == tag])))}
                    for name, tag in (("LV", 1), ("RV", 0))},
        "mixed_vertex_membership_tetrahedra": int(mixed.sum()),
        "interface_note": "LVepi is a post-tagging reference, not an internal PLC interface; crossing cells use their centroid",
        "geometry_and_element_order_preserved": True, "tag_island_relabeling_performed": False,
    }


def write_tetrahedral_vtk(grid, path):
    """Write legacy ASCII tetrahedra with lossless double-coordinate precision."""
    if not isinstance(grid, pv.UnstructuredGrid) or not np.all(grid.celltypes == pv.CellType.TETRA):
        raise ValueError("The volume writer accepts tetrahedral grids only.")
    for values in grid.cell_data.values():
        if np.asarray(values).shape != (grid.n_cells,) or np.asarray(values).dtype.kind not in "biu":
            raise ValueError("The volume writer expects integer cell scalars.")
    with Path(path).open("w", encoding="ascii") as stream:
        stream.write("# vtk DataFile Version 3.0\nLVRV tetrahedral mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {grid.n_points} double\n")
        np.savetxt(stream, grid.points, fmt="%.17g")
        stream.write(f"CELLS {grid.n_cells} {5 * grid.n_cells}\n")
        np.savetxt(stream, grid.cells.reshape(-1, 5), fmt="%d")
        stream.write(f"CELL_TYPES {grid.n_cells}\n")
        np.savetxt(stream, np.full(grid.n_cells, pv.CellType.TETRA), fmt="%d")
        if len(grid.cell_data):
            stream.write(f"CELL_DATA {grid.n_cells}\n")
            for name, values in grid.cell_data.items():
                stream.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
                np.savetxt(stream, np.asarray(values), fmt="%d")


def step_06_volmesh_generate(subject_dir, frames=("ED", "ES"), overwrite=False,
                             minratio=1.2, mindihedral=5.0, max_volume=0.0):
    """Generate one connected mesh, then tag it with the moved closed LVepi."""
    from model_vol_septum.generate_step09_volume import validate_volume
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface

    subject_dir = Path(subject_dir).resolve()
    subject, frames = subject_dir.name, parse_frames(",".join(frames))
    root = subject_dir / "volmesh_pipline"
    output_dir = root / "step_06_volmesh_generate"
    inputs = {frame: root / "step_05_LVRV_surf_clean" / f"{subject}_{frame}_LVRV_surf.vtk" for frame in frames}
    references = {frame: root / "step_03_move_RVepi" / f"{subject}_{frame}_LVepi_moved.vtk" for frame in frames}
    missing = [str(path) for path in (*inputs.values(), *references.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 05 surface or Step 03 LVepi_moved: " + ", ".join(missing))
    outputs = {(frame, role): output_dir / f"{subject}_{frame}_{suffix}.vtk"
               for frame in frames for role, suffix in (("untagged", "volmesh_notag"), ("tagged", "volmesh"))}
    summaries = {frame: output_dir / f"{subject}_{frame}_step06_volmesh_generate_summary.json" for frame in frames}
    existing = [path for path in (*outputs.values(), *summaries.values()) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 06 outputs already exist. Use --overwrite to back them up and replace them.")
    backup = (output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
              if existing else None)
    hashes = {path: checksum(path) for path in (*inputs.values(), *references.values())}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step06_volume_", dir=output_dir) as temporary:
        staged = []
        for frame, source in inputs.items():
            surfaces = []
            for path in (source, references[frame]):
                mesh = load_surface(path)
                surface = (pv.PolyData(mesh.points, mesh.cells) if isinstance(mesh, pv.UnstructuredGrid)
                           else mesh.copy(deep=True))
                surface.cell_data.update(dict(mesh.cell_data))
                surfaces.append(surface)
            boundary, reference = surfaces
            validate_clean_surface(reference)
            print(f"{subject} {frame}: generate connected TetGen myocardium", flush=True)
            grid, meshing = tetrahedralize_lvrv_surface(boundary, minratio, mindihedral, max_volume)
            untagged_path = Path(temporary) / outputs[frame, "untagged"].name
            write_tetrahedral_vtk(grid, untagged_path)
            saved = pv.read(untagged_path)
            if not np.array_equal(saved.points, grid.points) or not np.array_equal(saved.cells, grid.cells):
                raise RuntimeError("Saving the untagged volume changed coordinates or tetrahedra.")
            verification = validate_volume(saved, boundary)
            verification["boundary_matches_step05"] = verification.pop("boundary_matches_step08")
            if verification["component_count"] != 1:
                raise RuntimeError("The saved myocardial mesh is not face-connected.")
            print(f"{subject} {frame}: assign LV=1/RV=0 using LVepi_moved", flush=True)
            tagged, tagging = tag_lvrv_volume(saved, reference)
            tagged_path = Path(temporary) / outputs[frame, "tagged"].name
            write_tetrahedral_vtk(tagged, tagged_path)
            tagged_saved = pv.read(tagged_path)
            if (not np.array_equal(tagged_saved.points, saved.points)
                    or not np.array_equal(tagged_saved.cells, saved.cells)
                    or not np.array_equal(tagged_saved.celltypes, saved.celltypes)):
                raise RuntimeError("Tagging changed the untagged mesh geometry or element order.")
            for name, values in tagged.cell_data.items():
                if not np.array_equal(tagged_saved.cell_data.get(name), values):
                    raise RuntimeError(f"Saved volume lost its {name} cell values.")
            summary = {
                "step": "step_06_volmesh_generate", "subject": subject, "frame": frame,
                "input_path": str(source), "input_sha256": hashes[source],
                "LVepi_reference_path": str(references[frame]), "LVepi_reference_sha256": hashes[references[frame]],
                "outputs": {role: {"path": str(outputs[frame, role]), "sha256": checksum(path)}
                            for role, path in (("untagged", untagged_path), ("tagged", tagged_path))},
                "meshing": meshing, "tagging": tagging, "saved_volume_validation": verification,
                "tag_legend": {"1": "LV myocardium", "0": "RV myocardium"},
                "vtk_format": "ASCII_UNSTRUCTURED_GRID_TETRAHEDRA_DOUBLE_17_DIGITS",
                "input_files_modified": False, "stopped_after": "step_06_volmesh_generate",
                "previous_output_backup": str(backup) if backup else None,
            }
            summary_copy = Path(temporary) / summaries[frame].name
            summary_copy.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
            staged.extend(((untagged_path, outputs[frame, "untagged"]),
                           (tagged_path, outputs[frame, "tagged"]), (summary_copy, summaries[frame])))
            results.append(summary)
        for path, digest in hashes.items():
            if checksum(path) != digest:
                raise RuntimeError(f"Step 06 input changed: {path}")
        if backup:
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
        for copy, destination in staged:
            copy.replace(destination)
    print(f"Step 06 complete: validated {len(frames)} connected and tagged meshes in {output_dir}", flush=True)
    return results


def _surface_axis_moments(surface):
    """Area-integrated centroid and covariance, independent of vertex density."""
    xyz = surface.points[surface.faces.reshape(-1, 4)[:, 1:]].astype(float)
    area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1) / 2
    center = np.average(xyz.mean(axis=1), axis=0, weights=area)
    delta = xyz - center
    sums = delta.sum(axis=1)
    moments = np.einsum("nvi,nvj->nij", delta, delta) + np.einsum("ni,nj->nij", sums, sums)
    return center, np.average(moments, axis=0, weights=area) / 12


def _section_sampler(surface, origin, axis):
    """Measure actual closed planar sections, including concavity and holes."""
    import vtk
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    plane = vtk.vtkPlane()
    plane.SetNormal(axis)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(surface)
    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputConnection(cutter.GetOutputPort())

    def sample(level):
        plane.SetOrigin(origin + level * axis)
        cutter.Update()
        contour = pv.wrap(cutter.GetOutput())
        if not contour.n_lines:
            return 0., 0
        lines = contour.lines
        if len(lines) != 3 * contour.n_lines or np.any(lines[::3] != 2):
            return np.nan, 0
        vertices, ids = np.unique(lines.reshape(-1, 3)[:, 1:], return_inverse=True)
        edges = ids.reshape(-1, 2)
        if np.any(np.bincount(edges.ravel(), minlength=len(vertices)) != 2):
            return np.nan, 0
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(vertices), len(vertices))).tocsr()
        loops = int(connected_components(graph, directed=False, return_labels=False))
        triangulator.Update()
        if triangulator.GetTriangulationError():
            return np.nan, 0
        return float(pv.wrap(triangulator.GetOutput()).area), loops

    return sample


def _terminal_base_evidence(surface, origin, axis, levels, areas, bounds):
    low, high = bounds
    span = high - low
    peaks, taper, caps, cap_axes = [], [], [], []
    xyz = surface.points[surface.faces.reshape(-1, 4)[:, 1:]].astype(float)
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    projection = cross @ axis
    projected_area = np.abs(projection) / 2
    aligned = projected_area >= np.cos(np.pi / 6) * np.linalg.norm(cross, axis=1) / 2
    face_levels = (xyz.mean(axis=1) - origin) @ axis
    for end, sign in ((low, 1), (high, -1)):
        depth = sign * (levels - end)
        finite = np.isfinite(areas) & (depth >= 0)
        band = areas[finite & (depth <= .15 * span)]
        near = areas[finite & (depth <= .05 * span)]
        if not len(band) or not len(near):
            raise RuntimeError("Unable to measure terminal cavity areas; check the base axis and surface.")
        peak = float(band.max())
        peaks.append(peak)
        taper.append(float(near.mean()) / max(peak, 1e-12))
        cap = aligned & (sign * (face_levels - end) <= .15 * span)
        caps.append(float(projected_area[cap].sum()))
        normal = (cross[cap] * np.sign(projection[cap])[:, None]).sum(axis=0)
        length = np.linalg.norm(normal)
        cap_axes.append((normal / length).tolist() if length > 0 else None)
    contrast = lambda values: (values[1] - values[0]) / max(max(values), 1e-12)
    score = .60 * contrast(peaks) + .25 * contrast(caps) + .15 * (taper[1] - taper[0])
    return {"terminal_peak_areas_mm2": peaks, "near_tip_to_peak_area_ratios": taper,
            "axis_aligned_terminal_face_areas_mm2": caps, "terminal_cap_axes": cap_axes,
            "high_vs_low_score": score}


def _cavity_axis_bounds(cavities, origin, axis):
    bounds = {}
    for name, surface in cavities.items():
        projection = (surface.points - origin) @ axis
        bounds[name] = float(projection.min()), float(projection.max())
    return bounds


def _common_basal_interval(cavities, origin, axis, basal_fraction):
    bounds = _cavity_axis_bounds(cavities, origin, axis)
    return (max(high - basal_fraction * (high - low) for low, high in bounds.values()),
            min(high for _, high in bounds.values()))


def _profile_cavity_evidence(cavities, origin, axis, samples):
    profiles, evidence, scores = {}, {}, []
    for frame, parts in cavities.items():
        bounds = _cavity_axis_bounds(parts, origin, axis)
        start, stop = min(pair[0] for pair in bounds.values()), max(pair[1] for pair in bounds.values())
        levels = start + (np.arange(samples) + .5) * (stop - start) / samples
        sections = {}
        for name, surface in parts.items():
            sample = _section_sampler(surface, origin, axis)
            sections[name] = np.array([sample(s) for s in levels])
        evidence[frame] = {name: _terminal_base_evidence(surface, origin, axis, levels,
                           sections[name][:, 0], bounds[name]) for name, surface in parts.items()}
        scores.append(.75 * evidence[frame]["LVendo"]["high_vs_low_score"]
                      + .25 * evidence[frame]["RVendo"]["high_vs_low_score"])
        profiles[frame] = {"levels": levels, "sections": sections}
    return profiles, evidence, scores


def _basal_cap_consensus(evidence, side):
    end, sign = (1, 1) if side == "high" else (0, -1)
    normals = []
    for frame in evidence.values():
        for chamber in frame.values():
            normal = chamber["terminal_cap_axes"][end]
            support = chamber["axis_aligned_terminal_face_areas_mm2"][end]
            if normal is None or support < .25 * chamber["terminal_peak_areas_mm2"][end]:
                return None
            normals.append(sign * np.asarray(normal))
    axis = np.mean(normals, axis=0)
    axis /= np.linalg.norm(axis)
    alignment = float(np.min(np.asarray(normals) @ axis))
    return (axis, alignment) if alignment >= np.cos(np.pi / 9) else None


def detect_subject_base(cavities, base_axis="auto", base_side="auto", samples=160, basal_fraction=.30):
    """Share one axis and one base/apex decision across available frames."""
    if samples < 40 or base_side not in ("auto", "high", "low"):
        raise ValueError("Use at least 40 samples and a base side of auto, high, or low.")
    if not 0 < basal_fraction <= .5:
        raise ValueError("Basal search fraction must be in (0, 0.5].")
    moments = [_surface_axis_moments(parts["LVendo"]) for parts in cavities.values()]
    origin = np.mean([center for center, _ in moments], axis=0)
    covariances = [cov / np.trace(cov) for _, cov in moments]
    eigenvalues, vectors = np.linalg.eigh(np.mean(covariances, axis=0))
    ratio = float(eigenvalues[-1] / max(eigenvalues[-2], 1e-12))
    if base_axis == "auto":
        axis = vectors[:, -1]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
    elif base_axis == "z":
        axis = np.array([0., 0., 1.])
    else:
        try:
            axis = np.array([float(value) for value in base_axis.split(",")])
        except ValueError as error:
            raise ValueError("--base-axis must be auto, z, or a comma-separated 3D vector.") from error
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
            raise ValueError("--base-axis must be auto, z, or a finite nonzero 3D vector.")
        axis /= np.linalg.norm(axis)
    alignment = min(abs(np.linalg.eigh(cov)[1][:, -1] @ axis) for cov in covariances)
    if base_axis == "auto" and (ratio < 1.20 or alignment < .85):
        raise RuntimeError(f"Ambiguous subject long axis (PCA ratio={ratio:.3f}, frame alignment={alignment:.3f}); set --base-axis.")

    evidence_axis = axis.copy()
    profiles, evidence, phase_scores = _profile_cavity_evidence(cavities, origin, axis, samples)
    score = float(np.mean(phase_scores))
    disagreement = min(phase_scores) < -.12 and max(phase_scores) > .12
    if base_side == "auto" and (abs(score) < .12 or disagreement):
        raise RuntimeError(f"Ambiguous base/apex decision (score={score:.3f}, frame scores={phase_scores}); set --base-side.")
    side = ("high" if score > 0 else "low") if base_side == "auto" else base_side
    if side == "low":
        axis = -axis
        for profile in profiles.values():
            profile["levels"] = -profile["levels"][::-1]
            profile["sections"] = {name: values[::-1] for name, values in profile["sections"].items()}
    intervals = {frame: _common_basal_interval(parts, origin, axis, basal_fraction)
                 for frame, parts in cavities.items()}
    refinement = None
    confidence = abs(score)
    if base_axis == "auto" and any(low >= high for low, high in intervals.values()):
        consensus = _basal_cap_consensus(evidence, side)
        if consensus is not None:
            refined_axis, cap_alignment = consensus
            refined_intervals = {frame: _common_basal_interval(parts, origin, refined_axis, basal_fraction)
                                 for frame, parts in cavities.items()}
            if all(low < high for low, high in refined_intervals.values()):
                refined_profiles, refined_evidence, refined_scores = _profile_cavity_evidence(
                    cavities, origin, refined_axis, samples)
                refined_score = float(np.mean(refined_scores))
                if base_side != "auto" or (refined_score >= .12 and min(refined_scores) >= -.12):
                    refinement = {"method": "consensus_of_basal_face_normals",
                                  "reason": "PCA axis did not yield a common basal interval in every frame",
                                  "original_apex_to_base_axis": axis.tolist(),
                                  "rotation_from_PCA_degrees": float(np.degrees(np.arccos(np.clip(axis @ refined_axis, -1, 1)))),
                                  "minimum_cap_axis_alignment": cap_alignment,
                                  "initial_basal_intervals_mm": intervals, "refined_basal_intervals_mm": refined_intervals,
                                  "refined_terminal_evidence": refined_evidence,
                                  "refined_frame_scores": dict(zip(cavities, refined_scores))}
                    axis, profiles, confidence = refined_axis, refined_profiles, abs(refined_score)
    return origin, axis, profiles, {
        "method": "subject_level_area_weighted_LV_axis_and_terminal_cavity_evidence",
        "axis_mode": base_axis, "base_side_relative_to_input_axis": side, "base_side_override": base_side,
        "apex_to_base_axis": axis.tolist(), "projection_origin_mm": origin.tolist(),
        "evidence_axis_before_sign_selection": evidence_axis.tolist(), "axis_refinement": refinement,
        "PCA_long_axis_eigenvalue_ratio": ratio, "minimum_frame_axis_alignment": float(alignment),
        "frames_used": list(cavities), "terminal_evidence": evidence,
        "frame_high_vs_low_scores": dict(zip(cavities, phase_scores)), "high_vs_low_score": score,
        "confidence_score": confidence, "automatic_acceptance_threshold": .12,
        "confidence_is_geometric_not_anatomical_certainty": True,
    }


def _basal_candidate_plane(grid, cavities, origin, axis, basal_fraction=.30, samples=64):
    """Choose a broad common opening without actually clipping tetrahedra."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from model_vol_septum.tag_step10_volume import shared_face_neighbors

    if not isinstance(grid, pv.UnstructuredGrid) or not grid.n_cells or not np.all(grid.celltypes == pv.CellType.TETRA):
        raise ValueError("Base detection requires a nonempty Step 06 tetrahedral mesh.")
    if not 0 < basal_fraction <= .5 or samples < 3:
        raise ValueError("Basal search fraction must be in (0, 0.5], with at least three candidates.")
    projection = (grid.points - origin) @ axis
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    if not np.isfinite(projection).all() or cells.min() < 0 or cells.max() >= len(projection):
        raise RuntimeError("Step 06 contains invalid points or tetrahedron indices.")
    a, b, faces = shared_face_neighbors(cells)
    minimum_tet = projection[cells].min(axis=1)
    maximum_tet = projection[cells].max(axis=1)
    minimum_face = projection[faces].min(axis=1)
    lo, hi = _common_basal_interval(cavities, origin, axis, basal_fraction)
    if lo >= hi:
        raise RuntimeError("No common basal interval for the two cavities; review the base axis or search fraction.")
    step = (hi - lo) / samples
    levels = lo + (np.arange(samples) + .5) * step
    nodes = np.unique(projection)
    index = np.clip(np.searchsorted(nodes, levels), 1, len(nodes) - 1)
    midpoint = (nodes[index - 1] + nodes[index]) / 2
    levels += np.clip(midpoint - levels, -.25 * step, .25 * step)
    samplers = [_section_sampler(cavities[name], origin, axis) for name in ("LVendo", "RVendo")]
    sections = np.array([[sample(level) for sample in samplers] for level in levels])
    areas, loops = sections[:, :, 0], sections[:, :, 1].astype(int)
    valid = np.all(np.isfinite(areas) & (areas > 0) & (loops == 1), axis=1)
    if not np.any(valid):
        raise RuntimeError("No candidate plane has one complete LV and one complete RV opening.")
    scores = np.zeros(samples)
    scores[valid] = (areas[valid] / areas[valid].max(axis=0)).min(axis=1)
    components = np.full(samples, -1, dtype=int)
    # A clipped tetrahedron/shared face has positive measure exactly when at
    # least one of its vertices lies strictly on the kept side of the plane.
    best = scores.max()
    preferred = np.flatnonzero(valid & (scores >= .98 * best))[::-1]
    remaining = np.flatnonzero(valid & (scores < .98 * best))
    order = np.r_[preferred, remaining[np.argsort(-scores[remaining], kind="stable")]]
    for index in order:
        level = levels[index]
        kept = minimum_tet < level
        mapping = np.full(len(cells), -1, dtype=int)
        mapping[kept] = np.arange(kept.sum())
        edges = minimum_face < level
        graph = coo_matrix((np.ones(edges.sum()), (mapping[a[edges]], mapping[b[edges]])),
                           shape=(kept.sum(), kept.sum())).tocsr()
        components[index] = connected_components(graph, directed=False, return_labels=False)
        if components[index] == 1 and np.any(~kept):
            break
    else:
        raise RuntimeError("No common basal plane keeps a connected myocardial volume; review the cut geometry.")
    record = {"axial_cut_mm": float(level), "plane_origin_mm": (origin + level * axis).tolist(),
              "plane_normal_toward_base": axis.tolist(), "keep_apex_side": "signed_distance <= 0",
              "LV_opening_area_mm2": float(areas[index, 0]), "RV_opening_area_mm2": float(areas[index, 1]),
              "balanced_opening_score": float(scores[index]), "basal_search_range_mm": [lo, hi],
              "predicted_face_connected_components": int(components[index]),
              "fully_kept_tetrahedra": int(np.count_nonzero(maximum_tet < level)),
              "crossing_tetrahedra": int(np.count_nonzero(kept & (maximum_tet >= level))),
              "fully_removed_tetrahedra": int(np.count_nonzero(~kept)),
              "minimum_plane_to_vertex_axial_distance_mm": float(np.min(np.abs(projection - level)))}
    rows = [(float(s), float(al), float(ar), int(nl), int(nr), float(score), int(count), i == index)
            for i, (s, (al, ar), (nl, nr), score, count) in enumerate(zip(levels, areas, loops, scores, components))]
    return record, rows


def step_07_base_detection(subject_dir, frames=("ED", "ES"), overwrite=False,
                           base_axis="auto", base_side="auto", samples=160, basal_fraction=.30):
    """Save the shared base direction and frame-specific plane previews only."""
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface

    subject_dir = Path(subject_dir).resolve()
    subject, frames = subject_dir.name, parse_frames(",".join(frames))
    root = subject_dir / "volmesh_pipline"
    output_dir = root / "step_07_base_detection"
    boundaries = {frame: root / "step_05_LVRV_surf_clean" / f"{subject}_{frame}_LVRV_surf.vtk"
                  for frame in ("ED", "ES")}
    volumes = {frame: root / "step_06_volmesh_generate" / f"{subject}_{frame}_volmesh_notag.vtk" for frame in frames}
    missing = [str(path) for frame in frames for path in (boundaries[frame], volumes[frame]) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 05/06 base-detection inputs: " + ", ".join(missing))
    boundaries = {frame: path for frame, path in boundaries.items() if path.is_file()}
    suffixes = ("base_plane.vtk", "apex_point.vtk", "area_profiles.csv", "basal_candidates.csv", "base_detection.json")
    names = {(frame, suffix): f"{subject}_{frame}_{suffix}" for frame in frames for suffix in suffixes}
    subject_name = f"{subject}_base_detection_summary.json"
    existing = [output_dir / name for name in (*names.values(), subject_name) if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 07 outputs already exist. Use --overwrite to back them up and replace them.")
    hashes = {path: checksum(path) for path in (*boundaries.values(), *volumes.values())}
    cavities = {}
    for frame, path in boundaries.items():
        mesh = load_surface(path)
        parts = _lvrv_surface_components(mesh)
        cavities[frame] = {name: parts[name] for name in ("LVendo", "RVendo")}
        for surface in cavities[frame].values():
            validate_clean_surface(surface)
    print(f"{subject}: detect base using {', '.join(cavities)} cavity surfaces", flush=True)
    origin, axis, profiles, detection = detect_subject_base(cavities, base_axis, base_side, samples, basal_fraction)
    if detection["axis_refinement"]:
        print(f"{subject}: refined the shared axis with consensus basal face normals", flush=True)
    backup = (output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
              if existing else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step07_base_", dir=output_dir) as temporary:
        temporary = Path(temporary)
        for frame in frames:
            grid = pv.read(volumes[frame])
            cut, candidates = _basal_candidate_plane(grid, cavities[frame], origin, axis, basal_fraction)
            level = cut["axial_cut_mm"]
            center = np.asarray(grid.center) + axis * (level - (np.asarray(grid.center) - origin) @ axis)
            u = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)
            size = .575 * grid.length
            corners = center + size * np.array([-u-v, u-v, u+v, -u+v])
            plane = pv.PolyData(corners, np.array([3, 0, 1, 2, 3, 0, 2, 3]))
            apex_points = cavities[frame]["LVendo"].points
            apex = apex_points[np.argmin((apex_points - origin) @ axis)]
            plane.save(temporary / names[frame, "base_plane.vtk"], binary=False)
            pv.PolyData(apex.reshape(1, 3)).save(temporary / names[frame, "apex_point.vtk"], binary=False)
            profile = profiles[frame]
            with (temporary / names[frame, "area_profiles.csv"]).open("w", newline="", encoding="ascii") as stream:
                writer = csv.writer(stream)
                writer.writerow(("axial_mm", "LV_area_mm2", "RV_area_mm2", "LV_contours", "RV_contours"))
                for s, lv, rv in zip(profile["levels"], profile["sections"]["LVendo"], profile["sections"]["RVendo"]):
                    writer.writerow((s, lv[0], rv[0], int(lv[1]), int(rv[1])))
            with (temporary / names[frame, "basal_candidates.csv"]).open("w", newline="", encoding="ascii") as stream:
                writer = csv.writer(stream)
                writer.writerow(("axial_mm", "LV_area_mm2", "RV_area_mm2", "LV_contours", "RV_contours",
                                 "balanced_score", "predicted_components_minus_one_if_unchecked", "selected"))
                writer.writerows(candidates)
            record = {"step": "step_07_base_detection", "subject": subject, "frame": frame,
                      "surface_path": str(boundaries[frame]), "volume_path": str(volumes[frame]),
                      "surface_sha256": hashes[boundaries[frame]], "volume_sha256": hashes[volumes[frame]],
                      "subject_detection": detection, "cut": cut, "LVendo_apex_point_mm": apex.tolist(),
                      "outputs": {suffix: str(output_dir / names[frame, suffix]) for suffix in suffixes[:-1]},
                      "volume_cut_performed": False, "previous_output_backup": str(backup) if backup else None}
            (temporary / names[frame, "base_detection.json"]).write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="ascii")
            results.append(record)
            print(f"{subject} {frame}: plane at axial {level:.3f} mm; "
                  f"LV/RV opening areas {cut['LV_opening_area_mm2']:.1f}/{cut['RV_opening_area_mm2']:.1f} mm^2", flush=True)
        (temporary / subject_name).write_text(json.dumps({"subject": subject, **detection,
            "surface_inputs": {frame: {"path": str(path), "sha256": hashes[path]} for frame, path in boundaries.items()},
            "cutting_planes": {record["frame"]: record["cut"] for record in results},
            "preview_only": True}, indent=2, allow_nan=False) + "\n", encoding="ascii")
        for path, digest in hashes.items():
            if checksum(path) != digest:
                raise RuntimeError(f"Base-detection input changed: {path}")
        if backup:
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
        for name in (*names.values(), subject_name):
            (temporary / name).replace(output_dir / name)
    print(f"Step 07 complete: plane previews in {output_dir}; volume meshes unchanged", flush=True)
    return results


def _clip_myocardium(grid, origin, normal):
    """Clip one volume and preserve the Step 06 parent ID of every output cell."""
    import vtk
    from model_vol_septum.generate_step09_volume import signed_volumes

    if not isinstance(grid, pv.UnstructuredGrid) or not grid.n_cells or not np.all(grid.celltypes == pv.CellType.TETRA):
        raise ValueError("Step 08 requires the nonempty Step 06 tetrahedral grid.")
    points = np.asarray(grid.points, dtype=np.float64)
    distance = (points - origin) @ normal
    tolerance = max(float(grid.length), 1.) * 1e-10
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    distances = distance[cells]
    kept = distances.max(axis=1) <= 0
    retained = distances.min(axis=1) < 0
    if not np.any(retained & ~kept) or not np.any(~retained):
        raise RuntimeError("Step 08 plane does not cut through the Step 06 volume.")

    source = grid.copy(deep=False)
    source.cell_data["parentCellId"] = np.arange(grid.n_cells, dtype=np.int32)
    source.point_data["planeDistance"] = distance
    clip = vtk.vtkTableBasedClipDataSet()
    clip.SetInputData(source)
    clip.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "planeDistance")
    clip.SetValue(0.)
    clip.InsideOutOn()
    clip.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    tetra = vtk.vtkDataSetTriangleFilter()
    tetra.SetInputConnection(clip.GetOutputPort())
    tetra.TetrahedraOnlyOn()
    tetra.Update()
    result = pv.wrap(tetra.GetOutput())
    if not result.n_cells or not np.all(result.celltypes == pv.CellType.TETRA):
        raise RuntimeError("Step 08 clipping did not produce a nonempty tetrahedral mesh.")
    parents = np.asarray(result.cell_data["parentCellId"]).copy()
    counts = np.bincount(parents, minlength=grid.n_cells)
    if not np.array_equal(counts > 0, retained) or np.any(counts[kept & retained] != 1):
        raise RuntimeError("Clipping unexpectedly lost, split or added an uncut Step 06 tetrahedron.")
    children = result.cells.reshape(-1, 5).copy()
    new_points = np.asarray(result.points, dtype=np.float64).copy()
    if np.max((new_points - origin) @ normal) > tolerance:
        raise RuntimeError("Clipping left points on the discarded side of the plane.")
    uncut = kept[parents]
    before = points[cells[parents[uncut]]]
    after = new_points[children[uncut, 1:]]
    # VTK may permute connectivity but must not move or replace any uncut node.
    if not np.all(np.any(np.all(after[:, :, None] == before[:, None, :], axis=-1), axis=-1)):
        raise RuntimeError("Clipping changed an uncut tetrahedron's geometry.")
    volumes = signed_volumes(new_points, children[:, 1:])
    if not np.isfinite(volumes).all() or np.any(volumes == 0):
        raise RuntimeError("Basal clipping produced degenerate tetrahedra; review the Step 07 plane.")
    reverse = volumes < 0
    children[reverse, 1:3] = children[reverse, 2:0:-1]
    clipped = pv.UnstructuredGrid(children.ravel(), result.celltypes.copy(), new_points)
    cut_children = children[~uncut, 1:]
    xyz = new_points[cut_children]
    edges2 = sum(np.sum((xyz[:, i] - xyz[:, j]) ** 2, axis=1)
                 for i in range(4) for j in range(i))
    cut_volumes = np.abs(volumes[~uncut])
    quality = 12 * (3 * cut_volumes) ** (2/3) / edges2
    return clipped, parents, tolerance, {
        "fully_kept_parent_cells": int(np.count_nonzero(uncut)),
        "crossing_parent_cells": int(np.count_nonzero(retained & ~kept)),
        "fully_removed_parent_cells": int(np.count_nonzero(~retained)),
        "cut_child_cells": int(len(cut_children)),
        "cut_child_minimum_volume_mm3": float(cut_volumes.min()),
        "cut_child_mean_ratio_quality_percentiles": dict(zip(
            ("min", "p01", "p05", "median"), np.percentile(quality, [0, 1, 5, 50]).tolist())),
        "quality_note": "Mean ratio is 1 for a regular tetrahedron; near 0 indicates a poorly shaped cell. No smoothing or remeshing.",
    }


def _validate_basal_openings(boundary, references, origin, normal, tolerance):
    """Check the cut is planar myocardium bounded by one outer and two cavity rims."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from model_vol_septum.fill_step01_surface_holes import topology, triangles

    stats = topology(boundary)
    if any(stats[key] for key in ("free_edges", "nonmanifold_edges",
                                  "inconsistent_orientation_edges", "degenerate_triangles")):
        raise RuntimeError(f"The extracted Step 08 boundary is not watertight and oriented: {stats}")
    face_ids = triangles(boundary)
    plane_distances = np.abs((boundary.points - origin) @ normal)
    on_plane = np.all(plane_distances[face_ids] <= tolerance, axis=1)
    if not np.any(on_plane):
        raise RuntimeError("The clipped myocardium has no planar basal faces.")
    basal = face_ids[on_plane]
    directed_edges = np.vstack((basal[:, [0, 1]], basal[:, [1, 2]], basal[:, [2, 0]]))
    edges, counts = np.unique(np.sort(directed_edges, axis=1), axis=0, return_counts=True)
    rim_edges = edges[counts == 1]
    vertices, inverse = np.unique(rim_edges, return_inverse=True)
    inverse = inverse.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(inverse)), (inverse[:, 0], inverse[:, 1])),
                       shape=(len(vertices), len(vertices))).tocsr()
    num_rims, labels = connected_components(graph, directed=False)
    if num_rims != 3 or np.any(np.bincount(inverse.ravel(), minlength=len(vertices)) != 2):
        raise RuntimeError("The basal patch must have three disjoint simple rims: outer, LV and RV.")

    section_areas = {}
    for name, reference in references.items():
        area, loops = _section_sampler(reference, origin, normal)(0.)
        if loops != 1 or not np.isfinite(area) or area <= 0:
            raise RuntimeError(f"The saved plane does not have a single positive {name} cross-section.")
        section_areas[name] = area
    expected = section_areas["outer"] - section_areas["LVendo"] - section_areas["RVendo"]
    xyz = boundary.points[basal]
    measured = float(np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1).sum() / 2)
    if expected <= 0 or not np.isclose(measured, expected, rtol=1e-6, atol=tolerance**2):
        raise RuntimeError(f"Basal face area {measured:g} does not match the expected myocardium area {expected:g}.")
    rims = {}
    for label in range(num_rims):
        edges_for_rim = inverse[labels[inverse[:, 0]] == label]
        ids, local = np.unique(edges_for_rim, return_inverse=True)
        coords = boundary.points[vertices[ids]]
        # Match both endpoints and segment midpoints to the corresponding Step 05 surface.
        samples = np.vstack((coords, boundary.points[vertices[edges_for_rim]].mean(axis=1)))
        matches = []
        for name, reference in references.items():
            _, closest = reference.find_closest_cell(samples, return_closest_point=True)
            if np.max(np.linalg.norm(samples - closest, axis=1)) <= tolerance:
                matches.append(name)
        if len(matches) != 1 or matches[0] in rims:
            raise RuntimeError("A basal rim does not uniquely match its Step 05 anatomical surface.")
        lines = np.column_stack((np.full(len(edges_for_rim), 2), local.reshape(-1, 2))).ravel()
        rims[matches[0]] = pv.PolyData(coords, lines=lines)
    return on_plane, rims, {
        "boundary_topology": stats, "basal_faces": int(on_plane.sum()),
        "basal_myocardial_area_mm2": measured, "step05_section_areas_mm2": section_areas,
        "rim_points": {name: int(rim.n_points) for name, rim in rims.items()},
    }


def _validate_unchanged_wall(source, boundary, parents, on_plane, tolerance):
    """Reject non-basal faces that are not subsets of exposed Step 06 faces."""
    from model_vol_septum.fill_step01_surface_holes import triangles

    source_skin = source.extract_surface()
    source_cells = source.cells.reshape(-1, 5)[:, 1:]
    exposed = np.zeros((source.n_cells, 4), dtype=bool)
    owners = source_skin.cell_data["vtkOriginalCellIds"]
    node_ids = source_skin.point_data["vtkOriginalPointIds"][triangles(source_skin)]
    opposite = ~np.any(source_cells[owners, :, None] == node_ids[:, None, :], axis=-1)
    exposed[owners, opposite.argmax(axis=1)] = True

    owner = parents[boundary.cell_data["vtkOriginalCellIds"][~on_plane]]
    wall = boundary.points[triangles(boundary)[~on_plane]]
    parent_nodes = source.points[source_cells[owner]]
    matched = np.zeros(len(owner), dtype=bool)
    for omit in range(4):
        eligible = exposed[owner, omit] & ~matched
        if not eligible.any():
            continue
        a, b, c = parent_nodes[eligible][:, np.arange(4) != omit].transpose(1, 0, 2)
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        error = np.einsum("ijk,ik->ij", wall[eligible] - a[:, None], normal)
        matched[eligible] = np.all(np.abs(error) <= tolerance, axis=1)
    if not np.all(matched):
        raise RuntimeError(f"Clipping exposed {np.count_nonzero(~matched)} unexpected non-basal faces.")


def step_08_LV_RV_openbase(subject_dir, frames=("ED", "ES"), overwrite=False):
    """Apply the saved common plane to the whole tetrahedral mesh of each phase."""
    from model_vol_septum.generate_step09_volume import validate_volume

    subject_dir = Path(subject_dir).resolve()
    subject, frames = subject_dir.name, parse_frames(",".join(frames))
    root = subject_dir / "volmesh_pipline"
    output_dir = root / "step_08_LV_RV_openbase"
    input_specs = {
        "plane": ("step_07_base_detection", "base_detection.json"),
        "volume": ("step_06_volmesh_generate", "volmesh_notag.vtk"),
        "boundary": ("step_05_LVRV_surf_clean", "LVRV_surf.vtk"),
        "LVepi_reference": ("step_03_move_RVepi", "LVepi_moved.vtk"),
    }
    inputs = {(frame, role): root / folder / f"{subject}_{frame}_{suffix}"
              for frame in frames for role, (folder, suffix) in input_specs.items()}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 08 inputs: " + ", ".join(missing))
    suffixes = {
        "untagged": "openbase_volmesh_notag.vtk", "tagged": "openbase_volmesh.vtk",
        "boundary": "openbase_boundary.vtk", "basal": "basal_myocardium.vtk",
        "outer": "basal_outer_rim.vtk", "LVendo": "LV_opening_rim.vtk", "RVendo": "RV_opening_rim.vtk",
        "summary": "step08_openbase_summary.json",
    }
    outputs = {(frame, role): output_dir / f"{subject}_{frame}_{suffix}"
               for frame in frames for role, suffix in suffixes.items()}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 08 outputs already exist. Use --overwrite to back them up and replace them.")
    backup = (output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
              if existing else None)
    hashes = {key: checksum(path) for key, path in inputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step08_openbase_", dir=output_dir) as temporary:
        temporary = Path(temporary)
        for frame in frames:
            plane = json.loads(inputs[frame, "plane"].read_text(encoding="ascii"))
            if (plane["subject"] != subject or plane["frame"] != frame
                    or plane["volume_sha256"] != hashes[frame, "volume"]
                    or plane["surface_sha256"] != hashes[frame, "boundary"]):
                raise RuntimeError(f"{frame} Step 07 plane does not match the current Step 05/06 inputs; rerun Step 07.")
            cut = plane["cut"]
            origin = np.asarray(cut["plane_origin_mm"], dtype=float)
            normal = np.asarray(cut["plane_normal_toward_base"], dtype=float)
            apex = np.asarray(plane["LVendo_apex_point_mm"], dtype=float)
            if (origin.shape != (3,) or normal.shape != (3,) or apex.shape != (3,)
                    or not np.isfinite(origin).all() or not np.isfinite(apex).all()
                    or not np.isclose(np.linalg.norm(normal), 1., atol=1e-10, rtol=0)
                    or (apex - origin) @ normal >= 0):
                raise RuntimeError(f"{frame} Step 07 plane has invalid coordinates, normal or apex side.")
            references = _lvrv_surface_components(load_surface(inputs[frame, "boundary"]))
            lv_epi = load_surface(inputs[frame, "LVepi_reference"]).extract_surface().triangulate()
            grid = pv.read(inputs[frame, "volume"])
            print(f"{subject} {frame}: clip the connected Step 06 volume at the saved basal plane", flush=True)
            clipped, parents, tolerance, clipping = _clip_myocardium(grid, origin, normal)
            boundary = clipped.extract_surface()
            validation = validate_volume(clipped, boundary)
            validation["boundary_matches_extracted_surface"] = validation.pop("boundary_matches_step08")
            if validation["component_count"] != 1:
                raise RuntimeError(f"{frame} opened myocardium is not connected by tetrahedron faces.")
            on_plane, rims, openings = _validate_basal_openings(boundary, references, origin, normal, tolerance)
            _validate_unchanged_wall(grid, boundary, parents, on_plane, tolerance)

            tagged, tagging = tag_lvrv_volume(clipped, lv_epi)
            tagged.cell_data["parentCellId"] = parents
            owner = np.asarray(boundary.cell_data["vtkOriginalCellIds"]).copy()
            boundary.clear_data()
            boundary.cell_data["elemTag"] = tagged.cell_data["elemTag"][owner]
            boundary.cell_data["parentCellId"] = parents[owner]
            boundary.cell_data["isBasalCut"] = on_plane.astype(np.int32)
            boundary.set_active_scalars("elemTag", preference="cell")
            basal = boundary.extract_cells(on_plane).extract_surface(
                pass_pointid=False, pass_cellid=False)
            for role, mesh in (("untagged", clipped), ("tagged", tagged)):
                path = temporary / outputs[frame, role].name
                write_tetrahedral_vtk(mesh, path)
                saved = pv.read(path)
                if (not np.array_equal(saved.points, mesh.points) or not np.array_equal(saved.cells, mesh.cells)
                        or not np.array_equal(saved.celltypes, mesh.celltypes)
                        or any(not np.array_equal(saved.cell_data.get(name), values)
                               for name, values in mesh.cell_data.items())):
                    raise RuntimeError(f"Writing {path.name} changed the validated volume or cell arrays.")
            for role, mesh in {"boundary": boundary, "basal": basal, **rims}.items():
                mesh.save(temporary / outputs[frame, role].name, binary=False)
            summary = {
                "step": "step_08_LV_RV_openbase", "subject": subject, "frame": frame,
                "inputs": {role: {"path": str(inputs[frame, role]), "sha256": hashes[frame, role]}
                           for role in input_specs},
                "plane_origin_mm": origin.tolist(), "plane_normal_toward_base": normal.tolist(),
                "kept_halfspace": "signed_distance <= 0", "plane_adjusted": False,
                "clipping": clipping, "openings": openings, "volume_validation": validation,
                "tagging": tagging, "nonbasal_wall_matches_step06": True,
                "outputs": {role: str(outputs[frame, role]) for role in suffixes if role != "summary"},
                "previous_output_backup": str(backup) if backup else None,
            }
            (temporary / outputs[frame, "summary"].name).write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="ascii")
            results.append(summary)
            print(f"{subject} {frame}: {clipped.n_cells} tetrahedra; validated two open cavities and one connected volume", flush=True)
        for key, path in inputs.items():
            if checksum(path) != hashes[key]:
                raise RuntimeError(f"Step 08 input changed: {path}")
        if backup:
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
        for path in outputs.values():
            (temporary / path.name).replace(path)
    print(f"Step 08 complete: opened meshes in {output_dir}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frames", type=parse_frames, default=["ED", "ES"])
    group.add_argument("--frame", choices=("ED", "ES"))
    parser.add_argument("--start-step", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--stop-after", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--min-wall-distance", "--rv-min-distance", dest="rv_min_distance", type=float,
                        default=3.0, help="Minimum epi/endo surface clearance for both LV and RV, in mm.")
    parser.add_argument("--rv-move-max-iterations", type=int, default=600)
    parser.add_argument("--tetgen-minratio", type=float, default=1.2)
    parser.add_argument("--tetgen-mindihedral", type=float, default=5.0)
    parser.add_argument("--tetgen-max-volume", type=float, default=0.0, help="Maximum tetrahedron volume in mm^3; 0 disables the limit.")
    parser.add_argument("--base-axis", default="auto", help="Subject long axis: auto, z, or nx,ny,nz.")
    parser.add_argument("--base-side", choices=("auto", "high", "low"), default="auto",
                        help="Which end along the chosen axis is the base.")
    parser.add_argument("--base-samples", type=int, default=160)
    parser.add_argument("--basal-search-fraction", type=float, default=.30,
                        help="Search the basal fraction of each cavity's axial extent for a common plane.")
    parser.add_argument("--overwrite", action="store_true", help="Back up and replace existing outputs of selected steps.")
    args = parser.parse_args(argv)
    if args.start_step > args.stop_after:
        parser.error("--start-step cannot exceed --stop-after.")
    frames = [args.frame] if args.frame else args.frames
    operations = {1: step_01_load_surf, 2: step_02_surf_clean, 8: step_08_LV_RV_openbase}
    for step in range(args.start_step, args.stop_after + 1):
        if step == 3:
            step_03_move_RVepi(args.subject_dir, frames, args.overwrite,
                              args.rv_min_distance, args.rv_move_max_iterations)
        elif step == 4:
            step_04_extract_LVRV_surf(args.subject_dir, frames, args.overwrite, args.rv_min_distance)
        elif step == 5:
            step_05_LVRV_surf_clean(args.subject_dir, frames, args.overwrite, args.rv_min_distance)
        elif step == 6:
            step_06_volmesh_generate(args.subject_dir, frames, args.overwrite,
                args.tetgen_minratio, args.tetgen_mindihedral, args.tetgen_max_volume)
        elif step == 7:
            step_07_base_detection(args.subject_dir, frames, args.overwrite, args.base_axis,
                                   args.base_side, args.base_samples, args.basal_search_fraction)
        else:
            operations[step](args.subject_dir, frames, args.overwrite)
    print(f"Stopped after Step {args.stop_after:02d}. No later stages were run.", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
