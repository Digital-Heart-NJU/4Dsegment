#!/usr/bin/env python3
"""New volumetric pipeline, implemented through Step 09.

Step 01: copy original surfaces without changing their geometry or arrays.
Step 02: MeshFix repair, closed-surface validation and LV/RV surface labels.
Step 03: move closed LVepi/RVepi outward, keeping both endocardia fixed and clearance >=3 mm.
Step 04: extract the outer envelope and append both complete inner surfaces.
Step 05: MeshFix-clean the combined surface, retaining both blood-pool cavities.
Step 06: TetGen mesh of the connected myocardium, then LVepi-based LV/RV tags.
Step 07: detect the subject base and save common cutting-plane previews.
Step 08: build the basal-open myocardial shell, then remesh it with TetGen.
Step 09: absorb small label islands by ordered tag 0 / tag 1 region growth.

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
        SUBJECT/volmesh_pipline/step_09_final_mesh/SUBJECT_FRAME_final_volmesh.vtk
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
Use --start-step 8 to open using the saved Step 07 apex-to-base direction.
Step 08 uses a full translation only to define reference opening rims. Their
inverse-translated curves split the original endocardia. Only the basal caps
are extended; the protected apical/lateral patches and retained epi stay fixed.
The original myocardial solid is subtracted by the locally extended cavities.
The resulting surface shell is meshed directly with TetGen. This is a new
tetrahedralization: original tetrahedral connectivity is NOT preserved.
Tags are transferred spatially from the existing tagged Step 06 volume;
referenceCellId records the source cell, not a clipped parent-child relation.
The removed-myocardium diagnostic is a raw Boolean TRIANGLE surface, which
may include coincident contact sheets; it is not used for meshing.
No C++ helper or strict local-clipping/accuracy checks are used in this preview.
Use --start-step 9 --stop-after 9 to correct only the saved Step 08 tags.
Step 09 starts from Step 08 tags, with no Step 06 overlap voting. Grow tag 1
into eligible tag 0 islands first, then tag 0 into eligible tag 1 islands.
Only complete shared triangular faces define neighborhoods. Protect both
initial main regions, and absorb only islands below a fixed volume limit.
--island-volume-percent defaults to 0.1, meaning 0.1% of the initial largest
same-tag component's volume. Larger or unreachable islands remain unchanged.
elemTag_step08, tagChanged and growthPass record the changes. JSON reports
both passes, component counts and remaining islands. growthPass=1 denotes
tag 1 growth, and growthPass=2 denotes tag 0 growth. No boundary erosion.
Step 09 preserves every point, tetrahedron, cell type and element order exactly.
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


def _solid_surface(surface, role):
    """Construct an outward-oriented solid without moving any input vertex."""
    import manifold3d

    points = np.asarray(surface.points, dtype=np.float64)
    faces = surface.faces.reshape(-1, 4)[:, 1:].copy()
    xyz = points[faces] - points.mean(axis=0)
    if np.einsum("ij,ij->i", xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    mesh = manifold3d.Mesh64(points, np.asarray(faces, dtype=np.uint64),
                            face_id=np.full(len(faces), role, dtype=np.uint64))
    solid = manifold3d.Manifold(mesh)
    if solid.status() != manifold3d.Error.NoError or solid.is_empty() or solid.volume() <= 0:
        raise RuntimeError(f"Surface role {role} does not define a valid positive solid: {solid.status()}")
    outward = pv.PolyData(points.copy(), np.column_stack((np.full(len(faces), 3), faces)).ravel())
    return solid, outward


def _basal_directional_translation(endo, outer, axis, basal_fraction=.30):
    """Use the maximum sampled basal forward exit distance, not a percentile."""
    import vtk

    if not 0 < basal_fraction <= .5:
        raise ValueError("The basal sampling fraction must be in (0, 0.5].")
    xyz = endo.points[endo.faces.reshape(-1, 4)[:, 1:]]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    projection = endo.points @ axis
    lower = float(projection.max() - basal_fraction * np.ptp(projection))
    centers = xyz.mean(axis=1)
    cap = ((centers @ axis >= lower) & (normals @ axis >= .5))
    if not np.any(cap):
        raise RuntimeError("No basal-facing endocardial triangles support an automatic displacement.")
    ids = np.unique(endo.faces.reshape(-1, 4)[cap, 1:])
    starts = np.vstack((endo.points[ids], centers[cap]))
    tree = vtk.vtkOBBTree()
    tree.SetDataSet(outer)
    tree.BuildLocator()
    ends, lengths = [], []
    tolerance = max(outer.length, 1.) * 1e-9
    for point in starts:
        hits = vtk.vtkPoints()
        hits.SetDataTypeToDouble()
        tree.IntersectWithLine(point, point + 2 * outer.length * axis, hits, vtk.vtkIdList())
        distances = np.array([(np.asarray(hits.GetPoint(i)) - point) @ axis
                              for i in range(hits.GetNumberOfPoints())])
        positive = distances[distances > tolerance]
        if not len(positive):
            raise RuntimeError("A basal endocardial sample has no forward exit through the outer surface.")
        # Use the last exit if a ray encounters a nonconvex outer envelope.
        length = float(positive.max())
        lengths.append(length)
        ends.append(point + length * axis)
    lengths = np.asarray(lengths)
    edge_mesh = endo.extract_all_edges()
    edges = edge_mesh.lines.reshape(-1, 3)[:, 1:]
    median_edge = float(np.median(np.linalg.norm(
        edge_mesh.points[edges[:, 0]] - edge_mesh.points[edges[:, 1]], axis=1)))
    margin = max(.05 * median_edge, 10 * tolerance)
    distance = float(lengths.max() + margin)
    samples = pv.PolyData(starts)
    samples.point_data["directionalThickness_mm"] = lengths
    samples.point_data["isMaximumSample"] = (lengths == lengths.max()).astype(np.int32)
    rays = pv.PolyData(np.vstack((starts, ends)),
        lines=np.column_stack((np.full(len(starts), 2), np.arange(len(starts)),
                               np.arange(len(starts)) + len(starts))).ravel())
    rays.cell_data["directionalThickness_mm"] = lengths
    return distance, samples, rays, {
        "displacement_mm": distance, "maximum_sampled_basal_directional_thickness_mm": float(lengths.max()),
        "numerical_margin_mm": margin, "median_endocardial_edge_mm": median_edge,
        "basal_fraction": basal_fraction, "basal_band_lower_axial_mm": lower,
        "basal_facing_triangle_normal_dot_minimum": .5, "sample_count": int(len(starts)),
        "measurement": "last forward epicardial exit at every basal-facing triangle vertex and centroid",
        "maximum_is_over_samples_not_a_continuous_surface_bound": True,
    }




def _manifold_surface(solid, role=None):
    """Extract source triangles, compacting unused vertices without welding."""
    mesh = solid.to_mesh64()
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    roles = np.asarray(mesh.face_id, dtype=np.int32)
    points = np.asarray(mesh.vert_properties, dtype=float)[:, :3]
    if role is not None:
        faces, roles = faces[roles == role], roles[roles == role]
    if not len(faces):
        raise RuntimeError("The Boolean result contains no triangles for the requested surface.")
    ids, inverse = np.unique(faces, return_inverse=True)
    result = pv.PolyData(points[ids].copy(), np.column_stack(
        (np.full(len(faces), 3), inverse.reshape(-1, 3))).ravel())
    result.cell_data["sourceSurface"] = roles
    return result


def _basal_patch_rim(patch):
    """Return the directed boundary edges of one disk-shaped basal patch."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = patch.faces.reshape(-1, 4)[:, 1:]
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                      return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    ids, local = np.unique(boundary, return_inverse=True)
    local = local.reshape(-1, 2)
    if (not len(ids) or np.any(counts > 2)
            or patch.n_points - len(edges) + patch.n_cells != 1
            or np.any(np.bincount(local.ravel(), minlength=len(ids)) != 2)):
        raise RuntimeError("The mapped rim must bound a single disk-shaped basal endocardial patch.")
    graph = coo_matrix((np.ones(len(local)), (local[:, 0], local[:, 1])),
                       shape=(len(ids), len(ids)))
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise RuntimeError("The basal endocardial patch has more than one boundary loop.")
    return boundary, pv.PolyData(patch.points[ids].copy(),
        lines=np.column_stack((np.full(len(local), 2), local)).ravel())


def _rim_distance(first, second):
    """Compare curve geometry in both directions, including segment midpoints."""
    errors = []
    for a, b in ((first, second), (second, first)):
        if a.n_cells != a.n_lines or len(a.lines) != 3 * a.n_lines or not a.n_lines:
            raise RuntimeError("An opening rim must consist of nonempty two-point line segments.")
        edges = a.lines.reshape(-1, 3)[:, 1:]
        samples = np.vstack((a.points, a.points[edges].mean(axis=1)))
        _, nearest = b.find_closest_cell(samples, return_closest_point=True)
        errors.append(float(np.linalg.norm(samples - nearest, axis=1).max()))
    return max(errors)




def _extended_basal_cavity(protected, basal, delta, original_role, extension_role):
    """Keep the protected wall and replace only its basal cap by an extension."""
    import manifold3d

    edges, _ = _basal_patch_rim(basal)
    a, b = edges.T
    n, k = protected.n_points, basal.n_points
    faces = np.vstack((protected.faces.reshape(-1, 4)[:, 1:],
        basal.faces.reshape(-1, 4)[:, 1:]+n+k,
        np.column_stack((a+n, b+n, b+n+k)),
        np.column_stack((a+n, b+n+k, a+n+k))))
    surface = pv.PolyData(np.vstack((protected.points, basal.points, basal.points+delta)),
                         np.column_stack((np.full(len(faces), 3), faces)).ravel())
    surface.cell_data["sourceSurface"] = np.r_[np.full(protected.n_cells, original_role),
        np.full(len(faces)-protected.n_cells, extension_role)].astype(np.int32)
    # Join the independently split copies of the same original rim.
    surface = surface.clean(tolerance=max(surface.length, 1.)*1e-11, absolute=True)
    solid = manifold3d.Manifold(manifold3d.Mesh64(np.asarray(surface.points, dtype=float),
        np.asarray(surface.faces.reshape(-1, 4)[:, 1:], dtype=np.uint64),
        face_id=np.asarray(surface.cell_data["sourceSurface"], dtype=np.uint64)))
    if solid.status() != manifold3d.Error.NoError or solid.is_empty():
        raise RuntimeError("The basal extension could not be assembled into a closed cavity.")
    return solid


def _shell_opening_rim(shell, role):
    faces = shell.faces.reshape(-1, 4)[:, 1:]
    kinds = np.asarray(shell.cell_data["sourceSurface"])
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    owners = np.tile(kinds, 3)
    epi = np.bincount(inverse, weights=owners == 1, minlength=len(unique)) > 0
    basal = np.bincount(inverse, weights=owners == role, minlength=len(unique)) > 0
    seam = unique[epi & basal]
    ids, local = np.unique(seam, return_inverse=True)
    return pv.PolyData(shell.points[ids].copy(),
        lines=np.column_stack((np.full(len(seam), 2), local.reshape(-1, 2))).ravel())


def build_open_myocardial_shell(parts, axis, basal_fraction=.30):
    """Earlier surface-based method: original myocardium minus basal extensions."""
    axis = np.asarray(axis, dtype=float)
    outer, outer_surface = _solid_surface(parts["outer"], 1)
    originals, extensions, artifacts, information = [], [], {}, {}
    for name, key, role, extension_role, tag in (
            ("LV", "LVendo", 2, 4, 1), ("RV", "RVendo", 3, 5, 0)):
        original_solid, original = _solid_surface(parts[key], role)
        distance, _, _, info = _basal_directional_translation(
            original, outer_surface, axis, basal_fraction)
        delta = distance * axis
        back_outer = outer.translate(-delta)
        basal = _manifold_surface(original_solid-back_outer, role)
        protected = _manifold_surface(original_solid ^ back_outer, role)
        _, mapped = _basal_patch_rim(basal)
        displaced = basal.translate(delta, inplace=False)
        for mesh in (basal, displaced):
            mesh.cell_data["elemTag"] = np.full(mesh.n_cells, tag, dtype=np.int32)
        artifacts.update({f"{name}_basal_patch": basal, f"{name}_basal_moved": displaced,
                          f"{name}_mapped_rim": mapped})
        originals.append(original_solid)
        extensions.append(_extended_basal_cavity(protected, basal, delta, role, extension_role))
        information[name] = info
        print(f"[Step08 {name}] basal cap {basal.n_cells} triangles; "
              f"extension {distance:.6f} mm", flush=True)
    myocardium = outer-originals[0]-originals[1]
    opened = myocardium-extensions[0]-extensions[1]
    shell = _manifold_surface(opened)
    # A raw surface diagnostic, not a tetrahedral mesh or a meshing constraint.
    removed = _manifold_surface(myocardium ^ (extensions[0]+extensions[1]))
    for name, role in (("LV", 4), ("RV", 5)):
        artifacts[f"{name}_rim"] = _shell_opening_rim(shell, role)
    return shell, removed, artifacts, information


def mesh_open_myocardial_shell(shell, minratio=1.2, mindihedral=5., max_volume=0.):
    """Pass the closed, open-base myocardial shell directly to TetGen."""
    import tetgen
    from model_vol_septum.generate_step09_volume import signed_volumes

    settings = dict(plc=True, quality=True, nobisect=True, order=1,
                    minratio=minratio, mindihedral=mindihedral, quiet=True)
    if max_volume > 0:
        settings.update(fixedvolume=True, maxvolume=max_volume)
    print(f"[Step08 TetGen] {shell.n_points} shell points, {shell.n_cells} triangles", flush=True)
    points, cells = tetgen.TetGen(shell).tetrahedralize(**settings)
    points, cells = np.asarray(points, dtype=float), np.asarray(cells, dtype=np.int64).copy()
    if not len(cells) or cells.shape[1] != 4:
        raise RuntimeError("TetGen did not produce tetrahedra.")
    volumes = signed_volumes(points, cells)
    cells[volumes < 0, 1:3] = cells[volumes < 0, 2:0:-1]
    used, local = np.unique(cells, return_inverse=True)
    return pv.UnstructuredGrid(np.column_stack((np.full(len(cells), 4), local.reshape(-1, 4))).ravel(),
        np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8), points[used])


def label_remeshed_volume(grid, original):
    """Spatially transfer the existing classification; no parent-child claim."""
    centers = grid.cell_centers().points
    reference = original.find_containing_cell(centers)
    missing = reference < 0
    if np.any(missing):
        reference[missing] = original.find_closest_cell(centers[missing])
    result = grid.copy(deep=True)
    result.cell_data["elemTag"] = np.asarray(original.cell_data["elemTag"])[reference]
    result.cell_data["referenceCellId"] = reference.astype(np.int32)
    result.set_active_scalars("elemTag", preference="cell")
    return result


def step_08_LV_RV_openbase(subject_dir, frames=("ED", "ES"), overwrite=False, basal_fraction=.30,
                         minratio=1.2, mindihedral=5., max_volume=0.):
    """Surface-based preview; save only the ten requested VTKs per frame."""
    subject_dir = Path(subject_dir).resolve()
    subject, frames = subject_dir.name, parse_frames(",".join(frames))
    root = subject_dir / "volmesh_pipline"
    output_dir = root / "step_08_LV_RV_openbase"
    specifications = {"base": ("step_07_base_detection", "base_detection.json"),
        "volume": ("step_06_volmesh_generate", "volmesh.vtk"),
        "notag": ("step_06_volmesh_generate", "volmesh_notag.vtk"),
        "surface": ("step_05_LVRV_surf_clean", "LVRV_surf.vtk")}
    inputs = {(frame, role): root / folder / f"{subject}_{frame}_{suffix}"
              for frame in frames for role, (folder, suffix) in specifications.items()}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 08 inputs: " + ", ".join(missing))
    suffixes = {"volume": "openbase_volmesh.vtk", "removed": "removed_myocardium_raw_diagnostic.vtk"}
    for name in ("LV", "RV"):
        suffixes.update({f"{name}_{role}": f"{name}_{suffix}.vtk" for role, suffix in (
            ("basal_patch", "endo_basal_patch"), ("basal_moved", "endo_basal_patch_displaced"),
            ("mapped_rim", "mapped_endo_rim"), ("rim", "opening_rim"))})
    outputs = {(frame, role): output_dir / f"{subject}_{frame}_{suffix}"
               for frame in frames for role, suffix in suffixes.items()}
    existing = [path for path in output_dir.glob(f"{subject}_*") if path.is_file()
                and any(path.name.startswith(f"{subject}_{frame}_") for frame in frames)]
    if existing and not overwrite:
        raise FileExistsError("Step 08 outputs already exist. Use --overwrite to archive them.")
    hashes = {key: checksum(path) for key, path in inputs.items()}
    backup = output_dir / "previous_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step08_shell_mesh_", dir=output_dir) as temporary:
        temporary = Path(temporary)
        for frame in frames:
            decision = json.loads(inputs[frame, "base"].read_text(encoding="ascii"))
            if (decision["subject"] != subject or decision["frame"] != frame
                    or decision["volume_sha256"] != hashes[frame, "notag"]
                    or decision["surface_sha256"] != hashes[frame, "surface"]):
                raise RuntimeError(f"{frame} Step 07 inputs changed; rerun Step 07 first.")
            original = pv.read(inputs[frame, "volume"])
            untagged = pv.read(inputs[frame, "notag"])
            if not np.array_equal(original.points, untagged.points) or not np.array_equal(original.cells, untagged.cells):
                raise RuntimeError("The tagged and untagged Step 06 meshes must have identical original tetrahedra.")
            source = load_surface(inputs[frame, "surface"])
            axis = np.asarray(decision["subject_detection"]["apex_to_base_axis"], dtype=float)
            shell, removed, artifacts, measurements = build_open_myocardial_shell(
                _lvrv_surface_components(source), axis, basal_fraction)
            volume = mesh_open_myocardial_shell(shell, minratio, mindihedral, max_volume)
            volume = label_remeshed_volume(volume, original)
            write_tetrahedral_vtk(volume, temporary / outputs[frame, "volume"].name)
            removed.save(temporary / outputs[frame, "removed"].name, binary=False)
            for role, mesh in artifacts.items():
                mesh.save(temporary / outputs[frame, role].name, binary=False)
            results.append({"subject": subject, "frame": frame,
                            "method": "basal_surface_extensions_then_TetGen",
                            "new_tetrahedralization_performed": True,
                            "tagging": "spatial_transfer_from_step06_elemTag",
                            "translation": measurements,
                            "outputs": {role: str(outputs[frame, role]) for role in suffixes}})
            print(f"{subject} {frame}: {volume.n_cells} new tetrahedra; "
                  f"LV/RV tags spatially transferred; raw removed diagnostic has {removed.n_cells} triangles", flush=True)
        for key, path in inputs.items():
            if checksum(path) != hashes[key]:
                raise RuntimeError(f"Step 08 input changed while meshing: {path}")
        if existing:
            backup.mkdir(parents=True)
            for path in existing:
                path.replace(backup / path.name)
        for path in outputs.values():
            (temporary / path.name).replace(path)
    print(f"Step 08 complete: only requested VTKs in {output_dir}", flush=True)
    return results


def _tagged_tetrahedra(grid, role):
    if not isinstance(grid, pv.UnstructuredGrid) or not grid.n_cells or not np.all(grid.celltypes == pv.CellType.TETRA):
        raise ValueError(f"{role} must be a nonempty tetrahedral mesh.")
    tags = np.asarray(grid.cell_data.get("elemTag", []))
    if tags.shape != (grid.n_cells,) or tags.dtype.kind not in "iu" or not np.isin(tags, (0, 1)).all():
        raise ValueError(f"{role} must have integer elemTag=1 (LV) or elemTag=0 (RV).")
    xyz = np.asarray(grid.points)[grid.cells.reshape(-1, 5)[:, 1:]]
    basis = np.swapaxes(xyz[:, 1:]-xyz[:, :1], 1, 2)
    if not np.isfinite(xyz).all() or np.any(np.linalg.det(basis) == 0):
        raise ValueError(f"{role} contains nonfinite or degenerate tetrahedra.")
    return xyz, basis, tags


def _assert_identical_geometry(before, after):
    for name in ("points", "cells", "celltypes"):
        a, b = np.asarray(getattr(before, name)), np.asarray(getattr(after, name))
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise RuntimeError(f"Step 09 changed {name}; the final mesh must match Step 08 exactly.")


def _grow_tag_components(tags, volumes, a, b, face_areas, island_fraction=1e-3):
    """Ordered, whole-component growth into the opposite protected main region."""
    from model_vol_septum.tag_step10_volume import component_labels

    if not np.isfinite(island_fraction) or not 0 <= island_fraction <= 1:
        raise ValueError("Island volume fraction must be between 0 and 1.")
    initial = np.asarray(tags)
    volumes = np.asarray(volumes, dtype=float)
    if initial.ndim != 1 or set(np.unique(initial)) != {0, 1}:
        raise ValueError("Bidirectional growth requires both elemTag=0 and elemTag=1.")
    if volumes.shape != initial.shape or not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Every growth cell must have a finite positive volume.")
    if not np.isfinite(face_areas).all() or np.any(face_areas <= 0):
        raise ValueError("Growth requires positive shared triangular-face areas.")
    result = initial.copy()
    growth_pass = np.zeros(len(initial), dtype=np.int32)
    labels = component_labels(result, a, b)

    def regions(current_labels):
        component_volumes = np.bincount(current_labels, weights=volumes)
        sizes = np.bincount(current_labels)
        summary = {}
        for tag in (0, 1):
            ids = np.unique(current_labels[result == tag])
            ranked = ids[np.argsort(-component_volumes[ids], kind="stable")]
            summary[str(tag)] = {
                "components": int(len(ids)), "volume_mm3": float(volumes[result == tag].sum()),
                "largest_component_volume_mm3": float(component_volumes[ranked[0]]),
                "components_ranked_by_volume": [
                    {"cells": int(sizes[i]), "volume_mm3": float(component_volumes[i])} for i in ranked]}
        return summary

    before = regions(labels)
    initial_component_count = int(labels.max())+1
    component_volumes = np.bincount(labels, weights=volumes)
    seeds, limits, protected = {}, {}, {}
    for tag in (0, 1):
        ids = np.unique(labels[result == tag])
        main = ids[np.argmax(component_volumes[ids])]
        protected[tag] = np.flatnonzero(labels == main)
        seeds[tag] = int(protected[tag][0])
        limits[tag] = float(component_volumes[main]*island_fraction)
    moves, passes = [], []
    cycle = 0
    while True:
        cycle += 1
        cycle_moves = len(moves)
        for phase, target_tag in ((1, 1), (2, 0)):
            source_tag = 1-target_tag
            pass_start = len(moves)
            components_before = int(labels.max())+1
            while True:
                component_volumes = np.bincount(labels, weights=volumes)
                roots = {tag: int(labels[seed]) for tag, seed in seeds.items()}
                component_tags = np.empty(len(component_volumes), dtype=initial.dtype)
                component_tags[labels] = result
                left, right = labels[a] == roots[target_tag], labels[b] == roots[target_tag]
                support = np.bincount(np.r_[labels[b[left]], labels[a[right]]],
                    weights=np.r_[face_areas[left], face_areas[right]], minlength=len(component_volumes))
                eligible = ((component_tags == source_tag) &
                            (component_volumes < limits[source_tag]) & (support > 0))
                eligible[list(roots.values())] = False
                candidates = np.flatnonzero(eligible)
                if not len(candidates):
                    break
                component = int(candidates[np.argmin(component_volumes[candidates])])
                ids = np.flatnonzero(labels == component)
                result[ids] = target_tag
                growth_pass[ids] = phase
                moves.append({"cycle": cycle, "pass": phase, "from_tag": source_tag, "to_tag": target_tag,
                    "cells": int(len(ids)), "cell_ids": ids.tolist(),
                    "volume_mm3": float(component_volumes[component]),
                    "supporting_face_area_mm2": float(support[component])})
                old_count = len(component_volumes)
                labels = component_labels(result, a, b)
                if int(labels.max())+1 >= old_count:
                    raise RuntimeError("Region growth did not reduce the component count.")
            passes.append({"cycle": cycle, "pass": phase, "growing_tag": target_tag,
                "components_before": components_before, "components_after": int(labels.max())+1,
                "absorbed_islands": len(moves)-pass_start,
                "changed_cells": sum(move["cells"] for move in moves[pass_start:])})
        if len(moves) == cycle_moves:
            break

    after = regions(labels)
    component_volumes = np.bincount(labels, weights=volumes)
    sizes = np.bincount(labels)
    roots = {tag: int(labels[seed]) for tag, seed in seeds.items()}
    unresolved = []
    for component in range(len(component_volumes)):
        ids = np.flatnonzero(labels == component)
        tag = int(result[ids[0]])
        if component == roots[tag]:
            continue
        reason = ("at_or_above_volume_limit" if component_volumes[component] >= limits[tag]
                  else "no_shared_face_with_opposite_main_region")
        unresolved.append({"elemTag": tag, "cells": int(sizes[component]),
            "volume_mm3": float(component_volumes[component]), "cell_ids": ids.tolist(), "reason": reason})
    for tag, ids in protected.items():
        if not np.all(result[ids] == tag):
            raise RuntimeError("Region growth changed an initial protected main component.")
    changed = result != initial
    if not np.array_equal(changed, growth_pass != 0):
        raise RuntimeError("A growth pass undid an earlier label change.")
    return result, growth_pass, {
        "method": "ordered_bidirectional_shared_face_component_growth",
        "growth_order": ["tag1_into_tag0_islands", "tag0_into_tag1_islands"],
        "growth_pass_legend": {"0": "unchanged", "1": "tag1_growth", "2": "tag0_growth"},
        "island_volume_fraction": float(island_fraction), "island_volume_percent": float(100*island_fraction),
        "island_volume_limit_mm3": {str(tag): value for tag, value in limits.items()},
        "size_rule": "island volume < fraction * initial largest same-tag component volume",
        "regions_before": before, "regions_after": after,
        "initial_component_count": initial_component_count, "final_component_count": int(labels.max())+1,
        "absorbed_islands": len(moves), "changes": moves, "growth_passes": passes,
        "cycles_with_changes": max((move["cycle"] for move in moves), default=0),
        "changed_cells": int(changed.sum()),
        "RV_to_LV": int(np.count_nonzero(changed & (result == 1))),
        "LV_to_RV": int(np.count_nonzero(changed & (result == 0))),
        "unchanged_cells": int(np.count_nonzero(~changed)),
        "unresolved_components": unresolved, "converged": True,
        "initial_main_regions_preserved": True, "step06_overlap_voting_performed": False,
        "limitation": "Wrong patches attached by faces to an initial main region are protected, not eroded."}


def correct_openbase_tags(opened, island_fraction=1e-3):
    """Grow tag 1 first, then tag 0, changing only Step 08 labels."""
    from model_vol_septum.tag_step10_volume import shared_face_neighbors

    if any(name in opened.cell_data for name in ("elemTag_step08", "tagChanged", "growthPass")):
        raise ValueError("Step 09 expects the original Step 08 file, not an already corrected result.")
    _, basis, initial = _tagged_tetrahedra(opened, "Step 08 input")
    cells = opened.cells.reshape(-1, 5)[:, 1:]
    a, b, faces = shared_face_neighbors(cells)
    triangles = opened.points[faces]
    face_areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                        triangles[:, 2]-triangles[:, 0]), axis=1)/2
    volumes = np.abs(np.linalg.det(basis))/6
    tags, growth_pass, report = _grow_tag_components(initial, volumes, a, b, face_areas, island_fraction)
    result = opened.copy(deep=True)
    result.cell_data["elemTag"] = tags
    result.cell_data["elemTag_step08"] = initial.copy()
    result.cell_data["tagChanged"] = (tags != initial).astype(np.int32)
    result.cell_data["growthPass"] = growth_pass
    result.set_active_scalars("elemTag", preference="cell")
    _assert_identical_geometry(opened, result)
    report.update({"tetrahedra": opened.n_cells, "geometry_connectivity_and_cell_order_unchanged": True})
    return result, report


def _write_step09_mesh(grid, path):
    """Lossless ASCII geometry and numeric arrays, without changing shared writers."""
    types = {"i1": "char", "u1": "unsigned_char", "i2": "short", "u2": "unsigned_short",
             "i4": "int", "u4": "unsigned_int", "i8": "vtktypeint64", "u8": "vtktypeuint64",
             "f4": "float", "f8": "double"}

    def field(stream, arrays):
        if not arrays:
            return
        stream.write(f"FIELD FieldData {len(arrays)}\n")
        for name, array in arrays.items():
            values = np.asarray(array)
            key = values.dtype.kind+str(values.dtype.itemsize)
            if key not in types or values.ndim not in (1, 2) or any(c.isspace() for c in name):
                raise ValueError(f"Unsupported numeric VTK array: {name}.")
            data = values.reshape(len(values), -1)
            stream.write(f"{name} {data.shape[1]} {len(values)} {types[key]}\n")
            np.savetxt(stream, data, fmt="%.17g" if values.dtype.kind == "f" else "%d")

    with Path(path).open("w", encoding="ascii") as stream:
        stream.write("# vtk DataFile Version 3.0\nStep 09 tag-only correction\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {grid.n_points} {'float' if grid.points.dtype.itemsize == 4 else 'double'}\n")
        np.savetxt(stream, grid.points, fmt="%.17g")
        stream.write(f"CELLS {grid.n_cells} {5*grid.n_cells}\n")
        np.savetxt(stream, grid.cells.reshape(-1, 5), fmt="%d")
        stream.write(f"CELL_TYPES {grid.n_cells}\n")
        np.savetxt(stream, grid.celltypes, fmt="%d")
        field(stream, dict(grid.field_data))
        stream.write(f"CELL_DATA {grid.n_cells}\n")
        tag_type = grid.cell_data["elemTag"].dtype
        stream.write(f"SCALARS elemTag {types[tag_type.kind+str(tag_type.itemsize)]} 1\nLOOKUP_TABLE default\n")
        np.savetxt(stream, grid.cell_data["elemTag"], fmt="%d")
        field(stream, {key: values for key, values in grid.cell_data.items() if key != "elemTag"})
        if len(grid.point_data):
            stream.write(f"POINT_DATA {grid.n_points}\n")
            field(stream, dict(grid.point_data))


def step_09_final_mesh(subject_dir, frames=("ED", "ES"), overwrite=False, island_volume_percent=.1):
    """Bidirectional island absorption on Step 08 tags, without changing geometry."""
    if not np.isfinite(island_volume_percent) or not 0 <= island_volume_percent <= 100:
        raise ValueError("Island volume percent must be between 0 and 100.")
    subject_dir = Path(subject_dir).resolve()
    subject, frames = subject_dir.name, parse_frames(",".join(frames))
    root = subject_dir / "volmesh_pipline"
    directory = root / "step_09_final_mesh"
    inputs = {frame: root/"step_08_LV_RV_openbase"/f"{subject}_{frame}_openbase_volmesh.vtk"
              for frame in frames}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 09 inputs: " + ", ".join(missing))
    outputs = {(frame, role): directory/f"{subject}_{frame}_{suffix}" for frame in frames
               for role, suffix in (("mesh", "final_volmesh.vtk"),
                                    ("summary", "step09_tag_correction_summary.json"))}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Step 09 outputs already exist. Use --overwrite to archive them.")
    hashes = {key: checksum(path) for key, path in inputs.items()}
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix=".step09_tags_", dir=directory) as temporary:
        temporary = Path(temporary)
        for frame in frames:
            print(f"{subject} {frame}: tag 1 then tag 0 growth, island volume < {island_volume_percent:g}%", flush=True)
            opened = pv.read(inputs[frame])
            result, report = correct_openbase_tags(opened, island_volume_percent/100)
            staged = temporary/outputs[frame, "mesh"].name
            _write_step09_mesh(result, staged)
            saved = pv.read(staged)
            _assert_identical_geometry(opened, saved)
            for attribute in ("cell_data", "point_data", "field_data"):
                before, after = getattr(result, attribute), getattr(saved, attribute)
                if set(before) != set(after) or any(not np.array_equal(before[key], after[key]) for key in before):
                    raise RuntimeError(f"Step 09 serialization changed {attribute}.")
            report.update({"subject": subject, "frame": frame,
                "inputs": {"opened": {"path": str(inputs[frame]), "sha256": hashes[frame]}},
                "output_path": str(outputs[frame, "mesh"]), "output_sha256": checksum(staged),
                "saved_geometry_matches_step08_exactly": True})
            (temporary/outputs[frame, "summary"].name).write_text(
                json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="ascii")
            results.append(report)
            print(f"{subject} {frame}: changed {report['changed_cells']} tags "
                  f"(RV->LV {report['RV_to_LV']}, LV->RV {report['LV_to_RV']}); "
                  f"regions 0/1: {report['regions_before']['0']['components']}/"
                  f"{report['regions_before']['1']['components']} -> "
                  f"{report['regions_after']['0']['components']}/{report['regions_after']['1']['components']}; "
                  "geometry unchanged", flush=True)
        if any(checksum(path) != hashes[key] for key, path in inputs.items()):
            raise RuntimeError("A Step 09 input changed during region growth.")
        if existing:
            backup = directory/"previous_outputs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup.mkdir(parents=True)
            for path in existing:
                path.replace(backup/path.name)
        for path in outputs.values():
            (temporary/path.name).replace(path)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frames", type=parse_frames, default=["ED", "ES"])
    group.add_argument("--frame", choices=("ED", "ES"))
    parser.add_argument("--start-step", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--stop-after", type=int, choices=range(1, 10), default=9)
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
                        help="Basal fraction used for Step 07 plane search and Step 08 directional-thickness samples.")
    parser.add_argument("--island-volume-percent", type=float, default=.1,
                        help="Step 09 island limit as percent of initial main same-tag volume; 0.1 means 0.1%%.")
    parser.add_argument("--overwrite", action="store_true", help="Back up and replace existing outputs of selected steps.")
    args = parser.parse_args(argv)
    if args.start_step > args.stop_after:
        parser.error("--start-step cannot exceed --stop-after.")
    frames = [args.frame] if args.frame else args.frames
    operations = {1: step_01_load_surf, 2: step_02_surf_clean}
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
        elif step == 8:
            step_08_LV_RV_openbase(args.subject_dir, frames, args.overwrite,
                                   args.basal_search_fraction, args.tetgen_minratio,
                                   args.tetgen_mindihedral, args.tetgen_max_volume)
        elif step == 9:
            step_09_final_mesh(args.subject_dir, frames, args.overwrite, args.island_volume_percent)
        else:
            operations[step](args.subject_dir, frames, args.overwrite)
    print(f"Stopped after Step {args.stop_after:02d}. No later stages were run.", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
