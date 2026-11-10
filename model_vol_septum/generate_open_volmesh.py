#!/usr/bin/env python3
"""New volumetric pipeline, implemented through Step 03.

Step 01: copy original surfaces without changing their geometry or arrays.
Step 02: MeshFix repair, closed-surface validation and LV/RV surface labels.
Step 03: move closed RVepi outward, keeping RVendo fixed and clearance >=3 mm.

Example:
    python model_vol_septum/generate_open_volmesh.py \
        --subject-dir data/subject_11 --frames ED,ES

Input:  SUBJECT/vtks/{LVendo,LVepi,RV,RVepi}_{ED,ES}.vtk
Output: SUBJECT/volmesh_pipline/step_01_load_surf/SUBJECT_FRAME_SURFACE.vtk
        SUBJECT/volmesh_pipline/step_02_surf_clean/SUBJECT_FRAME_SURFACE.vtk
        SUBJECT/volmesh_pipline/step_03_move_RVepi/SUBJECT_FRAME_RVepi_moved.vtk
RV is the RV endocardium. Copies retain the original VTK format and arrays.
Cleaned surfaces use ASCII VTK triangles with LV elemTag=1 and RV elemTag=0.
Use --start-step 2 to clean existing Step 01 files without copying them again.
Use --start-step 3 to move RVepi using existing Step 02 surfaces.
Use --stop-after 1 or 2 to stop earlier. Steps 04 onwards are not implemented.
Use --overwrite to back up and replace existing outputs of the selected steps.
"""

import argparse
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


def save_clean_surface(surface, path, tag):
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
    if len(tags) != saved.n_cells or not np.all(tags == tag):
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
        raise RuntimeError("Non-finite RVepi/RVendo surface distance.")
    return distance, result.nearest_points[0]


def validate_closed_rvepi(epi, endo, minimum=3.0, source=None):
    """Validate the closed moved surface without changing either input."""
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface

    quality = validate_clean_surface(epi)
    validate_clean_surface(endo)
    distance, _ = _surface_gap(epi, _surface_object(endo))
    if distance < minimum:
        raise RuntimeError(f"RVepi/RVendo surface distance is {distance:.9g} mm, below {minimum:g} mm.")
    outside = pv.PolyData(epi.points).select_enclosed_points(endo, tolerance=0.0)["SelectedPoints"] == 0
    inside = pv.PolyData(endo.points).select_enclosed_points(epi, tolerance=0.0)["SelectedPoints"] != 0
    if not np.all(outside):
        raise RuntimeError("Moved RVepi contains vertices inside RVendo.")
    if not np.all(inside):
        raise RuntimeError("Moved RVepi does not enclose the complete RVendo.")
    info = {"minimum_surface_distance_mm": distance, "self_intersection_pairs": 0,
            "RVepi_vertices_inside_RVendo": 0, "RVendo_vertices_outside_RVepi": 0,
            "free_edges": quality["free_edges"], "distance_method": "FCL_triangle_triangle"}
    if source is not None:
        if epi.points.shape != source.points.shape or not np.array_equal(epi.faces, source.faces):
            raise RuntimeError("RVepi point numbering or triangle connectivity changed.")
        reference = source.compute_normals(point_normals=True, cell_normals=False,
            consistent_normals=True, auto_orient_normals=True, split_vertices=False).point_data["Normals"]
        outward = np.einsum("ij,ij->i", epi.points - source.points, reference)
        if np.any(outward < -1e-9):
            raise RuntimeError("RVepi displacement contains inward movement.")
        info.update({"connectivity_preserved": True, "inward_displacement_vertices": 0,
                     "maximum_displacement_mm": float(np.linalg.norm(epi.points - source.points, axis=1).max())})
    return info


def move_closed_rvepi(epi, endo, minimum=3.0, max_iterations=600):
    """Displace closed RVepi along its outward directions; keep RVendo fixed."""
    import vtk
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import factorized
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface
    from model_vol_septum.mesh_intersections import confirmed_self_intersections

    if not np.isfinite(minimum) or minimum <= 0 or max_iterations < 1:
        raise ValueError("RV clearance and iteration count must be positive.")
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
    target, max_step = minimum + 0.1, min(1.0, max(0.5 * scale, 1e-3))
    for iteration in range(max_iterations + 1):
        distances = signed(result.points)
        gap, witness = _surface_gap(result, fixed_object)
        if np.all(distances >= minimum) and gap >= minimum:
            validation = validate_closed_rvepi(result, endo, minimum, source=epi)
            break
        if iteration == max_iterations:
            raise RuntimeError(f"RVepi movement did not converge; minimum vertex distance={distances.min():.6g} mm.")
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
            raise RuntimeError(f"Outward RVepi displacement stopped at iteration {iteration + 1}: "
                f"{rejected}; no invalid result saved.")
        if np.max(np.linalg.norm(step, axis=1)) < 1e-9:
            raise RuntimeError("RVepi movement stalled before meeting the requested clearance.")
        if iteration % 5 == 0:
            print(f"[Step03 RVepi] iteration={iteration + 1} minimum_signed_distance={distances.min():.6g} mm "
                  f"maximum_step={np.linalg.norm(step, axis=1).max():.6g} mm", flush=True)
    result = result.compute_normals(point_normals=True, cell_normals=True,
        consistent_normals=False, auto_orient_normals=False, split_vertices=False)
    result.cell_data["elemTag"] = np.zeros(result.n_cells, dtype=np.int32)
    result.point_data["displacement"] = result.points - original
    result.point_data["outwardDisplacement"] = np.einsum("ij,ij->i", result.points - original, reference)
    result.point_data["initialSignedDistanceToRVendo"] = initial
    result.point_data["signedDistanceToRVendo"] = distances
    result.point_data["insideRVendo"] = (distances < 0).astype(np.int32)
    return result, {"method": "outward_RVepi_normal_displacement", "iterations": iteration,
                    "minimum_requested_distance_mm": minimum,
                    "initial_vertices_inside_RVendo": int(np.count_nonzero(initial < 0)),
                    "validation": validation}


def step_03_move_RVepi(subject_dir, frames=("ED", "ES"), overwrite=False,
                       minimum=3.0, max_iterations=600):
    """Move only closed RVepi, and publish only validated saved results."""
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
    outputs = {frame: output_dir / f"{subject}_{frame}_RVepi_moved.vtk" for frame in frames}
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
                for name in ("RV", "RVepi"):
                    mesh = load_surface(inputs[frame, name])
                    if isinstance(mesh, pv.UnstructuredGrid):
                        surface = pv.PolyData(mesh.points, mesh.cells)
                        surface.point_data.update(dict(mesh.point_data))
                        surface.cell_data.update(dict(mesh.cell_data))
                    else:
                        surface = mesh.copy(deep=True)
                    surfaces[name] = surface
                print(f"{subject} {frame}: move closed RVepi outward from fixed RVendo", flush=True)
                moved, info = move_closed_rvepi(surfaces["RVepi"], surfaces["RV"],
                                               minimum=minimum, max_iterations=max_iterations)
                copy = Path(temporary) / outputs[frame].name
                save_clean_surface(moved, copy, 0)
                saved = load_surface(copy)
                verification = validate_closed_rvepi(pv.PolyData(saved.points, saved.cells),
                    surfaces["RV"], minimum, source=surfaces["RVepi"])
                for key, source in inputs.items():
                    if checksum(source) != hashes[key]:
                        raise RuntimeError(f"Step 02 input changed during movement: {source}")
                summary = {
                    "step": "step_03_move_RVepi", "subject": subject, "frame": frame,
                    "inputs": {name: str(inputs[frame, name]) for name in SURFACES},
                    "input_sha256": {name: hashes[frame, name] for name in SURFACES},
                    "output_path": str(outputs[frame]), "output_sha256": checksum(copy),
                    "movement": info, "saved_validation": verification,
                    "fixed_surfaces": ["LVendo", "LVepi", "RV"],
                    "input_files_modified": False, "LV_intersection_handling": False,
                    "stopped_after": "step_03_move_RVepi",
                    "previous_output_backup": str(backup) if backup else None,
                }
                summary_copy = Path(temporary) / summaries[frame].name
                summary_copy.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
                staged.extend(((copy, outputs[frame]), (summary_copy, summaries[frame])))
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
    print(f"Step 03 complete: validated {len(frames)} moved RVepi surfaces in {output_dir}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frames", type=parse_frames, default=["ED", "ES"])
    group.add_argument("--frame", choices=("ED", "ES"))
    parser.add_argument("--start-step", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--stop-after", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--rv-min-distance", type=float, default=3.0, help="Minimum RVepi/RVendo surface clearance in mm.")
    parser.add_argument("--rv-move-max-iterations", type=int, default=600)
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
        else:
            operations[step](args.subject_dir, frames, args.overwrite)
    print(f"Stopped after Step {args.stop_after:02d}. No later stages were run.", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
