#!/usr/bin/env python3
"""Fix RV/RVepi surface intersections against fixed LVepi and save tagged VTKs.

LVendo and LVepi are preserved geometrically. RV and RVepi are corrected with a
signed-distance clearance from LVepi, then all four surfaces are written as VTK
legacy ASCII unstructured-grid triangle meshes with a cell scalar named elemTag:

  LVendo = 1
  LVepi  = 2
  RV     = 3
  RVepi  = 4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyvista as pv
import vtk


SURFACE_TAGS = {
    "LVendo": 1,
    "LVepi": 2,
    "RV": 3,
    "RVepi": 4,
}


def load_surface(path: Path, tolerance: float) -> pv.PolyData:
    if not path.exists():
        raise FileNotFoundError(path)
    mesh = pv.read(path).extract_surface().triangulate().clean(tolerance=tolerance)
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise RuntimeError(f"Loaded empty surface: {path}")
    return orient_surface(mesh, tolerance=tolerance)


def orient_surface(mesh: pv.PolyData, tolerance: float) -> pv.PolyData:
    return mesh.compute_normals(
        cell_normals=False,
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
        inplace=False,
    ).extract_surface().triangulate().clean(tolerance=tolerance)


def triangle_faces(mesh: pv.PolyData) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.int64)
    cells = faces.reshape((-1, 4))
    if not np.all(cells[:, 0] == 3):
        raise ValueError("Expected a triangulated surface with 3-point cells.")
    return cells[:, 1:4].astype(np.int64)


def faces_from_triangles(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    return np.column_stack([np.full(len(triangles), 3, dtype=np.int64), triangles]).ravel()


def make_surface(points: np.ndarray, triangles: np.ndarray, tolerance: float) -> pv.PolyData:
    return pv.PolyData(
        np.asarray(points, dtype=float),
        faces_from_triangles(np.asarray(triangles, dtype=np.int64)),
    ).extract_surface().triangulate().clean(tolerance=tolerance)


def write_custom_surface_vtk(filename: Path, points: np.ndarray, triangles: np.ndarray, elem_tag: int) -> None:
    """Write a triangle surface in the same legacy ASCII style as custom_vtk.write_custom_vtk."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    elem_tags = np.full(len(triangles), int(elem_tag), dtype=np.int32)

    with filename.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")

        f.write(f"POINTS {len(points)} float\n")
        np.savetxt(f, points, fmt="%.5f", delimiter=" ")
        f.write("\n")

        f.write(f"CELL_TYPES {len(triangles)}\n")
        np.savetxt(f, np.full(len(triangles), 5, dtype=np.int32), fmt="%d")
        f.write("\n")

        f.write(f"CELLS {len(triangles)} {4 * len(triangles)}\n")
        cell_lines = np.column_stack([np.full(len(triangles), 3, dtype=np.int32), triangles])
        np.savetxt(f, cell_lines, fmt="%d")
        f.write("\n")

        f.write(f"CELL_DATA {len(triangles)}\n")
        f.write("SCALARS elemTag int 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, elem_tags.reshape(-1, 1), fmt="%d")
        f.write("\n")


def write_tag_legend(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["surface", "elemTag"])
        for name, tag in SURFACE_TAGS.items():
            writer.writerow([name, tag])


def enclosed_mask(points: np.ndarray, shell: pv.PolyData) -> np.ndarray:
    selected = pv.PolyData(np.asarray(points, dtype=float)).select_enclosed_points(
        shell,
        tolerance=1e-5,
        check_surface=False,
    )
    return selected.point_data["SelectedPoints"].astype(bool)


def signed_distance_and_gradient(reference: pv.PolyData, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(reference)
    pts = np.asarray(points, dtype=float)
    distances = np.empty(len(pts), dtype=float)
    gradients = np.empty((len(pts), 3), dtype=float)
    for index, point in enumerate(pts):
        p = [float(point[0]), float(point[1]), float(point[2])]
        grad = [0.0, 0.0, 0.0]
        distances[index] = float(implicit.EvaluateFunction(p))
        implicit.EvaluateGradient(p, grad)
        gradients[index] = grad

    norms = np.linalg.norm(gradients, axis=1)
    bad = norms < 1e-12
    if np.any(bad):
        center = np.asarray(reference.center, dtype=float)
        fallback = pts[bad] - center[None, :]
        fallback_norm = np.maximum(np.linalg.norm(fallback, axis=1), 1e-12)
        gradients[bad] = fallback / fallback_norm[:, None]
        norms[bad] = 1.0
    return distances, gradients / np.maximum(norms[:, None], 1e-12)


def intersection_curve(first: pv.PolyData, second: pv.PolyData, tolerance: float) -> pv.PolyData:
    intersector = vtk.vtkIntersectionPolyDataFilter()
    intersector.SetInputData(0, first)
    intersector.SetInputData(1, second)
    if hasattr(intersector, "SetTolerance"):
        intersector.SetTolerance(float(tolerance))
    if hasattr(intersector, "SplitFirstOutputOff"):
        intersector.SplitFirstOutputOff()
    if hasattr(intersector, "SplitSecondOutputOff"):
        intersector.SplitSecondOutputOff()
    intersector.Update()
    return pv.wrap(intersector.GetOutput()).clean(tolerance=tolerance)


def intersection_summary(
    first_name: str,
    first: pv.PolyData,
    second_name: str,
    second: pv.PolyData,
    close_distance: float,
    intersection_tolerance: float,
) -> Tuple[Dict[str, object], pv.PolyData, pv.PolyData, pv.PolyData]:
    curve = intersection_curve(first, second, intersection_tolerance)
    first_points = np.asarray(first.points)
    second_points = np.asarray(second.points)
    first_inside = enclosed_mask(first_points, second)
    second_inside = enclosed_mask(second_points, first)
    first_distances, _ = signed_distance_and_gradient(second, first_points)
    second_distances, _ = signed_distance_and_gradient(first, second_points)

    first_inside_points = pv.PolyData(first_points[first_inside]) if np.any(first_inside) else pv.PolyData()
    second_inside_points = pv.PolyData(second_points[second_inside]) if np.any(second_inside) else pv.PolyData()

    summary = {
        "pair": f"{first_name}_vs_{second_name}",
        "has_intersection_or_penetration": bool(
            curve.n_points > 0 or curve.n_cells > 0 or np.any(first_inside) or np.any(second_inside)
        ),
        "intersection_points": int(curve.n_points),
        "intersection_cells": int(curve.n_cells),
        "intersection_length": float(curve.length) if curve.n_points else 0.0,
        f"{first_name}_points_inside_{second_name}": int(np.count_nonzero(first_inside)),
        f"{second_name}_points_inside_{first_name}": int(np.count_nonzero(second_inside)),
        f"{first_name}_points_close_to_{second_name}": int(np.count_nonzero(np.abs(first_distances) <= close_distance)),
        f"{second_name}_points_close_to_{first_name}": int(np.count_nonzero(np.abs(second_distances) <= close_distance)),
        f"{first_name}_min_signed_distance_to_{second_name}": float(first_distances.min()) if len(first_distances) else None,
        f"{second_name}_min_signed_distance_to_{first_name}": float(second_distances.min()) if len(second_distances) else None,
    }
    return summary, curve, first_inside_points, second_inside_points


def vertex_neighbors(triangles: np.ndarray, point_count: int) -> List[set]:
    neighbors = [set() for _ in range(point_count)]
    for tri in triangles:
        a, b, c = (int(v) for v in tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = set()
    for tri in triangles:
        a, b, c = (int(v) for v in tri)
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.asarray(sorted(edges), dtype=np.int64)


def nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        return cKDTree(reference).query(query)[1].astype(np.int64)
    except Exception:
        indices: List[int] = []
        query = np.asarray(query, dtype=float)
        reference = np.asarray(reference, dtype=float)
        for start in range(0, len(query), 256):
            chunk = query[start:start + 256]
            dist2 = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            indices.extend(np.argmin(dist2, axis=1).astype(int).tolist())
        return np.asarray(indices, dtype=np.int64)


def blend_displacement(
    points: np.ndarray,
    triangles: np.ndarray,
    displacement: np.ndarray,
    source_mask: np.ndarray,
    rings: int,
    falloff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if rings <= 0 or not np.any(source_mask):
        return displacement, source_mask.copy()

    source_ids = np.flatnonzero(source_mask)
    source_points = points[source_ids]
    source_displacement = displacement[source_ids]
    neighbors = vertex_neighbors(triangles, len(points))
    blended = displacement.copy()
    moving = source_mask.copy()
    visited = set(int(v) for v in source_ids)
    frontier = set(visited)

    for ring in range(1, rings + 1):
        next_ring = set()
        for vertex_id in frontier:
            for nb in neighbors[vertex_id]:
                if nb not in visited:
                    next_ring.add(int(nb))
        if not next_ring:
            break

        ring_ids = np.asarray(sorted(next_ring), dtype=np.int64)
        nearest = nearest_indices(points[ring_ids], source_points)
        weight = float(falloff) ** float(ring)
        blended[ring_ids] += source_displacement[nearest] * weight
        moving[ring_ids] = True
        visited.update(next_ring)
        frontier = next_ring

    return blended, moving


def smooth_displacement(
    triangles: np.ndarray,
    displacement: np.ndarray,
    moving_mask: np.ndarray,
    iterations: int,
    relaxation: float,
) -> np.ndarray:
    if iterations <= 0 or not np.any(moving_mask):
        return displacement

    neighbors = vertex_neighbors(triangles, len(displacement))
    moving_ids = np.flatnonzero(moving_mask)
    out = displacement.copy()
    for _ in range(iterations):
        updated = out.copy()
        for vertex_id in moving_ids:
            nb = list(neighbors[int(vertex_id)])
            if not nb:
                continue
            avg = out[np.asarray(nb, dtype=np.int64)].mean(axis=0)
            updated[vertex_id] = out[vertex_id] + float(relaxation) * (avg - out[vertex_id])
        out = updated
    return out


def add_conflict_displacement(
    displacement: np.ndarray,
    counts: np.ndarray,
    vertex_ids: np.ndarray,
    required: np.ndarray,
    gradients: np.ndarray,
) -> None:
    if len(vertex_ids) == 0:
        return
    disp = gradients * required[:, None]
    repeated_disp = np.repeat(disp, vertex_ids.shape[1], axis=0)
    flat_ids = vertex_ids.reshape(-1)
    np.add.at(displacement, flat_ids, repeated_disp)
    np.add.at(counts, flat_ids, 1.0)


def correction_iteration(
    mesh: pv.PolyData,
    protected: pv.PolyData,
    clearance: float,
    relaxation: float,
    max_step: float,
    blend_rings: int,
    blend_falloff: float,
    displacement_smooth_iterations: int,
    displacement_smooth_relaxation: float,
    tolerance: float,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    points = np.asarray(mesh.points).copy()
    triangles = triangle_faces(mesh)
    displacement = np.zeros_like(points)
    counts = np.zeros(len(points), dtype=float)

    distances, gradients = signed_distance_and_gradient(protected, points)
    inside = enclosed_mask(points, protected)
    required = np.maximum(float(clearance) - distances, 0.0)
    vertex_conflict = (required > 0.0) | inside
    add_conflict_displacement(
        displacement,
        counts,
        np.flatnonzero(vertex_conflict)[:, None],
        required[vertex_conflict],
        gradients[vertex_conflict],
    )

    centers = points[triangles].mean(axis=1)
    center_distances, center_gradients = signed_distance_and_gradient(protected, centers)
    center_required = np.maximum(float(clearance) - center_distances, 0.0)
    center_conflict = center_required > 0.0
    add_conflict_displacement(
        displacement,
        counts,
        triangles[center_conflict],
        center_required[center_conflict],
        center_gradients[center_conflict],
    )

    edges = unique_edges(triangles)
    midpoints = points[edges].mean(axis=1)
    edge_distances, edge_gradients = signed_distance_and_gradient(protected, midpoints)
    edge_required = np.maximum(float(clearance) - edge_distances, 0.0)
    edge_conflict = edge_required > 0.0
    add_conflict_displacement(
        displacement,
        counts,
        edges[edge_conflict],
        edge_required[edge_conflict],
        edge_gradients[edge_conflict],
    )

    source_mask = counts > 0.0
    displacement[source_mask] /= counts[source_mask, None]
    displacement *= float(relaxation)

    step_lengths = np.linalg.norm(displacement, axis=1)
    too_large = step_lengths > float(max_step)
    if np.any(too_large):
        displacement[too_large] *= (float(max_step) / step_lengths[too_large])[:, None]

    displacement, moving_mask = blend_displacement(
        points,
        triangles,
        displacement,
        source_mask,
        rings=blend_rings,
        falloff=blend_falloff,
    )
    displacement = smooth_displacement(
        triangles,
        displacement,
        moving_mask,
        iterations=displacement_smooth_iterations,
        relaxation=displacement_smooth_relaxation,
    )

    corrected_points = points + displacement
    corrected_distances, corrected_gradients = signed_distance_and_gradient(protected, corrected_points)
    final_required = np.maximum(float(clearance) - corrected_distances, 0.0)
    final_mask = final_required > 0.0
    if np.any(final_mask):
        corrected_points[final_mask] += corrected_gradients[final_mask] * final_required[final_mask, None]
        moving_mask[final_mask] = True

    corrected = make_surface(corrected_points, triangles, tolerance=tolerance)
    step_lengths = np.linalg.norm(corrected_points - points, axis=1)
    info = {
        "vertex_conflicts": int(np.count_nonzero(vertex_conflict)),
        "center_conflicts": int(np.count_nonzero(center_conflict)),
        "edge_conflicts": int(np.count_nonzero(edge_conflict)),
        "moved_vertices": int(np.count_nonzero(moving_mask)),
        "max_step": float(step_lengths.max()) if len(step_lengths) else 0.0,
        "mean_step_on_moved_vertices": float(step_lengths[moving_mask].mean()) if np.any(moving_mask) else 0.0,
    }
    return orient_surface(corrected, tolerance=tolerance), info


def fix_surface_against_lv_epi(
    name: str,
    mesh: pv.PolyData,
    lv_epi: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    current = mesh
    iterations: List[Dict[str, object]] = []
    initial, _, _, _ = intersection_summary(
        name,
        current,
        "LVepi",
        lv_epi,
        close_distance=args.close_distance,
        intersection_tolerance=args.intersection_tolerance,
    )

    for index in range(1, args.max_iterations + 1):
        current, iter_info = correction_iteration(
            current,
            protected=lv_epi,
            clearance=args.clearance,
            relaxation=args.relaxation,
            max_step=args.max_step,
            blend_rings=args.blend_rings,
            blend_falloff=args.blend_falloff,
            displacement_smooth_iterations=args.displacement_smooth_iterations,
            displacement_smooth_relaxation=args.displacement_smooth_relaxation,
            tolerance=args.clean_tolerance,
        )
        after, _, _, _ = intersection_summary(
            name,
            current,
            "LVepi",
            lv_epi,
            close_distance=args.close_distance,
            intersection_tolerance=args.intersection_tolerance,
        )
        iter_info["iteration"] = int(index)
        iter_info["after"] = after
        iterations.append(iter_info)

        if (
            after["intersection_points"] == 0
            and after["intersection_cells"] == 0
            and after[f"{name}_points_inside_LVepi"] == 0
        ):
            break

    final, _, _, _ = intersection_summary(
        name,
        current,
        "LVepi",
        lv_epi,
        close_distance=args.close_distance,
        intersection_tolerance=args.intersection_tolerance,
    )
    return current, {
        "method": "signed_distance_clearance_projection_with_face_and_edge_sampling",
        "protected_surface": "LVepi",
        "moving_surface": name,
        "initial": initial,
        "final": final,
        "iterations": iterations,
        "parameters": {
            "clearance": float(args.clearance),
            "max_iterations": int(args.max_iterations),
            "relaxation": float(args.relaxation),
            "max_step": float(args.max_step),
            "blend_rings": int(args.blend_rings),
            "blend_falloff": float(args.blend_falloff),
            "displacement_smooth_iterations": int(args.displacement_smooth_iterations),
            "displacement_smooth_relaxation": float(args.displacement_smooth_relaxation),
        },
    }


def save_debug_pair(
    output_dir: Path,
    label: str,
    first_name: str,
    first: pv.PolyData,
    second_name: str,
    second: pv.PolyData,
    args: argparse.Namespace,
) -> Dict[str, object]:
    summary, curve, first_inside_points, second_inside_points = intersection_summary(
        first_name,
        first,
        second_name,
        second,
        close_distance=args.close_distance,
        intersection_tolerance=args.intersection_tolerance,
    )
    paths: Dict[str, object] = {}
    if curve.n_points:
        path = output_dir / f"{label}_{first_name}_{second_name}_intersection_curve.vtk"
        curve.save(path)
        paths["intersection_curve_vtk"] = str(path.resolve())
    if first_inside_points.n_points:
        path = output_dir / f"{label}_{first_name}_points_inside_{second_name}.vtk"
        first_inside_points.save(path)
        paths[f"{first_name}_inside_{second_name}_vtk"] = str(path.resolve())
    if second_inside_points.n_points:
        path = output_dir / f"{label}_{second_name}_points_inside_{first_name}.vtk"
        second_inside_points.save(path)
        paths[f"{second_name}_inside_{first_name}_vtk"] = str(path.resolve())
    summary["debug_files"] = paths
    return summary


def surface_summary(mesh: pv.PolyData, tag: int, path: Path) -> Dict[str, object]:
    return {
        "path": str(path.resolve()),
        "elemTag": int(tag),
        "points": int(mesh.n_points),
        "triangles": int(mesh.n_cells),
        "surface_area": float(mesh.area),
        "bounds": [float(value) for value in mesh.bounds],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clearance", type=float, default=0.35)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--max-step", type=float, default=2.0)
    parser.add_argument("--blend-rings", type=int, default=3)
    parser.add_argument("--blend-falloff", type=float, default=0.45)
    parser.add_argument("--displacement-smooth-iterations", type=int, default=8)
    parser.add_argument("--displacement-smooth-relaxation", type=float, default=0.25)
    parser.add_argument("--clean-tolerance", type=float, default=1e-7)
    parser.add_argument("--intersection-tolerance", type=float, default=1e-6)
    parser.add_argument("--close-distance", type=float, default=0.25)
    parser.add_argument("--skip-debug-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_dir = args.subject_dir
    frame = args.frame
    input_dir = subject_dir / "vtks"
    output_dir = args.output_dir or subject_dir / "vtks_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "LVendo": input_dir / f"LVendo_{frame}.vtk",
        "LVepi": input_dir / f"LVepi_{frame}.vtk",
        "RV": input_dir / f"RV_{frame}.vtk",
        "RVepi": input_dir / f"RVepi_{frame}.vtk",
    }
    surfaces = {
        name: load_surface(path, args.clean_tolerance)
        for name, path in input_paths.items()
    }

    before_checks = {}
    if not args.skip_debug_files:
        before_checks["LVepi_vs_RV"] = save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_before", "LVepi", surfaces["LVepi"], "RV", surfaces["RV"], args
        )
        before_checks["LVepi_vs_RVepi"] = save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_before", "LVepi", surfaces["LVepi"], "RVepi", surfaces["RVepi"], args
        )

    fixed_rv, rv_fix = fix_surface_against_lv_epi("RV", surfaces["RV"], surfaces["LVepi"], args)
    fixed_rvepi, rvepi_fix = fix_surface_against_lv_epi("RVepi", surfaces["RVepi"], surfaces["LVepi"], args)

    fixed_surfaces = {
        "LVendo": surfaces["LVendo"],
        "LVepi": surfaces["LVepi"],
        "RV": fixed_rv,
        "RVepi": fixed_rvepi,
    }

    output_paths = {
        name: output_dir / f"{name}_{frame}.vtk"
        for name in ("LVendo", "LVepi", "RV", "RVepi")
    }
    for name, mesh in fixed_surfaces.items():
        write_custom_surface_vtk(
            output_paths[name],
            np.asarray(mesh.points),
            triangle_faces(mesh),
            SURFACE_TAGS[name],
        )

    write_tag_legend(output_dir / f"{subject_dir.name}_{frame}_surface_elemTag_legend.csv")

    after_checks = {}
    if not args.skip_debug_files:
        after_checks["LVepi_vs_RV"] = save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_after", "LVepi", fixed_surfaces["LVepi"], "RV", fixed_surfaces["RV"], args
        )
        after_checks["LVepi_vs_RVepi"] = save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_after", "LVepi", fixed_surfaces["LVepi"], "RVepi", fixed_surfaces["RVepi"], args
        )
        after_checks["RV_vs_RVepi"] = save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_after", "RV", fixed_surfaces["RV"], "RVepi", fixed_surfaces["RVepi"], args
        )

    outputs = {
        name: surface_summary(fixed_surfaces[name], SURFACE_TAGS[name], output_paths[name])
        for name in ("LVendo", "LVepi", "RV", "RVepi")
    }
    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
        "output_dir": str(output_dir.resolve()),
        "outputs": outputs,
        "method": {
            "protected_geometry": "LVendo and LVepi are not moved. LVepi is used as the protected collision surface.",
            "moving_geometry": "RV and RVepi are moved only where their vertices, face centers, or edge midpoints violate the LVepi signed-distance clearance.",
            "output_format": "Legacy ASCII UNSTRUCTURED_GRID triangle surfaces, matching custom_vtk.py style, with CELL_DATA scalar elemTag.",
        },
        "fixes": {
            "RV": rv_fix,
            "RVepi": rvepi_fix,
        },
        "before_checks": before_checks,
        "after_checks": after_checks,
    }
    summary_path = output_dir / f"{subject_dir.name}_{frame}_surface_intersection_fix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "summary": str(summary_path),
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "elemTags": SURFACE_TAGS,
        "RV_final": rv_fix["final"],
        "RVepi_final": rvepi_fix["final"],
    }, indent=2))


if __name__ == "__main__":
    main()
