#!/usr/bin/env python3
"""Merge separate open-base LV/RV volume meshes while editing only RV.

This post-processing script starts from the outputs of
model_vol_septum/generate_open_volmesh_seperate.py:

  SUBJECT_DIR/volmesh_LVRV/SUBJECT_FRAME_openbase_LV_volmesh.vtk
  SUBJECT_DIR/volmesh_LVRV/SUBJECT_FRAME_openbase_RV_volmesh.vtk

Rules:
  - LV points/cells are kept unchanged.
  - RV tetrahedra overlapping the LV volume are removed.
  - Small disconnected RV parts after removal are discarded.
  - RV vertices near the LV/RV interface are retracted away from LV and then
    smoothed with a volume-quality guard.
  - The final merged VTK contains elemTag=1 for LV and elemTag=0 for RV.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MESH_VOL_DIR = REPO_ROOT / "mesh_vol"
if str(MESH_VOL_DIR) not in sys.path:
    sys.path.insert(0, str(MESH_VOL_DIR))

from custom_vtk import write_custom_vtk


TETRA_CELL_TYPE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--frames", default=None, help="Comma-separated frames, e.g. ED,ES.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Defaults to SUBJECT_DIR/volmesh_LVRV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to SUBJECT_DIR/volmesh_LVRV_merged_clean.")
    parser.add_argument("--lv-tag", type=int, default=1)
    parser.add_argument("--rv-tag", type=int, default=0)
    parser.add_argument("--inside-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--overlap-vertex-fraction",
        type=float,
        default=0.25,
        help="Remove an RV tet when this fraction of its vertices is inside LV. 0.25 means at least one vertex.",
    )
    parser.add_argument("--remove-overlap-rings", type=int, default=1)
    parser.add_argument("--keep-rv-components", choices=["largest", "large"], default="largest")
    parser.add_argument("--min-component-fraction", type=float, default=0.02)
    parser.add_argument("--retract-distance-mm", type=float, default=None)
    parser.add_argument("--retract-distance-edge-multiplier", type=float, default=0.60)
    parser.add_argument("--retract-active-distance-mm", type=float, default=None)
    parser.add_argument("--retract-active-edge-multiplier", type=float, default=2.0)
    parser.add_argument("--retract-rings", type=int, default=2)
    parser.add_argument("--smooth-iterations", type=int, default=10)
    parser.add_argument("--smooth-relaxation", type=float, default=0.12)
    parser.add_argument("--smooth-rings", type=int, default=3)
    parser.add_argument("--max-move-edge-multiplier", type=float, default=0.35)
    parser.add_argument("--min-positive-volume", type=float, default=0.0)
    parser.add_argument("--quality-line-search-steps", type=int, default=8)
    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


def parse_csv_values(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def tet_faces(cell: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    a, b, c, d = (int(v) for v in cell)
    return (
        tuple(sorted((a, b, c))),
        tuple(sorted((a, b, d))),
        tuple(sorted((a, c, d))),
        tuple(sorted((b, c, d))),
    )


def tet_edges(cell: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    a, b, c, d = (int(v) for v in cell)
    return (
        tuple(sorted((a, b))),
        tuple(sorted((a, c))),
        tuple(sorted((a, d))),
        tuple(sorted((b, c))),
        tuple(sorted((b, d))),
        tuple(sorted((c, d))),
    )


def tetra_connectivity(grid: pv.UnstructuredGrid) -> np.ndarray:
    if not np.all(np.asarray(grid.celltypes) == TETRA_CELL_TYPE):
        grid = grid.extract_cells(np.flatnonzero(np.asarray(grid.celltypes) == TETRA_CELL_TYPE))
    raw = np.asarray(grid.cells, dtype=np.int64)
    cells: List[np.ndarray] = []
    i = 0
    while i < len(raw):
        size = int(raw[i])
        if size == 4:
            cells.append(raw[i + 1:i + 5])
        i += size + 1
    if not cells:
        return np.empty((0, 4), dtype=np.int64)
    return np.asarray(cells, dtype=np.int64)


def make_grid(points: np.ndarray, cells: np.ndarray, tags: Optional[np.ndarray] = None) -> pv.UnstructuredGrid:
    cell_sizes = np.full((len(cells), 1), 4, dtype=np.int64)
    cells_flat = np.hstack([cell_sizes, cells.astype(np.int64)]).ravel()
    celltypes = np.full(len(cells), TETRA_CELL_TYPE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells_flat, celltypes, np.asarray(points, dtype=float))
    if tags is not None:
        grid.cell_data["elemTag"] = np.asarray(tags, dtype=np.int32)
    return grid


def signed_tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    a = points[cells[:, 0]]
    b = points[cells[:, 1]]
    c = points[cells[:, 2]]
    d = points[cells[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def reorient_tetrahedra(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    out = np.asarray(cells, dtype=np.int64).copy()
    volumes = signed_tet_volumes(points, out)
    negative = volumes < 0.0
    if np.any(negative):
        ids = np.flatnonzero(negative)
        first = out[ids, 0].copy()
        out[ids, 0] = out[ids, 1]
        out[ids, 1] = first
    after = signed_tet_volumes(points, out)
    return out, {
        "negative_tets_reoriented": int(np.count_nonzero(negative)),
        "nonpositive_after_reorientation": int(np.count_nonzero(after <= 0.0)),
    }


def median_tet_edge_length(points: np.ndarray, cells: np.ndarray) -> float:
    sample = cells if len(cells) <= 25000 else cells[np.linspace(0, len(cells) - 1, 25000).astype(np.int64)]
    lengths: List[np.ndarray] = []
    for edge_id in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        lengths.append(np.linalg.norm(points[sample[:, edge_id[0]]] - points[sample[:, edge_id[1]]], axis=1))
    return float(np.median(np.concatenate(lengths))) if lengths else 1.0


def nearest_distances_and_indices(query: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(np.asarray(reference, dtype=float)).query(np.asarray(query, dtype=float))
        return distances.astype(float), indices.astype(np.int64)
    except Exception:
        distances: List[np.ndarray] = []
        indices: List[np.ndarray] = []
        ref = np.asarray(reference, dtype=float)
        for start in range(0, len(query), 512):
            chunk = np.asarray(query[start:start + 512], dtype=float)
            dist2 = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(dist2, axis=1)
            indices.append(idx.astype(np.int64))
            distances.append(np.sqrt(dist2[np.arange(len(chunk)), idx]))
        return np.concatenate(distances), np.concatenate(indices)


def enclosed_mask(points: np.ndarray, closed_surface: pv.PolyData, tolerance: float) -> np.ndarray:
    pdata = pv.PolyData(np.asarray(points, dtype=float))
    selected = pdata.select_enclosed_points(closed_surface, tolerance=float(tolerance), check_surface=False)
    return np.asarray(selected.point_data["SelectedPoints"], dtype=bool)


def face_adjacency(cells: np.ndarray) -> List[List[int]]:
    face_owner: Dict[Tuple[int, int, int], int] = {}
    adjacency = [set() for _ in range(len(cells))]
    for cell_id, cell in enumerate(cells):
        for face in tet_faces(cell):
            other = face_owner.get(face)
            if other is None:
                face_owner[face] = int(cell_id)
            else:
                adjacency[cell_id].add(other)
                adjacency[other].add(int(cell_id))
    return [sorted(values) for values in adjacency]


def connected_components(mask: np.ndarray, adjacency: Sequence[Sequence[int]]) -> List[np.ndarray]:
    remaining = set(int(v) for v in np.flatnonzero(mask))
    components: List[np.ndarray] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            cell_id = stack.pop()
            for neighbor in adjacency[cell_id]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(int(neighbor))
                    comp.append(int(neighbor))
        components.append(np.asarray(comp, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def expand_cell_mask(seed: np.ndarray, adjacency: Sequence[Sequence[int]], rings: int) -> np.ndarray:
    expanded = np.asarray(seed, dtype=bool).copy()
    frontier = set(int(v) for v in np.flatnonzero(seed))
    for _ in range(max(0, int(rings))):
        next_frontier = set()
        for cell_id in frontier:
            for neighbor in adjacency[cell_id]:
                if not expanded[neighbor]:
                    expanded[neighbor] = True
                    next_frontier.add(int(neighbor))
        if not next_frontier:
            break
        frontier = next_frontier
    return expanded


def vertex_neighbors_from_cells(cells: np.ndarray, point_count: int) -> List[set]:
    neighbors = [set() for _ in range(point_count)]
    for cell in cells:
        for a, b in tet_edges(cell):
            neighbors[a].add(b)
            neighbors[b].add(a)
    return neighbors


def boundary_vertex_mask(cells: np.ndarray, point_count: int) -> np.ndarray:
    counts: Dict[Tuple[int, int, int], int] = {}
    for cell in cells:
        for face in tet_faces(cell):
            counts[face] = counts.get(face, 0) + 1
    mask = np.zeros(point_count, dtype=bool)
    for face, count in counts.items():
        if count == 1:
            mask[list(face)] = True
    return mask


def expand_vertex_weights(seed: np.ndarray, neighbors: Sequence[set], rings: int) -> np.ndarray:
    weights = np.zeros(len(seed), dtype=float)
    weights[seed] = 1.0
    visited = set(int(v) for v in np.flatnonzero(seed))
    frontier = set(visited)
    for ring in range(1, max(0, int(rings)) + 1):
        next_frontier = set()
        for vertex in frontier:
            for neighbor in neighbors[vertex]:
                if neighbor not in visited:
                    next_frontier.add(int(neighbor))
        if not next_frontier:
            break
        weight = float(rings - ring + 1) / float(rings + 1)
        ids = np.fromiter(sorted(next_frontier), dtype=np.int64)
        weights[ids] = np.maximum(weights[ids], weight)
        visited.update(next_frontier)
        frontier = next_frontier
    return weights


def compact_mesh(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    used = np.unique(cells.ravel())
    point_map = -np.ones(len(points), dtype=np.int64)
    point_map[used] = np.arange(len(used), dtype=np.int64)
    return np.asarray(points, dtype=float)[used], point_map[cells], used


def remove_overlapping_rv_cells(
    lv_surface: pv.PolyData,
    rv_points: np.ndarray,
    rv_cells: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    centroids = rv_points[rv_cells].mean(axis=1)
    vertices = rv_points[rv_cells.reshape(-1)].reshape((len(rv_cells), 4, 3))
    centroid_inside = enclosed_mask(centroids, lv_surface, args.inside_tolerance)
    vertex_inside = enclosed_mask(vertices.reshape((-1, 3)), lv_surface, args.inside_tolerance).reshape((len(rv_cells), 4))
    vertex_fraction = vertex_inside.mean(axis=1)
    overlap = centroid_inside | (vertex_fraction >= float(args.overlap_vertex_fraction))
    adjacency = face_adjacency(rv_cells)
    expanded_overlap = expand_cell_mask(overlap, adjacency, args.remove_overlap_rings)
    keep = ~expanded_overlap
    info = {
        "rv_cells_before_overlap_removal": int(len(rv_cells)),
        "centroid_inside_lv_cells": int(np.count_nonzero(centroid_inside)),
        "vertex_inside_lv_cells": int(np.count_nonzero(vertex_fraction > 0.0)),
        "overlap_cells_before_ring_expansion": int(np.count_nonzero(overlap)),
        "remove_overlap_rings": int(args.remove_overlap_rings),
        "overlap_cells_after_ring_expansion": int(np.count_nonzero(expanded_overlap)),
        "rv_cells_after_overlap_removal": int(np.count_nonzero(keep)),
    }
    return keep, info


def keep_main_rv_component(cells: np.ndarray, initial_keep: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object]]:
    adjacency = face_adjacency(cells)
    components = connected_components(initial_keep, adjacency)
    keep = np.zeros(len(cells), dtype=bool)
    if components:
        if args.keep_rv_components == "largest":
            keep[components[0]] = True
        else:
            min_size = max(1, int(round(len(components[0]) * float(args.min_component_fraction))))
            for component in components:
                if len(component) >= min_size:
                    keep[component] = True
    info = {
        "component_count_after_overlap_removal": int(len(components)),
        "component_sizes_after_overlap_removal": [int(len(component)) for component in components[:20]],
        "keep_rule": args.keep_rv_components,
        "min_component_fraction": float(args.min_component_fraction),
        "rv_cells_after_component_cleanup": int(np.count_nonzero(keep)),
        "removed_small_component_cells": int(np.count_nonzero(initial_keep) - np.count_nonzero(keep)),
    }
    return keep, info


def line_search_valid_move(
    points: np.ndarray,
    cells: np.ndarray,
    displacement: np.ndarray,
    min_positive_volume: float,
    steps: int,
) -> Tuple[np.ndarray, float, Dict[str, int]]:
    base_volumes = signed_tet_volumes(points, cells)
    if np.any(base_volumes < 0.0):
        raise ValueError("RV cells must be positively oriented before smoothing/retraction.")
    for step in range(max(1, int(steps)) + 1):
        alpha = 1.0 / (2.0 ** step)
        candidate = points + displacement * alpha
        volumes = signed_tet_volumes(candidate, cells)
        if np.all(volumes > float(min_positive_volume)):
            return candidate, alpha, {
                "line_search_halvings": int(step),
                "nonpositive_candidate_tets": int(np.count_nonzero(volumes <= float(min_positive_volume))),
            }
    return points, 0.0, {
        "line_search_halvings": int(steps),
        "nonpositive_candidate_tets": int(np.count_nonzero(base_volumes <= float(min_positive_volume))),
    }


def retract_and_smooth_rv(
    rv_points: np.ndarray,
    rv_cells: np.ndarray,
    lv_surface: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    rv_points = np.asarray(rv_points, dtype=float)
    rv_cells = np.asarray(rv_cells, dtype=np.int64)
    median_edge = median_tet_edge_length(rv_points, rv_cells)
    retract_distance = (
        float(args.retract_distance_mm)
        if args.retract_distance_mm is not None
        else median_edge * float(args.retract_distance_edge_multiplier)
    )
    active_distance = (
        float(args.retract_active_distance_mm)
        if args.retract_active_distance_mm is not None
        else median_edge * float(args.retract_active_edge_multiplier)
    )
    max_move = median_edge * float(args.max_move_edge_multiplier)

    lv_points = np.asarray(lv_surface.points, dtype=float)
    distances, nearest = nearest_distances_and_indices(rv_points, lv_points)
    boundary_mask = boundary_vertex_mask(rv_cells, len(rv_points))
    seed = boundary_mask & (distances <= active_distance)
    neighbors = vertex_neighbors_from_cells(rv_cells, len(rv_points))
    weights = expand_vertex_weights(seed, neighbors, args.retract_rings)

    directions = rv_points - lv_points[nearest]
    norms = np.linalg.norm(directions, axis=1)
    valid_direction = norms > 1e-8
    directions[valid_direction] /= norms[valid_direction, None]
    directions[~valid_direction] = 0.0
    needed = np.maximum(0.0, retract_distance - distances)
    displacement = directions * (needed * weights)[:, None]
    displacement_norm = np.linalg.norm(displacement, axis=1)
    too_large = displacement_norm > max_move
    if np.any(too_large):
        displacement[too_large] *= (max_move / displacement_norm[too_large])[:, None]
    moved_points, retract_alpha, retract_info = line_search_valid_move(
        rv_points,
        rv_cells,
        displacement,
        args.min_positive_volume,
        args.quality_line_search_steps,
    )

    smooth_points = moved_points.copy()
    smooth_weights = expand_vertex_weights(seed, neighbors, args.smooth_rings)
    smooth_history: List[Dict[str, object]] = []
    for iteration in range(max(0, int(args.smooth_iterations))):
        lap = np.zeros_like(smooth_points)
        for vertex_id, linked in enumerate(neighbors):
            if linked:
                ids = np.fromiter(linked, dtype=np.int64)
                lap[vertex_id] = smooth_points[ids].mean(axis=0) - smooth_points[vertex_id]
        step = lap * (float(args.smooth_relaxation) * smooth_weights)[:, None]
        step_norm = np.linalg.norm(step, axis=1)
        too_large = step_norm > max_move
        if np.any(too_large):
            step[too_large] *= (max_move / step_norm[too_large])[:, None]
        smooth_points, alpha, info = line_search_valid_move(
            smooth_points,
            rv_cells,
            step,
            args.min_positive_volume,
            args.quality_line_search_steps,
        )
        smooth_history.append({
            "iteration": int(iteration + 1),
            "accepted_alpha": float(alpha),
            **info,
        })
        if alpha == 0.0:
            break

    total_displacement = smooth_points - rv_points
    final_distances, _ = nearest_distances_and_indices(smooth_points, lv_points)
    info = {
        "median_rv_tet_edge_length": float(median_edge),
        "retract_distance_mm": float(retract_distance),
        "retract_active_distance_mm": float(active_distance),
        "max_move_per_step_mm": float(max_move),
        "seed_boundary_vertices": int(np.count_nonzero(seed)),
        "weighted_retract_vertices": int(np.count_nonzero(weights > 0.0)),
        "weighted_smooth_vertices": int(np.count_nonzero(smooth_weights > 0.0)),
        "retraction": {
            "accepted_alpha": float(retract_alpha),
            **retract_info,
        },
        "smooth_iterations_requested": int(args.smooth_iterations),
        "smooth_history": smooth_history,
        "distance_to_lv_before": {
            "min": float(distances.min()),
            "p05": float(np.percentile(distances, 5)),
            "p50": float(np.percentile(distances, 50)),
            "p95": float(np.percentile(distances, 95)),
        },
        "distance_to_lv_after": {
            "min": float(final_distances.min()),
            "p05": float(np.percentile(final_distances, 5)),
            "p50": float(np.percentile(final_distances, 50)),
            "p95": float(np.percentile(final_distances, 95)),
        },
        "total_displacement_mm": {
            "max": float(np.linalg.norm(total_displacement, axis=1).max()),
            "p95": float(np.percentile(np.linalg.norm(total_displacement, axis=1), 95)),
            "mean_active": float(np.linalg.norm(total_displacement[weights > 0.0], axis=1).mean()) if np.any(weights > 0.0) else 0.0,
        },
    }
    return smooth_points, info


def load_tet_mesh(path: Path, expected_tag: int) -> Tuple[pv.UnstructuredGrid, np.ndarray, np.ndarray, np.ndarray]:
    grid = pv.read(path)
    cells, orientation = reorient_tetrahedra(np.asarray(grid.points, dtype=float), tetra_connectivity(grid))
    if "elemTag" in grid.cell_data:
        tags = np.asarray(grid.cell_data["elemTag"], dtype=np.int32)
        if len(tags) != len(cells):
            tags = np.full(len(cells), int(expected_tag), dtype=np.int32)
    else:
        tags = np.full(len(cells), int(expected_tag), dtype=np.int32)
    return grid, np.asarray(grid.points, dtype=float), cells, tags


def merge_lv_rv(
    lv_points: np.ndarray,
    lv_cells: np.ndarray,
    rv_points: np.ndarray,
    rv_cells: np.ndarray,
    lv_tag: int,
    rv_tag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.vstack([lv_points, rv_points])
    cells = np.vstack([lv_cells, rv_cells + len(lv_points)])
    tags = np.concatenate([
        np.full(len(lv_cells), int(lv_tag), dtype=np.int32),
        np.full(len(rv_cells), int(rv_tag), dtype=np.int32),
    ])
    return points, cells, tags


def process_frame(args: argparse.Namespace, frame: str) -> Dict[str, object]:
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    input_dir = args.input_dir or subject_dir / "volmesh_LVRV"
    output_dir = args.output_dir or subject_dir / "volmesh_LVRV_merged_clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    lv_path = input_dir / f"{subject_id}_{frame}_openbase_LV_volmesh.vtk"
    rv_path = input_dir / f"{subject_id}_{frame}_openbase_RV_volmesh.vtk"
    _, lv_points, lv_cells, _ = load_tet_mesh(lv_path, args.lv_tag)
    _, rv_points_raw, rv_cells_raw, _ = load_tet_mesh(rv_path, args.rv_tag)

    lv_surface = make_grid(lv_points, lv_cells).extract_surface().triangulate().clean(tolerance=1e-7)
    overlap_keep, overlap_info = remove_overlapping_rv_cells(lv_surface, rv_points_raw, rv_cells_raw, args)
    component_keep, component_info = keep_main_rv_component(rv_cells_raw, overlap_keep, args)
    rv_cells_kept_raw = rv_cells_raw[component_keep]
    rv_points, rv_cells, used_rv_points = compact_mesh(rv_points_raw, rv_cells_kept_raw)
    rv_cells, rv_orientation = reorient_tetrahedra(rv_points, rv_cells)
    rv_points_clean, rv_edit_info = retract_and_smooth_rv(rv_points, rv_cells, lv_surface, args)
    rv_cells, rv_final_orientation = reorient_tetrahedra(rv_points_clean, rv_cells)

    merged_points, merged_cells, merged_tags = merge_lv_rv(
        lv_points,
        lv_cells,
        rv_points_clean,
        rv_cells,
        args.lv_tag,
        args.rv_tag,
    )
    merged_cells, merged_orientation = reorient_tetrahedra(merged_points, merged_cells)

    output_mesh = output_dir / f"{subject_id}_{frame}_openbase_LVRV_merged_clean_volmesh.vtk"
    write_custom_vtk(output_mesh, merged_points, merged_cells, merged_tags)

    debug_paths: Dict[str, str] = {}
    if args.save_debug:
        rv_removed_path = output_dir / f"{subject_id}_{frame}_RV_after_overlap_component_cleanup_debug.vtk"
        rv_clean_path = output_dir / f"{subject_id}_{frame}_RV_after_retract_smooth_debug.vtk"
        write_custom_vtk(rv_removed_path, rv_points, rv_cells, np.full(len(rv_cells), args.rv_tag, dtype=np.int32))
        write_custom_vtk(rv_clean_path, rv_points_clean, rv_cells, np.full(len(rv_cells), args.rv_tag, dtype=np.int32))
        debug_paths = {
            "rv_after_overlap_component_cleanup": str(rv_removed_path.resolve()),
            "rv_after_retract_smooth": str(rv_clean_path.resolve()),
        }

    lv_volumes = signed_tet_volumes(lv_points, lv_cells)
    rv_volumes = signed_tet_volumes(rv_points_clean, rv_cells)
    merged_volumes = signed_tet_volumes(merged_points, merged_cells)
    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "inputs": {
            "lv": str(lv_path.resolve()),
            "rv": str(rv_path.resolve()),
        },
        "output_mesh": str(output_mesh.resolve()),
        "debug_outputs": debug_paths,
        "method": {
            "lv": "LV points and tetrahedra are kept intact.",
            "rv_overlap_removal": "RV tetrahedra whose centroid or configured vertex fraction lies inside the LV closed volume are removed, then removal is expanded by face-neighbor rings.",
            "rv_component_cleanup": "After overlap removal, small disconnected RV parts are removed; default keeps only the largest RV component.",
            "rv_retraction": "Remaining RV vertices near LV are moved away from the LV surface by an automatic median-edge-based clearance target.",
            "rv_smoothing": "Weighted local RV volume Laplacian smoothing is applied with line search so tetrahedra stay positive.",
            "merge": "LV and edited RV are appended into one VTK without moving LV points.",
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "overlap": overlap_info,
        "component_cleanup": component_info,
        "rv_orientation_before_edit": rv_orientation,
        "rv_retract_smooth": rv_edit_info,
        "rv_orientation_after_edit": rv_final_orientation,
        "merged_orientation": merged_orientation,
        "counts": {
            "lv_points": int(len(lv_points)),
            "lv_cells": int(len(lv_cells)),
            "rv_points_raw": int(len(rv_points_raw)),
            "rv_cells_raw": int(len(rv_cells_raw)),
            "rv_points_after_cleanup": int(len(rv_points)),
            "rv_cells_after_cleanup": int(len(rv_cells)),
            "merged_points": int(len(merged_points)),
            "merged_cells": int(len(merged_cells)),
            "merged_elemTag": {
                str(k): int(v)
                for k, v in sorted(Counter(map(int, merged_tags)).items())
            },
        },
        "quality": {
            "lv_nonpositive_signed_tets": int(np.count_nonzero(lv_volumes <= 0.0)),
            "rv_nonpositive_signed_tets": int(np.count_nonzero(rv_volumes <= 0.0)),
            "merged_nonpositive_signed_tets": int(np.count_nonzero(merged_volumes <= 0.0)),
            "lv_min_abs_volume": float(np.abs(lv_volumes).min()) if len(lv_volumes) else 0.0,
            "rv_min_abs_volume": float(np.abs(rv_volumes).min()) if len(rv_volumes) else 0.0,
            "merged_min_abs_volume": float(np.abs(merged_volumes).min()) if len(merged_volumes) else 0.0,
        },
    }
    summary_path = output_dir / f"{subject_id}_{frame}_openbase_LVRV_merged_clean_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return {
        "frame": frame,
        "output_mesh": str(output_mesh),
        "summary": str(summary_path),
        "lv_cells": int(len(lv_cells)),
        "rv_cells_raw": int(len(rv_cells_raw)),
        "rv_cells_after_cleanup": int(len(rv_cells)),
        "merged_cells": int(len(merged_cells)),
        "overlap_removed_cells": int(overlap_info["overlap_cells_after_ring_expansion"]),
        "small_component_removed_cells": int(component_info["removed_small_component_cells"]),
        "merged_nonpositive_signed_tets": int(np.count_nonzero(merged_volumes <= 0.0)),
    }


def main() -> None:
    args = parse_args()
    frames = parse_csv_values(args.frames) or [args.frame]
    results = [process_frame(args, frame) for frame in frames]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
