#!/usr/bin/env python3
"""Fuse the RV septal target into LV by locally rebuilding tetrahedra.

This is a more geometric follow-up to retag_rv_septum_to_lv_v2.py.

The script first detects the RV septal/contact target and writes it as tag 2.
Then it selects the closest LV interface patch, removes the local LV+target
tetrahedra, rebuilds that local volume with a local Delaunay fill, and merges
the rebuilt region back as LV tag 1.

Input labels:
  elemTag = 1: LV myocardium
  elemTag = 0: RV myocardium

Intermediate target labels:
  elemTag = 1: original LV myocardium
  elemTag = 2: RV septal/contact target plus bridge cells
  elemTag = 0: protected RV free wall

Final labels:
  elemTag = 1: LV myocardium + rebuilt fused septum
  elemTag = 0: protected RV free wall
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import retag_rv_septum_to_lv as v1
import retag_rv_septum_to_lv_v2 as v2
from custom_vtk import write_custom_vtk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-mesh",
        type=Path,
        default=Path("data/subject_1/volmesh_open_stable/subject_1_ED_biventricular_open_stable_custom_tag_volmesh.vtk"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lv-tag", type=int, default=1)
    parser.add_argument("--rv-tag", type=int, default=0)
    parser.add_argument("--target-tag", type=int, default=2)
    parser.add_argument("--cutoff-mm", type=float, default=None)
    parser.add_argument("--cutoff-multiplier", type=float, default=3.0)
    parser.add_argument("--min-candidate-fraction", type=float, default=0.005)
    parser.add_argument("--max-cutoff-multiplier", type=float, default=4.0)
    parser.add_argument("--integration-tolerance-mm", type=float, default=None)
    parser.add_argument("--integration-tolerance-multiplier", type=float, default=2.5)
    parser.add_argument("--bridge-rings", type=int, default=2)
    parser.add_argument("--bridge-lv-distance-multiplier", type=float, default=1.0)
    parser.add_argument("--lv-patch-rings", type=int, default=2)
    parser.add_argument("--min-lv-patch-faces", type=int, default=250)
    parser.add_argument("--max-lv-patch-distance-multiplier", type=float, default=1.75)
    parser.add_argument("--delaunay-alpha-multiplier", type=float, default=2.5)
    parser.add_argument("--delaunay-tolerance", type=float, default=0.001)
    parser.add_argument("--delaunay-offset", type=float, default=2.5)
    parser.add_argument("--surface-clean-tolerance", type=float, default=1e-7)
    parser.add_argument("--volume-eps", type=float, default=1e-10)
    return parser.parse_args()


def infer_output_paths(input_mesh: Path, output_dir: Optional[Path]) -> Dict[str, Path]:
    if output_dir is None:
        output_dir = input_mesh.parent.parent / "vol_mesh_septum_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_stem = input_mesh.stem.replace("_volmesh", "")
    return {
        "target": output_dir / f"{base_stem}_septum_target_tag2_volmesh.vtk",
        "lv_patch": output_dir / f"{base_stem}_lv_rebuild_patch_debug.vtk",
        "local_shell": output_dir / f"{base_stem}_local_rebuild_shell.vtk",
        "local_rebuilt": output_dir / f"{base_stem}_local_rebuilt_lv_region.vtk",
        "final": output_dir / f"{base_stem}_septum_fused_rebuilt_lvtag_volmesh.vtk",
        "debug": output_dir / f"{base_stem}_septum_rebuild_debug.vtk",
        "summary": output_dir / f"{base_stem}_septum_rebuild_summary.json",
    }


def make_grid(points: np.ndarray, cells: np.ndarray, tags: Optional[np.ndarray] = None) -> pv.UnstructuredGrid:
    cell_sizes = np.full((len(cells), 1), 4, dtype=np.int64)
    cells_flat = np.hstack([cell_sizes, cells.astype(np.int64)]).ravel()
    celltypes = np.full(len(cells), v1.TETRA_CELL_TYPE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells_flat, celltypes, np.asarray(points, dtype=float))
    if tags is not None:
        grid.cell_data["elemTag"] = np.asarray(tags, dtype=np.int32)
    return grid


def boundary_faces_and_owners(cells: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    face_to_owner: Dict[Tuple[int, int, int], int] = {}
    face_counts: Dict[Tuple[int, int, int], int] = {}
    for cell_id in np.flatnonzero(mask):
        for face in v1.tet_faces(cells[cell_id]):
            face_counts[face] = face_counts.get(face, 0) + 1
            face_to_owner[face] = int(cell_id)
    faces: List[Tuple[int, int, int]] = []
    owners: List[int] = []
    for face, count in face_counts.items():
        if count == 1:
            faces.append(face)
            owners.append(face_to_owner[face])
    if not faces:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64), np.asarray(owners, dtype=np.int64)


def expand_cell_mask(seed_mask: np.ndarray, allowed_mask: np.ndarray, adjacency: Sequence[Sequence[int]], rings: int) -> np.ndarray:
    expanded = np.asarray(seed_mask, dtype=bool).copy()
    frontier = set(int(v) for v in np.flatnonzero(seed_mask))
    for _ in range(max(0, int(rings))):
        next_frontier = set()
        for cell_id in frontier:
            for neighbor in adjacency[cell_id]:
                if allowed_mask[neighbor] and not expanded[neighbor]:
                    expanded[neighbor] = True
                    next_frontier.add(int(neighbor))
        if not next_frontier:
            break
        frontier = next_frontier
    return expanded


def detect_target_region(
    points: np.ndarray,
    cells: np.ndarray,
    original_tags: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    lv_tag = int(args.lv_tag)
    rv_tag = int(args.rv_tag)
    lv_mask = original_tags == lv_tag
    rv_mask = original_tags == rv_tag
    centroids = v1.tetra_centroids(points, cells)
    median_edge = v1.median_tet_edge_length(points, cells)

    lv_boundary_centers = v1.boundary_face_centers_for_label(points, cells, original_tags, lv_tag)
    rv_indices = np.flatnonzero(rv_mask)
    rv_to_lv_distance = v1.nearest_distances(centroids[rv_indices], lv_boundary_centers)
    rv_distance_to_lv = np.full(len(cells), np.inf, dtype=float)
    rv_distance_to_lv[rv_indices] = rv_to_lv_distance
    cutoff, cutoff_info = v1.choose_cutoff(
        rv_to_lv_distance,
        median_edge,
        args.cutoff_mm,
        args.cutoff_multiplier,
        args.min_candidate_fraction,
        args.max_cutoff_multiplier,
    )

    septum_candidate = np.zeros(len(cells), dtype=bool)
    septum_candidate[rv_indices[rv_to_lv_distance <= cutoff]] = True
    remaining_rv = rv_mask & ~septum_candidate
    remaining_components = v1.connected_components(remaining_rv, adjacency)
    if not remaining_components:
        raise RuntimeError("No RV main body remains after the septum cutoff.")

    protected_rv_initial = np.zeros(len(cells), dtype=bool)
    protected_rv_initial[remaining_components[0]] = True
    initial_target = rv_mask & ~protected_rv_initial

    if args.integration_tolerance_mm is None:
        integration_tolerance = float(args.integration_tolerance_multiplier) * float(median_edge)
    else:
        integration_tolerance = float(args.integration_tolerance_mm)
    max_bridge_lv_distance = float(cutoff) + float(args.bridge_lv_distance_multiplier) * integration_tolerance
    expanded_target, bridge_info = v2.expand_target_bridge(
        initial_target,
        protected_rv_initial,
        rv_mask,
        adjacency,
        rv_distance_to_lv,
        max_bridge_lv_distance=max_bridge_lv_distance,
        bridge_rings=args.bridge_rings,
    )

    final_remaining_rv = rv_mask & ~expanded_target
    final_components = v1.connected_components(final_remaining_rv, adjacency)
    if not final_components:
        raise RuntimeError("No RV main body remains after bridge expansion.")

    protected_rv = np.zeros(len(cells), dtype=bool)
    protected_rv[final_components[0]] = True
    target_mask = rv_mask & ~protected_rv
    bridge_mask = target_mask & ~initial_target

    info = {
        "median_tet_edge_length": float(median_edge),
        "cutoff": cutoff_info,
        "cutoff_mm": float(cutoff),
        "integration_tolerance_mm": float(integration_tolerance),
        "bridge": bridge_info,
        "initial_septum_candidate_cells": int(np.count_nonzero(septum_candidate)),
        "initial_target_cells": int(np.count_nonzero(initial_target)),
        "target_cells_after_bridge": int(np.count_nonzero(target_mask)),
        "bridge_cells": int(np.count_nonzero(bridge_mask)),
        "protected_rv_cells": int(np.count_nonzero(protected_rv)),
        "initial_remaining_rv_component_sizes": [int(len(c)) for c in remaining_components[:20]],
        "final_remaining_rv_component_sizes": [int(len(c)) for c in final_components[:20]],
    }
    return target_mask, bridge_mask, protected_rv, info


def select_lv_rebuild_patch(
    points: np.ndarray,
    cells: np.ndarray,
    lv_mask: np.ndarray,
    target_mask: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    integration_tolerance: float,
    min_lv_patch_faces: int,
    max_distance_multiplier: float,
    rings: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    lv_faces, lv_owners = boundary_faces_and_owners(cells, lv_mask)
    target_faces, _ = boundary_faces_and_owners(cells, target_mask)
    if len(lv_faces) == 0 or len(target_faces) == 0:
        raise RuntimeError("Cannot select LV rebuild patch because LV or target boundary faces are empty.")

    lv_centers = points[lv_faces].mean(axis=1)
    target_centers = points[target_faces].mean(axis=1)
    nearest_lv_ids, distances = v2.nearest_indices_and_distances(target_centers, lv_centers)

    max_distance = max(float(integration_tolerance), float(integration_tolerance) * float(max_distance_multiplier))
    close = distances <= max_distance
    if int(np.count_nonzero(close)) < int(min_lv_patch_faces):
        order = np.argsort(distances)
        keep_count = min(len(order), max(int(min_lv_patch_faces), int(np.count_nonzero(close))))
        close_ids = order[:keep_count]
    else:
        close_ids = np.flatnonzero(close)

    seed_lv_cells = np.zeros(len(cells), dtype=bool)
    seed_lv_cells[lv_owners[nearest_lv_ids[close_ids]]] = True
    lv_patch = expand_cell_mask(seed_lv_cells, lv_mask, adjacency, rings)

    return lv_patch, {
        "lv_boundary_faces": int(len(lv_faces)),
        "target_boundary_faces": int(len(target_faces)),
        "selected_target_faces_for_nearest_lv": int(len(close_ids)),
        "seed_lv_patch_cells": int(np.count_nonzero(seed_lv_cells)),
        "lv_patch_cells_after_ring_expansion": int(np.count_nonzero(lv_patch)),
        "min_distance_to_lv_patch": float(np.min(distances)) if len(distances) else None,
        "median_distance_to_lv_patch": float(np.median(distances)) if len(distances) else None,
        "max_selected_distance_to_lv_patch": float(np.max(distances[close_ids])) if len(close_ids) else None,
        "max_allowed_distance": float(max_distance),
        "lv_patch_rings": int(rings),
    }


def local_reindex(points: np.ndarray, cells: np.ndarray, selected_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_global_points = np.unique(cells[selected_cells].ravel())
    global_to_local = -np.ones(len(points), dtype=np.int64)
    global_to_local[local_global_points] = np.arange(len(local_global_points), dtype=np.int64)
    local_points = points[local_global_points].copy()
    local_cells = global_to_local[cells[selected_cells]]
    return local_points, local_cells, local_global_points



def reorient_and_filter(points: np.ndarray, cells: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    duplicate_mask = np.asarray([len(set(map(int, cell))) < 4 for cell in cells], dtype=bool)
    volumes = v2.tet_volumes(points, cells)
    negative_mask = volumes < -float(eps)
    out_cells = cells.copy()
    if np.any(negative_mask):
        neg = np.flatnonzero(negative_mask)
        first = out_cells[neg, 0].copy()
        out_cells[neg, 0] = out_cells[neg, 1]
        out_cells[neg, 1] = first
        volumes = v2.tet_volumes(points, out_cells)
    small_mask = np.abs(volumes) <= float(eps)
    keep = ~(duplicate_mask | small_mask)
    return out_cells[keep], keep, {
        "input_cells": int(len(cells)),
        "duplicate_vertex_cells": int(np.count_nonzero(duplicate_mask)),
        "negative_orientation_cells_reoriented": int(np.count_nonzero(negative_mask)),
        "small_volume_cells": int(np.count_nonzero(small_mask)),
        "kept_cells": int(np.count_nonzero(keep)),
        "dropped_cells": int(np.count_nonzero(~keep)),
    }



def merge_with_coordinate_dedup(
    retained_points: np.ndarray,
    retained_cells: np.ndarray,
    retained_tags: np.ndarray,
    local_points: np.ndarray,
    local_cells: np.ndarray,
    local_tag: int,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    all_points = np.vstack([retained_points, local_points])
    local_cells_offset = local_cells + len(retained_points)
    all_cells = np.vstack([retained_cells, local_cells_offset])
    all_tags = np.concatenate([retained_tags, np.full(len(local_cells), int(local_tag), dtype=np.int32)])

    rounded = np.round(all_points / float(tolerance)).astype(np.int64)
    mapping: Dict[Tuple[int, int, int], int] = {}
    old_to_new = np.empty(len(all_points), dtype=np.int64)
    unique_points: List[np.ndarray] = []
    duplicates = 0
    for point_id, key_values in enumerate(rounded):
        key = tuple(int(v) for v in key_values)
        existing = mapping.get(key)
        if existing is None:
            existing = len(unique_points)
            mapping[key] = existing
            unique_points.append(all_points[point_id])
        else:
            duplicates += 1
        old_to_new[point_id] = existing

    merged_points = np.asarray(unique_points, dtype=float)
    merged_cells = old_to_new[all_cells]

    used_points = np.unique(merged_cells.ravel())
    compact_map = -np.ones(len(merged_points), dtype=np.int64)
    compact_map[used_points] = np.arange(len(used_points), dtype=np.int64)
    compact_cells = compact_map[merged_cells]
    compact_points = merged_points[used_points]

    return compact_points, compact_cells, all_tags, {
        "input_points_before_coordinate_dedup": int(len(all_points)),
        "output_points_after_coordinate_dedup": int(len(unique_points)),
        "output_points_after_compaction": int(len(compact_points)),
        "unused_points_removed_after_merge": int(len(unique_points) - len(compact_points)),
        "deduplicated_points": int(duplicates),
        "retained_cells": int(len(retained_cells)),
        "local_rebuilt_cells": int(len(local_cells)),
        "merged_cells_before_filter": int(len(compact_cells)),
    }


def delaunay_rebuild_local_region(
    local_points: np.ndarray,
    median_edge: float,
    args: argparse.Namespace,
    lv_tag: int,
) -> Tuple[np.ndarray, np.ndarray, pv.PolyData, Dict[str, object]]:
    alpha = float(args.delaunay_alpha_multiplier) * float(median_edge)
    cloud = pv.PolyData(np.asarray(local_points, dtype=float))
    rebuilt = cloud.delaunay_3d(
        alpha=alpha,
        tol=float(args.delaunay_tolerance),
        offset=float(args.delaunay_offset),
    )
    tet_ids = np.flatnonzero(np.asarray(rebuilt.celltypes) == v1.TETRA_CELL_TYPE)
    if len(tet_ids) == 0:
        raise RuntimeError("Local Delaunay rebuild produced no tetrahedra.")
    tet_grid = rebuilt.extract_cells(tet_ids)
    rebuilt_cells = v1.tetra_connectivity(tet_grid)
    rebuilt_points = np.asarray(tet_grid.points, dtype=float)
    rebuilt_cells, keep, filter_info = reorient_and_filter(
        rebuilt_points,
        rebuilt_cells,
        eps=float(args.volume_eps),
    )
    rebuilt_grid = make_grid(rebuilt_points, rebuilt_cells, np.full(len(rebuilt_cells), lv_tag, dtype=np.int32))
    rebuilt_surface = rebuilt_grid.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    return rebuilt_points, rebuilt_cells, rebuilt_surface, {
        "method": "pyvista_delaunay_3d_point_cloud",
        "alpha": float(alpha),
        "alpha_multiplier": float(args.delaunay_alpha_multiplier),
        "tolerance": float(args.delaunay_tolerance),
        "offset": float(args.delaunay_offset),
        "input_points": int(len(local_points)),
        "raw_cells": int(rebuilt.n_cells),
        "raw_tetrahedra": int(len(tet_ids)),
        "kept_tetrahedra_after_filter": int(len(rebuilt_cells)),
        "filter": filter_info,
    }


def write_debug_original_grid(
    path: Path,
    mesh: pv.UnstructuredGrid,
    original_tags: np.ndarray,
    target_mask: np.ndarray,
    bridge_mask: np.ndarray,
    protected_rv: np.ndarray,
    lv_patch: np.ndarray,
    rebuild_mask: np.ndarray,
) -> None:
    debug = mesh.copy(deep=True)
    debug.cell_data["original_elemTag"] = original_tags.astype(np.int32)
    debug.cell_data["septumTargetTag2"] = target_mask.astype(np.int32)
    debug.cell_data["bridgeCells"] = bridge_mask.astype(np.int32)
    debug.cell_data["protectedRVMainBody"] = protected_rv.astype(np.int32)
    debug.cell_data["lvRebuildPatch"] = lv_patch.astype(np.int32)
    debug.cell_data["localRebuildMask"] = rebuild_mask.astype(np.int32)
    debug.save(path)


def main() -> None:
    args = parse_args()
    mesh = pv.read(args.input_mesh)
    points = np.asarray(mesh.points, dtype=float)
    cells = v1.tetra_connectivity(mesh)
    if "elemTag" not in mesh.cell_data:
        raise KeyError("Input mesh must contain cell_data['elemTag'].")
    original_tags = np.asarray(mesh.cell_data["elemTag"], dtype=np.int32)
    if len(original_tags) != len(cells):
        raise ValueError("elemTag length does not match tetra cell count.")

    lv_tag = int(args.lv_tag)
    rv_tag = int(args.rv_tag)
    target_tag = int(args.target_tag)
    if target_tag in {lv_tag, rv_tag}:
        raise ValueError("--target-tag must differ from LV and RV tags.")

    lv_mask = original_tags == lv_tag
    rv_mask = original_tags == rv_tag
    if not np.any(lv_mask) or not np.any(rv_mask):
        raise ValueError("Input mesh must contain both LV and RV tags.")

    adjacency = v1.build_adjacency(cells)
    target_mask, bridge_mask, protected_rv, target_info = detect_target_region(points, cells, original_tags, adjacency, args)
    integration_tolerance = float(target_info["integration_tolerance_mm"])

    intermediate_tags = original_tags.copy()
    intermediate_tags[target_mask] = target_tag
    intermediate_tags[protected_rv] = rv_tag

    lv_patch, lv_patch_info = select_lv_rebuild_patch(
        points,
        cells,
        lv_mask,
        target_mask,
        adjacency,
        integration_tolerance=integration_tolerance,
        min_lv_patch_faces=args.min_lv_patch_faces,
        max_distance_multiplier=args.max_lv_patch_distance_multiplier,
        rings=args.lv_patch_rings,
    )
    rebuild_mask = target_mask | lv_patch
    retained_mask = ~rebuild_mask
    if not np.any(rebuild_mask):
        raise RuntimeError("No cells selected for local rebuild.")

    paths = infer_output_paths(args.input_mesh, args.output_dir)
    write_custom_vtk(paths["target"], points, cells, intermediate_tags, fiber_data=mesh.cell_data.get("fiber", None))

    patch_debug_tags = np.full(len(cells), -1, dtype=np.int32)
    patch_debug_tags[retained_mask] = original_tags[retained_mask]
    patch_debug_tags[target_mask] = target_tag
    patch_debug_tags[lv_patch] = lv_tag
    write_custom_vtk(paths["lv_patch"], points, cells, patch_debug_tags)
    write_debug_original_grid(paths["debug"], mesh, original_tags, target_mask, bridge_mask, protected_rv, lv_patch, rebuild_mask)

    selected_cell_ids = np.flatnonzero(rebuild_mask)
    local_points, local_cells, local_global_points = local_reindex(points, cells, selected_cell_ids)
    rebuilt_points, rebuilt_cells, rebuilt_surface, rebuild_info = delaunay_rebuild_local_region(
        local_points,
        median_edge=float(target_info["median_tet_edge_length"]),
        args=args,
        lv_tag=lv_tag,
    )
    rebuilt_surface.save(paths["local_shell"])
    make_grid(
        rebuilt_points,
        rebuilt_cells,
        np.full(len(rebuilt_cells), lv_tag, dtype=np.int32),
    ).save(paths["local_rebuilt"])

    fiber_data = mesh.cell_data.get("fiber", None)
    retained_points = points
    retained_cells = cells[retained_mask]
    retained_tags = original_tags[retained_mask].copy()
    retained_tags[retained_tags == target_tag] = lv_tag
    retained_fiber = fiber_data[retained_mask] if fiber_data is not None else None

    merged_points, merged_cells, merged_tags, merge_info = merge_with_coordinate_dedup(
        retained_points,
        retained_cells,
        retained_tags,
        rebuilt_points,
        rebuilt_cells,
        local_tag=lv_tag,
        tolerance=max(args.surface_clean_tolerance, 1e-6),
    )
    merged_cells, final_keep, final_filter_info = reorient_and_filter(
        merged_points,
        merged_cells,
        eps=float(args.volume_eps),
    )
    final_tags = merged_tags[final_keep]
    final_fiber = None
    if retained_fiber is not None:
        final_fiber = np.vstack([
            retained_fiber,
            np.zeros((len(rebuilt_cells), retained_fiber.shape[1]), dtype=retained_fiber.dtype),
        ])[final_keep]

    write_custom_vtk(paths["final"], merged_points, merged_cells, final_tags, fiber_data=final_fiber)

    final_grid = make_grid(merged_points, merged_cells, final_tags)
    final_adjacency = v1.build_adjacency(merged_cells)
    lv_components = [int(len(c)) for c in v1.connected_components(final_tags == lv_tag, final_adjacency)]
    rv_components = [int(len(c)) for c in v1.connected_components(final_tags == rv_tag, final_adjacency)]

    original_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, original_tags)).items())}
    intermediate_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, intermediate_tags)).items())}
    final_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, final_tags)).items())}
    final_volumes = v2.tet_volumes(merged_points, merged_cells)

    summary = {
        "script": str(Path(__file__).resolve()),
        "input_mesh": str(args.input_mesh.resolve()),
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        "method": {
            "target_detection": "Same stable target rule as v2: RV cells close to LV boundary are candidates; largest remaining RV component is protected.",
            "lv_patch_selection": "Closest LV boundary faces to the tag-2 target define LV interface cells, expanded by LV adjacency rings.",
            "local_rebuild": "Target cells and closest LV interface cells are removed from the original mesh. The selected patch point cloud is tetrahedralized with PyVista Delaunay so the gap is filled by newly generated LV-tagged tetrahedra.",
            "final_fusion": "The rebuilt local volume is merged back with the retained original mesh and marked as LV tag 1.",
        },
        "labels": {
            "rv_tag": rv_tag,
            "lv_tag": lv_tag,
            "target_tag_intermediate": target_tag,
            "rebuilt_region_final_tag": lv_tag,
        },
        "counts": {
            "original_elemTag": original_counts,
            "intermediate_elemTag": intermediate_counts,
            "final_elemTag": final_counts,
            "target_cells": int(np.count_nonzero(target_mask)),
            "bridge_cells": int(np.count_nonzero(bridge_mask)),
            "lv_patch_cells": int(np.count_nonzero(lv_patch)),
            "local_rebuild_input_cells": int(np.count_nonzero(rebuild_mask)),
            "retained_original_cells": int(np.count_nonzero(retained_mask)),
            "final_cells": int(len(merged_cells)),
            "final_points": int(len(merged_points)),
        },
        "target": target_info,
        "lv_patch": lv_patch_info,
        "local_rebuild": rebuild_info,
        "merge": merge_info,
        "final_filter": final_filter_info,
        "final_quality": {
            "negative_or_zero_signed_tets": int(np.count_nonzero(final_volumes <= 0)),
            "min_abs_tet_volume": float(np.abs(final_volumes).min()) if len(final_volumes) else 0.0,
            "median_abs_tet_volume": float(np.median(np.abs(final_volumes))) if len(final_volumes) else 0.0,
        },
        "connectivity": {
            "lv_components_by_face": lv_components[:20],
            "rv_components_by_face": rv_components[:20],
            "lv_component_count": int(len(lv_components)),
            "rv_component_count": int(len(rv_components)),
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "input_mesh": str(args.input_mesh),
        "target_tag2_mesh": str(paths["target"]),
        "local_shell": str(paths["local_shell"]),
        "local_rebuilt": str(paths["local_rebuilt"]),
        "final_mesh": str(paths["final"]),
        "summary": str(paths["summary"]),
        "original_elemTag": original_counts,
        "intermediate_elemTag": intermediate_counts,
        "final_elemTag": final_counts,
        "target_cells": int(np.count_nonzero(target_mask)),
        "bridge_cells": int(np.count_nonzero(bridge_mask)),
        "lv_patch_cells": int(np.count_nonzero(lv_patch)),
        "rebuilt_cells": int(len(rebuilt_cells)),
        "final_cells": int(len(merged_cells)),
        "final_points": int(len(merged_points)),
        "final_negative_or_zero_signed_tets": int(np.count_nonzero(final_volumes <= 0)),
        "lv_components_by_face": lv_components[:5],
        "rv_components_by_face": rv_components[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
