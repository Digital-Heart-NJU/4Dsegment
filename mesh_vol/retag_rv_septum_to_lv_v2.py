#!/usr/bin/env python3
"""Build a visible septal target region, then fuse it into LV myocardium.

This is a post-processor for the open biventricular volume mesh.

Input labels:
  elemTag = 1: LV myocardium
  elemTag = 0: RV myocardium

Intermediate output:
  elemTag = 1: original LV myocardium
  elemTag = 2: detected RV septal/contact target plus bridge cells
  elemTag = 0: protected RV free wall

Final fused output:
  elemTag = 1: LV myocardium + fused septal target/bridge
  elemTag = 0: protected RV free wall

The fusion step can optionally remap target-side interface points to nearby LV
boundary points. This closes small geometric gaps caused by separate LV/RV
tetrahedralization while leaving the original LV points fixed.
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
    parser.add_argument("--drop-degenerate-cells", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-interface-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-debug-vtk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--volume-eps", type=float, default=1e-10)
    return parser.parse_args()


def tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    a = pts[cells[:, 0]]
    b = pts[cells[:, 1]]
    c = pts[cells[:, 2]]
    d = pts[cells[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def boundary_faces_for_mask(cells: np.ndarray, mask: np.ndarray) -> np.ndarray:
    face_counts: Dict[Tuple[int, int, int], int] = {}
    for cell_id in np.flatnonzero(mask):
        for face in v1.tet_faces(cells[cell_id]):
            face_counts[face] = face_counts.get(face, 0) + 1
    faces = [face for face, count in face_counts.items() if count == 1]
    if not faces:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def nearest_indices_and_distances(query: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(reference).query(query, k=1)
        return indices.astype(np.int64), distances.astype(float)
    except Exception:
        indices = np.empty(len(query), dtype=np.int64)
        distances = np.empty(len(query), dtype=float)
        for start in range(0, len(query), 2048):
            chunk = query[start:start + 2048]
            dist2 = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            local = np.argmin(dist2, axis=1)
            indices[start:start + len(chunk)] = local
            distances[start:start + len(chunk)] = np.sqrt(dist2[np.arange(len(chunk)), local])
        return indices, distances


def infer_output_paths(input_mesh: Path, output_dir: Optional[Path]) -> Tuple[Path, Path, Path, Path]:
    if output_dir is None:
        output_dir = input_mesh.parent.parent / "vol_mesh_septum"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_mesh.stem
    base_stem = stem.replace("_volmesh", "")
    intermediate = output_dir / f"{base_stem}_septum_target_tag2_volmesh.vtk"
    final = output_dir / f"{base_stem}_septum_fused_lvtag_volmesh.vtk"
    debug = output_dir / f"{base_stem}_septum_integration_debug.vtk"
    summary = output_dir / f"{base_stem}_septum_fusion_summary.json"
    return intermediate, final, debug, summary


def expand_target_bridge(
    target_mask: np.ndarray,
    protected_rv: np.ndarray,
    rv_mask: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    rv_distance_to_lv: np.ndarray,
    max_bridge_lv_distance: float,
    bridge_rings: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    expanded = np.asarray(target_mask, dtype=bool).copy()
    ring_counts: List[int] = []
    frontier = set(int(v) for v in np.flatnonzero(expanded))

    for _ in range(max(0, int(bridge_rings))):
        next_ring = set()
        for cell_id in frontier:
            for neighbor in adjacency[cell_id]:
                if (
                    protected_rv[neighbor]
                    and rv_mask[neighbor]
                    and not expanded[neighbor]
                    and rv_distance_to_lv[neighbor] <= max_bridge_lv_distance
                ):
                    next_ring.add(int(neighbor))
        if not next_ring:
            break
        ids = np.asarray(sorted(next_ring), dtype=np.int64)
        expanded[ids] = True
        ring_counts.append(int(len(ids)))
        frontier = next_ring

    return expanded, {
        "requested_bridge_rings": int(bridge_rings),
        "completed_bridge_rings": int(len(ring_counts)),
        "bridge_cell_counts_by_ring": ring_counts,
        "bridge_added_cells": int(np.count_nonzero(expanded & ~target_mask)),
        "max_bridge_lv_distance": float(max_bridge_lv_distance),
    }


def compact_points_and_cells(
    points: np.ndarray,
    cells: np.ndarray,
    tags: np.ndarray,
    keep_mask: np.ndarray,
    fiber_data: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    kept_cells = cells[keep_mask]
    kept_tags = tags[keep_mask]
    used = np.unique(kept_cells.ravel())
    new_ids = -np.ones(len(points), dtype=np.int64)
    new_ids[used] = np.arange(len(used), dtype=np.int64)
    compact_cells = new_ids[kept_cells]
    compact_points = points[used]
    compact_fiber = fiber_data[keep_mask] if fiber_data is not None else None
    return compact_points, compact_cells, kept_tags, compact_fiber, {
        "input_points": int(len(points)),
        "output_points": int(len(compact_points)),
        "input_cells": int(len(cells)),
        "output_cells": int(len(compact_cells)),
        "dropped_cells": int(np.count_nonzero(~keep_mask)),
        "dropped_unused_points": int(len(points) - len(compact_points)),
    }


def fuse_target_points_to_lv(
    points: np.ndarray,
    cells: np.ndarray,
    intermediate_tags: np.ndarray,
    lv_tag: int,
    target_tag: int,
    tolerance: float,
    merge_points: bool,
    drop_degenerate: bool,
    volume_eps: float,
    fiber_data: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, object], np.ndarray, np.ndarray]:
    lv_mask = intermediate_tags == lv_tag
    target_mask = intermediate_tags == target_tag
    lv_faces = boundary_faces_for_mask(cells, lv_mask)
    target_faces = boundary_faces_for_mask(cells, target_mask)

    if len(lv_faces) == 0 or len(target_faces) == 0:
        final_tags = intermediate_tags.copy()
        final_tags[target_mask] = lv_tag
        keep_mask = np.ones(len(cells), dtype=bool)
        compact = {
            "input_points": int(len(points)),
            "output_points": int(len(points)),
            "input_cells": int(len(cells)),
            "output_cells": int(len(cells)),
            "dropped_cells": 0,
            "dropped_unused_points": 0,
        }
        info = {
            "enabled": bool(merge_points),
            "reason": "LV or target boundary faces were empty; only label fusion was applied.",
            "integration_tolerance": float(tolerance),
            "candidate_target_interface_faces": 0,
            "snapped_target_points": 0,
            "compact": compact,
        }
        return points, cells, final_tags, fiber_data, info, keep_mask, np.zeros(len(points), dtype=bool)

    lv_face_centers = points[lv_faces].mean(axis=1)
    target_face_centers = points[target_faces].mean(axis=1)
    nearest_face_ids, face_distances = nearest_indices_and_distances(target_face_centers, lv_face_centers)
    interface_face_mask = face_distances <= tolerance
    interface_target_faces = target_faces[interface_face_mask]
    interface_lv_faces = lv_faces[nearest_face_ids[interface_face_mask]]

    point_was_snapped = np.zeros(len(points), dtype=bool)
    point_map = np.arange(len(points), dtype=np.int64)
    snapped_distances: List[float] = []

    if merge_points and len(interface_target_faces):
        target_point_ids = np.unique(interface_target_faces.ravel())
        lv_point_ids = np.unique(interface_lv_faces.ravel())
        nearest_point_local, point_distances = nearest_indices_and_distances(points[target_point_ids], points[lv_point_ids])
        snap_mask = point_distances <= tolerance
        source_ids = target_point_ids[snap_mask]
        destination_ids = lv_point_ids[nearest_point_local[snap_mask]]
        point_map[source_ids] = destination_ids
        point_was_snapped[source_ids] = True
        snapped_distances = [float(value) for value in point_distances[snap_mask]]

    fused_cells = point_map[cells]
    final_tags = intermediate_tags.copy()
    final_tags[target_mask] = lv_tag

    duplicate_vertex_mask = np.asarray([len(set(map(int, cell))) < 4 for cell in fused_cells], dtype=bool)
    mapped_volumes = tet_volumes(points, fused_cells)
    negative_orientation_mask = mapped_volumes < -float(volume_eps)
    if np.any(negative_orientation_mask):
        fused_cells = fused_cells.copy()
        neg_ids = np.flatnonzero(negative_orientation_mask)
        first = fused_cells[neg_ids, 0].copy()
        fused_cells[neg_ids, 0] = fused_cells[neg_ids, 1]
        fused_cells[neg_ids, 1] = first
        mapped_volumes = tet_volumes(points, fused_cells)
    small_volume_mask = np.abs(mapped_volumes) <= float(volume_eps)
    invalid_mask = duplicate_vertex_mask | small_volume_mask
    keep_mask = ~invalid_mask if drop_degenerate else np.ones(len(fused_cells), dtype=bool)

    compact_points, compact_cells, compact_tags, compact_fiber, compact_info = compact_points_and_cells(
        points,
        fused_cells,
        final_tags,
        keep_mask,
        fiber_data,
    )
    info = {
        "enabled": bool(merge_points),
        "integration_tolerance": float(tolerance),
        "candidate_target_boundary_faces": int(len(target_faces)),
        "candidate_target_interface_faces": int(np.count_nonzero(interface_face_mask)),
        "candidate_lv_boundary_faces": int(len(lv_faces)),
        "snapped_target_points": int(np.count_nonzero(point_was_snapped)),
        "mean_snap_distance": float(np.mean(snapped_distances)) if snapped_distances else 0.0,
        "max_snap_distance": float(np.max(snapped_distances)) if snapped_distances else 0.0,
        "drop_degenerate_cells": bool(drop_degenerate),
        "negative_orientation_cells_reoriented": int(np.count_nonzero(negative_orientation_mask)),
        "duplicate_vertex_cells_after_mapping": int(np.count_nonzero(duplicate_vertex_mask)),
        "small_volume_cells_after_mapping": int(np.count_nonzero(small_volume_mask)),
        "dropped_invalid_cells": int(np.count_nonzero(~keep_mask)),
        "compact": compact_info,
    }
    return compact_points, compact_cells, compact_tags, compact_fiber, info, keep_mask, point_was_snapped


def write_debug_mesh(
    path: Path,
    mesh: pv.UnstructuredGrid,
    original_tags: np.ndarray,
    intermediate_tags: np.ndarray,
    final_tags_before_compact: np.ndarray,
    target_mask: np.ndarray,
    bridge_mask: np.ndarray,
    protected_rv: np.ndarray,
    snapped_points: np.ndarray,
) -> None:
    debug = mesh.copy(deep=True)
    debug.cell_data["original_elemTag"] = original_tags.astype(np.int32)
    debug.cell_data["intermediate_elemTag"] = intermediate_tags.astype(np.int32)
    debug.cell_data["finalElemTagBeforeCompact"] = final_tags_before_compact.astype(np.int32)
    debug.cell_data["septumTargetTag2"] = target_mask.astype(np.int32)
    debug.cell_data["bridgeCells"] = bridge_mask.astype(np.int32)
    debug.cell_data["protectedRVMainBody"] = protected_rv.astype(np.int32)
    debug.point_data["snappedToLVInterface"] = snapped_points.astype(np.int32)
    debug.save(path)


def main() -> None:
    args = parse_args()
    input_mesh = args.input_mesh
    mesh = pv.read(input_mesh)
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
        raise ValueError("--target-tag must be different from --lv-tag and --rv-tag.")

    lv_mask = original_tags == lv_tag
    rv_mask = original_tags == rv_tag
    if not np.any(lv_mask) or not np.any(rv_mask):
        raise ValueError("Input mesh must contain both LV and RV tags.")

    centroids = v1.tetra_centroids(points, cells)
    median_edge = v1.median_tet_edge_length(points, cells)
    adjacency = v1.build_adjacency(cells)

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
        raise RuntimeError("No RV main-body cells remain after applying the septum cutoff.")
    protected_rv_initial = np.zeros(len(cells), dtype=bool)
    protected_rv_initial[remaining_components[0]] = True

    initial_target = rv_mask & ~protected_rv_initial

    if args.integration_tolerance_mm is None:
        integration_tolerance = float(args.integration_tolerance_multiplier) * float(median_edge)
    else:
        integration_tolerance = float(args.integration_tolerance_mm)

    max_bridge_lv_distance = float(cutoff) + float(args.bridge_lv_distance_multiplier) * integration_tolerance
    expanded_target, bridge_info = expand_target_bridge(
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
        raise RuntimeError("No RV main-body cells remain after bridge expansion.")
    protected_rv = np.zeros(len(cells), dtype=bool)
    protected_rv[final_components[0]] = True
    target_mask = rv_mask & ~protected_rv
    bridge_mask = target_mask & ~initial_target

    intermediate_tags = original_tags.copy()
    intermediate_tags[target_mask] = target_tag
    intermediate_tags[protected_rv] = rv_tag

    final_tags_before_compact = intermediate_tags.copy()
    final_tags_before_compact[final_tags_before_compact == target_tag] = lv_tag

    fiber_data = mesh.cell_data.get("fiber", None)
    (
        fused_points,
        fused_cells,
        fused_tags,
        fused_fiber,
        fusion_info,
        keep_mask,
        snapped_points,
    ) = fuse_target_points_to_lv(
        points,
        cells,
        intermediate_tags,
        lv_tag=lv_tag,
        target_tag=target_tag,
        tolerance=integration_tolerance,
        merge_points=bool(args.merge_interface_points),
        drop_degenerate=bool(args.drop_degenerate_cells),
        volume_eps=float(args.volume_eps),
        fiber_data=fiber_data,
    )

    intermediate_path, final_path, debug_path, summary_path = infer_output_paths(input_mesh, args.output_dir)
    write_custom_vtk(intermediate_path, points, cells, intermediate_tags, fiber_data=fiber_data)
    write_custom_vtk(final_path, fused_points, fused_cells, fused_tags, fiber_data=fused_fiber)

    if args.write_debug_vtk:
        write_debug_mesh(
            debug_path,
            mesh,
            original_tags,
            intermediate_tags,
            final_tags_before_compact,
            target_mask,
            bridge_mask,
            protected_rv,
            snapped_points,
        )

    original_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, original_tags)).items())}
    intermediate_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, intermediate_tags)).items())}
    final_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, fused_tags)).items())}
    summary = {
        "script": str(Path(__file__).resolve()),
        "input_mesh": str(input_mesh.resolve()),
        "intermediate_target_tag2_mesh": str(intermediate_path.resolve()),
        "final_fused_lvtag_mesh": str(final_path.resolve()),
        "debug_mesh": str(debug_path.resolve()) if args.write_debug_vtk else None,
        "method": {
            "target_detection": "RV cells close to LV boundary are found by adaptive centroid-to-LV-boundary distance. The largest remaining RV component is protected as RV free wall.",
            "tag2_stage": "All RV cells outside the protected RV main body are written as elemTag=2 for inspection.",
            "bridge_stage": "The tag2 target can expand by a limited number of RV adjacency rings, but only through RV cells still near the LV boundary.",
            "fusion_stage": "The final mesh converts elemTag=2 to elemTag=1. If enabled, tag2-side interface points near LV boundary points are remapped to those LV points to close small LV-target gaps.",
            "lv_preservation": "Original LV point coordinates are never moved; only target-side interface points may be remapped to LV points.",
        },
        "labels": {
            "rv_tag": rv_tag,
            "lv_tag": lv_tag,
            "target_tag_intermediate": target_tag,
            "target_tag_after_fusion": lv_tag,
        },
        "mesh": {
            "input_points": int(mesh.n_points),
            "input_tetrahedra": int(len(cells)),
            "median_tet_edge_length": float(median_edge),
        },
        "cutoff": cutoff_info,
        "integration": {
            "integration_tolerance": float(integration_tolerance),
            "integration_tolerance_multiplier": None if args.integration_tolerance_mm is not None else float(args.integration_tolerance_multiplier),
            "integration_tolerance_mode": "manual" if args.integration_tolerance_mm is not None else "adaptive",
            "bridge": bridge_info,
            "fusion": fusion_info,
        },
        "counts": {
            "original_elemTag": original_counts,
            "intermediate_elemTag": intermediate_counts,
            "final_elemTag": final_counts,
            "initial_septum_candidate_cells": int(np.count_nonzero(septum_candidate)),
            "initial_target_cells": int(np.count_nonzero(initial_target)),
            "bridge_cells": int(np.count_nonzero(bridge_mask)),
            "final_target_tag2_cells_before_fusion": int(np.count_nonzero(target_mask)),
            "protected_rv_main_body_cells": int(np.count_nonzero(protected_rv)),
            "final_cells_after_degenerate_filter": int(len(fused_cells)),
        },
        "rv_connectivity": {
            "initial_remaining_rv_components_after_cutoff": int(len(remaining_components)),
            "initial_remaining_rv_component_sizes_after_cutoff": [int(len(component)) for component in remaining_components[:20]],
            "final_remaining_rv_components_after_bridge": int(len(final_components)),
            "final_remaining_rv_component_sizes_after_bridge": [int(len(component)) for component in final_components[:20]],
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "input_mesh": str(input_mesh),
        "intermediate_target_tag2_mesh": str(intermediate_path),
        "final_fused_lvtag_mesh": str(final_path),
        "summary": str(summary_path),
        "median_tet_edge_length": float(median_edge),
        "cutoff_mm": float(cutoff),
        "integration_tolerance_mm": float(integration_tolerance),
        "original_elemTag": original_counts,
        "intermediate_elemTag": intermediate_counts,
        "final_elemTag": final_counts,
        "target_tag2_cells_before_fusion": int(np.count_nonzero(target_mask)),
        "bridge_cells": int(np.count_nonzero(bridge_mask)),
        "snapped_target_points": int(fusion_info.get("snapped_target_points", 0)),
        "dropped_invalid_cells": int(fusion_info.get("dropped_invalid_cells", 0)),
    }, indent=2))


if __name__ == "__main__":
    main()
