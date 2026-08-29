#!/usr/bin/env python3
"""Retag RV septal/contact tetrahedra as LV in an open biventricular volume mesh.

The script is a label-only post-processor. It does not move mesh points and does
not delete tetrahedra. The intended input is the open-base volume mesh generated
by generate_subject_open_base_max_interface_smooth_boundary_custom_tag_volmesh.py.

Default labels:
  elemTag = 1: LV myocardium
  elemTag = 0: RV myocardium

Pipeline:
  input open vol mesh
  -> build tetra adjacency and median edge-length scale
  -> detect RV cells close to the LV boundary with an adaptive cutoff
  -> keep the largest non-candidate RV connected component as RV main body
  -> retag remaining RV/contact cells to LV
  -> optionally remove small isolated RV components
  -> write custom ASCII VTK with elemTag
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from custom_vtk import write_custom_vtk


TETRA_CELL_TYPE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-mesh",
        type=Path,
        default=Path("data/subject_1/volmesh_open_stable/subject_1_ED_biventricular_open_stable_custom_tag_volmesh.vtk"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--lv-tag", type=int, default=1)
    parser.add_argument("--rv-tag", type=int, default=0)
    parser.add_argument("--cutoff-mm", type=float, default=None)
    parser.add_argument("--cutoff-multiplier", type=float, default=3.0)
    parser.add_argument("--min-candidate-fraction", type=float, default=0.005)
    parser.add_argument("--max-cutoff-multiplier", type=float, default=4.0)
    parser.add_argument("--small-rv-component-ratio", type=float, default=0.001)
    parser.add_argument("--min-small-rv-component-cells", type=int, default=30)
    parser.add_argument("--write-debug-vtk", action="store_true")
    return parser.parse_args()


def tetra_connectivity(mesh: pv.UnstructuredGrid) -> np.ndarray:
    celltypes = np.asarray(mesh.celltypes)
    if len(celltypes) == 0:
        raise ValueError("Input mesh has no cells.")
    if not np.all(celltypes == TETRA_CELL_TYPE):
        unique = sorted(set(int(value) for value in celltypes))
        raise ValueError(f"Only tetrahedral VTK cell type 10 is supported; found cell types {unique}.")

    cells = np.asarray(mesh.cells, dtype=np.int64).reshape((-1, 5))
    if not np.all(cells[:, 0] == 4):
        raise ValueError("Expected tetrahedral VTK cell connectivity with 4 vertices per cell.")
    return cells[:, 1:5].astype(np.int64)


def tetra_centroids(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float)[cells].mean(axis=1)


def median_tet_edge_length(points: np.ndarray, cells: np.ndarray) -> float:
    tet_points = np.asarray(points, dtype=float)[cells]
    edges = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        edges.append(np.linalg.norm(tet_points[:, i] - tet_points[:, j], axis=1))
    return float(np.median(np.concatenate(edges)))


def tet_faces(cell: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    a, b, c, d = (int(v) for v in cell)
    return (
        tuple(sorted((b, c, d))),
        tuple(sorted((a, c, d))),
        tuple(sorted((a, b, d))),
        tuple(sorted((a, b, c))),
    )


def build_adjacency(cells: np.ndarray) -> List[List[int]]:
    face_to_cells: Dict[Tuple[int, int, int], List[int]] = {}
    for cell_id, cell in enumerate(cells):
        for face in tet_faces(cell):
            face_to_cells.setdefault(face, []).append(int(cell_id))

    adjacency_sets = [set() for _ in range(len(cells))]
    for ids in face_to_cells.values():
        if len(ids) < 2:
            continue
        for i in ids:
            for j in ids:
                if i != j:
                    adjacency_sets[i].add(j)
    return [sorted(values) for values in adjacency_sets]


def boundary_face_centers_for_label(
    points: np.ndarray,
    cells: np.ndarray,
    tags: np.ndarray,
    label: int,
) -> np.ndarray:
    face_counts: Dict[Tuple[int, int, int], int] = {}
    label_cell_ids = np.flatnonzero(tags == label)
    for cell_id in label_cell_ids:
        for face in tet_faces(cells[cell_id]):
            face_counts[face] = face_counts.get(face, 0) + 1

    boundary_faces = [face for face, count in face_counts.items() if count == 1]
    if not boundary_faces:
        raise RuntimeError(f"No boundary faces found for label {label}.")
    face_points = np.asarray(points, dtype=float)[np.asarray(boundary_faces, dtype=np.int64)]
    return face_points.mean(axis=1)


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(reference).query(query, k=1)
        return distances.astype(float)
    except Exception:
        output = np.empty(len(query), dtype=float)
        for start in range(0, len(query), 2048):
            chunk = query[start:start + 2048]
            dist2 = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            output[start:start + len(chunk)] = np.sqrt(dist2.min(axis=1))
        return output


def connected_components(mask: np.ndarray, adjacency: Sequence[Sequence[int]]) -> List[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(len(mask), dtype=bool)
    components: List[np.ndarray] = []

    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        component: List[int] = []
        while stack:
            cell_id = stack.pop()
            component.append(cell_id)
            for neighbor in adjacency[cell_id]:
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        components.append(np.asarray(component, dtype=np.int64))

    components.sort(key=len, reverse=True)
    return components


def choose_cutoff(
    distances: np.ndarray,
    median_edge: float,
    explicit_cutoff: Optional[float],
    cutoff_multiplier: float,
    min_candidate_fraction: float,
    max_cutoff_multiplier: float,
) -> Tuple[float, Dict[str, object]]:
    if explicit_cutoff is not None:
        cutoff = float(explicit_cutoff)
        return cutoff, {
            "mode": "manual",
            "cutoff_mm": cutoff,
            "cutoff_multiplier": None,
            "target_min_candidate_fraction": None,
        }

    base_cutoff = float(cutoff_multiplier) * float(median_edge)
    cutoff = base_cutoff
    min_candidates = int(np.ceil(len(distances) * float(min_candidate_fraction)))
    candidate_count = int(np.count_nonzero(distances <= cutoff))
    max_cutoff = float(max_cutoff_multiplier) * float(median_edge)

    if min_candidates > 0 and candidate_count < min_candidates:
        sorted_distances = np.sort(distances)
        required_index = min(max(min_candidates - 1, 0), len(sorted_distances) - 1)
        cutoff = min(max(base_cutoff, float(sorted_distances[required_index])), max_cutoff)
        candidate_count = int(np.count_nonzero(distances <= cutoff))

    return cutoff, {
        "mode": "adaptive",
        "base_cutoff_mm": base_cutoff,
        "cutoff_mm": float(cutoff),
        "cutoff_multiplier": float(cutoff_multiplier),
        "max_cutoff_mm": float(max_cutoff),
        "max_cutoff_multiplier": float(max_cutoff_multiplier),
        "target_min_candidate_fraction": float(min_candidate_fraction),
        "candidate_count_after_adaptation": int(candidate_count),
    }


def cleanup_small_rv_components(
    tags: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    rv_tag: int,
    lv_tag: int,
    min_component_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    cleaned = np.asarray(tags, dtype=np.int32).copy()
    rv_components = connected_components(cleaned == rv_tag, adjacency)
    changed = 0
    removed_components = 0
    removed_sizes: List[int] = []

    for component in rv_components[1:]:
        if len(component) >= min_component_size:
            continue
        neighbor_tags: List[int] = []
        component_set = set(int(c) for c in component)
        for cell_id in component:
            for neighbor in adjacency[int(cell_id)]:
                if neighbor not in component_set:
                    neighbor_tags.append(int(cleaned[neighbor]))
        if neighbor_tags and Counter(neighbor_tags).most_common(1)[0][0] == lv_tag:
            cleaned[component] = lv_tag
            changed += int(len(component))
            removed_components += 1
            removed_sizes.append(int(len(component)))

    return cleaned, {
        "enabled": True,
        "min_component_size": int(min_component_size),
        "removed_small_rv_components": int(removed_components),
        "removed_small_rv_cells": int(changed),
        "removed_component_sizes": removed_sizes,
    }


def infer_output_paths(input_mesh: Path, output_dir: Optional[Path], output_name: Optional[str]) -> Tuple[Path, Path]:
    if output_dir is None:
        output_dir = input_mesh.parent.parent / "vol_mesh_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        stem = input_mesh.stem
        output_name = stem.replace("_volmesh", "_septum_lvtag_volmesh")
        if output_name == stem:
            output_name = f"{stem}_septum_lvtag"
        output_name = f"{output_name}.vtk"

    output_mesh = output_dir / output_name
    summary_path = output_dir / output_name.replace(".vtk", "_summary.json")
    return output_mesh, summary_path


def write_debug_mesh(
    path: Path,
    mesh: pv.UnstructuredGrid,
    original_tags: np.ndarray,
    final_tags: np.ndarray,
    septum_candidate: np.ndarray,
    protected_rv: np.ndarray,
) -> None:
    debug = mesh.copy(deep=True)
    debug.cell_data["original_elemTag"] = original_tags.astype(np.int32)
    debug.cell_data["elemTag"] = final_tags.astype(np.int32)
    debug.cell_data["septumCandidate"] = septum_candidate.astype(np.int32)
    debug.cell_data["protectedRVMainBody"] = protected_rv.astype(np.int32)
    debug.save(path)


def main() -> None:
    args = parse_args()
    input_mesh = args.input_mesh
    mesh = pv.read(input_mesh)
    points = np.asarray(mesh.points, dtype=float)
    cells = tetra_connectivity(mesh)

    if "elemTag" not in mesh.cell_data:
        raise KeyError("Input mesh must contain cell_data['elemTag'].")
    original_tags = np.asarray(mesh.cell_data["elemTag"], dtype=np.int32)
    if len(original_tags) != len(cells):
        raise ValueError("elemTag length does not match tetra cell count.")

    lv_mask = original_tags == int(args.lv_tag)
    rv_mask = original_tags == int(args.rv_tag)
    if not np.any(lv_mask) or not np.any(rv_mask):
        raise ValueError("Input mesh must contain both LV and RV tags.")

    centroids = tetra_centroids(points, cells)
    median_edge = median_tet_edge_length(points, cells)
    adjacency = build_adjacency(cells)

    lv_boundary_centers = boundary_face_centers_for_label(points, cells, original_tags, int(args.lv_tag))
    rv_indices = np.flatnonzero(rv_mask)
    rv_to_lv_distance = nearest_distances(centroids[rv_indices], lv_boundary_centers)
    cutoff, cutoff_info = choose_cutoff(
        rv_to_lv_distance,
        median_edge,
        args.cutoff_mm,
        args.cutoff_multiplier,
        args.min_candidate_fraction,
        args.max_cutoff_multiplier,
    )

    rv_candidate_local = rv_to_lv_distance <= cutoff
    septum_candidate = np.zeros(len(cells), dtype=bool)
    septum_candidate[rv_indices[rv_candidate_local]] = True

    remaining_rv = rv_mask & ~septum_candidate
    remaining_components = connected_components(remaining_rv, adjacency)
    if not remaining_components:
        raise RuntimeError("No RV main-body cells remain after applying the septum cutoff. Reduce --cutoff-mm or --cutoff-multiplier.")

    protected_rv = np.zeros(len(cells), dtype=bool)
    protected_rv[remaining_components[0]] = True

    final_tags = original_tags.copy()
    convert_to_lv = rv_mask & ~protected_rv
    final_tags[convert_to_lv] = int(args.lv_tag)

    min_small_size = max(
        int(args.min_small_rv_component_cells),
        int(round(len(cells) * float(args.small_rv_component_ratio))),
    )
    cleaned_tags, cleanup_info = cleanup_small_rv_components(
        final_tags,
        adjacency,
        rv_tag=int(args.rv_tag),
        lv_tag=int(args.lv_tag),
        min_component_size=min_small_size,
    )

    output_mesh, summary_path = infer_output_paths(input_mesh, args.output_dir, args.output_name)
    fiber_data = mesh.cell_data.get("fiber", None)
    write_custom_vtk(output_mesh, points, cells, cleaned_tags, fiber_data=fiber_data)

    if args.write_debug_vtk:
        write_debug_mesh(
            output_mesh.with_name(output_mesh.stem + "_debug.vtk"),
            mesh,
            original_tags,
            cleaned_tags,
            septum_candidate,
            protected_rv,
        )

    original_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, original_tags)).items())}
    final_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, cleaned_tags)).items())}
    summary = {
        "script": str(Path(__file__).resolve()),
        "input_mesh": str(input_mesh.resolve()),
        "output_mesh": str(output_mesh.resolve()),
        "method": {
            "description": "Distance-based RV septal/contact retagging with RV main-body connectivity protection.",
            "distance_rule": "RV tetra centroids are compared with LV boundary-face centroids. RV cells within the cutoff become septum candidates.",
            "rv_protection_rule": "After removing septum candidates, the largest remaining RV connected component is treated as the RV main body and kept as RV.",
            "retag_rule": "All original RV cells outside the protected RV main body are retagged to LV.",
            "cleanup_rule": "Small isolated RV components are converted to LV only when their neighboring majority tag is LV.",
        },
        "labels": {
            "lv_tag": int(args.lv_tag),
            "rv_tag": int(args.rv_tag),
        },
        "mesh": {
            "points": int(mesh.n_points),
            "tetrahedra": int(len(cells)),
            "median_tet_edge_length": float(median_edge),
        },
        "cutoff": cutoff_info,
        "counts": {
            "original_elemTag": original_counts,
            "final_elemTag": final_counts,
            "rv_cells": int(np.count_nonzero(rv_mask)),
            "lv_cells": int(np.count_nonzero(lv_mask)),
            "septum_candidate_rv_cells": int(np.count_nonzero(septum_candidate)),
            "protected_rv_main_body_cells": int(np.count_nonzero(protected_rv)),
            "retagged_rv_to_lv_before_cleanup": int(np.count_nonzero(convert_to_lv)),
            "total_changed_cells_after_cleanup": int(np.count_nonzero(cleaned_tags != original_tags)),
        },
        "rv_connectivity": {
            "remaining_rv_components_after_cutoff": int(len(remaining_components)),
            "remaining_rv_component_sizes_after_cutoff": [int(len(component)) for component in remaining_components[:20]],
            "kept_component_policy": "largest_remaining_rv_component",
        },
        "cleanup": cleanup_info,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "input_mesh": str(input_mesh),
        "output_mesh": str(output_mesh),
        "summary": str(summary_path),
        "cutoff_mm": float(cutoff),
        "median_tet_edge_length": float(median_edge),
        "original_elemTag": original_counts,
        "final_elemTag": final_counts,
        "septum_candidate_rv_cells": int(np.count_nonzero(septum_candidate)),
        "retagged_rv_to_lv": int(np.count_nonzero(cleaned_tags != original_tags)),
    }, indent=2))


if __name__ == "__main__":
    main()
