#!/usr/bin/env python3
"""Generate an open-base biventricular volume mesh with explicit RV/LVepi fixing.

Pipeline:
  1. Load LVendo, LVepi, RV, RVepi surfaces.
  2. Detect subject base side and write LV/RV Z-area profiles.
  3. Build the open-base LV myocardial shell.
  4. Build the initial open-base RV myocardial shell.
  5. Remove RVepi-open faces intersecting LVepi.
  6. Remove RV-open faces intersecting LVepi.
  7. Reconstruct a new RV open myocardial shell from the trimmed RV/RVepi.
  8. Keep RVepi fixed and move RV inward until RV-to-RVepi distance is >= 3 mm.
  9. Tetrahedralize LV and RV shells separately, using TetGen quality refinement.
 10. Merge LV/RV tets and delete RV tets lying inside LV.
 11. If LV/RV are not face-connected, grow RV tets toward LV with local bridge cells.
 12. Write the final VTK with elemTag=1 for LV and elemTag=0 for RV.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import vtk


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MESH_VOL_DIR = REPO_ROOT / "mesh_vol"
CODE_EXPLAIN_TMPS_DIR = REPO_ROOT / "code_explain" / "tmps"
for extra_path in (REPO_ROOT, MESH_VOL_DIR, CODE_EXPLAIN_TMPS_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import generate_subject_open_base_max_interface_smooth_boundary_custom_tag_volmesh as openbase
from custom_vtk import write_custom_vtk


base = openbase.base
cap_utils = openbase.cap_utils
TETRA_CELL_TYPE = 10


@dataclass
class RebuiltShell:
    name: str
    endo_open: pv.PolyData
    epi_open: pv.PolyData
    cut_surfaces: List[pv.PolyData]
    shell: pv.PolyData
    loop_pairs: List[Dict[str, object]]
    unmatched_loops: Dict[str, List[Dict[str, object]]]
    free_edges_before_repair: int
    free_edges_after_repair: int


@dataclass
class TetMesh:
    name: str
    points: np.ndarray
    cells: np.ndarray
    tag: int
    seed: np.ndarray
    quality: Dict[str, object]


def parse_csv_values(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--frames", default=None, help="Comma-separated frames, for example ED,ES.")
    parser.add_argument("--vtk-dir", type=Path, default=None, help="Defaults to SUBJECT_DIR/vtks.")
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to SUBJECT_DIR/volmesh_LVRV_septum_fine.")
    parser.add_argument("--save-debug", action="store_true", help="Save intermediate surfaces and chamber meshes.")

    parser.add_argument("--subject-base-side", choices=["auto", "low_z", "high_z", "off"], default="auto")
    parser.add_argument("--subject-base-frames", default="ED,ES")
    parser.add_argument("--lv-base-side", choices=["low_z", "high_z"], default=None)
    parser.add_argument("--rv-base-side", choices=["low_z", "high_z"], default=None)

    parser.add_argument("--opening-strategy", choices=["stable_thickness", "max_interface"], default="stable_thickness")
    parser.add_argument("--stable-thickness-percentile", type=float, default=95.0)
    parser.add_argument("--stable-thickness-candidate-percentiles", default="90,92.5,95,97.5,98,99,99.5,100")
    parser.add_argument("--stable-thickness-seed-band-multiplier", type=float, default=2.0)
    parser.add_argument("--stable-thickness-min-basal-face-fraction", type=float, default=0.01)
    parser.add_argument("--stable-thickness-max-search-fraction", type=float, default=0.30)
    parser.add_argument("--stable-thickness-search-steps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--slab-fraction", type=float, default=0.035)
    parser.add_argument("--end-band-fraction", type=float, default=0.12)
    parser.add_argument("--stable-fraction", type=float, default=0.45)
    parser.add_argument("--seed-band-fraction", type=float, default=0.035)
    parser.add_argument("--min-shift", type=float, default=0.75)
    parser.add_argument("--min-shift-fraction", type=float, default=0.01)
    parser.add_argument("--max-shift-fraction", type=float, default=0.25)
    parser.add_argument("--shift-steps", type=int, default=40)
    parser.add_argument("--interface-candidate-rank", type=int, default=0)
    parser.add_argument("--max-interface-candidate-rank", type=int, default=4)
    parser.add_argument("--target-intersection-area-fraction", type=float, default=0.08)
    parser.add_argument("--min-intersection-area", type=float, default=250.0)
    parser.add_argument("--min-intersection-faces", type=int, default=80)
    parser.add_argument("--max-intersection-depth-fraction", type=float, default=0.45)
    parser.add_argument("--max-secondary-loop-area-fraction", type=float, default=0.05)
    parser.add_argument("--boundary-smooth-iterations", type=int, default=40)
    parser.add_argument("--boundary-smooth-relaxation", type=float, default=0.18)
    parser.add_argument("--boundary-blend-rings", type=int, default=2)
    parser.add_argument("--boundary-blend-relaxation", type=float, default=0.35)
    parser.add_argument("--control-loop-points", type=int, default=120)
    parser.add_argument("--control-smooth-iterations", type=int, default=20)
    parser.add_argument("--control-smooth-relaxation", type=float, default=0.20)
    parser.add_argument("--annulus-loop-points", type=int, default=120)
    parser.add_argument("--annulus-radial-layers", type=int, default=4)
    parser.add_argument("--annulus-smooth-iterations", type=int, default=30)
    parser.add_argument("--annulus-smooth-relaxation", type=float, default=0.20)

    parser.add_argument("--surface-lvepi-clearance", type=float, default=0.0)
    parser.add_argument("--surface-removal-rings", type=int, default=1)
    parser.add_argument("--surface-component-min-fraction", type=float, default=0.01)
    parser.add_argument("--boundary-loop-min-length-fraction", type=float, default=0.02)
    parser.add_argument("--surface-clean-tolerance", type=float, default=1e-7)

    parser.add_argument("--rv-min-distance-to-rvepi", type=float, default=3.0)
    parser.add_argument("--rv-distance-padding", type=float, default=0.05)
    parser.add_argument("--rv-distance-iterations", type=int, default=8)
    parser.add_argument("--rv-distance-relaxation", type=float, default=0.85)
    parser.add_argument("--rv-distance-max-step", type=float, default=2.0)
    parser.add_argument("--rv-distance-blend-rings", type=int, default=2)
    parser.add_argument("--rv-distance-blend-falloff", type=float, default=0.5)
    parser.add_argument("--rv-distance-smooth-iterations", type=int, default=4)
    parser.add_argument("--rv-distance-smooth-relaxation", type=float, default=0.25)

    parser.add_argument("--lv-tag", type=int, default=1)
    parser.add_argument("--rv-tag", type=int, default=0)
    parser.add_argument("--inside-tolerance", type=float, default=1e-5)
    parser.add_argument("--rv-overlap-vertex-fraction", type=float, default=0.25)
    parser.add_argument("--rv-overlap-removal-rings", type=int, default=1)
    parser.add_argument("--growth-max-distance", type=float, default=None)
    parser.add_argument("--growth-edge-multiplier", type=float, default=8.0)
    parser.add_argument("--growth-nearest-face-count", type=int, default=2500)
    parser.add_argument("--bridge-min-abs-volume", type=float, default=1e-8)
    parser.add_argument("--max-edge-length-multiplier", type=float, default=8.0)
    parser.add_argument("--max-volume-multiplier", type=float, default=50.0)
    return parser.parse_args()


def resolve_vtk_dir(subject_dir: Path, vtk_dir: Optional[Path]) -> Path:
    if vtk_dir is None:
        return subject_dir / "vtks"
    if vtk_dir.is_absolute():
        return vtk_dir
    subject_relative = subject_dir / vtk_dir
    return subject_relative if subject_relative.exists() else vtk_dir


def resolve_output_dir(subject_dir: Path, output_root: Optional[Path]) -> Path:
    output_dir = output_root or subject_dir / "volmesh_LVRV_septum_fine"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def input_paths_for_frame(vtk_dir: Path, frame: str) -> Dict[str, Path]:
    return {
        "lv_endo": vtk_dir / f"LVendo_{frame}.vtk",
        "lv_epi": vtk_dir / f"LVepi_{frame}.vtk",
        "rv_endo": vtk_dir / f"RV_{frame}.vtk",
        "rv_epi": vtk_dir / f"RVepi_{frame}.vtk",
    }


def load_surfaces(vtk_dir: Path, frame: str) -> Dict[str, pv.PolyData]:
    paths = input_paths_for_frame(vtk_dir, frame)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required surface files: " + ", ".join(missing))
    return {name: base.load_surface(path) for name, path in paths.items()}


def serializable_args(args: argparse.Namespace) -> Dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def signed_distances(reference: pv.PolyData, points: np.ndarray) -> np.ndarray:
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(reference)
    pts = np.asarray(points, dtype=float)
    out = np.empty(len(pts), dtype=float)
    for index, point in enumerate(pts):
        out[index] = float(implicit.EvaluateFunction([float(point[0]), float(point[1]), float(point[2])]))
    return out


def nearest_distances_and_indices(query: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(np.asarray(reference, dtype=float)).query(np.asarray(query, dtype=float))
        return distances.astype(float), indices.astype(np.int64)
    except Exception:
        ref = np.asarray(reference, dtype=float)
        distances: List[np.ndarray] = []
        indices: List[np.ndarray] = []
        for start in range(0, len(query), 512):
            chunk = np.asarray(query[start:start + 512], dtype=float)
            dist2 = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(dist2, axis=1)
            indices.append(idx.astype(np.int64))
            distances.append(np.sqrt(dist2[np.arange(len(chunk)), idx]))
        return np.concatenate(distances), np.concatenate(indices)


def unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = set()
    for tri in np.asarray(triangles, dtype=np.int64):
        a, b, c = (int(v) for v in tri)
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.asarray(sorted(edges), dtype=np.int64)


def face_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def connected_face_components(mask: np.ndarray, triangles: np.ndarray) -> List[np.ndarray]:
    adjacency = base.face_adjacency(np.asarray(triangles, dtype=np.int64))
    visited = np.zeros(len(mask), dtype=bool)
    components: List[np.ndarray] = []
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        queue: deque[int] = deque([int(start)])
        visited[start] = True
        ids: List[int] = []
        while queue:
            current = queue.popleft()
            ids.append(current)
            for nb in adjacency[current]:
                if mask[nb] and not visited[nb]:
                    visited[nb] = True
                    queue.append(int(nb))
        components.append(np.asarray(ids, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def expand_face_mask(mask: np.ndarray, triangles: np.ndarray, rings: int) -> np.ndarray:
    expanded = np.asarray(mask, dtype=bool).copy()
    if rings <= 0 or not np.any(expanded):
        return expanded
    adjacency = base.face_adjacency(np.asarray(triangles, dtype=np.int64))
    frontier = set(int(v) for v in np.flatnonzero(expanded))
    for _ in range(int(rings)):
        next_frontier = set()
        for face_id in frontier:
            for nb in adjacency[face_id]:
                if not expanded[nb]:
                    expanded[nb] = True
                    next_frontier.add(int(nb))
        if not next_frontier:
            break
        frontier = next_frontier
    return expanded


def keep_large_surface_components(mesh: pv.PolyData, min_fraction: float) -> Tuple[pv.PolyData, Dict[str, object]]:
    triangles = base.triangle_faces(mesh)
    points = np.asarray(mesh.points, dtype=float)
    if len(triangles) == 0:
        raise RuntimeError("Surface has no triangles after intersection removal.")
    areas = face_areas(points, triangles)
    components = connected_face_components(np.ones(len(triangles), dtype=bool), triangles)
    if not components:
        raise RuntimeError("Surface has no connected face component after intersection removal.")
    max_area = float(areas[components[0]].sum())
    keep = np.zeros(len(triangles), dtype=bool)
    kept_components = []
    for component in components:
        area = float(areas[component].sum())
        if area >= max_area * float(min_fraction):
            keep[component] = True
            kept_components.append({"faces": int(len(component)), "area": area})
    out = base.subset_mesh_faces(mesh, keep)
    return out.extract_surface().triangulate().clean(tolerance=1e-7), {
        "component_count": int(len(components)),
        "kept_component_count": int(len(kept_components)),
        "kept_components": kept_components[:20],
        "removed_component_faces": int(np.count_nonzero(~keep)),
    }


def remove_surface_intersection_with_lvepi(
    name: str,
    surface: pv.PolyData,
    lv_epi: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    triangles = base.triangle_faces(surface)
    points = np.asarray(surface.points, dtype=float)
    clearance = float(args.surface_lvepi_clearance)

    vertex_distances = signed_distances(lv_epi, points)
    vertex_hit = vertex_distances <= clearance
    centers = points[triangles].mean(axis=1)
    center_hit = signed_distances(lv_epi, centers) <= clearance
    edges = unique_edges(triangles)
    edge_midpoints = points[edges].mean(axis=1)
    edge_hit = signed_distances(lv_epi, edge_midpoints) <= clearance

    remove = center_hit | vertex_hit[triangles].any(axis=1)
    if np.any(edge_hit):
        edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for face_id, tri in enumerate(triangles):
            a, b, c = (int(v) for v in tri)
            for edge in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
                edge_to_faces[edge].append(int(face_id))
        for edge in edges[edge_hit]:
            for face_id in edge_to_faces.get(tuple(int(v) for v in edge), []):
                remove[face_id] = True

    remove = expand_face_mask(remove, triangles, args.surface_removal_rings)
    keep = ~remove
    if not np.any(keep):
        raise RuntimeError(f"{name}: all faces were removed by LVepi intersection deletion.")
    trimmed = base.subset_mesh_faces(surface, keep)
    trimmed, component_info = keep_large_surface_components(trimmed, args.surface_component_min_fraction)
    info = {
        "surface": name,
        "input_points": int(surface.n_points),
        "input_faces": int(len(triangles)),
        "clearance_mm": clearance,
        "vertex_hits": int(np.count_nonzero(vertex_hit)),
        "face_center_hits": int(np.count_nonzero(center_hit)),
        "edge_midpoint_hits": int(np.count_nonzero(edge_hit)),
        "removed_faces_after_dilation": int(np.count_nonzero(remove)),
        "kept_faces_before_component_cleanup": int(np.count_nonzero(keep)),
        "output_points": int(trimmed.n_points),
        "output_faces": int(trimmed.n_cells),
        "component_cleanup": component_info,
        "min_signed_distance_to_LVepi": float(vertex_distances.min()) if len(vertex_distances) else None,
    }
    return trimmed, info


def loop_points(loop: pv.PolyData) -> np.ndarray:
    return cap_utils.remove_duplicate_sequential_points(np.asarray(loop.points, dtype=float))


def loop_record(loop: pv.PolyData) -> Dict[str, object]:
    points = loop_points(loop)
    return {
        "points": int(len(points)),
        "length": float(loop.length),
        "area_best_plane": float(abs(base.signed_area_in_best_plane(points))),
        "centroid": [float(x) for x in points.mean(axis=0)] if len(points) else [0.0, 0.0, 0.0],
    }


def filtered_boundary_loops(mesh: pv.PolyData, min_fraction: float) -> List[pv.PolyData]:
    loops = base.boundary_loops(mesh)
    if not loops:
        return []
    longest = max(float(loop.length) for loop in loops)
    selected = [loop for loop in loops if float(loop.length) >= longest * float(min_fraction)]
    return selected or [loops[0]]


def pair_boundary_loops(
    endo_loops: Sequence[pv.PolyData],
    epi_loops: Sequence[pv.PolyData],
) -> Tuple[List[Tuple[pv.PolyData, pv.PolyData, float]], List[pv.PolyData], List[pv.PolyData]]:
    if not endo_loops or not epi_loops:
        return [], list(endo_loops), list(epi_loops)
    candidates: List[Tuple[float, int, int]] = []
    for endo_id, endo_loop in enumerate(endo_loops):
        endo_center = loop_points(endo_loop).mean(axis=0)
        for epi_id, epi_loop in enumerate(epi_loops):
            epi_center = loop_points(epi_loop).mean(axis=0)
            candidates.append((float(np.linalg.norm(endo_center - epi_center)), endo_id, epi_id))
    candidates.sort(key=lambda item: item[0])
    used_endo = set()
    used_epi = set()
    pairs: List[Tuple[pv.PolyData, pv.PolyData, float]] = []
    for distance, endo_id, epi_id in candidates:
        if endo_id in used_endo or epi_id in used_epi:
            continue
        used_endo.add(endo_id)
        used_epi.add(epi_id)
        pairs.append((endo_loops[endo_id], epi_loops[epi_id], distance))
    unmatched_endo = [loop for index, loop in enumerate(endo_loops) if index not in used_endo]
    unmatched_epi = [loop for index, loop in enumerate(epi_loops) if index not in used_epi]
    return pairs, unmatched_endo, unmatched_epi


def cap_boundary_loop(loop: pv.PolyData) -> pv.PolyData:
    points = loop_points(loop)
    if len(points) < 3:
        raise RuntimeError("Cannot cap a boundary loop with fewer than three points.")
    center = points.mean(axis=0, keepdims=True)
    cap_points = np.vstack([points, center])
    center_id = len(points)
    triangles = [[i, (i + 1) % len(points), center_id] for i in range(len(points))]
    return pv.PolyData(cap_points, base.faces_from_triangles(np.asarray(triangles, dtype=np.int64))).extract_surface().triangulate().clean(tolerance=1e-7)


def reconstruct_open_myocardial_shell(
    name: str,
    endo_open: pv.PolyData,
    epi_open: pv.PolyData,
    args: argparse.Namespace,
) -> RebuiltShell:
    endo_open = endo_open.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    epi_open = epi_open.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    endo_loops = filtered_boundary_loops(endo_open, args.boundary_loop_min_length_fraction)
    epi_loops = filtered_boundary_loops(epi_open, args.boundary_loop_min_length_fraction)
    pairs, unmatched_endo, unmatched_epi = pair_boundary_loops(endo_loops, epi_loops)
    if not pairs:
        raise RuntimeError(f"{name}: no endocardial/epicardial boundary-loop pairs available for shell reconstruction.")

    cut_surfaces: List[pv.PolyData] = []
    loop_pairs: List[Dict[str, object]] = []
    for pair_id, (endo_loop, epi_loop, distance) in enumerate(pairs):
        cut = cap_utils.smooth_myocardium_cut_surface(
            epi_loop,
            endo_loop,
            sample_count=args.annulus_loop_points,
            radial_layers=args.annulus_radial_layers,
            smooth_iterations=args.annulus_smooth_iterations,
            smooth_relaxation=args.annulus_smooth_relaxation,
        )
        cut_surfaces.append(cut)
        loop_pairs.append({
            "pair_id": int(pair_id),
            "centroid_distance": float(distance),
            "endo_loop": loop_record(endo_loop),
            "epi_loop": loop_record(epi_loop),
            "cut_surface_points": int(cut.n_points),
            "cut_surface_faces": int(cut.n_cells),
        })

    unmatched_records = {"endo": [loop_record(loop) for loop in unmatched_endo], "epi": [loop_record(loop) for loop in unmatched_epi]}
    for loop in unmatched_endo:
        cut_surfaces.append(cap_boundary_loop(loop))
    for loop in unmatched_epi:
        cut_surfaces.append(cap_boundary_loop(loop))

    shell_before_repair = base.merge_surfaces([epi_open, endo_open, *cut_surfaces])
    free_before = len(base.extract_boundary_edges(shell_before_repair))
    shell = base.repair_closed_surface(shell_before_repair)
    free_after = len(base.extract_boundary_edges(shell))
    if free_after:
        raise RuntimeError(f"{name}: reconstructed shell still has {free_after} free edges after repair.")
    return RebuiltShell(
        name=name,
        endo_open=endo_open,
        epi_open=epi_open,
        cut_surfaces=cut_surfaces,
        shell=shell,
        loop_pairs=loop_pairs,
        unmatched_loops=unmatched_records,
        free_edges_before_repair=int(free_before),
        free_edges_after_repair=int(free_after),
    )


def surface_vertex_neighbors(triangles: np.ndarray, point_count: int) -> List[set]:
    neighbors = [set() for _ in range(point_count)]
    for tri in np.asarray(triangles, dtype=np.int64):
        a, b, c = (int(v) for v in tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def surface_sample_sets(mesh: pv.PolyData) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(mesh.points, dtype=float)
    triangles = base.triangle_faces(mesh)
    edges = unique_edges(triangles)
    return {
        "vertices": (points, np.arange(len(points), dtype=np.int64)[:, None]),
        "face_centers": (points[triangles].mean(axis=1), triangles),
        "edge_midpoints": (points[edges].mean(axis=1), edges),
    }


def add_displacement(displacement: np.ndarray, counts: np.ndarray, vertex_ids: np.ndarray, vectors: np.ndarray) -> None:
    if len(vertex_ids) == 0:
        return
    repeated = np.repeat(vectors, vertex_ids.shape[1], axis=0)
    flat_ids = vertex_ids.reshape(-1)
    np.add.at(displacement, flat_ids, repeated)
    np.add.at(counts, flat_ids, 1.0)


def normalize_vectors(vectors: np.ndarray, fallback_center: np.ndarray, samples: np.ndarray) -> np.ndarray:
    out = np.asarray(vectors, dtype=float).copy()
    norms = np.linalg.norm(out, axis=1)
    bad = norms < 1e-12
    if np.any(bad):
        fallback = samples[bad] - fallback_center[None, :]
        fallback_norm = np.linalg.norm(fallback, axis=1)
        fallback_bad = fallback_norm < 1e-12
        fallback[fallback_bad] = np.array([0.0, 0.0, -1.0])
        fallback_norm[fallback_bad] = 1.0
        out[bad] = fallback / fallback_norm[:, None]
        norms[bad] = 1.0
    return out / np.maximum(norms[:, None], 1e-12)


def blend_displacement(
    points: np.ndarray,
    triangles: np.ndarray,
    displacement: np.ndarray,
    source_mask: np.ndarray,
    rings: int,
    falloff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    blended = displacement.copy()
    moving = source_mask.copy()
    if rings <= 0 or not np.any(source_mask):
        return blended, moving
    source_ids = np.flatnonzero(source_mask)
    source_points = points[source_ids]
    source_displacement = displacement[source_ids]
    neighbors = surface_vertex_neighbors(triangles, len(points))
    visited = set(int(v) for v in source_ids)
    frontier = set(visited)
    for ring in range(1, int(rings) + 1):
        next_frontier = set()
        for vertex_id in frontier:
            for nb in neighbors[vertex_id]:
                if nb not in visited:
                    next_frontier.add(int(nb))
        if not next_frontier:
            break
        ring_ids = np.asarray(sorted(next_frontier), dtype=np.int64)
        _, nearest = nearest_distances_and_indices(points[ring_ids], source_points)
        blended[ring_ids] += source_displacement[nearest] * (float(falloff) ** float(ring))
        moving[ring_ids] = True
        visited.update(next_frontier)
        frontier = next_frontier
    return blended, moving


def smooth_displacement(triangles: np.ndarray, displacement: np.ndarray, moving_mask: np.ndarray, iterations: int, relaxation: float) -> np.ndarray:
    if iterations <= 0 or not np.any(moving_mask):
        return displacement
    neighbors = surface_vertex_neighbors(triangles, len(displacement))
    moving_ids = np.flatnonzero(moving_mask)
    out = displacement.copy()
    for _ in range(int(iterations)):
        updated = out.copy()
        for vertex_id in moving_ids:
            linked = list(neighbors[int(vertex_id)])
            if not linked:
                continue
            ids = np.asarray(linked, dtype=np.int64)
            updated[vertex_id] = out[vertex_id] + float(relaxation) * (out[ids].mean(axis=0) - out[vertex_id])
        out = updated
    return out


def rv_to_rvepi_distance_summary(rv: pv.PolyData, rvepi: pv.PolyData, args: argparse.Namespace) -> Dict[str, object]:
    rvepi_points = np.asarray(rvepi.points, dtype=float)
    required = float(args.rv_min_distance_to_rvepi)
    summary: Dict[str, object] = {"required_min_distance_mm": required}
    passes = True
    for name, (sample_points, _) in surface_sample_sets(rv).items():
        distances, _ = nearest_distances_and_indices(sample_points, rvepi_points)
        below = distances < required
        passes = passes and not np.any(below)
        summary[name] = {
            "samples": int(len(sample_points)),
            "below_min_distance": int(np.count_nonzero(below)),
            "min_distance_mm": float(distances.min()) if len(distances) else None,
            "p05_distance_mm": float(np.percentile(distances, 5)) if len(distances) else None,
            "p50_distance_mm": float(np.percentile(distances, 50)) if len(distances) else None,
        }
    summary["passes_rule"] = bool(passes)
    return summary


def move_rv_to_min_rvepi_distance(rv: pv.PolyData, rvepi: pv.PolyData, args: argparse.Namespace) -> Tuple[pv.PolyData, Dict[str, object]]:
    current = rv.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    rvepi_points = np.asarray(rvepi.points, dtype=float)
    target = float(args.rv_min_distance_to_rvepi) + float(args.rv_distance_padding)
    initial = rv_to_rvepi_distance_summary(current, rvepi, args)
    iterations: List[Dict[str, object]] = []

    for iteration in range(1, int(args.rv_distance_iterations) + 1):
        points = np.asarray(current.points, dtype=float).copy()
        triangles = base.triangle_faces(current)
        displacement = np.zeros_like(points)
        counts = np.zeros(len(points), dtype=float)
        sample_reports = {}
        for sample_name, (sample_points, sample_vertex_ids) in surface_sample_sets(current).items():
            distances, nearest = nearest_distances_and_indices(sample_points, rvepi_points)
            required = np.maximum(target - distances, 0.0)
            conflict = required > 0.0
            if np.any(conflict):
                directions = normalize_vectors(sample_points[conflict] - rvepi_points[nearest[conflict]], np.asarray(rvepi.center, dtype=float), sample_points[conflict])
                vectors = directions * required[conflict, None]
                add_displacement(displacement, counts, sample_vertex_ids[conflict], vectors)
            sample_reports[sample_name] = {
                "samples": int(len(sample_points)),
                "conflicts": int(np.count_nonzero(conflict)),
                "min_distance_mm": float(distances.min()) if len(distances) else None,
            }

        source_mask = counts > 0.0
        if not np.any(source_mask):
            iterations.append({"iteration": iteration, "source_vertices": 0, "sample_reports": sample_reports})
            break
        displacement[source_mask] /= counts[source_mask, None]
        displacement *= float(args.rv_distance_relaxation)
        lengths = np.linalg.norm(displacement, axis=1)
        too_large = lengths > float(args.rv_distance_max_step)
        if np.any(too_large):
            displacement[too_large] *= (float(args.rv_distance_max_step) / lengths[too_large])[:, None]
        displacement, moving_mask = blend_displacement(
            points,
            triangles,
            displacement,
            source_mask,
            args.rv_distance_blend_rings,
            args.rv_distance_blend_falloff,
        )
        displacement = smooth_displacement(
            triangles,
            displacement,
            moving_mask,
            args.rv_distance_smooth_iterations,
            args.rv_distance_smooth_relaxation,
        )
        moved_points = points + displacement

        vertex_distances, nearest = nearest_distances_and_indices(moved_points, rvepi_points)
        hard_required = np.maximum(target - vertex_distances, 0.0)
        hard_conflict = hard_required > 0.0
        if np.any(hard_conflict):
            directions = normalize_vectors(moved_points[hard_conflict] - rvepi_points[nearest[hard_conflict]], np.asarray(rvepi.center, dtype=float), moved_points[hard_conflict])
            moved_points[hard_conflict] += directions * hard_required[hard_conflict, None]
            moving_mask[hard_conflict] = True

        current = pv.PolyData(moved_points, base.faces_from_triangles(triangles)).extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
        after = rv_to_rvepi_distance_summary(current, rvepi, args)
        step_lengths = np.linalg.norm(moved_points - points, axis=1)
        iterations.append({
            "iteration": int(iteration),
            "sample_reports": sample_reports,
            "source_vertices": int(np.count_nonzero(source_mask)),
            "moved_vertices": int(np.count_nonzero(moving_mask)),
            "hard_projected_vertices": int(np.count_nonzero(hard_conflict)),
            "max_step_mm": float(step_lengths.max()) if len(step_lengths) else 0.0,
            "mean_step_on_moved_vertices_mm": float(step_lengths[moving_mask].mean()) if np.any(moving_mask) else 0.0,
            "after": after,
        })
        if after["passes_rule"]:
            break

    final = rv_to_rvepi_distance_summary(current, rvepi, args)
    return current, {
        "method": "nearest-distance RV inward projection with fixed RVepi",
        "target_distance_mm": float(target),
        "initial": initial,
        "final": final,
        "iterations": iterations,
    }


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


def signed_tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    a = points[cells[:, 0]]
    b = points[cells[:, 1]]
    c = points[cells[:, 2]]
    d = points[cells[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def reorient_tetrahedra(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    out = np.asarray(cells, dtype=np.int64).copy()
    if len(out) == 0:
        return out, {"negative_tets_reoriented": 0, "nonpositive_after_reorientation": 0}
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


def make_grid(points: np.ndarray, cells: np.ndarray, tags: Optional[np.ndarray] = None) -> pv.UnstructuredGrid:
    cells = np.asarray(cells, dtype=np.int64)
    cells_flat = np.column_stack([np.full(len(cells), 4, dtype=np.int64), cells]).ravel()
    celltypes = np.full(len(cells), TETRA_CELL_TYPE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells_flat, celltypes, np.asarray(points, dtype=float))
    if tags is not None:
        grid.cell_data["elemTag"] = np.asarray(tags, dtype=np.int32)
    return grid


def median_tet_edge_length(points: np.ndarray, cells: np.ndarray) -> float:
    if len(cells) == 0:
        return 1.0
    sample = cells if len(cells) <= 25000 else cells[np.linspace(0, len(cells) - 1, 25000).astype(np.int64)]
    values = []
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        values.append(np.linalg.norm(points[sample[:, a]] - points[sample[:, b]], axis=1))
    return float(np.median(np.concatenate(values)))


def tet_quality_report(name: str, points: np.ndarray, cells: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    if len(cells) == 0:
        return {"name": name, "tetrahedra": 0}
    lengths = []
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        lengths.append(np.linalg.norm(points[cells[:, a]] - points[cells[:, b]], axis=1))
    edge_lengths = np.column_stack(lengths)
    max_edges = edge_lengths.max(axis=1)
    volumes = np.abs(signed_tet_volumes(points, cells))
    median_edge = float(np.median(max_edges))
    median_volume = float(np.median(volumes))
    large_edge_threshold = median_edge * float(args.max_edge_length_multiplier)
    large_volume_threshold = median_volume * float(args.max_volume_multiplier)
    return {
        "name": name,
        "tetrahedra": int(len(cells)),
        "nonpositive_signed_tets": int(np.count_nonzero(signed_tet_volumes(points, cells) <= 0.0)),
        "min_abs_volume": float(volumes.min()),
        "median_abs_volume": median_volume,
        "max_abs_volume": float(volumes.max()),
        "median_max_edge_length": median_edge,
        "max_edge_length": float(max_edges.max()),
        "large_edge_threshold": large_edge_threshold,
        "large_volume_threshold": large_volume_threshold,
        "large_edge_tets": int(np.count_nonzero(max_edges > large_edge_threshold)),
        "large_volume_tets": int(np.count_nonzero(volumes > large_volume_threshold)),
        "tetgen_refinement": "base.tetrahedralize_shell uses TetGen quality=True with switches pq1.2/5AaY, including refinement, smoothing, and mesh improvement.",
        "tet_quality": base.tet_quality(points, cells),
    }


def find_seed_inside_shell(shell: pv.PolyData, samples: int = 10000) -> np.ndarray:
    bounds = np.asarray(shell.bounds, dtype=float)
    mins = bounds[[0, 2, 4]]
    maxs = bounds[[1, 3, 5]]
    rng = np.random.default_rng(20260901)
    candidates = rng.uniform(mins, maxs, size=(samples, 3))
    inside = base.enclosed_mask(candidates, shell)
    if np.any(inside):
        return candidates[np.flatnonzero(inside)[0]]
    fallback = np.vstack([np.asarray(shell.center, dtype=float), np.asarray(shell.points, dtype=float)[:: max(1, shell.n_points // 1000)]])
    inside = base.enclosed_mask(fallback, shell)
    if np.any(inside):
        return fallback[np.flatnonzero(inside)[0]]
    raise RuntimeError("Could not find a TetGen seed inside the reconstructed shell.")


def tetrahedralize_checked(name: str, shell: pv.PolyData, tag: int, args: argparse.Namespace) -> TetMesh:
    seed = find_seed_inside_shell(shell)
    points, cells = base.tetrahedralize_shell(shell, seed)
    cells, orientation = reorient_tetrahedra(points, cells[:, :4])
    quality = tet_quality_report(name, points, cells, args)
    quality["orientation"] = orientation
    return TetMesh(name=name, points=points, cells=cells, tag=int(tag), seed=seed, quality=quality)


def raw_tet_face_owners(cells: np.ndarray) -> Dict[Tuple[int, int, int], List[int]]:
    owners: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for cell_id, cell in enumerate(np.asarray(cells, dtype=np.int64)):
        for face in tet_faces(cell):
            owners[face].append(int(cell_id))
    return owners


def boundary_faces(cells: np.ndarray) -> np.ndarray:
    faces = [face for face, owners in raw_tet_face_owners(cells).items() if len(owners) == 1]
    return np.asarray(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)


def tet_face_adjacency(cells: np.ndarray) -> List[List[int]]:
    first_owner: Dict[Tuple[int, int, int], int] = {}
    adjacency = [set() for _ in range(len(cells))]
    for cell_id, cell in enumerate(np.asarray(cells, dtype=np.int64)):
        for face in tet_faces(cell):
            other = first_owner.get(face)
            if other is None:
                first_owner[face] = int(cell_id)
            else:
                adjacency[cell_id].add(other)
                adjacency[other].add(int(cell_id))
    return [sorted(values) for values in adjacency]


def connected_tet_components(mask: np.ndarray, cells: np.ndarray) -> List[np.ndarray]:
    adjacency = tet_face_adjacency(cells)
    remaining = set(int(v) for v in np.flatnonzero(mask))
    components: List[np.ndarray] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            cell_id = stack.pop()
            for nb in adjacency[cell_id]:
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(int(nb))
                    comp.append(int(nb))
        components.append(np.asarray(comp, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def expand_cell_mask(mask: np.ndarray, cells: np.ndarray, rings: int) -> np.ndarray:
    expanded = np.asarray(mask, dtype=bool).copy()
    if rings <= 0 or not np.any(expanded):
        return expanded
    adjacency = tet_face_adjacency(cells)
    frontier = set(int(v) for v in np.flatnonzero(expanded))
    for _ in range(int(rings)):
        next_frontier = set()
        for cell_id in frontier:
            for nb in adjacency[cell_id]:
                if not expanded[nb]:
                    expanded[nb] = True
                    next_frontier.add(int(nb))
        if not next_frontier:
            break
        frontier = next_frontier
    return expanded


def compact_tet_mesh(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(cells) == 0:
        point_map = -np.ones(len(points), dtype=np.int64)
        return np.empty((0, 3), dtype=float), np.empty((0, 4), dtype=np.int64), np.empty(0, dtype=np.int64), point_map
    used = np.unique(cells.ravel())
    point_map = -np.ones(len(points), dtype=np.int64)
    point_map[used] = np.arange(len(used), dtype=np.int64)
    return np.asarray(points, dtype=float)[used], point_map[cells], used, point_map


def remove_rv_tets_inside_lv(lv: TetMesh, rv: TetMesh, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object]]:
    lv_surface = make_grid(lv.points, lv.cells).extract_surface().triangulate().clean(tolerance=1e-7)
    centroids = rv.points[rv.cells].mean(axis=1)
    vertices = rv.points[rv.cells.reshape(-1)].reshape((len(rv.cells), 4, 3))
    centroid_inside = base.enclosed_mask(centroids, lv_surface)
    vertex_inside = base.enclosed_mask(vertices.reshape((-1, 3)), lv_surface).reshape((len(rv.cells), 4))
    vertex_fraction = vertex_inside.mean(axis=1)
    intersect = centroid_inside | (vertex_fraction >= float(args.rv_overlap_vertex_fraction))
    expanded = expand_cell_mask(intersect, rv.cells, args.rv_overlap_removal_rings)
    keep = ~expanded
    components = connected_tet_components(keep, rv.cells)
    final_keep = np.zeros(len(rv.cells), dtype=bool)
    if components:
        final_keep[components[0]] = True
    return final_keep, {
        "rv_cells_before": int(len(rv.cells)),
        "centroid_inside_lv": int(np.count_nonzero(centroid_inside)),
        "vertex_inside_lv": int(np.count_nonzero(vertex_fraction > 0.0)),
        "removed_before_ring_expansion": int(np.count_nonzero(intersect)),
        "removed_after_ring_expansion": int(np.count_nonzero(expanded)),
        "component_count_after_removal": int(len(components)),
        "component_sizes_after_removal": [int(len(component)) for component in components[:20]],
        "rv_cells_after": int(np.count_nonzero(final_keep)),
        "removed_small_component_cells": int(np.count_nonzero(keep) - np.count_nonzero(final_keep)),
    }


def extract_cut_faces_from_removed_cells(cells: np.ndarray, kept: np.ndarray, removed: np.ndarray) -> np.ndarray:
    cut_faces: List[Tuple[int, int, int]] = []
    for face, owners in raw_tet_face_owners(cells).items():
        if len(owners) != 2:
            continue
        a, b = owners
        if bool(kept[a]) == bool(kept[b]):
            continue
        removed_id = b if kept[a] else a
        if bool(removed[removed_id]):
            cut_faces.append(face)
    return np.asarray(cut_faces, dtype=np.int64) if cut_faces else np.empty((0, 3), dtype=np.int64)


def best_anchor_permutation(rv_triangle_points: np.ndarray, lv_face: Sequence[int], lv_points: np.ndarray) -> np.ndarray:
    lv_ids = np.asarray(lv_face, dtype=np.int64)
    best = lv_ids
    best_score = float("inf")
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        candidate = lv_ids[list(order)]
        score = float(np.linalg.norm(rv_triangle_points - lv_points[candidate], axis=1).sum())
        if score < best_score:
            best = candidate
            best_score = score
    return best.astype(np.int64)


def prism_split_tets(rv_ids: Sequence[int], lv_ids: Sequence[int]) -> List[List[int]]:
    r0, r1, r2 = (int(v) for v in rv_ids)
    l0, l1, l2 = (int(v) for v in lv_ids)
    return [[r0, r1, r2, l0], [r1, l1, l2, l0], [r1, r2, l2, l0]]


def select_growth_faces(lv_points: np.ndarray, lv_cells: np.ndarray, rv_points: np.ndarray, rv_cells: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object]]:
    lv_boundary = boundary_faces(lv_cells)
    rv_boundary = boundary_faces(rv_cells)
    if len(rv_boundary) == 0 or len(lv_boundary) == 0:
        return np.empty((0, 3), dtype=np.int64), {"selected_by": "none", "reason": "missing boundary faces"}
    lv_centers = lv_points[lv_boundary].mean(axis=1)
    rv_centers = rv_points[rv_boundary].mean(axis=1)
    distances, _ = nearest_distances_and_indices(rv_centers, lv_centers)
    if args.growth_max_distance is not None:
        limit = float(args.growth_max_distance)
    else:
        limit = median_tet_edge_length(rv_points, rv_cells) * float(args.growth_edge_multiplier)
    selected = np.flatnonzero(distances <= limit)
    selected_by = "distance_limit"
    if len(selected) == 0:
        count = min(max(1, int(args.growth_nearest_face_count)), len(rv_boundary))
        selected = np.argsort(distances)[:count]
        selected_by = "nearest_fallback"
    return rv_boundary[selected], {
        "selected_by": selected_by,
        "distance_limit_mm": float(limit),
        "selected_faces": int(len(selected)),
        "nearest_distance_min": float(distances.min()) if len(distances) else None,
        "nearest_distance_p50": float(np.percentile(distances, 50)) if len(distances) else None,
    }


def build_growth_tets(
    lv_points: np.ndarray,
    lv_cells: np.ndarray,
    rv_points: np.ndarray,
    rv_cells: np.ndarray,
    rv_cut_faces: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    lv_boundary = boundary_faces(lv_cells)
    if len(lv_boundary) == 0:
        raise RuntimeError("LV boundary faces are unavailable for RV growth.")
    selected_by = {"selected_by": "cut_faces", "selected_faces": int(len(rv_cut_faces))}
    source_faces = np.asarray(rv_cut_faces, dtype=np.int64)
    if len(source_faces) == 0:
        source_faces, selected_by = select_growth_faces(lv_points, lv_cells, rv_points, rv_cells, args)
    if len(source_faces) == 0:
        return np.empty((0, 4), dtype=np.int64), {"growth_cells": 0, **selected_by}

    rv_offset = len(lv_points)
    lv_centers = lv_points[lv_boundary].mean(axis=1)
    rv_centers = rv_points[source_faces].mean(axis=1)
    distances, nearest_faces = nearest_distances_and_indices(rv_centers, lv_centers)
    candidates: List[List[int]] = []
    for face, nearest_face in zip(source_faces, nearest_faces):
        rv_ids = np.asarray(face, dtype=np.int64) + rv_offset
        lv_ids = best_anchor_permutation(rv_points[face], lv_boundary[int(nearest_face)], lv_points)
        for tet in prism_split_tets(rv_ids, lv_ids):
            if len(set(tet)) == 4:
                candidates.append(tet)

    unique: List[List[int]] = []
    seen = set()
    for tet in candidates:
        key = tuple(sorted(tet))
        if key not in seen:
            seen.add(key)
            unique.append(tet)
    bridge = np.asarray(unique, dtype=np.int64) if unique else np.empty((0, 4), dtype=np.int64)
    all_points = np.vstack([lv_points, rv_points])
    if len(bridge):
        bridge, orientation = reorient_tetrahedra(all_points, bridge)
        volumes = np.abs(signed_tet_volumes(all_points, bridge))
        keep = volumes > float(args.bridge_min_abs_volume)
        filtered = int(np.count_nonzero(~keep))
        bridge = bridge[keep]
    else:
        orientation = {"negative_tets_reoriented": 0, "nonpositive_after_reorientation": 0}
        filtered = 0
    return bridge, {
        **selected_by,
        "candidate_cells": int(len(candidates)),
        "duplicate_cells_removed": int(len(candidates) - len(unique)),
        "volume_filtered_cells": int(filtered),
        "growth_cells": int(len(bridge)),
        "anchor_distance_min": float(distances.min()) if len(distances) else None,
        "anchor_distance_p50": float(np.percentile(distances, 50)) if len(distances) else None,
        "anchor_distance_p95": float(np.percentile(distances, 95)) if len(distances) else None,
        "orientation": orientation,
    }


def count_shared_tag_faces(cells: np.ndarray, tags: np.ndarray, lv_tag: int, rv_tag: int) -> Dict[str, int]:
    shared = 0
    nonmanifold = 0
    for owners in raw_tet_face_owners(cells).values():
        if len(owners) > 2:
            nonmanifold += 1
        tag_set = {int(tags[cell_id]) for cell_id in owners}
        if int(lv_tag) in tag_set and int(rv_tag) in tag_set:
            shared += 1
    return {"shared_lv_rv_faces": int(shared), "nonmanifold_faces": int(nonmanifold)}


def component_report(cells: np.ndarray, tags: np.ndarray) -> Dict[str, object]:
    components = connected_tet_components(np.ones(len(cells), dtype=bool), cells) if len(cells) else []
    return {
        "component_count": int(len(components)),
        "components": [
            {"cells": int(len(comp)), "elemTag": {str(k): int(v) for k, v in sorted(Counter(map(int, tags[comp])).items())}}
            for comp in components[:20]
        ],
    }


def compact_tagged_mesh(points: np.ndarray, cells: np.ndarray, tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(cells) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 4), dtype=np.int64), np.empty(0, dtype=np.int32)
    used = np.unique(cells.ravel())
    point_map = -np.ones(len(points), dtype=np.int64)
    point_map[used] = np.arange(len(used), dtype=np.int64)
    return np.asarray(points, dtype=float)[used], point_map[cells], np.asarray(tags, dtype=np.int32)


def keep_largest_mixed_component(points: np.ndarray, cells: np.ndarray, tags: np.ndarray, lv_tag: int, rv_tag: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    components = connected_tet_components(np.ones(len(cells), dtype=bool), cells) if len(cells) else []
    selected = None
    for component in components:
        tag_set = {int(tag) for tag in tags[component]}
        if int(lv_tag) in tag_set and int(rv_tag) in tag_set:
            selected = component
            break
    if selected is None or len(components) <= 1:
        return points, cells, tags, {"applied": False, "component_count_before": int(len(components))}
    keep = np.zeros(len(cells), dtype=bool)
    keep[selected] = True
    out_points, out_cells, out_tags = compact_tagged_mesh(points, cells[keep], tags[keep])
    return out_points, out_cells, out_tags, {
        "applied": True,
        "component_count_before": int(len(components)),
        "kept_cells": int(len(selected)),
        "removed_cells": int(len(cells) - len(selected)),
    }


def merge_lv_rv_with_cleanup(lv: TetMesh, rv: TetMesh, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rv_keep, removal_info = remove_rv_tets_inside_lv(lv, rv, args)
    removed = ~rv_keep
    cut_faces = extract_cut_faces_from_removed_cells(rv.cells, rv_keep, removed)
    rv_points_clean, rv_cells_clean, _, rv_point_map = compact_tet_mesh(rv.points, rv.cells[rv_keep])
    rv_cells_clean, rv_orientation = reorient_tetrahedra(rv_points_clean, rv_cells_clean)
    if len(cut_faces):
        valid_cut_faces = cut_faces[np.all(rv_point_map[cut_faces] >= 0, axis=1)]
        mapped_cut_faces = rv_point_map[valid_cut_faces] if len(valid_cut_faces) else np.empty((0, 3), dtype=np.int64)
    else:
        mapped_cut_faces = np.empty((0, 3), dtype=np.int64)

    growth_cells, growth_info = build_growth_tets(lv.points, lv.cells, rv_points_clean, rv_cells_clean, mapped_cut_faces, args)
    merged_points = np.vstack([lv.points, rv_points_clean])
    rv_global = rv_cells_clean + len(lv.points)
    cells = [lv.cells, rv_global]
    tags = [
        np.full(len(lv.cells), int(args.lv_tag), dtype=np.int32),
        np.full(len(rv_global), int(args.rv_tag), dtype=np.int32),
    ]
    if len(growth_cells):
        cells.append(growth_cells)
        tags.append(np.full(len(growth_cells), int(args.rv_tag), dtype=np.int32))
    merged_cells = np.vstack(cells).astype(np.int64)
    merged_tags = np.concatenate(tags).astype(np.int32)
    merged_cells, merged_orientation = reorient_tetrahedra(merged_points, merged_cells)

    report_before = component_report(merged_cells, merged_tags)
    merged_points, merged_cells, merged_tags, component_cleanup = keep_largest_mixed_component(
        merged_points,
        merged_cells,
        merged_tags,
        args.lv_tag,
        args.rv_tag,
    )
    report_after = component_report(merged_cells, merged_tags)
    contact = count_shared_tag_faces(merged_cells, merged_tags, args.lv_tag, args.rv_tag)
    quality = tet_quality_report("merged", merged_points, merged_cells, args)

    debug = {
        "rv_after_lv_overlap_removal": (rv_points_clean, rv_cells_clean, np.full(len(rv_cells_clean), int(args.rv_tag), dtype=np.int32)),
        "rv_growth_cells": (merged_points, growth_cells, np.full(len(growth_cells), int(args.rv_tag), dtype=np.int32)),
    }
    return merged_points, merged_cells, merged_tags, {
        "rv_inside_lv_removal": removal_info,
        "rv_cut_faces_from_removed_area": int(len(cut_faces)),
        "rv_cut_faces_after_compaction_mapping": int(len(mapped_cut_faces)),
        "rv_orientation_after_cleanup": rv_orientation,
        "growth_to_lv": growth_info,
        "merged_orientation": merged_orientation,
        "component_before_cleanup": report_before,
        "component_cleanup": component_cleanup,
        "component_after_cleanup": report_after,
        "lv_rv_contact": contact,
        "quality": quality,
        "counts": {
            "lv_cells": int(len(lv.cells)),
            "rv_cells_after_cleanup": int(len(rv_cells_clean)),
            "growth_cells": int(len(growth_cells)),
            "merged_points": int(len(merged_points)),
            "merged_cells": int(len(merged_cells)),
            "elemTag": {str(k): int(v) for k, v in sorted(Counter(map(int, merged_tags)).items())},
        },
    }, debug


def write_elem_tag_legend(path: Path, lv_tag: int, rv_tag: int) -> None:
    path.write_text("elemTag,region\n" f"{int(lv_tag)},LV myocardium\n" f"{int(rv_tag)},RV myocardium\n")


def write_debug_surface(path: Path, mesh: pv.PolyData) -> str:
    mesh.save(path)
    return str(path.resolve())


def write_debug_tet(path: Path, points: np.ndarray, cells: np.ndarray, tags: np.ndarray) -> str:
    if len(cells) == 0:
        return ""
    p, c, t = compact_tagged_mesh(points, cells, tags)
    write_custom_vtk(path, p, c, t)
    return str(path.resolve())


def shell_summary(shell: RebuiltShell) -> Dict[str, object]:
    return {
        "endo_open_points": int(shell.endo_open.n_points),
        "endo_open_faces": int(shell.endo_open.n_cells),
        "epi_open_points": int(shell.epi_open.n_points),
        "epi_open_faces": int(shell.epi_open.n_cells),
        "cut_surface_count": int(len(shell.cut_surfaces)),
        "loop_pairs": shell.loop_pairs,
        "unmatched_loops": shell.unmatched_loops,
        "free_edges_before_repair": int(shell.free_edges_before_repair),
        "free_edges_after_repair": int(shell.free_edges_after_repair),
        "shell_points": int(shell.shell.n_points),
        "shell_faces": int(shell.shell.n_cells),
    }


def process_frame(args: argparse.Namespace, frame: str) -> Dict[str, object]:
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    vtk_dir = resolve_vtk_dir(subject_dir, args.vtk_dir)
    output_dir = resolve_output_dir(subject_dir, args.output_root)
    input_paths = input_paths_for_frame(vtk_dir, frame)

    surfaces = load_surfaces(vtk_dir, frame)
    frame_args = openbase.namespace_with_overrides(args, {"frame": frame})
    subject_base_side, subject_base_detection = openbase.detect_subject_base_side(vtk_dir, frame_args)
    lv_base_side = args.lv_base_side or subject_base_side
    rv_base_side = args.rv_base_side or subject_base_side
    subject_base_detection["applied_base_side"] = {"lv": lv_base_side, "rv": rv_base_side}

    lv_profile = base.compute_area_profile(
        surfaces["lv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=lv_base_side,
    )
    rv_profile = base.compute_area_profile(
        surfaces["rv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=rv_base_side,
    )
    base.write_area_profile(output_dir / f"{subject_id}_{frame}_LV_z_area_profile.csv", lv_profile)
    base.write_area_profile(output_dir / f"{subject_id}_{frame}_RV_z_area_profile.csv", rv_profile)

    lv_build = openbase.build_ventricle_shell("LV", surfaces["lv_endo"], surfaces["lv_epi"], lv_profile, frame_args)
    rv_initial_build = openbase.build_ventricle_shell("RV", surfaces["rv_endo"], surfaces["rv_epi"], rv_profile, frame_args)

    rv_epi_trimmed, rv_epi_removal = remove_surface_intersection_with_lvepi(
        "RVepi_open", rv_initial_build.epi_open, surfaces["lv_epi"], args
    )
    rv_endo_trimmed, rv_endo_removal = remove_surface_intersection_with_lvepi(
        "RV_open", rv_initial_build.endo_open, surfaces["lv_epi"], args
    )

    rv_shell_after_removal = reconstruct_open_myocardial_shell("RV_after_LVepi_intersection_removal", rv_endo_trimmed, rv_epi_trimmed, args)
    rv_endo_moved, rv_distance_fix = move_rv_to_min_rvepi_distance(rv_endo_trimmed, rv_epi_trimmed, args)
    rv_final_shell = reconstruct_open_myocardial_shell("RV_final_after_3mm_fix", rv_endo_moved, rv_epi_trimmed, args)

    lv_mesh = tetrahedralize_checked("LV", lv_build.shell, args.lv_tag, args)
    rv_mesh = tetrahedralize_checked("RV", rv_final_shell.shell, args.rv_tag, args)
    merged_points, merged_cells, merged_tags, merge_info, debug_tets = merge_lv_rv_with_cleanup(lv_mesh, rv_mesh, args)

    output_mesh = output_dir / f"{subject_id}_{frame}_openbase_LVRV_septum_fine_volmesh.vtk"
    write_custom_vtk(output_mesh, merged_points, merged_cells, merged_tags)
    legend_path = output_dir / f"{subject_id}_{frame}_septum_fine_elemTag_legend.csv"
    write_elem_tag_legend(legend_path, args.lv_tag, args.rv_tag)

    debug_paths: Dict[str, str] = {}
    if args.save_debug:
        debug_dir = output_dir / f"{subject_id}_{frame}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_surfaces = {
            "LV_shell": lv_build.shell,
            "RV_initial_shell": rv_initial_build.shell,
            "RVepi_after_LVepi_removal": rv_epi_trimmed,
            "RV_after_LVepi_removal": rv_endo_trimmed,
            "RV_shell_after_LVepi_removal": rv_shell_after_removal.shell,
            "RV_after_3mm_fix": rv_endo_moved,
            "RV_final_shell": rv_final_shell.shell,
        }
        for name, mesh in debug_surfaces.items():
            debug_paths[name] = write_debug_surface(debug_dir / f"{subject_id}_{frame}_{name}.vtk", mesh)
        for name, (points, cells, tags) in debug_tets.items():
            path = write_debug_tet(debug_dir / f"{subject_id}_{frame}_{name}.vtk", points, cells, tags)
            if path:
                debug_paths[name] = path

    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "vtk_dir": str(vtk_dir.resolve()),
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "outputs": {
            "biventricular_volmesh": str(output_mesh.resolve()),
            "elemTag_legend": str(legend_path.resolve()),
            "debug": debug_paths,
        },
        "pipeline": [
            "load_surfaces",
            "base_side_detection_and_z_area_profile",
            "open_base_LV_shell_generation",
            "open_base_RV_shell_generation",
            "remove_RVepi_open_intersection_with_LVepi",
            "remove_RV_open_intersection_with_LVepi",
            "reconstruct_RV_shell_after_surface_removal",
            "keep_RVepi_fixed_and_move_RV_to_min_3mm_distance",
            "reconstruct_final_RV_shell_after_RV_motion",
            "separate_LV_RV_tetrahedralization_with_TetGen_quality_refinement",
            "merge_LV_RV_and_delete_RV_tets_inside_LV",
            "grow_RV_tets_toward_LV_when_connection_is_needed",
            "assign_elemTag_and_save_final_VTK",
        ],
        "parameters": serializable_args(frame_args),
        "subject_base_detection": subject_base_detection,
        "profiles": {"lv": base.profile_to_dict(lv_profile), "rv": base.profile_to_dict(rv_profile)},
        "lv_open_base": openbase.ventricle_summary(lv_build),
        "rv_initial_open_base": openbase.ventricle_summary(rv_initial_build),
        "surface_intersection_removal": {"RVepi": rv_epi_removal, "RV": rv_endo_removal},
        "rv_shell_after_intersection_removal": shell_summary(rv_shell_after_removal),
        "rv_distance_fix": rv_distance_fix,
        "rv_final_shell": shell_summary(rv_final_shell),
        "tetrahedralization": {
            "LV": {"seed": [float(x) for x in lv_mesh.seed], "quality": lv_mesh.quality},
            "RV": {"seed": [float(x) for x in rv_mesh.seed], "quality": rv_mesh.quality},
        },
        "merge_and_connection": merge_info,
    }
    summary_path = output_dir / f"{subject_id}_{frame}_openbase_LVRV_septum_fine_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "frame": frame,
        "biventricular_volmesh": str(output_mesh),
        "summary": str(summary_path),
        "merged_cells": int(len(merged_cells)),
        "elemTag": {str(k): int(v) for k, v in sorted(Counter(map(int, merged_tags)).items())},
        "component_count": int(merge_info["component_after_cleanup"]["component_count"]),
        "shared_lv_rv_faces": int(merge_info["lv_rv_contact"]["shared_lv_rv_faces"]),
        "nonpositive_signed_tets": int(merge_info["quality"]["nonpositive_signed_tets"]),
        "rv_3mm_passes": bool(rv_distance_fix["final"]["passes_rule"]),
    }


def main() -> None:
    args = parse_args()
    frames = parse_csv_values(args.frames) or [args.frame]
    results = [process_frame(args, frame) for frame in frames]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
