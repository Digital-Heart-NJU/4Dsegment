#!/usr/bin/env python3
"""Generate an open-base biventricular myocardium mesh with one TetGen PLC.

Requested pipeline:
  1. Load LVendo, LVepi, RV, RVepi surfaces.
  2. Detect subject base side and write LV/RV Z-area profiles.
  3. Build the open-base LV myocardial shell.
  4. Build the initial open-base RV myocardial shell.
  5. Remove RVepi-open points intersecting LVepi, then keep only valid triangles.
  6. Remove RV-open points intersecting LVepi, then keep only valid triangles.
  7. Keep RVepi fixed and move RVendo until RV-to-RVepi distance is >= 3 mm.
  8. Merge new RVepi+LVepi as the whole epi surface and merge
     new RVendo+RV open-base annulus+LVendo+LV open-base annulus as
     the whole endo/annulus surface.  TetGen also receives the cut closure
     surfaces required to make a valid myocardial PLC.
  9. Tetrahedralize the whole volume mesh with TetGen.
 10. Write final VTK with elemTag=1 for LV and elemTag=0 for RV.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import tetgen
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


def log_stage(message: str) -> None:
    print(f"[generate_open_volmesh_stable] {message}", flush=True)


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
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to SUBJECT_DIR/volmesh_tetgen.")
    parser.add_argument("--pipeline-output-root", type=Path, default=None, help="Defaults to SUBJECT_DIR/volmesh_generate_pipline/step3open_base.")
    parser.add_argument("--repair-open-shells-before-step5", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-open-shell-stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-intersection-removal-stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rvendo-distance-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-merge-surface-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-open-shells", action="store_true", help="Save LV/RV open-base shell stage outputs and stop before later RV trimming/TetGen stages.")
    parser.add_argument("--stop-after-intersection-removal", action="store_true", help="Save step 5/6 triangle-surface outputs and stop before RV distance fixing/TetGen stages.")
    parser.add_argument("--stop-after-rvendo-distance", action="store_true", help="Save step 7 RVepi/RVendo triangle surfaces and stop before whole-surface merging/TetGen stages.")
    parser.add_argument("--stop-after-merge-surface", action="store_true", help="Save step 8 whole epi/endo triangle surfaces and stop before closure/TetGen stages.")
    parser.add_argument("--save-debug", action="store_true")

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
    parser.add_argument("--surface-removal-rings", type=int, default=0)
    parser.add_argument("--surface-component-min-fraction", type=float, default=0.01)
    parser.add_argument("--surface-clean-tolerance", type=float, default=1e-7)
    parser.add_argument("--combined-surface-merge-tolerance", type=float, default=1e-6)
    parser.add_argument("--boundary-match-tolerance", type=float, default=1e-3)
    parser.add_argument("--rv-min-distance-to-rvepi", type=float, default=3.0)
    parser.add_argument("--rv-distance-padding", type=float, default=0.05)
    parser.add_argument("--rv-distance-iterations", type=int, default=8)
    parser.add_argument("--rv-distance-max-step", type=float, default=2.0)

    parser.add_argument("--tetgen-input-mode", choices=["raw", "repair", "raw_then_repair"], default="raw")
    parser.add_argument("--strict-lv-preservation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lv-preservation-tolerance", type=float, default=1.0)
    parser.add_argument("--tetgen-minratio", type=float, default=1.2)
    parser.add_argument("--tetgen-mindihedral", type=float, default=5.0)
    parser.add_argument("--tetgen-max-volume", type=float, default=0.0)
    parser.add_argument("--tetgen-verbose", action="store_true")
    parser.add_argument("--strict-tag-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-region-cell-fraction", type=float, default=0.03)
    parser.add_argument("--drop-small-component-fraction", type=float, default=0.001)
    parser.add_argument("--drop-small-component-min-cells", type=int, default=8)
    parser.add_argument("--max-edge-length-multiplier", type=float, default=8.0)
    parser.add_argument("--max-volume-multiplier", type=float, default=50.0)
    parser.add_argument("--lv-tag", type=int, default=1)
    parser.add_argument("--rv-tag", type=int, default=0)
    return parser.parse_args()


def resolve_vtk_dir(subject_dir: Path, vtk_dir: Optional[Path]) -> Path:
    if vtk_dir is None:
        return subject_dir / "vtks"
    if vtk_dir.is_absolute():
        return vtk_dir
    subject_relative = subject_dir / vtk_dir
    return subject_relative if subject_relative.exists() else vtk_dir


def resolve_output_dir(subject_dir: Path, output_root: Optional[Path]) -> Path:
    output_dir = output_root or subject_dir / "volmesh_tetgen"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_pipeline_stage_dir(subject_dir: Path, pipeline_output_root: Optional[Path], stage_name: str) -> Path:
    root_dir = pipeline_output_root or subject_dir / "volmesh_generate_pipline"
    if root_dir.name.startswith("step"):
        root_dir = root_dir.parent
    output_dir = root_dir / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_pipeline_output_dir(subject_dir: Path, pipeline_output_root: Optional[Path]) -> Path:
    return resolve_pipeline_stage_dir(subject_dir, pipeline_output_root, "step3open_base")


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
    implicit.SetInput(reference.extract_surface().triangulate().clean(tolerance=1e-7))
    pts = np.asarray(points, dtype=float)
    out = np.empty(len(pts), dtype=float)
    for index, point in enumerate(pts):
        out[index] = float(implicit.EvaluateFunction([float(point[0]), float(point[1]), float(point[2])]))
    return out


def unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = set()
    for tri in np.asarray(triangles, dtype=np.int64):
        a, b, c = (int(v) for v in tri)
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.asarray(sorted(edges), dtype=np.int64)


def triangle_edge_face_map(triangles: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for face_id, tri in enumerate(np.asarray(triangles, dtype=np.int64)):
        a, b, c = (int(v) for v in tri)
        for edge in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
            edge_to_faces[edge].append(int(face_id))
    return edge_to_faces


def face_adjacency(triangles: np.ndarray) -> List[List[int]]:
    edge_to_faces = triangle_edge_face_map(triangles)
    adjacency = [set() for _ in range(len(triangles))]
    for owners in edge_to_faces.values():
        for a in owners:
            for b in owners:
                if a != b:
                    adjacency[a].add(b)
    return [sorted(values) for values in adjacency]


def expand_face_mask(mask: np.ndarray, triangles: np.ndarray, rings: int) -> np.ndarray:
    expanded = np.asarray(mask, dtype=bool).copy()
    if rings <= 0 or not np.any(expanded):
        return expanded
    adjacency = face_adjacency(triangles)
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


def face_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        return np.empty((0,), dtype=float)
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def connected_face_components(mask: np.ndarray, triangles: np.ndarray) -> List[np.ndarray]:
    adjacency = face_adjacency(triangles)
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


def keep_large_surface_components(mesh: pv.PolyData, min_fraction: float) -> Tuple[pv.PolyData, Dict[str, object]]:
    mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    triangles = base.triangle_faces(mesh)
    points = np.asarray(mesh.points, dtype=float)
    if len(triangles) == 0:
        raise RuntimeError("Surface has no triangles after trimming.")
    areas = face_areas(points, triangles)
    components = connected_face_components(np.ones(len(triangles), dtype=bool), triangles)
    if not components:
        raise RuntimeError("Surface has no connected component after trimming.")
    max_area = float(areas[components[0]].sum())
    keep = np.zeros(len(triangles), dtype=bool)
    records = []
    for component in components:
        area = float(areas[component].sum())
        if area >= max_area * float(min_fraction):
            keep[component] = True
            records.append({"faces": int(len(component)), "area": area})
    out = base.subset_mesh_faces(mesh, keep)
    return out.extract_surface().triangulate().clean(tolerance=1e-7), {
        "component_count": int(len(components)),
        "kept_component_count": int(len(records)),
        "kept_components": records[:20],
        "removed_component_faces": int(np.count_nonzero(~keep)),
    }


def cut_boundary_edges_from_face_mask(triangles: np.ndarray, remove: np.ndarray) -> np.ndarray:
    edge_to_faces = triangle_edge_face_map(triangles)
    remove = np.asarray(remove, dtype=bool)
    edges: List[Tuple[int, int]] = []
    for edge, owners in edge_to_faces.items():
        flags = [bool(remove[face_id]) for face_id in owners]
        if any(flags) and not all(flags):
            edges.append(edge)
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def map_source_points_to_target_mask(source_points: np.ndarray, target_points: np.ndarray, tolerance: float) -> np.ndarray:
    mask = np.zeros(len(target_points), dtype=bool)
    if len(source_points) == 0 or len(target_points) == 0:
        return mask
    tree = cKDTree(np.asarray(source_points, dtype=float))
    distances, _ = tree.query(np.asarray(target_points, dtype=float), k=1)
    mask[distances <= float(tolerance)] = True
    return mask


def inside_or_near_reference_surface(reference: pv.PolyData, query_points: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    points = np.asarray(query_points, dtype=float)
    if len(points) == 0:
        empty_bool = np.empty((0,), dtype=bool)
        return empty_bool, np.empty((0,), dtype=float), empty_bool
    inside = base.enclosed_mask(points, reference)
    distances = signed_distances(reference, points)
    near_tol = max(float(args.surface_lvepi_clearance), float(args.surface_clean_tolerance) * 100.0, 1e-5)
    near = np.abs(distances) <= near_tol
    return inside | near, distances, inside


def closest_points_on_surface(reference: pv.PolyData, query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference = reference.extract_surface().triangulate().clean(tolerance=1e-7)
    points = np.asarray(query_points, dtype=float)
    if len(points) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(reference)
    locator.BuildLocator()

    projected = np.empty_like(points, dtype=float)
    distances = np.empty(len(points), dtype=float)
    cell = vtk.vtkGenericCell()
    for index, point in enumerate(points):
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        locator.FindClosestPoint(
            [float(point[0]), float(point[1]), float(point[2])],
            closest,
            cell,
            cell_id,
            sub_id,
            dist2,
        )
        projected[index] = closest
        distances[index] = float(dist2) ** 0.5
    return projected, distances


def clipped_edge_point(
    outside_id: int,
    inside_id: int,
    points: np.ndarray,
    signed_distances: np.ndarray,
) -> np.ndarray:
    outside_distance = float(signed_distances[outside_id])
    inside_distance = float(signed_distances[inside_id])
    if np.isfinite(outside_distance) and np.isfinite(inside_distance) and abs(outside_distance - inside_distance) > 1e-12:
        if outside_distance * inside_distance <= 0.0:
            t = outside_distance / (outside_distance - inside_distance)
        else:
            denominator = abs(outside_distance) + abs(inside_distance)
            t = abs(outside_distance) / denominator if denominator > 1e-12 else 0.5
    else:
        t = 0.5
    t = float(np.clip(t, 0.02, 0.98))
    return points[outside_id] + t * (points[inside_id] - points[outside_id])


def compact_reconstructed_triangles(
    points: np.ndarray,
    triangles: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    triangles_array = np.asarray(triangles, dtype=np.int64)
    if len(triangles_array) == 0:
        raise RuntimeError("No valid triangles remain after clipped reconstruction.")
    used = np.unique(triangles_array.ravel())
    point_map = -np.ones(len(points), dtype=np.int64)
    point_map[used] = np.arange(len(used), dtype=np.int64)
    return np.asarray(points, dtype=float)[used], point_map[triangles_array]


def remove_intersecting_points_from_surface(
    name: str,
    surface: pv.PolyData,
    lv_epi_reference: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, np.ndarray, Dict[str, object]]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_epi_reference = lv_epi_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    triangles = base.triangle_faces(surface)
    points = np.asarray(surface.points, dtype=float)
    clearance = float(args.surface_lvepi_clearance)

    vertex_hit, vertex_distances, vertex_inside = inside_or_near_reference_surface(lv_epi_reference, points, args)
    centers = points[triangles].mean(axis=1) if len(triangles) else np.empty((0, 3), dtype=float)
    center_hit, _, center_inside = inside_or_near_reference_surface(lv_epi_reference, centers, args) if len(centers) else (np.empty((0,), dtype=bool), np.empty((0,), dtype=float), np.empty((0,), dtype=bool))

    output_points: List[np.ndarray] = [point.copy() for point in points]
    output_triangles: List[List[int]] = []
    cut_edge_cache: Dict[Tuple[int, int], int] = {}
    projected_cut_points: List[np.ndarray] = []
    projection_distances: List[float] = []
    fully_kept_faces = 0
    fully_deleted_faces = 0
    mixed_faces = 0
    reconstructed_from_one_bad_vertex = 0
    reconstructed_from_two_bad_vertices = 0
    degenerate_reconstructed_triangles = 0
    all_outside_center_hit_faces = 0

    def output_vertex(vertex_id: int) -> int:
        return int(vertex_id)

    def cut_vertex(outside_id: int, inside_id: int) -> int:
        key = tuple(sorted((int(outside_id), int(inside_id))))
        cached = cut_edge_cache.get(key)
        if cached is not None:
            return cached
        raw_point = clipped_edge_point(int(outside_id), int(inside_id), points, vertex_distances)
        projected_point, projected_distance = closest_points_on_surface(lv_epi_reference, raw_point[None, :])
        new_id = len(output_points)
        output_points.append(projected_point[0])
        projected_cut_points.append(projected_point[0])
        projection_distances.append(float(projected_distance[0]))
        cut_edge_cache[key] = new_id
        return new_id

    def triangle_area(a: int, b: int, c: int) -> float:
        pa, pb, pc = output_points[a], output_points[b], output_points[c]
        return 0.5 * float(np.linalg.norm(np.cross(pb - pa, pc - pa)))

    def add_triangle(a: int, b: int, c: int) -> int:
        if triangle_area(a, b, c) <= 1e-10:
            return 0
        output_triangles.append([int(a), int(b), int(c)])
        return 1

    for face_id, tri in enumerate(triangles):
        flags = vertex_hit[tri]
        bad_count = int(np.count_nonzero(flags))
        if bad_count == 0:
            output_triangles.append([int(tri[0]), int(tri[1]), int(tri[2])])
            fully_kept_faces += 1
            if len(center_hit) and bool(center_hit[face_id]):
                all_outside_center_hit_faces += 1
            continue
        if bad_count == 3:
            fully_deleted_faces += 1
            continue

        mixed_faces += 1
        if bad_count == 1:
            reconstructed_from_one_bad_vertex += 1
        else:
            reconstructed_from_two_bad_vertices += 1

        polygon: List[int] = []
        for local_id in range(3):
            current = int(tri[local_id])
            nxt = int(tri[(local_id + 1) % 3])
            current_keep = not bool(vertex_hit[current])
            next_keep = not bool(vertex_hit[nxt])
            if current_keep and next_keep:
                polygon.append(output_vertex(nxt))
            elif current_keep and not next_keep:
                polygon.append(cut_vertex(current, nxt))
            elif not current_keep and next_keep:
                polygon.append(cut_vertex(nxt, current))
                polygon.append(output_vertex(nxt))

        deduped: List[int] = []
        for vertex_id in polygon:
            if not deduped or deduped[-1] != vertex_id:
                deduped.append(vertex_id)
        if len(deduped) > 2 and deduped[0] == deduped[-1]:
            deduped = deduped[:-1]

        before_count = len(output_triangles)
        if len(deduped) == 3:
            add_triangle(deduped[0], deduped[1], deduped[2])
        elif len(deduped) == 4:
            add_triangle(deduped[0], deduped[1], deduped[2])
            add_triangle(deduped[0], deduped[2], deduped[3])
        if len(output_triangles) == before_count:
            degenerate_reconstructed_triangles += 1

    if not output_triangles:
        raise RuntimeError(f"{name}: clipped reconstruction produced no valid triangles.")

    all_points = np.vstack(output_points)
    compact_points, compact_triangles = compact_reconstructed_triangles(all_points, output_triangles)
    trimmed = pv.PolyData(compact_points, base.faces_from_triangles(compact_triangles))
    trimmed = trimmed.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)

    cut_points = np.asarray(projected_cut_points, dtype=float) if projected_cut_points else np.empty((0, 3), dtype=float)
    cut_mask = map_source_points_to_target_mask(
        cut_points,
        np.asarray(trimmed.points, dtype=float),
        max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0),
    )
    output_point_kept_from_original = map_source_points_to_target_mask(
        points[~vertex_hit],
        np.asarray(trimmed.points, dtype=float),
        max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0),
    )
    return trimmed, cut_mask, {
        "surface": name,
        "method": "clip_mixed_triangles_against_closed_LVepi_and_project_cut_vertices_to_LVepi",
        "input_points": int(surface.n_points),
        "input_faces": int(len(triangles)),
        "clearance_mm": clearance,
        "reference_LVepi_free_edges": int(len(base.extract_boundary_edges(lv_epi_reference))),
        "removed_points_by_LVepi_test": int(np.count_nonzero(vertex_hit)),
        "removed_points_inside_closed_LVepi": int(np.count_nonzero(vertex_inside)),
        "kept_original_outside_points_after_reconstruction": int(np.count_nonzero(output_point_kept_from_original)),
        "additional_outside_points_unreferenced_after_reconstruction": int(np.count_nonzero(~vertex_hit) - np.count_nonzero(output_point_kept_from_original)),
        "face_center_hits_not_used_for_removal": int(np.count_nonzero(center_hit)),
        "face_centers_inside_closed_LVepi_not_used_for_removal": int(np.count_nonzero(center_inside)),
        "all_outside_vertex_faces_with_hit_center_left_unchanged": int(all_outside_center_hit_faces),
        "fully_kept_faces": int(fully_kept_faces),
        "fully_deleted_faces_all_vertices_inside_or_near_LVepi": int(fully_deleted_faces),
        "mixed_faces_reconstructed": int(mixed_faces),
        "mixed_faces_with_one_bad_vertex": int(reconstructed_from_one_bad_vertex),
        "mixed_faces_with_two_bad_vertices": int(reconstructed_from_two_bad_vertices),
        "new_projected_cut_vertices_before_cleanup": int(len(projected_cut_points)),
        "unique_cut_edges": int(len(cut_edge_cache)),
        "cut_boundary_vertices_after_cleanup": int(np.count_nonzero(cut_mask)),
        "degenerate_reconstructed_polygons_skipped": int(degenerate_reconstructed_triangles),
        "reconstructed_triangles_before_cleanup": int(len(output_triangles)),
        "output_points": int(trimmed.n_points),
        "output_faces": int(trimmed.n_cells),
        "min_signed_distance_to_LVepi": float(vertex_distances.min()) if len(vertex_distances) else None,
        "cut_projection_to_LVepi": {
            "method": "signed_distance_edge_crossing_then_vtkCellLocator_projection",
            "raw_to_projected_distance_min": float(np.min(projection_distances)) if projection_distances else None,
            "raw_to_projected_distance_mean": float(np.mean(projection_distances)) if projection_distances else None,
            "raw_to_projected_distance_max": float(np.max(projection_distances)) if projection_distances else None,
        },
        "component_cleanup": {"applied": False, "reason": "clipped_reconstruction_preserves_local_surface_components"},
    }


def enforce_rvendo_distance(
    rv_endo: pv.PolyData,
    rv_epi: pv.PolyData,
    args: argparse.Namespace,
    frozen_mask: Optional[np.ndarray] = None,
) -> Tuple[pv.PolyData, np.ndarray, Dict[str, object]]:
    rv_endo = rv_endo.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    rv_epi = rv_epi.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    points = np.asarray(rv_endo.points, dtype=float).copy()

    if frozen_mask is None:
        frozen = np.zeros(len(points), dtype=bool)
    else:
        frozen = np.asarray(frozen_mask, dtype=bool)
        if len(frozen) != len(points):
            raise ValueError(f"frozen_mask has {len(frozen)} values, expected {len(points)} RVendo points.")
        frozen = frozen.copy()

    rv_epi_points = np.asarray(rv_epi.points, dtype=float)
    rv_epi_tree = cKDTree(rv_epi_points)
    requested_distance = float(args.rv_min_distance_to_rvepi)
    target = requested_distance + float(args.rv_distance_padding)
    total_moved = np.zeros(len(points), dtype=bool)
    iteration_records = []
    initial_distances = rv_epi_tree.query(points, k=1)[0] if len(points) else np.empty((0,), dtype=float)

    for iteration in range(int(args.rv_distance_iterations)):
        distances, ids = rv_epi_tree.query(points, k=1)
        movable = (distances < target) & (distances > 1e-8) & (~frozen)
        if not np.any(movable):
            iteration_records.append({
                "iteration": int(iteration),
                "moved_vertices": 0,
                "frozen_opening_boundary_vertices": int(np.count_nonzero(frozen)),
                "min_distance_before": float(np.min(distances)) if len(distances) else None,
                "p05_distance_before": float(np.percentile(distances, 5)) if len(distances) else None,
            })
            break
        nearest = rv_epi_points[ids[movable]]
        vectors = points[movable] - nearest
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)[:, None]
        steps = np.minimum(target - distances[movable], float(args.rv_distance_max_step))
        points[movable] += unit * steps[:, None]
        total_moved[movable] = True
        iteration_records.append({
            "iteration": int(iteration),
            "moved_vertices": int(np.count_nonzero(movable)),
            "frozen_opening_boundary_vertices": int(np.count_nonzero(frozen)),
            "min_distance_before": float(np.min(distances)) if len(distances) else None,
            "p05_distance_before": float(np.percentile(distances, 5)) if len(distances) else None,
            "max_step_mm": float(np.max(steps)) if len(steps) else 0.0,
        })

    moved = rv_endo.copy(deep=True)
    moved.points = points
    moved = moved.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    moved_mask = map_source_points_to_target_mask(
        points[total_moved],
        np.asarray(moved.points, dtype=float),
        max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0),
    )
    frozen_after_cleanup = map_source_points_to_target_mask(
        points[frozen],
        np.asarray(moved.points, dtype=float),
        max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0),
    )
    final_distances = cKDTree(np.asarray(rv_epi.points, dtype=float)).query(np.asarray(moved.points, dtype=float), k=1)[0]
    below_requested = final_distances < requested_distance
    return moved, moved_mask, {
        "method": "fix_RVepi_and_move_RVendo_to_minimum_distance_freezing_basal_opening_boundary",
        "minimum_requested_distance_mm": requested_distance,
        "target_distance_with_padding_mm": target,
        "rv_epi_fixed": True,
        "rvendo_basal_opening_boundary_frozen": True,
        "frozen_opening_boundary_vertices": int(np.count_nonzero(frozen)),
        "frozen_opening_boundary_vertices_after_cleanup": int(np.count_nonzero(frozen_after_cleanup)),
        "lv_epi_boundary_replacement": False,
        "initial_vertices_below_requested_distance": int(np.count_nonzero(initial_distances < requested_distance)) if len(initial_distances) else 0,
        "initial_nonfrozen_vertices_below_requested_distance": int(np.count_nonzero((initial_distances < requested_distance) & (~frozen))) if len(initial_distances) else 0,
        "initial_frozen_vertices_below_requested_distance": int(np.count_nonzero((initial_distances < requested_distance) & frozen)) if len(initial_distances) else 0,
        "initial_min_distance_all_vertices": float(np.min(initial_distances)) if len(initial_distances) else None,
        "initial_p05_distance_all_vertices": float(np.percentile(initial_distances, 5)) if len(initial_distances) else None,
        "iterations": iteration_records,
        "total_vertices_moved": int(np.count_nonzero(total_moved)),
        "final_vertices_below_requested_distance": int(np.count_nonzero(below_requested)) if len(final_distances) else 0,
        "final_nonfrozen_vertices_below_requested_distance": int(np.count_nonzero(below_requested & (~frozen_after_cleanup))) if len(final_distances) else 0,
        "final_frozen_vertices_below_requested_distance": int(np.count_nonzero(below_requested & frozen_after_cleanup)) if len(final_distances) else 0,
        "final_min_distance_all_vertices": float(np.min(final_distances)) if len(final_distances) else None,
        "final_p05_distance_all_vertices": float(np.percentile(final_distances, 5)) if len(final_distances) else None,
    }


def loop_points(loop: pv.PolyData) -> np.ndarray:
    return base.remove_duplicate_loop_endpoint(np.asarray(loop.points, dtype=float))


def loop_record(loop: pv.PolyData) -> Dict[str, object]:
    pts = loop_points(loop)
    return {
        "points": int(len(pts)),
        "length": float(loop.length),
        "area_best_plane": float(abs(base.signed_area_in_best_plane(pts))),
        "centroid": [float(v) for v in pts.mean(axis=0)] if len(pts) else [0.0, 0.0, 0.0],
    }


def opening_loop_mask_on_surface(surface: pv.PolyData, opening_loop: pv.PolyData, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object]]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    loop_pts = loop_points(opening_loop)
    surface_pts = np.asarray(surface.points, dtype=float)
    tolerance = max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0)
    mask = map_source_points_to_target_mask(loop_pts, surface_pts, tolerance)
    fallback_used = False
    fallback_max_distance = None

    if len(loop_pts) and np.count_nonzero(mask) < max(3, int(0.5 * len(loop_pts))):
        distances, ids = cKDTree(surface_pts).query(loop_pts, k=1)
        fallback_used = True
        fallback_max_distance = float(np.max(distances)) if len(distances) else None
        fallback_tol = max(tolerance, float(np.percentile(distances, 95)) if len(distances) else tolerance)
        mask = np.zeros(len(surface_pts), dtype=bool)
        mask[np.asarray(ids, dtype=np.int64)[distances <= fallback_tol]] = True

    return mask, {
        "method": "match_RVendo_basal_opening_loop_points_to_step7_input_surface",
        "loop_points": int(len(loop_pts)),
        "matched_surface_vertices": int(np.count_nonzero(mask)),
        "match_tolerance_mm": float(tolerance),
        "fallback_nearest_point_matching_used": bool(fallback_used),
        "fallback_max_loop_to_surface_distance_mm": fallback_max_distance,
    }


def significant_boundary_loops(mesh: pv.PolyData, min_fraction: float = 0.02) -> List[pv.PolyData]:
    loops = base.boundary_loops(mesh.extract_surface().triangulate().clean(tolerance=1e-7))
    if not loops:
        return []
    longest = max(float(loop.length) for loop in loops)
    selected = [loop for loop in loops if float(loop.length) >= longest * float(min_fraction)]
    return selected or [loops[0]]


def pair_loops(epi_loops: Sequence[pv.PolyData], endo_loops: Sequence[pv.PolyData]) -> List[Tuple[pv.PolyData, pv.PolyData, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for epi_id, epi_loop in enumerate(epi_loops):
        epi_center = loop_points(epi_loop).mean(axis=0)
        for endo_id, endo_loop in enumerate(endo_loops):
            endo_center = loop_points(endo_loop).mean(axis=0)
            candidates.append((float(np.linalg.norm(epi_center - endo_center)), epi_id, endo_id))
    candidates.sort(key=lambda item: item[0])
    used_epi = set()
    used_endo = set()
    pairs: List[Tuple[pv.PolyData, pv.PolyData, float]] = []
    for distance, epi_id, endo_id in candidates:
        if epi_id in used_epi or endo_id in used_endo:
            continue
        used_epi.add(epi_id)
        used_endo.add(endo_id)
        pairs.append((epi_loops[epi_id], endo_loops[endo_id], distance))
    return pairs


def cap_loop(loop: pv.PolyData) -> pv.PolyData:
    pts = loop_points(loop)
    if len(pts) < 3:
        raise RuntimeError("Cannot cap loop with fewer than three points.")
    center = pts.mean(axis=0, keepdims=True)
    cap_points = np.vstack([pts, center])
    center_id = len(pts)
    triangles = [[i, (i + 1) % len(pts), center_id] for i in range(len(pts))]
    return pv.PolyData(cap_points, base.faces_from_triangles(np.asarray(triangles, dtype=np.int64))).extract_surface().triangulate().clean(tolerance=1e-7)


def build_rv_closure_surfaces(rv_epi: pv.PolyData, rv_endo: pv.PolyData, args: argparse.Namespace) -> Tuple[List[pv.PolyData], Dict[str, object]]:
    epi_loops = significant_boundary_loops(rv_epi, min_fraction=0.02)
    endo_loops = significant_boundary_loops(rv_endo, min_fraction=0.02)
    pairs = pair_loops(epi_loops, endo_loops)
    used_epi = {id(pair[0]) for pair in pairs}
    used_endo = {id(pair[1]) for pair in pairs}
    surfaces: List[pv.PolyData] = []
    records = []
    for pair_id, (epi_loop, endo_loop, distance) in enumerate(pairs):
        cut = cap_utils.smooth_myocardium_cut_surface(
            epi_loop,
            endo_loop,
            sample_count=args.annulus_loop_points,
            radial_layers=args.annulus_radial_layers,
            smooth_iterations=args.annulus_smooth_iterations,
            smooth_relaxation=args.annulus_smooth_relaxation,
        )
        surfaces.append(cut)
        records.append({
            "pair_id": int(pair_id),
            "centroid_distance": float(distance),
            "epi_loop": loop_record(epi_loop),
            "endo_loop": loop_record(endo_loop),
            "cut_surface_points": int(cut.n_points),
            "cut_surface_faces": int(cut.n_cells),
        })
    unmatched_epi = [loop for loop in epi_loops if id(loop) not in used_epi]
    unmatched_endo = [loop for loop in endo_loops if id(loop) not in used_endo]
    for loop in unmatched_epi:
        surfaces.append(cap_loop(loop))
    for loop in unmatched_endo:
        surfaces.append(cap_loop(loop))
    return surfaces, {
        "epi_loop_count": int(len(epi_loops)),
        "endo_loop_count": int(len(endo_loops)),
        "paired_loop_count": int(len(pairs)),
        "unmatched_epi_loop_count": int(len(unmatched_epi)),
        "unmatched_endo_loop_count": int(len(unmatched_endo)),
        "pairs": records,
        "unmatched_epi": [loop_record(loop) for loop in unmatched_epi],
        "unmatched_endo": [loop_record(loop) for loop in unmatched_endo],
    }


def merge_surfaces(parts: Sequence[pv.PolyData], tolerance: float) -> pv.PolyData:
    if not parts:
        raise ValueError("No surfaces supplied for merge.")
    merged = parts[0].extract_surface().triangulate().clean(tolerance=float(tolerance))
    for part in parts[1:]:
        merged = merged.merge(
            part.extract_surface().triangulate().clean(tolerance=float(tolerance)),
            merge_points=True,
            tolerance=float(tolerance),
        )
    return merged.extract_surface().triangulate().clean(tolerance=float(tolerance))


def surface_summary(mesh: pv.PolyData) -> Dict[str, object]:
    mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    return {
        "points": int(mesh.n_points),
        "triangles": int(mesh.n_cells),
        "free_edges": int(len(base.extract_boundary_edges(mesh))),
        "bounds": [float(v) for v in mesh.bounds],
        "surface_area": float(mesh.area),
    }


def midwall_seed(epi: pv.PolyData, endo: pv.PolyData) -> Tuple[np.ndarray, Dict[str, object]]:
    epi_points = np.asarray(epi.points, dtype=float)
    endo_points = np.asarray(endo.points, dtype=float)
    tree = cKDTree(epi_points)
    distances, nearest_epi = tree.query(endo_points, k=1)
    valid = np.flatnonzero(np.isfinite(distances) & (distances > 1e-6))
    if len(valid) == 0:
        raise RuntimeError("Could not create mid-wall material seed from epi/endo points.")
    target = np.percentile(distances[valid], 50)
    endo_id = int(valid[np.argmin(np.abs(distances[valid] - target))])
    epi_id = int(nearest_epi[endo_id])
    seed = 0.5 * (endo_points[endo_id] + epi_points[epi_id])
    return seed, {
        "method": "nearest_epi_endo_midwall_seed",
        "seed": [float(v) for v in seed],
        "endo_point_id": endo_id,
        "epi_point_id": epi_id,
        "local_distance": float(distances[endo_id]),
        "distance_p50": float(np.percentile(distances[valid], 50)),
        "distance_p95": float(np.percentile(distances[valid], 95)),
    }


def signed_tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    if len(cells) == 0:
        return np.empty((0,), dtype=float)
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


def tet_faces(cell: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    a, b, c, d = (int(v) for v in cell)
    return (
        tuple(sorted((a, b, c))),
        tuple(sorted((a, b, d))),
        tuple(sorted((a, c, d))),
        tuple(sorted((b, c, d))),
    )


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


def compact_tet_mesh(points: np.ndarray, cells: np.ndarray, tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(cells) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 4), dtype=np.int64), np.empty((0,), dtype=np.int32)
    used = np.unique(cells.ravel())
    point_map = -np.ones(len(points), dtype=np.int64)
    point_map[used] = np.arange(len(used), dtype=np.int64)
    return np.asarray(points, dtype=float)[used], point_map[cells], np.asarray(tags, dtype=np.int32)


def drop_small_tet_components(points: np.ndarray, cells: np.ndarray, tags: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if len(cells) == 0:
        return points, cells, tags, {"applied": False, "reason": "empty"}
    components = connected_tet_components(np.ones(len(cells), dtype=bool), cells)
    if len(components) <= 1:
        return points, cells, tags, {"applied": False, "component_count_before": int(len(components)), "dropped_cells": 0}
    largest = int(len(components[0]))
    threshold = max(int(args.drop_small_component_min_cells), int(np.ceil(largest * float(args.drop_small_component_fraction))))
    keep = np.zeros(len(cells), dtype=bool)
    kept = []
    dropped = []
    for component_id, comp in enumerate(components):
        record = {"component_id": int(component_id), "cells": int(len(comp)), "elemTag": {str(k): int(v) for k, v in sorted(Counter(map(int, tags[comp])).items())}}
        if len(comp) >= threshold:
            keep[comp] = True
            kept.append(record)
        else:
            dropped.append(record)
    if np.all(keep):
        return points, cells, tags, {"applied": False, "component_count_before": int(len(components)), "threshold_cells": int(threshold), "dropped_cells": 0, "kept_components": kept}
    new_points, new_cells, new_tags = compact_tet_mesh(points, cells[keep], tags[keep])
    return new_points, new_cells, new_tags, {
        "applied": True,
        "component_count_before": int(len(components)),
        "threshold_cells": int(threshold),
        "dropped_cells": int(np.count_nonzero(~keep)),
        "kept_components": kept,
        "dropped_components": dropped,
    }


def component_report(cells: np.ndarray, tags: np.ndarray) -> Dict[str, object]:
    components = connected_tet_components(np.ones(len(cells), dtype=bool), cells) if len(cells) else []
    return {
        "component_count": int(len(components)),
        "components": [
            {"cells": int(len(comp)), "elemTag": {str(k): int(v) for k, v in sorted(Counter(map(int, tags[comp])).items())}}
            for comp in components[:20]
        ],
    }


def tet_quality_report(name: str, points: np.ndarray, cells: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    if len(cells) == 0:
        return {"name": name, "tetrahedra": 0}
    p = points[cells]
    lengths = []
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        lengths.append(np.linalg.norm(p[:, a] - p[:, b], axis=1))
    edge_lengths = np.column_stack(lengths)
    max_edges = edge_lengths.max(axis=1)
    volumes_signed = signed_tet_volumes(points, cells)
    volumes = np.abs(volumes_signed)
    median_edge = float(np.median(max_edges))
    median_volume = float(np.median(volumes))
    return {
        "name": name,
        "tetrahedra": int(len(cells)),
        "nonpositive_signed_tets": int(np.count_nonzero(volumes_signed <= 0.0)),
        "min_abs_volume": float(volumes.min()),
        "median_abs_volume": median_volume,
        "max_abs_volume": float(volumes.max()),
        "median_max_edge_length": median_edge,
        "max_edge_length": float(max_edges.max()),
        "large_edge_tets": int(np.count_nonzero(max_edges > median_edge * float(args.max_edge_length_multiplier))),
        "large_volume_tets": int(np.count_nonzero(volumes > median_volume * float(args.max_volume_multiplier))),
        "tet_quality": base.tet_quality(points, cells),
    }


def classify_by_lv_shell(points: np.ndarray, cells: np.ndarray, lv_shell: pv.PolyData, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object]]:
    centroids = points[cells].mean(axis=1)
    inside_lv = base.enclosed_mask(centroids, lv_shell)
    tags = np.full(len(cells), int(args.rv_tag), dtype=np.int32)
    tags[inside_lv] = int(args.lv_tag)
    counts = Counter(map(int, tags))
    total = int(len(tags))
    min_cells = int(np.ceil(total * float(args.min_region_cell_fraction))) if total else 0
    lv_count = int(counts.get(int(args.lv_tag), 0))
    rv_count = int(counts.get(int(args.rv_tag), 0))
    passes = bool(total > 0 and lv_count >= min_cells and rv_count >= min_cells)
    info = {
        "method": "tet_centroid_inside_LV_open_base_shell",
        "total_tetrahedra": total,
        "inside_lv_count": int(np.count_nonzero(inside_lv)),
        "elemTag_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "min_region_cell_fraction": float(args.min_region_cell_fraction),
        "passes_min_region_fraction": passes,
    }
    if args.strict_tag_validation and not passes:
        raise RuntimeError(f"Tag validation failed: LV cells={lv_count}, RV cells={rv_count}, total={total}.")
    return tags, info


def lv_surface_preservation_report(candidate: pv.PolyData, lv_reference: pv.PolyData, tolerance: float) -> Dict[str, object]:
    candidate = candidate.extract_surface().triangulate().clean(tolerance=1e-7)
    lv_reference = lv_reference.extract_surface().triangulate().clean(tolerance=1e-7)
    candidate_bounds = np.asarray(candidate.bounds, dtype=float)
    reference_bounds = np.asarray(lv_reference.bounds, dtype=float)
    axis_names = ("x", "y", "z")
    bound_violations = []
    for axis, axis_name in enumerate(axis_names):
        low = axis * 2
        high = low + 1
        if candidate_bounds[low] > reference_bounds[low] + tolerance:
            bound_violations.append({
                "axis": axis_name,
                "side": "min",
                "candidate": float(candidate_bounds[low]),
                "reference": float(reference_bounds[low]),
            })
        if candidate_bounds[high] < reference_bounds[high] - tolerance:
            bound_violations.append({
                "axis": axis_name,
                "side": "max",
                "candidate": float(candidate_bounds[high]),
                "reference": float(reference_bounds[high]),
            })

    if candidate.n_points and lv_reference.n_points:
        distances = cKDTree(np.asarray(candidate.points, dtype=float)).query(np.asarray(lv_reference.points, dtype=float), k=1)[0]
        p99_distance = float(np.percentile(distances, 99))
        max_distance = float(np.max(distances))
    else:
        p99_distance = None
        max_distance = None

    passes = (
        len(bound_violations) == 0
        and p99_distance is not None
        and p99_distance <= float(tolerance)
    )
    return {
        "method": "candidate_surface_must_preserve_LV_reference_bounds_and_points",
        "tolerance_mm": float(tolerance),
        "passes": bool(passes),
        "candidate_bounds": [float(v) for v in candidate_bounds],
        "reference_bounds": [float(v) for v in reference_bounds],
        "bound_violations": bound_violations,
        "lv_reference_to_candidate_distance_p99": p99_distance,
        "lv_reference_to_candidate_distance_max": max_distance,
    }


def run_tetgen_whole_surface(
    surface: pv.PolyData,
    seeds: Sequence[Tuple[int, np.ndarray]],
    args: argparse.Namespace,
    lv_shell_reference: Optional[pv.PolyData] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, object], pv.PolyData]:
    raw = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    candidate_names = []
    if args.tetgen_input_mode in {"raw", "raw_then_repair"}:
        candidate_names.append("raw_whole_epi_endo_plc")
    if args.tetgen_input_mode in {"repair", "raw_then_repair"}:
        candidate_names.append("repaired_whole_epi_endo_plc")

    attempts = []
    last_error: Optional[BaseException] = None
    for name in candidate_names:
        candidate = raw
        if name.startswith("repaired"):
            log_stage("repair whole TetGen PLC candidate")
            candidate = base.repair_closed_surface(raw).extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
        preservation = None
        try:
            if bool(args.strict_lv_preservation) and lv_shell_reference is not None:
                preservation = lv_surface_preservation_report(candidate, lv_shell_reference, args.lv_preservation_tolerance)
                if not preservation["passes"]:
                    raise RuntimeError("TetGen input rejected because repaired/raw PLC does not preserve the LV open-base shell: " + json.dumps(preservation, indent=2))
            log_stage(f"TetGen on {name}: {candidate.n_points} points, {candidate.n_cells} faces, {len(base.extract_boundary_edges(candidate))} free edges")
            tg = tetgen.TetGen(candidate)
            for region_id, seed in seeds:
                tg.add_region(int(region_id), np.asarray(seed, dtype=float).tolist(), float(args.tetgen_max_volume))
            switches = f"pq{float(args.tetgen_minratio):g}/{float(args.tetgen_mindihedral):g}Y"
            if seeds:
                switches += "A"
            result = tg.tetrahedralize(
                plc=True,
                quality=True,
                minratio=float(args.tetgen_minratio),
                mindihedral=float(args.tetgen_mindihedral),
                nobisect=False,
                regionattrib=bool(seeds),
                switches=switches,
                verbose=1 if args.tetgen_verbose else 0,
            )
            points = np.asarray(result[0], dtype=float)
            cells = np.asarray(result[1], dtype=np.int64)[:, :4]
            labels = None
            if len(result) >= 3 and len(np.asarray(result[2]).reshape(-1)) == len(cells):
                labels = np.rint(np.asarray(result[2]).reshape(-1)).astype(np.int32)
            attempts.append({
                "name": name,
                "status": "success",
                "switches": switches,
                "input_surface_points": int(candidate.n_points),
                "input_surface_faces": int(candidate.n_cells),
                "input_surface_free_edges": int(len(base.extract_boundary_edges(candidate))),
                "lv_preservation": preservation,
                "raw_tetrahedra": int(len(cells)),
                "region_label_counts": {str(k): int(v) for k, v in sorted(Counter(map(int, labels)).items())} if labels is not None else None,
            })
            return points, cells, labels, {"selected_input": name, "attempts": attempts}, candidate
        except Exception as exc:
            last_error = exc
            attempts.append({
                "name": name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "input_surface_points": int(candidate.n_points),
                "input_surface_faces": int(candidate.n_cells),
                "input_surface_free_edges": int(len(base.extract_boundary_edges(candidate))),
                "lv_preservation": preservation,
            })
    raise RuntimeError("TetGen failed for every whole-PLC candidate: " + json.dumps(attempts, indent=2)) from last_error


def write_elem_tag_legend(path: Path, lv_tag: int, rv_tag: int) -> None:
    path.write_text("elemTag,region\n" f"{int(lv_tag)},LV myocardium\n" f"{int(rv_tag)},RV myocardium\n")


def build_ventricle_shell_for_pre_step5(
    name: str,
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile,
    args: argparse.Namespace,
):
    if args.repair_open_shells_before_step5:
        return openbase.build_ventricle_shell(name, endo, epi, profile, args)

    original_repair = base.repair_closed_surface

    def passthrough_closed_surface(mesh: pv.PolyData) -> pv.PolyData:
        return mesh.extract_surface().triangulate().clean(tolerance=1e-7)

    base.repair_closed_surface = passthrough_closed_surface
    try:
        return openbase.build_ventricle_shell(name, endo, epi, profile, args)
    finally:
        base.repair_closed_surface = original_repair


def save_debug_surfaces(debug_dir: Path, subject_id: str, frame: str, surfaces: Dict[str, pv.PolyData]) -> Dict[str, str]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, mesh in surfaces.items():
        path = debug_dir / f"{subject_id}_{frame}_{name}.vtk"
        mesh.save(path)
        paths[name] = str(path.resolve())
    return paths


def save_open_shell_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    lv_build,
    rv_build,
    lv_profile,
    rv_profile,
    lv_base_side: str,
    rv_base_side: str,
    repair_open_shells_before_step5: bool,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    lv_openbase_myo_shell = base.merge_surfaces([lv_build.epi_open, lv_build.endo_open, lv_build.annulus])
    rv_openbase_myo_shell = base.merge_surfaces([rv_build.epi_open, rv_build.endo_open, rv_build.annulus])
    stage_surfaces = {
        "stage03_LV_openbase_myo_shell": lv_openbase_myo_shell,
        "stage03_LV_openbase_epi": lv_build.epi_open,
        "stage03_LV_openbase_endo": lv_build.endo_open,
        "stage03_LV_openbase_annulus": lv_build.annulus,
        "stage04_RV_initial_openbase_myo_shell": rv_openbase_myo_shell,
        "stage04_RV_initial_openbase_epi": rv_build.epi_open,
        "stage04_RV_initial_openbase_endo": rv_build.endo_open,
        "stage04_RV_initial_openbase_annulus": rv_build.annulus,
    }
    paths: Dict[str, str] = {}
    surface_records: Dict[str, object] = {}
    for name, mesh in stage_surfaces.items():
        clean_mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
        path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        clean_mesh.save(path)
        paths[name] = str(path.resolve())
        surface_records[name] = surface_summary(clean_mesh)

    summary = {
        "description": "Open-base LV/RV myocardial shell outputs saved immediately after shell construction, before any LVepi intersection removal or closed-shell repair changes.",
        "myo_shell_saved_from": "raw_merge_of_epi_open_plus_endo_open_plus_annulus_before_repair_closed_surface",
        "pre_step5_repair_closed_surface_enabled": bool(repair_open_shells_before_step5),
        "helper_repaired_shell_not_used_for_saved_myo_shells": True,
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "base_side": {"LV": lv_base_side, "RV": rv_base_side},
        "paths": paths,
        "surfaces": surface_records,
        "profiles": {
            "LV": base.profile_to_dict(lv_profile),
            "RV": base.profile_to_dict(rv_profile),
        },
        "open_base": {
            "LV": openbase.ventricle_summary(lv_build),
            "RV": openbase.ventricle_summary(rv_build),
        },
        "raw_openbase_shell_free_edges_before_helper_repair": {
            "LV": int(lv_build.shell_free_edges_before_repair),
            "RV": int(rv_build.shell_free_edges_before_repair),
        },
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage03_04_openbase_shell_outputs.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage03_04_summary"] = str(summary_path.resolve())
    return paths, summary


def save_triangle_surface_vtk(
    path: Path,
    mesh: pv.PolyData,
    cut_mask: Optional[np.ndarray] = None,
    point_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    surface = mesh.extract_surface().triangulate()
    cut_boundary_vertices = None
    if cut_mask is not None:
        cut_mask = np.asarray(cut_mask, dtype=bool)
        if len(cut_mask) == surface.n_points:
            surface.point_data["cutBoundary"] = cut_mask.astype(np.int32)
            cut_boundary_vertices = int(np.count_nonzero(cut_mask))
    if point_data:
        for name, values in point_data.items():
            array = np.asarray(values)
            if len(array) != surface.n_points:
                raise ValueError(f"Point-data array {name!r} has length {len(array)}, expected {surface.n_points}.")
            surface.point_data[name] = array
    surface.save(path)
    record = surface_summary(surface)
    record.update({
        "path": str(path.resolve()),
        "point_data": list(surface.point_data.keys()),
        "cut_boundary_vertices": cut_boundary_vertices,
    })
    return record


def save_intersection_removal_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    rv_epi_trimmed: pv.PolyData,
    rv_epi_cut_mask: np.ndarray,
    rv_epi_removal: Dict[str, object],
    rv_endo_trimmed: pv.PolyData,
    rv_endo_cut_mask: np.ndarray,
    rv_endo_removal: Dict[str, object],
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}

    stage_items = [
        ("stage05_RVepi_after_LVepi_intersection_removal", rv_epi_trimmed, rv_epi_cut_mask, rv_epi_removal),
        ("stage06_RVendo_after_LVepi_intersection_removal", rv_endo_trimmed, rv_endo_cut_mask, rv_endo_removal),
    ]
    for name, mesh, cut_mask, removal_info in stage_items:
        surface_path = pipeline_dir / f"{subject_id}_{frame}_{name}_surface.vtk"
        records[name] = save_triangle_surface_vtk(surface_path, mesh, cut_mask)
        paths[name] = str(surface_path.resolve())
        records[name]["removal_info"] = removal_info

    summary = {
        "description": "Step 5/6 outputs saved as triangle surface VTKs after LVepi intersection removal.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage05_06_remove_intersection_surfaces_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage05_06_summary"] = str(summary_path.resolve())
    return paths, summary


def save_rvendo_distance_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    rv_epi: pv.PolyData,
    rv_endo: pv.PolyData,
    rv_endo_moved_mask: np.ndarray,
    rv_distance_fix: Dict[str, object],
    args: argparse.Namespace,
    rv_endo_frozen_mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}

    rv_epi_surface = rv_epi.extract_surface().triangulate()
    rv_endo_surface = rv_endo.extract_surface().triangulate()
    rv_epi_path = pipeline_dir / f"{subject_id}_{frame}_stage07_RVepi_fixed_surface.vtk"
    rv_endo_path = pipeline_dir / f"{subject_id}_{frame}_stage07_RVendo_after_3mm_distance_surface.vtk"

    records["stage07_RVepi_fixed"] = save_triangle_surface_vtk(rv_epi_path, rv_epi_surface)
    paths["stage07_RVepi_fixed"] = str(rv_epi_path.resolve())

    distances = cKDTree(np.asarray(rv_epi_surface.points, dtype=float)).query(np.asarray(rv_endo_surface.points, dtype=float), k=1)[0]
    moved_mask = np.asarray(rv_endo_moved_mask, dtype=bool)
    point_data = {
        "movedByStep7": moved_mask.astype(np.int32),
        "distanceToRVepi": distances.astype(np.float32),
        "belowRequested3mm": (distances < float(args.rv_min_distance_to_rvepi)).astype(np.int32),
    }
    if rv_endo_frozen_mask is not None:
        frozen_mask = np.asarray(rv_endo_frozen_mask, dtype=bool)
        if len(frozen_mask) == rv_endo_surface.n_points:
            point_data["frozenOpeningBoundary"] = frozen_mask.astype(np.int32)
    records["stage07_RVendo_after_3mm_distance"] = save_triangle_surface_vtk(
        rv_endo_path,
        rv_endo_surface,
        point_data=point_data,
    )
    paths["stage07_RVendo_after_3mm_distance"] = str(rv_endo_path.resolve())

    summary = {
        "description": "Step 7 outputs after keeping RVepi fixed and moving RVendo by the >=3 mm rule while freezing only the basal RVendo opening boundary.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "rv_distance_fix": rv_distance_fix,
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage07_move_RVendo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage07_summary"] = str(summary_path.resolve())
    return paths, summary


def save_merge_surface_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    whole_epi: pv.PolyData,
    whole_endo: pv.PolyData,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}

    stage_items = [
        ("stage08_whole_epi_surface", whole_epi),
        ("stage08_whole_endo_surface", whole_endo),
    ]
    for name, mesh in stage_items:
        surface_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = save_triangle_surface_vtk(surface_path, mesh)
        paths[name] = str(surface_path.resolve())

    summary = {
        "description": "Step 8 outputs after merging LVepi+RVepi as whole epi and RVendo+RV open-base annulus+LVendo+LV open-base annulus as whole endo/annulus surface.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage08_merge_surface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage08_summary"] = str(summary_path.resolve())
    return paths, summary


def process_frame(args: argparse.Namespace, frame: str) -> Dict[str, object]:
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    vtk_dir = resolve_vtk_dir(subject_dir, args.vtk_dir)
    output_dir = resolve_output_dir(subject_dir, args.output_root)
    open_shell_stage_dir = resolve_pipeline_output_dir(subject_dir, args.pipeline_output_root)
    intersection_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step5remove_intersection")
    rvendo_distance_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step7moveRVendo")
    merge_surface_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step8merge_surface")
    input_paths = input_paths_for_frame(vtk_dir, frame)

    log_stage(f"{subject_id} {frame}: load LV/RV surfaces")
    surfaces = load_surfaces(vtk_dir, frame)

    frame_args = openbase.namespace_with_overrides(args, {"frame": frame})
    log_stage(f"{subject_id} {frame}: detect subject base side")
    subject_base_side, subject_base_detection = openbase.detect_subject_base_side(vtk_dir, frame_args)
    lv_base_side = args.lv_base_side or subject_base_side
    rv_base_side = args.rv_base_side or subject_base_side
    subject_base_detection["applied_base_side"] = {"lv": lv_base_side, "rv": rv_base_side}

    log_stage(f"{subject_id} {frame}: write LV/RV Z-area profiles")
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

    log_stage(f"{subject_id} {frame}: build open-base LV shell")
    lv_build = build_ventricle_shell_for_pre_step5("LV", surfaces["lv_endo"], surfaces["lv_epi"], lv_profile, frame_args)
    log_stage(f"{subject_id} {frame}: build initial open-base RV shell")
    rv_build = build_ventricle_shell_for_pre_step5("RV", surfaces["rv_endo"], surfaces["rv_epi"], rv_profile, frame_args)

    open_shell_stage_paths: Dict[str, str] = {}
    open_shell_stage_summary: Dict[str, object] = {}
    if args.save_open_shell_stages:
        log_stage(f"{subject_id} {frame}: save open-base LV/RV shell stage outputs")
        open_shell_stage_paths, open_shell_stage_summary = save_open_shell_stage_outputs(
            open_shell_stage_dir,
            subject_id,
            frame,
            lv_build,
            rv_build,
            lv_profile,
            rv_profile,
            lv_base_side,
            rv_base_side,
            bool(args.repair_open_shells_before_step5),
        )

    if args.stop_after_open_shells:
        return {
            "frame": frame,
            "stopped_after": "open_base_LV_RV_shell_generation",
            "stage_outputs": open_shell_stage_paths,
        }

    lv_shell = lv_build.shell.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_epi_open = lv_build.epi_open.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_epi_intersection_reference = surfaces["lv_epi"].extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)

    log_stage(f"{subject_id} {frame}: remove RVepi points intersecting LVepi")
    rv_epi_trimmed, rv_epi_cut_mask, rv_epi_removal = remove_intersecting_points_from_surface(
        "RVepi_open", rv_build.epi_open, lv_epi_intersection_reference, args
    )
    log_stage(f"{subject_id} {frame}: remove RVendo points intersecting LVepi")
    rv_endo_trimmed, rv_endo_cut_mask, rv_endo_removal = remove_intersecting_points_from_surface(
        "RV_open", rv_build.endo_open, lv_epi_intersection_reference, args
    )

    intersection_stage_paths: Dict[str, str] = {}
    intersection_stage_summary: Dict[str, object] = {}
    if args.save_intersection_removal_stages:
        log_stage(f"{subject_id} {frame}: save step 5/6 intersection-removal triangle-surface outputs")
        intersection_stage_paths, intersection_stage_summary = save_intersection_removal_stage_outputs(
            intersection_stage_dir,
            subject_id,
            frame,
            rv_epi_trimmed,
            rv_epi_cut_mask,
            rv_epi_removal,
            rv_endo_trimmed,
            rv_endo_cut_mask,
            rv_endo_removal,
        )

    if args.stop_after_intersection_removal:
        return {
            "frame": frame,
            "stopped_after": "step05_06_intersection_removal",
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
        }

    log_stage(f"{subject_id} {frame}: keep RVepi fixed and move RVendo to >=3 mm distance while freezing basal opening boundary")
    rv_endo_opening_mask, rv_endo_opening_match = opening_loop_mask_on_surface(
        rv_endo_trimmed,
        rv_build.endo_opening_loop,
        args,
    )
    rv_endo_fixed, rv_endo_moved_mask, rv_distance_fix = enforce_rvendo_distance(
        rv_endo_trimmed,
        rv_epi_trimmed,
        args,
        frozen_mask=rv_endo_opening_mask,
    )
    rv_distance_fix["opening_boundary_freeze"] = rv_endo_opening_match

    rvendo_distance_stage_paths: Dict[str, str] = {}
    rvendo_distance_stage_summary: Dict[str, object] = {}
    if args.save_rvendo_distance_stage:
        log_stage(f"{subject_id} {frame}: save step 7 RVepi/RVendo triangle-surface outputs")
        rvendo_distance_stage_paths, rvendo_distance_stage_summary = save_rvendo_distance_stage_outputs(
            rvendo_distance_stage_dir,
            subject_id,
            frame,
            rv_epi_trimmed,
            rv_endo_fixed,
            rv_endo_moved_mask,
            rv_distance_fix,
            args,
            rv_endo_opening_mask,
        )

    if args.stop_after_rvendo_distance:
        return {
            "frame": frame,
            "stopped_after": "step07_move_RVendo",
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
            "rvendo_distance_stage_outputs": rvendo_distance_stage_paths,
        }

    log_stage(f"{subject_id} {frame}: merge whole epi and whole endo surfaces")
    whole_epi = merge_surfaces([lv_epi_open, rv_epi_trimmed], args.combined_surface_merge_tolerance)
    whole_endo = merge_surfaces(
        [rv_endo_fixed, rv_build.annulus, lv_build.endo_open, lv_build.annulus],
        args.combined_surface_merge_tolerance,
    )

    merge_surface_stage_paths: Dict[str, str] = {}
    merge_surface_stage_summary: Dict[str, object] = {}
    if args.save_merge_surface_stage:
        log_stage(f"{subject_id} {frame}: save step 8 whole epi/endo triangle-surface outputs")
        merge_surface_stage_paths, merge_surface_stage_summary = save_merge_surface_stage_outputs(
            merge_surface_stage_dir, subject_id, frame, whole_epi, whole_endo
        )

    if args.stop_after_merge_surface:
        return {
            "frame": frame,
            "stopped_after": "step08_merge_surface",
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
            "rvendo_distance_stage_outputs": rvendo_distance_stage_paths,
            "merge_surface_stage_outputs": merge_surface_stage_paths,
        }

    rv_cut_surfaces, rv_cut_info = build_rv_closure_surfaces(rv_epi_trimmed, rv_endo_fixed, args)
    overall_surface = merge_surfaces(
        [whole_epi, whole_endo, lv_build.annulus, *rv_cut_surfaces],
        args.combined_surface_merge_tolerance,
    )

    lv_seed, lv_seed_info = midwall_seed(lv_build.epi_open, lv_build.endo_open)
    rv_seed, rv_seed_info = midwall_seed(rv_epi_trimmed, rv_endo_fixed)

    debug_paths: Dict[str, str] = {}
    if args.save_debug:
        debug_surfaces = {
            "LV_shell_reference": lv_shell,
            "LVepi_open": lv_epi_open,
            "LVepi_closed_intersection_reference": lv_epi_intersection_reference,
            "LVendo_open": lv_build.endo_open,
            "RV_initial_shell": rv_build.shell,
            "RVepi_after_LVepi_intersection_removal": rv_epi_trimmed,
            "RVendo_after_LVepi_intersection_removal": rv_endo_trimmed,
            "RVendo_after_3mm_fix": rv_endo_fixed,
            "whole_epi_surface": whole_epi,
            "whole_endo_surface": whole_endo,
            "overall_TetGen_PLC_before_tetgen": overall_surface,
        }
        for index, cut in enumerate(rv_cut_surfaces):
            debug_surfaces[f"RV_closure_cut_surface_{index:02d}"] = cut
        debug_paths = save_debug_surfaces(output_dir / f"{subject_id}_{frame}_debug", subject_id, frame, debug_surfaces)

    log_stage(f"{subject_id} {frame}: TetGen whole volume")
    tet_points, tet_cells, region_labels, tetgen_info, tetgen_surface = run_tetgen_whole_surface(
        overall_surface,
        [(1, lv_seed), (2, rv_seed)],
        args,
        lv_shell_reference=lv_shell,
    )

    log_stage(f"{subject_id} {frame}: assign elemTag and write VTK")
    tet_cells, orientation = reorient_tetrahedra(tet_points, tet_cells)
    elem_tags, tag_info = classify_by_lv_shell(tet_points, tet_cells, lv_shell, args)
    tet_points, tet_cells, elem_tags, component_filter = drop_small_tet_components(tet_points, tet_cells, elem_tags, args)
    quality = tet_quality_report("whole_biventricular_tetgen", tet_points, tet_cells, args)
    quality["orientation"] = orientation
    components = component_report(tet_cells, elem_tags)

    output_mesh = output_dir / f"{subject_id}_{frame}_openbase_LVRV_tetgen_volmesh.vtk"
    write_custom_vtk(output_mesh, tet_points, tet_cells, elem_tags)
    legend_path = output_dir / f"{subject_id}_{frame}_tetgen_elemTag_legend.csv"
    write_elem_tag_legend(legend_path, args.lv_tag, args.rv_tag)

    region_label_counts = None
    if region_labels is not None:
        region_label_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, region_labels)).items())}

    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "vtk_dir": str(vtk_dir.resolve()),
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "outputs": {
            "biventricular_volmesh": str(output_mesh.resolve()),
            "elemTag_legend": str(legend_path.resolve()),
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
            "rvendo_distance_stage_outputs": rvendo_distance_stage_paths,
            "merge_surface_stage_outputs": merge_surface_stage_paths,
            "debug": debug_paths,
        },
        "pipeline": [
            "load_LVendo_LVepi_RV_RVepi_surfaces",
            "detect_subject_base_side_and_write_LV_RV_z_area_profiles",
            "build_open_base_LV_myocardial_shell",
            "build_initial_open_base_RV_myocardial_shell",
            "remove_RVepi_open_points_intersecting_LVepi",
            "remove_RV_open_points_intersecting_LVepi",
            "fix_RVepi_and_move_RVendo_to_min_3mm_RVepi_distance_freezing_basal_opening_boundary",
            "merge_new_RVepi_plus_LVepi_as_whole_epi_surface",
            "merge_new_RVendo_plus_RV_openbase_annulus_plus_LVendo_plus_LV_openbase_annulus_as_whole_endo_annulus_surface",
            "add_required_RV_cut_closure_surfaces_for_TetGen_PLC",
            "tetgen_whole_volume_mesh",
            "post_tag_centroids_inside_LV_shell_as_elemTag_1_else_0",
            "write_final_custom_VTK",
        ],
        "parameters": serializable_args(frame_args),
        "subject_base_detection": subject_base_detection,
        "profiles": {"lv": base.profile_to_dict(lv_profile), "rv": base.profile_to_dict(rv_profile)},
        "lv_open_base": openbase.ventricle_summary(lv_build),
        "rv_initial_open_base": openbase.ventricle_summary(rv_build),
        "open_shell_stage_summary": open_shell_stage_summary,
        "intersection_removal_stage_summary": intersection_stage_summary,
        "rvendo_distance_stage_summary": rvendo_distance_stage_summary,
        "merge_surface_stage_summary": merge_surface_stage_summary,
        "surface_intersection_removal": {"RVepi": rv_epi_removal, "RV": rv_endo_removal},
        "rv_distance_fix": rv_distance_fix,
        "rv_closure_surfaces": rv_cut_info,
        "surface_summaries": {
            "lv_shell_reference": surface_summary(lv_shell),
            "lv_epi_closed_intersection_reference": surface_summary(lv_epi_intersection_reference),
            "rv_epi_trimmed": surface_summary(rv_epi_trimmed),
            "rv_endo_trimmed": surface_summary(rv_endo_trimmed),
            "rv_endo_fixed": surface_summary(rv_endo_fixed),
            "whole_epi": surface_summary(whole_epi),
            "whole_endo": surface_summary(whole_endo),
            "overall_surface_before_tetgen": surface_summary(overall_surface),
            "tetgen_surface_used": surface_summary(tetgen_surface),
        },
        "region_seeds": {"LV": lv_seed_info, "RV": rv_seed_info},
        "tetgen": tetgen_info,
        "tetgen_region_label_counts": region_label_counts,
        "tagging": tag_info,
        "component_filter": component_filter,
        "components": components,
        "quality": quality,
    }
    summary_path = output_dir / f"{subject_id}_{frame}_openbase_LVRV_tetgen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "frame": frame,
        "biventricular_volmesh": str(output_mesh),
        "summary": str(summary_path),
        "tetrahedra": int(len(tet_cells)),
        "elemTag": {str(k): int(v) for k, v in sorted(Counter(map(int, elem_tags)).items())},
        "component_count": int(components["component_count"]),
        "nonpositive_signed_tets": int(quality["nonpositive_signed_tets"]),
        "tetgen_input": tetgen_info["selected_input"],
    }


def main() -> None:
    args = parse_args()
    frames = parse_csv_values(args.frames) or [args.frame]
    results = [process_frame(args, frame) for frame in frames]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
