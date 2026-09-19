#!/usr/bin/env python3
"""Generate an open-base biventricular myocardium mesh with one TetGen PLC.

Requested pipeline:
  1. Load LVendo, LVepi, RV, RVepi surfaces.
  2. Detect subject base side and write LV/RV Z-area profiles.
  3. Build the open-base LV myocardial shell.
  4. Build the initial open-base RV myocardial shell.
  5. Remove RVepi-open points intersecting LVepi, then keep only valid triangles.
  6. Remove RV-open points intersecting LVepi, then keep only valid triangles.
  7. Keep RVepi fixed, move RVendo until RV-to-RVepi distance is >= 3 mm,
     adaptively update and smooth the RVendo open-base ring from neighboring
     displacement, then rebuild the RV open-base annulus.
  8. Re-check/fix RVepi and RVendo intersections against fixed LVepi, then
     rebuild LV/RV open-base annuli from the final fixed surface loops.
  9. Merge new RVepi+LVepi as the whole epi surface and merge
     new RVendo+rebuilt RV open-base annulus+LVendo+LV open-base annulus as
     the whole endo/annulus surface.  Also derive a new surface from the
     saved whole_epi_endo VTK by removing the LVepi patch enclosed between
     the RVepi/LVepi and RVendo/LVepi intersection boundaries, then fill
     the resulting patch holes.
 10. Tetrahedralize the saved Step 9 filled-holes surface with TetGen.
 11. Write final VTK with elemTag=1 for LV and elemTag=0 for RV.
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
from scipy.spatial import Delaunay, cKDTree
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
    parser.add_argument("--save-fix-intersection-new-surf-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-merge-surface-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-tetgen-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-open-shells", action="store_true", help="Save LV/RV open-base shell stage outputs and stop before later RV trimming/TetGen stages.")
    parser.add_argument("--stop-after-intersection-removal", action="store_true", help="Save step 5/6 triangle-surface outputs and stop before RV distance fixing/TetGen stages.")
    parser.add_argument("--stop-after-rvendo-distance", action="store_true", help="Save step 7 RVepi/RVendo triangle surfaces and stop before whole-surface merging/TetGen stages.")
    parser.add_argument("--stop-after-fix-intersection-new-surf", action="store_true", help="Save step 8 fixed LV/RV surfaces and rebuilt annuli, then stop before whole-surface merging/TetGen stages.")
    parser.add_argument("--stop-after-merge-surface", action="store_true", help="Save step 9 whole epi/endo triangle surfaces and stop before closure/TetGen stages.")
    parser.add_argument("--stop-after-tetgen", action="store_true", help="Save the step 10 raw whole TetGen volume mesh and stop before step 11 LV/RV elemTag classification.")
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
    parser.add_argument("--stage8-intersection-check-tolerance", type=float, default=1e-6)
    parser.add_argument("--step9-lvepi-face-match-tolerance", type=float, default=0.25)
    parser.add_argument("--step9-rv-openbase-loop-match-tolerance", type=float, default=0.5)
    parser.add_argument("--step9-lvepi-patch-loop-tolerance", type=float, default=3.0)
    parser.add_argument("--step9-lvepi-patch-component-min-fraction", type=float, default=0.02)
    parser.add_argument("--step9-fill-lvepi-patch-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step9-hole-fill-boundary-match-tolerance", type=float, default=3.0)
    parser.add_argument("--combined-surface-merge-tolerance", type=float, default=1e-6)
    parser.add_argument("--boundary-match-tolerance", type=float, default=1e-3)
    parser.add_argument("--rv-min-distance-to-rvepi", type=float, default=3.0)
    parser.add_argument("--rv-distance-padding", type=float, default=0.05)
    parser.add_argument("--rv-distance-iterations", type=int, default=8)
    parser.add_argument("--rv-distance-max-step", type=float, default=2.0)
    parser.add_argument("--rv-opening-adaptive-rings", type=int, default=3)
    parser.add_argument("--rv-opening-adaptive-k-neighbors", type=int, default=12)
    parser.add_argument("--rv-opening-adaptive-max-displacement-percentile", type=float, default=95.0)
    parser.add_argument("--rv-opening-adaptive-max-displacement-multiplier", type=float, default=1.25)
    parser.add_argument("--rv-opening-ring-smooth-iterations", type=int, default=15)
    parser.add_argument("--rv-opening-ring-smooth-relaxation", type=float, default=0.20)
    parser.add_argument("--rv-opening-band-smooth-iterations", type=int, default=8)
    parser.add_argument("--rv-opening-band-smooth-relaxation", type=float, default=0.12)
    parser.add_argument("--rv-opening-ring-floor-iterations", type=int, default=6)

    parser.add_argument("--tetgen-input-mode", choices=["raw", "repair", "raw_then_repair"], default="repair")
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


def vertex_neighbors_from_surface(mesh: pv.PolyData) -> List[set]:
    triangles = base.triangle_faces(mesh.extract_surface().triangulate())
    neighbors = [set() for _ in range(mesh.n_points)]
    for tri in triangles:
        a, b, c = (int(v) for v in tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def vertex_ring_levels(start_mask: np.ndarray, neighbors: Sequence[set], rings: int) -> np.ndarray:
    start_mask = np.asarray(start_mask, dtype=bool)
    levels = -np.ones(len(start_mask), dtype=np.int32)
    current = set(int(v) for v in np.flatnonzero(start_mask))
    for vertex_id in current:
        levels[vertex_id] = 0
    visited = set(current)
    for ring in range(1, max(int(rings), 0) + 1):
        next_ring = set()
        for vertex_id in current:
            for nb in neighbors[vertex_id]:
                nb = int(nb)
                if nb not in visited:
                    next_ring.add(nb)
        if not next_ring:
            break
        for vertex_id in next_ring:
            levels[vertex_id] = int(ring)
        visited.update(next_ring)
        current = next_ring
    return levels


def smooth_closed_vectors(vectors: np.ndarray, iterations: int, relaxation: float) -> np.ndarray:
    out = np.asarray(vectors, dtype=float).copy()
    if len(out) < 3 or int(iterations) <= 0 or float(relaxation) <= 0.0:
        return out
    for _ in range(int(iterations)):
        out = out + float(relaxation) * ((np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0)) * 0.5 - out)
    return out


def boundary_loop_match_records(loops: Sequence[pv.PolyData], reference_loop: pv.PolyData) -> List[Dict[str, object]]:
    reference_points = loop_points(reference_loop)
    if len(reference_points) == 0:
        return []
    reference_tree = cKDTree(reference_points)
    records: List[Dict[str, object]] = []
    for loop_id, loop in enumerate(loops):
        points = loop_points(loop)
        if len(points) == 0:
            continue
        loop_tree = cKDTree(points)
        loop_to_reference = reference_tree.query(points, k=1)[0]
        reference_to_loop = loop_tree.query(reference_points, k=1)[0]
        centroid_distance = float(np.linalg.norm(points.mean(axis=0) - reference_points.mean(axis=0)))
        score = float(np.percentile(loop_to_reference, 95) + np.percentile(reference_to_loop, 95) + 0.01 * centroid_distance)
        records.append({
            "loop_id": int(loop_id),
            "score": score,
            "points": int(len(points)),
            "length": float(loop.length),
            "area_best_plane": float(abs(base.signed_area_in_best_plane(points))) if len(points) >= 3 else 0.0,
            "centroid_distance_to_reference": centroid_distance,
            "loop_to_reference_distance_p95": float(np.percentile(loop_to_reference, 95)),
            "reference_to_loop_distance_p95": float(np.percentile(reference_to_loop, 95)),
        })
    records.sort(key=lambda item: float(item["score"]))
    return records


def select_boundary_loop_by_reference(name: str, mesh: pv.PolyData, reference_loop: pv.PolyData) -> Tuple[pv.PolyData, Dict[str, object]]:
    surface = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    loops = base.boundary_loops(surface)
    if not loops:
        raise RuntimeError(f"{name}: no boundary loop found on surface.")
    records = boundary_loop_match_records(loops, reference_loop)
    if records:
        selected_id = int(records[0]["loop_id"])
        selected = loops[selected_id]
        method = "closest_boundary_loop_to_reference_openbase_loop"
    else:
        lengths = [float(loop.length) for loop in loops]
        selected_id = int(np.argmax(lengths))
        selected = loops[selected_id]
        method = "fallback_longest_boundary_loop"
    selected_points = loop_points(selected)
    selected_loop = base.make_polyline(selected_points)
    return selected_loop, {
        "name": name,
        "method": method,
        "boundary_loop_count": int(len(loops)),
        "selected_loop_id": int(selected_id),
        "selected_loop": loop_record(selected_loop),
        "candidate_matches": records[:10],
    }


def ordered_loop_vertex_indices(
    surface: pv.PolyData,
    loop: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    surface_points = np.asarray(surface.points, dtype=float)
    loop_pts = loop_points(loop)
    if len(loop_pts) < 3:
        raise RuntimeError("Open-base loop has fewer than three points.")
    distances, ids = cKDTree(surface_points).query(loop_pts, k=1)
    ordered: List[int] = []
    seen = set()
    for vertex_id in np.asarray(ids, dtype=np.int64):
        vertex_id = int(vertex_id)
        if ordered and ordered[-1] == vertex_id:
            continue
        if vertex_id in seen:
            continue
        ordered.append(vertex_id)
        seen.add(vertex_id)
    if len(ordered) < 3:
        raise RuntimeError("Mapped open-base loop has fewer than three unique surface vertices.")
    ordered_ids = np.asarray(ordered, dtype=np.int64)
    mask = np.zeros(surface.n_points, dtype=bool)
    mask[ordered_ids] = True
    return ordered_ids, mask, {
        "method": "ordered_nearest_surface_vertices_for_selected_openbase_loop",
        "selected_loop_points": int(len(loop_pts)),
        "unique_surface_vertices": int(len(ordered_ids)),
        "nearest_distance_min": float(np.min(distances)) if len(distances) else None,
        "nearest_distance_mean": float(np.mean(distances)) if len(distances) else None,
        "nearest_distance_p95": float(np.percentile(distances, 95)) if len(distances) else None,
        "nearest_distance_max": float(np.max(distances)) if len(distances) else None,
    }


def match_openbase_loop_on_surface(
    name: str,
    surface: pv.PolyData,
    reference_loop: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, np.ndarray, np.ndarray, Dict[str, object]]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    selected_loop, selection_info = select_boundary_loop_by_reference(name, surface, reference_loop)
    ordered_ids, mask, index_info = ordered_loop_vertex_indices(surface, selected_loop, args)
    exact_loop = base.make_polyline(np.asarray(surface.points, dtype=float)[ordered_ids])
    return exact_loop, ordered_ids, mask, {
        "selection": selection_info,
        "indexing": index_info,
    }


def weighted_displacements_from_support(
    query_points: np.ndarray,
    support_points: np.ndarray,
    support_displacements: np.ndarray,
    k_neighbors: int,
) -> np.ndarray:
    if len(query_points) == 0:
        return np.empty((0, 3), dtype=float)
    if len(support_points) == 0:
        return np.zeros((len(query_points), 3), dtype=float)
    k = min(max(int(k_neighbors), 1), len(support_points))
    distances, indices = cKDTree(support_points).query(query_points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-6) ** 2
    weighted = support_displacements[np.asarray(indices, dtype=np.int64)] * weights[:, :, None]
    return weighted.sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)[:, None]


def limit_displacement_vectors(vectors: np.ndarray, max_length: float) -> Tuple[np.ndarray, int]:
    out = np.asarray(vectors, dtype=float).copy()
    if len(out) == 0 or not np.isfinite(max_length) or max_length <= 0.0:
        return out, 0
    lengths = np.linalg.norm(out, axis=1)
    limited = lengths > float(max_length)
    if np.any(limited):
        out[limited] *= (float(max_length) / np.maximum(lengths[limited], 1e-12))[:, None]
    return out, int(np.count_nonzero(limited))


def enforce_rvendo_distance_adaptive_opening_ring(
    rv_endo: pv.PolyData,
    rv_epi: pv.PolyData,
    opening_loop: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, np.ndarray, np.ndarray, np.ndarray, pv.PolyData, Dict[str, object]]:
    rv_endo = rv_endo.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    rv_epi = rv_epi.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    opening_loop_current, opening_ids, opening_mask, opening_match = match_openbase_loop_on_surface(
        "RVendo_openbase_ring_before_step7",
        rv_endo,
        opening_loop,
        args,
    )

    original_points = np.asarray(rv_endo.points, dtype=float).copy()
    points = original_points.copy()
    neighbors = vertex_neighbors_from_surface(rv_endo)
    ring_levels = vertex_ring_levels(opening_mask, neighbors, int(args.rv_opening_adaptive_rings))
    transition_band_mask = ring_levels >= 0

    rv_epi_points = np.asarray(rv_epi.points, dtype=float)
    rv_epi_tree = cKDTree(rv_epi_points)
    requested_distance = float(args.rv_min_distance_to_rvepi)
    target = requested_distance + float(args.rv_distance_padding)
    total_direct_moved = np.zeros(len(points), dtype=bool)
    iteration_records = []
    initial_distances = rv_epi_tree.query(points, k=1)[0] if len(points) else np.empty((0,), dtype=float)

    for iteration in range(int(args.rv_distance_iterations)):
        distances, ids = rv_epi_tree.query(points, k=1)
        movable = (distances < target) & (distances > 1e-8) & (~opening_mask)
        if not np.any(movable):
            iteration_records.append({
                "iteration": int(iteration),
                "direct_moved_vertices": 0,
                "adaptive_opening_boundary_vertices": int(np.count_nonzero(opening_mask)),
                "min_distance_before": float(np.min(distances)) if len(distances) else None,
                "p05_distance_before": float(np.percentile(distances, 5)) if len(distances) else None,
            })
            break
        nearest = rv_epi_points[ids[movable]]
        vectors = points[movable] - nearest
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)[:, None]
        steps = np.minimum(target - distances[movable], float(args.rv_distance_max_step))
        points[movable] += unit * steps[:, None]
        total_direct_moved[movable] = True
        iteration_records.append({
            "iteration": int(iteration),
            "direct_moved_vertices": int(np.count_nonzero(movable)),
            "adaptive_opening_boundary_vertices": int(np.count_nonzero(opening_mask)),
            "min_distance_before": float(np.min(distances)) if len(distances) else None,
            "p05_distance_before": float(np.percentile(distances, 5)) if len(distances) else None,
            "max_step_mm": float(np.max(steps)) if len(steps) else 0.0,
        })

    displacement = points - original_points
    displacement_lengths = np.linalg.norm(displacement, axis=1)
    support_ids = np.flatnonzero((ring_levels > 0) & total_direct_moved)
    support_source = "moved_vertices_in_opening_neighborhood"
    if len(support_ids) < 3:
        support_ids = np.flatnonzero((~opening_mask) & total_direct_moved)
        support_source = "all_directly_moved_non_opening_vertices"
    if len(support_ids) < 3:
        support_ids = np.flatnonzero(~opening_mask)
        support_source = "all_non_opening_vertices_fallback"

    ring_displacement_raw = weighted_displacements_from_support(
        original_points[opening_ids],
        original_points[support_ids],
        displacement[support_ids],
        int(args.rv_opening_adaptive_k_neighbors),
    )
    support_lengths = np.linalg.norm(displacement[support_ids], axis=1) if len(support_ids) else np.empty((0,), dtype=float)
    if len(support_lengths):
        max_ring_displacement = float(np.percentile(support_lengths, float(args.rv_opening_adaptive_max_displacement_percentile)))
        max_ring_displacement *= float(args.rv_opening_adaptive_max_displacement_multiplier)
    else:
        max_ring_displacement = 0.0
    ring_displacement_limited, limited_count = limit_displacement_vectors(ring_displacement_raw, max_ring_displacement)
    ring_displacement_smoothed = smooth_closed_vectors(
        ring_displacement_limited,
        int(args.rv_opening_ring_smooth_iterations),
        float(args.rv_opening_ring_smooth_relaxation),
    )
    displacement[opening_ids] = ring_displacement_smoothed

    extra_opening_ids = np.setdiff1d(np.flatnonzero(opening_mask), opening_ids, assume_unique=False)
    if len(extra_opening_ids):
        nearest = cKDTree(original_points[opening_ids]).query(original_points[extra_opening_ids], k=1)[1]
        displacement[extra_opening_ids] = ring_displacement_smoothed[np.asarray(nearest, dtype=np.int64)]

    band_iterations = int(args.rv_opening_band_smooth_iterations)
    band_relaxation = float(args.rv_opening_band_smooth_relaxation)
    if band_iterations > 0 and band_relaxation > 0.0 and np.any(ring_levels > 0):
        smooth_ids = np.flatnonzero(ring_levels > 0)
        max_level = max(int(args.rv_opening_adaptive_rings), 1)
        for _ in range(band_iterations):
            next_displacement = displacement.copy()
            for vertex_id in smooth_ids:
                nb = list(neighbors[int(vertex_id)])
                if not nb:
                    continue
                level = max(int(ring_levels[vertex_id]), 1)
                weight = band_relaxation * float(max_level - level + 1) / float(max_level + 1)
                neighbor_mean = displacement[np.asarray(nb, dtype=np.int64)].mean(axis=0)
                next_displacement[vertex_id] = displacement[vertex_id] + weight * (neighbor_mean - displacement[vertex_id])
            next_displacement[opening_ids] = ring_displacement_smoothed
            displacement = next_displacement

    ring_floor_records = []
    max_level = max(int(args.rv_opening_adaptive_rings), 1)
    ring_tree = cKDTree(original_points[opening_ids])
    ring_floor_padding = max(1e-3, min(float(args.rv_distance_padding), 0.02))
    for floor_iteration in range(max(int(args.rv_opening_ring_floor_iterations), 0)):
        current_points = original_points + displacement
        ring_distances, ring_epi_ids = rv_epi_tree.query(current_points[opening_ids], k=1)
        ring_below = ring_distances < requested_distance
        record = {
            "iteration": int(floor_iteration),
            "opening_vertices_below_before": int(np.count_nonzero(ring_below)),
            "min_opening_distance_before": float(np.min(ring_distances)) if len(ring_distances) else None,
        }
        if not np.any(ring_below):
            ring_floor_records.append(record)
            break
        nearest = rv_epi_points[ring_epi_ids[ring_below]]
        vectors = current_points[opening_ids][ring_below] - nearest
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)[:, None]
        steps = np.minimum(requested_distance - ring_distances[ring_below] + ring_floor_padding, float(args.rv_distance_max_step) * 0.5)
        correction = np.zeros((len(opening_ids), 3), dtype=float)
        correction[ring_below] = unit * steps[:, None]
        correction = smooth_closed_vectors(
            correction,
            max(3, int(args.rv_opening_ring_smooth_iterations) // 3),
            float(args.rv_opening_ring_smooth_relaxation),
        )
        correction, limited_floor = limit_displacement_vectors(correction, float(args.rv_distance_max_step) * 0.75)
        displacement[opening_ids] += correction
        if np.any(ring_levels > 0):
            for level in range(1, max_level + 1):
                level_ids = np.flatnonzero(ring_levels == level)
                if len(level_ids) == 0:
                    continue
                nearest_ring = ring_tree.query(original_points[level_ids], k=1)[1]
                weight = 0.5 * float(max_level - level + 1) / float(max_level + 1)
                displacement[level_ids] += correction[np.asarray(nearest_ring, dtype=np.int64)] * weight
        record.update({
            "max_correction_mm": float(np.max(np.linalg.norm(correction, axis=1))) if len(correction) else 0.0,
            "mean_correction_mm": float(np.mean(np.linalg.norm(correction, axis=1))) if len(correction) else 0.0,
            "limited_vertices": int(limited_floor),
        })
        ring_floor_records.append(record)

    points = original_points + displacement
    final_correction_moved = np.zeros(len(points), dtype=bool)
    for correction_iteration in range(2):
        distances, ids = rv_epi_tree.query(points, k=1)
        correct = (distances < requested_distance) & (distances > 1e-8) & (~opening_mask)
        if not np.any(correct):
            break
        nearest = rv_epi_points[ids[correct]]
        vectors = points[correct] - nearest
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)[:, None]
        steps = np.minimum(requested_distance - distances[correct] + 1e-3, float(args.rv_distance_max_step) * 0.5)
        points[correct] += unit * steps[:, None]
        final_correction_moved[correct] = True

    final_source_moved = np.linalg.norm(points - original_points, axis=1) > max(float(args.boundary_match_tolerance), 1e-6)
    moved = rv_endo.copy(deep=True)
    moved.points = points
    moved = moved.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    cleanup_tolerance = max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0)
    moved_mask = map_source_points_to_target_mask(points[final_source_moved], np.asarray(moved.points, dtype=float), cleanup_tolerance)
    opening_after_cleanup = map_source_points_to_target_mask(points[opening_mask], np.asarray(moved.points, dtype=float), cleanup_tolerance)
    band_after_cleanup = map_source_points_to_target_mask(points[transition_band_mask], np.asarray(moved.points, dtype=float), cleanup_tolerance)
    final_opening_loop = base.make_polyline(points[opening_ids])

    final_distances = rv_epi_tree.query(np.asarray(moved.points, dtype=float), k=1)[0]
    below_requested = final_distances < requested_distance
    return moved, moved_mask, opening_after_cleanup, band_after_cleanup, final_opening_loop, {
        "method": "fix_RVepi_and_move_RVendo_to_minimum_distance_with_adaptive_openbase_ring",
        "minimum_requested_distance_mm": requested_distance,
        "target_distance_with_padding_mm": target,
        "rv_epi_fixed": True,
        "rvendo_basal_opening_boundary_frozen": False,
        "rvendo_basal_opening_boundary_adaptive": True,
        "opening_loop_match": opening_match,
        "opening_loop_before_step7": loop_record(opening_loop_current),
        "opening_loop_after_adaptive_smoothing": loop_record(final_opening_loop),
        "adaptive_opening_boundary_vertices": int(np.count_nonzero(opening_mask)),
        "adaptive_opening_boundary_vertices_after_cleanup": int(np.count_nonzero(opening_after_cleanup)),
        "transition_band_vertices": int(np.count_nonzero(transition_band_mask)),
        "transition_band_vertices_after_cleanup": int(np.count_nonzero(band_after_cleanup)),
        "adaptive_ring_support_source": support_source,
        "adaptive_ring_support_vertices": int(len(support_ids)),
        "adaptive_ring_k_neighbors": int(args.rv_opening_adaptive_k_neighbors),
        "adaptive_ring_max_displacement_mm": float(max_ring_displacement),
        "adaptive_ring_limited_vertices": int(limited_count),
        "adaptive_ring_raw_displacement_mean_mm": float(np.mean(np.linalg.norm(ring_displacement_raw, axis=1))) if len(ring_displacement_raw) else 0.0,
        "adaptive_ring_raw_displacement_max_mm": float(np.max(np.linalg.norm(ring_displacement_raw, axis=1))) if len(ring_displacement_raw) else 0.0,
        "adaptive_ring_final_displacement_mean_mm": float(np.mean(np.linalg.norm(ring_displacement_smoothed, axis=1))) if len(ring_displacement_smoothed) else 0.0,
        "adaptive_ring_final_displacement_max_mm": float(np.max(np.linalg.norm(ring_displacement_smoothed, axis=1))) if len(ring_displacement_smoothed) else 0.0,
        "ring_smooth_iterations": int(args.rv_opening_ring_smooth_iterations),
        "ring_smooth_relaxation": float(args.rv_opening_ring_smooth_relaxation),
        "band_smooth_iterations": int(args.rv_opening_band_smooth_iterations),
        "band_smooth_relaxation": float(args.rv_opening_band_smooth_relaxation),
        "opening_ring_distance_floor_iterations_requested": int(args.rv_opening_ring_floor_iterations),
        "opening_ring_distance_floor_padding_mm": float(ring_floor_padding),
        "opening_ring_distance_floor_iterations": ring_floor_records,
        "initial_vertices_below_requested_distance": int(np.count_nonzero(initial_distances < requested_distance)) if len(initial_distances) else 0,
        "initial_opening_vertices_below_requested_distance": int(np.count_nonzero((initial_distances < requested_distance) & opening_mask)) if len(initial_distances) else 0,
        "initial_nonopening_vertices_below_requested_distance": int(np.count_nonzero((initial_distances < requested_distance) & (~opening_mask))) if len(initial_distances) else 0,
        "initial_min_distance_all_vertices": float(np.min(initial_distances)) if len(initial_distances) else None,
        "initial_p05_distance_all_vertices": float(np.percentile(initial_distances, 5)) if len(initial_distances) else None,
        "iterations": iteration_records,
        "direct_vertices_moved_before_ring_adaptation": int(np.count_nonzero(total_direct_moved)),
        "final_correction_nonopening_vertices_moved": int(np.count_nonzero(final_correction_moved)),
        "total_vertices_moved_after_adaptation": int(np.count_nonzero(final_source_moved)),
        "final_vertices_below_requested_distance": int(np.count_nonzero(below_requested)) if len(final_distances) else 0,
        "final_nonopening_vertices_below_requested_distance": int(np.count_nonzero(below_requested & (~opening_after_cleanup))) if len(final_distances) else 0,
        "final_adaptive_opening_vertices_below_requested_distance": int(np.count_nonzero(below_requested & opening_after_cleanup)) if len(final_distances) else 0,
        "final_min_distance_all_vertices": float(np.min(final_distances)) if len(final_distances) else None,
        "final_p05_distance_all_vertices": float(np.percentile(final_distances, 5)) if len(final_distances) else None,
    }


def rebuild_rv_openbase_annulus_after_step7(
    rv_epi: pv.PolyData,
    rv_endo: pv.PolyData,
    rv_epi_reference_loop: pv.PolyData,
    rv_endo_reference_loop: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, pv.PolyData, pv.PolyData, Dict[str, object]]:
    rv_epi_loop, _, _, epi_match = match_openbase_loop_on_surface(
        "RVepi_openbase_ring_for_rebuilt_annulus",
        rv_epi,
        rv_epi_reference_loop,
        args,
    )
    rv_endo_loop, _, _, endo_match = match_openbase_loop_on_surface(
        "RVendo_openbase_ring_for_rebuilt_annulus",
        rv_endo,
        rv_endo_reference_loop,
        args,
    )
    annulus = cap_utils.smooth_myocardium_cut_surface(
        rv_epi_loop,
        rv_endo_loop,
        sample_count=args.annulus_loop_points,
        radial_layers=args.annulus_radial_layers,
        smooth_iterations=args.annulus_smooth_iterations,
        smooth_relaxation=args.annulus_smooth_relaxation,
    )
    annulus = annulus.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    return rv_epi_loop, rv_endo_loop, annulus, {
        "method": "rebuild_RV_openbase_annulus_after_adaptive_RVendo_ring_smoothing",
        "rv_epi_loop_match": epi_match,
        "rv_endo_loop_match": endo_match,
        "annulus": surface_summary(annulus),
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

    clean_parts = [part.extract_surface().triangulate().clean(tolerance=float(tolerance)) for part in parts]
    if all("elemTag" in part.cell_data and len(part.cell_data["elemTag"]) == part.n_cells for part in clean_parts):
        points: List[np.ndarray] = []
        faces: List[np.ndarray] = []
        tags: List[np.ndarray] = []
        point_offset = 0
        for part in clean_parts:
            triangles = base.triangle_faces(part)
            points.append(np.asarray(part.points, dtype=float))
            faces.append(np.column_stack([np.full(len(triangles), 3, dtype=np.int64), triangles + point_offset]).ravel())
            tags.append(np.rint(np.asarray(part.cell_data["elemTag"])).astype(np.int32))
            point_offset += int(part.n_points)
        merged = pv.PolyData(np.vstack(points), np.concatenate(faces))
        merged.cell_data["elemTag"] = np.concatenate(tags)
        return merged.extract_surface().triangulate().clean(tolerance=float(tolerance))

    merged = clean_parts[0]
    for part in clean_parts[1:]:
        merged = merged.merge(part, merge_points=True, tolerance=float(tolerance))
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


def classify_by_lv_reference_surface(
    points: np.ndarray,
    cells: np.ndarray,
    lv_reference: pv.PolyData,
    args: argparse.Namespace,
    reference_name: str = "original_closed_LVepi",
) -> Tuple[np.ndarray, Dict[str, object]]:
    centroids = points[cells].mean(axis=1)
    reference = lv_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    inside_lv = base.enclosed_mask(centroids, reference)
    tags = np.full(len(cells), int(args.rv_tag), dtype=np.int32)
    tags[inside_lv] = int(args.lv_tag)
    counts = Counter(map(int, tags))
    total = int(len(tags))
    min_cells = int(np.ceil(total * float(args.min_region_cell_fraction))) if total else 0
    lv_count = int(counts.get(int(args.lv_tag), 0))
    rv_count = int(counts.get(int(args.rv_tag), 0))
    passes = bool(total > 0 and lv_count >= min_cells and rv_count >= min_cells)
    info = {
        "method": "tet_centroid_inside_original_closed_LVepi_reference",
        "reference_name": reference_name,
        "reference_surface": surface_summary(reference),
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
        save_triangle_surface_vtk(path, mesh, elem_tag=surface_tag_from_name(name))
        paths[name] = str(path.resolve())
    return paths



def surface_tag_from_name(name: str, default: int = -1) -> int:
    upper = name.upper()
    lv_index = upper.find("LV")
    rv_index = upper.find("RV")
    if lv_index >= 0 and rv_index >= 0:
        return 1 if lv_index < rv_index else 0
    if lv_index >= 0:
        return 1
    if rv_index >= 0:
        return 0
    return int(default)


def tagged_surface(mesh: pv.PolyData, elem_tag: int) -> pv.PolyData:
    surface = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    surface.cell_data["elemTag"] = np.full(surface.n_cells, int(elem_tag), dtype=np.int32)
    return surface


def cell_tags_for_surface(surface: pv.PolyData, path: Path, elem_tag: Optional[int]) -> np.ndarray:
    if "elemTag" in surface.cell_data and len(surface.cell_data["elemTag"]) == surface.n_cells:
        return np.rint(np.asarray(surface.cell_data["elemTag"])).astype(np.int32)
    tag = surface_tag_from_name(path.stem) if elem_tag is None else int(elem_tag)
    return np.full(surface.n_cells, int(tag), dtype=np.int32)


def write_custom_surface_vtk(
    filename: Path,
    points: np.ndarray,
    triangles: np.ndarray,
    elem_tags: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write triangle surfaces in the same ASCII legacy style as write_custom_vtk."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    elem_tags = np.asarray(elem_tags, dtype=np.int32).reshape(-1)
    if len(triangles) != len(elem_tags):
        raise ValueError(f"Triangle count {len(triangles)} does not match elemTag count {len(elem_tags)}.")

    with Path(filename).open("w") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("vtk output\n")
        handle.write("ASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n\n")

        handle.write(f"POINTS {len(points)} float\n")
        np.savetxt(handle, points, fmt="%.5f", delimiter=" ")
        handle.write("\n")

        handle.write(f"CELL_TYPES {len(triangles)}\n")
        np.savetxt(handle, np.full(len(triangles), 5, dtype=np.int32), fmt="%d")
        handle.write("\n")

        handle.write(f"CELLS {len(triangles)} {4 * len(triangles)}\n")
        cell_lines = np.column_stack([np.full(len(triangles), 3, dtype=np.int32), triangles])
        np.savetxt(handle, cell_lines, fmt="%d")
        handle.write("\n")

        handle.write(f"CELL_DATA {len(triangles)}\n")
        handle.write("SCALARS elemTag int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        np.savetxt(handle, elem_tags.reshape(-1, 1), fmt="%d")
        handle.write("\n")

        if point_data:
            valid_point_data = []
            for name, values in point_data.items():
                array = np.asarray(values)
                if len(array) == len(points):
                    valid_point_data.append((name, array))
            if valid_point_data:
                handle.write(f"POINT_DATA {len(points)}\n")
                for name, array in valid_point_data:
                    vtk_name = "".join(char if char.isalnum() or char == "_" else "_" for char in str(name))
                    if array.ndim == 2 and array.shape[1] == 3:
                        handle.write(f"VECTORS {vtk_name} float\n")
                        np.savetxt(handle, array.astype(float), fmt="%.6g")
                    else:
                        flat = array.reshape((len(points), -1))[:, 0]
                        if np.issubdtype(flat.dtype, np.integer) or np.issubdtype(flat.dtype, np.bool_):
                            handle.write(f"SCALARS {vtk_name} int 1\n")
                            handle.write("LOOKUP_TABLE default\n")
                            np.savetxt(handle, flat.astype(np.int32).reshape(-1, 1), fmt="%d")
                        else:
                            handle.write(f"SCALARS {vtk_name} float 1\n")
                            handle.write("LOOKUP_TABLE default\n")
                            np.savetxt(handle, flat.astype(float).reshape(-1, 1), fmt="%.6g")
                        handle.write("\n")

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
        surface_records[name] = save_triangle_surface_vtk(path, clean_mesh, elem_tag=surface_tag_from_name(name))
        paths[name] = str(path.resolve())

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
    elem_tag: Optional[int] = None,
) -> Dict[str, object]:
    surface = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
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

    triangles = base.triangle_faces(surface)
    elem_tags = cell_tags_for_surface(surface, path, elem_tag)
    surface.cell_data["elemTag"] = elem_tags
    write_custom_surface_vtk(path, np.asarray(surface.points, dtype=float), triangles, elem_tags, dict(surface.point_data))

    record = surface_summary(surface)
    tag_counts = Counter(map(int, elem_tags))
    record.update({
        "path": str(path.resolve()),
        "writer": "custom_ascii_unstructured_grid_triangle_surface",
        "cell_type": 5,
        "cell_data": ["elemTag"],
        "elemTag_counts": {str(k): int(v) for k, v in sorted(tag_counts.items())},
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
    rv_endo_opening_mask: Optional[np.ndarray] = None,
    rv_endo_transition_band_mask: Optional[np.ndarray] = None,
    rv_openbase_annulus: Optional[pv.PolyData] = None,
    rv_openbase_annulus_info: Optional[Dict[str, object]] = None,
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
    if rv_endo_opening_mask is not None:
        opening_mask = np.asarray(rv_endo_opening_mask, dtype=bool)
        if len(opening_mask) == rv_endo_surface.n_points:
            point_data["adaptiveOpeningBoundary"] = opening_mask.astype(np.int32)
    if rv_endo_transition_band_mask is not None:
        transition_band_mask = np.asarray(rv_endo_transition_band_mask, dtype=bool)
        if len(transition_band_mask) == rv_endo_surface.n_points:
            point_data["openingTransitionBand"] = transition_band_mask.astype(np.int32)
    records["stage07_RVendo_after_3mm_distance"] = save_triangle_surface_vtk(
        rv_endo_path,
        rv_endo_surface,
        point_data=point_data,
    )
    paths["stage07_RVendo_after_3mm_distance"] = str(rv_endo_path.resolve())

    if rv_openbase_annulus is not None:
        rv_annulus_path = pipeline_dir / f"{subject_id}_{frame}_stage07_RV_openbase_annulus_after_adaptive_RVendo_ring_surface.vtk"
        records["stage07_RV_openbase_annulus_after_adaptive_RVendo_ring"] = save_triangle_surface_vtk(
            rv_annulus_path,
            rv_openbase_annulus,
        )
        paths["stage07_RV_openbase_annulus_after_adaptive_RVendo_ring"] = str(rv_annulus_path.resolve())

    summary = {
        "description": "Step 7 outputs after keeping RVepi fixed, moving RVendo by the >=3 mm rule, adaptively smoothing the RVendo open-base ring, and rebuilding the RV open-base annulus.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "rv_distance_fix": rv_distance_fix,
        "rv_openbase_annulus": rv_openbase_annulus_info or {},
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage07_move_RVendo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage07_summary"] = str(summary_path.resolve())
    return paths, summary


def direct_surface_intersection_metrics(
    name_a: str,
    surface_a: pv.PolyData,
    name_b: str,
    surface_b: pv.PolyData,
    tolerance: float,
) -> Dict[str, object]:
    first = surface_a.extract_surface().triangulate().clean(tolerance=1e-7)
    second = surface_b.extract_surface().triangulate().clean(tolerance=1e-7)
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
    curve = pv.wrap(intersector.GetOutput()).clean(tolerance=float(tolerance))
    return {
        "surface_a": name_a,
        "surface_b": name_b,
        "has_intersection_curve": bool(curve.n_points > 0 or curve.n_cells > 0),
        "intersection_points": int(curve.n_points),
        "intersection_cells": int(curve.n_cells),
        "intersection_length": float(curve.length) if curve.n_points else 0.0,
    }


def select_openbase_loop_direct(
    name: str,
    surface: pv.PolyData,
    profile,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    loops = base.boundary_loops(surface)
    if not loops:
        raise RuntimeError(f"{name}: no boundary loop found for direct open-base annulus rebuild.")

    points = np.asarray(surface.points, dtype=float)
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    height = max(z_max - z_min, 1e-6)
    records = []
    for loop_id, loop in enumerate(loops):
        pts = loop_points(loop)
        if len(pts) < 3:
            continue
        if profile.base_side == "high_z":
            base_distance = z_max - float(np.max(pts[:, 2]))
        else:
            base_distance = float(np.min(pts[:, 2])) - z_min
        area = float(abs(base.signed_area_in_best_plane(pts)))
        records.append({
            "loop_id": int(loop_id),
            "base_distance": float(base_distance),
            "relative_base_distance": float(base_distance / height),
            "area_best_plane": area,
            "length": float(loop.length),
            "points": int(len(pts)),
            "centroid": [float(v) for v in pts.mean(axis=0)],
        })
    if not records:
        raise RuntimeError(f"{name}: all boundary loops have fewer than three points.")

    basal_band = max(0.12 * height, height / 40.0)
    basal_records = [record for record in records if float(record["base_distance"]) <= basal_band]
    candidates = basal_records or records
    candidates.sort(key=lambda record: (float(record["base_distance"]), -float(record["area_best_plane"]), -float(record["length"])))
    selected_id = int(candidates[0]["loop_id"])
    selected = base.make_polyline(loop_points(loops[selected_id]))
    return selected, {
        "method": "direct_boundary_loop_selection_from_current_fixed_open_surface",
        "surface": name,
        "base_side": profile.base_side,
        "basal_band_mm": float(basal_band),
        "boundary_loop_count": int(len(loops)),
        "selected_loop_id": selected_id,
        "selected_loop": loop_record(selected),
        "candidate_loops": records,
    }


def rebuild_openbase_annulus_from_current_surfaces(
    name: str,
    epi: pv.PolyData,
    endo: pv.PolyData,
    profile,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, pv.PolyData, pv.PolyData, Dict[str, object]]:
    epi_loop, epi_selection = select_openbase_loop_direct(
        f"{name}epi_openbase_ring_for_step8_annulus",
        epi,
        profile,
        args,
    )
    endo_loop, endo_selection = select_openbase_loop_direct(
        f"{name}endo_openbase_ring_for_step8_annulus",
        endo,
        profile,
        args,
    )
    annulus = base.zipper_annulus(epi_loop, endo_loop)
    annulus = annulus.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    return epi_loop, endo_loop, annulus, {
        "method": "direct_zipper_annulus_from_current_fixed_openbase_boundary_loops_no_reference_loop",
        "chamber": name,
        "surface_points_not_moved_by_annulus_rebuild": True,
        "epi_loop_selection": epi_selection,
        "endo_loop_selection": endo_selection,
        "epi_loop": loop_record(epi_loop),
        "endo_loop": loop_record(endo_loop),
        "annulus": surface_summary(annulus),
    }


def fix_intersections_and_rebuild_new_surfaces(
    lv_epi_open: pv.PolyData,
    lv_endo_open: pv.PolyData,
    rv_epi: pv.PolyData,
    rv_endo: pv.PolyData,
    lv_epi_intersection_reference: pv.PolyData,
    lv_profile,
    rv_profile,
    args: argparse.Namespace,
) -> Tuple[Dict[str, pv.PolyData], Dict[str, np.ndarray], Dict[str, object]]:
    lv_epi_fixed = lv_epi_open.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_endo_fixed = lv_endo_open.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_epi_reference = lv_epi_intersection_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)

    rv_epi_fixed, rv_epi_cut_mask, rv_epi_fix = remove_intersecting_points_from_surface(
        "step8_RVepi_against_fixed_LVepi",
        rv_epi,
        lv_epi_reference,
        args,
    )
    rv_endo_fixed, rv_endo_cut_mask, rv_endo_fix = remove_intersecting_points_from_surface(
        "step8_RVendo_against_fixed_LVepi",
        rv_endo,
        lv_epi_reference,
        args,
    )

    lv_epi_loop, lv_endo_loop, lv_annulus, lv_annulus_info = rebuild_openbase_annulus_from_current_surfaces(
        "LV",
        lv_epi_fixed,
        lv_endo_fixed,
        lv_profile,
        args,
    )
    rv_epi_loop, rv_endo_loop, rv_annulus, rv_annulus_info = rebuild_openbase_annulus_from_current_surfaces(
        "RV",
        rv_epi_fixed,
        rv_endo_fixed,
        rv_profile,
        args,
    )

    intersection_checks = {
        "RVepi_fixed_vs_LVepi_fixed": direct_surface_intersection_metrics(
            "RVepi_fixed",
            rv_epi_fixed,
            "LVepi_fixed",
            lv_epi_fixed,
            args.stage8_intersection_check_tolerance,
        ),
        "RVendo_fixed_vs_LVepi_fixed": direct_surface_intersection_metrics(
            "RVendo_fixed",
            rv_endo_fixed,
            "LVepi_fixed",
            lv_epi_fixed,
            args.stage8_intersection_check_tolerance,
        ),
    }

    surfaces = {
        "LVendo": lv_endo_fixed,
        "LVepi": lv_epi_fixed,
        "RVendo": rv_endo_fixed,
        "RVepi": rv_epi_fixed,
        "LV_openbase_annulus": lv_annulus,
        "RV_openbase_annulus": rv_annulus,
    }
    masks = {
        "RVepi_cut_boundary": rv_epi_cut_mask,
        "RVendo_cut_boundary": rv_endo_cut_mask,
    }
    info = {
        "description": "Step 8 keeps LVendo/LVepi fixed, re-fixes RVepi/RVendo against fixed LVepi, then rebuilds LV/RV open-base annuli from the final fixed boundary loops.",
        "LV_surfaces_moved": False,
        "RV_surfaces_moved_by_step8": False,
        "RV_surface_fix_method": "clipped_reconstruction_against_fixed_closed_LVepi_reference",
        "LVepi_intersection_reference": surface_summary(lv_epi_reference),
        "RVepi_fix": rv_epi_fix,
        "RVendo_fix": rv_endo_fix,
        "annuli_rebuilt_from_final_surfaces": True,
        "LV_annulus": lv_annulus_info,
        "RV_annulus": rv_annulus_info,
        "surface_intersection_checks_after_step8": intersection_checks,
        "surface_summaries": {name: surface_summary(mesh) for name, mesh in surfaces.items()},
        "loops": {
            "LVepi_openbase": loop_record(lv_epi_loop),
            "LVendo_openbase": loop_record(lv_endo_loop),
            "RVepi_openbase": loop_record(rv_epi_loop),
            "RVendo_openbase": loop_record(rv_endo_loop),
        },
    }
    return surfaces, masks, info


def save_fix_intersection_new_surface_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    surfaces: Dict[str, pv.PolyData],
    masks: Dict[str, np.ndarray],
    fix_info: Dict[str, object],
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}

    stage_items = [
        ("stage08_LVendo_openbase_surface", surfaces["LVendo"], None),
        ("stage08_LVepi_openbase_surface", surfaces["LVepi"], None),
        ("stage08_RVendo_fixed_openbase_surface", surfaces["RVendo"], masks.get("RVendo_cut_boundary")),
        ("stage08_RVepi_fixed_openbase_surface", surfaces["RVepi"], masks.get("RVepi_cut_boundary")),
        ("stage08_LV_openbase_annulus_surface", surfaces["LV_openbase_annulus"], None),
        ("stage08_RV_openbase_annulus_surface", surfaces["RV_openbase_annulus"], None),
    ]
    for name, mesh, cut_mask in stage_items:
        surface_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = save_triangle_surface_vtk(surface_path, mesh, cut_mask)
        paths[name] = str(surface_path.resolve())

    summary = {
        "description": "Step 8 fixed/intersection-cleaned final open-base surfaces and newly rebuilt LV/RV open-base annuli.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "fix_intersection_new_surface": fix_info,
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage08_fix_intersection_new_surfaces_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage08_summary"] = str(summary_path.resolve())
    return paths, summary


def tagged_surface_face_arrays(mesh: pv.PolyData) -> Tuple[pv.PolyData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    surface = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    triangles = base.triangle_faces(surface)
    points = np.asarray(surface.points, dtype=float)
    centers = points[triangles].mean(axis=1) if len(triangles) else np.empty((0, 3), dtype=float)
    tags = surface.cell_data.get("elemTag")
    if tags is None or len(tags) != len(triangles):
        raise RuntimeError("Surface must contain one elemTag value for each triangle cell.")
    tags = np.rint(np.asarray(tags).reshape(-1)).astype(np.int32)
    return surface, points, triangles, centers, tags


def compact_tagged_triangle_surface(points: np.ndarray, triangles: np.ndarray, elem_tags: np.ndarray) -> pv.PolyData:
    triangles = np.asarray(triangles, dtype=np.int64)
    elem_tags = np.asarray(elem_tags, dtype=np.int32).reshape(-1)
    if len(triangles) == 0:
        raise RuntimeError("Cannot build a triangle surface with zero triangles.")
    used = np.unique(triangles.ravel())
    point_map = -np.ones(len(points), dtype=np.int64)
    point_map[used] = np.arange(len(used), dtype=np.int64)
    out = pv.PolyData(np.asarray(points, dtype=float)[used], base.faces_from_triangles(point_map[triangles]))
    out.cell_data["elemTag"] = elem_tags
    return out.extract_surface().triangulate().clean(tolerance=1e-7)


def loop_to_point_cloud_distance_record(loop: pv.PolyData, target_points: np.ndarray) -> Dict[str, object]:
    pts = loop_points(loop)
    if len(pts) == 0 or len(target_points) == 0:
        return {"loop_points": int(len(pts)), "min": None, "mean": None, "p95": None, "max": None}
    distances = cKDTree(np.asarray(target_points, dtype=float)).query(pts, k=1)[0]
    return {
        "loop_points": int(len(pts)),
        "min": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def rv_lvepi_intersection_boundary_loops(
    name: str,
    rv_surface: pv.PolyData,
    rv_openbase_annulus: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[List[pv.PolyData], Dict[str, object]]:
    surface = rv_surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    annulus = rv_openbase_annulus.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    loops = base.boundary_loops(surface)
    if not loops:
        raise RuntimeError(f"{name}: no boundary loops detected for LVepi patch removal.")
    annulus_points = np.asarray(annulus.points, dtype=float)
    if len(annulus_points) == 0:
        raise RuntimeError(f"{name}: RV open-base annulus has no points; cannot separate basal loop from LVepi-intersection loop.")

    annulus_tree = cKDTree(annulus_points)
    records: List[Dict[str, object]] = []
    gap_loops: List[pv.PolyData] = []
    for loop_id, loop in enumerate(loops):
        pts = loop_points(loop)
        distances = annulus_tree.query(pts, k=1)[0] if len(pts) else np.empty((0,), dtype=float)
        p95 = float(np.percentile(distances, 95)) if len(distances) else float("inf")
        is_openbase = p95 <= float(args.step9_rv_openbase_loop_match_tolerance)
        records.append({
            "loop_id": int(loop_id),
            "points": int(len(pts)),
            "length": float(loop.length),
            "annulus_distance_p95": p95,
            "annulus_distance_max": float(np.max(distances)) if len(distances) else None,
            "classified_as": "RV_openbase_boundary" if is_openbase else "RV_LVepi_intersection_boundary",
        })
        if not is_openbase and len(pts) >= 3:
            gap_loops.append(base.make_polyline(pts))

    if not gap_loops:
        fallback = sorted(records, key=lambda item: float(item["annulus_distance_p95"]), reverse=True)[0]
        fallback_id = int(fallback["loop_id"])
        gap_loops = [base.make_polyline(loop_points(loops[fallback_id]))]
        records[fallback_id]["classified_as"] = "RV_LVepi_intersection_boundary_fallback_farthest_from_openbase_annulus"

    return gap_loops, {
        "surface": name,
        "method": "RV_boundary_loops_not_matching_RV_openbase_annulus_are_used_as_LVepi_intersection_boundaries",
        "loop_count": int(len(loops)),
        "gap_loop_count": int(len(gap_loops)),
        "rv_openbase_loop_match_tolerance_mm": float(args.step9_rv_openbase_loop_match_tolerance),
        "loops": records,
    }


def remove_lvepi_patch_between_rv_boundaries_from_whole_epi_endo(
    whole_epi_endo_surface: pv.PolyData,
    lv_epi_surface: pv.PolyData,
    rv_epi_surface: pv.PolyData,
    rv_endo_surface: pv.PolyData,
    rv_openbase_annulus: pv.PolyData,
    rv_epi_reference: pv.PolyData,
    rv_endo_reference: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, pv.PolyData, Dict[str, object]]:
    whole, points, triangles, centers, tags = tagged_surface_face_arrays(whole_epi_endo_surface)
    lv_epi = lv_epi_surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    rv_epi_ref = rv_epi_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    rv_endo_ref = rv_endo_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)

    rv_epi_gap_loops, rv_epi_loop_info = rv_lvepi_intersection_boundary_loops(
        "RVepi_step9", rv_epi_surface, rv_openbase_annulus, args
    )
    rv_endo_gap_loops, rv_endo_loop_info = rv_lvepi_intersection_boundary_loops(
        "RVendo_step9", rv_endo_surface, rv_openbase_annulus, args
    )

    _, distances_to_lv_epi = closest_points_on_surface(lv_epi, centers)
    lv_epi_face_mask = (tags == int(args.lv_tag)) & (distances_to_lv_epi <= float(args.step9_lvepi_face_match_tolerance))
    inside_rv_epi = base.enclosed_mask(centers, rv_epi_ref)
    inside_rv_endo = base.enclosed_mask(centers, rv_endo_ref)
    candidate_mask = lv_epi_face_mask & inside_rv_epi & (~inside_rv_endo)
    candidate_source = "LVepi_faces_from_saved_whole_epi_endo_with_centers_inside_original_RVepi_and_outside_original_RVendo"

    if not np.any(candidate_mask):
        loop_point_sets = [loop_points(loop) for loop in list(rv_epi_gap_loops) + list(rv_endo_gap_loops)]
        loop_point_sets = [pts for pts in loop_point_sets if len(pts)]
        if not loop_point_sets:
            raise RuntimeError("No RV/LVepi boundary loop points available for LVepi patch fallback selection.")
        all_loop_points = np.vstack(loop_point_sets)
        loop_distances = cKDTree(all_loop_points).query(centers, k=1)[0]
        candidate_mask = lv_epi_face_mask & (loop_distances <= float(args.step9_lvepi_patch_loop_tolerance))
        candidate_source = "fallback_LVepi_faces_from_saved_whole_epi_endo_near_RV_LVepi_boundary_loops"

    components = connected_face_components(candidate_mask, triangles)
    if not components:
        raise RuntimeError("No LVepi patch component found between RVendo/RVepi intersection boundaries.")

    areas = face_areas(points, triangles)
    component_records: List[Dict[str, object]] = []
    selected_components: List[np.ndarray] = []
    loop_tolerance = float(args.step9_lvepi_patch_loop_tolerance)
    for component_id, component in enumerate(components):
        component_vertices = np.unique(triangles[component].ravel())
        component_points = points[component_vertices]
        epi_distances = [loop_to_point_cloud_distance_record(loop, component_points) for loop in rv_epi_gap_loops]
        endo_distances = [loop_to_point_cloud_distance_record(loop, component_points) for loop in rv_endo_gap_loops]
        epi_p95 = [float(record["p95"]) for record in epi_distances if record["p95"] is not None]
        endo_p95 = [float(record["p95"]) for record in endo_distances if record["p95"] is not None]
        close_to_epi = bool(epi_p95 and min(epi_p95) <= loop_tolerance)
        close_to_endo = bool(endo_p95 and min(endo_p95) <= loop_tolerance)
        record = {
            "component_id": int(component_id),
            "faces": int(len(component)),
            "vertices": int(len(component_vertices)),
            "area": float(areas[component].sum()),
            "close_to_RVepi_LVepi_boundary": close_to_epi,
            "close_to_RVendo_LVepi_boundary": close_to_endo,
            "RVepi_boundary_distances": epi_distances,
            "RVendo_boundary_distances": endo_distances,
        }
        component_records.append(record)
        if close_to_epi and close_to_endo:
            selected_components.append(component)

    selection_method = "connected_LVepi_components_between_RVepi_and_RVendo_LVepi_boundaries"
    if not selected_components:
        selected_components = [components[0]]
        selection_method = "fallback_largest_connected_LVepi_candidate_component"

    max_selected_area = max(float(areas[component].sum()) for component in selected_components)
    min_selected_area = max_selected_area * float(args.step9_lvepi_patch_component_min_fraction)
    selected_components = [component for component in selected_components if float(areas[component].sum()) >= min_selected_area]

    remove_mask = np.zeros(len(triangles), dtype=bool)
    for component in selected_components:
        remove_mask[component] = True
    if not np.any(remove_mask):
        raise RuntimeError("LVepi patch selection produced an empty removal mask.")

    kept_surface = compact_tagged_triangle_surface(points, triangles[~remove_mask], tags[~remove_mask])
    patch_surface = compact_tagged_triangle_surface(points, triangles[remove_mask], tags[remove_mask])
    info = {
        "method": "derive_from_saved_whole_epi_endo_and_remove_LVepi_patch_between_RVepi_LVepi_and_RVendo_LVepi_boundaries",
        "source_surface": "saved_stage09_whole_epi_endo_surface_vtk",
        "candidate_source": candidate_source,
        "whole_surface": surface_summary(whole),
        "whole_surface_elemTag_counts": {str(k): int(v) for k, v in sorted(Counter(map(int, tags)).items())},
        "LVepi_face_match_tolerance_mm": float(args.step9_lvepi_face_match_tolerance),
        "RV_boundary_loop_tolerance_mm": loop_tolerance,
        "LVepi_faces_in_saved_whole_surface": int(np.count_nonzero(lv_epi_face_mask)),
        "candidate_patch_faces_before_component_selection": int(np.count_nonzero(candidate_mask)),
        "candidate_component_count": int(len(components)),
        "selected_component_count": int(len(selected_components)),
        "selection_method": selection_method,
        "removed_patch_faces": int(np.count_nonzero(remove_mask)),
        "removed_patch_area": float(areas[remove_mask].sum()),
        "rv_epi_intersection_boundary_detection": rv_epi_loop_info,
        "rv_endo_intersection_boundary_detection": rv_endo_loop_info,
        "component_records": component_records[:30],
        "output_surface": surface_summary(kept_surface),
        "removed_patch_surface": surface_summary(patch_surface),
    }
    return kept_surface, patch_surface, info


def point_segment_distance_2d(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    denom = float(np.dot(vector, vector))
    if denom <= 1e-20:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, vector) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * vector)))


def point_inside_or_on_polygon_2d(point: np.ndarray, polygon: np.ndarray, tolerance: float = 1e-8) -> bool:
    if len(polygon) < 3:
        return False
    for index in range(len(polygon)):
        if point_segment_distance_2d(point, polygon[index], polygon[(index + 1) % len(polygon)]) <= tolerance:
            return True

    x, y = float(point[0]), float(point[1])
    inside = False
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        xi, yi = float(polygon[current, 0]), float(polygon[current, 1])
        xj, yj = float(polygon[previous, 0]), float(polygon[previous, 1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersection = (xj - xi) * (y - yi) / max(yj - yi, 1e-20) + xi
            if x < x_intersection:
                inside = not inside
        previous = current
    return inside


def clean_loop_points(points: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    cleaned: List[np.ndarray] = []
    for point in np.asarray(points, dtype=float):
        if not cleaned or np.linalg.norm(point - cleaned[-1]) > tolerance:
            cleaned.append(point.copy())
    if len(cleaned) > 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tolerance:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def fan_cap_loop_points(loop_points_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    pts = clean_loop_points(loop_points_array)
    if len(pts) < 3:
        raise RuntimeError("Cannot fan-cap a boundary loop with fewer than three points.")
    center = pts.mean(axis=0, keepdims=True)
    out_points = np.vstack([pts, center])
    center_id = len(pts)
    triangles = np.asarray([[index, (index + 1) % len(pts), center_id] for index in range(len(pts))], dtype=np.int64)
    return out_points, triangles, {
        "method": "fallback_center_fan_cap",
        "loop_points": int(len(pts)),
        "triangles": int(len(triangles)),
    }


def triangulate_boundary_loop_points(loop_points_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    pts = clean_loop_points(loop_points_array)
    if len(pts) < 3:
        raise RuntimeError("Cannot triangulate a boundary loop with fewer than three points.")
    if len(pts) == 3:
        return pts, np.asarray([[0, 1, 2]], dtype=np.int64), {
            "method": "single_triangle_loop_cap",
            "loop_points": 3,
            "triangles": 1,
            "fallback_used": False,
        }

    centered = pts - pts.mean(axis=0)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if len(singular_values) < 2 or float(singular_values[1]) <= 1e-10:
            raise RuntimeError("Boundary loop is nearly collinear in best-fit plane.")
        basis = vh[:2].T
        coords = centered @ basis
        if len(np.unique(np.round(coords, decimals=8), axis=0)) < 3:
            raise RuntimeError("Boundary loop projection has fewer than three unique points.")
        delaunay = Delaunay(coords)
        polygon_tolerance = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])), 1.0) * 1e-8
        selected: List[List[int]] = []
        rejected = 0
        for simplex in np.asarray(delaunay.simplices, dtype=np.int64):
            tri_coords = coords[simplex]
            centroid = tri_coords.mean(axis=0)
            midpoints = np.asarray([
                0.5 * (tri_coords[0] + tri_coords[1]),
                0.5 * (tri_coords[1] + tri_coords[2]),
                0.5 * (tri_coords[2] + tri_coords[0]),
            ])
            probes = [centroid, *midpoints]
            if all(point_inside_or_on_polygon_2d(probe, coords, polygon_tolerance) for probe in probes):
                area = 0.5 * float(np.linalg.norm(np.cross(pts[simplex[1]] - pts[simplex[0]], pts[simplex[2]] - pts[simplex[0]])))
                if area > 1e-10:
                    selected.append([int(v) for v in simplex])
                else:
                    rejected += 1
            else:
                rejected += 1
        if not selected:
            raise RuntimeError("Delaunay loop triangulation produced no interior triangles.")
        triangles = np.asarray(selected, dtype=np.int64)
        expected_edges = {tuple(sorted((index, (index + 1) % len(pts)))) for index in range(len(pts))}
        observed_edge_map = triangle_edge_face_map(triangles)
        observed_boundary_edges = {edge for edge, owners in observed_edge_map.items() if len(owners) == 1}
        if observed_boundary_edges != expected_edges:
            missing = len(expected_edges - observed_boundary_edges)
            extra = len(observed_boundary_edges - expected_edges)
            raise RuntimeError(
                "Delaunay loop triangulation did not close exactly on the source loop "
                f"(missing_boundary_edges={missing}, extra_boundary_edges={extra})."
            )
        return pts, triangles, {
            "method": "best_fit_plane_delaunay_loop_cap",
            "loop_points": int(len(pts)),
            "triangles": int(len(triangles)),
            "rejected_delaunay_triangles": int(rejected),
            "fallback_used": False,
            "best_fit_plane_singular_values": [float(v) for v in singular_values[:3]],
        }
    except Exception as exc:
        cap_points, cap_triangles, cap_info = fan_cap_loop_points(pts)
        cap_info.update({
            "fallback_used": True,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        })
        return cap_points, cap_triangles, cap_info


def loop_pair_distance_score(first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    first = clean_loop_points(first)
    second = clean_loop_points(second)
    if len(first) == 0 or len(second) == 0:
        return {"score": float("inf"), "mean": float("inf"), "p95": float("inf"), "max": float("inf"), "centroid": float("inf")}
    first_tree = cKDTree(first)
    second_tree = cKDTree(second)
    first_to_second = second_tree.query(first, k=1)[0]
    second_to_first = first_tree.query(second, k=1)[0]
    centroid = float(np.linalg.norm(first.mean(axis=0) - second.mean(axis=0)))
    p95 = max(float(np.percentile(first_to_second, 95)), float(np.percentile(second_to_first, 95)))
    mean = 0.5 * (float(np.mean(first_to_second)) + float(np.mean(second_to_first)))
    max_distance = max(float(np.max(first_to_second)), float(np.max(second_to_first)))
    return {
        "score": float(p95 + 0.05 * centroid),
        "mean": mean,
        "p95": p95,
        "max": max_distance,
        "centroid": centroid,
    }


def annulus_between_boundary_loop_points(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    first_pts = clean_loop_points(first)
    second_pts = clean_loop_points(second)
    if len(first_pts) < 3 or len(second_pts) < 3:
        raise RuntimeError("Cannot stitch boundary loops with fewer than three points.")
    annulus = base.zipper_annulus(base.make_polyline(first_pts), base.make_polyline(second_pts))
    annulus = annulus.extract_surface().triangulate().clean(tolerance=1e-7)
    triangles = base.triangle_faces(annulus)
    if len(triangles) == 0:
        raise RuntimeError("Boundary-loop annulus stitching produced no triangles.")
    return np.asarray(annulus.points, dtype=float), triangles, {
        "method": "zipper_annulus_between_paired_patch_and_RV_boundary_loops",
        "first_loop_points": int(len(first_pts)),
        "second_loop_points": int(len(second_pts)),
        "annulus_points": int(annulus.n_points),
        "annulus_triangles": int(len(triangles)),
        "annulus_free_edges_before_merge": int(len(base.extract_boundary_edges(annulus))),
    }


def greedy_pair_loop_ids(primary_ids: Sequence[int], secondary_ids: Sequence[int], loop_points_by_id: Dict[int, np.ndarray]) -> List[Tuple[int, int, Dict[str, float]]]:
    available_primary = set(int(v) for v in primary_ids)
    available_secondary = set(int(v) for v in secondary_ids)
    pairs: List[Tuple[int, int, Dict[str, float]]] = []
    while available_primary and available_secondary:
        candidates: List[Tuple[float, int, int, Dict[str, float]]] = []
        for first_id in available_primary:
            for second_id in available_secondary:
                score_record = loop_pair_distance_score(loop_points_by_id[first_id], loop_points_by_id[second_id])
                candidates.append((float(score_record["score"]), int(first_id), int(second_id), score_record))
        candidates.sort(key=lambda item: item[0])
        _, first_id, second_id, score_record = candidates[0]
        pairs.append((first_id, second_id, score_record))
        available_primary.remove(first_id)
        available_secondary.remove(second_id)
    return pairs


def fill_lvepi_patch_removed_holes(
    surface: pv.PolyData,
    removed_patch: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    surface, points, triangles, _, tags = tagged_surface_face_arrays(surface)
    free_edges_before = int(len(base.extract_boundary_edges(surface)))
    boundary_loops = base.boundary_loops(surface)
    patch_boundary_loops = base.boundary_loops(removed_patch.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance))
    patch_boundary_points = [loop_points(loop) for loop in patch_boundary_loops]
    patch_boundary_points = [pts for pts in patch_boundary_points if len(pts) >= 3]
    if free_edges_before == 0 or not boundary_loops or not patch_boundary_points or not bool(args.step9_fill_lvepi_patch_holes):
        return surface, {
            "enabled": bool(args.step9_fill_lvepi_patch_holes),
            "method": "pair_patch_and_RV_boundary_loops_then_stitch_annuli",
            "applied": False,
            "reason": "no_matching_input_boundary" if free_edges_before else "surface_already_has_no_free_edges",
            "free_edges_before": free_edges_before,
            "boundary_loop_count_before": int(len(boundary_loops)),
            "removed_patch_boundary_loop_count": int(len(patch_boundary_points)),
            "output_surface": surface_summary(surface),
        }

    patch_tree = cKDTree(np.vstack(patch_boundary_points))
    match_tolerance = max(
        float(args.step9_hole_fill_boundary_match_tolerance),
        float(args.boundary_match_tolerance),
        float(args.surface_clean_tolerance) * 10.0,
    )
    exact_patch_tolerance = max(float(args.boundary_match_tolerance) * 10.0, float(args.surface_clean_tolerance) * 100.0, 1e-4)
    selected_loop_ids: List[int] = []
    exact_patch_loop_ids: List[int] = []
    partner_loop_ids: List[int] = []
    loop_points_by_id: Dict[int, np.ndarray] = {}
    loop_records: List[Dict[str, object]] = []

    for loop_id, loop in enumerate(boundary_loops):
        pts = clean_loop_points(loop_points(loop))
        if len(pts) < 3:
            loop_records.append({
                "loop_id": int(loop_id),
                "points": int(len(pts)),
                "selected_for_fill": False,
                "reason": "fewer_than_3_points",
            })
            continue
        distances = patch_tree.query(pts, k=1)[0]
        p95 = float(np.percentile(distances, 95)) if len(distances) else float("inf")
        max_distance = float(np.max(distances)) if len(distances) else float("inf")
        selected = p95 <= match_tolerance
        exact_patch_boundary = p95 <= exact_patch_tolerance and max_distance <= max(exact_patch_tolerance * 10.0, 1e-3)
        loop_points_by_id[int(loop_id)] = pts
        if selected:
            selected_loop_ids.append(int(loop_id))
            if exact_patch_boundary:
                exact_patch_loop_ids.append(int(loop_id))
            else:
                partner_loop_ids.append(int(loop_id))
        loop_records.append({
            "loop_id": int(loop_id),
            "points": int(len(pts)),
            "length": float(loop.length),
            "area_best_plane": float(abs(base.signed_area_in_best_plane(pts))),
            "distance_to_removed_patch_boundary_min": float(np.min(distances)) if len(distances) else None,
            "distance_to_removed_patch_boundary_mean": float(np.mean(distances)) if len(distances) else None,
            "distance_to_removed_patch_boundary_p95": p95,
            "distance_to_removed_patch_boundary_max": max_distance,
            "selected_for_fill": bool(selected),
            "classified_as_exact_removed_patch_boundary": bool(exact_patch_boundary),
        })

    if not selected_loop_ids:
        return surface, {
            "enabled": bool(args.step9_fill_lvepi_patch_holes),
            "method": "pair_patch_and_RV_boundary_loops_then_stitch_annuli",
            "applied": False,
            "reason": "no_surface_boundary_loop_matched_removed_patch_boundary",
            "match_tolerance_mm": float(match_tolerance),
            "exact_patch_tolerance_mm": float(exact_patch_tolerance),
            "free_edges_before": free_edges_before,
            "boundary_loop_count_before": int(len(boundary_loops)),
            "removed_patch_boundary_loop_count": int(len(patch_boundary_points)),
            "loop_records": loop_records,
            "output_surface": surface_summary(surface),
        }

    pairs: List[Tuple[int, int, Dict[str, float]]] = []
    if exact_patch_loop_ids and partner_loop_ids:
        pairs = greedy_pair_loop_ids(exact_patch_loop_ids, partner_loop_ids, loop_points_by_id)
        pair_method = "pair_exact_removed_patch_boundary_loops_to_nearby_RV_boundary_loops"
    else:
        unpaired_ids = set(selected_loop_ids)
        pair_method = "fallback_greedy_pair_all_selected_boundary_loops"
        while len(unpaired_ids) >= 2:
            candidates: List[Tuple[float, int, int, Dict[str, float]]] = []
            ids = sorted(unpaired_ids)
            for pos, first_id in enumerate(ids):
                for second_id in ids[pos + 1:]:
                    score_record = loop_pair_distance_score(loop_points_by_id[first_id], loop_points_by_id[second_id])
                    candidates.append((float(score_record["score"]), int(first_id), int(second_id), score_record))
            candidates.sort(key=lambda item: item[0])
            _, first_id, second_id, score_record = candidates[0]
            pairs.append((first_id, second_id, score_record))
            unpaired_ids.remove(first_id)
            unpaired_ids.remove(second_id)

    paired_ids = {loop_id for pair in pairs for loop_id in pair[:2]}
    unpaired_selected_ids = [loop_id for loop_id in selected_loop_ids if loop_id not in paired_ids]

    fill_points_blocks: List[np.ndarray] = [points]
    fill_triangles_blocks: List[np.ndarray] = [triangles]
    fill_tags_blocks: List[np.ndarray] = [tags]
    fill_operations: List[Dict[str, object]] = []
    point_offset = len(points)
    added_triangles = 0
    added_points = 0

    for pair_id, (first_id, second_id, score_record) in enumerate(pairs):
        try:
            fill_points, fill_triangles, fill_info = annulus_between_boundary_loop_points(
                loop_points_by_id[first_id],
                loop_points_by_id[second_id],
            )
        except Exception as exc:
            fill_points, fill_triangles, fill_info = fan_cap_loop_points(loop_points_by_id[first_id])
            fill_info.update({
                "method": "fallback_center_fan_cap_for_first_loop_after_annulus_failure",
                "annulus_failure": f"{type(exc).__name__}: {exc}",
            })
            unpaired_selected_ids.append(second_id)
        fill_points_blocks.append(fill_points)
        fill_triangles_blocks.append(fill_triangles + point_offset)
        fill_tags_blocks.append(np.full(len(fill_triangles), int(args.lv_tag), dtype=np.int32))
        point_offset += len(fill_points)
        added_points += int(len(fill_points))
        added_triangles += int(len(fill_triangles))
        fill_operations.append({
            "operation": "stitch_boundary_loop_pair",
            "pair_id": int(pair_id),
            "first_loop_id": int(first_id),
            "second_loop_id": int(second_id),
            "pair_score": score_record,
            "fill": fill_info,
        })

    for loop_id in unpaired_selected_ids:
        fill_points, fill_triangles, fill_info = fan_cap_loop_points(loop_points_by_id[loop_id])
        fill_points_blocks.append(fill_points)
        fill_triangles_blocks.append(fill_triangles + point_offset)
        fill_tags_blocks.append(np.full(len(fill_triangles), int(args.lv_tag), dtype=np.int32))
        point_offset += len(fill_points)
        added_points += int(len(fill_points))
        added_triangles += int(len(fill_triangles))
        fill_operations.append({
            "operation": "fallback_cap_unpaired_boundary_loop",
            "loop_id": int(loop_id),
            "fill": fill_info,
        })

    if not fill_operations:
        return surface, {
            "enabled": bool(args.step9_fill_lvepi_patch_holes),
            "method": "pair_patch_and_RV_boundary_loops_then_stitch_annuli",
            "applied": False,
            "reason": "selected_boundary_loops_could_not_be_paired_or_filled",
            "match_tolerance_mm": float(match_tolerance),
            "exact_patch_tolerance_mm": float(exact_patch_tolerance),
            "free_edges_before": free_edges_before,
            "boundary_loop_count_before": int(len(boundary_loops)),
            "removed_patch_boundary_loop_count": int(len(patch_boundary_points)),
            "selected_boundary_loop_count": int(len(selected_loop_ids)),
            "loop_records": loop_records,
            "output_surface": surface_summary(surface),
        }

    merged_points = np.vstack(fill_points_blocks)
    merged_triangles = np.vstack(fill_triangles_blocks)
    merged_tags = np.concatenate(fill_tags_blocks)
    filled = compact_tagged_triangle_surface(merged_points, merged_triangles, merged_tags)
    free_edges_after = int(len(base.extract_boundary_edges(filled)))
    return filled, {
        "enabled": bool(args.step9_fill_lvepi_patch_holes),
        "method": "pair_patch_and_RV_boundary_loops_then_stitch_annuli",
        "applied": True,
        "pair_method": pair_method,
        "fill_tag": int(args.lv_tag),
        "match_tolerance_mm": float(match_tolerance),
        "exact_patch_tolerance_mm": float(exact_patch_tolerance),
        "free_edges_before": free_edges_before,
        "free_edges_after": free_edges_after,
        "free_edges_reduced_by": int(free_edges_before - free_edges_after),
        "boundary_loop_count_before": int(len(boundary_loops)),
        "removed_patch_boundary_loop_count": int(len(patch_boundary_points)),
        "selected_boundary_loop_count": int(len(selected_loop_ids)),
        "exact_removed_patch_boundary_loop_count": int(len(exact_patch_loop_ids)),
        "partner_boundary_loop_count": int(len(partner_loop_ids)),
        "stitched_pair_count": int(len(pairs)),
        "unpaired_fallback_cap_count": int(len(unpaired_selected_ids)),
        "added_points_before_cleanup": int(added_points),
        "added_triangles_before_cleanup": int(added_triangles),
        "loop_records": loop_records,
        "fill_operations": fill_operations,
        "output_surface": surface_summary(filled),
    }


def save_merge_surface_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    whole_epi: pv.PolyData,
    whole_endo: pv.PolyData,
    whole_epi_endo: pv.PolyData,
    lv_epi: pv.PolyData,
    rv_epi: pv.PolyData,
    rv_endo: pv.PolyData,
    rv_openbase_annulus: pv.PolyData,
    rv_epi_reference: pv.PolyData,
    rv_endo_reference: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}

    stage_items = [
        ("stage09_whole_epi_surface", whole_epi),
        ("stage09_whole_endo_surface", whole_endo),
        ("stage09_whole_epi_endo_surface", whole_epi_endo),
    ]
    for name, mesh in stage_items:
        surface_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = save_triangle_surface_vtk(surface_path, mesh)
        paths[name] = str(surface_path.resolve())

    saved_whole_epi_endo = pv.read(paths["stage09_whole_epi_endo_surface"]).extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lvepi_patch_removed, removed_lvepi_patch, lvepi_patch_info = remove_lvepi_patch_between_rv_boundaries_from_whole_epi_endo(
        saved_whole_epi_endo,
        lv_epi,
        rv_epi,
        rv_endo,
        rv_openbase_annulus,
        rv_epi_reference,
        rv_endo_reference,
        args,
    )
    patch_removed_path = pipeline_dir / f"{subject_id}_{frame}_stage09_whole_epi_endo_without_LVepi_RV_interface_patch.vtk"
    records["stage09_whole_epi_endo_without_LVepi_RV_interface_patch"] = save_triangle_surface_vtk(patch_removed_path, lvepi_patch_removed)
    paths["stage09_whole_epi_endo_without_LVepi_RV_interface_patch"] = str(patch_removed_path.resolve())

    removed_patch_path = pipeline_dir / f"{subject_id}_{frame}_stage09_removed_LVepi_patch_between_RVendo_RVepi_boundaries.vtk"
    records["stage09_removed_LVepi_patch_between_RVendo_RVepi_boundaries"] = save_triangle_surface_vtk(removed_patch_path, removed_lvepi_patch)
    paths["stage09_removed_LVepi_patch_between_RVendo_RVepi_boundaries"] = str(removed_patch_path.resolve())

    lvepi_patch_removed_filled, lvepi_patch_hole_fill_info = fill_lvepi_patch_removed_holes(
        lvepi_patch_removed,
        removed_lvepi_patch,
        args,
    )
    patch_removed_filled_path = pipeline_dir / f"{subject_id}_{frame}_stage09_whole_epi_endo_without_LVepi_RV_interface_patch_filled_holes.vtk"
    records["stage09_whole_epi_endo_without_LVepi_RV_interface_patch_filled_holes"] = save_triangle_surface_vtk(
        patch_removed_filled_path,
        lvepi_patch_removed_filled,
    )
    paths["stage09_whole_epi_endo_without_LVepi_RV_interface_patch_filled_holes"] = str(patch_removed_filled_path.resolve())

    summary = {
        "description": "Step 9 outputs after merging LVepi+RVepi as whole epi, RVendo+rebuilt RV open-base annulus+LVendo+rebuilt LV open-base annulus as whole endo/annulus surface, whole_epi+whole_endo as one combined inspection surface, a derived whole_epi_endo surface with the LVepi patch between RVendo/RVepi intersection boundaries removed, and a hole-filled version of that derived surface.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "lvepi_patch_removed_from_saved_whole_epi_endo": lvepi_patch_info,
        "lvepi_patch_removed_hole_fill": lvepi_patch_hole_fill_info,
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage09_merge_surface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage09_summary"] = str(summary_path.resolve())
    return paths, summary


def save_raw_tetgen_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    overall_surface: pv.PolyData,
    tetgen_surface: pv.PolyData,
    raw_points: np.ndarray,
    raw_cells: np.ndarray,
    raw_region_labels: Optional[np.ndarray],
    tetgen_info: Dict[str, object],
    orientation: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}

    overall_surface_path = pipeline_dir / f"{subject_id}_{frame}_stage10_overall_TetGen_PLC_surface.vtk"
    records["stage10_overall_TetGen_PLC_surface"] = save_triangle_surface_vtk(overall_surface_path, overall_surface)
    paths["stage10_overall_TetGen_PLC_surface"] = str(overall_surface_path.resolve())

    tetgen_surface_path = pipeline_dir / f"{subject_id}_{frame}_stage10_tetgen_surface_used.vtk"
    records["stage10_tetgen_surface_used"] = save_triangle_surface_vtk(tetgen_surface_path, tetgen_surface)
    paths["stage10_tetgen_surface_used"] = str(tetgen_surface_path.resolve())

    whole_tags = np.zeros(len(raw_cells), dtype=np.int32)
    raw_tet_path = pipeline_dir / f"{subject_id}_{frame}_stage10_whole_tetgen_volmesh.vtk"
    write_custom_vtk(raw_tet_path, np.asarray(raw_points, dtype=float), np.asarray(raw_cells, dtype=np.int64), whole_tags)
    paths["stage10_whole_tetgen_volmesh"] = str(raw_tet_path.resolve())

    region_label_counts = None
    if raw_region_labels is not None and len(raw_region_labels) == len(raw_cells):
        region_label_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, raw_region_labels)).items())}

    quality = tet_quality_report(
        "step10_raw_whole_tetgen_before_step11_tagging",
        np.asarray(raw_points, dtype=float),
        np.asarray(raw_cells, dtype=np.int64),
        args,
    )
    quality["orientation"] = orientation

    summary = {
        "description": "Step 10 only: raw whole TetGen volume mesh saved before LV/RV post-classification.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "whole_tetgen_volmesh": {
            "path": str(raw_tet_path.resolve()),
            "points": int(len(raw_points)),
            "tetrahedra": int(len(raw_cells)),
            "cell_scalar_name": "elemTag",
            "cell_scalar_meaning": "Neutral placeholder for unclassified whole TetGen mesh; not LV/RV tags.",
            "elemTag_counts": {"0": int(len(raw_cells))},
            "tetgen_region_label_counts_not_written_to_vtk": region_label_counts,
            "quality": quality,
            "tetgen": tetgen_info,
        },
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage10_whole_tetgen_volmesh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage10_summary"] = str(summary_path.resolve())
    return paths, summary


def save_tetgen_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    overall_surface: pv.PolyData,
    tetgen_surface: pv.PolyData,
    raw_points: np.ndarray,
    raw_cells: np.ndarray,
    raw_region_labels: Optional[np.ndarray],
    final_points: np.ndarray,
    final_cells: np.ndarray,
    final_elem_tags: np.ndarray,
    tetgen_info: Dict[str, object],
    tag_info: Dict[str, object],
    component_filter: Dict[str, object],
    components: Dict[str, object],
    quality: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}

    overall_surface_path = pipeline_dir / f"{subject_id}_{frame}_stage10_overall_TetGen_PLC_surface.vtk"
    records["stage10_overall_TetGen_PLC_surface"] = save_triangle_surface_vtk(overall_surface_path, overall_surface)
    paths["stage10_overall_TetGen_PLC_surface"] = str(overall_surface_path.resolve())

    tetgen_surface_path = pipeline_dir / f"{subject_id}_{frame}_stage10_tetgen_surface_used.vtk"
    records["stage10_tetgen_surface_used"] = save_triangle_surface_vtk(tetgen_surface_path, tetgen_surface)
    paths["stage10_tetgen_surface_used"] = str(tetgen_surface_path.resolve())

    if raw_region_labels is not None and len(raw_region_labels) == len(raw_cells):
        raw_tags = np.asarray(raw_region_labels, dtype=np.int32)
        raw_scalar_meaning = "TetGen region attribute from add_region seeds; not the final LV/RV elemTag."
    else:
        raw_tags = np.zeros(len(raw_cells), dtype=np.int32)
        raw_scalar_meaning = "No TetGen region attributes were returned; scalar values are placeholder zeros."
    raw_tet_path = pipeline_dir / f"{subject_id}_{frame}_stage10_raw_tetgen_region_volmesh.vtk"
    write_custom_vtk(raw_tet_path, np.asarray(raw_points, dtype=float), np.asarray(raw_cells, dtype=np.int64), raw_tags)
    paths["stage10_raw_tetgen_region_volmesh"] = str(raw_tet_path.resolve())

    final_tet_path = pipeline_dir / f"{subject_id}_{frame}_stage11_final_elemTag_volmesh.vtk"
    write_custom_vtk(final_tet_path, np.asarray(final_points, dtype=float), np.asarray(final_cells, dtype=np.int64), np.asarray(final_elem_tags, dtype=np.int32))
    paths["stage11_final_elemTag_volmesh"] = str(final_tet_path.resolve())

    filled_holes_tagged_tet_path = pipeline_dir / f"{subject_id}_{frame}_stage10_filled_holes_elemTag_volmesh.vtk"
    write_custom_vtk(
        filled_holes_tagged_tet_path,
        np.asarray(final_points, dtype=float),
        np.asarray(final_cells, dtype=np.int64),
        np.asarray(final_elem_tags, dtype=np.int32),
    )
    paths["stage10_filled_holes_elemTag_volmesh"] = str(filled_holes_tagged_tet_path.resolve())

    legend_path = pipeline_dir / f"{subject_id}_{frame}_stage11_elemTag_legend.csv"
    write_elem_tag_legend(legend_path, args.lv_tag, args.rv_tag)
    paths["stage11_elemTag_legend"] = str(legend_path.resolve())

    raw_quality = tet_quality_report("raw_tetgen_before_step11_tagging", np.asarray(raw_points, dtype=float), np.asarray(raw_cells, dtype=np.int64), args)
    raw_label_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, raw_tags)).items())}
    final_tag_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, final_elem_tags)).items())}
    summary = {
        "description": "Step 10/11 outputs: TetGen PLC surface, TetGen-used surface, raw TetGen volume, and final custom LV/RV elemTag volume mesh.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "step10": {
            "operation": "tetrahedralize_whole_open_base_biventricular_PLC_with_TetGen",
            "raw_tetgen_points": int(len(raw_points)),
            "raw_tetgen_tetrahedra": int(len(raw_cells)),
            "raw_scalar_name_in_vtk": "elemTag",
            "raw_scalar_meaning": raw_scalar_meaning,
            "raw_region_label_counts": raw_label_counts,
            "raw_quality": raw_quality,
            "tetgen": tetgen_info,
        },
        "step11": {
            "operation": "classify_tet_centroids_inside_original_closed_LVepi_as_elemTag_1_else_RV_elemTag_0_then_write_custom_VTK",
            "final_points": int(len(final_points)),
            "final_tetrahedra": int(len(final_cells)),
            "final_elemTag_counts": final_tag_counts,
            "tagging": tag_info,
            "component_filter": component_filter,
            "components": components,
            "quality": quality,
            "elemTag_legend": {
                str(int(args.lv_tag)): "LV myocardium",
                str(int(args.rv_tag)): "RV myocardium",
            },
        },
    }
    summary_path = pipeline_dir / f"{subject_id}_{frame}_stage10_11_tetgen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage10_11_summary"] = str(summary_path.resolve())
    return paths, summary


def process_frame(args: argparse.Namespace, frame: str) -> Dict[str, object]:
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    vtk_dir = resolve_vtk_dir(subject_dir, args.vtk_dir)
    output_dir = resolve_output_dir(subject_dir, args.output_root)
    open_shell_stage_dir = resolve_pipeline_output_dir(subject_dir, args.pipeline_output_root)
    intersection_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step5remove_intersection")
    rvendo_distance_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step7moveRVendo")
    fix_new_surface_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step8fix_intersection_new_surf")
    merge_surface_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step9merge_surface")
    tetgen_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step10tetgen")
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

    log_stage(f"{subject_id} {frame}: keep RVepi fixed and adaptively move/smooth RVendo open-base ring to >=3 mm distance")
    rv_endo_fixed, rv_endo_moved_mask, rv_endo_opening_mask, rv_endo_transition_band_mask, rv_endo_step7_opening_loop, rv_distance_fix = enforce_rvendo_distance_adaptive_opening_ring(
        rv_endo_trimmed,
        rv_epi_trimmed,
        rv_build.endo_opening_loop,
        args,
    )
    rv_step7_epi_opening_loop, rv_step7_endo_opening_loop, rv_step7_annulus, rv_step7_annulus_info = rebuild_rv_openbase_annulus_after_step7(
        rv_epi_trimmed,
        rv_endo_fixed,
        rv_build.epi_opening_loop,
        rv_endo_step7_opening_loop,
        args,
    )
    rv_distance_fix["rv_openbase_annulus_rebuild"] = rv_step7_annulus_info

    rvendo_distance_stage_paths: Dict[str, str] = {}
    rvendo_distance_stage_summary: Dict[str, object] = {}
    if args.save_rvendo_distance_stage:
        log_stage(f"{subject_id} {frame}: save step 7 RVepi/RVendo/rebuilt-annulus triangle-surface outputs")
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
            rv_endo_transition_band_mask,
            rv_step7_annulus,
            rv_step7_annulus_info,
        )

    if args.stop_after_rvendo_distance:
        return {
            "frame": frame,
            "stopped_after": "step07_move_RVendo",
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
            "rvendo_distance_stage_outputs": rvendo_distance_stage_paths,
        }

    log_stage(f"{subject_id} {frame}: step 8 re-fix RV/LVepi intersections and rebuild LV/RV open-base annuli")
    step8_surfaces, step8_masks, step8_fix_info = fix_intersections_and_rebuild_new_surfaces(
        lv_epi_open,
        lv_build.endo_open,
        rv_epi_trimmed,
        rv_endo_fixed,
        lv_epi_intersection_reference,
        lv_profile,
        rv_profile,
        args,
    )
    lv_endo_step8 = step8_surfaces["LVendo"]
    lv_epi_step8 = step8_surfaces["LVepi"]
    rv_endo_step8 = step8_surfaces["RVendo"]
    rv_epi_step8 = step8_surfaces["RVepi"]
    lv_step8_annulus = step8_surfaces["LV_openbase_annulus"]
    rv_step8_annulus = step8_surfaces["RV_openbase_annulus"]
    lv_shell = merge_surfaces([tagged_surface(lv_epi_step8, 1), tagged_surface(lv_endo_step8, 1), tagged_surface(lv_step8_annulus, 1)], args.combined_surface_merge_tolerance)

    fix_new_surface_stage_paths: Dict[str, str] = {}
    fix_new_surface_stage_summary: Dict[str, object] = {}
    if args.save_fix_intersection_new_surf_stage:
        log_stage(f"{subject_id} {frame}: save step 8 fixed surface and rebuilt-annulus outputs")
        fix_new_surface_stage_paths, fix_new_surface_stage_summary = save_fix_intersection_new_surface_stage_outputs(
            fix_new_surface_stage_dir,
            subject_id,
            frame,
            step8_surfaces,
            step8_masks,
            step8_fix_info,
        )

    if args.stop_after_fix_intersection_new_surf:
        return {
            "frame": frame,
            "stopped_after": "step08_fix_intersection_new_surface",
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
            "rvendo_distance_stage_outputs": rvendo_distance_stage_paths,
            "fix_intersection_new_surface_stage_outputs": fix_new_surface_stage_paths,
        }

    log_stage(f"{subject_id} {frame}: step 9 merge whole epi and whole endo surfaces")
    whole_epi = merge_surfaces([tagged_surface(lv_epi_step8, 1), tagged_surface(rv_epi_step8, 0)], args.combined_surface_merge_tolerance)
    whole_endo = merge_surfaces(
        [tagged_surface(rv_endo_step8, 0), tagged_surface(rv_step8_annulus, 0), tagged_surface(lv_endo_step8, 1), tagged_surface(lv_step8_annulus, 1)],
        args.combined_surface_merge_tolerance,
    )
    whole_epi_endo = merge_surfaces([whole_epi, whole_endo], args.combined_surface_merge_tolerance)

    merge_surface_stage_paths: Dict[str, str] = {}
    merge_surface_stage_summary: Dict[str, object] = {}
    if args.save_merge_surface_stage:
        log_stage(f"{subject_id} {frame}: save step 9 whole epi/endo/combined triangle-surface outputs")
        merge_surface_stage_paths, merge_surface_stage_summary = save_merge_surface_stage_outputs(
            merge_surface_stage_dir,
            subject_id,
            frame,
            whole_epi,
            whole_endo,
            whole_epi_endo,
            lv_epi_step8,
            rv_epi_step8,
            rv_endo_step8,
            rv_step8_annulus,
            surfaces["rv_epi"],
            surfaces["rv_endo"],
            args,
        )

    if args.stop_after_merge_surface:
        return {
            "frame": frame,
            "stopped_after": "step09_merge_surface",
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
            "rvendo_distance_stage_outputs": rvendo_distance_stage_paths,
            "fix_intersection_new_surface_stage_outputs": fix_new_surface_stage_paths,
            "merge_surface_stage_outputs": merge_surface_stage_paths,
        }

    filled_holes_key = "stage09_whole_epi_endo_without_LVepi_RV_interface_patch_filled_holes"
    filled_holes_path_text = merge_surface_stage_paths.get(filled_holes_key)
    if not filled_holes_path_text:
        raise RuntimeError(
            "Step 10 requires the Step 9 filled-holes VTK file. "
            "Keep --save-merge-surface-stage enabled so the file is written before TetGen."
        )
    filled_holes_path = Path(filled_holes_path_text)
    if not filled_holes_path.exists():
        raise FileNotFoundError(f"Step 10 TetGen input surface does not exist: {filled_holes_path}")

    log_stage(f"{subject_id} {frame}: load Step 9 filled-holes surface as TetGen input: {filled_holes_path.name}")
    overall_surface = pv.read(filled_holes_path).extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    overall_surface_source = {
        "method": "use_saved_step09_filled_holes_surface_directly_as_step10_TetGen_PLC_input",
        "path": str(filled_holes_path.resolve()),
        "surface": surface_summary(overall_surface),
        "free_edges": int(len(base.extract_boundary_edges(overall_surface))),
        "lv_preservation_report_not_used_as_rejection_gate": lv_surface_preservation_report(
            overall_surface,
            lv_shell,
            args.lv_preservation_tolerance,
        ),
        "expected_cell_data": "elemTag from Step 9 surface; TetGen does not preserve surface tags onto tetrahedra, so Step 11 reassigns volume elemTag values by LV reference classification.",
    }
    rv_cut_surfaces: List[pv.PolyData] = []
    rv_cut_info = {
        "method": "not_used",
        "reason": "Step 10 now uses the saved Step 9 filled-holes whole_epi_endo surface directly as the TetGen PLC input.",
        "replaced_by": overall_surface_source,
    }

    lv_seed, lv_seed_info = midwall_seed(lv_epi_step8, lv_endo_step8)
    rv_seed, rv_seed_info = midwall_seed(rv_epi_step8, rv_endo_step8)

    debug_paths: Dict[str, str] = {}
    if args.save_debug:
        debug_surfaces = {
            "LV_shell_reference": lv_shell,
            "LVepi_open_before_step8": lv_epi_open,
            "LVepi_closed_intersection_reference": lv_epi_intersection_reference,
            "LVendo_open_before_step8": lv_build.endo_open,
            "LVepi_step8_fixed": lv_epi_step8,
            "LVendo_step8_fixed": lv_endo_step8,
            "LV_openbase_annulus_step8_rebuilt": lv_step8_annulus,
            "RV_initial_shell": rv_build.shell,
            "RVepi_after_step5_intersection_removal": rv_epi_trimmed,
            "RVendo_after_step6_intersection_removal": rv_endo_trimmed,
            "RVendo_after_step7_3mm_fix": rv_endo_fixed,
            "RV_openbase_annulus_after_step7": rv_step7_annulus,
            "RVepi_step8_fixed": rv_epi_step8,
            "RVendo_step8_fixed": rv_endo_step8,
            "RV_openbase_annulus_step8_rebuilt": rv_step8_annulus,
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
        lv_shell_reference=None,
    )
    tetgen_info["input_surface_source"] = overall_surface_source

    raw_tet_points = np.asarray(tet_points, dtype=float).copy()
    raw_tet_cells = np.asarray(tet_cells, dtype=np.int64).copy()
    raw_region_labels = None if region_labels is None else np.asarray(region_labels, dtype=np.int32).copy()
    raw_tet_cells, raw_orientation = reorient_tetrahedra(raw_tet_points, raw_tet_cells)

    tetgen_stage_paths: Dict[str, str] = {}
    tetgen_stage_summary: Dict[str, object] = {}
    if args.save_tetgen_stage:
        log_stage(f"{subject_id} {frame}: save step 10 raw whole TetGen volume mesh output")
        tetgen_stage_paths, tetgen_stage_summary = save_raw_tetgen_stage_outputs(
            tetgen_stage_dir,
            subject_id,
            frame,
            overall_surface,
            tetgen_surface,
            raw_tet_points,
            raw_tet_cells,
            raw_region_labels,
            tetgen_info,
            raw_orientation,
            args,
        )

    if args.stop_after_tetgen:
        return {
            "frame": frame,
            "stopped_after": "step10_raw_tetgen_whole_volume",
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
            "rvendo_distance_stage_outputs": rvendo_distance_stage_paths,
            "fix_intersection_new_surface_stage_outputs": fix_new_surface_stage_paths,
            "merge_surface_stage_outputs": merge_surface_stage_paths,
            "tetgen_stage_outputs": tetgen_stage_paths,
        }

    log_stage(f"{subject_id} {frame}: assign elemTag and write VTK")
    tet_cells, orientation = reorient_tetrahedra(tet_points, tet_cells)
    elem_tags, tag_info = classify_by_lv_reference_surface(tet_points, tet_cells, lv_epi_intersection_reference, args)
    tet_points, tet_cells, elem_tags, component_filter = drop_small_tet_components(tet_points, tet_cells, elem_tags, args)
    quality = tet_quality_report("whole_biventricular_tetgen", tet_points, tet_cells, args)
    quality["orientation"] = orientation
    components = component_report(tet_cells, elem_tags)

    output_mesh = output_dir / f"{subject_id}_{frame}_openbase_LVRV_tetgen_volmesh.vtk"
    write_custom_vtk(output_mesh, tet_points, tet_cells, elem_tags)
    legend_path = output_dir / f"{subject_id}_{frame}_tetgen_elemTag_legend.csv"
    write_elem_tag_legend(legend_path, args.lv_tag, args.rv_tag)

    tetgen_stage_paths: Dict[str, str] = {}
    tetgen_stage_summary: Dict[str, object] = {}
    if args.save_tetgen_stage:
        log_stage(f"{subject_id} {frame}: save step 10/11 TetGen and final tagged mesh outputs")
        tetgen_stage_paths, tetgen_stage_summary = save_tetgen_stage_outputs(
            tetgen_stage_dir,
            subject_id,
            frame,
            overall_surface,
            tetgen_surface,
            raw_tet_points,
            raw_tet_cells,
            raw_region_labels,
            tet_points,
            tet_cells,
            elem_tags,
            tetgen_info,
            tag_info,
            component_filter,
            components,
            quality,
            args,
        )

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
            "fix_intersection_new_surface_stage_outputs": fix_new_surface_stage_paths,
            "merge_surface_stage_outputs": merge_surface_stage_paths,
            "tetgen_stage_outputs": tetgen_stage_paths,
            "debug": debug_paths,
        },
        "pipeline": [
            "load_LVendo_LVepi_RV_RVepi_surfaces",
            "detect_subject_base_side_and_write_LV_RV_z_area_profiles",
            "build_open_base_LV_myocardial_shell",
            "build_initial_open_base_RV_myocardial_shell",
            "remove_RVepi_open_points_intersecting_LVepi",
            "remove_RV_open_points_intersecting_LVepi",
            "fix_RVepi_and_move_RVendo_to_min_3mm_RVepi_distance_with_adaptive_basal_opening_ring",
            "step8_refix_RVepi_and_RVendo_against_fixed_LVepi",
            "step8_rebuild_LV_and_RV_openbase_annuli_from_final_fixed_surface_loops",
            "step9_merge_new_RVepi_plus_LVepi_as_whole_epi_surface",
            "step9_merge_new_RVendo_plus_rebuilt_RV_openbase_annulus_plus_LVendo_plus_rebuilt_LV_openbase_annulus_as_whole_endo_annulus_surface",
            "step10_load_saved_step09_filled_holes_surface_as_TetGen_PLC_input",
            "step10_tetgen_whole_volume_mesh",
            "save_step10_raw_tetgen_outputs_to_pipeline_folder",
            "step11_post_tag_centroids_inside_original_closed_LVepi_as_elemTag_1_else_0",
            "step11_write_final_custom_VTK",
            "save_step11_final_elemTag_outputs_to_pipeline_folder",
        ],
        "parameters": serializable_args(frame_args),
        "subject_base_detection": subject_base_detection,
        "profiles": {"lv": base.profile_to_dict(lv_profile), "rv": base.profile_to_dict(rv_profile)},
        "lv_open_base": openbase.ventricle_summary(lv_build),
        "rv_initial_open_base": openbase.ventricle_summary(rv_build),
        "open_shell_stage_summary": open_shell_stage_summary,
        "intersection_removal_stage_summary": intersection_stage_summary,
        "rvendo_distance_stage_summary": rvendo_distance_stage_summary,
        "fix_intersection_new_surface_stage_summary": fix_new_surface_stage_summary,
        "merge_surface_stage_summary": merge_surface_stage_summary,
        "tetgen_stage_summary": tetgen_stage_summary,
        "surface_intersection_removal": {"step05_RVepi": rv_epi_removal, "step06_RV": rv_endo_removal, "step08": step8_fix_info},
        "rv_distance_fix": rv_distance_fix,
        "rv_closure_surfaces": rv_cut_info,
        "surface_summaries": {
            "lv_shell_reference": surface_summary(lv_shell),
            "lv_epi_closed_intersection_reference": surface_summary(lv_epi_intersection_reference),
            "rv_epi_step5_trimmed": surface_summary(rv_epi_trimmed),
            "rv_endo_step6_trimmed": surface_summary(rv_endo_trimmed),
            "rv_endo_step7_distance_fixed": surface_summary(rv_endo_fixed),
            "lv_epi_step8_fixed": surface_summary(lv_epi_step8),
            "lv_endo_step8_fixed": surface_summary(lv_endo_step8),
            "rv_epi_step8_fixed": surface_summary(rv_epi_step8),
            "rv_endo_step8_fixed": surface_summary(rv_endo_step8),
            "lv_step8_annulus": surface_summary(lv_step8_annulus),
            "rv_step8_annulus": surface_summary(rv_step8_annulus),
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
