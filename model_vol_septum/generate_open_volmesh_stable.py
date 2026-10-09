#!/usr/bin/env python3
"""Build and save the open-base LV/RV surface stages.

Active pipeline:
  1. Load the four input surfaces and save copies per subject and frame.
  2. Detect the subject-level base side and save the Z-area profiles.
  3. Build and save the LV open-base epi, endo, and annulus surfaces.
  4. Build and save the initial RV open-base epi, endo, and annulus surfaces.
  5. Keep RVendo fixed and move RVepi to at least 3 mm from RVendo, while
     adaptively displacing and smoothing the RVepi opening ring.
  6. Rebuild the LV and RV annuli by directly connecting each open-base epi
     boundary to its corresponding open-base endo boundary.
  7. Remove RVepi/RVendo points intersecting the Step 6 LVepi reference and
     save the LV/RV surfaces as point-only VTK files.

The previous intersection-repair, surface-merge, TetGen, and volume-tagging
stages remain in this file for reference but are not called by the active
pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
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
    parser.add_argument("--pipeline-output-root", type=Path, default=None, help="Defaults to SUBJECT_DIR/volmesh_pipline.")
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
    parser.add_argument(
        "--step7-lvepi-point-tolerance",
        type=float,
        default=0.05,
        help="Maximum point-to-LVepi distance treated as an intersecting Step 7 RV point.",
    )
    parser.add_argument("--step6-ball-pivot-normal-k", type=int, default=30)
    parser.add_argument("--step6-ball-pivot-radii-multipliers", default="0.75,1.5,3.0")
    parser.add_argument("--surface-removal-rings", type=int, default=0)
    parser.add_argument("--surface-component-min-fraction", type=float, default=0.01)
    parser.add_argument("--step56-component-min-area-fraction", type=float, default=0.0)
    parser.add_argument("--step56-cut-loop-min-length-fraction", type=float, default=0.0)
    parser.add_argument("--step56-guide-point-fraction", type=float, default=0.25)
    parser.add_argument("--step56-guide-min-points", type=int, default=24)
    parser.add_argument("--step56-guide-max-points", type=int, default=80)
    parser.add_argument("--step56-guide-smooth-iterations", type=int, default=12)
    parser.add_argument("--step56-guide-smooth-relaxation", type=float, default=0.12)
    parser.add_argument("--step56-candidate-guide-scales", default="0.7,1.0,1.35")
    parser.add_argument("--step56-transition-layers", type=int, default=-1, help="Use a non-negative value to override adaptive internal transition-layer selection.")
    parser.add_argument("--step56-min-transition-layers", type=int, default=1)
    parser.add_argument("--step56-max-transition-layers", type=int, default=6)
    parser.add_argument("--step56-transition-width-percentile", type=float, default=90.0)
    parser.add_argument("--step56-target-edge-percentile", type=float, default=50.0)
    parser.add_argument("--step56-lvepi-path-proximity-weight", type=float, default=0.35)
    parser.add_argument("--step56-internal-smooth-iterations", type=int, default=8)
    parser.add_argument("--step56-internal-smooth-relaxation", type=float, default=0.12)
    parser.add_argument("--step56-bad-aspect-threshold", type=float, default=12.0)
    parser.add_argument("--surface-clean-tolerance", type=float, default=1e-7)
    parser.add_argument("--stage8-intersection-check-tolerance", type=float, default=1e-6)
    parser.add_argument("--step9-lvepi-face-match-tolerance", type=float, default=0.25)
    parser.add_argument("--step9-rv-openbase-loop-match-tolerance", type=float, default=0.5)
    parser.add_argument("--step9-lvepi-patch-loop-tolerance", type=float, default=3.0)
    parser.add_argument("--step9-lvepi-patch-component-min-fraction", type=float, default=0.02)
    parser.add_argument("--step9-fill-lvepi-patch-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step9-fill-small-local-seams", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step9-small-local-seam-max-loop-points", type=int, default=12)
    parser.add_argument("--step9-small-local-seam-max-area", type=float, default=25.0)
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
    parser.add_argument("--step7-interface-ring-candidate-k", type=int, default=256)
    parser.add_argument("--step7-interface-ring-distance-tolerance", type=float, default=1e-5)

    parser.add_argument("--tetgen-input-mode", choices=["raw", "repair", "raw_then_repair"], default="repair")
    parser.add_argument("--strict-lv-preservation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lv-preservation-tolerance", type=float, default=1.0)
    parser.add_argument("--tetgen-minratio", type=float, default=1.2)
    parser.add_argument("--tetgen-mindihedral", type=float, default=5.0)
    parser.add_argument("--tetgen-max-volume", type=float, default=0.0)
    parser.add_argument("--tetgen-verbose", action="store_true")
    parser.add_argument("--strict-tag-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-region-cell-fraction", type=float, default=0.03)
    parser.add_argument("--lv-tag-vertex-inside-threshold", type=int, default=3)
    parser.add_argument("--tag-island-cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag-island-min-cells", type=int, default=32)
    parser.add_argument("--tag-island-min-fraction", type=float, default=0.0005)
    parser.add_argument("--tag-island-cleanup-iterations", type=int, default=2)
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








def ordered_unique_indices(indices: np.ndarray) -> np.ndarray:
    ordered: List[int] = []
    seen = set()
    for value in np.asarray(indices, dtype=np.int64).reshape(-1):
        item = int(value)
        if ordered and ordered[-1] == item:
            continue
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    if len(ordered) > 2 and ordered[0] == ordered[-1]:
        ordered = ordered[:-1]
    return np.asarray(ordered, dtype=np.int64)




def select_lvepi_cut_boundary_loops(
    name: str,
    surface: pv.PolyData,
    opening_loop: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[List[pv.PolyData], Dict[str, object]]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    loops = base.boundary_loops(surface)
    if not loops:
        raise RuntimeError(f"{name}: no boundary loop remains after intersecting-face deletion.")
    ref_pts = loop_points(opening_loop)
    if len(ref_pts) < 3:
        raise RuntimeError(f"{name}: open-base reference loop has fewer than three points.")
    ref_tree = cKDTree(ref_pts)
    openbase_tol = max(
        float(args.step9_rv_openbase_loop_match_tolerance),
        float(args.boundary_match_tolerance) * 10.0,
        float(args.surface_clean_tolerance) * 100.0,
    )
    records: List[Dict[str, object]] = []
    cut_candidates: List[Tuple[int, pv.PolyData]] = []
    for loop_id, loop in enumerate(loops):
        pts = loop_points(loop)
        distances = ref_tree.query(pts, k=1)[0] if len(pts) else np.empty((0,), dtype=float)
        p95 = float(np.percentile(distances, 95)) if len(distances) else float("inf")
        is_openbase = p95 <= openbase_tol
        record = {
            "loop_id": int(loop_id),
            "points": int(len(pts)),
            "length": float(loop.length),
            "area_best_plane": float(abs(base.signed_area_in_best_plane(pts))) if len(pts) >= 3 else 0.0,
            "openbase_reference_distance_min": float(np.min(distances)) if len(distances) else None,
            "openbase_reference_distance_mean": float(np.mean(distances)) if len(distances) else None,
            "openbase_reference_distance_p95": p95,
            "openbase_reference_distance_max": float(np.max(distances)) if len(distances) else None,
            "classified_as": "openbase_boundary" if is_openbase else "LVepi_intersection_cut_boundary",
        }
        records.append(record)
        if not is_openbase and len(pts) >= 3:
            cut_candidates.append((int(loop_id), base.make_polyline(pts)))

    if not cut_candidates:
        fallback = max(records, key=lambda item: float(item["openbase_reference_distance_p95"]))
        fallback_id = int(fallback["loop_id"])
        cut_candidates = [(fallback_id, base.make_polyline(loop_points(loops[fallback_id])))]
        records[fallback_id]["classified_as"] = "LVepi_intersection_cut_boundary_fallback_farthest_from_openbase"

    longest = max(float(loop.length) for _, loop in cut_candidates)
    cut_loop_min_fraction = max(float(getattr(args, "step56_cut_loop_min_length_fraction", args.surface_component_min_fraction)), 0.0)
    min_length = longest * cut_loop_min_fraction
    selected: List[pv.PolyData] = []
    selected_ids = []
    for loop_id, loop in cut_candidates:
        if float(loop.length) >= min_length:
            selected.append(loop)
            selected_ids.append(int(loop_id))
        else:
            records[loop_id]["filtered_out_reason"] = "short_cut_boundary_component"
    if not selected:
        loop_id, loop = max(cut_candidates, key=lambda item: float(item[1].length))
        selected = [loop]
        selected_ids = [int(loop_id)]
        records[loop_id]["filtered_out_reason"] = None

    return selected, {
        "method": "boundary_loops_not_matching_openbase_loop_after_intersecting_face_vertex_deletion",
        "surface": name,
        "boundary_loop_count": int(len(loops)),
        "selected_cut_loop_count": int(len(selected)),
        "selected_cut_loop_ids": selected_ids,
        "openbase_match_tolerance_mm": float(openbase_tol),
        "cut_loop_min_length_fraction": float(cut_loop_min_fraction),
        "loops": records,
    }











def empty_tagged_triangle_surface(elem_tag: int = 0) -> pv.PolyData:
    mesh = pv.PolyData(np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.int64))
    mesh.cell_data["elemTag"] = np.empty((0,), dtype=np.int32)
    mesh.point_data["lvepiSubstituteVertex"] = np.empty((0,), dtype=np.int32)
    mesh.point_data["lvepiSubstituteSourceId"] = np.empty((0,), dtype=np.int32)
    mesh.point_data["keptOriginalRVVertex"] = np.empty((0,), dtype=np.int32)
    return mesh


def compact_tagged_triangle_surface_with_point_data(
    points: np.ndarray,
    triangles: np.ndarray,
    elem_tags: np.ndarray,
    point_data: Dict[str, np.ndarray],
    clean_tolerance: float,
) -> pv.PolyData:
    triangles = np.asarray(triangles, dtype=np.int64)
    elem_tags = np.asarray(elem_tags, dtype=np.int32).reshape(-1)
    if len(triangles) == 0:
        return empty_tagged_triangle_surface()
    used = np.unique(triangles.ravel())
    point_map = -np.ones(len(points), dtype=np.int64)
    point_map[used] = np.arange(len(used), dtype=np.int64)
    out = pv.PolyData(np.asarray(points, dtype=float)[used], base.faces_from_triangles(point_map[triangles]))
    out.cell_data["elemTag"] = elem_tags
    for key, values in point_data.items():
        array = np.asarray(values)
        if len(array) == len(points):
            out.point_data[key] = array[used]
    return out.extract_surface().triangulate().clean(tolerance=float(clean_tolerance))


def loop_vertex_ids_without_duplicate_endpoint(vertex_ids: Sequence[int]) -> np.ndarray:
    ordered: List[int] = []
    for value in np.asarray(vertex_ids, dtype=np.int64).reshape(-1):
        item = int(value)
        if ordered and ordered[-1] == item:
            continue
        ordered.append(item)
    if len(ordered) > 2 and ordered[0] == ordered[-1]:
        ordered.pop()
    return np.asarray(ordered, dtype=np.int64)


def closed_loop_edge_count_record(vertex_ids: Sequence[int], edge_set: set) -> Dict[str, object]:
    ids = loop_vertex_ids_without_duplicate_endpoint(vertex_ids)
    total = 0
    exact = 0
    missing_edges: List[Tuple[int, int]] = []
    for index in range(len(ids)):
        a = int(ids[index])
        b = int(ids[(index + 1) % len(ids)])
        if a == b:
            continue
        total += 1
        edge = sorted_edge_key(a, b)
        if edge in edge_set:
            exact += 1
        elif len(missing_edges) < 20:
            missing_edges.append(edge)
    return {
        "loop_vertices": int(len(ids)),
        "closed_loop_edges": int(total),
        "exact_existing_LVepi_edges": int(exact),
        "non_LVepi_edges": int(total - exact),
        "exact_existing_LVepi_edge_fraction": float(exact / max(total, 1)),
        "missing_edge_sample": [[int(a), int(b)] for a, b in missing_edges],
    }


def shortest_weighted_path_on_lvepi_graph(
    start: int,
    goal: int,
    neighbors: Sequence[set],
    lv_points: np.ndarray,
    allowed_mask: Optional[np.ndarray] = None,
    node_penalty: Optional[np.ndarray] = None,
    blocked_mask: Optional[np.ndarray] = None,
) -> List[int]:
    start = int(start)
    goal = int(goal)
    if start == goal:
        return [start]
    n = len(neighbors)
    points = np.asarray(lv_points, dtype=float)
    distances = np.full(n, np.inf, dtype=float)
    previous = -np.ones(n, dtype=np.int64)
    distances[start] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start)]
    penalty = None if node_penalty is None else np.asarray(node_penalty, dtype=float)
    while heap:
        current_distance, current = heappop(heap)
        if current_distance > distances[current]:
            continue
        if current == goal:
            break
        for nb in neighbors[current]:
            nb = int(nb)
            if allowed_mask is not None and not bool(allowed_mask[nb]) and nb != goal:
                continue
            if blocked_mask is not None and bool(blocked_mask[nb]) and nb != goal:
                continue
            edge_cost = float(np.linalg.norm(points[nb] - points[current]))
            if penalty is not None:
                edge_cost += 0.5 * float(max(penalty[current], 0.0) + max(penalty[nb], 0.0))
            next_distance = current_distance + edge_cost
            if next_distance < distances[nb]:
                distances[nb] = next_distance
                previous[nb] = current
                heappush(heap, (next_distance, nb))
    if not np.isfinite(distances[goal]):
        return []
    path = [goal]
    current = goal
    while current != start:
        current = int(previous[current])
        if current < 0:
            return []
        path.append(current)
    path.reverse()
    return path


def loop_duplicate_vertex_count(vertex_ids: Sequence[int]) -> int:
    ids = loop_vertex_ids_without_duplicate_endpoint(vertex_ids)
    return int(len(ids) - len(set(int(v) for v in ids)))


def step56_candidate_guide_counts(cut_point_count: int, args: argparse.Namespace) -> Tuple[List[int], Dict[str, object]]:
    cut_count = int(cut_point_count)
    if cut_count < 3:
        return [], {"method": "too_few_cut_points", "cut_boundary_points": cut_count}
    fraction = float(np.clip(float(getattr(args, "step56_guide_point_fraction", 0.25)), 0.02, 1.0))
    min_points = max(3, int(getattr(args, "step56_guide_min_points", 24)))
    max_points = max(min_points, int(getattr(args, "step56_guide_max_points", 80)))
    min_points = min(min_points, cut_count)
    max_points = min(max_points, cut_count)
    base_count = int(np.clip(int(round(cut_count * fraction)), min_points, max_points))
    scales: List[float] = []
    for item in parse_csv_values(str(getattr(args, "step56_candidate_guide_scales", "0.7,1.0,1.35"))):
        try:
            value = float(item)
        except ValueError:
            continue
        if np.isfinite(value) and value > 0.0:
            scales.append(value)
    if not scales:
        scales = [1.0]
    candidates = {base_count, min_points, max_points}
    for scale in scales:
        candidates.add(int(np.clip(int(round(base_count * scale)), min_points, max_points)))
    ordered = sorted(candidates, key=lambda count: (abs(count - base_count), count))
    return ordered, {
        "method": "arc_length_sparse_RV_cut_loop_guide_counts_from_boundary_point_count",
        "cut_boundary_points": int(cut_count),
        "guide_point_fraction": float(fraction),
        "guide_min_points": int(min_points),
        "guide_max_points": int(max_points),
        "base_guide_count": int(base_count),
        "candidate_guide_scales": [float(v) for v in scales],
        "candidate_guide_counts": [int(v) for v in ordered],
    }


def sparse_closed_loop_guide_points(points: np.ndarray, count: int) -> np.ndarray:
    pts = clean_loop_points(points)
    count = int(count)
    if len(pts) < 3:
        return pts
    if count >= len(pts):
        return pts.copy()
    return base.resample_closed_loop(pts, max(3, count))


def closed_loop_self_intersection_count_2d(points: np.ndarray, tolerance: float = 1e-10) -> int:
    pts = clean_loop_points(points)
    n = len(pts)
    if n < 4:
        return 0
    centered = pts - pts.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vh[:2].T
    except Exception:
        coords = centered[:, :2]

    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def boxes_overlap(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        return bool(
            max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0])) + tolerance
            and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1])) + tolerance
        )

    def crosses(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        if not boxes_overlap(a, b, c, d):
            return False
        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)
        if abs(o1) <= tolerance or abs(o2) <= tolerance or abs(o3) <= tolerance or abs(o4) <= tolerance:
            return False
        return bool((o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0))

    count = 0
    for i in range(n):
        a = coords[i]
        b = coords[(i + 1) % n]
        for j in range(i + 1, n):
            if i == j or (i + 1) % n == j or (j + 1) % n == i:
                continue
            c = coords[j]
            d = coords[(j + 1) % n]
            if crosses(a, b, c, d):
                count += 1
    return int(count)


def closed_loop_turn_angle_summary(points: np.ndarray) -> Dict[str, object]:
    pts = clean_loop_points(points)
    if len(pts) < 3:
        return {"points": int(len(pts)), "turn_angle_mean_deg": None, "turn_angle_p95_deg": None, "turn_angle_max_deg": None}
    prev_vec = pts - np.roll(pts, 1, axis=0)
    next_vec = np.roll(pts, -1, axis=0) - pts
    prev_norm = np.linalg.norm(prev_vec, axis=1)
    next_norm = np.linalg.norm(next_vec, axis=1)
    valid = (prev_norm > 1e-12) & (next_norm > 1e-12)
    if not np.any(valid):
        return {"points": int(len(pts)), "turn_angle_mean_deg": None, "turn_angle_p95_deg": None, "turn_angle_max_deg": None}
    unit_prev = prev_vec[valid] / prev_norm[valid, None]
    unit_next = next_vec[valid] / next_norm[valid, None]
    cos_values = np.sum(unit_prev * unit_next, axis=1)
    turn = np.degrees(np.arccos(np.clip(cos_values, -1.0, 1.0)))
    return {
        "points": int(len(pts)),
        "turn_angle_mean_deg": float(np.mean(turn)),
        "turn_angle_p95_deg": float(np.percentile(turn, 95)),
        "turn_angle_max_deg": float(np.max(turn)),
    }


def closed_loop_edge_length_summary(points: np.ndarray) -> Dict[str, object]:
    lengths = closed_loop_edge_lengths(points)
    if len(lengths) == 0:
        return {"edge_count": 0}
    median = float(np.median(lengths))
    return {
        "edge_count": int(len(lengths)),
        "edge_length_min": float(np.min(lengths)),
        "edge_length_mean": float(np.mean(lengths)),
        "edge_length_median": median,
        "edge_length_p95": float(np.percentile(lengths, 95)),
        "edge_length_max": float(np.max(lengths)),
        "edge_length_max_over_median": float(np.max(lengths) / max(median, 1e-12)),
    }


def lvepi_line_loop_quality_record(
    vertex_ids: Sequence[int],
    lv_points: np.ndarray,
    edge_set: set,
    guide_distances: Optional[np.ndarray],
    target_boundary_point_count: int,
) -> Dict[str, object]:
    ids = loop_vertex_ids_without_duplicate_endpoint(vertex_ids)
    points = np.asarray(lv_points, dtype=float)[ids] if len(ids) else np.empty((0, 3), dtype=float)
    edge_record = closed_loop_edge_count_record(ids, edge_set)
    edge_record.update(closed_loop_edge_length_summary(points))
    turn_record = closed_loop_turn_angle_summary(points)
    duplicate_count = loop_duplicate_vertex_count(ids)
    self_intersections = closed_loop_self_intersection_count_2d(points)
    density_ratio = float(len(ids) / max(int(target_boundary_point_count), 1))
    distance_record: Dict[str, object]
    if guide_distances is not None and len(ids):
        local = np.asarray(guide_distances, dtype=float)[ids]
        distance_record = {
            "guide_distance_min": float(np.min(local)),
            "guide_distance_mean": float(np.mean(local)),
            "guide_distance_p95": float(np.percentile(local, 95)),
            "guide_distance_max": float(np.max(local)),
        }
    else:
        distance_record = {"guide_distance_min": None, "guide_distance_mean": None, "guide_distance_p95": None, "guide_distance_max": None}
    valid = bool(
        len(ids) >= 3
        and duplicate_count == 0
        and int(edge_record.get("non_LVepi_edges", 0) or 0) == 0
        and self_intersections == 0
    )
    score = 0.0
    score += float(turn_record.get("turn_angle_p95_deg") or 180.0) * 0.04
    score += float(edge_record.get("edge_length_max_over_median") or 10.0) * 1.5
    score += float(distance_record.get("guide_distance_p95") or 0.0) * 0.4
    score += abs(density_ratio - 1.0) * 2.0
    score += duplicate_count * 1000.0 + self_intersections * 1000.0
    if not valid:
        score += 10000.0
    return {
        "valid": bool(valid),
        "quality_score": float(score),
        "target_RV_cut_boundary_points": int(target_boundary_point_count),
        "density_ratio_vs_RV_cut_boundary": density_ratio,
        "duplicate_vertices": int(duplicate_count),
        "self_intersections_in_best_plane": int(self_intersections),
        "edge_quality": edge_record,
        "turn_quality": turn_record,
        "guide_distance_quality": distance_record,
    }


def trace_lvepi_seed_loop_on_graph(
    ordered_seed_ids: Sequence[int],
    neighbors: Sequence[set],
    lv_points: np.ndarray,
    allowed_mask: Optional[np.ndarray],
    node_penalty: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, object]]:
    seeds = loop_vertex_ids_without_duplicate_endpoint(ordered_seed_ids)
    traced: List[int] = []
    used_internal_vertices = set()
    path_records: List[Dict[str, object]] = []
    fallback_paths = 0
    failed_paths = 0
    repeated_vertex_fallback_paths = 0
    for index in range(len(seeds)):
        start_id = int(seeds[index])
        goal_id = int(seeds[(index + 1) % len(seeds)])
        if start_id == goal_id:
            continue
        blocked_mask = np.zeros(len(lv_points), dtype=bool)
        blocked_mask[np.asarray(seeds, dtype=np.int64)] = True
        if used_internal_vertices:
            blocked_mask[np.asarray(sorted(used_internal_vertices), dtype=np.int64)] = True
        blocked_mask[start_id] = False
        blocked_mask[goal_id] = False
        path_ids = shortest_weighted_path_on_lvepi_graph(
            start_id,
            goal_id,
            neighbors,
            lv_points,
            allowed_mask,
            node_penalty,
            blocked_mask,
        )
        path_method = "weighted_allowed_graph_near_smoothed_sparse_LVepi_seed_loop_avoiding_existing_loop_vertices"
        used_fallback = False
        if len(path_ids) < 2:
            path_ids = shortest_weighted_path_on_lvepi_graph(
                start_id,
                goal_id,
                neighbors,
                lv_points,
                allowed_mask,
                node_penalty,
                None,
            )
            path_method = "weighted_allowed_graph_near_smoothed_sparse_LVepi_seed_loop"
            used_fallback = True
            repeated_vertex_fallback_paths += 1
        if len(path_ids) < 2:
            path_ids = shortest_weighted_path_on_lvepi_graph(
                start_id,
                goal_id,
                neighbors,
                lv_points,
                None,
                node_penalty,
                None,
            )
            path_method = "weighted_unrestricted_graph_fallback"
            used_fallback = True
            fallback_paths += 1
        if len(path_ids) < 2:
            failed_paths += 1
            path_records.append({
                "segment_id": int(index),
                "start": int(start_id),
                "goal": int(goal_id),
                "failed": True,
                "path_vertices": int(len(path_ids)),
            })
            continue
        segment_ids = path_ids if not traced else path_ids[1:]
        if index == len(seeds) - 1 and segment_ids and traced and segment_ids[-1] == traced[0]:
            segment_ids = segment_ids[:-1]
        traced.extend(int(v) for v in segment_ids)
        used_internal_vertices.update(int(v) for v in path_ids[1:-1])
        path_records.append({
            "segment_id": int(index),
            "start": int(start_id),
            "goal": int(goal_id),
            "method": path_method,
            "used_fallback": bool(used_fallback),
            "path_vertices": int(len(path_ids)),
            "path_length_mm": float(sum(np.linalg.norm(lv_points[int(b)] - lv_points[int(a)]) for a, b in zip(path_ids[:-1], path_ids[1:]))),
        })
    return loop_vertex_ids_without_duplicate_endpoint(traced), {
        "repeated_vertex_avoidance_fallback_path_count": int(repeated_vertex_fallback_paths),
        "fallback_unrestricted_path_count": int(fallback_paths),
        "failed_path_count": int(failed_paths),
        "path_records_sample": path_records[:30],
    }


def trace_lvepi_seed_loop_on_graph_strict_allowed(
    ordered_seed_ids: Sequence[int],
    neighbors: Sequence[set],
    lv_points: np.ndarray,
    allowed_mask: np.ndarray,
    node_penalty: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, object]]:
    seeds = loop_vertex_ids_without_duplicate_endpoint(ordered_seed_ids)
    traced: List[int] = []
    used_internal_vertices = set()
    path_records: List[Dict[str, object]] = []
    failed_paths = 0
    repeated_vertex_fallback_paths = 0
    for index in range(len(seeds)):
        start_id = int(seeds[index])
        goal_id = int(seeds[(index + 1) % len(seeds)])
        if start_id == goal_id:
            continue
        blocked_mask = np.zeros(len(lv_points), dtype=bool)
        blocked_mask[np.asarray(seeds, dtype=np.int64)] = True
        if used_internal_vertices:
            blocked_mask[np.asarray(sorted(used_internal_vertices), dtype=np.int64)] = True
        blocked_mask[start_id] = False
        blocked_mask[goal_id] = False
        path_ids = shortest_weighted_path_on_lvepi_graph(
            start_id,
            goal_id,
            neighbors,
            lv_points,
            allowed_mask,
            node_penalty,
            blocked_mask,
        )
        path_method = "strict_allowed_graph_avoiding_existing_loop_vertices"
        used_fallback = False
        if len(path_ids) < 2:
            path_ids = shortest_weighted_path_on_lvepi_graph(
                start_id,
                goal_id,
                neighbors,
                lv_points,
                allowed_mask,
                node_penalty,
                None,
            )
            path_method = "strict_allowed_graph_without_repeat_blocking"
            used_fallback = True
            repeated_vertex_fallback_paths += 1
        if len(path_ids) < 2:
            failed_paths += 1
            path_records.append({
                "segment_id": int(index),
                "start": int(start_id),
                "goal": int(goal_id),
                "failed": True,
                "path_vertices": int(len(path_ids)),
            })
            continue
        segment_ids = path_ids if not traced else path_ids[1:]
        if index == len(seeds) - 1 and segment_ids and traced and segment_ids[-1] == traced[0]:
            segment_ids = segment_ids[:-1]
        traced.extend(int(v) for v in segment_ids)
        used_internal_vertices.update(int(v) for v in path_ids[1:-1])
        path_records.append({
            "segment_id": int(index),
            "start": int(start_id),
            "goal": int(goal_id),
            "method": path_method,
            "used_fallback": bool(used_fallback),
            "path_vertices": int(len(path_ids)),
            "path_length_mm": float(sum(np.linalg.norm(lv_points[int(b)] - lv_points[int(a)]) for a, b in zip(path_ids[:-1], path_ids[1:]))),
        })
    return loop_vertex_ids_without_duplicate_endpoint(traced), {
        "method": "strict_allowed_LVepi_graph_trace_no_unrestricted_fallback",
        "repeated_vertex_avoidance_fallback_path_count": int(repeated_vertex_fallback_paths),
        "failed_path_count": int(failed_paths),
        "path_records_sample": path_records[:30],
    }


def trace_existing_lvepi_edge_loop_from_cut_loop(
    name: str,
    cut_loop: pv.PolyData,
    lv_epi_reference: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, pv.PolyData, Dict[str, object]]:
    cut_pts = clean_loop_points(loop_points(cut_loop))
    lv_epi = lv_epi_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_points = np.asarray(lv_epi.points, dtype=float)
    lv_triangles = base.triangle_faces(lv_epi)
    if len(cut_pts) < 3 or len(lv_points) < 3 or len(lv_triangles) == 0:
        raise RuntimeError(f"{name}: cannot trace an LVepi topology loop from the RV cut boundary.")

    neighbors, edge_set = lvepi_vertex_edge_graph(lv_epi.n_points, lv_triangles)
    guide_counts, guide_count_info = step56_candidate_guide_counts(len(cut_pts), args)
    if not guide_counts:
        raise RuntimeError(f"{name}: cannot sparsify the RV cut boundary.")

    lv_edge_lengths = []
    for a, b in edge_set:
        lv_edge_lengths.append(float(np.linalg.norm(lv_points[int(a)] - lv_points[int(b)])))
    median_lv_edge = float(np.median(lv_edge_lengths)) if lv_edge_lengths else 1.0
    proximity_weight = max(float(getattr(args, "step56_lvepi_path_proximity_weight", 0.35)), 0.0)
    best_candidate: Optional[Dict[str, object]] = None
    candidate_records: List[Dict[str, object]] = []

    for guide_count in guide_counts:
        sparse_guide_pts = sparse_closed_loop_guide_points(cut_pts, int(guide_count))
        if len(sparse_guide_pts) < 3:
            continue
        seed_distances, nearest_ids = cKDTree(lv_points).query(sparse_guide_pts, k=1)
        seed_ids = ordered_unique_indices(np.asarray(nearest_ids, dtype=np.int64))
        if len(seed_ids) < 3:
            candidate_records.append({
                "guide_count": int(guide_count),
                "created": False,
                "reason": "fewer_than_three_unique_LVepi_seed_vertices",
                "unique_seed_vertices": int(len(seed_ids)),
            })
            continue
        ordered_seed_ids, order_info = order_lvepi_seam_vertex_ids(
            f"{name}_seed_order_{guide_count}",
            lv_points,
            seed_ids,
            edge_set,
        )
        seed_points = lv_points[ordered_seed_ids]
        smooth_iterations = int(getattr(args, "step56_guide_smooth_iterations", 12))
        smooth_relaxation = float(getattr(args, "step56_guide_smooth_relaxation", 0.12))
        smoothed_seed_points = smooth_closed_vectors(seed_points, smooth_iterations, smooth_relaxation)
        guide_sample_count = max(48, min(400, len(smoothed_seed_points) * 4))
        smoothed_seed_curve = base.resample_closed_loop(smoothed_seed_points, guide_sample_count)
        guide_distances = cKDTree(smoothed_seed_curve).query(lv_points, k=1)[0]
        seed_p95 = float(np.percentile(seed_distances, 95)) if len(seed_distances) else 0.0
        allowed_radius = max(
            float(args.step9_lvepi_patch_loop_tolerance),
            seed_p95 * 2.0,
            median_lv_edge * 4.0,
            float(args.boundary_match_tolerance) * 10.0,
            1.0,
        )
        allowed_mask = guide_distances <= allowed_radius
        allowed_mask[np.asarray(ordered_seed_ids, dtype=np.int64)] = True
        node_penalty = guide_distances * proximity_weight
        traced_ids, path_info = trace_lvepi_seed_loop_on_graph(
            ordered_seed_ids,
            neighbors,
            lv_points,
            allowed_mask,
            node_penalty,
        )
        if len(traced_ids) < 3:
            candidate_records.append({
                "guide_count": int(guide_count),
                "created": False,
                "reason": "LVepi_graph_trace_produced_fewer_than_three_vertices",
                "unique_seed_vertices": int(len(seed_ids)),
                "path_trace": path_info,
            })
            continue
        quality = lvepi_line_loop_quality_record(
            traced_ids,
            lv_points,
            edge_set,
            guide_distances,
            len(cut_pts),
        )
        candidate = {
            "guide_count": int(guide_count),
            "created": True,
            "sparse_RV_guide_points": int(len(sparse_guide_pts)),
            "nearest_seed_vertices_before_unique": int(len(nearest_ids)),
            "unique_seed_vertices": int(len(seed_ids)),
            "ordered_seed_vertices": int(len(ordered_seed_ids)),
            "traced_existing_LVepi_vertices": int(len(traced_ids)),
            "allowed_graph_radius_mm": float(allowed_radius),
            "median_LVepi_graph_edge_length_mm": float(median_lv_edge),
            "nearest_sparse_guide_to_LVepi_distance_min": float(np.min(seed_distances)) if len(seed_distances) else None,
            "nearest_sparse_guide_to_LVepi_distance_mean": float(np.mean(seed_distances)) if len(seed_distances) else None,
            "nearest_sparse_guide_to_LVepi_distance_p95": seed_p95 if len(seed_distances) else None,
            "nearest_sparse_guide_to_LVepi_distance_max": float(np.max(seed_distances)) if len(seed_distances) else None,
            "seed_ordering": order_info,
            "path_trace": path_info,
            "quality": quality,
            "lv_epi_vertex_ids": [int(v) for v in traced_ids.tolist()],
        }
        candidate_records.append(candidate)
        if best_candidate is None or float(quality["quality_score"]) < float(best_candidate["quality"]["quality_score"]):
            best_candidate = candidate
        elif bool(quality["valid"]) and not bool(best_candidate["quality"].get("valid", False)):
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(f"{name}: no LVepi loop candidate could be traced from sparse RV guide points.")

    traced_ids = np.asarray(best_candidate["lv_epi_vertex_ids"], dtype=np.int64)
    if len(traced_ids) < 3:
        raise RuntimeError(f"{name}: selected LVepi graph loop has fewer than three vertices.")
    seam_loop = base.make_polyline(lv_points[traced_ids])
    selected_quality = best_candidate["quality"]
    return traced_ids, seam_loop, {
        "method": "sparse_RV_cut_boundary_guides_initial_LVepi_seeds_then_refined_smooth_LVepi_graph_line_loop_using_original_LVepi_vertices_only",
        "cut_loop": loop_record(cut_loop),
        "cut_loop_points": int(len(cut_pts)),
        "sparse_guide_count_selection": guide_count_info,
        "selected_guide_count": int(best_candidate["guide_count"]),
        "selected_candidate_valid": bool(selected_quality.get("valid", False)),
        "selected_candidate_quality_score": float(selected_quality.get("quality_score", 0.0)),
        "selected_loop_quality": selected_quality,
        "nearest_seed_vertices_before_unique": int(best_candidate["nearest_seed_vertices_before_unique"]),
        "unique_seed_vertices": int(best_candidate["unique_seed_vertices"]),
        "traced_existing_LVepi_vertices": int(len(traced_ids)),
        "allowed_graph_radius_mm": float(best_candidate["allowed_graph_radius_mm"]),
        "path_proximity_weight": float(proximity_weight),
        "guide_smooth_iterations": int(getattr(args, "step56_guide_smooth_iterations", 12)),
        "guide_smooth_relaxation": float(getattr(args, "step56_guide_smooth_relaxation", 0.12)),
        "repeated_vertex_avoidance_fallback_path_count": int(best_candidate["path_trace"].get("repeated_vertex_avoidance_fallback_path_count", 0)),
        "fallback_unrestricted_path_count": int(best_candidate["path_trace"].get("fallback_unrestricted_path_count", 0)),
        "failed_path_count": int(best_candidate["path_trace"].get("failed_path_count", 0)),
        "duplicate_vertices_in_final_line_loop": int(loop_duplicate_vertex_count(traced_ids)),
        "path_block_rule": "avoid all other sparse LVepi seed vertices and already-used internal path vertices when tracing each segment; fallback only if no path exists",
        "seed_ordering": best_candidate["seed_ordering"],
        "nearest_existing_LVepi_distance_min": best_candidate["nearest_sparse_guide_to_LVepi_distance_min"],
        "nearest_existing_LVepi_distance_mean": best_candidate["nearest_sparse_guide_to_LVepi_distance_mean"],
        "nearest_existing_LVepi_distance_p95": best_candidate["nearest_sparse_guide_to_LVepi_distance_p95"],
        "nearest_existing_LVepi_distance_max": best_candidate["nearest_sparse_guide_to_LVepi_distance_max"],
        "edge_conformity": selected_quality.get("edge_quality", closed_loop_edge_count_record(traced_ids, edge_set)),
        "lv_epi_vertex_ids": [int(v) for v in traced_ids.tolist()],
        "seam_loop": loop_record(seam_loop),
        "candidate_records": candidate_records,
        "path_records_sample": best_candidate["path_trace"].get("path_records_sample", []),
    }


def mark_lvepi_substitute_vertices(
    surface: pv.PolyData,
    lv_epi_reference: pv.PolyData,
    substitute_source_ids: Sequence[int],
    args: argparse.Namespace,
) -> pv.PolyData:
    mesh = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    source_ids = np.unique(np.asarray(substitute_source_ids, dtype=np.int64).reshape(-1))
    source_ids = source_ids[(source_ids >= 0) & (source_ids < lv_epi_reference.n_points)]
    substitute_mask = np.zeros(mesh.n_points, dtype=np.int32)
    source_array = np.full(mesh.n_points, -1, dtype=np.int32)
    if len(source_ids):
        lv_points = np.asarray(lv_epi_reference.points, dtype=float)
        candidate_points = lv_points[source_ids]
        distances, local_ids = cKDTree(candidate_points).query(np.asarray(mesh.points, dtype=float), k=1)
        exact_tolerance = max(float(args.surface_clean_tolerance) * 100.0, 1e-5)
        matched = distances <= exact_tolerance
        substitute_mask[matched] = 1
        source_array[matched] = source_ids[np.asarray(local_ids, dtype=np.int64)[matched]].astype(np.int32)
    mesh.point_data["lvepiSubstituteVertex"] = substitute_mask
    mesh.point_data["lvepiSubstituteSourceId"] = source_array
    mesh.point_data["keptOriginalRVVertex"] = (substitute_mask == 0).astype(np.int32)
    return mesh


def mesh_with_default_step56_point_data(mesh: pv.PolyData, elem_tag: int) -> pv.PolyData:
    out = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    out.cell_data["elemTag"] = np.full(out.n_cells, int(elem_tag), dtype=np.int32)
    out.point_data["lvepiSubstituteVertex"] = np.zeros(out.n_points, dtype=np.int32)
    out.point_data["lvepiSubstituteSourceId"] = np.full(out.n_points, -1, dtype=np.int32)
    out.point_data["keptOriginalRVVertex"] = np.ones(out.n_points, dtype=np.int32)
    return out


def merge_tagged_triangle_surfaces_with_point_data(parts: Sequence[pv.PolyData], clean_tolerance: float) -> pv.PolyData:
    clean_parts = [part.extract_surface().triangulate().clean(tolerance=float(clean_tolerance)) for part in parts if part.n_cells > 0]
    if not clean_parts:
        return empty_tagged_triangle_surface()
    keys = sorted({key for part in clean_parts for key in part.point_data.keys()})
    point_blocks: List[np.ndarray] = []
    triangle_blocks: List[np.ndarray] = []
    tag_blocks: List[np.ndarray] = []
    data_blocks: Dict[str, List[np.ndarray]] = {key: [] for key in keys}
    offset = 0
    for part in clean_parts:
        triangles = base.triangle_faces(part)
        points = np.asarray(part.points, dtype=float)
        point_blocks.append(points)
        triangle_blocks.append(triangles + offset)
        if "elemTag" in part.cell_data and len(part.cell_data["elemTag"]) == len(triangles):
            tag_blocks.append(np.rint(np.asarray(part.cell_data["elemTag"])).astype(np.int32))
        else:
            tag_blocks.append(np.zeros(len(triangles), dtype=np.int32))
        for key in keys:
            if key in part.point_data and len(part.point_data[key]) == part.n_points:
                data_blocks[key].append(np.asarray(part.point_data[key]))
            elif key.endswith("SourceId"):
                data_blocks[key].append(np.full(part.n_points, -1, dtype=np.int32))
            else:
                data_blocks[key].append(np.zeros(part.n_points, dtype=np.int32))
        offset += int(part.n_points)
    point_data = {key: np.concatenate(values) for key, values in data_blocks.items()}
    return compact_tagged_triangle_surface_with_point_data(
        np.vstack(point_blocks),
        np.vstack(triangle_blocks),
        np.concatenate(tag_blocks),
        point_data,
        float(clean_tolerance),
    )


def closed_loop_fraction_positions(points: np.ndarray) -> np.ndarray:
    pts = clean_loop_points(points)
    if len(pts) == 0:
        return np.empty((0,), dtype=float)
    closed = np.vstack([pts, pts[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(pts), endpoint=False)
    return np.concatenate([[0.0], np.cumsum(lengths[:-1])]) / total


def sample_closed_loop_at_fractions(points: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    pts = clean_loop_points(points)
    fractions = np.asarray(fractions, dtype=float).reshape(-1)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)
    if len(pts) == 1:
        return np.repeat(pts[:1], len(fractions), axis=0)
    closed = np.vstack([pts, pts[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-12:
        return np.repeat(pts[:1], len(fractions), axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = (fractions % 1.0) * total
    segment_ids = np.searchsorted(cumulative, targets, side="right") - 1
    segment_ids = np.clip(segment_ids, 0, len(lengths) - 1)
    local = (targets - cumulative[segment_ids]) / np.maximum(lengths[segment_ids], 1e-12)
    return closed[segment_ids] * (1.0 - local[:, None]) + closed[segment_ids + 1] * local[:, None]


def align_lvepi_vertex_id_loop_to_points(
    target_points: np.ndarray,
    seam_ids: Sequence[int],
    lv_points: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    target = clean_loop_points(target_points)
    ids = loop_vertex_ids_without_duplicate_endpoint(seam_ids)
    if len(target) < 3 or len(ids) < 3:
        raise RuntimeError("Cannot align LVepi seam loop with fewer than three points.")
    sample_count = min(200, max(48, min(len(target), len(ids))))
    target_sample = base.resample_closed_loop(target, sample_count)
    best_score = float("inf")
    best_ids = ids.copy()
    best_reversed = False
    best_shift = 0
    for reversed_flag, candidate in ((False, ids), (True, ids[::-1].copy())):
        for shift in range(len(candidate)):
            shifted = np.roll(candidate, -int(shift))
            shifted_sample = base.resample_closed_loop(np.asarray(lv_points, dtype=float)[shifted], sample_count)
            score = float(np.mean(np.linalg.norm(target_sample - shifted_sample, axis=1)))
            if score < best_score:
                best_score = score
                best_ids = shifted.copy()
                best_reversed = bool(reversed_flag)
                best_shift = int(shift)
    return best_ids, {
        "method": "loop_orientation_and_cyclic_shift_minimizing_resampled_RV_to_LVepi_distance",
        "sample_count": int(sample_count),
        "reversed": bool(best_reversed),
        "cyclic_shift": int(best_shift),
        "mean_resampled_alignment_distance": float(best_score),
    }


def arc_length_zipper_triangles(outer_count: int, inner_count: int, outer_offset: int, inner_offset: int) -> np.ndarray:
    outer_count = int(outer_count)
    inner_count = int(inner_count)
    if outer_count < 3 or inner_count < 3:
        raise RuntimeError("Cannot arc-length stitch loops with fewer than three points.")
    triangles: List[List[int]] = []
    i = 0
    j = 0
    outer_steps = 0
    inner_steps = 0
    while outer_steps < outer_count or inner_steps < inner_count:
        next_i = (i + 1) % outer_count
        next_j = (j + 1) % inner_count
        if outer_steps >= outer_count:
            advance_outer = False
        elif inner_steps >= inner_count:
            advance_outer = True
        else:
            advance_outer = ((outer_steps + 1) / float(outer_count)) <= ((inner_steps + 1) / float(inner_count))
        if advance_outer:
            triangles.append([int(outer_offset + i), int(outer_offset + next_i), int(inner_offset + j)])
            i = next_i
            outer_steps += 1
        else:
            triangles.append([int(outer_offset + i), int(inner_offset + next_j), int(inner_offset + j)])
            j = next_j
            inner_steps += 1
    return np.asarray(triangles, dtype=np.int64)


def triangle_aspect_for_ids(points: np.ndarray, ids: Sequence[int]) -> float:
    pts = np.asarray(points, dtype=float)
    a, b, c = (pts[int(ids[0])], pts[int(ids[1])], pts[int(ids[2])])
    lengths = np.asarray([
        np.linalg.norm(b - a),
        np.linalg.norm(c - b),
        np.linalg.norm(a - c),
    ], dtype=float)
    area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    if area <= 1e-12:
        return float("inf")
    return float(np.max(lengths) / max(np.min(lengths), 1e-12))




def same_count_ring_triangles_best_diagonal(points: np.ndarray, count: int, first_offset: int, second_offset: int) -> np.ndarray:
    triangles: List[List[int]] = []
    for index in range(int(count)):
        nxt = (index + 1) % int(count)
        a = int(first_offset + index)
        b = int(first_offset + nxt)
        c = int(second_offset + index)
        d = int(second_offset + nxt)
        candidate_ac = ([a, b, c], [b, d, c])
        candidate_bd = ([a, b, d], [a, d, c])
        score_ac = max(triangle_aspect_for_ids(points, candidate_ac[0]), triangle_aspect_for_ids(points, candidate_ac[1]))
        score_bd = max(triangle_aspect_for_ids(points, candidate_bd[0]), triangle_aspect_for_ids(points, candidate_bd[1]))
        selected = candidate_bd if score_bd < score_ac else candidate_ac
        triangles.extend([[int(v) for v in tri] for tri in selected])
    return np.asarray(triangles, dtype=np.int64)


def triangle_quality_summary(points: np.ndarray, triangles: np.ndarray, bad_aspect_threshold: float) -> Dict[str, object]:
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(triangles, dtype=np.int64)
    if len(tri) == 0:
        return {"triangles": 0}
    pa = pts[tri[:, 0]]
    pb = pts[tri[:, 1]]
    pc = pts[tri[:, 2]]
    lengths = np.column_stack([
        np.linalg.norm(pb - pa, axis=1),
        np.linalg.norm(pc - pb, axis=1),
        np.linalg.norm(pa - pc, axis=1),
    ])
    areas = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa), axis=1)
    longest = lengths.max(axis=1)
    shortest = np.maximum(lengths.min(axis=1), 1e-12)
    aspect = longest / shortest
    threshold = float(bad_aspect_threshold)
    bad = (aspect > threshold) | (areas <= 1e-10)
    return {
        "triangles": int(len(tri)),
        "edge_length_min": float(np.min(lengths)),
        "edge_length_mean": float(np.mean(lengths)),
        "edge_length_p95": float(np.percentile(lengths, 95)),
        "edge_length_max": float(np.max(lengths)),
        "area_min": float(np.min(areas)),
        "area_mean": float(np.mean(areas)),
        "area_p05": float(np.percentile(areas, 5)),
        "aspect_min": float(np.min(aspect)),
        "aspect_mean": float(np.mean(aspect)),
        "aspect_p95": float(np.percentile(aspect, 95)),
        "aspect_max": float(np.max(aspect)),
        "bad_aspect_threshold": threshold,
        "bad_or_degenerate_triangle_count": int(np.count_nonzero(bad)),
        "bad_or_degenerate_triangle_fraction": float(np.count_nonzero(bad) / max(len(tri), 1)),
    }


def closed_loop_edge_lengths(points: np.ndarray) -> np.ndarray:
    pts = clean_loop_points(points)
    if len(pts) < 2:
        return np.empty((0,), dtype=float)
    closed = np.vstack([pts, pts[0]])
    return np.linalg.norm(np.diff(closed, axis=0), axis=1)


def choose_step56_transition_layer_count(
    rv_points: np.ndarray,
    lv_ring: np.ndarray,
    lv_at_rv_fractions: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, object]]:
    fixed_layers = int(getattr(args, "step56_transition_layers", -1))
    min_layers = max(0, int(getattr(args, "step56_min_transition_layers", 1)))
    max_layers = max(min_layers, int(getattr(args, "step56_max_transition_layers", 6)))
    width_percentile = float(getattr(args, "step56_transition_width_percentile", 90.0))
    edge_percentile = float(getattr(args, "step56_target_edge_percentile", 50.0))

    rv_edge_lengths = closed_loop_edge_lengths(rv_points)
    lv_edge_lengths = closed_loop_edge_lengths(lv_ring)
    edge_pool = np.concatenate([rv_edge_lengths, lv_edge_lengths])
    edge_pool = edge_pool[np.isfinite(edge_pool) & (edge_pool > 1e-8)]
    target_edge_length = float(np.percentile(edge_pool, edge_percentile)) if len(edge_pool) else 1.0

    rv_to_lv_widths = np.linalg.norm(np.asarray(lv_at_rv_fractions, dtype=float) - np.asarray(rv_points, dtype=float), axis=1)
    lv_fractions = closed_loop_fraction_positions(lv_ring)
    rv_at_lv_fractions = sample_closed_loop_at_fractions(rv_points, lv_fractions)
    lv_to_rv_widths = np.linalg.norm(np.asarray(lv_ring, dtype=float) - rv_at_lv_fractions, axis=1)
    width_pool = np.concatenate([rv_to_lv_widths, lv_to_rv_widths])
    width_pool = width_pool[np.isfinite(width_pool)]
    transition_width = float(np.percentile(width_pool, width_percentile)) if len(width_pool) else 0.0

    if fixed_layers >= 0:
        layer_count = int(np.clip(fixed_layers, min_layers, max_layers))
        method = "fixed_user_step56_transition_layers_clamped_to_min_max"
    else:
        estimated_segments = int(np.ceil(transition_width / max(target_edge_length, 1e-8)))
        layer_count = int(np.clip(max(min_layers, estimated_segments - 1), min_layers, max_layers))
        method = "adaptive_internal_support_layers_from_transition_width_over_local_edge_length"

    return layer_count, {
        "method": method,
        "fixed_step56_transition_layers": int(fixed_layers),
        "min_transition_layers": int(min_layers),
        "max_transition_layers": int(max_layers),
        "selected_transition_layers": int(layer_count),
        "transition_width_percentile": float(width_percentile),
        "target_edge_percentile": float(edge_percentile),
        "transition_width_mm": float(transition_width),
        "target_edge_length_mm": float(target_edge_length),
        "estimated_width_segments": int(np.ceil(transition_width / max(target_edge_length, 1e-8))) if fixed_layers < 0 else None,
        "rv_boundary_edge_length_min": float(np.min(rv_edge_lengths)) if len(rv_edge_lengths) else None,
        "rv_boundary_edge_length_median": float(np.median(rv_edge_lengths)) if len(rv_edge_lengths) else None,
        "rv_boundary_edge_length_max": float(np.max(rv_edge_lengths)) if len(rv_edge_lengths) else None,
        "lvepi_boundary_edge_length_min": float(np.min(lv_edge_lengths)) if len(lv_edge_lengths) else None,
        "lvepi_boundary_edge_length_median": float(np.median(lv_edge_lengths)) if len(lv_edge_lengths) else None,
        "lvepi_boundary_edge_length_max": float(np.max(lv_edge_lengths)) if len(lv_edge_lengths) else None,
        "rv_to_lvepi_width_min": float(np.min(rv_to_lv_widths)) if len(rv_to_lv_widths) else None,
        "rv_to_lvepi_width_median": float(np.median(rv_to_lv_widths)) if len(rv_to_lv_widths) else None,
        "rv_to_lvepi_width_p95": float(np.percentile(rv_to_lv_widths, 95)) if len(rv_to_lv_widths) else None,
        "rv_to_lvepi_width_max": float(np.max(rv_to_lv_widths)) if len(rv_to_lv_widths) else None,
    }


def smooth_internal_transition_ring(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    iterations = int(getattr(args, "step56_internal_smooth_iterations", 0))
    relaxation = float(getattr(args, "step56_internal_smooth_relaxation", 0.0))
    if iterations <= 0 or relaxation <= 0.0 or len(points) < 4:
        return np.asarray(points, dtype=float).copy()
    return smooth_closed_vectors(np.asarray(points, dtype=float), iterations, relaxation)


def build_layered_lvepi_edge_conforming_transition_strip(
    name: str,
    rv_cut_loop: pv.PolyData,
    lv_seam_vertex_ids: Sequence[int],
    lv_epi_reference: pv.PolyData,
    elem_tag: int,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, np.ndarray, Dict[str, object]]:
    rv_points = clean_loop_points(loop_points(rv_cut_loop))
    lv_epi = lv_epi_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_points = np.asarray(lv_epi.points, dtype=float)
    lv_triangles = base.triangle_faces(lv_epi)
    if len(rv_points) < 3:
        raise RuntimeError(f"{name}: RV cut loop has fewer than three points.")
    if len(lv_triangles) == 0:
        raise RuntimeError(f"{name}: LVepi reference has no triangles.")

    aligned_ids, align_info = align_lvepi_vertex_id_loop_to_points(rv_points, lv_seam_vertex_ids, lv_points)
    lv_ring = lv_points[aligned_ids]
    rv_fractions = closed_loop_fraction_positions(rv_points)
    lv_at_rv_fractions = sample_closed_loop_at_fractions(lv_ring, rv_fractions)
    layer_count, layer_info = choose_step56_transition_layer_count(
        rv_points,
        lv_ring,
        lv_at_rv_fractions,
        args,
    )

    rings: List[np.ndarray] = [rv_points]
    internal_ring_records: List[Dict[str, object]] = []
    for layer_id in range(1, layer_count + 1):
        alpha = float(layer_id) / float(layer_count + 1)
        raw_ring = (1.0 - alpha) * rv_points + alpha * lv_at_rv_fractions
        smoothed_ring = smooth_internal_transition_ring(raw_ring, args)
        rings.append(smoothed_ring)
        displacement = np.linalg.norm(smoothed_ring - raw_ring, axis=1)
        internal_ring_records.append({
            "internal_layer_id": int(layer_id),
            "alpha_from_RV_to_LVepi": float(alpha),
            "points": int(len(smoothed_ring)),
            "smoothing_iterations": int(getattr(args, "step56_internal_smooth_iterations", 0)),
            "smoothing_relaxation": float(getattr(args, "step56_internal_smooth_relaxation", 0.0)),
            "smoothing_displacement_mean": float(np.mean(displacement)) if len(displacement) else 0.0,
            "smoothing_displacement_max": float(np.max(displacement)) if len(displacement) else 0.0,
        })
    rings.append(lv_ring)

    point_blocks: List[np.ndarray] = []
    offsets: List[int] = []
    offset = 0
    for ring in rings:
        offsets.append(offset)
        point_blocks.append(ring)
        offset += len(ring)
    all_points = np.vstack(point_blocks)

    triangle_blocks: List[np.ndarray] = []
    if layer_count > 0:
        for ring_id in range(layer_count):
            triangle_blocks.append(same_count_ring_triangles_best_diagonal(all_points, len(rv_points), offsets[ring_id], offsets[ring_id + 1]))
        triangle_blocks.append(arc_length_zipper_triangles(len(rv_points), len(lv_ring), offsets[layer_count], offsets[layer_count + 1]))
    else:
        triangle_blocks.append(arc_length_zipper_triangles(len(rv_points), len(lv_ring), offsets[0], offsets[1]))
    triangles = np.vstack(triangle_blocks)
    tags = np.full(len(triangles), int(elem_tag), dtype=np.int32)

    substitute = np.zeros(len(all_points), dtype=np.int32)
    source_ids = np.full(len(all_points), -1, dtype=np.int32)
    kept_original = np.ones(len(all_points), dtype=np.int32)
    lv_offset = offsets[-1]
    substitute[lv_offset:lv_offset + len(aligned_ids)] = 1
    source_ids[lv_offset:lv_offset + len(aligned_ids)] = np.asarray(aligned_ids, dtype=np.int32)
    kept_original[lv_offset:lv_offset + len(aligned_ids)] = 0
    for ring_id in range(1, layer_count + 1):
        start = offsets[ring_id]
        stop = start + len(rv_points)
        kept_original[start:stop] = 0

    strip = compact_tagged_triangle_surface_with_point_data(
        all_points,
        triangles,
        tags,
        {
            "lvepiSubstituteVertex": substitute,
            "lvepiSubstituteSourceId": source_ids,
            "keptOriginalRVVertex": kept_original,
        },
        args.surface_clean_tolerance,
    )
    strip = mark_lvepi_substitute_vertices(strip, lv_epi_reference, aligned_ids, args)
    _, edge_set = lvepi_vertex_edge_graph(lv_epi.n_points, lv_triangles)
    edge_record = closed_loop_edge_count_record(aligned_ids, edge_set)
    quality = triangle_quality_summary(all_points, triangles, float(args.step56_bad_aspect_threshold))
    return strip, aligned_ids, {
        "method": "adaptive_internal_support_ring_transition_strip_from_RV_cut_loop_to_existing_LVepi_edge_loop",
        "rv_cut_loop": loop_record(rv_cut_loop),
        "lv_epi_edge_path_loop": loop_record(base.make_polyline(lv_ring)),
        "rv_cut_loop_points": int(len(rv_points)),
        "lv_epi_path_vertices": int(len(aligned_ids)),
        "transition_layer_count": int(layer_count),
        "internal_support_points_created": int(layer_count * len(rv_points)),
        "boundary_points_created": 0,
        "boundary_rule": "Final RV/LV interface boundary is the generated LVepi line loop using existing LVepi vertices; the rough RV cut loop is only the internal attachment side of the repair band.",
        "alignment": align_info,
        "adaptive_layer_selection": layer_info,
        "internal_support_rings": internal_ring_records,
        "lv_epi_edge_conformity": edge_record,
        "raw_transition_strip_quality_before_cleanup": quality,
        "stitch_surface": surface_summary(strip),
    }

def select_step56_intersection_removal_faces(
    name: str,
    triangles: np.ndarray,
    points: np.ndarray,
    vertex_hit: np.ndarray,
    center_hit: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    triangles = np.asarray(triangles, dtype=np.int64)
    vertex_hit = np.asarray(vertex_hit, dtype=bool)
    invalid_face_mask = np.any(vertex_hit[triangles], axis=1) if len(triangles) else np.empty((0,), dtype=bool)
    remove_vertex_mask = vertex_hit.copy()
    if not np.any(remove_vertex_mask):
        return invalid_face_mask, remove_vertex_mask, {
            "method": "no_vertex_intersection_candidates",
            "vertex_deletion_rule": "delete_only_RV_vertices_that_intersect_or_touch_LVepi",
            "face_validity_rule": "remove_triangles_that_reference_deleted_vertices",
            "candidate_faces": 0,
            "selected_faces": 0,
            "deleted_vertices": 0,
        }

    areas = face_areas(points, triangles) if len(triangles) else np.empty((0,), dtype=float)
    components = connected_face_components(invalid_face_mask, triangles) if np.any(invalid_face_mask) else []
    records: List[Dict[str, object]] = []
    for component_id, component in enumerate(components):
        component = np.asarray(component, dtype=np.int64)
        component_vertices = np.unique(triangles[component].ravel()) if len(component) else np.empty((0,), dtype=np.int64)
        hit_vertices = component_vertices[vertex_hit[component_vertices]] if len(component_vertices) else np.empty((0,), dtype=np.int64)
        records.append({
            "component_id": int(component_id),
            "invalid_faces": int(len(component)),
            "area": float(areas[component].sum()) if len(component) else 0.0,
            "vertices_on_invalid_faces": int(len(component_vertices)),
            "deleted_hit_vertices": int(len(hit_vertices)),
            "retained_non_hit_vertices_on_invalid_faces_if_referenced_by_valid_faces": int(len(component_vertices) - len(hit_vertices)),
            "center_hit_faces_diagnostic": int(np.count_nonzero(center_hit[component])) if len(center_hit) == len(triangles) else 0,
        })

    invalid_face_vertices = np.unique(triangles[invalid_face_mask].ravel()) if np.any(invalid_face_mask) else np.empty((0,), dtype=np.int64)
    non_hit_vertices_on_invalid_faces = invalid_face_vertices[~vertex_hit[invalid_face_vertices]] if len(invalid_face_vertices) else np.empty((0,), dtype=np.int64)
    valid_face_mask = ~np.any(remove_vertex_mask[triangles], axis=1) if len(triangles) else np.empty((0,), dtype=bool)
    retained_by_valid_faces = np.zeros(len(points), dtype=bool)
    if np.any(valid_face_mask):
        retained_by_valid_faces[np.unique(triangles[valid_face_mask].ravel())] = True
    return invalid_face_mask, remove_vertex_mask, {
        "method": "delete_only_RV_vertices_intersecting_LVepi_then_remove_invalid_faces_that_reference_deleted_vertices",
        "vertex_deletion_rule": "delete_intersecting_vertices_only_non_intersecting_vertices_are_not_marked_for_deletion",
        "face_validity_rule": "triangles_referencing_any_deleted_vertex_are_removed_because_they_are_invalid",
        "center_hit_usage": "diagnostic_only_not_used_for_vertex_deletion",
        "candidate_faces": int(np.count_nonzero(invalid_face_mask)),
        "selected_faces": int(np.count_nonzero(invalid_face_mask)),
        "invalid_faces_removed_after_vertex_deletion": int(np.count_nonzero(~valid_face_mask)),
        "valid_faces_kept_after_vertex_deletion": int(np.count_nonzero(valid_face_mask)),
        "deleted_vertices": int(np.count_nonzero(remove_vertex_mask)),
        "non_hit_vertices_on_invalid_faces": int(len(non_hit_vertices_on_invalid_faces)),
        "non_hit_vertices_on_invalid_faces_retained_by_other_valid_faces": int(np.count_nonzero(retained_by_valid_faces[non_hit_vertices_on_invalid_faces])) if len(non_hit_vertices_on_invalid_faces) else 0,
        "candidate_component_count": int(len(components)),
        "component_records": records[:40],
    }


def make_lvepi_edge_conforming_stitch_surfaces(
    name: str,
    rv_cut_loops: Sequence[pv.PolyData],
    lv_seam_vertex_id_loops: Sequence[np.ndarray],
    lv_epi_reference: pv.PolyData,
    elem_tag: int,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    if len(rv_cut_loops) != len(lv_seam_vertex_id_loops):
        raise ValueError(f"{name}: RV cut loop count and LVepi seam vertex-id loop count differ.")
    stitches: List[pv.PolyData] = []
    records: List[Dict[str, object]] = []
    all_source_ids: List[int] = []
    for loop_id, (cut_loop, seam_ids) in enumerate(zip(rv_cut_loops, lv_seam_vertex_id_loops)):
        seam_ids = loop_vertex_ids_without_duplicate_endpoint(seam_ids)
        if len(seam_ids) < 3:
            records.append({
                "loop_id": int(loop_id),
                "created": False,
                "reason": "fewer_than_three_LVepi_path_vertices",
            })
            continue
        stitch, aligned_ids, stitch_info = build_layered_lvepi_edge_conforming_transition_strip(
            f"{name}_loop_{loop_id}",
            cut_loop,
            seam_ids,
            lv_epi_reference,
            elem_tag,
            args,
        )
        stitches.append(stitch)
        all_source_ids.extend(int(v) for v in aligned_ids)
        stitch_info["loop_id"] = int(loop_id)
        stitch_info["created"] = True
        records.append(stitch_info)
    if not stitches:
        empty = empty_tagged_triangle_surface(elem_tag)
        return empty, {"method": "no_LVepi_edge_conforming_stitches_created", "stitch_count": 0, "stitches": records}
    merged = merge_tagged_triangle_surfaces_with_point_data(stitches, args.surface_clean_tolerance)
    merged = mark_lvepi_substitute_vertices(merged, lv_epi_reference, all_source_ids, args)
    return merged, {
        "method": "merge_all_LVepi_line_loop_boundary_repair_strips_with_internal_support_points",
        "stitch_count": int(len(stitches)),
        "transition_layers": "adaptive" if int(getattr(args, "step56_transition_layers", -1)) < 0 else int(args.step56_transition_layers),
        "unique_existing_LVepi_vertices_used_by_stitches": int(len(set(all_source_ids))),
        "stitches": records,
        "merged_stitch_surface": surface_summary(merged),
    }


def substitute_intersecting_vertices_with_lvepi_existing_vertices(
    name: str,
    surface: pv.PolyData,
    lv_epi_reference: pv.PolyData,
    opening_loop: pv.PolyData,
    elem_tag: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_epi_reference = lv_epi_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    triangles = base.triangle_faces(surface)
    points = np.asarray(surface.points, dtype=float)
    lv_points = np.asarray(lv_epi_reference.points, dtype=float)
    if len(triangles) == 0:
        raise RuntimeError(f"{name}: input surface has no triangles.")
    if len(lv_points) == 0:
        raise RuntimeError(f"{name}: LVepi reference has no points.")

    vertex_hit, vertex_distances, vertex_inside = inside_or_near_reference_surface(lv_epi_reference, points, args)
    centers = points[triangles].mean(axis=1) if len(triangles) else np.empty((0, 3), dtype=float)
    center_hit, _, center_inside = inside_or_near_reference_surface(lv_epi_reference, centers, args) if len(centers) else (np.empty((0,), dtype=bool), np.empty((0,), dtype=float), np.empty((0,), dtype=bool))
    hit_counts = np.count_nonzero(vertex_hit[triangles], axis=1)

    if not np.any(vertex_hit):
        tagged = mesh_with_default_step56_point_data(surface, elem_tag)
        return {
            "surface": tagged,
            "trimmed_surface": tagged,
            "stitch_surface": empty_tagged_triangle_surface(elem_tag),
            "cut_mask": np.zeros(tagged.n_points, dtype=bool),
            "trimmed_cut_mask": np.zeros(tagged.n_points, dtype=bool),
            "rv_cut_loops": [],
            "lv_seam_loops": [],
            "info": {
                "surface": name,
                "method": "no_RV_vertices_intersect_or_touch_LVepi",
                "input_points": int(surface.n_points),
                "input_faces": int(len(triangles)),
                "clearance_mm": float(args.surface_lvepi_clearance),
                "surface_after_existing_LVepi_vertex_substitution": surface_summary(tagged),
            },
        }

    selected_remove_faces, remove_vertex_mask, component_info = select_step56_intersection_removal_faces(
        name,
        triangles,
        points,
        vertex_hit,
        center_hit,
        args,
    )
    keep_faces = ~np.any(remove_vertex_mask[triangles], axis=1)
    if not np.any(keep_faces):
        raise RuntimeError(f"{name}: deleting whole triangles with any LVepi-intersecting vertex would remove every valid triangle.")

    original_vertices_kept_by_face_rule = np.zeros(len(points), dtype=bool)
    original_vertices_kept_by_face_rule[np.unique(triangles[keep_faces].ravel())] = True
    original_vertices_on_deleted_faces = np.zeros(len(points), dtype=bool)
    if np.any(selected_remove_faces):
        original_vertices_on_deleted_faces[np.unique(triangles[selected_remove_faces].ravel())] = True

    kept_raw = base.subset_mesh_faces(surface, keep_faces).extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    kept_surface = mesh_with_default_step56_point_data(kept_raw, elem_tag)
    rv_cut_loops, cut_loop_info = select_lvepi_cut_boundary_loops(name, kept_surface, opening_loop, args)

    lv_seam_loops: List[pv.PolyData] = []
    lv_seam_vertex_id_loops: List[np.ndarray] = []
    seam_records: List[Dict[str, object]] = []
    all_lvepi_source_ids: List[int] = []
    for loop_id, cut_loop in enumerate(rv_cut_loops):
        traced_ids, seam_loop, seam_info = trace_existing_lvepi_edge_loop_from_cut_loop(
            f"{name}_LVepi_intersection_cut_loop_{loop_id}",
            cut_loop,
            lv_epi_reference,
            args,
        )
        lv_seam_loops.append(seam_loop)
        lv_seam_vertex_id_loops.append(traced_ids)
        all_lvepi_source_ids.extend(int(v) for v in traced_ids)
        seam_records.append(seam_info)

    stitch_surface, stitch_info = make_lvepi_edge_conforming_stitch_surfaces(
        name,
        rv_cut_loops,
        lv_seam_vertex_id_loops,
        lv_epi_reference,
        elem_tag,
        args,
    )
    reconstructed = merge_tagged_triangle_surfaces_with_point_data(
        [kept_surface, stitch_surface],
        args.surface_clean_tolerance,
    )
    reconstructed = mark_lvepi_substitute_vertices(reconstructed, lv_epi_reference, all_lvepi_source_ids, args)

    cut_loop_points_blocks = [loop_points(loop) for loop in rv_cut_loops if len(loop_points(loop))]
    seam_loop_points_blocks = [loop_points(loop) for loop in lv_seam_loops if len(loop_points(loop))]
    rv_cut_point_cloud = np.vstack(cut_loop_points_blocks) if cut_loop_points_blocks else np.empty((0, 3), dtype=float)
    full_cut_point_cloud = np.vstack(cut_loop_points_blocks + seam_loop_points_blocks) if (cut_loop_points_blocks or seam_loop_points_blocks) else np.empty((0, 3), dtype=float)
    mask_tol = max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0, 1e-5)
    cut_mask = map_source_points_to_target_mask(
        full_cut_point_cloud,
        np.asarray(reconstructed.points, dtype=float),
        mask_tol,
    ) if reconstructed.n_points else np.empty((0,), dtype=bool)
    kept_cut_mask = map_source_points_to_target_mask(
        rv_cut_point_cloud,
        np.asarray(kept_surface.points, dtype=float),
        mask_tol,
    ) if kept_surface.n_points else np.empty((0,), dtype=bool)

    output_substitute_mask = np.asarray(reconstructed.point_data.get("lvepiSubstituteVertex", np.zeros(reconstructed.n_points, dtype=np.int32))).reshape(-1).astype(bool)
    output_source_ids = np.asarray(reconstructed.point_data.get("lvepiSubstituteSourceId", np.full(reconstructed.n_points, -1, dtype=np.int32))).reshape(-1).astype(np.int64)
    unique_output_source_ids = np.unique(output_source_ids[output_substitute_mask & (output_source_ids >= 0)])
    kept_original_points_after_reconstruction = map_source_points_to_target_mask(
        points[original_vertices_kept_by_face_rule],
        np.asarray(reconstructed.points, dtype=float),
        max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0, 1e-5),
    )
    lvepi_edge_records = [record.get("edge_conformity", {}) for record in seam_records]
    total_lvepi_edges = int(sum(int(record.get("closed_loop_edges", 0) or 0) for record in lvepi_edge_records))
    exact_lvepi_edges = int(sum(int(record.get("exact_existing_LVepi_edges", 0) or 0) for record in lvepi_edge_records))

    ignored_hit_vertices = int(np.count_nonzero(vertex_hit & (~remove_vertex_mask)))
    info = {
        "surface": name,
        "method": "delete_intersecting_RV_vertices_keep_non_intersecting_RV_vertices_then_reconstruct_to_smooth_sparse_guided_LVepi_original_vertex_loop",
        "new_projected_points_created": False,
        "substitute_points_source": "existing_LVepi_vertices_only",
        "substitute_edges_source": "actual_existing_LVepi_triangle_edges_traced_on_LVepi_graph",
        "input_points": int(surface.n_points),
        "input_faces": int(len(triangles)),
        "clearance_mm": float(args.surface_lvepi_clearance),
        "reference_LVepi_free_edges": int(len(base.extract_boundary_edges(lv_epi_reference))),
        "intersecting_vertices_initial_candidates": int(np.count_nonzero(vertex_hit)),
        "intersecting_vertices_on_deleted_faces": int(np.count_nonzero(remove_vertex_mask)),
        "intersecting_vertices_ignored_by_optional_small_component_filter": ignored_hit_vertices,
        "intersecting_vertices_inside_closed_LVepi": int(np.count_nonzero(vertex_inside)),
        "face_center_hits_diagnostic_only": int(np.count_nonzero(center_hit)),
        "face_centers_inside_closed_LVepi_diagnostic_only": int(np.count_nonzero(center_inside)),
        "initial_faces_with_no_intersecting_vertices": int(np.count_nonzero(hit_counts == 0)),
        "initial_mixed_faces": int(np.count_nonzero((hit_counts > 0) & (hit_counts < 3))),
        "initial_faces_all_vertices_intersect_LVepi": int(np.count_nonzero(hit_counts == 3)),
        "vertex_deletion_and_invalid_face_removal": component_info,
        "whole_triangle_deletion": component_info,
        "kept_faces_after_vertex_deletion_validity_rule": int(np.count_nonzero(keep_faces)),
        "deleted_faces_after_vertex_deletion_validity_rule": int(np.count_nonzero(~keep_faces)),
        "deleted_intersecting_vertices_by_vertex_rule": int(np.count_nonzero(remove_vertex_mask)),
        "kept_faces_after_any_vertex_face_deletion_rule": int(np.count_nonzero(keep_faces)),
        "deleted_faces_after_any_vertex_face_deletion_rule": int(np.count_nonzero(~keep_faces)),
        "original_vertices_on_deleted_faces": int(np.count_nonzero(original_vertices_on_deleted_faces)),
        "original_vertices_still_used_by_kept_faces": int(np.count_nonzero(original_vertices_kept_by_face_rule)),
        "kept_original_vertices_preserved_in_reconstructed_surface": int(np.count_nonzero(kept_original_points_after_reconstruction)),
        "kept_original_vertices_unreferenced_after_reconstruction": int(np.count_nonzero(original_vertices_kept_by_face_rule) - np.count_nonzero(kept_original_points_after_reconstruction)),
        "cut_boundary_vertices_on_RV_kept_surface": int(np.count_nonzero(kept_cut_mask)),
        "cut_boundary_vertices_after_stitch_cleanup": int(np.count_nonzero(cut_mask)),
        "unique_existing_LVepi_vertices_requested_by_seams": int(len(set(all_lvepi_source_ids))),
        "unique_existing_LVepi_vertices_present_after_cleanup": int(len(unique_output_source_ids)),
        "LVepi_seam_edges_total": int(total_lvepi_edges),
        "LVepi_seam_edges_exact_existing_LVepi_edges": int(exact_lvepi_edges),
        "LVepi_seam_edge_exact_fraction": float(exact_lvepi_edges / max(total_lvepi_edges, 1)),
        "min_signed_distance_to_LVepi_before_deletion": float(vertex_distances.min()) if len(vertex_distances) else None,
        "kept_original_surface": surface_summary(kept_surface) if kept_surface.n_cells else {"points": 0, "triangles": 0, "free_edges": 0},
        "edge_conforming_interface_strip_surface": surface_summary(stitch_surface) if stitch_surface.n_cells else {"points": 0, "triangles": 0, "free_edges": 0},
        "surface_after_existing_LVepi_vertex_substitution": surface_summary(reconstructed),
        "cut_boundary_selection": cut_loop_info,
        "lv_epi_seam_mapping_and_graph_trace": seam_records,
        "stitch": stitch_info,
        "saved_LVepi_interface_seam_loops": [loop_record(loop) for loop in lv_seam_loops],
        "component_cleanup": {"applied": False, "reason": "intersecting_vertices_removed_then_RV_LVepi_interface_rebuilt_from_sparse_guided_original_LVepi_vertex_line_loop"},
    }
    return {
        "surface": reconstructed,
        "trimmed_surface": kept_surface,
        "stitch_surface": stitch_surface,
        "cut_mask": cut_mask,
        "trimmed_cut_mask": kept_cut_mask,
        "rv_cut_loops": rv_cut_loops,
        "lv_seam_loops": lv_seam_loops,
        "info": info,
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
    protected_vertex_mask: Optional[np.ndarray] = None,
    force_outward_from_fixed_surface: bool = False,
) -> Tuple[pv.PolyData, np.ndarray, np.ndarray, np.ndarray, pv.PolyData, Dict[str, object]]:
    rv_endo = rv_endo.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    rv_epi = rv_epi.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    opening_loop_current, opening_ids, opening_mask, opening_match = match_openbase_loop_on_surface(
        "RVendo_openbase_ring_before_step7",
        rv_endo,
        opening_loop,
        args,
    )
    if protected_vertex_mask is None:
        protected_vertex_mask = np.zeros(rv_endo.n_points, dtype=bool)
    else:
        protected_vertex_mask = np.asarray(protected_vertex_mask, dtype=bool).reshape(-1)
        if len(protected_vertex_mask) != rv_endo.n_points:
            raise ValueError(
                "protected_vertex_mask must match the cleaned RVendo point count "
                f"({len(protected_vertex_mask)} != {rv_endo.n_points})."
            )

    original_points = np.asarray(rv_endo.points, dtype=float).copy()
    points = original_points.copy()
    neighbors = vertex_neighbors_from_surface(rv_endo)
    ring_levels = vertex_ring_levels(opening_mask, neighbors, int(args.rv_opening_adaptive_rings))
    transition_band_mask = ring_levels >= 0

    rv_epi_points = np.asarray(rv_epi.points, dtype=float)
    rv_epi_tree = cKDTree(rv_epi_points)
    fixed_surface_center = rv_epi_points.mean(axis=0) if len(rv_epi_points) else np.zeros(3, dtype=float)
    rv_endo_center = np.asarray(rv_endo.points, dtype=float).mean(axis=0) if rv_endo.n_points else np.zeros(3, dtype=float)
    try:
        rv_epi_normal_mesh = rv_epi.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            inplace=False,
        )
        rv_epi_normals = np.asarray(rv_epi_normal_mesh.point_data.get("Normals", np.zeros_like(rv_epi_points)), dtype=float)
    except Exception:
        rv_epi_normals = np.zeros_like(rv_epi_points)

    def separation_unit_vectors(
        query_points: np.ndarray,
        nearest_ids: np.ndarray,
        prefer_outward: bool = False,
    ) -> np.ndarray:
        """Return a separating direction, optionally restricted to fixed-surface outside."""
        query_points = np.asarray(query_points, dtype=float)
        nearest_ids = np.asarray(nearest_ids, dtype=np.int64).reshape(-1)
        nearest = rv_epi_points[nearest_ids]
        vectors = query_points - nearest
        lengths = np.linalg.norm(vectors, axis=1)
        if prefer_outward:
            radial = nearest - fixed_surface_center[None, :]
            radial_lengths = np.linalg.norm(radial, axis=1)
            outward = rv_epi_normals[nearest_ids].copy()
            normal_lengths = np.linalg.norm(outward, axis=1)
            normal_valid = normal_lengths > 1e-8
            if np.any(normal_valid):
                normal_dot_radial = np.sum(outward[normal_valid] * radial[normal_valid], axis=1)
                outward[normal_valid] *= np.where(normal_dot_radial >= 0.0, 1.0, -1.0)[:, None]
            radial_valid = radial_lengths > 1e-8
            use_radial = (~normal_valid) & radial_valid
            if np.any(use_radial):
                outward[use_radial] = radial[use_radial]
                normal_lengths[use_radial] = radial_lengths[use_radial]
            outward_lengths = np.linalg.norm(outward, axis=1)
            outward_valid = outward_lengths > 1e-8
            if np.any(outward_valid):
                outward[outward_valid] /= outward_lengths[outward_valid, None]
                dot_outward = np.sum(vectors * outward, axis=1)
                use_outward = (~(lengths > 1e-8)) | (dot_outward <= 0.0)
                vectors[use_outward] = outward[use_outward]
                lengths[use_outward] = 1.0
        zero = lengths <= 1e-8
        if np.any(zero):
            fallback = rv_endo_center[None, :] - nearest[zero]
            fallback_lengths = np.linalg.norm(fallback, axis=1)
            fallback_normals = rv_epi_normals[nearest_ids[zero]]
            normal_lengths = np.linalg.norm(fallback_normals, axis=1)
            normal_valid = normal_lengths > 1e-8
            if np.any(normal_valid):
                normal_dot_center = np.sum(
                    fallback_normals[normal_valid] * fallback[normal_valid], axis=1
                )
                normal_sign = np.where(normal_dot_center >= 0.0, 1.0, -1.0)
                fallback[normal_valid] = (
                    fallback_normals[normal_valid]
                    * normal_sign[:, None]
                )
                fallback_lengths[normal_valid] = normal_lengths[normal_valid]
            valid_fallback = fallback_lengths > 1e-8
            zero_indices = np.flatnonzero(zero)
            if np.any(valid_fallback):
                vectors[zero_indices[valid_fallback]] = fallback[valid_fallback]
                lengths[zero_indices[valid_fallback]] = fallback_lengths[valid_fallback]
            unresolved = zero_indices[~valid_fallback]
            if len(unresolved):
                vectors[unresolved] = np.array([0.0, 0.0, 1.0], dtype=float)
                lengths[unresolved] = 1.0
        return vectors / np.maximum(lengths, 1e-12)[:, None]

    def apply_uniform_ring_outward_offset(
        ring_candidate: np.ndarray,
        target_distance: float,
        protected_ring_mask: np.ndarray,
        max_iterations: int,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        """Move the opening ring with a smooth, shared step per iteration.

        Independent vertex corrections can create local kinks when the nearest
        fixed-surface feature changes from one ring vertex to the next.  The
        ring direction is therefore smoothed around the closed loop, while a
        single scalar step is applied to all movable ring vertices.
        """
        candidate = np.asarray(ring_candidate, dtype=float).copy()
        protected_ring_mask = np.asarray(protected_ring_mask, dtype=bool).reshape(-1)
        records: List[Dict[str, object]] = []
        direction_iterations = max(2, int(args.rv_opening_ring_smooth_iterations) // 3)
        direction_relaxation = min(float(args.rv_opening_ring_smooth_relaxation), 0.25)
        for iteration in range(max(int(max_iterations), 1)):
            distances, nearest_ids = rv_epi_tree.query(candidate, k=1)
            movable = ~protected_ring_mask
            minimum_distance = float(np.min(distances[movable])) if np.any(movable) else None
            record = {
                "iteration": int(iteration),
                "vertices_below_target_before": int(np.count_nonzero((distances < target_distance) & movable)),
                "minimum_distance_before": minimum_distance,
                "uniform_step_mm": 0.0,
            }
            if not np.any(movable) or minimum_distance is None or minimum_distance >= target_distance:
                records.append(record)
                break

            raw_directions = separation_unit_vectors(
                candidate[movable],
                nearest_ids[movable],
                prefer_outward=False,
            )
            smooth_directions = smooth_closed_vectors(
                raw_directions,
                direction_iterations,
                direction_relaxation,
            )
            direction_lengths = np.linalg.norm(smooth_directions, axis=1)
            valid = direction_lengths > 1e-8
            smooth_directions[valid] /= direction_lengths[valid, None]
            smooth_directions[~valid] = raw_directions[~valid]
            # Keep the smoothed field pointing in the local outward direction.
            direction_sign = np.sum(smooth_directions * raw_directions, axis=1) < 0.0
            smooth_directions[direction_sign] *= -1.0

            step = min(
                target_distance - minimum_distance + 1e-3,
                float(args.rv_distance_max_step) * 0.5,
            )
            candidate[movable] += smooth_directions * step
            record.update({
                "vertices_corrected": int(np.count_nonzero(movable)),
                "uniform_step_mm": float(step),
            })
            records.append(record)
        return candidate, records

    requested_distance = float(args.rv_min_distance_to_rvepi)
    target = requested_distance + float(args.rv_distance_padding)
    total_direct_moved = np.zeros(len(points), dtype=bool)
    iteration_records = []
    initial_distances = rv_epi_tree.query(points, k=1)[0] if len(points) else np.empty((0,), dtype=float)

    for iteration in range(int(args.rv_distance_iterations)):
        distances, ids = rv_epi_tree.query(points, k=1)
        movable = (distances < target) & (~opening_mask) & (~protected_vertex_mask)
        if not np.any(movable):
            iteration_records.append({
                "iteration": int(iteration),
                "direct_moved_vertices": 0,
                "adaptive_opening_boundary_vertices": int(np.count_nonzero(opening_mask)),
                "min_distance_before": float(np.min(distances)) if len(distances) else None,
                "p05_distance_before": float(np.percentile(distances, 5)) if len(distances) else None,
            })
            break
        unit = separation_unit_vectors(
            points[movable],
            ids[movable],
            prefer_outward=force_outward_from_fixed_surface,
        )
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
    support_ids = np.flatnonzero((ring_levels > 0) & total_direct_moved & (~protected_vertex_mask))
    support_source = "moved_vertices_in_opening_neighborhood"
    if len(support_ids) < 3:
        support_ids = np.flatnonzero((~opening_mask) & total_direct_moved & (~protected_vertex_mask))
        support_source = "all_directly_moved_non_opening_vertices"
    if len(support_ids) < 3:
        support_ids = np.flatnonzero((~opening_mask) & (~protected_vertex_mask))
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
    protected_opening = protected_vertex_mask[opening_ids]
    ring_displacement_smoothed[protected_opening] = 0.0
    displacement[opening_ids[~protected_opening]] = ring_displacement_smoothed[~protected_opening]

    extra_opening_ids = np.setdiff1d(
        np.flatnonzero(opening_mask & (~protected_vertex_mask)),
        opening_ids,
        assume_unique=False,
    )
    if len(extra_opening_ids):
        nearest = cKDTree(original_points[opening_ids]).query(original_points[extra_opening_ids], k=1)[1]
        displacement[extra_opening_ids] = ring_displacement_smoothed[np.asarray(nearest, dtype=np.int64)]

    band_iterations = int(args.rv_opening_band_smooth_iterations)
    band_relaxation = float(args.rv_opening_band_smooth_relaxation)
    if band_iterations > 0 and band_relaxation > 0.0 and np.any(ring_levels > 0):
        smooth_ids = np.flatnonzero((ring_levels > 0) & (~protected_vertex_mask))
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
            next_displacement[opening_ids[~protected_opening]] = ring_displacement_smoothed[~protected_opening]
            next_displacement[protected_vertex_mask] = 0.0
            displacement = next_displacement

    ring_floor_records = []
    max_level = max(int(args.rv_opening_adaptive_rings), 1)
    ring_tree = cKDTree(original_points[opening_ids])
    ring_floor_padding = max(1e-3, min(float(args.rv_distance_padding), 0.02))
    for floor_iteration in range(max(int(args.rv_opening_ring_floor_iterations), 0)):
        current_points = original_points + displacement
        ring_distances, ring_epi_ids = rv_epi_tree.query(current_points[opening_ids], k=1)
        ring_below = (ring_distances < requested_distance) & (~protected_opening)
        protected_ring_below = (ring_distances < requested_distance) & protected_opening
        record = {
            "iteration": int(floor_iteration),
            "opening_vertices_below_before": int(np.count_nonzero(ring_below)),
            "protected_opening_vertices_below_before": int(np.count_nonzero(protected_ring_below)),
            "min_opening_distance_before": float(np.min(ring_distances)) if len(ring_distances) else None,
        }
        if not np.any(ring_below):
            ring_floor_records.append(record)
            break
        unit = separation_unit_vectors(
            current_points[opening_ids][ring_below],
            ring_epi_ids[ring_below],
            prefer_outward=force_outward_from_fixed_surface,
        )
        steps = np.minimum(requested_distance - ring_distances[ring_below] + ring_floor_padding, float(args.rv_distance_max_step) * 0.5)
        correction = np.zeros((len(opening_ids), 3), dtype=float)
        correction[ring_below] = unit * steps[:, None]
        correction = smooth_closed_vectors(
            correction,
            max(3, int(args.rv_opening_ring_smooth_iterations) // 3),
            float(args.rv_opening_ring_smooth_relaxation),
        )
        correction, limited_floor = limit_displacement_vectors(correction, float(args.rv_distance_max_step) * 0.75)
        correction[protected_opening] = 0.0
        displacement[opening_ids[~protected_opening]] += correction[~protected_opening]
        if np.any(ring_levels > 0):
            for level in range(1, max_level + 1):
                level_ids = np.flatnonzero((ring_levels == level) & (~protected_vertex_mask))
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
    for correction_iteration in range(max(2, int(args.rv_distance_iterations))):
        distances, ids = rv_epi_tree.query(points, k=1)
        correct = (
            (distances < requested_distance)
            & (~protected_vertex_mask)
            & ((~opening_mask) | force_outward_from_fixed_surface)
        )
        if not np.any(correct):
            break
        unit = separation_unit_vectors(
            points[correct],
            ids[correct],
            prefer_outward=force_outward_from_fixed_surface,
        )
        steps = np.minimum(requested_distance - distances[correct] + 1e-3, float(args.rv_distance_max_step) * 0.5)
        points[correct] += unit * steps[:, None]
        final_correction_moved[correct] = True

    opening_ring_post_smoothing_records = []
    if force_outward_from_fixed_surface and len(opening_ids) >= 3:
        ring_source_points = original_points[opening_ids].copy()
        ring_displacement = points[opening_ids] - ring_source_points
        ring_displacement = smooth_closed_vectors(
            ring_displacement,
            int(args.rv_opening_ring_smooth_iterations),
            float(args.rv_opening_ring_smooth_relaxation),
        )
        ring_candidate = ring_source_points + ring_displacement
        ring_smoothing_target = target
        ring_candidate, opening_ring_post_smoothing_records = apply_uniform_ring_outward_offset(
            ring_candidate,
            ring_smoothing_target,
            protected_opening,
            max(4, int(args.rv_distance_iterations) * 3),
        )
        points[opening_ids[~protected_opening]] = ring_candidate[~protected_opening]

    displacement = points - original_points
    displacement[protected_vertex_mask] = 0.0
    points = original_points + displacement
    final_source_moved = np.linalg.norm(points - original_points, axis=1) > max(float(args.boundary_match_tolerance), 1e-6)
    moved = rv_endo.copy(deep=True)
    moved.points = points
    moved = moved.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    post_cleanup_correction_records = []
    post_cleanup_iterations = max(4, int(args.rv_distance_iterations) * 4)
    for correction_iteration in range(post_cleanup_iterations):
        moved_points = np.asarray(moved.points, dtype=float)
        distances, ids = rv_epi_tree.query(moved_points, k=1)
        correct = distances < requested_distance
        if len(protected_vertex_mask) == len(distances):
            protected_after_cleanup_now = map_source_points_to_target_mask(
                points[protected_vertex_mask],
                moved_points,
                max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0),
            )
            correct &= ~protected_after_cleanup_now
        # The opening ring is handled by the closed-loop smoothing pass below;
        # never apply independent cleanup corrections to its vertices.
        opening_after_cleanup_now = map_source_points_to_target_mask(
            points[opening_mask],
            moved_points,
            max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0),
        )
        correct &= ~opening_after_cleanup_now
        record = {
            "iteration": int(correction_iteration),
            "vertices_below_before": int(np.count_nonzero(correct)),
            "minimum_distance_before": float(np.min(distances)) if len(distances) else None,
        }
        if not np.any(correct):
            post_cleanup_correction_records.append(record)
            break
        unit = separation_unit_vectors(
            moved_points[correct],
            ids[correct],
            prefer_outward=force_outward_from_fixed_surface,
        )
        steps = np.minimum(
            requested_distance - distances[correct] + float(args.rv_distance_padding),
            float(args.rv_distance_max_step),
        )
        moved_points[correct] += unit * steps[:, None]
        moved.points = moved_points
        record.update({
            "vertices_corrected": int(np.count_nonzero(correct)),
            "maximum_step_mm": float(np.max(steps)) if len(steps) else 0.0,
        })
        post_cleanup_correction_records.append(record)
    cleanup_tolerance = max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0)
    moved_mask = map_source_points_to_target_mask(points[final_source_moved], np.asarray(moved.points, dtype=float), cleanup_tolerance)
    opening_after_cleanup = map_source_points_to_target_mask(points[opening_mask], np.asarray(moved.points, dtype=float), cleanup_tolerance)
    if force_outward_from_fixed_surface and np.any(opening_after_cleanup):
        moved_points = np.asarray(moved.points, dtype=float).copy()
        cleaned_ring_ids = np.flatnonzero(opening_after_cleanup)
        cleaned_ring_candidate, cleaned_ring_records = apply_uniform_ring_outward_offset(
            moved_points[cleaned_ring_ids],
            target,
            np.zeros(len(cleaned_ring_ids), dtype=bool),
            max(4, int(args.rv_distance_iterations) * 3),
        )
        moved_points[cleaned_ring_ids] = cleaned_ring_candidate
        moved.points = moved_points
        opening_ring_post_smoothing_records.append({
            "post_cleanup_uniform_offset": True,
            "records": cleaned_ring_records,
        })
        opening_after_cleanup = map_source_points_to_target_mask(
            points[opening_mask],
            np.asarray(moved.points, dtype=float),
            cleanup_tolerance,
        )
    band_after_cleanup = map_source_points_to_target_mask(points[transition_band_mask], np.asarray(moved.points, dtype=float), cleanup_tolerance)
    protected_after_cleanup = map_source_points_to_target_mask(
        points[protected_vertex_mask],
        np.asarray(moved.points, dtype=float),
        cleanup_tolerance,
    )
    final_opening_loop = base.make_polyline(points[opening_ids])

    final_distances = rv_epi_tree.query(np.asarray(moved.points, dtype=float), k=1)[0]
    below_requested = final_distances < requested_distance
    protected_below_requested = below_requested & protected_after_cleanup
    movable_below_requested = below_requested & (~protected_after_cleanup)
    return moved, moved_mask, opening_after_cleanup, band_after_cleanup, final_opening_loop, {
        "method": "fix_RVepi_and_move_RVendo_to_minimum_distance_with_adaptive_openbase_ring",
        "minimum_requested_distance_mm": requested_distance,
        "target_distance_with_padding_mm": target,
        "rv_epi_fixed": True,
        "force_outward_from_fixed_surface": bool(force_outward_from_fixed_surface),
        "rvendo_lv_epi_intersection_vertices_frozen": True,
        "protected_intersection_vertices": int(np.count_nonzero(protected_vertex_mask)),
        "protected_intersection_vertices_after_cleanup": int(np.count_nonzero(protected_after_cleanup)),
        "protected_opening_vertices": int(np.count_nonzero(protected_opening)),
        "movable_vertices": int(np.count_nonzero(~protected_vertex_mask)),
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
        "opening_ring_post_movement_smoothing_applied": bool(force_outward_from_fixed_surface),
        "opening_ring_post_movement_smoothing_iterations": int(args.rv_opening_ring_smooth_iterations) if force_outward_from_fixed_surface else 0,
        "opening_ring_post_movement_smoothing_records": opening_ring_post_smoothing_records,
        "ring_smooth_iterations": int(args.rv_opening_ring_smooth_iterations),
        "ring_smooth_relaxation": float(args.rv_opening_ring_smooth_relaxation),
        "band_smooth_iterations": int(args.rv_opening_band_smooth_iterations),
        "band_smooth_relaxation": float(args.rv_opening_band_smooth_relaxation),
        "opening_ring_distance_floor_iterations_requested": int(args.rv_opening_ring_floor_iterations),
        "opening_ring_distance_floor_padding_mm": float(ring_floor_padding),
        "opening_ring_distance_floor_iterations": ring_floor_records,
        "initial_vertices_below_requested_distance": int(np.count_nonzero(initial_distances < requested_distance)) if len(initial_distances) else 0,
        "initial_protected_vertices_below_requested_distance": int(np.count_nonzero(initial_distances[protected_vertex_mask] < requested_distance)) if np.any(protected_vertex_mask) else 0,
        "initial_movable_vertices_below_requested_distance": int(np.count_nonzero(initial_distances[~protected_vertex_mask] < requested_distance)) if np.any(~protected_vertex_mask) else 0,
        "initial_opening_vertices_below_requested_distance": int(np.count_nonzero((initial_distances < requested_distance) & opening_mask)) if len(initial_distances) else 0,
        "initial_nonopening_vertices_below_requested_distance": int(np.count_nonzero((initial_distances < requested_distance) & (~opening_mask))) if len(initial_distances) else 0,
        "initial_min_distance_all_vertices": float(np.min(initial_distances)) if len(initial_distances) else None,
        "initial_p05_distance_all_vertices": float(np.percentile(initial_distances, 5)) if len(initial_distances) else None,
        "iterations": iteration_records,
        "direct_vertices_moved_before_ring_adaptation": int(np.count_nonzero(total_direct_moved)),
        "final_correction_nonopening_vertices_moved": int(np.count_nonzero(final_correction_moved)),
        "final_correction_opening_vertices_moved": int(np.count_nonzero(final_correction_moved & opening_mask)),
        "post_cleanup_correction_iterations": post_cleanup_correction_records,
        "total_vertices_moved_after_adaptation": int(np.count_nonzero(final_source_moved)),
        "final_vertices_below_requested_distance": int(np.count_nonzero(below_requested)) if len(final_distances) else 0,
        "final_protected_vertices_below_requested_distance": int(np.count_nonzero(protected_below_requested)) if len(final_distances) else 0,
        "final_movable_vertices_below_requested_distance": int(np.count_nonzero(movable_below_requested)) if len(final_distances) else 0,
        "final_nonopening_vertices_below_requested_distance": int(np.count_nonzero(below_requested & (~opening_after_cleanup))) if len(final_distances) else 0,
        "final_adaptive_opening_vertices_below_requested_distance": int(np.count_nonzero(below_requested & opening_after_cleanup)) if len(final_distances) else 0,
        "final_min_distance_all_vertices": float(np.min(final_distances)) if len(final_distances) else None,
        "final_p05_distance_all_vertices": float(np.percentile(final_distances, 5)) if len(final_distances) else None,
    }


def enforce_rvepi_distance_adaptive_opening_ring(
    rv_epi: pv.PolyData,
    rv_endo: pv.PolyData,
    opening_loop: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, np.ndarray, np.ndarray, np.ndarray, pv.PolyData, Dict[str, object]]:
    """Move RVepi away from fixed RVendo using the existing adaptive-ring solver."""
    moved, moved_mask, opening_mask, transition_band_mask, final_opening_loop, info = (
        enforce_rvendo_distance_adaptive_opening_ring(
            rv_epi,
            rv_endo,
            opening_loop,
            args,
            protected_vertex_mask=np.zeros(
                rv_epi.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance).n_points,
                dtype=bool,
            ),
            force_outward_from_fixed_surface=True,
        )
    )
    info = dict(info)
    info.update({
        "method": "move_RVepi_to_minimum_distance_from_fixed_RVendo_with_adaptive_openbase_ring",
        "moving_surface": "RVepi",
        "fixed_surface": "RVendo",
        "rv_endo_fixed": True,
        "rv_epi_fixed": False,
        "movement_side_constraint": "outward_from_fixed_RVendo_surface",
        "LVepi_intersection_handling": "disabled_for_active_step5",
    })
    return moved, moved_mask, opening_mask, transition_band_mask, final_opening_loop, info


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


def nearest_eligible_lvepi_vertices_for_guide_points(
    guide_points: np.ndarray,
    lv_points: np.ndarray,
    eligible_mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    guide = np.asarray(guide_points, dtype=float)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    if len(guide) == 0 or len(lv_points) == 0:
        return np.empty((0,), dtype=np.int64), {"method": "empty_input", "guide_points": int(len(guide))}
    eligible_ids = np.flatnonzero(eligible_mask)
    if len(eligible_ids) < 3:
        raise RuntimeError("Fewer than three LVepi vertices satisfy the Step 7 interface-ring distance constraint.")

    k = int(np.clip(int(getattr(args, "step7_interface_ring_candidate_k", 256)), 1, len(lv_points)))
    distances, ids = cKDTree(lv_points).query(guide, k=k)
    if k == 1:
        distances = distances.reshape((-1, 1))
        ids = ids.reshape((-1, 1))
    selected: List[int] = []
    selected_distances: List[float] = []
    fallback_global = 0
    eligible_tree = cKDTree(lv_points[eligible_ids])
    for row_id in range(len(guide)):
        row_ids = np.asarray(ids[row_id], dtype=np.int64)
        row_distances = np.asarray(distances[row_id], dtype=float)
        valid = eligible_mask[row_ids]
        if np.any(valid):
            first = int(np.flatnonzero(valid)[0])
            selected.append(int(row_ids[first]))
            selected_distances.append(float(row_distances[first]))
            continue
        fallback_distance, fallback_local_id = eligible_tree.query(guide[row_id], k=1)
        selected.append(int(eligible_ids[int(fallback_local_id)]))
        selected_distances.append(float(fallback_distance))
        fallback_global += 1
    ordered = ordered_unique_indices(np.asarray(selected, dtype=np.int64))
    return ordered, {
        "method": "nearest_original_LVepi_vertices_satisfying_step7_distance_constraints",
        "guide_points": int(len(guide)),
        "candidate_k": int(k),
        "eligible_LVepi_vertices": int(len(eligible_ids)),
        "selected_vertices_before_unique": int(len(selected)),
        "unique_selected_vertices": int(len(ordered)),
        "guide_points_requiring_global_eligible_fallback": int(fallback_global),
        "selected_distance_to_guide_min": float(np.min(selected_distances)) if selected_distances else None,
        "selected_distance_to_guide_mean": float(np.mean(selected_distances)) if selected_distances else None,
        "selected_distance_to_guide_p95": float(np.percentile(selected_distances, 95)) if selected_distances else None,
        "selected_distance_to_guide_max": float(np.max(selected_distances)) if selected_distances else None,
    }


def select_step7_lvepi_interface_loop_away_from_rvepi_boundary(
    name: str,
    guide_loop: pv.PolyData,
    rv_epi_boundary_loops: Sequence[pv.PolyData],
    rv_epi_surface: pv.PolyData,
    lv_epi_reference: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, pv.PolyData, Dict[str, object]]:
    guide_pts = clean_loop_points(loop_points(guide_loop))
    if len(guide_pts) < 3:
        raise RuntimeError(f"{name}: guide RVendo/LVepi loop has fewer than three points.")
    rv_epi_boundary_blocks = [clean_loop_points(loop_points(loop)) for loop in rv_epi_boundary_loops]
    rv_epi_boundary_blocks = [pts for pts in rv_epi_boundary_blocks if len(pts) >= 3]
    if not rv_epi_boundary_blocks:
        raise RuntimeError(f"{name}: no fixed RVepi/LVepi boundary ring was supplied for the 3 mm constraint.")

    lv_epi = lv_epi_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_points = np.asarray(lv_epi.points, dtype=float)
    lv_triangles = base.triangle_faces(lv_epi)
    if len(lv_points) < 3 or len(lv_triangles) == 0:
        raise RuntimeError(f"{name}: LVepi reference surface is empty.")
    neighbors, edge_set = lvepi_vertex_edge_graph(lv_epi.n_points, lv_triangles)

    rv_epi_boundary_points = np.vstack(rv_epi_boundary_blocks)
    boundary_distances = cKDTree(rv_epi_boundary_points).query(lv_points, k=1)[0]
    _, rv_epi_surface_distances = closest_points_on_surface(rv_epi_surface, lv_points)
    requested = float(args.rv_min_distance_to_rvepi)
    tolerance = max(float(getattr(args, "step7_interface_ring_distance_tolerance", 1e-5)), float(args.surface_clean_tolerance) * 100.0)
    eligible_mask = (boundary_distances + tolerance >= requested) & (rv_epi_surface_distances + tolerance >= requested)
    if np.count_nonzero(eligible_mask) < 3:
        raise RuntimeError(
            f"{name}: fewer than three LVepi vertices satisfy both Step 7 distance constraints "
            f"(requested={requested:g} mm)."
        )

    guide_counts, guide_count_info = step56_candidate_guide_counts(len(guide_pts), args)
    if not guide_counts:
        raise RuntimeError(f"{name}: cannot create sparse guide counts for the RVendo interface loop.")
    lv_edge_lengths = [float(np.linalg.norm(lv_points[int(a)] - lv_points[int(b)])) for a, b in edge_set]
    median_lv_edge = float(np.median(lv_edge_lengths)) if lv_edge_lengths else 1.0
    proximity_weight = max(float(getattr(args, "step56_lvepi_path_proximity_weight", 0.35)), 0.0)

    best_candidate: Optional[Dict[str, object]] = None
    candidate_records: List[Dict[str, object]] = []
    for guide_count in guide_counts:
        sparse_guide = sparse_closed_loop_guide_points(guide_pts, int(guide_count))
        if len(sparse_guide) < 3:
            continue
        smooth_iterations = int(getattr(args, "step56_guide_smooth_iterations", 12))
        smooth_relaxation = float(getattr(args, "step56_guide_smooth_relaxation", 0.12))
        smoothed_guide = smooth_closed_vectors(sparse_guide, smooth_iterations, smooth_relaxation)
        seed_ids, seed_info = nearest_eligible_lvepi_vertices_for_guide_points(
            smoothed_guide,
            lv_points,
            eligible_mask,
            args,
        )
        if len(seed_ids) < 3:
            candidate_records.append({
                "guide_count": int(guide_count),
                "created": False,
                "reason": "fewer_than_three_unique_eligible_LVepi_seed_vertices",
                "seed_selection": seed_info,
            })
            continue
        ordered_seed_ids, order_info = order_lvepi_seam_vertex_ids(
            f"{name}_step7_eligible_seed_order_{guide_count}",
            lv_points,
            seed_ids,
            edge_set,
        )
        guide_curve_sample_count = max(48, min(400, len(smoothed_guide) * 4))
        guide_curve = base.resample_closed_loop(smoothed_guide, guide_curve_sample_count)
        guide_distances = cKDTree(guide_curve).query(lv_points, k=1)[0]
        node_penalty = guide_distances * proximity_weight
        traced_ids, path_info = trace_lvepi_seed_loop_on_graph_strict_allowed(
            ordered_seed_ids,
            neighbors,
            lv_points,
            eligible_mask,
            node_penalty,
        )
        if len(traced_ids) < 3:
            candidate_records.append({
                "guide_count": int(guide_count),
                "created": False,
                "reason": "strict_eligible_LVepi_graph_trace_failed",
                "seed_selection": seed_info,
                "seed_ordering": order_info,
                "path_trace": path_info,
            })
            continue
        quality = lvepi_line_loop_quality_record(
            traced_ids,
            lv_points,
            edge_set,
            guide_distances,
            len(guide_pts),
        )
        traced_boundary_distances = boundary_distances[traced_ids]
        traced_surface_distances = rv_epi_surface_distances[traced_ids]
        distance_record = {
            "requested_min_distance_mm": float(requested),
            "tolerance_mm": float(tolerance),
            "min_distance_to_fixed_RVepi_boundary_ring": float(np.min(traced_boundary_distances)) if len(traced_boundary_distances) else None,
            "mean_distance_to_fixed_RVepi_boundary_ring": float(np.mean(traced_boundary_distances)) if len(traced_boundary_distances) else None,
            "p05_distance_to_fixed_RVepi_boundary_ring": float(np.percentile(traced_boundary_distances, 5)) if len(traced_boundary_distances) else None,
            "min_distance_to_fixed_RVepi_surface": float(np.min(traced_surface_distances)) if len(traced_surface_distances) else None,
            "mean_distance_to_fixed_RVepi_surface": float(np.mean(traced_surface_distances)) if len(traced_surface_distances) else None,
            "p05_distance_to_fixed_RVepi_surface": float(np.percentile(traced_surface_distances, 5)) if len(traced_surface_distances) else None,
            "vertices_below_boundary_ring_distance": int(np.count_nonzero(traced_boundary_distances + tolerance < requested)),
            "vertices_below_surface_distance": int(np.count_nonzero(traced_surface_distances + tolerance < requested)),
        }
        valid = bool(
            quality.get("valid", False)
            and distance_record["vertices_below_boundary_ring_distance"] == 0
            and distance_record["vertices_below_surface_distance"] == 0
        )
        score = float(quality.get("quality_score", 0.0))
        score += float(distance_record["vertices_below_boundary_ring_distance"] + distance_record["vertices_below_surface_distance"]) * 10000.0
        if distance_record["min_distance_to_fixed_RVepi_boundary_ring"] is not None:
            score += max(0.0, requested - float(distance_record["min_distance_to_fixed_RVepi_boundary_ring"])) * 1000.0
        if distance_record["min_distance_to_fixed_RVepi_surface"] is not None:
            score += max(0.0, requested - float(distance_record["min_distance_to_fixed_RVepi_surface"])) * 1000.0
        candidate = {
            "guide_count": int(guide_count),
            "created": True,
            "valid": bool(valid),
            "score": float(score),
            "sparse_guide_points": int(len(sparse_guide)),
            "ordered_seed_vertices": int(len(ordered_seed_ids)),
            "traced_existing_LVepi_vertices": int(len(traced_ids)),
            "median_LVepi_graph_edge_length_mm": float(median_lv_edge),
            "seed_selection": seed_info,
            "seed_ordering": order_info,
            "path_trace": path_info,
            "quality": quality,
            "distance_constraints": distance_record,
            "lv_epi_vertex_ids": [int(v) for v in traced_ids.tolist()],
        }
        candidate_records.append(candidate)
        if not valid:
            continue
        if best_candidate is None or float(candidate["score"]) < float(best_candidate["score"]):
            best_candidate = candidate

    if best_candidate is None:
        valid_summary = [{k: rec.get(k) for k in ("guide_count", "created", "valid", "score", "reason")} for rec in candidate_records]
        raise RuntimeError(f"{name}: no valid original-LVepi RVendo boundary loop satisfied the 3 mm RVepi constraint: {json.dumps(valid_summary, indent=2)}")

    selected_ids = np.asarray(best_candidate["lv_epi_vertex_ids"], dtype=np.int64)
    selected_loop = base.make_polyline(lv_points[selected_ids])
    return selected_ids, selected_loop, {
        "method": "select_new_RVendo_LVepi_interface_boundary_ring_from_original_LVepi_vertices_with_min_3mm_distance_to_fixed_RVepi_boundary_ring",
        "boundary_points_created": 0,
        "boundary_points_source": "original_LVepi_vertices_only",
        "RVepi_fixed": True,
        "requested_min_distance_to_RVepi_boundary_ring_mm": float(requested),
        "requested_min_distance_to_RVepi_surface_mm": float(requested),
        "eligible_LVepi_vertices": int(np.count_nonzero(eligible_mask)),
        "total_LVepi_vertices": int(len(lv_points)),
        "guide_loop": loop_record(guide_loop),
        "fixed_RVepi_boundary_loops": [loop_record(loop) for loop in rv_epi_boundary_loops],
        "sparse_guide_count_selection": guide_count_info,
        "selected_guide_count": int(best_candidate["guide_count"]),
        "selected_loop": loop_record(selected_loop),
        "selected_loop_quality": best_candidate["quality"],
        "selected_loop_distance_constraints": best_candidate["distance_constraints"],
        "selected_loop_vertex_ids": [int(v) for v in selected_ids.tolist()],
        "candidate_records": candidate_records,
    }


def select_step7_lvepi_interface_loops_away_from_rvepi_boundary(
    rv_endo_guide_loops: Sequence[pv.PolyData],
    rv_epi_boundary_loops: Sequence[pv.PolyData],
    rv_epi_surface: pv.PolyData,
    lv_epi_reference: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], List[pv.PolyData], Dict[str, object]]:
    if not rv_endo_guide_loops:
        raise RuntimeError("Step 7 requires the Step 6 RVendo/LVepi seam loop as the guide for the new boundary ring.")
    vertex_id_loops: List[np.ndarray] = []
    output_loops: List[pv.PolyData] = []
    records: List[Dict[str, object]] = []
    for loop_id, guide_loop in enumerate(rv_endo_guide_loops):
        ids, loop, info = select_step7_lvepi_interface_loop_away_from_rvepi_boundary(
            f"RVendo_step7_LVepi_interface_loop_{loop_id}",
            guide_loop,
            rv_epi_boundary_loops,
            rv_epi_surface,
            lv_epi_reference,
            args,
        )
        vertex_id_loops.append(ids)
        output_loops.append(loop)
        info["loop_id"] = int(loop_id)
        records.append(info)
    return vertex_id_loops, output_loops, {
        "method": "select_all_step7_RVendo_LVepi_interface_boundary_rings_from_original_LVepi_vertices_with_3mm_RVepi_distance_constraint",
        "loop_count": int(len(output_loops)),
        "loops": records,
    }


def enforce_surface_distance_to_rvepi_preserving_lvepi_boundary(
    name: str,
    surface: pv.PolyData,
    rv_epi_surface: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, np.ndarray, Dict[str, object]]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    rv_epi_surface = rv_epi_surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    points = np.asarray(surface.points, dtype=float).copy()
    original = points.copy()
    triangles = base.triangle_faces(surface)
    tags = np.rint(np.asarray(surface.cell_data.get("elemTag", np.full(len(triangles), int(args.rv_tag), dtype=np.int32))).reshape(-1)).astype(np.int32)
    point_data = {key: np.asarray(value).copy() for key, value in surface.point_data.items() if len(value) == surface.n_points}
    fixed_mask = np.asarray(point_data.get("lvepiSubstituteVertex", np.zeros(surface.n_points, dtype=np.int32))).reshape(-1).astype(bool)
    rv_epi_points = np.asarray(rv_epi_surface.points, dtype=float)
    rv_epi_tree = cKDTree(rv_epi_points)
    requested = float(args.rv_min_distance_to_rvepi)
    target = requested + float(args.rv_distance_padding)
    iteration_records: List[Dict[str, object]] = []
    for iteration in range(max(1, int(args.rv_distance_iterations))):
        distances, ids = rv_epi_tree.query(points, k=1)
        movable = (distances < target) & (distances > 1e-8) & (~fixed_mask)
        iteration_records.append({
            "iteration": int(iteration),
            "movable_vertices_below_target": int(np.count_nonzero(movable)),
            "fixed_LVepi_boundary_vertices": int(np.count_nonzero(fixed_mask)),
            "min_distance_before": float(np.min(distances)) if len(distances) else None,
            "p05_distance_before": float(np.percentile(distances, 5)) if len(distances) else None,
        })
        if not np.any(movable):
            break
        nearest = rv_epi_points[ids[movable]]
        vectors = points[movable] - nearest
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)[:, None]
        steps = np.minimum(target - distances[movable], float(args.rv_distance_max_step))
        points[movable] += unit * steps[:, None]
        iteration_records[-1]["max_step_mm"] = float(np.max(steps)) if len(steps) else 0.0
    moved_mask_source = np.linalg.norm(points - original, axis=1) > max(float(args.boundary_match_tolerance), 1e-6)
    corrected = compact_tagged_triangle_surface_with_point_data(
        points,
        triangles,
        tags,
        point_data,
        args.surface_clean_tolerance,
    )
    cleanup_tolerance = max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0)
    moved_mask = map_source_points_to_target_mask(points[moved_mask_source], np.asarray(corrected.points, dtype=float), cleanup_tolerance)
    fixed_after = np.asarray(corrected.point_data.get("lvepiSubstituteVertex", np.zeros(corrected.n_points, dtype=np.int32))).reshape(-1).astype(bool)
    final_distances = rv_epi_tree.query(np.asarray(corrected.points, dtype=float), k=1)[0] if corrected.n_points else np.empty((0,), dtype=float)
    return corrected, moved_mask, {
        "method": "final_RVendo_distance_correction_preserving_selected_original_LVepi_interface_boundary_vertices",
        "fixed_boundary_rule": "lvepiSubstituteVertex points are not moved by final distance correction",
        "requested_min_distance_mm": float(requested),
        "target_distance_with_padding_mm": float(target),
        "fixed_boundary_vertices_before_cleanup": int(np.count_nonzero(fixed_mask)),
        "fixed_boundary_vertices_after_cleanup": int(np.count_nonzero(fixed_after)),
        "iterations": iteration_records,
        "moved_nonfixed_vertices_after_interface_rebuild": int(np.count_nonzero(moved_mask)),
        "final_vertices_below_requested_distance": int(np.count_nonzero(final_distances < requested)) if len(final_distances) else 0,
        "final_nonfixed_vertices_below_requested_distance": int(np.count_nonzero((final_distances < requested) & (~fixed_after))) if len(final_distances) else 0,
        "final_fixed_boundary_vertices_below_requested_distance": int(np.count_nonzero((final_distances < requested) & fixed_after)) if len(final_distances) else 0,
        "final_min_distance_all_vertices": float(np.min(final_distances)) if len(final_distances) else None,
        "final_p05_distance_all_vertices": float(np.percentile(final_distances, 5)) if len(final_distances) else None,
    }


def rebuild_rvendo_lvepi_interface_after_step7(
    name: str,
    moved_core: pv.PolyData,
    reference_cut_loops: Sequence[pv.PolyData],
    lv_seam_vertex_id_loops: Sequence[np.ndarray],
    lv_seam_loops: Sequence[pv.PolyData],
    lv_epi_reference: pv.PolyData,
    rv_epi_surface: pv.PolyData,
    elem_tag: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    moved_core = moved_core.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    moved_cut_loops: List[pv.PolyData] = []
    match_records: List[Dict[str, object]] = []
    for loop_id, reference_loop in enumerate(reference_cut_loops):
        selected_loop, selected_info = select_boundary_loop_by_reference(
            f"{name}_RV_cut_loop_{loop_id}_after_core_distance_move",
            moved_core,
            reference_loop,
        )
        moved_cut_loops.append(selected_loop)
        match_records.append(selected_info)
    if len(moved_cut_loops) != len(lv_seam_vertex_id_loops):
        raise RuntimeError(
            f"{name}: moved RV cut loop count ({len(moved_cut_loops)}) does not match new LVepi seam loop count ({len(lv_seam_vertex_id_loops)})."
        )
    stitch_surface, stitch_info = make_lvepi_edge_conforming_stitch_surfaces(
        name,
        moved_cut_loops,
        lv_seam_vertex_id_loops,
        lv_epi_reference,
        elem_tag,
        args,
    )
    core_tagged = mesh_with_default_step56_point_data(moved_core, elem_tag)
    stitched = merge_tagged_triangle_surfaces_with_point_data(
        [core_tagged, stitch_surface],
        args.surface_clean_tolerance,
    )
    corrected, corrected_moved_mask, distance_info = enforce_surface_distance_to_rvepi_preserving_lvepi_boundary(
        name,
        stitched,
        rv_epi_surface,
        args,
    )
    substitute_mask = np.asarray(corrected.point_data.get("lvepiSubstituteVertex", np.zeros(corrected.n_points, dtype=np.int32))).reshape(-1).astype(bool)
    return {
        "surface": corrected,
        "trimmed_surface": moved_core,
        "stitch_surface": stitch_surface,
        "rv_cut_loops": moved_cut_loops,
        "lv_seam_loops": list(lv_seam_loops),
        "cut_mask": substitute_mask,
        "corrected_moved_mask": corrected_moved_mask,
        "info": {
            "method": "rebuild_RVendo_LVepi_interface_after_step7_using_new_original_LVepi_boundary_ring_satisfying_RVepi_3mm_distance",
            "RVepi_fixed": True,
            "boundary_points_created": 0,
            "boundary_points_source": "original_LVepi_vertices_only",
            "moved_core_cut_loop_match": match_records,
            "new_RVendo_LVepi_boundary_loops": [loop_record(loop) for loop in lv_seam_loops],
            "stitch": stitch_info,
            "distance_correction_after_stitch": distance_info,
            "surface_with_new_lvepi_boundary_ring": surface_summary(corrected),
        },
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


def connected_tet_components_from_adjacency(mask: np.ndarray, adjacency: Sequence[Sequence[int]]) -> List[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(len(mask), dtype=bool)
    components: List[np.ndarray] = []
    for start in np.flatnonzero(mask):
        start = int(start)
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: List[int] = []
        while stack:
            current = stack.pop()
            comp.append(int(current))
            for nb in adjacency[current]:
                nb = int(nb)
                if mask[nb] and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(np.asarray(comp, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def cleanup_small_tag_islands_by_adjacency(
    tags: np.ndarray,
    cells: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    tags = np.asarray(tags, dtype=np.int32).copy()
    if len(tags) == 0 or not bool(args.tag_island_cleanup):
        return tags, {
            "applied": False,
            "reason": "disabled_or_empty",
            "elemTag_counts_before": {str(k): int(v) for k, v in sorted(Counter(map(int, tags)).items())},
            "elemTag_counts_after": {str(k): int(v) for k, v in sorted(Counter(map(int, tags)).items())},
        }

    adjacency = tet_face_adjacency(cells)
    iterations = max(1, int(args.tag_island_cleanup_iterations))
    min_cells = max(1, int(args.tag_island_min_cells))
    min_fraction = max(0.0, float(args.tag_island_min_fraction))
    tag_values = [int(args.lv_tag), int(args.rv_tag)]
    counts_before = Counter(map(int, tags))
    iteration_records: List[Dict[str, object]] = []
    total_changed = 0

    for iteration in range(iterations):
        changed_this_iteration = 0
        records: List[Dict[str, object]] = []
        updated = tags.copy()
        for tag_value in tag_values:
            mask = tags == int(tag_value)
            components = connected_tet_components_from_adjacency(mask, adjacency)
            if not components:
                records.append({
                    "tag": int(tag_value),
                    "component_count": 0,
                    "threshold_cells": int(min_cells),
                    "reassigned_component_count": 0,
                })
                continue
            largest = int(len(components[0]))
            threshold = max(min_cells, int(np.ceil(largest * min_fraction)))
            component_records: List[Dict[str, object]] = []
            reassigned_component_count = 0
            reassigned_cell_count = 0
            for component_id, comp in enumerate(components):
                comp_size = int(len(comp))
                keep_component = component_id == 0 or comp_size >= threshold
                record: Dict[str, object] = {
                    "component_id": int(component_id),
                    "cells": comp_size,
                    "kept": bool(keep_component),
                }
                if not keep_component:
                    neighbor_tags: List[int] = []
                    comp_set = set(int(v) for v in comp.tolist())
                    for cell_id in comp:
                        for nb in adjacency[int(cell_id)]:
                            if int(nb) not in comp_set and int(tags[int(nb)]) != int(tag_value):
                                neighbor_tags.append(int(tags[int(nb)]))
                    if neighbor_tags:
                        replacement = Counter(neighbor_tags).most_common(1)[0][0]
                        updated[comp] = int(replacement)
                        changed_this_iteration += comp_size
                        reassigned_component_count += 1
                        reassigned_cell_count += comp_size
                        record.update({
                            "reassigned": True,
                            "replacement_tag": int(replacement),
                            "neighbor_tag_counts": {str(k): int(v) for k, v in sorted(Counter(neighbor_tags).items())},
                        })
                    else:
                        record.update({
                            "reassigned": False,
                            "reason": "no_neighboring_opposite_tag_cells_found",
                        })
                component_records.append(record)
            records.append({
                "tag": int(tag_value),
                "component_count": int(len(components)),
                "largest_component_cells": largest,
                "threshold_cells": int(threshold),
                "reassigned_component_count": int(reassigned_component_count),
                "reassigned_cell_count": int(reassigned_cell_count),
                "components_sample": component_records[:30],
            })
        tags = updated
        total_changed += int(changed_this_iteration)
        iteration_records.append({
            "iteration": int(iteration + 1),
            "changed_cells": int(changed_this_iteration),
            "tag_records": records,
            "elemTag_counts_after_iteration": {str(k): int(v) for k, v in sorted(Counter(map(int, tags)).items())},
        })
        if changed_this_iteration == 0:
            break

    return tags, {
        "applied": bool(total_changed > 0),
        "method": "remove_small_isolated_LV_RV_tag_components_by_tet_face_adjacency_majority_neighbor_reassignment",
        "iterations_requested": int(iterations),
        "iterations_run": int(len(iteration_records)),
        "tag_island_min_cells": int(min_cells),
        "tag_island_min_fraction_of_largest_same_tag_component": float(min_fraction),
        "total_reassigned_cells": int(total_changed),
        "elemTag_counts_before": {str(k): int(v) for k, v in sorted(counts_before.items())},
        "elemTag_counts_after": {str(k): int(v) for k, v in sorted(Counter(map(int, tags)).items())},
        "iterations": iteration_records,
    }


def classify_by_lv_reference_surface(
    points: np.ndarray,
    cells: np.ndarray,
    lv_reference: pv.PolyData,
    args: argparse.Namespace,
    reference_name: str = "original_closed_LVepi",
) -> Tuple[np.ndarray, Dict[str, object]]:
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    centroids = points[cells].mean(axis=1) if len(cells) else np.empty((0, 3), dtype=float)
    reference = lv_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)

    centroid_inside_lv = base.enclosed_mask(centroids, reference) if len(centroids) else np.empty((0,), dtype=bool)
    vertex_inside_points = base.enclosed_mask(points, reference) if len(points) else np.empty((0,), dtype=bool)
    vertex_inside_counts = np.count_nonzero(vertex_inside_points[cells], axis=1) if len(cells) else np.empty((0,), dtype=np.int32)
    vertex_threshold = int(np.clip(int(args.lv_tag_vertex_inside_threshold), 1, 4))

    lv_initial = np.asarray(centroid_inside_lv, dtype=bool) | (vertex_inside_counts >= vertex_threshold)
    tags_initial = np.full(len(cells), int(args.rv_tag), dtype=np.int32)
    tags_initial[lv_initial] = int(args.lv_tag)
    tags, cleanup_info = cleanup_small_tag_islands_by_adjacency(tags_initial, cells, args)

    counts_initial = Counter(map(int, tags_initial))
    counts = Counter(map(int, tags))
    total = int(len(tags))
    min_cells = int(np.ceil(total * float(args.min_region_cell_fraction))) if total else 0
    lv_count = int(counts.get(int(args.lv_tag), 0))
    rv_count = int(counts.get(int(args.rv_tag), 0))
    passes = bool(total > 0 and lv_count >= min_cells and rv_count >= min_cells)

    centroid_inside_count = int(np.count_nonzero(centroid_inside_lv))
    vertex_threshold_count = int(np.count_nonzero(vertex_inside_counts >= vertex_threshold))
    centroid_and_vertex_count = int(np.count_nonzero(centroid_inside_lv & (vertex_inside_counts >= vertex_threshold))) if len(cells) else 0
    centroid_only_count = int(np.count_nonzero(centroid_inside_lv & (vertex_inside_counts < vertex_threshold))) if len(cells) else 0
    vertex_only_count = int(np.count_nonzero((~centroid_inside_lv) & (vertex_inside_counts >= vertex_threshold))) if len(cells) else 0

    vertex_hist = Counter(map(int, vertex_inside_counts))
    info = {
        "method": "original_closed_LVepi_constraint_centroid_or_vertex_majority_then_small_tag_island_cleanup",
        "reference_name": reference_name,
        "reference_surface": surface_summary(reference),
        "total_tetrahedra": total,
        "lv_tag_rule": f"elemTag={int(args.lv_tag)} when tet centroid is inside original closed LVepi OR at least {vertex_threshold}/4 vertices are inside original closed LVepi; otherwise elemTag={int(args.rv_tag)} before cleanup",
        "centroid_inside_lv_count": centroid_inside_count,
        "vertex_inside_threshold": vertex_threshold,
        "vertex_threshold_inside_lv_count": vertex_threshold_count,
        "centroid_and_vertex_threshold_agree_lv_count": centroid_and_vertex_count,
        "centroid_only_lv_count": centroid_only_count,
        "vertex_threshold_only_lv_count": vertex_only_count,
        "vertex_inside_count_histogram": {str(k): int(v) for k, v in sorted(vertex_hist.items())},
        "initial_elemTag_counts_before_cleanup": {str(k): int(v) for k, v in sorted(counts_initial.items())},
        "cleanup": cleanup_info,
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


def write_custom_point_cloud_vtk(
    filename: Path,
    points: np.ndarray,
    elem_tags: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a legacy VTK point cloud using one VTK_VERTEX cell per point."""
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    elem_tags = np.asarray(elem_tags, dtype=np.int32).reshape(-1)
    if len(points) != len(elem_tags):
        raise ValueError(f"Point count {len(points)} does not match elemTag count {len(elem_tags)}.")

    with Path(filename).open("w") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("surface point cloud\n")
        handle.write("ASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n\n")

        handle.write(f"POINTS {len(points)} float\n")
        if len(points):
            np.savetxt(handle, points, fmt="%.5f", delimiter=" ")
        handle.write("\n")

        handle.write(f"CELLS {len(points)} {2 * len(points)}\n")
        if len(points):
            cell_lines = np.column_stack([
                np.ones(len(points), dtype=np.int32),
                np.arange(len(points), dtype=np.int32),
            ])
            np.savetxt(handle, cell_lines, fmt="%d")
        handle.write("\n")

        handle.write(f"CELL_TYPES {len(points)}\n")
        if len(points):
            np.savetxt(handle, np.ones(len(points), dtype=np.int32), fmt="%d")
        handle.write("\n")

        handle.write(f"CELL_DATA {len(points)}\n")
        handle.write("SCALARS elemTag int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        if len(elem_tags):
            np.savetxt(handle, elem_tags.reshape(-1, 1), fmt="%d")
        handle.write("\n")

        valid_point_data = []
        for name, values in (point_data or {}).items():
            array = np.asarray(values)
            if len(array) == len(points):
                valid_point_data.append((name, array))
        if not valid_point_data:
            return

        handle.write(f"POINT_DATA {len(points)}\n")
        for name, array in valid_point_data:
            vtk_name = "".join(char if char.isalnum() or char == "_" else "_" for char in str(name))
            flat = array.reshape((len(points), -1))[:, 0] if array.ndim != 1 else array
            if np.issubdtype(flat.dtype, np.integer) or np.issubdtype(flat.dtype, np.bool_):
                handle.write(f"SCALARS {vtk_name} int 1\n")
                handle.write("LOOKUP_TABLE default\n")
                if len(flat):
                    np.savetxt(handle, flat.astype(np.int32).reshape(-1, 1), fmt="%d")
            else:
                handle.write(f"SCALARS {vtk_name} float 1\n")
                handle.write("LOOKUP_TABLE default\n")
                if len(flat):
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

def write_custom_line_loop_vtk(
    filename: Path,
    loops: Sequence[pv.PolyData],
    loop_tag: int = 0,
) -> Dict[str, object]:
    point_blocks: List[np.ndarray] = []
    line_rows: List[np.ndarray] = []
    loop_ids: List[int] = []
    offset = 0
    loop_records: List[Dict[str, object]] = []
    for loop_id, loop in enumerate(loops):
        pts = clean_loop_points(loop_points(loop))
        if len(pts) < 2:
            continue
        closed_pts = np.vstack([pts, pts[0]])
        ids = np.arange(offset, offset + len(closed_pts), dtype=np.int64)
        point_blocks.append(closed_pts)
        line_rows.append(np.concatenate([[len(ids)], ids]).astype(np.int64))
        loop_ids.append(int(loop_id))
        loop_records.append(loop_record(base.make_polyline(pts)))
        offset += len(closed_pts)

    all_points = np.vstack(point_blocks) if point_blocks else np.empty((0, 3), dtype=float)
    with Path(filename).open("w") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("vtk output\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n\n")
        handle.write(f"POINTS {len(all_points)} float\n")
        if len(all_points):
            np.savetxt(handle, all_points, fmt="%.5f", delimiter=" ")
        handle.write("\n")
        total_line_size = int(sum(len(row) for row in line_rows))
        handle.write(f"LINES {len(line_rows)} {total_line_size}\n")
        for row in line_rows:
            handle.write(" ".join(str(int(v)) for v in row) + "\n")
        handle.write("\n")
        if line_rows:
            handle.write(f"CELL_DATA {len(line_rows)}\n")
            handle.write("SCALARS elemTag int 1\n")
            handle.write("LOOKUP_TABLE default\n")
            np.savetxt(handle, np.full((len(line_rows), 1), int(loop_tag), dtype=np.int32), fmt="%d")
            handle.write("\n")
            handle.write("SCALARS loopId int 1\n")
            handle.write("LOOKUP_TABLE default\n")
            np.savetxt(handle, np.asarray(loop_ids, dtype=np.int32).reshape(-1, 1), fmt="%d")
            handle.write("\n")
    return {
        "path": str(Path(filename).resolve()),
        "writer": "custom_ascii_polydata_line_loop",
        "points": int(len(all_points)),
        "line_loops": int(len(line_rows)),
        "elemTag": int(loop_tag),
        "loops": loop_records,
    }


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
    extra_surfaces: Optional[Sequence[Tuple[str, pv.PolyData, Optional[np.ndarray], Optional[int]]]] = None,
    extra_line_loops: Optional[Sequence[Tuple[str, Sequence[pv.PolyData], int]]] = None,
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

    for name, mesh, cut_mask, elem_tag in extra_surfaces or []:
        surface_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = save_triangle_surface_vtk(surface_path, mesh, cut_mask, elem_tag=elem_tag)
        paths[name] = str(surface_path.resolve())

    for name, loops, elem_tag in extra_line_loops or []:
        loop_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = write_custom_line_loop_vtk(loop_path, loops, loop_tag=int(elem_tag))
        paths[name] = str(loop_path.resolve())

    summary = {
        "description": "Step 5/6 outputs saved after deleting whole RV triangles with any LVepi-intersecting vertex, generating LVepi-based boundary line loops, and repairing RV boundaries with internal support-point transition bands.",
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
    extra_surfaces: Optional[Sequence[Tuple[str, pv.PolyData, Optional[np.ndarray], Optional[int]]]] = None,
    extra_line_loops: Optional[Sequence[Tuple[str, Sequence[pv.PolyData], int]]] = None,
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

    for name, mesh, cut_mask, elem_tag in extra_surfaces or []:
        surface_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = save_triangle_surface_vtk(surface_path, mesh, cut_mask, elem_tag=elem_tag)
        paths[name] = str(surface_path.resolve())

    for name, loops, elem_tag in extra_line_loops or []:
        loop_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = write_custom_line_loop_vtk(loop_path, loops, loop_tag=int(elem_tag))
        paths[name] = str(loop_path.resolve())

    summary = {
        "description": "Step 7 outputs after keeping RVepi fixed, moving RVendo by the >=3 mm rule, selecting a new RVendo LVepi-interface boundary ring from original LVepi vertices that is >=3 mm from the fixed RVepi boundary ring, and rebuilding the RV open-base annulus.",
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
    lv_epi_fixed = tagged_surface(lv_epi_open.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance), args.lv_tag)
    lv_endo_fixed = tagged_surface(lv_endo_open.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance), args.lv_tag)
    rv_epi_fixed = tagged_surface(rv_epi.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance), args.rv_tag)
    rv_endo_fixed = tagged_surface(rv_endo.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance), args.rv_tag)
    lv_epi_reference = lv_epi_intersection_reference.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)

    rv_epi_cut_mask = np.zeros(rv_epi_fixed.n_points, dtype=bool)
    rv_endo_cut_mask = np.zeros(rv_endo_fixed.n_points, dtype=bool)

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
        "RVepi_step5_stitched_vs_LVepi_fixed": direct_surface_intersection_metrics(
            "RVepi_step5_stitched",
            rv_epi_fixed,
            "LVepi_fixed",
            lv_epi_fixed,
            args.stage8_intersection_check_tolerance,
        ),
        "RVendo_step7_stitched_vs_LVepi_fixed": direct_surface_intersection_metrics(
            "RVendo_step7_stitched",
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
        "LV_openbase_annulus": tagged_surface(lv_annulus, args.lv_tag),
        "RV_openbase_annulus": tagged_surface(rv_annulus, args.rv_tag),
    }
    masks = {
        "RVepi_cut_boundary": rv_epi_cut_mask,
        "RVendo_cut_boundary": rv_endo_cut_mask,
    }
    info = {
        "description": "Step 8 no longer re-runs projection-based intersection removal. It preserves the Step 5/6 existing-LVepi-vertex seam stitches, keeps LV surfaces fixed, and only rebuilds basal open-base annuli from the current surface loops.",
        "LV_surfaces_moved": False,
        "RV_surfaces_moved_by_step8": False,
        "RV_surface_fix_method": "pass_through_after_step5_6_face_vertex_deletion_and_LVepi_existing_vertex_stitch",
        "LVepi_intersection_reference": surface_summary(lv_epi_reference),
        "RVepi_fix": {"method": "not_retrimmed_in_step8", "reason": "Step 5 already deleted intersecting RVepi faces/vertices and stitched to existing LVepi vertices."},
        "RVendo_fix": {"method": "not_retrimmed_in_step8", "reason": "Step 6/7 already deleted intersecting RVendo faces/vertices and rebuilt the LVepi stitch after distance movement."},
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









def sorted_edge_key(first: int, second: int) -> Tuple[int, int]:
    a, b = int(first), int(second)
    return (a, b) if a < b else (b, a)


def lvepi_vertex_edge_graph(n_points: int, triangles: np.ndarray) -> Tuple[List[set], set]:
    neighbors = [set() for _ in range(int(n_points))]
    edges = set()
    for tri in np.asarray(triangles, dtype=np.int64):
        for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
            a = int(tri[local_a])
            b = int(tri[local_b])
            if a == b:
                continue
            edge = sorted_edge_key(a, b)
            edges.add(edge)
            neighbors[a].add(b)
            neighbors[b].add(a)
    return neighbors, edges


def closed_loop_direct_edge_fraction(vertex_ids: np.ndarray, edge_set: set) -> float:
    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    if len(ids) < 2:
        return 0.0
    hits = 0
    total = 0
    for index in range(len(ids)):
        a = int(ids[index])
        b = int(ids[(index + 1) % len(ids)])
        if a == b:
            continue
        total += 1
        if sorted_edge_key(a, b) in edge_set:
            hits += 1
    return float(hits / max(total, 1))


def order_lvepi_seam_vertex_ids(
    name: str,
    lv_points: np.ndarray,
    vertex_ids: np.ndarray,
    edge_set: set,
) -> Tuple[np.ndarray, Dict[str, object]]:
    input_ids = ordered_unique_indices(np.asarray(vertex_ids, dtype=np.int64))
    if len(input_ids) < 3:
        raise RuntimeError(f"{name}: fewer than three unique LVepi seam vertices.")
    input_fraction = closed_loop_direct_edge_fraction(input_ids, edge_set)

    pts = np.asarray(lv_points, dtype=float)[input_ids]
    centered = pts - pts.mean(axis=0)
    order_method = "input_loop_order"
    angle_ids = input_ids.copy()
    angle_fraction = 0.0
    singular_values: List[float] = []
    try:
        _, sv, vh = np.linalg.svd(centered, full_matrices=False)
        singular_values = [float(v) for v in sv[:3]]
        if len(sv) >= 2 and float(sv[1]) > 1e-10:
            basis = vh[:2].T
            coords = centered @ basis
            angles = np.arctan2(coords[:, 1], coords[:, 0])
            angle_order = np.argsort(angles)
            angle_ids = input_ids[angle_order]
            angle_fraction = closed_loop_direct_edge_fraction(angle_ids, edge_set)
            if angle_fraction > input_fraction:
                order_method = "best_fit_plane_angle_order"
    except Exception:
        angle_fraction = 0.0

    ordered_ids = angle_ids if order_method == "best_fit_plane_angle_order" else input_ids
    return ordered_ids, {
        "method": order_method,
        "unique_vertices": int(len(input_ids)),
        "input_order_direct_edge_fraction": float(input_fraction),
        "angle_order_direct_edge_fraction": float(angle_fraction),
        "best_fit_plane_singular_values": singular_values,
    }


def shortest_path_on_allowed_lvepi_graph(
    start: int,
    goal: int,
    neighbors: Sequence[set],
    allowed_mask: Optional[np.ndarray] = None,
) -> List[int]:
    start = int(start)
    goal = int(goal)
    if start == goal:
        return [start]
    n = len(neighbors)
    visited = np.zeros(n, dtype=bool)
    previous = -np.ones(n, dtype=np.int64)
    queue: deque[int] = deque([start])
    visited[start] = True
    while queue:
        current = queue.popleft()
        for nb in neighbors[current]:
            nb = int(nb)
            if visited[nb]:
                continue
            if allowed_mask is not None and not bool(allowed_mask[nb]) and nb != goal:
                continue
            visited[nb] = True
            previous[nb] = current
            if nb == goal:
                queue.clear()
                break
            queue.append(nb)
    if not visited[goal]:
        return []
    path = [goal]
    current = goal
    while current != start:
        current = int(previous[current])
        if current < 0:
            return []
        path.append(current)
    path.reverse()
    return path


def map_seam_loops_to_lvepi_vertices(
    name: str,
    seam_loops: Sequence[pv.PolyData],
    lv_epi: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    lv_points = np.asarray(lv_epi.points, dtype=float)
    lv_tree = cKDTree(lv_points)
    mapped: List[np.ndarray] = []
    records: List[Dict[str, object]] = []
    exact_tolerance = max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 100.0, 1e-4)
    for loop_id, loop in enumerate(seam_loops):
        pts = clean_loop_points(loop_points(loop))
        if len(pts) < 3:
            records.append({
                "loop_id": int(loop_id),
                "input_points": int(len(pts)),
                "used": False,
                "reason": "fewer_than_3_points",
            })
            continue
        distances, ids = lv_tree.query(pts, k=1)
        ids = ordered_unique_indices(np.asarray(ids, dtype=np.int64))
        used = len(ids) >= 3
        if used:
            mapped.append(ids)
        records.append({
            "loop_id": int(loop_id),
            "input_points": int(len(pts)),
            "used": bool(used),
            "unique_LVepi_vertices": int(len(ids)),
            "exact_match_tolerance_mm": float(exact_tolerance),
            "exact_match_vertices": int(np.count_nonzero(distances <= exact_tolerance)),
            "nearest_distance_min": float(np.min(distances)) if len(distances) else None,
            "nearest_distance_mean": float(np.mean(distances)) if len(distances) else None,
            "nearest_distance_p95": float(np.percentile(distances, 95)) if len(distances) else None,
            "nearest_distance_max": float(np.max(distances)) if len(distances) else None,
        })
    if not mapped:
        raise RuntimeError(f"{name}: no usable LVepi seam vertices were mapped from the supplied boundary loops.")
    return mapped, {
        "method": "nearest_existing_LVepi_vertex_ids_from_RV_substitute_boundary_points",
        "name": name,
        "input_loop_count": int(len(seam_loops)),
        "used_loop_count": int(len(mapped)),
        "loops": records,
    }






def build_lvepi_barrier_edges_from_saved_line_loops(
    name: str,
    seam_loops: Sequence[pv.PolyData],
    lv_epi: pv.PolyData,
    neighbors: Sequence[set],
    edge_set: set,
    args: argparse.Namespace,
) -> Tuple[set, np.ndarray, Dict[str, object]]:
    lv_epi = lv_epi.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_points = np.asarray(lv_epi.points, dtype=float)
    mapped_loops, mapping_info = map_seam_loops_to_lvepi_vertices(name, seam_loops, lv_epi, args)
    barrier_edges = set()
    barrier_vertices: List[int] = []
    loop_records: List[Dict[str, object]] = []
    exact_direct_edges = 0
    graph_completed_edges = 0
    failed_graph_edges = 0
    same_vertex_edges = 0
    for loop_id, raw_ids in enumerate(mapped_loops):
        ids = loop_vertex_ids_without_duplicate_endpoint(raw_ids)
        if len(ids) < 3:
            loop_records.append({
                "loop_id": int(loop_id),
                "used": False,
                "reason": "fewer_than_three_unique_LVepi_vertices",
                "vertices": int(len(ids)),
            })
            continue
        edge_record = closed_loop_edge_count_record(ids, edge_set)
        path_records: List[Dict[str, object]] = []
        for index in range(len(ids)):
            lv_a = int(ids[index])
            lv_b = int(ids[(index + 1) % len(ids)])
            if lv_a == lv_b:
                same_vertex_edges += 1
                continue
            edge = sorted_edge_key(lv_a, lv_b)
            if edge in edge_set:
                barrier_edges.add(edge)
                barrier_vertices.extend([lv_a, lv_b])
                exact_direct_edges += 1
                path_records.append({
                    "segment_id": int(index),
                    "start": lv_a,
                    "goal": lv_b,
                    "method": "direct_saved_LVepi_loop_edge",
                    "path_vertices": 2,
                })
                continue
            path_ids = shortest_path_on_allowed_lvepi_graph(lv_a, lv_b, neighbors, None)
            if len(path_ids) < 2:
                failed_graph_edges += 1
                path_records.append({
                    "segment_id": int(index),
                    "start": lv_a,
                    "goal": lv_b,
                    "method": "failed_LVepi_graph_completion_for_missing_saved_loop_edge",
                    "path_vertices": int(len(path_ids)),
                    "failed": True,
                })
                continue
            for a, b in zip(path_ids[:-1], path_ids[1:]):
                barrier_edges.add(sorted_edge_key(a, b))
            barrier_vertices.extend(int(v) for v in path_ids)
            graph_completed_edges += 1
            path_records.append({
                "segment_id": int(index),
                "start": lv_a,
                "goal": lv_b,
                "method": "graph_completion_for_missing_saved_loop_edge",
                "path_vertices": int(len(path_ids)),
            })
        loop_records.append({
            "loop_id": int(loop_id),
            "used": True,
            "vertices": int(len(ids)),
            "edge_conformity": edge_record,
            "duplicate_vertices": int(loop_duplicate_vertex_count(ids)),
            "line_loop": loop_record(base.make_polyline(lv_points[ids])),
            "path_records_sample": path_records[:30],
        })
    if len(barrier_edges) < 3:
        raise RuntimeError(f"{name}: saved Step 5/6 LVepi line loop did not produce enough barrier edges.")
    return barrier_edges, np.unique(np.asarray(barrier_vertices, dtype=np.int64)), {
        "method": "direct_saved_step5_step6_LVepi_line_loop_edges",
        "name": name,
        "input_boundary_loop_source": "saved Step 5 RVepi line loop or saved Step 6 RVendo line loop",
        "mapping": mapping_info,
        "barrier_edge_count": int(len(barrier_edges)),
        "barrier_vertex_count": int(len(set(barrier_vertices))),
        "exact_direct_saved_loop_edges": int(exact_direct_edges),
        "graph_completed_missing_saved_loop_edges": int(graph_completed_edges),
        "failed_graph_completion_edges": int(failed_graph_edges),
        "same_LVepi_vertex_edges_skipped": int(same_vertex_edges),
        "loops": loop_records,
    }


def face_components_split_by_barrier_edges(triangles: np.ndarray, barrier_edges: set) -> List[np.ndarray]:
    edge_to_faces = triangle_edge_face_map(triangles)
    adjacency = [set() for _ in range(len(triangles))]
    for edge, owners in edge_to_faces.items():
        if edge in barrier_edges:
            continue
        for a in owners:
            for b in owners:
                if a != b:
                    adjacency[int(a)].add(int(b))
    visited = np.zeros(len(triangles), dtype=bool)
    components: List[np.ndarray] = []
    for start in range(len(triangles)):
        if visited[start]:
            continue
        queue: deque[int] = deque([int(start)])
        visited[start] = True
        ids: List[int] = []
        while queue:
            current = queue.popleft()
            ids.append(current)
            for nb in adjacency[current]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(int(nb))
        components.append(np.asarray(ids, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def component_barrier_edge_contacts(component: np.ndarray, edge_to_faces: Dict[Tuple[int, int], List[int]], barrier_edges: set) -> int:
    component_faces = set(int(v) for v in np.asarray(component, dtype=np.int64))
    contacts = 0
    for edge in barrier_edges:
        if any(int(face_id) in component_faces for face_id in edge_to_faces.get(edge, [])):
            contacts += 1
    return int(contacts)








def clean_loop_points(points: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    cleaned: List[np.ndarray] = []
    for point in np.asarray(points, dtype=float):
        if not cleaned or np.linalg.norm(point - cleaned[-1]) > tolerance:
            cleaned.append(point.copy())
    if len(cleaned) > 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tolerance:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)
















def edge_topology_report(mesh: pv.PolyData, args: argparse.Namespace) -> Dict[str, object]:
    surface = mesh.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    triangles = base.triangle_faces(surface)
    edge_map = triangle_edge_face_map(triangles)
    free_edges = [edge for edge, owners in edge_map.items() if len(owners) == 1]
    nonmanifold_edges = [edge for edge, owners in edge_map.items() if len(owners) > 2]
    loops = base.boundary_loops(surface)
    return {
        "points": int(surface.n_points),
        "triangles": int(surface.n_cells),
        "unique_edges": int(len(edge_map)),
        "free_edges": int(len(free_edges)),
        "nonmanifold_edges": int(len(nonmanifold_edges)),
        "boundary_loop_count": int(len(loops)),
        "boundary_loop_sizes": [int(len(clean_loop_points(loop_points(loop)))) for loop in loops[:20]],
        "free_edge_sample": [[int(a), int(b)] for a, b in free_edges[:20]],
        "nonmanifold_edge_sample": [[int(a), int(b), int(len(edge_map[(a, b)]))] for a, b in nonmanifold_edges[:20]],
    }


def loop_to_surface_point_match_record(
    name: str,
    loops: Sequence[pv.PolyData],
    surface: pv.PolyData,
    args: argparse.Namespace,
) -> Dict[str, object]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    target_points = np.asarray(surface.points, dtype=float)
    tolerance = max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 100.0, 1e-5)
    if len(target_points) == 0:
        return {"name": name, "passes": False, "reason": "target_surface_has_no_points", "tolerance_mm": float(tolerance)}
    target_tree = cKDTree(target_points)
    records: List[Dict[str, object]] = []
    all_distances: List[np.ndarray] = []
    for loop_id, loop in enumerate(loops):
        pts = clean_loop_points(loop_points(loop))
        if len(pts) == 0:
            records.append({"loop_id": int(loop_id), "points": 0, "passes": False, "reason": "empty_loop"})
            continue
        distances, _ = target_tree.query(pts, k=1)
        all_distances.append(np.asarray(distances, dtype=float))
        records.append({
            "loop_id": int(loop_id),
            "points": int(len(pts)),
            "tolerance_mm": float(tolerance),
            "matched_points": int(np.count_nonzero(distances <= tolerance)),
            "passes": bool(np.all(distances <= tolerance)),
            "distance_min": float(np.min(distances)),
            "distance_mean": float(np.mean(distances)),
            "distance_p95": float(np.percentile(distances, 95)),
            "distance_max": float(np.max(distances)),
        })
    if all_distances:
        merged = np.concatenate(all_distances)
        passes = bool(np.all(merged <= tolerance))
        return {
            "name": name,
            "method": "boundary_loop_points_to_nearest_surface_point_distance",
            "passes": passes,
            "tolerance_mm": float(tolerance),
            "total_loop_points": int(len(merged)),
            "matched_points": int(np.count_nonzero(merged <= tolerance)),
            "distance_min": float(np.min(merged)),
            "distance_mean": float(np.mean(merged)),
            "distance_p95": float(np.percentile(merged, 95)),
            "distance_max": float(np.max(merged)),
            "loops": records,
        }
    return {
        "name": name,
        "method": "boundary_loop_points_to_nearest_surface_point_distance",
        "passes": False,
        "reason": "no_nonempty_loops",
        "tolerance_mm": float(tolerance),
        "loops": records,
    }


def tagged_lvepi_subset(
    lv_points: np.ndarray,
    lv_triangles: np.ndarray,
    face_mask: np.ndarray,
    elem_tag: int,
    name: str,
) -> pv.PolyData:
    mask = np.asarray(face_mask, dtype=bool)
    if len(mask) != len(lv_triangles):
        raise ValueError(f"{name}: face mask length {len(mask)} does not match triangle count {len(lv_triangles)}.")
    if not np.any(mask):
        raise RuntimeError(f"{name}: selected zero LVepi faces.")
    return compact_tagged_triangle_surface(
        np.asarray(lv_points, dtype=float),
        np.asarray(lv_triangles, dtype=np.int64)[mask],
        np.full(int(np.count_nonzero(mask)), int(elem_tag), dtype=np.int32),
    )


def split_lvepi_open_into_septum_and_noseptum(
    lv_epi: pv.PolyData,
    rv_epi_lvepi_seam_loops: Sequence[pv.PolyData],
    rv_endo_lvepi_seam_loops: Sequence[pv.PolyData],
    args: argparse.Namespace,
) -> Tuple[Dict[str, pv.PolyData], Dict[str, object]]:
    if not rv_epi_lvepi_seam_loops or not rv_endo_lvepi_seam_loops:
        raise RuntimeError("Step 9 LVepi split requires the RVepi Step 5/6 boundary ring and the displaced RVendo Step 7 boundary ring.")

    lv_epi = tagged_surface(lv_epi.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance), args.lv_tag)
    lv_points = np.asarray(lv_epi.points, dtype=float)
    lv_triangles = base.triangle_faces(lv_epi)
    if len(lv_triangles) == 0:
        raise RuntimeError("Step 9 LVepi split received an LVepi surface with no triangles.")

    neighbors, edge_set = lvepi_vertex_edge_graph(lv_epi.n_points, lv_triangles)
    rv_epi_barrier_edges, rv_epi_barrier_vertices, rv_epi_barrier_info = build_lvepi_barrier_edges_from_saved_line_loops(
        "RVepi_step6_state_LVepi_boundary_ring",
        rv_epi_lvepi_seam_loops,
        lv_epi,
        neighbors,
        edge_set,
        args,
    )
    rv_endo_barrier_edges, rv_endo_barrier_vertices, rv_endo_barrier_info = build_lvepi_barrier_edges_from_saved_line_loops(
        "RVendo_step7_displaced_LVepi_boundary_ring",
        rv_endo_lvepi_seam_loops,
        lv_epi,
        neighbors,
        edge_set,
        args,
    )

    all_barriers = set(rv_epi_barrier_edges) | set(rv_endo_barrier_edges)
    components = face_components_split_by_barrier_edges(lv_triangles, all_barriers)
    if not components:
        raise RuntimeError("No LVepi face components found after splitting by RVepi/RVendo barrier loops.")

    edge_to_faces = triangle_edge_face_map(lv_triangles)
    areas = face_areas(lv_points, lv_triangles)
    records: List[Dict[str, object]] = []
    interface_component_ids: List[int] = []
    septum_candidates: List[Tuple[int, float, int, np.ndarray]] = []
    for component_id, component in enumerate(components):
        epi_contacts = component_barrier_edge_contacts(component, edge_to_faces, rv_epi_barrier_edges)
        endo_contacts = component_barrier_edge_contacts(component, edge_to_faces, rv_endo_barrier_edges)
        area = float(areas[component].sum()) if len(component) else 0.0
        touches_epi = epi_contacts > 0
        touches_endo = endo_contacts > 0
        touches_both = touches_epi and touches_endo
        classification = "LVepi_noseptum_candidate"
        if touches_both:
            classification = "interface_patch_between_RVepi_and_RVendo_rings"
            interface_component_ids.append(int(component_id))
        elif touches_endo:
            classification = "septum_candidate_inside_RVendo_ring"
            septum_candidates.append((int(endo_contacts), float(area), int(component_id), component))
        elif touches_epi:
            classification = "LVepi_noseptum_candidate_outside_RVepi_ring"
        records.append({
            "component_id": int(component_id),
            "faces": int(len(component)),
            "area": area,
            "RVepi_barrier_edge_contacts": int(epi_contacts),
            "RVendo_barrier_edge_contacts": int(endo_contacts),
            "touches_RVepi_barrier": bool(touches_epi),
            "touches_RVendo_barrier": bool(touches_endo),
            "touches_both_barriers": bool(touches_both),
            "classification_before_septum_selection": classification,
        })

    if not interface_component_ids:
        raise RuntimeError("No LVepi component was found between the RVepi and displaced RVendo boundary rings.")
    if not septum_candidates:
        raise RuntimeError("No LVepi component was found inside/wrapped by the displaced RVendo boundary ring.")

    septum_candidates.sort(key=lambda item: (-int(item[0]), -float(item[1])))
    _, _, septum_component_id, septum_component = septum_candidates[0]

    interface_mask = np.zeros(len(lv_triangles), dtype=bool)
    for component_id in interface_component_ids:
        interface_mask[components[int(component_id)]] = True
    septum_mask = np.zeros(len(lv_triangles), dtype=bool)
    septum_mask[septum_component] = True
    noseptum_mask = ~(interface_mask | septum_mask)

    if not np.any(noseptum_mask):
        raise RuntimeError("LVepi_noseptum selection is empty after removing interface patch and septum.")

    for record in records:
        if int(record["component_id"]) in interface_component_ids:
            record["final_classification"] = "removed_interface_patch"
        elif int(record["component_id"]) == int(septum_component_id):
            record["final_classification"] = "septum"
        else:
            record["final_classification"] = "LVepi_noseptum"

    surfaces = {
        "septum": tagged_lvepi_subset(lv_points, lv_triangles, septum_mask, args.lv_tag, "septum"),
        "LVepi_noseptum": tagged_lvepi_subset(lv_points, lv_triangles, noseptum_mask, args.lv_tag, "LVepi_noseptum"),
        "removed_LVepi_interface_patch": tagged_lvepi_subset(lv_points, lv_triangles, interface_mask, args.lv_tag, "removed_LVepi_interface_patch"),
    }
    info = {
        "method": "split_open_LVepi_by_RVepi_step5_boundary_and_displaced_RVendo_step7_boundary_before_surface_merge",
        "LVepi_input": surface_summary(lv_epi),
        "RVepi_boundary_on_LVepi": rv_epi_barrier_info,
        "RVendo_boundary_on_LVepi": rv_endo_barrier_info,
        "interface_component_ids_removed": [int(v) for v in interface_component_ids],
        "selected_septum_component_id": int(septum_component_id),
        "noseptum_face_count": int(np.count_nonzero(noseptum_mask)),
        "septum_face_count": int(np.count_nonzero(septum_mask)),
        "removed_interface_patch_face_count": int(np.count_nonzero(interface_mask)),
        "component_records": records[:80],
        "output_surfaces": {name: surface_summary(mesh) for name, mesh in surfaces.items()},
    }
    return surfaces, info


def build_partitioned_step9_surface_set(
    lv_epi: pv.PolyData,
    lv_endo: pv.PolyData,
    rv_epi: pv.PolyData,
    rv_endo: pv.PolyData,
    lv_openbase_annulus: pv.PolyData,
    rv_openbase_annulus: pv.PolyData,
    rv_epi_lvepi_seam_loops: Sequence[pv.PolyData],
    rv_endo_lvepi_seam_loops: Sequence[pv.PolyData],
    args: argparse.Namespace,
) -> Tuple[Dict[str, pv.PolyData], Dict[str, object]]:
    split_surfaces, split_info = split_lvepi_open_into_septum_and_noseptum(
        lv_epi,
        rv_epi_lvepi_seam_loops,
        rv_endo_lvepi_seam_loops,
        args,
    )
    septum = tagged_surface(split_surfaces["septum"], args.lv_tag)
    lvepi_noseptum = tagged_surface(split_surfaces["LVepi_noseptum"], args.lv_tag)
    removed_interface_patch = tagged_surface(split_surfaces["removed_LVepi_interface_patch"], args.lv_tag)
    rv_epi_tagged = tagged_surface(rv_epi, args.rv_tag)
    rv_endo_tagged = tagged_surface(rv_endo, args.rv_tag)
    lv_endo_tagged = tagged_surface(lv_endo, args.lv_tag)
    lv_annulus_tagged = tagged_surface(lv_openbase_annulus, args.lv_tag)
    rv_annulus_tagged = tagged_surface(rv_openbase_annulus, args.rv_tag)

    lvrv_epi_surface = merge_surfaces([lvepi_noseptum, rv_epi_tagged], args.combined_surface_merge_tolerance)
    rv_endo_surface = merge_surfaces([rv_endo_tagged, septum], args.combined_surface_merge_tolerance)
    lv_endo_surface = lv_endo_tagged
    whole_endo_surface = merge_surfaces(
        [rv_endo_surface, lv_endo_surface, lv_annulus_tagged, rv_annulus_tagged],
        args.combined_surface_merge_tolerance,
    )
    whole_epi_endo_surface = merge_surfaces(
        [lvrv_epi_surface, rv_endo_surface, lv_endo_surface, lv_annulus_tagged, rv_annulus_tagged],
        args.combined_surface_merge_tolerance,
    )

    surfaces = {
        "septum": septum,
        "LVepi_noseptum": lvepi_noseptum,
        "removed_LVepi_interface_patch": removed_interface_patch,
        "LVRV_epi_surface": lvrv_epi_surface,
        "RV_endo_surface": rv_endo_surface,
        "LV_endo_surface": lv_endo_surface,
        "whole_endo_surface": whole_endo_surface,
        "whole_epi_endo_surface": whole_epi_endo_surface,
    }
    boundary_validation = {
        "RVepi_step5_ring_on_LVepi_noseptum": loop_to_surface_point_match_record("RVepi_step5_ring_on_LVepi_noseptum", rv_epi_lvepi_seam_loops, lvepi_noseptum, args),
        "RVepi_step5_ring_on_RVepi": loop_to_surface_point_match_record("RVepi_step5_ring_on_RVepi", rv_epi_lvepi_seam_loops, rv_epi_tagged, args),
        "RVendo_step7_ring_on_septum": loop_to_surface_point_match_record("RVendo_step7_ring_on_septum", rv_endo_lvepi_seam_loops, septum, args),
        "RVendo_step7_ring_on_RVendo": loop_to_surface_point_match_record("RVendo_step7_ring_on_RVendo", rv_endo_lvepi_seam_loops, rv_endo_tagged, args),
    }
    topology = {name: edge_topology_report(mesh, args) for name, mesh in surfaces.items()}
    info = {
        "method": "partition_LVepi_before_merging_then_build_LVRV_epi_RV_endo_LV_endo_and_final_whole_surface",
        "split_LVepi": split_info,
        "boundary_matching": boundary_validation,
        "topology": topology,
        "elemTag_policy": {
            "septum": int(args.lv_tag),
            "LVepi_noseptum": int(args.lv_tag),
            "LVendo": int(args.lv_tag),
            "LV_openbase_annulus": int(args.lv_tag),
            "RVepi": int(args.rv_tag),
            "RVendo": int(args.rv_tag),
            "RV_openbase_annulus": int(args.rv_tag),
            "RV_endo_surface_contains_mixed_tags": True,
            "septum_tag_is_preserved_when_merged_into_RV_endo_surface": int(args.lv_tag),
        },
        "surface_summaries": {name: surface_summary(mesh) for name, mesh in surfaces.items()},
    }
    return surfaces, info


def save_merge_surface_stage_outputs(
    pipeline_dir: Path,
    subject_id: str,
    frame: str,
    step9_surfaces: Dict[str, pv.PolyData],
    step9_build_info: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}

    stage_items = [
        ("stage09_septum", step9_surfaces["septum"]),
        ("stage09_LVepi_noseptum", step9_surfaces["LVepi_noseptum"]),
        ("stage09_removed_LVepi_interface_patch", step9_surfaces["removed_LVepi_interface_patch"]),
        ("stage09_LVRV_epi_surface", step9_surfaces["LVRV_epi_surface"]),
        ("stage09_RV_endo_surface", step9_surfaces["RV_endo_surface"]),
        ("stage09_LV_endo_surface", step9_surfaces["LV_endo_surface"]),
        ("stage09_whole_endo_surface", step9_surfaces["whole_endo_surface"]),
        ("stage09_whole_epi_endo_surface", step9_surfaces["whole_epi_endo_surface"]),
    ]
    for name, mesh in stage_items:
        surface_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = save_triangle_surface_vtk(surface_path, mesh)
        paths[name] = str(surface_path.resolve())

    # Backward-compatible aliases are overwritten with the new topology-first surfaces
    # so stale files from previous Step 9 implementations do not remain misleading.
    compatibility_items = [
        ("stage09_whole_epi_surface", step9_surfaces["LVRV_epi_surface"]),
        ("stage09_whole_epi_endo_without_LVepi_RV_interface_patch", step9_surfaces["whole_epi_endo_surface"]),
        ("stage09_whole_epi_endo_without_LVepi_RV_interface_patch_filled_holes", step9_surfaces["whole_epi_endo_surface"]),
        ("stage09_removed_LVepi_patch_between_RVendo_RVepi_boundaries", step9_surfaces["removed_LVepi_interface_patch"]),
    ]
    for name, mesh in compatibility_items:
        surface_path = pipeline_dir / f"{subject_id}_{frame}_{name}.vtk"
        records[name] = save_triangle_surface_vtk(surface_path, mesh)
        paths[name] = str(surface_path.resolve())

    summary = {
        "description": "Step 9 topology-first outputs: split open LVepi by the RVepi Step 5/6 boundary ring and displaced RVendo Step 7 boundary ring, save septum and LVepi_noseptum, then merge LVRV_epi_surface, RV_endo_surface, LV_endo_surface, and LV/RV open-base annuli into the final whole_epi_endo_surface. Septum keeps elemTag=1 even inside RV_endo_surface.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(pipeline_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "step9_build": step9_build_info,
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
    step11_pipeline_dir: Path,
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
    step11_pipeline_dir.mkdir(parents=True, exist_ok=True)
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

    final_tet_path = step11_pipeline_dir / f"{subject_id}_{frame}_stage11_final_elemTag_volmesh.vtk"
    write_custom_vtk(final_tet_path, np.asarray(final_points, dtype=float), np.asarray(final_cells, dtype=np.int64), np.asarray(final_elem_tags, dtype=np.int32))
    paths["stage11_final_elemTag_volmesh"] = str(final_tet_path.resolve())

    filled_holes_tagged_tet_path = step11_pipeline_dir / f"{subject_id}_{frame}_stage10_filled_holes_elemTag_volmesh.vtk"
    write_custom_vtk(
        filled_holes_tagged_tet_path,
        np.asarray(final_points, dtype=float),
        np.asarray(final_cells, dtype=np.int64),
        np.asarray(final_elem_tags, dtype=np.int32),
    )
    paths["stage10_filled_holes_elemTag_volmesh"] = str(filled_holes_tagged_tet_path.resolve())

    legend_path = step11_pipeline_dir / f"{subject_id}_{frame}_stage11_elemTag_legend.csv"
    write_elem_tag_legend(legend_path, args.lv_tag, args.rv_tag)
    paths["stage11_elemTag_legend"] = str(legend_path.resolve())

    raw_quality = tet_quality_report("raw_tetgen_before_step11_tagging", np.asarray(raw_points, dtype=float), np.asarray(raw_cells, dtype=np.int64), args)
    raw_label_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, raw_tags)).items())}
    final_tag_counts = {str(k): int(v) for k, v in sorted(Counter(map(int, final_elem_tags)).items())}
    summary = {
        "description": "Step 10/11 outputs: TetGen PLC surface, TetGen-used surface, raw TetGen volume, and final custom LV/RV elemTag volume mesh.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(step11_pipeline_dir.resolve()),
        "step10_output_dir": str(pipeline_dir.resolve()),
        "step11_output_dir": str(step11_pipeline_dir.resolve()),
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
            "operation": "classify_tets_with_original_closed_LVepi_centroid_or_vertex_majority_rule_then_cleanup_and_write_custom_VTK",
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
    summary_path = step11_pipeline_dir / f"{subject_id}_{frame}_stage11_final_elemTag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["stage11_summary"] = str(summary_path.resolve())
    return paths, summary


def save_pipeline_summary(stage_dir: Path, subject_id: str, frame: str, name: str, summary: Dict[str, object]) -> str:
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"{subject_id}_{frame}_{name}.json"
    path.write_text(json.dumps(summary, indent=2))
    return str(path.resolve())


def save_step1_loaded_surfaces(
    stage_dir: Path,
    subject_id: str,
    frame: str,
    surfaces: Dict[str, pv.PolyData],
    input_paths: Dict[str, Path],
) -> Tuple[Dict[str, str], Dict[str, object]]:
    items = [
        ("LVendo", "lv_endo", int(1)),
        ("LVepi", "lv_epi", int(1)),
        ("RV", "rv_endo", int(0)),
        ("RVepi", "rv_epi", int(0)),
    ]
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}
    for output_name, surface_name, tag in items:
        path = stage_dir / f"{subject_id}_{frame}_{output_name}.vtk"
        records[output_name] = save_triangle_surface_vtk(path, surfaces[surface_name], elem_tag=tag)
        records[output_name]["input_path"] = str(input_paths[surface_name].resolve())
        paths[output_name] = str(path.resolve())
    summary = {
        "description": "Original four input surfaces loaded for this subject and frame.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(stage_dir.resolve()),
        "paths": paths,
        "surfaces": records,
    }
    paths["summary"] = save_pipeline_summary(stage_dir, subject_id, frame, "step01_load_surf_summary", summary)
    return paths, summary


def save_step2_base_detection(
    stage_dir: Path,
    subject_id: str,
    frame: str,
    subject_base_detection: Dict[str, object],
    lv_profile,
    rv_profile,
    lv_base_side: str,
    rv_base_side: str,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    paths = {
        "LV_z_area_profile": str((stage_dir / f"{subject_id}_{frame}_LV_z_area_profile.csv").resolve()),
        "RV_z_area_profile": str((stage_dir / f"{subject_id}_{frame}_RV_z_area_profile.csv").resolve()),
    }
    base.write_area_profile(Path(paths["LV_z_area_profile"]), lv_profile)
    base.write_area_profile(Path(paths["RV_z_area_profile"]), rv_profile)
    summary = {
        "description": "Subject-level base-side detection and LV/RV Z-area profiles.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(stage_dir.resolve()),
        "paths": paths,
        "base_side": {"subject": subject_base_detection, "LV": lv_base_side, "RV": rv_base_side},
        "profiles": {"LV": base.profile_to_dict(lv_profile), "RV": base.profile_to_dict(rv_profile)},
    }
    paths["summary"] = save_pipeline_summary(stage_dir, subject_id, frame, "step02_base_detection_summary", summary)
    return paths, summary


def save_openbase_surface_stage(
    stage_dir: Path,
    subject_id: str,
    frame: str,
    chamber: str,
    build,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    chamber = chamber.upper()
    tag = 1 if chamber == "LV" else 0
    surfaces = {
        f"{chamber}_openbase_epi": build.epi_open,
        f"{chamber}_openbase_endo": build.endo_open,
        f"{chamber}_openbase_annulus": build.annulus,
    }
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}
    for output_key, mesh in surfaces.items():
        path = stage_dir / f"{subject_id}_{frame}_{output_key}.vtk"
        records[output_key] = save_triangle_surface_vtk(path, mesh, elem_tag=tag)
        paths[output_key] = str(path.resolve())
    summary = {
        "description": f"Step {3 if chamber == 'LV' else 4} {chamber} open-base surfaces saved immediately after shell construction.",
        "subject": subject_id,
        "frame": frame,
        "chamber": chamber,
        "output_dir": str(stage_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "open_base": openbase.ventricle_summary(build),
    }
    paths["summary"] = save_pipeline_summary(
        stage_dir,
        subject_id,
        frame,
        f"step{3 if chamber == 'LV' else 4:02d}_{chamber}_openbase_summary",
        summary,
    )
    return paths, summary


def save_step5_moved_rvendo(
    stage_dir: Path,
    subject_id: str,
    frame: str,
    rv_endo_moved: pv.PolyData,
    rv_epi_fixed: pv.PolyData,
    moved_mask: np.ndarray,
    opening_mask: np.ndarray,
    transition_band_mask: np.ndarray,
    protected_mask: np.ndarray,
    distance_info: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    moved = rv_endo_moved.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    rv_epi = rv_epi_fixed.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    distances = cKDTree(np.asarray(rv_epi.points, dtype=float)).query(np.asarray(moved.points, dtype=float), k=1)[0]

    def fit_mask(mask: np.ndarray) -> np.ndarray:
        values = np.asarray(mask, dtype=bool).reshape(-1)
        if len(values) == moved.n_points:
            return values
        return np.zeros(moved.n_points, dtype=bool)

    point_data = {
        "movedByStep5": fit_mask(moved_mask).astype(np.int32),
        "adaptiveOpeningBoundary": fit_mask(opening_mask).astype(np.int32),
        "openingTransitionBand": fit_mask(transition_band_mask).astype(np.int32),
        "protectedLVepiIntersection": fit_mask(protected_mask).astype(np.int32),
        "distanceToRVepi": np.asarray(distances, dtype=np.float32),
        "belowRequested3mm": (distances < float(args.rv_min_distance_to_rvepi)).astype(np.int32),
    }
    path = stage_dir / f"{subject_id}_{frame}_RV_openbase_endo_moved.vtk"
    record = save_triangle_surface_vtk(path, moved, point_data=point_data, elem_tag=int(args.rv_tag))
    paths = {"RV_openbase_endo_moved": str(path.resolve())}
    protected_output = fit_mask(protected_mask)
    below_requested = distances < float(args.rv_min_distance_to_rvepi)
    record["distance_summary"] = {
        "minimum_distance_mm": float(np.min(distances)) if len(distances) else None,
        "p05_distance_mm": float(np.percentile(distances, 5)) if len(distances) else None,
        "vertices_below_3mm": int(np.count_nonzero(below_requested)),
        "protected_vertices_below_3mm": int(np.count_nonzero(below_requested & protected_output)),
        "movable_vertices_below_3mm": int(np.count_nonzero(below_requested & (~protected_output))),
    }
    summary = {
        "description": "RVendo moved away from fixed RVepi with an adaptive, smoothed open-base ring.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(stage_dir.resolve()),
        "paths": paths,
        "surface": record,
        "distance_fix": distance_info,
        "rv_epi_fixed": True,
        "protected_intersection_vertices_output": int(np.count_nonzero(protected_output)),
        "minimum_requested_distance_mm": float(args.rv_min_distance_to_rvepi),
        "opening_ring_output": "Opening-ring points are included in the moved RVendo surface and marked by adaptiveOpeningBoundary.",
    }
    paths["summary"] = save_pipeline_summary(stage_dir, subject_id, frame, "step05_move_RVendo_summary", summary)
    return paths, summary


def save_step5_moved_rvepi(
    stage_dir: Path,
    subject_id: str,
    frame: str,
    rv_epi_moved: pv.PolyData,
    rv_endo_fixed: pv.PolyData,
    moved_mask: np.ndarray,
    opening_mask: np.ndarray,
    transition_band_mask: np.ndarray,
    distance_info: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    moved = rv_epi_moved.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    fixed = rv_endo_fixed.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    distances = cKDTree(np.asarray(fixed.points, dtype=float)).query(np.asarray(moved.points, dtype=float), k=1)[0]

    def fit_mask(mask: np.ndarray) -> np.ndarray:
        values = np.asarray(mask, dtype=bool).reshape(-1)
        if len(values) == moved.n_points:
            return values
        return np.zeros(moved.n_points, dtype=bool)

    point_data = {
        "movedByStep5": fit_mask(moved_mask).astype(np.int32),
        "adaptiveOpeningBoundary": fit_mask(opening_mask).astype(np.int32),
        "openingTransitionBand": fit_mask(transition_band_mask).astype(np.int32),
        "distanceToRVendo": np.asarray(distances, dtype=np.float32),
        "belowRequested3mm": (distances < float(args.rv_min_distance_to_rvepi)).astype(np.int32),
    }
    path = stage_dir / f"{subject_id}_{frame}_RV_openbase_epi_moved.vtk"
    record = save_triangle_surface_vtk(path, moved, point_data=point_data, elem_tag=int(args.rv_tag))
    below_requested = distances < float(args.rv_min_distance_to_rvepi)
    record["distance_summary"] = {
        "minimum_distance_mm": float(np.min(distances)) if len(distances) else None,
        "p05_distance_mm": float(np.percentile(distances, 5)) if len(distances) else None,
        "vertices_below_3mm": int(np.count_nonzero(below_requested)),
    }
    paths = {"RV_openbase_epi_moved": str(path.resolve())}
    summary = {
        "description": "RVepi moved away from fixed RVendo with an adaptive, smoothed open-base ring.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(stage_dir.resolve()),
        "paths": paths,
        "surface": record,
        "distance_fix": distance_info,
        "rv_endo_fixed": True,
        "lv_epi_intersection_handling": "not applied",
        "minimum_requested_distance_mm": float(args.rv_min_distance_to_rvepi),
    }
    paths["summary"] = save_pipeline_summary(stage_dir, subject_id, frame, "step05_move_RVepi_summary", summary)
    return paths, summary


def save_step6_reconstructed_annuli(
    stage_dir: Path,
    subject_id: str,
    frame: str,
    lv_epi: pv.PolyData,
    lv_endo: pv.PolyData,
    lv_annulus: pv.PolyData,
    rv_epi: pv.PolyData,
    rv_endo: pv.PolyData,
    rv_annulus: pv.PolyData,
    lv_info: Dict[str, object],
    rv_info: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    items = [
        ("LV_openbase_epi", lv_epi, int(args.lv_tag)),
        ("LV_openbase_endo", lv_endo, int(args.lv_tag)),
        ("RV_openbase_epi_moved", rv_epi, int(args.rv_tag)),
        ("RV_openbase_endo", rv_endo, int(args.rv_tag)),
        ("LV_openbase_annulus_recon", lv_annulus, int(args.lv_tag)),
        ("RV_openbase_annulus_recon", rv_annulus, int(args.rv_tag)),
    ]
    paths: Dict[str, str] = {}
    records: Dict[str, object] = {}
    for output_key, mesh, tag in items:
        path = stage_dir / f"{subject_id}_{frame}_{output_key}.vtk"
        records[output_key] = save_triangle_surface_vtk(path, mesh, elem_tag=tag)
        paths[output_key] = str(path.resolve())
    summary = {
        "description": "LV/RV open-base annuli reconstructed by directly connecting the current epi and endo boundary loops.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(stage_dir.resolve()),
        "paths": paths,
        "surfaces": records,
        "LV_annulus_reconstruction": lv_info,
        "RV_annulus_reconstruction": rv_info,
        "direct_boundary_connection": True,
        "tetgen_or_downstream_processing": False,
    }
    paths["summary"] = save_pipeline_summary(stage_dir, subject_id, frame, "step06_reconstruct_annulus_summary", summary)
    return paths, summary


def load_step6_point_surface(
    stage_dir: Path,
    subject_id: str,
    frame: str,
    surface_name: str,
    fallback_names: Sequence[str] = (),
) -> Tuple[pv.PolyData, Path]:
    candidate_names = [surface_name, *fallback_names]
    candidates = [stage_dir / f"{subject_id}_{frame}_{name}.vtk" for name in candidate_names]
    for path in candidates:
        if path.exists():
            return base.load_surface(path), path
    raise FileNotFoundError(
        f"Missing Step 6 surface {surface_name!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def save_step7_whole_points_surface(
    stage_dir: Path,
    step6_dir: Path,
    subject_id: str,
    frame: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Remove RV points near LVepi and save all Step 7 outputs as point clouds."""
    stage_dir.mkdir(parents=True, exist_ok=True)

    input_specs = [
        ("LV_openbase_epi", "LV_openbase_epi", (), 1),
        ("LV_openbase_endo", "LV_openbase_endo", (), 1),
        ("LV_openbase_annulus_recon", "LV_openbase_annulus_recon", (), 1),
        ("RV_openbase_epi", "RV_openbase_epi", ("RV_openbase_epi_moved",), 0),
        ("RV_openbase_endo", "RV_openbase_endo", (), 0),
        ("RV_openbase_annulus_recon", "RV_openbase_annulus_recon", (), 0),
    ]
    inputs: Dict[str, pv.PolyData] = {}
    input_paths: Dict[str, Path] = {}
    for logical_name, primary_name, fallback_names, _ in input_specs:
        mesh, path = load_step6_point_surface(
            step6_dir,
            subject_id,
            frame,
            primary_name,
            fallback_names,
        )
        inputs[logical_name] = mesh
        input_paths[logical_name] = path

    lv_epi_reference = inputs["LV_openbase_epi"].extract_surface().triangulate().clean(
        tolerance=float(args.surface_clean_tolerance)
    )
    point_tolerance = max(
        float(args.step7_lvepi_point_tolerance),
        float(args.surface_lvepi_clearance),
        float(args.surface_clean_tolerance) * 100.0,
        1e-5,
    )

    def clean_rv_points(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(inputs[name].points, dtype=float).copy()
        _, distances = closest_points_on_surface(lv_epi_reference, points)
        intersecting = distances <= point_tolerance
        kept_ids = np.flatnonzero(~intersecting).astype(np.int64)
        return points[kept_ids], kept_ids, distances[kept_ids], intersecting

    rv_epi_points, rv_epi_source_ids, rv_epi_distances, rv_epi_removed = clean_rv_points("RV_openbase_epi")
    rv_endo_points, rv_endo_source_ids, rv_endo_distances, rv_endo_removed = clean_rv_points("RV_openbase_endo")
    rv_annulus_points, rv_annulus_source_ids, rv_annulus_distances, rv_annulus_removed = clean_rv_points(
        "RV_openbase_annulus_recon"
    )

    rv_epi_path = stage_dir / f"{subject_id}_{frame}_RVepi_clean_surf.vtk"
    rv_endo_path = stage_dir / f"{subject_id}_{frame}_RVendo_clean_surf.vtk"
    write_custom_point_cloud_vtk(
        rv_epi_path,
        rv_epi_points,
        np.full(len(rv_epi_points), int(args.rv_tag), dtype=np.int32),
        point_data={
            "sourcePointId": rv_epi_source_ids,
            "distanceToLVepi": rv_epi_distances.astype(np.float32),
        },
    )
    write_custom_point_cloud_vtk(
        rv_endo_path,
        rv_endo_points,
        np.full(len(rv_endo_points), int(args.rv_tag), dtype=np.int32),
        point_data={
            "sourcePointId": rv_endo_source_ids,
            "distanceToLVepi": rv_endo_distances.astype(np.float32),
        },
    )

    point_blocks: List[np.ndarray] = []
    tag_blocks: List[np.ndarray] = []
    source_surface_blocks: List[np.ndarray] = []
    source_point_blocks: List[np.ndarray] = []
    merged_source_names: List[str] = []
    merged_point_counts: Dict[str, int] = {}
    merged_removed_counts: Dict[str, int] = {}
    merged_distance_minima: Dict[str, Optional[float]] = {}
    cleaned_rv = {
        "RV_openbase_epi": (rv_epi_points, rv_epi_source_ids, rv_epi_removed, rv_epi_distances),
        "RV_openbase_endo": (rv_endo_points, rv_endo_source_ids, rv_endo_removed, rv_endo_distances),
        "RV_openbase_annulus_recon": (
            rv_annulus_points,
            rv_annulus_source_ids,
            rv_annulus_removed,
            rv_annulus_distances,
        ),
    }
    for source_index, (logical_name, _, _, tag) in enumerate(input_specs):
        if logical_name in cleaned_rv:
            points, source_ids, removed_mask, distances = cleaned_rv[logical_name]
            merged_removed_counts[logical_name] = int(np.count_nonzero(removed_mask))
            merged_distance_minima[logical_name] = float(np.min(distances)) if len(distances) else None
        else:
            points = np.asarray(inputs[logical_name].points, dtype=float).copy()
            source_ids = np.arange(len(points), dtype=np.int64)
            merged_removed_counts[logical_name] = 0
            merged_distance_minima[logical_name] = None
        point_blocks.append(points)
        tag_blocks.append(np.full(len(points), int(tag), dtype=np.int32))
        source_surface_blocks.append(np.full(len(points), int(source_index), dtype=np.int32))
        source_point_blocks.append(source_ids)
        merged_source_names.append(logical_name)
        merged_point_counts[logical_name] = int(len(points))

    merged_points = np.vstack(point_blocks) if point_blocks else np.empty((0, 3), dtype=float)
    merged_tags = np.concatenate(tag_blocks) if tag_blocks else np.empty((0,), dtype=np.int32)
    merged_source_surface = (
        np.concatenate(source_surface_blocks) if source_surface_blocks else np.empty((0,), dtype=np.int32)
    )
    merged_source_point = (
        np.concatenate(source_point_blocks) if source_point_blocks else np.empty((0,), dtype=np.int64)
    )
    merged_rv_mask = np.isin(merged_source_surface, np.asarray([3, 4, 5], dtype=np.int32))
    merged_rv_distances = (
        closest_points_on_surface(lv_epi_reference, merged_points[merged_rv_mask])[1]
        if np.any(merged_rv_mask)
        else np.empty((0,), dtype=float)
    )
    merged_rv_intersections = merged_rv_distances <= point_tolerance
    if np.any(merged_rv_intersections):
        raise RuntimeError(
            "Step 7 merged point cloud still contains RV points intersecting LVepi: "
            f"{int(np.count_nonzero(merged_rv_intersections))} points."
        )
    merged_path = stage_dir / f"{subject_id}_{frame}_LVRV_surf.vtk"
    write_custom_point_cloud_vtk(
        merged_path,
        merged_points,
        merged_tags,
        point_data={
            "sourceSurface": merged_source_surface,
            "sourcePointId": merged_source_point,
        },
    )

    paths = {
        "RVendo_clean_surf": str(rv_endo_path.resolve()),
        "RVepi_clean_surf": str(rv_epi_path.resolve()),
        "LVRV_surf": str(merged_path.resolve()),
    }
    records = {
        "RVendo_clean_surf": {
            "points": int(len(rv_endo_points)),
            "removed_intersection_points": int(np.count_nonzero(rv_endo_removed)),
            "point_cloud_only": True,
            "cell_type": 1,
        },
        "RVepi_clean_surf": {
            "points": int(len(rv_epi_points)),
            "removed_intersection_points": int(np.count_nonzero(rv_epi_removed)),
            "point_cloud_only": True,
            "cell_type": 1,
        },
        "LVRV_surf": {
            "points": int(len(merged_points)),
            "point_cloud_only": True,
            "cell_type": 1,
            "elemTag_counts": {
                "1": int(np.count_nonzero(merged_tags == int(args.lv_tag))),
                "0": int(np.count_nonzero(merged_tags == int(args.rv_tag))),
            },
            "source_surfaces": merged_source_names,
        },
    }
    summary = {
        "description": "Step 7 point-only LV/RV surface cloud after removing RV points intersecting the Step 6 LVepi surface.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(stage_dir.resolve()),
        "input_step6_dir": str(step6_dir.resolve()),
        "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
        "paths": paths,
        "records": records,
        "reference_surface": "LV_openbase_epi",
        "intersection_rule": "remove_RV_points_with_point_to_LVepi_surface_distance_at_or_below_tolerance",
        "point_intersection_tolerance_mm": float(point_tolerance),
        "removed_intersection_points": merged_removed_counts,
        "minimum_kept_distance_to_LVepi_mm": merged_distance_minima,
        "merged_RV_point_count_checked_against_LVepi": int(np.count_nonzero(merged_rv_mask)),
        "merged_RV_points_intersecting_LVepi_after_cleaning": int(np.count_nonzero(merged_rv_intersections)),
        "merged_point_counts": merged_point_counts,
        "triangle_faces_written": False,
        "vtk_cell_type": "VTK_VERTEX",
    }
    paths["summary"] = save_pipeline_summary(stage_dir, subject_id, frame, "step07_whole_points_surf_summary", summary)
    return paths, summary


def save_step6_merged_point_cloud(
    stage_dir: Path,
    subject_id: str,
    frame: str,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, object]]:
    """Merge all six Step 6 surfaces as a raw point-only VTK cloud."""
    input_specs = [
        ("LV_openbase_epi", "LV_openbase_epi", (), int(args.lv_tag)),
        ("LV_openbase_endo", "LV_openbase_endo", (), int(args.lv_tag)),
        ("LV_openbase_annulus_recon", "LV_openbase_annulus_recon", (), int(args.lv_tag)),
        ("RV_openbase_epi", "RV_openbase_epi", ("RV_openbase_epi_moved",), int(args.rv_tag)),
        ("RV_openbase_endo", "RV_openbase_endo", (), int(args.rv_tag)),
        ("RV_openbase_annulus_recon", "RV_openbase_annulus_recon", (), int(args.rv_tag)),
    ]
    point_blocks: List[np.ndarray] = []
    tag_blocks: List[np.ndarray] = []
    source_surface_blocks: List[np.ndarray] = []
    source_point_blocks: List[np.ndarray] = []
    source_names: List[str] = []
    input_paths: Dict[str, str] = {}
    point_counts: Dict[str, int] = {}
    for source_index, (logical_name, primary_name, fallback_names, tag) in enumerate(input_specs):
        mesh, path = load_step6_point_surface(
            stage_dir,
            subject_id,
            frame,
            primary_name,
            fallback_names,
        )
        points = np.asarray(mesh.points, dtype=float).copy()
        point_blocks.append(points)
        tag_blocks.append(np.full(len(points), int(tag), dtype=np.int32))
        source_surface_blocks.append(np.full(len(points), int(source_index), dtype=np.int32))
        source_point_blocks.append(np.arange(len(points), dtype=np.int64))
        source_names.append(logical_name)
        input_paths[logical_name] = str(path.resolve())
        point_counts[logical_name] = int(len(points))

    merged_points = np.vstack(point_blocks) if point_blocks else np.empty((0, 3), dtype=float)
    merged_tags = np.concatenate(tag_blocks) if tag_blocks else np.empty((0,), dtype=np.int32)
    merged_source_surface = (
        np.concatenate(source_surface_blocks) if source_surface_blocks else np.empty((0,), dtype=np.int32)
    )
    merged_source_point = (
        np.concatenate(source_point_blocks) if source_point_blocks else np.empty((0,), dtype=np.int64)
    )
    output_path = stage_dir / f"{subject_id}_{frame}_LVRV.vtk"
    write_custom_point_cloud_vtk(
        output_path,
        merged_points,
        merged_tags,
        point_data={
            "sourceSurface": merged_source_surface,
            "sourcePointId": merged_source_point,
        },
    )
    summary = {
        "description": "Raw point-only merge of all six Step 6 LV/RV surfaces; no intersection removal is applied.",
        "subject": subject_id,
        "frame": frame,
        "output_dir": str(stage_dir.resolve()),
        "input_paths": input_paths,
        "output_path": str(output_path.resolve()),
        "source_surfaces": source_names,
        "point_counts": point_counts,
        "total_points": int(len(merged_points)),
        "point_cloud_only": True,
        "vtk_cell_type": "VTK_VERTEX",
        "triangle_faces_written": False,
        "elemTag_counts": {
            "1": int(np.count_nonzero(merged_tags == int(args.lv_tag))),
            "0": int(np.count_nonzero(merged_tags == int(args.rv_tag))),
        },
    }
    summary_path = stage_dir / f"{subject_id}_{frame}_LVRV_point_cloud_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path.resolve())
    return str(output_path.resolve()), summary


def save_step6_open3d_ball_pivoting_reconstruction(
    stage_dir: Path,
    point_cloud_path: str,
    subject_id: str,
    frame: str,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, object]]:
    """Reconstruct source point groups with Open3D Ball Pivoting."""
    point_cloud = pv.read(point_cloud_path)
    source_points = np.asarray(point_cloud.points, dtype=float)
    source_tags = np.asarray(
        point_cloud.cell_data.get("elemTag", np.zeros(len(source_points), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if len(source_tags) != len(source_points):
        source_tags = np.zeros(len(source_points), dtype=np.int32)

    source_surface_ids = np.asarray(
        point_cloud.point_data.get("sourceSurface", np.full(len(source_points), -1, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if len(source_surface_ids) != len(source_points):
        source_surface_ids = np.full(len(source_points), -1, dtype=np.int32)
    source_names = [
        "LV_openbase_epi",
        "LV_openbase_endo",
        "LV_openbase_annulus_recon",
        "RV_openbase_epi",
        "RV_openbase_endo",
        "RV_openbase_annulus_recon",
    ]
    source_tags_by_id = np.asarray([
        int(args.lv_tag),
        int(args.lv_tag),
        int(args.lv_tag),
        int(args.rv_tag),
        int(args.rv_tag),
        int(args.rv_tag),
    ], dtype=np.int32)
    radii_multipliers = [
        float(value)
        for value in parse_csv_values(args.step6_ball_pivot_radii_multipliers)
    ]
    if not radii_multipliers or any(value <= 0.0 for value in radii_multipliers):
        raise ValueError("--step6-ball-pivot-radii-multipliers must contain positive values.")

    reconstructed_point_blocks: List[np.ndarray] = []
    reconstructed_triangle_blocks: List[np.ndarray] = []
    reconstructed_tag_blocks: List[np.ndarray] = []
    source_records: List[Dict[str, object]] = []
    point_offset = 0
    for source_id, source_name in enumerate(source_names):
        source_mask = source_surface_ids == int(source_id)
        points = source_points[source_mask]
        source_record: Dict[str, object] = {
            "source_surface": source_name,
            "source_id": int(source_id),
            "input_points": int(len(points)),
            "input_tag": int(source_tags_by_id[source_id]),
        }
        if len(points) < 3:
            source_record.update({
                "output_points": 0,
                "output_triangles": 0,
                "status": "skipped_fewer_than_three_points",
            })
            source_records.append(source_record)
            continue

        nearest_distances = cKDTree(points).query(points, k=2)[0][:, 1]
        median_spacing = float(np.median(nearest_distances[nearest_distances > 1e-8])) if np.any(nearest_distances > 1e-8) else 0.0
        if median_spacing <= 0.0:
            source_record.update({
                "output_points": 0,
                "output_triangles": 0,
                "status": "skipped_duplicate_points_only",
            })
            source_records.append(source_record)
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        normal_k = min(max(int(args.step6_ball_pivot_normal_k), 3), len(points) - 1)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_k),
            fast_normal_computation=False,
        )
        orientation_k = min(max(normal_k, 3), len(points) - 1)
        pcd.orient_normals_consistent_tangent_plane(orientation_k)
        radii = [median_spacing * multiplier for multiplier in radii_multipliers]
        ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii),
        )
        ball_points = np.asarray(ball_mesh.vertices, dtype=float)
        ball_triangles = np.asarray(ball_mesh.triangles, dtype=np.int64)
        source_record.update({
            "median_neighbor_spacing_mm": median_spacing,
            "normal_k": int(normal_k),
            "ball_pivot_radii_mm": radii,
            "output_points": int(len(ball_points)),
            "output_triangles": int(len(ball_triangles)),
            "status": "reconstructed" if len(ball_triangles) else "no_triangles",
        })
        source_records.append(source_record)
        if len(ball_points) == 0 or len(ball_triangles) == 0:
            continue
        reconstructed_point_blocks.append(ball_points)
        reconstructed_triangle_blocks.append(ball_triangles + int(point_offset))
        reconstructed_tag_blocks.append(
            np.full(len(ball_triangles), int(source_tags_by_id[source_id]), dtype=np.int32)
        )
        point_offset += len(ball_points)

    if not reconstructed_point_blocks or not reconstructed_triangle_blocks:
        raise RuntimeError("Open3D Ball Pivoting produced no triangles from the Step 6 point cloud.")
    reconstructed_points = np.vstack(reconstructed_point_blocks)
    triangles = np.vstack(reconstructed_triangle_blocks)
    reconstructed_tags = np.concatenate(reconstructed_tag_blocks)
    reconstructed_surface = pv.PolyData(reconstructed_points, base.faces_from_triangles(triangles))
    reconstructed_surface.cell_data["elemTag"] = reconstructed_tags

    output_path = stage_dir / f"{subject_id}_{frame}_LVRV_surf.vtk"
    record = save_triangle_surface_vtk(output_path, reconstructed_surface, elem_tag=None)
    summary = {
        "description": "Triangle surface reconstructed from the six Step 6 point groups with Open3D Ball Pivoting.",
        "subject": subject_id,
        "frame": frame,
        "input_point_cloud": str(Path(point_cloud_path).resolve()),
        "output_path": str(output_path.resolve()),
        "method": "open3d_ball_pivoting_per_source_surface_group",
        "normal_k": int(args.step6_ball_pivot_normal_k),
        "radii_multipliers": radii_multipliers,
        "source_points": int(len(source_points)),
        "source_surface_records": source_records,
        "reconstructed_surface": record,
        "tag_assignment": "source-surface tag assigned to each reconstructed patch",
    }
    summary_path = stage_dir / f"{subject_id}_{frame}_LVRV_surface_ball_pivoting_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path.resolve())
    return str(output_path.resolve()), summary


def process_frame_new_pipeline(args: argparse.Namespace, frame: str) -> Dict[str, object]:
    """Run the active seven-stage surface point-cloud pipeline for one subject/frame."""
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    vtk_dir = resolve_vtk_dir(subject_dir, args.vtk_dir)
    pipeline_root = args.pipeline_output_root or subject_dir / "volmesh_pipline"
    if pipeline_root.name.startswith("step"):
        pipeline_root = pipeline_root.parent
    pipeline_root.mkdir(parents=True, exist_ok=True)

    step1_dir = resolve_pipeline_stage_dir(subject_dir, pipeline_root, "step_01_load_surf")
    step2_dir = resolve_pipeline_stage_dir(subject_dir, pipeline_root, "step_02_base_detection")
    step3_dir = resolve_pipeline_stage_dir(subject_dir, pipeline_root, "step_03_LV_openbase")
    step4_dir = resolve_pipeline_stage_dir(subject_dir, pipeline_root, "step_04_RV_openbase")
    step5_dir = resolve_pipeline_stage_dir(subject_dir, pipeline_root, "step_05_move_RVepi")
    step6_dir = resolve_pipeline_stage_dir(subject_dir, pipeline_root, "step_06_reconstruct_annulus")
    step7_dir = resolve_pipeline_stage_dir(subject_dir, pipeline_root, "step_07_whole_points_surf")

    input_paths = input_paths_for_frame(vtk_dir, frame)
    log_stage(f"{subject_id} {frame}: Step 1 load four surfaces")
    surfaces = load_surfaces(vtk_dir, frame)
    step1_paths, _ = save_step1_loaded_surfaces(step1_dir, subject_id, frame, surfaces, input_paths)

    frame_args = openbase.namespace_with_overrides(args, {"frame": frame})
    log_stage(f"{subject_id} {frame}: Step 2 detect subject-level base side")
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
    step2_paths, _ = save_step2_base_detection(
        step2_dir,
        subject_id,
        frame,
        subject_base_detection,
        lv_profile,
        rv_profile,
        lv_base_side,
        rv_base_side,
    )

    log_stage(f"{subject_id} {frame}: Step 3 build LV open-base myocardial shell")
    lv_build = build_ventricle_shell_for_pre_step5(
        "LV", surfaces["lv_endo"], surfaces["lv_epi"], lv_profile, frame_args
    )
    step3_paths, _ = save_openbase_surface_stage(step3_dir, subject_id, frame, "LV", lv_build)

    log_stage(f"{subject_id} {frame}: Step 4 build initial RV open-base myocardial shell")
    rv_build = build_ventricle_shell_for_pre_step5(
        "RV", surfaces["rv_endo"], surfaces["rv_epi"], rv_profile, frame_args
    )
    step4_paths, _ = save_openbase_surface_stage(step4_dir, subject_id, frame, "RV", rv_build)

    if args.stop_after_open_shells:
        return {
            "subject": subject_id,
            "frame": frame,
            "stopped_after": "step04_RV_openbase",
            "stage_outputs": {
                "step_01_load_surf": step1_paths,
                "step_02_base_detection": step2_paths,
                "step_03_LV_openbase": step3_paths,
                "step_04_RV_openbase": step4_paths,
            },
        }

    rv_endo_step5_surface = rv_build.endo_open.extract_surface().triangulate().clean(tolerance=frame_args.surface_clean_tolerance)
    rv_epi_step5_surface = rv_build.epi_open.extract_surface().triangulate().clean(tolerance=frame_args.surface_clean_tolerance)
    log_stage(f"{subject_id} {frame}: Step 5 move RVepi away from fixed RVendo by at least 3 mm")
    rv_epi_moved, moved_mask, opening_mask, transition_band_mask, moved_opening_loop, distance_info = enforce_rvepi_distance_adaptive_opening_ring(
        rv_epi_step5_surface,
        rv_endo_step5_surface,
        rv_build.epi_opening_loop,
        frame_args,
    )
    distance_info["step"] = "step05_move_RVepi"
    distance_info["opening_loop_after_step5"] = loop_record(moved_opening_loop)
    step5_paths, _ = save_step5_moved_rvepi(
        step5_dir,
        subject_id,
        frame,
        rv_epi_moved,
        rv_endo_step5_surface,
        moved_mask,
        opening_mask,
        transition_band_mask,
        distance_info,
        frame_args,
    )

    log_stage(f"{subject_id} {frame}: Step 6 reconstruct LV/RV annuli from current epi/endo boundaries")
    lv_epi_loop, lv_endo_loop, lv_annulus, lv_annulus_info = rebuild_openbase_annulus_from_current_surfaces(
        "LV",
        lv_build.epi_open,
        lv_build.endo_open,
        lv_profile,
        frame_args,
    )
    rv_epi_loop, rv_endo_loop, rv_annulus, rv_annulus_info = rebuild_openbase_annulus_from_current_surfaces(
        "RV",
        rv_epi_moved,
        rv_endo_step5_surface,
        rv_profile,
        frame_args,
    )
    step6_paths, _ = save_step6_reconstructed_annuli(
        step6_dir,
        subject_id,
        frame,
        lv_build.epi_open,
        lv_build.endo_open,
        lv_annulus,
        rv_epi_moved,
        rv_endo_step5_surface,
        rv_annulus,
        {
            **lv_annulus_info,
            "selected_epi_boundary_loop": loop_record(lv_epi_loop),
            "selected_endo_boundary_loop": loop_record(lv_endo_loop),
        },
        {
            **rv_annulus_info,
            "selected_epi_boundary_loop": loop_record(rv_epi_loop),
            "selected_endo_boundary_loop": loop_record(rv_endo_loop),
        },
        frame_args,
    )
    log_stage(f"{subject_id} {frame}: merge the six Step 6 surfaces into a raw point cloud")
    step6_lvrv_point_cloud_path, step6_lvrv_point_cloud_summary = save_step6_merged_point_cloud(
        step6_dir,
        subject_id,
        frame,
        frame_args,
    )
    step6_paths["LVRV_point_cloud"] = step6_lvrv_point_cloud_path
    step6_paths["LVRV_point_cloud_summary"] = step6_lvrv_point_cloud_summary["summary_path"]
    log_stage(f"{subject_id} {frame}: reconstruct a triangle surface from the Step 6 point cloud with Open3D Ball Pivoting")
    step6_lvrv_surface_path, step6_lvrv_surface_summary = save_step6_open3d_ball_pivoting_reconstruction(
        step6_dir,
        step6_lvrv_point_cloud_path,
        subject_id,
        frame,
        frame_args,
    )
    step6_paths["LVRV_surface_ball_pivoting"] = step6_lvrv_surface_path
    step6_paths["LVRV_surface_ball_pivoting_summary"] = step6_lvrv_surface_summary["summary_path"]

    log_stage(f"{subject_id} {frame}: Step 7 remove RV points intersecting LVepi and save merged point cloud")
    step7_paths, _ = save_step7_whole_points_surface(
        step7_dir,
        step6_dir,
        subject_id,
        frame,
        frame_args,
    )

    return {
        "subject": subject_id,
        "frame": frame,
        "stopped_after": "step07_whole_points_surf",
        "stage_outputs": {
            "step_01_load_surf": step1_paths,
            "step_02_base_detection": step2_paths,
            "step_03_LV_openbase": step3_paths,
            "step_04_RV_openbase": step4_paths,
            "step_05_move_RVepi": step5_paths,
            "step_06_reconstruct_annulus": step6_paths,
            "step_07_whole_points_surf": step7_paths,
        },
        "active_pipeline": [
            "load_four_surfaces",
            "subject_level_base_side_detection",
            "build_LV_openbase_shell",
            "build_initial_RV_openbase_shell",
            "move_RVepi_to_minimum_3mm_from_fixed_RVendo_with_adaptive_opening_ring",
            "directly_reconstruct_LV_and_RV_openbase_annuli",
            "merge_six_step06_surfaces_to_raw_point_only_LVRV_cloud",
            "reconstruct_LVRV_surface_with_open3d_ball_pivoting",
            "remove_RV_points_intersecting_LVepi_and_save_point_only_LVRV_cloud",
        ],
        "disabled_downstream_stages": [
            "intersection_removal",
            "surface_merge",
            "TetGen",
            "volume_tagging",
        ],
    }


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
    step11_stage_dir = resolve_pipeline_stage_dir(subject_dir, args.pipeline_output_root, "step11volmesh_LVRV")
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

    log_stage(f"{subject_id} {frame}: step 5 delete intersecting RVepi vertices and rebuild sparse-guided smooth LVepi-edge transition strip")
    rv_epi_step5 = substitute_intersecting_vertices_with_lvepi_existing_vertices(
        "RVepi_open",
        rv_build.epi_open,
        lv_epi_intersection_reference,
        rv_build.epi_opening_loop,
        int(args.rv_tag),
        args,
    )
    log_stage(f"{subject_id} {frame}: step 6 delete intersecting RVendo vertices and rebuild sparse-guided smooth LVepi-edge transition strip")
    rv_endo_step6 = substitute_intersecting_vertices_with_lvepi_existing_vertices(
        "RV_open",
        rv_build.endo_open,
        lv_epi_intersection_reference,
        rv_build.endo_opening_loop,
        int(args.rv_tag),
        args,
    )

    rv_epi_trimmed = rv_epi_step5["surface"]
    rv_epi_trimmed_core = rv_epi_step5["trimmed_surface"]
    rv_epi_cut_mask = rv_epi_step5["cut_mask"]
    rv_epi_removal = rv_epi_step5["info"]
    rv_endo_step6_surface = rv_endo_step6["surface"]
    rv_endo_trimmed_core = rv_endo_step6["trimmed_surface"]
    rv_endo_cut_mask = rv_endo_step6["cut_mask"]
    rv_endo_removal = rv_endo_step6["info"]

    intersection_stage_paths: Dict[str, str] = {}
    intersection_stage_summary: Dict[str, object] = {}
    if args.save_intersection_removal_stages:
        log_stage(f"{subject_id} {frame}: save step 5/6 vertex deletion and sparse-guided LVepi-edge transition-strip outputs")
        intersection_stage_paths, intersection_stage_summary = save_intersection_removal_stage_outputs(
            intersection_stage_dir,
            subject_id,
            frame,
            rv_epi_trimmed,
            rv_epi_cut_mask,
            rv_epi_removal,
            rv_endo_step6_surface,
            rv_endo_cut_mask,
            rv_endo_removal,
            extra_surfaces=[
                ("stage05_RVepi_kept_original_faces_before_LVepi_substitution_surface", rv_epi_step5["trimmed_surface"], rv_epi_step5["trimmed_cut_mask"], int(args.rv_tag)),
                ("stage05_RVepi_rebuilt_mixed_faces_with_existing_LVepi_vertices_surface", rv_epi_step5["stitch_surface"], None, int(args.rv_tag)),
                ("stage06_RVendo_kept_original_faces_before_LVepi_substitution_surface", rv_endo_step6["trimmed_surface"], rv_endo_step6["trimmed_cut_mask"], int(args.rv_tag)),
                ("stage06_RVendo_rebuilt_mixed_faces_with_existing_LVepi_vertices_surface", rv_endo_step6["stitch_surface"], None, int(args.rv_tag)),
            ],
            extra_line_loops=[
                ("stage05_RVepi_new_boundary_LVepi_line_loop", rv_epi_step5["lv_seam_loops"], int(args.rv_tag)),
                ("stage06_RVendo_new_boundary_LVepi_line_loop", rv_endo_step6["lv_seam_loops"], int(args.rv_tag)),
            ],
        )

    if args.stop_after_intersection_removal:
        return {
            "frame": frame,
            "stopped_after": "step05_06_intersection_removal",
            "open_shell_stage_outputs": open_shell_stage_paths,
            "intersection_removal_stage_outputs": intersection_stage_paths,
        }

    log_stage(f"{subject_id} {frame}: keep RVepi fixed, move RVendo core to >=3 mm, then rebuild RVendo LVepi-interface ring from original LVepi vertices")
    rv_endo_step7_lvepi_vertex_id_loops, rv_endo_step7_lvepi_loops, rv_endo_step7_lvepi_ring_info = select_step7_lvepi_interface_loops_away_from_rvepi_boundary(
        rv_endo_step6["lv_seam_loops"],
        rv_epi_step5["lv_seam_loops"],
        rv_epi_trimmed,
        lv_epi_intersection_reference,
        args,
    )
    rv_endo_fixed_core, rv_endo_moved_mask_core, rv_endo_opening_mask_core, rv_endo_transition_band_mask_core, rv_endo_step7_opening_loop, rv_distance_fix = enforce_rvendo_distance_adaptive_opening_ring(
        rv_endo_trimmed_core,
        rv_epi_trimmed,
        rv_build.endo_opening_loop,
        args,
    )
    rv_endo_step7_interface = rebuild_rvendo_lvepi_interface_after_step7(
        "RVendo_step7",
        rv_endo_fixed_core,
        rv_endo_step6["rv_cut_loops"],
        rv_endo_step7_lvepi_vertex_id_loops,
        rv_endo_step7_lvepi_loops,
        lv_epi_intersection_reference,
        rv_epi_trimmed,
        int(args.rv_tag),
        args,
    )
    rv_endo_fixed = rv_endo_step7_interface["surface"]
    mask_map_tol = max(float(args.boundary_match_tolerance), float(args.surface_clean_tolerance) * 10.0)
    rv_endo_moved_mask_from_core = map_source_points_to_target_mask(
        np.asarray(rv_endo_fixed_core.points, dtype=float)[np.asarray(rv_endo_moved_mask_core, dtype=bool)],
        np.asarray(rv_endo_fixed.points, dtype=float),
        mask_map_tol,
    )
    rv_endo_moved_mask = np.asarray(rv_endo_moved_mask_from_core, dtype=bool) | np.asarray(rv_endo_step7_interface["corrected_moved_mask"], dtype=bool)
    rv_endo_opening_mask = map_source_points_to_target_mask(
        np.asarray(rv_endo_fixed_core.points, dtype=float)[np.asarray(rv_endo_opening_mask_core, dtype=bool)],
        np.asarray(rv_endo_fixed.points, dtype=float),
        mask_map_tol,
    )
    rv_endo_transition_band_mask = map_source_points_to_target_mask(
        np.asarray(rv_endo_fixed_core.points, dtype=float)[np.asarray(rv_endo_transition_band_mask_core, dtype=bool)],
        np.asarray(rv_endo_fixed.points, dtype=float),
        mask_map_tol,
    )
    rv_endo_lvepi_cut_mask = np.asarray(rv_endo_step7_interface["cut_mask"], dtype=bool)
    rv_distance_fix["rvendo_lvepi_interface_boundary_ring_step7"] = rv_endo_step7_lvepi_ring_info
    rv_distance_fix["rvendo_lvepi_interface_rebuild_after_step7"] = rv_endo_step7_interface["info"]

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
            extra_surfaces=[
                ("stage07_RVendo_moved_core_before_LVepi_interface_rebuild_surface", rv_endo_fixed_core, rv_endo_moved_mask_core, int(args.rv_tag)),
                ("stage07_RVendo_new_LVepi_interface_ring_rebuilt_surface", rv_endo_fixed, rv_endo_lvepi_cut_mask, int(args.rv_tag)),
            ],
            extra_line_loops=[
                ("stage07_RVepi_fixed_LVepi_interface_boundary_ring", rv_epi_step5["lv_seam_loops"], int(args.rv_tag)),
                ("stage07_RVendo_new_LVepi_interface_boundary_ring", rv_endo_step7_lvepi_loops, int(args.rv_tag)),
            ],
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

    log_stage(f"{subject_id} {frame}: step 9 split LVepi septum/noseptum and merge final whole surface")
    step9_surfaces, step9_build_info = build_partitioned_step9_surface_set(
        lv_epi_step8,
        lv_endo_step8,
        rv_epi_step8,
        rv_endo_step8,
        lv_step8_annulus,
        rv_step8_annulus,
        rv_epi_step5["lv_seam_loops"],
        rv_endo_step7_lvepi_loops,
        args,
    )
    whole_epi = step9_surfaces["LVRV_epi_surface"]
    whole_endo = step9_surfaces["whole_endo_surface"]
    whole_epi_endo = step9_surfaces["whole_epi_endo_surface"]

    merge_surface_stage_paths: Dict[str, str] = {}
    merge_surface_stage_summary: Dict[str, object] = {}
    if args.save_merge_surface_stage:
        log_stage(f"{subject_id} {frame}: save step 9 partitioned LVepi and final whole-surface outputs")
        merge_surface_stage_paths, merge_surface_stage_summary = save_merge_surface_stage_outputs(
            merge_surface_stage_dir,
            subject_id,
            frame,
            step9_surfaces,
            step9_build_info,
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

    tetgen_input_key = "stage09_whole_epi_endo_surface"
    tetgen_input_path_text = merge_surface_stage_paths.get(tetgen_input_key)
    if not tetgen_input_path_text:
        raise RuntimeError(
            "Step 10 requires the Step 9 final whole_epi_endo_surface VTK file. "
            "Keep --save-merge-surface-stage enabled so the obstruction-free surface is written before TetGen."
        )
    tetgen_input_path = Path(tetgen_input_path_text)
    if not tetgen_input_path.exists():
        raise FileNotFoundError(f"Step 10 TetGen input surface does not exist: {tetgen_input_path}")

    log_stage(f"{subject_id} {frame}: load Step 9 topology-first whole_epi_endo_surface as TetGen input: {tetgen_input_path.name}")
    overall_surface = pv.read(tetgen_input_path).extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    overall_surface_source = {
        "method": "use_saved_step09_topology_first_whole_epi_endo_surface_as_step10_TetGen_PLC_input",
        "path": str(tetgen_input_path.resolve()),
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
        "reason": "Step 10 now uses the saved Step 9 LVepi-patch-removed whole_epi_endo surface directly as the TetGen PLC input.",
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
            "RVendo_after_step6_intersection_removal": rv_endo_step6_surface,
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

    output_mesh = step11_stage_dir / f"{subject_id}_{frame}_openbase_LVRV_tetgen_volmesh.vtk"
    write_custom_vtk(output_mesh, tet_points, tet_cells, elem_tags)
    legend_path = step11_stage_dir / f"{subject_id}_{frame}_tetgen_elemTag_legend.csv"
    write_elem_tag_legend(legend_path, args.lv_tag, args.rv_tag)

    tetgen_stage_paths: Dict[str, str] = {}
    tetgen_stage_summary: Dict[str, object] = {}
    if args.save_tetgen_stage:
        log_stage(f"{subject_id} {frame}: save step 10/11 TetGen and final tagged mesh outputs")
        tetgen_stage_paths, tetgen_stage_summary = save_tetgen_stage_outputs(
            tetgen_stage_dir,
            step11_stage_dir,
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
            "step9_split_LVepi_into_septum_and_LVepi_noseptum_by_RVepi_step5_and_RVendo_step7_boundary_rings",
            "step9_merge_LVepi_noseptum_plus_RVepi_as_LVRV_epi_surface",
            "step9_merge_RVendo_plus_septum_as_RV_endo_surface_preserving_septum_elemTag_1",
            "step9_merge_LVRV_epi_RV_endo_LV_endo_and_openbase_annuli_as_whole_epi_endo_surface",
            "step10_load_saved_step09_whole_epi_endo_surface_as_TetGen_PLC_input",
            "step10_tetgen_whole_volume_mesh",
            "save_step10_raw_tetgen_outputs_to_pipeline_folder",
            "step11_post_tag_using_original_closed_LVepi_centroid_or_3of4_vertex_inside_rule_then_cleanup",
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
            "rv_epi_step5_trimmed_core": surface_summary(rv_epi_trimmed_core),
            "rv_epi_step5_stitched": surface_summary(rv_epi_trimmed),
            "rv_endo_step6_trimmed_core": surface_summary(rv_endo_trimmed_core),
            "rv_endo_step6_stitched": surface_summary(rv_endo_step6_surface),
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
    summary_path = step11_stage_dir / f"{subject_id}_{frame}_openbase_LVRV_tetgen_summary.json"
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
    results = [process_frame_new_pipeline(args, frame) for frame in frames]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
