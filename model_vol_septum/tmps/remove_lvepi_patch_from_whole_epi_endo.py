#!/usr/bin/env python3
"""Remove the LVepi septal patch from an existing whole epi/endo surface.

The input mesh is the already saved step-9 whole epi/endo surface, for example:
  subject_1_ED_stage09_whole_epi_endo_surface.vtk

The script identifies the LVepi patch enclosed by the RVepi/LVepi and
RVendo/LVepi clipped intersection boundaries, removes that patch from the
existing whole surface, and writes a new triangle-surface VTK in the same
custom ASCII style used by the current pipeline.
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
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_open_volmesh_stable import (  # noqa: E402
    base,
    connected_face_components,
    face_areas,
    loop_points,
    surface_summary,
    write_custom_surface_vtk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--pipeline-root", type=Path, default=None, help="Defaults to SUBJECT_DIR/volmesh_generate_pipline.")
    parser.add_argument("--whole-epi-endo", type=Path, default=None, help="Existing stage09 whole epi/endo surface VTK.")
    parser.add_argument("--lv-epi", type=Path, default=None, help="Stage08 LVepi open-base surface VTK.")
    parser.add_argument("--rv-epi", type=Path, default=None, help="Stage08 RVepi fixed open-base surface VTK.")
    parser.add_argument("--rv-endo", type=Path, default=None, help="Stage08 RVendo fixed open-base surface VTK.")
    parser.add_argument("--rv-openbase-annulus", type=Path, default=None, help="Stage08 RV open-base annulus surface VTK.")
    parser.add_argument("--rv-epi-reference", type=Path, default=None, help="Closed/original RVepi surface used to identify LVepi patch between RVepi and RVendo.")
    parser.add_argument("--rv-endo-reference", type=Path, default=None, help="Closed/original RVendo surface used to identify LVepi patch between RVepi and RVendo.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--patch-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--lv-tag", type=int, default=1)
    parser.add_argument("--rv-tag", type=int, default=0)
    parser.add_argument("--clean-tolerance", type=float, default=1e-7)
    parser.add_argument("--lvepi-match-tolerance", type=float, default=0.25, help="Max distance for a whole-surface face center to be treated as an LVepi face.")
    parser.add_argument("--annulus-loop-match-tolerance", type=float, default=0.5, help="Used to exclude the basal RV open-base loop from the RV intersection loops.")
    parser.add_argument("--loop-component-tolerance", type=float, default=3.0, help="Max p95 distance from RV gap-loop points to selected LVepi patch component.")
    parser.add_argument("--component-min-area-fraction", type=float, default=0.02)
    parser.add_argument("--allow-largest-component-fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_pipeline_root(subject_dir: Path, pipeline_root: Optional[Path]) -> Path:
    return pipeline_root if pipeline_root is not None else subject_dir / "volmesh_generate_pipline"


def default_paths(args: argparse.Namespace) -> Dict[str, Path]:
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    frame = args.frame
    root = resolve_pipeline_root(subject_dir, args.pipeline_root)
    step8 = root / "step8fix_intersection_new_surf"
    step9 = root / "step9merge_surface"
    vtk_dir = subject_dir / "vtks"
    return {
        "whole_epi_endo": args.whole_epi_endo or step9 / f"{subject_id}_{frame}_stage09_whole_epi_endo_surface.vtk",
        "lv_epi": args.lv_epi or step8 / f"{subject_id}_{frame}_stage08_LVepi_openbase_surface.vtk",
        "rv_epi": args.rv_epi or step8 / f"{subject_id}_{frame}_stage08_RVepi_fixed_openbase_surface.vtk",
        "rv_endo": args.rv_endo or step8 / f"{subject_id}_{frame}_stage08_RVendo_fixed_openbase_surface.vtk",
        "rv_openbase_annulus": args.rv_openbase_annulus or step8 / f"{subject_id}_{frame}_stage08_RV_openbase_annulus_surface.vtk",
        "rv_epi_reference": args.rv_epi_reference or vtk_dir / f"RVepi_{frame}.vtk",
        "rv_endo_reference": args.rv_endo_reference or vtk_dir / f"RV_{frame}.vtk",
        "output": args.output or step9 / f"{subject_id}_{frame}_stage09_whole_epi_endo_without_LVepi_RV_interface_patch.vtk",
        "patch_output": args.patch_output or step9 / f"{subject_id}_{frame}_stage09_removed_LVepi_patch_between_RVendo_RVepi_boundaries.vtk",
        "summary_output": args.summary_output or step9 / f"{subject_id}_{frame}_stage09_remove_LVepi_patch_between_RV_boundaries_summary.json",
    }


def load_surface(path: Path, tolerance: float) -> pv.PolyData:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pv.read(path).extract_surface().triangulate().clean(tolerance=float(tolerance))


def cell_center_points(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface = mesh.extract_surface().triangulate()
    triangles = base.triangle_faces(surface)
    points = np.asarray(surface.points, dtype=float)
    centers = points[triangles].mean(axis=1) if len(triangles) else np.empty((0, 3), dtype=float)
    return points, triangles, centers


def closest_distances_to_tri_surface(surface: pv.PolyData, query_points: np.ndarray) -> np.ndarray:
    surface_points = np.asarray(surface.points, dtype=float)
    if len(query_points) == 0 or len(surface_points) == 0:
        return np.empty((0,), dtype=float)
    # Cell-center distances only need to distinguish exact LVepi faces from LVendo/annulus.
    # A point-tree is enough here because all candidate faces come from saved pipeline surfaces.
    return cKDTree(surface_points).query(np.asarray(query_points, dtype=float), k=1)[0]


def write_surface(path: Path, points: np.ndarray, triangles: np.ndarray, elem_tags: np.ndarray) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_custom_surface_vtk(path, np.asarray(points, dtype=float), np.asarray(triangles, dtype=np.int64), np.asarray(elem_tags, dtype=np.int32))
    mesh = pv.PolyData(np.asarray(points, dtype=float), base.faces_from_triangles(np.asarray(triangles, dtype=np.int64)))
    mesh.cell_data["elemTag"] = np.asarray(elem_tags, dtype=np.int32)
    record = surface_summary(mesh)
    record.update({
        "path": str(path.resolve()),
        "cell_type": 5,
        "elemTag_counts": {str(k): int(v) for k, v in sorted(Counter(map(int, elem_tags)).items())},
    })
    return record


def compact_surface(points: np.ndarray, triangles: np.ndarray, elem_tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(triangles) == 0:
        raise RuntimeError("No triangles remain after LVepi patch removal.")
    used = np.unique(np.asarray(triangles, dtype=np.int64).ravel())
    point_map = -np.ones(len(points), dtype=np.int64)
    point_map[used] = np.arange(len(used), dtype=np.int64)
    return np.asarray(points, dtype=float)[used], point_map[triangles], np.asarray(elem_tags, dtype=np.int32)


def loop_distance_record(loop: pv.PolyData, target_points: np.ndarray) -> Dict[str, object]:
    pts = loop_points(loop)
    if len(pts) == 0 or len(target_points) == 0:
        return {"points": int(len(pts)), "min": None, "mean": None, "p95": None, "max": None}
    distances = cKDTree(np.asarray(target_points, dtype=float)).query(pts, k=1)[0]
    return {
        "points": int(len(pts)),
        "min": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def classify_rv_boundary_loops(
    name: str,
    rv_surface: pv.PolyData,
    rv_openbase_annulus: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[List[pv.PolyData], Dict[str, object]]:
    loops = base.boundary_loops(rv_surface.extract_surface().triangulate().clean(tolerance=float(args.clean_tolerance)))
    annulus_points = np.asarray(rv_openbase_annulus.points, dtype=float)
    if not loops:
        raise RuntimeError(f"{name}: no boundary loops detected.")
    if len(annulus_points) == 0:
        raise RuntimeError("RV open-base annulus has no points; cannot identify basal loop.")

    annulus_tree = cKDTree(annulus_points)
    records = []
    gap_loops: List[pv.PolyData] = []
    for loop_id, loop in enumerate(loops):
        pts = loop_points(loop)
        distances = annulus_tree.query(pts, k=1)[0] if len(pts) else np.empty((0,), dtype=float)
        p95 = float(np.percentile(distances, 95)) if len(distances) else float("inf")
        is_openbase = p95 <= float(args.annulus_loop_match_tolerance)
        record = {
            "loop_id": int(loop_id),
            "points": int(len(pts)),
            "length": float(loop.length),
            "annulus_distance_p95": p95,
            "annulus_distance_max": float(np.max(distances)) if len(distances) else None,
            "classified_as": "RV_openbase_annulus_boundary" if is_openbase else "RV_LVepi_intersection_boundary",
        }
        records.append(record)
        if not is_openbase and len(pts) >= 3:
            gap_loops.append(base.make_polyline(pts))

    if not gap_loops:
        fallback_records = sorted(records, key=lambda item: float(item["annulus_distance_p95"]), reverse=True)
        fallback_id = int(fallback_records[0]["loop_id"])
        gap_loops = [base.make_polyline(loop_points(loops[fallback_id]))]
        records[fallback_id]["classified_as"] = "RV_LVepi_intersection_boundary_fallback_farthest_from_openbase_annulus"

    return gap_loops, {
        "surface": name,
        "method": "boundary_loops_not_matching_RV_openbase_annulus_are_RV_LVepi_intersection_boundaries",
        "loop_count": int(len(loops)),
        "gap_loop_count": int(len(gap_loops)),
        "annulus_loop_match_tolerance_mm": float(args.annulus_loop_match_tolerance),
        "loops": records,
    }


def component_loop_proximity(component_points: np.ndarray, loops: Sequence[pv.PolyData]) -> Tuple[bool, List[Dict[str, object]]]:
    records = [loop_distance_record(loop, component_points) for loop in loops]
    return bool(records), records


def select_lvepi_patch_faces(
    whole_surface: pv.PolyData,
    lv_epi: pv.PolyData,
    rv_epi_reference: pv.PolyData,
    rv_endo_reference: pv.PolyData,
    rv_epi_gap_loops: Sequence[pv.PolyData],
    rv_endo_gap_loops: Sequence[pv.PolyData],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    points, triangles, centers = cell_center_points(whole_surface)
    if len(triangles) == 0:
        raise RuntimeError("Whole epi/endo surface has no triangle faces.")

    tags = whole_surface.cell_data.get("elemTag")
    if tags is None or len(tags) != len(triangles):
        raise RuntimeError("Whole epi/endo surface must contain one elemTag value per triangle cell.")
    tags = np.rint(np.asarray(tags).reshape(-1)).astype(np.int32)

    distance_to_lvepi = closest_distances_to_tri_surface(lv_epi, centers)
    lv_epi_face_mask = (tags == int(args.lv_tag)) & (distance_to_lvepi <= float(args.lvepi_match_tolerance))

    inside_rv_epi = base.enclosed_mask(centers, rv_epi_reference)
    inside_rv_endo = base.enclosed_mask(centers, rv_endo_reference)
    candidate_mask = lv_epi_face_mask & inside_rv_epi & (~inside_rv_endo)

    candidate_source = "whole_surface_LVepi_faces_with_centers_inside_original_RVepi_and_outside_original_RVendo"
    if not np.any(candidate_mask):
        gap_points = np.vstack([loop_points(loop) for loop in list(rv_epi_gap_loops) + list(rv_endo_gap_loops) if len(loop_points(loop))])
        distances_to_gap = cKDTree(gap_points).query(centers, k=1)[0] if len(gap_points) else np.full(len(centers), np.inf)
        candidate_mask = lv_epi_face_mask & (distances_to_gap <= float(args.loop_component_tolerance))
        candidate_source = "fallback_whole_surface_LVepi_faces_near_RV_gap_boundaries"

    components = connected_face_components(candidate_mask, triangles)
    if not components:
        raise RuntimeError("No connected LVepi patch component found between RVendo/RVepi boundaries.")

    areas = face_areas(points, triangles)
    component_records = []
    selected_components: List[np.ndarray] = []
    for component_id, component in enumerate(components):
        component_vertices = np.unique(triangles[component].ravel())
        component_points = points[component_vertices]
        has_epi_loops, epi_loop_distances = component_loop_proximity(component_points, rv_epi_gap_loops)
        has_endo_loops, endo_loop_distances = component_loop_proximity(component_points, rv_endo_gap_loops)
        epi_p95 = [float(record["p95"]) for record in epi_loop_distances if record["p95"] is not None]
        endo_p95 = [float(record["p95"]) for record in endo_loop_distances if record["p95"] is not None]
        close_to_epi_boundary = bool(has_epi_loops and epi_p95 and min(epi_p95) <= float(args.loop_component_tolerance))
        close_to_endo_boundary = bool(has_endo_loops and endo_p95 and min(endo_p95) <= float(args.loop_component_tolerance))
        area = float(areas[component].sum())
        record = {
            "component_id": int(component_id),
            "faces": int(len(component)),
            "vertices": int(len(component_vertices)),
            "area": area,
            "close_to_RVepi_LVepi_boundary": close_to_epi_boundary,
            "close_to_RVendo_LVepi_boundary": close_to_endo_boundary,
            "RVepi_boundary_distances": epi_loop_distances,
            "RVendo_boundary_distances": endo_loop_distances,
        }
        component_records.append(record)
        if close_to_epi_boundary and close_to_endo_boundary:
            selected_components.append(component)

    selected_method = "components_inside_RV_myo_band_and_touching_both_RVepi_RVendo_gap_boundaries"
    if not selected_components:
        if not bool(args.allow_largest_component_fallback):
            raise RuntimeError("No LVepi patch component touches both RV boundaries and fallback is disabled.")
        selected_components = [components[0]]
        selected_method = "fallback_largest_connected_candidate_component"

    max_area = max(float(areas[component].sum()) for component in selected_components)
    min_area = max_area * float(args.component_min_area_fraction)
    selected_components = [component for component in selected_components if float(areas[component].sum()) >= min_area]

    remove_mask = np.zeros(len(triangles), dtype=bool)
    for component in selected_components:
        remove_mask[component] = True

    return remove_mask, {
        "method": "remove_LVepi_patch_enclosed_by_RVepi_LVepi_and_RVendo_LVepi_intersection_boundaries_from_existing_whole_epi_endo_surface",
        "candidate_source": candidate_source,
        "whole_surface_faces": int(len(triangles)),
        "whole_surface_elemTag_counts": {str(k): int(v) for k, v in sorted(Counter(map(int, tags)).items())},
        "LVepi_face_match_tolerance_mm": float(args.lvepi_match_tolerance),
        "LVepi_faces_in_whole_surface": int(np.count_nonzero(lv_epi_face_mask)),
        "candidate_patch_faces_before_component_selection": int(np.count_nonzero(candidate_mask)),
        "candidate_component_count": int(len(components)),
        "selected_component_count": int(len(selected_components)),
        "selected_method": selected_method,
        "selected_patch_faces": int(np.count_nonzero(remove_mask)),
        "selected_patch_area": float(areas[remove_mask].sum()) if np.any(remove_mask) else 0.0,
        "loop_component_tolerance_mm": float(args.loop_component_tolerance),
        "component_min_area_fraction": float(args.component_min_area_fraction),
        "component_records": component_records[:30],
    }


def main() -> None:
    args = parse_args()
    paths = default_paths(args)
    for key, path in paths.items():
        if key in {"output", "patch_output", "summary_output"}:
            continue
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")

    whole = load_surface(paths["whole_epi_endo"], args.clean_tolerance)
    lv_epi = load_surface(paths["lv_epi"], args.clean_tolerance)
    rv_epi = load_surface(paths["rv_epi"], args.clean_tolerance)
    rv_endo = load_surface(paths["rv_endo"], args.clean_tolerance)
    rv_annulus = load_surface(paths["rv_openbase_annulus"], args.clean_tolerance)
    rv_epi_reference = load_surface(paths["rv_epi_reference"], args.clean_tolerance)
    rv_endo_reference = load_surface(paths["rv_endo_reference"], args.clean_tolerance)

    rv_epi_gap_loops, rv_epi_loop_info = classify_rv_boundary_loops("RVepi", rv_epi, rv_annulus, args)
    rv_endo_gap_loops, rv_endo_loop_info = classify_rv_boundary_loops("RVendo", rv_endo, rv_annulus, args)
    remove_mask, patch_info = select_lvepi_patch_faces(
        whole,
        lv_epi,
        rv_epi_reference,
        rv_endo_reference,
        rv_epi_gap_loops,
        rv_endo_gap_loops,
        args,
    )

    points, triangles, _ = cell_center_points(whole)
    tags = np.rint(np.asarray(whole.cell_data["elemTag"]).reshape(-1)).astype(np.int32)

    keep_mask = ~remove_mask
    out_points, out_triangles, out_tags = compact_surface(points, triangles[keep_mask], tags[keep_mask])
    patch_points, patch_triangles, patch_tags = compact_surface(points, triangles[remove_mask], tags[remove_mask])

    output_record = write_surface(paths["output"], out_points, out_triangles, out_tags)
    patch_record = write_surface(paths["patch_output"], patch_points, patch_triangles, patch_tags)

    summary = {
        "subject_dir": str(args.subject_dir.resolve()),
        "frame": args.frame,
        "inputs": {key: str(path.resolve()) for key, path in paths.items() if key not in {"output", "patch_output", "summary_output"}},
        "outputs": {
            "whole_epi_endo_without_LVepi_RV_interface_patch": str(paths["output"].resolve()),
            "removed_LVepi_patch_between_RVendo_RVepi_boundaries": str(paths["patch_output"].resolve()),
            "summary": str(paths["summary_output"].resolve()),
        },
        "RVepi_boundary_detection": rv_epi_loop_info,
        "RVendo_boundary_detection": rv_endo_loop_info,
        "LVepi_patch_removal": patch_info,
        "surface_summaries": {
            "input_whole_epi_endo": surface_summary(whole),
            "input_LVepi_reference_for_cell_matching": surface_summary(lv_epi),
            "output_whole_epi_endo_without_patch": output_record,
            "removed_patch": patch_record,
        },
        "parameters": {
            "lv_tag": int(args.lv_tag),
            "rv_tag": int(args.rv_tag),
            "clean_tolerance": float(args.clean_tolerance),
            "lvepi_match_tolerance": float(args.lvepi_match_tolerance),
            "annulus_loop_match_tolerance": float(args.annulus_loop_match_tolerance),
            "loop_component_tolerance": float(args.loop_component_tolerance),
            "component_min_area_fraction": float(args.component_min_area_fraction),
        },
    }
    paths["summary_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_output"].write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "output": str(paths["output"].resolve()),
        "removed_patch": str(paths["patch_output"].resolve()),
        "summary": str(paths["summary_output"].resolve()),
        "removed_faces": int(patch_info["selected_patch_faces"]),
        "output_faces": int(output_record["triangles"]),
    }, indent=2))


if __name__ == "__main__":
    main()
