#!/usr/bin/env python3
"""Fix LV/RV surface intersections with RV constrained inside fixed RVepi.

This version preserves LVendo/LVepi geometry, fixes RVepi against LVepi, then
uses the fixed RVepi as the packaging surface for RV. Final surfaces are saved
as legacy ASCII VTK triangle unstructured grids with CELL_DATA scalar elemTag:

  LVendo = 1
  LVepi  = 2
  RV     = 3
  RVepi  = 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code_explain.tmps import fix_lv_rv_surface_intersections as surface_fix


def sample_sets(mesh: pv.PolyData) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(mesh.points, dtype=float)
    triangles = surface_fix.triangle_faces(mesh)
    edges = surface_fix.unique_edges(triangles)
    return {
        "vertices": (points, np.arange(len(points), dtype=np.int64)[:, None]),
        "face_centers": (points[triangles].mean(axis=1), triangles),
        "edge_midpoints": (points[edges].mean(axis=1), edges),
    }


def add_sample_constraint(
    displacement: np.ndarray,
    counts: np.ndarray,
    sample_vertex_ids: np.ndarray,
    required: np.ndarray,
    directions: np.ndarray,
) -> int:
    conflict = required > 0.0
    if not np.any(conflict):
        return 0
    surface_fix.add_conflict_displacement(
        displacement,
        counts,
        sample_vertex_ids[conflict],
        required[conflict],
        directions[conflict],
    )
    return int(np.count_nonzero(conflict))


def rv_inside_rvepi_summary(
    rv: pv.PolyData,
    rvepi: pv.PolyData,
    min_distance: float,
    intersection_tolerance: float,
) -> Tuple[Dict[str, object], pv.PolyData]:
    curve = surface_fix.intersection_curve(rv, rvepi, intersection_tolerance)
    samples = sample_sets(rv)
    summary: Dict[str, object] = {
        "pair": "RV_inside_RVepi",
        "required_min_distance_mm": float(min_distance),
        "direct_intersection_points": int(curve.n_points),
        "direct_intersection_cells": int(curve.n_cells),
        "direct_intersection_length": float(curve.length) if curve.n_points else 0.0,
    }

    for name, (sample_points, _) in samples.items():
        distances, _ = surface_fix.signed_distance_and_gradient(rvepi, sample_points)
        inside = distances < 0.0
        inward_distance = -distances
        too_close = distances > -float(min_distance)
        summary[f"{name}_count"] = int(len(sample_points))
        summary[f"{name}_outside_RVepi"] = int(np.count_nonzero(~inside))
        summary[f"{name}_inside_RVepi"] = int(np.count_nonzero(inside))
        summary[f"{name}_less_than_min_distance"] = int(np.count_nonzero(too_close))
        summary[f"{name}_min_signed_distance_to_RVepi"] = float(distances.min()) if len(distances) else None
        summary[f"{name}_max_signed_distance_to_RVepi"] = float(distances.max()) if len(distances) else None
        summary[f"{name}_min_inward_distance_to_RVepi"] = float(inward_distance.min()) if len(inward_distance) else None
        summary[f"{name}_p05_inward_distance_to_RVepi"] = float(np.percentile(inward_distance, 5)) if len(inward_distance) else None

    summary["passes_rule"] = bool(
        summary["direct_intersection_points"] == 0
        and summary["direct_intersection_cells"] == 0
        and summary["vertices_outside_RVepi"] == 0
        and summary["vertices_less_than_min_distance"] == 0
        and summary["face_centers_less_than_min_distance"] == 0
        and summary["edge_midpoints_less_than_min_distance"] == 0
    )
    return summary, curve


def lvepi_clearance_summary(
    moving_name: str,
    moving: pv.PolyData,
    lv_epi: pv.PolyData,
    clearance: float,
    intersection_tolerance: float,
) -> Tuple[Dict[str, object], pv.PolyData]:
    base_summary, curve, _, _ = surface_fix.intersection_summary(
        moving_name,
        moving,
        "LVepi",
        lv_epi,
        close_distance=clearance,
        intersection_tolerance=intersection_tolerance,
    )
    samples = sample_sets(moving)
    for name, (sample_points, _) in samples.items():
        distances, _ = surface_fix.signed_distance_and_gradient(lv_epi, sample_points)
        base_summary[f"{name}_below_clearance_to_LVepi"] = int(np.count_nonzero(distances < float(clearance)))
        base_summary[f"{name}_min_signed_distance_to_LVepi"] = float(distances.min()) if len(distances) else None
    base_summary["passes_rule"] = bool(
        base_summary["intersection_points"] == 0
        and base_summary["intersection_cells"] == 0
        and base_summary[f"{moving_name}_points_inside_LVepi"] == 0
        and base_summary["vertices_below_clearance_to_LVepi"] == 0
        and base_summary["face_centers_below_clearance_to_LVepi"] == 0
        and base_summary["edge_midpoints_below_clearance_to_LVepi"] == 0
    )
    return base_summary, curve


def fast_lvepi_clearance_summary(
    moving_name: str,
    moving: pv.PolyData,
    lv_epi: pv.PolyData,
    clearance: float,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "pair": f"{moving_name}_vs_LVepi",
        "direct_intersection_checked": False,
    }
    samples = sample_sets(moving)
    for name, (sample_points, _) in samples.items():
        distances, _ = surface_fix.signed_distance_and_gradient(lv_epi, sample_points)
        below = distances < float(clearance)
        inside = distances < 0.0
        summary[f"{name}_count"] = int(len(sample_points))
        summary[f"{name}_inside_LVepi"] = int(np.count_nonzero(inside))
        summary[f"{name}_below_clearance_to_LVepi"] = int(np.count_nonzero(below))
        summary[f"{name}_min_signed_distance_to_LVepi"] = float(distances.min()) if len(distances) else None
    summary["passes_rule"] = bool(
        summary["vertices_below_clearance_to_LVepi"] == 0
        and summary["face_centers_below_clearance_to_LVepi"] == 0
        and summary["edge_midpoints_below_clearance_to_LVepi"] == 0
    )
    return summary


def fast_rv_inside_rvepi_summary(
    rv: pv.PolyData,
    rvepi: pv.PolyData,
    min_distance: float,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "pair": "RV_inside_RVepi",
        "required_min_distance_mm": float(min_distance),
        "direct_intersection_checked": False,
    }
    for name, (sample_points, _) in sample_sets(rv).items():
        distances, _ = surface_fix.signed_distance_and_gradient(rvepi, sample_points)
        inside = distances < 0.0
        inward_distance = -distances
        too_close = distances > -float(min_distance)
        summary[f"{name}_count"] = int(len(sample_points))
        summary[f"{name}_outside_RVepi"] = int(np.count_nonzero(~inside))
        summary[f"{name}_less_than_min_distance"] = int(np.count_nonzero(too_close))
        summary[f"{name}_min_inward_distance_to_RVepi"] = float(inward_distance.min()) if len(inward_distance) else None
        summary[f"{name}_p05_inward_distance_to_RVepi"] = float(np.percentile(inward_distance, 5)) if len(inward_distance) else None
    summary["passes_rule"] = bool(
        summary["vertices_outside_RVepi"] == 0
        and summary["vertices_less_than_min_distance"] == 0
        and summary["face_centers_less_than_min_distance"] == 0
        and summary["edge_midpoints_less_than_min_distance"] == 0
    )
    return summary


def correction_iteration(
    rv: pv.PolyData,
    lv_epi: pv.PolyData,
    rvepi: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    points = np.asarray(rv.points, dtype=float).copy()
    triangles = surface_fix.triangle_faces(rv)
    displacement = np.zeros_like(points)
    counts = np.zeros(len(points), dtype=float)

    conflict_counts = {}
    for sample_name, (sample_points, sample_vertex_ids) in sample_sets(rv).items():
        lv_distances, lv_gradients = surface_fix.signed_distance_and_gradient(lv_epi, sample_points)
        lv_required = np.maximum(float(args.lv_clearance) - lv_distances, 0.0)
        conflict_counts[f"{sample_name}_lv_clearance_conflicts"] = add_sample_constraint(
            displacement,
            counts,
            sample_vertex_ids,
            lv_required,
            lv_gradients,
        )

        rvepi_distances, rvepi_gradients = surface_fix.signed_distance_and_gradient(rvepi, sample_points)
        # Negative signed distance is inside RVepi. The RV surface must be at
        # least args.rv_min_distance_to_rvepi mm inward from RVepi.
        target_rvepi_distance = float(args.rv_min_distance_to_rvepi) + float(args.rv_target_distance_padding)
        rvepi_required = np.maximum(rvepi_distances + target_rvepi_distance, 0.0)
        conflict_counts[f"{sample_name}_rvepi_distance_conflicts"] = add_sample_constraint(
            displacement,
            counts,
            sample_vertex_ids,
            rvepi_required,
            -rvepi_gradients,
        )

    source_mask = counts > 0.0
    displacement[source_mask] /= counts[source_mask, None]
    displacement *= float(args.relaxation)

    lengths = np.linalg.norm(displacement, axis=1)
    too_large = lengths > float(args.max_step)
    if np.any(too_large):
        displacement[too_large] *= (float(args.max_step) / lengths[too_large])[:, None]

    displacement, moving_mask = surface_fix.blend_displacement(
        points,
        triangles,
        displacement,
        source_mask,
        rings=args.blend_rings,
        falloff=args.blend_falloff,
    )
    displacement = surface_fix.smooth_displacement(
        triangles,
        displacement,
        moving_mask,
        iterations=args.displacement_smooth_iterations,
        relaxation=args.displacement_smooth_relaxation,
    )

    corrected_points = points + displacement

    # Hard projection at vertices after smoothing prevents the blend from
    # leaving obvious point-wise violations behind.
    rvepi_distances, rvepi_gradients = surface_fix.signed_distance_and_gradient(rvepi, corrected_points)
    target_rvepi_distance = float(args.rv_min_distance_to_rvepi) + float(args.rv_target_distance_padding)
    rvepi_required = np.maximum(rvepi_distances + target_rvepi_distance, 0.0)
    rvepi_conflict = rvepi_required > 0.0
    if np.any(rvepi_conflict):
        corrected_points[rvepi_conflict] -= rvepi_gradients[rvepi_conflict] * rvepi_required[rvepi_conflict, None]
        moving_mask[rvepi_conflict] = True

    lv_distances, lv_gradients = surface_fix.signed_distance_and_gradient(lv_epi, corrected_points)
    lv_required = np.maximum(float(args.lv_clearance) - lv_distances, 0.0)
    lv_conflict = lv_required > 0.0
    if np.any(lv_conflict):
        corrected_points[lv_conflict] += lv_gradients[lv_conflict] * lv_required[lv_conflict, None]
        moving_mask[lv_conflict] = True

    step = np.linalg.norm(corrected_points - points, axis=1)
    corrected = surface_fix.make_surface(corrected_points, triangles, tolerance=args.clean_tolerance)
    corrected = surface_fix.orient_surface(corrected, tolerance=args.clean_tolerance)
    info = {
        **conflict_counts,
        "source_vertices": int(np.count_nonzero(source_mask)),
        "moved_vertices": int(np.count_nonzero(moving_mask)),
        "hard_projected_to_RVepi_vertices": int(np.count_nonzero(rvepi_conflict)),
        "hard_projected_to_LVepi_vertices": int(np.count_nonzero(lv_conflict)),
        "max_step": float(step.max()) if len(step) else 0.0,
        "mean_step_on_moved_vertices": float(step[moving_mask].mean()) if np.any(moving_mask) else 0.0,
    }
    return corrected, info


def fix_rv_against_lv_and_rvepi(
    rv: pv.PolyData,
    lv_epi: pv.PolyData,
    rvepi: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    current = rv
    initial_lvepi = fast_lvepi_clearance_summary("RV", current, lv_epi, args.lv_clearance)
    initial_rvepi = fast_rv_inside_rvepi_summary(current, rvepi, args.rv_min_distance_to_rvepi)
    iterations: List[Dict[str, object]] = []

    for index in range(1, args.max_iterations + 1):
        current, step_info = correction_iteration(current, lv_epi, rvepi, args)
        lv_summary = fast_lvepi_clearance_summary("RV", current, lv_epi, args.lv_clearance)
        rvepi_summary = fast_rv_inside_rvepi_summary(current, rvepi, args.rv_min_distance_to_rvepi)
        step_info["iteration"] = int(index)
        step_info["after_LVepi_rule"] = lv_summary
        step_info["after_RVepi_rule"] = rvepi_summary
        iterations.append(step_info)
        if lv_summary["passes_rule"] and rvepi_summary["passes_rule"]:
            break

    final_lvepi, _ = lvepi_clearance_summary("RV", current, lv_epi, args.lv_clearance, args.intersection_tolerance)
    final_rvepi, _ = rv_inside_rvepi_summary(current, rvepi, args.rv_min_distance_to_rvepi, args.intersection_tolerance)
    return current, {
        "method": "multi_constraint_signed_distance_projection",
        "moving_surface": "RV",
        "protected_surfaces": {
            "LVepi": f"RV must stay outside LVepi by at least {args.lv_clearance} mm",
            "RVepi": f"RV must validate inside RVepi by at least {args.rv_min_distance_to_rvepi} mm; correction target includes {args.rv_target_distance_padding} mm padding",
        },
        "initial": {
            "LVepi_rule": initial_lvepi,
            "RVepi_rule": initial_rvepi,
        },
        "final": {
            "LVepi_rule": final_lvepi,
            "RVepi_rule": final_rvepi,
        },
        "iterations": iterations,
    }


def write_outputs(
    output_dir: Path,
    frame: str,
    surfaces: Dict[str, pv.PolyData],
) -> Dict[str, str]:
    output_paths = {
        name: output_dir / f"{name}_{frame}.vtk"
        for name in ("LVendo", "LVepi", "RV", "RVepi")
    }
    for name, mesh in surfaces.items():
        surface_fix.write_custom_surface_vtk(
            output_paths[name],
            np.asarray(mesh.points),
            surface_fix.triangle_faces(mesh),
            surface_fix.SURFACE_TAGS[name],
        )
    return {name: str(path.resolve()) for name, path in output_paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lv-clearance", type=float, default=0.35)
    parser.add_argument("--rv-min-distance-to-rvepi", type=float, default=3.0)
    parser.add_argument("--rv-target-distance-padding", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=20)
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
    args.clearance = args.lv_clearance
    subject_dir = args.subject_dir
    frame = args.frame
    input_dir = subject_dir / "vtks"
    output_dir = args.output_dir or subject_dir / "vtk_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "LVendo": input_dir / f"LVendo_{frame}.vtk",
        "LVepi": input_dir / f"LVepi_{frame}.vtk",
        "RV": input_dir / f"RV_{frame}.vtk",
        "RVepi": input_dir / f"RVepi_{frame}.vtk",
    }
    original = {
        name: surface_fix.load_surface(path, args.clean_tolerance)
        for name, path in input_paths.items()
    }

    before_checks = {}
    if not args.skip_debug_files:
        before_checks["LVepi_vs_RV"] = surface_fix.save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_before", "LVepi", original["LVepi"], "RV", original["RV"], args
        )
        before_checks["LVepi_vs_RVepi"] = surface_fix.save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_before", "LVepi", original["LVepi"], "RVepi", original["RVepi"], args
        )
        before_checks["RV_vs_RVepi_direct"] = surface_fix.save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_before", "RV", original["RV"], "RVepi", original["RVepi"], args
        )

    fixed_rvepi, rvepi_fix = surface_fix.fix_surface_against_lv_epi("RVepi", original["RVepi"], original["LVepi"], args)
    fixed_rv_seed, rv_lvepi_first_fix = surface_fix.fix_surface_against_lv_epi("RV", original["RV"], original["LVepi"], args)
    fixed_rv, rv_multi_fix = fix_rv_against_lv_and_rvepi(fixed_rv_seed, original["LVepi"], fixed_rvepi, args)

    fixed = {
        "LVendo": original["LVendo"],
        "LVepi": original["LVepi"],
        "RV": fixed_rv,
        "RVepi": fixed_rvepi,
    }
    output_paths = write_outputs(output_dir, frame, fixed)
    surface_fix.write_tag_legend(output_dir / f"{subject_dir.name}_{frame}_surface_elemTag_legend.csv")

    after_checks = {}
    if not args.skip_debug_files:
        after_checks["LVepi_vs_RV"] = surface_fix.save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_after", "LVepi", fixed["LVepi"], "RV", fixed["RV"], args
        )
        after_checks["LVepi_vs_RVepi"] = surface_fix.save_debug_pair(
            debug_dir, f"{subject_dir.name}_{frame}_after", "LVepi", fixed["LVepi"], "RVepi", fixed["RVepi"], args
        )
        rv_rvepi_summary, rv_rvepi_curve = rv_inside_rvepi_summary(
            fixed["RV"],
            fixed["RVepi"],
            args.rv_min_distance_to_rvepi,
            args.intersection_tolerance,
        )
        after_checks["RV_inside_RVepi_min_distance"] = rv_rvepi_summary
        if rv_rvepi_curve.n_points:
            curve_path = debug_dir / f"{subject_dir.name}_{frame}_after_RV_RVepi_intersection_curve.vtk"
            rv_rvepi_curve.save(curve_path)
            after_checks["RV_inside_RVepi_min_distance"]["intersection_curve_vtk"] = str(curve_path.resolve())

    outputs = {
        name: surface_fix.surface_summary(fixed[name], surface_fix.SURFACE_TAGS[name], Path(output_paths[name]))
        for name in ("LVendo", "LVepi", "RV", "RVepi")
    }
    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
        "output_dir": str(output_dir.resolve()),
        "outputs": outputs,
        "elemTags": surface_fix.SURFACE_TAGS,
        "method": {
            "fixed_surfaces": "LVendo and LVepi are copied without geometric movement.",
            "first_stage": "RVepi is fixed against LVepi. RV is also first moved outside LVepi.",
            "second_stage": "Fixed RVepi is treated as the packaging shell. RV is further modified so it validates inside RVepi with at least 3 mm inward distance while maintaining LVepi clearance; correction uses a small target padding for numerical robustness.",
            "output_format": "Legacy ASCII UNSTRUCTURED_GRID triangle surfaces with CELL_DATA scalar elemTag, matching custom_vtk.py style.",
        },
        "fixes": {
            "RVepi_against_LVepi": rvepi_fix,
            "RV_initial_against_LVepi": rv_lvepi_first_fix,
            "RV_against_LVepi_and_RVepi": rv_multi_fix,
        },
        "before_checks": before_checks,
        "after_checks": after_checks,
    }
    summary_path = output_dir / f"{subject_dir.name}_{frame}_surface_intersection_fix_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    rv_final_lvepi = rv_multi_fix["final"]["LVepi_rule"]
    rv_final_rvepi = rv_multi_fix["final"]["RVepi_rule"]
    print(json.dumps({
        "summary": str(summary_path),
        "outputs": output_paths,
        "elemTags": surface_fix.SURFACE_TAGS,
        "RVepi_final_against_LVepi": rvepi_fix["final"],
        "RV_final_against_LVepi": rv_final_lvepi,
        "RV_final_inside_RVepi": rv_final_rvepi,
    }, indent=2))


if __name__ == "__main__":
    main()
