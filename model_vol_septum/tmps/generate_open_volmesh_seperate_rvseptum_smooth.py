#!/usr/bin/env python3
"""Generate separate LV/RV open-base meshes with RV septal pre-smoothing.

This is a new variant of generate_open_volmesh_seperate.py. It keeps the LV
workflow unchanged, but smooths the RV myocardial septal-side shell after
open-base shell construction and before TetGen.

The RV smoothing is local and thickness-protected:
  1. LVepi is used as the septal reference surface.
  2. RV/RVepi vertices nearest to LVepi define the septal smoothing core.
  3. Mesh-neighbor rings around the core receive tapered smoothing weights.
  4. The RV myocardial shell is locally blended toward a Taubin-smoothed shell target.
  5. Moved shell vertices are clamped against the opposite RV wall surface so
     nearest RVendo/RVepi distance does not expand beyond its original value
     plus a small tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MESH_VOL_DIR = REPO_ROOT / "mesh_vol"
CODE_EXPLAIN_TMPS_DIR = REPO_ROOT / "code_explain" / "tmps"
for path in (REPO_ROOT, MESH_VOL_DIR, CODE_EXPLAIN_TMPS_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_open_volmesh_seperate as separate


openbase = separate.openbase
base = separate.base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--frames", default=None, help="Comma-separated frames to process, e.g. ED,ES.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--vtk-dir", type=Path, default=None, help="Surface input folder; defaults to SUBJECT_DIR/vtks.")
    parser.add_argument("--save-debug-surfaces", action="store_true")
    parser.add_argument("--subject-base-side", choices=["auto", "low_z", "high_z", "off"], default="auto")
    parser.add_argument("--subject-base-frames", default="ED,ES")
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
    parser.add_argument("--lv-base-side", choices=["low_z", "high_z"], default=None)
    parser.add_argument("--rv-base-side", choices=["low_z", "high_z"], default=None)

    parser.add_argument("--disable-rv-septum-smoothing", action="store_true")
    parser.add_argument("--rv-septum-distance-mm", type=float, default=None)
    parser.add_argument("--rv-septum-core-fraction", type=float, default=0.05)
    parser.add_argument("--rv-septum-blend-rings", type=int, default=12)
    parser.add_argument("--rv-septum-smooth-iterations", type=int, default=30)
    parser.add_argument("--rv-septum-smooth-relaxation", type=float, default=0.15)
    parser.add_argument("--rv-septum-taubin-pass-band", type=float, default=0.12)
    parser.add_argument("--rv-septum-max-step-mm", type=float, default=None)
    parser.add_argument("--rv-septum-max-step-edge-multiplier", type=float, default=0.35)
    parser.add_argument("--rv-septum-max-expansion-mm", type=float, default=0.05)
    parser.add_argument("--rv-septum-min-thickness-mm", type=float, default=0.0)
    parser.add_argument("--rv-septum-distance-binary-steps", type=int, default=12)
    return parser.parse_args()


def nearest_distances_and_indices(query: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(reference).query(query)
        return distances.astype(float), indices.astype(np.int64)
    except Exception:
        distances: List[np.ndarray] = []
        indices: List[np.ndarray] = []
        for start in range(0, len(query), 512):
            chunk = query[start:start + 512]
            dist2 = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(dist2, axis=1)
            indices.append(idx.astype(np.int64))
            distances.append(np.sqrt(dist2[np.arange(len(chunk)), idx]))
        return np.concatenate(distances), np.concatenate(indices)


def median_surface_edge_length(mesh: pv.PolyData) -> float:
    triangles = base.triangle_faces(mesh)
    points = np.asarray(mesh.points, dtype=float)
    edges = set()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add(tuple(sorted((int(a), int(b)))))
    if not edges:
        return 1.0
    edge_array = np.asarray(sorted(edges), dtype=np.int64)
    lengths = np.linalg.norm(points[edge_array[:, 0]] - points[edge_array[:, 1]], axis=1)
    return float(np.median(lengths))


def laplacian_displacement(points: np.ndarray, neighbors: Sequence[set]) -> np.ndarray:
    displacement = np.zeros_like(points, dtype=float)
    for idx, linked in enumerate(neighbors):
        if linked:
            neighbor_ids = np.fromiter(linked, dtype=np.int64)
            displacement[idx] = points[neighbor_ids].mean(axis=0) - points[idx]
    return displacement


def septal_weights(mesh: pv.PolyData, lv_epi: pv.PolyData, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object]]:
    points = np.asarray(mesh.points, dtype=float)
    lv_points = np.asarray(lv_epi.points, dtype=float)
    distances, _ = nearest_distances_and_indices(points, lv_points)
    if args.rv_septum_distance_mm is None:
        fraction = min(max(float(args.rv_septum_core_fraction), 0.01), 0.95)
        threshold = float(np.percentile(distances, 100.0 * fraction))
        threshold_rule = f"distance percentile from --rv-septum-core-fraction={fraction:g}"
    else:
        threshold = float(args.rv_septum_distance_mm)
        threshold_rule = "manual --rv-septum-distance-mm"

    core = distances <= threshold
    weights = np.zeros(len(points), dtype=float)
    weights[core] = 1.0
    neighbors = openbase.vertex_neighbors(mesh)
    visited = set(int(v) for v in np.flatnonzero(core))
    frontier = set(visited)
    for ring in range(1, max(0, int(args.rv_septum_blend_rings)) + 1):
        next_frontier = set()
        for vertex_id in frontier:
            for neighbor in neighbors[vertex_id]:
                if neighbor not in visited:
                    next_frontier.add(int(neighbor))
        if not next_frontier:
            break
        weight = float(args.rv_septum_blend_rings - ring + 1) / float(args.rv_septum_blend_rings + 1)
        ids = np.fromiter(sorted(next_frontier), dtype=np.int64)
        weights[ids] = np.maximum(weights[ids], weight)
        visited.update(next_frontier)
        frontier = next_frontier

    return weights, {
        "threshold_mm": threshold,
        "threshold_rule": threshold_rule,
        "core_vertices": int(np.count_nonzero(core)),
        "weighted_vertices": int(np.count_nonzero(weights > 0.0)),
        "mean_weight": float(weights[weights > 0.0].mean()) if np.any(weights > 0.0) else 0.0,
        "distance_to_lv_epi": {
            "min": float(distances.min()),
            "p10": float(np.percentile(distances, 10)),
            "p50": float(np.percentile(distances, 50)),
            "p90": float(np.percentile(distances, 90)),
            "max": float(distances.max()),
        },
    }


def one_sided_distance_summary(query: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    distances, _ = nearest_distances_and_indices(query, reference)
    return {
        "min": float(distances.min()),
        "p05": float(np.percentile(distances, 5)),
        "p50": float(np.percentile(distances, 50)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(distances.max()),
    }


def clamp_points_to_distance_bounds(
    original_points: np.ndarray,
    proposed_points: np.ndarray,
    opposite_points: np.ndarray,
    original_distances: np.ndarray,
    active_mask: np.ndarray,
    min_thickness: float,
    max_expansion: float,
    binary_steps: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if not np.any(active_mask):
        return proposed_points, {"checked_vertices": 0, "clamped_vertices": 0}

    active_ids = np.flatnonzero(active_mask)
    lower = np.where(
        original_distances[active_ids] >= float(min_thickness),
        float(min_thickness),
        original_distances[active_ids],
    )
    upper = original_distances[active_ids] + float(max_expansion)
    distances, _ = nearest_distances_and_indices(proposed_points[active_ids], opposite_points)
    bad = (distances < lower) | (distances > upper)
    if not np.any(bad):
        return proposed_points, {"checked_vertices": int(len(active_ids)), "clamped_vertices": 0}

    bad_ids = active_ids[bad]
    bad_lower = lower[bad]
    bad_upper = upper[bad]
    original = original_points[bad_ids]
    displacement = proposed_points[bad_ids] - original
    best_alpha = np.zeros(len(bad_ids), dtype=float)
    lo = np.zeros(len(bad_ids), dtype=float)
    hi = np.ones(len(bad_ids), dtype=float)
    for _ in range(max(1, int(binary_steps))):
        mid = 0.5 * (lo + hi)
        candidate = original + displacement * mid[:, None]
        candidate_distances, _ = nearest_distances_and_indices(candidate, opposite_points)
        ok = (candidate_distances >= bad_lower) & (candidate_distances <= bad_upper)
        best_alpha[ok] = mid[ok]
        lo[ok] = mid[ok]
        hi[~ok] = mid[~ok]

    clamped = proposed_points.copy()
    clamped[bad_ids] = original + displacement * best_alpha[:, None]
    return clamped, {"checked_vertices": int(len(active_ids)), "clamped_vertices": int(len(bad_ids))}


def weighted_mesh_copy(mesh: pv.PolyData, points: np.ndarray, weights: np.ndarray, displacement: np.ndarray) -> pv.PolyData:
    out = mesh.copy(deep=True)
    out.points = np.asarray(points, dtype=float)
    out.point_data["rvSeptalSmoothWeight"] = np.asarray(weights, dtype=float)
    out.point_data["rvSeptalSmoothDisplacement"] = np.linalg.norm(np.asarray(displacement, dtype=float), axis=1)
    return out.extract_surface().triangulate().clean(tolerance=1e-7)


def limit_displacement(displacement: np.ndarray, max_distance: float) -> np.ndarray:
    out = np.asarray(displacement, dtype=float).copy()
    norms = np.linalg.norm(out, axis=1)
    too_large = norms > float(max_distance)
    if np.any(too_large):
        out[too_large] *= (float(max_distance) / norms[too_large])[:, None]
    return out


def taubin_target_points(mesh: pv.PolyData, args: argparse.Namespace) -> np.ndarray:
    smoothed = mesh.smooth_taubin(
        n_iter=int(args.rv_septum_smooth_iterations),
        pass_band=float(args.rv_septum_taubin_pass_band),
        boundary_smoothing=True,
        feature_smoothing=False,
        non_manifold_smoothing=True,
        normalize_coordinates=True,
    )
    return np.asarray(smoothed.points, dtype=float)


def smooth_rv_septal_pair(
    rv_endo: pv.PolyData,
    rv_epi: pv.PolyData,
    lv_epi: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, pv.PolyData, Dict[str, object]]:
    endo = rv_endo.extract_surface().triangulate().clean(tolerance=1e-7)
    epi = rv_epi.extract_surface().triangulate().clean(tolerance=1e-7)
    endo_original = np.asarray(endo.points, dtype=float).copy()
    epi_original = np.asarray(epi.points, dtype=float).copy()
    endo_weights, endo_weight_info = septal_weights(endo, lv_epi, args)
    epi_weights, epi_weight_info = septal_weights(epi, lv_epi, args)
    original_endo_to_epi, _ = nearest_distances_and_indices(endo_original, epi_original)
    original_epi_to_endo, _ = nearest_distances_and_indices(epi_original, endo_original)

    median_edge = float(np.median([median_surface_edge_length(endo), median_surface_edge_length(epi)]))
    if args.rv_septum_max_step_mm is None:
        max_step = median_edge * float(args.rv_septum_max_step_edge_multiplier)
    else:
        max_step = float(args.rv_septum_max_step_mm)

    strength = min(max(float(args.rv_septum_smooth_relaxation), 0.0), 1.0)
    endo_target = taubin_target_points(endo, args)
    epi_target = taubin_target_points(epi, args)
    endo_displacement = (endo_target - endo_original) * (strength * endo_weights[:, None])
    epi_displacement = (epi_target - epi_original) * (strength * epi_weights[:, None])
    endo_displacement = limit_displacement(endo_displacement, max_step)
    epi_displacement = limit_displacement(epi_displacement, max_step)

    endo_proposed = endo_original + endo_displacement
    epi_proposed = epi_original + epi_displacement
    endo_points, endo_clamp = clamp_points_to_distance_bounds(
        endo_original,
        endo_proposed,
        epi_proposed,
        original_endo_to_epi,
        endo_weights > 0.0,
        min_thickness=float(args.rv_septum_min_thickness_mm),
        max_expansion=float(args.rv_septum_max_expansion_mm),
        binary_steps=int(args.rv_septum_distance_binary_steps),
    )
    epi_points, epi_clamp = clamp_points_to_distance_bounds(
        epi_original,
        epi_proposed,
        endo_points,
        original_epi_to_endo,
        epi_weights > 0.0,
        min_thickness=float(args.rv_septum_min_thickness_mm),
        max_expansion=float(args.rv_septum_max_expansion_mm),
        binary_steps=int(args.rv_septum_distance_binary_steps),
    )

    endo_displacement = endo_points - endo_original
    epi_displacement = epi_points - epi_original
    smoothed_endo = weighted_mesh_copy(endo, endo_points, endo_weights, endo_displacement)
    smoothed_epi = weighted_mesh_copy(epi, epi_points, epi_weights, epi_displacement)
    endo_norms = np.linalg.norm(endo_displacement, axis=1)
    epi_norms = np.linalg.norm(epi_displacement, axis=1)
    info = {
        "enabled": True,
        "method": "local_taubin_target_blend_with_no_expansion_distance_clamp",
        "reference_surface": "LVepi",
        "endo_weighting": endo_weight_info,
        "epi_weighting": epi_weight_info,
        "taubin_iterations": int(args.rv_septum_smooth_iterations),
        "taubin_pass_band": float(args.rv_septum_taubin_pass_band),
        "local_blend_strength": float(strength),
        "median_surface_edge_length": float(median_edge),
        "max_step_mm": float(max_step),
        "max_expansion_mm": float(args.rv_septum_max_expansion_mm),
        "min_thickness_mm": float(args.rv_septum_min_thickness_mm),
        "clamped_vertices": {
            "endo": endo_clamp,
            "epi": epi_clamp,
        },
        "displacement_mm": {
            "endo_max": float(endo_norms.max()),
            "endo_p95": float(np.percentile(endo_norms, 95)),
            "epi_max": float(epi_norms.max()),
            "epi_p95": float(np.percentile(epi_norms, 95)),
        },
        "thickness_distance_before": {
            "endo_to_epi": one_sided_distance_summary(endo_original, epi_original),
            "epi_to_endo": one_sided_distance_summary(epi_original, endo_original),
        },
        "thickness_distance_after": {
            "endo_to_epi": one_sided_distance_summary(np.asarray(smoothed_endo.points), np.asarray(smoothed_epi.points)),
            "epi_to_endo": one_sided_distance_summary(np.asarray(smoothed_epi.points), np.asarray(smoothed_endo.points)),
        },
    }
    return smoothed_endo, smoothed_epi, info


def reorient_tetrahedra(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    out = np.asarray(cells, dtype=np.int64).copy()
    volumes = base.tet_volumes(points, out)
    negative = volumes < 0.0
    if np.any(negative):
        ids = np.flatnonzero(negative)
        first = out[ids, 0].copy()
        out[ids, 0] = out[ids, 1]
        out[ids, 1] = first
    after = base.tet_volumes(points, out)
    return out, {
        "negative_tets_reoriented": int(np.count_nonzero(negative)),
        "nonpositive_after_reorientation": int(np.count_nonzero(after <= 0.0)),
    }


def write_chamber_mesh(path: Path, points: np.ndarray, cells: np.ndarray, tag: int) -> Dict[str, object]:
    fixed_cells, orientation = reorient_tetrahedra(points, cells)
    elem_tags = np.full(len(fixed_cells), int(tag), dtype=np.int32)
    separate.write_custom_vtk(path, points, fixed_cells, elem_tags)
    quality = base.tet_quality(points, fixed_cells)
    volume = float(np.abs(base.tet_volumes(points, fixed_cells)).sum())
    return {
        "path": str(path.resolve()),
        "points": int(len(points)),
        "tetrahedra": int(len(fixed_cells)),
        "elemTag": int(tag),
        "abs_volume": volume,
        "tet_quality": quality,
        "orientation_fix": orientation,
    }


def smooth_rv_septal_shell(
    shell: pv.PolyData,
    endo_reference: pv.PolyData,
    epi_reference: pv.PolyData,
    lv_epi: pv.PolyData,
    args: argparse.Namespace,
) -> Tuple[pv.PolyData, Dict[str, object]]:
    shell = shell.extract_surface().triangulate().clean(tolerance=1e-7)
    original = np.asarray(shell.points, dtype=float).copy()
    weights, weight_info = septal_weights(shell, lv_epi, args)
    median_edge = median_surface_edge_length(shell)
    if args.rv_septum_max_step_mm is None:
        max_step = median_edge * float(args.rv_septum_max_step_edge_multiplier)
    else:
        max_step = float(args.rv_septum_max_step_mm)

    target = taubin_target_points(shell, args)
    strength = min(max(float(args.rv_septum_smooth_relaxation), 0.0), 1.0)
    displacement = (target - original) * (strength * weights[:, None])
    displacement = limit_displacement(displacement, max_step)
    proposed = original + displacement

    endo_points = np.asarray(endo_reference.points, dtype=float)
    epi_points = np.asarray(epi_reference.points, dtype=float)
    dist_to_endo, _ = nearest_distances_and_indices(original, endo_points)
    dist_to_epi, _ = nearest_distances_and_indices(original, epi_points)
    endo_like = dist_to_endo <= dist_to_epi
    active = weights > 0.0

    clamped_once, endo_clamp = clamp_points_to_distance_bounds(
        original,
        proposed,
        epi_points,
        dist_to_epi,
        active & endo_like,
        min_thickness=float(args.rv_septum_min_thickness_mm),
        max_expansion=float(args.rv_septum_max_expansion_mm),
        binary_steps=int(args.rv_septum_distance_binary_steps),
    )
    clamped, epi_clamp = clamp_points_to_distance_bounds(
        original,
        clamped_once,
        endo_points,
        dist_to_endo,
        active & ~endo_like,
        min_thickness=float(args.rv_septum_min_thickness_mm),
        max_expansion=float(args.rv_septum_max_expansion_mm),
        binary_steps=int(args.rv_septum_distance_binary_steps),
    )

    smoothed = weighted_mesh_copy(shell, clamped, weights, clamped - original)
    free_edges_before_repair = len(base.extract_boundary_edges(smoothed))
    repaired = base.repair_closed_surface(smoothed)
    free_edges_after_repair = len(base.extract_boundary_edges(repaired))
    disp_norm = np.linalg.norm(clamped - original, axis=1)
    return repaired, {
        "enabled": True,
        "method": "local_rv_myocardial_shell_taubin_blend_before_tetgen",
        "reference_surface": "LVepi",
        "weighting": weight_info,
        "taubin_iterations": int(args.rv_septum_smooth_iterations),
        "taubin_pass_band": float(args.rv_septum_taubin_pass_band),
        "local_blend_strength": float(strength),
        "median_shell_edge_length": float(median_edge),
        "max_step_mm": float(max_step),
        "max_expansion_mm": float(args.rv_septum_max_expansion_mm),
        "min_thickness_mm": float(args.rv_septum_min_thickness_mm),
        "classified_vertices": {
            "endo_like": int(np.count_nonzero(endo_like)),
            "epi_like": int(np.count_nonzero(~endo_like)),
        },
        "clamped_vertices": {
            "endo_like": endo_clamp,
            "epi_like": epi_clamp,
        },
        "displacement_mm": {
            "max": float(disp_norm.max()),
            "p95": float(np.percentile(disp_norm, 95)),
            "mean_weighted": float(disp_norm[active].mean()) if np.any(active) else 0.0,
        },
        "free_edges": {
            "before_repair": int(free_edges_before_repair),
            "after_repair": int(free_edges_after_repair),
        },
        "distance_to_opposite_wall_before": {
            "endo_like_to_epi": one_sided_distance_summary(original[endo_like], epi_points) if np.any(endo_like) else {},
            "epi_like_to_endo": one_sided_distance_summary(original[~endo_like], endo_points) if np.any(~endo_like) else {},
        },
        "distance_to_opposite_wall_after": {
            "endo_like_to_epi": one_sided_distance_summary(clamped[endo_like], epi_points) if np.any(endo_like) else {},
            "epi_like_to_endo": one_sided_distance_summary(clamped[~endo_like], endo_points) if np.any(~endo_like) else {},
        },
    }


def build_rv_shell_smooth_and_tetrahedralize_adaptive(
    name: str,
    endo: pv.PolyData,
    epi: pv.PolyData,
    lv_epi: pv.PolyData,
    profile: base.AreaProfile,
    args: argparse.Namespace,
) -> Tuple[openbase.VentricleBuild, np.ndarray, np.ndarray, np.ndarray, Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    failures: List[Dict[str, object]] = []
    for attempt_index, overrides in enumerate(openbase.unique_attempt_overrides(args), start=1):
        attempt_args = openbase.namespace_with_overrides(args, overrides)
        try:
            build = openbase.build_ventricle_shell(name, endo, epi, profile, attempt_args)
            if args.disable_rv_septum_smoothing:
                smoothing_info = {"enabled": False}
            else:
                build.shell, smoothing_info = smooth_rv_septal_shell(
                    build.shell,
                    build.endo_open,
                    build.epi_open,
                    lv_epi,
                    args,
                )
            seed = base.find_shell_seed(build.epi_open, build.endo_open, build.shell)
            points, cells = base.tetrahedralize_shell(build.shell, seed)
            metadata = {
                "attempt_index": attempt_index,
                "overrides": overrides,
                "seed": [float(x) for x in seed],
                "tetrahedra": int(len(cells)),
                "points": int(len(points)),
                "rv_septal_shell_smoothing": smoothing_info,
            }
            return build, seed, points, cells, metadata, failures, smoothing_info
        except Exception as exc:
            failures.append({
                "attempt_index": attempt_index,
                "overrides": overrides,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"{name} adaptive attempt {attempt_index} failed: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"{name}: all adaptive shell smoothing/tetrahedralization attempts failed. Last error: {failures[-1] if failures else 'none'}")


def process_frame(args: argparse.Namespace, frame: str) -> Dict[str, object]:
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    frame_args = openbase.namespace_with_overrides(args, {"frame": frame})
    vtk_dir = separate.resolve_vtk_dir(subject_dir, args.vtk_dir)
    output_dir = separate.resolve_output_dir(subject_dir, args.output_root or subject_dir / "volmesh_LVRV_rvseptum_shell_smooth")
    input_paths = separate.input_paths_for_frame(vtk_dir, frame)
    surfaces = separate.load_frame_surfaces(vtk_dir, frame)

    subject_base_side, subject_base_detection = openbase.detect_subject_base_side(vtk_dir, frame_args)
    lv_base_side = args.lv_base_side or subject_base_side
    rv_base_side = args.rv_base_side or subject_base_side
    subject_base_detection["applied_base_side"] = {"lv": lv_base_side, "rv": rv_base_side}
    subject_base_detection["explicit_chamber_overrides"] = {"lv": args.lv_base_side, "rv": args.rv_base_side}

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

    lv_build, lv_seed, lv_points, lv_cells, lv_attempt, lv_failures = openbase.build_and_tetrahedralize_adaptive(
        "LV", surfaces["lv_endo"], surfaces["lv_epi"], lv_profile, frame_args
    )
    rv_build, rv_seed, rv_points, rv_cells, rv_attempt, rv_failures, rv_smoothing_info = build_rv_shell_smooth_and_tetrahedralize_adaptive(
        "RV", surfaces["rv_endo"], surfaces["rv_epi"], surfaces["lv_epi"], rv_profile, frame_args
    )

    lv_path = output_dir / f"{subject_id}_{frame}_openbase_LV_volmesh.vtk"
    rv_path = output_dir / f"{subject_id}_{frame}_openbase_RV_volmesh.vtk"
    lv_mesh_summary = write_chamber_mesh(lv_path, lv_points, lv_cells, tag=1)
    rv_mesh_summary = write_chamber_mesh(rv_path, rv_points, rv_cells, tag=0)
    openbase.write_custom_elem_tag_legend(output_dir / f"{subject_id}_{frame}_separate_elemTag_legend.csv")

    if not args.disable_rv_septum_smoothing:
        rv_build.shell.save(output_dir / f"{subject_id}_{frame}_RV_myocardium_boundary_rvseptum_smoothed_for_tetgen.vtk")
    if args.save_debug_surfaces:
        separate.save_debug_surfaces(output_dir, subject_id, frame, lv_build, rv_build)

    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "vtk_dir": str(vtk_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "outputs": {
            "lv_volmesh": lv_mesh_summary,
            "rv_volmesh": rv_mesh_summary,
            "rv_smoothed_shell_for_tetgen": str((output_dir / f"{subject_id}_{frame}_RV_myocardium_boundary_rvseptum_smoothed_for_tetgen.vtk").resolve()),
        },
        "method": {
            "input_rule": "Read LVendo/LVepi/RV/RVepi surfaces from SUBJECT_DIR/vtks by default.",
            "rv_pre_tetgen_smoothing": "Build the RV open-base myocardial shell first, then smooth only the shell band nearest LVepi before TetGen. Shell vertices are classified as endo-like or epi-like and clamped against the opposite RV wall to avoid thickness expansion.",
            "opening": "Use the same open-base clipping, boundary smoothing, annulus construction, and adaptive TetGen flow as the existing generator.",
            "labels": "LV output cells are elemTag=1. RV output cells are elemTag=0.",
        },
        "parameters": separate.serializable_args(frame_args),
        "subject_base_detection": subject_base_detection,
        "rv_septum_smoothing": rv_smoothing_info,
        "ventricles": {
            "lv": openbase.ventricle_summary(lv_build),
            "rv": openbase.ventricle_summary(rv_build),
        },
        "adaptive_attempts": {
            "lv_selected": lv_attempt,
            "rv_selected": rv_attempt,
            "lv_failures": lv_failures,
            "rv_failures": rv_failures,
        },
        "tetgen_seeds": {
            "lv": [float(x) for x in lv_seed],
            "rv": [float(x) for x in rv_seed],
        },
    }
    summary_path = output_dir / f"{subject_id}_{frame}_openbase_rvseptum_shell_smooth_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "frame": frame,
        "lv_volmesh": str(lv_path),
        "rv_volmesh": str(rv_path),
        "summary": str(summary_path),
        "lv_tetrahedra": int(len(lv_cells)),
        "rv_tetrahedra": int(len(rv_cells)),
        "lv_shift_distance": float(lv_build.intersection.shift_distance),
        "rv_shift_distance": float(rv_build.intersection.shift_distance),
        "rv_septum_smoothing": {
            "enabled": bool(rv_smoothing_info.get("enabled", False)),
            "weighted_vertices": int(rv_smoothing_info.get("weighting", {}).get("weighted_vertices", 0)),
        },
    }


def main() -> None:
    args = parse_args()
    frames = separate.parse_csv_values(args.frames) or [args.frame]
    results = [process_frame(args, frame) for frame in frames]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
