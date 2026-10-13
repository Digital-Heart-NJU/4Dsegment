#!/usr/bin/env python3
"""Generate an open-base biventricular myocardium mesh with custom LV/RV elemTag labels.

Pipeline:
  raw LV/RV endocardial + epicardial surfaces
  -> detect anatomical base side from Z-area profiles
  -> estimate basal directional myocardium thickness for each chamber/frame
  -> translate each endocardial blood-pool surface toward the subject-level base by a maximum-thickness-limited distance
  -> choose the largest valid basal shifted-endo/epi interface candidate up to the maximum basal thickness
  -> explicitly smooth the clipped basal rim while preserving opening area
  -> build a layered myocardium cut surface
  -> repair closed myocardial shells and tetrahedralize LV/RV myocardium
  -> save custom VTK elemTag labels: LV myocardium = 1, RV myocardium = 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv

from code_explain.tmps import generate_subject_open_base_zrule_volmesh as base, \
    generate_subject_open_base_bloodpool_cap_volmesh as cap_utils
from custom_vtk import write_custom_vtk


@dataclass
class ShiftCandidate:
    distance: float
    face_count: int
    area: float
    opening_loop_area: float
    boundary_loop_areas: List[float]
    boundary_loop_lengths: List[float]
    material_secondary_loop_count: int
    depth_from_base: float
    removed_z_min: float
    removed_z_max: float
    component_count: int
    boundary_loop_count: int
    boundary_loop_length: float
    accepted: bool


@dataclass
class ShiftIntersectionResult:
    shift_distance: float
    direction: np.ndarray
    moved_endo: pv.PolyData
    shifted_intersection_cap: pv.PolyData
    original_endo_intersection_cap: pv.PolyData
    original_endo_face_open: pv.PolyData
    boundary_loop: pv.PolyData
    candidate_history: List[ShiftCandidate]
    selected_candidate: ShiftCandidate
    selection_metadata: Dict[str, object]


@dataclass
class VentricleBuild:
    name: str
    profile: base.AreaProfile
    intersection: ShiftIntersectionResult
    endo_open: pv.PolyData
    epi_open: pv.PolyData
    endo_deleted_cap: pv.PolyData
    epi_deleted_cap: pv.PolyData
    endo_opening_loop: pv.PolyData
    epi_opening_loop: pv.PolyData
    endo_wall_boundary_loop: pv.PolyData
    epi_wall_boundary_loop: pv.PolyData
    original_control_loop: pv.PolyData
    shifted_control_loop: pv.PolyData
    endo_raw_wall_boundary_loop: pv.PolyData
    epi_raw_wall_boundary_loop: pv.PolyData
    endo_boundary_smoothing: Dict[str, object]
    epi_boundary_smoothing: Dict[str, object]
    annulus: pv.PolyData
    shell: pv.PolyData
    shell_free_edges_before_repair: int
    shell_free_edges_after_repair: int


REGION_LEGEND = base.REGION_LEGEND
CUSTOM_ELEMTAG_LEGEND = {1: "LV myocardium", 0: "RV myocardium"}


def parse_csv_values(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def detect_subject_base_side(
    vtk_dir: Path,
    args: argparse.Namespace,
    surface_loader: Optional[Callable[[Path], pv.PolyData]] = None,
) -> Tuple[Optional[str], Dict[str, object]]:
    """Choose one base side for the subject from the selected surface input folder."""
    mode = getattr(args, "subject_base_side", "auto")
    if mode == "off":
        return None, {
            "mode": "off",
            "source_vtk_dir": str(vtk_dir.resolve()),
            "selected_base_side": None,
            "reason": "Subject-level base side disabled; using per-frame/per-chamber detection unless explicit overrides are set.",
        }
    if mode in {"low_z", "high_z"}:
        return mode, {
            "mode": "manual",
            "source_vtk_dir": str(vtk_dir.resolve()),
            "selected_base_side": mode,
            "reason": "Manual subject-level base-side override.",
        }

    frames = parse_csv_values(getattr(args, "subject_base_frames", "ED,ES")) or [getattr(args, "frame", "ED")]
    chamber_files = [("lv", "LVendo"), ("rv", "RV")]
    records: List[Dict[str, object]] = []
    high_score = 0.0
    low_score = 0.0
    high_votes = 0
    low_votes = 0

    for scan_frame in frames:
        for chamber, prefix in chamber_files:
            mesh_path = vtk_dir / f"{prefix}_{scan_frame}.vtk"
            if not mesh_path.exists():
                records.append({
                    "frame": scan_frame,
                    "chamber": chamber,
                    "surface": prefix,
                    "path": str(mesh_path),
                    "available": False,
                })
                continue
            profile = base.compute_area_profile(
                (surface_loader or base.load_surface)(mesh_path),
                samples=args.samples,
                slab_fraction=args.slab_fraction,
                end_band_fraction=args.end_band_fraction,
                stable_fraction=args.stable_fraction,
            )
            low_area = float(profile.low_end_area)
            high_area = float(profile.high_end_area)
            denominator = max(low_area, high_area, 1e-6)
            signed_margin = (high_area - low_area) / denominator
            confidence = abs(signed_margin)
            auto_side = "high_z" if signed_margin >= 0.0 else "low_z"
            if auto_side == "high_z":
                high_votes += 1
                high_score += confidence
            else:
                low_votes += 1
                low_score += confidence
            records.append({
                "frame": scan_frame,
                "chamber": chamber,
                "surface": prefix,
                "path": str(mesh_path.resolve()),
                "available": True,
                "low_end_area": low_area,
                "high_end_area": high_area,
                "signed_high_minus_low_relative_margin": float(signed_margin),
                "confidence": float(confidence),
                "single_surface_base_side": auto_side,
            })

    available = [record for record in records if record.get("available")]
    if not available:
        raise RuntimeError(f"No LVendo/RV ED/ES surfaces found in {vtk_dir} for subject-level base detection.")

    if high_score == low_score:
        selected = "high_z" if high_votes >= low_votes else "low_z"
    else:
        selected = "high_z" if high_score > low_score else "low_z"

    return selected, {
        "mode": "auto",
        "source_vtk_dir": str(vtk_dir.resolve()),
        "frames": frames,
        "selected_base_side": selected,
        "high_score": float(high_score),
        "low_score": float(low_score),
        "high_votes": int(high_votes),
        "low_votes": int(low_votes),
        "records": records,
        "rule": "Score ED/ES LVendo and RV surfaces from --vtk-dir; use one base side for the whole subject. Each surface contributes abs(high-low)/max(high,low) to its voted side.",
    }


def custom_elem_tags(region_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(region_labels, dtype=np.int32)
    tags = np.zeros(len(labels), dtype=np.int32)
    tags[labels == 2] = 1
    tags[labels == 4] = 0
    unknown = sorted(set(int(value) for value in labels) - {2, 4})
    if unknown:
        raise ValueError(f"Unsupported region labels for custom elemTag mapping: {unknown}")
    return tags


def write_custom_elem_tag_legend(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elemTag", "description"])
        for tag, description in sorted(CUSTOM_ELEMTAG_LEGEND.items(), reverse=True):
            writer.writerow([tag, description])


def base_direction(profile: base.AreaProfile) -> np.ndarray:
    sign = 1.0 if profile.base_side == "high_z" else -1.0
    return np.array([0.0, 0.0, sign], dtype=float)


def translated_surface(mesh: pv.PolyData, direction: np.ndarray, distance: float) -> pv.PolyData:
    moved = mesh.copy(deep=True)
    moved.points = np.asarray(mesh.points) + direction[None, :] * float(distance)
    return moved.extract_surface().triangulate().clean(tolerance=1e-7)


def component_patch_for_shift(
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile: base.AreaProfile,
    distance: float,
    seed_band_fraction: float,
    max_secondary_loop_area_fraction: float,
) -> Optional[Tuple[np.ndarray, ShiftCandidate]]:
    triangles = base.triangle_faces(endo)
    points = np.asarray(endo.points)
    centers = points[triangles].mean(axis=1)
    direction = base_direction(profile)
    shifted_centers = centers + direction[None, :] * float(distance)

    inside_epi = base.enclosed_mask(shifted_centers, epi)
    candidate_mask = ~inside_epi
    if not np.any(candidate_mask):
        return None

    z_centers = centers[:, 2]
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    height = max(z_max - z_min, 1e-6)
    distance_from_base = z_max - z_centers if profile.base_side == "high_z" else z_centers - z_min
    seed_depth = max(seed_band_fraction * height, height / 80.0)
    seed_mask = distance_from_base <= seed_depth

    neighbors = base.face_adjacency(triangles)
    components = cap_utils.connected_components(candidate_mask, neighbors)
    seeded_components = [component for component in components if np.any(seed_mask[component])]
    usable_components = seeded_components or components
    if not usable_components:
        return None

    areas = cap_utils.face_areas(points, triangles)
    largest = max(usable_components, key=lambda component: float(areas[component].sum()))
    patch_mask = np.zeros(len(triangles), dtype=bool)
    patch_mask[largest] = True

    face_open = base.subset_mesh_faces(endo, ~patch_mask)
    loops = base.boundary_loops(face_open)
    loop_count = len(loops)
    loop_areas = [abs(base.signed_area_in_best_plane(np.asarray(loop.points))) for loop in loops]
    loop_lengths = [float(loop.length) for loop in loops]
    if loop_areas:
        dominant_index = int(np.argmax(loop_areas))
        opening_loop_area = float(loop_areas[dominant_index])
        loop_length = float(loop_lengths[dominant_index])
        secondary_limit = opening_loop_area * float(max_secondary_loop_area_fraction)
        material_secondary_loop_count = sum(
            1 for index, area in enumerate(loop_areas)
            if index != dominant_index and area > secondary_limit
        )
    else:
        opening_loop_area = 0.0
        loop_length = 0.0
        material_secondary_loop_count = 0
    candidate = ShiftCandidate(
        distance=float(distance),
        face_count=int(patch_mask.sum()),
        area=float(areas[patch_mask].sum()),
        opening_loop_area=float(opening_loop_area),
        boundary_loop_areas=[float(area) for area in loop_areas],
        boundary_loop_lengths=[float(length) for length in loop_lengths],
        material_secondary_loop_count=int(material_secondary_loop_count),
        depth_from_base=float(distance_from_base[patch_mask].max()),
        removed_z_min=float(z_centers[patch_mask].min()),
        removed_z_max=float(z_centers[patch_mask].max()),
        component_count=int(len(usable_components)),
        boundary_loop_count=int(loop_count),
        boundary_loop_length=loop_length,
        accepted=False,
    )
    return patch_mask, candidate


def parse_float_values(value: object) -> List[float]:
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).split(",")
    values: List[float] = []
    for item in raw:
        text = str(item).strip()
        if text:
            values.append(float(text))
    return values


def candidate_is_topologically_valid(candidate: ShiftCandidate, height: float, args: argparse.Namespace) -> bool:
    max_depth = height * float(args.max_intersection_depth_fraction)
    return bool(
        candidate.face_count >= int(args.min_intersection_faces)
        and candidate.depth_from_base <= max_depth
        and candidate.boundary_loop_count >= 1
        and candidate.boundary_loop_length > 0.0
        and candidate.opening_loop_area > 0.0
        and candidate.material_secondary_loop_count == 0
    )


def estimate_basal_directional_thickness(
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile: base.AreaProfile,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Estimate chamber-specific basal wall thickness along the selected base direction."""
    triangles = base.triangle_faces(endo)
    points = np.asarray(endo.points)
    centers = points[triangles].mean(axis=1)
    height = max(profile.z_max - profile.z_min, 1e-6)
    direction = base_direction(profile)

    if profile.base_side == "high_z":
        distance_from_base = profile.z_max - centers[:, 2]
    else:
        distance_from_base = centers[:, 2] - profile.z_min

    seed_depth = max(float(args.seed_band_fraction) * height, height / 80.0)
    basal_depth = min(
        height * float(args.max_intersection_depth_fraction),
        seed_depth * float(args.stable_thickness_seed_band_multiplier),
    )
    basal_mask = distance_from_base <= basal_depth
    min_basal_faces = max(
        int(args.min_intersection_faces),
        int(round(len(triangles) * float(args.stable_thickness_min_basal_face_fraction))),
    )
    if int(np.count_nonzero(basal_mask)) < min_basal_faces:
        order = np.argsort(distance_from_base)
        basal_mask = np.zeros(len(triangles), dtype=bool)
        basal_mask[order[: min(min_basal_faces, len(order))]] = True

    samples = centers[basal_mask]
    max_search = max(seed_depth, height * float(args.stable_thickness_max_search_fraction))
    steps = max(8, int(args.stable_thickness_search_steps))
    scan_distances = np.linspace(max_search / steps, max_search, steps)
    exit_distances = np.full(len(samples), np.nan, dtype=float)
    active = np.ones(len(samples), dtype=bool)

    for distance in scan_distances:
        if not np.any(active):
            break
        active_ids = np.flatnonzero(active)
        shifted = samples[active_ids] + direction[None, :] * float(distance)
        inside = base.enclosed_mask(shifted, epi)
        exited_ids = active_ids[~inside]
        if len(exited_ids):
            exit_distances[exited_ids] = float(distance)
            active[exited_ids] = False

    valid_exits = exit_distances[np.isfinite(exit_distances)]
    if len(valid_exits) == 0:
        raise RuntimeError("No basal directional endocardium-to-epicardium exits were found for stable thickness opening.")

    percentile_requests = sorted(set(
        [float(args.stable_thickness_percentile)]
        + parse_float_values(args.stable_thickness_candidate_percentiles)
        + [50.0, 80.0, 90.0, 95.0, 98.0, 100.0]
    ))
    percentiles = {
        f"p{percentile:g}": float(np.percentile(valid_exits, percentile))
        for percentile in percentile_requests
        if 0.0 <= percentile <= 100.0
    }
    target_distance = float(np.percentile(valid_exits, float(args.stable_thickness_percentile)))

    return {
        "method": "basal_directional_thickness",
        "base_side": profile.base_side,
        "direction": [float(v) for v in direction],
        "height": float(height),
        "seed_depth": float(seed_depth),
        "basal_depth": float(basal_depth),
        "basal_face_count": int(np.count_nonzero(basal_mask)),
        "exit_sample_count": int(len(samples)),
        "valid_exit_count": int(len(valid_exits)),
        "unresolved_exit_count": int(np.count_nonzero(~np.isfinite(exit_distances))),
        "max_search_distance": float(max_search),
        "search_steps": int(steps),
        "target_percentile": float(args.stable_thickness_percentile),
        "target_distance": float(target_distance),
        "percentile_distances": percentiles,
    }


def finalize_shift_intersection(
    endo: pv.PolyData,
    profile: base.AreaProfile,
    selected_mask: np.ndarray,
    selected_candidate: ShiftCandidate,
    history: List[ShiftCandidate],
    selection_metadata: Dict[str, object],
) -> ShiftIntersectionResult:
    direction = base_direction(profile)
    moved_endo = translated_surface(endo, direction, selected_candidate.distance)
    original_cap = base.subset_mesh_faces(endo, selected_mask)
    shifted_cap = translated_surface(original_cap, direction, selected_candidate.distance)
    face_open = base.subset_mesh_faces(endo, ~selected_mask)
    loops = base.boundary_loops(face_open)
    if not loops:
        raise RuntimeError("Selected shifted-interface candidate has no opening boundary loop.")
    loop_areas = [abs(base.signed_area_in_best_plane(np.asarray(loop.points))) for loop in loops]
    boundary_loop = loops[int(np.argmax(loop_areas))]

    return ShiftIntersectionResult(
        shift_distance=selected_candidate.distance,
        direction=direction,
        moved_endo=moved_endo,
        shifted_intersection_cap=shifted_cap,
        original_endo_intersection_cap=original_cap,
        original_endo_face_open=face_open,
        boundary_loop=boundary_loop,
        candidate_history=history,
        selected_candidate=selected_candidate,
        selection_metadata=selection_metadata,
    )


def find_stable_thickness_intersection(
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile: base.AreaProfile,
    args: argparse.Namespace,
) -> ShiftIntersectionResult:
    height = max(profile.z_max - profile.z_min, 1e-6)
    thickness = estimate_basal_directional_thickness(endo, epi, profile, args)
    percentile_distances = thickness["percentile_distances"]
    candidate_distances: List[float] = []
    for percentile in parse_float_values(args.stable_thickness_candidate_percentiles):
        key = f"p{percentile:g}"
        if key in percentile_distances:
            candidate_distances.append(float(percentile_distances[key]))
    candidate_distances.append(float(thickness["target_distance"]))

    max_search = float(thickness["max_search_distance"])
    max_basal_thickness = float(percentile_distances.get("p100", thickness["target_distance"]))
    lower_bound = min(
        float(percentile_distances.get("p90", thickness["target_distance"])),
        float(thickness["target_distance"]),
    )
    lower_bound = max(lower_bound, max_search / max(int(args.stable_thickness_search_steps), 1))
    upper_bound = max(lower_bound, max_basal_thickness)
    candidate_distances.append(lower_bound)
    candidate_distances.append(upper_bound)
    candidate_distances = sorted({
        round(max(0.0, min(float(distance), upper_bound)), 6)
        for distance in candidate_distances
        if distance > 0.0
    })

    history: List[ShiftCandidate] = []
    accepted: List[Tuple[np.ndarray, ShiftCandidate]] = []

    def test_distances(distances: Sequence[float]) -> None:
        tested = {round(candidate.distance, 6) for candidate in history}
        for distance in distances:
            rounded = round(float(distance), 6)
            if rounded in tested or rounded <= 0.0:
                continue
            result = component_patch_for_shift(
                endo,
                epi,
                profile,
                float(rounded),
                seed_band_fraction=args.seed_band_fraction,
                max_secondary_loop_area_fraction=args.max_secondary_loop_area_fraction,
            )
            if result is None:
                continue
            patch_mask, candidate = result
            candidate.accepted = candidate_is_topologically_valid(candidate, height, args)
            history.append(candidate)
            tested.add(rounded)
            if candidate.accepted:
                accepted.append((patch_mask, candidate))

    test_distances(candidate_distances)
    dense_fallback_used = False
    if not accepted:
        dense_fallback_used = True
        scan = np.linspace(lower_bound, upper_bound, max(int(args.shift_steps), 1))
        test_distances(scan)

    if not accepted:
        raise RuntimeError("Stable thickness opening did not find any topologically valid shifted-interface candidate.")

    accepted.sort(
        key=lambda item: (
            -item[1].distance,
            item[1].boundary_loop_count != 1,
            item[1].depth_from_base,
            -item[1].opening_loop_area,
        )
    )
    rank = min(max(int(getattr(args, "interface_candidate_rank", 0)), 0), len(accepted) - 1)
    selected_mask, selected_candidate = accepted[rank]
    selection_metadata = {
        "strategy": "stable_thickness",
        "selection_rule": "Use the P100 basal directional thickness as the upper displacement limit and choose the largest topologically valid shifted-endo/epi interface candidate; this avoids fixed distances while forcing thick basal myocardium regions to be included in the opening.",
        "rank": int(rank),
        "accepted_candidate_count": int(len(accepted)),
        "tested_candidate_count": int(len(history)),
        "dense_fallback_used": bool(dense_fallback_used),
        "lower_search_bound": float(lower_bound),
        "upper_search_bound": float(upper_bound),
        "max_basal_directional_thickness": float(max_basal_thickness),
        "accepted_distances": [float(candidate.distance) for _, candidate in accepted],
        "thickness": thickness,
    }
    return finalize_shift_intersection(endo, profile, selected_mask, selected_candidate, history, selection_metadata)


def find_auto_shift_intersection(
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile: base.AreaProfile,
    args: argparse.Namespace,
) -> ShiftIntersectionResult:
    height = max(profile.z_max - profile.z_min, 1e-6)
    min_shift = max(args.min_shift, height * args.min_shift_fraction)
    max_shift = max(min_shift, height * args.max_shift_fraction)
    target_area = max(args.min_intersection_area, endo.area * args.target_intersection_area_fraction)
    max_depth = height * args.max_intersection_depth_fraction

    distances = np.linspace(min_shift, max_shift, args.shift_steps)
    history: List[ShiftCandidate] = []
    accepted: List[Tuple[np.ndarray, ShiftCandidate]] = []
    fallback: List[Tuple[np.ndarray, ShiftCandidate]] = []

    for distance in distances:
        result = component_patch_for_shift(
            endo,
            epi,
            profile,
            float(distance),
            seed_band_fraction=args.seed_band_fraction,
            max_secondary_loop_area_fraction=args.max_secondary_loop_area_fraction,
        )
        if result is None:
            continue
        patch_mask, candidate = result
        candidate.accepted = (
            candidate.area >= target_area
            and candidate.face_count >= args.min_intersection_faces
            and candidate.depth_from_base <= max_depth
            and candidate.boundary_loop_count >= 1
            and candidate.boundary_loop_length > 0.0
            and candidate.opening_loop_area > 0.0
            and candidate.material_secondary_loop_count == 0
        )
        history.append(candidate)

        if candidate.depth_from_base <= max_depth and candidate.boundary_loop_length > 0.0:
            fallback.append((patch_mask, candidate))
        if candidate.accepted:
            accepted.append((patch_mask, candidate))

    ranked = accepted or fallback
    if not ranked:
        raise RuntimeError("Could not find a valid shifted endocardium/epicardium intersection region.")

    ranked = sorted(
        ranked,
        key=lambda item: (item[1].opening_loop_area, item[1].area, -item[1].depth_from_base),
        reverse=True,
    )
    rank = min(max(int(getattr(args, "interface_candidate_rank", 0)), 0), len(ranked) - 1)
    selected_mask, selected_candidate = ranked[rank]
    selection_metadata = {
        "strategy": "max_interface",
        "selection_rule": "Rank valid shifted-endo/epi interface candidates by opening-loop area, then area, then shallower depth.",
        "rank": int(rank),
        "accepted_candidate_count": int(len(accepted)),
        "fallback_candidate_count": int(len(fallback)),
        "tested_candidate_count": int(len(history)),
        "target_area": float(target_area),
        "min_shift": float(min_shift),
        "max_shift": float(max_shift),
    }
    return finalize_shift_intersection(endo, profile, selected_mask, selected_candidate, history, selection_metadata)


def smooth_control_loop(points: np.ndarray, sample_count: int, iterations: int, relaxation: float) -> np.ndarray:
    resampled = base.resample_closed_loop(np.asarray(points, dtype=float), sample_count)
    target_area = abs(base.signed_area_in_best_plane(resampled))
    smoothed = cap_utils.smooth_closed_loop(resampled, iterations=iterations, relaxation=relaxation)
    smoothed_area = abs(base.signed_area_in_best_plane(smoothed))
    if target_area > 1e-8 and smoothed_area > 1e-8:
        scale = float((target_area / smoothed_area) ** 0.5)
        centroid = smoothed.mean(axis=0)
        smoothed = centroid + (smoothed - centroid) * scale
    return smoothed


def closed_loop_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    closed = np.vstack([pts, pts[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def area_preserving_same_count_smooth(points: np.ndarray, iterations: int, relaxation: float) -> np.ndarray:
    pts = cap_utils.remove_duplicate_sequential_points(np.asarray(points, dtype=float))
    if len(pts) < 3 or iterations <= 0:
        return pts.copy()
    target_area = abs(base.signed_area_in_best_plane(pts))
    target_centroid = pts.mean(axis=0)
    smoothed = cap_utils.smooth_closed_loop(pts, iterations=iterations, relaxation=relaxation)
    smoothed_area = abs(base.signed_area_in_best_plane(smoothed))
    if target_area > 1e-8 and smoothed_area > 1e-8:
        scale = float((target_area / smoothed_area) ** 0.5)
        smoothed = target_centroid + (smoothed - smoothed.mean(axis=0)) * scale
    return smoothed


def boundary_vertex_loops(mesh: pv.PolyData) -> List[np.ndarray]:
    edges = base.extract_boundary_edges(mesh)
    if not edges:
        return []

    adjacency: Dict[int, List[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    unused = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    loops: List[np.ndarray] = []
    while unused:
        start, nxt = next(iter(unused))
        unused.remove((start, nxt))
        loop = [start, nxt]
        prev = start
        current = nxt
        for _ in range(len(edges) + 5):
            chosen = None
            for candidate in adjacency.get(current, []):
                if candidate == prev:
                    continue
                key = tuple(sorted((current, candidate)))
                if key in unused:
                    chosen = candidate
                    break
            if chosen is None:
                break
            unused.remove(tuple(sorted((current, chosen))))
            if chosen == start:
                break
            loop.append(chosen)
            prev, current = current, chosen
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))

    points = np.asarray(mesh.points)
    loops.sort(key=lambda ids: abs(base.signed_area_in_best_plane(points[ids])), reverse=True)
    return loops


def nearest_reference_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    try:
        from scipy.spatial import cKDTree

        return cKDTree(reference).query(query)[1].astype(np.int64)
    except Exception:
        indices: List[int] = []
        for start in range(0, len(query), 256):
            chunk = query[start:start + 256]
            dist2 = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            indices.extend(np.argmin(dist2, axis=1).astype(int).tolist())
        return np.asarray(indices, dtype=np.int64)


def vertex_neighbors(mesh: pv.PolyData) -> List[set]:
    triangles = base.triangle_faces(mesh)
    neighbors = [set() for _ in range(mesh.n_points)]
    for tri in triangles:
        a, b, c = (int(v) for v in tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def smooth_open_surface_boundary(
    mesh: pv.PolyData,
    iterations: int,
    relaxation: float,
    blend_rings: int,
    blend_relaxation: float,
) -> Tuple[pv.PolyData, pv.PolyData, Dict[str, object]]:
    mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    loops = boundary_vertex_loops(mesh)
    if not loops:
        raise RuntimeError("Cannot smooth opening boundary because the clipped surface has no boundary loop.")

    loop_ids = loops[0]
    points = np.asarray(mesh.points).copy()
    exact_loop = points[loop_ids].copy()
    smoothed_loop = area_preserving_same_count_smooth(exact_loop, iterations=iterations, relaxation=relaxation)
    displacement = smoothed_loop - exact_loop

    if blend_rings > 0 and blend_relaxation > 0.0:
        neighbors = vertex_neighbors(mesh)
        visited = set(int(v) for v in loop_ids)
        current = set(visited)
        for ring in range(1, blend_rings + 1):
            next_ring = set()
            for vertex_id in current:
                for nb in neighbors[vertex_id]:
                    if nb not in visited:
                        next_ring.add(int(nb))
            if not next_ring:
                break
            ring_ids = np.asarray(sorted(next_ring), dtype=np.int64)
            nearest = nearest_reference_indices(points[ring_ids], exact_loop)
            weight = float(blend_relaxation) * float(blend_rings - ring + 1) / float(blend_rings + 1)
            points[ring_ids] += displacement[nearest] * weight
            visited.update(next_ring)
            current = next_ring

    points[loop_ids] = smoothed_loop
    smoothed_mesh = mesh.copy(deep=True)
    smoothed_mesh.points = points
    smoothed_mesh = smoothed_mesh.extract_surface().triangulate().clean(tolerance=1e-7)

    info = {
        "boundary_points": int(len(loop_ids)),
        "boundary_loop_count": int(len(loops)),
        "area_before": float(abs(base.signed_area_in_best_plane(exact_loop))),
        "area_after": float(abs(base.signed_area_in_best_plane(smoothed_loop))),
        "length_before": closed_loop_length(exact_loop),
        "length_after": closed_loop_length(smoothed_loop),
        "max_boundary_displacement": float(np.linalg.norm(displacement, axis=1).max()),
        "mean_boundary_displacement": float(np.linalg.norm(displacement, axis=1).mean()),
        "blend_rings": int(blend_rings),
        "blend_relaxation": float(blend_relaxation),
        "iterations": int(iterations),
        "relaxation": float(relaxation),
    }
    return smoothed_mesh, base.make_polyline(smoothed_loop), info


def build_ventricle_shell(
    name: str,
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile: base.AreaProfile,
    args: argparse.Namespace,
) -> VentricleBuild:
    if args.opening_strategy == "stable_thickness":
        intersection = find_stable_thickness_intersection(endo, epi, profile, args)
    else:
        intersection = find_auto_shift_intersection(endo, epi, profile, args)

    original_control_points = smooth_control_loop(
        np.asarray(intersection.boundary_loop.points),
        sample_count=args.control_loop_points,
        iterations=args.control_smooth_iterations,
        relaxation=args.control_smooth_relaxation,
    )
    shifted_control_points = original_control_points + intersection.direction[None, :] * intersection.shift_distance

    endo_loop_points = cap_utils.nearest_surface_loop(original_control_points, endo)
    epi_loop_points = cap_utils.nearest_surface_loop(shifted_control_points, epi)

    endo_open_raw = cap_utils.clip_surface_by_loop(endo, endo_loop_points, keep_large_region=True)
    epi_open_raw = cap_utils.clip_surface_by_loop(epi, epi_loop_points, keep_large_region=True)
    endo_deleted_cap = cap_utils.clip_surface_by_loop(endo, endo_loop_points, keep_large_region=False)
    epi_deleted_cap = cap_utils.clip_surface_by_loop(epi, epi_loop_points, keep_large_region=False)

    endo_raw_boundary = base.select_boundary_loops(endo_open_raw)[0]
    epi_raw_boundary = base.select_boundary_loops(epi_open_raw)[0]
    endo_open, endo_smoothed_boundary, endo_boundary_smoothing = smooth_open_surface_boundary(
        endo_open_raw,
        iterations=args.boundary_smooth_iterations,
        relaxation=args.boundary_smooth_relaxation,
        blend_rings=args.boundary_blend_rings,
        blend_relaxation=args.boundary_blend_relaxation,
    )
    epi_open, epi_smoothed_boundary, epi_boundary_smoothing = smooth_open_surface_boundary(
        epi_open_raw,
        iterations=args.boundary_smooth_iterations,
        relaxation=args.boundary_smooth_relaxation,
        blend_rings=args.boundary_blend_rings,
        blend_relaxation=args.boundary_blend_relaxation,
    )
    endo_boundary = base.select_boundary_loops(endo_open)[0]
    epi_boundary = base.select_boundary_loops(epi_open)[0]
    annulus = cap_utils.smooth_myocardium_cut_surface(
        epi_boundary,
        endo_boundary,
        sample_count=args.annulus_loop_points,
        radial_layers=args.annulus_radial_layers,
        smooth_iterations=args.annulus_smooth_iterations,
        smooth_relaxation=args.annulus_smooth_relaxation,
    )

    shell_before_repair = base.merge_surfaces([epi_open, endo_open, annulus])
    free_before = len(base.extract_boundary_edges(shell_before_repair))

    shell = base.repair_closed_surface(shell_before_repair)
    free_after = len(base.extract_boundary_edges(shell))
    if free_after:
        raise RuntimeError(f"{name}: shell has {free_after} free edges after repair.")

    return VentricleBuild(
        name=name,
        profile=profile,
        intersection=intersection,
        endo_open=endo_open,
        epi_open=epi_open,
        endo_deleted_cap=endo_deleted_cap,
        epi_deleted_cap=epi_deleted_cap,
        endo_opening_loop=endo_smoothed_boundary,
        epi_opening_loop=epi_smoothed_boundary,
        endo_wall_boundary_loop=endo_boundary,
        epi_wall_boundary_loop=epi_boundary,
        original_control_loop=base.make_polyline(original_control_points),
        shifted_control_loop=base.make_polyline(shifted_control_points),
        endo_raw_wall_boundary_loop=endo_raw_boundary,
        epi_raw_wall_boundary_loop=epi_raw_boundary,
        endo_boundary_smoothing=endo_boundary_smoothing,
        epi_boundary_smoothing=epi_boundary_smoothing,
        annulus=annulus,
        shell=shell,
        shell_free_edges_before_repair=free_before,
        shell_free_edges_after_repair=free_after,
    )


def unique_attempt_overrides(args: argparse.Namespace) -> List[Dict[str, object]]:
    templates: List[Dict[str, object]] = [
        {},
        {"control_loop_points": 120, "annulus_loop_points": 120},
        {"control_loop_points": 96, "annulus_loop_points": 96},
        {"control_loop_points": 80, "annulus_loop_points": 80},
        {"control_loop_points": 64, "annulus_loop_points": 64},
        {"control_loop_points": 120, "annulus_loop_points": 96},
        {"control_loop_points": 96, "annulus_loop_points": 120},
        {"control_loop_points": 120, "annulus_loop_points": 120, "control_smooth_iterations": 10, "annulus_smooth_iterations": 15},
        {"control_loop_points": 96, "annulus_loop_points": 96, "control_smooth_iterations": 10, "annulus_smooth_iterations": 15},
        {"control_loop_points": 120, "annulus_loop_points": 120, "control_smooth_iterations": 30, "annulus_smooth_iterations": 45},
        {"control_loop_points": 96, "annulus_loop_points": 96, "control_smooth_iterations": 30, "annulus_smooth_iterations": 45},
        {"control_loop_points": 120, "annulus_loop_points": 120, "annulus_radial_layers": 2},
        {"control_loop_points": 96, "annulus_loop_points": 96, "annulus_radial_layers": 2},
        {"control_loop_points": 120, "annulus_loop_points": 120, "target_intersection_area_fraction": 0.12},
        {"control_loop_points": 96, "annulus_loop_points": 96, "target_intersection_area_fraction": 0.12},
        {"control_loop_points": 120, "annulus_loop_points": 120, "target_intersection_area_fraction": 0.05},
        {"control_loop_points": 96, "annulus_loop_points": 96, "target_intersection_area_fraction": 0.05},
        {"control_loop_points": 80, "annulus_loop_points": 80, "target_intersection_area_fraction": 0.12},
        {"control_loop_points": 80, "annulus_loop_points": 80, "target_intersection_area_fraction": 0.05},
    ]

    max_rank = int(getattr(args, "max_interface_candidate_rank", 4))
    templates = [dict(template, interface_candidate_rank=rank) for rank in range(max_rank + 1) for template in templates]

    seen = set()
    unique: List[Dict[str, object]] = []
    base_values = vars(args)
    for template in templates:
        merged = {
            "control_loop_points": template.get("control_loop_points", base_values["control_loop_points"]),
            "annulus_loop_points": template.get("annulus_loop_points", base_values["annulus_loop_points"]),
            "control_smooth_iterations": template.get("control_smooth_iterations", base_values["control_smooth_iterations"]),
            "annulus_smooth_iterations": template.get("annulus_smooth_iterations", base_values["annulus_smooth_iterations"]),
            "annulus_radial_layers": template.get("annulus_radial_layers", base_values["annulus_radial_layers"]),
            "target_intersection_area_fraction": template.get("target_intersection_area_fraction", base_values["target_intersection_area_fraction"]),
            "interface_candidate_rank": template.get("interface_candidate_rank", base_values.get("interface_candidate_rank", 0)),
        }
        key = tuple(sorted(merged.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(merged)
    return unique


def namespace_with_overrides(args: argparse.Namespace, overrides: Dict[str, object]) -> argparse.Namespace:
    values = dict(vars(args))
    values.update(overrides)
    return argparse.Namespace(**values)


def build_and_tetrahedralize_adaptive(
    name: str,
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile: base.AreaProfile,
    args: argparse.Namespace,
) -> Tuple[VentricleBuild, np.ndarray, np.ndarray, np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    failures: List[Dict[str, object]] = []
    for attempt_index, overrides in enumerate(unique_attempt_overrides(args), start=1):
        attempt_args = namespace_with_overrides(args, overrides)
        try:
            build = build_ventricle_shell(name, endo, epi, profile, attempt_args)
            seed = base.find_shell_seed(epi, endo, build.shell)
            points, cells = base.tetrahedralize_shell(build.shell, seed)
            metadata = {
                "attempt_index": attempt_index,
                "overrides": overrides,
                "seed": [float(x) for x in seed],
                "tetrahedra": int(len(cells)),
                "points": int(len(points)),
            }
            return build, seed, points, cells, metadata, failures
        except Exception as exc:
            failures.append({
                "attempt_index": attempt_index,
                "overrides": overrides,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"{name} adaptive attempt {attempt_index} failed: {type(exc).__name__}: {exc}", flush=True)

    raise RuntimeError(f"{name}: all adaptive build/tetrahedralization attempts failed. Last error: {failures[-1] if failures else 'none'}")


def candidate_to_dict(candidate: ShiftCandidate) -> Dict[str, object]:
    return {
        "distance": candidate.distance,
        "face_count": candidate.face_count,
        "area": candidate.area,
        "opening_loop_area": candidate.opening_loop_area,
        "boundary_loop_areas": candidate.boundary_loop_areas,
        "boundary_loop_lengths": candidate.boundary_loop_lengths,
        "material_secondary_loop_count": candidate.material_secondary_loop_count,
        "depth_from_base": candidate.depth_from_base,
        "removed_z_min": candidate.removed_z_min,
        "removed_z_max": candidate.removed_z_max,
        "component_count": candidate.component_count,
        "boundary_loop_count": candidate.boundary_loop_count,
        "boundary_loop_length": candidate.boundary_loop_length,
        "accepted": candidate.accepted,
    }


def intersection_summary(result: ShiftIntersectionResult) -> Dict[str, object]:
    return {
        "shift_distance": result.shift_distance,
        "shift_direction": [float(v) for v in result.direction],
        "selected_candidate": candidate_to_dict(result.selected_candidate),
        "selection_metadata": result.selection_metadata,
        "candidate_history": [candidate_to_dict(candidate) for candidate in result.candidate_history],
        "moved_endo": base.surface_summary(result.moved_endo),
        "shifted_intersection_cap": base.surface_summary(result.shifted_intersection_cap),
        "original_endo_intersection_cap": base.surface_summary(result.original_endo_intersection_cap),
        "original_endo_face_open": base.surface_summary(result.original_endo_face_open),
        "boundary_loop_points": int(result.boundary_loop.n_points),
        "boundary_loop_length": float(result.boundary_loop.length),
    }


def ventricle_summary(build: VentricleBuild) -> Dict[str, object]:
    return {
        "z_rule": base.profile_to_dict(build.profile),
        "shifted_endo_intersection": intersection_summary(build.intersection),
        "surfaces": {
            "endo_open": base.surface_summary(build.endo_open),
            "epi_open": base.surface_summary(build.epi_open),
            "endo_deleted_cap": base.surface_summary(build.endo_deleted_cap),
            "epi_deleted_cap": base.surface_summary(build.epi_deleted_cap),
            "myocardium_base_cut": base.surface_summary(build.annulus),
            "myocardium_shell": base.surface_summary(build.shell),
        },
        "loops": {
            "original_control_loop_points": int(build.original_control_loop.n_points),
            "shifted_control_loop_points": int(build.shifted_control_loop.n_points),
            "endo_opening_loop_points": int(build.endo_opening_loop.n_points),
            "epi_opening_loop_points": int(build.epi_opening_loop.n_points),
            "endo_wall_boundary_loop_points": int(build.endo_wall_boundary_loop.n_points),
            "epi_wall_boundary_loop_points": int(build.epi_wall_boundary_loop.n_points),
            "endo_raw_wall_boundary_loop_points": int(build.endo_raw_wall_boundary_loop.n_points),
            "epi_raw_wall_boundary_loop_points": int(build.epi_raw_wall_boundary_loop.n_points),
        },
        "boundary_smoothing": {
            "endo": build.endo_boundary_smoothing,
            "epi": build.epi_boundary_smoothing,
        },
        "free_edges": {
            "before_repair": build.shell_free_edges_before_repair,
            "after_repair": build.shell_free_edges_after_repair,
        },
    }


def write_region_legend(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region_id", "description"])
        for region_id, description in REGION_LEGEND.items():
            writer.writerow([region_id, description])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--vtk-dir", type=Path, default=None, help="Surface input folder; defaults to SUBJECT_DIR/vtk_fixed")
    parser.add_argument(
        "--subject-base-side",
        choices=["auto", "low_z", "high_z", "off"],
        default="auto",
        help="Use one base side for the whole subject. auto scores ED/ES LV/RV surfaces from --vtk-dir; off uses the old per-frame/per-chamber rule.",
    )
    parser.add_argument("--subject-base-frames", default="ED,ES", help="Comma-separated frames used for auto subject-base detection.")
    parser.add_argument(
        "--opening-strategy",
        choices=["stable_thickness", "max_interface"],
        default="stable_thickness",
        help="stable_thickness uses the maximum basal directional myocardium thickness as the displacement limit; max_interface keeps the old maximum-area opening rule.",
    )
    parser.add_argument("--stable-thickness-percentile", type=float, default=95.0)
    parser.add_argument("--stable-thickness-candidate-percentiles", default="90,92.5,95,97.5,98,99,99.5,100")
    parser.add_argument("--stable-thickness-seed-band-multiplier", type=float, default=2.0)
    parser.add_argument("--stable-thickness-min-basal-face-fraction", type=float, default=0.01)
    parser.add_argument("--stable-thickness-max-search-fraction", type=float, default=0.30)
    parser.add_argument("--stable-thickness-search-steps", type=int, default=60)
    parser.add_argument("--flat-output", action="store_true")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_dir = args.subject_dir
    frame = args.frame
    vtk_dir = args.vtk_dir or subject_dir / "vtk_fixed"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or subject_dir / ("volmesh_open_stable" if args.opening_strategy == "stable_thickness" else "volmesh_open_base_max_interface_smooth_boundary_custom_tag")
    output_token = "open_stable" if args.opening_strategy == "stable_thickness" else "max_interface_smooth_boundary_custom_tag_open_base"
    if args.flat_output:
        output_dir = output_root
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_root / f"{subject_dir.name}_{frame}_{output_token}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)

    input_paths = {
        "lv_endo": vtk_dir / f"LVendo_{frame}.vtk",
        "lv_epi": vtk_dir / f"LVepi_{frame}.vtk",
        "rv_endo": vtk_dir / f"RV_{frame}.vtk",
        "rv_epi": vtk_dir / f"RVepi_{frame}.vtk",
    }
    surfaces = {name: base.load_surface(path) for name, path in input_paths.items()}
    subject_base_side, subject_base_detection = detect_subject_base_side(vtk_dir, args)
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
    base.write_area_profile(output_dir / f"{subject_dir.name}_{frame}_LV_z_area_profile.csv", lv_profile)
    base.write_area_profile(output_dir / f"{subject_dir.name}_{frame}_RV_z_area_profile.csv", rv_profile)

    lv_build, lv_seed, lv_points, lv_cells, lv_attempt, lv_failures = build_and_tetrahedralize_adaptive(
        "LV", surfaces["lv_endo"], surfaces["lv_epi"], lv_profile, args
    )
    rv_build, rv_seed, rv_points, rv_cells, rv_attempt, rv_failures = build_and_tetrahedralize_adaptive(
        "RV", surfaces["rv_endo"], surfaces["rv_epi"], rv_profile, args
    )
    points, cells, labels = base.combine_tet_meshes([(lv_points, lv_cells, 2), (rv_points, rv_cells, 4)])

    elem_tags = custom_elem_tags(labels)
    custom_counts = {str(tag): int(np.count_nonzero(elem_tags == tag)) for tag in sorted(CUSTOM_ELEMTAG_LEGEND, reverse=True)}
    mesh_name = (
        f"{subject_dir.name}_{frame}_biventricular_open_stable_custom_tag_volmesh.vtk"
        if args.opening_strategy == "stable_thickness"
        else f"{subject_dir.name}_{frame}_biventricular_open_base_max_interface_smooth_boundary_custom_tag_volmesh.vtk"
    )
    mesh_path = output_dir / mesh_name
    write_custom_vtk(mesh_path, points, cells, elem_tags)
    write_region_legend(output_dir / f"{subject_dir.name}_{frame}_open_base_region_legend.csv")
    write_custom_elem_tag_legend(output_dir / f"{subject_dir.name}_{frame}_custom_elemTag_legend.csv")

    debug_surfaces = {
        f"LVendo_{frame}_shifted_to_base.vtk": lv_build.intersection.moved_endo,
        f"RV_{frame}_shifted_to_base.vtk": rv_build.intersection.moved_endo,
        f"LV_shifted_endo_epi_intersection_cap_{frame}.vtk": lv_build.intersection.shifted_intersection_cap,
        f"RV_shifted_endo_epi_intersection_cap_{frame}.vtk": rv_build.intersection.shifted_intersection_cap,
        f"LV_endo_intersection_cap_{frame}.vtk": lv_build.intersection.original_endo_intersection_cap,
        f"RV_endo_intersection_cap_{frame}.vtk": rv_build.intersection.original_endo_intersection_cap,
        f"LV_endo_face_open_from_intersection_{frame}.vtk": lv_build.intersection.original_endo_face_open,
        f"RV_endo_face_open_from_intersection_{frame}.vtk": rv_build.intersection.original_endo_face_open,
        f"LVendo_{frame}_open.vtk": lv_build.endo_open,
        f"LVepi_{frame}_open.vtk": lv_build.epi_open,
        f"RV_{frame}_open.vtk": rv_build.endo_open,
        f"RVepi_{frame}_open.vtk": rv_build.epi_open,
        f"LV_endo_deleted_cap_{frame}.vtk": lv_build.endo_deleted_cap,
        f"LV_epi_deleted_cap_{frame}.vtk": lv_build.epi_deleted_cap,
        f"RV_endo_deleted_cap_{frame}.vtk": rv_build.endo_deleted_cap,
        f"RV_epi_deleted_cap_{frame}.vtk": rv_build.epi_deleted_cap,
        f"LV_original_control_loop_{frame}.vtk": lv_build.original_control_loop,
        f"LV_shifted_control_loop_{frame}.vtk": lv_build.shifted_control_loop,
        f"RV_original_control_loop_{frame}.vtk": rv_build.original_control_loop,
        f"RV_shifted_control_loop_{frame}.vtk": rv_build.shifted_control_loop,
        f"LV_endo_opening_loop_{frame}.vtk": lv_build.endo_opening_loop,
        f"LV_epi_opening_loop_{frame}.vtk": lv_build.epi_opening_loop,
        f"RV_endo_opening_loop_{frame}.vtk": rv_build.endo_opening_loop,
        f"RV_epi_opening_loop_{frame}.vtk": rv_build.epi_opening_loop,
        f"LV_endo_wall_boundary_loop_{frame}.vtk": lv_build.endo_wall_boundary_loop,
        f"LV_epi_wall_boundary_loop_{frame}.vtk": lv_build.epi_wall_boundary_loop,
        f"RV_endo_wall_boundary_loop_{frame}.vtk": rv_build.endo_wall_boundary_loop,
        f"RV_epi_wall_boundary_loop_{frame}.vtk": rv_build.epi_wall_boundary_loop,
        f"LV_endo_raw_wall_boundary_loop_{frame}.vtk": lv_build.endo_raw_wall_boundary_loop,
        f"LV_epi_raw_wall_boundary_loop_{frame}.vtk": lv_build.epi_raw_wall_boundary_loop,
        f"RV_endo_raw_wall_boundary_loop_{frame}.vtk": rv_build.endo_raw_wall_boundary_loop,
        f"RV_epi_raw_wall_boundary_loop_{frame}.vtk": rv_build.epi_raw_wall_boundary_loop,
        f"LV_myocardium_base_cut_{frame}.vtk": lv_build.annulus,
        f"RV_myocardium_base_cut_{frame}.vtk": rv_build.annulus,
        f"LV_myocardium_boundary_{frame}.vtk": lv_build.shell,
        f"RV_myocardium_boundary_{frame}.vtk": rv_build.shell,
    }
    for name, surface in debug_surfaces.items():
        surface.save(output_dir / name)

    quality = base.tet_quality(points, cells)
    per_region: Dict[str, Dict[str, object]] = {}
    for region_id in sorted(REGION_LEGEND):
        mask = labels == region_id
        per_region[str(region_id)] = {
            "description": REGION_LEGEND[region_id],
            "tetrahedra": int(mask.sum()),
            "abs_volume": float(np.abs(base.tet_volumes(points, cells[mask])).sum()),
        }

    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "output_dir": str(output_dir.resolve()),
        "output_mesh": str(mesh_path.resolve()),
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "method": {
            "base_detection": "Use one subject-level base side by default: score ED/ES LVendo and RV low-Z/high-Z end areas from --vtk-dir, then apply the selected side to both LV and RV for the current frame. Explicit --lv-base-side/--rv-base-side overrides still take precedence.",
            "auto_shift": "Stable mode estimates basal directional myocardium thickness and tests endocardial translations capped by the P100 basal thickness; max_interface mode keeps the old area-ranked scan.",
            "deletion_region": "Stable mode uses the maximum basal directional thickness as the upper displacement limit and selects the largest topologically valid basal shifted-endo outside-epi interface candidate. Adaptive retries may choose the next valid candidate if shell/TetGen validation fails.",
            "smoothing": "Use area-preserving control-loop smoothing, then smooth the actual clipped basal rim vertices with local neighbor-ring blending before building the layered myocardium base cut.",
            "custom_vtk_labels": "Final VTK is written with custom_vtk.write_custom_vtk. Cell scalar elemTag uses 1 for LV myocardium and 0 for RV myocardium.",
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "subject_base_detection": subject_base_detection,
        "ventricles": {
            "lv": ventricle_summary(lv_build),
            "rv": ventricle_summary(rv_build),
        },
        "adaptive_attempts": {
            "lv_selected": lv_attempt,
            "rv_selected": rv_attempt,
            "lv_failures": lv_failures,
            "rv_failures": rv_failures,
        },
        "tet_quality": quality,
        "per_region": per_region,
        "custom_elemTag": {"1": "LV myocardium", "0": "RV myocardium"},
        "custom_elemTag_counts": custom_counts,
        "tetgen_seeds": {
            "lv": [float(x) for x in lv_seed],
            "rv": [float(x) for x in rv_seed],
        },
        "notes": [
            "Input LV/RV surfaces are read from --vtk-dir; this run uses the repaired vtk_fixed LVendo/LVepi/RV/RVepi VTK files.",
            "In stable_thickness mode, the shifted endocardial surface is moved up to the maximum basal directional thickness and the largest topologically valid basal exit region is used; the final wall surfaces use the original endo/epi geometry except for a local smoothed basal rim band.",
            "The myocardium has a closed basal cut surface for tetrahedral meshing, while the blood-pool side is open at the base.",
            "Debug shifted surfaces, intersection caps, opening loops, deleted caps, base cut surfaces, and repaired myocardial shells are saved beside the final mesh.",
            "For ParaView coloring, load the final VTK and color by the cell scalar elemTag: 1 is LV myocardium, 0 is RV myocardium.",
        ],
    }
    summary_name = (
        f"{subject_dir.name}_{frame}_open_stable_custom_tag_summary.json"
        if args.opening_strategy == "stable_thickness"
        else f"{subject_dir.name}_{frame}_open_base_max_interface_smooth_boundary_custom_tag_summary.json"
    )
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "output_mesh": str(mesh_path),
        "summary": str(summary_path),
        "quality": quality,
        "per_region": per_region,
        "custom_elemTag_counts": custom_counts,
        "lv_shift_distance": lv_build.intersection.shift_distance,
        "rv_shift_distance": rv_build.intersection.shift_distance,
        "lv_selected_intersection": candidate_to_dict(lv_build.intersection.selected_candidate),
        "rv_selected_intersection": candidate_to_dict(rv_build.intersection.selected_candidate),
    }, indent=2))


if __name__ == "__main__":
    main()
