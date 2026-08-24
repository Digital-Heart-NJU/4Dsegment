#!/usr/bin/env python3
"""Generate an open-base biventricular myocardium mesh from the maximum shifted endocardium/epicardium interface.

Pipeline:
  raw LV/RV endocardial + epicardial surfaces
  -> detect anatomical base side from Z-area profiles
  -> translate each endocardial blood-pool surface toward the base by an automatic distance
  -> find every valid basal connected shifted-endo/epi interface candidate
  -> choose the maximum valid interface boundary to define endocardial and epicardial opening loops
  -> smooth the opening loops and build a layered myocardium cut surface
  -> repair closed myocardial shells and tetrahedralize LV/RV myocardium
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import meshio
import numpy as np
import pyvista as pv

import generate_subject_open_base_bloodpool_cap_volmesh as cap_utils
import generate_subject_open_base_zrule_volmesh as base


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
    annulus: pv.PolyData
    shell: pv.PolyData
    shell_free_edges_before_repair: int
    shell_free_edges_after_repair: int


REGION_LEGEND = base.REGION_LEGEND


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
    )


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


def build_ventricle_shell(
    name: str,
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile: base.AreaProfile,
    args: argparse.Namespace,
) -> VentricleBuild:
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

    endo_open = cap_utils.clip_surface_by_loop(endo, endo_loop_points, keep_large_region=True)
    epi_open = cap_utils.clip_surface_by_loop(epi, epi_loop_points, keep_large_region=True)
    endo_deleted_cap = cap_utils.clip_surface_by_loop(endo, endo_loop_points, keep_large_region=False)
    epi_deleted_cap = cap_utils.clip_surface_by_loop(epi, epi_loop_points, keep_large_region=False)

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
        endo_opening_loop=base.make_polyline(endo_loop_points),
        epi_opening_loop=base.make_polyline(epi_loop_points),
        endo_wall_boundary_loop=endo_boundary,
        epi_wall_boundary_loop=epi_boundary,
        original_control_loop=base.make_polyline(original_control_points),
        shifted_control_loop=base.make_polyline(shifted_control_points),
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
    vtk_dir = subject_dir / "vtks"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or subject_dir / "volmesh_open_base_max_interface"
    if args.flat_output:
        output_dir = output_root
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_root / f"{subject_dir.name}_{frame}_max_interface_open_base_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)

    input_paths = {
        "lv_endo": vtk_dir / f"LVendo_{frame}.vtk",
        "lv_epi": vtk_dir / f"LVepi_{frame}.vtk",
        "rv_endo": vtk_dir / f"RV_{frame}.vtk",
        "rv_epi": vtk_dir / f"RVepi_{frame}.vtk",
    }
    surfaces = {name: base.load_surface(path) for name, path in input_paths.items()}

    lv_profile = base.compute_area_profile(
        surfaces["lv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=args.lv_base_side,
    )
    rv_profile = base.compute_area_profile(
        surfaces["rv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=args.rv_base_side,
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

    mesh_path = output_dir / f"{subject_dir.name}_{frame}_biventricular_open_base_max_interface_volmesh.vtk"
    base.save_vtk(mesh_path, points, cells, labels)
    write_region_legend(output_dir / f"{subject_dir.name}_{frame}_open_base_region_legend.csv")

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
            "base_detection": "Compare low-Z and high-Z blood-pool end areas; larger side is base.",
            "auto_shift": "Scan all base-directed endocardial translations and rank valid shifted-endo/epi interfaces by opening-loop area.",
            "deletion_region": "Use the maximum valid basal shifted-endo outside-epi interface as the endocardial deletion cap; if it fails shell/TetGen validation, try the next largest candidate.",
            "smoothing": "Use area-preserving control-loop smoothing and smooth only the layered myocardium base cut, while preserving ventricular wall surfaces.",
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
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
        "tetgen_seeds": {
            "lv": [float(x) for x in lv_seed],
            "rv": [float(x) for x in rv_seed],
        },
        "notes": [
            "Input LV/RV surfaces are the raw LVendo/LVepi/RV/RVepi VTK files.",
            "The shifted endocardial surface is used only to rank maximum basal intersection/deletion candidates; the final wall surfaces use the original endo/epi geometry clipped by area-preserved smoothed loops.",
            "The myocardium has a closed basal cut surface for tetrahedral meshing, while the blood-pool side is open at the base.",
            "Debug shifted surfaces, intersection caps, opening loops, deleted caps, base cut surfaces, and repaired myocardial shells are saved beside the final mesh.",
        ],
    }
    summary_path = output_dir / f"{subject_dir.name}_{frame}_open_base_max_interface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "output_mesh": str(mesh_path),
        "summary": str(summary_path),
        "quality": quality,
        "per_region": per_region,
        "lv_shift_distance": lv_build.intersection.shift_distance,
        "rv_shift_distance": rv_build.intersection.shift_distance,
        "lv_selected_intersection": candidate_to_dict(lv_build.intersection.selected_candidate),
        "rv_selected_intersection": candidate_to_dict(rv_build.intersection.selected_candidate),
    }, indent=2))


if __name__ == "__main__":
    main()
