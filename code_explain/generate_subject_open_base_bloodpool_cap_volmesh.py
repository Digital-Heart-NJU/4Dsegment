#!/usr/bin/env python3
"""Generate an open-base biventricular myocardium mesh from basal blood-pool caps.

Pipeline:
  raw LV/RV endocardial + epicardial surfaces
  -> detect anatomical base side from Z-area profiles
  -> find the maximum connected basal cap on each blood-pool surface
  -> use that cap boundary as the control curve for endocardial/epicardial opening
  -> build a layered, smoothed myocardial base cut surface
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

import numpy as np
import pyvista as pv
import vtk

import generate_subject_open_base_zrule_volmesh as base


@dataclass
class BasalCapResult:
    cap_surface: pv.PolyData
    face_open_surface: pv.PolyData
    boundary_loop: pv.PolyData
    face_count: int
    cap_area: float
    removed_z_min: float
    removed_z_max: float
    effective_z_limit: float
    normal_threshold: float
    component_count: int


@dataclass
class VentricleBuild:
    name: str
    profile: base.AreaProfile
    bloodpool_cap: BasalCapResult
    endo_open: pv.PolyData
    epi_open: pv.PolyData
    endo_clipped_cap: pv.PolyData
    epi_clipped_cap: pv.PolyData
    endo_opening_loop: pv.PolyData
    epi_opening_loop: pv.PolyData
    endo_wall_boundary_loop: pv.PolyData
    epi_wall_boundary_loop: pv.PolyData
    control_loop: pv.PolyData
    annulus: pv.PolyData
    shell: pv.PolyData
    shell_free_edges_before_repair: int
    shell_free_edges_after_repair: int


REGION_LEGEND = base.REGION_LEGEND


def face_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def connected_components(mask: np.ndarray, neighbors: Sequence[Sequence[int]]) -> List[np.ndarray]:
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
            for nb in neighbors[current]:
                if mask[nb] and not visited[nb]:
                    visited[nb] = True
                    queue.append(int(nb))
        components.append(np.asarray(ids, dtype=np.int64))
    return components


def detect_max_connected_bloodpool_cap(
    mesh: pv.PolyData,
    profile: base.AreaProfile,
    seed_band_fraction: float,
    cap_depth_scale: float,
    normal_threshold: float,
) -> BasalCapResult:
    mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    triangles = base.triangle_faces(mesh)
    points = np.asarray(mesh.points)
    centers = points[triangles].mean(axis=1)
    z_centers = centers[:, 2]
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    height = max(z_max - z_min, 1e-6)

    with_normals = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
        inplace=False,
    )
    normals = np.asarray(with_normals.cell_data.get("Normals"))
    if normals.shape[0] != len(triangles):
        normals = np.zeros((len(triangles), 3), dtype=float)

    if profile.base_side == "high_z":
        distance_from_base = z_max - z_centers
        detected_depth = max(z_max - profile.z_cut, 0.0)
        base_normal_component = normals[:, 2]
    else:
        distance_from_base = z_centers - z_min
        detected_depth = max(profile.z_cut - z_min, 0.0)
        base_normal_component = -normals[:, 2]

    seed_depth = max(seed_band_fraction * height, height / 80.0)
    max_depth = max(detected_depth * cap_depth_scale, seed_depth * 1.75)
    max_depth = min(max_depth, height * 0.45)
    effective_z_limit = z_max - max_depth if profile.base_side == "high_z" else z_min + max_depth

    def choose_patch(threshold: float) -> Tuple[np.ndarray, int]:
        seed_mask = distance_from_base <= seed_depth
        candidate_mask = (distance_from_base <= max_depth) & (base_normal_component >= threshold)
        candidate_mask = candidate_mask | seed_mask
        neighbors = base.face_adjacency(triangles)
        components = connected_components(candidate_mask, neighbors)
        seeded = [comp for comp in components if np.any(seed_mask[comp])]
        usable = seeded or components
        if not usable:
            return np.zeros(len(triangles), dtype=bool), 0
        areas = face_areas(points, triangles)
        best = max(usable, key=lambda comp: float(areas[comp].sum()))
        patch = np.zeros(len(triangles), dtype=bool)
        patch[best] = True
        return patch, len(usable)

    patch_mask, component_count = choose_patch(normal_threshold)
    active_threshold = normal_threshold
    min_patch_faces = max(8, int(round(len(triangles) * 0.003)))
    if int(patch_mask.sum()) < min_patch_faces:
        patch_mask, component_count = choose_patch(-1.0)
        active_threshold = -1.0

    if int(patch_mask.sum()) == 0:
        raise RuntimeError("No connected basal blood-pool cap could be detected.")

    cap_surface = base.subset_mesh_faces(mesh, patch_mask)
    face_open = base.subset_mesh_faces(mesh, ~patch_mask)
    loops = base.select_boundary_loops(face_open)
    boundary_loop = loops[0]
    cap_area = float(face_areas(points, triangles)[patch_mask].sum())
    removed_z = z_centers[patch_mask]

    return BasalCapResult(
        cap_surface=cap_surface,
        face_open_surface=face_open,
        boundary_loop=boundary_loop,
        face_count=int(patch_mask.sum()),
        cap_area=cap_area,
        removed_z_min=float(removed_z.min()),
        removed_z_max=float(removed_z.max()),
        effective_z_limit=float(effective_z_limit),
        normal_threshold=float(active_threshold),
        component_count=int(component_count),
    )


def smooth_closed_loop(points: np.ndarray, iterations: int, relaxation: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        pts = pts + relaxation * ((np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0)) * 0.5 - pts)
    return pts


def remove_duplicate_sequential_points(points: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    output: List[np.ndarray] = []
    for point in np.asarray(points, dtype=float):
        if not output or np.linalg.norm(point - output[-1]) > eps:
            output.append(point)
    if len(output) > 3 and np.linalg.norm(output[0] - output[-1]) <= eps:
        output = output[:-1]
    return np.asarray(output, dtype=float)


def nearest_surface_loop(loop_points: np.ndarray, mesh: pv.PolyData) -> np.ndarray:
    projected: List[np.ndarray] = []
    last_id = -1
    for point in loop_points:
        point_id = int(mesh.find_closest_point(point))
        if point_id != last_id:
            projected.append(np.asarray(mesh.points[point_id], dtype=float))
            last_id = point_id
    projected_arr = remove_duplicate_sequential_points(np.asarray(projected, dtype=float))
    if len(projected_arr) < 12:
        raise RuntimeError("Projected opening loop has too few unique points.")
    return projected_arr


def clip_surface_by_loop(mesh: pv.PolyData, loop_points: np.ndarray, keep_large_region: bool) -> pv.PolyData:
    vtk_points = vtk.vtkPoints()
    for point in loop_points:
        vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))

    selector = vtk.vtkSelectPolyData()
    selector.SetInputData(mesh)
    selector.SetLoop(vtk_points)
    selector.GenerateSelectionScalarsOn()
    selector.SetSelectionModeToSmallestRegion()
    selector.Update()

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputConnection(selector.GetOutputPort())
    clipper.SetValue(0.0)
    if keep_large_region:
        clipper.InsideOutOff()
    else:
        clipper.InsideOutOn()
    clipper.Update()

    clipped = pv.wrap(clipper.GetOutput()).extract_surface().triangulate().clean(tolerance=1e-7)
    if clipped.n_cells == 0:
        raise RuntimeError("Loop clipping returned an empty surface.")
    return clipped


def equal_count_strip(loop_a: np.ndarray, loop_b: np.ndarray) -> pv.PolyData:
    if len(loop_a) != len(loop_b):
        raise ValueError("equal_count_strip requires loops with equal point counts.")
    n = len(loop_a)
    points = np.vstack([loop_a, loop_b])
    triangles: List[List[int]] = []
    for i in range(n):
        nxt = (i + 1) % n
        triangles.append([i, nxt, n + i])
        triangles.append([nxt, n + nxt, n + i])
    return pv.PolyData(points, base.faces_from_triangles(np.asarray(triangles, dtype=np.int64))).extract_surface().triangulate().clean(tolerance=1e-7)


def smooth_myocardium_cut_surface(
    epi_boundary_loop: pv.PolyData,
    endo_boundary_loop: pv.PolyData,
    sample_count: int,
    radial_layers: int,
    smooth_iterations: int,
    smooth_relaxation: float,
) -> pv.PolyData:
    endo_exact = remove_duplicate_sequential_points(np.asarray(endo_boundary_loop.points, dtype=float))
    epi_exact = remove_duplicate_sequential_points(np.asarray(epi_boundary_loop.points, dtype=float))

    endo_smooth = smooth_closed_loop(
        base.resample_closed_loop(endo_exact, sample_count),
        iterations=smooth_iterations,
        relaxation=smooth_relaxation,
    )
    epi_aligned = base.align_inner_loop_to_outer(endo_smooth, epi_exact)
    epi_smooth = smooth_closed_loop(
        base.resample_closed_loop(epi_aligned, sample_count),
        iterations=smooth_iterations,
        relaxation=smooth_relaxation,
    )

    parts: List[pv.PolyData] = []
    parts.append(base.zipper_annulus(base.make_polyline(endo_smooth), base.make_polyline(endo_exact)))

    rings = [endo_smooth]
    for layer in range(1, radial_layers + 1):
        t = layer / float(radial_layers + 1)
        rings.append((1.0 - t) * endo_smooth + t * epi_smooth)
    rings.append(epi_smooth)

    for first, second in zip(rings[:-1], rings[1:]):
        parts.append(equal_count_strip(first, second))

    parts.append(base.zipper_annulus(base.make_polyline(epi_exact), base.make_polyline(epi_smooth)))
    return base.merge_surfaces(parts)


def build_ventricle_shell(
    name: str,
    endo: pv.PolyData,
    epi: pv.PolyData,
    profile: base.AreaProfile,
    args: argparse.Namespace,
) -> VentricleBuild:
    bloodpool_cap = detect_max_connected_bloodpool_cap(
        endo,
        profile,
        seed_band_fraction=args.cap_seed_band_fraction,
        cap_depth_scale=args.cap_depth_scale,
        normal_threshold=args.cap_normal_threshold,
    )

    control_points = smooth_closed_loop(
        base.resample_closed_loop(np.asarray(bloodpool_cap.boundary_loop.points), args.control_loop_points),
        iterations=args.control_smooth_iterations,
        relaxation=args.control_smooth_relaxation,
    )
    endo_loop_points = nearest_surface_loop(control_points, endo)
    epi_loop_points = nearest_surface_loop(control_points, epi)

    endo_open = clip_surface_by_loop(endo, endo_loop_points, keep_large_region=True)
    epi_open = clip_surface_by_loop(epi, epi_loop_points, keep_large_region=True)
    endo_clipped_cap = clip_surface_by_loop(endo, endo_loop_points, keep_large_region=False)
    epi_clipped_cap = clip_surface_by_loop(epi, epi_loop_points, keep_large_region=False)

    endo_boundary = base.select_boundary_loops(endo_open)[0]
    epi_boundary = base.select_boundary_loops(epi_open)[0]
    annulus = smooth_myocardium_cut_surface(
        epi_boundary,
        endo_boundary,
        sample_count=args.annulus_loop_points,
        radial_layers=args.annulus_radial_layers,
        smooth_iterations=args.annulus_smooth_iterations,
        smooth_relaxation=args.annulus_smooth_relaxation,
    )

    shell_before_repair = base.merge_surfaces([epi_open, endo_open, annulus])
    free_before = len(base.extract_boundary_edges(shell_before_repair))
    if free_before:
        raise RuntimeError(f"{name}: shell has {free_before} free edges before repair.")

    shell = base.repair_closed_surface(shell_before_repair)
    free_after = len(base.extract_boundary_edges(shell))
    if free_after:
        raise RuntimeError(f"{name}: shell has {free_after} free edges after repair.")

    return VentricleBuild(
        name=name,
        profile=profile,
        bloodpool_cap=bloodpool_cap,
        endo_open=endo_open,
        epi_open=epi_open,
        endo_clipped_cap=endo_clipped_cap,
        epi_clipped_cap=epi_clipped_cap,
        endo_opening_loop=base.make_polyline(endo_loop_points),
        epi_opening_loop=base.make_polyline(epi_loop_points),
        endo_wall_boundary_loop=endo_boundary,
        epi_wall_boundary_loop=epi_boundary,
        control_loop=base.make_polyline(control_points),
        annulus=annulus,
        shell=shell,
        shell_free_edges_before_repair=free_before,
        shell_free_edges_after_repair=free_after,
    )


def basal_cap_summary(result: BasalCapResult) -> Dict[str, object]:
    return {
        "face_count": result.face_count,
        "area": result.cap_area,
        "removed_z_min": result.removed_z_min,
        "removed_z_max": result.removed_z_max,
        "effective_z_limit": result.effective_z_limit,
        "normal_threshold": result.normal_threshold,
        "component_count_considered": result.component_count,
        "surface": base.surface_summary(result.cap_surface),
        "boundary_loop_points": int(result.boundary_loop.n_points),
        "boundary_loop_length": float(result.boundary_loop.length),
    }


def ventricle_summary(build: VentricleBuild) -> Dict[str, object]:
    return {
        "z_rule": base.profile_to_dict(build.profile),
        "bloodpool_basal_cap": basal_cap_summary(build.bloodpool_cap),
        "surfaces": {
            "endo_open": base.surface_summary(build.endo_open),
            "epi_open": base.surface_summary(build.epi_open),
            "endo_clipped_cap": base.surface_summary(build.endo_clipped_cap),
            "epi_clipped_cap": base.surface_summary(build.epi_clipped_cap),
            "myocardium_base_cut": base.surface_summary(build.annulus),
            "myocardium_shell": base.surface_summary(build.shell),
        },
        "loops": {
            "control_loop_points": int(build.control_loop.n_points),
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


def save_loop(path: Path, loop: pv.PolyData) -> None:
    loop.save(path)


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
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--slab-fraction", type=float, default=0.035)
    parser.add_argument("--end-band-fraction", type=float, default=0.12)
    parser.add_argument("--stable-fraction", type=float, default=0.45)
    parser.add_argument("--cap-seed-band-fraction", type=float, default=0.035)
    parser.add_argument("--cap-depth-scale", type=float, default=1.35)
    parser.add_argument("--cap-normal-threshold", type=float, default=0.15)
    parser.add_argument("--control-loop-points", type=int, default=160)
    parser.add_argument("--control-smooth-iterations", type=int, default=15)
    parser.add_argument("--control-smooth-relaxation", type=float, default=0.20)
    parser.add_argument("--annulus-loop-points", type=int, default=160)
    parser.add_argument("--annulus-radial-layers", type=int, default=4)
    parser.add_argument("--annulus-smooth-iterations", type=int, default=25)
    parser.add_argument("--annulus-smooth-relaxation", type=float, default=0.20)
    parser.add_argument("--lv-z-cut", type=float, default=None)
    parser.add_argument("--rv-z-cut", type=float, default=None)
    parser.add_argument("--lv-base-side", choices=["low_z", "high_z"], default=None)
    parser.add_argument("--rv-base-side", choices=["low_z", "high_z"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_dir = args.subject_dir
    frame = args.frame
    vtk_dir = subject_dir / "vtks"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or subject_dir / "volmesh_open_base_bloodpool_cap"
    output_dir = output_root / f"{subject_dir.name}_{frame}_bloodpool_cap_open_base_{timestamp}"
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
        override_z_cut=args.lv_z_cut,
    )
    rv_profile = base.compute_area_profile(
        surfaces["rv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=args.rv_base_side,
        override_z_cut=args.rv_z_cut,
    )
    base.write_area_profile(output_dir / f"{subject_dir.name}_{frame}_LV_z_area_profile.csv", lv_profile)
    base.write_area_profile(output_dir / f"{subject_dir.name}_{frame}_RV_z_area_profile.csv", rv_profile)

    lv_build = build_ventricle_shell("LV", surfaces["lv_endo"], surfaces["lv_epi"], lv_profile, args)
    rv_build = build_ventricle_shell("RV", surfaces["rv_endo"], surfaces["rv_epi"], rv_profile, args)

    lv_seed = base.find_shell_seed(surfaces["lv_epi"], surfaces["lv_endo"], lv_build.shell)
    rv_seed = base.find_shell_seed(surfaces["rv_epi"], surfaces["rv_endo"], rv_build.shell)
    lv_points, lv_cells = base.tetrahedralize_shell(lv_build.shell, lv_seed)
    rv_points, rv_cells = base.tetrahedralize_shell(rv_build.shell, rv_seed)
    points, cells, labels = base.combine_tet_meshes([(lv_points, lv_cells, 2), (rv_points, rv_cells, 4)])

    mesh_path = output_dir / f"{subject_dir.name}_{frame}_biventricular_open_base_bloodpool_cap_volmesh.vtk"
    base.save_vtk(mesh_path, points, cells, labels)
    write_region_legend(output_dir / f"{subject_dir.name}_{frame}_open_base_region_legend.csv")

    debug_surfaces = {
        f"LV_bloodpool_basal_cap_{frame}.vtk": lv_build.bloodpool_cap.cap_surface,
        f"RV_bloodpool_basal_cap_{frame}.vtk": rv_build.bloodpool_cap.cap_surface,
        f"LV_endo_face_open_from_cap_{frame}.vtk": lv_build.bloodpool_cap.face_open_surface,
        f"RV_endo_face_open_from_cap_{frame}.vtk": rv_build.bloodpool_cap.face_open_surface,
        f"LVendo_{frame}_open.vtk": lv_build.endo_open,
        f"LVepi_{frame}_open.vtk": lv_build.epi_open,
        f"RV_{frame}_open.vtk": rv_build.endo_open,
        f"RVepi_{frame}_open.vtk": rv_build.epi_open,
        f"LV_endo_clipped_cap_{frame}.vtk": lv_build.endo_clipped_cap,
        f"LV_epi_clipped_cap_{frame}.vtk": lv_build.epi_clipped_cap,
        f"RV_endo_clipped_cap_{frame}.vtk": rv_build.endo_clipped_cap,
        f"RV_epi_clipped_cap_{frame}.vtk": rv_build.epi_clipped_cap,
        f"LV_myocardium_base_cut_{frame}.vtk": lv_build.annulus,
        f"RV_myocardium_base_cut_{frame}.vtk": rv_build.annulus,
        f"LV_myocardium_boundary_{frame}.vtk": lv_build.shell,
        f"RV_myocardium_boundary_{frame}.vtk": rv_build.shell,
        f"LV_bloodpool_cap_control_loop_{frame}.vtk": lv_build.control_loop,
        f"RV_bloodpool_cap_control_loop_{frame}.vtk": rv_build.control_loop,
        f"LV_endo_opening_loop_{frame}.vtk": lv_build.endo_opening_loop,
        f"LV_epi_opening_loop_{frame}.vtk": lv_build.epi_opening_loop,
        f"RV_endo_opening_loop_{frame}.vtk": rv_build.endo_opening_loop,
        f"RV_epi_opening_loop_{frame}.vtk": rv_build.epi_opening_loop,
        f"LV_endo_wall_boundary_loop_{frame}.vtk": lv_build.endo_wall_boundary_loop,
        f"LV_epi_wall_boundary_loop_{frame}.vtk": lv_build.epi_wall_boundary_loop,
        f"RV_endo_wall_boundary_loop_{frame}.vtk": rv_build.endo_wall_boundary_loop,
        f"RV_epi_wall_boundary_loop_{frame}.vtk": rv_build.epi_wall_boundary_loop,
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
            "base_detection": "Compare low-Z and high-Z end cross-section areas on blood-pool surfaces; larger end is the base.",
            "cap_detection": "Select the maximum connected basal cap component on LV/RV blood-pool surfaces.",
            "opening_definition": "Smooth the blood-pool cap boundary, project it to endocardium and epicardium, and use those loops for surface clipping.",
            "myocardium_cut": "Layered annular surface with exact wall-boundary collars and smoothed interior rings.",
        },
        "parameters": {
            "samples": args.samples,
            "slab_fraction": args.slab_fraction,
            "end_band_fraction": args.end_band_fraction,
            "stable_fraction": args.stable_fraction,
            "cap_seed_band_fraction": args.cap_seed_band_fraction,
            "cap_depth_scale": args.cap_depth_scale,
            "cap_normal_threshold": args.cap_normal_threshold,
            "control_loop_points": args.control_loop_points,
            "control_smooth_iterations": args.control_smooth_iterations,
            "control_smooth_relaxation": args.control_smooth_relaxation,
            "annulus_loop_points": args.annulus_loop_points,
            "annulus_radial_layers": args.annulus_radial_layers,
            "annulus_smooth_iterations": args.annulus_smooth_iterations,
            "annulus_smooth_relaxation": args.annulus_smooth_relaxation,
        },
        "ventricles": {
            "lv": ventricle_summary(lv_build),
            "rv": ventricle_summary(rv_build),
        },
        "tet_quality": quality,
        "per_region": per_region,
        "tetgen_seeds": {
            "lv": [float(x) for x in lv_seed],
            "rv": [float(x) for x in rv_seed],
        },
        "notes": [
            "Input LV/RV surfaces are LVendo/LVepi/RV/RVepi raw VTK files.",
            "The blood-pool basal cap is detected on endocardial surfaces before myocardial shell construction.",
            "The saved LV/RV bloodpool_basal_cap files show the maximum connected cap surfaces used to define the openings.",
            "The myocardium is still closed at the basal cut surface so TetGen can produce a valid myocardium volume mesh; the blood-pool side is open to exterior.",
            "Debug loops and clipped cap/open surfaces are saved beside the final VTK for visual inspection.",
        ],
    }
    summary_path = output_dir / f"{subject_dir.name}_{frame}_open_base_bloodpool_cap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "output_mesh": str(mesh_path),
        "summary": str(summary_path),
        "quality": quality,
        "per_region": per_region,
        "lv_cap": basal_cap_summary(lv_build.bloodpool_cap),
        "rv_cap": basal_cap_summary(rv_build.bloodpool_cap),
        "lv_base_side": lv_profile.base_side,
        "rv_base_side": rv_profile.base_side,
    }, indent=2))


if __name__ == "__main__":
    main()
