#!/usr/bin/env python3
"""Generate an open-base biventricular myocardium tetrahedral mesh for one subject.

This script uses the raw 4Dsegment ED/ES surface meshes:
  LVendo_<frame>.vtk, LVepi_<frame>.vtk, RV_<frame>.vtk, RVepi_<frame>.vtk

The default opening method is a connected basal-cap removal. Z-area profiles are used
only to decide which end is the anatomical base and to estimate the deepest basal
cap search depth. The actual opening boundary follows the connected cap patch, so
it is not constrained to a single flat clipping plane.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import meshio
import numpy as np
import pyvista as pv
import pymeshfix
import tetgen
import vtk


@dataclass
class AreaProfile:
    z_values: np.ndarray
    areas: np.ndarray
    z_min: float
    z_max: float
    low_end_area: float
    high_end_area: float
    base_side: str
    z_cut: float
    base_area: float
    apex_area: float
    area_threshold: float


@dataclass
class OpenSurfaceResult:
    mesh: pv.PolyData
    method: str
    removed_faces: int
    kept_faces: int
    removed_z_min: Optional[float]
    removed_z_max: Optional[float]
    effective_z_limit: Optional[float]
    normal_threshold: Optional[float]


@dataclass
class OpenShell:
    shell: pv.PolyData
    epi_open: pv.PolyData
    endo_open: pv.PolyData
    annulus: pv.PolyData
    epi_info: OpenSurfaceResult
    endo_info: OpenSurfaceResult
    loops: Dict[str, List[pv.PolyData]]


REGION_LEGEND = {
    2: "LV myocardium, blood pool open at base",
    4: "RV myocardium, blood pool open at base",
}


def load_surface(path: Path) -> pv.PolyData:
    if not path.exists():
        raise FileNotFoundError(path)
    mesh = pv.read(path).extract_surface().triangulate().clean(tolerance=1e-7)
    mesh = mesh.compute_normals(
        point_normals=True,
        cell_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
        inplace=False,
    )
    return mesh.extract_surface().triangulate().clean(tolerance=1e-7)


def triangle_faces(mesh: pv.PolyData) -> np.ndarray:
    faces = mesh.faces.reshape((-1, 4))
    if not np.all(faces[:, 0] == 3):
        mesh = mesh.triangulate()
        faces = mesh.faces.reshape((-1, 4))
    return faces[:, 1:4].astype(np.int64)


def faces_from_triangles(triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        return np.empty((0,), dtype=np.int64)
    prefix = np.full((len(triangles), 1), 3, dtype=np.int64)
    return np.hstack([prefix, triangles.astype(np.int64)]).ravel()


def convex_hull_area_xy(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    pts = pts[:, :2]
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    unique = [pts[0]]
    for point in pts[1:]:
        if np.linalg.norm(point - unique[-1]) > 1e-8:
            unique.append(point)
    pts = np.asarray(unique)
    if len(pts) < 3:
        return 0.0

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_area_profile(
    mesh: pv.PolyData,
    samples: int = 160,
    slab_fraction: float = 0.035,
    end_band_fraction: float = 0.12,
    stable_fraction: float = 0.45,
    override_base_side: Optional[str] = None,
    override_z_cut: Optional[float] = None,
) -> AreaProfile:
    pts = np.asarray(mesh.points)
    z = pts[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    height = max(z_max - z_min, 1e-6)
    slab = max(height * slab_fraction, height / max(samples, 1))
    z_values = np.linspace(z_min, z_max, samples)
    areas = []
    for center in z_values:
        mask = np.abs(z - center) <= slab / 2.0
        if int(mask.sum()) < 8:
            nearest = np.argsort(np.abs(z - center))[: min(24, len(z))]
            section_pts = pts[nearest]
        else:
            section_pts = pts[mask]
        areas.append(convex_hull_area_xy(section_pts))
    areas_arr = np.asarray(areas, dtype=float)

    band_count = max(3, int(round(samples * end_band_fraction)))
    low_end_area = float(np.nanmax(areas_arr[:band_count]))
    high_end_area = float(np.nanmax(areas_arr[-band_count:]))

    if override_base_side:
        base_side = override_base_side
    else:
        base_side = "high_z" if high_end_area >= low_end_area else "low_z"

    base_area = high_end_area if base_side == "high_z" else low_end_area
    apex_area = low_end_area if base_side == "high_z" else high_end_area
    area_threshold = max(base_area * stable_fraction, apex_area * 1.10, 1e-6)

    if override_z_cut is not None:
        z_cut = float(override_z_cut)
    elif base_side == "high_z":
        search = np.arange(samples - 1, -1, -1)
        fallback = max(0, samples - 1 - int(round(samples * 0.08)))
        selected = fallback
        for idx in search:
            if areas_arr[idx] >= area_threshold:
                selected = int(idx)
                break
        z_cut = float(z_values[selected])
    else:
        search = np.arange(samples)
        fallback = min(samples - 1, int(round(samples * 0.08)))
        selected = fallback
        for idx in search:
            if areas_arr[idx] >= area_threshold:
                selected = int(idx)
                break
        z_cut = float(z_values[selected])

    return AreaProfile(
        z_values=z_values,
        areas=areas_arr,
        z_min=z_min,
        z_max=z_max,
        low_end_area=low_end_area,
        high_end_area=high_end_area,
        base_side=base_side,
        z_cut=z_cut,
        base_area=float(base_area),
        apex_area=float(apex_area),
        area_threshold=float(area_threshold),
    )


def write_area_profile(path: Path, profile: AreaProfile) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["z", "xy_convex_hull_area", "base_side", "z_cut"])
        for z_value, area in zip(profile.z_values, profile.areas):
            writer.writerow([float(z_value), float(area), profile.base_side, profile.z_cut])


def subset_mesh_faces(mesh: pv.PolyData, keep_mask: np.ndarray) -> pv.PolyData:
    triangles = triangle_faces(mesh)
    kept = triangles[np.asarray(keep_mask, dtype=bool)]
    out = pv.PolyData(np.asarray(mesh.points).copy(), faces_from_triangles(kept))
    return out.extract_surface().triangulate().clean(tolerance=1e-7)


def face_adjacency(triangles: np.ndarray) -> List[List[int]]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for face_id, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_to_faces[edge].append(face_id)

    neighbors: List[List[int]] = [[] for _ in range(len(triangles))]
    for ids in edge_to_faces.values():
        if len(ids) < 2:
            continue
        for i in ids:
            for j in ids:
                if i != j:
                    neighbors[i].append(j)
    return neighbors


def remove_basal_cap_patch(
    mesh: pv.PolyData,
    profile: AreaProfile,
    seed_band_fraction: float = 0.035,
    cap_depth_scale: float = 1.35,
    normal_threshold: float = 0.15,
) -> OpenSurfaceResult:
    """Remove a connected cap patch from the anatomical base.

    The region starts at the basal extreme and grows across neighboring faces only
    inside the basal search depth. A normal filter keeps the boundary closer to the
    true cap-wall transition. If that filter is too restrictive for a surface, the
    function relaxes it to depth-only connected growth rather than returning a tiny
    opening.
    """
    mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    triangles = triangle_faces(mesh)
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
        effective_z_limit = z_max - detected_depth * cap_depth_scale
    else:
        distance_from_base = z_centers - z_min
        detected_depth = max(profile.z_cut - z_min, 0.0)
        base_normal_component = -normals[:, 2]
        effective_z_limit = z_min + detected_depth * cap_depth_scale

    seed_depth = max(seed_band_fraction * height, height / 80.0)
    max_depth = max(detected_depth * cap_depth_scale, seed_depth * 1.75)
    max_depth = min(max_depth, height * 0.45)

    if profile.base_side == "high_z":
        effective_z_limit = z_max - max_depth
    else:
        effective_z_limit = z_min + max_depth

    seed_mask = distance_from_base <= seed_depth
    depth_mask = distance_from_base <= max_depth

    min_patch_faces = max(8, int(round(len(triangles) * 0.003)))
    patch_mask = grow_connected_cap(triangles, seed_mask, depth_mask, base_normal_component, normal_threshold)
    active_threshold = normal_threshold

    if int(patch_mask.sum()) < min_patch_faces:
        patch_mask = grow_connected_cap(triangles, seed_mask, depth_mask, base_normal_component, -1.0)
        active_threshold = -1.0

    if int(patch_mask.sum()) >= len(triangles) * 0.55:
        tighter_depth = distance_from_base <= max(seed_depth * 2.25, detected_depth)
        patch_mask = grow_connected_cap(triangles, seed_mask, tighter_depth, base_normal_component, normal_threshold)
        active_threshold = normal_threshold

    if int(patch_mask.sum()) == 0:
        raise RuntimeError("Basal cap detection removed no faces; check surface orientation or Z rule parameters.")

    keep_mask = ~patch_mask
    opened = subset_mesh_faces(mesh, keep_mask)
    removed_z = z_centers[patch_mask]
    return OpenSurfaceResult(
        mesh=opened,
        method="connected_basal_cap_patch",
        removed_faces=int(patch_mask.sum()),
        kept_faces=int(keep_mask.sum()),
        removed_z_min=float(removed_z.min()),
        removed_z_max=float(removed_z.max()),
        effective_z_limit=float(effective_z_limit),
        normal_threshold=float(active_threshold),
    )


def grow_connected_cap(
    triangles: np.ndarray,
    seed_mask: np.ndarray,
    depth_mask: np.ndarray,
    base_normal_component: np.ndarray,
    normal_threshold: float,
) -> np.ndarray:
    accept_mask = depth_mask & (base_normal_component >= normal_threshold)
    accept_mask = accept_mask | seed_mask
    seeds = np.flatnonzero(seed_mask)
    neighbors = face_adjacency(triangles)
    visited = np.zeros(len(triangles), dtype=bool)
    queue: deque[int] = deque()

    for seed in seeds:
        if accept_mask[seed]:
            visited[seed] = True
            queue.append(int(seed))

    while queue:
        current = queue.popleft()
        for nb in neighbors[current]:
            if not visited[nb] and accept_mask[nb]:
                visited[nb] = True
                queue.append(int(nb))
    return visited


def clip_surface_at_base(mesh: pv.PolyData, profile: AreaProfile) -> OpenSurfaceResult:
    mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    triangles = triangle_faces(mesh)
    z_centers = np.asarray(mesh.points)[triangles].mean(axis=1)[:, 2]
    if profile.base_side == "high_z":
        keep_mask = z_centers < profile.z_cut
    else:
        keep_mask = z_centers > profile.z_cut
    opened = subset_mesh_faces(mesh, keep_mask)
    removed_z = z_centers[~keep_mask]
    return OpenSurfaceResult(
        mesh=opened,
        method="flat_z_cell_removal",
        removed_faces=int((~keep_mask).sum()),
        kept_faces=int(keep_mask.sum()),
        removed_z_min=float(removed_z.min()) if len(removed_z) else None,
        removed_z_max=float(removed_z.max()) if len(removed_z) else None,
        effective_z_limit=float(profile.z_cut),
        normal_threshold=None,
    )


def extract_boundary_edges(mesh: pv.PolyData) -> List[Tuple[int, int]]:
    triangles = triangle_faces(mesh)
    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_counts[edge] += 1
    return [edge for edge, count in edge_counts.items() if count == 1]


def boundary_loops(mesh: pv.PolyData) -> List[pv.PolyData]:
    edges = extract_boundary_edges(mesh)
    if not edges:
        return []

    adjacency: Dict[int, List[int]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    unused = {tuple(sorted(edge)) for edge in edges}
    loops: List[pv.PolyData] = []
    points = np.asarray(mesh.points)

    while unused:
        start_edge = next(iter(unused))
        unused.remove(start_edge)
        start, nxt = start_edge
        loop = [start, nxt]
        prev = start
        current = nxt

        for _ in range(len(edges) + 5):
            candidates = [node for node in adjacency[current] if node != prev]
            if not candidates:
                break
            chosen = None
            for cand in candidates:
                key = tuple(sorted((current, cand)))
                if key in unused:
                    chosen = cand
                    break
            if chosen is None:
                break
            unused.remove(tuple(sorted((current, chosen))))
            if chosen == start:
                break
            loop.append(chosen)
            prev, current = current, chosen

        if len(loop) >= 3:
            loop_points = points[np.asarray(loop, dtype=np.int64)]
            lines = np.hstack([[len(loop_points)], np.arange(len(loop_points), dtype=np.int64)]).astype(np.int64)
            loops.append(pv.PolyData(loop_points, lines=lines))

    loops.sort(key=lambda poly: poly.length, reverse=True)
    return loops


def select_boundary_loops(mesh: pv.PolyData, min_length_fraction: float = 0.05) -> List[pv.PolyData]:
    loops = boundary_loops(mesh)
    if not loops:
        raise RuntimeError("No boundary loop found after basal opening.")
    longest = max(loop.length for loop in loops)
    selected = [loop for loop in loops if loop.length >= longest * min_length_fraction]
    return selected or [loops[0]]


def signed_area_in_best_plane(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh[:2]
    uv = centered @ axes.T
    x = uv[:, 0]
    y = uv[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def orient_loop_points(points: np.ndarray, positive: bool) -> np.ndarray:
    area = signed_area_in_best_plane(points)
    if (area >= 0) != positive:
        return points[::-1].copy()
    return points.copy()


def make_polyline(points: np.ndarray) -> pv.PolyData:
    lines = np.hstack([[len(points)], np.arange(len(points), dtype=np.int64)]).astype(np.int64)
    return pv.PolyData(points, lines=lines)


def remove_duplicate_loop_endpoint(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) > 3 and np.linalg.norm(pts[0] - pts[-1]) < 1e-8:
        return pts[:-1].copy()
    return pts.copy()


def resample_closed_loop(points: np.ndarray, samples: int) -> np.ndarray:
    pts = remove_duplicate_loop_endpoint(points)
    closed = np.vstack([pts, pts[0]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg_lengths.sum())
    if total <= 1e-12:
        return np.repeat(pts[:1], samples, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    targets = np.linspace(0.0, total, samples, endpoint=False)
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(seg_lengths) - 1)
    local = (targets - cumulative[indices]) / np.maximum(seg_lengths[indices], 1e-12)
    return closed[indices] * (1.0 - local[:, None]) + closed[indices + 1] * local[:, None]


def rotate_loop(points: np.ndarray, shift: int) -> np.ndarray:
    pts = remove_duplicate_loop_endpoint(points)
    return np.roll(pts, -int(shift), axis=0)


def align_inner_loop_to_outer(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    outer_pts = remove_duplicate_loop_endpoint(outer)
    inner_pts = remove_duplicate_loop_endpoint(inner)
    sample_count = min(160, max(48, min(len(outer_pts), len(inner_pts))))
    outer_sample = resample_closed_loop(outer_pts, sample_count)

    best_score = math.inf
    best = inner_pts
    for candidate in (inner_pts, inner_pts[::-1].copy()):
        for shift in range(len(candidate)):
            shifted = rotate_loop(candidate, shift)
            shifted_sample = resample_closed_loop(shifted, sample_count)
            score = float(np.mean(np.linalg.norm(outer_sample - shifted_sample, axis=1)))
            if score < best_score:
                best_score = score
                best = shifted
    return best


def zipper_annulus(epi_loop: pv.PolyData, endo_loop: pv.PolyData) -> pv.PolyData:
    outer = remove_duplicate_loop_endpoint(np.asarray(epi_loop.points, dtype=float))
    inner = align_inner_loop_to_outer(outer, np.asarray(endo_loop.points, dtype=float))
    if len(outer) < 3 or len(inner) < 3:
        raise RuntimeError("Basal annulus loops must each contain at least three points.")

    n_outer = len(outer)
    n_inner = len(inner)
    points = np.vstack([outer, inner])
    triangles: List[List[int]] = []
    i = 0
    j = 0
    outer_steps = 0
    inner_steps = 0

    while outer_steps < n_outer or inner_steps < n_inner:
        next_i = (i + 1) % n_outer
        next_j = (j + 1) % n_inner
        if outer_steps >= n_outer:
            advance_outer = False
        elif inner_steps >= n_inner:
            advance_outer = True
        else:
            outer_advance_length = np.linalg.norm(outer[next_i] - inner[j])
            inner_advance_length = np.linalg.norm(outer[i] - inner[next_j])
            advance_outer = bool(outer_advance_length <= inner_advance_length)

        if advance_outer:
            triangles.append([i, next_i, n_outer + j])
            i = next_i
            outer_steps += 1
        else:
            triangles.append([i, n_outer + next_j, n_outer + j])
            j = next_j
            inner_steps += 1

    annulus = pv.PolyData(points, faces_from_triangles(np.asarray(triangles, dtype=np.int64)))
    return annulus.extract_surface().triangulate().clean(tolerance=1e-7)


def triangulate_annulus(epi_loops: Sequence[pv.PolyData], endo_loops: Sequence[pv.PolyData]) -> pv.PolyData:
    """Triangulate the myocardium basal annulus bounded by epi and endo loops."""
    if len(epi_loops) == 1 and len(endo_loops) == 1:
        return zipper_annulus(epi_loops[0], endo_loops[0])

    vtk_points = vtk.vtkPoints()
    vtk_lines = vtk.vtkCellArray()

    def add_loop(loop: pv.PolyData, positive: bool) -> None:
        pts = orient_loop_points(np.asarray(loop.points), positive=positive)
        ids = vtk.vtkIdList()
        first_id = None
        for point in pts:
            point_id = vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            if first_id is None:
                first_id = point_id
            ids.InsertNextId(point_id)
        if first_id is not None:
            ids.InsertNextId(first_id)
        vtk_lines.InsertNextCell(ids)

    # The largest epicardial loop acts as the outside contour. Endocardial loops are holes.
    for loop in epi_loops:
        add_loop(loop, positive=True)
    for loop in endo_loops:
        add_loop(loop, positive=False)

    contour = vtk.vtkPolyData()
    contour.SetPoints(vtk_points)
    contour.SetLines(vtk_lines)

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(contour)
    triangulator.Update()
    annulus = pv.wrap(triangulator.GetOutput()).extract_surface().triangulate().clean(tolerance=1e-7)
    if annulus.n_cells == 0:
        raise RuntimeError("Failed to triangulate basal annulus from boundary loops.")
    return annulus


def merge_surfaces(parts: Sequence[pv.PolyData]) -> pv.PolyData:
    merged = parts[0].copy()
    for part in parts[1:]:
        merged = merged.merge(part, merge_points=True, tolerance=1e-6)
    return merged.extract_surface().triangulate().clean(tolerance=1e-7)


def make_open_shell(
    epi: pv.PolyData,
    endo: pv.PolyData,
    profile: AreaProfile,
    opening_method: str,
    seed_band_fraction: float,
    cap_depth_scale: float,
    normal_threshold: float,
) -> OpenShell:
    if opening_method == "z_clip":
        epi_info = clip_surface_at_base(epi, profile)
        endo_info = clip_surface_at_base(endo, profile)
    elif opening_method == "cap_patch":
        epi_info = remove_basal_cap_patch(epi, profile, seed_band_fraction, cap_depth_scale, normal_threshold)
        endo_info = remove_basal_cap_patch(endo, profile, seed_band_fraction, cap_depth_scale, normal_threshold)
    else:
        raise ValueError(f"Unknown opening method: {opening_method}")

    epi_loops = select_boundary_loops(epi_info.mesh)
    endo_loops = select_boundary_loops(endo_info.mesh)
    annulus = triangulate_annulus(epi_loops, endo_loops)
    shell = merge_surfaces([epi_info.mesh, endo_info.mesh, annulus])

    free_edges = extract_boundary_edges(shell)
    if free_edges:
        raise RuntimeError(f"Constructed shell still has {len(free_edges)} free boundary edges before repair.")

    shell = repair_closed_surface(shell)
    free_edges = extract_boundary_edges(shell)
    if free_edges:
        raise RuntimeError(f"Constructed shell still has {len(free_edges)} free boundary edges after repair.")

    return OpenShell(
        shell=shell,
        epi_open=epi_info.mesh,
        endo_open=endo_info.mesh,
        annulus=annulus,
        epi_info=epi_info,
        endo_info=endo_info,
        loops={"epi": epi_loops, "endo": endo_loops},
    )


def repair_closed_surface(mesh: pv.PolyData) -> pv.PolyData:
    faces = triangle_faces(mesh)
    fixer = pymeshfix.MeshFix(np.asarray(mesh.points), faces)
    fixer.repair(joincomp=True, remove_smallest_components=False)
    fixer.clean(max_iters=60, inner_loops=10)
    fixer.intersection_removal(max_iter=10)
    fixer.degeneracy_removal(max_iter=10)
    repaired = pv.PolyData(fixer.points, faces_from_triangles(fixer.faces))
    return repaired.extract_surface().triangulate().clean(tolerance=1e-7)


def surface_summary(mesh: pv.PolyData) -> Dict[str, object]:
    return {
        "points": int(mesh.n_points),
        "triangles": int(mesh.n_cells),
        "bounds": [float(v) for v in mesh.bounds],
        "surface_area": float(mesh.area),
        "volume_if_closed": float(mesh.volume) if mesh.is_all_triangles else None,
    }


def open_surface_summary(result: OpenSurfaceResult) -> Dict[str, object]:
    data = surface_summary(result.mesh)
    data.update(
        {
            "opening_method": result.method,
            "removed_faces": result.removed_faces,
            "kept_faces": result.kept_faces,
            "removed_z_min": result.removed_z_min,
            "removed_z_max": result.removed_z_max,
            "effective_z_limit": result.effective_z_limit,
            "normal_threshold": result.normal_threshold,
        }
    )
    return data


def enclosed_mask(points: np.ndarray, shell: pv.PolyData) -> np.ndarray:
    pdata = pv.PolyData(points)
    selected = pdata.select_enclosed_points(shell, tolerance=1e-5, check_surface=False)
    return selected.point_data["SelectedPoints"].astype(bool)


def find_shell_seed(outer: pv.PolyData, inner: pv.PolyData, shell: pv.PolyData, samples: int = 8000) -> np.ndarray:
    bounds = np.asarray(shell.bounds, dtype=float)
    mins = bounds[[0, 2, 4]]
    maxs = bounds[[1, 3, 5]]
    rng = np.random.default_rng(20260818)
    candidates = rng.uniform(mins, maxs, size=(samples, 3))
    inside_outer = enclosed_mask(candidates, outer)
    inside_inner = enclosed_mask(candidates, inner)
    inside_shell = enclosed_mask(candidates, shell)
    mask = inside_outer & ~inside_inner & inside_shell
    if np.any(mask):
        return candidates[np.flatnonzero(mask)[0]]

    center = np.asarray(shell.center, dtype=float)
    candidate_cloud = np.vstack([center, np.asarray(shell.points)[:: max(1, shell.n_points // 500)]])
    inside_shell = enclosed_mask(candidate_cloud, shell)
    if np.any(inside_shell):
        return candidate_cloud[np.flatnonzero(inside_shell)[0]]
    raise RuntimeError("Could not find a valid myocardial seed point for TetGen.")


def tetrahedralize_shell(shell: pv.PolyData, seed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tg = tetgen.TetGen(shell)
    tg.add_region(1, seed.tolist(), 0.0)
    result = tg.tetrahedralize(
        plc=True,
        quality=True,
        minratio=1.2,
        mindihedral=5.0,
        nobisect=False,
        regionattrib=True,
        switches="pq1.2/5AaY",
        verbose=0,
    )
    nodes = result[0]
    elems = result[1]
    return np.asarray(nodes, dtype=float), np.asarray(elems, dtype=np.int64)


def combine_tet_meshes(parts: Sequence[Tuple[np.ndarray, np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_all: List[np.ndarray] = []
    cells_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    offset = 0
    for points, cells, label in parts:
        points_all.append(points)
        cells_all.append(cells[:, :4] + offset)
        labels_all.append(np.full(len(cells), int(label), dtype=np.int32))
        offset += len(points)
    return np.vstack(points_all), np.vstack(cells_all), np.concatenate(labels_all)


def tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    a = points[cells[:, 0]]
    b = points[cells[:, 1]]
    c = points[cells[:, 2]]
    d = points[cells[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def tet_quality(points: np.ndarray, cells: np.ndarray) -> Dict[str, float]:
    vols = tet_volumes(points, cells)
    p = points[cells]
    edges = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        edges.append(np.linalg.norm(p[:, i] - p[:, j], axis=1))
    edge_arr = np.vstack(edges).T
    min_edge = np.maximum(edge_arr.min(axis=1), 1e-12)
    aspect_proxy = edge_arr.max(axis=1) / min_edge
    return {
        "tetrahedra": int(len(cells)),
        "total_abs_volume": float(np.abs(vols).sum()),
        "negative_or_zero_signed_tets": int(np.count_nonzero(vols <= 0)),
        "min_abs_tet_volume": float(np.abs(vols).min()) if len(vols) else 0.0,
        "median_abs_tet_volume": float(np.median(np.abs(vols))) if len(vols) else 0.0,
        "max_edge_ratio_proxy": float(aspect_proxy.max()) if len(aspect_proxy) else 0.0,
        "p95_edge_ratio_proxy": float(np.percentile(aspect_proxy, 95)) if len(aspect_proxy) else 0.0,
    }


def save_vtk(path: Path, points: np.ndarray, cells: np.ndarray, labels: np.ndarray) -> None:
    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", cells.astype(np.int64))],
        cell_data={"region_id": [labels.astype(np.int32)]},
    )
    mesh.write(path)


def write_region_legend(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "description"])
        for region_id, description in REGION_LEGEND.items():
            writer.writerow([region_id, description])


def profile_to_dict(profile: AreaProfile) -> Dict[str, object]:
    return {
        "z_min": profile.z_min,
        "z_max": profile.z_max,
        "low_end_area": profile.low_end_area,
        "high_end_area": profile.high_end_area,
        "base_side": profile.base_side,
        "z_cut": profile.z_cut,
        "base_area": profile.base_area,
        "apex_area": profile.apex_area,
        "area_threshold": profile.area_threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--opening-method", choices=["cap_patch", "z_clip"], default="cap_patch")
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--slab-fraction", type=float, default=0.035)
    parser.add_argument("--end-band-fraction", type=float, default=0.12)
    parser.add_argument("--stable-fraction", type=float, default=0.45)
    parser.add_argument("--cap-seed-band-fraction", type=float, default=0.035)
    parser.add_argument("--cap-depth-scale", type=float, default=1.35)
    parser.add_argument("--cap-normal-threshold", type=float, default=0.15)
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
    output_root = args.output_root or subject_dir / "volmesh_open_base"
    output_dir = output_root / f"{subject_dir.name}_{frame}_open_base_{args.opening_method}_zrule_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    input_paths = {
        "lv_endo": vtk_dir / f"LVendo_{frame}.vtk",
        "lv_epi": vtk_dir / f"LVepi_{frame}.vtk",
        "rv_endo": vtk_dir / f"RV_{frame}.vtk",
        "rv_epi": vtk_dir / f"RVepi_{frame}.vtk",
    }

    surfaces = {name: load_surface(path) for name, path in input_paths.items()}

    lv_profile = compute_area_profile(
        surfaces["lv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=args.lv_base_side,
        override_z_cut=args.lv_z_cut,
    )
    rv_profile = compute_area_profile(
        surfaces["rv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=args.rv_base_side,
        override_z_cut=args.rv_z_cut,
    )

    write_area_profile(output_dir / f"{subject_dir.name}_{frame}_LV_z_area_profile.csv", lv_profile)
    write_area_profile(output_dir / f"{subject_dir.name}_{frame}_RV_z_area_profile.csv", rv_profile)

    lv_open = make_open_shell(
        epi=surfaces["lv_epi"],
        endo=surfaces["lv_endo"],
        profile=lv_profile,
        opening_method=args.opening_method,
        seed_band_fraction=args.cap_seed_band_fraction,
        cap_depth_scale=args.cap_depth_scale,
        normal_threshold=args.cap_normal_threshold,
    )
    rv_open = make_open_shell(
        epi=surfaces["rv_epi"],
        endo=surfaces["rv_endo"],
        profile=rv_profile,
        opening_method=args.opening_method,
        seed_band_fraction=args.cap_seed_band_fraction,
        cap_depth_scale=args.cap_depth_scale,
        normal_threshold=args.cap_normal_threshold,
    )

    lv_seed = find_shell_seed(surfaces["lv_epi"], surfaces["lv_endo"], lv_open.shell)
    rv_seed = find_shell_seed(surfaces["rv_epi"], surfaces["rv_endo"], rv_open.shell)

    lv_points, lv_cells = tetrahedralize_shell(lv_open.shell, lv_seed)
    rv_points, rv_cells = tetrahedralize_shell(rv_open.shell, rv_seed)
    points, cells, labels = combine_tet_meshes([(lv_points, lv_cells, 2), (rv_points, rv_cells, 4)])

    mesh_path = output_dir / f"{subject_dir.name}_{frame}_biventricular_open_base_volmesh.vtk"
    save_vtk(mesh_path, points, cells, labels)
    write_region_legend(output_dir / f"{subject_dir.name}_{frame}_open_base_region_legend.csv")

    debug_surfaces = {
        f"LVendo_{frame}_open.vtk": lv_open.endo_open,
        f"LVepi_{frame}_open.vtk": lv_open.epi_open,
        f"RV_{frame}_open.vtk": rv_open.endo_open,
        f"RVepi_{frame}_open.vtk": rv_open.epi_open,
        f"LV_basal_annulus_{frame}.vtk": lv_open.annulus,
        f"RV_basal_annulus_{frame}.vtk": rv_open.annulus,
        f"LV_myocardium_boundary_{frame}.vtk": lv_open.shell,
        f"RV_myocardium_boundary_{frame}.vtk": rv_open.shell,
    }
    for name, surface in debug_surfaces.items():
        surface.save(output_dir / name)

    quality = tet_quality(points, cells)
    per_region = {}
    for region_id in sorted(REGION_LEGEND):
        mask = labels == region_id
        per_region[str(region_id)] = {
            "description": REGION_LEGEND[region_id],
            "tetrahedra": int(mask.sum()),
            "abs_volume": float(np.abs(tet_volumes(points, cells[mask])).sum()),
        }

    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "opening_method": args.opening_method,
        "output_dir": str(output_dir.resolve()),
        "output_mesh": str(mesh_path.resolve()),
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "z_rule": {
            "lv": profile_to_dict(lv_profile),
            "rv": profile_to_dict(rv_profile),
        },
        "cap_patch_parameters": {
            "seed_band_fraction": args.cap_seed_band_fraction,
            "cap_depth_scale": args.cap_depth_scale,
            "normal_threshold": args.cap_normal_threshold,
        },
        "open_surface_summaries": {
            "lv_endo_open": open_surface_summary(lv_open.endo_info),
            "lv_epi_open": open_surface_summary(lv_open.epi_info),
            "rv_endo_open": open_surface_summary(rv_open.endo_info),
            "rv_epi_open": open_surface_summary(rv_open.epi_info),
        },
        "shell_summaries": {
            "lv_shell": surface_summary(lv_open.shell),
            "rv_shell": surface_summary(rv_open.shell),
            "lv_annulus": surface_summary(lv_open.annulus),
            "rv_annulus": surface_summary(rv_open.annulus),
        },
        "tet_quality": quality,
        "per_region": per_region,
        "tetgen_seeds": {
            "lv": [float(x) for x in lv_seed],
            "rv": [float(x) for x in rv_seed],
        },
        "notes": [
            "Input surfaces are the raw LV/RV VTK files, not N_LV prefixed variants.",
            "The mesh contains myocardium regions only; blood pools are intentionally open to the exterior at the base.",
            "Z profiles determine basal side by comparing end-area proxies; connected cap removal defines the final opening boundary.",
            "The closed myocardial boundary is repaired with MeshFix before TetGen to remove local annulus self-intersections.",
            "Debug open surfaces and annulus surfaces are saved beside the volume mesh for inspection.",
        ],
    }
    summary_path = output_dir / f"{subject_dir.name}_{frame}_open_base_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "output_mesh": str(mesh_path),
        "summary": str(summary_path),
        "quality": quality,
        "per_region": per_region,
        "z_rule": summary["z_rule"],
        "open_surface_summaries": summary["open_surface_summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
