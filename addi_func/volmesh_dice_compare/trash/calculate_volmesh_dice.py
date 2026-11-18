#!/usr/bin/env python3
"""Compare complete tetrahedral volumes after rigid ICP or landmark registration.

Example (from addi_func):
    conda run --no-capture-output -n cme-dt-env python \
        volmesh_dice_compare/calculate_volmesh_dice.py --case-dir volmesh_dice_compare/1

The source is the moving volume; the reference defines the fixed comparison frame.
Landmark mode moves subject_11 to subject0081, fixes the LV endo apex exactly, and
jointly minimizes squared mitral/tricuspid landmark errors using rotation only.
Landmark sidecars define registration coordinates, never the Dice volumes.
Auto clipping detects large, planar, supporting caps on the REFERENCE only.
It never deletes source tissue because it fails to overlap the reference. Automatic
cap detection is a geometric hypothesis, not semantic valve segmentation: inspect
comparison.png and basal_planes.vtp before interpreting the conditional Dice.

All coordinates and --spacing/--plane-offset-mm values are assumed to be in mm.
Numeric tags correspond by value unless --label-map is supplied; tag 0 is tissue,
not background. Output label codes start at 1, with 0 reserved for outside tissue.
Only geometry and comparison labels are copied to output meshes. Inputs are read
only. Dependencies: numpy, scipy, pyvista/VTK, matplotlib (no GUI required).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def log(message):
    print(message, flush=True)


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def validate_rigid(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6)):
        raise ValueError("Transform must contain a proper rotation and translation, with no scaling/reflection.")
    return matrix


def read_volume(path, label_array):
    mesh = pv.read(path)
    if not isinstance(mesh, pv.UnstructuredGrid) or not mesh.n_cells:
        raise ValueError(f"{path}: expected a nonempty unstructured tetrahedral grid")
    if np.any(mesh.celltypes != pv.CellType.TETRA):
        raise ValueError(f"{path}: input must contain only tetrahedra")
    if not np.isfinite(mesh.points).all():
        raise ValueError(f"{path}: nonfinite coordinates")
    if label_array not in mesh.cell_data:
        raise ValueError(f"{path}: missing cell label array {label_array!r}")
    labels = np.asarray(mesh.cell_data[label_array])
    if labels.ndim != 1 or not np.isfinite(labels).all() or not np.equal(labels, np.round(labels)).all():
        raise ValueError("Cell labels must be scalar integers")
    labels = labels.astype(np.int64)
    mesh.clear_data()
    mesh.cell_data["elemTag"] = labels
    mesh.cell_data["original_cell_id"] = np.arange(mesh.n_cells, dtype=np.int64)
    return mesh


def volume_stats(mesh):
    if not mesh.n_cells:
        return {"cells": 0, "volume_mm3": 0.0, "tag_volumes_mm3": {}}
    volumes = np.asarray(mesh.compute_cell_sizes(length=False, area=False)["Volume"])
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Mesh contains cells with nonpositive or nonfinite volume")
    tags = np.asarray(mesh.cell_data["elemTag"])
    return {"cells": mesh.n_cells, "points": mesh.n_points, "bounds_mm": list(mesh.bounds),
            "volume_mm3": float(volumes.sum()),
            "tag_volumes_mm3": {str(t): float(volumes[tags == t].sum()) for t in np.unique(tags)}}


def surface_arrays(mesh):
    surface = mesh.extract_surface().triangulate()
    triangles = np.asarray(surface.points, dtype=float)[surface.faces.reshape(-1, 4)[:, 1:]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    valid = length > 1e-12
    return (surface, triangles[valid], length[valid] / 2,
            cross[valid] / length[valid, None], np.asarray(surface.cell_data["elemTag"])[valid])


def detect_basal_planes(mesh, min_area_fraction=0.015, tolerance=0.05):
    """Find large, truly planar support patches; never inspect the source overlap."""
    _, triangles, area, normals, _ = surface_arrays(mesh)
    centers = triangles.mean(axis=1)
    major = np.argmax(np.abs(normals), axis=1)
    sign = np.where(normals[np.arange(len(normals)), major] < 0, -1, 1)
    canonical = normals * sign[:, None]
    offsets = np.einsum("ij,ij->i", canonical, centers)
    keys = np.column_stack([np.round(canonical, 3), np.round(offsets, 1)])
    _, groups = np.unique(keys, axis=0, return_inverse=True)
    group_area = np.bincount(groups, weights=area)
    planes = []
    for group in np.argsort(group_area)[::-1]:
        if group_area[group] < min_area_fraction * area.sum():
            break
        selected = groups == group
        if selected.sum() < 15:
            continue
        vertices = triangles[selected].reshape(-1, 3)
        center = np.average(centers[selected], axis=0, weights=area[selected])
        _, _, vt = np.linalg.svd(vertices - center, full_matrices=False)
        normal = vt[-1]
        if np.median((mesh.points - center) @ normal) > 0:
            normal = -normal
        offset = float(center @ normal)
        residual = np.abs((vertices - center) @ normal)
        if residual.max() > tolerance or np.max(mesh.points @ normal - offset) > tolerance:
            continue
        if any(np.dot(normal, p["normal"]) > 0.99999 and abs(offset - p["offset_mm"]) < tolerance for p in planes):
            continue
        planes.append({"normal": normal.tolist(), "offset_mm": offset,
                       "origin_mm": (normal * offset).tolist(),
                       "reference_patch_area_mm2": float(group_area[group]),
                       "reference_surface_area_fraction": float(group_area[group] / area.sum()),
                       "planarity_max_error_mm": float(residual.max()),
                       "method": "reference_planar_support_patch"})
    if not planes:
        raise ValueError("No large supporting basal cap found. Supply --plane NX NY NZ D, or use --no-trim.")
    if len(planes) > 1 and planes[0]["reference_patch_area_mm2"] < 3 * planes[1]["reference_patch_area_mm2"]:
        raise ValueError("Multiple substantial supporting planes: basal anatomy is ambiguous. Supply explicit --plane boundaries.")
    # A single dominant cap is identifiable in these inputs. Other flat patches
    # are not automatically assumed to be anatomical truncation boundaries.
    return planes[:1]


def normalized_plane(values):
    normal = np.asarray(values[:3], dtype=float)
    norm = np.linalg.norm(normal)
    if not np.isfinite(values).all() or norm < 1e-12:
        raise ValueError("Plane requires a finite nonzero normal and finite offset")
    normal /= norm
    offset = float(values[3] / norm)
    return {"normal": normal.tolist(), "offset_mm": offset,
            "origin_mm": (normal * offset).tolist(), "method": "explicit_plane"}


def sample_surface(mesh, count, seed, exclude_planes=()):
    _, triangles, areas, normals, tags = surface_arrays(mesh)
    eligible = np.ones(len(areas), dtype=bool)
    for plane in exclude_planes:
        distances = triangles @ np.asarray(plane["normal"]) - plane["offset_mm"]
        eligible &= np.max(np.abs(distances), axis=1) > 0.06
    ids = np.flatnonzero(eligible)
    if not len(ids):
        raise ValueError("No surface available for registration")
    rng = np.random.default_rng(seed)
    selected = rng.choice(ids, count, p=areas[ids] / areas[ids].sum())
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    xyz = triangles[selected]
    points = xyz[:, 0] + uv[:, :1] * (xyz[:, 1] - xyz[:, 0]) + uv[:, 1:] * (xyz[:, 2] - xyz[:, 0])
    return points, normals[selected], tags[selected]


def label_trees(points, labels):
    return {int(tag): (np.flatnonzero(labels == tag), cKDTree(points[labels == tag])) for tag in np.unique(labels)}


def matches(source, source_normals, source_tags, target, target_normals, target_tags, target_trees,
            trim_fraction, max_distance=np.inf, normal_compatibility=False):
    """Symmetric, tag-constrained correspondences, trimmed for alignment ONLY."""
    source_trees = label_trees(source, source_tags)
    parts = []
    for tag in sorted(set(source_trees) & set(target_trees)):
        si, stree = source_trees[tag]
        ti, ttree = target_trees[tag]
        for reverse in (False, True):
            query = target[ti] if reverse else source[si]
            tree = stree if reverse else ttree
            lookup = si if reverse else ti
            dataset_normals = source_normals if reverse else target_normals
            query_normals = target_normals[ti] if reverse else source_normals[si]
            if normal_compatibility:
                d, nn = tree.query(query, k=min(12, len(lookup)))
                if d.ndim == 1:
                    d, nn = d[:, None], nn[:, None]
                compatible = np.einsum("ijk,ik->ij", dataset_normals[lookup[nn]], query_normals) > 0.25
                d[~compatible] = np.inf
                chosen = np.argmin(d, axis=1)
                d, nn = d[np.arange(len(d)), chosen], nn[np.arange(len(nn)), chosen]
            else:
                d, nn = tree.query(query)
            if reverse:
                ii, jj, normal = si[nn], ti, source_normals[si[nn]]
            else:
                ii, jj, normal = si, ti[nn], target_normals[ti[nn]]
            finite = np.isfinite(d)
            if not finite.any():
                continue
            limit = min(float(np.quantile(d[finite], trim_fraction)), max_distance)
            keep = finite & (d <= limit)
            if keep.sum() >= 6:
                # Equalize tags/directions to avoid the larger chamber dominating.
                weight = np.full(keep.sum(), 1.0 / keep.sum())
                parts.append((source[ii[keep]], target[jj[keep]], normal[keep], d[keep], weight))
    if not parts:
        raise ValueError("Insufficient registration correspondences; check tag mapping and initial alignment")
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(5))


def kabsch(source, target, weight):
    weight = weight / weight.sum()
    cs, ct = weight @ source, weight @ target
    u, _, vt = np.linalg.svd(((source - cs) * weight[:, None]).T @ (target - ct))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = ct - rotation @ cs
    return matrix


def icp(source_data, target_data, initial, iterations, trim_fraction, point_to_plane=False, max_distance=np.inf,
        planes=()):
    sp, sn, sl = source_data
    tp, tn, tl = target_data
    trees = label_trees(tp, tl)
    matrix = initial.copy()
    for _ in range(iterations):
        moved = transform_points(sp, matrix)
        normals = sn @ matrix[:3, :3].T
        active = np.ones(len(moved), dtype=bool)
        for plane in planes:
            active &= moved @ np.asarray(plane["normal"]) <= plane["offset_mm"] + 2.0
        p, q, n, d, weights = matches(moved[active], normals[active], sl[active], tp, tn, tl, trees,
                                     trim_fraction, max_distance, normal_compatibility=point_to_plane)
        if point_to_plane:
            center = p.mean(axis=0)
            residual = np.einsum("ij,ij->i", p - q, n)
            weights *= np.minimum(1.0, 1.0 / np.maximum(np.abs(residual), 1e-12))
            design = np.column_stack([np.cross(p - center, n), n])
            weighted = design * np.sqrt(weights[:, None])
            step, *_ = np.linalg.lstsq(weighted, -residual * np.sqrt(weights), rcond=None)
            step *= min(1.0, 0.15 / max(np.linalg.norm(step[:3]), 1e-12),
                        5.0 / max(np.linalg.norm(step[3:]), 1e-12))
            rotation = Rotation.from_rotvec(step[:3]).as_matrix()
            delta = np.eye(4)
            delta[:3, :3] = rotation
            delta[:3, 3] = center + step[3:] - rotation @ center
        else:
            weights *= np.minimum(1.0, 3.0 / np.maximum(d, 1e-12))
            delta = kabsch(p, q, weights)
        matrix = delta @ matrix
        if np.linalg.norm(delta[:3, :3] - np.eye(3)) < 1e-7 and np.linalg.norm(delta[:3, 3]) < 1e-5:
            break
    return validate_rigid(matrix)


def alignment_score(source_data, target_data, matrix, fraction, planes=()):
    sp, sn, sl = source_data
    tp, tn, tl = target_data
    moved = transform_points(sp, matrix)
    active = np.ones(len(moved), dtype=bool)
    for plane in planes:
        active &= moved @ np.asarray(plane["normal"]) <= plane["offset_mm"] + 2.0
    _, _, _, distances, weights = matches(moved[active], (sn @ matrix[:3, :3].T)[active],
                                         sl[active], tp, tn, tl, label_trees(tp, tl), fraction,
                                         normal_compatibility=bool(planes))
    return float(np.average(distances, weights=weights))


def initial_alignments(source, target):
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    _, bs = np.linalg.eigh(np.cov(source.T))
    _, bt = np.linalg.eigh(np.cov(target.T))
    initial = np.eye(4)
    initial[:3, 3] = ct - cs
    yield initial
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = (bt[:, order] * np.asarray(signs)) @ bs.T
            if np.linalg.det(rotation) > 0:
                initial = np.eye(4)
                initial[:3, :3] = rotation
                initial[:3, 3] = ct - rotation @ cs
                yield initial


def register(source_data, target_data, fraction, planes=()):
    coarse_s = tuple(x[:2500] for x in source_data)
    coarse_t = tuple(x[:2500] for x in target_data)
    candidates = []
    for index, initial in enumerate(initial_alignments(coarse_s[0], coarse_t[0])):
        matrix = icp(coarse_s, coarse_t, initial, 35, fraction)
        score = alignment_score(coarse_s, coarse_t, matrix, fraction)
        candidates.append((score, index, matrix))
    candidates.sort(key=lambda item: item[0])
    refined = []
    for _, index, matrix in candidates[:8]:
        matrix = icp(source_data, target_data, matrix, 25, fraction, planes=planes)
        matrix = icp(source_data, target_data, matrix, 60, fraction, point_to_plane=True, max_distance=10.0, planes=planes)
        refined.append((alignment_score(source_data, target_data, matrix, fraction, planes), index, matrix))
    score, index, matrix = min(refined, key=lambda item: item[0])
    return matrix, {"method": "25_initializations_symmetric_tag_constrained_trimmed_ICP_then_Huber_point_to_plane",
                    "trim_fraction_for_alignment_only": fraction,
                    "source_registration_region": "reference basal halfspaces plus 2 mm margin" if planes else "full surface",
                    "normal_compatibility_in_refinement": "dot(source_normal, reference_normal) > 0.25",
                    "mean_trimmed_symmetric_surface_distance_mm": score,
                    "selected_initialization": index,
                    "coarse_candidates": [{"initialization": i, "distance_mm": s} for s, i, _ in candidates],
                    "refined_candidates": [{"initialization": i, "distance_mm": s} for s, i, _ in sorted(refined, key=lambda x: x[0])],
                    "rotation_determinant": float(np.linalg.det(matrix[:3, :3]))}


LANDMARK_NAMES = ["lv_endo_apex", "lv_mitral_point", "rv_tricuspid_point"]
LANDMARK_FILES = {
    "fixed_lv_endo": "Mesh_LV_endo_timeframe001.vtk",
    "fixed_aorta_valve": "Mesh_aorta_valve_timeframe001.vtk",
    "fixed_tricuspid": "Mesh_tricuspid_valve_timeframe001.vtk",
    "moving_apex": "subject_11_ED_apex_point.vtk",
    "moving_lv_rim": "subject_11_ED_LV_opening_rim.vtk",
    "moving_rv_rim": "subject_11_ED_RV_opening_rim.vtk",
}


def referenced_surface(mesh):
    """Compact only polygon-referenced points; named VTK surfaces can share a full point table."""
    if not isinstance(mesh, pv.PolyData):
        raise ValueError("Landmark surfaces must be PolyData")
    triangulated = mesh.triangulate()
    if not len(triangulated.faces):
        raise ValueError("Landmark surface has no polygon faces")
    faces = triangulated.faces.reshape(-1, 4)[:, 1:]
    ids, inverse = np.unique(faces, return_inverse=True)
    points = np.asarray(triangulated.points, dtype=float)[ids]
    if not np.isfinite(points).all():
        raise ValueError("Landmark surface has nonfinite referenced points")
    compact_faces = np.column_stack([np.full(len(faces), 3), inverse.reshape(-1, 3)])
    surface = pv.PolyData(points, compact_faces.ravel())
    return surface, ids


def valve_centroid(mesh):
    surface, ids = referenced_surface(mesh)
    triangles = surface.points[surface.faces.reshape(-1, 4)[:, 1:]]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or area.sum() <= 1e-12:
        raise ValueError("Valve surface has no positive area")
    center = np.average(triangles.mean(axis=1), axis=0, weights=area)
    return center, {"method": "triangle_area_weighted_centroid", "area_mm2": float(area.sum()),
                    "stored_points": mesh.n_points, "referenced_vertices": len(ids)}


def rim_centroid(mesh):
    if not isinstance(mesh, pv.PolyData):
        raise ValueError("Opening rim must be PolyData with line connectivity")
    connectivity = mesh.lines
    segments, offset = [], 0
    while offset < len(connectivity):
        count = int(connectivity[offset])
        ids = connectivity[offset + 1:offset + count + 1]
        if count >= 2:
            segments.extend(zip(ids[:-1], ids[1:]))
        offset += count + 1
    if not segments:
        raise ValueError("Opening rim has no line segments")
    segments = np.asarray(segments, dtype=int)
    endpoints = np.asarray(mesh.points, dtype=float)[segments]
    if not np.isfinite(endpoints).all():
        raise ValueError("Opening rim contains nonfinite referenced points")
    length = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
    if length.sum() <= 1e-12:
        raise ValueError("Opening rim has zero length")
    center = np.average(endpoints.mean(axis=1), axis=0, weights=length)
    return center, {"method": "arc_length_weighted_centroid", "length_mm": float(length.sum()),
                    "stored_points": mesh.n_points, "referenced_vertices": len(np.unique(segments))}


def endo_apex(endo_mesh, aorta_mesh):
    endo, original_ids = referenced_surface(endo_mesh)
    valve, _ = referenced_surface(aorta_mesh)
    _, closest = valve.find_closest_cell(endo.points, return_closest_point=True)
    distances = np.linalg.norm(endo.points - closest, axis=1)
    index = int(np.argmax(distances))
    return np.asarray(endo.points[index]), {
        "method": "referenced_LV_endo_vertex_maximizing_minimum_distance_to_aorta_triangles",
        "original_point_id": int(original_ids[index]), "distance_to_valve_surface_mm": float(distances[index]),
        "closest_valve_surface_point_mm": np.asarray(closest[index]).tolist(),
        "stored_points": endo_mesh.n_points, "referenced_vertices": endo.n_points,
    }


def extract_landmarks(directory):
    paths = {key: (directory / filename).resolve() for key, filename in LANDMARK_FILES.items()}
    meshes = {key: pv.read(path) for key, path in paths.items()}
    apex_mesh = meshes["moving_apex"]
    if apex_mesh.n_points != 1 or not np.isfinite(apex_mesh.points).all():
        raise ValueError("apex_point must contain exactly one finite point")
    fixed_apex, apex_info = endo_apex(meshes["fixed_lv_endo"], meshes["fixed_aorta_valve"])
    fixed_mitral, mitral_info = valve_centroid(meshes["fixed_aorta_valve"])
    fixed_tricuspid, tricuspid_info = valve_centroid(meshes["fixed_tricuspid"])
    moving_mitral, lv_info = rim_centroid(meshes["moving_lv_rim"])
    moving_tricuspid, rv_info = rim_centroid(meshes["moving_rv_rim"])
    moving = np.vstack([apex_mesh.points[0], moving_mitral, moving_tricuspid]).astype(float)
    fixed = np.vstack([fixed_apex, fixed_mitral, fixed_tricuspid]).astype(float)
    metadata = {"names": LANDMARK_NAMES, "moving_subject": "subject_11", "fixed_subject": "subject0081",
                "files": {key: str(path) for key, path in paths.items()},
                "moving_points_mm": moving.tolist(), "fixed_points_mm": fixed.tolist(),
                "extraction": {"fixed_lv_endo_apex": apex_info, "fixed_lv_mitral_point": mitral_info,
                               "fixed_rv_tricuspid_point": tricuspid_info,
                               "moving_lv_endo_apex": {"method": "apex_point_coordinate"},
                               "moving_lv_mitral_point": lv_info, "moving_rv_tricuspid_point": rv_info},
                "lv_mitral_definition": "Fixed point is the aorta_valve centroid as explicitly requested."}
    return moving, fixed, metadata


def register_anchored_landmarks(moving, fixed):
    """Solve min_R sum_i ||R(p_i-apex_m)-(q_i-apex_f)||^2, with no centering/scaling."""
    moving, fixed = np.asarray(moving, dtype=float), np.asarray(fixed, dtype=float)
    if moving.shape != (3, 3) or fixed.shape != (3, 3) or not np.isfinite([moving, fixed]).all():
        raise ValueError("Expected three finite landmark pairs in order: apex, mitral, tricuspid")
    relative_moving, relative_fixed = moving[1:] - moving[0], fixed[1:] - fixed[0]
    for vectors in (relative_moving, relative_fixed):
        lengths = np.linalg.norm(vectors, axis=1)
        if (np.any(lengths < 1e-8)
                or np.linalg.norm(np.cross(*vectors)) / np.prod(lengths) < 1e-6):
            raise ValueError("Apex and valve landmarks must be distinct and non-collinear")
    u, singular_values, vt = np.linalg.svd(relative_moving.T @ relative_fixed)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = fixed[0] - rotation @ moving[0]
    validate_rigid(matrix)
    aligned = transform_points(moving, matrix)
    errors = np.linalg.norm(aligned - fixed, axis=1)
    if errors[0] > 1e-6:
        raise ValueError("Hard LV endo-apex coincidence was not satisfied")
    pairs = [(0, 1), (0, 2), (1, 2)]
    report = {"method": "apex_fixed_rigid_landmark_SVD", "landmark_names": LANDMARK_NAMES,
              "apex_is_hard_constraint": True, "apex_tolerance_mm": 1e-6,
              "objective": "squared_mitral_error + squared_tricuspid_error with equal weights",
              "weights": {"lv_mitral_point": 1.0, "rv_tricuspid_point": 1.0},
              "icp_refinement": False, "moving_points_mm": moving.tolist(),
              "fixed_points_mm": fixed.tolist(), "aligned_moving_points_mm": aligned.tolist(),
              "errors_before_mm": dict(zip(LANDMARK_NAMES, np.linalg.norm(moving - fixed, axis=1).tolist())),
              "errors_after_mm": dict(zip(LANDMARK_NAMES, errors.tolist())),
              "valve_squared_error_sum_mm2": float(np.sum(errors[1:] ** 2)),
              "valve_rms_error_mm": float(np.sqrt(np.mean(errors[1:] ** 2))),
              "rotation_determinant": float(np.linalg.det(rotation)),
              "orthogonality_max_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
              "cross_covariance_singular_values": singular_values.tolist(),
              "landmark_pair_distances_mm": [
                  {"pair": [LANDMARK_NAMES[i], LANDMARK_NAMES[j]],
                   "moving": float(np.linalg.norm(moving[i] - moving[j])),
                   "fixed": float(np.linalg.norm(fixed[i] - fixed[j]))} for i, j in pairs]}
    return matrix, report


def associate_landmarks_with_volume(points, mesh, mesh_path):
    """Record geometric association without snapping centroids or changing the given definitions."""
    bounds = np.asarray(mesh.bounds).reshape(3, 2)
    if np.any(points < bounds[:, 0] - 1.0) or np.any(points > bounds[:, 1] + 1.0):
        raise ValueError(f"Landmarks fall outside {mesh_path.name}'s bounds; check coordinate frames")
    surface = mesh.extract_surface().triangulate()
    cell_ids, closest = surface.find_closest_cell(points, return_closest_point=True)
    vertex_distances, vertex_ids = cKDTree(mesh.points).query(points)
    parent_ids = np.asarray(surface.cell_data["original_cell_id"])[cell_ids]
    return {"volume_mesh": str(mesh_path), "coordinates_are_in_volume_native_frame": True,
            "coordinates_snapped": False,
            "note": "Centroids are geometric reference points and need not lie on material; nearest geometry is diagnostic only.",
            "landmarks": [
                {"name": name, "coordinate_mm": np.asarray(point).tolist(),
                 "nearest_surface_point_mm": np.asarray(q).tolist(),
                 "distance_to_surface_mm": float(np.linalg.norm(point - q)),
                 "nearest_original_volume_cell_id": int(cell_id),
                 "nearest_volume_vertex_id": int(vertex_id), "distance_to_vertex_mm": float(vertex_distance)}
                for name, point, q, cell_id, vertex_id, vertex_distance
                in zip(LANDMARK_NAMES, points, closest, parent_ids, vertex_ids, vertex_distances)]}


def save_landmark_artifacts(output, moving, fixed, matrix, metadata):
    aligned = transform_points(moving, matrix)
    for filename, points in [("moving_landmarks_original.vtp", moving),
                             ("moving_landmarks_aligned.vtp", aligned), ("fixed_landmarks.vtp", fixed)]:
        cloud = pv.PolyData(points)
        cloud.point_data["landmark_id"] = np.arange(3)
        cloud.point_data["landmark_name"] = np.asarray(LANDMARK_NAMES)
        cloud.save(output / filename)
    lines = np.array([[2, i, i + 3] for i in range(3)]).ravel()
    residuals = pv.PolyData(np.vstack([aligned, fixed]), lines=lines)
    residuals.cell_data["landmark_name"] = np.asarray(LANDMARK_NAMES)
    residuals.cell_data["error_mm"] = np.linalg.norm(aligned - fixed, axis=1)
    residuals.save(output / "landmark_residuals.vtp")
    (output / "landmarks.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def hash_files(paths):
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path)] = digest.hexdigest()
    return result


def clip_volume(mesh, planes):
    retained = mesh
    removed_parts = []
    for plane in planes:
        if not retained.n_cells:
            break
        normal, offset = np.asarray(plane["normal"]), plane["offset_mm"]
        if np.max(retained.points @ normal - offset) <= 1e-7:
            continue
        retained, removed = retained.clip(normal=normal, origin=normal * offset,
                                          invert=True, return_clipped=True)
        if removed.n_cells:
            removed_parts.append(removed)
    if not retained.n_cells:
        raise ValueError("Clipping removed the entire volume; check plane direction/offset")
    retained = retained.triangulate()
    if removed_parts:
        removed = pv.merge(removed_parts, merge_points=False).triangulate()
    else:
        removed = pv.UnstructuredGrid()
    return retained, removed


def encode_volume(mesh, tag_codes):
    probe = mesh.copy(deep=False)
    probe.clear_data()
    tags = np.asarray(mesh.cell_data["elemTag"])
    codes = np.zeros(len(tags), dtype=np.int32)
    for tag, code in tag_codes.items():
        codes[tags == tag] = code
    if np.any(codes == 0):
        raise ValueError("Missing tag encoding")
    probe.cell_data["dice_code"] = codes
    return probe


def grid_definition(source, reference, spacing):
    bounds = np.array([source.bounds, reference.bounds]).reshape(2, 3, 2)
    low = np.floor(bounds[:, :, 0].min(axis=0) / spacing) * spacing - spacing
    high = np.ceil(bounds[:, :, 1].max(axis=0) / spacing) * spacing + spacing
    shape = np.ceil((high - low) / spacing).astype(int)
    return low, shape


def sample_volume(mesh, low, shape, spacing):
    # Image POINTS are voxel centers here. VTK tests actual cell containment and
    # copies cell labels; nearest-surface/nearest-vertex distances are never used.
    centers = pv.ImageData(dimensions=tuple(shape), spacing=(spacing,) * 3, origin=low + spacing / 2)
    sampled = centers.sample(mesh, tolerance=1e-8, locator="static_cell", mark_blank=False,
                             pass_point_data=False, pass_cell_data=False, pass_field_data=False)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    codes = np.asarray(sampled.point_data["dice_code"]).astype(np.int32, copy=True)
    codes[~valid] = 0
    return codes


def region_mask(low, shape, spacing, planes):
    x, y, z = [low[i] + (np.arange(shape[i]) + 0.5) * spacing for i in range(3)]
    keep = np.ones(tuple(shape[::-1]), dtype=bool)
    for plane in planes:
        nx, ny, nz = plane["normal"]
        keep &= nz * z[:, None, None] + ny * y[None, :, None] + nx * x[None, None, :] <= plane["offset_mm"] + 1e-9
    return keep.ravel()


def dice_rows(source, reference, tag_codes, spacing, stage):
    rows = []
    for region, code in [("all_tissue", None)] + [(f"tag_{tag}", c) for tag, c in tag_codes.items()]:
        a = source != 0 if code is None else source == code
        b = reference != 0 if code is None else reference == code
        na, nb, intersection = int(a.sum()), int(b.sum()), int(np.count_nonzero(a & b))
        denominator = na + nb
        rows.append({"stage": stage, "spacing_mm": float(spacing), "region": region,
                     "dice": 2 * intersection / denominator if denominator else None,
                     "source_volume_mm3": na * spacing ** 3,
                     "reference_volume_mm3": nb * spacing ** 3,
                     "intersection_volume_mm3": intersection * spacing ** 3,
                     "source_voxels": na, "reference_voxels": nb, "intersection_voxels": intersection})
    return rows


def preview(source_data, target_data, matrix, planes, output, landmarks=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = transform_points(source_data[0], matrix)
    b = target_data[0]
    retained_a, retained_b = np.ones(len(a), bool), np.ones(len(b), bool)
    for plane in planes:
        retained_a &= a @ np.asarray(plane["normal"]) <= plane["offset_mm"] + 1e-7
        retained_b &= b @ np.asarray(plane["normal"]) <= plane["offset_mm"] + 1e-7
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for col, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        axes[0, col].scatter(b[:, i], b[:, j], s=.6, alpha=.25, c="#176ea3", label="Reference")
        axes[0, col].scatter(a[:, i], a[:, j], s=.6, alpha=.25, c="#e89b23", label="Aligned source")
        ax = axes[1, col]
        if planes:
            ax.scatter(b[retained_b, i], b[retained_b, j], s=.6, alpha=.25, c="#176ea3", label="Reference retained")
            ax.scatter(a[retained_a, i], a[retained_a, j], s=.6, alpha=.25, c="#e89b23", label="Source retained")
            ax.scatter(a[~retained_a, i], a[~retained_a, j], s=1, alpha=.45, c="#cf3434", label="Source removed")
        else:
            original = source_data[0]
            ax.scatter(b[:, i], b[:, j], s=.6, alpha=.25, c="#176ea3", label="Fixed volume")
            ax.scatter(original[:, i], original[:, j], s=.6, alpha=.25, c="#e89b23", label="Moving volume before alignment")
        for row in range(2):
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xlabel("XYZ"[i] + " (mm)")
            axes[row, col].set_ylabel("XYZ"[j] + " (mm)")
            axes[row, col].set_title("Aligned full surfaces" if row == 0 else (
                "Shared basal clipping; red = removed" if planes else "Before alignment (native coordinates)"))
            if landmarks is not None:
                moving, fixed = landmarks
                moved = transform_points(moving, matrix) if row == 0 or planes else moving
                current = axes[row, col]
                current.scatter(fixed[:, i], fixed[:, j], marker="*", s=110, c="#176ea3", edgecolors="black", linewidths=.5, zorder=5)
                current.scatter(moved[:, i], moved[:, j], marker="o", s=45, c="#e89b23", edgecolors="black", linewidths=.5, zorder=5)
                for label, p, q in zip(["A", "M", "T"], moved, fixed):
                    current.plot([p[i], q[i]], [p[j], q[j]], "k--", linewidth=.9, alpha=.7)
                    current.annotate(label + " fixed", (q[i], q[j]), xytext=(5, 5), textcoords="offset points", fontsize=8)
                    current.annotate(label + " moving", (p[i], p[j]), xytext=(5, -11), textcoords="offset points", fontsize=8)
        axes[0, col].legend(markerscale=4, loc="best")
        ax.legend(markerscale=4, loc="best")
    title = "Blue: fixed reference; orange: moving volume."
    if landmarks is not None:
        title += " A = LV endo apex; M = specified LV mitral point; T = RV tricuspid point."
    elif planes:
        title += " Inspect the inferred basal boundary."
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def save_planes(planes, reference, output):
    if not planes:
        return
    length = float(np.linalg.norm(np.ptp(reference.points, axis=0)))
    patches = []
    for index, plane in enumerate(planes):
        normal = np.asarray(plane["normal"])
        center = np.asarray(reference.center)
        center -= normal * (center @ normal - plane["offset_mm"])
        patch = pv.Plane(center=center, direction=normal, i_size=length, j_size=length, i_resolution=1, j_resolution=1)
        patch.cell_data["plane_index"] = np.full(patch.n_cells, index)
        patches.append(patch)
    pv.merge(patches, merge_points=False).save(output / "basal_planes.vtp")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-dir", type=Path, default=Path(__file__).resolve().parent / "1")
    p.add_argument("--source", type=Path, help="Moving tetrahedral volume")
    p.add_argument("--reference", type=Path, help="Fixed tetrahedral volume")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--registration-method", choices=["icp", "landmarks"], default="icp",
                   help="landmarks moves subject_11 to subject0081 with exact LV apex coincidence; default icp")
    p.add_argument("--landmarks-dir", type=Path, help="Directory containing the six anatomical sidecars; default --case-dir")
    p.add_argument("--label-array", default="elemTag")
    p.add_argument("--label-map", action="append", default=[], metavar="SOURCE:REFERENCE", help="Explicit source tag correspondence; repeat as needed")
    p.add_argument("--spacing", nargs="+", type=float, default=[1.0, 0.5], help="Shared grid spacings in mm; default 1.0 0.5")
    p.add_argument("--plane", action="append", nargs=4, type=float, metavar=("NX", "NY", "NZ", "D"), help="Reference-frame plane: retain dot(normal,x)<=D; repeat for multiple planes")
    p.add_argument("--no-trim", action="store_true", help="Compute aligned full-volume Dice only")
    p.add_argument("--plane-offset-mm", type=float, default=0.0, help="Move every boundary outward by this amount; negative clips deeper")
    p.add_argument("--transform-in", type=Path, help="Reuse transform.json or a JSON 4x4 rigid matrix")
    p.add_argument("--samples", type=int, default=12000)
    p.add_argument("--registration-trim-fraction", type=float, default=0.85)
    p.add_argument("--max-grid-points", type=int, default=80000000)
    p.add_argument("--convergence-tolerance", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=81)
    p.add_argument("--overwrite", action="store_true", help="Replace artifacts in an existing output directory")
    return p


def run(args):
    started = time.monotonic()
    if (not all(np.isfinite(s) and s > 0 for s in args.spacing) or args.samples < 1000
            or not 0.5 <= args.registration_trim_fraction <= 1.0
            or not np.isfinite(args.plane_offset_mm) or args.convergence_tolerance < 0):
        raise ValueError("Invalid spacing, sample count, trimming fraction, plane offset, or convergence tolerance")
    if args.no_trim and args.plane:
        raise ValueError("--no-trim and --plane cannot be used together")
    landmark_mode = args.registration_method == "landmarks"
    if landmark_mode and args.transform_in:
        raise ValueError("Landmark fitting and --transform-in are mutually exclusive")
    source_name = "subject_11_volmesh.vtk" if landmark_mode else "subject0081_mesh_all.vtk"
    reference_name = "subject0081_mesh_all.vtk" if landmark_mode else "subject_11_volmesh.vtk"
    source_path = (args.source or args.case_dir / source_name).resolve()
    reference_path = (args.reference or args.case_dir / reference_name).resolve()
    output_name = "dice_results_landmarks" if landmark_mode else "dice_results"
    output = (args.output_dir or args.case_dir / output_name).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError(f"Output directory is not empty: {output}. Use a new directory or --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    input_paths = [source_path, reference_path]
    if landmark_mode:
        landmark_directory = args.landmarks_dir or args.case_dir
        input_paths.extend((landmark_directory / name).resolve() for name in LANDMARK_FILES.values())
    input_hashes = hash_files(input_paths)
    log("Reading tetrahedral volumes ...")
    source, reference = read_volume(source_path, args.label_array), read_volume(reference_path, args.label_array)
    original_source_tags = sorted(int(t) for t in np.unique(source["elemTag"]))
    mapping = {t: t for t in original_source_tags}
    for item in args.label_map:
        a, b = item.split(":")
        if int(a) not in mapping:
            raise ValueError(f"Source tag {a} is not present")
        mapping[int(a)] = int(b)
    original_labels = np.asarray(source["elemTag"]).copy()
    source["elemTag"] = np.array([mapping[int(t)] for t in original_labels], dtype=np.int64)
    reference_tags = sorted(int(t) for t in np.unique(reference["elemTag"]))
    if set(mapping.values()) != set(reference_tags):
        raise ValueError("Source/reference tag sets differ. Supply --label-map to establish anatomical correspondence.")
    source_stats, reference_stats = volume_stats(source), volume_stats(reference)
    no_trim = args.no_trim or (landmark_mode and not args.plane)
    planes = [] if no_trim else ([normalized_plane(v) for v in args.plane] if args.plane else detect_basal_planes(reference))
    registration_planes = [dict(plane) for plane in planes]
    for plane in planes:
        plane["offset_mm"] += args.plane_offset_mm
        plane["origin_mm"] = (np.asarray(plane["normal"]) * plane["offset_mm"]).tolist()
    log(f"Basal clipping planes: {len(planes)}; sampling boundary surfaces ...")
    source_data = sample_surface(source, args.samples, args.seed)
    target_data = sample_surface(reference, args.samples, args.seed + 1, registration_planes)
    landmarks, landmark_metadata = None, None
    if landmark_mode:
        log("Extracting anatomical landmarks and associating them with the full volumes ...")
        moving_landmarks, fixed_landmarks, landmark_metadata = extract_landmarks(landmark_directory)
        landmark_metadata["moving_volume_association"] = associate_landmarks_with_volume(moving_landmarks, source, source_path)
        landmark_metadata["fixed_volume_association"] = associate_landmarks_with_volume(fixed_landmarks, reference, reference_path)
        matrix, registration = register_anchored_landmarks(moving_landmarks, fixed_landmarks)
        landmark_metadata["registration"] = registration
        landmark_metadata["source_to_reference"] = matrix.tolist()
        landmarks = (moving_landmarks, fixed_landmarks)
        save_landmark_artifacts(output, moving_landmarks, fixed_landmarks, matrix, landmark_metadata)
        log(f"Landmark errors after fitting (mm): {registration['errors_after_mm']}")
    elif args.transform_in:
        data = json.loads(args.transform_in.read_text())
        matrix = validate_rigid(data["source_to_reference"] if isinstance(data, dict) else data)
        registration = {"method": "provided_rigid_transform", "transform_file": str(args.transform_in.resolve())}
    else:
        log("Aligning ventricular bodies (25 rigid initializations; no scaling/deformation) ...")
        matrix, registration = register(source_data, target_data, args.registration_trim_fraction, registration_planes)
        log(f"Registration trimmed mean surface distance: {registration['mean_trimmed_symmetric_surface_distance_mm']:.4f} mm")
    (output / "transform.json").write_text(json.dumps({"source": str(source_path), "reference": str(reference_path),
                                                    "source_to_reference": matrix.tolist(), "registration": registration}, indent=2) + "\n")
    (output / "clipping_planes.json").write_text(json.dumps({"keep_rule": "all dot(normal,x) <= offset_mm in reference coordinates",
                                                          "planes": planes}, indent=2) + "\n")
    aligned = source.copy(deep=True)
    aligned.points = transform_points(source.points, matrix)
    aligned_stats = volume_stats(aligned)
    if not np.isclose(aligned_stats["volume_mm3"], source_stats["volume_mm3"], rtol=1e-8, atol=1e-6):
        raise ValueError("Rigid alignment failed native-volume preservation")
    if not np.array_equal(aligned.cells, source.cells):
        raise ValueError("Rigid alignment changed volume connectivity")
    log("Clipping tetrahedral volumes at shared anatomical boundaries ...")
    retained_a, removed_a = clip_volume(aligned, planes)
    retained_b, removed_b = clip_volume(reference, planes)
    stats_a, stats_b = volume_stats(retained_a), volume_stats(retained_b)
    removed_stats_a, removed_stats_b = volume_stats(removed_a), volume_stats(removed_b)
    for whole, kept, removed in [(source_stats, stats_a, removed_stats_a), (reference_stats, stats_b, removed_stats_b)]:
        if not np.isclose(whole["volume_mm3"], kept["volume_mm3"] + removed["volume_mm3"], rtol=1e-5, atol=1e-4):
            raise ValueError("Clipping failed volume conservation")
    aligned.save(output / "source_aligned.vtu")
    retained_a.save(output / "source_retained.vtu")
    retained_b.save(output / "reference_retained.vtu")
    for removed_mesh, name in [(removed_a, "source_removed.vtu"), (removed_b, "reference_removed.vtu")]:
        path = output / name
        if removed_mesh.n_cells:
            removed_mesh.save(path)
        elif path.exists():
            path.unlink()  # Remove a stale artifact from a previous --overwrite run.
    save_planes(planes, reference, output)
    if not planes and (output / "basal_planes.vtp").exists():
        (output / "basal_planes.vtp").unlink()
    preview(source_data, target_data, matrix, planes, output, landmarks)
    log(f"Source removed: {100 * removed_stats_a['volume_mm3'] / source_stats['volume_mm3']:.3f}% of native volume")
    tag_codes = {t: i + 1 for i, t in enumerate(reference_tags)}
    probe_a, probe_b = encode_volume(aligned, tag_codes), encode_volume(reference, tag_codes)
    rows, grids = [], []
    spacings = sorted(set(args.spacing), reverse=True)
    for spacing in spacings:
        low, shape = grid_definition(aligned, reference, spacing)
        count = int(np.prod(shape, dtype=np.int64))
        if count > args.max_grid_points:
            raise ValueError(f"Grid would contain {count:,} points; choose coarser spacing or increase --max-grid-points")
        log(f"Voxelizing at {spacing:g} mm: {tuple(shape)} = {count:,} voxel centers ...")
        a, b = sample_volume(probe_a, low, shape, spacing), sample_volume(probe_b, low, shape, spacing)
        current_rows = dice_rows(a, b, tag_codes, spacing, "aligned_full")
        keep = region_mask(low, shape, spacing, planes)
        trimmed_a, trimmed_b = a.copy(), b.copy()
        trimmed_a[~keep], trimmed_b[~keep] = 0, 0
        if planes:
            current_rows.extend(dice_rows(trimmed_a, trimmed_b, tag_codes, spacing, "basal_trimmed"))
        rows.extend(current_rows)
        grids.append({"spacing_mm": spacing, "lower_corner_mm": low.tolist(), "dimensions_voxels": shape.tolist(),
                      "sample_rule": "tetrahedral containment at voxel centers; x varies fastest"})
        if spacing == min(spacings):
            image = pv.ImageData(dimensions=tuple(shape + 1), spacing=(spacing,) * 3, origin=low)
            image.cell_data["source_aligned_full"] = a
            image.cell_data["reference_full"] = b
            image.cell_data["source_retained"] = trimmed_a
            image.cell_data["reference_retained"] = trimmed_b
            image.cell_data["comparison_region"] = keep.astype(np.uint8)
            image.save(output / "comparison_masks.vti")
        for row in current_rows:
            log(f"  {row['stage']} {row['region']}: Dice={row['dice']}")
    finest = min(spacings)
    convergence = {"checked": len(spacings) >= 2, "tolerance": args.convergence_tolerance,
                   "max_absolute_dice_change": None, "within_tolerance": None}
    if len(spacings) >= 2:
        pair = sorted(spacings)[:2]
        keyed = {(r["spacing_mm"], r["stage"], r["region"]): r["dice"] for r in rows}
        changes = [abs(value - keyed[(pair[1], stage, region)]) for (h, stage, region), value in keyed.items()
                   if h == pair[0] and value is not None and keyed[(pair[1], stage, region)] is not None]
        if changes:
            maximum = max(changes)
            convergence.update({"compared_spacings_mm": pair, "max_absolute_dice_change": maximum,
                                "within_tolerance": maximum <= args.convergence_tolerance})
    warnings = ["Automatic basal clipping is geometric, not semantic valve identification; inspect the saved overlay and plane."] if any(p["method"].startswith("reference_") for p in planes) else []
    warnings.append("Tags correspond by numeric value unless explicitly mapped; anatomical LV/RV meanings are not inferred.")
    warnings.append("Scores quantify shape overlap after rigid registration; subject identity and cardiac phase are not inferred.")
    if not convergence["checked"] or convergence["within_tolerance"] is not True:
        warnings.append("Voxel-spacing convergence is unconfirmed; run at a finer spacing before interpreting small differences.")
    inputs_unchanged = hash_files(input_paths) == input_hashes
    if not inputs_unchanged:
        raise ValueError("An input file changed during the run")
    report = {"source": str(source_path), "reference": str(reference_path), "units_assumed": "mm",
              "dice_input_volumes": {"moving": str(source_path), "fixed": str(reference_path),
                                     "landmark_sidecars_used_for_dice": False},
              "input_sha256": input_hashes, "inputs_unchanged": inputs_unchanged,
              "rigid_volume_check": {"original_mm3": source_stats["volume_mm3"],
                                     "aligned_mm3": aligned_stats["volume_mm3"], "connectivity_preserved": True},
              "source_tag_mapping": mapping, "mask_code_to_reference_tag": {v: k for k, v in tag_codes.items()},
              "mask_background_code": 0, "registration": registration, "source_to_reference": matrix.tolist(),
              "clipping": {"planes": planes, "source_full": source_stats, "reference_full": reference_stats,
                           "source_retained": stats_a, "reference_retained": stats_b,
                           "source_removed": removed_stats_a, "reference_removed": removed_stats_b,
                           "source_removed_fraction": removed_stats_a["volume_mm3"] / source_stats["volume_mm3"],
                           "reference_removed_fraction": removed_stats_b["volume_mm3"] / reference_stats["volume_mm3"]},
              "grids": grids, "dice": rows, "convergence": convergence, "warnings": warnings,
              "finest_spacing_mm": finest, "elapsed_seconds": time.monotonic() - started}
    if landmark_metadata is not None:
        report["landmarks"] = landmark_metadata
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "dice_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log(f"Results saved to {output}; spacing convergence: {convergence}")
    return report


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        p.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
