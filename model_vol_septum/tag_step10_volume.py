"""Validate and save LV/RV cell labels without changing Step 09 geometry."""
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from model_vol_septum.clean_step02_surfaces import validate_clean_surface
from model_vol_septum.generate_step09_volume import signed_volumes


def shared_face_neighbors(cells):
    """Return cell pairs and the triangle they share (vertex/edge contacts excluded)."""
    faces = np.sort(np.vstack([cells[:, ids] for ids in
                              ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]), axis=1)
    unique, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError("Step 09 tetrahedra have nonmanifold shared faces.")
    owners = np.tile(np.arange(len(cells)), 4)
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
    a, b = owners[order[offsets]], owners[order[offsets + 1]]
    return a, b, unique[counts == 2]


def component_labels(tags, a, b):
    keep = tags[a] == tags[b]
    graph = coo_matrix((np.ones(np.count_nonzero(keep)), (a[keep], b[keep])),
                       shape=(len(tags), len(tags))).tocsr()
    return connected_components(graph, directed=False)[1]


def cleanup_tag_islands(points, cells, tags, *, enabled=True, min_cells=32, min_fraction=0.0005,
                        neighbors=None):
    """Sequentially absorb small islands into the opposite protected main region.

    Each accepted move merges components and never removes cells from either main
    region. Thus component count strictly decreases and alternating labels cannot
    oscillate. Thresholds are fixed from the initial largest regions.
    """
    if min_cells < 0 or not np.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
        raise ValueError("Island thresholds require min_cells >= 0 and 0 <= min_fraction <= 1.")
    initial = np.asarray(tags, dtype=np.int32)
    if set(np.unique(initial)) != {0, 1}:
        raise ValueError("Island cleanup requires both LV=1 and RV=0.")
    result = initial.copy()
    a, b, faces = shared_face_neighbors(cells) if neighbors is None else neighbors
    triangles = points[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) * .5
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise RuntimeError("Tag cleanup requires nondegenerate shared faces.")
    labels = component_labels(result, a, b)
    sizes = np.bincount(labels)
    seeds, thresholds = {}, {}
    for tag in (0, 1):
        components = np.unique(labels[result == tag])
        largest = components[np.argmax(sizes[components])]
        seeds[tag] = int(np.flatnonzero(labels == largest)[0])
        thresholds[tag] = max(int(min_cells), int(np.ceil(sizes[largest] * min_fraction)))

    changes = []
    while True:
        sizes = np.bincount(labels)
        roots = {tag: int(labels[seed]) for tag, seed in seeds.items()}
        # Areas of each component's interface with each protected main region.
        support = {}
        for tag in (0, 1):
            left, right = labels[a] == roots[tag], labels[b] == roots[tag]
            support[tag] = np.bincount(
                np.r_[labels[b[left]], labels[a[right]]],
                weights=np.r_[areas[left], areas[right]], minlength=len(sizes))
        candidates, unresolved = [], []
        component_tags = np.zeros(len(sizes), dtype=np.int32)
        component_tags[labels[result == 1]] = 1
        for component in np.argsort(sizes, kind="stable"):
            tag = int(component_tags[component])
            if component == roots[tag]:
                continue
            reason = ("cleanup_disabled" if not enabled else
                      "at_or_above_size_threshold" if sizes[component] >= thresholds[tag] else
                      "no_shared_face_with_opposite_main_region" if support[1 - tag][component] <= 0 else None)
            if reason:
                unresolved.append({"elemTag": tag, "cells": int(sizes[component]),
                                   "component": int(component), "reason": reason})
            else:
                candidates.append((component, tag, float(support[1 - tag][component])))
        if not candidates:
            for item in unresolved:
                item["cell_ids"] = np.flatnonzero(labels == item.pop("component")).tolist()
            break
        component, tag, area = candidates[0]
        ids = np.flatnonzero(labels == component)
        previous_count = len(sizes)
        result[ids] = 1 - tag
        changes.append({"from_tag": tag, "to_tag": 1 - tag, "cells": int(len(ids)),
                        "cell_ids": ids.tolist(), "supporting_face_area_mm2": area})
        labels = component_labels(result, a, b)
        if int(labels.max()) + 1 >= previous_count:
            raise RuntimeError("Tag cleanup did not reduce disconnected components.")
    return result, {
        "enabled": bool(enabled), "method": "sequential_shared_face_area_supported_main_region_absorption",
        "threshold_rule": "non-main component cells < max(min_cells, ceil(initial_main_cells * min_fraction))",
        "min_cells": int(min_cells), "min_fraction": float(min_fraction),
        "threshold_cells": {str(tag): value for tag, value in thresholds.items()},
        "relabeled_cells": int(np.count_nonzero(initial != result)),
        "relabeled_components": len(changes), "changes": changes,
        "unresolved_components": unresolved, "converged": True,
    }


def tag_components(cells, tags, volumes, neighbors=None):
    """Report face-connected components; a vertex contact does not join regions."""
    a, b, _ = shared_face_neighbors(cells) if neighbors is None else neighbors
    result = {}
    for tag, name in ((1, "LV"), (0, "RV")):
        ids = np.flatnonzero(tags == tag)
        mapping = np.full(len(cells), -1, dtype=np.int64)
        mapping[ids] = np.arange(len(ids))
        keep = (tags[a] == tag) & (tags[b] == tag)
        graph = coo_matrix((np.ones(np.count_nonzero(keep)), (mapping[a[keep]], mapping[b[keep]])),
                           shape=(len(ids), len(ids))).tocsr()
        count, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        component_volumes = np.bincount(labels, weights=volumes[ids])
        ranked = np.argsort(-sizes, kind="stable")
        largest = int(ranked[0])
        result[name] = {
            "elemTag": tag, "tetrahedra": int(len(ids)), "volume_mm3": float(volumes[ids].sum()),
            "component_count": int(count),
            "components": [{"cells": int(sizes[i]), "volume_mm3": float(component_volumes[i])} for i in ranked],
            "cells_outside_largest_component": ids[labels != largest].tolist(),
        }
    return result


def save_tagged_volume(input_path: Path, reference_path: Path, output_path: Path, classify,
                       *, cleanup_enabled=True, min_cells=32, min_fraction=0.0005):
    before_path = output_path.with_name(output_path.stem + "_before_tag_cleanup.vtk")
    if output_path.resolve() in (input_path.resolve(), reference_path.resolve()):
        raise ValueError("Step 10 output must differ from its input and reference files.")
    if before_path.resolve() in (input_path.resolve(), reference_path.resolve()):
        raise ValueError("Step 10 comparison output must differ from its inputs.")
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (input_path, reference_path)}
    source = pv.read(input_path)
    if not isinstance(source, pv.UnstructuredGrid) or not source.n_cells or not np.all(source.celltypes == pv.CellType.TETRA):
        raise RuntimeError("Step 10 requires a nonempty tetrahedron-only Step 09 mesh.")
    if "elemTag" in source.cell_data:
        raise RuntimeError("Step 10 expects the untagged Step 09 mesh.")
    points = np.asarray(source.points)
    cells = np.asarray(source.cells).reshape(-1, 5)[:, 1:]
    if not np.isfinite(points).all() or cells.min() < 0 or cells.max() >= len(points):
        raise RuntimeError("Invalid Step 09 coordinates or connectivity.")
    volumes = signed_volumes(points, cells)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise RuntimeError("Step 09 has inverted or degenerate tetrahedra; tagging cannot repair them.")
    reference = pv.read(reference_path).extract_surface().triangulate()
    reference_validation = validate_clean_surface(reference)
    tags, classification = classify(points, cells, reference)
    tags = np.asarray(tags)
    if tags.shape != (source.n_cells,) or not np.isin(tags, [0, 1]).all():
        raise RuntimeError("Each tetrahedron must have exactly one LV=1 or RV=0 label.")
    if set(np.unique(tags)) != {0, 1}:
        raise RuntimeError("Step 10 classification must contain both LV and RV regions.")
    tags = tags.astype(np.int32)
    neighbors = shared_face_neighbors(cells)
    regions_before = tag_components(cells, tags, volumes, neighbors)
    initial_tags = tags.copy()
    tags, cleanup = cleanup_tag_islands(points, cells, tags, enabled=cleanup_enabled,
                                       min_cells=min_cells, min_fraction=min_fraction, neighbors=neighbors)
    regions = tag_components(cells, tags, volumes, neighbors)
    classification["island_relabeling_applied"] = bool(cleanup["relabeled_cells"])
    changed = (tags != initial_tags).astype(np.uint8)
    before = source.copy(deep=True)
    before.cell_data["elemTag"] = initial_tags
    before.set_active_scalars("elemTag", preference="cell")
    final = source.copy(deep=True)
    final.cell_data["elemTag"] = tags
    final.cell_data["tagChanged"] = changed
    final.set_active_scalars("elemTag", preference="cell")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".step10_validate_", dir=output_path.parent) as directory:
        staged = Path(directory) / output_path.name
        staged_before = Path(directory) / before_path.name
        before.save(staged_before, binary=True)
        final.save(staged, binary=True)
        for path, expected in ((staged_before, initial_tags), (staged, tags)):
            saved = pv.read(path)
            if not (np.array_equal(saved.points, source.points) and np.array_equal(saved.cells, source.cells)
                    and np.array_equal(saved.celltypes, source.celltypes)):
                raise RuntimeError("Step 10 serialization changed mesh geometry or element order.")
            if not np.array_equal(saved.cell_data.get("elemTag"), expected):
                raise RuntimeError("Step 10 saved labels differ from the validated labels.")
        if not np.array_equal(saved.cell_data.get("tagChanged"), changed):
            raise RuntimeError("Step 10 saved change flags differ from the validated flags.")
        for path, digest in hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Step 10 input changed during tagging: {path}")
        staged_before.replace(before_path)
        staged.replace(output_path)
    return {
        "input_path": str(input_path.resolve()), "input_sha256": hashes[input_path],
        "reference_path": str(reference_path.resolve()), "reference_sha256": hashes[reference_path],
        "reference_stage": "step_02_surf_clean", "reference_validation": reference_validation,
        "output_path": str(output_path.resolve()), "cell_array": "elemTag", "tag_legend": {"1": "LV", "0": "RV"},
        "before_cleanup_path": str(before_path.resolve()), "change_array": "tagChanged",
        "points": int(source.n_points), "tetrahedra": int(source.n_cells),
        "geometry_and_element_order_preserved": True, "inputs_unchanged": True,
        "nonpositive_tetrahedra": 0, "total_volume_mm3": float(volumes.sum()),
        "regions": regions, "regions_before_cleanup": regions_before,
        "classification": classification, "tag_cleanup": cleanup,
        "saved_validation": {"geometry_matches_step09": True, "tags_match_cleanup": True,
                             "before_tags_match_classification": True, "change_flags_match": True,
                             "both_regions_present": True, "only_tags_0_and_1": True},
    }
