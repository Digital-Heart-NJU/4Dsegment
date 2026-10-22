"""MeshFix repair and validation shared by pipeline Steps 02 and 08."""

from __future__ import annotations

import numpy as np
import pymeshfix
import pyvista as pv

from model_vol_septum.fill_step01_surface_holes import topology, triangles
from model_vol_septum.mesh_intersections import confirmed_self_intersections


def surface_quality(surface: pv.PolyData) -> dict:
    if surface.n_points == 0 or surface.n_cells == 0:
        raise RuntimeError("Surface is empty.")
    if not np.isfinite(surface.points).all():
        raise RuntimeError("Surface contains non-finite coordinates.")
    faces = triangles(surface)
    return {
        **topology(surface),
        "self_intersection_pairs": int(len(confirmed_self_intersections(surface.points, faces))),
    }


def validate_clean_surface(surface: pv.PolyData) -> dict:
    stats = surface_quality(surface)
    failures = {key: stats[key] for key in (
        "free_edges", "nonmanifold_edges", "degenerate_triangles",
        "inconsistent_orientation_edges", "self_intersection_pairs",
    ) if stats[key]}
    if failures:
        raise RuntimeError(f"MeshFix surface validation failed: {failures}")
    return stats


def clean_surface_meshfix(surface: pv.PolyData, tag: int | None) -> tuple[pv.PolyData, dict]:
    """Repair a copy; MeshFix may change point locations and connectivity."""
    source = surface.extract_surface().triangulate()
    before = surface_quality(source)
    fixer = pymeshfix.MeshFix(np.asarray(source.points, dtype=float).copy(), triangles(source).copy())
    fixer.repair(joincomp=False, remove_smallest_components=False)
    extra_cleanup = False
    if len(confirmed_self_intersections(fixer.points, fixer.faces)):
        fixer.clean(max_iters=60, inner_loops=10)
        fixer.intersection_removal(max_iter=10)
        fixer.degeneracy_removal(max_iter=10)
        extra_cleanup = True
    repaired = fixer.mesh.extract_surface().triangulate().compute_normals(
        point_normals=True, cell_normals=True, consistent_normals=True,
        auto_orient_normals=True, split_vertices=False,
    )
    # A merged LV/RV surface has no single chamber tag. MeshFix may change
    # connectivity, so its original per-face labels cannot be reused directly.
    if tag is None:
        repaired.clear_data()
    else:
        repaired.cell_data["elemTag"] = np.full(repaired.n_cells, tag, dtype=np.int32)
    after = validate_clean_surface(repaired)
    return repaired, {
        "method": "pymeshfix.MeshFix.repair",
        "meshfix_version": pymeshfix.__version__,
        "repair_options": {"joincomp": False, "remove_smallest_components": False},
        "additional_intersection_cleanup": extra_cleanup,
        "intersection_validation": "Open3D_candidates_confirmed_by_FCL_double_precision_triangle_distance",
        "before": before, "after": after,
        "elemTag": int(tag) if tag is not None else None,
        "input_file_modified": False,
    }
