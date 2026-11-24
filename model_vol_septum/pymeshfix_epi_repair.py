"""Use MeshFix geometry as a local displacement target without remeshing the input.

MeshFix receives a complete closed trial surface: an extracted open patch can
have its outer boundary capped by MeshFix. Only vertices near the reported bad
triangles are projected onto the repaired target. The target's connectivity is
discarded. All original point IDs, faces, and proposed coordinates outside the
patch are retained, and acceptance is checked against the complete input mesh.
"""
from __future__ import annotations

import numpy as np


def repair_closed_epi_pymeshfix(points, faces, delta, edges, weights,
                                reference, normals, original, bad_faces,
                                max_attempts=6, min_accepted_step=.01, final_check=None,
                                fixed_vertices=None):
    """Return a locally corrected step, or None when coordinate transfer fails.

    ``fixed_vertices`` excludes locked coordinates from both the proposed step
    and target projection; the complete candidate still undergoes validation.
    """
    import pymeshfix
    from model_vol_septum.mesh_intersections import confirmed_self_intersections

    points, faces = np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64)
    frozen = np.zeros(len(points), dtype=bool) if fixed_vertices is None else np.asarray(fixed_vertices, dtype=bool)
    if frozen.shape != (len(points),):
        raise ValueError("Frozen-vertex mask must match the point count.")
    delta = np.asarray(delta, dtype=float).copy()
    delta[frozen] = 0.
    bad_faces = np.unique(np.asarray(bad_faces, dtype=np.int64))
    core = np.zeros(len(points), dtype=bool)
    core[np.unique(faces[bad_faces])] = True
    affected = np.zeros(len(points), dtype=bool)
    a, b = edges[:, 0], edges[:, 1]
    xyz = points[faces]
    cross = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    info = {"method": "pymeshfix_local_coordinate_transfer", "accepted": False,
            "pymeshfix_version": pymeshfix.__version__, "attempts": [],
            "backend_scope": "closed_trial_surface", "connectivity_preserved": True,
            "point_numbering_preserved": True, "target_connectivity_discarded": True,
            "initial_problem_triangle_ids": bad_faces.tolist(), "affected_vertex_ids": [],
            "affected_triangle_ids": []}
    info.update(frozen_vertex_count=int(frozen.sum()), frozen_vertex_ids=np.flatnonzero(frozen).tolist())
    candidate = points.copy()
    for attempt in range(max_attempts):
        # Increase the correction neighborhood as the proposed update decreases.
        mask = core.copy()
        for _ in range(attempt+1):
            incident = mask[a] | mask[b]
            mask[edges[incident].ravel()] = True
        mask[frozen] = False
        affected |= mask
        ids = np.flatnonzero(mask)
        info["affected_vertex_ids"] = np.flatnonzero(affected).tolist()
        info["affected_triangle_ids"] = np.flatnonzero(np.any(affected[faces], axis=1)).tolist()
        proposed_step = delta * .5**attempt
        trial = points+proposed_step
        trial[frozen] = points[frozen]
        record = {"attempt": attempt+1, "proposed_step_scale": .5**attempt,
                  "patch_rings": attempt+1, "patch_vertex_count": int(len(ids)),
                  "coordinate_transfer_trials": []}
        info["attempts"].append(record)
        try:
            correction = np.empty((0, 3))
            if len(ids):
                fixer = pymeshfix.MeshFix(trial.copy(), faces.copy())
                fixer.repair(joincomp=False, remove_smallest_components=False)
                target = fixer.mesh
                if not target.n_points or not target.n_cells or not np.isfinite(target.points).all():
                    raise RuntimeError("MeshFix returned an empty or non-finite target")
                record.update(target_points=int(target.n_points), target_triangles=int(target.n_cells))
                _, projected = target.find_closest_cell(trial[ids], return_closest_point=True)
                correction = np.asarray(projected)-trial[ids]
            else:
                record["skipped_target_no_movable_patch_vertices"] = True
            record["maximum_target_correction_mm"] = float(np.linalg.norm(correction, axis=1).max(initial=0))
        except (RuntimeError, ValueError, AssertionError) as error:
            record["error"] = str(error)
            continue
        # Blending can retain usable original triangle shapes when projection
        # alone places several source vertices on a single repaired target face.
        for blend in (1., .75, .5, .25):
            candidate = trial.copy()
            candidate[ids] += blend*correction
            candidate[frozen] = points[frozen]
            step = candidate-points
            new = candidate[faces]
            new_cross = np.cross(new[:, 1]-new[:, 0], new[:, 2]-new[:, 0])
            folded = np.flatnonzero((np.linalg.norm(new_cross, axis=1) <= 1e-12)
                                   | (np.einsum("ij,ij->i", cross, new_cross) <= 0))
            intersections = confirmed_self_intersections(candidate, faces)
            outward = all(np.all(np.einsum("ij,ij->i", step, direction) >= -1e-12)
                          for direction in (reference, normals))
            outward &= bool(np.all(np.einsum("ij,ij->i", candidate-original, reference) >= -1e-9))
            maximum_step = float(np.linalg.norm(step, axis=1).max())
            geometry_ok = not len(folded) and not len(intersections) and outward
            final_small_step = bool(geometry_ok and maximum_step < min_accepted_step
                                    and final_check is not None and final_check(candidate))
            check = {"blend": blend, "folded_triangle_ids": folded.tolist(),
                     "intersection_pairs": intersections.tolist(), "outward": bool(outward),
                     "maximum_step_mm": maximum_step, "final_small_step": final_small_step,
                     "step_large_enough": maximum_step >= min_accepted_step,
                     "outside_patch_correction_exactly_zero": bool(np.array_equal(candidate[~mask], trial[~mask]))}
            check["frozen_vertices_preserved_exactly"] = bool(np.array_equal(candidate[frozen], points[frozen]))
            record["coordinate_transfer_trials"].append(check)
            if geometry_ok and (maximum_step >= min_accepted_step or final_small_step):
                record["accepted_blend"] = blend
                info["accepted"] = True
                info["frozen_vertices_preserved_exactly"] = True
                return step, info, candidate
            problem = np.unique(np.r_[folded, intersections.ravel()]).astype(np.int64)
            if len(problem):
                core[np.unique(faces[problem])] = True
    return None, info, candidate
