"""Connectivity-preserving epicardial optimization with self-contact constraints.

The geometric clearance objective and patch constraints use SciPy. Official
IPC Toolkit (ipctk 1.4.0) supplies self-contact barrier values/derivatives and
Tight Inclusion continuous collision detection. All units are mm. No
remeshing, welding, subdivision, or point reordering is performed.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
import time
from importlib.metadata import version

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.optimize import minimize


def point_triangle_distance(point, triangle):
    """Batched closest points, including edge/vertex Voronoi regions."""
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    ab, ac, ap = b-a, c-a, point-a
    normal = np.cross(ab, ac)
    determinant = np.einsum('ij,ij->i', normal, normal)
    good = determinant > 1e-300
    denominator = np.where(good, determinant, 1.)
    # Cross-product formulas avoid cancellation of aa*cc-bb*bb on skinny
    # triangles, where treating a valid face as its edges would overestimate
    # distance and invalidate conservative advancement.
    s = np.einsum('ij,ij->i', np.cross(ap, ac), normal)/denominator
    t = np.einsum('ij,ij->i', np.cross(ab, ap), normal)/denominator
    bary = np.column_stack((1-s-t, s, t))
    closest = np.einsum('ij,ijk->ik', bary, triangle)
    squared = np.einsum('ij,ij->i', point-closest, point-closest)
    squared[~good | (bary.min(axis=1) < 0)] = np.inf
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = triangle[:, j]-triangle[:, i]
        fraction = np.clip(np.einsum('ij,ij->i', point-triangle[:, i], edge)
                           / np.maximum(np.einsum('ij,ij->i', edge, edge), 1e-300), 0, 1)
        q = triangle[:, i]+fraction[:, None]*edge
        d2 = np.einsum('ij,ij->i', point-q, point-q)
        use = d2 < squared
        closest[use], squared[use] = q[use], d2[use]
        bary[use] = 0
        bary[use, i], bary[use, j] = 1-fraction[use], fraction[use]
    return np.sqrt(np.maximum(squared, 0)), point-closest, bary


def edge_edge_distance(first, second):
    """Batched segment distances and closest-point interpolation fractions."""
    a, c = first[:, 0], second[:, 0]
    u, v, w = first[:, 1]-a, second[:, 1]-c, a-c
    uu, uv, vv = (np.einsum('ij,ij->i', x, y) for x, y in ((u, u), (u, v), (v, v)))
    uw, vw = np.einsum('ij,ij->i', u, w), np.einsum('ij,ij->i', v, w)
    normal = np.cross(u, v)
    determinant = np.einsum('ij,ij->i', normal, normal)
    good = determinant > 1e-300
    denominator = np.where(good, determinant, 1.)
    s = np.einsum('ij,ij->i', np.cross(v, w), normal)/denominator
    t = np.einsum('ij,ij->i', np.cross(u, w), normal)/denominator
    difference = w+s[:, None]*u-t[:, None]*v
    squared = np.einsum('ij,ij->i', difference, difference)
    squared[~good | (s < 0) | (s > 1) | (t < 0) | (t > 1)] = np.inf
    for ss, tt in ((np.zeros(len(a)), np.clip(vw/np.maximum(vv, 1e-300), 0, 1)),
                   (np.ones(len(a)), np.clip((vw+uv)/np.maximum(vv, 1e-300), 0, 1)),
                   (np.clip(-uw/np.maximum(uu, 1e-300), 0, 1), np.zeros(len(a))),
                   (np.clip((uv-uw)/np.maximum(uu, 1e-300), 0, 1), np.ones(len(a)))):
        diff = w+ss[:, None]*u-tt[:, None]*v
        d2 = np.einsum('ij,ij->i', diff, diff)
        use = d2 < squared
        difference[use], squared[use], s[use], t[use] = diff[use], d2[use], ss[use], tt[use]
    return np.sqrt(np.maximum(squared, 0)), difference, s, t


def contact_pairs(points, faces, edges, displacement=None, padding=0.):
    """Conservative swept-AABB broad phase; exclude only incident primitives."""
    end = points if displacement is None else points+displacement
    lower, upper = np.minimum(points, end), np.maximum(points, end)
    vlo, vhi = lower, upper
    flo, fhi = lower[faces].min(axis=1), upper[faces].max(axis=1)
    elo, ehi = lower[edges].min(axis=1), upper[edges].max(axis=1)
    vc, fc, ec = (vlo+vhi)*.5, (flo+fhi)*.5, (elo+ehi)*.5
    vr, fr, er = (np.linalg.norm(hi-lo, axis=1)*.5 for lo, hi in ((vlo, vhi), (flo, fhi), (elo, ehi)))
    neighbors = cKDTree(fc).query_ball_point(vc, vr+fr.max()+padding)
    counts = np.fromiter(map(len, neighbors), dtype=np.int64, count=len(points))
    vi = np.repeat(np.arange(len(points)), counts)
    fi = np.concatenate(neighbors).astype(np.int64) if counts.sum() else np.empty(0, dtype=np.int64)
    keep = (np.all(vlo[vi] <= fhi[fi]+padding, axis=1)
            & np.all(flo[fi] <= vhi[vi]+padding, axis=1)
            & ~np.any(faces[fi] == vi[:, None], axis=1))
    vf = np.column_stack((vi[keep], fi[keep]))
    ee = cKDTree(ec).query_pairs(2*er.max()+padding, output_type='ndarray')
    a, b = ee[:, 0], ee[:, 1]
    keep = (np.all(elo[a] <= ehi[b]+padding, axis=1)
            & np.all(elo[b] <= ehi[a]+padding, axis=1)
            & ~np.any(edges[a, :, None] == edges[b, None, :], axis=(1, 2)))
    return vf, ee[keep]


def contact_geometry(points, faces, edges, vf, ee):
    """Distances and their sparse vertex-coordinate Jacobian."""
    dv, vv, bary = point_triangle_distance(points[vf[:, 0]], points[faces[vf[:, 1]]])
    de, ve, s, t = edge_edge_distance(points[edges[ee[:, 0]]], points[edges[ee[:, 1]]])
    distance = np.r_[dv, de]
    ids = np.vstack((np.column_stack((vf[:, 0], faces[vf[:, 1]])),
                     np.column_stack((edges[ee[:, 0]], edges[ee[:, 1]]))))
    coefficients = np.vstack((np.column_stack((np.ones(len(vf)), -bary)),
                              np.column_stack((1-s, s, t-1, -t))))
    difference = np.vstack((vv, ve))
    direction = difference/np.maximum(distance[:, None], 1e-300)
    return distance, ids, coefficients, direction


def _jacobian(ids, coefficients, directions, npoints):
    values = coefficients[:, :, None]*directions[:, None, :]
    columns = ids[:, :, None]*3+np.arange(3)
    rows = np.broadcast_to(np.arange(len(ids))[:, None, None], values.shape)
    return coo_matrix((values.ravel(), (rows.ravel(), columns.ravel())),
                      shape=(len(ids), 3*npoints)).tocsr()


class _IpcContact:
    """Official IPC Toolkit 1.4 contact barrier and continuous collision API."""

    def __init__(self, original, faces, edges, activation_distance):
        try:
            import ipctk
        except ImportError as error:
            command = shlex.join([sys.executable, '-m', 'pip', 'install', 'ipctk==1.4.0'])
            raise RuntimeError(f'Constrained optimization requires IPC Toolkit in {sys.executable}. '
                f'Import failed: {error}. Install with {command}') from error
        self.ipc = ipctk
        self.package_version = version('ipctk')
        if self.package_version != '1.4.0':
            raise RuntimeError(f'Unsupported ipctk version {self.package_version}; this backend is tested with ipctk==1.4.0.')
        self.center = np.asarray(original, dtype=float).mean(axis=0)
        self.shape = np.shape(original)
        self.mesh = ipctk.CollisionMesh(self.positions(original),
            np.asfortranarray(edges, dtype=np.int32), np.asfortranarray(faces, dtype=np.int32))
        if self.mesh.num_vertices != len(original):
            raise RuntimeError('IPC collision mesh changed the vertex count.')
        self.potential = ipctk.BarrierPotential(float(activation_distance))
        # IPC 1.4 does not include stiffness in BarrierPotential. Apply it to
        # the value, gradient, and PSD-projected Hessian consistently.
        self.stiffness = .05/float(activation_distance)**2
        self.ccd_tolerance = 1e-9
        self.narrow_phase = ipctk.TightInclusionCCD(tolerance=self.ccd_tolerance,
                                                   conservative_rescaling=.9)

    def positions(self, points):
        values = np.asarray(points, dtype=float)
        if values.shape != self.shape or not np.all(np.isfinite(values)):
            raise ValueError('IPC positions must be finite and preserve the original point-array shape.')
        return np.asfortranarray(values-self.center)

    def validate_initial(self, points):
        if self.ipc.has_intersections(self.mesh, self.positions(points), broad_phase=self.ipc.HashGrid()):
            raise RuntimeError('IPC Toolkit detected intersections in the initial constrained surface.')

    def evaluate(self, points, derivatives=False):
        x = self.positions(points)
        collisions = self.ipc.NormalCollisions()
        collisions.build(self.mesh, x, self.potential.dhat, broad_phase=self.ipc.HashGrid())
        energy = self.stiffness*float(self.potential(collisions, self.mesh, x))
        if not derivatives:
            return energy
        gradient = self.stiffness*np.asarray(self.potential.gradient(collisions, self.mesh, x)).ravel()
        hessian = self.stiffness*self.potential.hessian(
            collisions, self.mesh, x, self.ipc.PSDProjectionMethod.CLAMP)
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian.data)):
            raise RuntimeError('IPC Toolkit produced non-finite contact derivatives.')
        return energy, gradient, hessian, len(collisions)

    def collision_free_step(self, points, displacement):
        first, last = self.positions(points), self.positions(points+displacement)
        candidates = self.ipc.Candidates()
        candidates.build(self.mesh, first, last, broad_phase=self.ipc.HashGrid())
        alpha = float(candidates.compute_collision_free_stepsize(
            self.mesh, first, last, narrow_phase_ccd=self.narrow_phase))
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise RuntimeError('IPC Toolkit returned an invalid collision-free step fraction.')
        details = {'backend': 'ipctk', 'version': self.package_version,
                   'narrow_phase': 'TightInclusionCCD', 'tolerance_mm': self.ccd_tolerance,
                   'candidate_pairs': len(candidates), 'unresolved_pairs': None,
                   'limiting_reason': 'complete' if alpha == 1 else 'ipc_collision_free_bound'}
        return alpha, len(candidates), details


def collision_free_step(points, displacement, faces, edges, return_details=False, contact=None):
    """Use official IPC Toolkit swept candidates and Tight Inclusion CCD."""
    if contact is None:
        contact = _IpcContact(points, faces, edges, .01)
    result = contact.collision_free_step(points, displacement)
    return result if return_details else result[:2]


def _unfolded_path(points, step, faces, alpha):
    """Check oriented triangle area throughout the linear update, not only its end."""
    xyz, delta = points[faces], step[faces]
    a, b = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
    da, db = delta[:, 1]-delta[:, 0], delta[:, 2]-delta[:, 0]
    cross = np.cross(a, b)
    area2 = np.einsum('ij,ij->i', cross, cross)
    linear = np.einsum('ij,ij->i', cross, np.cross(da, b)+np.cross(a, db))
    quadratic = np.einsum('ij,ij->i', cross, np.cross(da, db))
    end = area2+alpha*linear+alpha*alpha*quadratic
    minimum = np.minimum(area2, end)
    convex = quadratic > 0
    t = np.zeros(len(faces))
    t[convex] = -linear[convex]/(2*quadratic[convex])
    inside = convex & (t > 0) & (t < alpha)
    minimum[inside] = np.minimum(minimum[inside],
        area2[inside]+t[inside]*linear[inside]+t[inside]**2*quadratic[inside])
    return bool(np.all(minimum > np.maximum(area2*1e-6, 1e-24)))


def _path_area_constraints(points, step, faces):
    """Minimum normalized oriented area over the path and its endpoint gradient."""
    xyz, delta = points[faces], step[faces]
    a, b = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
    da, db = delta[:, 1]-delta[:, 0], delta[:, 2]-delta[:, 0]
    cross = np.cross(a, b)
    area = np.linalg.norm(cross, axis=1)
    normal = cross/area[:, None]
    linear = np.einsum('ij,ij->i', normal, np.cross(da, b)+np.cross(a, db))/area
    quadratic = np.einsum('ij,ij->i', normal, np.cross(da, db))/area
    when = np.ones(len(faces))
    positive = quadratic > 0
    when[positive] = np.clip(-linear[positive]/(2*quadratic[positive]), 0, 1)
    value = 1+when*linear+when*when*quadratic
    # If the minimum is the initial point, its derivative is zero.
    when[value > 1] = 0
    value = np.minimum(value, 1.)
    first = np.cross(b+when[:, None]*db, normal)
    second = np.cross(normal, a+when[:, None]*da)
    gradient = np.stack((-first-second, first, second), axis=1)
    gradient *= (when/area)[:, None, None]
    columns = faces[:, :, None]*3+np.arange(3)
    rows = np.broadcast_to(np.arange(len(faces))[:, None, None], gradient.shape)
    J = coo_matrix((gradient.ravel(), (rows.ravel(), columns.ravel())),
                   shape=(len(faces), points.size)).tocsr()
    return value, J


def _constrain_direction(points, proposed, faces, edges, original, reference, contact_distance,
                          max_patch_vertices=512, fixed_vertices=None):
    """Repair only threatened patches using contact-plane and path-area constraints.

    Contact inequalities retain half the current separating-plane gap for
    every cross-primitive vertex pair. Triangle inequalities retain 20% of
    oriented area throughout the linear path, including interior-time folds.
    All vertices outside each selected patch retain the proposed direction.
    The final global CCD/area/energy checks remain mandatory.
    """
    step, goal_step = proposed.copy(), proposed.copy()
    frozen = np.zeros(len(points), dtype=bool) if fixed_vertices is None else np.asarray(fixed_vertices, dtype=bool)
    if frozen.shape != (len(points),):
        raise ValueError('Frozen-vertex mask must match the point count.')
    step[frozen], goal_step[frozen] = 0., 0.
    vf, ee = contact_pairs(points, faces, edges, padding=contact_distance)
    distance, ids, _, normals = contact_geometry(points, faces, edges, vf, ee)
    near = distance < contact_distance
    vf_kind = np.arange(len(distance)) < len(vf)
    contact_ids, contact_normals, contact_lower = [], [], []
    for k in range(4):
        valid = near & ((k < 3) | ~vf_kind)
        selected = np.flatnonzero(valid)
        if not len(selected):
            continue
        is_vf = vf_kind[selected]
        i = np.where(is_vf, ids[selected, 0], ids[selected, k//2])
        j = np.where(is_vf, ids[selected, min(k+1, 3)], ids[selected, 2+k % 2])
        gap = np.einsum('ij,ij->i', points[i]-points[j], normals[selected])
        contact_ids.append(np.column_stack((i, j)))
        contact_normals.append(normals[selected])
        contact_lower.append(-.5*gap)
    pairs = np.vstack(contact_ids) if contact_ids else np.empty((0, 2), dtype=int)
    directions = np.vstack(contact_normals) if contact_normals else np.empty((0, 3))
    lower = np.concatenate(contact_lower) if contact_lower else np.empty(0)
    contact_J = _jacobian(pairs, np.tile([1., -1.], (len(pairs), 1)), directions, len(points))
    active_contact = np.zeros(len(pairs), dtype=bool)
    active_faces = np.zeros(len(faces), dtype=bool)
    info = {'patch_vertices': 0, 'contact_constraints': 0, 'area_constraints': 0,
            'projection_passes': 0, 'projection_success': True, 'trust_region_reductions': 0}
    for refinement in range(4):
        violated = contact_J @ step.ravel() < lower-1e-11
        area_values, _ = _path_area_constraints(points, step, faces)
        threatened = area_values < .2
        if not np.any(violated) and not np.any(threatened):
            break
        active_contact |= violated
        active_faces |= threatened
        selected_faces = faces[active_faces]
        patch_ids = np.unique(np.r_[pairs[active_contact].ravel(), selected_faces.ravel()])
        vertex_ids = patch_ids[~frozen[patch_ids]]
        if not len(vertex_ids):
            info['projection_success'] = False
            break
        # Bound dense subproblem memory before converting any sparse rows.
        # Large patches reduce the trust region and rebuild their active set.
        if len(vertex_ids) > max_patch_vertices or active_contact.sum() > 4096 or len(selected_faces) > 4096:
            goal_step *= .5
            step *= .5
            active_contact[:] = False
            active_faces[:] = False
            info['trust_region_reductions'] += 1
            info['projection_success'] = False
            continue
        columns = (vertex_ids[:, None]*3+np.arange(3)).ravel()
        matrix = contact_J[active_contact][:, columns].toarray()
        rhs = lower[active_contact]
        # Every selected contact and triangle is wholly inside the patch;
        # zero patch displacement is a feasible starting point.
        outward = np.zeros((len(vertex_ids), len(columns)))
        outward[np.arange(len(vertex_ids))[:, None], np.arange(len(columns)).reshape(-1, 3)] = reference[vertex_ids]
        outward_rhs = -np.einsum('ij,ij->i', points[vertex_ids]-original[vertex_ids], reference[vertex_ids])
        matrix = np.vstack((matrix, outward))
        rhs = np.r_[rhs, outward_rhs]
        goal = goal_step[vertex_ids].ravel()
        def objective(value):
            difference = value-goal
            return .5*float(difference @ difference), difference
        def area(value, jacobian=False):
            trial = step.copy()
            trial[vertex_ids] = value.reshape(-1, 3)
            values, J = _path_area_constraints(points, trial, selected_faces)
            return J[:, columns].toarray() if jacobian else values-.2
        constraints = [{'type': 'ineq', 'fun': lambda x: matrix @ x-rhs,
                        'jac': lambda x: matrix}]
        if len(selected_faces):
            constraints.append({'type': 'ineq', 'fun': area,
                                'jac': lambda x: area(x, jacobian=True)})
        result = minimize(objective, np.zeros(len(columns)), jac=True, method='SLSQP',
                          constraints=constraints, options={'ftol': 1e-12, 'maxiter': 150})
        feasible = (np.all(np.isfinite(result.x)) and np.min(matrix @ result.x-rhs) >= -1e-10
                    and (not len(selected_faces) or np.min(area(result.x)) >= -1e-8))
        step[vertex_ids] = result.x.reshape(-1, 3) if feasible else 0.
        info.update(patch_vertices=len(vertex_ids), contact_constraints=int(active_contact.sum()),
                    area_constraints=int(active_faces.sum()), projection_passes=refinement+1,
                    projection_success=bool(feasible))
    # Projection can increase an individual vertex's movement. Rescaling the
    # complete direction preserves all already-certified path inequalities.
    proposed_max = float(np.linalg.norm(goal_step, axis=1).max())
    actual_max = float(np.linalg.norm(step, axis=1).max())
    step *= min(1., proposed_max/max(actual_max, 1e-30))
    info['remaining_contact_violations'] = int(np.sum(contact_J @ step.ravel() < lower-1e-10))
    info['remaining_area_violations'] = int(np.sum(_path_area_constraints(points, step, faces)[0] < .2-1e-8))
    return step, info


def _outward_newton_step(hessian, gradient, displacement, normals, fixed_vertices=None):
    """Active-set quadratic solve for cumulative outward halfspace bounds.

    Euclidean projection of an unconstrained Newton direction can destroy its
    self-contact response. Solve the bounds in the Hessian metric instead.
    """
    # Rotate each vertex into two tangent coordinates and one normal
    # coordinate. Active bounds can then be eliminated from an SPD system,
    # avoiding an indefinite saddle-point factorization with extra DOFs.
    n = len(normals)
    frozen = np.zeros(n, dtype=bool) if fixed_vertices is None else np.asarray(fixed_vertices, dtype=bool)
    if frozen.shape != (n,):
        raise ValueError('Frozen-vertex mask must match the point count.')
    frozen_dofs = np.repeat(frozen, 3)
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    tangent = np.cross(normals, axis)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    blocks = np.stack((tangent, np.cross(normals, tangent), normals), axis=2)
    rows = np.repeat(np.arange(3*n), 3)
    columns = np.repeat(3*np.arange(n), 9)+np.tile(np.tile(np.arange(3), 3), n)
    rotation = coo_matrix((blocks.ravel(), (rows, columns)), shape=(3*n, 3*n)).tocsr()
    hessian = (rotation.T @ hessian @ rotation).tocsc()
    gradient = rotation.T @ gradient
    lower = -np.einsum('ij,ij->i', displacement, normals)
    step = np.zeros(len(gradient))
    if np.any(~frozen_dofs):
        step[~frozen_dofs] = spsolve(hessian[~frozen_dofs][:, ~frozen_dofs].tocsc(), -gradient[~frozen_dofs])
    active = np.zeros(len(normals), dtype=bool)
    release = np.empty(0, dtype=int)
    for _ in range(30):
        violation = (step[2::3] < lower-1e-10) & ~frozen
        if np.any(violation & ~active):
            active |= violation
        elif len(release):
            active[release] = False
        else:
            return (rotation @ step).reshape(-1, 3)
        ids = np.flatnonzero(active)
        if not len(ids):
            step = np.zeros(len(gradient))
            if np.any(~frozen_dofs):
                step[~frozen_dofs] = spsolve(hessian[~frozen_dofs][:, ~frozen_dofs].tocsc(), -gradient[~frozen_dofs])
            release = np.empty(0, dtype=int)
            continue
        fixed = ids*3+2
        free = ~frozen_dofs.copy()
        free[fixed] = False
        step = np.zeros(len(gradient))
        step[fixed] = lower[ids]
        matrix = hessian[free]
        rhs = -gradient[free]-matrix[:, fixed] @ step[fixed]
        if np.any(free):
            step[free] = spsolve(matrix[:, free].tocsc(), rhs)
        multipliers = (hessian @ step+gradient)[fixed]
        release = ids[multipliers < -1e-9]
    raise RuntimeError('Outward halfspace quadratic subproblem did not converge')


def optimize_closed_epi(epi, endo, minimum=3., max_iterations=600,
                        diagnostics_dir=None, chamber='RV', contact_distance=None,
                        smooth_weight=.02, initial_points=None, freeze_satisfied=False):
    """Solve clearance and self-contact; optionally resume an accepted state.

    ``epi`` always defines the original connectivity, displacement reference,
    and contact rest distances, even when ``initial_points`` supplies a warm start.
    ``freeze_satisfied`` locks entry vertices with signed clearance >= minimum
    to their entry coordinates for the entire solve.
    """
    import open3d as o3d
    import pyvista as pv
    from model_vol_septum.clean_step02_surfaces import validate_clean_surface
    from model_vol_septum.generate_open_volmesh import (
        validate_closed_epi, _surface_gap, _surface_object, _clearance_freeze_state,
        _validate_frozen_vertices,
    )
    from model_vol_septum.mesh_intersections import confirmed_self_intersections

    started = time.monotonic()
    if not np.isfinite(minimum) or minimum <= 0 or max_iterations < 1:
        raise ValueError('Clearance and iteration limit must be positive.')
    if not np.isfinite(smooth_weight) or smooth_weight < 0:
        raise ValueError('Optimization smoothing weight must be finite and nonnegative.')
    validate_clean_surface(epi)
    validate_clean_surface(endo)
    original = np.asarray(epi.points, dtype=float).copy()
    faces = epi.faces.reshape(-1, 4)[:, 1:].copy()
    edges = np.unique(np.sort(np.vstack((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1), axis=0)
    n = len(original)
    lengths = np.linalg.norm(original[edges[:, 0]]-original[edges[:, 1]], axis=1)
    scale = float(np.median(lengths))
    dhat = .05*scale if contact_distance is None else contact_distance
    if not np.isfinite(dhat) or dhat <= 0:
        raise ValueError('Contact activation distance must be positive and finite.')
    reference = np.asarray(epi.compute_normals(point_normals=True, cell_normals=False,
        consistent_normals=True, auto_orient_normals=True, split_vertices=False).point_data['Normals'], dtype=float)
    reference /= np.linalg.norm(reference, axis=1)[:, None]
    center = np.mean(endo.points, axis=0)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor((endo.points-center).astype(np.float32)),
        o3d.core.Tensor(endo.faces.reshape(-1, 4)[:, 1:].astype(np.uint32)))
    # Sample vertices, unique edge midpoints, and triangle centers. Final
    # acceptance additionally checks the complete triangle-to-triangle gap.
    sample_ids = np.vstack((np.column_stack((np.arange(n),)*3),
                           np.column_stack((edges, edges[:, 1])), faces))
    sample_weights = np.vstack((np.tile([1., 0., 0.], (n, 1)),
                               np.tile([.5, .5, 0.], (len(edges), 1)),
                               np.full((len(faces), 3), 1/3)))
    a, b = edges[:, 0], edges[:, 1]
    weights = np.minimum(scale/np.maximum(lengths, scale*1e-3), 10.)
    laplace = coo_matrix((np.r_[weights, weights, -weights, -weights],
        (np.r_[a, b, a, b], np.r_[a, b, b, a])), shape=(n, n)).tocsr()
    regularizer = kron(smooth_weight*laplace+eye(n, format='csr')*1e-5, eye(3), format='csr')
    target = minimum+max(.1, minimum*.03)
    points = original.copy() if initial_points is None else np.asarray(initial_points, dtype=float).copy()
    if points.shape != original.shape or not np.all(np.isfinite(points)):
        raise ValueError('Constrained initial points must be finite and match the original point array.')
    if np.any(np.einsum('ij,ij->i', points-original, reference) < -1e-9):
        raise ValueError('Constrained initial points violate the original outward displacement bounds.')
    if initial_points is not None:
        initial_surface = epi.copy(deep=True)
        initial_surface.points = points.copy()
        validate_clean_surface(initial_surface)
    frozen = np.zeros(n, dtype=bool)
    frozen_reference = points.copy()
    freeze_info = {'enabled': bool(freeze_satisfied), 'activated': bool(freeze_satisfied), 'frozen_vertex_count': 0}
    if freeze_satisfied:
        entry_surface = epi.copy(deep=True)
        entry_surface.points = points.copy()
        frozen, freeze_info = _clearance_freeze_state(entry_surface, endo, minimum)
    movable_samples = np.any((sample_weights != 0) & ~frozen[sample_ids], axis=1)
    contact_backend = _IpcContact(original, faces, edges, dhat)
    contact_backend.validate_initial(points)
    print(f'[Step03 {chamber}epi constrained] contact_backend=ipctk '
          f'version={contact_backend.package_version} ccd=TightInclusionCCD '
          f'frozen_vertices={int(frozen.sum())}', flush=True)
    initial_displacement = float(np.linalg.norm(points-original, axis=1).max())
    history = []
    last_direction = np.zeros_like(points)
    max_step = min(1., .5*scale)
    fixed_object = _surface_object(endo)

    def sdf(samples):
        query = o3d.core.Tensor((samples-center).astype(np.float32))
        distance = scene.compute_signed_distance(query, nthreads=4, nsamples=3).numpy().astype(float)
        closest = scene.compute_closest_points(query, nthreads=4)
        difference = samples-(closest['points'].numpy().astype(float)+center)
        direction = difference/np.maximum(np.linalg.norm(difference, axis=1)[:, None], 1e-30)
        direction[distance < 0] *= -1
        on_surface = np.abs(distance) < 1e-7
        direction[on_surface] = closest['primitive_normals'].numpy()[on_surface]
        return distance, direction

    def evaluate(x, derivatives=False):
        samples = np.einsum('ij,ijk->ik', sample_weights, x[sample_ids])
        distances, direction = sdf(samples)
        deficit = np.maximum(target-distances, 0)
        u = (x-original).ravel()
        energy = .5*float(u @ (regularizer @ u))+.5*float(deficit @ deficit)
        if not derivatives:
            return energy+contact_backend.evaluate(x)
        barrier_energy, barrier_gradient, barrier_hessian, contacts = contact_backend.evaluate(x, derivatives=True)
        energy += barrier_energy
        J = _jacobian(sample_ids, sample_weights, direction, n)
        J = diags((deficit > 0).astype(float)) @ J
        gradient = regularizer @ u-J.T @ deficit
        hessian = regularizer+J.T @ J
        gradient += barrier_gradient
        hessian += barrier_hessian
        return energy, gradient, hessian, distances, contacts

    def fail(message):
        report = {'method': 'collision_aware_constrained_optimization', 'status': 'failed',
                  'error': message, 'iterations': len(history), 'history': history,
                  'elapsed_seconds': time.monotonic()-started,
                  'warm_start': initial_points is not None,
                  'initial_maximum_displacement_mm': initial_displacement,
                  'contact_backend': {'package': 'ipctk', 'version': contact_backend.package_version},
                  'freeze': freeze_info,
                  'connectivity_preserved': True, 'final_output_published': False}
        if diagnostics_dir is not None:
            folder = Path(diagnostics_dir)/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            folder.mkdir(parents=True)
            mesh = epi.copy(deep=True)
            mesh.points = points.copy()
            if freeze_satisfied:
                mesh.point_data['clearanceFrozen'] = frozen.astype(np.int32)
                mesh.point_data['freezeReferencePosition'] = frozen_reference
            mesh.point_data['proposedDirection'] = last_direction
            mesh.field_data['diagnosticOnly'] = np.array([1], dtype=np.int32)
            mesh.save(folder/f'{chamber}epi_last_accepted.vtk', binary=False)
            (folder/'optimization_summary.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'[Step03 {chamber}epi constrained] Failure diagnostics: {folder}', flush=True)
        raise RuntimeError(message+'; no final output published.')

    validation = None
    if freeze_info.get('minimum_frozen_triangle_distance_mm') is not None and freeze_info['minimum_frozen_triangle_distance_mm'] < minimum:
        fail('Frozen vertices prevent the required triangle clearance: a fully frozen triangle is too close to the endocardium')
    for iteration in range(max_iterations+1):
        energy, gradient, hessian, distances, contacts = evaluate(points, derivatives=True)
        if not np.isfinite(energy):
            fail('Non-finite optimization energy or unresolved initial contact')
        if np.all(distances[movable_samples] >= minimum):
            trial = epi.copy(deep=True)
            trial.points = points.copy()
            gap, _ = _surface_gap(trial, fixed_object)
            if gap >= minimum:
                try:
                    validation = validate_closed_epi(trial, endo, minimum, source=epi, chamber=chamber)
                except RuntimeError as error:
                    fail(str(error))
                break
        if iteration == max_iterations:
            fail(f'{chamber}epi constrained optimization exceeded {max_iterations} iterations')
        if np.all(frozen):
            fail('All vertices are frozen but final clearance/containment validation is not satisfied')
        try:
            step = _outward_newton_step(hessian, gradient, points-original, reference, fixed_vertices=frozen)
        except RuntimeError as error:
            fail(str(error))
        if not np.all(np.isfinite(step)):
            fail('Non-finite sparse optimization direction')
        # Remove only numerical halfspace residuals from the constrained solve.
        proposed = points+step-original
        step -= np.minimum(np.einsum('ij,ij->i', proposed, reference), 0)[:, None]*reference
        step[frozen] = 0.
        magnitude = float(np.linalg.norm(step, axis=1).max())
        if magnitude > max_step:
            step *= max_step/magnitude
        slope = float(gradient @ step.ravel())
        if slope >= 0:
            step = -gradient.reshape(-1, 3)/np.maximum(hessian.diagonal().reshape(-1, 3), 1e-12)
            proposed = points+step-original
            step -= np.minimum(np.einsum('ij,ij->i', proposed, reference), 0)[:, None]*reference
            step[frozen] = 0.
            step *= min(1., max_step/max(float(np.linalg.norm(step, axis=1).max()), 1e-30))
            slope = float(gradient @ step.ravel())
        if slope >= 0:
            fail('No feasible descent direction under outward constraints')
        proposed_maximum = float(np.linalg.norm(step, axis=1).max())
        step, direction_info = _constrain_direction(points, step, faces, edges, original, reference, dhat,
                                                   fixed_vertices=frozen)
        slope = float(gradient @ step.ravel())
        if slope >= 0:
            # Projection of a scaled negative gradient has the zero direction
            # as a feasible competitor. Retry with that descent target if the
            # Newton correction conflicts with contact/area inequalities.
            descent = -gradient.reshape(-1, 3)
            descent[frozen] = 0.
            descent *= max_step/max(float(np.linalg.norm(descent, axis=1).max()), 1e-30)
            step, direction_info = _constrain_direction(points, descent, faces, edges, original, reference, dhat,
                                                       fixed_vertices=frozen)
            direction_info['gradient_recovery'] = True
            slope = float(gradient @ step.ravel())
        if slope >= 0:
            fail('Contact and area constraints leave no feasible descent direction')
        last_direction = step.copy()
        alpha, candidates, ccd_info = collision_free_step(points, step, faces, edges,
                                                         return_details=True, contact=contact_backend)
        ccd_fraction = alpha
        accepted = False
        for half in range(25):
            if alpha < 1e-12:
                break
            if _unfolded_path(points, step, faces, alpha):
                candidate = points+alpha*step
                candidate[frozen] = frozen_reference[frozen]
                value = evaluate(candidate)
                if np.isfinite(value) and value <= energy+1e-4*alpha*slope:
                    if not len(confirmed_self_intersections(candidate, faces)):
                        accepted = True
                        break
            alpha *= .5
        record = {'iteration': iteration+1, 'energy': float(energy),
                  'minimum_sample_distance_mm': float(np.min(distances)),
                  'proposed_maximum_step_mm': proposed_maximum,
                  'active_contacts': contacts, 'ccd_candidates': candidates,
                  'ccd': ccd_info, 'direction_constraints': direction_info,
                  'ccd_step_fraction': ccd_fraction,
                  'step_fraction': alpha, 'line_search_halvings': half,
                  'maximum_step_mm': float(alpha*np.linalg.norm(step, axis=1).max())}
        history.append(record)
        if iteration % 5 == 0 or not accepted:
            print(f'[Step03 {chamber}epi constrained] iteration={iteration+1} '
                  f'minimum_sample_distance={distances.min():.6g} mm '
                  f'contacts={contacts} step={record["maximum_step_mm"]:.6g} mm '
                  f'fraction={alpha:.6g} ccd_limit={ccd_info["limiting_reason"]} '
                  f'unresolved={ccd_info["unresolved_pairs"]} '
                  f'patch_vertices={direction_info["patch_vertices"]}', flush=True)
        if not accepted:
            fail('Collision-aware line search could not find a feasible descent step')
        points = candidate
    result = epi.copy(deep=True)
    result.points = points
    for data in (result.point_data, result.cell_data):
        if 'Normals' in data:
            del data['Normals']
    result.point_data['displacement'] = points-original
    result.point_data['outwardDisplacement'] = np.einsum('ij,ij->i', points-original, reference)
    if freeze_satisfied:
        result.point_data['clearanceFrozen'] = frozen.astype(np.int32)
        result.point_data['freezeReferencePosition'] = frozen_reference
        validation.update(_validate_frozen_vertices(result))
    result.cell_data['elemTag'] = np.full(result.n_cells, int(chamber == 'LV'), dtype=np.int32)
    return result, {'method': 'collision_aware_constrained_optimization', 'iterations': iteration,
                    'elapsed_seconds': time.monotonic()-started,
                    'warm_start': initial_points is not None,
                    'initial_maximum_displacement_mm': initial_displacement,
                    'freeze': freeze_info,
                    'minimum_requested_distance_mm': minimum, 'target_distance_mm': target,
                    'contact_activation_distance_mm': dhat, 'smooth_weight': smooth_weight,
                    'contact_backend': {'package': 'ipctk', 'version': contact_backend.package_version,
                                        'barrier': 'BarrierPotential', 'stiffness': contact_backend.stiffness},
                    'collision_detection': 'ipctk.Candidates_TightInclusionCCD',
                    'direction_constraints': 'contact_halfspaces_and_minimum_path_area',
                    'connectivity_preserved': True, 'point_numbering_preserved': True,
                    'validation': validation, 'history': history}
