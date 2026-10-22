"""Outward RVepi deformation against fixed open RVendo (coordinates in mm).

The signed solver uses the cleaned closed RVendo for inside/outside. The
legacy normal-only trial uses diagnostic basal closures. Only real open
triangles enter clearance queries. Invalid inputs are rejected.
"""
from __future__ import annotations

import importlib
import numpy as np
import open3d as o3d
import pyvista as pv
import shapely
import vtk
from scipy.sparse import coo_matrix, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized
from shapely.geometry import Polygon
from model_vol_septum.mesh_intersections import confirmed_self_intersections


def triangles(mesh):
    if not mesh.is_all_triangles or mesh.n_cells == 0:
        raise RuntimeError("Step 06 requires a nonempty triangle-only surface.")
    return np.asarray(mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int64)


def _surface(points, faces):
    return pv.PolyData(np.asarray(points, dtype=np.float64),
                       np.column_stack((np.full(len(faces), 3), faces)).ravel())


def scene(mesh):
    result = o3d.t.geometry.RaycastingScene()
    result.add_triangles(o3d.core.Tensor(np.asarray(mesh.points, dtype=np.float32)),
                         o3d.core.Tensor(triangles(mesh).astype(np.uint32)))
    return result


def query(solver, points):
    return o3d.core.Tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3))


def _edges(faces):
    return np.unique(np.sort(np.vstack((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1), axis=0)


def _opening_indices(mesh):
    faces = triangles(mesh)
    edges, counts = np.unique(np.sort(np.vstack((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1),
                              axis=0, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError("Open-base surface has nonmanifold edges.")
    adjacent = {}
    for a, b in edges[counts == 1]:
        adjacent.setdefault(int(a), []).append(int(b))
        adjacent.setdefault(int(b), []).append(int(a))
    if not adjacent or any(len(neighbors) != 2 for neighbors in adjacent.values()):
        raise RuntimeError("The opening must be a simple closed boundary loop.")
    ordered, previous = [min(adjacent)], None
    for _ in range(len(adjacent)):
        current = ordered[-1]
        nxt = next(v for v in adjacent[current] if v != previous)
        if nxt == ordered[0]:
            break
        previous = current
        ordered.append(nxt)
    if len(ordered) != len(adjacent):
        raise RuntimeError("Step 06 expects one basal opening, but found additional boundary loops.")
    return np.asarray(ordered, dtype=np.int64)


def _self_intersections(points, faces):
    return confirmed_self_intersections(points, faces)


def _check_geometry(mesh, name, closed=False):
    points, faces = np.asarray(mesh.points, dtype=float), triangles(mesh)
    if not np.all(np.isfinite(points)):
        raise RuntimeError(f"{name} contains non-finite coordinates.")
    xyz = points[faces]
    area2 = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if np.any(area2 <= 1e-12):
        raise RuntimeError(f"{name} contains degenerate triangles {np.flatnonzero(area2 <= 1e-12)[:12].tolist()}.")
    edges = _edges(faces)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points), len(points)))
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise RuntimeError(f"{name} must be one connected surface without unused vertices.")
    if closed:
        if mesh.n_open_edges or not mesh.is_manifold:
            raise RuntimeError(f"{name} is not a closed manifold.")
    else:
        _opening_indices(mesh)
    pairs = _self_intersections(points, faces)
    if len(pairs):
        raise RuntimeError(f"{name} contains {len(pairs)} self-intersections; repair Step 05 first. "
                           f"Triangle pairs: {pairs[:12].tolist()}.")


def _basal_reference(mesh, ring_ids):
    points = np.asarray(mesh.points, dtype=float)
    ring = points[ring_ids]
    _, _, axes = np.linalg.svd(ring - ring.mean(axis=0), full_matrices=False)
    axis = axes[-1]
    if np.dot(axis, ring.mean(axis=0) - points.mean(axis=0)) < 0:
        axis *= -1
    return axis, float(np.max(points @ axis) + mesh.length)


def _plane_basis(axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    u = np.eye(3)[np.argmin(np.abs(axis))]
    u = u - np.dot(u, axis) * axis
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(axis, u)))


def _ring_polygon(ring, basis):
    polygon = Polygon(np.asarray(ring) @ basis)
    if not polygon.is_valid or polygon.area <= 1e-10:
        raise RuntimeError("Basal ring projection is not a simple polygon; cannot define an outward opening.")
    return polygon


def close_for_inside_test(mesh, axis=None, plane=None):
    """Close a diagnostic copy while preserving real points and face indices."""
    ring_ids = _opening_indices(mesh)
    if axis is None:
        axis, default_plane = _basal_reference(mesh, ring_ids)
        if plane is None:
            plane = default_plane
    if plane is None:
        plane = float(np.max(np.asarray(mesh.points) @ axis) + mesh.length)
    _ring_polygon(mesh.points[ring_ids], _plane_basis(axis))
    points = np.asarray(mesh.points, dtype=np.float64)
    ring = points[ring_ids]
    if np.any(points @ axis >= plane):
        raise RuntimeError("Diagnostic basal plane must lie beyond all real points.")
    extended = ring + (plane - ring @ axis)[:, None] * axis
    outer = np.arange(len(points), len(points) + len(ring), dtype=np.int64)
    nxt, outer_nxt = np.roll(ring_ids, -1), np.roll(outer, -1)
    curtain = np.vstack((np.column_stack((ring_ids, nxt, outer_nxt)),
                         np.column_stack((ring_ids, outer_nxt, outer))))
    copy = _surface(np.vstack((points, extended)), np.vstack((triangles(mesh), curtain)))
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(copy)
    filler.SetHoleSize(float(copy.length) * 2)
    filler.Update()
    closed = pv.wrap(filler.GetOutput()).triangulate()
    if not np.array_equal(points, closed.points[:len(points)]):
        raise RuntimeError("Diagnostic capping changed original vertices.")
    _check_geometry(closed, "Diagnostic basal closure", closed=True)
    return closed.compute_normals(point_normals=False, cell_normals=True, consistent_normals=True,
                                  auto_orient_normals=True, split_vertices=False)


def _vertex_normals(points, faces):
    xyz = points[faces]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    normals = np.zeros_like(points, dtype=float)
    for k in range(3):
        np.add.at(normals, faces[:, k], cross)
    length = np.linalg.norm(normals, axis=1)
    if np.any(length <= 1e-12):
        raise RuntimeError("RVepi has undefined outward vertex normals.")
    return normals / length[:, None]


def outward_normals(mesh, axis=None, plane=None):
    oriented = close_for_inside_test(mesh, axis, plane)
    return _vertex_normals(np.asarray(mesh.points, dtype=float), triangles(oriented)[:mesh.n_cells])


class FixedEndo:
    def __init__(self, mesh, axis=None, plane=None, closed_reference=None):
        self.points = np.asarray(mesh.points, dtype=float).copy()
        self.faces = triangles(mesh).copy()
        self.reference_supplied = closed_reference is not None
        if closed_reference is None:
            self.closed_mesh = close_for_inside_test(mesh, axis, plane)
            oriented_open = _surface(self.points, triangles(self.closed_mesh)[:mesh.n_cells])
        else:
            _check_geometry(closed_reference, "Closed RVendo reference", closed=True)
            self.closed_mesh = closed_reference.copy(deep=True).compute_normals(
                point_normals=False, consistent_normals=True, auto_orient_normals=True)
            oriented_open = mesh
        self.open, self.closed = scene(oriented_open), scene(self.closed_mesh)
        self.distance_function = vtk.vtkImplicitPolyDataDistance()
        self.distance_function.SetInput(oriented_open)
        try:
            self.fcl = importlib.import_module("fcl")
        except ImportError as error:
            raise RuntimeError("Step 06 requires python-fcl in the active environment.") from error
        self.object = self.collision_object(self.points, self.faces)
        self.closed_object = self.collision_object(self.closed_mesh.points, triangles(self.closed_mesh))

    def assert_unchanged(self, mesh):
        if not np.array_equal(self.points, mesh.points) or not np.array_equal(self.faces, triangles(mesh)):
            raise RuntimeError("Step 06 modified the fixed RVendo geometry.")

    def collision_object(self, points, faces):
        model = self.fcl.BVHModel()
        model.beginModel()
        model.addSubModel(np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int32))
        model.endModel()
        return self.fcl.CollisionObject(model)

    def values(self, points, closest=False):
        x = query(self.open, points)
        nearest = self.open.compute_closest_points(x)
        q = np.asarray(nearest["points"].numpy(), dtype=float)
        d = np.linalg.norm(np.asarray(points).reshape(-1, 3) - q, axis=1)
        inside = self.closed.compute_occupancy(x, nsamples=5).numpy() > .5
        signed = np.where(inside, -d, d)
        return (signed, q) if closest else signed

    def gradient(self, points, signed, closest):
        gradient = np.asarray(points) - closest
        inside = signed < 0
        if np.any(inside):
            gradient[inside] *= -1
        length = np.linalg.norm(gradient, axis=1)
        good = length > 1e-8
        gradient[good] /= length[good, None]
        if np.any(~good):
            nearest = self.closed.compute_closest_points(query(self.closed, points[~good]))
            gradient[~good] = nearest["primitive_normals"].numpy()
        return gradient

    def distances(self, points):
        return np.asarray(self.open.compute_distance(query(self.open, points)).numpy(), dtype=float)

    def exact_distances(self, points):
        return np.asarray([abs(self.distance_function.FunctionValue(p)) for p in points], dtype=float)


def _band_levels(faces, ring_ids, count, layers):
    edges = _edges(faces)
    levels = np.full(count, -1, dtype=np.int32)
    levels[ring_ids] = 0
    for level in range(1, layers + 1):
        hits = edges[(levels[edges] == level - 1).any(axis=1)].ravel()
        levels[hits[levels[hits] < 0]] = level
    return levels


def _scalar_smoother(points, faces, vector=False):
    """An M-matrix smooths positive amounts without changing directions."""
    edges = _edges(faces)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    typical = float(np.median(lengths))
    weights = np.minimum((typical / np.maximum(lengths, typical * 1e-5)) ** 2, 1e4)
    a, b = edges[:, 0], edges[:, 1]
    laplace = coo_matrix((np.r_[weights, weights, -weights, -weights],
                          (np.r_[a, b, a, b], np.r_[a, b, b, a])), shape=(len(points), len(points))).tocsc()
    solve = factorized(eye(len(points), format="csc") + 2 * laplace)
    return (solve if vector else lambda amounts: np.maximum(solve(amounts), 0.0)), typical


def _ring_directions(ring, axis):
    _ring_polygon(ring, _plane_basis(axis))
    centre = ring.mean(axis=0)
    area = np.sum(np.cross(ring - centre, np.roll(ring, -1, axis=0) - centre) @ axis)
    tangent = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
    directions = np.cross(tangent, axis) * np.sign(area)
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths < 1e-10):
        raise RuntimeError("Basal ring has an undefined outward direction.")
    return directions / lengths[:, None]


def _ring_deficits(ring, endo_polygon, basis, target):
    xy = ring @ basis
    probes = np.vstack((xy, (xy + np.roll(xy, -1, axis=0)) * .5))
    samples = shapely.points(probes)
    d = shapely.distance(samples, endo_polygon.boundary)
    signed = np.where(shapely.contains(endo_polygon, samples), -d, d)
    deficits = np.maximum(target - signed, 0.0)
    segment_d = shapely.distance(shapely.linestrings(np.stack((xy, np.roll(xy, -1, axis=0)), axis=1)), endo_polygon)
    edge_deficit = np.maximum(deficits[len(ring):], np.maximum(target - segment_d, 0.0))
    vertex_deficit = np.maximum(deficits[:len(ring)], np.maximum(edge_deficit, np.roll(edge_deficit, 1)))
    if not _ring_polygon(ring, basis).covers(endo_polygon) and not np.any(vertex_deficit > 0):
        vertex_deficit[:] = target
    return vertex_deficit


def face_clearance(points, faces, fixed, minimum):
    """Global double-precision triangle distance and failing-face witnesses."""
    moving = fixed.collision_object(points, faces)
    result = fixed.fcl.DistanceResult()
    distance = float(fixed.fcl.distance(moving, fixed.object,
                                      fixed.fcl.DistanceRequest(enable_nearest_points=True), result))
    minima = np.full(len(faces), float("inf"))
    if not np.isfinite(distance):
        raise RuntimeError("FCL returned a non-finite surface distance.")
    if distance >= minimum:
        return np.empty(0, dtype=np.int64), distance, minima
    xyz = points[faces]
    probes = np.stack((xyz.mean(axis=1), (xyz[:, 0] + xyz[:, 1]) * .5,
                       (xyz[:, 1] + xyz[:, 2]) * .5, (xyz[:, 2] + xyz[:, 0]) * .5), axis=1)
    minima = fixed.values(probes.reshape(-1, 3)).reshape(-1, 4).min(axis=1)
    failing = minima < minimum
    contact = fixed.fcl.CollisionResult()
    fixed.fcl.collide(moving, fixed.object, fixed.fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True), contact)
    for item in contact.contacts:
        if 0 <= item.b1 < len(faces):
            failing[item.b1], minima[item.b1] = True, min(minima[item.b1], 0.0)
    if not contact.is_collision:
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(_surface(points, faces))
        locator.BuildLocator()
        cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        locator.FindClosestPoint(result.nearest_points[0], [0., 0., 0.], cell_id, sub_id, dist2)
        index = int(cell_id)
        failing[index], minima[index] = True, min(minima[index], distance)
    if not np.any(failing):
        raise RuntimeError("Clearance failed without a usable triangle witness.")
    return np.flatnonzero(failing), distance, minima


def validate_rvepi(epi, endo, minimum=3.0, axis=None, plane=None, fixed=None, closed_reference=None):
    """Prove full-surface side/containment via intersection exclusion and classification.

The side of a connected surface cannot change without crossing the closed
boundary. Vertex classification is used only after excluding all crossings.
"""
    _check_geometry(epi, "Moved RVepi")
    _check_geometry(endo, "Fixed RVendo")
    if axis is None:
        axis, plane = _basal_reference(epi, _opening_indices(epi))
        plane = max(plane, float(np.max(endo.points @ axis) + endo.length))
    if fixed is None:
        fixed = FixedEndo(endo, axis, plane, closed_reference)
    fixed.assert_unchanged(endo)
    bad, distance, _ = face_clearance(np.asarray(epi.points, dtype=float), triangles(epi), fixed, minimum)
    if len(bad):
        raise RuntimeError(f"RVepi surface clearance is {distance:.9g} mm, below {minimum:g}; faces {bad[:12].tolist()}.")
    if fixed.reference_supplied:
        contact = fixed.fcl.CollisionResult()
        fixed.fcl.collide(fixed.collision_object(epi.points, triangles(epi)), fixed.closed_object,
                          fixed.fcl.CollisionRequest(), contact)
        if contact.is_collision or np.any(fixed.values(epi.points) <= 0):
            raise RuntimeError("RVepi crosses or lies inside the closed Step 02 RVendo reference.")
        return {
            "minimum_certified_surface_distance_mm": distance,
            "surface_distance_method": "FCL_double_precision_global_triangle_triangle_distance",
            "whole_surface_containment_method": "closed_Step02_RVendo_intersection_exclusion_and_vertex_classification",
            "RVepi_entirely_outside_RVendo": True,
            "final_RVepi_self_intersection_pairs": 0, "final_vertices_inside_RVendo": 0,
            "rv_endo_fixed": True,
        }
    closed_epi = close_for_inside_test(epi, axis, plane)
    epi_object = fixed.collision_object(epi.points, triangles(epi))
    closed_epi_object = fixed.collision_object(closed_epi.points, triangles(closed_epi))
    for a, b, name in ((epi_object, fixed.closed_object, "RVepi vs closed RVendo"),
                       (fixed.object, closed_epi_object, "RVendo vs closed RVepi")):
        contact = fixed.fcl.CollisionResult()
        fixed.fcl.collide(a, b, fixed.fcl.CollisionRequest(), contact)
        if contact.is_collision:
            raise RuntimeError(f"Whole-surface containment failed: {name} intersects, including diagnostic curtains.")
    if np.any(fixed.values(epi.points) <= 0):
        raise RuntimeError("RVepi is not entirely exterior to RVendo.")
    epi_scene = scene(closed_epi)
    inside = epi_scene.compute_occupancy(query(epi_scene, endo.points), nsamples=5).numpy() > .5
    if not np.all(inside):
        raise RuntimeError(f"RVepi does not enclose RVendo ({np.count_nonzero(~inside)} exterior vertices).")
    return {
        "minimum_certified_surface_distance_mm": distance,
        "surface_distance_method": "FCL_double_precision_global_triangle_triangle_distance",
        "whole_surface_containment_method": "triangle_intersection_exclusion_and_connected_surface_classification",
        "RVepi_entirely_outside_RVendo": True, "RVendo_entirely_inside_RVepi": True,
        "final_RVepi_self_intersection_pairs": 0, "final_vertices_inside_RVendo": 0,
        "RVendo_containment_samples": int(len(endo.points)), "rv_endo_fixed": True,
    }


def move_rvepi(epi, endo, ring_ids, minimum, padding, max_step,
               ring_smooth_iterations, ring_relaxation, max_iterations=600, band_layers=3,
               basal_axis=None):
    """Expand wall/ring outward, without unrestricted repair or vector smoothing."""
    if not np.isfinite([minimum, padding, max_step, ring_relaxation]).all() or minimum <= 0 or padding < 0 or max_step <= 0:
        raise ValueError("Clearance/max_step must be positive and padding nonnegative.")
    if max_iterations < 1 or ring_smooth_iterations < 0 or not 0 <= ring_relaxation <= 1:
        raise ValueError("Invalid Step 06 iteration or ring smoothing settings.")
    _check_geometry(epi, "Step 05 RVepi")
    _check_geometry(endo, "Step 05 RVendo")
    original = np.asarray(epi.points, dtype=float).copy()
    original_faces = triangles(epi).copy()
    ring_ids = np.asarray(ring_ids, dtype=np.int64)
    boundary = _opening_indices(epi)
    if len(ring_ids) != len(boundary) or set(ring_ids.tolist()) != set(boundary.tolist()):
        raise RuntimeError("Step 06 ring IDs must cover exactly the RVepi boundary.")
    ring_edges = {tuple(e) for e in _edges(original_faces)}
    if any(tuple(sorted((int(a), int(b)))) not in ring_edges for a, b in zip(ring_ids, np.roll(ring_ids, -1))):
        raise RuntimeError("Step 06 ring IDs are not in boundary order.")
    axis, _ = _basal_reference(epi, ring_ids)
    if basal_axis is not None:
        axis = np.asarray(basal_axis, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-12:
            raise ValueError("basal_axis must be a finite nonzero 3-vector.")
        axis = axis / np.linalg.norm(axis)
    plane = max(float(np.max(endo.points @ axis)), float(np.max(original @ axis))) + max(epi.length, endo.length) * 2
    oriented = close_for_inside_test(epi, axis, plane)
    faces = triangles(oriented)[:epi.n_cells]
    reference = _vertex_normals(original, faces)
    fixed = FixedEndo(endo, axis, plane)
    points = original.copy()
    smoother, typical = _scalar_smoother(original, faces)
    trust_step = min(max_step, max(typical * .5, 1e-3))
    target = minimum + max(padding, 1e-3)
    basis = _plane_basis(axis)
    endo_polygon = _ring_polygon(endo.points[_opening_indices(endo)], basis)
    ring_mask = np.zeros(len(points), dtype=bool)
    ring_mask[ring_ids] = True
    levels = _band_levels(faces, ring_ids, len(points), band_layers)
    initial_signed = fixed.values(points)
    minimum_step_dot = np.zeros(len(points), dtype=float)
    minimum_ring_dot = 0.0
    accepted_steps = 0
    failing = np.empty(0, dtype=np.int64)
    minima = np.full(len(faces), float("inf"))
    for iteration in range(max_iterations):
        signed, closest = fixed.values(points, closest=True)
        ring_deficit = _ring_deficits(points[ring_ids], endo_polygon, basis, target)
        if np.all(signed >= target) and np.max(ring_deficit) <= 1e-5:
            failing, _, minima = face_clearance(points, faces, fixed, target)
            if not len(failing):
                break
        normals = _vertex_normals(points, faces)
        directions = normals.copy()
        directions[ring_ids] = _ring_directions(points[ring_ids], axis)
        outward_dot = np.einsum("ij,ij->i", directions, normals)
        reference_dot = np.einsum("ij,ij->i", directions, reference)
        if np.any(outward_dot < -1e-8) or np.any(reference_dot < -1e-8):
            bad = np.flatnonzero((outward_dot < -1e-8) | (reference_dot < -1e-8))
            raise RuntimeError(f"Outward ring/wall motion is infeasible at vertices {bad[:12].tolist()}; no inward fallback applied.")
        gradient = fixed.gradient(points, signed, closest)
        gain = np.einsum("ij,ij->i", directions, gradient)
        deficit = np.maximum(target - signed, 0.0)
        # Nonpositive derivatives trigger outward exploration, never gradient fallback.
        raw = np.minimum(deficit / np.maximum(gain, .25), trust_step)
        raw[ring_ids] = np.maximum(raw[ring_ids], np.minimum(ring_deficit, trust_step))
        xyz = points[faces]
        probes = np.stack((xyz.mean(axis=1), (xyz[:, 0] + xyz[:, 1]) * .5,
                           (xyz[:, 1] + xyz[:, 2]) * .5, (xyz[:, 2] + xyz[:, 0]) * .5), axis=1)
        face_values = fixed.values(probes.reshape(-1, 3)).reshape(-1, 4).min(axis=1)
        face_steps = np.minimum(np.maximum(target - face_values, 0.0), trust_step)
        if len(failing):
            face_steps[failing] = np.maximum(face_steps[failing], np.minimum(np.maximum(target - minima[failing], .001), trust_step))
        np.maximum.at(raw, faces.ravel(), np.repeat(face_steps, 3))
        amounts = smoother(raw)
        for _ in range(ring_smooth_iterations):
            s = amounts[ring_ids]
            amounts[ring_ids] = (1 - ring_relaxation) * s + ring_relaxation * .5 * (np.roll(s, 1) + np.roll(s, -1))
        delta = directions * np.minimum(amounts, trust_step)[:, None]
        current_cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        accepted = False
        rejected = []
        for half in range(16):
            step_delta = delta * .5 ** half
            candidate = points + step_delta
            if np.max(np.linalg.norm(step_delta, axis=1)) < 1e-8:
                rejected.append("stalled")
                break
            new_xyz = candidate[faces]
            new_cross = np.cross(new_xyz[:, 1] - new_xyz[:, 0], new_xyz[:, 2] - new_xyz[:, 0])
            if np.any(np.linalg.norm(new_cross, axis=1) <= 1e-12) or np.any(np.einsum("ij,ij->i", new_cross, current_cross) <= 0):
                rejected.append("folded_face")
                continue
            if len(_self_intersections(candidate, faces)):
                rejected.append("self_intersection")
                continue
            try:
                _ring_polygon(candidate[ring_ids], basis)
            except RuntimeError:
                rejected.append("invalid_ring")
                continue
            if np.any(candidate @ axis >= plane):
                rejected.append("diagnostic_plane_exceeded")
                continue
            step_dot = np.einsum("ij,ij->i", step_delta, normals)
            net_dot = np.einsum("ij,ij->i", candidate - original, reference)
            if np.min(step_dot) < -1e-9 or np.min(net_dot) < -1e-9:
                rejected.append("inward_displacement")
                continue
            points = candidate
            minimum_step_dot = np.minimum(minimum_step_dot, step_dot)
            minimum_ring_dot = min(minimum_ring_dot, float(np.min(np.einsum("ij,ij->i", step_delta[ring_ids], directions[ring_ids]))))
            accepted_steps += 1
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f"Step 06 outward deformation stalled at iteration {iteration}: "
                               f"min_signed={signed.min():.6g}, below_target={np.count_nonzero(signed < target)}, "
                               f"failing_faces={failing[:12].tolist()}, rejected={rejected}; no inward repair applied.")
        failing = np.empty(0, dtype=np.int64)
        if iteration % 20 == 0:
            print(f"[Step06 RVepi] iteration={iteration} below_target={np.count_nonzero(signed < target)} "
                  f"inside={np.count_nonzero(signed < 0)} max_step={np.linalg.norm(step_delta, axis=1).max():.5g}", flush=True)
    else:
        remaining = fixed.values(points)
        raise RuntimeError(f"Step 06 did not converge after {max_iterations} outward iterations: "
                           f"min_signed={remaining.min():.6g}, below_target={np.count_nonzero(remaining < target)}.")
    result = epi.copy(deep=True)
    result.points = points
    displacement = points - original
    outward_displacement = np.einsum("ij,ij->i", displacement, reference)
    result.point_data["step6SourcePointId"] = np.arange(len(points), dtype=np.int64)
    result.point_data["step6ReferenceOutwardNormal"] = reference
    result.point_data["step6Displacement"] = displacement
    result.point_data["step6OutwardDisplacement"] = outward_displacement
    result.point_data["step6MinimumStepOutwardDot"] = minimum_step_dot
    final_signed = np.where(fixed.values(points) < 0, -1., 1.) * fixed.exact_distances(points)
    result.point_data["signedDistanceToRVendo"] = final_signed
    result.point_data["adaptiveOpeningBoundary"] = ring_mask.astype(np.int32)
    result.point_data["openingTransitionBand"] = (levels >= 0).astype(np.int32)
    verification = validate_rvepi(result, endo, minimum, axis, plane, fixed)
    fixed.assert_unchanged(endo)
    if not np.array_equal(triangles(result), original_faces):
        raise RuntimeError("Step 06 modified RVepi connectivity.")
    return result, {
        "method": "RVepi_nonnegative_normal_displacement_with_outward_basal_ring",
        "deformation": "scalar_M_matrix_smoothing_and_geometry_backtracking",
        "minimum_requested_distance_mm": float(minimum), "target_distance_with_padding_mm": float(target),
        "minimum_vertex_signed_distance_mm": float(final_signed.min()),
        "initial_vertices_inside_RVendo": int(np.count_nonzero(initial_signed < 0)),
        "initial_vertices_below_3mm": int(np.count_nonzero(initial_signed < minimum)),
        "initial_RVepi_self_intersection_pairs": 0, "final_vertices_below_3mm": int(np.count_nonzero(final_signed < minimum)),
        "outward_step_violations": int(np.count_nonzero(minimum_step_dot < -1e-9)),
        "minimum_step_outward_dot_mm": float(minimum_step_dot.min()),
        "minimum_ring_outward_dot_mm": minimum_ring_dot,
        "minimum_total_outward_displacement_mm": float(outward_displacement.min()),
        "total_inward_displacement_vertices": int(np.count_nonzero(outward_displacement < -1e-9)),
        "opening_vertex_count_preserved": True, "opening_edge_connections_preserved": True,
        "ring_vertices": int(len(ring_ids)), "iterations": accepted_steps,
        "max_displacement_mm": float(np.linalg.norm(displacement, axis=1).max()),
        "_opening_ring_ids": ring_ids.tolist(), "basal_reference_axis": axis.tolist(), "basal_reference_plane": float(plane),
        **verification,
    }


def move_rvepi_signed(epi, endo, ring_ids, minimum, padding, max_step,
                       ring_smooth_iterations, ring_relaxation, max_iterations=600,
                       band_layers=3, closed_reference=None):
    """Correct signed clearance using a connected, smoothed vector displacement.

    The closed Step 02 endocardium supplies the side test; distances and
    witnesses use the actual fixed open endocardial triangles. No artificial
    closure is added to either exported surface.
    """
    if closed_reference is None:
        raise ValueError("Signed Step 06 requires the cleaned closed Step 02 RVendo reference.")
    if not np.isfinite([minimum, padding, max_step]).all() or minimum <= 0 or max_step <= 0 or padding < 0 or max_iterations < 1:
        raise ValueError("Invalid signed-clearance parameters.")
    _check_geometry(epi, "Step 05 RVepi")
    _check_geometry(endo, "Step 05 RVendo")
    original = np.asarray(epi.points, dtype=float).copy()
    faces = triangles(epi).copy()
    ring_ids = np.asarray(ring_ids, dtype=np.int64)
    boundary = _opening_indices(epi)
    if len(ring_ids) != len(boundary) or set(ring_ids.tolist()) != set(boundary.tolist()):
        raise ValueError("Opening IDs must match the RVepi boundary.")
    fixed = FixedEndo(endo, closed_reference=closed_reference)
    smoother, typical = _scalar_smoother(original, faces, vector=True)
    trust = min(max_step, max(typical * .5, .001))
    target = minimum + max(padding, .001)
    points = original.copy()
    initial = fixed.values(points)
    failing = np.empty(0, dtype=np.int64)
    minima = np.full(len(faces), np.inf)
    for iteration in range(max_iterations):
        signed, closest = fixed.values(points, closest=True)
        if np.all(signed >= minimum):
            failing, distance, minima = face_clearance(points, faces, fixed, minimum)
            if not len(failing):
                # Also exclude crossings of the retained closed basal anatomy.
                validate_rvepi(_surface(points, faces), endo, minimum, fixed=fixed)
                break
        gradient = fixed.gradient(points, signed, closest)
        raw = gradient * np.minimum(np.maximum(target - signed, 0), trust)[:, None]
        xyz = points[faces]
        probes = np.stack((xyz.mean(axis=1), (xyz[:, 0] + xyz[:, 1]) * .5,
                           (xyz[:, 1] + xyz[:, 2]) * .5, (xyz[:, 2] + xyz[:, 0]) * .5), axis=1)
        values, nearest = fixed.values(probes.reshape(-1, 3), closest=True)
        face_vectors = fixed.gradient(probes.reshape(-1, 3), values, nearest)
        face_vectors *= np.minimum(np.maximum(target - values, 0), trust)[:, None]
        face_vectors = face_vectors.reshape(-1, 4, 3)
        worst = np.argmin(values.reshape(-1, 4), axis=1)
        contributions = face_vectors[np.arange(len(faces)), worst]
        if len(failing):
            for index in failing:
                if np.linalg.norm(contributions[index]) < 1e-8:
                    direction = gradient[faces[index]].mean(axis=0)
                    direction /= max(np.linalg.norm(direction), 1e-12)
                    contributions[index] = direction * min(target - minima[index], trust)
        sums = np.zeros_like(points)
        counts = np.zeros(len(points))
        for k in range(3):
            np.add.at(sums, faces[:, k], contributions)
            np.add.at(counts, faces[:, k], np.linalg.norm(contributions, axis=1) > 0)
        averaged = sums / np.maximum(counts, 1)[:, None]
        use = np.linalg.norm(averaged, axis=1) > np.linalg.norm(raw, axis=1)
        raw[use] = averaged[use]
        delta = smoother(raw)
        # The opening participates in the same connected field; no frozen rim.
        length = np.linalg.norm(delta, axis=1)
        delta *= np.minimum(1, trust / np.maximum(length, 1e-12))[:, None]
        cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        accepted = False
        for half in range(17):
            step = delta * .5 ** half
            candidate = points + step
            if np.max(np.linalg.norm(step, axis=1)) < 1e-8:
                break
            new = candidate[faces]
            new_cross = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
            if np.any(np.einsum("ij,ij->i", cross, new_cross) <= 0):
                continue
            if len(_self_intersections(candidate, faces)):
                continue
            points = candidate
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f"Signed Step 06 stalled at iteration {iteration}: min_signed={signed.min():.6g} mm; geometry guards rejected displacement.")
        if iteration % 20 == 0:
            print(f"[Step06 signed] iteration={iteration} inside={np.count_nonzero(signed < 0)} "
                  f"below_target={np.count_nonzero(signed < target)} min_signed={signed.min():.5g} "
                  f"max_step={np.linalg.norm(step, axis=1).max():.5g}", flush=True)
    else:
        raise RuntimeError(f"Signed Step 06 did not converge after {max_iterations} iterations; min_signed={fixed.values(points).min():.6g} mm.")
    result = epi.copy(deep=True)
    result.points = points
    final = np.where(fixed.values(points) < 0, -1., 1.) * fixed.exact_distances(points)
    levels = _band_levels(faces, ring_ids, len(points), band_layers)
    result.point_data["step6SourcePointId"] = np.arange(len(points))
    result.point_data["step6Displacement"] = points - original
    result.point_data["step6DisplacementMagnitude"] = np.linalg.norm(points - original, axis=1)
    result.point_data["initialSignedDistanceToRVendo"] = initial
    result.point_data["initialInsideRVendo"] = (initial < 0).astype(np.int32)
    result.point_data["signedDistanceToRVendo"] = final
    result.point_data["insideRVendo"] = (final < 0).astype(np.int32)
    result.point_data["adaptiveOpeningBoundary"] = (levels == 0).astype(np.int32)
    result.point_data["openingTransitionBand"] = (levels >= 0).astype(np.int32)
    verification = validate_rvepi(result, endo, minimum, fixed=fixed)
    fixed.assert_unchanged(endo)
    return result, {
        "method": "signed_triangle_clearance_with_connected_vector_smoothing",
        "side_reference": "cleaned_closed_Step02_RVendo",
        "minimum_requested_distance_mm": minimum,
        "target_distance_with_padding_mm": target,
        "initial_vertices_inside_RVendo": int(np.count_nonzero(initial < 0)),
        "initial_vertices_below_3mm": int(np.count_nonzero(initial < minimum)),
        "final_vertices_below_3mm": int(np.count_nonzero(final < minimum)),
        "minimum_vertex_signed_distance_mm": float(final.min()),
        "max_displacement_mm": float(np.linalg.norm(points - original, axis=1).max()),
        "opening_vertex_count_preserved": True, "opening_edge_connections_preserved": True,
        "iterations": iteration, "_opening_ring_ids": ring_ids.tolist(), **verification,
    }
