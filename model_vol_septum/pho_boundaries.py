"""Map source surfaces and opening rims to disjoint PHO volume boundaries."""
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


SURFACE_CLASSES = ("epi", "LVendo", "RVendo", "LVannulus", "RVannulus")
PHO_BOUNDARY_RULE_VERSION = "current_tag_surfaces_remaining_tag0_basal_zero_v2"


def load_surface(path):
    mesh = pv.read(path).extract_surface().triangulate().clean(tolerance=0.0)
    if (not mesh.n_cells or not mesh.is_all_triangles
            or not np.all(np.isfinite(mesh.points))):
        raise ValueError(f"PHO reference must be a finite triangle surface, not a curve/cloud: {path}")
    return mesh


def face_samples(surface):
    triangles = surface.points[surface.faces.reshape(-1, 4)[:, 1:]]
    samples = np.concatenate((
        triangles,
        0.5 * (triangles + np.roll(triangles, -1, axis=1)),
        triangles.mean(axis=1)[:, None, :],
    ), axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        raise ValueError("PHO boundary contains degenerate triangles.")
    edges = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)
    return samples, normals / lengths[:, None], np.median(edges, axis=1)


def nearest_triangles(surface, points):
    ids, closest = surface.find_closest_cell(np.asarray(points).reshape(-1, 3), return_closest_point=True)
    distances = np.linalg.norm(np.asarray(points).reshape(-1, 3) - closest, axis=1)
    return np.asarray(ids, dtype=np.int64), distances


def save_subset(surface, mask, path):
    selected = surface.extract_cells(np.flatnonzero(mask)).extract_surface() if np.any(mask) else pv.PolyData()
    selected.save(path, binary=True)


def save_node_cloud(points, ids, value, path):
    cloud = pv.PolyData(np.asarray(points)[ids])
    cloud.point_data["volumeNodeId"] = np.asarray(ids, dtype=np.int64)
    cloud.point_data["PHO_BC"] = np.full(len(ids), value, dtype=np.int32)
    cloud.set_active_scalars("PHO_BC")
    cloud.save(path, binary=True)


def _face_components(faces):
    """Connect surface triangles only across complete shared edges."""
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError("The unmapped PHO boundary has nonmanifold edges.")
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
    owners = np.tile(np.arange(len(faces)), 3)
    a, b = owners[order[offsets]], owners[order[offsets + 1]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(faces), len(faces)))
    return connected_components(graph, directed=False)[1]


def _tag_interface_faces(volume):
    """Extract only complete shared faces between tag 0 and tag 1 tetrahedra."""
    from .tag_step10_volume import shared_face_neighbors

    if (not isinstance(volume, pv.UnstructuredGrid) or not volume.n_cells
            or not np.all(volume.celltypes == pv.CellType.TETRA)
            or "elemTag" not in volume.cell_data):
        raise RuntimeError("PHO LVepi extraction requires tetrahedra with elemTag cell labels.")
    tags = np.asarray(volume.cell_data["elemTag"])
    if not set(np.unique(tags)).issubset({0, 1}) or not np.any(tags == 1):
        raise RuntimeError("PHO LVepi extraction requires tag 1 and only tag 0/1 cell labels.")
    a, b, shared = shared_face_neighbors(volume.cells_dict[pv.CellType.TETRA])
    between = tags[a] != tags[b]
    a, b = a[between], b[between]
    return (shared[between], np.where(tags[a] == 1, a, b),
            np.where(tags[a] == 0, a, b))


def _epicardial_rim_ears(triangles, face_ids, rim_nodes, rim_edges, labels):
    """Recognize exact rim-edge ears without reclassifying epicardial faces."""
    nodes = set(np.unique(triangles[face_ids]).tolist())
    if set(rim_nodes) <= nodes:
        return []
    patch = triangles[face_ids]
    edges, counts = np.unique(np.sort(np.vstack((
        patch[:, [0, 1]], patch[:, [1, 2]], patch[:, [2, 0]])), axis=1),
        axis=0, return_counts=True)
    boundary_edges = set(map(tuple, edges[counts == 1]))
    candidates = np.flatnonzero((labels == 1) & np.all(np.isin(triangles, rim_nodes), axis=1))
    added = []
    for face_id in candidates:
        face = triangles[face_id]
        if set(face) <= nodes:
            continue
        face_edges = {tuple(sorted(edge)) for edge in
                      ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
        attached = face_edges & boundary_edges
        if len(attached) == 1 and face_edges - attached <= rim_edges:
            boundary_edges.symmetric_difference_update(face_edges)
            nodes.update(face.tolist())
            added.append(int(face_id))
    return added if set(rim_nodes) <= nodes else None


def _volume_face_surface(volume, faces, parts, tag1_cells, tag0_cells, lvendo_nodes):
    """Use original volume triangles, recording provenance and effective nodal BCs."""
    if not len(faces):
        return pv.PolyData()
    nodes, local = np.unique(faces, return_inverse=True)
    packed = np.column_stack((np.full(len(faces), 3), local.reshape(-1, 3)))
    surface = pv.PolyData(volume.points[nodes].copy(), faces=packed.ravel())
    bc = (~np.isin(nodes, lvendo_nodes)).astype(np.int32)
    surface.point_data["volumeNodeId"] = nodes
    surface.point_data["PHO_BC"] = bc
    surface.cell_data["boundaryPart"] = np.asarray(parts, dtype=np.int32)
    surface.cell_data["volumeFaceNodeIds"] = np.sort(faces, axis=1)
    surface.cell_data["tag1CellId"] = tag1_cells
    surface.cell_data["tag0CellId"] = tag0_cells
    surface.cell_data["PHO_requested_BC"] = np.ones(len(faces), dtype=np.int32)
    face_bc = bc[local.reshape(-1, 3)]
    surface.cell_data["PHO_BC"] = np.where(
        np.all(face_bc == 1, axis=1), 1, np.where(np.all(face_bc == 0, axis=1), 0, -1),
    ).astype(np.int32)
    surface.set_active_scalars("PHO_BC", preference="point")
    return surface


def _extract_tag_boundary_conditions(volume, boundary, labels, lvendo_nodes, output_dir):
    """Classify exact current tag surfaces; never project to the old LVepi."""
    faces = np.asarray(boundary.point_data["volumeNodeId"])[
        boundary.faces.reshape(-1, 4)[:, 1:]]
    owners = np.asarray(boundary.cell_data["volumeCellId"], dtype=np.int64)
    tags = np.asarray(volume.cell_data["elemTag"])
    junction, junction_tag1, junction_tag0 = _tag_interface_faces(volume)
    owner_tags = tags[owners]
    # RV basal faces transferred into tag 1 now belong to its condition-1 wall.
    exterior_one = (labels == 1) | ((owner_tags == 1) & ~np.isin(labels, [2, 4]))
    requested = exterior_one.astype(np.int32)
    one_faces = np.vstack((faces[exterior_one], junction))
    zero_faces = faces[~exterior_one]
    tree = cKDTree(np.asarray(volume.points))

    def expand(ids):
        if not len(ids):
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(
            tree.query_ball_point(volume.points[ids], r=1e-6))).astype(np.int64)

    raw_zero = expand(np.unique(zero_faces))
    raw_one = expand(np.unique(one_faces))
    protected = np.intersect1d(raw_one, lvendo_nodes)
    one_nodes = np.setdiff1d(raw_one, lvendo_nodes)
    zero_nodes = np.setdiff1d(np.union1d(raw_zero, lvendo_nodes), one_nodes)
    effective = np.full(volume.n_points, -1, dtype=np.int32)
    effective[zero_nodes], effective[one_nodes] = 0, 1

    def save_faces(filename, triangles, parts, tag1, tag0, prescribed):
        surface = _volume_face_surface(
            volume, triangles, parts, tag1, tag0, lvendo_nodes)
        if surface.n_cells:
            ids = surface.point_data["volumeNodeId"]
            surface.point_data["PHO_BC"] = effective[ids]
            surface.cell_data["PHO_requested_BC"] = np.asarray(prescribed, dtype=np.int32)
            bc = effective[triangles]
            surface.cell_data["PHO_BC"] = np.where(
                np.all(bc == 1, axis=1), 1,
                np.where(np.all(bc == 0, axis=1), 0, -1)).astype(np.int32)
        surface.save(output_dir / filename, binary=True)
        return surface

    minus = np.full(len(faces), -1, dtype=np.int64)
    mapped_mask = (owner_tags == 1) & exterior_one
    mapped = faces[mapped_mask]
    save_faces(
        "LVepi_mapped.vtk", mapped, np.ones(len(mapped)), owners[mapped_mask],
        minus[mapped_mask], np.ones(len(mapped)))
    save_faces("LVepi_interface.vtk", junction, np.full(len(junction), 2),
               junction_tag1, junction_tag0, np.ones(len(junction)))
    combined_faces = np.vstack((mapped, junction))
    save_faces(
        "LVepi_openbase.vtk", combined_faces,
        np.r_[np.ones(len(mapped)), np.full(len(junction), 2)],
        np.r_[owners[mapped_mask], junction_tag1],
        np.r_[minus[mapped_mask], junction_tag0], np.ones(len(combined_faces)))
    for tag, chamber in ((1, "LV"), (0, "RV")):
        mask = owner_tags == tag
        exterior_owners = owners[mask]
        tag1 = exterior_owners if tag == 1 else minus[mask]
        tag0 = exterior_owners if tag == 0 else minus[mask]
        save_faces(
            chamber + "_tag" + str(tag) + "_whole_surface.vtk",
            np.vstack((faces[mask], junction)),
            np.r_[np.ones(mask.sum()), np.full(len(junction), 2)],
            np.r_[tag1, junction_tag1], np.r_[tag0, junction_tag0],
            np.r_[requested[mask], np.ones(len(junction))])

    mapped_nodes, junction_nodes = np.unique(mapped), np.unique(junction)
    lv_nodes = np.setdiff1d(np.union1d(mapped_nodes, junction_nodes), lvendo_nodes)
    for filename, ids in (("LVepi_mapped_nodes.vtk", mapped_nodes),
                          ("LVepi_interface_nodes.vtk", junction_nodes)):
        cloud = pv.PolyData(volume.points[ids].copy())
        cloud.point_data["volumeNodeId"] = ids
        cloud.point_data["PHO_BC"] = effective[ids]
        cloud.save(output_dir / filename, binary=True)
    internal = np.setdiff1d(lv_nodes, boundary.point_data["volumeNodeId"])
    save_node_cloud(volume.points, internal, 1, output_dir / "LVepi_internal_one_nodes.vtk")
    info = {
        "method": "current_tag_surfaces_without_LVepi_source_mapping",
        "uses_Step03_LVepi": False, "uses_tag1_surface_selection": True,
        "source_surface_cut_performed": False,
        "mapped_face_count": int(len(mapped)), "interface_face_count": int(len(junction)),
        "internal_node_count": int(len(internal)),
        "mapped_node_count": int(len(mapped_nodes)),
        "interface_node_count": int(len(junction_nodes)),
        "node_count": int(len(lv_nodes)),
        "tag1_whole_surface_face_count": int(np.count_nonzero(owner_tags == 1) + len(junction)),
        "tag0_whole_surface_face_count": int(np.count_nonzero(owner_tags == 0) + len(junction)),
        "tag1_RV_basal_faces_condition_one": int(np.count_nonzero((owner_tags == 1) & (labels == 5))),
        "tag0_RV_basal_faces_condition_zero": int(np.count_nonzero((owner_tags == 0) & (labels == 5))),
        "tag1_LV_basal_faces_condition_zero": int(np.count_nonzero((owner_tags == 1) & (labels == 4))),
        "tag1_RV_endo_faces_condition_one": int(np.count_nonzero((owner_tags == 1) & (labels == 3))),
        "geometry_and_volume_node_ids_preserved": True,
        "internal_interface_requested_condition": 1,
        "LVendo_priority_removed_one_node_count": int(len(protected)),
    }
    (output_dir / "tag_surface_boundary_summary.json").write_text(json.dumps(info, indent=2))
    return zero_nodes, one_nodes, lv_nodes, requested, info


def build_pho_boundaries(volume, full_points, surface_paths, output_dir, edge_scale):
    """Use current tag surfaces, with the shared interface=1 and LVendo=0."""
    from .compute_UVC_fiber import _map_contour_to_mesh_nodes, _opening_rim_loop

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_points = np.asarray(full_points, dtype=float)
    if full_points.shape != volume.points.shape or not np.allclose(full_points, volume.points, rtol=0., atol=1e-4):
        raise RuntimeError("CARP and VTK full-mesh point order or coordinates do not match.")
    references = {name: load_surface(surface_paths[name]) for name in SURFACE_CLASSES[:3]}
    boundary = volume.extract_surface(pass_pointid=True, pass_cellid=True).triangulate()
    original_ids = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.point_data["volumeNodeId"] = original_ids
    # VTK may replace vtkOriginalCellIds during subsequent subset extraction.
    boundary.cell_data["volumeCellId"] = np.asarray(boundary.cell_data["vtkOriginalCellIds"]).copy()
    triangles = original_ids[boundary.faces.reshape(-1, 4)[:, 1:]]
    samples, normals, local_scale = face_samples(boundary)
    scores, agreement = [], []
    for reference in references.values():
        ids, distances = nearest_triangles(reference, samples)
        _, ref_normals, _ = face_samples(reference)
        scores.append(np.sqrt(np.mean(distances.reshape(-1, 7) ** 2, axis=1)))
        center_ids = ids.reshape(-1, 7)[:, -1]
        agreement.append(np.abs(np.einsum("ij,ij->i", normals, ref_normals[center_ids])))
    scores, agreement = np.column_stack(scores), np.column_stack(agreement)
    winners = np.argmin(scores, axis=1)
    rows = np.arange(boundary.n_cells)
    best = scores[rows, winners]
    tolerance = np.maximum(1e-4, np.minimum(.2 * local_scale, .75 * edge_scale))
    compatible = (agreement[rows, winners] >= .5) | (best <= 1e-5)
    # A triangle must be closer to the winning zero/one class, not just to a vertex.
    opposite = np.where(winners == 0, scores[:, 1:].min(axis=1), scores[:, 0])
    matched = (best <= tolerance) & compatible & (opposite - best > max(1e-8, 1e-8 * edge_scale))
    labels = np.where(matched, winners + 1, 0).astype(np.int32)
    details = {
        "method": "current_tag_surfaces_with_reference_classification_of_exterior_faces",
        "source_surfaces": {name: str(Path(path).resolve()) for name, path in surface_paths.items()},
        "class_ids": {str(i + 1): name for i, name in enumerate(SURFACE_CLASSES)},
        "matching_tolerance_max_mm": float(tolerance.max()),
        "matched_reference_distance_max_mm": float(best[matched].max()) if np.any(matched) else None,
        "surface_mapping": "seven_samples_per_target_triangle_with_triangle_distance_and_normal_checks",
        "internal_LVepi_constraint": True,
        "boundary_rule_version": PHO_BOUNDARY_RULE_VERSION,
        "basal_components": [],
    }
    report_path = output_dir / "boundary_classification_summary.json"

    def save_diagnostics():
        boundary.cell_data["surfaceClass"] = labels
        boundary.cell_data["PHO_BC"] = np.where(labels == 0, -1, (labels == 1).astype(np.int32))
        boundary.cell_data["referenceDistance_mm"] = best
        boundary.cell_data["referenceNormalAgreement"] = agreement[rows, winners]
        boundary.save(output_dir / "volume_boundary_classified.vtk", binary=True)
        save_subset(boundary, labels == 0, output_dir / "unresolved_volume_faces.vtk")
        details["unresolved_faces"] = int(np.count_nonzero(labels == 0))
        details["face_counts"] = {name: int(np.count_nonzero(labels == i + 1))
                                 for i, name in enumerate(SURFACE_CLASSES)}

    def fail(message):
        save_diagnostics()
        details["error"] = message
        report_path.write_text(json.dumps(details, indent=2))
        raise RuntimeError(f"{message} See {report_path}")

    for i, name in enumerate(SURFACE_CLASSES[:3], start=1):
        if not np.any(labels == i):
            fail(f"PHO surface mapping did not match any volume faces for {name}.")
    rim_ids, rim_edges = {}, {}
    for name in ("LV", "RV"):
        points, loop_info = _opening_rim_loop(str(surface_paths[name + "annulus"]))
        node_ids, mapping = _map_contour_to_mesh_nodes(
            points, np.asarray(volume.points), edge_scale, name + " PHO opening rim",
        )
        if not np.all(np.isin(node_ids, original_ids)):
            fail(f"{name} opening rim mapped to interior volume nodes.")
        rim_ids[name] = node_ids
        rim = pv.read(surface_paths[name + "annulus"]).clean(tolerance=0.0)
        _, mapped = cKDTree(np.asarray(volume.points)).query(rim.points)
        edges, offset = set(), 0
        while offset < len(rim.lines):
            count = int(rim.lines[offset])
            line = mapped[rim.lines[offset + 1:offset + count + 1]]
            edges.update(tuple(sorted(edge)) for edge in zip(line[:-1], line[1:]))
            offset += count + 1
        rim_edges[name] = edges
        details[name + "_opening_rim"] = {"loop": loop_info, "mapping": mapping}

    # New basal walls are absent from the original closed endocardia. Identify
    # each only if its edge-connected component contains one complete saved rim.
    unknown = np.flatnonzero(labels == 0)
    if len(unknown):
        components = _face_components(triangles[unknown])
        for component in np.unique(components):
            face_ids = unknown[components == component]
            support = {name: _epicardial_rim_ears(
                triangles, face_ids, ids, rim_edges[name], labels)
                for name, ids in rim_ids.items()}
            chambers = [name for name, ears in support.items() if ears is not None]
            if len(chambers) != 1:
                fail("Unmapped PHO patch is not bounded by exactly one complete LV/RV opening rim.")
            chamber = chambers[0]
            labels[face_ids] = 4 if chamber == "LV" else 5
            details["basal_components"].append({
                "chamber": chamber, "face_count": int(len(face_ids)),
                "complete_opening_rim_covered": True,
                "epicardial_rim_ear_face_ids": support[chamber],
                "epicardial_rim_ears_keep_condition_one": True,
            })
    save_diagnostics()
    tree = cKDTree(np.asarray(volume.points))
    def expand(ids):
        return np.unique(np.concatenate(tree.query_ball_point(volume.points[ids], r=1e-6))).astype(np.int64)
    lvendo_nodes = expand(np.unique(triangles[labels == 2]))
    outer_nodes = np.unique(triangles[labels == 1])
    zero_nodes, one_nodes, lv_nodes, requested, tag_details = _extract_tag_boundary_conditions(
        volume, boundary, labels, lvendo_nodes, output_dir,
    )
    details["tag_surface_conditions"] = tag_details
    old_zero = np.unique(triangles[labels > 1])
    lv_overrides = np.intersect1d(old_zero, lv_nodes)
    shared_rim = np.intersect1d(old_zero, one_nodes)
    if not np.all(np.isin(lvendo_nodes, zero_nodes)):
        fail("LVendo=0 priority was not preserved for every LV endocardial node.")
    if not len(zero_nodes) or not len(one_nodes):
        fail("Both PHO=0 and PHO=1 mapped boundary sets must be nonempty.")
    save_node_cloud(volume.points, zero_nodes, 0, output_dir / "PHO_zero_nodes.vtk")
    save_node_cloud(volume.points, one_nodes, 1, output_dir / "PHO_one_nodes.vtk")
    save_node_cloud(volume.points, lvendo_nodes, 0, output_dir / "LVendo_zero_nodes.vtk")
    save_node_cloud(volume.points, lv_nodes, 1, output_dir / "LVepi_one_nodes.vtk")
    save_node_cloud(volume.points, lv_overrides, 1, output_dir / "LVepi_zero_boundary_overrides.vtk")
    save_node_cloud(volume.points, shared_rim, 1, output_dir / "shared_rim_one_nodes.vtk")
    effective_bc = np.full(volume.n_points, -1, dtype=np.int32)
    effective_bc[zero_nodes] = 0
    effective_bc[one_nodes] = 1
    boundary.point_data["PHO_BC"] = effective_bc[original_ids]
    face_bc = effective_bc[triangles]
    boundary.cell_data["PHO_requested_BC"] = requested
    boundary.cell_data["PHO_BC"] = np.where(
        np.all(face_bc == 1, axis=1), 1, np.where(np.all(face_bc == 0, axis=1), 0, -1),
    ).astype(np.int32)
    boundary.cell_data["ownerTag"] = np.asarray(volume.cell_data["elemTag"])[
        boundary.cell_data["volumeCellId"]]
    boundary.save(output_dir / "volume_boundary_classified.vtk", binary=True)
    for i, name in enumerate(SURFACE_CLASSES, start=1):
        save_subset(boundary, labels == i, output_dir / (name + "_mapped.vtk"))
    save_subset(boundary, np.isin(labels, [4, 5]), output_dir / "basal_opening_faces.vtk")
    owner_tags = np.asarray(boundary.cell_data["ownerTag"])
    save_subset(boundary, (owner_tags == 0) & np.isin(labels, [4, 5]),
                output_dir / "remaining_tag0_basal_walls.vtk")
    save_subset(boundary, (owner_tags == 1) & (labels == 5),
                output_dir / "RV_basal_transferred_to_tag1.vtk")
    details.update({
        "zero_node_count": int(len(zero_nodes)), "one_node_count": int(len(one_nodes)),
        "combined_outer_node_count": int(len(outer_nodes)),
        "LVepi_node_count": int(len(lv_nodes)),
        "LVepi_additional_one_node_count": int(len(np.setdiff1d(lv_nodes, outer_nodes))),
        "LVepi_zero_boundary_override_count": int(len(lv_overrides)),
        "LVendo_zero_node_count": int(len(lvendo_nodes)),
        "LVendo_priority_removed_one_node_count": tag_details["LVendo_priority_removed_one_node_count"],
        "shared_rim_node_count": int(len(shared_rim)),
        "constraint_overlap_count": 0,
        "shared_rim_rule": "LVendo_zero_first_then_current_tag1_wall_interface_and_combined_epi_one",
        "diagnostic_cell_PHO_BC_minus_one": "mixed_zero_and_one_vertex_values",
        "volume_geometry_modified": False,
    })
    report_path.write_text(json.dumps(details, indent=2))
    return zero_nodes, one_nodes, details
