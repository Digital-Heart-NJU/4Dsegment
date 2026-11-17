"""Map source surfaces and opening rims to disjoint PHO volume boundaries."""
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


SURFACE_CLASSES = ("epi", "LVendo", "RVendo", "LVannulus", "RVannulus")
PHO_BOUNDARY_RULE_VERSION = "mapped_tag1_LVepi_without_junction_plus_exact_interface_v1"


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


def _map_lvepi_and_extract_interface(volume, boundary, labels, source_path,
                                      lvendo_nodes, output_dir):
    """Map the exposed tag-1 LVepi and add its disjoint, directly extracted interface."""
    source = load_surface(source_path)
    junction, junction_tag1, junction_tag0 = _tag_interface_faces(volume)
    original_ids = np.asarray(boundary.point_data["volumeNodeId"], dtype=np.int64)
    owners = np.asarray(boundary.cell_data["volumeCellId"], dtype=np.int64)
    tags = np.asarray(volume.cell_data["elemTag"])
    # The full-volume exterior excludes every internal tag-0/tag-1 interface
    # triangle by connectivity. Protect LVendo and both basal opening walls.
    candidates = (tags[owners] == 1) & ~np.isin(labels, [2, 4, 5])
    selected = boundary.extract_cells(np.flatnonzero(candidates)).extract_surface()
    if not selected.n_cells:
        raise RuntimeError("No exposed tag-1 LVepi faces remain after LVendo and basal exclusion.")
    samples, normals, _ = face_samples(selected)
    ref_ids, distances = nearest_triangles(source, samples)
    distances = distances.reshape(-1, 7)
    scores = np.sqrt(np.mean(distances ** 2, axis=1))
    _, ref_normals, ref_edge_scale = face_samples(source)
    agreement = np.abs(np.einsum(
        "ij,ij->i", normals, ref_normals[ref_ids.reshape(-1, 7)[:, -1]]))
    mapped = selected.point_data["volumeNodeId"][selected.faces.reshape(-1, 4)[:, 1:]]
    mapped_owners = np.asarray(selected.cell_data["volumeCellId"], dtype=np.int64)
    cells = np.asarray(volume.cells_dict[pv.CellType.TETRA])[mapped_owners]
    tolerance = np.zeros(len(cells))
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        tolerance = np.maximum(tolerance, np.linalg.norm(
            volume.points[cells[:, a]] - volume.points[cells[:, b]], axis=1))
    # Tiny target slivers can be much finer than their source triangles.
    # Check mapping at the local resolution of both meshes.
    source_spacing = ref_edge_scale[ref_ids.reshape(-1, 7)].max(axis=1)
    tolerance = np.maximum(np.maximum(tolerance, source_spacing), 1e-4)
    # Retain the complete exposed wall. Stop with diagnostics if its mapping
    # fails, rather than making holes by silently dropping distant faces.
    # A nearest source triangle can lie across a sharp basal crease, so
    # normal agreement is diagnostic; tag-1 topology determines wall membership.
    invalid = distances.max(axis=1) > tolerance
    a_surface = _volume_face_surface(
        volume, mapped, np.ones(len(mapped)), mapped_owners,
        np.full(len(mapped), -1, dtype=np.int64), lvendo_nodes)
    a_surface.cell_data["sourceSurfaceRMSDistance_mm"] = scores
    a_surface.cell_data["sourceSurfaceMaxDistance_mm"] = distances.max(axis=1)
    a_surface.cell_data["sourceNormalAgreement"] = agreement
    a_surface.cell_data["sourceMappingTolerance_mm"] = tolerance
    mapped_path = output_dir / "LVepi_mapped.vtk"
    a_surface.save(mapped_path, binary=True)
    if np.any(invalid):
        save_subset(a_surface, invalid, output_dir / "LVepi_unmapped_faces.vtk")
        raise RuntimeError("Exposed tag-1 LVepi faces exceed the local tetrahedral-edge mapping tolerance.")
    (output_dir / "LVepi_unmapped_faces.vtk").unlink(missing_ok=True)
    a_keys = {tuple(sorted(face)) for face in mapped}
    b_keys = {tuple(sorted(face)) for face in junction}
    if a_keys & b_keys:
        raise RuntimeError("Mapped LVepi contains junction triangles; the two parts must be disjoint.")
    b_surface = _volume_face_surface(
        volume, junction, np.full(len(junction), 2), junction_tag1, junction_tag0, lvendo_nodes)
    junction_path = output_dir / "LVepi_interface.vtk"
    b_surface.save(junction_path, binary=True)
    combined_faces = np.vstack((mapped, junction))
    combined_parts = np.r_[np.ones(len(mapped)), np.full(len(junction), 2)]
    combined = _volume_face_surface(
        volume, combined_faces, combined_parts, np.r_[mapped_owners, junction_tag1],
        np.r_[np.full(len(mapped), -1, dtype=np.int64), junction_tag0], lvendo_nodes)
    combined_path = output_dir / "LVepi_openbase.vtk"
    combined.save(combined_path, binary=True)
    mapped_nodes, junction_nodes = np.unique(mapped), np.unique(junction)
    nodes = np.union1d(mapped_nodes, junction_nodes)
    protected = np.intersect1d(nodes, lvendo_nodes)
    one_nodes = np.setdiff1d(nodes, lvendo_nodes)
    for name, ids in (("LVepi_mapped_nodes.vtk", mapped_nodes),
                      ("LVepi_interface_nodes.vtk", junction_nodes)):
        cloud = pv.PolyData(volume.points[ids].copy())
        cloud.point_data["volumeNodeId"] = ids
        cloud.point_data["PHO_BC"] = (~np.isin(ids, lvendo_nodes)).astype(np.int32)
        cloud.set_active_scalars("PHO_BC")
        cloud.save(output_dir / name, binary=True)
    internal = np.setdiff1d(one_nodes, original_ids)
    save_node_cloud(volume.points, internal, 1, output_dir / "LVepi_internal_one_nodes.vtk")
    info = {
        "method": "mapped_tag1_exterior_plus_extracted_tag0_tag1_interface",
        "source": str(Path(source_path).resolve()), "uses_Step03_LVepi": True,
        "uses_tag1_surface_selection": True, "source_surface_cut_performed": False,
        "mapped_surface": str(mapped_path), "interface_surface": str(junction_path),
        "combined_surface": str(combined_path),
        "boundary_part_legend": {"1": "mapped_LVepi_without_junction", "2": "extracted_tag0_tag1_interface"},
        "mapped_face_count": int(len(mapped)), "interface_face_count": int(len(junction)),
        "combined_face_count": int(len(combined_faces)), "overlapping_face_count": 0,
        "mapped_junction_faces_excluded_by": "full_volume_exterior_face_connectivity",
        "interface_extraction_rule": "complete_triangle_shared_by_one_tag0_and_one_tag1_tetrahedron",
        "mapped_excludes_surface_classes": ["LVendo", "LVannulus", "RVannulus"],
        "mapped_node_count": int(len(mapped_nodes)), "interface_node_count": int(len(junction_nodes)),
        "shared_seam_node_count": int(len(np.intersect1d(mapped_nodes, junction_nodes))),
        "node_count": int(len(nodes)), "condition_one_node_count": int(len(one_nodes)),
        "shared_LVendo_zero_node_count": int(len(protected)), "internal_node_count": int(len(internal)),
        "maximum_mapping_distance_mm": float(distances.max()),
        "maximum_distance_to_tolerance_ratio": float(np.max(distances.max(axis=1) / tolerance)),
        "mapping_tolerance_rule": "max_incident_tag1_tetra_edge_and_local_source_triangle_median_edge",
        "unmapped_face_count": int(np.count_nonzero(invalid)),
        "combined_connected_components": int(len(np.unique(combined.connectivity().cell_data["RegionId"]))),
        "combined_open_edge_count": int(combined.n_open_edges),
        "geometry_and_volume_node_ids_preserved": True,
    }
    (output_dir / "LVepi_mapping_summary.json").write_text(json.dumps(info, indent=2))
    return one_nodes, info


def build_pho_boundaries(volume, full_points, surface_paths, output_dir, edge_scale):
    """Set mapped LVepi plus the separate tag interface to 1, protecting LVendo=0."""
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
        "method": "source_triangles_to_volume_boundary_then_opening_rim_basal_components",
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
    rim_ids = {}
    for name in ("LV", "RV"):
        points, loop_info = _opening_rim_loop(str(surface_paths[name + "annulus"]))
        node_ids, mapping = _map_contour_to_mesh_nodes(
            points, np.asarray(volume.points), edge_scale, name + " PHO opening rim",
        )
        if not np.all(np.isin(node_ids, original_ids)):
            fail(f"{name} opening rim mapped to interior volume nodes.")
        rim_ids[name] = node_ids
        details[name + "_opening_rim"] = {"loop": loop_info, "mapping": mapping}

    # New basal walls are absent from the original closed endocardia. Identify
    # each only if its edge-connected component contains one complete saved rim.
    unknown = np.flatnonzero(labels == 0)
    if len(unknown):
        components = _face_components(triangles[unknown])
        for component in np.unique(components):
            face_ids = unknown[components == component]
            nodes = np.unique(triangles[face_ids])
            chambers = [name for name, ids in rim_ids.items() if np.all(np.isin(ids, nodes))]
            if len(chambers) != 1:
                fail("Unmapped PHO patch is not bounded by exactly one complete LV/RV opening rim.")
            chamber = chambers[0]
            labels[face_ids] = 4 if chamber == "LV" else 5
            details["basal_components"].append({
                "chamber": chamber, "face_count": int(len(face_ids)),
                "complete_opening_rim_covered": True,
            })
    save_diagnostics()
    tree = cKDTree(np.asarray(volume.points))
    def expand(ids):
        return np.unique(np.concatenate(tree.query_ball_point(volume.points[ids], r=1e-6))).astype(np.int64)
    lvendo_nodes = expand(np.unique(triangles[labels == 2]))
    outer_nodes = np.unique(triangles[labels == 1])
    lv_nodes, lv_mapping = _map_lvepi_and_extract_interface(
        volume, boundary, labels, surface_paths["LVepi"], lvendo_nodes, output_dir,
    )
    details["LVepi_mapping"] = lv_mapping
    one_nodes = expand(np.union1d(outer_nodes, lv_nodes))
    zero_nodes = np.unique(np.concatenate((triangles[labels > 1].ravel(), *rim_ids.values())))
    lv_overrides = np.intersect1d(zero_nodes, lv_nodes)
    zero_nodes = expand(zero_nodes)
    lvendo_priority_nodes = np.intersect1d(one_nodes, lvendo_nodes)
    one_nodes = np.setdiff1d(one_nodes, lvendo_nodes)
    shared_rim = np.intersect1d(zero_nodes, one_nodes)
    zero_nodes = np.setdiff1d(zero_nodes, one_nodes)
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
    boundary.cell_data["PHO_requested_BC"] = (labels == 1).astype(np.int32)
    boundary.cell_data["PHO_BC"] = np.where(
        np.all(face_bc == 1, axis=1), 1, np.where(np.all(face_bc == 0, axis=1), 0, -1),
    ).astype(np.int32)
    boundary.save(output_dir / "volume_boundary_classified.vtk", binary=True)
    for i, name in enumerate(SURFACE_CLASSES, start=1):
        save_subset(boundary, labels == i, output_dir / (name + "_mapped.vtk"))
    save_subset(boundary, np.isin(labels, [4, 5]), output_dir / "basal_opening_faces.vtk")
    details.update({
        "zero_node_count": int(len(zero_nodes)), "one_node_count": int(len(one_nodes)),
        "combined_outer_node_count": int(len(outer_nodes)),
        "LVepi_node_count": int(len(lv_nodes)),
        "LVepi_additional_one_node_count": int(len(np.setdiff1d(lv_nodes, outer_nodes))),
        "LVepi_zero_boundary_override_count": int(len(lv_overrides)),
        "LVendo_zero_node_count": int(len(lvendo_nodes)),
        "LVendo_priority_removed_one_node_count": int(len(lvendo_priority_nodes)),
        "shared_rim_node_count": int(len(shared_rim)),
        "constraint_overlap_count": 0,
        "shared_rim_rule": "LVendo_zero_first_then_mapped_LVepi_extracted_interface_and_combined_epicardium_one",
        "diagnostic_cell_PHO_BC_minus_one": "mixed_zero_and_one_vertex_values",
        "volume_geometry_modified": False,
    })
    report_path.write_text(json.dumps(details, indent=2))
    return zero_nodes, one_nodes, details
