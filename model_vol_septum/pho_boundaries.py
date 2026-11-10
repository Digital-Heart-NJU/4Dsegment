"""Classify a myocardial shell and map disjoint PHO constraints to volume nodes."""
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


SURFACE_CLASSES = ("LVepi", "RVepi", "LVendo", "RVendo", "LVannulus", "RVannulus")


def load_surface(path):
    mesh = pv.read(path).extract_surface().triangulate().clean(tolerance=0.0)
    if not mesh.is_all_triangles or not np.all(np.isfinite(mesh.points)):
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


def classify_shell(shell, references, edge_scale):
    """Use actual reference triangles, face samples and normal compatibility."""
    samples, normals, local_scale = face_samples(shell)
    scores, alignment = [], []
    for name in SURFACE_CLASSES:
        reference = references[name]
        ids, distances = nearest_triangles(reference, samples)
        _, ref_normals, _ = face_samples(reference)
        scores.append(np.sqrt(np.mean(distances.reshape(-1, 7) ** 2, axis=1)))
        dots = np.abs(np.einsum("ij,ij->i", np.repeat(normals, 7, axis=0), ref_normals[ids]))
        # Shared-edge nearest cells may belong to a different side of a rim.
        alignment.append(dots.reshape(-1, 7)[:, -1])
    scores, alignment = np.column_stack(scores), np.column_stack(alignment)
    winners = np.argmin(scores, axis=1)
    rows = np.arange(shell.n_cells)
    best = scores[rows, winners]
    opposite = np.where((np.arange(6)[None, :] < 2) != (winners[:, None] < 2), scores, np.inf).min(axis=1)
    tolerance = np.maximum(1e-4, np.minimum(0.2 * local_scale, 0.75 * edge_scale))
    good_distance = best <= tolerance
    good_normal = ((alignment[rows, winners] >= 0.5)
                   | (best <= np.maximum(1e-5, 0.01 * tolerance))
                   | (opposite - best > 0.1 * tolerance))
    clear = opposite - best > max(1e-8, 1e-8 * edge_scale)
    labels = np.where(good_distance & good_normal & clear, winners + 1, 0).astype(np.int32)

    # Boolean seam slivers can lie numerically on both original surfaces.
    # Apply the agreed epicardial priority only below the mesh area resolution.
    triangles = shell.points[shell.faces.reshape(-1, 4)[:, 1:]]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
    seam_slivers = ((labels == 0) & (areas <= 1e-6 * edge_scale ** 2)
                    & (scores[:, :2].min(axis=1) <= 1e-5)
                    & (scores[:, 2:4].min(axis=1) <= 1e-5))
    labels[seam_slivers] = np.argmin(scores[seam_slivers, :2], axis=1) + 1

    # Only resolve a geometric tie when the neighboring region agrees; never
    # propagate labels across an unmatched or incompatible surface patch.
    for _ in range(2):
        resolved = []
        for cell_id in np.flatnonzero((labels == 0) & good_distance & good_normal):
            neighbors = labels[shell.cell_neighbors(int(cell_id), connections="edges")]
            neighbors = neighbors[neighbors > 0]
            if len(neighbors) >= 2 and np.all((neighbors <= 2) == (neighbors[0] <= 2)):
                candidates = np.arange(6)[(np.arange(6) < 2) == (neighbors[0] <= 2)]
                candidate = int(candidates[np.argmin(scores[cell_id, candidates])])
                if scores[cell_id, candidate] <= best[cell_id] + max(1e-8, 1e-8 * edge_scale):
                    resolved.append((cell_id, candidate + 1))
        for cell_id, label in resolved:
            labels[cell_id] = label
    shell.cell_data["surfaceClass"] = labels
    shell.cell_data["PHO_BC"] = np.where(labels == 0, -1, (labels <= 2).astype(int))
    shell.cell_data["referenceDistance_mm"] = best
    shell.cell_data["referenceNormalAgreement"] = alignment[rows, winners]
    return labels, {
        "class_ids": {str(index + 1): name for index, name in enumerate(SURFACE_CLASSES)},
        "face_counts": {name: int(np.count_nonzero(labels == index + 1)) for index, name in enumerate(SURFACE_CLASSES)},
        "unresolved_faces": int(np.count_nonzero(labels == 0)),
        "subresolution_shared_seam_faces_given_epi_priority": int(np.count_nonzero(seam_slivers)),
        "normal_disagreement_resolved_by_clear_geometry": int(np.count_nonzero(
            (labels > 0) & (alignment[rows, winners] < 0.5))),
        "maximum_reference_distance_mm": float(best.max()),
        "p95_reference_distance_mm": float(np.percentile(best, 95)),
        "maximum_matching_tolerance_mm": float(tolerance.max()),
    }


def save_subset(surface, mask, path):
    selected = surface.extract_cells(np.flatnonzero(mask)).extract_surface() if np.any(mask) else pv.PolyData()
    selected.save(path, binary=False)


def save_node_cloud(points, ids, value, path):
    cloud = pv.PolyData(np.asarray(points)[ids])
    cloud.point_data["volumeNodeId"] = np.asarray(ids, dtype=np.int64)
    cloud.point_data["PHO_BC"] = np.full(len(ids), value, dtype=np.int32)
    cloud.save(path, binary=False)


def build_pho_boundaries(volume, full_points, surface_paths, output_dir, edge_scale):
    """Return PHO=0/1 node sets, keeping the supplied volume mesh unchanged."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = {name: load_surface(surface_paths[name]) for name in SURFACE_CLASSES}
    shell = load_surface(surface_paths["epi"])
    labels, details = classify_shell(shell, references, edge_scale)
    report_path = output_dir / "boundary_classification_summary.json"

    def report_failure(message):
        details["error"] = message
        report_path.write_text(json.dumps(details, indent=2))
        raise RuntimeError(f"{message} See {report_path}")

    shell.save(output_dir / "combined_shell_classified.vtk", binary=False)
    save_subset(shell, (labels > 0) & (labels <= 2), output_dir / "external_epicardium.vtk")
    save_subset(shell, labels == 0, output_dir / "unresolved_shell_faces.vtk")
    if np.any(labels == 0):
        report_failure("Combined-shell PHO classification has unresolved faces.")

    full_points = np.asarray(full_points, dtype=float)
    if full_points.shape != volume.points.shape or not np.allclose(full_points, volume.points, rtol=0.0, atol=1e-4):
        report_failure("CARP and VTK full-mesh point order or coordinates do not match.")
    boundary = volume.extract_surface(pass_pointid=True, pass_cellid=False).triangulate()
    original_ids = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    samples, normals, _ = face_samples(boundary)
    source_ids, distances = nearest_triangles(shell, samples)
    source_ids, distances = source_ids.reshape(-1, 7), distances.reshape(-1, 7)
    boundary_labels = labels[source_ids[:, -1]]
    _, shell_normals, _ = face_samples(shell)
    agreement = np.abs(np.einsum("ij,ij->i", normals, shell_normals[source_ids[:, -1]]))
    transfer_tolerance = max(1e-4, 0.3 * edge_scale)
    invalid = (distances.max(axis=1) > transfer_tolerance) | (agreement < 0.5)
    boundary.cell_data["surfaceClass"] = boundary_labels
    boundary.cell_data["PHO_BC"] = (boundary_labels <= 2).astype(np.int32)
    boundary.cell_data["shellMappingDistance_mm"] = distances.max(axis=1)
    boundary.save(output_dir / "volume_boundary_classified.vtk", binary=False)
    save_subset(boundary, invalid, output_dir / "unresolved_volume_faces.vtk")
    details["volume_boundary_maximum_mapping_distance_mm"] = float(distances.max())
    details["volume_boundary_transfer_tolerance_mm"] = transfer_tolerance
    if np.any(invalid):
        report_failure("Volume boundary does not reliably match the classified combined shell.")

    triangles = original_ids[boundary.faces.reshape(-1, 4)[:, 1:]]
    epicardial_nodes = np.unique(triangles[boundary_labels <= 2])
    zero_nodes = np.unique(triangles[boundary_labels > 2])
    shared_rim = np.intersect1d(zero_nodes, epicardial_nodes)
    zero_nodes = np.setdiff1d(zero_nodes, shared_rim)

    # Boolean union removes internal LVepi faces from the external shell.
    # Sample that original surface separately and map only to eligible nodes.
    lv_samples, _, _ = face_samples(references["LVepi"])
    lv_samples = lv_samples.reshape(-1, 3)
    _, raw_ids = cKDTree(full_points).query(lv_samples)
    conflicts = np.intersect1d(raw_ids, zero_nodes)
    save_node_cloud(full_points, conflicts, -1, output_dir / "LVepi_mapping_conflicts.vtk")
    if len(conflicts):
        if not len(shared_rim):
            report_failure("Internal LVepi maps to endocardial/annular nodes away from any shared rim.")
        _, surface_distances = nearest_triangles(references["LVepi"], full_points[conflicts])
        actual_shared = conflicts[surface_distances <= max(1e-5, 1e-6 * edge_scale)]
        rim_distances, _ = cKDTree(full_points[shared_rim]).query(full_points[actual_shared])
        details["LVepi_initial_conflicts"] = int(len(conflicts))
        details["LVepi_rejected_nearest_zero_candidates"] = int(len(conflicts) - len(actual_shared))
        details["LVepi_true_shared_curve_nodes_added"] = int(len(actual_shared))
        details["LVepi_conflict_maximum_distance_to_shared_rim_mm"] = float(rim_distances.max()) if len(rim_distances) else 0.0
        if np.any(rim_distances > 1.5 * edge_scale):
            report_failure("Internal LVepi conflicts with PHO=0 patch interiors, not just their shared rims.")
        shared_rim = np.union1d(shared_rim, actual_shared)
        epicardial_nodes = np.union1d(epicardial_nodes, actual_shared)
        zero_nodes = np.setdiff1d(zero_nodes, actual_shared)

    eligible = np.setdiff1d(np.arange(len(full_points)), zero_nodes)
    distances, mapped = cKDTree(full_points[eligible]).query(lv_samples)
    lv_nodes = np.unique(eligible[mapped])
    tetrahedra = np.asarray(volume.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    cell_scale = np.zeros(len(tetrahedra))
    for first, second in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        cell_scale = np.maximum(cell_scale, np.linalg.norm(
            full_points[tetrahedra[:, first]] - full_points[tetrahedra[:, second]], axis=1))
    node_scale = np.zeros(len(full_points))
    for column in range(4):
        np.maximum.at(node_scale, tetrahedra[:, column], cell_scale)
    mapping_tolerance = np.maximum(1e-4, node_scale[eligible[mapped]])
    details["LVepi_mapping_maximum_distance_mm"] = float(distances.max())
    details["LVepi_mapping_p95_distance_mm"] = float(np.percentile(distances, 95))
    details["LVepi_mapping_tolerance_mm"] = float(mapping_tolerance.max())
    details["LVepi_mapping_tolerance_rule"] = "one_maximum_incident_tetrahedral_edge_at_the_mapped_node"
    details["LVepi_mapping_is_approximate"] = True
    if np.any(distances > mapping_tolerance):
        save_node_cloud(lv_samples, np.flatnonzero(distances > mapping_tolerance), -1,
                        output_dir / "unmapped_LVepi_samples.vtk")
        report_failure("LVepi-to-volume mapping lacks coverage within the local tetrahedral spacing.")
    one_nodes = np.union1d(epicardial_nodes, lv_nodes)
    neighbors = cKDTree(full_points).query_ball_point(full_points[one_nodes], r=1e-6)
    one_nodes = np.unique(np.concatenate(neighbors)).astype(np.int64)
    if len(np.intersect1d(zero_nodes, one_nodes)):
        report_failure("Coincident volume nodes have contradictory PHO constraints.")
    if not len(one_nodes) or not len(zero_nodes):
        report_failure("Both PHO=0 and PHO=1 boundary sets must be nonempty.")

    save_node_cloud(full_points, zero_nodes, 0, output_dir / "PHO_zero_nodes.vtk")
    save_node_cloud(full_points, one_nodes, 1, output_dir / "PHO_one_nodes.vtk")
    save_node_cloud(full_points, lv_nodes, 1, output_dir / "LVepi_one_nodes.vtk")
    save_node_cloud(full_points, shared_rim, 1, output_dir / "shared_rim_one_nodes.vtk")
    details.update({
        "zero_node_count": int(len(zero_nodes)), "one_node_count": int(len(one_nodes)),
        "LVepi_node_count": int(len(lv_nodes)), "shared_rim_node_count": int(len(shared_rim)),
        "constraint_overlap_count": 0, "shared_rim_rule": "epicardium_one_has_priority",
        "volume_geometry_modified": False,
    })
    report_path.write_text(json.dumps(details, indent=2))
    return zero_nodes, one_nodes, details
