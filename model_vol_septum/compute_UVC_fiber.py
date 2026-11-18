#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:36:35 2022

@author: sq20
"""
import subprocess

import pandas as pd
import numpy as np
import os
import glob
import math
import random
import shutil
import json
import vtk
import pyvista as pv
import logging
from pathlib import Path
from typing import Dict, Tuple

from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

from . import meshIO
from . import meshtool_func as meshtool
from . import carpfunc
from . import geometrical
from . import Coord_UVC_function

'''
LV_ENDOCARDIAL = 0
RV_SEPTUM = 1
RV_FREEWALL = 2
EPICARDIAL = 3
MITRAL_VALVE = 4
AORTA_VALVE = 5
TRICUSPID_VALVE = 6
PULMONARY_VALVE = 7
RV_INSERT = 8
'''

def coord_z(folder, meshtoolloc, mesh, input_dir, meshnames, surfloc, UVCfolder):
    # "folder" is where the labeled mesh stored,"mesh" is the mesh name
    # input_dir is where surface mesh stored
    # "surfloc" is where to store surfaces
    # find apex point in LV only mesh
    # output is the x,y,z coord of the apex point
    # mesh_names: ['Mesh_LV_endo_timeframe001', 'Mesh_RV_FW_timeframe001', 'Mesh_RV_septum_timeframe001',
    #              'Mesh_aorta_valve_timeframe001', 'Mesh_epi_timeframe001', 'Mesh_mitral_valve_timeframe001',
    #              'Mesh_pulmonary_valve_timeframe001', 'Mesh_tricuspid_valve_timeframe001']
    os.chdir(folder)

    if not meshtool.extract_mesh(meshtoolloc, mesh, "1", "LV_all"): # 只提取LV_all部分,此部分的tag为1
        return None

    Mesh_mitral_valve_surf = meshnames[5]

    if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_mitral_valve_surf}", input_dir+f"/{Mesh_mitral_valve_surf}", "vtk", "carp_txt"):
        return None

    if not os.path.exists(surfloc):
        os.mkdir(surfloc)

    # extract all surface in biv mesh (only LV and RV) to store in folder surfloc/biv_i
    if not os.path.exists(surfloc + "/biv_i/"):
        os.mkdir(surfloc + "/biv_i/")

    mitral_valve_files = [
        (input_dir + f"/{Mesh_mitral_valve_surf}" + ".pts",
        input_dir + f"/{Mesh_mitral_valve_surf}" + ".elem")
    ]

    if not Coord_UVC_function.find_intersection_base(mitral_valve_files, "LV_all", surfloc + "/biv_i/lv_base"):
        return None
    # if not meshtool.extract_surf(meshtoolloc, "LV_all", surfloc + "/mitral", "4"):
    #     return None
    #1待修改:此处先convert Mesh_mitral_valve_timeframe001的vtk为carp_txt
    #2待修改:python find_intersection_lv_base.py 即通过二尖瓣的.pts/.elem寻找瓣膜环lv_base_i

    d_from_mitral = "d_mitraltoLVall.dat"
    # if not meshtool.generate_dfield(meshtoolloc, "LV_all", surfloc + "/mitral", d_from_mitral):
    #     return None
    #3待修改:if not meshtool.generate_dfield(meshtoolloc, "LV_all", "lv_base_i", d_from_mitral):
    #     return None

    if not meshtool.generate_dfield(meshtoolloc, "LV_all", surfloc + "/biv_i/lv_base", d_from_mitral):
        return None

    d_LVtomitral = np.genfromtxt(d_from_mitral)
    # find the index of the furthest point from the mitral valve as apex point
    index = np.where(d_LVtomitral == np.amax(d_LVtomitral))[0]

    if not meshtool.extract_mesh(meshtoolloc, mesh, "0,1", mesh + "_i"):  # 此处有valve的mesh进行左右心室提取,无valve的情况下提取就是本身(可复制替代本操作)
        return None

    pts_LV = np.array(meshIO.read_pts(basename="LV_all"))
    pts_biv = np.array(meshIO.read_pts(basename=mesh + "_i"))

    # find the apex point's index in biv mesh
    ptsindex_biv = np.where(np.all(pts_biv == pts_LV[index], axis=1))[0]

    if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surfloc + "/biv_i/myoall", "0,1"):
        return None

    surf = meshIO.read_surf(basename=surfloc + "/biv_i/myoall")
    # find the surface including the apex point and write to mesh+"_i_apex" in both .surf and .vtx
    surfnew = surf.loc[
        (surf[1] == int(ptsindex_biv)) | (surf[2] == int(ptsindex_biv)) | (surf[3] == int(ptsindex_biv)), :]
    surf_vtx = np.unique(surfnew.loc[:, 1:3])

    Filename = surfloc + "/biv_i/apex_i"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)

    # extract base surf in myo only mesh which is the intersection region of valves with myo
    # first extract the "biv_i_all-biv_allvalves" in the whole mesh and map it to biv myo only mesh
    # if not meshtool.extract_surf(meshtoolloc, mesh, surfloc + "/intersection", "0,1:4,5,6,7"):
    #     return None
    #4待修改:此处先convert Mesh_mitral_valve(之前已完成)/Mesh_tricuspid_valve/Mesh_aorta_valve/Mesh_pulmonary_valve的vtk为carp_txt

    #5待修改:python find_intersection.py 即通过多个瓣膜的.pts/.elem寻找瓣膜环的整体base_i
    # if not meshtool.map_file(meshtoolloc, mesh + "_i", surfloc + "/intersection.surf", surfloc + "/biv_i"):
    #     return None
    # if not meshtool.map_file(meshtoolloc, mesh + "_i", surfloc + "/intersection.surf.vtx", surfloc + "/biv_i"):
    #     return None

    # Mesh_mitral_valve_surf = meshnames[5] # 上面已经完成
    Mesh_tricuspid_valve = meshnames[7]
    Mesh_aorta_valve = meshnames[3]
    Mesh_pulmonary_valve = meshnames[6]

    if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_tricuspid_valve}", input_dir+f"/{Mesh_tricuspid_valve}", "vtk", "carp_txt"):
        return None
    if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_aorta_valve}", input_dir+f"/{Mesh_aorta_valve}", "vtk", "carp_txt"):
        return None
    if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_pulmonary_valve}", input_dir+f"/{Mesh_pulmonary_valve}", "vtk", "carp_txt"):
        return None

    all_valve_files = [
        (input_dir + f"/{Mesh_mitral_valve_surf}" + ".pts",
        input_dir + f"/{Mesh_mitral_valve_surf}" + ".elem"),
        (input_dir + f"/{Mesh_tricuspid_valve}" + ".pts",
         input_dir + f"/{Mesh_tricuspid_valve}" + ".elem"),
        (input_dir + f"/{Mesh_aorta_valve}" + ".pts",
         input_dir + f"/{Mesh_aorta_valve}" + ".elem"),
        (input_dir + f"/{Mesh_pulmonary_valve}" + ".pts",
         input_dir + f"/{Mesh_pulmonary_valve}" + ".elem"),
    ]

    if not Coord_UVC_function.find_intersection_base(all_valve_files, mesh + "_i", surfloc + "/biv_i/base_i"):
        return None

    # calculate the distance from base to apex and store the results in
    apexsurf = surfloc + "/biv_i/apex_i"
    # basesurf = surfloc + "/biv_i/intersection"
    #6待修改:basesurf = surfloc + "/biv_i/base_i"
    basesurf = surfloc + "/biv_i/base_i"
    outputFilename = UVCfolder + "/COORDS_Z.dat"

    if not meshtool.generate_dfield(meshtoolloc, mesh + "_i", apexsurf, outputFilename, "-esurf=" + basesurf):
        return None

    if not meshtool.collect_nodal(meshtoolloc, mesh + "_i", "carp_txt", UVCfolder + "/COORDS_Z", '',UVCfolder + "/COORDS_Z.dat"):
        return None
    #7待修改:可视化COORDS_Z.dat可通过collect -imsh=subject0071_mesh_all -omsh=result -nod=COORDS_Z.dat
    # 产物: result.geo / result.case / result.tags.ens / result.fiber.ens / result_COORDS_Z.ens
    return outputFilename


def _annulus_high_z_boundary_loop(vtk_path: str) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the complete connected high-Z boundary loop of an annular surface."""
    annulus = pv.read(vtk_path).extract_surface().triangulate().clean(tolerance=0.0)
    boundary = annulus.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    if boundary.n_cells == 0:
        raise RuntimeError(f"Annulus has no boundary edges: {vtk_path}")

    connected = boundary.connectivity()
    region_ids = np.asarray(connected.cell_data.get("RegionId", []), dtype=np.int64)
    regions = np.unique(region_ids)
    if len(regions) != 2:
        raise RuntimeError(
            f"Expected exactly two closed boundary loops in {vtk_path}, found {len(regions)}."
        )

    records = []
    loop_points = []
    for region_id in regions:
        component = connected.extract_cells(np.flatnonzero(region_ids == region_id)).extract_surface()
        lines = np.asarray(component.lines, dtype=np.int64)
        if not len(lines) or len(lines) % 3 != 0:
            raise RuntimeError(f"Invalid boundary-line topology in {vtk_path}, region {region_id}.")
        line_cells = lines.reshape(-1, 3)
        if np.any(line_cells[:, 0] != 2):
            raise RuntimeError(f"Non-segment boundary cell in {vtk_path}, region {region_id}.")
        edges = line_cells[:, 1:]
        used, degree = np.unique(edges, return_counts=True)
        if len(used) < 3 or np.any(degree != 2):
            raise RuntimeError(f"Boundary region {region_id} in {vtk_path} is not one closed loop.")
        points = np.asarray(component.points, dtype=float)[used]
        z = points[:, 2]
        record = {
            "region_id": int(region_id),
            "point_count": int(len(points)),
            "mean_z": float(np.mean(z)),
            "median_z": float(np.median(z)),
            "p90_z": float(np.percentile(z, 90)),
            "min_z": float(np.min(z)),
            "max_z": float(np.max(z)),
            "length": float(component.length),
        }
        records.append(record)
        loop_points.append(points)

    selected_index = max(
        range(len(records)),
        key=lambda index: (
            records[index]["median_z"], records[index]["mean_z"], records[index]["p90_z"]
        ),
    )
    return loop_points[selected_index], {
        "source": str(Path(vtk_path).resolve()),
        "selection": "connected_closed_boundary_loop_with_highest_median_then_mean_z",
        "selected_region_id": records[selected_index]["region_id"],
        "selected_loop": records[selected_index],
        "candidate_loops": records,
    }


def _opening_rim_loop(vtk_path: str) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read one supplied closed opening ring without choosing a boundary by Z."""
    rim = pv.read(vtk_path)
    if not isinstance(rim, pv.PolyData):
        raise RuntimeError(f"Opening rim must be a PolyData curve: {vtk_path}")
    rim = rim.clean(tolerance=0.0)
    if rim.n_lines == 0 or rim.n_cells != rim.n_lines:
        raise RuntimeError(f"Opening rim must contain only line cells: {vtk_path}")
    if not np.all(np.isfinite(rim.points)):
        raise RuntimeError(f"Opening rim contains non-finite coordinates: {vtk_path}")

    segments = []
    lines = np.asarray(rim.lines, dtype=np.int64)
    offset = 0
    while offset < len(lines):
        count = int(lines[offset])
        ids = lines[offset + 1:offset + 1 + count]
        if count < 2 or len(ids) != count:
            raise RuntimeError(f"Invalid opening-rim line cell: {vtk_path}")
        segments.extend(zip(ids[:-1], ids[1:]))
        offset += count + 1
    edges = np.asarray(segments, dtype=np.int64)
    used, degree = np.unique(edges, return_counts=True)
    regions = np.unique(rim.connectivity().cell_data["RegionId"])
    if (len(regions) != 1 or len(used) < 3 or np.any(degree != 2)
            or np.any(edges[:, 0] == edges[:, 1])):
        raise RuntimeError(f"Opening rim must be exactly one connected closed ring: {vtk_path}")
    points = np.asarray(rim.points, dtype=float)[used]
    lengths = np.linalg.norm(rim.points[edges[:, 0]] - rim.points[edges[:, 1]], axis=1)
    return points, {
        "source": str(Path(vtk_path).resolve()),
        "selection": "single_closed_opening_rim_used_directly",
        "point_count": int(len(points)),
        "segment_count": int(len(edges)),
        "length_mm": float(lengths.sum()),
        "length_weighted_centroid": np.average(
            rim.points[edges].mean(axis=1), axis=0, weights=lengths
        ).tolist(),
    }


def _curve_enclosed_area(curve):
    """Measure a closed line network in its least-squares best-fit plane."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    lines = np.asarray(curve.lines, dtype=np.int64)
    if (curve.n_points < 3 or curve.n_cells != curve.n_lines or not len(lines)
            or len(lines) % 3 or np.any(lines.reshape(-1, 3)[:, 0] != 2)
            or not np.all(np.isfinite(curve.points))):
        raise RuntimeError("Interface curve must contain finite two-node line segments.")
    edges = lines.reshape(-1, 3)[:, 1:]
    degree = np.bincount(edges.ravel(), minlength=curve.n_points)
    if np.any(degree < 2) or np.any(degree % 2) or np.any(edges[:, 0] == edges[:, 1]):
        raise RuntimeError("Interface curve has an open boundary; enclosed area is undefined.")
    points = np.asarray(curve.points, dtype=float)
    origin = points.mean(axis=0)
    centered = points - origin
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T
    # Polygonization also handles small loops attached at a shared boundary node.
    # No endpoint connections or convex hulls are introduced to invent an area.
    segments = unary_union([LineString(projected[edge]) for edge in edges])
    polygons = list(polygonize(segments))
    area = float(unary_union(polygons).area)
    if not np.isfinite(area) or area <= np.finfo(float).eps:
        raise RuntimeError("Interface curve has no positive enclosed projected area.")
    deviation = centered @ axes[2]
    return {
        "enclosed_area_mm2": area,
        "area_method": "union_of_enclosed_polygons_in_best_fit_plane",
        "best_fit_plane_origin": origin.tolist(),
        "best_fit_plane_normal": axes[2].tolist(),
        "out_of_plane_rms_mm": float(np.sqrt(np.mean(deviation ** 2))),
        "out_of_plane_max_mm": float(np.max(np.abs(deviation))),
        "junction_node_count": int(np.count_nonzero(degree > 2)),
        "projected_polygon_count": len(polygons),
        "point_count": int(curve.n_points),
        "segment_count": int(curve.n_lines),
    }


def extract_tag_interface_curve(volume_vtk, output_dir):
    """Select the largest-area curve from any number of tag-interface curves."""
    from .tag_step10_volume import shared_face_neighbors

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    volume = pv.read(volume_vtk)
    if (not isinstance(volume, pv.UnstructuredGrid) or not volume.n_cells
            or not np.all(volume.celltypes == pv.CellType.TETRA)
            or not np.all(np.isfinite(volume.points))):
        raise RuntimeError("PHI interface extraction requires a finite tetrahedral volume.")
    if "elemTag" not in volume.cell_data:
        raise RuntimeError("PHI interface extraction requires the elemTag cell array.")
    tags = np.asarray(volume.cell_data["elemTag"])
    if set(np.unique(tags).tolist()) != {0, 1}:
        raise RuntimeError("PHI interface extraction requires exactly tag 0 and tag 1.")
    a, b, shared_faces = shared_face_neighbors(volume.cells_dict[pv.CellType.TETRA])
    between = tags[a] != tags[b]
    triangles = shared_faces[between]
    if not len(triangles):
        raise RuntimeError("Tag 0 and tag 1 do not share any complete triangular faces.")
    node_ids, local = np.unique(triangles, return_inverse=True)
    faces = np.column_stack((np.full(len(triangles), 3), local.reshape(-1, 3)))
    interface = pv.PolyData(volume.points[node_ids].copy(), faces=faces.ravel())
    interface.point_data["volumeNodeId"] = node_ids
    interface.cell_data["tag0CellId"] = np.where(tags[a[between]] == 0, a[between], b[between])
    interface.cell_data["tag1CellId"] = np.where(tags[a[between]] == 1, a[between], b[between])
    interface_path = output_dir / "tag_interface_surface.vtk"
    interface.save(interface_path, binary=True)

    triangle_edges = np.sort(np.vstack((
        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
    )), axis=1)
    edges, incidence = np.unique(triangle_edges, axis=0, return_counts=True)
    # At a nonmanifold interface junction, three incident triangles leave one
    # boundary edge after cancellation. Dropping these edges opens real curves.
    boundary_edges = edges[incidence % 2 == 1]
    if not len(boundary_edges):
        raise RuntimeError("The tag-0/tag-1 interface has no boundary curves.")
    boundary_ids, local = np.unique(boundary_edges, return_inverse=True)
    boundary = pv.PolyData(volume.points[boundary_ids].copy(), lines=np.column_stack((
        np.full(len(boundary_edges), 2), local.reshape(-1, 2),
    )).ravel())
    boundary.point_data["volumeNodeId"] = boundary_ids
    boundary.cell_data["interfaceTriangleIncidence"] = incidence[incidence % 2 == 1]
    boundary_path = output_dir / "tag_interface_boundary_curves.vtk"
    boundary.save(boundary_path, binary=True)
    connected = boundary.connectivity()
    region_ids = np.asarray(connected.cell_data["RegionId"], dtype=np.int64)
    candidates, curves = [], []
    for region_id in np.unique(region_ids):
        curve = connected.extract_cells(np.flatnonzero(region_ids == region_id)).extract_surface()
        record = _curve_enclosed_area(curve)
        record["region_id"] = int(region_id)
        path = output_dir / f"tag_interface_curve_candidate_{region_id}.vtk"
        curve.field_data["enclosed_area_mm2"] = np.array([record["enclosed_area_mm2"]])
        curve.save(path, binary=True)
        record["path"] = str(path)
        candidates.append(record)
        curves.append(curve)
    if not curves:
        raise RuntimeError(
            "No tag-interface boundary curves were found. "
            f"Inspect {boundary_path}."
        )
    selected = max(range(len(curves)), key=lambda index: candidates[index]["enclosed_area_mm2"])
    selected_path = output_dir / "tag_interface_curve.vtk"
    curves[selected].save(selected_path, binary=True)
    summary = {
        "source_volume": str(Path(volume_vtk).resolve()),
        "method": "shared_tag_0_tag_1_faces_then_boundary_curves",
        "selection": "largest_enclosed_area" if len(curves) > 1 else "only_interface_curve",
        "interface_surface": str(interface_path),
        "interface_triangle_count": int(len(triangles)),
        "boundary_curves": str(boundary_path),
        "odd_nonmanifold_boundary_edge_count": int(np.count_nonzero(incidence[incidence % 2 == 1] > 1)),
        "curve_count": len(curves),
        "candidates": candidates,
        "selected_region_id": candidates[selected]["region_id"],
        "selected_enclosed_area_mm2": candidates[selected]["enclosed_area_mm2"],
        "selected_curve": str(selected_path),
        "source_volume_modified": False,
    }
    (output_dir / "tag_interface_curve_summary.json").write_text(json.dumps(summary, indent=2))
    return str(selected_path), summary


def _tetra_edge_scale(volume: pv.UnstructuredGrid) -> float:
    tetrahedra = volume.cells_dict.get(vtk.VTK_TETRA)
    if tetrahedra is None or not len(tetrahedra):
        raise RuntimeError("COORDS_Z input VTK contains no tetrahedral cells.")
    stride = max(1, len(tetrahedra) // 50000)
    tetrahedra = np.asarray(tetrahedra[::stride], dtype=np.int64)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    lengths = np.concatenate([
        np.linalg.norm(volume.points[tetrahedra[:, first]] - volume.points[tetrahedra[:, second]], axis=1)
        for first, second in pairs
    ])
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if not len(lengths):
        raise RuntimeError("Cannot determine a positive edge scale from the volume mesh.")
    return float(np.median(lengths))


def _map_contour_to_mesh_nodes(
    contour_points: np.ndarray,
    mesh_points: np.ndarray,
    edge_scale: float,
    name: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    distances, node_ids = cKDTree(np.asarray(mesh_points, dtype=float)).query(
        np.asarray(contour_points, dtype=float), k=1
    )
    tolerance = max(1e-4, 2.0 * float(edge_scale))
    if not np.all(np.isfinite(distances)) or float(np.max(distances)) > tolerance:
        raise RuntimeError(
            f"{name}: contour-to-volume mapping distance {np.max(distances):.6g} mm "
            f"exceeds the adaptive tolerance {tolerance:.6g} mm."
        )
    unique_ids = np.unique(np.asarray(node_ids, dtype=np.int64))
    if len(unique_ids) < 3:
        raise RuntimeError(f"{name}: fewer than three unique volume nodes were mapped.")
    return unique_ids, {
        "source_point_count": int(len(contour_points)),
        "mapped_unique_node_count": int(len(unique_ids)),
        "distance_min_mm": float(np.min(distances)),
        "distance_mean_mm": float(np.mean(distances)),
        "distance_p95_mm": float(np.percentile(distances, 95)),
        "distance_max_mm": float(np.max(distances)),
        "adaptive_tolerance_mm": tolerance,
    }


def _write_carp_node_surface(basename: str, node_ids: np.ndarray) -> None:
    """Write a CARP node boundary as .surf and .surf.vtx files."""
    node_ids = np.unique(np.asarray(node_ids, dtype=np.int64))
    if not len(node_ids):
        raise RuntimeError(f"Cannot write an empty CARP node surface: {basename}")
    Path(basename).parent.mkdir(parents=True, exist_ok=True)
    meshIO.write_vtx_File(
        vtxFilename=basename + ".surf",
        vtx=pd.Series(node_ids, dtype=np.int64),
    )
    degenerate_triangles = pd.DataFrame(
        np.repeat(node_ids[:, None], 3, axis=1), columns=(1, 2, 3)
    )
    meshIO.write_surf(surfFilename=basename, surf=degenerate_triangles, shapes="Tr")


def _expand_coincident_node_ids(full_points, node_ids, tolerance_mm=1e-6):
    """Include coincident duplicate volume nodes on an interface surface."""
    full_points = np.asarray(full_points, dtype=float)
    node_ids = np.unique(np.asarray(node_ids, dtype=np.int64))
    if not len(node_ids):
        return node_ids
    neighbors = cKDTree(full_points).query_ball_point(
        full_points[node_ids], r=float(tolerance_mm)
    )
    expanded = np.unique(
        np.concatenate([np.asarray(ids, dtype=np.int64) for ids in neighbors if len(ids)])
    )
    return expanded


def _project_surface_points_to_full_mesh(source_vtk, full_points, output_surface):
    """Project a source surface onto all full-mesh nodes, including internal interfaces.

    The usual PHO surface extraction operates on ``extract_surface()`` and therefore
    cannot represent an LVepi patch that is internal to a combined LV/RV volume.
    This path uses the source vertices and triangle centroids directly, then expands
    coincident full-mesh nodes so an interface is represented on both sides.
    """
    source = pv.read(str(source_vtk))
    if isinstance(source, pv.UnstructuredGrid):
        source = source.extract_surface()
    source = source.triangulate().clean(tolerance=0.0)
    source_points = np.asarray(source.points, dtype=float)
    if source.n_points == 0 or source.n_cells == 0:
        raise RuntimeError(f"LVepi projection source is empty: {source_vtk}")

    samples = [source_points]
    centers = np.asarray(source.cell_centers().points, dtype=float)
    if len(centers):
        samples.append(centers)
    samples = np.vstack(samples)
    distances, node_ids = cKDTree(np.asarray(full_points, dtype=float)).query(samples, k=1)
    if not np.all(np.isfinite(distances)):
        raise RuntimeError(f"Non-finite LVepi-to-volume projection distance: {source_vtk}")

    mapped_ids = _expand_coincident_node_ids(
        full_points, np.unique(np.asarray(node_ids, dtype=np.int64)), tolerance_mm=1e-6
    )
    if len(mapped_ids) < 3:
        raise RuntimeError(f"LVepi projection mapped fewer than three full-mesh nodes: {source_vtk}")

    _write_carp_node_surface(str(output_surface), mapped_ids)
    return {
        "source": str(Path(source_vtk).resolve()),
        "output": str(output_surface),
        "source_point_count": int(source.n_points),
        "source_triangle_count": int(source.n_cells),
        "projection_sample_count": int(len(samples)),
        "node_count": int(len(mapped_ids)),
        "max_mapping_distance_mm": float(np.max(distances)),
        "p95_mapping_distance_mm": float(np.percentile(distances, 95)),
        "mapping": "source_vertices_and_triangle_centroids_to_full_mesh_with_coincident_node_expansion",
    }


def coord_z_openbase_annuli(
    folder: str,
    meshtoolloc: str,
    meshname: str,
    volume_vtk: str,
    lv_annulus_vtk: str,
    rv_annulus_vtk: str,
    surfloc: str,
    UVCfolder: str,
) -> Tuple[str, Dict[str, object]]:
    """Generate COORDS_Z using the complete supplied LV/RV opening rings."""
    folder_path = Path(folder).resolve()
    surface_root = folder_path / surfloc / "biv_i"
    uvc_root = Path(UVCfolder).resolve()
    surface_root.mkdir(parents=True, exist_ok=True)
    uvc_root.mkdir(parents=True, exist_ok=True)
    os.chdir(folder_path)

    volume = pv.read(volume_vtk)
    if not isinstance(volume, pv.UnstructuredGrid):
        raise RuntimeError(f"COORDS_Z volume input is not an unstructured grid: {volume_vtk}")
    edge_scale = _tetra_edge_scale(volume)

    lv_loop, lv_loop_info = _opening_rim_loop(lv_annulus_vtk)
    rv_loop, rv_loop_info = _opening_rim_loop(rv_annulus_vtk)

    lv_meshname = "LV_all"
    if not meshtool.extract_mesh(meshtoolloc, meshname, "1", lv_meshname):
        raise RuntimeError("meshtool failed to extract the elemTag=1 LV mesh.")
    full_points = np.asarray(meshIO.read_pts(basename=meshname), dtype=float)
    lv_points = np.asarray(meshIO.read_pts(basename=lv_meshname), dtype=float)

    lv_base_ids, lv_base_map = _map_contour_to_mesh_nodes(
        lv_loop, lv_points, edge_scale, "LV opening rim to LV mesh"
    )
    lv_base = str(surface_root / "lv_base")
    _write_carp_node_surface(lv_base, lv_base_ids)

    lv_distance_file = str(folder_path / "d_LVbasetoLVall.dat")
    if not meshtool.generate_dfield(meshtoolloc, lv_meshname, lv_base, lv_distance_file):
        raise RuntimeError("meshtool failed to calculate the LV base distance field.")
    lv_distance = np.atleast_1d(np.genfromtxt(lv_distance_file)).reshape(-1)
    if len(lv_distance) != len(lv_points) or not np.all(np.isfinite(lv_distance)):
        raise RuntimeError("LV base distance field has an invalid size or non-finite values.")

    lv_outer = str(surface_root / "LV_outer")
    if not meshtool.extract_surf(meshtoolloc, lv_meshname, lv_outer, "1"):
        raise RuntimeError("meshtool failed to extract the LV exterior surface.")
    lv_surface_ids = np.atleast_1d(meshIO.read_vtx_File(lv_outer + ".surf.vtx")).astype(np.int64)
    lv_surface_ids = lv_surface_ids[(lv_surface_ids >= 0) & (lv_surface_ids < len(lv_points))]
    if not len(lv_surface_ids):
        raise RuntimeError("The extracted LV exterior surface has no valid vertices.")
    apex_lv_id = int(lv_surface_ids[np.argmax(lv_distance[lv_surface_ids])])
    apex_coordinate = lv_points[apex_lv_id]

    apex_distance, apex_full_id = cKDTree(full_points).query(apex_coordinate, k=1)
    mapping_tolerance = max(1e-4, 2.0 * edge_scale)
    if not np.isfinite(apex_distance) or float(apex_distance) > mapping_tolerance:
        raise RuntimeError("The LV apex could not be mapped back to the full biventricular mesh.")
    apex_surface = str(surface_root / "apex_i")
    _write_carp_node_surface(apex_surface, np.array([int(apex_full_id)], dtype=np.int64))

    lv_full_ids, lv_full_map = _map_contour_to_mesh_nodes(
        lv_loop, full_points, edge_scale, "LV opening rim to biventricular mesh"
    )
    rv_full_ids, rv_full_map = _map_contour_to_mesh_nodes(
        rv_loop, full_points, edge_scale, "RV opening rim to biventricular mesh"
    )
    combined_base_ids = np.unique(np.concatenate((lv_full_ids, rv_full_ids)))
    if int(apex_full_id) in combined_base_ids:
        raise RuntimeError("The selected LV apex is also an opening-rim base node.")
    comprehensive_base = str(surface_root / "base_i")
    _write_carp_node_surface(comprehensive_base, combined_base_ids)

    output_filename = str(uvc_root / "COORDS_Z.dat")
    if not meshtool.generate_dfield(
        meshtoolloc, meshname, apex_surface, output_filename, "-esurf=" + comprehensive_base
    ):
        raise RuntimeError("meshtool failed to calculate COORDS_Z.")
    coords_z = np.atleast_1d(np.genfromtxt(output_filename)).reshape(-1)
    if len(coords_z) != len(full_points) or not np.all(np.isfinite(coords_z)):
        raise RuntimeError("COORDS_Z.dat has an invalid size or non-finite values.")
    if float(np.ptp(coords_z)) <= 0.0:
        raise RuntimeError("COORDS_Z.dat has a zero coordinate range.")
    apex_error = float(abs(coords_z[int(apex_full_id)]))
    base_error = float(np.max(np.abs(coords_z[combined_base_ids] - 1.0)))
    if apex_error > 1e-6 or base_error > 1e-6:
        raise RuntimeError(
            f"COORDS_Z does not enforce apex=0 and opening rims=1: "
            f"apex error {apex_error:.6g}, base error {base_error:.6g}."
        )
    if float(np.min(coords_z)) < -1e-6 or float(np.max(coords_z)) > 1.0 + 1e-6:
        raise RuntimeError("COORDS_Z values are outside the relative-distance range [0, 1].")

    output_basename = str(uvc_root / "COORDS_Z")
    if not meshtool.collect_nodal(
        meshtoolloc, meshname, "carp_txt", output_basename, "", output_filename
    ):
        raise RuntimeError("meshtool failed to create the COORDS_Z visualization files.")

    volume.point_data["COORDS_Z"] = coords_z
    volume.save(uvc_root / "COORDS_Z.vtk", binary=True)

    return output_filename, {
        "method": "single_opening_rims_to_apex_base_distancefield",
        "volume_vtk": str(Path(volume_vtk).resolve()),
        "mesh_point_count": int(len(full_points)),
        "median_tetra_edge_length_mm": edge_scale,
        "LV_annulus": lv_loop_info,
        "RV_annulus": rv_loop_info,
        "LV_base_mapping": lv_base_map,
        "LV_full_mapping": lv_full_map,
        "RV_full_mapping": rv_full_map,
        "combined_base_node_count": int(len(combined_base_ids)),
        "apex_LV_node_id": apex_lv_id,
        "apex_full_mesh_node_id": int(apex_full_id),
        "apex_coordinate": [float(value) for value in apex_coordinate],
        "apex_mapping_distance_mm": float(apex_distance),
        "apex_zero_max_abs_error": apex_error,
        "base_one_max_abs_error": base_error,
        "COORDS_Z_min": float(np.min(coords_z)),
        "COORDS_Z_max": float(np.max(coords_z)),
    }


def _merge_openbase_surface_vtks(input_paths, output_path):
    """Merge triangle VTK surfaces while preserving geometry and removing duplicates."""
    merged = None
    for input_path in input_paths:
        part = pv.read(input_path)
        if isinstance(part, pv.UnstructuredGrid):
            part = part.extract_surface()
        part = part.triangulate().clean(tolerance=0.0)
        if part.n_points == 0 or part.n_cells == 0:
            raise RuntimeError(f"Empty PHO boundary surface: {input_path}")
        merged = part if merged is None else merged.merge(part, merge_points=True)
    merged = merged.triangulate().clean(tolerance=0.0)
    if merged.n_points == 0 or merged.n_cells == 0:
        raise RuntimeError(f"Merged PHO boundary surface is empty: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.save(output_path, binary=False)
    return output_path


def _extract_openbase_boundary_on_submesh(
    coarse_vtk, fine_vtk, output_prefix, max_distance_mm,
):
    """Use the existing coarse-to-fine surface extraction for one boundary."""
    if not Coord_UVC_function.extract_fine_by_distance(
        coarse_vtk,
        fine_vtk,
        output_prefix,
        0.0,
        max_distance_mm,
        90.0,
    ):
        raise RuntimeError(f"Failed to select fine-mesh points near {coarse_vtk}.")
    points_path = output_prefix + ".pts"
    if not os.path.isfile(points_path) or os.path.getsize(points_path) == 0:
        raise RuntimeError(f"No fine-mesh points were selected for {coarse_vtk}.")
    result = Coord_UVC_function.extract_surface_by_pts(
        fine_vtk, points_path, output_prefix, 1e-3, False
    )
    if not result:
        raise RuntimeError(f"Failed to reconstruct CARP faces for {coarse_vtk}.")
    surface = meshIO.read_surf(basename=output_prefix)
    if len(surface) == 0:
        raise RuntimeError(f"No CARP triangles were generated for {coarse_vtk}.")
    return output_prefix


def _remap_surface_to_full_mesh(local_mesh, local_surface, full_mesh, output_surface):
    """Remap a CARP surface from a tag submesh to full-mesh node numbering."""
    local_points = np.asarray(meshIO.read_pts(basename=local_mesh), dtype=float)
    full_points = np.asarray(meshIO.read_pts(basename=full_mesh), dtype=float)
    local_surf = meshIO.read_surf(basename=local_surface)
    triangles = np.asarray(local_surf.iloc[:, 1:4], dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError(f"Invalid CARP surface topology: {local_surface}")
    if np.any(triangles < 0) or np.any(triangles >= len(local_points)):
        raise RuntimeError(f"Surface indices are outside the local mesh: {local_surface}")

    local_triangle_points = local_points[triangles]
    distances, full_ids = cKDTree(full_points).query(
        local_triangle_points.reshape(-1, 3), k=1
    )
    distances = np.asarray(distances, dtype=float).reshape(-1, 3)
    full_ids = np.asarray(full_ids, dtype=np.int64).reshape(-1, 3)
    if not np.all(np.isfinite(distances)) or float(np.max(distances)) > 1e-3:
        raise RuntimeError(
            f"Local-to-full PHO surface mapping exceeds 1e-3 mm: "
            f"{float(np.max(distances)):.6g} mm for {local_surface}."
        )
    mapped = full_ids
    valid = (
        (mapped[:, 0] != mapped[:, 1])
        & (mapped[:, 0] != mapped[:, 2])
        & (mapped[:, 1] != mapped[:, 2])
    )
    mapped = mapped[valid]
    if not len(mapped):
        raise RuntimeError(f"All mapped PHO triangles became degenerate: {local_surface}")
    _, unique_rows = np.unique(np.sort(mapped, axis=1), axis=0, return_index=True)
    mapped = mapped[np.sort(unique_rows)]

    output_path = Path(output_surface)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshIO.write_surf(
        surfFilename=str(output_path),
        surf=pd.DataFrame(mapped, columns=(1, 2, 3)),
        shapes="Tr",
    )
    meshIO.write_vtx_File(
        vtxFilename=str(output_path) + ".surf",
        vtx=pd.Series(np.unique(mapped), dtype=np.int64),
    )
    return {
        "triangle_count": int(len(mapped)),
        "node_count": int(len(np.unique(mapped))),
        "max_mapping_distance_mm": float(np.max(distances)),
        "source": str(local_surface),
        "output": str(output_path),
    }


def transmural_openbase(
    folder,
    meshtoolloc,
    meshname,
    volume_vtk,
    surface_paths,
    MPIEXEC,
    OPTS,
    CARP,
    parfile,
    IGBEXTRACT,
    UVCfolder,
):
    """Solve PHO with mapped LVepi and extracted junction=1, protecting LVendo=0."""
    from .pho_boundaries import build_pho_boundaries
    folder_path = Path(folder).resolve()
    work_root = folder_path / "PHO_work"
    surface_root = work_root / "surfaces"
    work_root.mkdir(parents=True, exist_ok=True)
    surface_root.mkdir(parents=True, exist_ok=True)
    uvc_root = Path(UVCfolder).resolve()
    uvc_root.mkdir(parents=True, exist_ok=True)
    os.chdir(folder_path)

    full_mesh = folder_path / meshname
    full_mesh_vtk = Path(volume_vtk).resolve()
    full_points = np.asarray(meshIO.read_pts(basename=str(full_mesh)), dtype=float)
    volume = pv.read(full_mesh_vtk)
    edge_scale = _tetra_edge_scale(volume)
    zero_nodes, outer_nodes, boundary_details = build_pho_boundaries(
        volume, full_points, surface_paths, work_root, edge_scale,
    )
    zero_boundary = str(surface_root / "endo_and_annuli_zero")
    outer_boundary = str(surface_root / "combined_epicardium_and_LVepi_one")
    _write_carp_node_surface(zero_boundary, zero_nodes)
    _write_carp_node_surface(outer_boundary, outer_nodes)

    # A single spatially varying Dirichlet electrode avoids interactions between
    # separate ground/clamp electrodes in the legacy openCARP RHS assembly.
    dirichlet_file = surface_root / "pho_dirichlet.vtx"
    boundary_nodes = np.concatenate((zero_nodes, outer_nodes))
    boundary_values = np.concatenate((np.zeros(len(zero_nodes)), np.ones(len(outer_nodes))))
    with dirichlet_file.open("w") as stream:
        stream.write(f"{len(boundary_nodes)}\nextra\n")
        np.savetxt(stream, np.column_stack((boundary_nodes, boundary_values)), fmt=["%d", "%.17g"])

    sim_id = str(work_root / "transmural_results")
    solver_boundary = os.path.relpath(dirichlet_file.with_suffix(""), folder_path)
    args = (
        "-cg_tol_ellip 1e-12 "
        "-gregion[0].num_IDs 2 -gregion[0].ID[0] 0 -gregion[0].ID[1] 1 "
        "-gregion[0].g_in 1 -gregion[0].g_en 1 "
        "-num_stim 1 -stimulus[0].stimtype 2 -stimulus[0].strength 1 -stimulus[0].duration 1 "
        "-stimulus[0].vtx_fcn 1 -stimulus[0].vtx_file " + solver_boundary + " "
    )
    if not carpfunc.Laplace_solver(
        MPIEXEC, OPTS, CARP, parfile, str(full_mesh), sim_id, args
    ):
        raise RuntimeError("openCARP failed while solving the open-base PHO field.")
    pho_values = np.atleast_1d(carpfunc.igb_todata(IGBEXTRACT, sim_id)).reshape(-1)
    if len(pho_values) != len(full_points) or not np.all(np.isfinite(pho_values)):
        raise RuntimeError(
            f"PHO solver output has {len(pho_values)} values; expected "
            f"{len(full_points)} finite nodal values."
        )

    raw_error = max(float(np.max(np.abs(pho_values[outer_nodes] - 1.0))),
                    float(np.max(np.abs(pho_values[zero_nodes]))))
    if raw_error > 1e-6:
        raise RuntimeError(f"PHO solver did not enforce its boundary conditions: error {raw_error:.6g}.")
    solver_range = [float(np.min(pho_values)), float(np.max(pho_values))]
    # Finite-element solutions on this mesh can overshoot between prescribed
    # boundaries. Preserve the unmodified solve for inspection before exporting
    # the normalized coordinate used for fiber-angle interpolation.
    raw_values = pho_values.copy()
    raw_path = uvc_root / "PHO_raw.dat"
    np.savetxt(raw_path, raw_values, fmt="%.17g")
    clipped_node_count = int(np.count_nonzero((pho_values < 0.0) | (pho_values > 1.0)))
    clipped_ids = np.flatnonzero((pho_values < 0.0) | (pho_values > 1.0))
    clipped_cloud = pv.PolyData(volume.points[clipped_ids].copy())
    clipped_cloud.point_data["volumeNodeId"] = clipped_ids
    clipped_cloud.point_data["PHO_raw"] = raw_values[clipped_ids]
    clipped_cloud.point_data["PHO"] = np.clip(raw_values[clipped_ids], 0.0, 1.0)
    clipped_cloud.save(work_root / "PHO_clipped_nodes.vtk", binary=True)
    pho_values = np.clip(pho_values, 0.0, 1.0)
    pho_values[outer_nodes] = 1.0
    pho_values[zero_nodes] = 0.0
    surface_error = float(np.max(np.abs(pho_values[outer_nodes] - 1.0)))
    if surface_error > 1e-12:
        raise RuntimeError(
            f"The exported PHO outer boundary is not exactly 1: error {surface_error:.6g}."
        )

    pho_path = str(uvc_root / "PHO.dat")
    np.savetxt(pho_path, pho_values, fmt="%.17g")
    pho_basename = str(uvc_root / "PHO")
    if not meshtool.collect_nodal(
        meshtoolloc, str(full_mesh), "carp_txt", pho_basename, "", pho_path
    ):
        raise RuntimeError("meshtool failed to create the PHO visualization files.")

    volume.point_data["PHO"] = pho_values
    volume.point_data["PHO_raw"] = raw_values
    volume.save(uvc_root / "PHO.vtk", binary=True)

    expected = [
        uvc_root / name for name in (
            "PHO.case", "PHO.dat", "PHO_raw.dat", "PHO.vtk", "PHO.fiber.ens", "PHO.geo",
            "PHO.tags.ens", "PHO_PHO.ens"
        )
    ]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"PHO generation did not create valid outputs: {missing}")
    return pho_path, {
        "method": "mapped_LVepi_and_extracted_interface_one_LVendo_zero_laplace",
        "mesh_point_count": int(len(full_points)),
        "epicardial_node_count": int(len(outer_nodes)),
        "LVepi_node_count": boundary_details["LVepi_node_count"],
        "LVepi_internal_node_count": boundary_details["LVepi_mapping"]["internal_node_count"],
        "LVepi_mapped_face_count": boundary_details["LVepi_mapping"]["mapped_face_count"],
        "LVepi_interface_face_count": boundary_details["LVepi_mapping"]["interface_face_count"],
        "zero_boundary_node_count": int(len(zero_nodes)),
        "unified_outer_boundary_node_count": int(len(outer_nodes)),
        "unified_outer_boundary_max_abs_error": surface_error,
        "solver_boundary_max_abs_error_before_export": raw_error,
        "solver_PHO_range_before_export": solver_range,
        "raw_solution_dat": str(raw_path),
        "normalization": "clip_to_unit_interval_with_unmodified_raw_solution_preserved",
        "normalization_clipped_node_count": clipped_node_count,
        "normalization_clipped_node_fraction": float(clipped_node_count / len(raw_values)),
        "normalization_maximum_adjustment": float(np.max(np.abs(raw_values - pho_values))),
        "isotropic_conductivity_tags": [0, 1],
        "median_tetra_edge_length_mm": float(edge_scale),
        "boundary_files": {"zero": zero_boundary, "one": outer_boundary,
                           "nodal_dirichlet": str(dirichlet_file)},
        "boundary_details": boundary_details,
        "PHO_range": [float(np.min(pho_values)), float(np.max(pho_values))],
        "outputs": [str(path) for path in expected],
    }


def transmural(folder, meshtoolloc, mesh, pts5810, input_dir, meshnames, surfloc, MPIEXEC, OPTS, CARP, parfile, SIMID, IGBEXTRACT, UVCfolder):
    #
    os.chdir(folder)

    et_vertex_start_end = np.array(
        [[0, 1499], [1500, 2164], [2165, 3223], [3224, 5581],
         [5582, 5630], [5631, 5655], [5656, 5696], [5697, 5729],
         [5730, 5809]])
    '''Class constant, surface index limits for vertices `et_pos`. 
        Surfaces are defined in the following order:

        LV_ENDOCARDIAL = 0 
        RV_SEPTUM = 1
        RV_FREEWALL = 2
        EPICARDIAL =3
        MITRAL_VALVE =4
        AORTA_VALVE = 5
        TRICUSPID_VALVE = 6
        PULMONARY_VALVE = 7
        RV_INSERT = 8
        '''
    # LVendo = et_vertex_start_end[0][0]
    # LVendocoord = str(pts5810.iloc[LVendo, 0]) + "," + str(pts5810.iloc[LVendo, 1]) + "," + str(pts5810.iloc[LVendo, 2])
    # RVendo = et_vertex_start_end[2][0]
    # RVendocoord = str(pts5810.iloc[RVendo, 0]) + "," + str(pts5810.iloc[RVendo, 1]) + "," + str(pts5810.iloc[RVendo, 2])
    # RVseptum = et_vertex_start_end[1][0]
    # RVseptumcoord = str(pts5810.iloc[RVseptum, 0]) + "," + str(pts5810.iloc[RVseptum, 1]) + "," + str(pts5810.iloc[RVseptum, 2])
    # epi = et_vertex_start_end[3][0]
    # epicoord = str(pts5810.iloc[epi, 0]) + "," + str(pts5810.iloc[epi, 1]) + "," + str(pts5810.iloc[epi, 2])


    # mesh_names: ['Mesh_LV_endo_timeframe001', 'Mesh_RV_FW_timeframe001', 'Mesh_RV_septum_timeframe001',
    #              'Mesh_aorta_valve_timeframe001', 'Mesh_epi_timeframe001', 'Mesh_mitral_valve_timeframe001',
    #              'Mesh_pulmonary_valve_timeframe001', 'Mesh_tricuspid_valve_timeframe001']
    Mesh_LV_endo_surf = meshnames[0]
    Mesh_RV_septum_surf = meshnames[2]
    Mesh_RV_endo_surf = meshnames[1]
    Mesh_epi_surf = meshnames[4]

    LV_endo_coarse_file = input_dir+f"/{Mesh_LV_endo_surf}" + ".vtk"
    RV_septum_coarse_file = input_dir+f"/{Mesh_RV_septum_surf}" + ".vtk"
    RV_endo_coarse_file = input_dir+f"/{Mesh_RV_endo_surf}" + ".vtk"
    epi_coarse_file = input_dir+f"/{Mesh_epi_surf}" + ".vtk"
    fine_file = mesh + ".vtk"
    output_prefix = surfloc + "/biv_i"
    # if not os.path.exists(output_prefix):
    #     os.mkdir(output_prefix)
    Coord_UVC_function.extract_fine_by_distance(LV_endo_coarse_file,fine_file,output_prefix + "/LV_endo",0.0, 0.3, 90)
    Coord_UVC_function.extract_fine_by_distance(RV_septum_coarse_file,fine_file,output_prefix + "/RV_septum",0.0, 0.3, 90)
    Coord_UVC_function.extract_fine_by_distance(RV_endo_coarse_file,fine_file,output_prefix + "/RV_endo",0.0, 0.3, 90)
    Coord_UVC_function.extract_fine_by_distance(epi_coarse_file,fine_file,output_prefix + "/epi",0.0, 0.3, 90)

    Coord_UVC_function.extract_surface_by_pts(fine_file, output_prefix + "/LV_endo" + ".pts", output_prefix + "/LV_endo", 1e-3, False)
    Coord_UVC_function.extract_surface_by_pts(fine_file, output_prefix + "/RV_septum" + ".pts", output_prefix + "/RV_septum", 1e-3, False)
    Coord_UVC_function.extract_surface_by_pts(fine_file, output_prefix + "/RV_endo" + ".pts", output_prefix + "/RV_endo", 1e-3, False)
    Coord_UVC_function.extract_surface_by_pts(fine_file, output_prefix + "/epi" + ".pts", output_prefix + "/epi", 1e-3, False)

    # surf_iloc = surfloc + "/Surface_pts/"
    surf_iloc = surfloc + "/biv_i/"
    # op = surf_iloc + "myoall-" + surf_iloc + "intersection"
    op = surf_iloc + "myoall" + "-" + surf_iloc + "base_i"
    #待修改：intersection实际就是coord z计算中生成的base

    LV_endo = surf_iloc + 'LV_endo'
    RV_septum = surf_iloc + 'RV_septum'
    RV_endo = surf_iloc + 'RV_endo'
    epi = surf_iloc + 'epi'

    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "LV_endo", op, "-coord=" + LVendocoord):
    #     return None
    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "RV_septum", surf_iloc + 'myoall_insertion',"-coord=" + RVseptumcoord, "-edge=10"):
    #     return None
    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "RV_endo1", op, "-coord=" + RVendocoord,"-edge=10"):
    #     return None
    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "RV_endo",surf_iloc + "RV_endo1-" + surf_iloc + 'RV_septum', "-coord=" + RVendocoord):
    #     return None
    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "epi", op, "-coord=" + epicoord):
    #     return None

    args = "-stimulus[0].vtx_file " + LV_endo + ".surf "
    args = args + " -stimulus[1].vtx_file " + RV_endo + ".surf "
    args = args + " -stimulus[2].vtx_file " + epi + ".surf "
    args = args + " -stimulus[3].vtx_file " + RV_septum + ".surf "
    # LV,RV_septum, RV_endo and epi Laplace solver
    if not carpfunc.Laplace_solver(MPIEXEC, OPTS, CARP, parfile, mesh + "_i", SIMID, args):
        return None

    gradMag = carpfunc.igb_todata(IGBEXTRACT, SIMID)
    file = UVCfolder + "/PHO.dat"
    meshIO.write_data(dataFilename=file, data=gradMag)
    # print (max(gradMag))
    # print (min(gradMag))
    if not meshtool.collect_nodal(meshtoolloc, mesh, "carp_txt", UVCfolder + "/PHO", '', UVCfolder + "/PHO.dat"):
        return None
    return file

def find_two_RV_insertion_pts_openbase(
    volume_vtk,
    coords_z_path,
    lv_annulus_vtk,
    rv_annulus_vtk,
    rv_epi_interface_vtk,
    ant_output_vtk,
    inf_output_vtk,
    coord_z_tolerance=0.01,
):
    """Find ANT/INF on the selected tag-interface curve using opening-rim centers."""

    def write_landmark(
        path,
        point,
        nearest_node_id,
        coords_z,
        mapping_distance,
        angle_degrees,
    ):
        landmark = pv.PolyData()
        landmark.points = np.asarray(point, dtype=float).reshape(1, 3)
        landmark.verts = np.array([1, 0], dtype=np.int64)
        landmark.point_data["nearest_volume_node_id"] = np.array(
            [nearest_node_id], dtype=np.int64
        )
        landmark.point_data["COORDS_Z"] = np.array([coords_z], dtype=float)
        landmark.point_data["mapping_distance_mm"] = np.array(
            [mapping_distance], dtype=float
        )
        landmark.point_data["projected_angle_deg"] = np.array(
            [angle_degrees], dtype=float
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        landmark.save(path, binary=False)

    volume = pv.read(volume_vtk)
    if not isinstance(volume, pv.UnstructuredGrid):
        raise RuntimeError(f"Input is not an unstructured volume mesh: {volume_vtk}")
    volume_points = np.asarray(volume.points, dtype=float)
    coords_z = np.atleast_1d(np.genfromtxt(coords_z_path)).reshape(-1)
    if len(coords_z) != len(volume_points) or not np.all(np.isfinite(coords_z)):
        raise RuntimeError("COORDS_Z does not match the volume-mesh points.")

    _, lv_rim = _opening_rim_loop(lv_annulus_vtk)
    _, rv_rim = _opening_rim_loop(rv_annulus_vtk)
    lv_centroid = np.asarray(lv_rim["length_weighted_centroid"])
    rv_centroid = np.asarray(rv_rim["length_weighted_centroid"])
    interface_points = np.asarray(pv.read(rv_epi_interface_vtk).points, dtype=float)
    if len(interface_points) < 3 or not np.all(np.isfinite(interface_points)):
        raise RuntimeError("Selected tag-interface curve has fewer than three valid points.")

    edge_scale = _tetra_edge_scale(volume)
    mapping_distances, mapped_ids = cKDTree(volume_points).query(interface_points, k=1)
    mapping_tolerance = max(1e-4, 2.0 * edge_scale)
    if not np.all(np.isfinite(mapping_distances)) or float(np.max(mapping_distances)) > mapping_tolerance:
        raise RuntimeError(
            f"Interface mapping distance {np.max(mapping_distances):.6g} mm exceeds "
            f"the tolerance {mapping_tolerance:.6g} mm."
        )
    mapped_ids = np.asarray(mapped_ids, dtype=np.int64)
    interface_coords_z = coords_z[mapped_ids]

    points_xy = interface_points[:, :2]
    vectors_lv = lv_centroid[:2] - points_xy
    vectors_rv = rv_centroid[:2] - points_xy
    denominator = np.linalg.norm(vectors_lv, axis=1) * np.linalg.norm(vectors_rv, axis=1)
    valid_vectors = np.isfinite(denominator) & (denominator > np.finfo(float).eps)
    cosines = np.full(len(interface_points), np.nan, dtype=float)
    cosines[valid_vectors] = np.einsum(
        "ij,ij->i", vectors_lv[valid_vectors], vectors_rv[valid_vectors]
    ) / denominator[valid_vectors]
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))

    if rv_centroid[0] < lv_centroid[0]:
        ant_direction = "negative_y"
        inf_direction = "positive_y"
        ant_side = interface_points[:, 1] < min(rv_centroid[1], lv_centroid[1])
        inf_side = interface_points[:, 1] > max(rv_centroid[1], lv_centroid[1])
    else:
        ant_direction = "positive_y"
        inf_direction = "negative_y"
        ant_side = interface_points[:, 1] > max(rv_centroid[1], lv_centroid[1])
        inf_side = interface_points[:, 1] < min(rv_centroid[1], lv_centroid[1])

    baseward_cutoff = float(np.quantile(interface_coords_z, 0.70))
    ant_candidates = np.flatnonzero(
        ant_side & (interface_coords_z >= baseward_cutoff) & np.isfinite(angles)
    )
    if not len(ant_candidates):
        raise RuntimeError(
            f"No interface point satisfies the {ant_direction} and base-near conditions."
        )

    ant_minimum_angle = float(np.min(angles[ant_candidates]))
    ant_tied = ant_candidates[
        np.isclose(angles[ant_candidates], ant_minimum_angle, rtol=1e-12, atol=1e-12)
    ]
    ant_angle_tie_count = int(len(ant_tied))
    ant_maximum_z = float(np.max(interface_coords_z[ant_tied]))
    ant_tied = ant_tied[
        np.isclose(interface_coords_z[ant_tied], ant_maximum_z, rtol=1e-12, atol=1e-12)
    ]
    ant_interface_id = int(np.min(ant_tied))
    ant_node_id = int(mapped_ids[ant_interface_id])
    ant_point = interface_points[ant_interface_id]
    ant_z = float(interface_coords_z[ant_interface_id])

    same_level = np.abs(interface_coords_z - ant_z) <= coord_z_tolerance
    inf_candidates = np.flatnonzero(
        inf_side & same_level & np.isfinite(angles)
    )
    inf_candidates = inf_candidates[inf_candidates != ant_interface_id]
    if not len(inf_candidates):
        raise RuntimeError(
            f"No interface point satisfies the {inf_direction} and "
            f"COORDS_Z +/- {coord_z_tolerance:g} conditions."
        )

    inf_minimum_angle = float(np.min(angles[inf_candidates]))
    inf_tied = inf_candidates[
        np.isclose(angles[inf_candidates], inf_minimum_angle, rtol=1e-12, atol=1e-12)
    ]
    inf_angle_tie_count = int(len(inf_tied))
    z_differences = np.abs(interface_coords_z[inf_tied] - ant_z)
    minimum_z_difference = float(np.min(z_differences))
    inf_tied = inf_tied[
        np.isclose(z_differences, minimum_z_difference, rtol=1e-12, atol=1e-12)
    ]
    inf_maximum_z = float(np.max(interface_coords_z[inf_tied]))
    inf_tied = inf_tied[
        np.isclose(interface_coords_z[inf_tied], inf_maximum_z, rtol=1e-12, atol=1e-12)
    ]
    inf_interface_id = int(np.min(inf_tied))
    inf_node_id = int(mapped_ids[inf_interface_id])
    inf_point = interface_points[inf_interface_id]
    inf_z = float(interface_coords_z[inf_interface_id])

    ant_angle_degrees = float(np.degrees(ant_minimum_angle))
    inf_angle_degrees = float(np.degrees(inf_minimum_angle))
    write_landmark(
        ant_output_vtk,
        ant_point,
        ant_node_id,
        ant_z,
        float(mapping_distances[ant_interface_id]),
        ant_angle_degrees,
    )
    write_landmark(
        inf_output_vtk,
        inf_point,
        inf_node_id,
        inf_z,
        float(mapping_distances[inf_interface_id]),
        inf_angle_degrees,
    )
    return {
        "centroid_method": "opening_rim_segment_length_weighted_midpoints",
        "lv_centroid": lv_centroid.tolist(),
        "rv_centroid": rv_centroid.tolist(),
        "ant_direction": ant_direction,
        "inf_direction": inf_direction,
        "baseward_coords_z_cutoff": baseward_cutoff,
        "ant_interface_point_id": ant_interface_id,
        "ant_nearest_volume_node_id": ant_node_id,
        "ant_point": ant_point.tolist(),
        "ant_coords_z": ant_z,
        "ant_angle_degrees": ant_angle_degrees,
        "ant_angle_tie_count": ant_angle_tie_count,
        "ant_mapping_distance_mm": float(mapping_distances[ant_interface_id]),
        "inf_interface_point_id": inf_interface_id,
        "inf_nearest_volume_node_id": inf_node_id,
        "inf_point": inf_point.tolist(),
        "inf_coords_z": inf_z,
        "inf_angle_degrees": inf_angle_degrees,
        "inf_angle_tie_count": inf_angle_tie_count,
        "inf_mapping_distance_mm": float(mapping_distances[inf_interface_id]),
        "coords_z_difference": abs(inf_z - ant_z),
    }


def find_two_RV_insertion_pts(folder, meshtoolloc, meshname, input_dir, meshnames, surfloc,  coordzdir):
    # this fuc is to find two pts in RV_insertion by
    # (1)finding the "anterior point" which is the closest point to the pulmonary valve
    # (2)finding the "interior point" which is the furtherest point to ANT in RV insertion surf and also have same z
    # ANT and INF is wrong. it needs to switch for later steps!!!
    os.chdir(folder)
    LVmyosurf = surfloc + "/LVmyo"
    if not os.path.exists(LVmyosurf):
        os.mkdir(LVmyosurf)

    LV = surfloc + '/LV'
    if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/LV", "1"):
        return False

    RVinsertion = surfloc + '/RVinsertion'
    if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/RVinsertion", "0:1"):
        return False
    if not meshtool.extract_surf(meshtoolloc, meshname + '_i', surfloc + "/biv_i/RVinsertion", "0:1"): #这两处无valve时是一样的
        return False

    # if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/RVinsertion.surf",surfloc+"/LVmyo"):
    #     return False
    # if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/RVinsertion.surf.vtx",surfloc+"/LVmyo"):
    #     return False

    LVvtx = meshIO.read_vtx_File(surfloc + "/LV.surf.vtx")
    LVsurf = meshIO.read_surf(surfloc + "/LV")
    pts = meshIO.read_pts(meshname)
    coord_LV = np.zeros((len(LVvtx), 3))
    for i in range(len(LVvtx)):
        coord_LV[i] = pts.loc[int(LVvtx[i])]
    #######################
    # find the centre point of Pulmonary valve

    # if not meshtool.extract_mesh(meshtoolloc, meshname, "7", "pulmonarymesh"):
    #     return False
    #8待修改:此处先用pulmonarymesh_pts_read.py读取surface mesh的点坐标

    # pulmonary_pts = meshIO.read_pts(folder + "/pulmonarymesh")
    #9待修改:pulmonary_pts应该改为读取上述pulmonarymesh_pts_read.py的输出
    Mesh_pulmonary_valve_surf = meshnames[6]
    Mesh_pulmonary_valve_surf_pts = input_dir + f"/{Mesh_pulmonary_valve_surf}" + ".vtk"
    pulmonary_pts = Coord_UVC_function.read_surface_from_vtk(Mesh_pulmonary_valve_surf_pts)

    pulmonary_centre = np.array(np.mean(pulmonary_pts, axis=0))
    ####################

    # 1. 提取 RV insertion 顶点坐标（与 LV 坐标同属 pts 空间）
    RVinsertionvtx = meshIO.read_vtx_File(surfloc + "/RVinsertion.surf.vtx")
    coord_RVins = pts.loc[RVinsertionvtx.astype(int)].values

    # 2. 使用 KDTree 高效计算 LV 表面各点到 RV insertion 的最近距离
    tree_rv = cKDTree(coord_RVins)
    dist_LV_to_RVins, _ = tree_rv.query(coord_LV)

    # 3. 自适应阈值：仅保留距离 RV insertion 最近的 5% LV 顶点作为候选
    # 百分比策略对网格尺度/分辨率不敏感，比固定绝对值更鲁棒
    tol_rv = np.percentile(dist_LV_to_RVins, 5)

    # 防呆回退：若阈值过严导致无候选点，自动放宽至 10%
    mask_near_RV = dist_LV_to_RVins <= tol_rv
    if not np.any(mask_near_RV):
        print("[Warning] RV proximity filter too strict. Relaxing to top 10%.")
        tol_rv = np.percentile(dist_LV_to_RVins, 10)
        mask_near_RV = dist_LV_to_RVins <= tol_rv

    # 4. 【核心修正】仅对候选点集合计算到肺动脉中心的距离
    coord_candidates = coord_LV[mask_near_RV]  # 形状: (N_candidates, 3)
    dist_candidates_to_pul = np.linalg.norm(coord_candidates - pulmonary_centre, axis=1)  # 形状: (N_candidates,)

    # 5. 在候选集中找距离肺动脉最近的点
    best_idx_in_candidates = np.argmin(dist_candidates_to_pul)

    # ⚠️ 关键：将候选集内的局部索引 映射回 原始 LV 顶点索引
    original_indices = np.where(mask_near_RV)[0]  # 所有候选点在 coord_LV 中的真实位置
    best_idx_original = original_indices[best_idx_in_candidates]

    NodeANT = int(LVvtx[best_idx_original])
    print(f"[Info] ANT selected from {len(coord_candidates)} LV candidates near RV insertion. "
          f"Min dist to Pul: {dist_candidates_to_pul[best_idx_in_candidates]:.4f}")

    # distances = np.linalg.norm(coord_LV - pulmonary_centre, axis=1)
    # NodeANT = int(LVvtx[np.argmin(distances)])
    ######################################
    # print this out for visualizing but not necesseary

    surfnew = LVsurf.loc[(LVsurf[1] == NodeANT) | (LVsurf[2] == NodeANT) | (LVsurf[3] == NodeANT), :]
    surf_vtx = np.unique(surfnew.loc[:, 1:3])
    Filename = surfloc + "/Node_ANT"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
    ######################################

    #################################
    # Node_INF is identified by
    # 1) find all surfaces in RV insertion having similar coordz as Node_ANT
    # 2) In those surfaces, find the furtherest point from the Node_ANT
    # note coordz.dat is for biv_i mesh, so suface need to be mapped

    # map to biv_i mesh
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/Node_ANT.vtx", surfloc + "/biv_i"):
        return False
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/Node_ANT.surf", surfloc + "/biv_i"):
        return False

    # RVinsertion = surfloc + '/RVinsertion'
    # if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/RVinsertion", "0:1"):
    #     return False
    #10待修改:与本function前面的RVinsertion赋值冗余,可做删除

    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/RVinsertion.surf", surfloc + "/biv_i"):
        return False
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/RVinsertion.surf.vtx", surfloc + "/biv_i"):
        return False

    coordz = np.genfromtxt(coordzdir)

    ANTvtx = meshIO.read_vtx_File(surfloc + "/biv_i/Node_ANT.vtx")
    ANTsurf = meshIO.read_surf(surfloc + "/biv_i/Node_ANT")

    ANT_coordz = coordz[int(ANTvtx[0])]

    RVinsertionvtx = meshIO.read_vtx_File(surfloc + "/biv_i/RVinsertion.surf.vtx")
    RVinsertion_coordz = np.zeros(len(RVinsertionvtx))
    vtx_similarz = np.empty((0, 1), int)
    print("RV insertion:", len(RVinsertionvtx))
    for i in range(len(RVinsertionvtx)):
        if coordz[int(RVinsertionvtx[i])] < (ANT_coordz + 0.005) and coordz[int(RVinsertionvtx[i])] > (ANT_coordz - 0.005): #Z值与ANT相差 <0.005 的顶点索引数组
        # if coordz[int(RVinsertionvtx[i])] < (ANT_coordz + 0.1) and coordz[int(RVinsertionvtx[i])] > (ANT_coordz - 0.1): #Z值与ANT相差 <0.005 的顶点索引数组
            vtx_similarz = np.append(vtx_similarz, [RVinsertionvtx[i]])
    print("line 359")
    print('vtx_similarz: {}'.format(vtx_similarz))
    RVinsertionsurf = meshIO.read_surf(surfloc + "/biv_i/RVinsertion")
    pts = meshIO.read_pts(meshname + "_i")
    coordz_ANT = np.array(pts.loc[ANTvtx[0]])
    coord_similarz = np.zeros((len(vtx_similarz), 3))
    for i in range(len(vtx_similarz)):
        coord_similarz[i] = pts.loc[int(vtx_similarz[i])]

    distances = np.linalg.norm(coord_similarz - coordz_ANT, axis=1)
    if len(distances) == 0:
        return False
    NodeINF = int(vtx_similarz[np.argmax(distances)]) #定位INF点（同Z平面内距ANT最远的点）
    print('Node INF: {}'.format(NodeINF))

    ######################################
    # print this out for visualizing but not necesseary
    surfnew = RVinsertionsurf.loc[(RVinsertionsurf[1] == NodeINF) | (RVinsertionsurf[2] == NodeINF) | (RVinsertionsurf[3] == NodeINF), :]
    surf_vtx = np.unique(surfnew.loc[:, 1:3])
    Filename = surfloc + "/biv_i/Node_INF"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
    ######################################

    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/biv_i/Node_INF.surf", surfloc, "s2m"):
        return False
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/biv_i/Node_INF.surf.vtx", surfloc, "s2m"):
        return False

    return True


#########################################
# Function to preprocessing data for rotational fuc for LV
#########################################
def extract_surf_forLV(folder, meshtoolloc, meshname, input_dir, meshnames, surfloc):
    # extract surface: apex,base(mitral_intersect), Node_ANT, Node_INF
    os.chdir(folder)

    LVmyosurf = surfloc + "/LVmyo"
    if not os.path.exists(LVmyosurf):
        os.mkdir(LVmyosurf)

    if not meshtool.extract_mesh(meshtoolloc, meshname + "_i", "1", "LVmyo_i"):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/apex_i.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/apex_i.vtx", LVmyosurf):
        return None

    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/RVinsertion.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/RVinsertion.vtx", LVmyosurf):
        return None

    if not meshtool.extract_surf(meshtoolloc, "LVmyo_i", LVmyosurf + '/myoall_insertion',
                                 '1-' + LVmyosurf + '/RVinsertion'):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", LVmyosurf + '/myoall_insertion.surf', surfloc + "/biv_i",
                             mode='s2m'):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", LVmyosurf + '/myoall_insertion.vtx', surfloc + "/biv_i",
                             mode='s2m'):
        return None

    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/Node_ANT.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/Node_ANT.vtx", LVmyosurf):
        return None

    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/Node_INF.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/Node_INF.vtx", LVmyosurf):
        return None

    # map the base surf from whole mesh as the mitral valve intersection with the LVmyo
    if not meshtool.extract_mesh(meshtoolloc, meshname, "1", "LVmyo"):
        return None
    # mitral_intersect = surfloc + '/mitral_intersect'

    Mesh_mitral_valve_surf = meshnames[5]

    mitral_valve_files = [
        (input_dir + f"/{Mesh_mitral_valve_surf}" + ".pts",
         input_dir + f"/{Mesh_mitral_valve_surf}" + ".elem")
    ]

    if not Coord_UVC_function.find_intersection_base(mitral_valve_files, "LVmyo", surfloc + "/mitral_intersect"):
        return None

    # if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/mitral_intersect", "1:4"):
    #     return None
    if not meshtool.map_file(meshtoolloc, "LVmyo", surfloc + "/mitral_intersect.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo", surfloc + "/mitral_intersect.surf.vtx", LVmyosurf):
        return None
    #11修改:使用find_intersection_lv_base生成mitral_intersect.surf/.surf.vtx
    
    Node_ANT = meshIO.read_vtx_File(LVmyosurf + "/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        return None
    NodeANTvtx = int(Node_ANT[0])

    Node_INF = meshIO.read_vtx_File(LVmyosurf + "/Node_INF.vtx")
    if len(Node_INF) == 0:
        return None
    NodeINFvtx = int(Node_INF[0])
    return [LVmyosurf + "/mitral_intersect", LVmyosurf + "/apex_i.vtx", NodeINFvtx, NodeANTvtx]


#########################################
# Function to preprocessing data for rotational fuc for RV
#########################################
def extract_surf_forRV(folder, meshtoolloc, meshname, input_dir, meshnames, surfloc, coordzdir):
    os.chdir(folder)
    RVmyosurf = surfloc + "/RVmyo"
    if not os.path.exists(RVmyosurf):
        os.mkdir(RVmyosurf)

    if not meshtool.extract_mesh(meshtoolloc, meshname + "_i", "0", "RVmyo_i"):
        return None

    RV_endo = surfloc + '/RV_all'
    if not meshtool.extract_surf(meshtoolloc, meshname + "_i", surfloc + "/RV_all", "0"):
        return None

    coordz = np.genfromtxt(coordzdir)
    # Node_RVapex=np.argmin(coordz)

    RVvtx = meshIO.read_vtx_File(surfloc + "/RV_all.surf.vtx")
    RVsurf = meshIO.read_surf(surfloc + "/RV_all")

    coordz_RV = np.zeros((len(RVvtx), 1))
    for i in range(len(RVvtx)):
        coordz_RV[i] = coordz[int(RVvtx[i])]

    Node_RVapex = int(RVvtx[np.argmin(coordz_RV)])
    ######################################
    # print this out for visualizing but not necesseary
    surfnew = RVsurf.loc[(RVsurf[1] == Node_RVapex) | (RVsurf[2] == Node_RVapex) | (RVsurf[3] == Node_RVapex), :]
    surf_vtx = np.unique(surfnew.loc[:, 1:3])
    Filename = surfloc + "/Node_RV_apex"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)

    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/Node_RV_apex.surf", surfloc + "/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/Node_RV_apex.vtx", surfloc + "/RVmyo"):
        return None

    # tricuspid_intersect = surfloc + '/tricuspid_intersect'
    # if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/tricuspid_intersect", "0:6"):
    #     return None
    #11待修改,python find_intersection_tricuspid.py
    Mesh_tricuspid_valve_surf = meshnames[7]

    tricuspid_valve_files = [
        (input_dir + f"/{Mesh_tricuspid_valve_surf}" + ".pts",
         input_dir + f"/{Mesh_tricuspid_valve_surf}" + ".elem")
    ]

    if not Coord_UVC_function.find_intersection_base(tricuspid_valve_files, meshname, surfloc + "/tricuspid_intersect"):
        return None

    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/tricuspid_intersect.surf", surfloc + "/biv_i"):
        return None
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/tricuspid_intersect.surf.vtx", surfloc + "/biv_i"):
        return None
    # 12待修改,surfloc + "/tricuspid_intersect.vtx"

    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/tricuspid_intersect.surf", surfloc + "/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/tricuspid_intersect.surf.vtx", surfloc + "/RVmyo"):
        return None
    # 13待修改,surfloc + "/tricuspid_intersect.vtx"

    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/Node_INF.surf", surfloc + "/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/Node_INF.vtx", surfloc + "/RVmyo"):
        return None

    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/Node_ANT.surf", surfloc + "/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/Node_ANT.vtx", surfloc + "/RVmyo"):
        return None

    RVsurfdir = surfloc + "/RVmyo"
    Node_ANT = meshIO.read_vtx_File(RVsurfdir + "/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        ########################
        # find the closest point in RVmyo_i to ANT node in biv_i mesh
        ptsbiv = meshIO.read_pts(meshname + '_i')
        bivvtx = meshIO.read_vtx_File(surfloc + "/biv_i/Node_ANT.vtx")
        ptscoordANT = ptsbiv.loc[int(bivvtx[0])]

        if not meshtool.extract_surf(meshtoolloc, "RVmyo_i", surfloc + "/RVmyo/myoall", "0"):
            return None

        ptsRV = meshIO.read_pts('RVmyo_i')
        RVvtx = meshIO.read_vtx_File(surfloc + "/RVmyo/myoall.surf.vtx")
        RVsurf = meshIO.read_surf(surfloc + "/RVmyo/myoall")
        ptscoordRV = ptsRV.loc[RVvtx]

        distances = np.linalg.norm(ptscoordRV - ptscoordANT, axis=1)
        NodeANT = int(RVvtx[np.argmin(distances)])

        ######################################
        # print this out for visualizing but not necesseary
        surfnew = RVsurf.loc[(RVsurf[1] == NodeANT) | (RVsurf[2] == NodeANT) | (RVsurf[3] == NodeANT), :]
        surf_vtx = np.unique(surfnew.loc[:, 1:3])
        Filename = surfloc + "/RVmyo/Node_ANT"
        meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
        meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
        RVsurfdir = surfloc + "/RVmyo"
        Node_ANT = meshIO.read_vtx_File(RVsurfdir + "/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        return None
    NodeANTvtx = int(Node_ANT[0])

    Node_INF = meshIO.read_vtx_File(RVsurfdir + "/Node_INF.vtx")
    if len(Node_INF) == 0:
        return None
    NodeINFvtx = int(Node_INF[0])

    return [surfloc + "/RVmyo/tricuspid_intersect", surfloc + "/RVmyo/Node_RV_apex.vtx", NodeINFvtx, NodeANTvtx]


#########################################
# Function to compute rotationall UVC coordinate by Martin
#########################################
# PS: function(find_two_RV_insertion_pts) gives NodeINFvtx and NodeANTvtx by turn
# but in function(compute_uvc_rotational) read NodeINFvtx and NodeANTvtx represent RVinsertionNodeANT and RVinsertionNodeINF
# The index of RVinsertionNodeANT and RVinsertionNodeINF actually flipped 20260418
def compute_uvc_rotational(folder, meshtoolloc, meshname, base_surfFilename, apex_pointFilename, RVinsertionNodeANT,
                           RVinsertionNodeINF, phiname):
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apex_pointFilename)
    apex_vtx = int(apex_vtx[0])
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]

    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    centroids = meshIO.create_centroids(elems, pts)

    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    base_surf = meshIO.read_surf(base_surfFilename)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(), pts.loc[base_nodes][1].mean(), pts.loc[base_nodes][2].mean()]

    # Computes long axis between apex and CoM
    longAxis = (apex_coord - CoM_coord) / np.linalg.norm(apex_coord - CoM_coord)

    # Translates mesh to have x,y point aligned with apex
    pts_t = pts - apex_coord

    # Rotates mesh to align long axis with z [001] direction
    z_axis = [0, 0, 1]
    R = geometrical.computeRotationMatrix(z_axis, longAxis)
    pts_r = pd.DataFrame(np.transpose(np.matmul(R, np.transpose(pts_t))))

    #########################################
    # Computes phi based on relative angles
    #########################################
    # Defines 2 RV insertion sites and coords
    RVinsPtANT_coord = pts_r.iloc[RVinsertionNodeANT]
    RVinsPtINF_coord = pts_r.iloc[RVinsertionNodeINF]

    # Picks out the xy coord (projecting it into xy plane)
    zero_phase_RVANT = [RVinsPtANT_coord[0], RVinsPtANT_coord[1]] / np.linalg.norm(
        [RVinsPtANT_coord[0], RVinsPtANT_coord[1]])
    zero_phase_RVINF = [RVinsPtINF_coord[0], RVinsPtINF_coord[1]] / np.linalg.norm(
        [RVinsPtINF_coord[0], RVinsPtINF_coord[1]])

    # find singularity
    sing_phase = -0.5 * (zero_phase_RVANT + zero_phase_RVINF)
    sing_phase = sing_phase / np.linalg.norm(sing_phase)

    # Projects all points in mesh into xy plane
    pts_proj = pts_r.loc[:, 0:1]
    mag_pts = np.linalg.norm(pts_proj, axis=1)
    safe_mag_pts = np.where(mag_pts > np.finfo(float).eps, mag_pts, 1.0)

    # computes the angles between every point in the mesh and...
    # ... anterior RV insertion site
    cosThetas_RVANT = np.clip(np.dot(pts_proj, zero_phase_RVANT) / safe_mag_pts, -1.0, 1.0)
    phi_RVANT = np.arccos(cosThetas_RVANT)
    # ... Interior RV insertion site
    cosThetas_RVINF = np.clip(np.dot(pts_proj, zero_phase_RVINF) / safe_mag_pts, -1.0, 1.0)
    phi_RVINF = np.arccos(cosThetas_RVINF)
    # ... centre of free wall
    cosThetas_SING = np.clip(np.dot(pts_proj, sing_phase) / safe_mag_pts, -1.0, 1.0)
    phi_SING = np.arccos(cosThetas_SING)

    # Defines array to store phi coords
    phi_coord = np.zeros(len(pts))

    # defines anterior nodes and interior nodes
    ant_LV_nodes = np.where(phi_RVANT < phi_RVINF)
    inf_LV_nodes = np.where(phi_RVANT >= phi_RVINF)

    phi_coord[ant_LV_nodes] = -np.pi / 2 - (np.pi / 2) * phi_RVANT[ant_LV_nodes] / (
                phi_RVANT[ant_LV_nodes] + phi_SING[ant_LV_nodes])
    phi_coord[inf_LV_nodes] = np.pi / 2 + (np.pi / 2) * phi_RVINF[inf_LV_nodes] / (
                phi_RVINF[inf_LV_nodes] + phi_SING[inf_LV_nodes])

    #############################
    # Computes phi for septum
    #############################
    septal_angle = np.arccos(np.dot(zero_phase_RVANT, zero_phase_RVINF))
    septal_nodes = np.intersect1d(
        np.intersect1d(np.where(phi_RVANT < septal_angle), np.where(phi_RVINF < septal_angle)),
        np.where(phi_SING > np.pi / 2))
    phi_coord[septal_nodes] = np.pi / 2 - np.pi * (
                phi_RVINF[septal_nodes] / (phi_RVANT[septal_nodes] + phi_RVINF[septal_nodes]))
    meshIO.write_data(dataFilename=phiname, data=phi_coord)

    if not meshtool.collect_nodal(meshtoolloc, meshname, "carp_txt", folder + "/Phi_LV", '', folder + "/" + phiname):
        return None

    return folder + "/" + phiname


########################################
# Function to compute rotationall UVC coordinate by Martin
#########################################
def compute_uvc_rotationalRV(folder, meshtoolloc, meshname, base_surfFilename, apex_pointFilename, RVinsertionNodeANT,
                             RVinsertionNodeINF, phiname):
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apex_pointFilename)
    apex_vtx = int(apex_vtx[0])
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]

    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    centroids = meshIO.create_centroids(elems, pts)

    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    base_surf = meshIO.read_surf(base_surfFilename)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(), pts.loc[base_nodes][1].mean(), pts.loc[base_nodes][2].mean()]

    # Computes long axis between apex and CoM
    longAxis = (apex_coord - CoM_coord) / np.linalg.norm(apex_coord - CoM_coord)

    # Translates mesh to have x,y point aligned with apex
    pts_t = pts - apex_coord

    # Rotates mesh to align long axis with z [001] direction
    z_axis = [0, 0, 1]
    R = geometrical.computeRotationMatrix(z_axis, longAxis)
    pts_r = pd.DataFrame(np.transpose(np.matmul(R, np.transpose(pts_t))))

    #########################################
    # Computes phi based on relative angles
    #########################################
    # Defines 2 RV insertion sites and coords
    RVinsPtANT_coord = pts_r.iloc[RVinsertionNodeANT]
    RVinsPtINF_coord = pts_r.iloc[RVinsertionNodeINF]

    # Picks out the xy coord (projecting it into xy plane)
    zero_phase_RVANT = [RVinsPtANT_coord[0], RVinsPtANT_coord[1]] / np.linalg.norm(
        [RVinsPtANT_coord[0], RVinsPtANT_coord[1]])
    zero_phase_RVINF = [RVinsPtINF_coord[0], RVinsPtINF_coord[1]] / np.linalg.norm(
        [RVinsPtINF_coord[0], RVinsPtINF_coord[1]])

    # find singularity
    sing_phase = -0.5 * (zero_phase_RVANT + zero_phase_RVINF)
    sing_phase = sing_phase / np.linalg.norm(sing_phase)

    # Projects all points in mesh into xy plane
    pts_proj = pts_r.loc[:, 0:1]
    mag_pts = np.linalg.norm(pts_proj, axis=1)
    safe_mag_pts = np.where(mag_pts > np.finfo(float).eps, mag_pts, 1.0)

    # computes the angles between every point in the mesh and...
    # ... anterior RV insertion site
    cosThetas_RVANT = np.clip(np.dot(pts_proj, zero_phase_RVANT) / safe_mag_pts, -1.0, 1.0)
    phi_RVANT = np.arccos(cosThetas_RVANT)
    # ... Interior RV insertion site
    cosThetas_RVINF = np.clip(np.dot(pts_proj, zero_phase_RVINF) / safe_mag_pts, -1.0, 1.0)
    phi_RVINF = np.arccos(cosThetas_RVINF)

    # Defines array to store phi coords
    phi_coord = np.zeros(len(pts))

    # defines anterior nodes and interior nodes
    ant_LV_nodes = np.where(phi_RVANT < phi_RVINF)
    inf_LV_nodes = np.where(phi_RVANT >= phi_RVINF)

    phi_coord = np.pi / 2 - np.pi * (phi_RVINF / (phi_RVANT + phi_RVINF))
    meshIO.write_data(dataFilename=phiname, data=phi_coord)

    if not meshtool.collect_nodal(meshtoolloc, meshname, "carp_txt", folder + "/Phi_RV", '', folder + "/" + phiname):
        return None
    return folder + "/" + phiname


def merge_PHI(folder, meshtoolloc, meshname, phiLV, phiRV, surfloc, UVCfolder):
    os.chdir(folder)
    if not meshtool.insert_data(meshtoolloc, 'LVmyo_i', phiLV, meshname + '_i', 'Phi_insertLV.dat', '0'):
        return None
    if not meshtool.insert_data(meshtoolloc, 'RVmyo_i', phiRV, meshname + '_i', 'Phi_insertRV.dat', '0'):
        return None
    Phi_insertLV = np.genfromtxt('Phi_insertLV.dat')
    Phi_insertRV = np.genfromtxt('Phi_insertRV.dat')
    if not meshtool.extract_surf(meshtoolloc, meshname + '_i', surfloc + "/biv_i/RVinsertion", "0:1"):
        return None
    RVinsertionvtx = meshIO.read_vtx_File(surfloc + "/biv_i/RVinsertion.surf.vtx")
    RVinsertionvtx = RVinsertionvtx.astype(int)
    Phi = Phi_insertLV + Phi_insertRV
    Phi[RVinsertionvtx] = Phi_insertLV[RVinsertionvtx]
    meshIO.write_data(dataFilename=UVCfolder + '/PHI.dat', data=Phi)
    if not meshtool.collect_nodal(meshtoolloc, meshname, "carp_txt", UVCfolder + "/PHI", '', UVCfolder + '/PHI.dat'):
        return None
    return UVCfolder + '/PHI.dat'



def compute_phi_openbase_annuli(
    folder, meshtoolloc, meshname, volume_vtk, coords_z_path,
    lv_annulus_vtk, rv_annulus_vtk, ant_landmark_vtk,
    inf_landmark_vtk, UVCfolder,
):
    """Compute PHI separately on tag-1 LV and tag-0 RV open-base submeshes."""

    def read_values(path, size, name):
        values = np.atleast_1d(np.genfromtxt(path)).reshape(-1)
        if len(values) != size or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{name} has {len(values)} values; expected {size} finite values."
            )
        return values

    def read_landmark(path, name):
        points = np.asarray(pv.read(path).points, dtype=float)
        if len(points) != 1 or not np.all(np.isfinite(points)):
            raise RuntimeError(f"{name} must contain exactly one finite point: {path}")
        return points[0]

    def prepare_chamber(name, mesh_basename, tag, annulus_path, full_coords_z, edge_scale):
        points = np.asarray(meshIO.read_pts(basename=mesh_basename), dtype=float)
        loop, loop_info = _opening_rim_loop(annulus_path)
        base_ids, base_mapping = _map_contour_to_mesh_nodes(
            loop, points, edge_scale, f"{name} opening rim to {name} mesh"
        )
        base = str(work_root / f"{name}_base")
        _write_carp_node_surface(base, base_ids)

        local_coords_z = str(work_root / f"COORDS_Z_{name}.dat")
        if not meshtool.extract_data(
            meshtoolloc, mesh_basename, full_coords_z, local_coords_z, "0"
        ):
            raise RuntimeError(f"meshtool failed to extract COORDS_Z for {name}.")
        coords_z = read_values(local_coords_z, len(points), f"{name} COORDS_Z")

        outer = str(work_root / f"{name}_outer")
        if not meshtool.extract_surf(meshtoolloc, mesh_basename, outer, str(tag)):
            raise RuntimeError(f"meshtool failed to extract the {name} exterior surface.")
        surface_ids = np.atleast_1d(meshIO.read_vtx_File(outer + ".surf.vtx")).astype(np.int64)
        surface_ids = np.unique(surface_ids[(surface_ids >= 0) & (surface_ids < len(points))])
        if not len(surface_ids):
            raise RuntimeError(f"The {name} exterior surface contains no valid nodes.")
        apex_id = int(surface_ids[np.argmin(coords_z[surface_ids])])
        if apex_id in set(base_ids.tolist()):
            raise RuntimeError(f"The selected {name} apex is also a base node.")
        apex = str(work_root / f"{name}_apex")
        meshIO.write_vtx_File(apex, pd.Series([apex_id], dtype=np.int64))
        return {
            "points": points,
            "base": base,
            "base_ids": base_ids,
            "base_mapping": base_mapping,
            "base_loop": loop_info,
            "coords_z": coords_z,
            "apex": apex,
            "apex_id": apex_id,
            "apex_coordinate": points[apex_id],
        }

    def map_landmarks(name, points, edge_scale, ant, inf):
        distances, ids = cKDTree(points).query(np.vstack((ant, inf)), k=1)
        tolerance = max(1e-4, 2.0 * float(edge_scale))
        if not np.all(np.isfinite(distances)) or float(np.max(distances)) > tolerance:
            raise RuntimeError(
                f"{name} landmark mapping distance {np.max(distances):.6g} mm "
                f"exceeds {tolerance:.6g} mm."
            )
        ids = np.asarray(ids, dtype=np.int64)
        if ids[0] == ids[1]:
            raise RuntimeError(f"ANT and INF map to the same {name} mesh node.")
        return ids, np.asarray(distances, dtype=float), tolerance

    folder_path = Path(folder).resolve()
    work_root = folder_path / "PHI_work"
    uvc_root = Path(UVCfolder).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    uvc_root.mkdir(parents=True, exist_ok=True)
    full_mesh = folder_path / meshname
    required = [full_mesh.with_suffix(ext) for ext in (".pts", ".elem", ".lon")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing full-mesh CARP files: {missing}")

    volume = pv.read(volume_vtk)
    if not isinstance(volume, pv.UnstructuredGrid):
        raise RuntimeError(f"PHI volume input is not an unstructured grid: {volume_vtk}")
    edge_scale = _tetra_edge_scale(volume)
    full_points = np.asarray(meshIO.read_pts(basename=str(full_mesh)), dtype=float)
    full_coords_z = read_values(coords_z_path, len(full_points), "Full-mesh COORDS_Z")

    lv_mesh = str(work_root / "LVmyo_i")
    rv_mesh = str(work_root / "RVmyo_i")
    if not meshtool.extract_mesh(meshtoolloc, str(full_mesh), "1", lv_mesh):
        raise RuntimeError("meshtool failed to extract the tag-1 LV mesh.")
    if not meshtool.extract_mesh(meshtoolloc, str(full_mesh), "0", rv_mesh):
        raise RuntimeError("meshtool failed to extract the tag-0 RV mesh.")

    lv = prepare_chamber("LV", lv_mesh, 1, lv_annulus_vtk, coords_z_path, edge_scale)
    rv = prepare_chamber("RV", rv_mesh, 0, rv_annulus_vtk, coords_z_path, edge_scale)
    ant = read_landmark(ant_landmark_vtk, "ANT landmark")
    inf = read_landmark(inf_landmark_vtk, "INF landmark")
    lv_ids, lv_distances, lv_tolerance = map_landmarks("LV", lv["points"], edge_scale, ant, inf)
    rv_ids, rv_distances, rv_tolerance = map_landmarks("RV", rv["points"], edge_scale, ant, inf)

    # The legacy rotational solvers receive the insertion indices in the
    # historical INF, ANT order, despite their parameter names being ANT, INF.
    lv_phi_path = compute_uvc_rotational(
        str(work_root), meshtoolloc, lv_mesh, lv["base"], lv["apex"] + ".vtx",
        int(lv_ids[1]), int(lv_ids[0]), "Phi_LV.dat"
    )
    rv_phi_path = compute_uvc_rotationalRV(
        str(work_root), meshtoolloc, rv_mesh, rv["base"], rv["apex"] + ".vtx",
        int(rv_ids[1]), int(rv_ids[0]), "Phi_RV.dat"
    )
    if lv_phi_path is None or rv_phi_path is None:
        raise RuntimeError("A chamber PHI calculation failed.")
    lv_phi = read_values(lv_phi_path, len(lv["points"]), "LV PHI")
    rv_phi = read_values(rv_phi_path, len(rv["points"]), "RV PHI")

    inserted_lv_path = str(work_root / "Phi_insertLV.dat")
    inserted_rv_path = str(work_root / "Phi_insertRV.dat")
    if not meshtool.insert_data(
        meshtoolloc, lv_mesh, lv_phi_path, str(full_mesh), inserted_lv_path, "0"
    ):
        raise RuntimeError("meshtool failed to insert LV PHI into the full mesh.")
    if not meshtool.insert_data(
        meshtoolloc, rv_mesh, rv_phi_path, str(full_mesh), inserted_rv_path, "0"
    ):
        raise RuntimeError("meshtool failed to insert RV PHI into the full mesh.")
    inserted_lv = read_values(inserted_lv_path, len(full_points), "Inserted LV PHI")
    inserted_rv = read_values(inserted_rv_path, len(full_points), "Inserted RV PHI")

    interface = str(work_root / "RV_LV_interface")
    if not meshtool.extract_surf(meshtoolloc, str(full_mesh), interface, "0:1"):
        raise RuntimeError("meshtool failed to extract the LV/RV interface.")
    interface_ids = np.atleast_1d(meshIO.read_vtx_File(interface + ".surf.vtx")).astype(np.int64)
    interface_ids = np.unique(interface_ids[(interface_ids >= 0) & (interface_ids < len(full_points))])
    if not len(interface_ids):
        raise RuntimeError("The LV/RV interface contains no valid nodes.")

    phi = inserted_lv + inserted_rv
    phi[interface_ids] = inserted_lv[interface_ids]
    if not np.all(np.isfinite(phi)):
        raise RuntimeError("Merged PHI contains non-finite values.")
    if float(np.min(phi)) < -np.pi - 1e-6 or float(np.max(phi)) > np.pi + 1e-6:
        raise RuntimeError("Merged PHI is outside the expected range [-pi, pi].")
    phi_path = str(uvc_root / "PHI.dat")
    meshIO.write_data(phi_path, phi)
    if not meshtool.collect_nodal(
        meshtoolloc, str(full_mesh), "carp_txt", str(uvc_root / "PHI"), "", phi_path
    ):
        raise RuntimeError("meshtool failed to create the PHI visualization files.")

    volume.point_data["PHI"] = phi
    volume.point_data["COORDS_Z"] = full_coords_z
    volume.save(uvc_root / "PHI.vtk", binary=True)

    expected = [
        uvc_root / name for name in (
            "PHI.case", "PHI.dat", "PHI.vtk", "PHI.fiber.ens", "PHI.geo",
            "PHI.tags.ens", "PHI_PHI.ens"
        )
    ]
    missing_outputs = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f"PHI generation did not create valid outputs: {missing_outputs}")
    os.chdir(folder_path)
    return phi_path, {
        "method": "separate_openbase_LV_RV_rotation_then_full_mesh_merge",
        "full_mesh_point_count": int(len(full_points)),
        "LV_point_count": int(len(lv["points"])),
        "RV_point_count": int(len(rv["points"])),
        "interface_node_count": int(len(interface_ids)),
        "median_tetra_edge_length_mm": float(edge_scale),
        "LV_base_loop": lv["base_loop"], "RV_base_loop": rv["base_loop"],
        "LV_base_mapping": lv["base_mapping"], "RV_base_mapping": rv["base_mapping"],
        "LV_apex_node_id": int(lv["apex_id"]), "RV_apex_node_id": int(rv["apex_id"]),
        "LV_apex_coordinate": [float(value) for value in lv["apex_coordinate"]],
        "RV_apex_coordinate": [float(value) for value in rv["apex_coordinate"]],
        "LV_apex_COORDS_Z": float(lv["coords_z"][lv["apex_id"]]),
        "RV_apex_COORDS_Z": float(rv["coords_z"][rv["apex_id"]]),
        "LV_ANT_node_id": int(lv_ids[0]), "LV_INF_node_id": int(lv_ids[1]),
        "RV_ANT_node_id": int(rv_ids[0]), "RV_INF_node_id": int(rv_ids[1]),
        "LV_landmark_mapping_distances_mm": lv_distances.tolist(),
        "RV_landmark_mapping_distances_mm": rv_distances.tolist(),
        "LV_landmark_mapping_tolerance_mm": float(lv_tolerance),
        "RV_landmark_mapping_tolerance_mm": float(rv_tolerance),
        "angular_convention": "ANT=+pi/2_INF=-pi/2_historical_argument_order_preserved",
        "interface_merge_rule": "LV_priority",
        "LV_PHI_range": [float(np.min(lv_phi)), float(np.max(lv_phi))],
        "RV_PHI_range": [float(np.min(rv_phi)), float(np.max(rv_phi))],
        "PHI_range": [float(np.min(phi)), float(np.max(phi))],
        "outputs": [str(path) for path in expected],
    }


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def _repair_degenerate_vectors(vectors, centroids, min_norm=1e-8, neighbors=8):
    """Repair zero/near-zero element vectors using nearby valid directions."""
    vectors = np.asarray(vectors, dtype=float).copy()
    centroids = np.asarray(centroids, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > min_norm)
    if np.all(valid):
        return vectors, 0
    valid_ids = np.flatnonzero(valid)
    if not len(valid_ids):
        raise RuntimeError("No valid element vectors are available for degenerate-vector repair.")

    tree = cKDTree(centroids[valid_ids])
    repaired = np.flatnonzero(~valid)
    k = min(int(neighbors), len(valid_ids))
    for element_id in repaired:
        distances, local_ids = tree.query(centroids[element_id], k=k)
        distances = np.atleast_1d(np.asarray(distances, dtype=float))
        local_ids = np.atleast_1d(np.asarray(local_ids, dtype=np.int64))
        neighbor_ids = valid_ids[local_ids]
        neighbor_vectors = vectors[neighbor_ids]
        neighbor_norms = np.linalg.norm(neighbor_vectors, axis=1)
        unit_vectors = neighbor_vectors / neighbor_norms[:, None]
        weights = 1.0 / np.maximum(distances, 1e-8)
        candidate = np.sum(unit_vectors * weights[:, None], axis=0)
        candidate_norm = np.linalg.norm(candidate)
        if not np.isfinite(candidate_norm) or candidate_norm <= min_norm:
            candidate = unit_vectors[0]
            candidate_norm = np.linalg.norm(candidate)
        scale = float(np.average(neighbor_norms, weights=weights))
        vectors[element_id] = candidate / candidate_norm * scale
    return vectors, int(len(repaired))


BIV_RULE_VERSION = 'step09_elemTag_LV_minus1_RV_plus1_LV_priority_v1'
FIBER_RULE_VERSION = 'step09_tagwise_P1_gradients_six_column_fiber_sheet_v2'
FIBER_ANGLES_DEG = {'alpha_endo': 60., 'alpha_epi': -60., 'beta_endo': -65., 'beta_epi': 25.}


def biv_openbase(volume):
    """Derive nodal chamber labels directly from Step 09, with LV priority."""
    if (not isinstance(volume, pv.UnstructuredGrid) or not volume.n_cells
            or not np.all(volume.celltypes == pv.CellType.TETRA)
            or not np.isfinite(volume.points).all() or 'elemTag' not in volume.cell_data):
        raise ValueError('biv requires a finite tetrahedral mesh with elemTag cell labels.')
    tags = np.asarray(volume.cell_data['elemTag'])
    if set(np.unique(tags)) != {0, 1}:
        raise ValueError('biv requires exactly RV tag 0 and LV tag 1.')
    cells = np.asarray(volume.cells_dict[pv.CellType.TETRA])
    lv = np.unique(cells[tags == 1])
    rv = np.unique(cells[tags == 0])
    if len(np.union1d(lv, rv)) != volume.n_points:
        raise ValueError('biv cannot label orphan points outside every tetrahedron.')
    shared = np.intersect1d(lv, rv)
    values = np.ones(volume.n_points, dtype=np.int32)
    values[lv] = -1
    return values, {
        'method': BIV_RULE_VERSION, 'LV_tag': 1, 'RV_tag': 0,
        'LV_value': -1, 'RV_value': 1, 'shared_node_rule': 'LV_minus_one_priority',
        'lv_nodes': int(np.count_nonzero(values == -1)),
        'rv_nodes': int(np.count_nonzero(values == 1)),
        'shared_interface_nodes': int(len(shared)),
        'LV_element_count': int(np.count_nonzero(tags == 1)),
        'RV_element_count': int(np.count_nonzero(tags == 0)),
    }


def _tetra_nodal_gradient(points, cells, values):
    """Exact element gradients of the piecewise-linear nodal field."""
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError('The nodal field must have one finite value per volume point.')
    corners = points[cells]
    sampled = values[cells]
    try:
        gradient = np.linalg.solve(corners[:, 1:] - corners[:, :1],
                                   (sampled[:, 1:] - sampled[:, :1])[..., None])[..., 0]
    except np.linalg.LinAlgError as error:
        raise ValueError('Fiber generation encountered a singular tetrahedron.') from error
    if not np.isfinite(gradient).all():
        raise ValueError('The tetrahedral coordinate gradients are not finite.')
    return gradient


def fiber_openbase(volume, coords_z, pho, angles=None):
    """Build unit fiber/sheet directions in original element order, separately per tag."""
    biv_openbase(volume)
    angles = dict(FIBER_ANGLES_DEG if angles is None else angles)
    cells = np.asarray(volume.cells_dict[pv.CellType.TETRA])
    tags = np.asarray(volume.cell_data['elemTag'])
    coords_z, pho = np.asarray(coords_z, dtype=float), np.asarray(pho, dtype=float)
    for name, values in (('COORDS_Z', coords_z), ('PHO', pho)):
        if values.shape != (volume.n_points,) or not np.isfinite(values).all():
            raise ValueError(f'{name} must contain one finite value per volume point.')
        if values.min() < -1e-6 or values.max() > 1 + 1e-6:
            raise ValueError(f'{name} must be a normalized coordinate in [0, 1].')
    grad_z = _tetra_nodal_gradient(volume.points, cells, coords_z)
    grad_pho = _tetra_nodal_gradient(volume.points, cells, pho)
    centroids = np.asarray(volume.points, dtype=float)[cells].mean(axis=1)
    repair_mask = np.zeros(volume.n_cells, dtype=np.uint8)
    circ = np.empty_like(grad_z)
    radial = np.empty_like(grad_pho)
    repairs = {}
    for tag, chamber in ((1, 'LV'), (0, 'RV')):
        ids = np.flatnonzero(tags == tag)
        center = centroids[ids]
        local_z, local_pho = grad_z[ids].copy(), grad_pho[ids].copy()
        bad_z = np.linalg.norm(local_z, axis=1) <= 1e-8
        bad_pho = np.linalg.norm(local_pho, axis=1) <= 1e-8
        try:
            local_z, nz = _repair_degenerate_vectors(local_z, center)
            local_pho, nr = _repair_degenerate_vectors(local_pho, center)
            n = local_pho / np.linalg.norm(local_pho, axis=1)[:, None]
            c = np.cross(n, -local_z)
            bad_circ = np.linalg.norm(c, axis=1) <= 1e-8
            c, nc = _repair_degenerate_vectors(c, center)
        except RuntimeError as error:
            raise RuntimeError(f'{chamber} cannot build a coordinate frame: {error}') from error
        # A borrowed circumferential vector must be tangent to this element's
        # radial direction. Record the rare deterministic fallback separately.
        c -= n * np.einsum('ij,ij->i', c, n)[:, None]
        fallback = np.linalg.norm(c, axis=1) <= 1e-8
        if np.any(fallback):
            reference = np.eye(3)[np.argmin(np.abs(n[fallback]), axis=1)]
            c[fallback] = np.cross(n[fallback], reference)
        c /= np.linalg.norm(c, axis=1)[:, None]
        circ[ids], radial[ids] = c, n
        repair_mask[ids[bad_pho]] |= 1
        repair_mask[ids[bad_z]] |= 2
        repair_mask[ids[bad_circ]] |= 4
        repair_mask[ids[fallback]] |= 8
        repairs[chamber] = {'elements': int(len(ids)), 'PHO_gradient': nr,
                            'COORDS_Z_gradient': nz, 'circumferential': nc,
                            'tangent_axis_fallback': int(np.count_nonzero(fallback)),
                            'any_repair': int(np.count_nonzero(repair_mask[ids]))}
    mean_pho = pho[cells].mean(axis=1)
    alpha = (1 - mean_pho) * angles['alpha_endo'] + mean_pho * angles['alpha_epi']
    beta = (1 - mean_pho) * angles['beta_endo'] + mean_pho * angles['beta_epi']
    a, b = np.deg2rad(alpha)[:, None], np.deg2rad(beta)[:, None]
    # Preserve the historical angle convention: helix about radial, then sheet
    # about the resulting fiber. Use independent arrays for the two directions.
    fibers = np.cos(a) * circ + np.sin(a) * np.cross(radial, circ)
    fibers /= np.linalg.norm(fibers, axis=1)[:, None]
    sheets = np.cos(b) * radial + np.sin(b) * np.cross(fibers, radial)
    sheets -= fibers * np.einsum('ij,ij->i', fibers, sheets)[:, None]
    sheets /= np.linalg.norm(sheets, axis=1)[:, None]
    sheet_normals = np.cross(fibers, sheets)
    lon = np.column_stack((fibers, sheets))
    unit_error = max(float(np.abs(np.linalg.norm(fibers, axis=1) - 1).max()),
                     float(np.abs(np.linalg.norm(sheets, axis=1) - 1).max()))
    dot_error = float(np.abs(np.einsum('ij,ij->i', fibers, sheets)).max())
    if not np.isfinite(lon).all() or unit_error > 1e-10 or dot_error > 1e-10:
        raise RuntimeError('Fiber and sheet directions must be finite, unit length, and perpendicular.')
    arrays = {'fiber': fibers, 'sheet': sheets, 'sheet_normal': sheet_normals,
              'helix_angle_deg': alpha, 'sheet_angle_deg': beta, 'fiber_repair_mask': repair_mask,
              'PHO_gradient_magnitude': np.linalg.norm(grad_pho, axis=1),
              'COORDS_Z_gradient_magnitude': np.linalg.norm(grad_z, axis=1)}
    return lon, arrays, {
        'method': FIBER_RULE_VERSION, 'angles_deg': angles,
        'orientation_fields': ['COORDS_Z', 'PHO'],
        'gradient_method': 'piecewise_linear_tetrahedral_element_gradient',
        'repair_rule': 'nearest_valid_directions_within_the_same_ventricle_only',
        'repair_mask_bits': {'1': 'PHO_gradient', '2': 'COORDS_Z_gradient',
                             '4': 'circumferential', '8': 'tangent_axis_fallback'},
        'repairs': repairs, 'repaired_element_count': int(np.count_nonzero(repair_mask)),
        'maximum_unit_norm_error': unit_error, 'maximum_fiber_sheet_abs_dot': dot_error,
        'longitudinal_norm_range': [float(np.linalg.norm(fibers, axis=1).min()), float(np.linalg.norm(fibers, axis=1).max())],
        'sheet_norm_range': [float(np.linalg.norm(sheets, axis=1).min()), float(np.linalg.norm(sheets, axis=1).max())],
        'lon_format': 'header_2_then_six_columns_fiber_xyz_sheet_xyz',
    }


# This function is a modified version of the original by Martin Bishop
def compute_fiber_new(folder, meshtoolloc, meshname, apexvtx, base_surf, PHOname, coordszname, alpha_epi, alpha_endo,
                      beta_epi, beta_endo):
    '''
    return the mesh with longitudinal fiber according to alpha_epi and alpha_endo (usually:pi/3 and -pi/3)
    the 'meshname' is only LV or RV
    '''
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apexvtx)
    apex_vtx = int(apex_vtx[0])
    # meshname="LVmyo"
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]

    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    element_nodes = elems.iloc[:, 0:4].to_numpy(dtype=int)
    point_array = np.asarray(pts, dtype=float)
    centroids = point_array[element_nodes].mean(axis=1)

    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    # base_surf=predata[0]
    base_surf = meshIO.read_surf(base_surf)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(), pts.loc[base_nodes][1].mean(), pts.loc[base_nodes][2].mean()]

    # Computes long axis between apex and CoM
    longAxis = (apex_coord - CoM_coord) / np.linalg.norm(apex_coord - CoM_coord)
    # extract the PHO and COORDS_Z for the specific mesh (LV or RV) to a new .dat
    if not meshtool.extract_data(meshtoolloc, meshname, PHOname, 'PHO_' + meshname + '.dat', "0"):
        return None
    if not meshtool.extract_data(meshtoolloc, meshname, coordszname, 'COORDS_Z' + meshname + '.dat', "0"):
        return None
    # calculate the gradient of the PHO and COORDS_Z

    if not meshtool.extract_gradient(meshtoolloc, meshname, 'COORDS_Z' + meshname + '.dat',
                                     'COORDS_Z_' + meshname + '_gradient', '1'):
        return None
    if not meshtool.extract_gradient(meshtoolloc, meshname, 'PHO_' + meshname + '.dat', 'PHO_' + meshname + '_gradient',
                                     '1'):
        return None

    transmuralFile = 'PHO_' + meshname + '.dat'
    coordz_vec = np.genfromtxt('COORDS_Z_' + meshname + '_gradient.grad.vec')
    transmural_vec = np.genfromtxt('PHO_' + meshname + '_gradient.grad.vec')
    coordz_vec, repaired_coordz = _repair_degenerate_vectors(coordz_vec, centroids)
    transmural_vec, repaired_transmural = _repair_degenerate_vectors(transmural_vec, centroids)
    print(
        f'{meshname}: repaired {repaired_coordz} COORDS_Z and '
        f'{repaired_transmural} PHO gradient vectors'
    )
    # calculate the fiber along the myo surface (circumferential direction)
    circumferential = np.cross(transmural_vec, -coordz_vec)
    circumferential, repaired_circumferential = _repair_degenerate_vectors(
        circumferential, centroids
    )
    if repaired_circumferential:
        print(f'{meshname}: repaired {repaired_circumferential} circumferential vectors')

    # calculate the PHO for each element by averaging the PHO for four pts
    transmural = np.genfromtxt(transmuralFile)
    trans_n0 = transmural[elems.iloc[:, 0]]
    trans_n1 = transmural[elems.iloc[:, 1]]
    trans_n2 = transmural[elems.iloc[:, 2]]
    trans_n3 = transmural[elems.iloc[:, 3]]
    mean_trans = (trans_n0 + trans_n1 + trans_n2 + trans_n3) * 0.25

    alpha_trans = np.zeros(len(circumferential))
    beta_trans = np.zeros(len(circumferential))

    lon_longitudinal = circumferential  # to initialize the sheet lon vector
    lonsheet = circumferential  # to initialize the sheet lon vector

    # CARP's 8-column fiber format uses columns 3 and 4 as separators.
    # Keep them zero rather than NaN so the merged full-mesh field is finite.
    lonall = np.zeros((len(circumferential), 8))
    for i in range(len(circumferential)):
        alpha_trans[i] = -((alpha_endo - alpha_epi)) * mean_trans[i] + alpha_endo
        # rotate the fiber vectors by alpha_trans (in radian) along transmural_vec
        lon_longitudinal[i] = np.dot(rotation_matrix(transmural_vec[i], alpha_trans[i]), circumferential[i])
        lon_longitudinal[i] = lon_longitudinal[i] / np.linalg.norm(lon_longitudinal[i])
        lonall[i, 0:3] = lon_longitudinal[i]

    for i in range(len(circumferential)):
        beta_trans[i] = -((beta_endo - beta_epi)) * mean_trans[i] + beta_endo
        lonsheet[i] = np.dot(rotation_matrix(circumferential[i], beta_trans[i]), transmural_vec[i])
        lonsheet[i] = lonsheet[i] / np.linalg.norm(lonsheet[i])
        lonall[i, 5:8] = lonsheet[i]

    meshIO.write_lon_includesheet(lonFilename=meshname + '_fiber', lon=pd.DataFrame(lonall))
    meshIO.write_lon(lonFilename=meshname + '_longitudinal_fiber', lon=pd.DataFrame(lon_longitudinal))
    meshIO.write_lon(lonFilename=meshname + '_sheet_fiber', lon=pd.DataFrame(lonsheet))
    # meshIO.write_lon(lonFilename=meshname+'_fiber', lon=pd.DataFrame(lon_longitudinal))

    shutil.copyfile(meshname + ".elem", meshname + "_fiber.elem")
    shutil.copyfile(meshname + ".pts", meshname + "_fiber.pts")
    if not meshtool.convert_mesh(meshtoolloc, meshname + "_fiber", meshname + "_fiber", ofmt="vtk"):
        return None
    return meshname + "_fiber"


def merge_fibertomyoonly(folder, meshtoolloc, meshname, fiberfolder):
    os.chdir(folder)
    meshall_fiber = meshIO.read_fibres(basename=meshname + '_i')
    lonall = np.zeros((len(meshall_fiber.index), 8))
    meshIO.write_lon_includesheet(lonFilename=meshname + '_i', lon=pd.DataFrame(lonall))
    submesh = 'LVmyo_i'
    shutil.copyfile(submesh + '.eidx', submesh + '_fiber.eidx')
    shutil.copyfile(submesh + '.nod', submesh + '_fiber.nod')
    if not meshtool.insert_submesh(meshtoolloc, submesh + '_fiber', meshname + '_i', meshname + '_insertLV'):
        return False
    if not meshtool.convert_mesh(meshtoolloc, meshname + '_insertLV', meshname + '_insertLV', ofmt="vtk"):
        return False
    submesh = 'RVmyo_i'
    shutil.copyfile(submesh + '.eidx', submesh + '_fiber.eidx')
    shutil.copyfile(submesh + '.nod', submesh + '_fiber.nod')
    finalmesh_i = meshname + '_i_fiber'
    if not meshtool.insert_submesh(meshtoolloc, submesh + '_fiber', meshname + '_insertLV', finalmesh_i):
        return False
    if not meshtool.convert_mesh(meshtoolloc, finalmesh_i, finalmesh_i, ofmt="vtk"):
        return False

    submesh = meshname + '_i'
    shutil.copyfile(submesh + '.eidx', finalmesh_i + '.eidx')
    shutil.copyfile(submesh + '.nod', finalmesh_i + '.nod')

    meshall = meshIO.read_fibres(basename=meshname)
    lonall1 = np.zeros((len(meshall.index), 8))
    meshIO.write_lon_includesheet(lonFilename=meshname, lon=pd.DataFrame(lonall1))

    finalmesh = fiberfolder + '/' + meshname + '_fiber'
    if not meshtool.insert_submesh(meshtoolloc, finalmesh_i, meshname, finalmesh):
        return False
    meshall_pts = meshIO.read_pts(basename=meshname)
    # change pts unit from 'mm' to 'um' to fit for carp simulations
    meshall_pts = meshIO.write_pts(ptsFilename=finalmesh, pts=meshall_pts * 1000)
    if not meshtool.convert_mesh(meshtoolloc, finalmesh, finalmesh, ofmt="vtk"):
        return False

    return True


def bivlabel(folder, meshname, UVCfolder, source_meshname=None):
    """Write the nodal biventricular coordinate field.

    The historical workflow labels the LV as ``-1`` and the RV as ``1``.
    ``meshIO.read_elems`` returns four node columns followed by the element
    tag, so this implementation also works with the current CARP text format.
    ``source_meshname`` allows the frame-based workflow to label the converted
    whole mesh directly while preserving the legacy ``*_i`` default.
    """
    mesh_basename = source_meshname or (meshname + '_i')
    mesh_basename = os.path.join(folder, mesh_basename)
    pts = meshIO.read_pts(basename=mesh_basename)
    elem = meshIO.read_elems(basename=mesh_basename)

    node_columns = list(elem.columns[:4])
    tag_column = elem.columns[4]
    lv_elements = elem.loc[elem[tag_column].astype(int) == 1, node_columns]
    lv_nodes = np.unique(lv_elements.to_numpy(dtype=int)) if len(lv_elements) else np.array([], dtype=int)
    if len(lv_nodes) and (lv_nodes.min() < 0 or lv_nodes.max() >= len(pts)):
        raise ValueError('Element node indices are outside the point range.')

    # RV is the default; assigning LV afterwards makes shared interface nodes
    # deterministic and preserves the legacy LV-priority behavior.
    bivcoord = np.ones(len(pts), dtype=int)
    bivcoord[lv_nodes] = -1
    output_path = os.path.join(UVCfolder, 'biv.dat')
    meshIO.write_list(dataFilename=output_path, data=bivcoord)
    return output_path
