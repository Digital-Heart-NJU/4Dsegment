#!/usr/bin/env python3
"""Generate ED LV/RV myocardial tetrahedral meshes from selected VTK surfaces."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np


@dataclass
class Surface:
    name: str
    path: Path
    points: np.ndarray
    triangles: np.ndarray


def read_legacy_vtk_polydata(path: Path, name: str) -> Surface:
    tokens = path.read_text().split()
    if "POLYDATA" not in tokens:
        raise ValueError(f"{path} is not a VTK POLYDATA file")
    try:
        i_points = tokens.index("POINTS")
        n_points = int(tokens[i_points + 1])
        i_coords = i_points + 3
        coords = np.asarray(tokens[i_coords : i_coords + 3 * n_points], dtype=float)
        points = coords.reshape((n_points, 3))

        i_polys = tokens.index("POLYGONS")
        n_polys = int(tokens[i_polys + 1])
        cursor = i_polys + 3
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not parse POINTS/POLYGONS from {path}") from exc

    tris = []
    for _ in range(n_polys):
        n = int(tokens[cursor])
        cursor += 1
        ids = [int(x) for x in tokens[cursor : cursor + n]]
        cursor += n
        if n == 3:
            tris.append(ids)
        elif n > 3:
            for j in range(1, n - 1):
                tris.append([ids[0], ids[j], ids[j + 1]])

    if not tris:
        raise ValueError(f"No triangular polygons found in {path}")

    triangles = np.asarray(tris, dtype=np.int64)
    return Surface(name=name, path=path, points=points, triangles=triangles)


def signed_surface_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return float(np.einsum("ij,ij->", p0, np.cross(p1, p2)) / 6.0)


def orient_surface(surface: Surface, outward: bool) -> Surface:
    tris = surface.triangles.copy()
    vol = signed_surface_volume(surface.points, tris)
    wants_positive = outward
    if (vol > 0) != wants_positive:
        tris = tris[:, [0, 2, 1]]
    return Surface(surface.name, surface.path, surface.points, tris)


def edge_diagnostics(triangles: np.ndarray) -> dict[str, int]:
    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
    }


def surface_summary(surface: Surface) -> dict[str, object]:
    return {
        "file": str(surface.path),
        "points": int(len(surface.points)),
        "triangles": int(len(surface.triangles)),
        "bounds_min": surface.points.min(axis=0).round(6).tolist(),
        "bounds_max": surface.points.max(axis=0).round(6).tolist(),
        "signed_volume": signed_surface_volume(surface.points, surface.triangles),
        **edge_diagnostics(surface.triangles),
    }


def write_ascii_stl(path: Path, name: str, surfaces: list[Surface]) -> None:
    with path.open("w") as f:
        f.write(f"solid {name}\n")
        for surface in surfaces:
            points = surface.points
            for tri in surface.triangles:
                p0, p1, p2 = points[tri]
                normal = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(normal)
                if norm <= 1e-12:
                    continue
                normal = normal / norm
                f.write(
                    "  facet normal {:.9g} {:.9g} {:.9g}\n".format(
                        normal[0], normal[1], normal[2]
                    )
                )
                f.write("    outer loop\n")
                for p in (p0, p1, p2):
                    f.write("      vertex {:.9g} {:.9g} {:.9g}\n".format(p[0], p[1], p[2]))
                f.write("    endloop\n")
                f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def tetra_volume(points: np.ndarray, tetra: np.ndarray) -> float:
    p0 = points[tetra[:, 0]]
    p1 = points[tetra[:, 1]]
    p2 = points[tetra[:, 2]]
    p3 = points[tetra[:, 3]]
    signed = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0
    return float(np.abs(signed).sum())


def points_inside_surface(query_points: np.ndarray, surface: Surface) -> np.ndarray:
    """Ray-cast along +x against a closed triangle surface."""
    tri_pts = surface.points[surface.triangles]
    a = tri_pts[:, 0]
    b = tri_pts[:, 1]
    c = tri_pts[:, 2]

    min_y = np.minimum.reduce([a[:, 1], b[:, 1], c[:, 1]])
    max_y = np.maximum.reduce([a[:, 1], b[:, 1], c[:, 1]])
    min_z = np.minimum.reduce([a[:, 2], b[:, 2], c[:, 2]])
    max_z = np.maximum.reduce([a[:, 2], b[:, 2], c[:, 2]])
    max_x = np.maximum.reduce([a[:, 0], b[:, 0], c[:, 0]])

    denom = (b[:, 1] - c[:, 1]) * (a[:, 2] - c[:, 2]) + (
        c[:, 2] - b[:, 2]
    ) * (a[:, 1] - c[:, 1])

    scale = max(float(np.ptp(surface.points, axis=0).max()), 1.0)
    eps = scale * 1e-10
    dedupe = scale * 1e-8
    inside = np.zeros(len(query_points), dtype=bool)

    for i, p in enumerate(query_points):
        candidates = (
            (p[1] >= min_y - eps)
            & (p[1] <= max_y + eps)
            & (p[2] >= min_z - eps)
            & (p[2] <= max_z + eps)
            & (max_x > p[0] + eps)
            & (np.abs(denom) > eps)
        )
        if not np.any(candidates):
            continue

        aa = a[candidates]
        bb = b[candidates]
        cc = c[candidates]
        den = denom[candidates]

        wa = ((bb[:, 1] - cc[:, 1]) * (p[2] - cc[:, 2]) + (cc[:, 2] - bb[:, 2]) * (p[1] - cc[:, 1])) / den
        wb = ((cc[:, 1] - aa[:, 1]) * (p[2] - cc[:, 2]) + (aa[:, 2] - cc[:, 2]) * (p[1] - cc[:, 1])) / den
        wc = 1.0 - wa - wb
        in_tri = (wa >= -eps) & (wb >= -eps) & (wc >= -eps)
        if not np.any(in_tri):
            continue

        x_hit = wa[in_tri] * aa[in_tri, 0] + wb[in_tri] * bb[in_tri, 0] + wc[in_tri] * cc[in_tri, 0]
        x_hit = x_hit[x_hit > p[0] + eps]
        if len(x_hit) == 0:
            continue

        unique_hits = np.unique(np.rint(x_hit / dedupe).astype(np.int64))
        inside[i] = len(unique_hits) % 2 == 1

    return inside


def voxel_tetra_shell(
    name: str,
    outer: Surface,
    inner: Surface,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    bounds_min = outer.points.min(axis=0)
    bounds_max = outer.points.max(axis=0)
    n_cells = np.maximum(np.ceil((bounds_max - bounds_min) / voxel_size).astype(int), 1)
    axes = [bounds_min[d] + voxel_size * np.arange(n_cells[d] + 1) for d in range(3)]
    centers_axes = [(axis[:-1] + axis[1:]) * 0.5 for axis in axes]
    grid = np.meshgrid(*centers_axes, indexing="ij")
    centers = np.column_stack([g.ravel() for g in grid])

    inside_outer = points_inside_surface(centers, outer)
    inside_inner = points_inside_surface(centers, inner)
    shell_mask = (inside_outer & ~inside_inner).reshape(tuple(n_cells))
    selected = np.argwhere(shell_mask)
    if len(selected) == 0:
        raise RuntimeError(f"No voxels selected for {name}; check the input shell surfaces")

    point_index: dict[tuple[int, int, int], int] = {}
    points: list[list[float]] = []
    tets: list[list[int]] = []
    corner_offsets = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    cube_tets = [
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    ]

    def node_id(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        if key not in point_index:
            point_index[key] = len(points)
            points.append([float(axes[0][i]), float(axes[1][j]), float(axes[2][k])])
        return point_index[key]

    for i, j, k in selected:
        cube_nodes = [node_id(i + di, j + dj, k + dk) for di, dj, dk in corner_offsets]
        for tet in cube_tets:
            tets.append([cube_nodes[idx] for idx in tet])

    points_arr = np.asarray(points, dtype=float)
    tets_arr = np.asarray(tets, dtype=np.int64)
    info = {
        "method": "voxel_tetra_shell",
        "name": name,
        "voxel_size": voxel_size,
        "grid_cells": n_cells.astype(int).tolist(),
        "selected_voxels": int(len(selected)),
    }
    return points_arr, tets_arr, info


def generate_tets_from_stl(
    name: str,
    stl_path: Path,
    mesh_size_max: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.01)

        gmsh.model.add(name)
        gmsh.merge(str(stl_path))
        gmsh.model.mesh.classifySurfaces(math.pi, True, False, math.pi)

        surface_tags = [tag for dim, tag in gmsh.model.getEntities(2)]
        if not surface_tags:
            raise RuntimeError(f"Gmsh found no surfaces in {stl_path}")

        loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
        volume = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [volume], 1)
        gmsh.model.mesh.generate(3)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coords, dtype=float).reshape((-1, 3))
        node_index = {int(tag): idx for idx, tag in enumerate(node_tags)}

        tets = []
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3)
        for elem_type, nodes in zip(elem_types, elem_node_tags):
            name_, dim, order, n_nodes, *_ = gmsh.model.mesh.getElementProperties(elem_type)
            if dim != 3 or name_ != "Tetrahedron" or order != 1 or n_nodes != 4:
                continue
            arr = np.asarray(nodes, dtype=np.int64).reshape((-1, 4))
            mapped = np.vectorize(node_index.__getitem__, otypes=[np.int64])(arr)
            tets.append(mapped)

        if not tets:
            raise RuntimeError(f"Gmsh generated no linear tetrahedra for {name}")
        return points, np.vstack(tets)
    finally:
        gmsh.finalize()


def write_legacy_vtk_unstructured(
    path: Path,
    points: np.ndarray,
    tets: np.ndarray,
    materials: np.ndarray,
    title: str,
) -> None:
    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(points)} float\n")
        for p in points:
            f.write("{:.9g} {:.9g} {:.9g}\n".format(p[0], p[1], p[2]))

        f.write(f"CELLS {len(tets)} {len(tets) * 5}\n")
        for tet in tets:
            f.write("4 {} {} {} {}\n".format(tet[0], tet[1], tet[2], tet[3]))

        f.write(f"CELL_TYPES {len(tets)}\n")
        for _ in range(len(tets)):
            f.write("10\n")

        f.write(f"CELL_DATA {len(tets)}\n")
        f.write("SCALARS material int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for m in materials:
            f.write(f"{int(m)}\n")


def combine_meshes(
    lv_points: np.ndarray,
    lv_tets: np.ndarray,
    rv_points: np.ndarray,
    rv_tets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.vstack([lv_points, rv_points])
    tets = np.vstack([lv_tets, rv_tets + len(lv_points)])
    materials = np.concatenate(
        [
            np.ones(len(lv_tets), dtype=np.int32),
            np.full(len(rv_tets), 2, dtype=np.int32),
        ]
    )
    return points, tets, materials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", default="data/subject_1")
    parser.add_argument("--out-dir", default="data/subject_1/volmesh")
    parser.add_argument("--mesh-size-max", type=float, default=4.0)
    parser.add_argument("--verbose-gmsh", action="store_true")
    args = parser.parse_args()

    subject_dir = Path(args.subject_dir)
    vtks = subject_dir / "vtks"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lv_endo = read_legacy_vtk_polydata(vtks / "F_LVendo_ED.vtk", "LV endocardium")
    lv_epi = read_legacy_vtk_polydata(vtks / "F_LVepi_ED.vtk", "LV epicardium")
    rv_endo = read_legacy_vtk_polydata(vtks / "S_RV_ED.vtk", "RV endocardium")
    rv_epi = read_legacy_vtk_polydata(vtks / "N_RVepi_ED.vtk", "RV epicardium")

    raw_surfaces = [lv_endo, lv_epi, rv_endo, rv_epi]
    raw_summary = {s.name: surface_summary(s) for s in raw_surfaces}

    lv_outer = orient_surface(lv_epi, outward=True)
    lv_inner = orient_surface(lv_endo, outward=False)
    rv_outer = orient_surface(rv_epi, outward=True)
    rv_inner = orient_surface(rv_endo, outward=False)

    expected_lv_volume = abs(signed_surface_volume(lv_epi.points, orient_surface(lv_epi, True).triangles)) - abs(
        signed_surface_volume(lv_endo.points, orient_surface(lv_endo, True).triangles)
    )
    expected_rv_volume = abs(signed_surface_volume(rv_epi.points, orient_surface(rv_epi, True).triangles)) - abs(
        signed_surface_volume(rv_endo.points, orient_surface(rv_endo, True).triangles)
    )

    method_notes: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="subject1_volmesh_") as tmp:
            tmp_dir = Path(tmp)
            lv_stl = tmp_dir / "lv_shell.stl"
            rv_stl = tmp_dir / "rv_shell.stl"
            write_ascii_stl(lv_stl, "lv_myo_ed", [lv_outer, lv_inner])
            write_ascii_stl(rv_stl, "rv_myo_ed", [rv_outer, rv_inner])

            lv_points, lv_tets = generate_tets_from_stl(
                "lv_myo_ed", lv_stl, args.mesh_size_max, args.verbose_gmsh
            )
            rv_points, rv_tets = generate_tets_from_stl(
                "rv_myo_ed", rv_stl, args.mesh_size_max, args.verbose_gmsh
            )
        lv_mesh_info = {"method": "gmsh_stl", "mesh_size_max": args.mesh_size_max}
        rv_mesh_info = {"method": "gmsh_stl", "mesh_size_max": args.mesh_size_max}
    except Exception as exc:
        method_notes.append(f"Gmsh STL tetrahedralization failed: {exc}")
        print(f"Gmsh meshing failed ({exc}); using voxel tetrahedral fallback.")
        lv_points, lv_tets, lv_mesh_info = voxel_tetra_shell(
            "lv_myo_ed", lv_outer, lv_inner, args.mesh_size_max
        )
        rv_points, rv_tets, rv_mesh_info = voxel_tetra_shell(
            "rv_myo_ed", rv_outer, rv_inner, args.mesh_size_max
        )

    lv_materials = np.ones(len(lv_tets), dtype=np.int32)
    rv_materials = np.full(len(rv_tets), 2, dtype=np.int32)

    lv_out = out_dir / "lv_myo_ED.vtk"
    rv_out = out_dir / "rv_myo_ED.vtk"
    combined_out = out_dir / "biv_myo_ED.vtk"

    write_legacy_vtk_unstructured(
        lv_out, lv_points, lv_tets, lv_materials, "subject_1 ED LV myocardium tetra mesh"
    )
    write_legacy_vtk_unstructured(
        rv_out, rv_points, rv_tets, rv_materials, "subject_1 ED RV myocardium tetra mesh"
    )
    points, tets, materials = combine_meshes(lv_points, lv_tets, rv_points, rv_tets)
    write_legacy_vtk_unstructured(
        combined_out,
        points,
        tets,
        materials,
        "subject_1 ED biventricular myocardium tetra mesh",
    )

    lv_tet_volume = tetra_volume(lv_points, lv_tets)
    rv_tet_volume = tetra_volume(rv_points, rv_tets)
    summary = {
        "inputs": raw_summary,
        "outputs": {
            "lv": str(lv_out),
            "rv": str(rv_out),
            "combined": str(combined_out),
        },
        "material_labels": {
            "1": "LV myocardium",
            "2": "RV myocardium",
        },
        "mesh_size_max": args.mesh_size_max,
        "meshing": {
            "lv": lv_mesh_info,
            "rv": rv_mesh_info,
            "notes": method_notes,
        },
        "lv": {
            "points": int(len(lv_points)),
            "tetrahedra": int(len(lv_tets)),
            "tetra_volume": lv_tet_volume,
            "expected_shell_volume_from_surfaces": expected_lv_volume,
            "volume_ratio_tets_to_shell": lv_tet_volume / expected_lv_volume
            if expected_lv_volume > 0
            else None,
        },
        "rv": {
            "points": int(len(rv_points)),
            "tetrahedra": int(len(rv_tets)),
            "tetra_volume": rv_tet_volume,
            "expected_shell_volume_from_surfaces": expected_rv_volume,
            "volume_ratio_tets_to_shell": rv_tet_volume / expected_rv_volume
            if expected_rv_volume > 0
            else None,
        },
        "combined": {
            "points": int(len(points)),
            "tetrahedra": int(len(tets)),
        },
    }
    (out_dir / "biv_myo_ED_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["lv"], indent=2))
    print(json.dumps(summary["rv"], indent=2))
    print(json.dumps(summary["combined"], indent=2))
    print(f"Wrote {combined_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
