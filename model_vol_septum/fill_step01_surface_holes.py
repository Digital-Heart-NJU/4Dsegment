#!/usr/bin/env python3
"""Fill holes in a subject's four Step 1 surfaces and replace validated files.

Example:
    conda run -n morphinet python model_vol_septum/fill_step01_surface_holes.py \
        --subject-dir data/subject_3 --frame ED

Only boundary caps are added. Existing vertices and triangles must be preserved.
Already closed surfaces are left unchanged. This does not repair the original
vtks folder or run any subsequent opening or volume-meshing stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv
import vtk

from generate_open_volmesh_stable import write_custom_surface_vtk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_3"))
    parser.add_argument("--frame", choices=("ED", "ES"), default="ED")
    parser.add_argument(
        "--surface-dir", type=Path, default=None,
        help="Defaults to SUBJECT_DIR/volmesh_pipline/step_01_load_surf.",
    )
    parser.add_argument(
        "--max-hole-radius", type=float, default=None,
        help="Maximum hole circumsphere radius in mesh units; default fills all holes.",
    )
    args = parser.parse_args()
    if args.max_hole_radius is not None and args.max_hole_radius <= 0:
        parser.error("--max-hole-radius must be positive.")
    return args


def triangles(mesh: pv.PolyData) -> np.ndarray:
    rows = mesh.faces.reshape(-1, 4)
    if not np.all(rows[:, 0] == 3) or mesh.n_cells != len(rows):
        raise RuntimeError("The surface must contain triangle cells only.")
    return rows[:, 1:]


def topology(mesh: pv.PolyData) -> dict:
    faces = triangles(mesh)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True,
    )
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    direction_sums = np.bincount(inverse, weights=directions)
    points = np.asarray(mesh.points, dtype=float)
    double_areas = np.linalg.norm(
        np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                 points[faces[:, 2]] - points[faces[:, 0]]), axis=1,
    )
    return {
        "points": int(mesh.n_points),
        "triangles": int(len(faces)),
        "free_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "inconsistent_orientation_edges": int(np.count_nonzero(
            (counts == 2) & (direction_sums != 0))),
        "degenerate_triangles": int(np.count_nonzero(double_areas <= 0)),
    }


def validate_result(source: pv.PolyData, result: pv.PolyData, tag: int) -> dict:
    if not np.array_equal(source.points, result.points):
        raise RuntimeError("Hole filling changed existing vertices or their order.")
    if not np.array_equal(triangles(source), triangles(result)[:source.n_cells]):
        raise RuntimeError("Hole filling changed existing triangle connectivity or winding.")
    stats = topology(result)
    for key in ("free_edges", "nonmanifold_edges", "inconsistent_orientation_edges",
                "degenerate_triangles"):
        if stats[key]:
            raise RuntimeError(f"Filled surface validation failed: {key}={stats[key]}.")
    tags = np.asarray(result.cell_data.get("elemTag", []))
    if len(tags) != result.n_cells or not np.all(tags == tag):
        raise RuntimeError("The filled surface has missing or incorrect elemTag values.")
    return stats


def fill_surface(source: pv.PolyData, tag: int, max_radius: float | None) -> pv.PolyData:
    before = topology(source)
    if before["nonmanifold_edges"] or before["degenerate_triangles"]:
        raise RuntimeError("Input has nonmanifold edges or degenerate triangles; hole filling alone is insufficient.")
    if before["free_edges"] == 0:
        return source.copy(deep=True)

    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(source)
    filler.SetHoleSize(float(source.length) if max_radius is None else max_radius)
    filler.Update()
    result = pv.wrap(filler.GetOutput()).copy(deep=True).triangulate()
    result = result.compute_normals(
        point_normals=True, cell_normals=True, consistent_normals=True,
        auto_orient_normals=False, split_vertices=False,
    )
    result.cell_data["elemTag"] = np.full(result.n_cells, tag, dtype=np.int32)
    validate_result(source, result, tag)
    return result


def main() -> None:
    args = parse_args()
    subject_id = args.subject_dir.resolve().name
    surface_dir = args.surface_dir or args.subject_dir / "volmesh_pipline" / "step_01_load_surf"
    surface_dir = surface_dir.resolve()
    specs = [(name, tag, surface_dir / f"{subject_id}_{args.frame}_{name}.vtk")
             for name, tag in (("LVendo", 1), ("LVepi", 1), ("RV", 0), ("RVepi", 0))]
    missing = [str(path) for _, _, path in specs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required surfaces: " + ", ".join(missing))

    records = {}
    staged_outputs = []
    with TemporaryDirectory(prefix=".hole_fill_", dir=surface_dir) as temporary:
        temporary_dir = Path(temporary)
        for name, tag, path in specs:
            source = pv.read(path).extract_surface().triangulate()
            source_tags = np.asarray(source.cell_data.get("elemTag", []))
            if len(source_tags) != source.n_cells or not np.all(source_tags == tag):
                raise RuntimeError(f"{path.name}: expected elemTag={tag} on every input face.")
            before = topology(source)
            result = fill_surface(source, tag, args.max_hole_radius)
            after = validate_result(source, result, tag)
            added = result.n_cells - source.n_cells
            if added:
                staged = temporary_dir / path.name
                write_custom_surface_vtk(
                    staged, result.points, triangles(result), result.cell_data["elemTag"],
                    dict(result.point_data), point_precision=9,
                )
                saved = pv.read(staged).extract_surface().triangulate()
                after = validate_result(source, saved, tag)
                staged_outputs.append((staged, path))
            cap_points = result.points[np.unique(triangles(result)[source.n_cells:])]
            records[name] = {
                "path": str(path), "elemTag": tag,
                "status": "filled" if added else "already_closed_unchanged",
                "before": before, "after": after, "added_triangles": int(added),
                "original_vertices_and_triangles_preserved": True,
                "cap_z_range": ([float(cap_points[:, 2].min()), float(cap_points[:, 2].max())]
                                if len(cap_points) else None),
            }
            print(f"{path.name}: free edges {before['free_edges']} -> {after['free_edges']}; "
                  f"added triangles={added}", flush=True)

        # Validate all four results, including serialized geometry, before replacing any input.
        for staged, destination in staged_outputs:
            staged.replace(destination)
        summary = {
            "subject": subject_id, "frame": args.frame, "surface_dir": str(surface_dir),
            "method": "vtkFillHolesFilter_with_consistent_cap_orientation",
            "surfaces": records,
        }
        summary_path = surface_dir / f"{subject_id}_{args.frame}_hole_fill_summary.json"
        summary_staged = temporary_dir / summary_path.name
        summary_staged.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
        summary_staged.replace(summary_path)
    print(f"Saved hole-fill summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
