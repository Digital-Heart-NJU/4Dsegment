#!/usr/bin/env python3
"""Check intersections between two triangle surface VTK files.

Default use checks the step 8 subject_1 ED whole epicardial and whole
endocardial surfaces:

  data/subject_1/volmesh_generate_pipline/step8merge_surface/
    subject_1_ED_stage08_whole_epi_surface.vtk
    subject_1_ED_stage08_whole_endo_surface.vtk

The script writes direct surface-intersection curves and point-level
close-contact evidence for ParaView inspection. Point containment is reported
as a diagnostic only; for nested epi/endo surfaces, containment is not itself
an intersection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyvista as pv
import vtk


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in value)


def default_step8_dir(subject_dir: Path) -> Path:
    return subject_dir / "volmesh_generate_pipline" / "step8merge_surface"


def default_mesh_path(subject_dir: Path, frame: str, surface_name: str) -> Path:
    return default_step8_dir(subject_dir) / f"{subject_dir.name}_{frame}_stage08_{surface_name}_surface.vtk"


def load_surface(path: Path, tolerance: float) -> pv.PolyData:
    if not path.exists():
        raise FileNotFoundError(path)
    mesh = pv.read(path).extract_surface().triangulate().clean(tolerance=float(tolerance))
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise RuntimeError(f"Surface is empty after loading: {path}")
    return mesh


def intersection_curve(first: pv.PolyData, second: pv.PolyData, tolerance: float) -> pv.PolyData:
    intersector = vtk.vtkIntersectionPolyDataFilter()
    intersector.SetInputData(0, first)
    intersector.SetInputData(1, second)
    if hasattr(intersector, "SetTolerance"):
        intersector.SetTolerance(float(tolerance))
    if hasattr(intersector, "SplitFirstOutputOff"):
        intersector.SplitFirstOutputOff()
    if hasattr(intersector, "SplitSecondOutputOff"):
        intersector.SplitSecondOutputOff()
    intersector.Update()
    return pv.wrap(intersector.GetOutput()).clean(tolerance=float(tolerance))


def enclosed_mask(points: np.ndarray, shell: pv.PolyData, tolerance: float) -> np.ndarray:
    selected = pv.PolyData(np.asarray(points, dtype=float)).select_enclosed_points(
        shell,
        tolerance=float(tolerance),
        check_surface=False,
    )
    return selected.point_data["SelectedPoints"].astype(bool)


def signed_distances(points: np.ndarray, shell: pv.PolyData) -> np.ndarray:
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(shell)
    out = np.empty(len(points), dtype=float)
    for index, point in enumerate(np.asarray(points, dtype=float)):
        out[index] = float(
            implicit.EvaluateFunction([float(point[0]), float(point[1]), float(point[2])])
        )
    return out


def point_cloud(points: np.ndarray, values: Dict[str, np.ndarray]) -> pv.PolyData:
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    for name, data in values.items():
        array = np.asarray(data)
        if len(array) == cloud.n_points:
            cloud.point_data[name] = array
    return cloud


def surface_against_surface(
    source_name: str,
    source: pv.PolyData,
    target_name: str,
    target: pv.PolyData,
    close_distance: float,
    enclosed_tolerance: float,
) -> Tuple[Dict[str, object], pv.PolyData, pv.PolyData]:
    points = np.asarray(source.points, dtype=float)
    inside = enclosed_mask(points, target, enclosed_tolerance)
    distances = signed_distances(points, target)
    abs_distances = np.abs(distances)
    close = abs_distances <= float(close_distance)

    inside_points = point_cloud(
        points[inside],
        {
            f"signedDistanceTo{target_name}": distances[inside].astype(np.float32),
            f"absDistanceTo{target_name}": abs_distances[inside].astype(np.float32),
        },
    )
    close_points = point_cloud(
        points[close],
        {
            f"signedDistanceTo{target_name}": distances[close].astype(np.float32),
            f"absDistanceTo{target_name}": abs_distances[close].astype(np.float32),
            f"inside{target_name}": inside[close].astype(np.int32),
        },
    )

    summary = {
        "source": source_name,
        "target": target_name,
        "source_points": int(source.n_points),
        "source_triangles": int(source.n_cells),
        "points_inside_target": int(np.count_nonzero(inside)),
        "points_close_to_target": int(np.count_nonzero(close)),
        "close_distance_mm": float(close_distance),
        "min_signed_distance": float(distances.min()) if len(distances) else None,
        "p01_signed_distance": float(np.percentile(distances, 1)) if len(distances) else None,
        "p05_signed_distance": float(np.percentile(distances, 5)) if len(distances) else None,
        "median_signed_distance": float(np.median(distances)) if len(distances) else None,
        "p95_signed_distance": float(np.percentile(distances, 95)) if len(distances) else None,
        "p99_signed_distance": float(np.percentile(distances, 99)) if len(distances) else None,
        "max_signed_distance": float(distances.max()) if len(distances) else None,
        "min_abs_distance": float(abs_distances.min()) if len(abs_distances) else None,
    }
    return summary, inside_points, close_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--mesh-a", type=Path, default=None)
    parser.add_argument("--mesh-b", type=Path, default=None)
    parser.add_argument("--name-a", default="whole_epi")
    parser.add_argument("--name-b", default="whole_endo")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clean-tolerance", type=float, default=1e-7)
    parser.add_argument("--intersection-tolerance", type=float, default=1e-6)
    parser.add_argument("--enclosed-tolerance", type=float, default=1e-5)
    parser.add_argument("--close-distance", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_dir = args.subject_dir
    frame = args.frame
    mesh_a_path = args.mesh_a or default_mesh_path(subject_dir, frame, "whole_epi")
    mesh_b_path = args.mesh_b or default_mesh_path(subject_dir, frame, "whole_endo")
    output_dir = args.output_dir or default_step8_dir(subject_dir) / "intersection_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    name_a = safe_name(args.name_a)
    name_b = safe_name(args.name_b)
    pair_name = f"{subject_dir.name}_{frame}_{name_a}_vs_{name_b}"

    mesh_a = load_surface(mesh_a_path, args.clean_tolerance)
    mesh_b = load_surface(mesh_b_path, args.clean_tolerance)

    curve = intersection_curve(mesh_a, mesh_b, args.intersection_tolerance)
    curve_path = output_dir / f"{pair_name}_intersection_curve.vtk"
    if curve.n_points or curve.n_cells:
        curve.save(curve_path)

    a_summary, a_inside, a_close = surface_against_surface(
        name_a,
        mesh_a,
        name_b,
        mesh_b,
        args.close_distance,
        args.enclosed_tolerance,
    )
    b_summary, b_inside, b_close = surface_against_surface(
        name_b,
        mesh_b,
        name_a,
        mesh_a,
        args.close_distance,
        args.enclosed_tolerance,
    )

    paths: Dict[str, str | None] = {
        "intersection_curve": str(curve_path.resolve()) if curve.n_points or curve.n_cells else None,
        f"{name_a}_points_inside_{name_b}": None,
        f"{name_b}_points_inside_{name_a}": None,
        f"{name_a}_points_close_to_{name_b}": None,
        f"{name_b}_points_close_to_{name_a}": None,
    }

    a_inside_path = output_dir / f"{pair_name}_{name_a}_points_inside_{name_b}.vtk"
    b_inside_path = output_dir / f"{pair_name}_{name_b}_points_inside_{name_a}.vtk"
    a_close_path = output_dir / f"{pair_name}_{name_a}_points_close_to_{name_b}.vtk"
    b_close_path = output_dir / f"{pair_name}_{name_b}_points_close_to_{name_a}.vtk"
    if a_inside.n_points:
        a_inside.save(a_inside_path)
        paths[f"{name_a}_points_inside_{name_b}"] = str(a_inside_path.resolve())
    if b_inside.n_points:
        b_inside.save(b_inside_path)
        paths[f"{name_b}_points_inside_{name_a}"] = str(b_inside_path.resolve())
    if a_close.n_points:
        a_close.save(a_close_path)
        paths[f"{name_a}_points_close_to_{name_b}"] = str(a_close_path.resolve())
    if b_close.n_points:
        b_close.save(b_close_path)
        paths[f"{name_b}_points_close_to_{name_a}"] = str(b_close_path.resolve())

    summary = {
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "inputs": {
            name_a: str(mesh_a_path.resolve()),
            name_b: str(mesh_b_path.resolve()),
        },
        "parameters": {
            "clean_tolerance": float(args.clean_tolerance),
            "intersection_tolerance": float(args.intersection_tolerance),
            "enclosed_tolerance": float(args.enclosed_tolerance),
            "close_distance_mm": float(args.close_distance),
        },
        "surface_summaries": {
            name_a: {
                "points": int(mesh_a.n_points),
                "triangles": int(mesh_a.n_cells),
                "bounds": [float(v) for v in mesh_a.bounds],
                "surface_area": float(mesh_a.area),
            },
            name_b: {
                "points": int(mesh_b.n_points),
                "triangles": int(mesh_b.n_cells),
                "bounds": [float(v) for v in mesh_b.bounds],
                "surface_area": float(mesh_b.area),
            },
        },
        "direct_surface_intersection": {
            "has_intersection_curve": bool(curve.n_points > 0 or curve.n_cells > 0),
            "intersection_points": int(curve.n_points),
            "intersection_cells": int(curve.n_cells),
            "intersection_length": float(curve.length) if curve.n_points else 0.0,
        },
        "point_containment_and_close_contact": {
            f"{name_a}_against_{name_b}": a_summary,
            f"{name_b}_against_{name_a}": b_summary,
        },
        "paths": paths,
    }
    summary["point_containment_note"] = (
        "Containment counts are diagnostic only. For whole_epi versus whole_endo, "
        "endo points may normally be inside epi, so containment is not used as "
        "the intersection decision."
    )
    summary["has_direct_surface_intersection"] = bool(
        summary["direct_surface_intersection"]["has_intersection_curve"]
    )
    summary["has_close_contact"] = bool(
        a_summary["points_close_to_target"] > 0
        or b_summary["points_close_to_target"] > 0
    )

    summary_path = output_dir / f"{pair_name}_intersection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "summary": str(summary_path.resolve()),
        "has_direct_surface_intersection": summary["has_direct_surface_intersection"],
        "has_close_contact": summary["has_close_contact"],
        "intersection_curve": paths["intersection_curve"],
        "intersection_points": summary["direct_surface_intersection"]["intersection_points"],
        "intersection_cells": summary["direct_surface_intersection"]["intersection_cells"],
        "intersection_length": summary["direct_surface_intersection"]["intersection_length"],
        f"{name_a}_points_inside_{name_b}": a_summary["points_inside_target"],
        f"{name_b}_points_inside_{name_a}": b_summary["points_inside_target"],
        f"{name_a}_points_close_to_{name_b}": a_summary["points_close_to_target"],
        f"{name_b}_points_close_to_{name_a}": b_summary["points_close_to_target"],
    }, indent=2))


if __name__ == "__main__":
    main()
