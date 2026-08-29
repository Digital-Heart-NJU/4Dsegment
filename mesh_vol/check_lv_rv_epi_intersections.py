#!/usr/bin/env python3
"""Check whether LV and RV epicardial VTK surfaces intersect.

The direct surface intersection is computed with vtkIntersectionPolyDataFilter.
The script also reports point containment in each direction, which is useful
when the surfaces overlap enough that one wall penetrates the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyvista as pv
import vtk


def load_surface(path: Path, tolerance: float) -> pv.PolyData:
    if not path.exists():
        raise FileNotFoundError(path)
    surface = pv.read(path).extract_surface().triangulate().clean(tolerance=tolerance)
    if surface.n_points == 0 or surface.n_cells == 0:
        raise RuntimeError(f"Surface is empty after loading: {path}")
    return surface


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in name)


def intersection_curve(
    first: pv.PolyData,
    second: pv.PolyData,
    tolerance: float,
) -> pv.PolyData:
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
    return pv.wrap(intersector.GetOutput()).clean(tolerance=tolerance)


def enclosed_mask(points: np.ndarray, shell: pv.PolyData) -> np.ndarray:
    selected = pv.PolyData(points).select_enclosed_points(
        shell,
        tolerance=1e-5,
        check_surface=False,
    )
    return selected.point_data["SelectedPoints"].astype(bool)


def signed_distances(points: np.ndarray, shell: pv.PolyData) -> np.ndarray:
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(shell)
    distances = np.empty(len(points), dtype=float)
    for index, point in enumerate(np.asarray(points, dtype=float)):
        distances[index] = float(
            implicit.EvaluateFunction(
                [float(point[0]), float(point[1]), float(point[2])]
            )
        )
    return distances


def surface_against_shell_summary(
    name: str,
    source: pv.PolyData,
    shell: pv.PolyData,
    close_distance: float,
) -> Tuple[Dict[str, object], pv.PolyData]:
    points = np.asarray(source.points)
    inside = enclosed_mask(points, shell)
    distances = signed_distances(points, shell)
    abs_distances = np.abs(distances)
    close = abs_distances <= close_distance

    selected_points = points[inside]
    if len(selected_points):
        inside_points = pv.PolyData(selected_points)
    else:
        inside_points = pv.PolyData()

    summary = {
        "name": name,
        "source_points": int(source.n_points),
        "source_triangles": int(source.n_cells),
        "inside_other_surface_points": int(np.count_nonzero(inside)),
        "close_to_other_surface_points": int(np.count_nonzero(close)),
        "close_distance": float(close_distance),
        "min_signed_distance": float(distances.min()) if len(distances) else None,
        "p01_signed_distance": float(np.percentile(distances, 1)) if len(distances) else None,
        "p05_signed_distance": float(np.percentile(distances, 5)) if len(distances) else None,
        "median_signed_distance": float(np.median(distances)) if len(distances) else None,
        "max_signed_distance": float(distances.max()) if len(distances) else None,
        "min_abs_distance": float(abs_distances.min()) if len(abs_distances) else None,
    }
    return summary, inside_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--first-surface", type=Path, default=None)
    parser.add_argument("--second-surface", type=Path, default=None)
    parser.add_argument("--first-name", default="LVepi")
    parser.add_argument("--second-name", default="RVepi")
    parser.add_argument("--lv-epi", type=Path, default=None)
    parser.add_argument("--rv-epi", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clean-tolerance", type=float, default=1e-7)
    parser.add_argument("--intersection-tolerance", type=float, default=1e-6)
    parser.add_argument("--close-distance", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_dir = args.subject_dir
    frame = args.frame
    vtk_dir = subject_dir / "vtks"
    first_name = args.first_name
    second_name = args.second_name
    lv_path = args.first_surface or args.lv_epi or vtk_dir / f"LVepi_{frame}.vtk"
    rv_path = args.second_surface or args.rv_epi or vtk_dir / f"RVepi_{frame}.vtk"
    output_dir = args.output_dir or subject_dir / "epi_intersection_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_name = f"{subject_dir.name}_{frame}_{safe_name(first_name)}_{safe_name(second_name)}"

    lv = load_surface(lv_path, args.clean_tolerance)
    rv = load_surface(rv_path, args.clean_tolerance)

    curve = intersection_curve(lv, rv, args.intersection_tolerance)
    curve_path = output_dir / f"{pair_name}_intersection_curve.vtk"
    if curve.n_points:
        curve.save(curve_path)

    lv_summary, lv_inside_points = surface_against_shell_summary(
        f"{first_name}_points_inside_{second_name}",
        lv,
        rv,
        args.close_distance,
    )
    rv_summary, rv_inside_points = surface_against_shell_summary(
        f"{second_name}_points_inside_{first_name}",
        rv,
        lv,
        args.close_distance,
    )

    lv_inside_path = output_dir / f"{pair_name}_{safe_name(first_name)}_points_inside_{safe_name(second_name)}.vtk"
    rv_inside_path = output_dir / f"{pair_name}_{safe_name(second_name)}_points_inside_{safe_name(first_name)}.vtk"
    if lv_inside_points.n_points:
        lv_inside_points.save(lv_inside_path)
    if rv_inside_points.n_points:
        rv_inside_points.save(rv_inside_path)

    summary = {
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "inputs": {
            first_name: str(lv_path.resolve()),
            second_name: str(rv_path.resolve()),
        },
        "direct_surface_intersection": {
            "has_intersection_curve": bool(curve.n_points > 0 or curve.n_cells > 0),
            "intersection_points": int(curve.n_points),
            "intersection_cells": int(curve.n_cells),
            "intersection_length": float(curve.length) if curve.n_points else 0.0,
            "intersection_curve_vtk": str(curve_path.resolve()) if curve.n_points else None,
        },
        "point_containment": {
            f"{first_name}_against_{second_name}": lv_summary,
            f"{second_name}_against_{first_name}": rv_summary,
            f"{first_name}_points_inside_{second_name}_vtk": str(lv_inside_path.resolve()) if lv_inside_points.n_points else None,
            f"{second_name}_points_inside_{first_name}_vtk": str(rv_inside_path.resolve()) if rv_inside_points.n_points else None,
        },
    }
    summary["has_intersection_or_penetration"] = bool(
        summary["direct_surface_intersection"]["has_intersection_curve"]
        or lv_summary["inside_other_surface_points"] > 0
        or rv_summary["inside_other_surface_points"] > 0
    )

    summary_path = output_dir / f"{pair_name}_intersection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "summary": str(summary_path),
        "has_intersection_or_penetration": summary["has_intersection_or_penetration"],
        "intersection_points": summary["direct_surface_intersection"]["intersection_points"],
        "intersection_cells": summary["direct_surface_intersection"]["intersection_cells"],
        "intersection_length": summary["direct_surface_intersection"]["intersection_length"],
        f"{first_name}_points_inside_{second_name}": lv_summary["inside_other_surface_points"],
        f"{second_name}_points_inside_{first_name}": rv_summary["inside_other_surface_points"],
        "intersection_curve_vtk": summary["direct_surface_intersection"]["intersection_curve_vtk"],
    }, indent=2))


if __name__ == "__main__":
    main()
