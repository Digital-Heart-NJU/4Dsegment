#!/usr/bin/env python3
"""Generate a labeled subject_1 ED biventricular tetrahedral mesh with TetGen."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv
import tetgen
import vtk


REGION_LABELS = {
    1: "LV_chamber",
    2: "LV_myocardium",
    3: "RV_chamber",
    4: "RV_myocardium",
}


@dataclass
class RegionSeed:
    label: int
    name: str
    point: np.ndarray
    clearance: float


def load_surface(path: Path, smooth_iterations: int = 0) -> pv.PolyData:
    mesh = pv.read(path).extract_surface().triangulate().clean(tolerance=0.0)
    if smooth_iterations > 0:
        mesh = mesh.smooth(
            n_iter=smooth_iterations,
            relaxation_factor=0.05,
            boundary_smoothing=False,
            feature_smoothing=False,
        )
        mesh = mesh.triangulate().clean(tolerance=0.0)

    mesh = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )
    return mesh


def scale_surface(mesh: pv.PolyData, factor: float) -> pv.PolyData:
    if abs(factor - 1.0) < 1e-12:
        return mesh
    scaled = mesh.copy(deep=True)
    center = np.asarray(mesh.center, dtype=float)
    scaled.points = center + (mesh.points - center) * factor
    return scaled.triangulate().clean(tolerance=0.0).compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )


def surface_summary(name: str, path: Path, mesh: pv.PolyData) -> dict[str, object]:
    feature_edges = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    return {
        "name": name,
        "file": str(path),
        "points": int(mesh.n_points),
        "triangles": int(mesh.n_cells),
        "bounds": [float(x) for x in mesh.bounds],
        "volume": float(abs(mesh.volume)),
        "open_or_nonmanifold_edges": int(feature_edges.n_cells),
        "all_triangles": bool(mesh.is_all_triangles),
    }


def combine_surfaces(surfaces: Iterable[pv.PolyData]) -> pv.PolyData:
    points = []
    faces = []
    point_offset = 0

    for surface in surfaces:
        surface = surface.triangulate()
        face_array = surface.faces.reshape((-1, 4))
        if not np.all(face_array[:, 0] == 3):
            raise ValueError("All TetGen input surfaces must be triangulated")

        points.append(surface.points)
        adjusted = face_array.copy()
        adjusted[:, 1:] += point_offset
        faces.append(adjusted.ravel())
        point_offset += surface.n_points

    return pv.PolyData(np.vstack(points), np.concatenate(faces)).triangulate()


def enclosed_mask(points: np.ndarray, surface: pv.PolyData) -> np.ndarray:
    cloud = pv.PolyData(points)
    selected = cloud.select_enclosed_points(surface, tolerance=1e-6, check_surface=False)
    return selected.point_data["SelectedPoints"].astype(bool)


def candidate_grid(bounds: tuple[float, float, float, float, float, float], resolution: int) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    spans = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
    pad = np.maximum(spans * 0.025, 1e-6)
    axes = [
        np.linspace(xmin + pad[0], xmax - pad[0], resolution),
        np.linspace(ymin + pad[1], ymax - pad[1], resolution),
        np.linspace(zmin + pad[2], zmax - pad[2], resolution),
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([axis.ravel() for axis in grid])


def implicit_distance(surface: pv.PolyData, points: np.ndarray) -> np.ndarray:
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(surface)
    return np.asarray([abs(distance.EvaluateFunction(point)) for point in points], dtype=float)


def find_region_seed(
    label: int,
    name: str,
    include_surfaces: list[pv.PolyData],
    exclude_surfaces: list[pv.PolyData],
    bounds_surface: pv.PolyData,
    base_resolution: int,
) -> RegionSeed:
    for resolution in (base_resolution, base_resolution + 8, base_resolution + 16):
        candidates = candidate_grid(bounds_surface.bounds, resolution)
        mask = np.ones(len(candidates), dtype=bool)

        for surface in include_surfaces:
            mask &= enclosed_mask(candidates, surface)
        for surface in exclude_surfaces:
            mask &= ~enclosed_mask(candidates, surface)

        region_points = candidates[mask]
        if len(region_points) == 0:
            continue

        distances = [implicit_distance(surface, region_points) for surface in include_surfaces]
        distances.extend(implicit_distance(surface, region_points) for surface in exclude_surfaces)
        clearance = np.min(np.vstack(distances), axis=0)
        best = int(np.argmax(clearance))
        return RegionSeed(label, name, region_points[best], float(clearance[best]))

    raise RuntimeError(f"Could not find an interior TetGen seed for {name}")


def make_region_seeds(
    lv_endo: pv.PolyData,
    lv_epi: pv.PolyData,
    rv_endo: pv.PolyData,
    rv_epi: pv.PolyData,
    seed_resolution: int,
) -> list[RegionSeed]:
    return [
        find_region_seed(1, "LV_chamber", [lv_endo], [], lv_endo, seed_resolution),
        find_region_seed(2, "LV_myocardium", [lv_epi], [lv_endo], lv_epi, seed_resolution),
        find_region_seed(3, "RV_chamber", [rv_endo], [], rv_endo, seed_resolution),
        find_region_seed(4, "RV_myocardium", [rv_epi], [rv_endo], rv_epi, seed_resolution),
    ]


def switch_string(min_ratio: float, max_volume: float, preserve_surface: bool) -> str:
    switches = "p"
    if preserve_surface:
        switches += "Y"
    switches += f"zq{min_ratio:g}A"
    if max_volume > 0.0:
        switches += "a"
    return switches


def compact_tets(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used, inverse = np.unique(tets.ravel(), return_inverse=True)
    return points[used], inverse.reshape(tets.shape)


def orient_positive_tets(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, int]:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    signed = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0
    negative = signed < 0
    oriented = tets.copy()
    if np.any(negative):
        tmp = oriented[negative, 1].copy()
        oriented[negative, 1] = oriented[negative, 2]
        oriented[negative, 2] = tmp
    return oriented, int(np.count_nonzero(negative))


def tetrahedralize_labeled(
    name: str,
    surfaces: list[pv.PolyData],
    seeds: list[RegionSeed],
    allowed_labels: set[int],
    min_ratio: float,
    max_volume: float,
    preserve_surface: bool,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    plc = combine_surfaces(surfaces)
    tgen = tetgen.TetGen(plc)
    for seed in seeds:
        tgen.add_region(seed.label, seed.point.tolist(), max_volume)

    result = tgen.tetrahedralize(
        switches=switch_string(min_ratio, max_volume, preserve_surface),
        quiet=not verbose,
        verbose=1 if verbose else 0,
    )
    if len(result) < 3:
        raise RuntimeError(f"TetGen did not return region attributes for {name}")

    points = np.asarray(result[0], dtype=float)
    tets = np.asarray(result[1], dtype=np.int64)
    labels = np.rint(np.asarray(result[2]).reshape(-1)).astype(np.int32)
    if len(labels) != len(tets):
        raise RuntimeError(f"TetGen region attribute count mismatch for {name}")

    keep = np.isin(labels, list(allowed_labels))
    dropped = int(np.count_nonzero(~keep))
    points, tets = compact_tets(points, tets[keep])
    labels = labels[keep]
    tets, flipped = orient_positive_tets(points, tets)

    present = {int(x) for x in np.unique(labels)}
    missing = sorted(allowed_labels - present)
    if missing:
        raise RuntimeError(f"TetGen output for {name} is missing region labels {missing}")

    info = {
        "name": name,
        "method": "combined_tetgen_plc" if len(allowed_labels) == 4 else "separate_tetgen_plc",
        "input_surface_points": int(plc.n_points),
        "input_surface_triangles": int(plc.n_cells),
        "switches": switch_string(min_ratio, max_volume, preserve_surface),
        "preserve_input_surface": preserve_surface,
        "dropped_unlabeled_or_unrequested_tets": dropped,
        "flipped_negative_tets": flipped,
    }
    return points, tets, labels, info


def combine_tet_meshes(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_points = []
    all_tets = []
    all_labels = []
    offset = 0
    for points, tets, labels in parts:
        all_points.append(points)
        all_tets.append(tets + offset)
        all_labels.append(labels)
        offset += len(points)
    return np.vstack(all_points), np.vstack(all_tets), np.concatenate(all_labels)


def make_grid(points: np.ndarray, tets: np.ndarray, labels: np.ndarray) -> pv.UnstructuredGrid:
    cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64), tets]).ravel()
    cell_types = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.cell_data["region_id"] = labels.astype(np.int32)
    grid.cell_data["material"] = labels.astype(np.int32)
    grid.field_data["region_1"] = np.array(["LV_chamber"])
    grid.field_data["region_2"] = np.array(["LV_myocardium"])
    grid.field_data["region_3"] = np.array(["RV_chamber"])
    grid.field_data["region_4"] = np.array(["RV_myocardium"])
    return grid


def tet_signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    return np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0


def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def quality_summary(points: np.ndarray, tets: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    signed = tet_signed_volumes(points, tets)
    volumes = np.abs(signed)
    tet_points = points[tets]
    edge_pairs = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=np.int64,
    )
    edge_lengths = np.linalg.norm(
        tet_points[:, edge_pairs[:, 0], :] - tet_points[:, edge_pairs[:, 1], :],
        axis=2,
    )
    min_edges = edge_lengths.min(axis=1)
    max_edges = edge_lengths.max(axis=1)
    edge_ratio = np.divide(
        max_edges,
        min_edges,
        out=np.full_like(max_edges, np.inf),
        where=min_edges > 0,
    )

    a0 = triangle_area(tet_points[:, 0], tet_points[:, 1], tet_points[:, 2])
    a1 = triangle_area(tet_points[:, 0], tet_points[:, 1], tet_points[:, 3])
    a2 = triangle_area(tet_points[:, 0], tet_points[:, 2], tet_points[:, 3])
    a3 = triangle_area(tet_points[:, 1], tet_points[:, 2], tet_points[:, 3])
    surface_area = a0 + a1 + a2 + a3
    inradius = np.divide(
        3.0 * volumes,
        surface_area,
        out=np.zeros_like(volumes),
        where=surface_area > 0,
    )
    aspect_proxy = np.divide(
        max_edges,
        inradius,
        out=np.full_like(max_edges, np.inf),
        where=inradius > 0,
    )

    region_counts = {
        str(int(label)): {
            "name": REGION_LABELS.get(int(label), "unknown"),
            "tetrahedra": int(np.count_nonzero(labels == label)),
            "volume": float(volumes[labels == label].sum()),
        }
        for label in sorted(np.unique(labels))
    }

    return {
        "points": int(len(points)),
        "tetrahedra": int(len(tets)),
        "negative_or_zero_signed_tets": int(np.count_nonzero(signed <= 0.0)),
        "volume": {
            "min": float(volumes.min()),
            "mean": float(volumes.mean()),
            "max": float(volumes.max()),
            "total": float(volumes.sum()),
        },
        "edge_ratio_max_over_min": {
            "mean": float(np.mean(edge_ratio)),
            "p95": float(np.percentile(edge_ratio, 95)),
            "max": float(np.max(edge_ratio)),
        },
        "aspect_proxy_max_edge_over_inradius": {
            "mean": float(np.mean(aspect_proxy)),
            "p95": float(np.percentile(aspect_proxy, 95)),
            "max": float(np.max(aspect_proxy)),
        },
        "regions": region_counts,
    }


def write_region_legend(path: Path) -> None:
    lines = ["region_id,name"]
    lines.extend(f"{label},{name}" for label, name in sorted(REGION_LABELS.items()))
    path.write_text("\n".join(lines) + "\n")


def build_mesh(
    surfaces: dict[str, pv.PolyData],
    min_ratio: float,
    max_volume: float,
    seed_resolution: int,
    verbose: bool,
    separate_only: bool,
    preserve_surface: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RegionSeed], list[dict[str, object]]]:
    seeds = make_region_seeds(
        surfaces["lv_endo"],
        surfaces["lv_epi"],
        surfaces["rv_endo"],
        surfaces["rv_epi"],
        seed_resolution,
    )

    attempts: list[dict[str, object]] = []
    if not separate_only:
        try:
            points, tets, labels, info = tetrahedralize_labeled(
                "biventricular",
                [surfaces["lv_endo"], surfaces["lv_epi"], surfaces["rv_endo"], surfaces["rv_epi"]],
                seeds,
                set(REGION_LABELS),
                min_ratio,
                max_volume,
                preserve_surface,
                verbose,
            )
            attempts.append({**info, "status": "ok"})
            return points, tets, labels, seeds, attempts
        except Exception as exc:
            attempts.append({"name": "biventricular", "method": "combined_tetgen_plc", "status": "failed", "error": str(exc)})
    else:
        attempts.append({"name": "biventricular", "method": "combined_tetgen_plc", "status": "skipped"})

    lv_points, lv_tets, lv_labels, lv_info = tetrahedralize_labeled(
        "lv",
        [surfaces["lv_endo"], surfaces["lv_epi"]],
        [seeds[0], seeds[1]],
        {1, 2},
        min_ratio,
        max_volume,
        preserve_surface,
        verbose,
    )
    rv_points, rv_tets, rv_labels, rv_info = tetrahedralize_labeled(
        "rv",
        [surfaces["rv_endo"], surfaces["rv_epi"]],
        [seeds[2], seeds[3]],
        {3, 4},
        min_ratio,
        max_volume,
        preserve_surface,
        verbose,
    )
    attempts.append({**lv_info, "status": "ok"})
    attempts.append({**rv_info, "status": "ok"})
    points, tets, labels = combine_tet_meshes(
        [(lv_points, lv_tets, lv_labels), (rv_points, rv_tets, rv_labels)]
    )
    return points, tets, labels, seeds, attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", default="data/subject_1")
    parser.add_argument("--out-dir", default="data/subject_1/volmesh")
    parser.add_argument("--output-name", default="subject_1_ED_biventricular_volmesh.vtk")
    parser.add_argument("--min-ratio", type=float, default=1.4)
    parser.add_argument("--max-volume", type=float, default=0.0)
    parser.add_argument("--seed-resolution", type=int, default=31)
    parser.add_argument("--rv-epi-scale", type=float, default=1.0)
    parser.add_argument("--allow-surface-refinement", action="store_true")
    parser.add_argument("--smooth-on-failure", action="store_true", default=True)
    parser.add_argument("--smooth-iterations", type=int, default=10)
    parser.add_argument("--separate-only", action="store_true")
    parser.add_argument("--verbose-tetgen", action="store_true")
    args = parser.parse_args()

    subject_dir = Path(args.subject_dir)
    vtk_dir = subject_dir / "vtks"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "lv_endo": vtk_dir / "N_LVendo_ED.vtk",
        "lv_epi": vtk_dir / "N_LVepi_ED.vtk",
        "rv_endo": vtk_dir / "RV_ED.vtk",
        "rv_epi": vtk_dir / "RVepi_ED.vtk",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    summaries: dict[str, object] = {}
    all_attempts: list[dict[str, object]] = []

    surfaces = {name: load_surface(path) for name, path in paths.items()}
    surfaces["rv_epi"] = scale_surface(surfaces["rv_epi"], args.rv_epi_scale)
    summaries["raw_surfaces"] = {
        name: surface_summary(name, paths[name], surface) for name, surface in surfaces.items()
    }

    try:
        points, tets, labels, seeds, attempts = build_mesh(
            surfaces,
            args.min_ratio,
            args.max_volume,
            args.seed_resolution,
            args.verbose_tetgen,
            args.separate_only,
            not args.allow_surface_refinement,
        )
        all_attempts.extend(attempts)
        smoothing_used = 0
    except Exception as raw_exc:
        all_attempts.append({"name": "raw_surfaces", "status": "failed", "error": str(raw_exc)})
        if not args.smooth_on_failure:
            raise
        smoothed = {
            name: load_surface(path, smooth_iterations=args.smooth_iterations)
            for name, path in paths.items()
        }
        smoothed["rv_epi"] = scale_surface(smoothed["rv_epi"], args.rv_epi_scale)
        summaries["smoothed_surfaces"] = {
            name: surface_summary(name, paths[name], surface) for name, surface in smoothed.items()
        }
        points, tets, labels, seeds, attempts = build_mesh(
            smoothed,
            args.min_ratio,
            args.max_volume,
            args.seed_resolution,
            args.verbose_tetgen,
            args.separate_only,
            not args.allow_surface_refinement,
        )
        all_attempts.extend(attempts)
        smoothing_used = args.smooth_iterations

    grid = make_grid(points, tets, labels)
    output_path = out_dir / args.output_name
    grid.save(output_path, binary=False)
    legend_path = out_dir / "subject_1_ED_biventricular_region_legend.csv"
    write_region_legend(legend_path)

    summary = {
        "inputs": {name: str(path) for name, path in paths.items()},
        "outputs": {
            "volumetric_mesh_vtk": str(output_path),
            "region_legend_csv": str(legend_path),
        },
        "region_labels": {str(k): v for k, v in REGION_LABELS.items()},
        "tetgen": {
            "min_ratio": args.min_ratio,
            "max_volume": args.max_volume,
            "switches": switch_string(args.min_ratio, args.max_volume, not args.allow_surface_refinement),
            "preserve_input_surface": not args.allow_surface_refinement,
        },
        "surface_corrections": {
            "rv_epi_uniform_scale_about_center": args.rv_epi_scale,
        },
        "smoothing_iterations_used": smoothing_used,
        "region_seeds": {
            str(seed.label): {
                "name": seed.name,
                "point": [float(x) for x in seed.point],
                "clearance": seed.clearance,
            }
            for seed in seeds
        },
        "surface_summaries": summaries,
        "attempts": all_attempts,
        "quality": quality_summary(points, tets, labels),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["quality"], indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {legend_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
