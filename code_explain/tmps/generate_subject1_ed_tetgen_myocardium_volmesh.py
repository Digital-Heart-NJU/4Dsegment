#!/usr/bin/env python3
"""Generate a myocardium-only subject_1 ED tetrahedral mesh with TetGen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import tetgen

from generate_subject1_ed_tetgen_volmesh import (
    RegionSeed,
    combine_surfaces,
    combine_tet_meshes,
    compact_tets,
    load_surface,
    make_grid,
    make_region_seeds,
    orient_positive_tets,
    quality_summary,
    scale_surface,
    surface_summary,
    switch_string,
)


MYOCARDIUM_LABELS = {
    2: "LV_myocardium",
    4: "RV_myocardium",
}


def tetrahedralize_myocardium(
    name: str,
    surfaces: list[pv.PolyData],
    chamber_seed: RegionSeed,
    myocardium_seed: RegionSeed,
    min_ratio: float,
    max_volume: float,
    preserve_surface: bool,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    plc = combine_surfaces(surfaces)
    tgen = tetgen.TetGen(plc)
    tgen.add_hole(chamber_seed.point.tolist())
    tgen.add_region(myocardium_seed.label, myocardium_seed.point.tolist(), max_volume)

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

    keep = labels == myocardium_seed.label
    dropped = int(np.count_nonzero(~keep))
    if not np.any(keep):
        raise RuntimeError(f"TetGen output for {name} has no myocardium cells")

    points, tets = compact_tets(points, tets[keep])
    labels = labels[keep]
    tets, flipped = orient_positive_tets(points, tets)

    return points, tets, labels, {
        "name": name,
        "method": "tetgen_shell_with_chamber_hole",
        "input_surface_points": int(plc.n_points),
        "input_surface_triangles": int(plc.n_cells),
        "switches": switch_string(min_ratio, max_volume, preserve_surface),
        "preserve_input_surface": preserve_surface,
        "hole": chamber_seed.name,
        "region": myocardium_seed.name,
        "dropped_non_myocardium_tets": dropped,
        "flipped_negative_tets": flipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", default="data/subject_1")
    parser.add_argument("--out-dir", default="data/subject_1/volmesh")
    parser.add_argument("--output-name", default="subject_1_ED_biventricular_volmesh.vtk")
    parser.add_argument("--min-ratio", type=float, default=2.0)
    parser.add_argument("--max-volume", type=float, default=0.0)
    parser.add_argument("--seed-resolution", type=int, default=31)
    parser.add_argument("--rv-epi-scale", type=float, default=1.005)
    parser.add_argument("--allow-surface-refinement", action="store_true")
    parser.add_argument("--verbose-tetgen", action="store_true")
    args = parser.parse_args()

    subject_dir = Path(args.subject_dir)
    vtk_dir = subject_dir / "vtks"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "lv_endo": vtk_dir / "LVendo_ED.vtk",
        "lv_epi": vtk_dir / "LVepi_ED.vtk",
        "rv_endo": vtk_dir / "RV_ED.vtk",
        "rv_epi": vtk_dir / "RVepi_ED.vtk",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    surfaces = {name: load_surface(path) for name, path in paths.items()}
    surfaces["rv_epi"] = scale_surface(surfaces["rv_epi"], args.rv_epi_scale)
    seeds = make_region_seeds(
        surfaces["lv_endo"],
        surfaces["lv_epi"],
        surfaces["rv_endo"],
        surfaces["rv_epi"],
        args.seed_resolution,
    )

    preserve_surface = not args.allow_surface_refinement
    lv_points, lv_tets, lv_labels, lv_info = tetrahedralize_myocardium(
        "lv_myocardium",
        [surfaces["lv_endo"], surfaces["lv_epi"]],
        seeds[0],
        seeds[1],
        args.min_ratio,
        args.max_volume,
        preserve_surface,
        args.verbose_tetgen,
    )
    rv_points, rv_tets, rv_labels, rv_info = tetrahedralize_myocardium(
        "rv_myocardium",
        [surfaces["rv_endo"], surfaces["rv_epi"]],
        seeds[2],
        seeds[3],
        args.min_ratio,
        args.max_volume,
        preserve_surface,
        args.verbose_tetgen,
    )

    points, tets, labels = combine_tet_meshes(
        [(lv_points, lv_tets, lv_labels), (rv_points, rv_tets, rv_labels)]
    )
    grid = make_grid(points, tets, labels)

    output_path = out_dir / args.output_name
    grid.save(output_path, binary=False)

    legend_path = out_dir / "subject_1_ED_biventricular_myocardium_region_legend.csv"
    legend_path.write_text("region_id,name\n2,LV_myocardium\n4,RV_myocardium\n")

    summary = {
        "inputs": {name: str(path) for name, path in paths.items()},
        "outputs": {
            "volumetric_mesh_vtk": str(output_path),
            "region_legend_csv": str(legend_path),
        },
        "mode": "myocardium_only_chambers_empty",
        "region_labels": {str(k): v for k, v in MYOCARDIUM_LABELS.items()},
        "tetgen": {
            "min_ratio": args.min_ratio,
            "max_volume": args.max_volume,
            "switches": switch_string(args.min_ratio, args.max_volume, preserve_surface),
            "preserve_input_surface": preserve_surface,
        },
        "surface_corrections": {
            "rv_epi_uniform_scale_about_center": args.rv_epi_scale,
        },
        "holes": {
            "LV_chamber": [float(x) for x in seeds[0].point],
            "RV_chamber": [float(x) for x in seeds[2].point],
        },
        "surface_summaries": {
            name: surface_summary(name, paths[name], surface)
            for name, surface in surfaces.items()
        },
        "attempts": [lv_info, rv_info],
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
