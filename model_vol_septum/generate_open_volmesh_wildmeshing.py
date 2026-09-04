#!/usr/bin/env python3
"""Generate an open-base biventricular myocardium mesh with WildMeshing.

This script implements the user's revised method:
  1. Load LVendo, LVepi, RV, RVepi surfaces from SUBJECT_DIR/vtks.
  2. Detect one subject-level base side and write LV/RV Z-area profiles.
  3. Build the open-base LV myocardial shell and keep it only as the LV tag reference.
  4. Build the initial open-base RV endo/epi surfaces.
  5. Delete RVepi and RVendo surface parts that intersect LVepi.
  6. Merge LV shell + trimmed RV endo/epi + RV basal annulus as one overall triangle soup.
  7. Tetrahedralize that single overall geometry with WildMeshing/fTetWild.
  8. Tag tetrahedra by centroid-in-LV-shell: LV=1, all other myocardium=0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import wildmeshing

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MESH_VOL_DIR = REPO_ROOT / "mesh_vol"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MESH_VOL_DIR) not in sys.path:
    sys.path.insert(0, str(MESH_VOL_DIR))

import generate_open_volmesh_stable as stable
from custom_vtk import write_custom_vtk


base = stable.base
openbase = stable.openbase


def parse_csv_values(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED")
    parser.add_argument("--frames", default=None, help="Comma-separated frames, for example ED,ES.")
    parser.add_argument("--vtk-dir", type=Path, default=None, help="Defaults to SUBJECT_DIR/vtks.")
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to SUBJECT_DIR/volmesh_final_LVRV_wildmeshing.")
    parser.add_argument("--save-debug", action="store_true")

    parser.add_argument("--subject-base-side", choices=["auto", "low_z", "high_z", "off"], default="auto")
    parser.add_argument("--subject-base-frames", default="ED,ES")
    parser.add_argument("--lv-base-side", choices=["low_z", "high_z"], default=None)
    parser.add_argument("--rv-base-side", choices=["low_z", "high_z"], default=None)

    parser.add_argument("--opening-strategy", choices=["stable_thickness", "max_interface"], default="stable_thickness")
    parser.add_argument("--stable-thickness-percentile", type=float, default=95.0)
    parser.add_argument("--stable-thickness-candidate-percentiles", default="90,92.5,95,97.5,98,99,99.5,100")
    parser.add_argument("--stable-thickness-seed-band-multiplier", type=float, default=2.0)
    parser.add_argument("--stable-thickness-min-basal-face-fraction", type=float, default=0.01)
    parser.add_argument("--stable-thickness-max-search-fraction", type=float, default=0.30)
    parser.add_argument("--stable-thickness-search-steps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--slab-fraction", type=float, default=0.035)
    parser.add_argument("--end-band-fraction", type=float, default=0.12)
    parser.add_argument("--stable-fraction", type=float, default=0.45)
    parser.add_argument("--seed-band-fraction", type=float, default=0.035)
    parser.add_argument("--min-shift", type=float, default=0.75)
    parser.add_argument("--min-shift-fraction", type=float, default=0.01)
    parser.add_argument("--max-shift-fraction", type=float, default=0.25)
    parser.add_argument("--shift-steps", type=int, default=40)
    parser.add_argument("--interface-candidate-rank", type=int, default=0)
    parser.add_argument("--max-interface-candidate-rank", type=int, default=4)
    parser.add_argument("--target-intersection-area-fraction", type=float, default=0.08)
    parser.add_argument("--min-intersection-area", type=float, default=250.0)
    parser.add_argument("--min-intersection-faces", type=int, default=80)
    parser.add_argument("--max-intersection-depth-fraction", type=float, default=0.45)
    parser.add_argument("--max-secondary-loop-area-fraction", type=float, default=0.05)
    parser.add_argument("--boundary-smooth-iterations", type=int, default=40)
    parser.add_argument("--boundary-smooth-relaxation", type=float, default=0.18)
    parser.add_argument("--boundary-blend-rings", type=int, default=2)
    parser.add_argument("--boundary-blend-relaxation", type=float, default=0.35)
    parser.add_argument("--control-loop-points", type=int, default=120)
    parser.add_argument("--control-smooth-iterations", type=int, default=20)
    parser.add_argument("--control-smooth-relaxation", type=float, default=0.20)
    parser.add_argument("--annulus-loop-points", type=int, default=120)
    parser.add_argument("--annulus-radial-layers", type=int, default=4)
    parser.add_argument("--annulus-smooth-iterations", type=int, default=30)
    parser.add_argument("--annulus-smooth-relaxation", type=float, default=0.20)

    parser.add_argument("--surface-lvepi-clearance", type=float, default=0.0)
    parser.add_argument("--surface-intersection-method", choices=["implicit_clip", "face_delete"], default="face_delete")
    parser.add_argument("--surface-removal-rings", type=int, default=0)
    parser.add_argument("--surface-component-min-fraction", type=float, default=0.01)
    parser.add_argument("--boundary-loop-min-length-fraction", type=float, default=0.02)
    parser.add_argument("--surface-clean-tolerance", type=float, default=1e-7)
    parser.add_argument("--combined-surface-merge-tolerance", type=float, default=1e-6)

    parser.add_argument("--wild-stop-quality", type=float, default=10.0)
    parser.add_argument("--wild-max-its", type=int, default=80)
    parser.add_argument("--wild-stage", type=int, default=2)
    parser.add_argument("--wild-max-threads", type=int, default=0)
    parser.add_argument("--wild-epsilon", type=float, default=0.001)
    parser.add_argument("--wild-edge-length-r", type=float, default=0.02)
    parser.add_argument("--wild-skip-simplify", action="store_true")
    parser.add_argument("--wild-no-coarsen", action="store_true")
    parser.add_argument("--wild-log-level", type=int, default=5)
    parser.add_argument("--wild-smooth-open-boundary", action="store_true")
    parser.add_argument("--wild-floodfill", action="store_true")
    parser.add_argument("--wild-use-input-for-wn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wild-manifold-surface", action="store_true")
    parser.add_argument("--wild-correct-surface-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wild-all-mesh", action="store_true")

    parser.add_argument("--lv-tag", type=int, default=1)
    parser.add_argument("--rv-tag", type=int, default=0)
    parser.add_argument("--min-region-cell-fraction", type=float, default=0.03)
    parser.add_argument("--strict-tag-validation", action="store_true")
    parser.add_argument("--drop-small-component-fraction", type=float, default=0.001)
    parser.add_argument("--drop-small-component-min-cells", type=int, default=8)
    parser.add_argument("--max-edge-length-multiplier", type=float, default=8.0)
    parser.add_argument("--max-volume-multiplier", type=float, default=50.0)
    return parser.parse_args()


def resolve_vtk_dir(subject_dir: Path, vtk_dir: Optional[Path]) -> Path:
    if vtk_dir is None:
        return subject_dir / "vtks"
    if vtk_dir.is_absolute():
        return vtk_dir
    subject_relative = subject_dir / vtk_dir
    return subject_relative if subject_relative.exists() else vtk_dir


def resolve_output_dir(subject_dir: Path, output_root: Optional[Path]) -> Path:
    output_dir = output_root or subject_dir / "volmesh_final_LVRV_wildmeshing"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def input_paths_for_frame(vtk_dir: Path, frame: str) -> Dict[str, Path]:
    return {
        "lv_endo": vtk_dir / f"LVendo_{frame}.vtk",
        "lv_epi": vtk_dir / f"LVepi_{frame}.vtk",
        "rv_endo": vtk_dir / f"RV_{frame}.vtk",
        "rv_epi": vtk_dir / f"RVepi_{frame}.vtk",
    }


def serializable_args(args: argparse.Namespace) -> Dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def surface_summary(mesh: pv.PolyData) -> Dict[str, object]:
    mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-7)
    return {
        "points": int(mesh.n_points),
        "triangles": int(mesh.n_cells),
        "free_edges": int(len(base.extract_boundary_edges(mesh))),
        "bounds": [float(v) for v in mesh.bounds],
        "surface_area": float(mesh.area),
    }


def combine_triangle_soup(parts: Sequence[pv.PolyData], tolerance: float) -> pv.PolyData:
    if not parts:
        raise ValueError("No surface parts were supplied.")
    combined = parts[0].extract_surface().triangulate().clean(tolerance=float(tolerance))
    for part in parts[1:]:
        combined = combined.merge(
            part.extract_surface().triangulate().clean(tolerance=float(tolerance)),
            merge_points=True,
            tolerance=float(tolerance),
        )
    return combined.extract_surface().triangulate().clean(tolerance=float(tolerance))


def run_wildmeshing(surface: pv.PolyData, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    surface = surface.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    points_in = np.asarray(surface.points, dtype=np.float64)
    faces_in = base.triangle_faces(surface).astype(np.int32)
    if len(points_in) == 0 or len(faces_in) == 0:
        raise RuntimeError("WildMeshing input surface is empty.")

    tetra = wildmeshing.Tetrahedralizer(
        stop_quality=float(args.wild_stop_quality),
        max_its=int(args.wild_max_its),
        stage=int(args.wild_stage),
        max_threads=int(args.wild_max_threads),
        epsilon=float(args.wild_epsilon),
        edge_length_r=float(args.wild_edge_length_r),
        skip_simplify=bool(args.wild_skip_simplify),
        coarsen=not bool(args.wild_no_coarsen),
    )
    tetra.set_log_level(int(args.wild_log_level))
    tetra.set_mesh(points_in, faces_in)
    tetra.tetrahedralize()
    result = tetra.get_tet_mesh(
        smooth_open_boundary=bool(args.wild_smooth_open_boundary),
        floodfill=bool(args.wild_floodfill),
        use_input_for_wn=bool(args.wild_use_input_for_wn),
        manifold_surface=bool(args.wild_manifold_surface),
        correct_surface_orientation=bool(args.wild_correct_surface_orientation),
        all_mesh=bool(args.wild_all_mesh),
    )
    if len(result) < 2:
        raise RuntimeError("WildMeshing did not return points and tetrahedra.")
    points = np.asarray(result[0], dtype=float)
    cells = np.asarray(result[1], dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] < 4:
        raise RuntimeError("WildMeshing did not return tetrahedral connectivity.")
    cells = cells[:, :4]
    raw_labels = None
    if len(result) >= 3:
        raw = np.asarray(result[2]).reshape(-1)
        if len(raw) == len(cells):
            raw_labels = raw.astype(np.int32)
    stats = tetra.get_stats()
    return points, cells, raw_labels, {
        "method": "wildmeshing_fTetWild",
        "input_surface_points": int(surface.n_points),
        "input_surface_faces": int(surface.n_cells),
        "wild_parameters": {
            "stop_quality": float(args.wild_stop_quality),
            "max_its": int(args.wild_max_its),
            "stage": int(args.wild_stage),
            "max_threads": int(args.wild_max_threads),
            "epsilon": float(args.wild_epsilon),
            "edge_length_r": float(args.wild_edge_length_r),
            "skip_simplify": bool(args.wild_skip_simplify),
            "coarsen": not bool(args.wild_no_coarsen),
            "smooth_open_boundary": bool(args.wild_smooth_open_boundary),
            "floodfill": bool(args.wild_floodfill),
            "use_input_for_wn": bool(args.wild_use_input_for_wn),
            "manifold_surface": bool(args.wild_manifold_surface),
            "correct_surface_orientation": bool(args.wild_correct_surface_orientation),
            "all_mesh": bool(args.wild_all_mesh),
        },
        "output_points": int(len(points)),
        "output_tetrahedra": int(len(cells)),
        "raw_label_counts": {str(k): int(v) for k, v in sorted(Counter(map(int, raw_labels)).items())} if raw_labels is not None else None,
        "stats": stats,
    }



def drop_small_tet_components(
    points: np.ndarray,
    cells: np.ndarray,
    elem_tags: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if len(cells) == 0:
        return points, cells, elem_tags, {"applied": False, "reason": "empty mesh"}
    components = stable.connected_tet_components(np.ones(len(cells), dtype=bool), cells)
    if len(components) <= 1:
        return points, cells, elem_tags, {
            "applied": False,
            "component_count_before": int(len(components)),
            "dropped_cells": 0,
        }
    largest = int(len(components[0]))
    threshold = max(
        int(args.drop_small_component_min_cells),
        int(np.ceil(largest * float(args.drop_small_component_fraction))),
    )
    keep = np.zeros(len(cells), dtype=bool)
    kept_records = []
    dropped_records = []
    for component_id, comp in enumerate(components):
        record = {
            "component_id": int(component_id),
            "cells": int(len(comp)),
            "elemTag": {str(k): int(v) for k, v in sorted(Counter(map(int, elem_tags[comp])).items())},
        }
        if len(comp) >= threshold:
            keep[comp] = True
            kept_records.append(record)
        else:
            dropped_records.append(record)
    if np.all(keep):
        return points, cells, elem_tags, {
            "applied": False,
            "component_count_before": int(len(components)),
            "threshold_cells": int(threshold),
            "dropped_cells": 0,
            "kept_components": kept_records,
        }
    filtered_points, filtered_cells, filtered_tags = stable.compact_tet_mesh(points, cells[keep], elem_tags[keep])
    return filtered_points, filtered_cells, filtered_tags, {
        "applied": True,
        "component_count_before": int(len(components)),
        "threshold_cells": int(threshold),
        "dropped_cells": int(np.count_nonzero(~keep)),
        "kept_components": kept_records,
        "dropped_components": dropped_records,
    }


def classify_by_lv_shell(points: np.ndarray, cells: np.ndarray, lv_shell: pv.PolyData, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object]]:
    centroids = points[cells].mean(axis=1)
    inside_lv = base.enclosed_mask(centroids, lv_shell)
    elem_tags = np.full(len(cells), int(args.rv_tag), dtype=np.int32)
    elem_tags[inside_lv] = int(args.lv_tag)
    counts = Counter(map(int, elem_tags))
    total = int(len(elem_tags))
    min_cells = int(np.ceil(total * float(args.min_region_cell_fraction))) if total else 0
    lv_count = int(counts.get(int(args.lv_tag), 0))
    rv_count = int(counts.get(int(args.rv_tag), 0))
    passes = bool(total > 0 and lv_count >= min_cells and rv_count >= min_cells)
    info = {
        "method": "tet_centroid_inside_LV_myocardial_shell",
        "lv_shell_points": int(lv_shell.n_points),
        "lv_shell_faces": int(lv_shell.n_cells),
        "total_tetrahedra": total,
        "inside_lv_count": int(np.count_nonzero(inside_lv)),
        "elemTag_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "min_region_cell_fraction": float(args.min_region_cell_fraction),
        "passes_min_region_fraction": passes,
    }
    if args.strict_tag_validation and not passes:
        raise RuntimeError(
            "WildMeshing output failed LV/RV tag-distribution validation: "
            f"LV cells={lv_count}, RV/other cells={rv_count}, total={total}."
        )
    return elem_tags, info


def write_elem_tag_legend(path: Path, lv_tag: int, rv_tag: int) -> None:
    path.write_text("elemTag,region\n" f"{int(lv_tag)},LV myocardium\n" f"{int(rv_tag)},RV or non-LV myocardium\n")


def process_frame(args: argparse.Namespace, frame: str) -> Dict[str, object]:
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    vtk_dir = resolve_vtk_dir(subject_dir, args.vtk_dir)
    output_dir = resolve_output_dir(subject_dir, args.output_root)
    input_paths = input_paths_for_frame(vtk_dir, frame)

    surfaces = stable.load_surfaces(vtk_dir, frame)
    frame_args = openbase.namespace_with_overrides(args, {"frame": frame})
    subject_base_side, subject_base_detection = openbase.detect_subject_base_side(vtk_dir, frame_args)
    lv_base_side = args.lv_base_side or subject_base_side
    rv_base_side = args.rv_base_side or subject_base_side
    subject_base_detection["applied_base_side"] = {"lv": lv_base_side, "rv": rv_base_side}

    lv_profile = base.compute_area_profile(
        surfaces["lv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=lv_base_side,
    )
    rv_profile = base.compute_area_profile(
        surfaces["rv_endo"],
        samples=args.samples,
        slab_fraction=args.slab_fraction,
        end_band_fraction=args.end_band_fraction,
        stable_fraction=args.stable_fraction,
        override_base_side=rv_base_side,
    )
    base.write_area_profile(output_dir / f"{subject_id}_{frame}_LV_z_area_profile.csv", lv_profile)
    base.write_area_profile(output_dir / f"{subject_id}_{frame}_RV_z_area_profile.csv", rv_profile)

    lv_build = openbase.build_ventricle_shell("LV", surfaces["lv_endo"], surfaces["lv_epi"], lv_profile, frame_args)
    rv_build = openbase.build_ventricle_shell("RV", surfaces["rv_endo"], surfaces["rv_epi"], rv_profile, frame_args)

    lv_shell_reference = lv_build.shell.extract_surface().triangulate().clean(tolerance=args.surface_clean_tolerance)
    lv_epi_reference = lv_build.epi_open
    rv_epi_trimmed, rv_epi_removal = stable.remove_surface_intersection_with_lvepi(
        "RVepi_open", rv_build.epi_open, lv_epi_reference, args
    )
    rv_endo_trimmed, rv_endo_removal = stable.remove_surface_intersection_with_lvepi(
        "RV_open", rv_build.endo_open, lv_epi_reference, args
    )

    overall_surface = combine_triangle_soup(
        [lv_shell_reference, rv_epi_trimmed, rv_endo_trimmed, rv_build.annulus],
        args.combined_surface_merge_tolerance,
    )
    wild_points, wild_cells, raw_labels, wild_info = run_wildmeshing(overall_surface, args)
    wild_cells, orientation = stable.reorient_tetrahedra(wild_points, wild_cells)
    elem_tags, tag_info = classify_by_lv_shell(wild_points, wild_cells, lv_shell_reference, args)
    wild_points, wild_cells, elem_tags, component_filter = drop_small_tet_components(wild_points, wild_cells, elem_tags, args)
    quality = stable.tet_quality_report("wildmeshing_biventricular", wild_points, wild_cells, args)
    quality["orientation"] = orientation
    components = stable.component_report(wild_cells, elem_tags)
    contact = stable.count_shared_tag_faces(wild_cells, elem_tags, args.lv_tag, args.rv_tag)

    output_mesh = output_dir / f"{subject_id}_{frame}_openbase_LVRV_wildmesh_volmesh.vtk"
    write_custom_vtk(output_mesh, wild_points, wild_cells, elem_tags)
    legend_path = output_dir / f"{subject_id}_{frame}_wildmesh_elemTag_legend.csv"
    write_elem_tag_legend(legend_path, args.lv_tag, args.rv_tag)

    debug_paths: Dict[str, str] = {}
    if args.save_debug:
        debug_dir = output_dir / f"{subject_id}_{frame}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_surfaces = {
            "LV_shell_reference": lv_shell_reference,
            "LVepi_reference": lv_epi_reference,
            "RV_initial_endo_open": rv_build.endo_open,
            "RV_initial_epi_open": rv_build.epi_open,
            "RV_endo_after_LVepi_deletion": rv_endo_trimmed,
            "RVepi_after_LVepi_deletion": rv_epi_trimmed,
            "RV_basal_annulus": rv_build.annulus,
            "overall_triangle_soup_for_wildmeshing": overall_surface,
        }
        for name, mesh in debug_surfaces.items():
            path = debug_dir / f"{subject_id}_{frame}_{name}.vtk"
            mesh.save(path)
            debug_paths[name] = str(path.resolve())

    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "vtk_dir": str(vtk_dir.resolve()),
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "outputs": {
            "biventricular_volmesh": str(output_mesh.resolve()),
            "elemTag_legend": str(legend_path.resolve()),
            "debug": debug_paths,
        },
        "pipeline": [
            "load_LVendo_LVepi_RV_RVepi_surfaces",
            "detect_subject_base_side_and_write_LV_RV_z_area_profiles",
            "build_open_base_LV_myocardial_shell_for_later_tagging",
            "build_initial_open_base_RV_surfaces",
            "delete_RVepi_open_surface_parts_intersecting_LVepi",
            "delete_RV_open_surface_parts_intersecting_LVepi",
            "merge_LV_shell_trimmed_RVendo_trimmed_RVepi_RV_annulus_as_single_triangle_soup",
            "wildmeshing_single_overall_tetrahedralization",
            "post_tag_tetrahedra_by_centroid_inside_LV_shell_elemTag_1_else_0",
        ],
        "parameters": serializable_args(frame_args),
        "subject_base_detection": subject_base_detection,
        "profiles": {"lv": base.profile_to_dict(lv_profile), "rv": base.profile_to_dict(rv_profile)},
        "lv_open_base_reference": openbase.ventricle_summary(lv_build),
        "rv_initial_open_base": openbase.ventricle_summary(rv_build),
        "surface_intersection_removal": {"RVepi": rv_epi_removal, "RV": rv_endo_removal},
        "surface_summaries": {
            "lv_shell_reference": surface_summary(lv_shell_reference),
            "rv_epi_trimmed": surface_summary(rv_epi_trimmed),
            "rv_endo_trimmed": surface_summary(rv_endo_trimmed),
            "rv_annulus": surface_summary(rv_build.annulus),
            "overall_triangle_soup_for_wildmeshing": surface_summary(overall_surface),
        },
        "wildmeshing": wild_info,
        "tagging": tag_info,
        "quality": quality,
        "component_filter": component_filter,
        "components": components,
        "contact": contact,
    }
    summary_path = output_dir / f"{subject_id}_{frame}_openbase_LVRV_wildmesh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "frame": frame,
        "biventricular_volmesh": str(output_mesh),
        "summary": str(summary_path),
        "tetrahedra": int(len(wild_cells)),
        "elemTag": {str(k): int(v) for k, v in sorted(Counter(map(int, elem_tags)).items())},
        "component_count": int(components["component_count"]),
        "shared_lv_rv_faces": int(contact["shared_lv_rv_faces"]),
        "nonpositive_signed_tets": int(quality["nonpositive_signed_tets"]),
        "tag_validation_passed": bool(tag_info["passes_min_region_fraction"]),
    }


def main() -> None:
    args = parse_args()
    frames = parse_csv_values(args.frames) or [args.frame]
    results = [process_frame(args, frame) for frame in frames]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
