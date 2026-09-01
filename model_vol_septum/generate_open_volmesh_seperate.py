#!/usr/bin/env python3
"""Generate separate LV/RV open-base myocardial volume meshes from vtks surfaces.

This script reuses the open-base rules from
mesh_vol/generate_subject_open_base_max_interface_smooth_boundary_custom_tag_volmesh.py,
but changes the final behavior:

  subject_x/vtks/LVendo_FRAME.vtk + LVepi_FRAME.vtk -> LV open-base volmesh, elemTag=1
  subject_x/vtks/RV_FRAME.vtk + RVepi_FRAME.vtk     -> RV open-base volmesh, elemTag=0

Outputs are written directly to SUBJECT_DIR/volmesh_LVRV by default:

  subject_x_FRAME_openbase_LV_volmesh.vtk
  subject_x_FRAME_openbase_RV_volmesh.vtk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MESH_VOL_DIR = REPO_ROOT / "mesh_vol"
CODE_EXPLAIN_TMPS_DIR = REPO_ROOT / "code_explain" / "tmps"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MESH_VOL_DIR) not in sys.path:
    sys.path.insert(0, str(MESH_VOL_DIR))
if str(CODE_EXPLAIN_TMPS_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_EXPLAIN_TMPS_DIR))

import generate_subject_open_base_max_interface_smooth_boundary_custom_tag_volmesh as openbase
from custom_vtk import write_custom_vtk


base = openbase.base


def parse_csv_values(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("data/subject_1"))
    parser.add_argument("--frame", default="ED", help="Single frame to process when --frames is not set.")
    parser.add_argument("--frames", default=None, help="Comma-separated frames to process, e.g. ED,ES.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--vtk-dir", type=Path, default=None, help="Surface input folder; defaults to SUBJECT_DIR/vtks.")
    parser.add_argument("--save-debug-surfaces", action="store_true")
    parser.add_argument(
        "--subject-base-side",
        choices=["auto", "low_z", "high_z", "off"],
        default="auto",
        help="Use one base side for the whole subject. auto scores ED/ES LVendo and RV surfaces from --vtk-dir.",
    )
    parser.add_argument("--subject-base-frames", default="ED,ES")
    parser.add_argument(
        "--opening-strategy",
        choices=["stable_thickness", "max_interface"],
        default="stable_thickness",
    )
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
    parser.add_argument("--lv-base-side", choices=["low_z", "high_z"], default=None)
    parser.add_argument("--rv-base-side", choices=["low_z", "high_z"], default=None)
    return parser.parse_args()


def resolve_vtk_dir(subject_dir: Path, vtk_dir: Optional[Path]) -> Path:
    if vtk_dir is None:
        return subject_dir / "vtks"
    if vtk_dir.is_absolute():
        return vtk_dir
    subject_relative = subject_dir / vtk_dir
    if subject_relative.exists():
        return subject_relative
    return vtk_dir


def resolve_output_dir(subject_dir: Path, output_root: Optional[Path]) -> Path:
    output_dir = output_root or subject_dir / "volmesh_LVRV"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def input_paths_for_frame(vtk_dir: Path, frame: str) -> Dict[str, Path]:
    return {
        "lv_endo": vtk_dir / f"LVendo_{frame}.vtk",
        "lv_epi": vtk_dir / f"LVepi_{frame}.vtk",
        "rv_endo": vtk_dir / f"RV_{frame}.vtk",
        "rv_epi": vtk_dir / f"RVepi_{frame}.vtk",
    }


def load_frame_surfaces(vtk_dir: Path, frame: str) -> Dict[str, object]:
    paths = input_paths_for_frame(vtk_dir, frame)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required surface files: " + ", ".join(missing))
    return {name: base.load_surface(path) for name, path in paths.items()}


def write_chamber_mesh(path: Path, points: np.ndarray, cells: np.ndarray, tag: int) -> Dict[str, object]:
    elem_tags = np.full(len(cells), int(tag), dtype=np.int32)
    write_custom_vtk(path, points, cells, elem_tags)
    quality = base.tet_quality(points, cells)
    volume = float(np.abs(base.tet_volumes(points, cells)).sum())
    return {
        "path": str(path.resolve()),
        "points": int(len(points)),
        "tetrahedra": int(len(cells)),
        "elemTag": int(tag),
        "abs_volume": volume,
        "tet_quality": quality,
    }


def save_debug_surfaces(output_dir: Path, subject_id: str, frame: str, lv_build, rv_build) -> None:
    debug_dir = output_dir / f"{subject_id}_{frame}_openbase_debug_surfaces"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_surfaces = {
        f"LVendo_{frame}_open.vtk": lv_build.endo_open,
        f"LVepi_{frame}_open.vtk": lv_build.epi_open,
        f"RV_{frame}_open.vtk": rv_build.endo_open,
        f"RVepi_{frame}_open.vtk": rv_build.epi_open,
        f"LV_myocardium_base_cut_{frame}.vtk": lv_build.annulus,
        f"RV_myocardium_base_cut_{frame}.vtk": rv_build.annulus,
        f"LV_myocardium_boundary_{frame}.vtk": lv_build.shell,
        f"RV_myocardium_boundary_{frame}.vtk": rv_build.shell,
        f"LV_endo_deleted_cap_{frame}.vtk": lv_build.endo_deleted_cap,
        f"RV_endo_deleted_cap_{frame}.vtk": rv_build.endo_deleted_cap,
        f"LV_epi_deleted_cap_{frame}.vtk": lv_build.epi_deleted_cap,
        f"RV_epi_deleted_cap_{frame}.vtk": rv_build.epi_deleted_cap,
        f"LV_endo_opening_loop_{frame}.vtk": lv_build.endo_opening_loop,
        f"RV_endo_opening_loop_{frame}.vtk": rv_build.endo_opening_loop,
        f"LV_epi_opening_loop_{frame}.vtk": lv_build.epi_opening_loop,
        f"RV_epi_opening_loop_{frame}.vtk": rv_build.epi_opening_loop,
    }
    for name, surface in debug_surfaces.items():
        surface.save(debug_dir / name)


def serializable_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def process_frame(args: argparse.Namespace, frame: str) -> Dict[str, object]:
    subject_dir = args.subject_dir
    subject_id = subject_dir.name
    frame_args = openbase.namespace_with_overrides(args, {"frame": frame})
    vtk_dir = resolve_vtk_dir(subject_dir, args.vtk_dir)
    output_dir = resolve_output_dir(subject_dir, args.output_root)
    input_paths = input_paths_for_frame(vtk_dir, frame)
    surfaces = load_frame_surfaces(vtk_dir, frame)

    subject_base_side, subject_base_detection = openbase.detect_subject_base_side(vtk_dir, frame_args)
    lv_base_side = args.lv_base_side or subject_base_side
    rv_base_side = args.rv_base_side or subject_base_side
    subject_base_detection["applied_base_side"] = {"lv": lv_base_side, "rv": rv_base_side}
    subject_base_detection["explicit_chamber_overrides"] = {"lv": args.lv_base_side, "rv": args.rv_base_side}

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

    lv_build, lv_seed, lv_points, lv_cells, lv_attempt, lv_failures = openbase.build_and_tetrahedralize_adaptive(
        "LV", surfaces["lv_endo"], surfaces["lv_epi"], lv_profile, frame_args
    )
    rv_build, rv_seed, rv_points, rv_cells, rv_attempt, rv_failures = openbase.build_and_tetrahedralize_adaptive(
        "RV", surfaces["rv_endo"], surfaces["rv_epi"], rv_profile, frame_args
    )

    lv_path = output_dir / f"{subject_id}_{frame}_openbase_LV_volmesh.vtk"
    rv_path = output_dir / f"{subject_id}_{frame}_openbase_RV_volmesh.vtk"
    lv_mesh_summary = write_chamber_mesh(lv_path, lv_points, lv_cells, tag=1)
    rv_mesh_summary = write_chamber_mesh(rv_path, rv_points, rv_cells, tag=0)
    openbase.write_custom_elem_tag_legend(output_dir / f"{subject_id}_{frame}_separate_elemTag_legend.csv")

    if args.save_debug_surfaces:
        save_debug_surfaces(output_dir, subject_id, frame, lv_build, rv_build)

    summary = {
        "script": str(Path(__file__).resolve()),
        "subject_dir": str(subject_dir.resolve()),
        "frame": frame,
        "vtk_dir": str(vtk_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "outputs": {
            "lv_volmesh": lv_mesh_summary,
            "rv_volmesh": rv_mesh_summary,
        },
        "method": {
            "input_rule": "Read LVendo/LVepi/RV/RVepi surfaces from SUBJECT_DIR/vtks by default.",
            "base_detection": "Use the same subject-level base-side rule as the existing open-base generator.",
            "opening": "Use the existing stable-thickness or max-interface opening rule to clip the base and smooth the endo/epi boundary loops.",
            "tetrahedralization": "Build and tetrahedralize LV and RV myocardium shells independently; no biventricular combine step is used.",
            "labels": "LV output cells are written with elemTag=1. RV output cells are written with elemTag=0.",
        },
        "parameters": serializable_args(frame_args),
        "subject_base_detection": subject_base_detection,
        "ventricles": {
            "lv": openbase.ventricle_summary(lv_build),
            "rv": openbase.ventricle_summary(rv_build),
        },
        "adaptive_attempts": {
            "lv_selected": lv_attempt,
            "rv_selected": rv_attempt,
            "lv_failures": lv_failures,
            "rv_failures": rv_failures,
        },
        "tetgen_seeds": {
            "lv": [float(x) for x in lv_seed],
            "rv": [float(x) for x in rv_seed],
        },
    }
    summary_path = output_dir / f"{subject_id}_{frame}_openbase_separate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "frame": frame,
        "lv_volmesh": str(lv_path),
        "rv_volmesh": str(rv_path),
        "summary": str(summary_path),
        "lv_tetrahedra": int(len(lv_cells)),
        "rv_tetrahedra": int(len(rv_cells)),
        "lv_shift_distance": float(lv_build.intersection.shift_distance),
        "rv_shift_distance": float(rv_build.intersection.shift_distance),
    }


def main() -> None:
    args = parse_args()
    frames = parse_csv_values(args.frames) or [args.frame]
    results = [process_frame(args, frame) for frame in frames]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
