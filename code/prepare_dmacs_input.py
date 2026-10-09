#!/usr/bin/env python3
"""Prepare new NIfTI inputs for the legacy DMACS pipeline.

The original example.nii.gz files are left unchanged. For each selected
subject, this script writes a compact example.nii that preserves the original
orientation, z-stack, z-spacing, and time frames, resamples only x/y to
0.9375 mm, and stores finite intensities as int16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


TARGET_XY_SPACING_MM = 0.9375
TARGET_INTENSITY_MAX = 4095


def parse_subjects(value: str) -> List[str]:
    subjects = [item.strip() for item in value.split(",") if item.strip()]
    if not subjects:
        raise argparse.ArgumentTypeError("At least one subject is required.")
    return subjects


def inplane_target_grid(img: nib.spatialimages.SpatialImage) -> tuple[tuple[int, int, int], np.ndarray]:
    """Build a target grid that preserves orientation and physical coverage."""
    source_affine = np.asarray(img.affine, dtype=float)
    source_linear = source_affine[:3, :3]
    source_zooms = img.header.get_zooms()
    if len(source_zooms) < 3 or not np.all(np.isfinite(source_zooms[:3])):
        raise ValueError("Input NIfTI has invalid spatial voxel spacing.")

    target_linear = source_linear.copy()
    target_shape = [int(img.shape[0]), int(img.shape[1]), int(img.shape[2])]
    for axis in (0, 1):
        old_spacing = float(np.linalg.norm(source_linear[:, axis]))
        if not np.isfinite(old_spacing) or old_spacing <= 0.0:
            raise ValueError(f"Input NIfTI has invalid spacing on axis {axis}.")
        target_shape[axis] = max(1, int(round((img.shape[axis] - 1) * old_spacing / TARGET_XY_SPACING_MM)) + 1)
        target_linear[:, axis] = source_linear[:, axis] * (TARGET_XY_SPACING_MM / old_spacing)

    target_affine = source_affine.copy()
    target_affine[:3, :3] = target_linear
    return tuple(target_shape), target_affine


def resample_spatial_grid(img: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray]:
    """Resample only x/y while preserving source orientation and z coverage."""
    if len(img.shape) != 4:
        raise ValueError(f"Expected a 4-D cine image, found shape {img.shape}.")
    if any(int(size) < 1 for size in img.shape):
        raise ValueError(f"Input image has an invalid shape: {img.shape}.")

    target_shape_3d, target_affine = inplane_target_grid(img)
    source_data = np.asarray(img.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(source_data)):
        source_data = np.nan_to_num(source_data, nan=0.0, posinf=0.0, neginf=0.0)

    target_data = np.empty(target_shape_3d + (int(img.shape[3]),), dtype=np.float32)
    for frame in range(int(img.shape[3])):
        frame_img = nib.Nifti1Image(
            source_data[:, :, :, frame], img.affine, img.header
        )
        resampled = resample_from_to(
            frame_img,
            (target_shape_3d, target_affine),
            order=1,
            mode="constant",
            cval=0.0,
        )
        target_data[:, :, :, frame] = np.asarray(resampled.dataobj, dtype=np.float32)
    return target_data, target_affine


def normalize_to_int16(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """Scale finite non-negative intensities to a safe 12-bit int16 range."""
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.maximum(data, 0.0)
    positive = data[data > 0.0]
    if positive.size == 0:
        raise ValueError("The image contains no positive intensity values.")

    robust_high = float(np.percentile(positive, 99.5))
    if not np.isfinite(robust_high) or robust_high <= 0.0:
        raise ValueError("Could not determine a positive robust intensity scale.")

    clipped = np.minimum(data, robust_high)
    normalized = np.rint(clipped * (TARGET_INTENSITY_MAX / robust_high)).astype(np.int16)
    return normalized, {
        "positive_voxel_count": int(positive.size),
        "source_positive_p99_5": robust_high,
        "target_intensity_min": int(normalized.min()),
        "target_intensity_max": int(normalized.max()),
        "target_dtype": "int16",
        "target_scale_max": TARGET_INTENSITY_MAX,
    }


def make_output_image(data: np.ndarray, affine: np.ndarray, source: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Create a compact output while retaining source metadata where possible."""
    output = nib.Nifti1Image(data, affine, header=source.header.copy())
    header = output.header
    header.set_data_dtype(np.int16)
    source_zooms = source.header.get_zooms()
    time_spacing = float(source_zooms[3]) if len(source_zooms) > 3 else 1.0
    if not np.isfinite(time_spacing) or time_spacing <= 0.0:
        time_spacing = 1.0
    header.set_zooms((TARGET_XY_SPACING_MM, TARGET_XY_SPACING_MM, float(source_zooms[2]), time_spacing))
    header["descrip"] = b"DMACS input: in-plane resampled to 0.9375mm; robust int16 scaling"
    qform_code = int(source.header["qform_code"])
    sform_code = int(source.header["sform_code"])
    output.set_qform(affine, code=qform_code)
    output.set_sform(affine, code=sform_code)
    output.update_header()
    return output


def prepare_subject(subject_dir: Path, input_name: str, output_name: str, overwrite: bool) -> dict:
    input_path = subject_dir / input_name
    output_path = subject_dir / output_name
    if not input_path.is_file():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output must not overwrite the original input.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_path}")

    source = nib.load(str(input_path))
    original_shape = tuple(int(x) for x in source.shape)
    original_zooms = tuple(float(x) for x in source.header.get_zooms())
    source_data, affine = resample_spatial_grid(source)
    normalized, intensity_info = normalize_to_int16(source_data)
    output = make_output_image(normalized, affine, source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp.nii")
    if temporary_output.exists():
        temporary_output.unlink()
    nib.save(output, str(temporary_output))
    temporary_output.replace(output_path)

    summary = {
        "subject": subject_dir.name,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "original_input_preserved": input_path.is_file(),
        "original_shape": original_shape,
        "original_zooms": original_zooms,
        "output_shape": tuple(int(x) for x in output.shape),
        "output_zooms": tuple(float(x) for x in output.header.get_zooms()),
        "output_xyzt_units": output.header.get_xyzt_units(),
        "output_qform_code": int(output.header["qform_code"]),
        "output_sform_code": int(output.header["sform_code"]),
        "output_dtype": str(output.get_data_dtype()),
        "output_affine": np.asarray(output.affine).tolist(),
        "intensity": intensity_info,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=[f"subject_{i}" for i in range(11, 19)],
        help="Comma-separated subject folder names.",
    )
    parser.add_argument("--input-name", default="example.nii.gz")
    parser.add_argument("--output-name", default="example.nii")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summaries = []
    failures = []
    for subject in args.subjects:
        subject_dir = args.data_dir / subject
        try:
            summary = prepare_subject(
                subject_dir, args.input_name, args.output_name, args.overwrite
            )
            summaries.append(summary)
            print(f"SUCCESS {subject}: {summary['output']}")
        except Exception as exc:
            failures.append({"subject": subject, "error": str(exc)})
            print(f"FAILED {subject}: {exc}")

    print(json.dumps({"successes": summaries, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
