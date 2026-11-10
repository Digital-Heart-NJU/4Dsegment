#!/usr/bin/env python3
"""Crop a 3D/4D NIfTI image using voxel indices, without resampling.

X/Y bounds are zero-based and inclusive, in the stored array axes (not world
coordinates or necessarily the axes shown by an image viewer). Z slices and
time frames default to all. Singleton dimensions are never squeezed.

Example, keeping every Z slice and time frame:
    python additional_function/crop_nifti_by_coordinates.py \
        --x-min 50 --x-max 249 --y-min 50 --y-max 249

Requires numpy and nibabel, available in the morphinet conda environment.
"""

import argparse
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _range_slice(name, start, stop, size):
    if not 0 <= start < stop <= size:
        raise ValueError(
            f"Invalid {name} range [{start}, {stop}): axis length is {size}; "
            "the range must be nonempty and within the image."
        )
    return slice(start, stop)


def _count_slice(name, start, count, size):
    stop = size if count is None else start + count
    return _range_slice(name, start, stop, size)


def _verify_saved(path, expected, raw, slope, intercept):
    saved = nib.load(path)
    if saved.shape != raw.shape or saved.get_data_dtype() != expected.get_data_dtype():
        raise RuntimeError("Saved image shape or data type changed unexpectedly.")
    if not np.array_equal(saved.dataobj.get_unscaled(), raw, equal_nan=True):
        raise RuntimeError("Saved voxel values differ from the selected input voxels.")
    if (saved.dataobj.slope, saved.dataobj.inter) != (slope, intercept):
        raise RuntimeError("Saved intensity scaling changed unexpectedly.")
    if not np.array_equal(saved.header.get_zooms(), expected.header.get_zooms()):
        raise RuntimeError("Saved voxel spacing changed unexpectedly.")
    if not np.allclose(saved.affine, expected.affine, rtol=1e-6, atol=1e-5):
        raise RuntimeError("Saved spatial coordinates do not match the cropped input.")
    for form in ("qform", "sform"):
        saved_matrix, saved_code = getattr(saved, f"get_{form}")(coded=True)
        expected_matrix, expected_code = getattr(expected, f"get_{form}")(coded=True)
        if saved_code != expected_code or (
            saved_code and not np.allclose(
                saved_matrix, expected_matrix, rtol=1e-6, atol=1e-5
            )
        ):
            raise RuntimeError(f"Saved {form} does not match the cropped input.")
    if saved.header.get_xyzt_units() != expected.header.get_xyzt_units():
        raise RuntimeError("Saved spatial/time units changed unexpectedly.")
    if saved.header["toffset"] != expected.header["toffset"]:
        raise RuntimeError("Saved time offset changed unexpectedly.")


def crop_nifti(
    input_path,
    output_path,
    *,
    x_min,
    x_max,
    y_min,
    y_max,
    z_start=0,
    z_count=None,
    frame_start=0,
    frame_count=None,
    overwrite=False,
    dry_run=False,
):
    """Write a validated crop and return its selection and metadata summary.

    Raw stored values and NIfTI intensity scaling are preserved separately.
    Spatial forms are translated by the crop offset. For an image with neither
    spatial form coded, an explicit sform is added to preserve its coordinates.
    A frame selection also updates toffset in the input's original time units.
    """
    source = Path(input_path).expanduser().resolve(strict=True)
    target = Path(output_path).expanduser().resolve()
    if source == target or (target.exists() and source.samefile(target)):
        raise ValueError("Input and output must be different files; the input is protected.")
    if not str(target).endswith((".nii", ".nii.gz")):
        raise ValueError("Output must end in .nii or .nii.gz.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {target}. Use --overwrite to replace it.")

    image = nib.load(source)
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError("Input must be a NIfTI image (.nii or .nii.gz).")
    if image.ndim not in (3, 4):
        raise ValueError(f"Expected a 3D/4D image, found shape {image.shape}.")
    selection = [
        _range_slice("X", x_min, x_max + 1, image.shape[0]),
        _range_slice("Y", y_min, y_max + 1, image.shape[1]),
        _count_slice("Z", z_start, z_count, image.shape[2]),
    ]
    time_slice = _count_slice(
        "time frame", frame_start, frame_count, image.shape[3] if image.ndim == 4 else 1
    )
    if image.ndim == 4:
        selection.append(time_slice)

    translation = np.eye(4)
    translation[:3, 3] = [x_min, y_min, z_start]
    new_affine = image.affine @ translation
    summary = {
        "input": str(source),
        "output": str(target),
        "input_shape": image.shape,
        "output_shape": tuple(s.stop - s.start for s in selection),
        "selection": tuple((s.start, s.stop - 1) for s in selection),
        "dtype": str(image.get_data_dtype()),
        "spacing": tuple(float(v) for v in image.header.get_zooms()),
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    # Avoid get_fdata(): it expands integers and may trigger lossy re-quantization.
    raw = np.array(image.dataobj.get_unscaled()[tuple(selection)], copy=True, order="F")
    slope, intercept = image.dataobj.slope, image.dataobj.inter
    cropped = type(image)(raw, new_affine, header=image.header.copy())
    qform, qcode = image.get_qform(coded=False), int(image.header["qform_code"])
    sform, scode = image.get_sform(coded=False), int(image.header["sform_code"])
    cropped.set_qform(qform @ translation, code=qcode, strip_shears=False)
    cropped.set_sform(sform @ translation, code=scode)
    if qcode == 0 and scode == 0:
        cropped.set_sform(new_affine, code=2)
    cropped.header.set_zooms(image.header.get_zooms())
    if image.ndim == 4:
        cropped.header["toffset"] = (
            float(image.header["toffset"]) + frame_start * float(image.header.get_zooms()[3])
        )
    cropped.header.set_slope_inter(slope, intercept)

    # Verify a temporary file before publishing it, leaving an existing output
    # untouched on failure. Hard-link publication refuses concurrent overwrites.
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nifti-crop-", dir=target.parent) as temp_dir:
        temporary = Path(temp_dir) / target.name
        nib.save(cropped, temporary)
        _verify_saved(temporary, cropped, raw, slope, intercept)
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Crop an X/Y rectangle without resampling. Indices are zero-based; "
        "X/Y maxima are inclusive. All Z slices and time frames are kept by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/subject_21/example.nii")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/subject_211/example.nii")
    for name in ("x-min", "x-max", "y-min", "y-max"):
        parser.add_argument(f"--{name}", required=True, type=int, help="Inclusive voxel index")
    parser.add_argument("--z-start", type=int, default=0, help="First Z slice index")
    parser.add_argument("--z-count", type=int, help="Number of Z slices; omitted means all remaining")
    parser.add_argument("--frame-start", type=int, default=0, help="First time frame index")
    parser.add_argument("--frame-count", type=int, help="Number of frames; omitted means all remaining")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the output, never the input")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing files")
    args = parser.parse_args(argv)
    try:
        result = crop_nifti(
            args.input, args.output,
            x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max,
            z_start=args.z_start, z_count=args.z_count,
            frame_start=args.frame_start, frame_count=args.frame_count,
            overwrite=args.overwrite, dry_run=args.dry_run,
        )
    except (OSError, ValueError, RuntimeError, nib.filebasedimages.ImageFileError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    print(f"Input: {result['input']}")
    print(f"Shape: {result['input_shape']} -> {result['output_shape']}")
    print(f"Inclusive ranges (X, Y, Z[, time]): {result['selection']}")
    print(f"Preserved dtype: {result['dtype']}; spacing: {result['spacing']}")
    status = "Dry run, would save" if result["dry_run"] else "Saved and verified"
    print(f"{status}: {result['output']}")


if __name__ == "__main__":
    main()
