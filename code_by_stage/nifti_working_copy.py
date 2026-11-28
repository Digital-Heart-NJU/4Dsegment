"""Consistent NIfTI working copies for nibabel and the legacy/SimpleITK readers.

Compatible with the Docker Python 2 runtime. The active nibabel affine is
authoritative; neither voxel order nor world coordinates are resampled.
"""
import nibabel as nib
import numpy as np


def consistent_image(values, reference, dtype=np.float32):
    """Materialize scaled values and synchronize spatial header representations."""
    affine = np.asarray(reference.affine, dtype=float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (not np.isfinite(affine).all() or np.any(spacing <= 0)
            or not np.allclose(affine[3], [0, 0, 0, 1])):
        raise ValueError('Invalid active NIfTI affine for the working copy.')
    direction = affine[:3, :3] / spacing
    if not np.allclose(np.dot(direction.T, direction), np.eye(3), rtol=0, atol=1e-5):
        raise ValueError('Active NIfTI affine has shear; a header-only correction is not sufficient.')
    values = np.asarray(values).astype(dtype, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite values in the NIfTI working copy.')
    image = nib.Nifti1Image(values, affine, header=reference.header.copy())
    image.set_data_dtype(dtype)
    image.header.set_slope_inter(1, 0)
    image.set_sform(affine, code=int(reference.header['sform_code']) or 2)
    image.set_qform(affine, code=int(reference.header['qform_code']) or 1)
    # qform cannot represent shear. Only numerical orthogonalization is allowed.
    if not np.allclose(image.get_qform(), affine, rtol=1e-6, atol=1e-4):
        raise ValueError('Cannot encode the active affine consistently in the working-copy qform.')
    return image


def normalization_report(reference, working):
    original_spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    spacing = np.asarray(working.header.get_zooms()[:3], dtype=float)
    return {'spatial_authority': 'original_active_nibabel_affine',
            'original_header_spacing': original_spacing.tolist(),
            'original_affine_spacing': np.linalg.norm(reference.affine[:3, :3], axis=0).tolist(),
            'working_spacing': spacing.tolist(),
            'spacing_header_corrected': not bool(np.allclose(original_spacing, spacing, rtol=1e-6, atol=1e-6)),
            'original_intensity_slope': float(getattr(reference.dataobj, 'slope', 1.0)),
            'original_intensity_intercept': float(getattr(reference.dataobj, 'inter', 0.0)),
            'working_intensity_slope': 1.0, 'working_intensity_intercept': 0.0,
            'intensities': 'scaled_original_values_materialized_before_legacy_processing',
            'voxel_resampling': False,
            'active_affine_preserved': bool(np.array_equal(working.affine, reference.affine))}
