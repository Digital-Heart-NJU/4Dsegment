"""Native acquisition-frame selection for the legacy phase-image coordinates.

Keep this module compatible with the Docker Python 2 runtime. Labels never
pass through temporalalign or intensity processing.
"""
from __future__ import print_function

import json
import os
import re
import subprocess

import nibabel as nib
import numpy as np


def temporal_frame_lookup(reference_path, native_image, temporary, processed_count):
    """Ask the actual temporalalign executable which original frames it selects."""
    count = native_image.shape[3]
    if count > np.iinfo(np.int16).max:
        raise ValueError('Too many native frames for the temporal index probe.')
    values = np.empty((2, 2, 2, count), dtype=np.int16)
    values[:] = np.arange(count, dtype=np.int16)
    probe = nib.Nifti1Image(values, native_image.affine, header=native_image.header.copy())
    probe.header.set_data_dtype(np.int16)
    probe.header.set_slope_inter(1, 0)
    source = os.path.join(temporary, 'time_indices.nii.gz')
    target = os.path.join(temporary, 'time_indices_aligned.nii.gz')
    nib.save(probe, source)
    subprocess.check_call(['temporalalign', reference_path, source, target,
                           '-St1', '0', '-St2', '0', '-Et1', '1', '-Et2', '1'])
    sampled = np.asarray(nib.load(target).dataobj)
    if sampled.shape != (2, 2, 2, processed_count):
        raise RuntimeError('Temporal index probe does not match the processed cine frame count.')
    lookup = sampled[0, 0, 0].astype(np.int64)
    if (not np.array_equal(sampled, np.broadcast_to(lookup, sampled.shape))
            or np.any(lookup < 0) or np.any(lookup >= count) or np.any(np.diff(lookup) < 0)):
        raise RuntimeError('Invalid or non-monotone original-frame lookup from temporalalign.')
    return lookup


def rebuild_native_phases(original_path, native_path, processed_path, phase_paths,
                          detection_output, atlas_dir, temporary, subject_dir):
    """Use one original acquisition frame for both each phase MRI and its labels."""
    from image_utils import validate_nonnegative_image

    original = nib.load(original_path)
    native = nib.load(native_path)
    processed = nib.load(processed_path)
    if len(native.shape) != 4 or native.shape[:3] != original.shape[:3]:
        raise RuntimeError('Native phase export requires the original 4D spatial grid.')
    match = re.search(r'ES\s+phase\s+is:\s*(\d+)', detection_output)
    if match is None:
        raise RuntimeError('cardiacphasedetection did not report its ES frame index.')
    selected = {'ED': 0, 'ES': int(match.group(1))}
    count = processed.shape[3]
    lookup = temporal_frame_lookup(os.path.join(atlas_dir, 'temporal.gipl'), native, temporary, count)
    # Verify the probe on a high-resolution ramp too: fractional interpolation
    # must not be mistaken for an original acquisition frame by integer rounding.
    ramp = np.empty((2, 2, 2, native.shape[3]), dtype=np.int16)
    step = max(1, 30000 // max(native.shape[3] - 1, 1))
    ramp[:] = np.arange(native.shape[3], dtype=np.int16) * step
    ramp_image = nib.Nifti1Image(ramp, native.affine, header=native.header.copy())
    ramp_image.header.set_data_dtype(np.int16)
    ramp_image.header.set_slope_inter(1, 0)
    ramp_source, ramp_target = (os.path.join(temporary, name) for name in ('ramp.nii.gz', 'ramp_aligned.nii.gz'))
    nib.save(ramp_image, ramp_source)
    subprocess.check_call(['temporalalign', os.path.join(atlas_dir, 'temporal.gipl'), ramp_source, ramp_target,
                           '-St1', '0', '-St2', '0', '-Et1', '1', '-Et2', '1'])
    if not np.array_equal(np.asarray(nib.load(ramp_target).dataobj),
                          np.broadcast_to(lookup * step, (2, 2, 2, count))):
        raise RuntimeError('temporalalign interpolates frames; native-label correspondence cannot be exact.')
    for frame, index in selected.items():
        if not 0 <= index < count:
            raise RuntimeError('Detected phase index is outside the processed cine: ' + frame)
        detected = np.asarray(nib.load(phase_paths[frame]).dataobj)[..., 0].astype(np.float64)
        reference = np.asarray(processed.dataobj[..., index]).astype(np.float64)
        if not np.allclose(detected, reference, rtol=0, atol=1.01):
            raise RuntimeError('Reported phase index does not match the extracted phase MRI: ' + frame)
    subprocess.check_call(['autocontrast', native_path, native_path])
    validate_nonnegative_image(native_path)
    native = nib.load(native_path)
    timing_unit = original.header.get_xyzt_units()[1]
    dt = float(original.header['pixdim'][4]) if len(original.shape) == 4 else 0.0
    offset = float(original.header['toffset'])
    transform = np.dot(native.affine, np.linalg.inv(original.affine))
    report = {'method': 'native_frame_inference_then_exact_legacy_frame_lookup',
              'index_convention': 'zero_based; one_based values also recorded',
              'original_frame_count': int(native.shape[3]), 'processed_frame_count': int(count),
              'processed_to_original_frame_indices': lookup.tolist(),
              'temporal_lookup': 'actual_temporalalign_index_and_high_resolution_ramp_probes',
              'label_alignment': 'same_voxel_indices_with_legacy_reset_affine_no_resampling',
              'source_to_phase_world_matrix': transform.tolist(),
              'original_time_spacing': dt, 'original_time_unit': timing_unit, 'frames': {}}
    for frame, processed_index in selected.items():
        index = int(lookup[processed_index])
        values = np.asarray(native.dataobj[..., index:index + 1])
        image = nib.Nifti1Image(values, native.affine, header=native.header.copy())
        image.header.set_data_dtype(values.dtype)
        image.header.set_slope_inter(1, 0)
        nib.save(image, phase_paths[frame])
        report['frames'][frame] = {'original_frame_index_zero_based': index,
                                   'original_frame_number_one_based': index + 1,
                                   'processed_frame_index_zero_based': processed_index,
                                   'processed_frame_number_one_based': processed_index + 1,
                                   'original_time': offset + index * dt}
    segs = os.path.join(subject_dir, 'segs')
    if not os.path.isdir(segs):
        os.makedirs(segs)
    with open(os.path.join(segs, 'ED_ES_frames.json'), 'w') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    with open(os.path.join(segs, 'ED_ES_frames.txt'), 'w') as stream:
        stream.write('ED/ES acquisition-frame information\n')
        stream.write('Original image: {0}\n'.format(original_path))
        stream.write('Original frames: {0}; processed cine frames: {1}\n'.format(native.shape[3], count))
        stream.write('Zero-based index 0 is the first frame; one-based frame 1 is the first frame.\n')
        for frame in ('ED', 'ES'):
            info = report['frames'][frame]
            stream.write('{0}: original index={1}, original frame={2}, processed index={3}, processed frame={4}, time={5:.12g} {6}\n'.format(
                frame, info['original_frame_index_zero_based'], info['original_frame_number_one_based'],
                info['processed_frame_index_zero_based'], info['processed_frame_number_one_based'], info['original_time'], timing_unit))
        stream.write('Labels are selected from original acquisitions; no temporal interpolation of labels.\n')
        stream.write('Spatial alignment retains voxel indices and changes only the coordinate reference.\n')
        stream.write('Mapping is verified using temporalalign index and high-resolution ramp probes.\n')
        stream.write('Original-to-phase world-coordinate matrix:\n')
        np.savetxt(stream, transform, fmt='%.17g')
    return report
