"""Stage input checks and file protection, compatible with legacy Python 2."""
from __future__ import print_function

from contextlib import contextmanager
import hashlib
import json
import os
import shutil

import nibabel as nib
import numpy as np


FRAMES = ('ED', 'ES')
SURFACES = ('LVendo', 'LVepi', 'RV', 'RVepi', 'LVmyo')
STAGE_FLAGS = ('stage01_seg_only', 'stage02_surf_only', 'stage03_addi_mot_fit')


def select_stage(flags):
    selected = [name for name in STAGE_FLAGS if getattr(flags, name, False)]
    if len(selected) > 1:
        raise ValueError('Enable only one stage flag at a time: ' + ', '.join(selected))
    return selected[0] if selected else 'full'


def subject_directories(test_dir):
    if not os.path.isdir(test_dir):
        raise IOError('Dataset parent directory does not exist: ' + test_dir)
    subjects = [os.path.join(test_dir, name) for name in sorted(os.listdir(test_dir))
                if os.path.isdir(os.path.join(test_dir, name))]
    if not subjects:
        raise ValueError('No subject directories found under ' + test_dir)
    return subjects


def require_file(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise IOError('Required nonempty file is missing: ' + path)
    return path


def segmentation_path(subject, frame, manual=False):
    suffix = '_manual' if manual else ''
    return os.path.join(subject, 'segs', 'seg_lvsa_{0}{1}.nii.gz'.format(frame, suffix))


def phase_image_path(subject, frame):
    name = 'lvsa_{0}.nii.gz'.format(frame)
    for directory in (subject, os.path.join(subject, 'sizes')):
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return require_file(candidate)
    raise IOError('Missing phase image {0}; run Stage 01 first.'.format(os.path.join(subject, name)))


def validate_segmentations(subjects, manual=False, require_foreground=False):
    """Check every subject before any resumed processing or phase restoration."""
    records = []
    for subject in subjects:
        for frame in FRAMES:
            path = require_file(segmentation_path(subject, frame, manual))
            phase_path = phase_image_path(subject, frame)
            image, phase = nib.load(path), nib.load(phase_path)
            if len(image.shape) not in (3, 4) or (len(image.shape) == 4 and image.shape[3] != 1):
                raise ValueError('Expected a 3D or singleton-time segmentation: ' + path)
            if image.shape[:3] != phase.shape[:3]:
                raise ValueError('Segmentation dimensions do not match phase MRI: ' + path)
            if not np.allclose(image.affine, phase.affine, rtol=1e-6, atol=1e-4):
                raise ValueError('Segmentation geometry does not match phase MRI: ' + path)
            data = np.asarray(image.dataobj)
            if not np.isfinite(data).all():
                raise ValueError('Nonfinite segmentation labels: ' + path)
            labels = np.unique(data)
            if not np.isin(labels, np.arange(5)).all():
                raise ValueError('Segmentation must contain integer labels 0..4: ' + path)
            if (manual or require_foreground) and not np.isin(np.arange(1, 5), labels).all():
                source = 'Manual' if manual else 'Automatic'
                raise ValueError(source + ' input needs labels 1, 2, 3, 4 for ventricular surfaces: ' + path)
            records.append({'subject': os.path.basename(subject), 'frame': frame,
                            'segmentation': path, 'phase_image': phase_path,
                            'labels': [int(value) for value in labels]})
    return records


def prepare_refinement_inputs(subjects):
    for subject in subjects:
        for directory in ('dofs', 'tmps', 'sizes', 'vtks'):
            path = os.path.join(subject, directory)
            if not os.path.isdir(path):
                os.makedirs(path)
        for frame in FRAMES:
            target = os.path.join(subject, 'lvsa_{0}.nii.gz'.format(frame))
            if not os.path.isfile(target):
                source = phase_image_path(subject, frame)
                shutil.copyfile(source, target)
                print('[DMACS stages] Restored phase image from archive: ' + target)


def reset_prediction_directories(subject):
    """Retain all segs files; the CNN replaces only its two automatic outputs."""
    for name in ('dofs', 'segs', 'tmps', 'sizes', 'motion', 'vtks'):
        path = os.path.join(subject, name)
        if name != 'segs' and os.path.isdir(path):
            shutil.rmtree(path)
        if not os.path.isdir(path):
            os.makedirs(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def existing_segmentations(subjects, include_automatic=False):
    paths = []
    for subject in subjects:
        for frame in FRAMES:
            for manual in ((False, True) if include_automatic else (True,)):
                path = segmentation_path(subject, frame, manual)
                if os.path.isfile(path):
                    paths.append(path)
    return paths


@contextmanager
def protect_files(paths):
    before = dict((path, file_hash(path)) for path in paths)
    try:
        yield before
    finally:
        for path, expected in before.items():
            if not os.path.isfile(path) or file_hash(path) != expected:
                raise RuntimeError('Protected input was removed or changed: ' + path)


def raw_surface_inputs(subjects):
    paths = []
    for subject in subjects:
        for frame in FRAMES:
            for name in SURFACES:
                paths.append(require_file(os.path.join(subject, 'vtks', name + '_' + frame + '.vtk')))
            for name in ('LV', 'RV'):
                paths.append(require_file(os.path.join(subject, 'tmps', 'vtk_' + name + '_' + frame + '.nii.gz')))
    return paths


def validate_stage03_inputs(subjects):
    paths = raw_surface_inputs(subjects)
    for subject in subjects:
        path = require_file(os.path.join(subject, 'lvsa_.nii.gz'))
        image = nib.load(path)
        if len(image.shape) != 4 or image.shape[3] < 2:
            raise ValueError('Motion tracking needs a 4D cine with at least two frames: ' + path)
    return paths


def validate_stage03_outputs(subjects, test_dir):
    for subject in subjects:
        for frame in FRAMES:
            for name in ('F_LVendo', 'F_LVepi', 'F_LVmyo', 'C_RV', 'W_RV', 'W_LVmyo'):
                require_file(os.path.join(subject, 'vtks', name + '_' + frame + '.vtk'))
        frames = nib.load(os.path.join(subject, 'lvsa_.nii.gz')).shape[3]
        for frame in range(frames):
            for extension in ('.vtk', '.txt'):
                require_file(os.path.join(subject, 'motion', 'RV_fr{0:02d}'.format(frame) + extension))
    # The unchanged non-parallel decimator processes only the first valid subject.
    for name in ('RV_fr00_new.vtk', 'RV_fr00_new_new.vtk'):
        require_file(os.path.join(subjects[0], 'motion', name))
    require_file(os.path.join(test_dir, 'matchedpointsnew.txt'))


def record_stage(subjects, stage, inputs, seg_manual=None):
    for subject in subjects:
        record = {'stage': stage, 'status': 'success', 'subject': os.path.basename(subject),
                  'input_sha256': dict((path, value) for path, value in inputs.items()
                                       if os.path.dirname(path).startswith(subject + os.sep))}
        if seg_manual is not None:
            record.update({'seg_manual': bool(seg_manual),
                           'segmentation_source': 'manual' if seg_manual else 'automatic',
                           'segmentation_inputs': {frame: segmentation_path(subject, frame, seg_manual)
                                                   for frame in FRAMES}})
        path = os.path.join(subject, 'dmacs_' + stage + '_summary.json')
        with open(path, 'w') as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        print('[DMACS stages] Saved stage summary: ' + path)
