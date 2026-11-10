from __future__ import print_function

import os
import sys
import time


def requested_segmentation_backend(argv):
    for index, item in enumerate(argv):
        if item.startswith('--segmentation_backend='):
            return item.split('=', 1)[1]
        if item == '--segmentation_backend' and index + 1 < len(argv):
            return argv[index + 1]
    return 'legacy'


# Route before importing TensorFlow: nnUNet runs in modern host Python, while
# the legacy implementation and registration tools remain in their Docker.
if __name__ == '__main__' and requested_segmentation_backend(sys.argv[1:]) == 'nnunet':
    if sys.version_info < (3, 9):
        sys.exit('Run the nnunet backend in the modern host environment (cme-dt-env), not legacy Docker Python.')
    # Spawned nnUNet workers must import the modern launcher, not this
    # TensorFlow entry point. Replace this process before inference begins.
    launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nnunet_pipeline.py')
    os.execv(sys.executable, [sys.executable, '-B', '-u', launcher] + sys.argv[1:])

import tensorflow as tf
from deepseg import deeplearningseg
from p2processing import multiatlasreg3D
from meshfitting import meshCoregstration
from motionEstimation import motionTracking
from decimation import decimate
from stage_utils import (select_stage, subject_directories, validate_segmentations,
                         prepare_refinement_inputs, existing_segmentations, protect_files,
                         raw_surface_inputs, validate_stage03_inputs,
                         validate_stage03_outputs, record_stage)

""" Deployment parameters """
FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('coreNo', 3, 'Number of CPUs.')
#tf.app.flags.DEFINE_string('test_dir', '/vol/medic02/users/jduan/New_BRIDGE_Cases4',
#                           'Path to the test set directory, under which images are organised in '
#                           'subdirectories for each subject.')
#tf.app.flags.DEFINE_string('model_path', '/vol/medic02/users/jduan/HHData/tensorflowFCNCodes/DeepRegionEdgeSegmentation'
#                           '/saver/model/vgg_RE_network/vgg_RE_network.ckpt-50000', 'Path to the saved trained model.')
#tf.app.flags.DEFINE_string('atlas_dir', '/vol/medic02/users/jduan/HHData/3Dshapes_new', 'Path to the atlas.')
#tf.app.flags.DEFINE_string('param_dir', '/vol/medic02/users/jduan/myPatchMatch/par', 'Path to the registration parameters.')
#tf.app.flags.DEFINE_string('template_dir', '/vol/medic02/users/jduan/myPatchMatch/template_wenzhe', 'Path to the template.')
#tf.app.flags.DEFINE_string('template_PH', '/vol/medic02/users/jduan/myPatchMatch/vtks', 'Path to the template.')



tf.app.flags.DEFINE_integer('coreNo', 8, 'Number of CPUs.')
tf.app.flags.DEFINE_string('test_dir', '/data',
                           'Path to the test set directory, under which images are organised in '
                           'subdirectories for each subject.')
tf.app.flags.DEFINE_string('model_path',  '/model/vgg_RE_network.ckpt-50000', 'Path to the saved trained model.')
tf.app.flags.DEFINE_string('atlas_dir',  '/refs', 'Path to the atlas.')
tf.app.flags.DEFINE_string('param_dir', '/par', 'Path to the registration parameters.')
tf.app.flags.DEFINE_string('template_dir', '/vtks/1', 'Path to the template.')
tf.app.flags.DEFINE_string('template_PH', '/vtks/2', 'Path to the template.')
tf.app.flags.DEFINE_boolean('irtk', True, 'use irtk or not')
tf.app.flags.DEFINE_boolean('stage01_seg_only', False, 'Preprocess and segment ED/ES, then stop.')
tf.app.flags.DEFINE_boolean('stage02_surf_only', False, 'Refine selected ED/ES segmentations and extract raw surfaces, then stop.')
tf.app.flags.DEFINE_boolean('seg_manual', False, 'Stage 02 input: true selects _manual labels; false selects automatic labels.')
tf.app.flags.DEFINE_boolean('stage03_addi_mot_fit', False, 'Fit saved raw surfaces, compute measurements and motion, then decimate.')
tf.app.flags.DEFINE_string('segmentation_backend', 'legacy', 'legacy in Docker, or nnunet through modern host Python.')


def run_pipeline(flags):
    if getattr(flags, 'segmentation_backend', 'legacy') != 'legacy':
        raise ValueError('Unknown segmentation backend; choose legacy or nnunet.')
    stage = select_stage(flags)
    subjects = subject_directories(flags.test_dir)
    print('[DMACS stages] Mode: ' + stage)
    start_time = time.time()
    protected = existing_segmentations(subjects, include_automatic=stage in (
        'stage02_surf_only', 'stage03_addi_mot_fit'))
    with protect_files(protected) as original_hashes:
        recorded_hashes = dict(original_hashes)
        if stage in ('full', 'stage01_seg_only'):
            deeplearningseg(flags.model_path, flags.test_dir, flags.atlas_dir)

        if stage == 'stage01_seg_only':
            validate_segmentations(subjects)
        elif stage == 'stage02_surf_only':
            seg_manual = getattr(flags, 'seg_manual', True)
            records = validate_segmentations(subjects, manual=seg_manual, require_foreground=True)
            for record in records:
                source = 'Manual' if seg_manual else 'Automatic'
                print('[DMACS stages] ' + source + ' refinement input: ' + record['segmentation'])
            prepare_refinement_inputs(subjects)
            multiatlasreg3D(flags.test_dir, flags.atlas_dir, flags.param_dir,
                            flags.coreNo, True, flags.irtk, segmentation_suffix='_manual' if seg_manual else '')
            meshCoregstration(flags.test_dir, flags.param_dir, flags.template_dir,
                             flags.coreNo, True, False, mode='extract')
            raw_surface_inputs(subjects)
        elif stage == 'stage03_addi_mot_fit':
            inputs = validate_stage03_inputs(subjects)
            with protect_files(inputs) as raw_hashes:
                meshCoregstration(flags.test_dir, flags.param_dir, flags.template_dir,
                                 flags.coreNo, True, False, mode='fit')
                motionTracking(flags.test_dir, flags.param_dir, flags.template_PH, flags.coreNo, True)
                decimate(flags.test_dir, flags.coreNo, False)
                validate_stage03_outputs(subjects, flags.test_dir)
                recorded_hashes.update(raw_hashes)
        else:
            multiatlasreg3D(flags.test_dir, flags.atlas_dir, flags.param_dir, flags.coreNo, True, flags.irtk)
            meshCoregstration(flags.test_dir, flags.param_dir, flags.template_dir, flags.coreNo, True, False)
            motionTracking(flags.test_dir, flags.param_dir, flags.template_PH, flags.coreNo, True)
            decimate(flags.test_dir, flags.coreNo, False)
            validate_stage03_outputs(subjects, flags.test_dir)
    if stage == 'stage02_surf_only':
        record_stage(subjects, stage, recorded_hashes, seg_manual=seg_manual)
    else:
        record_stage(subjects, stage, recorded_hashes)
    if stage == 'stage01_seg_only':
        print('[DMACS stages] Stage 01 complete. Save edited labels as segs/seg_lvsa_ED_manual.nii.gz and seg_lvsa_ES_manual.nii.gz.')
    elif stage == 'stage02_surf_only':
        print('[DMACS stages] Stage 02 complete. Raw ED/ES surfaces and fitting masks are saved; fitting and motion were not run.')
    print('[DMACS stages] {0} completed in {1:.3f}s.'.format(stage, time.time() - start_time))


if __name__ == '__main__':
    try:
        run_pipeline(FLAGS)
    except Exception as error:
        print('[DMACS stages] ERROR: ' + str(error), file=sys.stderr)
        raise
