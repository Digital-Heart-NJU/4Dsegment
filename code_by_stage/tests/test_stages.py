"""Run with the supplied Docker's Python 2 or Python 3 plus mock-compatible APIs."""
from __future__ import print_function

import imp
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    from unittest import mock
except ImportError:
    import mock

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import DMACS
import image_utils
import meshfitting
import p2processing
import stage_utils as stages


class Flags(object):
    stage01_seg_only = False
    stage02_surf_only = False
    stage03_addi_mot_fit = False
    model_path = '/model/checkpoint'
    atlas_dir = '/refs'
    param_dir = '/par'
    template_dir = '/vtks/1'
    template_PH = '/vtks/2'
    coreNo = 8
    irtk = True
    seg_manual = True


class StageTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.subject = os.path.join(self.root, 'subject_test')
        os.makedirs(os.path.join(self.subject, 'segs'))
        self.affine = np.diag([0.9375, 0.9375, 8.0, 1.0])
        self.labels = (np.arange(60).reshape(4, 5, 3) % 5).astype(np.int16)
        for frame in stages.FRAMES:
            self.save_phase(frame)
            self.save_segmentation(frame, manual=False)
            self.save_segmentation(frame, manual=True)

    def save_phase(self, frame):
        image = nib.Nifti1Image(np.zeros((4, 5, 3, 1), dtype=np.int16), self.affine)
        nib.save(image, os.path.join(self.subject, 'lvsa_' + frame + '.nii.gz'))

    def save_segmentation(self, frame='ED', manual=True, labels=None, affine=None):
        image = nib.Nifti1Image(self.labels if labels is None else labels,
                               self.affine if affine is None else affine)
        nib.save(image, stages.segmentation_path(self.subject, frame, manual))

    def test_default_and_individual_stage_flags(self):
        flags = Flags()
        self.assertEqual(stages.select_stage(flags), 'full')
        for name in stages.STAGE_FLAGS:
            selected = Flags()
            setattr(selected, name, True)
            self.assertEqual(stages.select_stage(selected), name)

    def test_conflicting_flags_rejected(self):
        flags = Flags()
        flags.stage01_seg_only = flags.stage02_surf_only = True
        with self.assertRaises(ValueError):
            stages.select_stage(flags)

    def test_manual_inputs_are_explicit_and_valid(self):
        records = stages.validate_segmentations([self.subject], manual=True)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(item['segmentation'].endswith('_manual.nii.gz') for item in records))

    def test_missing_manual_has_no_fallback(self):
        os.unlink(stages.segmentation_path(self.subject, 'ES', True))
        with self.assertRaises(IOError):
            stages.validate_segmentations([self.subject], manual=True)

    def test_geometry_mismatch_rejected(self):
        shifted = self.affine.copy()
        shifted[0, 3] = 10
        self.save_segmentation(affine=shifted)
        with self.assertRaises(ValueError):
            stages.validate_segmentations([self.subject], manual=True)

    def test_dimensions_mismatch_rejected(self):
        self.save_segmentation(labels=np.zeros((3, 5, 3), dtype=np.int16))
        with self.assertRaises(ValueError):
            stages.validate_segmentations([self.subject], manual=True)

    def test_invalid_noninteger_nan_and_out_of_range_labels(self):
        for value in (0.5, 5, np.nan):
            labels = self.labels.astype(np.float32)
            labels[0, 0, 0] = value
            self.save_segmentation(labels=labels)
            with self.assertRaises(ValueError):
                stages.validate_segmentations([self.subject], manual=True)

    def test_singleton_time_manual_is_accepted(self):
        self.save_segmentation(labels=self.labels[..., None])
        stages.validate_segmentations([self.subject], manual=True)

    def test_bad_cnn_prediction_can_still_be_saved_for_manual_editing(self):
        self.save_segmentation(manual=False, labels=np.zeros_like(self.labels))
        stages.validate_segmentations([self.subject], manual=False)
        self.save_segmentation(labels=np.zeros_like(self.labels))
        with self.assertRaises(ValueError):
            stages.validate_segmentations([self.subject], manual=True)

    def run_selected_stage02(self, manual):
        flags = Flags()
        flags.test_dir = self.root
        flags.stage02_surf_only = True
        flags.seg_manual = manual
        selected = []

        def refine(*args, **kwargs):
            self.assertEqual(kwargs['segmentation_suffix'], '_manual' if manual else '')
            for frame in stages.FRAMES:
                path = stages.segmentation_path(self.subject, frame, manual)
                selected.append(path)
                with mock.patch.object(image_utils.os, 'system', return_value=0):
                    image_utils.moveVolumes(self.subject, os.path.join(self.subject, 'sizes'), frame,
                                            segmentation_input=path)

        with mock.patch.object(DMACS, 'multiatlasreg3D', side_effect=refine), \
                mock.patch.object(DMACS, 'meshCoregstration') as mesh, \
                mock.patch.object(DMACS, 'raw_surface_inputs', return_value=[]), \
                mock.patch.object(DMACS, 'deeplearningseg') as inference:
            DMACS.run_pipeline(flags)
            inference.assert_not_called()
            self.assertEqual(mesh.call_args[1], {'mode': 'extract'})
        for frame, path in zip(stages.FRAMES, selected):
            archive = os.path.join(self.subject, 'sizes', '2D_segmentation_' + frame + '.nii.gz')
            self.assertEqual(stages.file_hash(path), stages.file_hash(archive))
        with open(os.path.join(self.subject, 'dmacs_stage02_surf_only_summary.json')) as handle:
            record = json.load(handle)
        self.assertEqual(record['seg_manual'], manual)
        self.assertEqual(record['segmentation_source'], 'manual' if manual else 'automatic')
        self.assertEqual(record['segmentation_inputs'], dict(zip(stages.FRAMES, selected)))
        return record

    def test_stage02_automatic_input_does_not_require_manual_files(self):
        for frame in stages.FRAMES:
            os.unlink(stages.segmentation_path(self.subject, frame, True))
        self.run_selected_stage02(False)
        for frame in stages.FRAMES:
            self.assertFalse(os.path.exists(stages.segmentation_path(self.subject, frame, True)))

    def test_stage02_automatic_input_ignores_different_manual_labels(self):
        for frame in stages.FRAMES:
            self.save_segmentation(frame, manual=True, labels=self.labels[::-1].copy())
        self.run_selected_stage02(False)

    def test_stage02_manual_input_ignores_different_automatic_labels(self):
        for frame in stages.FRAMES:
            self.save_segmentation(frame, manual=False, labels=self.labels[::-1].copy())
        self.run_selected_stage02(True)

    def test_stage02_manual_input_does_not_require_automatic_files(self):
        for frame in stages.FRAMES:
            os.unlink(stages.segmentation_path(self.subject, frame, False))
        self.run_selected_stage02(True)

    def test_stage02_selected_source_missing_never_falls_back(self):
        for manual in (False, True):
            path = stages.segmentation_path(self.subject, 'ES', manual)
            os.unlink(path)
            flags = Flags()
            flags.test_dir, flags.stage02_surf_only, flags.seg_manual = self.root, True, manual
            with mock.patch.object(DMACS, 'multiatlasreg3D') as refine, \
                    mock.patch.object(DMACS, 'meshCoregstration') as mesh:
                with self.assertRaises(IOError):
                    DMACS.run_pipeline(flags)
                refine.assert_not_called()
                mesh.assert_not_called()
            self.save_segmentation('ES', manual=manual)

    def test_stage02_automatic_input_requires_all_foreground_classes(self):
        self.save_segmentation('ES', manual=False, labels=np.zeros_like(self.labels))
        flags = Flags()
        flags.test_dir, flags.stage02_surf_only, flags.seg_manual = self.root, True, False
        with mock.patch.object(DMACS, 'multiatlasreg3D') as refine, \
                mock.patch.object(DMACS, 'meshCoregstration') as mesh:
            with self.assertRaises(ValueError):
                DMACS.run_pipeline(flags)
            refine.assert_not_called()
            mesh.assert_not_called()

    def test_legacy_cli_parses_seg_manual_default_true_and_false(self):
        code = 'import DMACS; print("SEG_MANUAL=" + str(DMACS.FLAGS.seg_manual))'
        for argument, expected in (([], 'True'), (['--seg_manual=true'], 'True'), (['--seg_manual=false'], 'False')):
            output = subprocess.check_output([sys.executable, '-B', '-c', code] + argument)
            self.assertIn('SEG_MANUAL=' + expected, output.decode('utf-8'))

    def test_archived_phases_are_validated_then_restored(self):
        sizes = os.path.join(self.subject, 'sizes')
        os.makedirs(sizes)
        source = os.path.join(self.subject, 'lvsa_ED.nii.gz')
        archive = os.path.join(sizes, 'lvsa_ED.nii.gz')
        shutil.move(source, archive)
        stages.validate_segmentations([self.subject], manual=True)
        self.assertFalse(os.path.exists(source))
        stages.prepare_refinement_inputs([self.subject])
        self.assertEqual(stages.file_hash(source), stages.file_hash(archive))

    def test_prediction_reset_preserves_manual_files_and_notes(self):
        hashes = dict((p, stages.file_hash(p)) for p in stages.existing_segmentations([self.subject]))
        note = os.path.join(self.subject, 'segs', 'notes.txt')
        with open(note, 'w') as handle:
            handle.write('retain')
        for name in ('dofs', 'tmps', 'sizes', 'motion', 'vtks'):
            os.makedirs(os.path.join(self.subject, name))
        stages.reset_prediction_directories(self.subject)
        for path, digest in hashes.items():
            self.assertEqual(stages.file_hash(path), digest)
        self.assertTrue(os.path.isfile(note))

    def test_file_protection_detects_mutation(self):
        path = stages.segmentation_path(self.subject, 'ED', True)
        with self.assertRaises(RuntimeError):
            with stages.protect_files([path]):
                with open(path, 'ab') as handle:
                    handle.write(b'changed')

    def test_stage03_preflight_rejects_missing_raw_files(self):
        with self.assertRaises(IOError):
            stages.validate_stage03_inputs([self.subject])

    def test_stage_dispatch_preserves_full_default_and_boundaries(self):
        names = ('deeplearningseg', 'multiatlasreg3D', 'meshCoregstration', 'motionTracking',
                 'decimate', 'validate_segmentations', 'prepare_refinement_inputs',
                 'raw_surface_inputs', 'validate_stage03_inputs', 'validate_stage03_outputs',
                 'record_stage')
        for stage in ('full',) + stages.STAGE_FLAGS:
            flags = Flags()
            flags.test_dir = self.root
            if stage != 'full':
                setattr(flags, stage, True)
            with mock.patch.multiple(DMACS, **dict((name, mock.DEFAULT) for name in names)) as calls:
                calls['validate_segmentations'].return_value = []
                calls['validate_stage03_inputs'].return_value = []
                DMACS.run_pipeline(flags)
                self.assertEqual(calls['deeplearningseg'].call_count, int(stage in ('full', 'stage01_seg_only')))
                self.assertEqual(calls['multiatlasreg3D'].call_count, int(stage in ('full', 'stage02_surf_only')))
                self.assertEqual(calls['meshCoregstration'].call_count, int(stage != 'stage01_seg_only'))
                self.assertEqual(calls['motionTracking'].call_count, int(stage in ('full', 'stage03_addi_mot_fit')))
                self.assertEqual(calls['decimate'].call_count, int(stage in ('full', 'stage03_addi_mot_fit')))
                if stage == 'full':
                    self.assertEqual(calls['multiatlasreg3D'].call_args[1], {})
                    self.assertEqual(calls['meshCoregstration'].call_args[1], {})
                elif stage == 'stage02_surf_only':
                    self.assertEqual(calls['multiatlasreg3D'].call_args[1], {'segmentation_suffix': '_manual'})
                    self.assertEqual(calls['meshCoregstration'].call_args[1], {'mode': 'extract'})
                    self.assertEqual(calls['validate_segmentations'].call_args[1],
                                     {'manual': True, 'require_foreground': True})
                elif stage == 'stage03_addi_mot_fit':
                    self.assertEqual(calls['meshCoregstration'].call_args[1], {'mode': 'fit'})
                self.assertEqual(calls['record_stage'].call_count, 1)

    def test_atlas_worker_and_nonparallel_branch_use_manual_paths(self):
        helpers = ('topSimilarAtlasShapeSelection', 'formHighResolutionImg', 'output3DRefinement',
                   'refineFusionResults', 'convertImageSegment', 'outputVolumes', 'moveVolumes',
                   'allAtlasShapeSelection')
        for parallel in (False, True):
            with mock.patch.multiple(p2processing, **dict((name, mock.DEFAULT) for name in helpers)) as calls:
                calls['topSimilarAtlasShapeSelection'].return_value = ([], [])
                atlases = dict((frame, []) for frame in stages.FRAMES)
                if parallel:
                    p2processing.apply_PC('subject_test', self.root, '/par', atlases, atlases,
                                          True, segmentation_suffix='_manual')
                else:
                    calls['allAtlasShapeSelection'].return_value = (atlases, atlases)
                    p2processing.multiatlasreg3D(self.root, '/refs', '/par', 8, False, True,
                                               segmentation_suffix='_manual')
                for frame, call in zip(stages.FRAMES, calls['topSimilarAtlasShapeSelection'].call_args_list):
                    self.assertEqual(call[0][5], stages.segmentation_path(self.subject, frame, True))
                for frame, call in zip(stages.FRAMES, calls['moveVolumes'].call_args_list):
                    self.assertEqual(call[1]['segmentation_input'], stages.segmentation_path(self.subject, frame, True))

    def test_archive_copies_actual_manual_input_without_deleting_either_segmentation(self):
        sizes = os.path.join(self.subject, 'sizes')
        os.makedirs(sizes)
        path = stages.segmentation_path(self.subject, 'ED', True)
        with mock.patch.object(image_utils.os, 'system', return_value=0):
            image_utils.moveVolumes(self.subject, sizes, 'ED', segmentation_input=path)
        self.assertEqual(stages.file_hash(path), stages.file_hash(os.path.join(sizes, '2D_segmentation_ED.nii.gz')))
        self.assertTrue(os.path.isfile(stages.segmentation_path(self.subject, 'ED', False)))

    def test_mesh_command_parity_and_fit_has_no_cleanup_or_extraction(self):
        original_root = os.environ.get('DMACS_ORIGINAL_CODE', os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'code')))
        original = imp.load_source('original_meshfitting', os.path.join(original_root, 'meshfitting.py'))
        args = ('/subject', '/template', '/par', '/subject/tmps', '/subject/vtks', '/subject/dofs')
        recorded = {}
        for name, function in (('original', original.meshGeneration), ('all', meshfitting.meshGeneration),
                               ('extract', meshfitting.extractSurfaceMeshes), ('fit', meshfitting.fitSurfaceMeshes)):
            recorded[name] = []
            with mock.patch.object(os, 'system', side_effect=lambda command: recorded[name].append(command) or 0):
                function(*args)
        self.assertEqual(recorded['original'], recorded['all'])
        self.assertEqual(recorded['all'], recorded['extract'] + recorded['fit'])
        self.assertTrue(any(command.startswith('mcubes ') for command in recorded['extract']))
        self.assertFalse(any(command.startswith(('rm ', 'mcubes ', 'binarize ')) for command in recorded['fit']))
        with self.assertRaises(ValueError):
            meshfitting.meshGeneration(*args, mode='invalid')


if __name__ == '__main__':
    unittest.main(verbosity=2)
