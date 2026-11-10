"""Modern-backend tests: unittest discover -s tests -p nnunet_checks.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_rv_myo_xy import remap_nnunet_labels, expand_rv_myo_xy, save_labels
import nnunet_pipeline
import prepare_manual_fixture
import validate_nnunet


class XYWallTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((15, 15, 4), dtype=np.int16)
        self.labels[7, 7, 1] = 4
        self.affine = np.diag([1., 1., 8., 1.])

    def expand(self, data=None, affine=None, **kwargs):
        return expand_rv_myo_xy(self.labels if data is None else data,
                                self.affine if affine is None else affine, **kwargs)

    def test_remap_is_not_in_place_and_reserves_label_three(self):
        data = np.array([0, 1, 2, 3], dtype=np.int16)
        np.testing.assert_array_equal(remap_nnunet_labels(data), [0, 1, 2, 4])
        np.testing.assert_array_equal(data, [0, 1, 2, 3])

    def test_growth_is_xy_only_and_empty_slices_unchanged(self):
        result, added, report = self.expand()
        self.assertTrue(added[:, :, 1].any())
        for z in (0, 2, 3):
            self.assertFalse(added[:, :, z].any())
            np.testing.assert_array_equal(result[:, :, z], self.labels[:, :, z])
        self.assertTrue(report['no_Z_growth'])

    def test_radius_is_euclidean_millimeters_not_pixels(self):
        _, added, _ = self.expand(affine=np.diag([1.5, 1., 8., 1.]))
        self.assertTrue(added[9, 7, 1])
        self.assertFalse(added[10, 7, 1])
        self.assertTrue(added[7, 10, 1])
        self.assertFalse(added[9, 8, 1])

    def test_all_existing_nonzero_labels_preserved(self):
        self.labels[8, 7, 1] = 1
        self.labels[6, 7, 1] = 2
        self.labels[7, 8, 1] = 3
        result, _, _ = self.expand()
        np.testing.assert_array_equal(result[self.labels != 0], self.labels[self.labels != 0])

    def test_background_behind_blocking_label_is_not_an_island(self):
        self.labels[8, :, 1] = 2
        result, _, report = self.expand()
        self.assertEqual(result[9, 7, 1], 0)
        self.assertGreater(report['slices'][1]['disconnected_candidates_removed'], 0)

    def test_z_spacing_does_not_affect_xy_growth(self):
        first = self.expand()[0]
        other = self.expand(affine=np.diag([1., 1., 100., 1.]))[0]
        np.testing.assert_array_equal(first, other)

    def test_rotated_and_reflected_affines(self):
        expected = self.expand()[0]
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        matrix = self.affine.copy()
        matrix[:3, :3] = rotation @ matrix[:3, :3]
        np.testing.assert_array_equal(expected, self.expand(affine=matrix)[0])
        matrix[:3, 0] *= -1
        np.testing.assert_array_equal(expected, self.expand(affine=matrix)[0])

    def test_sheared_affine_uses_physical_positions(self):
        matrix = self.affine.copy()
        matrix[0, 1] = 0.5
        _, added, report = self.expand(affine=matrix)
        self.assertEqual(report['distance_method'], 'inplane_world_coordinate_KDTree')
        coords = np.argwhere(added[:, :, 1]) - [7, 7]
        distances = np.linalg.norm(coords @ matrix[:3, :2].T, axis=1)
        self.assertLessEqual(distances.max(), 3.0 + 1e-8)

    def test_known_units_and_explicit_unknown_unit_assumption(self):
        expected = self.expand()[0]
        matrix = self.affine.copy()
        matrix[:3, :3] /= 1000
        np.testing.assert_array_equal(expected, self.expand(affine=matrix, spatial_unit='meter')[0])
        with self.assertRaises(ValueError):
            self.expand(spatial_unit='unknown')
        result, _, report = self.expand(spatial_unit='unknown', assume_mm=True)
        np.testing.assert_array_equal(expected, result)
        self.assertTrue(report['assumed_mm'])

    def test_empty_rv_and_idempotence(self):
        empty = np.zeros_like(self.labels)
        np.testing.assert_array_equal(self.expand(data=empty)[0], empty)
        first = self.expand()[0]
        second, added, _ = self.expand(data=first)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(added.any())

    def test_invalid_labels_affine_radius_and_dimension(self):
        for value in (0.5, 5, np.nan):
            data = self.labels.astype(float)
            data[0, 0, 0] = value
            with self.assertRaises(ValueError):
                self.expand(data=data)
        for radius in (0, -1, np.inf):
            with self.assertRaises(ValueError):
                self.expand(radius_mm=radius)
        with self.assertRaises(ValueError):
            self.expand(affine=np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            self.expand(data=self.labels[..., None])

    def test_image_edge_does_not_change_field_of_view(self):
        data = np.zeros_like(self.labels)
        data[0, 0, 1] = 4
        result, _, report = self.expand(data=data)
        self.assertEqual(result.shape, data.shape)
        self.assertTrue(report['slices'][1]['growth_or_RV_touches_image_edge'])

    def test_saved_labels_preserve_geometry_dtype_forms_and_values(self):
        source = nib.Nifti1Image(np.zeros(self.labels.shape + (1,), dtype=np.int16), self.affine)
        source.header.set_xyzt_units('mm', 'sec')
        source.set_qform(self.affine, code=1)
        source.set_sform(self.affine, code=2)
        labels = self.expand()[0]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'labels.nii.gz'
            save_labels(path, labels, source)
            image = nib.load(path)
            np.testing.assert_array_equal(np.asarray(image.dataobj), labels)
            np.testing.assert_allclose(image.affine, source.affine)
            self.assertEqual(image.get_data_dtype(), np.dtype('int16'))
            self.assertEqual(image.header.get_xyzt_units(), ('mm', 'sec'))
            self.assertEqual(int(image.header['qform_code']), 1)
            self.assertEqual(int(image.header['sform_code']), 2)
            with self.assertRaises(FileExistsError):
                save_labels(path, labels, source)


class LauncherTests(unittest.TestCase):
    def test_modern_launcher_can_be_imported_by_spawned_worker(self):
        entry = Path(__file__).resolve().parents[1] / 'nnunet_pipeline.py'
        code = 'import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2], run_name="__mp_main__")'
        result = subprocess.run([sys.executable, '-B', '-c', code, str(entry.parent), str(entry)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_modern_entry_help_does_not_require_tensorflow(self):
        entry = Path(__file__).resolve().parents[1] / 'DMACS.py'
        result = subprocess.run([sys.executable, '-B', str(entry), '--segmentation_backend=nnunet', '--help'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--rv-wall-radius-mm', result.stdout)

    def test_conflicting_true_flags_rejected(self):
        with self.assertRaises(SystemExit):
            nnunet_pipeline.main(['--subject', 'unused', '--stage01_seg_only=true', '--stage02_surf_only=true'])

    def test_false_flag_does_not_conflict_with_true_flag(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'subject_test').mkdir()
            with mock.patch.object(nnunet_pipeline, 'run_logged') as call:
                nnunet_pipeline.main(['--data-dir', root, '--subject', 'subject_test',
                                      '--run-dir', str(Path(root) / 'run'),
                                      '--stage01_seg_only=false', '--stage03_addi_mot_fit=true'])
            self.assertEqual(call.call_count, 1)
            self.assertIn('--stage03_addi_mot_fit=true', call.call_args[0][0])

    def test_full_mode_uses_automatic_refinement_without_manual_copies(self):
        with tempfile.TemporaryDirectory() as root:
            subject = Path(root) / 'subject_test'
            subject.mkdir()
            with mock.patch.object(nnunet_pipeline, 'stage01', return_value={'all_foreground_classes_present': True}), \
                    mock.patch.object(nnunet_pipeline, 'run_logged') as call:
                nnunet_pipeline.main(['--data-dir', root, '--subject', subject.name,
                                      '--run-dir', str(Path(root) / 'run')])
            self.assertEqual(call.call_count, 2)
            self.assertIn('refine_nnunet_automatic.py', call.call_args_list[0][0][0])
            self.assertIn('--stage03_addi_mot_fit=true', call.call_args_list[1][0][0])
            self.assertFalse(list(subject.rglob('*_manual.nii.gz')))

    def test_full_mode_stops_before_refinement_when_prediction_is_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'subject_test').mkdir()
            with mock.patch.object(nnunet_pipeline, 'stage01', return_value={'all_foreground_classes_present': False}), \
                    mock.patch.object(nnunet_pipeline, 'run_logged') as call:
                with self.assertRaisesRegex(RuntimeError, 'predictions were saved but are incomplete'):
                    nnunet_pipeline.main(['--data-dir', root, '--subject', 'subject_test',
                                          '--run-dir', str(Path(root) / 'run')])
            call.assert_not_called()


class TerminalSliceTests(unittest.TestCase):
    def prepare(self, z_count=5, occupied=(0, 2, 4)):
        mapped = np.zeros((11, 11, z_count), dtype=np.int16)
        for z in occupied:
            mapped[3, 3, z], mapped[4, 3, z], mapped[7, 7, z] = 1, 2, 4
        grown, added, report = expand_rv_myo_xy(mapped, np.diag([1., 1., 8., 1.]))
        return mapped, grown, added, report

    def test_clears_all_labels_at_both_stack_ends_only(self):
        mapped, grown, added, report = self.prepare()
        original_grown, original_added, original_report = grown.copy(), added.copy(), json.dumps(report, sort_keys=True)
        final, retained, summary = nnunet_pipeline.finalize_stage01_labels(mapped, grown, added, report, 5)
        self.assertFalse(final[:, :, 0].any())
        self.assertFalse(final[:, :, -1].any())
        self.assertFalse(retained[:, :, (0, 4)].any())
        np.testing.assert_array_equal(final[:, :, 1:-1], grown[:, :, 1:-1])
        np.testing.assert_array_equal(retained[:, :, 1:-1], added[:, :, 1:-1])
        np.testing.assert_array_equal(grown, original_grown)
        np.testing.assert_array_equal(added, original_added)
        self.assertEqual(json.dumps(report, sort_keys=True), original_report)
        self.assertEqual(final.shape, grown.shape)
        self.assertEqual(summary['terminal_z_cleanup']['slice_indices'], [0, 4])
        self.assertEqual(summary['terminal_z_cleanup']['removed_foreground_voxels'],
                         int(np.count_nonzero(grown[:, :, (0, 4)])))
        self.assertFalse(summary['protected_labels_unchanged'])
        self.assertTrue(summary['nonterminal_protected_labels_unchanged'])
        self.assertEqual(summary['added_voxels'], int(retained.sum()))
        self.assertEqual(sum(row['added_voxels'] for row in summary['slices']), int(retained.sum()))

    def test_does_not_clear_first_or_last_foreground_slice_inside_stack(self):
        mapped, grown, added, report = self.prepare(occupied=(1, 3))
        final, retained, summary = nnunet_pipeline.finalize_stage01_labels(mapped, grown, added, report, 5)
        np.testing.assert_array_equal(final, grown)
        np.testing.assert_array_equal(retained, added)
        self.assertTrue(final[:, :, 1].any())
        self.assertTrue(final[:, :, 3].any())
        self.assertEqual(summary['terminal_z_cleanup']['removed_foreground_voxels'], 0)

    def test_clears_each_terminal_slice_independently(self):
        for occupied in ((0,), (4,)):
            mapped, grown, added, report = self.prepare(occupied=occupied)
            final, _, summary = nnunet_pipeline.finalize_stage01_labels(mapped, grown, added, report, 5)
            self.assertFalse(final.any())
            rows = summary['terminal_z_cleanup']['slices']
            self.assertEqual(sum(row['foreground_voxels_removed'] > 0 for row in rows), 1)

    def test_single_and_two_slice_stacks_are_cleared_without_double_counting(self):
        for count in (1, 2):
            mapped, grown, added, report = self.prepare(count, tuple(range(count)))
            final, retained, summary = nnunet_pipeline.finalize_stage01_labels(mapped, grown, added, report, count)
            self.assertFalse(final.any())
            self.assertFalse(retained.any())
            self.assertEqual(len(summary['terminal_z_cleanup']['slice_indices']), count)
            self.assertEqual(summary['terminal_z_cleanup']['removed_foreground_voxels'], int(np.count_nonzero(grown)))

    def test_rejects_original_stack_or_mask_shape_mismatch(self):
        mapped, grown, added, report = self.prepare()
        with self.assertRaises(ValueError):
            nnunet_pipeline.finalize_stage01_labels(mapped, grown, added, report, 6)
        with self.assertRaises(ValueError):
            nnunet_pipeline.finalize_stage01_labels(mapped, grown, added[:, :, :-1], report, 5)

    def test_empty_stack_labels_remain_empty(self):
        mapped, grown, added, report = self.prepare(occupied=())
        final, retained, summary = nnunet_pipeline.finalize_stage01_labels(mapped, grown, added, report, 5)
        self.assertFalse(final.any())
        self.assertFalse(retained.any())
        self.assertEqual(summary['terminal_z_cleanup']['removed_foreground_voxels'], 0)


class Stage01FileTests(unittest.TestCase):
    def run_stage01(self, root, with_anatomy):
        root = Path(root)
        subject, model, run = root / 'subject_test', root / 'model', root / 'run'
        subject.mkdir()
        model.mkdir()
        run.mkdir()
        (model / 'dataset.json').write_text(json.dumps({'labels': {
            'background': 0, 'LV blood pool': 1, 'LV myocardium': 2, 'RV blood pool': 3}}))
        (model / 'plans.json').write_text(json.dumps({'configurations': {'2d': {}}}))
        (model / 'fold_0').mkdir()
        (model / 'fold_0/checkpoint_final.pth').write_bytes(b'not loaded in this unit test')
        shape, affine = (17, 17, 3), np.diag([1., 1., 8., 1.])
        phase = nib.Nifti1Image(np.ones(shape + (1,), dtype=np.int16), affine)
        phase.header.set_xyzt_units('mm', 'sec')
        nib.save(phase, subject / 'example.nii')
        source_hash = nnunet_pipeline.sha256(subject / 'example.nii')
        (subject / 'segs').mkdir()
        manual = subject / 'segs/seg_lvsa_ED_manual.nii.gz'
        nib.save(phase, manual)
        manual_hash = nnunet_pipeline.sha256(manual)

        def preprocess(*unused):
            for frame in ('ED', 'ES'):
                nib.save(phase, subject / ('lvsa_' + frame + '.nii.gz'))

        def predict(args, inputs, outputs):
            for frame in ('ED', 'ES'):
                native = np.zeros(shape, dtype=np.uint8)
                if with_anatomy:
                    for z in ((0, 1, 2) if frame == 'ED' else (1, 2)):
                        native[4, 4, z], native[5, 4, z], native[9, 9, z] = 1, 2, 3
                nib.save(nib.Nifti1Image(native, affine), outputs / (subject.name + '_sax_' + frame + '.nii.gz'))

        from argparse import Namespace
        args = Namespace(input_name='example.nii', nnunet_model_dir=model, folds=[0],
                         checkpoint='checkpoint_final.pth', overwrite=False, coreNo=1,
                         docker_image='unused', rv_wall_radius_mm=3., assume_mm_if_unknown=False)
        with mock.patch.object(nnunet_pipeline, 'run_logged', side_effect=preprocess), \
                mock.patch.object(nnunet_pipeline, 'infer_cases', side_effect=predict):
            report = nnunet_pipeline.stage01(args, subject, run)
        self.assertEqual(source_hash, nnunet_pipeline.sha256(subject / 'example.nii'))
        self.assertEqual(manual_hash, nnunet_pipeline.sha256(manual))
        self.assertFalse((subject / 'segs/seg_lvsa_ES_manual.nii.gz').exists())
        for frame in ('ED', 'ES'):
            final = nib.load(subject / 'segs' / ('seg_lvsa_' + frame + '.nii.gz'))
            self.assertEqual(final.shape, shape)
            labels = np.asarray(final.dataobj)
            self.assertFalse(labels[:, :, 0].any())
            self.assertFalse(labels[:, :, -1].any())
            if with_anatomy:
                self.assertEqual(labels[9, 9, 1], 4)
                self.assertEqual(labels[10, 9, 1], 3)
                self.assertEqual(set(np.unique(labels)), {0, 1, 2, 3, 4})
                self.assertFalse((labels[:, :, 0] == 3).any())
                self.assertGreater(report['frames'][frame]['terminal_z_cleanup']['removed_foreground_voxels'], 0)
            else:
                self.assertFalse(labels.any())
                self.assertEqual(report['frames'][frame]['added_voxels'], 0)
                self.assertTrue(report['frames'][frame]['missing_predicted_anatomy'])
        validate_nnunet.verify(subject, run / 'validation_stage01.json')
        return report

    def test_native_prediction_to_five_labels_preserves_manual_and_original_files(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self.run_stage01(root, True)['status'], 'success')

    def test_empty_prediction_is_saved_for_manual_correction_without_inventing_labels(self):
        with tempfile.TemporaryDirectory() as root:
            report = self.run_stage01(root, False)
            self.assertEqual(report['status'], 'completed_requires_manual_correction')
            self.assertFalse(report['all_foreground_classes_present'])


class SameSubjectCopyTests(unittest.TestCase):
    def make_subject(self, root):
        subject = Path(root) / 'subject_test'
        (subject / 'segs').mkdir(parents=True)
        labels = np.zeros((9, 9, 3), dtype=np.int16)
        labels[3, 3, 1], labels[3, 4, 1], labels[4, 3, 1], labels[4, 4, 1] = 1, 2, 3, 4
        affine = np.diag([1., 1., 8., 1.])
        phase = nib.Nifti1Image(np.ones(labels.shape + (1,), dtype=np.int16), affine)
        nib.save(phase, subject / 'example.nii')
        for frame in ('ED', 'ES'):
            nib.save(phase, subject / ('lvsa_' + frame + '.nii.gz'))
            nib.save(nib.Nifti1Image(labels, affine), subject / 'segs' / ('seg_lvsa_' + frame + '.nii.gz'))
        return subject

    def test_copies_only_same_subject_predictions_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as root:
            subject = self.make_subject(root)
            report_path = Path(root) / 'copy.json'
            prepare_manual_fixture.prepare(subject, report_path)
            report = json.loads(report_path.read_text())
            self.assertTrue(report['source_is_same_subject_automatic_prediction'])
            self.assertFalse(report['human_corrected'])
            for frame in ('ED', 'ES'):
                automatic = subject / 'segs' / ('seg_lvsa_' + frame + '.nii.gz')
                manual = subject / 'segs' / ('seg_lvsa_' + frame + '_manual.nii.gz')
                self.assertEqual(automatic.read_bytes(), manual.read_bytes())
                self.assertEqual(Path(report['frames'][frame]['source']), automatic)

    def test_existing_manual_file_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            subject = self.make_subject(root)
            manual = subject / 'segs/seg_lvsa_ED_manual.nii.gz'
            manual.write_bytes(b'protected human edit')
            with self.assertRaises(FileExistsError):
                prepare_manual_fixture.prepare(subject, Path(root) / 'copy.json')
            self.assertEqual(manual.read_bytes(), b'protected human edit')
            self.assertFalse((subject / 'segs/seg_lvsa_ES_manual.nii.gz').exists())

    def test_incomplete_automatic_labels_do_not_create_partial_manual_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            subject = self.make_subject(root)
            nib.save(nib.Nifti1Image(np.zeros((9, 9, 3), dtype=np.int16), np.diag([1., 1., 8., 1.])),
                     subject / 'segs/seg_lvsa_ES.nii.gz')
            with self.assertRaisesRegex(ValueError, 'include classes'):
                prepare_manual_fixture.prepare(subject, Path(root) / 'copy.json')
            self.assertFalse(list((subject / 'segs').glob('*_manual.nii.gz')))

    def test_phase_geometry_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            subject = self.make_subject(root)
            nib.save(nib.Nifti1Image(np.ones((9, 9, 3, 1), dtype=np.int16), np.eye(4)),
                     subject / 'lvsa_ES.nii.gz')
            with self.assertRaises(ValueError):
                prepare_manual_fixture.prepare(subject, Path(root) / 'copy.json')
            self.assertFalse(list((subject / 'segs').glob('*_manual.nii.gz')))

    def test_symlink_to_other_subject_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            subject = self.make_subject(root)
            original = subject / 'segs/seg_lvsa_ED.nii.gz'
            outside = Path(root) / 'other_subject_prediction.nii.gz'
            original.rename(outside)
            original.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'local to this subject'):
                prepare_manual_fixture.prepare(subject, Path(root) / 'copy.json')
            self.assertFalse(list((subject / 'segs').glob('*_manual.nii.gz')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
