"""Acquisition-frame provenance and categorical-label alignment safeguards."""
from __future__ import print_function

import hashlib
import json
import os
import shutil
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
import image_utils
import native_phase_frames


class NativePhaseTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.original = os.path.join(self.root, 'example.nii')
        self.native = os.path.join(self.root, 'native.nii.gz')
        self.processed = os.path.join(self.root, 'processed.nii.gz')
        self.phases = dict((frame, os.path.join(self.root, frame + '.nii.gz')) for frame in ('ED', 'ES'))
        self.values = np.broadcast_to(np.arange(5, dtype=np.int16) * 100, (3, 4, 2, 5)).copy()
        self.affine = np.diag([1., 1., 8., 1.])
        image = nib.Nifti1Image(self.values, self.affine)
        image.header.set_xyzt_units('mm', 'sec')
        image.header['pixdim'][4] = .04
        nib.save(image, self.original)
        self.reset = self.affine.copy()
        self.reset[:3, 3] = [12., -9., 7.]
        nib.save(nib.Nifti1Image(self.values, self.reset, header=image.header.copy()), self.native)
        processed = self.values[..., [0, 2, 4]]
        nib.save(nib.Nifti1Image(processed, self.reset), self.processed)
        for frame, index in (('ED', 0), ('ES', 1)):
            nib.save(nib.Nifti1Image(processed[..., index:index + 1], self.reset), self.phases[frame])

    def command(self, argv):
        if argv[0] == 'temporalalign':
            image = nib.load(argv[2])
            values = np.asarray(image.dataobj)[..., [0, 2, 4]]
            nib.save(nib.Nifti1Image(values, image.affine), argv[3])
        elif argv[0] != 'autocontrast':
            raise AssertionError(argv)

    def rebuild(self, output='ES phase is: 1'):
        return native_phase_frames.rebuild_native_phases(
            self.original, self.native, self.processed, self.phases,
            output, '/refs', self.root, self.root)

    def test_frame_record_and_phase_images_use_the_original_acquisitions(self):
        with open(self.original, 'rb') as stream:
            before = hashlib.sha256(stream.read()).hexdigest()
        with mock.patch.object(native_phase_frames.subprocess, 'check_call', side_effect=self.command):
            report = self.rebuild()
        self.assertEqual(report['processed_to_original_frame_indices'], [0, 2, 4])
        self.assertEqual(report['frames']['ES']['original_frame_index_zero_based'], 2)
        self.assertEqual(report['frames']['ES']['original_frame_number_one_based'], 3)
        self.assertAlmostEqual(report['frames']['ES']['original_time'], .08)
        for frame, index in (('ED', 0), ('ES', 2)):
            saved = nib.load(self.phases[frame])
            np.testing.assert_array_equal(np.asarray(saved.dataobj), self.values[..., index:index + 1])
            np.testing.assert_array_equal(saved.affine, self.reset)
        np.testing.assert_array_equal(report['source_to_phase_world_matrix'], np.dot(self.reset, np.linalg.inv(self.affine)))
        with open(os.path.join(self.root, 'segs/ED_ES_frames.txt')) as stream:
            text = stream.read()
        self.assertIn('ES: original index=2, original frame=3, processed index=1, processed frame=2', text)
        with open(os.path.join(self.root, 'segs/ED_ES_frames.json')) as stream:
            self.assertEqual(json.load(stream), report)
        with open(self.original, 'rb') as stream:
            self.assertEqual(hashlib.sha256(stream.read()).hexdigest(), before)

    def test_unreported_or_invalid_phase_indices_are_rejected(self):
        for output in ('no phase index', 'ES phase is: 8', 'ES phase is: 2'):
            with mock.patch.object(native_phase_frames.subprocess, 'check_call', side_effect=self.command):
                with self.assertRaises(RuntimeError):
                    self.rebuild(output)
        self.assertFalse(os.path.exists(os.path.join(self.root, 'segs/ED_ES_frames.json')))

    def test_temporal_interpolation_is_not_silently_treated_as_a_native_frame(self):
        def interpolating(argv):
            self.command(argv)
            if argv[0] == 'temporalalign' and argv[2].endswith('ramp.nii.gz'):
                image = nib.load(argv[3])
                values = np.asarray(image.dataobj).copy()
                values[..., 1] += 1
                nib.save(nib.Nifti1Image(values, image.affine), argv[3])
        with mock.patch.object(native_phase_frames.subprocess, 'check_call', side_effect=interpolating):
            with self.assertRaises(RuntimeError):
                self.rebuild()

    def test_header_reset_cannot_silently_reorder_original_voxels(self):
        def reset(argv):
            self.assertEqual(argv[0], 'headertool')
            nib.save(nib.Nifti1Image(self.values[..., ::-1], self.reset), argv[2])
        with mock.patch.object(image_utils.subprocess, 'check_call', side_effect=reset):
            with self.assertRaises(RuntimeError):
                image_utils.imagePreprocessing(self.original, self.root, '/refs', native_frame_phases=True)


if __name__ == '__main__':
    unittest.main()
