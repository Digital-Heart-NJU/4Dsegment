"""Safe legacy intensity conversion; compatible with host and Docker Python."""
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


class IntensitySafetyTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.path = os.path.join(self.root, 'working.nii.gz')
        self.affine = np.diag([.9375, .9375, 8., 1.])

    def save(self, values):
        image = nib.Nifti1Image(values, self.affine)
        image.header.set_xyzt_units('mm', 'sec')
        image.header['pixdim'][4] = .04
        nib.save(image, self.path)

    def digest(self, path):
        with open(path, 'rb') as stream:
            return hashlib.sha256(stream.read()).hexdigest()

    def test_safe_input_is_not_rewritten(self):
        self.save(np.array([0., .25, 100., 32767.]).reshape(2, 2, 1))
        before = self.digest(self.path)
        report = image_utils.ensure_temporal_int16_range(self.path)
        self.assertFalse(report['rescaled'])
        self.assertEqual(self.digest(self.path), before)

    def test_unsafe_input_preserves_zero_mask_shape_geometry_and_time_spacing(self):
        data = np.linspace(0., 156607.14, 120).reshape(3, 4, 2, 5)
        data[0] = 0
        self.save(data)
        before = nib.load(self.path)
        report = image_utils.ensure_temporal_int16_range(self.path)
        saved = nib.load(self.path)
        actual = np.asarray(saved.dataobj)
        self.assertTrue(report['rescaled'])
        self.assertEqual(saved.get_data_dtype(), np.dtype(np.int16))
        self.assertEqual(saved.shape, data.shape)
        np.testing.assert_array_equal(saved.affine, before.affine)
        np.testing.assert_array_equal(saved.header['pixdim'], before.header['pixdim'])
        self.assertEqual(saved.header.get_xyzt_units(), before.header.get_xyzt_units())
        np.testing.assert_array_equal(actual == 0, data == 0)
        self.assertEqual(actual.min(), 0)
        self.assertEqual(actual.max(), 4095)
        self.assertTrue(np.all(actual[data > 0] >= 1))
        expected_low, expected_high = np.percentile(data[data > 0], (1, 99))
        self.assertEqual(report['percentile_values'], [expected_low, expected_high])

    def test_constant_bright_input_does_not_divide_by_zero(self):
        data = np.zeros((2, 2, 2))
        data[1] = 100000.
        self.save(data)
        image_utils.ensure_temporal_int16_range(self.path)
        values = np.asarray(nib.load(self.path).dataobj)
        np.testing.assert_array_equal(values[data == 0], 0)
        np.testing.assert_array_equal(values[data != 0], 4095)

    def test_empty_foreground_needs_no_scaling(self):
        self.save(np.zeros((2, 2, 2)))
        self.assertFalse(image_utils.ensure_temporal_int16_range(self.path)['rescaled'])

    def test_invalid_intensities_are_rejected_without_rewriting(self):
        for value in (-1., np.nan, np.inf):
            data = np.zeros((2, 2, 2))
            data[0, 0, 0] = value
            self.save(data)
            before = self.digest(self.path)
            with self.assertRaises(ValueError):
                image_utils.ensure_temporal_int16_range(self.path)
            self.assertEqual(self.digest(self.path), before)

    def test_pipeline_scales_before_temporalalign_and_preserves_original(self):
        self.save(np.linspace(0., 156607., 120).reshape(3, 4, 2, 5))
        original = self.path
        original_hash = self.digest(original)
        calls = []

        def command(argv):
            calls.append(argv[0])
            if argv[0] == 'headertool':
                shutil.copyfile(argv[1], argv[2])
            elif argv[0] == 'temporalalign':
                source = nib.load(argv[2])
                values = np.asarray(source.dataobj)
                self.assertEqual(source.get_data_dtype(), np.dtype(np.int16))
                self.assertLessEqual(values.max(), 4095)
                self.assertGreaterEqual(values.min(), 0)
            elif argv[0] == 'cardiacphasedetection':
                source = nib.load(argv[1])
                values = np.asarray(source.dataobj)
                for t, path in enumerate(argv[2:]):
                    nib.save(nib.Nifti1Image(values[..., t:t + 1], source.affine), path)

        with mock.patch.object(image_utils.subprocess, 'check_call', side_effect=command):
            image_utils.imagePreprocessing(original, self.root, '/refs')
        self.assertEqual(calls, ['headertool', 'temporalalign', 'autocontrast', 'cardiacphasedetection'])
        self.assertEqual(self.digest(original), original_hash)
        with open(os.path.join(self.root, 'lvsa_intensity_preprocessing.json')) as stream:
            self.assertTrue(json.load(stream)['rescaled'])

    def test_temporalalign_failure_does_not_continue(self):
        self.save(np.zeros((2, 2, 2, 3)))
        calls = []

        def command(argv):
            calls.append(argv[0])
            if argv[0] == 'headertool':
                shutil.copyfile(argv[1], argv[2])
            else:
                raise image_utils.subprocess.CalledProcessError(1, argv)

        with mock.patch.object(image_utils.subprocess, 'check_call', side_effect=command):
            with self.assertRaises(image_utils.subprocess.CalledProcessError):
                image_utils.imagePreprocessing(self.path, self.root, '/refs')
        self.assertEqual(calls, ['headertool', 'temporalalign'])

    def test_original_cannot_be_its_own_working_copy(self):
        with self.assertRaises(ValueError):
            image_utils.imagePreprocessing(os.path.join(self.root, 'lvsa_.nii.gz'), self.root, '/refs')


if __name__ == '__main__':
    unittest.main()
