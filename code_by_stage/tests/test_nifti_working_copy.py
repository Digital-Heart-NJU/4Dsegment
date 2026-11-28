"""Regression tests for scaled voxels and conflicting NIfTI spacing headers."""
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nifti_working_copy import consistent_image, normalization_report


class WorkingCopyTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def test_materializes_original_slope_and_intercept_without_contrast_scaling(self):
        path = os.path.join(self.root, 'original.nii')
        values = np.arange(24, dtype=np.int16).reshape(2, 3, 2, 2)
        image = nib.Nifti1Image(values, np.diag([1., 1., 8., 1.]))
        image.header.set_slope_inter(160.5867, 23.)
        nib.save(image, path)
        original = nib.load(path)
        with open(path, 'rb') as stream:
            original_bytes = stream.read()
        expected = np.asarray(original.dataobj).astype(np.float32)
        working = consistent_image(np.asarray(original.dataobj), original)
        candidate = os.path.join(self.root, 'working.nii.gz')
        nib.save(working, candidate)
        saved = nib.load(candidate)
        np.testing.assert_array_equal(np.asarray(saved.dataobj), expected)
        np.testing.assert_array_equal(saved.affine, original.affine)
        self.assertEqual(saved.dataobj.slope, 1.)
        self.assertEqual(saved.dataobj.inter, 0.)
        self.assertEqual(saved.get_data_dtype(), np.dtype(np.float32))
        self.assertGreater(normalization_report(original, working)['original_intensity_slope'], 160.)
        with open(path, 'rb') as stream:
            self.assertEqual(stream.read(), original_bytes)

    def test_reconciles_pixdim_and_qform_to_the_active_sform_without_resampling(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
        affine = np.diag([.9375, .9375, 8., 1.])
        affine[:3, 3] = [59., 161., -58.]
        original = nib.Nifti1Image(values, affine)
        original.set_qform(np.diag([1., 1., 8., 1.]), code=0)
        original.set_sform(affine, code=2)
        original.header.set_xyzt_units('mm', 'sec')
        original.header['pixdim'][4] = .04
        working = consistent_image(values, original)
        np.testing.assert_array_equal(np.asarray(working.dataobj), values)
        np.testing.assert_array_equal(working.affine, affine)
        np.testing.assert_array_equal(working.header.get_zooms()[:3], [.9375, .9375, 8.])
        np.testing.assert_array_equal(working.get_qform(), affine)
        self.assertEqual(working.header['pixdim'][4], original.header['pixdim'][4])
        self.assertEqual(working.header.get_xyzt_units(), original.header.get_xyzt_units())
        report = normalization_report(original, working)
        self.assertTrue(report['spacing_header_corrected'])
        self.assertTrue(report['active_affine_preserved'])
        self.assertFalse(report['voxel_resampling'])

    def test_consistent_headers_retain_values_and_timing(self):
        values = np.ones((2, 3, 2, 4), dtype=np.int16)
        original = nib.Nifti1Image(values, np.diag([1., 1., 8., 1.]))
        original.header['pixdim'][4] = .035
        image = consistent_image(values, original)
        np.testing.assert_array_equal(image.header['pixdim'], original.header['pixdim'])
        self.assertFalse(normalization_report(original, image)['spacing_header_corrected'])

    def test_label_archive_stays_uint8_and_keeps_exact_labels(self):
        values = np.arange(24, dtype=np.uint8).reshape(2, 3, 2, 2) % 4
        reference = nib.Nifti1Image(values, np.eye(4))
        image = consistent_image(values, reference, dtype=np.uint8)
        self.assertEqual(image.get_data_dtype(), np.dtype(np.uint8))
        np.testing.assert_array_equal(np.asarray(image.dataobj), values)

    def test_real_shear_is_rejected_instead_of_silently_changing_geometry(self):
        affine = np.eye(4)
        affine[0, 1] = .2
        reference = nib.Nifti1Image(np.zeros((2, 2, 2)), affine)
        with self.assertRaises(ValueError):
            consistent_image(np.asarray(reference.dataobj), reference)

    def test_nonfinite_working_values_are_rejected(self):
        reference = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
        values = np.ones((2, 2, 2))
        values[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            consistent_image(values, reference)


if __name__ == '__main__':
    unittest.main()
