"""Regression tests for voxel-index NIfTI cropping (run with unittest)."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop_nifti_by_coordinates import crop_nifti


class CropNiftiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.nii"
        self.target = self.root / "subject_211/example.nii"
        self.raw = np.arange(8 * 9 * 5 * 4, dtype=np.int16).reshape(8, 9, 5, 4)
        rotation = np.array([
            [np.cos(0.3), -np.sin(0.3), 0.0],
            [np.sin(0.3), np.cos(0.3), 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.affine = np.eye(4)
        self.affine[:3, :3] = rotation @ np.diag([0.9375, 0.9375, 8.0])
        self.affine[:3, 3] = [42.0, -11.0, 17.0]
        image = nib.Nifti1Image(self.raw, self.affine)
        image.header.set_zooms((0.9375, 0.9375, 8.0, 0.04))
        image.header.set_xyzt_units("mm", "sec")
        image.header["toffset"] = 1.25
        image.header.set_slope_inter(2.5, -7.0)
        nib.save(image, self.source)
        self.original_bytes = self.source.read_bytes()

    def crop(self, **kwargs):
        options = dict(x_min=1, x_max=5, y_min=2, y_max=6)
        options.update(kwargs)
        return crop_nifti(self.source, self.target, **options)

    def test_inclusive_crop_keeps_all_slices_frames_values_and_scaling(self):
        result = self.crop()
        image = nib.load(self.target)
        source = nib.load(self.source)
        self.assertEqual(result["output_shape"], (5, 5, 5, 4))
        np.testing.assert_array_equal(image.dataobj.get_unscaled(), self.raw[1:6, 2:7])
        np.testing.assert_array_equal(image.get_fdata(), source.get_fdata()[1:6, 2:7])
        self.assertEqual(image.get_data_dtype(), source.get_data_dtype())
        self.assertEqual((image.dataobj.slope, image.dataobj.inter), (2.5, -7.0))
        self.assertEqual(image.header.get_zooms(), source.header.get_zooms())
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_oblique_spatial_coordinates_and_temporal_offset(self):
        self.crop(z_start=2, z_count=2, frame_start=1, frame_count=2)
        image, source = nib.load(self.target), nib.load(self.source)
        self.assertEqual(image.shape, (5, 5, 2, 2))
        for point in ([0, 0, 0], [4, 4, 1]):
            expected = nib.affines.apply_affine(source.affine, np.array(point) + [1, 2, 2])
            np.testing.assert_allclose(nib.affines.apply_affine(image.affine, point), expected, atol=1e-5)
        self.assertAlmostEqual(float(image.header["toffset"]), 1.29, places=6)
        self.assertEqual(image.header.get_xyzt_units(), ("mm", "sec"))
        np.testing.assert_array_equal(image.dataobj.get_unscaled(), self.raw[1:6, 2:7, 2:4, 1:3])

    def test_singleton_dimensions_preserved(self):
        self.crop(z_start=4, z_count=1, frame_start=3, frame_count=1)
        self.assertEqual(nib.load(self.target).shape, (5, 5, 1, 1))

    def test_three_dimensional_image(self):
        nib.save(nib.Nifti1Image(self.raw[..., 0], self.affine), self.source)
        self.crop(z_count=1)
        self.assertEqual(nib.load(self.target).shape, (5, 5, 1))

    def test_invalid_ranges_do_not_create_output_directory(self):
        for options in (
            {"x_min": -1}, {"x_max": 8}, {"x_min": 6, "x_max": 5},
            {"y_max": 9}, {"z_count": 0}, {"z_start": 5}, {"z_count": 6},
            {"frame_count": -1}, {"frame_start": 4}, {"frame_start": 2, "frame_count": 3},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.crop(**options)
        self.assertFalse(self.target.parent.exists())

    def test_protects_existing_output_and_source_even_with_overwrite(self):
        self.crop()
        before = self.target.read_bytes()
        with self.assertRaises(FileExistsError):
            self.crop(x_max=6)
        self.assertEqual(self.target.read_bytes(), before)
        self.crop(x_max=6, overwrite=True)
        self.assertEqual(nib.load(self.target).shape, (6, 5, 5, 4))
        with self.assertRaises(ValueError):
            crop_nifti(self.source, self.source, x_min=0, x_max=1, y_min=0, y_max=1, overwrite=True)
        alias = self.root / "hardlink.nii"
        alias.hardlink_to(self.source)
        with self.assertRaises(ValueError):
            crop_nifti(self.source, alias, x_min=0, x_max=1, y_min=0, y_max=1, overwrite=True)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_dry_run_does_not_write(self):
        result = self.crop(dry_run=True)
        self.assertEqual(result["output_shape"], (5, 5, 5, 4))
        self.assertFalse(self.target.parent.exists())

    def test_distinct_qform_and_sform_are_shifted_independently(self):
        image = nib.load(self.source)
        raw = image.dataobj.get_unscaled()
        image = nib.Nifti1Image(raw, self.affine, header=image.header.copy())
        qform = self.affine.copy()
        qform[:3, 3] += [8, 6, 4]
        image.set_qform(qform, code=1)
        image.set_sform(self.affine, code=2)
        nib.save(image, self.source)
        self.crop()
        source, output = nib.load(self.source), nib.load(self.target)
        shift = np.eye(4)
        shift[:3, 3] = [1, 2, 0]
        for name, code in (("qform", 1), ("sform", 2)):
            actual, actual_code = getattr(output, f"get_{name}")(coded=True)
            expected = getattr(source, f"get_{name}")() @ shift
            self.assertEqual(actual_code, code)
            np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_uncoded_forms_still_preserve_world_coordinates(self):
        image = nib.Nifti1Image(self.raw, None)
        image.header.set_zooms((0.9375, 0.9375, 8.0, 0.04))
        nib.save(image, self.source)
        source = nib.load(self.source)
        self.assertEqual(int(source.header["sform_code"]), 0)
        self.crop()
        output = nib.load(self.target)
        self.assertEqual(int(output.header["sform_code"]), 2)
        np.testing.assert_allclose(output.affine[:3, 3], nib.affines.apply_affine(source.affine, [1, 2, 0]))

    def test_compressed_output(self):
        self.target = self.target.with_suffix(".nii.gz")
        self.crop()
        np.testing.assert_array_equal(nib.load(self.target).dataobj.get_unscaled(), self.raw[1:6, 2:7])

    def test_validation_failure_preserves_existing_output(self):
        self.crop()
        before = self.target.read_bytes()
        with patch("crop_nifti_by_coordinates._verify_saved", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                self.crop(overwrite=True, x_max=6)
        self.assertEqual(self.target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
