"""Verify saved native labels, remapping and physical XY-only RV wall growth."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_rv_myo_xy import expand_rv_myo_xy


def verify(subject, output):
    summary = json.loads((subject / 'dmacs_nnunet_stage01_summary.json').read_text())
    original_path = subject / 'example.nii'
    original_shape = nib.load(original_path).shape
    original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
    assert original_hash == summary['input_image_sha256']
    report = {'subject': subject.name, 'status': 'passed_file_and_growth_checks',
              'model_output_status': summary['status'], 'original_image_unchanged': True,
              'scope': 'Integrity and algorithm checks, not anatomical accuracy.', 'frames': {}}
    for frame in ('ED', 'ES'):
        phase_path = subject / ('lvsa_' + frame + '.nii.gz')
        if not phase_path.is_file():
            phase_path = subject / 'sizes' / phase_path.name
        phase = nib.load(phase_path)
        root = subject / 'segs'
        names = {'native': root / 'nnunet_native' / ('seg_lvsa_' + frame + '.nii.gz'),
                 'mapped': root / ('nnunet_mapped_' + frame + '.nii.gz'),
                 'final': root / ('seg_lvsa_' + frame + '.nii.gz'),
                 'added': root / ('rv_myo_added_' + frame + '.nii.gz')}
        images = {name: nib.load(path) for name, path in names.items()}
        arrays = {name: np.asarray(image.dataobj) for name, image in images.items()}
        for name, image in images.items():
            assert image.shape == phase.shape[:3], name
            np.testing.assert_allclose(image.affine, phase.affine, atol=1e-4)
            assert np.isfinite(arrays[name]).all()
        native, mapped, final, added = [arrays[name] for name in ('native', 'mapped', 'final', 'added')]
        wall_report = json.loads((root / ('rv_myo_added_' + frame + '_summary.json')).read_text())
        assert np.isin(native, (0, 1, 2, 3)).all()
        expected = native.astype(np.int16)
        expected[native == 3] = 4
        np.testing.assert_array_equal(mapped, expected)
        assert np.isin(final, (0, 1, 2, 3, 4)).all() and np.isin(added, (0, 1)).all()
        grown, all_added, growth_report = expand_rv_myo_xy(
            mapped, phase.affine, wall_report['radius_mm'], phase.header.get_xyzt_units()[0], wall_report['assumed_mm'])
        expected_final, expected_added = grown.copy(), all_added.copy()
        boundary = np.zeros(final.shape, dtype=bool)
        cleanup = wall_report.get('terminal_z_cleanup')
        if cleanup:
            assert final.shape[2] == original_shape[2] == cleanup['original_z_slices']
            indices = sorted({0, original_shape[2] - 1})
            assert indices == cleanup['slice_indices']
            boundary[:, :, indices] = True
            expected_final[boundary] = 0
            expected_added[boundary] = False
            assert not final[boundary].any() and not added[boundary].any()
            np.testing.assert_array_equal(final[:, :, 1:-1], grown[:, :, 1:-1])
            assert cleanup['removed_foreground_voxels'] == int(np.count_nonzero(grown[boundary]))
            assert cleanup['removed_added_wall_voxels'] == int(np.count_nonzero(all_added[boundary]))
            before = wall_report['growth_before_terminal_z_cleanup']
            assert before['label_counts'] == growth_report['label_counts']
            assert before['added_voxels'] == growth_report['added_voxels']
            assert wall_report['nonterminal_protected_labels_unchanged']
        np.testing.assert_array_equal(final, expected_final)
        np.testing.assert_array_equal(added.astype(bool), expected_added)
        changed = (final != mapped) & ~boundary
        np.testing.assert_array_equal(changed, added.astype(bool))
        assert (mapped[changed] == 0).all() and (final[changed] == 3).all()
        protected = (mapped != 0) & ~boundary
        np.testing.assert_array_equal(mapped[protected], final[protected])
        assert wall_report['protected_labels_unchanged'] == bool(np.array_equal(mapped[mapped != 0], final[mapped != 0]))
        assert images['final'].get_data_dtype() == np.dtype('int16')
        assert images['final'].header.get_xyzt_units() == phase.header.get_xyzt_units()
        for form in ('qform', 'sform'):
            actual, actual_code = getattr(images['final'], 'get_' + form)(coded=True)
            original, original_code = getattr(phase, 'get_' + form)(coded=True)
            assert int(actual_code) == int(original_code)
            if actual is not None:
                np.testing.assert_allclose(actual, original, atol=1e-4)
        scales = {'mm': 1., 'meter': 1000., 'micron': .001}
        scale = scales.get(phase.header.get_xyzt_units()[0], 1. if wall_report['assumed_mm'] else None)
        assert scale is not None
        basis = phase.affine[:3, :2] * scale
        orthogonal = abs(np.dot(basis[:, 0], basis[:, 1])) <= 1e-7 * np.prod(np.linalg.norm(basis, axis=0))
        distances = []
        for z in range(final.shape[2]):
            rv, new = mapped[:, :, z] == 4, changed[:, :, z]
            if not rv.any():
                assert not new.any()
            if new.any():
                source = np.argwhere(rv)
                target = np.argwhere(new)
                if orthogonal:
                    values = ndimage.distance_transform_edt(~rv, sampling=np.linalg.norm(basis, axis=0))[new]
                else:
                    # Keep this independent of the production KD-tree calculation.
                    values = np.array([np.linalg.norm((source - point) @ basis.T, axis=1).min() for point in target])
                distances.extend(values.tolist())
                assert np.all(values <= wall_report['radius_mm'] + 1e-8)
        counts = {str(int(value)): int(count) for value, count in zip(*np.unique(final, return_counts=True))}
        assert counts == wall_report['label_counts']
        assert int(changed.sum()) == wall_report['added_voxels']
        report['frames'][frame] = {'shape': list(final.shape), 'label_counts': counts,
                                    'added_RV_myocardium_voxels': int(changed.sum()),
                                    'max_added_center_distance_mm': max(distances) if distances else None,
                                    'missing_predicted_anatomy': wall_report['missing_predicted_anatomy'],
                                    'terminal_z_cleanup': cleanup,
                                    'native_RV_3_remapped_to_4': True, 'only_background_changed_to_3': True,
                                    'no_Z_growth': True, 'NIfTI_grid_and_forms_preserved': True}
    report['raw_surface_files'] = [p.name for p in (subject / 'vtks').glob('*.vtk')]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    verify(args.subject_dir.resolve(), args.output.resolve())
