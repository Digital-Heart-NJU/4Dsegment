"""Inspect staged outputs without rerunning the model (numpy, nibabel, VTK)."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def inspect_surface(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if mesh.GetNumberOfPoints() == 0 or mesh.GetNumberOfCells() == 0:
        raise ValueError('Empty or unreadable surface: ' + str(path))
    if not np.isfinite(vtk_to_numpy(mesh.GetPoints().GetData())).all():
        raise ValueError('Nonfinite surface points: ' + str(path))
    return {'points': mesh.GetNumberOfPoints(), 'cells': mesh.GetNumberOfCells()}


def verify(subject, run_dir, stage, require_manual_copy=False):
    frames = ('ED', 'ES')
    raw_names = {name + '_' + frame + '.vtk'
                 for name in ('LVendo', 'LVepi', 'RV', 'RVepi', 'LVmyo') for frame in frames}
    report = {'subject': subject.name, 'stage': stage, 'status': 'passed'}
    log = (run_dir / ('stage' + stage + '.log')).read_text(errors='replace')
    mode = 'stage02_surf_only' if stage == '02' else 'stage03_addi_mot_fit'
    assert '[DMACS stages] Mode: ' + mode in log
    assert mode + ' completed in ' in log
    assert 'Traceback (most recent call last)' not in log
    assert 'Segmenting ED frame' not in log and 'Segmenting ES frame' not in log
    report['no_CNN_execution_in_resume_log'] = True
    report['elapsed_seconds'] = float(re.search(re.escape(mode) + r' completed in ([0-9.]+)s', log).group(1))

    stage_inputs = json.loads((subject / ('dmacs_' + mode + '_summary.json')).read_text())
    for path, expected in stage_inputs['input_sha256'].items():
        prefix = '/data/' + subject.name + '/'
        actual = subject / path[len(prefix):] if path.startswith(prefix) else Path(path)
        assert digest(actual) == expected, str(actual)
    report['stage_input_hashes_match'] = True

    actual_vtks = set(path.name for path in (subject / 'vtks').glob('*.vtk'))
    assert raw_names <= actual_vtks
    report['raw_surfaces'] = {name: inspect_surface(subject / 'vtks' / name) for name in sorted(raw_names)}

    if stage == '02':
        assert actual_vtks == raw_names, 'Stage 02 produced unexpected prefixed surfaces.'
        assert not list((subject / 'motion').glob('RV_fr*.vtk'))
        seg_manual = stage_inputs.get('seg_manual', True)
        source = 'Manual' if seg_manual else 'Automatic'
        assert source + ' refinement input:' in log
        report['seg_manual'] = seg_manual
        report['segmentation_source'] = source.lower()
        if require_manual_copy and not seg_manual:
            raise ValueError('--require-manual-copy applies only to manual-input Stage 02 runs.')
        files = [subject / 'vtks' / name for name in raw_names]
        files += list((subject / 'tmps').glob('vtk_*.nii.gz'))
        files += list((subject / 'segs').glob('seg_lvsa_*.nii.gz'))
        report['preserved_inputs_for_stage03'] = {
            str(path.relative_to(subject)): digest(path) for path in sorted(files)
        }
        report['no_fitted_surfaces_or_motion'] = True
        if seg_manual:
            report['manual_test_copies_are_unedited'] = {}
        for frame in frames:
            manual = subject / 'segs' / ('seg_lvsa_' + frame + '_manual.nii.gz')
            automatic = subject / 'segs' / ('seg_lvsa_' + frame + '.nii.gz')
            archive = subject / 'sizes' / ('2D_segmentation_' + frame + '.nii.gz')
            selected = manual if seg_manual else automatic
            assert digest(selected) == digest(archive)
            if seg_manual:
                report['manual_test_copies_are_unedited'][frame] = bool(np.array_equal(
                    np.asarray(nib.load(manual).dataobj), np.asarray(nib.load(automatic).dataobj))) if automatic.is_file() else None
        report['selected_inputs_match_archived_2D_segmentations'] = True
        if seg_manual:
            report['manual_inputs_match_archived_2D_segmentations'] = True
        if require_manual_copy:
            provenance = json.loads((run_dir / 'manual_test_fixture.json').read_text())
            assert provenance['subject'] == subject.name
            assert provenance['source_is_same_subject_automatic_prediction'] is True
            for frame in frames:
                automatic = subject / 'segs' / ('seg_lvsa_' + frame + '.nii.gz')
                manual = subject / 'segs' / ('seg_lvsa_' + frame + '_manual.nii.gz')
                record = provenance['frames'][frame]
                assert Path(record['source']).resolve() == automatic.resolve()
                assert Path(record['destination']).resolve() == manual.resolve()
                assert digest(automatic) == digest(manual) == record['sha256']
            report['manual_copies_match_own_automatic_prediction_sha256'] = True
    else:
        previous = json.loads((run_dir / 'validation_stage02.json').read_text())
        for path, expected in previous['preserved_inputs_for_stage03'].items():
            assert digest(subject / path) == expected, path
        report['all_stage02_raw_surfaces_masks_and_segmentations_unchanged'] = True
        assert 'Select all the shape atlases' not in log
        report['no_atlas_refinement_in_stage03_log'] = True
        for frame in frames:
            for name in ('F_LVendo', 'F_LVepi', 'F_LVmyo', 'C_RV', 'W_LVmyo', 'W_RV'):
                assert name + '_' + frame + '.vtk' in actual_vtks
            for prefix in ('rv_' + frame.lower(), 'lv_myo' + frame.lower()):
                for quantity in ('curvature', 'wallthickness', 'signeddistances'):
                    path = subject / (prefix + '_' + quantity + '.txt')
                    assert path.is_file() and path.stat().st_size > 0, str(path)
        for name in actual_vtks:
            inspect_surface(subject / 'vtks' / name)
        report['all_subject_surface_VTKs_readable'] = len(actual_vtks)
        cine = nib.load(subject / 'lvsa_.nii.gz')
        for frame in range(cine.shape[3]):
            path = subject / 'motion' / ('RV_fr%02d.vtk' % frame)
            inspect_surface(path)
            assert path.with_suffix('.txt').is_file()
        report['motion_meshes_and_text_files'] = cine.shape[3]
        for name in ('RV_fr00_new.vtk', 'RV_fr00_new_new.vtk'):
            inspect_surface(subject / 'motion' / name)
        assert (run_dir / 'matchedpointsnew.txt').stat().st_size > 0
        report['decimation_and_correspondences_written'] = True
        report['noninvertible_transformation_warning_count'] = log.count('input mapping non-invertible')
        report['scope'] = 'Execution and file integrity; not anatomical or geometric quality certification.'
    output = run_dir / ('validation_stage' + stage + '.json')
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    print('Saved validation report:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('02', '03'))
    parser.add_argument('--require-manual-copy', action='store_true',
                        help='For unedited tests only: require same-subject automatic/manual byte identity and provenance.')
    args = parser.parse_args()
    verify(args.subject_dir.resolve(), args.run_dir.resolve(), args.stage, args.require_manual_copy)
