"""Modern host launcher: SAX nnUNet inference plus legacy staged Docker tools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

from build_rv_myo_xy import remap_nnunet_labels, expand_rv_myo_xy, save_labels


CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parent
FRAMES = ('ED', 'ES')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false.')


def run_logged(command, path):
    print('[nnUNet SAX] Command:', ' '.join(map(str, command)), flush=True)
    with Path(path).open('w') as log:
        log.write('COMMAND: ' + ' '.join(map(str, command)) + '\n')
        log.flush()
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def docker_command(args, subject, run_dir, script, extra):
    dataset = run_dir / 'dataset'
    (dataset / subject.name).mkdir(parents=True, exist_ok=True)
    return ['docker', 'run', '--rm', '--user', str(os.getuid()) + ':' + str(os.getgid()),
            '-e', 'LD_LIBRARY_PATH=/lib64', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUNBUFFERED=1',
            '-e', 'OMP_NUM_THREADS=' + str(args.coreNo),
            '-v', str(dataset) + ':/data', '-v', str(subject) + ':/data/' + subject.name,
            '-v', str(CODE_DIR) + ':/code:ro', '-w', '/code', args.docker_image,
            'python', '-B', '-u', script] + list(extra)


def infer_cases(args, inputs, outputs):
    workspace = args.nnunet_workspace.resolve()
    for name, folder in (('nnUNet_raw', 'nnUNet_raw'), ('nnUNet_preprocessed', 'nnUNet_preprocessed'),
                         ('nnUNet_results', 'nnUNet_results')):
        os.environ[name] = str(workspace / folder)
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; choose --device cpu explicitly.')
    torch.set_num_threads(args.coreNo)
    predictor = nnUNetPredictor(device=torch.device(args.device), allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(str(args.nnunet_model_dir), tuple(args.folds),
                                                   checkpoint_name=args.checkpoint)
    predictor.predict_from_files(str(inputs), str(outputs), save_probabilities=False,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1)


def finalize_stage01_labels(mapped, grown, added, growth_report, original_z_slices):
    """Clear whole-image terminal slices, retaining the unmodified growth evidence."""
    if (grown.ndim != 3 or grown.shape != mapped.shape or grown.shape != added.shape or
            grown.shape[2] < 1 or grown.shape[2] != original_z_slices):
        raise ValueError('Segmentation and wall mask must match the original image Z stack.')
    final, retained = grown.copy(), added.copy()
    indices = sorted({0, original_z_slices - 1})
    removed = []
    for z in indices:
        plane = grown[:, :, z]
        values, counts = np.unique(plane[plane != 0], return_counts=True)
        removed.append({'z': z, 'foreground_voxels_removed': int(np.count_nonzero(plane)),
                        'added_wall_voxels_removed': int(np.count_nonzero(added[:, :, z])),
                        'removed_label_counts': {str(int(v)): int(n) for v, n in zip(values, counts)}})
        final[:, :, z] = 0
        retained[:, :, z] = False
    report = dict(growth_report)
    report['growth_before_terminal_z_cleanup'] = growth_report
    report['terminal_z_cleanup'] = {
        'method': 'clear_first_and_last_slices_of_whole_original_Z_stack',
        'original_z_slices': int(original_z_slices), 'slice_indices': indices,
        'removed_foreground_voxels': sum(row['foreground_voxels_removed'] for row in removed),
        'removed_added_wall_voxels': sum(row['added_wall_voxels_removed'] for row in removed),
        'slices': removed, 'intermediate_slices_unchanged': bool(np.array_equal(final[:, :, 1:-1], grown[:, :, 1:-1])),
        'shape_preserved': final.shape == grown.shape}
    report['added_voxels'] = int(np.count_nonzero(retained))
    report['label_counts'] = {str(int(v)): int(n) for v, n in zip(*np.unique(final, return_counts=True))}
    report['protected_labels_unchanged'] = bool(np.array_equal(final[mapped != 0], mapped[mapped != 0]))
    interior, original_interior = final[:, :, 1:-1], mapped[:, :, 1:-1]
    report['nonterminal_protected_labels_unchanged'] = bool(np.array_equal(
        interior[original_interior != 0], original_interior[original_interior != 0]))
    report['slices'] = [dict(row) for row in growth_report['slices']]
    for row in report['slices']:
        row['terminal_slice_cleared'] = row['z'] in indices
        if row['terminal_slice_cleared']:
            row.update({'RV_voxels': 0, 'added_voxels': 0, 'skipped': True,
                        'max_added_center_distance_mm': None, 'growth_or_RV_touches_image_edge': False})
    return final, retained, report


def stage01(args, subject, run_dir):
    source = subject / args.input_name
    if not source.is_file():
        raise FileNotFoundError(source)
    source_shape = nib.load(source).shape
    if len(source_shape) < 3 or source_shape[2] < 1:
        raise ValueError('The original image must have a nonempty Z stack.')
    original_z_slices = source_shape[2]
    plans = json.loads((args.nnunet_model_dir / 'plans.json').read_text())
    dataset = json.loads((args.nnunet_model_dir / 'dataset.json').read_text())
    expected = {'background': 0, 'LV blood pool': 1, 'LV myocardium': 2, 'RV blood pool': 3}
    if dataset.get('labels') != expected or '2d' not in plans.get('configurations', {}):
        raise ValueError('Expected the supplied four-label, 2D SAX model.')
    for fold in args.folds:
        if not (args.nnunet_model_dir / ('fold_' + str(fold)) / args.checkpoint).is_file():
            raise FileNotFoundError('Missing SAX fold/checkpoint: ' + str(fold))
    targets = [subject / 'segs' / ('seg_lvsa_' + frame + '.nii.gz') for frame in FRAMES]
    if not args.overwrite and any(path.exists() for path in targets + [subject / 'lvsa_.nii.gz']):
        raise FileExistsError('Stage 01 outputs already exist; use --overwrite deliberately.')
    original_hash = sha256(source)
    existing_manual = {path: sha256(path) for path in (subject / 'segs').glob('*_manual.nii.gz')}
    command = docker_command(args, subject, run_dir, 'prepare_nnunet_phases.py',
                             ['--subject-dir', '/data/' + subject.name, '--input-name', args.input_name])
    run_logged(command, run_dir / 'stage01_preprocess.log')
    work = Path(tempfile.mkdtemp(prefix='prediction_', dir=run_dir))
    inputs, predictions = work / 'imagesTs', work / 'labelsTs'
    inputs.mkdir()
    predictions.mkdir()
    phase_images = {}
    for frame in FRAMES:
        phase = nib.load(subject / ('lvsa_' + frame + '.nii.gz'))
        data = np.asarray(phase.dataobj, dtype=np.float32)
        if data.ndim != 4 or data.shape[3] != 1:
            raise ValueError('Expected a singleton-time ED/ES image.')
        if data.shape[2] != original_z_slices:
            raise ValueError('Preprocessing changed the original image Z slice count: ' + frame)
        image = nib.Nifti1Image(data[..., 0], phase.affine, header=phase.header.copy())
        image.set_data_dtype(np.float32)
        nib.save(image, inputs / (subject.name + '_sax_' + frame + '_0000.nii.gz'))
        phase_images[frame] = phase
    infer_cases(args, inputs, predictions)
    frame_reports = {}
    for frame in FRAMES:
        predicted = predictions / (subject.name + '_sax_' + frame + '.nii.gz')
        native = nib.load(predicted)
        phase = phase_images[frame]
        if native.shape != phase.shape[:3] or not np.allclose(native.affine, phase.affine, rtol=1e-6, atol=1e-4):
            raise RuntimeError('nnUNet export does not match the original phase grid: ' + frame)
        mapped = remap_nnunet_labels(np.asarray(native.dataobj))
        missing = [name for value, name in ((1, 'LV blood pool'), (2, 'LV myocardium'), (4, 'RV blood pool'))
                   if not np.any(mapped == value)]
        if missing:
            print('[nnUNet SAX] WARNING:', frame, 'has no', ', '.join(missing),
                  '; saving predictions for inspection/manual correction, not inventing missing anatomy.', flush=True)
        final, added, report = expand_rv_myo_xy(mapped, phase.affine, args.rv_wall_radius_mm,
                                               phase.header.get_xyzt_units()[0], args.assume_mm_if_unknown)
        final, added, report = finalize_stage01_labels(mapped, final, added, report, original_z_slices)
        print('[nnUNet SAX]', frame, 'terminal Z cleanup:', report['terminal_z_cleanup']['slice_indices'],
              'removed foreground voxels:', report['terminal_z_cleanup']['removed_foreground_voxels'], flush=True)
        output_dir = subject / 'segs'
        native_dir = output_dir / 'nnunet_native'
        native_dir.mkdir(exist_ok=True)
        shutil.copy2(predicted, native_dir / ('seg_lvsa_' + frame + '.nii.gz'))
        save_labels(output_dir / ('nnunet_mapped_' + frame + '.nii.gz'), mapped, phase, args.overwrite)
        save_labels(output_dir / ('rv_myo_added_' + frame + '.nii.gz'), added.astype(np.int16), phase, args.overwrite)
        final_path = output_dir / ('seg_lvsa_' + frame + '.nii.gz')
        save_labels(final_path, final, phase, args.overwrite)
        report.update({'native_label_mapping': {'0': 0, '1': 1, '2': 2, '3': 4},
                       'RV_myocardium_is_synthetic': True, 'output': str(final_path),
                       'native_prediction': str(native_dir / ('seg_lvsa_' + frame + '.nii.gz')),
                       'missing_predicted_anatomy': missing,
                       'missing_final_anatomy': [name for value, name in ((1, 'LV blood pool'), (2, 'LV myocardium'),
                                                                       (3, 'RV myocardium'), (4, 'RV blood pool'))
                                                 if not np.any(final == value)],
                       'all_foreground_classes_present': all(np.any(final == value) for value in (1, 2, 3, 4))})
        (output_dir / ('rv_myo_added_' + frame + '_summary.json')).write_text(json.dumps(report, indent=2) + '\n')
        frame_reports[frame] = report
    if sha256(source) != original_hash or any(sha256(p) != h for p, h in existing_manual.items()):
        raise RuntimeError('Original image or protected manual input changed.')
    ready = all(report['all_foreground_classes_present'] for report in frame_reports.values())
    result = {'backend': 'nnunet', 'stage': 'stage01_seg_only',
              'status': 'success' if ready else 'completed_requires_manual_correction',
              'all_foreground_classes_present': ready,
              'subject': subject.name, 'SAX_only': True, 'configuration': '2d',
              'original_image_shape': list(source_shape),
              'folds': args.folds, 'checkpoint': args.checkpoint, 'model_dir': str(args.nnunet_model_dir),
              'input_image_sha256': original_hash, 'original_and_manual_inputs_unchanged': True,
              'work_dir': str(work), 'frames': frame_reports}
    (subject / 'dmacs_nnunet_stage01_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segmentation_backend', choices=('nnunet',), default='nnunet')
    parser.add_argument('--data-dir', '--test_dir', dest='data_dir', type=Path, default=REPO / 'data')
    parser.add_argument('--subject', required=True, help='One subject directory name, e.g. subject_444.')
    parser.add_argument('--input-name', default='example.nii')
    for name in ('stage01_seg_only', 'stage02_surf_only', 'stage03_addi_mot_fit'):
        parser.add_argument('--' + name, type=boolean, nargs='?', const=True, default=False)
    parser.add_argument('--seg_manual', type=boolean, nargs='?', const=True, default=True,
                        help='Stage 02 only: true selects _manual labels; false selects automatic labels.')
    parser.add_argument('--nnunet-workspace', type=Path, default=REPO / 'model_nnunet/nnunet_workspace')
    parser.add_argument('--nnunet-model-dir', type=Path)
    parser.add_argument('--checkpoint', default='checkpoint_final.pth')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--rv-wall-radius-mm', type=float, default=3.0)
    parser.add_argument('--assume-mm-if-unknown', action='store_true')
    parser.add_argument('--docker-image', default='jinmingduan/segmentationmeshmotion:latest')
    parser.add_argument('--coreNo', type=int, default=8)
    parser.add_argument('--irtk', type=boolean, default=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if sum(getattr(args, name) for name in ('stage01_seg_only', 'stage02_surf_only', 'stage03_addi_mot_fit')) > 1:
        parser.error('Enable only one stage flag at a time.')
    subject = (args.data_dir / args.subject).resolve()
    if not subject.is_dir() or subject.parent != args.data_dir.resolve():
        parser.error('Subject must be an existing direct child of --data-dir.')
    if args.coreNo < 1:
        parser.error('--coreNo must be positive.')
    if not np.isfinite(args.rv_wall_radius_mm) or args.rv_wall_radius_mm <= 0:
        parser.error('--rv-wall-radius-mm must be positive and finite.')
    args.nnunet_model_dir = (args.nnunet_model_dir or args.nnunet_workspace / 'nnUNet_results' /
                            'Dataset100_UKBB_Petersen_SAX/nnUNetTrainer__nnUNetPlans__2d').resolve()
    run_dir = (args.run_dir or CODE_DIR / 'runs' / ('nnunet_' + subject.name)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stage = next((name for name in ('stage01_seg_only', 'stage02_surf_only', 'stage03_addi_mot_fit')
                  if getattr(args, name)), 'full')
    started = time.time()
    print('[nnUNet SAX] Subject:', subject.name, 'mode:', stage, flush=True)
    if stage in ('full', 'stage01_seg_only'):
        result = stage01(args, subject, run_dir)
        if stage == 'full' and not result['all_foreground_classes_present']:
            raise RuntimeError('Stage 01 predictions were saved but are incomplete. Correct the labels, '
                               'save both _manual files and resume with --stage02_surf_only=true.')
    if stage in ('stage02_surf_only', 'stage03_addi_mot_fit', 'full'):
        if stage == 'full':
            command = docker_command(args, subject, run_dir, 'refine_nnunet_automatic.py',
                                     ['--coreNo', str(args.coreNo), '--irtk', str(args.irtk)])
            run_logged(command, run_dir / 'stage02.log')
            stage = 'stage03_addi_mot_fit'
        extra = ['--' + stage + '=true', '--coreNo', str(args.coreNo),
                 '--irtk', str(args.irtk), '--model_path', '/must_not_be_loaded']
        if stage == 'stage02_surf_only':
            extra.append('--seg_manual=' + str(args.seg_manual).lower())
        command = docker_command(args, subject, run_dir, 'DMACS.py', extra)
        run_logged(command, run_dir / ('stage02.log' if stage == 'stage02_surf_only' else 'stage03.log'))
    print('[nnUNet SAX] Completed in %.3f seconds.' % (time.time() - started), flush=True)


if __name__ == '__main__':
    main()
