"""Copy a subject's actual automatic labels for an unedited Stage 02 test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare(subject, report_path):
    subject = subject.resolve()
    records = {}
    for frame in ('ED', 'ES'):
        original = subject / 'segs' / ('seg_lvsa_' + frame + '.nii.gz')
        destination = subject / 'segs' / ('seg_lvsa_' + frame + '_manual.nii.gz')
        if original.resolve().parent != (subject / 'segs').resolve():
            raise ValueError('Automatic labels must be local to this subject: ' + str(original))
        if destination.exists():
            raise FileExistsError('Never overwrite existing manual input: ' + str(destination))
        phase = subject / ('lvsa_' + frame + '.nii.gz')
        if not phase.is_file():
            phase = subject / 'sizes' / phase.name
        phase_image = nib.load(phase)
        labels = nib.load(original)
        values = np.unique(np.asarray(labels.dataobj))
        if (len(labels.shape) not in (3, 4) or (len(labels.shape) == 4 and labels.shape[3] != 1) or
                labels.shape[:3] != phase_image.shape[:3] or
                not np.allclose(labels.affine, phase_image.affine, atol=1e-5) or
                not np.isin(values, (0, 1, 2, 3, 4)).all() or
                not np.isin((1, 2, 3, 4), values).all()):
            raise ValueError('Automatic labels must match this phase and include classes 1..4: ' + frame)
        records[frame] = {'source': str(original.resolve()), 'destination': str(destination.resolve()),
                          'sha256': digest(original)}
    if report_path.exists():
        raise FileExistsError(report_path)
    for record in records.values():
        destination = Path(record['destination'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as output, Path(record['source']).open('rb') as source_file:
            shutil.copyfileobj(source_file, output)
        if digest(destination) != record['sha256']:
            raise RuntimeError('Fixture copy changed unexpectedly.')
    report = {'purpose': 'Stage 02 test using unedited copies of this subject\'s actual automatic segmentation.',
              'subject': subject.name, 'source_is_same_subject_automatic_prediction': True,
              'example_sha256': digest(subject / 'example.nii'), 'human_corrected': False,
              'frames': records}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-dir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.subject_dir, args.report)
