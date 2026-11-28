"""Run only the original cine preprocessing in legacy Docker Python."""
from __future__ import print_function

import argparse
import os
import nibabel as nib
from image_utils import imagePreprocessing, validate_nonnegative_image
from stage_utils import existing_segmentations, protect_files, require_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-dir', required=True)
    parser.add_argument('--input-name', default='example.nii')
    parser.add_argument('--atlas-dir', default='/refs')
    parser.add_argument('--native-frame-phases', action='store_true',
                        help='Rebuild ED/ES MRIs from the selected original acquisition frames.')
    args = parser.parse_args()
    source = require_file(os.path.join(args.subject_dir, args.input_name))
    with protect_files([source] + existing_segmentations([args.subject_dir])):
        for name in ('segs', 'dofs', 'tmps', 'sizes', 'vtks', 'motion'):
            path = os.path.join(args.subject_dir, name)
            if not os.path.isdir(path):
                os.makedirs(path)
        imagePreprocessing(source, args.subject_dir, args.atlas_dir,
                           native_frame_phases=args.native_frame_phases)
        for name in ('lvsa_.nii.gz', 'lvsa_ED.nii.gz', 'lvsa_ES.nii.gz'):
            image = nib.load(require_file(os.path.join(args.subject_dir, name)))
            if len(image.shape) != 4 or (name != 'lvsa_.nii.gz' and image.shape[3] != 1):
                raise ValueError('Unexpected preprocessed cine/phase shape: ' + name)
            validate_nonnegative_image(os.path.join(args.subject_dir, name))
    print('[nnUNet SAX] Legacy preprocessing complete; no TensorFlow inference was run.')


if __name__ == '__main__':
    main()
