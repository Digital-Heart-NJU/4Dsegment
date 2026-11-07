"""Continue automatic nnUNet labels through original refinement/extraction."""
from __future__ import print_function

import argparse
from p2processing import multiatlasreg3D
from meshfitting import meshCoregstration
from stage_utils import (subject_directories, validate_segmentations, prepare_refinement_inputs,
                         existing_segmentations, protect_files, raw_surface_inputs, record_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test_dir', default='/data')
    parser.add_argument('--coreNo', type=int, default=8)
    parser.add_argument('--irtk', default='True')
    args = parser.parse_args()
    subjects = subject_directories(args.test_dir)
    validate_segmentations(subjects, manual=False)
    with protect_files(existing_segmentations(subjects, include_automatic=True)) as hashes:
        prepare_refinement_inputs(subjects)
        multiatlasreg3D(args.test_dir, '/refs', '/par', args.coreNo, True, args.irtk.lower() == 'true')
        meshCoregstration(args.test_dir, '/par', '/vtks/1', args.coreNo, True, False, mode='extract')
        raw_surface_inputs(subjects)
    record_stage(subjects, 'nnunet_automatic_refine_extract', hashes)


if __name__ == '__main__':
    main()
