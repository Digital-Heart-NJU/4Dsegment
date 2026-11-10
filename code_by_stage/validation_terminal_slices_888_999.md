# Stage 01 Terminal-Slice Cleanup: Subjects 888 and 999

Date: 2026-11-10. Source edits are confined to `code_by_stage/`.

## Implemented Rule

The nnUNet Stage 01 path now runs this order:

```text
Original example.nii
  -> existing cine preprocessing and ED/ES detection
  -> supplied SAX nnUNet inference
  -> native RV blood 3 remapped to canonical 4
  -> existing XY-only 3 mm RV myocardium expansion
  -> clear all foreground labels at whole-image Z indices 0 and N_z-1
  -> save final seg_lvsa_ED.nii.gz and seg_lvsa_ES.nii.gz
```

Only the two terminal segmentation planes are cleared. The first/last
heart-containing planes inside the volume are not selected for deletion.
All other label voxels are unchanged relative to the existing pipeline.
Image slices are not removed; shape, spacing, affine, qform/sform, and the
original MRI file remain unchanged by this cleanup.

The slice count is read from the original `example.nii`; a mismatch in the
preprocessed phase image causes an error. ED and ES are checked independently.
Already empty terminal planes remain empty. One-/two-slice inputs have all
their segmentation cleared and any missing anatomy is reported.

The helper is `finalize_stage01_labels` in `nnunet_pipeline.py`. It is used
by nnUNet Stage 01 and nnUNet full-automatic mode. The legacy CNN path and
the Stage 02/03 algorithms were not modified.

## Saved Diagnostics

- `segs/seg_lvsa_ED/ES.nii.gz`: final segmentation with cleared endpoints.
- `segs/rv_myo_added_ED/ES.nii.gz`: only the synthetic RV wall retained after cleanup.
- `segs/nnunet_native/seg_lvsa_ED/ES.nii.gz`: unchanged native predictions.
- `segs/nnunet_mapped_ED/ES.nii.gz`: unchanged mapped predictions before growth/cleanup.
- `segs/rv_myo_added_ED/ES_summary.json`: retained voxel counts and explicit
  `terminal_z_cleanup` statistics; the unmodified growth statistics are
  preserved under `growth_before_terminal_z_cleanup`.

Because clearing the endpoints intentionally removes original labels,
`protected_labels_unchanged` may be false globally. The separate
`nonterminal_protected_labels_unchanged` and
`terminal_z_cleanup.intermediate_slices_unchanged` checks must remain true.

## Subject 888: Stage 01 Passed

```bash
conda run --no-capture-output -n cme-dt-env python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_888 --stage01_seg_only=true
```

The final arrays remain `300 x 300 x 12`. The original MRI remains
`300 x 300 x 12 x 31`. Both final segmentation planes at `z=0` and `z=11`
are entirely background for ED and ES. Every voxel in slices 1-10 matches
the reconstructed pre-cleanup segmentation exactly.

| Phase | Foreground removed at z=0 | Foreground removed at z=11 | Synthetic wall removed |
| --- | ---: | ---: | ---: |
| ED | 0 | 4,080 | 304 |
| ES | 0 | 0 | 0 |

ED's removed labels were LV blood=2,313, LV myocardium=881, synthetic RV
myocardium=304, and RV blood=582. ES required no deletion because both
endpoints were already empty. Both phases retain all four foreground classes.

Final label counts:

| Phase | Background 0 | LV blood 1 | LV myo 2 | RV myo 3 | RV blood 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ED | 1,020,890 | 25,843 | 12,584 | 3,829 | 16,854 |
| ES | 1,039,398 | 17,602 | 12,954 | 2,648 | 7,398 |

Outputs are under `data/subject_888/segs/`. Stage 01 took `20.873 seconds`
as measured by the launcher. No surface extraction, fitting, or motion was
run for subject 888.

Validation:

```bash
conda run --no-capture-output -n cme-dt-env python -B code_by_stage/tests/validate_nnunet.py \
  --subject-dir data/subject_888 \
  --output code_by_stage/runs/nnunet_subject_888/validation_stage01.json
```

The validator reconstructs pre-cleanup growth, checks exact intermediate
array equality, terminal background, native mapping, retained-wall masks,
physical distances, source-image hashes and NIfTI geometry/forms.

## Subject 999: Own Cleaned Inputs

Stage 01 was executed independently on subject 999 (`20.819 seconds`) and
passed the same cleanup and label checks. Although the two original input
images have identical checksums, no prediction was copied from subject 888.
These are repeated technical test images, not independent validation of
anatomical generalization.

```bash
conda run --no-capture-output -n cme-dt-env python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_999 --stage01_seg_only=true
conda run --no-capture-output -n cme-dt-env python -B code_by_stage/tests/prepare_manual_fixture.py \
  --subject-dir data/subject_999 \
  --report code_by_stage/runs/nnunet_subject_999/manual_test_fixture.json
```

Its `_manual` files are unedited, byte-identical copies of its own cleaned
automatic predictions. The automatic/manual SHA-256 values are:

```text
ED 277cde89d90227025be3bdd9aef8943252b4779325de7270d390199705f4d5d8
ES 00d03a92d4ddec5d640c26804e049ca0ba7887685197212b0508cc0b8b56f903
```

No labels from another subject are used, and the existing copy helper
refuses to overwrite any manual inputs. Both manual inputs have empty
first and last planes, matching the cleaned automatic files.

## Subject 999: Stage 02

```bash
conda run --no-capture-output -n cme-dt-env python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_999 --stage02_surf_only=true
```

Status: **completed and validated for ED and ES**. Stage 02 took
`943.920 seconds` (about 15.7 minutes).
The original atlas selection, registration, label fusion, and marching-cubes
operations are retained. Expected raw outputs under `data/subject_999/vtks/`:

```text
LVendo_ED.vtk  LVepi_ED.vtk  RV_ED.vtk  RVepi_ED.vtk  LVmyo_ED.vtk
LVendo_ES.vtk  LVepi_ES.vtk  RV_ES.vtk  RVepi_ES.vtk  LVmyo_ES.vtk
```

All ten VTKs reload successfully with nonempty cells and finite coordinates:

| File | Points | Cells |
| --- | ---: | ---: |
| LVendo_ED.vtk | 9,964 | 19,924 |
| LVepi_ED.vtk | 14,876 | 29,748 |
| RV_ED.vtk | 9,414 | 18,824 |
| RVepi_ED.vtk | 11,924 | 23,844 |
| LVmyo_ED.vtk | 22,396 | 44,788 |
| LVendo_ES.vtk | 6,284 | 12,564 |
| LVepi_ES.vtk | 12,398 | 24,792 |
| RV_ES.vtk | 5,638 | 11,272 |
| RVepi_ES.vtk | 7,772 | 15,540 |
| LVmyo_ES.vtk | 16,576 | 33,148 |

Final verification passed:

- Automatic predictions, manual copies and `sizes/2D_segmentation_*` archives
  retain the recorded SHA-256 values, independently for ED and ES.
- All six automatic/manual/archive arrays retain shape `300 x 300 x 12`
  and zero foreground on their first and last Z planes.
- The Stage 01 validator passes again after Stage 02, including exact
  intermediate-slice preservation, retained-wall masks and image geometry.
- There are exactly ten raw surfaces, with no fitted surfaces or motion.
- The resume log contains no Python traceback, no CNN execution and zero
  `input mapping non-invertible` warnings.

Commands used for final validation:

```bash
conda run --no-capture-output -n morphinet python -B code_by_stage/tests/validate_subject.py \
  --subject-dir data/subject_999 --run-dir code_by_stage/runs/nnunet_subject_999 \
  --stage 02 --require-manual-copy
conda run --no-capture-output -n cme-dt-env python -B code_by_stage/tests/validate_nnunet.py \
  --subject-dir data/subject_999 \
  --output code_by_stage/runs/nnunet_subject_999/validation_stage01_after_stage02.json
```

Machine-readable evidence in `runs/nnunet_subject_999/` includes
`validation_stage02.json`, `validation_stage01_after_stage02.json`,
`terminal_inputs_after_stage02.json` and `manual_test_fixture.json`.

Stage 03 is not requested. This change does not re-clear the atlas-refined
segmentation: Stage 02 operates on its own resampled grid and can alter
boundaries. The endpoint guarantee applies to Stage 01's final canonical
files and their unchanged manual/archive copies, not to every later grid.

## Automated Checks

- All 32 modern tests passed, including six new terminal-slice tests.
- All 17 legacy staged-pipeline regression tests passed in Docker.
- Stage 01 output validation passed for subjects 888 and 999.
- Stage 02 output validation passed for subject 999.
- All 25 checked original source/model/input files matched their initial
  SHA-256 values; see `runs/nnunet_subject_999/input_preservation.json`.
- Tests cover clearing all foreground classes, unchanged intermediate
  slices, retained first/last interior heart-containing slices, independent
  endpoints, empty endpoints, one-/two-slice stacks, mismatch rejection,
  raw diagnostic preservation, retained-wall masks, and final-file geometry.

Execution logs and JSON evidence remain in
`code_by_stage/runs/nnunet_subject_888/` and `nnunet_subject_999/`.
These are integrity and workflow checks, not clinical accuracy certification.
