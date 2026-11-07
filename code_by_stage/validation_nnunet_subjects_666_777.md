# SAX nnUNet Stage Tests: Subjects 666 and 777

Date: 2026-11-07. The earlier subject_444/555 tests were stopped at the
user's request. Their partial files were retained, and no further processing
of those subjects is part of this test.

## Implementation and Input Rules

All modified/new source files are under `code_by_stage/`. The new backend
uses the supplied SAX-only nnUNet model, preserves atlas refinement, and
keeps the original Stage 03 implementation. The original `code/` and model
files are not modified. Select the new backend explicitly with
`--segmentation_backend=nnunet`; legacy execution remains the default.

The canonical labels are background=0, LV blood=1, LV myocardium=2,
synthetic RV myocardium=3, and RV blood=4. Native nnUNet RV blood is 3 and
is remapped to 4 before the wall is constructed. RV myocardium is added
only into background, within a 3 mm voxel-center radius in each XY slice.
No existing nonzero label is overwritten and no Z growth is performed.
This is not a guarantee of 3 mm continuous wall thickness after atlas
refinement; see [readme_model.md](readme_model.md#9-sax-nnunet-backend-and-xy-only-rv-wall).

Both new input images have shape `300 x 300 x 12 x 31`, with original
spacings approximately `0.9375 x 0.9375 x 8.000003 mm` in the supplied
affine. Their original spatial/time unit declarations are unknown; the
existing preprocessing helper produces explicitly millimeter-based phase
grids, which the new wall builder validates. Original images are preserved.

The original image checksum is identical for the two test folders:

```text
292a4f2b8c9263548815daf6855a25eb287ba2066926774a02d9cfb47ff35c33
```

Despite identical images, inference was executed independently for 666 and
777. No segmentation from another subject was reused.

## Subject 666: Stage 01 Completed

```bash
conda run --no-capture-output -n cme-dt-env python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_666 --stage01_seg_only=true
```

Runtime reported by the launcher: `20.867 seconds` for preprocessing,
five-fold ED/ES inference, mapping, and wall construction, excluding earlier
launcher imports. This is an observed test timing, not a runtime guarantee.

Final outputs:

```text
data/subject_666/segs/seg_lvsa_ED.nii.gz
data/subject_666/segs/seg_lvsa_ES.nii.gz
```

Both have shape `300 x 300 x 12`, int16 labels, and the same spatial grid,
units, qform, and sform as their corresponding preprocessed phase images.

| Phase | Background 0 | LV blood 1 | LV myo 2 | Added RV myo 3 | RV blood 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ED | 1,016,810 | 28,156 | 13,465 | 4,133 | 17,436 |
| ES | 1,039,398 | 17,602 | 12,954 | 2,648 | 7,398 |

The maximum added voxel-center distance is `2.9646353064078554 mm`, within
the requested 3 mm expansion radius. This is expected on a discrete grid;
the threshold does not require an added voxel center to lie exactly at 3 mm.
All protected native anatomical labels are unchanged, and slices without RV
blood have no added wall. Neither fitting nor raw surface generation ran
for subject 666.

Additional native/mapped labels, added-wall masks and JSON reports are in
its `segs/` folder. Execution and verification evidence:

- `runs/nnunet_subject_666/stage01.log`.
- `runs/nnunet_subject_666/validation_stage01.json`.
- `runs/nnunet_subject_666/ED_ES_segmentation_preview.png`.
- `data/subject_666/dmacs_nnunet_stage01_summary.json` (relative to project root).

The checks establish file integrity, label semantics, geometry preservation,
and growth behavior, not clinical segmentation accuracy.

## Subject 777: Own Predictions and Manual Copies

Stage 01 was run independently for subject 777 to obtain its actual inputs
for the requested Stage 02 test:

```bash
conda run --no-capture-output -n cme-dt-env python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_777 --stage01_seg_only=true

conda run --no-capture-output -n cme-dt-env python -B code_by_stage/tests/prepare_manual_fixture.py \
  --subject-dir data/subject_777 \
  --report code_by_stage/runs/nnunet_subject_777/manual_test_fixture.json
```

Its Stage 01 launcher time was `20.902 seconds`; label counts and physical
growth checks match the table above. Each `_manual` file is a byte-identical
copy of **that same subject's own canonical automatic segmentation**:

```text
subject_777/segs/seg_lvsa_ED.nii.gz
  -> subject_777/segs/seg_lvsa_ED_manual.nii.gz
subject_777/segs/seg_lvsa_ES.nii.gz
  -> subject_777/segs/seg_lvsa_ES_manual.nii.gz
```

The copied automatic/manual checksums are:

```text
ED ccfda454966058e07446f08c6be8c28b7945d9bede006e93103c88ac90e4605f
ES 00d03a92d4ddec5d640c26804e049ca0ba7887685197212b0508cc0b8b56f903
```

These copies are unedited predictions, not human corrections. No labels from
subjects 333, 444, 555, or 666 are used. The copy helper now accepts only a
single `--subject-dir`; it cannot select a different source subject, rejects
external source symlinks, validates both phases before copying, and refuses
to replace an existing manual file. Copying is an explicit test setup step,
not a production Stage 02 fallback.

## Subject 777: Stage 02

```bash
conda run --no-capture-output -n cme-dt-env python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_777 --stage02_surf_only=true
```

Status: **completed and validated for both ED and ES**. Stage 02 took
`991.368 seconds` (about 16.5 minutes), mostly the unchanged atlas search
and registration. The log confirms that its two own `_manual` files are
the refinement inputs.
The old CNN checkpoint path is deliberately set to `/must_not_be_loaded` in
the Docker resume command; Stage 02 must not load a segmentation checkpoint.

The required final raw surface files under `data/subject_777/vtks/` are:

```text
LVendo_ED.vtk  LVepi_ED.vtk  RV_ED.vtk  RVepi_ED.vtk  LVmyo_ED.vtk
LVendo_ES.vtk  LVepi_ES.vtk  RV_ES.vtk  RVepi_ES.vtk  LVmyo_ES.vtk
```

All ten VTKs were reloaded successfully and contain finite coordinates:

| File | Points | Cells |
| --- | ---: | ---: |
| LVendo_ED.vtk | 9,268 | 18,362 |
| LVepi_ED.vtk | 12,974 | 25,718 |
| RV_ED.vtk | 8,537 | 16,910 |
| RVepi_ED.vtk | 11,012 | 21,838 |
| LVmyo_ED.vtk | 22,170 | 43,938 |
| LVendo_ES.vtk | 6,282 | 12,560 |
| LVepi_ES.vtk | 12,390 | 24,776 |
| RV_ES.vtk | 5,642 | 11,280 |
| RVepi_ES.vtk | 7,774 | 15,544 |
| LVmyo_ES.vtk | 16,564 | 33,124 |

`RV_<phase>.vtk` is the original RV endocardial filename. The auxiliary
LVmyo files and fitting masks are retained. Stage 03 fitting/motion is not
requested or started.

Final checks passed:

- Both manual inputs remain byte-identical to subject 777's own automatic
  predictions and the recorded pre-refinement checksums.
- Both `sizes/2D_segmentation_<phase>.nii.gz` archives match those inputs.
- The original Stage 02 input hashes are unchanged.
- There are exactly ten raw VTKs, with no fitted/prefixed surfaces or motion.
- The resume log contains no CNN execution or Python traceback, and zero
  `input mapping non-invertible` warnings.
- The Stage 01 output validator passes again after Stage 02, including
  native labels, physical XY growth, image forms, and the source-image hash.

The final validator command was:

```bash
conda run --no-capture-output -n morphinet python -B code_by_stage/tests/validate_subject.py \
  --subject-dir data/subject_777 --run-dir code_by_stage/runs/nnunet_subject_777 \
  --stage 02 --require-manual-copy
```

`--require-manual-copy` is an optional validation-only check for this unedited
test setup. It is not required for genuinely human-edited manual inputs in
normal use. The final report is `runs/nnunet_subject_777/validation_stage02.json`.

## Automated Tests

- All 26 modern backend, file-handling, XY-growth, and same-subject-copy
  tests passed in `cme-dt-env`.
- All 17 legacy staged-pipeline regression tests passed in the original
  Docker environment.
- Stage 01 output validators passed for both 666 and 777, including native
  label mapping, nonzero-label preservation, no Z growth, physical distances,
  and NIfTI geometry/form preservation.
- All 25 checked original source/model/input files matched their starting
  SHA-256 values. The complete record is
  `runs/nnunet_subject_777/input_preservation.json`.

Test logs are `runs/nnunet_subject_777/unit_tests_modern.log` and
`unit_tests_legacy.log`. Stage 02 logs are `stage02.log` and
`stage02_console.log` in the same folder. The additional post-refinement
label/geometry check is `validation_stage01_after_stage02.json`.

Logs and machine-readable reports are retained under the ignored
`code_by_stage/runs/nnunet_subject_666/` and `nnunet_subject_777/` folders.
