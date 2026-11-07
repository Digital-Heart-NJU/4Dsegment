# SAX nnUNet Integration: Subjects 444 and 555

Date: 2026-11-07. All source changes are under `code_by_stage/`.

## Important Outcome

The new inference and postprocessing path executes, but the supplied
nnUNet ensemble predicts **only background for ED and ES** of these test
images. These are not usable cardiac segmentations. Stage 01 saves the
actual predictions for inspection and marks them as
`completed_requires_manual_correction`; it does not invent missing labels.

The original images of subjects 333, 444, and 555 have the identical SHA-256:

```text
5d79ce7f2d9019be944d4ea064a56d3bad80f914f3a8f5267fd8bc2cdc66b338
```

These are therefore repeated technical test cases, not independent evidence
of anatomical generalization.

## Implemented Behavior

- Explicit `--segmentation_backend=nnunet` selects the modern SAX backend.
- ED/ES detection and phase preprocessing reuse the original Docker helper.
- Inference uses the supplied 2D SAX model and all five final checkpoints.
- Native RV blood label 3 is remapped to canonical label 4.
- Synthetic RV myocardium becomes label 3 by up-to-3 mm physical expansion
  into background, independently per XY slice. Other nonzero labels are fixed.
- Original images and existing manual labels are protected.
- Stage 02 keeps the original manual-label atlas refinement and extraction.
- Stage 03 is unchanged and was not run for subjects 444 or 555.
- Default legacy execution remains available without the new backend flag.

See [readme_model.md](readme_model.md#9-sax-nnunet-backend-and-xy-only-rv-wall)
for runtime details, file meanings, and limitations of the voxel-center radius.

## Subject 444: Stage 01

Command from the project root:

```bash
conda run --no-capture-output -n cme-dt-env python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_444 --stage01_seg_only=true
```

The final verification rerun added `--overwrite` to replace this test's
earlier preprocessing outputs. Existing manual files are never overwritten
by that option.

Outputs:

```text
data/subject_444/segs/seg_lvsa_ED.nii.gz
data/subject_444/segs/seg_lvsa_ES.nii.gz
data/subject_444/segs/nnunet_native/seg_lvsa_ED.nii.gz
data/subject_444/segs/nnunet_native/seg_lvsa_ES.nii.gz
data/subject_444/segs/nnunet_mapped_ED.nii.gz
data/subject_444/segs/nnunet_mapped_ES.nii.gz
data/subject_444/segs/rv_myo_added_ED.nii.gz
data/subject_444/segs/rv_myo_added_ES.nii.gz
```

Both phase label maps have shape `151 x 121 x 12`, int16 final labels, and
219,252 background voxels. Neither contains foreground; no synthetic RV wall
is added without a predicted RV blood pool. JSON summaries are saved beside
the masks and at the subject root. No raw surfaces, fitting, or motion were
produced for subject 444.

Saved validation evidence:

- `runs/nnunet_subject_444/stage01.log`.
- `runs/nnunet_subject_444/validation_stage01.json`.
- `runs/nnunet_subject_444/stage01_preprocess.log`.

The validator checks the native-to-canonical mapping, preservation of
nonzero labels, phase geometry/forms/units, and physical XY growth rules.
Passing these integrity checks does not make an empty prediction valid.

## Inference Diagnostic Controls

The initial multiprocessing entry-point problem was fixed by re-executing
the modern launcher before nnUNet starts workers. They no longer import the
legacy TensorFlow entry point.

After that correction:

- A CPU fold-0 prediction on subject 444 also returned only background.
- Preprocessed image arrays and network logits/weights were finite.
- A fold-0 control prediction on the existing reference image
  `model_nnunet/nnunet_workspace/nnUNet_raw/Dataset12345_temp/imagesTs/`
  `subject003_sax_fr_00_0000.nii.gz` produced all four native classes.
- Applying the new wall builder to that control prediction added 2,584
  label-3 voxels, preserved all existing anatomical labels, and produced
  no Z growth. Maximum added voxel-center distance was
  `2.999999951962255 mm`.

The control outputs are isolated under
`runs/nnunet_subject_444/reference_probe/`; they are not subject 444 outputs.
The positive wall-growth check is recorded in `validation_wall_growth.json`.
The CPU predictions are retained in the earlier test prediction directory's
`labelsTs_cpu_fold0/` folder.

This demonstrates an input-dependent inference failure before label remapping
or RV wall construction. It does **not** establish the exact cause; image
context/domain compatibility needs further investigation. No weights, source
images, or training plans were modified to force a foreground result.

## Subject 555: Separate Stage 02 Test

Subject 555 initially contained only `example.nii`. Its preparatory nnUNet
Stage 01 run also produced background-only predictions, which remain saved
under its own `segs/` folder and were not replaced with other predictions.

To test the manual-input Stage 02 path independently, existing validation
labels from subject 333 were explicitly copied to subject 555's two
`*_manual.nii.gz` paths. Before copying, `tests/prepare_manual_fixture.py`
verified identical original image hashes, identical ED/ES phase arrays and
geometries, and presence of all required labels. It refuses to overwrite
existing manual files.

These are **unedited, pre-existing validation labels, not human corrections
or new nnUNet predictions**. Copying them is a test setup operation, not a
fallback in the production pipeline. Provenance and exact copied checksums
are saved in `runs/nnunet_subject_555/manual_test_fixture.json`.

```bash
conda run --no-capture-output -n cme-dt-env python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_555 --stage02_surf_only=true
```

Status: stopped at the user's request before Stage 02 completed. Partial
outputs are retained, but no successful final Stage 02 validation is claimed.
Testing moved to subjects 666 and 777. The current copy helper no longer
supports cross-subject fixtures; it accepts only same-subject automatic labels.

The requested raw surface names are `LVendo_ED/ES.vtk`, `LVepi_ED/ES.vtk`,
`RV_ED/ES.vtk`, and `RVepi_ED/ES.vtk` under `data/subject_555/vtks/`.
The original auxiliary `LVmyo_ED/ES.vtk` and fitting masks are also required
outputs. No fitting or motion is requested.

## Regression Tests and Input Preservation

- 21 modern nnUNet/postprocessing/launcher tests passed.
- All 17 existing legacy staged-pipeline regression tests passed in Docker.
- Original `code/` sources and documentation, legacy model files, supplied
  `segment_predict.py`, nnUNet plans/dataset metadata/five final weights,
  and the two test input images matched the starting 25-file SHA-256 snapshot.
- The user's `code_by_stage_backup/` folder was not modified.

Runtime logs and machine-readable validation reports are under the ignored
`code_by_stage/runs/` folder and remain available locally.
