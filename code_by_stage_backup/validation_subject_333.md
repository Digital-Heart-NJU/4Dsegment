# Subject 333: Three-Stage Execution Verification

Date: 2026-11-05. Environment: the local
`jinmingduan/segmentationmeshmotion:latest` Docker image, using the staged
sources mounted as `/code` and the unchanged shared `model/` checkpoint.

## Results

| Check | Result |
| --- | --- |
| Unit/regression tests in the legacy environment | 17 passed. |
| Stage 01 | Passed; ED/ES CNN predictions saved; no surface or motion outputs. 4.833 seconds. |
| Stage 01 rerun with manual files already present | Passed; manual file hashes unchanged. 4.857 seconds. |
| Stage 02 | Passed; manual inputs used with atlas refinement; stopped at ten raw surfaces. 735.130 seconds. |
| Stage 03 | Passed; fitting, measurements, motion, and decimation completed. 1322.014 seconds. |
| Missing manual input | Correctly rejected, exit code 1, without falling back to automatic labels. |
| Premature Stage 03 without surfaces | Correctly rejected, exit code 1, before fitting. |
| Conflicting stage flags | Correctly rejected, exit code 1, before dataset processing. |
| Original `code/` Python and Markdown files | SHA-256 values unchanged. |
| Original `model/` files | SHA-256 values unchanged. |
| Original Subject 333 `example.nii` | SHA-256 unchanged. |

Both resume stages were deliberately invoked with
`--model_path /must_not_be_loaded`. They completed without restoring that
nonexistent checkpoint, confirming that they do not rerun CNN inference.

The default no-stage mode was checked by dispatch tests and exact mesh-command
parity against the original `meshGeneration`. A second complete automatic
production run was not performed. The three separate stages exercised the
actual preprocessing, CNN, atlas, extraction, fitting, and motion operations.

## Manual Inputs Used for This Technical Test

No human-corrected labels were provided. These files are explicitly
**uncorrected copies of the Stage 01 predictions**, not claimed manual edits:

```text
data/subject_333/segs/seg_lvsa_ED_manual.nii.gz
data/subject_333/segs/seg_lvsa_ES_manual.nii.gz
```

The files were created before the Stage 01 preservation rerun. Their hashes
remained unchanged through that rerun and through Stages 02 and 03. The two
automatic predictions were also unchanged during both resume stages.
The archived `sizes/2D_segmentation_<phase>.nii.gz` files match the selected
manual input files byte-for-byte.

Subject 333's input voxel array matched the existing Subject 211 example.
The new automatic ED/ES label arrays matched Subject 211's predictions from
the original pipeline, confirming unchanged neural inference for this input.

This test verifies the manual-file execution path and preservation rules;
it does not assess the accuracy of a human correction.

## Raw Surfaces at the Stage 02 Stop Point

`vtks/` contained exactly the following eight requested surfaces plus the
two auxiliary `LVmyo_ED.vtk` and `LVmyo_ES.vtk` files. No prefixed fitting
surfaces or motion meshes existed at that boundary.

| File | Points | Cells |
| --- | --- | --- |
| `LVendo_ED.vtk` | 10478 | 20952 |
| `LVepi_ED.vtk` | 15346 | 30688 |
| `RV_ED.vtk` | 9930 | 19798 |
| `RVepi_ED.vtk` | 12507 | 24952 |
| `LVendo_ES.vtk` | 6454 | 12904 |
| `LVepi_ES.vtk` | 13122 | 26240 |
| `RV_ES.vtk` | 6832 | 13660 |
| `RVepi_ES.vtk` | 9186 | 18368 |

After Stage 03, all ten raw surfaces, twelve extraction masks, and four
automatic/manual segmentation files matched their Stage 02 hashes: **26
files preserved**. Fitting wrote separate prefixed outputs instead of
replacing the raw surfaces or clearing the mask cache.

## Final Output Checks

- All 46 subject-level surface VTK files were readable, nonempty, and had
  finite point coordinates.
- All 20 normalized-cine RV motion meshes and their matching text files
  were present and the meshes were readable.
- ED/ES measurement text files, both reduced reference meshes, and
  dataset-level `matchedpointsnew.txt` were present.
- Stage 03's log contained no CNN segmentation or atlas-refinement execution.

**Quality limitation:** the unchanged registration implementation emitted
18,421 `input mapping non-invertible` warnings. Successful execution and
readable files do not certify anatomical accuracy, absence of surface
intersections, or valid deformation fields. No registration-method changes
or automatic geometric repairs were made as part of this staging task.

The final [subject VTK folder](../data/subject_333/vtks/) now contains both
the preserved raw surfaces and the additional Stage 03 fitting outputs.

## Evidence and Reproduction

- [Unit test log](validation/subject_333_run/unit_tests.log)
- [Stage 01 log](validation/subject_333_run/stage01.log)
- [Stage 01 manual-preservation rerun](validation/subject_333_run/stage01_manual_preservation_rerun.log)
- [Stage 02 log](validation/subject_333_run/stage02.log)
- [Stage 02 saved-file validation](validation/subject_333_run/validation_stage02.json)
- [Stage 03 log](validation/subject_333_run/stage03.log)
- [Stage 03 saved-file validation](validation/subject_333_run/validation_stage03.json)
- [Missing manual input test](validation/subject_333_run/stage02_missing_manual_expected_error.log)
- [Premature Stage 03 test](validation/subject_333_run/stage03_missing_surfaces_expected_error.log)
- [Conflicting flags test](validation/subject_333_run/conflicting_flags_expected_error.log)

The isolated dataset root and logs are in `validation/subject_333_run/`;
actual subject outputs are under `data/subject_333/`. Validation artifacts
are intentionally ignored by Git. See [readme_model.md](readme_model.md)
for stage commands, Docker mounts, file requirements, and validator commands.
