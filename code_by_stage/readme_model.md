# 4Dsegment With Three Execution Stages

This is a separate, staged version of the original pipeline. All runtime
Python modules are in `code_by_stage/`. The original `code/` sources and the
shared pretrained checkpoint under `model/` are not modified.

For the original image-to-mesh algorithms and file families, see
[the original pipeline explanation](../code/readme_model_origin_end-to-end.md).
This document explains the new execution boundaries and manual-input rules.

## 1. Commands and Boundaries

Run these commands from the staged code directory **inside the supplied
legacy Docker environment**, as shown in Section 6.

| Command | Inputs and processing | Where it stops |
| --- | --- | --- |
| `python DMACS.py` | The complete original automatic pipeline using CNN predictions. Existing manual files are preserved but not selected. | After fitting, measurements, motion tracking, and decimation. |
| `python DMACS.py --stage01_seg_only=true` | Original cine preprocessing and ED/ES CNN inference. | After saving the two automatic segmentations. |
| `python DMACS.py --stage02_surf_only=true` | Manual ED/ES labels, original atlas refinement, and original raw surface extraction. | After raw surfaces for both phases have been saved, before any fitting or motion. |
| `python DMACS.py --stage03_addi_mot_fit=true` | Saved raw surfaces and image masks from Stage 02, followed by the original remaining operations. | After fitting, measurements, motion tracking, and decimation. |

Enable at most one stage flag. All three default to `false`; with no enabled
flag, the full automatic workflow is used. The shorter proposed names
`--seg_only` and `--surf_only` are not the implemented flags.

The overall staged workflow is:

```text
example.nii
    |
    v
Stage 01: preprocessing + CNN segmentation of ED/ES
    |
    +--> segs/seg_lvsa_ED.nii.gz
    +--> segs/seg_lvsa_ES.nii.gz
    |
    STOP: inspect and edit labels in an image editor
    |
    +--> segs/seg_lvsa_ED_manual.nii.gz
    +--> segs/seg_lvsa_ES_manual.nii.gz
    |
    v
Stage 02: manual labels -> atlas refinement -> marching-cubes surfaces
    |
    +--> unprefixed ED/ES surfaces and saved fitting masks
    |
    STOP: inspect raw surfaces
    |
    v
Stage 03: template fitting -> measurements -> motion -> decimation
    |
    +--> fitted surface families, measurement text files, motion sequence
```

## 2. Stage 01: Automatic Segmentation

```bash
python DMACS.py --stage01_seg_only=true --coreNo 8 --irtk True
```

This calls the original preprocessing and slice-wise neural inference:
`headertool`, temporal alignment, contrast processing, ED/ES detection,
checkpoint restoration, normalization, padding, and region prediction.

The automatic results are:

```text
data/<subject>/segs/seg_lvsa_ED.nii.gz
data/<subject>/segs/seg_lvsa_ES.nii.gz
```

The phase images `lvsa_ED.nii.gz` and `lvsa_ES.nii.gz`, the preprocessed
`lvsa_.nii.gz` cine, and the working directories are also retained.
No atlas refinement, raw surface extraction, fitting, or motion tracking
is performed in this mode.

Inspect the predictions against the **preprocessed phase images**, not an
unregistered original-scanner image. Save edited labels separately as:

```text
data/<subject>/segs/seg_lvsa_ED_manual.nii.gz
data/<subject>/segs/seg_lvsa_ES_manual.nii.gz
```

The program never automatically creates these manual files. It does not
rename them over the automatic predictions. The original predictions remain
available for comparison.

An empty or incomplete but otherwise valid CNN label map can still be saved
at Stage 01 so that it can be manually corrected. It is not rejected merely
because one anatomical class is absent.

**Rerun warning:** Stage 01 and the full automatic mode still reset generated
`dofs`, `tmps`, `sizes`, `motion`, and `vtks` directories, as the original
segmentation path does. Back up downstream results before rerunning them.
The `segs` directory is no longer removed wholesale: its manual files and
other files are retained, while the two automatic predictions are rewritten.

## 3. Stage 02: Manual Labels to Raw Surfaces

```bash
python DMACS.py --stage02_surf_only=true --coreNo 8 --irtk True
```

### Required inputs

For every subject directory under `test_dir`:

- `segs/seg_lvsa_ED_manual.nii.gz` and `segs/seg_lvsa_ES_manual.nii.gz`.
- The corresponding preprocessed phase MRI files, either in the subject
  root or archived under `sizes/lvsa_ED.nii.gz` and `sizes/lvsa_ES.nii.gz`.
- The same atlas and registration resources used by the original pipeline.

Manual inputs may be 3D or 4D with a singleton time dimension. They must
match the original phase image's spatial dimensions and affine geometry.
Labels must be finite integer values in `0..4`, with all four foreground
classes present for generation of all ventricular surfaces:

| Label | Meaning |
| --- | --- |
| 0 | Background |
| 1 | LV blood pool |
| 2 | LV myocardium |
| 3 | RV myocardium |
| 4 | RV blood pool |

All subjects' manual inputs are checked before resuming processing. Missing
manual files cause an error; there is no fallback to automatic predictions
and no automatic rerun of the CNN. If phase MRI files were archived by an
earlier Stage 02 run, their copies are restored to the working root before
refinement. Neither the archives nor manual labels are modified.

### Processing and outputs

The manual filename is passed directly into the existing atlas-selection
and refinement functions. Those functions retain the original candidate
atlases, best-three selection, affine/nonrigid registration, label fusion,
resampling, and mask-cleanup settings.

The refined outputs remain:

```text
PHsegmentation_ED.gipl
PHsegmentation_ES.gipl
sizes/3D_segmentation_ED.nii.gz
sizes/3D_segmentation_ES.nii.gz
```

`sizes/2D_segmentation_<phase>.nii.gz` archives the **actual manual input**
used for refinement. The old root-versus-`segs/` backup path inconsistency
is corrected in this copy of the code. Neither the manual input nor its
automatic counterpart is deleted after archiving.

The requested raw surfaces are:

```text
vtks/LVendo_ED.vtk    vtks/LVendo_ES.vtk
vtks/LVepi_ED.vtk     vtks/LVepi_ES.vtk
vtks/RV_ED.vtk        vtks/RV_ES.vtk
vtks/RVepi_ED.vtk     vtks/RVepi_ES.vtk
```

`RV_<phase>.vtk` is the original code's RV endocardial filename. No anatomical
output is renamed to `RVendo_<phase>.vtk` by this staged implementation.

The auxiliary `LVmyo_ED.vtk` and `LVmyo_ES.vtk` and extraction masks under
`tmps/` are also saved because subsequent fitting depends on them. In
particular, retain `tmps/vtk_LV_<phase>.nii.gz` and
`tmps/vtk_RV_<phase>.nii.gz` for both phases.

The extraction still uses the same masks and `mcubes ... 120 -blur 2`.
Stage 02 stops here: it does not create `N_`, `F_`, curvature/measurement
families, or motion meshes.

**Atlas refinement remains enabled.** Manual labels guide this refinement;
they are not hard boundary constraints that guarantee the final mesh exactly
follows every manually edited voxel.

Rerunning Stage 02 regenerates raw surfaces and clears the extraction's
generated `vtks/` and `tmps/` contents before rebuilding them. It does not
reset `segs/`, perform CNN inference, or delete manual segmentations.
The original extraction also removes subject-root `*.txt` files; keep
unrelated notes and backups outside that location before rerunning Stage 02.

## 4. Stage 03: Saved Surfaces to Fitting and Motion

```bash
python DMACS.py --stage03_addi_mot_fit=true --coreNo 8 --irtk True
```

This requires Stage 02's ten raw surfaces, the saved LV/RV image masks for
both phases, and `lvsa_.nii.gz` for motion estimation. Missing inputs produce
an error before fitting starts. Stage 03 does not silently reconstruct them
or rerun an earlier stage.

It runs the remaining original operations in their original order:

```text
Rigid/template alignment
    -> affine and nonrigid image/surface fitting
    -> template surface mapping
    -> wall thickness, template distance, and curvature measurements
    -> cine motion tracking and RV mesh propagation
    -> reference RV decimation and point matching
```

Outputs include the original `N_`, `F_`, `C_`, `FC_`, `S_`, and `W_` surface
families, measurement text files, and `motion/RV_frNN.vtk`/`.txt`.
The fitting-only function does not execute the raw-extraction cleanup,
`binarize`, or `mcubes` operations. Raw surface and required mask hashes are
checked before and after the stage, as are existing segmentation files.

This stage does not read the neural checkpoint, run atlas refinement, or
select new manual labels. If manual labels have changed, rerun Stage 02
before Stage 03 so the saved raw surfaces correspond to those edits.

The original limitation is retained: motion runs for all valid subjects,
but the default non-parallel decimation processes only the first valid
subject under `test_dir`.

## 5. File Protection and Validation

- Only one stage flag may be enabled.
- Manual files are protected by SHA-256 checks in every mode.
- Both automatic and manual segmentation files are protected in resume modes.
- Stage 03 additionally protects raw surfaces and its required saved masks.
- All stage runs create a `dmacs_<mode>_summary.json` under each subject
  after successful stage processing and input-preservation checks.
- Summary hashes refer to paths as seen by the process, usually `/data/...`
  inside Docker. Map them through the Docker bind mounts when inspecting
  the same files on the host.

Post-run file-existence checks and input hashes are execution checks, not
certificates of mesh quality or anatomical accuracy. Most legacy external
commands still use the original `os.system` handling. Review registration
warnings and inspect the meshes; non-invertible-transform warnings can occur
even when required outputs are written.

## 6. Docker Setup and Example Commands

Use the existing `jinmingduan/segmentationmeshmotion:latest` image. The code
still requires its legacy TensorFlow/NiBabel versions and compiled tools.
`model/` remains shared, read-only, and unchanged. Resume modes skip checkpoint
loading, but the current legacy program still imports TensorFlow for flags
and module imports.

Create a persistent dataset parent containing only the intended subject:

```bash
PROJECT=/home/mengao/Code/4Dsegment
SUBJECT=subject_333
RUN_ROOT="$PROJECT/code_by_stage/runs/$SUBJECT"
mkdir -p "$RUN_ROOT/$SUBJECT"

docker run --rm -it --name "dmacs_stages_$SUBJECT" \
  --user "$(id -u):$(id -g)" \
  -e LD_LIBRARY_PATH=/lib64 \
  -e OMP_NUM_THREADS=8 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "$RUN_ROOT:/data" \
  -v "$PROJECT/data/$SUBJECT:/data/$SUBJECT" \
  -v "$PROJECT/code_by_stage:/code:ro" \
  -v "$PROJECT/model:/model:ro" \
  -w /code \
  jinmingduan/segmentationmeshmotion:latest /bin/bash
```

The `/code` container mount above points to **`code_by_stage/`**, not the
original `code/` folder. `/data` is the writable parent dataset directory;
do not set `--test_dir` directly to a populated subject folder.

Inside the container:

```bash
# Stage 01: then edit the saved labels on the host and save _manual files.
python DMACS.py --stage01_seg_only=true

# After both manual files are available:
python DMACS.py --stage02_surf_only=true

# After checking the saved raw ED/ES surfaces:
python DMACS.py --stage03_addi_mot_fit=true
```

Each command returns to the shell at its boundary. The container can stay
open while an image editor on the host modifies the bind-mounted files.
Alternatively, exit and launch the same mounts for each separate command.
Do not change manual files while a stage using them is running.

The main subject outputs are saved under `data/<subject>/`. Dataset-level
volume summaries, `subjnames.txt`, and `matchedpointsnew.txt` remain under
`RUN_ROOT`, which survives container removal. Logs can also be kept there.

To run fully automatically instead, invoke `python DMACS.py` with no enabled
stage flags. It uses the automatic CNN predictions, even if manual files exist.

## 7. Changed and Copied Modules

| File | Role |
| --- | --- |
| [DMACS.py](DMACS.py) | Stage flags, dispatch, boundary checks, protection, summaries. |
| [stage_utils.py](stage_utils.py) | Manual-label validation, phase restoration, protected hashes, stage output checks. |
| [deepseg.py](deepseg.py) | Original inference; cleanup adjusted to retain `segs/` contents. |
| [p2processing.py](p2processing.py) | Original refinement; optional filename suffix selects manual inputs in Stage 02. |
| [image_utils.py](image_utils.py) | Original helpers; backup uses the actual selected segmentation path. |
| [meshfitting.py](meshfitting.py) | Extraction and fitting split into separate functions; default wrapper calls both in the original order. |
| [motionEstimation.py](motionEstimation.py) | Unchanged copy of the original motion implementation. |
| [decimation.py](decimation.py) | Unchanged copy of the original decimation implementation. |
| [p1processing.py](p1processing.py) | Copied optional 2D atlas code; still not used by the default runner. |

Neural weights, normalization, padding, atlas search, deformation settings,
mask rules, marching-cubes parameters, fitting commands, motion, and decimation
methods are not replaced by new algorithms.

## 8. Tests and Subject 333 Verification

Run the unit/regression tests against the actual legacy environment:

```bash
PROJECT=/home/mengao/Code/4Dsegment
docker run --rm --network none \
  -e LD_LIBRARY_PATH=/lib64 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e DMACS_ORIGINAL_CODE=/original_code \
  -v "$PROJECT/code_by_stage:/code:ro" \
  -v "$PROJECT/code:/original_code:ro" \
  -w /code \
  jinmingduan/segmentationmeshmotion:latest \
  python -B -m unittest discover -s tests -v
```

The tests cover flag dispatch, missing manual inputs, label and geometry
validation, phase restoration, manual-file retention, direct manual-path
selection, archive correctness, and exact command-list parity between the
original mesh generator and the extracted/fitting functions combined.

Subject 333 execution evidence is saved under `validation/subject_333_run/`.
For this technical integration test, no human-corrected segmentation was
provided, so `_manual` files are explicitly **uncorrected copies of Stage 01
predictions**. This verifies the manual-file loading path and preservation
rules, not the clinical quality of an actual manual correction.

After Stage 02 and before running Stage 03, capture the boundary and hashes
with the standalone validator (requires numpy, nibabel, and VTK; these are
available in the local `morphinet` environment):

```bash
conda run --no-capture-output -n morphinet python -B \
  code_by_stage/tests/validate_subject.py \
  --subject-dir data/subject_333 \
  --run-dir code_by_stage/validation/subject_333_run --stage 02
```

After Stage 03, run the same command with `--stage 03`. It compares all
saved Stage 02 surface, mask, and segmentation hashes, reloads surface and
motion VTKs, checks measurements and decimation outputs, and writes
`validation_stage02.json` / `validation_stage03.json` in the run directory.
It expects the stage execution logs named `stage02.log` and `stage03.log`.

The completed execution results, limitations, and links to evidence are
summarized in [validation_subject_333.md](validation_subject_333.md).
