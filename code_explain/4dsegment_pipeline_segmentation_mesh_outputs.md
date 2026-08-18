# 4Dsegment Pipeline: Input To Segmentation And Mesh Outputs

This note explains the runtime pipeline in `code/DMACS.py`, from the input cine MRI file to the segmentation and mesh outputs. It focuses on files produced by the original 4Dsegment pipeline, not on the extra experimental volume-mesh scripts in `code_explain/`.

The active execution order is:

```text
deeplearningseg
multiatlasreg3D
meshCoregstration
motionTracking
decimate
```

The 2D atlas refinement function `multiatlasreg2D` exists in `code/p1processing.py`, but it is commented out in `DMACS.py`, so it is not part of the default run.

## Runtime And Path Assumptions

The code is meant to run inside the Docker image:

```text
jinmingduan/segmentationmeshmotion:latest
```

Inside Docker, the default paths are:

| Flag | Default path | Meaning |
|---|---|---|
| `test_dir` | `/data` | Subject input/output folders. This is usually mounted from local `data/`. |
| `model_path` | `/model/vgg_RE_network.ckpt-50000` | TensorFlow 1 checkpoint for CNN segmentation. |
| `atlas_dir` | `/refs` | Atlas segmentations and atlas landmarks used for shape refinement. |
| `param_dir` | `/par` | Registration parameter files. |
| `template_dir` | `/vtks/1` | ED/ES static mesh template used by `meshfitting.py`. |
| `template_PH` | `/vtks/2` | RV motion template used by `motionEstimation.py`. |

For normal running, local `data/`, `code/`, and `model/` are mounted into Docker as `/data`, `/code`, and `/model`.

## Expected Input Layout

Each subject should be a subfolder under `/data`:

```text
/data/
  subject_1/
    example.nii
```

or:

```text
/data/
  subject_1/
    example.nii.gz
```

The code searches for `*.nii` first. If none exist, it searches for `*.nii.gz`. The selected file is treated as the original 4D cine short-axis MRI input.

## Segmentation Labels

The label meaning is inferred from how `meshfitting.py` extracts structures:

| Label | Meaning | Evidence in mesh extraction |
|---:|---|---|
| 0 | Background | Implicit background. |
| 1 | LV cavity / LV endocardium | `binarize ... 1 1` creates `LVendo`. |
| 2 | LV myocardium | `binarize ... 2 2` creates `LVmyo`. |
| 3 | RV myocardium / RV wall | `RVepi` uses labels `3 4`. |
| 4 | RV cavity / RV endocardium | `binarize ... 4 4` creates `RV`. |

Important RV naming detail: the code does not output a file named `RVmyo`. The final RV cavity/endocardial surface is named `RV`, and the RV outer/wall surface is named `RVepi`.

## Stage 1: Preprocessing And ED/ES Frame Extraction

Source functions:

```text
code/deepseg.py: deeplearningseg
code/image_utils.py: imagePreprocessing
```

Input:

```text
/data/<subject>/*.nii
or
/data/<subject>/*.nii.gz
```

Main operations:

1. Reset image header with `headertool`.
2. Save the preprocessed full cine sequence as `lvsa_.nii.gz`.
3. Temporally align the cine sequence with `/refs/temporal.gipl`.
4. Apply `autocontrast`.
5. Detect ED and ES frames with `cardiacphasedetection`.

Outputs inside each subject folder:

| Output | Meaning |
|---|---|
| `/data/<subject>/lvsa_.nii.gz` | Preprocessed full 4D cine sequence. Used later by motion tracking. |
| `/data/<subject>/lvsa_ED.nii.gz` | ED image frame extracted from the cine sequence. Later moved to `sizes/`. |
| `/data/<subject>/lvsa_ES.nii.gz` | ES image frame extracted from the cine sequence. Later moved to `sizes/`. |

Outputs under `/data`:

| Output | Meaning |
|---|---|
| `/data/subjnames.txt` | List of subject folders processed by `deeplearningseg`. Existing top-level `.txt` files in `/data` are removed before this file is recreated. |

Folders created or recreated per subject:

```text
/data/<subject>/dofs/
/data/<subject>/segs/
/data/<subject>/tmps/
/data/<subject>/sizes/
/data/<subject>/motion/
/data/<subject>/vtks/
```

If any of these folders already exist, the code removes and recreates all of them. That means old intermediate segmentation, registration, mesh, and motion outputs are deleted at the start of a new run.

## Stage 2: CNN Segmentation Of ED And ES

Source function:

```text
code/deepseg.py: deeplearningseg
```

Input:

```text
/data/<subject>/lvsa_ED.nii.gz
/data/<subject>/lvsa_ES.nii.gz
/model/vgg_RE_network.ckpt-50000
```

Main operations:

1. Load the TensorFlow graph from `vgg_RE_network.ckpt-50000.meta`.
2. Restore checkpoint weights.
3. Process ED and ES slice-by-slice.
4. Rescale intensities with the 1st and 99th percentiles.
5. Pad each 2D slice to a multiple of 16.
6. Run the network tensors `probE:0` and `predR:0`.
7. Crop predictions back to the original slice size.
8. Reassemble the 3D prediction volume.

Outputs:

| Output | Meaning |
|---|---|
| `/data/<subject>/segs/seg_lvsa_ED.nii.gz` | Raw CNN ED segmentation. This is before 3D atlas shape refinement. |
| `/data/<subject>/segs/seg_lvsa_ES.nii.gz` | Raw CNN ES segmentation. This is before 3D atlas shape refinement. |

These files are useful for debugging the deep-learning model. They are not the final segmentation files used for final mesh extraction.

## Stage 3: 3D Atlas Shape Refinement

Source functions:

```text
code/p2processing.py: multiatlasreg3D
code/p2processing.py: output3DRefinement
code/image_utils.py: allAtlasShapeSelection
code/image_utils.py: topSimilarAtlasShapeSelection
code/image_utils.py: formHighResolutionImg
code/image_utils.py: refineFusionResults
code/image_utils.py: convertImageSegment
code/image_utils.py: outputVolumes
code/image_utils.py: moveVolumes
```

Input:

```text
/data/<subject>/segs/seg_lvsa_ED.nii.gz
/data/<subject>/segs/seg_lvsa_ES.nii.gz
/refs/<atlas>/PHsegmentation_ED.nii.gz
/refs/<atlas>/PHsegmentation_ES.nii.gz
/par/segareg.txt
/par/ffd_label_1.cfg
```

Main operations for each frame, `ED` and `ES`:

1. Load atlas segmentation candidates from `/refs`.
2. Register each atlas label image to the raw CNN segmentation.
3. Evaluate atlas similarity with `cardiacimageevaluation`.
4. Select the top 3 most similar atlases.
5. Resample the ED/ES image to high resolution.
6. Nonrigidly propagate the top 3 atlas segmentations to the subject.
7. Fuse propagated labels with `combineLabels`.
8. Smooth and clean labels with `refineFusionResults`.
9. Convert the final refined segmentation to `.gipl`.
10. Move high-resolution image and segmentation archives into `sizes/`.

Intermediate registration outputs:

| Output | Meaning |
|---|---|
| `/data/<subject>/dofs/shapelandmarks_<atlas_index>.dof.gz` | Initial atlas-to-subject transform for atlas selection. Despite the name, the active code uses segmentation-label registration because `landmarks = False`. |
| `/data/<subject>/tmps/shapenmi_<atlas_index>.txt` | Similarity scores from `cardiacimageevaluation`, used for top-atlas selection. |
| `/data/<subject>/dofs/shapeffd_0_ED.dof.gz` | Nonrigid transform for selected ED atlas 0. |
| `/data/<subject>/dofs/shapeffd_1_ED.dof.gz` | Nonrigid transform for selected ED atlas 1. |
| `/data/<subject>/dofs/shapeffd_2_ED.dof.gz` | Nonrigid transform for selected ED atlas 2. |
| `/data/<subject>/dofs/shapeffd_0_ES.dof.gz` | Nonrigid transform for selected ES atlas 0. |
| `/data/<subject>/dofs/shapeffd_1_ES.dof.gz` | Nonrigid transform for selected ES atlas 1. |
| `/data/<subject>/dofs/shapeffd_2_ES.dof.gz` | Nonrigid transform for selected ES atlas 2. |

Intermediate segmentation outputs:

| Output | Meaning |
|---|---|
| `/data/<subject>/lvsa_SR_ED.nii.gz` | High-resolution ED image before being moved to `sizes/lvsa_ED_SR.nii.gz`. |
| `/data/<subject>/lvsa_SR_ES.nii.gz` | High-resolution ES image before being moved to `sizes/lvsa_ES_SR.nii.gz`. |
| `/data/<subject>/tmps/seg_lvsa_SR_0_ED.nii.gz` | Selected ED atlas 0 warped into subject high-resolution ED space. |
| `/data/<subject>/tmps/seg_lvsa_SR_1_ED.nii.gz` | Selected ED atlas 1 warped into subject high-resolution ED space. |
| `/data/<subject>/tmps/seg_lvsa_SR_2_ED.nii.gz` | Selected ED atlas 2 warped into subject high-resolution ED space. |
| `/data/<subject>/tmps/seg_lvsa_SR_0_ES.nii.gz` | Selected ES atlas 0 warped into subject high-resolution ES space. |
| `/data/<subject>/tmps/seg_lvsa_SR_1_ES.nii.gz` | Selected ES atlas 1 warped into subject high-resolution ES space. |
| `/data/<subject>/tmps/seg_lvsa_SR_2_ES.nii.gz` | Selected ES atlas 2 warped into subject high-resolution ES space. |
| `/data/<subject>/seg_lvsa_SR_ED.nii.gz` | Fused and refined high-resolution ED segmentation before conversion and archiving. |
| `/data/<subject>/seg_lvsa_SR_ES.nii.gz` | Fused and refined high-resolution ES segmentation before conversion and archiving. |

Final segmentation outputs:

| Output | Meaning |
|---|---|
| `/data/<subject>/PHsegmentation_ED.gipl` | Final high-resolution, atlas-refined ED segmentation. This is the main ED segmentation used for mesh generation. |
| `/data/<subject>/PHsegmentation_ES.gipl` | Final high-resolution, atlas-refined ES segmentation. This is the main ES segmentation used for mesh generation. |
| `/data/<subject>/lvsa_ED_enlarged_SR.nii.gz` | Final high-resolution ED image companion for `PHsegmentation_ED.gipl`. |
| `/data/<subject>/lvsa_ES_enlarged_SR.nii.gz` | Final high-resolution ES image companion for `PHsegmentation_ES.gipl`. |

Archived image and segmentation outputs in `sizes/`:

| Output | Meaning |
|---|---|
| `/data/<subject>/sizes/lvsa_ED.nii.gz` | Original-resolution ED frame archive. |
| `/data/<subject>/sizes/lvsa_ES.nii.gz` | Original-resolution ES frame archive. |
| `/data/<subject>/sizes/lvsa_ED_SR.nii.gz` | High-resolution ED frame archive. |
| `/data/<subject>/sizes/lvsa_ES_SR.nii.gz` | High-resolution ES frame archive. |
| `/data/<subject>/sizes/3D_segmentation_ED.nii.gz` | Final refined ED segmentation archive before `.gipl` conversion. |
| `/data/<subject>/sizes/3D_segmentation_ES.nii.gz` | Final refined ES segmentation archive before `.gipl` conversion. |

Top-level volume summaries:

| Output | Meaning |
|---|---|
| `/data/<subject>_ED.txt` | ED volume summary calculated from `PHsegmentation_ED.gipl`. |
| `/data/<subject>_ES.txt` | ES volume summary calculated from `PHsegmentation_ES.gipl`. |

The volume summary is written label-by-label using `cardiacvolumecount`: label 1, label 2 with scale 1.05, label 4, and label 3 with scale 1.05.

## Stage 4: Static Mesh Extraction And Template Fitting

Source functions:

```text
code/meshfitting.py: meshCoregstration
code/meshfitting.py: meshGeneration
```

Input:

```text
/data/<subject>/PHsegmentation_ED.gipl
/data/<subject>/PHsegmentation_ES.gipl
/vtks/1/RV_ED.vtk
/vtks/1/RV_ES.vtk
/vtks/1/LVendo_ED.vtk
/vtks/1/LVendo_ES.vtk
/vtks/1/LVepi_ED.vtk
/vtks/1/LVepi_ES.vtk
/vtks/1/LVmyo_ED.vtk
/vtks/1/LVmyo_ES.vtk
/vtks/1/vtk_RV_ED.nii.gz
/vtks/1/vtk_RV_ES.nii.gz
/vtks/1/vtk_LV_ED.nii.gz
/vtks/1/vtk_LV_ES.nii.gz
```

At the start of this stage, the code removes existing subject root `*.txt`, clears `vtks/`, and clears `tmps/`.

### 4.1 Binary Mask Extraction From Final Segmentation

For each frame, `ED` and `ES`, the final `PHsegmentation_<frame>.gipl` is split into masks:

| Output | Segmentation labels used | Meaning |
|---|---:|---|
| `/data/<subject>/tmps/vtk_RV_<frame>.nii.gz` | 4 | RV cavity/endocardium mask. |
| `/data/<subject>/tmps/vtk_RVepi_<frame>.nii.gz` | 3-4 | RV wall plus cavity outer-region mask. |
| `/data/<subject>/tmps/vtk_LV_<frame>.nii.gz` | 1-2 after removing labels 3 and 4 | Combined LV mask used for registration. |
| `/data/<subject>/tmps/vtk_LVendo_<frame>.nii.gz` | 1 | LV cavity/endocardium mask. |
| `/data/<subject>/tmps/vtk_LVepi_<frame>.nii.gz` | 1-2 | LV cavity plus myocardium outer-region mask. |
| `/data/<subject>/tmps/vtk_LVmyo_<frame>.nii.gz` | 2 | LV myocardium mask. |

### 4.2 Raw Marching-Cubes Surface Meshes

The binary masks are converted to VTK surfaces with `mcubes`:

| Output | Meaning |
|---|---|
| `/data/<subject>/vtks/RV_ED.vtk` | Raw RV endocardial/cavity surface at ED. |
| `/data/<subject>/vtks/RV_ES.vtk` | Raw RV endocardial/cavity surface at ES. |
| `/data/<subject>/vtks/RVepi_ED.vtk` | Raw RV outer/wall surface at ED. |
| `/data/<subject>/vtks/RVepi_ES.vtk` | Raw RV outer/wall surface at ES. |
| `/data/<subject>/vtks/LVendo_ED.vtk` | Raw LV endocardial surface at ED. |
| `/data/<subject>/vtks/LVendo_ES.vtk` | Raw LV endocardial surface at ES. |
| `/data/<subject>/vtks/LVepi_ED.vtk` | Raw LV epicardial surface at ED. |
| `/data/<subject>/vtks/LVepi_ES.vtk` | Raw LV epicardial surface at ES. |
| `/data/<subject>/vtks/LVmyo_ED.vtk` | Raw LV myocardium surface at ED. |
| `/data/<subject>/vtks/LVmyo_ES.vtk` | Raw LV myocardium surface at ES. |

These raw meshes follow the subject segmentation geometry, but they do not yet have the template vertex correspondence needed for consistent feature analysis across subjects.

### 4.3 Mesh Registration And Normalized Subject Meshes

The code performs surface and image registration against the template:

| Output | Meaning |
|---|---|
| `/data/<subject>/tmps/ED.dof.gz` | Initial ED multi-surface registration transform. |
| `/data/<subject>/tmps/ES.dof.gz` | Initial ES multi-surface registration transform. |
| `/data/<subject>/tmps/lv_ED_rreg.dof.gz` | LV ED rigid/surface registration transform. |
| `/data/<subject>/tmps/lv_ES_rreg.dof.gz` | LV ES rigid/surface registration transform. |
| `/data/<subject>/tmps/rv_ED_rreg.dof.gz` | RV ED rigid/surface registration transform. |
| `/data/<subject>/tmps/rv_ES_rreg.dof.gz` | RV ES rigid/surface registration transform. |
| `/data/<subject>/vtks/N_RV_ED.vtk` | Registered/normalized RV surface at ED. |
| `/data/<subject>/vtks/N_RV_ES.vtk` | Registered/normalized RV surface at ES. |
| `/data/<subject>/vtks/N_RVepi_ED.vtk` | Registered/normalized RV outer/wall surface at ED. |
| `/data/<subject>/vtks/N_RVepi_ES.vtk` | Registered/normalized RV outer/wall surface at ES. |
| `/data/<subject>/vtks/N_LVendo_ED.vtk` | Registered/normalized LV endocardial surface at ED. |
| `/data/<subject>/vtks/N_LVendo_ES.vtk` | Registered/normalized LV endocardial surface at ES. |
| `/data/<subject>/vtks/N_LVepi_ED.vtk` | Registered/normalized LV epicardial surface at ED. |
| `/data/<subject>/vtks/N_LVepi_ES.vtk` | Registered/normalized LV epicardial surface at ES. |
| `/data/<subject>/vtks/N_LVmyo_ED.vtk` | Registered/normalized LV myocardium surface at ED. |
| `/data/<subject>/vtks/N_LVmyo_ES.vtk` | Registered/normalized LV myocardium surface at ES. |
| `/data/<subject>/tmps/N_vtk_RV_ED.nii.gz` | Registered/normalized RV binary mask at ED. |
| `/data/<subject>/tmps/N_vtk_RV_ES.nii.gz` | Registered/normalized RV binary mask at ES. |
| `/data/<subject>/tmps/N_vtk_LV_ED.nii.gz` | Registered/normalized LV binary mask at ED. |
| `/data/<subject>/tmps/N_vtk_LV_ES.nii.gz` | Registered/normalized LV binary mask at ES. |

Additional affine and nonrigid fitting transforms:

| Output | Meaning |
|---|---|
| `/data/<subject>/tmps/rv_ED_areg.dof.gz` | RV ED affine registration transform. |
| `/data/<subject>/tmps/rv_ES_areg.dof.gz` | RV ES affine registration transform. |
| `/data/<subject>/tmps/lv_ED_areg.dof.gz` | LV ED affine registration transform. |
| `/data/<subject>/tmps/lv_ES_areg.dof.gz` | LV ES affine registration transform. |
| `/data/<subject>/tmps/rv_ED_nreg.dof.gz` | RV ED nonrigid image registration transform. |
| `/data/<subject>/tmps/rv_ES_nreg.dof.gz` | RV ES nonrigid image registration transform. |
| `/data/<subject>/tmps/lv_ED_nreg.dof.gz` | LV ED nonrigid image registration transform. |
| `/data/<subject>/tmps/lv_ES_nreg.dof.gz` | LV ES nonrigid image registration transform. |
| `/data/<subject>/tmps/rvEDds8.dof.gz` | Final RV ED surface-fitting transform. |
| `/data/<subject>/tmps/rvESds8.dof.gz` | Final RV ES surface-fitting transform. |
| `/data/<subject>/tmps/lvEDfinal.dof.gz` | Final LV ED surface-fitting transform. |
| `/data/<subject>/tmps/lvESfinal.dof.gz` | Final LV ES surface-fitting transform. |

These transforms are mainly useful for debugging registration failures.

### 4.4 Fitted Template-Correspondence Meshes

The final fitting step maps template surfaces into the subject space while preserving template point correspondence.

LV fitted meshes:

| Output | Meaning |
|---|---|
| `/data/<subject>/vtks/F_LVendo_ED.vtk` | Fitted LV endocardial mesh at ED with template correspondence. |
| `/data/<subject>/vtks/F_LVendo_ES.vtk` | Fitted LV endocardial mesh at ES with template correspondence. |
| `/data/<subject>/vtks/F_LVepi_ED.vtk` | Fitted LV epicardial mesh at ED with template correspondence. |
| `/data/<subject>/vtks/F_LVepi_ES.vtk` | Fitted LV epicardial mesh at ES with template correspondence. |
| `/data/<subject>/vtks/F_LVmyo_ED.vtk` | Fitted LV myocardium mesh at ED with template correspondence. |
| `/data/<subject>/vtks/F_LVmyo_ES.vtk` | Fitted LV myocardium mesh at ES with template correspondence. |

RV fitted meshes:

| Output | Meaning |
|---|---|
| `/data/<subject>/vtks/C_RV_ED.vtk` | RV fitted correspondence mesh at ED before/after curvature scalar calculation. |
| `/data/<subject>/vtks/C_RV_ES.vtk` | RV fitted correspondence mesh at ES before/after curvature scalar calculation. |

For RV, the code does not create `F_RV_ED.vtk`. Instead, `cardiacsurfacemap` writes the fitted RV surface to `C_RV_<frame>.vtk`, then copies it to `S_RV_<frame>.vtk` and `W_RV_<frame>.vtk` before scalar calculations.

### 4.5 Scalar Mesh Outputs

After fitting, the code copies fitted meshes into scalar-specific files and computes wall thickness, signed distance, and curvature.

Main signed-distance meshes:

| Output | Meaning |
|---|---|
| `/data/<subject>/vtks/S_LVmyo_ED.vtk` | LV myocardium fitted mesh at ED with signed-distance scalar relative to template. |
| `/data/<subject>/vtks/S_LVmyo_ES.vtk` | LV myocardium fitted mesh at ES with signed-distance scalar relative to template. |
| `/data/<subject>/vtks/S_RV_ED.vtk` | RV fitted mesh at ED with signed-distance scalar relative to template. |
| `/data/<subject>/vtks/S_RV_ES.vtk` | RV fitted mesh at ES with signed-distance scalar relative to template. |

Wall-thickness meshes:

| Output | Meaning |
|---|---|
| `/data/<subject>/vtks/W_LVmyo_ED.vtk` | LV myocardium fitted mesh at ED with wall-thickness scalar. |
| `/data/<subject>/vtks/W_LVmyo_ES.vtk` | LV myocardium fitted mesh at ES with wall-thickness scalar. |
| `/data/<subject>/vtks/W_RV_ED.vtk` | RV fitted mesh at ED with wall-thickness scalar. |
| `/data/<subject>/vtks/W_RV_ES.vtk` | RV fitted mesh at ES with wall-thickness scalar. |

Curvature meshes:

| Output | Meaning |
|---|---|
| `/data/<subject>/vtks/C_LVmyo_ED.vtk` | LV myocardium fitted mesh at ED with curvature scalar. |
| `/data/<subject>/vtks/C_LVmyo_ES.vtk` | LV myocardium fitted mesh at ES with curvature scalar. |
| `/data/<subject>/vtks/C_RV_ED.vtk` | RV fitted mesh at ED with curvature scalar. |
| `/data/<subject>/vtks/C_RV_ES.vtk` | RV fitted mesh at ES with curvature scalar. |
| `/data/<subject>/vtks/FC_LVmyo_ED.vtk` | Temporary curvature-estimator output for LV ED. Usually ignore. |
| `/data/<subject>/vtks/FC_LVmyo_ES.vtk` | Temporary curvature-estimator output for LV ES. Usually ignore. |
| `/data/<subject>/vtks/FC_RV_ED.vtk` | Temporary curvature-estimator output for RV ED. Usually ignore. |
| `/data/<subject>/vtks/FC_RV_ES.vtk` | Temporary curvature-estimator output for RV ES. Usually ignore. |

Scalar text exports in the subject root:

| Output | Meaning |
|---|---|
| `/data/<subject>/lv_myoed_curvature.txt` | Per-vertex LV myocardium curvature at ED. |
| `/data/<subject>/lv_myoes_curvature.txt` | Per-vertex LV myocardium curvature at ES. |
| `/data/<subject>/lv_myoed_wallthickness.txt` | Per-vertex LV myocardium wall thickness at ED. |
| `/data/<subject>/lv_myoes_wallthickness.txt` | Per-vertex LV myocardium wall thickness at ES. |
| `/data/<subject>/lv_myoed_signeddistances.txt` | Per-vertex LV myocardium signed distance at ED. |
| `/data/<subject>/lv_myoes_signeddistances.txt` | Per-vertex LV myocardium signed distance at ES. |
| `/data/<subject>/rv_ed_curvature.txt` | Per-vertex RV curvature at ED. |
| `/data/<subject>/rv_es_curvature.txt` | Per-vertex RV curvature at ES. |
| `/data/<subject>/rv_ed_wallthickness.txt` | Per-vertex RV wall thickness at ED. |
| `/data/<subject>/rv_es_wallthickness.txt` | Per-vertex RV wall thickness at ES. |
| `/data/<subject>/rv_ed_signeddistances.txt` | Per-vertex RV signed distance at ED. |
| `/data/<subject>/rv_es_signeddistances.txt` | Per-vertex RV signed distance at ES. |

## Stage 5: RV Motion Tracking Meshes

Source function:

```text
code/motionEstimation.py: motionTracking
```

Input:

```text
/data/<subject>/lvsa_.nii.gz
/data/<subject>/vtks/RV_ED.vtk
/vtks/2/RVendo_ED.vtk
/par/ffd_motion.cfg
/par/ffd_refine.cfg
```

Main operations:

1. Split the full cine sequence into frame images.
2. Register consecutive frames.
3. Compose transforms from frame 00 to later frames.
4. Refine each frame-to-ED transform.
5. Fit the RV ED mesh to the RV template.
6. Propagate the RV mesh through all cine frames.
7. Export each frame mesh as both VTK and text.

Outputs in `motion/`:

| Output | Meaning |
|---|---|
| `/data/<subject>/motion/lvsa_00.nii.gz` to `/data/<subject>/motion/lvsa_XX.nii.gz` | Cine frames split from `lvsa_.nii.gz`. Many sample subjects have 20 frames, `00` to `19`. |
| `/data/<subject>/motion/ffd_00_to_01.dof.gz`, etc. | Frame-to-frame registration transforms. |
| `/data/<subject>/motion/ffd_comp_00_to_02.dof.gz`, etc. | Composed transforms from frame 00 to later frames. |
| `/data/<subject>/motion/ffd_00_to_02.dof.gz`, etc. | Refined transforms from ED/frame 00 to later frames. |
| `/data/<subject>/motion/RV_srreg.dof.gz` | RV template-to-subject surface registration helper. |
| `/data/<subject>/motion/RV_sareg.dof.gz` | RV affine surface registration helper. |
| `/data/<subject>/motion/RV_snreg.dof.gz` | RV nonrigid surface registration helper. |
| `/data/<subject>/motion/RV_ED_srreg.vtk` | Template RV mesh registered toward subject ED before final propagation. |
| `/data/<subject>/motion/RV_fr00.vtk` to `/data/<subject>/motion/RV_frXX.vtk` | RV mesh propagated through the cine cardiac cycle. |
| `/data/<subject>/motion/RV_fr00.txt` to `/data/<subject>/motion/RV_frXX.txt` | Text export of the corresponding per-frame RV mesh. |

This stage is RV-only. It does not propagate LV meshes through the full cardiac cycle.

## Stage 6: Decimation

Source function:

```text
code/decimation.py: decimate
```

Input:

```text
/data/<subject>/motion/RV_fr00.vtk
```

Main operations:

1. Decimate the RV frame-00 mesh with `mirtk decimate-surface`.
2. Register the decimated mesh back to the original RV frame-00 mesh.
3. Match decimated points to the original point set.

Outputs:

| Output | Meaning |
|---|---|
| `/data/<subject>/motion/RV_fr00_new.vtk` | Decimated RV frame-00 mesh. |
| `/data/<subject>/motion/RV_fr00_new_new.vtk` | Decimated mesh after transform back to the original frame-00 mesh. |
| `/data/<subject>/motion/transformation.vtk` | Transform/correspondence helper for decimation. |
| `/data/matchedpointsnew.txt` | Matched-point index output in the default non-parallel `DMACS.py` run. |

Caveat: `DMACS.py` calls `decimate(..., parallel=False)`. In that branch, `decimation.py` only decimates the first valid subject because of the `i == 0` guard. The parallel branch would process each subject separately, but that is not the default call.

## Which Outputs To Use

For final segmentation:

```text
/data/<subject>/PHsegmentation_ED.gipl
/data/<subject>/PHsegmentation_ES.gipl
```

Use these with their high-resolution image companions:

```text
/data/<subject>/lvsa_ED_enlarged_SR.nii.gz
/data/<subject>/lvsa_ES_enlarged_SR.nii.gz
```

For raw CNN segmentation debugging:

```text
/data/<subject>/segs/seg_lvsa_ED.nii.gz
/data/<subject>/segs/seg_lvsa_ES.nii.gz
```

For raw ED/ES segmentation-derived surfaces:

```text
/data/<subject>/vtks/RV_ED.vtk
/data/<subject>/vtks/RV_ES.vtk
/data/<subject>/vtks/RVepi_ED.vtk
/data/<subject>/vtks/RVepi_ES.vtk
/data/<subject>/vtks/LVendo_ED.vtk
/data/<subject>/vtks/LVendo_ES.vtk
/data/<subject>/vtks/LVepi_ED.vtk
/data/<subject>/vtks/LVepi_ES.vtk
/data/<subject>/vtks/LVmyo_ED.vtk
/data/<subject>/vtks/LVmyo_ES.vtk
```

For final template-correspondence static meshes:

```text
/data/<subject>/vtks/F_LVendo_ED.vtk
/data/<subject>/vtks/F_LVendo_ES.vtk
/data/<subject>/vtks/F_LVepi_ED.vtk
/data/<subject>/vtks/F_LVepi_ES.vtk
/data/<subject>/vtks/F_LVmyo_ED.vtk
/data/<subject>/vtks/F_LVmyo_ES.vtk
/data/<subject>/vtks/S_RV_ED.vtk
/data/<subject>/vtks/S_RV_ES.vtk
```

For signed-distance shape analysis:

```text
/data/<subject>/vtks/S_LVmyo_ED.vtk
/data/<subject>/vtks/S_LVmyo_ES.vtk
/data/<subject>/vtks/S_RV_ED.vtk
/data/<subject>/vtks/S_RV_ES.vtk
```

For wall-thickness analysis:

```text
/data/<subject>/vtks/W_LVmyo_ED.vtk
/data/<subject>/vtks/W_LVmyo_ES.vtk
/data/<subject>/vtks/W_RV_ED.vtk
/data/<subject>/vtks/W_RV_ES.vtk
```

For curvature analysis:

```text
/data/<subject>/vtks/C_LVmyo_ED.vtk
/data/<subject>/vtks/C_LVmyo_ES.vtk
/data/<subject>/vtks/C_RV_ED.vtk
/data/<subject>/vtks/C_RV_ES.vtk
```

For RV motion through the full cine sequence:

```text
/data/<subject>/motion/RV_fr00.vtk
/data/<subject>/motion/RV_fr01.vtk
...
/data/<subject>/motion/RV_frXX.vtk
```

## Important Caveats

- The default pipeline does not calculate Dice, Hausdorff distance, or HD95. It produces segmentations, meshes, motion meshes, volume summaries, and per-vertex scalar feature exports.
- The final segmentation used for mesh generation is `PHsegmentation_ED.gipl` and `PHsegmentation_ES.gipl`, not the raw CNN outputs in `segs/`.
- The original pipeline outputs surface meshes in `.vtk` format. The `volmesh/` files under local sample data are from separate later scripts in `code_explain/`, not from the default `DMACS.py` pipeline.
- The RV output naming is easy to misread: `S_RV_*`, `W_RV_*`, and `C_RV_*` refer to the fitted RV surface; `RVepi_*` and `N_RVepi_*` are the RV outer/wall surface context.
- Most files in `dofs/` and `tmps/` are registration intermediates. Keep them for debugging, but use `PHsegmentation_*`, `vtks/`, root scalar `.txt` files, and `motion/RV_fr*` for analysis.
