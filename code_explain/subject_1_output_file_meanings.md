# Subject 1 Output File Meanings

This note maps the two papers in `` to the code and to the files generated for `../data/subject_1`.

The short version:

- For final biventricular segmentation, use `PHsegmentation_ED.gipl` and `PHsegmentation_ES.gipl`.
- For final static LV myocardium mesh, use `vtks/S_LVmyo_ED.vtk` and `vtks/S_LVmyo_ES.vtk`.
- For final static RV surface mesh, use `vtks/S_RV_ED.vtk` and `vtks/S_RV_ES.vtk`.
- For RV wall context, also inspect `vtks/N_RVepi_ED.vtk`, `vtks/N_RVepi_ES.vtk`, `vtks/W_RV_ED.vtk`, and `vtks/W_RV_ES.vtk`.
- Ignore most `tmps`, `dofs/`, and `motion/` files unless debugging registration or RV motion.

## Paper To Code Mapping

The segmentation paper describes a pipeline with:

1. deep-learning segmentation,
2. anatomical/shape refinement by atlas propagation,
3. label fusion into smooth high-resolution 3D biventricular segmentation.

In this repository, that corresponds to:

- `../code/DMACS.py`: main runner.
- `../code/deepseg.py`: CNN segmentation of ED and ES frames.
- `../code/p2processing.py`: 3D atlas shape refinement.
- `../code/meshfitting.py`: mesh extraction, template fitting, scalar calculation.

The motion/survival paper describes:

1. automated biventricular segmentation,
2. registration-based RV motion tracking through the cine sequence,
3. per-frame RV meshes with point correspondence,
4. later use of these RV motion meshes for survival prediction.

In this repository, that corresponds mainly to:

- `../code/motionEstimation.py`: RV motion tracking across frames.
- `../code/decimation.py`: lower-resolution RV motion mesh generation.

The actual execution order is in `../code/DMACS.py`:

```text
deeplearningseg
multiatlasreg3D
meshCoregstration
motionTracking
decimate
```

## Segmentation Label Meaning

The mesh code reveals the final segmentation labels:

| Label | Meaning | Evidence in code |
|---:|---|---|
| 0 | background | implicit |
| 1 | LV cavity / LV endocardium | `binarize ... 1 1` creates `LVendo` |
| 2 | LV myocardium | `binarize ... 2 2` creates `LVmyo` |
| 3 | RV myocardium / RV wall | `RVepi` uses labels `3 4` |
| 4 | RV cavity / RV endocardium | `binarize ... 4 4` creates `RV` |

Important distinction: the code does not output a file named `S_RVmyo`. It outputs RV endocardial/cavity surface as `S_RV_*`, and RV wall/epicardial context through `RVepi_*`, `N_RVepi_*`, and `W_RV_*`.

## Top-Level `../data` Files

These files are outside `subject_1`, but generated for the subject:

| File | Meaning |
|---|---|
| `../data/subjnames.txt` | List of processed subject folder names. Created by `deepseg.py`. |
| `../data/subject_1_ED.txt` | ED volume summary from final `PHsegmentation_ED.gipl`. Values are written in label order: LV cavity, LV myocardium, RV cavity, RV myocardium. Myocardium values are scaled by 1.05 in the code. |
| `../data/subject_1_ES.txt` | Same as above for ES. |

For current `subject_1`, these are:

```text
subject_1_ED.txt: 79371.9 106739 252175 51436.9
subject_1_ES.txt: 31284.4 107838 202766 58632.7
```

## `../data/subject_1` Root Files

| File pattern | Stage | Meaning |
|---|---|---|
| `example.nii` | input | Original 4D cine short-axis CMR input. |
| `lvsa_.nii.gz` | preprocessing | Preprocessed full cine sequence after header reset, temporal alignment, and contrast normalization. Used later for motion tracking. |
| `PHsegmentation_ED.gipl` | final segmentation | Final high-resolution, atlas-refined ED segmentation. Main ED segmentation file to trust. |
| `PHsegmentation_ES.gipl` | final segmentation | Final high-resolution, atlas-refined ES segmentation. Main ES segmentation file to trust. |
| `lvsa_ED_enlarged_SR.nii.gz` | final image companion | High-resolution ED image aligned with `PHsegmentation_ED.gipl`. |
| `lvsa_ES_enlarged_SR.nii.gz` | final image companion | High-resolution ES image aligned with `PHsegmentation_ES.gipl`. |
| `lv_myoed_curvature.txt` | mesh scalar export | Per-vertex LV myocardium curvature at ED, exported from `vtks/C_LVmyo_ED.vtk`. |
| `lv_myoes_curvature.txt` | mesh scalar export | Per-vertex LV myocardium curvature at ES. |
| `lv_myoed_wallthickness.txt` | mesh scalar export | Per-vertex LV myocardium wall thickness at ED, from `vtks/W_LVmyo_ED.vtk`. |
| `lv_myoes_wallthickness.txt` | mesh scalar export | Per-vertex LV myocardium wall thickness at ES. |
| `lv_myoed_signeddistances.txt` | mesh scalar export | Per-vertex LV myocardium signed distance relative to template at ED, from `vtks/S_LVmyo_ED.vtk`. |
| `lv_myoes_signeddistances.txt` | mesh scalar export | Per-vertex LV myocardium signed distance relative to template at ES. |
| `rv_ed_curvature.txt` | mesh scalar export | Per-vertex RV curvature at ED, from `vtks/C_RV_ED.vtk`. |
| `rv_es_curvature.txt` | mesh scalar export | Per-vertex RV curvature at ES. |
| `rv_ed_wallthickness.txt` | mesh scalar export | Per-vertex RV wall thickness at ED, from `vtks/W_RV_ED.vtk`. |
| `rv_es_wallthickness.txt` | mesh scalar export | Per-vertex RV wall thickness at ES. |
| `rv_ed_signeddistances.txt` | mesh scalar export | Per-vertex RV signed distance relative to template at ED, from `vtks/S_RV_ED.vtk`. |
| `rv_es_signeddistances.txt` | mesh scalar export | Per-vertex RV signed distance relative to template at ES. |

## `../data/subject_1/segs`

| File | Stage | Meaning |
|---|---|---|
| `seg_lvsa_ED.nii.gz` | CNN segmentation | Raw 2D CNN segmentation for ED. It is before 3D atlas/shape refinement. |
| `seg_lvsa_ES.nii.gz` | CNN segmentation | Raw 2D CNN segmentation for ES. It is before 3D atlas/shape refinement. |

These are useful for debugging the network, but they are not the final segmentation.

## `../data/subject_1/sizes`

| File | Stage | Meaning |
|---|---|---|
| `lvsa_ED.nii.gz` | preprocessing archive | ED image extracted by cardiac phase detection. |
| `lvsa_ES.nii.gz` | preprocessing archive | ES image extracted by cardiac phase detection. |
| `lvsa_ED_SR.nii.gz` | shape refinement | ED image resampled to high resolution, used by atlas refinement. |
| `lvsa_ES_SR.nii.gz` | shape refinement | ES image resampled to high resolution, used by atlas refinement. |
| `3D_segmentation_ED.nii.gz` | shape refinement archive | Final fused/refined ED segmentation before conversion to `.gipl`. |
| `3D_segmentation_ES.nii.gz` | shape refinement archive | Final fused/refined ES segmentation before conversion to `.gipl`. |

The code attempts to archive `2D_segmentation_ED/ES.nii.gz`, but in this run the raw CNN files remained in `segs/` instead.

## `../data/subject_1/dofs`

These are deformation/registration transform files. They are not meshes or segmentations.

| File pattern | Meaning |
|---|---|
| `shapelandmarks_*.dof.gz` | Atlas-to-subject affine transforms used during atlas selection. The name says `landmarks`, but in this code `landmarks = False`, so these are generated from segmentation-label registration, not saved CNN landmark outputs. There is one per atlas candidate. |
| `shapeffd_0_ED.dof.gz`, `shapeffd_1_ED.dof.gz`, `shapeffd_2_ED.dof.gz` | Nonrigid transforms for the top 3 selected ED atlases. |
| `shapeffd_0_ES.dof.gz`, `shapeffd_1_ES.dof.gz`, `shapeffd_2_ES.dof.gz` | Nonrigid transforms for the top 3 selected ES atlases. |

You can usually ignore `dofs/` unless checking why atlas refinement failed.

## `../data/subject_1/tmps`

Temporary registration and mesh-extraction workspace. Most files here are intermediates.

| File pattern | Meaning |
|---|---|
| `vtk_RV_ED.nii.gz`, `vtk_RV_ES.nii.gz` | Binary masks from final segmentation for RV cavity, label `4`, used by marching cubes. |
| `vtk_RVepi_ED.nii.gz`, `vtk_RVepi_ES.nii.gz` | Binary masks for RV cavity plus wall, labels `3-4`, used to estimate RV wall thickness. |
| `vtk_LVendo_ED/ES.nii.gz` | Binary masks for LV cavity, label `1`. |
| `vtk_LVepi_ED/ES.nii.gz` | Binary masks for LV cavity plus LV myocardium, labels `1-2`. |
| `vtk_LVmyo_ED/ES.nii.gz` | Binary masks for LV myocardium, label `2`. |
| `N_vtk_RV_ED/ES.nii.gz`, `N_vtk_LV_ED/ES.nii.gz` | Registered/normalized versions of the binary masks used for template fitting. |
| `ED.dof.gz`, `ES.dof.gz` | Initial multi-surface registration transforms for ED/ES. |
| `lv_ED_rreg.dof.gz`, `rv_ED_rreg.dof.gz`, etc. | Rigid/surface registration transforms for LV/RV. |
| `lv_ED_areg.dof.gz`, `rv_ED_areg.dof.gz`, etc. | Affine registration transforms. |
| `lv_ED_nreg.dof.gz`, `rv_ED_nreg.dof.gz`, etc. | Nonrigid image registration transforms. |
| `lvEDfinal.dof.gz`, `lvESfinal.dof.gz` | Final LV mesh fitting transforms. |
| `rvEDds8.dof.gz`, `rvESds8.dof.gz` | Final RV surface fitting transforms. |

For normal analysis, ignore `tmps`.

## `../data/subject_1/vtks`

This is the main mesh folder. Suffixes:

- `ED`: end-diastole.
- `ES`: end-systole.

Anatomical names:

| Name | Meaning |
|---|---|
| `LVendo` | LV endocardial surface, from label `1`. |
| `LVepi` | LV epicardial surface, from labels `1-2`. |
| `LVmyo` | LV myocardium surface, from label `2`. |
| `RV` | RV endocardial/cavity surface, from label `4`. |
| `RVepi` | RV epicardial/wall outer surface, from labels `3-4`. |

Prefix meanings:

| Prefix | Meaning | Use? |
|---|---|---|
| no prefix, for example `LVmyo_ED.vtk` | Raw marching-cubes mesh extracted directly from the segmentation. | Useful for visual QA, but not final correspondence mesh. |
| `N_` | Normalized/registered raw subject mesh after initial surface registration. | Mostly intermediate. |
| `F_` | Fitted template mesh in subject space, with consistent vertex correspondence. LV files use this prefix. | Useful if you need correspondence without scalar emphasis. |
| `S_` | Same fitted/corresponding mesh with signed-distance scalar values added. | Main shape-analysis mesh. |
| `C_` | Same fitted/corresponding mesh with curvature scalar values added. | Use for curvature analysis. |
| `W_` | Same fitted/corresponding mesh with wall-thickness scalar values added. | Use for wall thickness analysis. |
| `FC_` | Temporary curvature-estimator output used to create `C_`. | Usually ignore. |

Important static biventricular meshes:

| File | Meaning |
|---|---|
| `S_LVmyo_ED.vtk` | Final LV myocardium mesh at ED with signed-distance scalar. |
| `S_LVmyo_ES.vtk` | Final LV myocardium mesh at ES with signed-distance scalar. |
| `S_RV_ED.vtk` | Final RV endocardial/cavity surface at ED with signed-distance scalar. |
| `S_RV_ES.vtk` | Final RV endocardial/cavity surface at ES with signed-distance scalar. |
| `W_LVmyo_ED.vtk`, `W_LVmyo_ES.vtk` | Final LV myocardium mesh carrying wall-thickness scalar. |
| `W_RV_ED.vtk`, `W_RV_ES.vtk` | Final RV surface carrying wall-thickness scalar. |
| `C_LVmyo_ED.vtk`, `C_LVmyo_ES.vtk` | Final LV myocardium mesh carrying curvature scalar. |
| `C_RV_ED.vtk`, `C_RV_ES.vtk` | Final RV surface carrying curvature scalar. |
| `N_RVepi_ED.vtk`, `N_RVepi_ES.vtk` | Registered RV outer/wall surface. Useful when inspecting RV wall geometry. |

If you only care about biventricular segmentation and static mesh, focus on:

```text
PHsegmentation_ED.gipl
PHsegmentation_ES.gipl
vtks/S_LVmyo_ED.vtk
vtks/S_LVmyo_ES.vtk
vtks/S_RV_ED.vtk
vtks/S_RV_ES.vtk
vtks/W_LVmyo_ED.vtk
vtks/W_LVmyo_ES.vtk
vtks/W_RV_ED.vtk
vtks/W_RV_ES.vtk
```

## `../data/subject_1/motion`

This folder corresponds to the motion/survival paper more than the segmentation paper. It is mostly RV motion tracking.

| File pattern | Meaning |
|---|---|
| `lvsa_00.nii.gz` to `lvsa_19.nii.gz` | The full 4D cine `lvsa_.nii.gz` split into 20 time frames. |
| `ffd_01_to_02.dof.gz`, etc. | Frame-to-frame free-form deformation field. |
| `ffd_comp_00_to_02.dof.gz`, etc. | Composed deformation from frame 00 to later frame. |
| `ffd_00_to_02.dof.gz`, etc. | Refined deformation from ED/frame 00 to later frame. |
| `RV_srreg.dof.gz` | Surface registration between subject RV ED and template RV. |
| `RV_sareg.dof.gz` | Affine RV surface registration. |
| `RV_snreg.dof.gz` | Nonrigid RV surface registration. |
| `RV_ED_srreg.vtk` | Template RV mesh registered toward subject ED before final motion propagation. |
| `RV_fr00.vtk` to `RV_fr19.vtk` | RV mesh propagated through the cine cardiac cycle. |
| `RV_fr00.txt` to `RV_fr19.txt` | Text export of the corresponding `RV_frXX.vtk` mesh. |
| `RV_fr00_new.vtk`, `RV_fr00_new_new.vtk` | Decimated/downsampled RV frame-00 meshes. |
| `transformation.vtk` | Point correspondence/transform helper from decimation. |

For your current focus on ED/ES biventricular segmentation and static meshes, `motion/` can mostly be ignored.

## Practical Interpretation For `subject_1`

1. Start by opening `PHsegmentation_ED.gipl` and `PHsegmentation_ES.gipl` with the matching images `lvsa_ED_enlarged_SR.nii.gz` and `lvsa_ES_enlarged_SR.nii.gz`.
2. For LV myocardium mesh, open `vtks/S_LVmyo_ED.vtk` and `vtks/S_LVmyo_ES.vtk`.
3. For RV mesh, open `vtks/S_RV_ED.vtk` and `vtks/S_RV_ES.vtk`. Add `vtks/N_RVepi_ED.vtk` or `vtks/N_RVepi_ES.vtk` if you need the RV outer wall.
4. Use `W_*` files when you care about wall thickness scalars.
5. Use `C_*` files when you care about curvature scalars.
6. Use root-level `*_signeddistances.txt`, `*_wallthickness.txt`, and `*_curvature.txt` only if you want numeric per-vertex features outside a VTK viewer.



• Plan before generation:

  1. Validate input surfaces
      - Check these files exist and are non-empty:
          - data/subject_1/vtks/F_LVendo_ED.vtk
          - data/subject_1/vtks/F_LVepi_ED.vtk
          - data/subject_1/vtks/S_RV_ED.vtk
          - data/subject_1/vtks/N_RVepi_ED.vtk

      - Read point/cell counts and bounding boxes.

  2. Prepare closed myocardium shells
      - LV myocardium shell:
          - outer surface = F_LVepi_ED.vtk
          - inner cavity boundary = F_LVendo_ED.vtk

      - RV myocardium shell:
          - outer surface = N_RVepi_ED.vtk
          - inner cavity boundary = S_RV_ED.vtk

      - Reverse normals for inner surfaces so tetrahedralization understands them as holes/cavities.
      - Clean duplicate points, triangulate surfaces, check holes/non-manifold edges.

  3. Generate tetrahedral volume mesh
      - Create two volume regions:
          - material 1: LV myocardium
          - material 2: RV myocardium

      - Prefer generating LV and RV separately first, then merge into one volumetric mesh with region labels.
      - Use TetGen if available; otherwise I’ll check available local mesh tools and choose the least invasive
        option.

  4. Save outputs
      - Main combined volumetric mesh:
          - data/subject_1/volmesh/biv_myo_ED.vtu

      - Optional separated meshes:
          - data/subject_1/volmesh/lv_myo_ED.vtu
          - data/subject_1/volmesh/rv_myo_ED.vtu

      - Also save surface-cleaning intermediates only if needed for debugging.

  5. Quality checks
      - Confirm nonzero tetrahedra.
      - Report number of points/cells for LV, RV, combined mesh.
      - Check mesh bounds match the original surfaces.
      - Check material labels exist.
      - If tetrahedralization fails, I’ll report whether the cause is open surface, self-intersection, or
        overlap.

  I will not run it until you confirm this plan.
• I stopped the previous direction after your interruption. The important finding is that post-processing the volume mesh itself
  is not the best primary path here; the failures are coming from the input surface geometry being locally difficult for
  tetrahedralization.

• Yes, your proposed method is more likely to solve the problem than my last post-processing direction.

  The better approach should be:

  1. Start again from acdc_anchor.obj, not from the already generated VTK.
  2. Clean only degenerate/duplicate faces.
  3. Apply mild surface smoothing before tetrahedralization.
  4. Preserve topology: keep LV/RV cavities and valve openings/through-holes, do not fill the blood pools.
  5. Generate the volume mesh from this repaired/smoothed surface.
  6. Validate the final boundary against the original surface.

  I would not use plain strong Laplacian smoothing directly, because it can shrink the myocardium and change the ventricle
  cavities. I would use one of these safer variants:

  - Taubin smoothing: preferred, because it smooths without strong shrinkage.
  - Volume-constrained Laplacian smoothing: acceptable if kept mild.
  - Local Laplacian smoothing only near bad/self-intersecting regions: best if we can identify the problematic areas.

  So the revised plan should be:

  - Generate several smoothed surface candidates with very small smoothing strength.
  - Check each one for watertightness, Euler number, volume difference, and surface displacement.
  - Only accept a surface if it still matches the original topology and volume closely.
  - Then tetrahedralize that accepted surface.
  - Save the new final .vtk.

  In short: yes, smoothing the surface before generating the body mesh is a reasonable and likely better fix. I would use mild
  Taubin or constrained Laplacian smoothing, not aggressive global Laplacian smoothing.
