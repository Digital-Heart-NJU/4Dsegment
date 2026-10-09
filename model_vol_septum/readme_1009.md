# Model-vol-septum work log

Date: 2026-10-09

This file records the main commands, pipeline changes, experiments, and output
locations used today for `generate_open_volmesh_stable.py`.

## Main source file

Current implementation:

`/home/mengao/Code/4Dsegment/model_vol_septum/generate_open_volmesh_stable.py`

The active implementation currently runs the pipeline through Step 7. The
later volumetric-mesh stages remain disabled in the active frame-processing
path.

## Environment and verification commands

The model was run in the `morphinet` Conda environment:

```bash
conda run -n morphinet python model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_1 \
  --frame ED
```

The same command was rerun after the relevant Step 5, Step 6, and Step 7
changes to regenerate the current Subject 1 ED outputs.

Syntax verification:

```bash
python -m py_compile model_vol_septum/generate_open_volmesh_stable.py
```

Package checks used during reconstruction testing:

```bash
conda run -n morphinet python -c "import pyvista as pv; print(pv.__version__)"
conda run -n morphinet python -c "import open3d as o3d; print(o3d.__version__)"
```

The tested versions were PyVista `0.44.2` and Open3D `0.19.0`.

## Current input convention

The command reads the subject surface inputs under the subject directory. The
four required surfaces are the selected frame variants of:

- `LVendo_*.vtk`
- `LVepi_*.vtk`
- `RV_*.vtk`
- `RVepi_*.vtk`

The active pipeline uses the original four surfaces as Step 1 inputs and
creates the intermediate files under:

`data/<subject>/volmesh_pipline/`

The actual Step 6 folder spelling in the code is:

`step_06_reconstruct_annulus`

This is different from the earlier requested spelling `step_06_reconstruct_anulus`.

## Active processing pipeline

```text
Load four surfaces
  -> subject-level base-side detection and LV/RV Z-area profiles
  -> LV open-base myocardial shell
  -> initial RV open-base myocardial shell
  -> move RVepi outward while keeping RVendo fixed and preserving the opening ring
  -> reconstruct LV/RV opening annuli
  -> merge six Step 6 surface point sets into one point cloud
  -> reconstruct a surface from the point cloud with Open3D Ball Pivoting
  -> remove RVendo/RVepi points close to LVepi
  -> save cleaned point clouds and the merged cleaned point cloud
```

The old downstream TetGen, tag-splitting, and final volume-mesh generation
operations are not part of the active execution path after Step 7.

## Step 1: load surfaces

The code loads the four frame-specific input surfaces and keeps the original
surface geometry available for the rest of the frame. The loaded data is also
used by the later base-side decision; base detection is subject-level rather
than independently guessed from each chamber after later modifications.

## Step 2: base-side detection and Z-area profiles

The code evaluates the Z-layer area profile for the LV and RV surfaces and
selects a common subject-level base direction. The base-side decision is made
before the open-base surfaces are constructed. The corresponding diagnostic
profiles are written to the pipeline output for inspection.

## Step 3: LV open-base shell

The LV endocardial and epicardial surfaces are clipped at the detected base
side. The opening boundary loops are retained, and an LV annulus is generated
between the LVendo and LVepi opening loops. The resulting LV open-base epi,
endo, and annulus surfaces are saved for later steps.

## Step 4: initial RV open-base shell

The same base-side rule is applied to RVendo and RVepi. The initial RV
open-base epi and endo surfaces are created, together with the RV opening
annulus. This stage performs the opening operation only; the current initial
RV outputs are not additionally clipped against LVepi at this point.

## Step 5: move RVepi while preserving the RVendo shell

The current Step 5 keeps the open-base RVendo surface fixed and moves the
open-base RVepi outward so that the RVepi-to-RVendo separation is at least
3 mm. The movement direction is outward from the RVendo blood-pool side, so
the epicardium is not displaced inward into the cavity.

The opening-ring handling was changed to avoid local pointwise kinks:

1. Detect the RVepi opening boundary loop.
2. Classify movable and protected points.
3. Estimate outward directions from the local geometry.
4. Smooth the ring displacement directions along the ordered loop.
5. Use a shared displacement scale for the movable ring points instead of
   unrelated per-point offsets.
6. Apply the minimum correction needed to satisfy the distance constraint.
7. Preserve the ring topology and validate the resulting boundary.

The helper implementing this controlled ring adjustment is
`apply_uniform_ring_outward_offset`.

The current output is:

`data/<subject>/volmesh_pipline/step_05_move_RVepi/<subject>_<frame>_RV_openbase_epi_moved.vtk`

The previous moved-RVendo output is not used by the current Step 5. RVendo is
read from the Step 4 output and remains fixed in this version.

## Step 6: reconstruct opening annuli

Step 6 reads the LV and RV open-base epi/endo surfaces. It reconstructs the
opening annuli by directly connecting the corresponding epi and endo boundary
loops. The current Step 6 output names are:

```text
<subject>_<frame>_LV_openbase_epi.vtk
<subject>_<frame>_LV_openbase_endo.vtk
<subject>_<frame>_LV_openbase_annulus_recon.vtk
<subject>_<frame>_RV_openbase_epi_moved.vtk
<subject>_<frame>_RV_openbase_endo.vtk
<subject>_<frame>_RV_openbase_annulus_recon.vtk
```

They are stored in:

`data/<subject>/volmesh_pipline/step_06_reconstruct_annulus/`

The annulus construction keeps the source epi and endo surfaces fixed. The
direct connection is checked for possible intersections, and the ring
geometry is adjusted only where the current implementation requires it.

## Step 6a: merge six surfaces into a point cloud

The six Step 6 point sets are merged without tetrahedralization. The merged
point cloud preserves a `sourceSurface` point-data array so that the later
reconstruction can process the six source groups separately:

```text
0 LV epi
1 LV endo
2 LV annulus
3 RV epi
4 RV endo
5 RV annulus
```

The merged raw point cloud is:

`data/subject_1/volmesh_pipline/step_06_reconstruct_annulus/subject_1_ED_LVRV.vtk`

The point-cloud merge also writes:

`subject_1_ED_LVRV_point_cloud_summary.json`

For the tested Subject 1 ED frame, the merged point cloud contained 43,134
points and 43,134 vertex cells, with point tags `[0, 1]` in the custom VTK
representation.

## Step 6b: surface reconstruction experiments

Several point-cloud reconstruction methods were tested against the merged
Step 6 point cloud.

### PyVista Delaunay3D

The tested alpha values were `0.0`, `0.5`, `1.0`, `3.0`, and `5.0`. The
results were sensitive to alpha and generally produced many free edges or
degenerate-triangle warnings. Representative results were:

| Alpha | Surface points | Triangles | Free edges |
|---:|---:|---:|---:|
| 0.0 | 6,470 | 12,936 | 0 |
| 0.5 | 16,725 | 8,167 | 15,173 |
| 1.0 | 36,376 | 69,551 | 25,382 |
| 3.0 | 37,237 | 141,959 | 6,549 |
| 5.0 | 35,209 | 145,128 | 5,349 |

The VTK implementation reported linear-system and degenerate-triangle
warnings during these trials. This implementation is no longer active.

### VTK surface reconstruction filter

`vtkSurfaceReconstructionFilter` followed by `vtkContourFilter` was also
tested. The main parameters were:

- neighborhood size: `20`
- contour value: `0.0`
- default VTK sample spacing
- clean tolerance: `1e-7`

The test produced approximately 15,272 points, 30,226 triangles, and 322
free edges. This implementation is also no longer active.

### Open3D Ball Pivoting

The current implementation uses Open3D Ball Pivoting. The reconstruction is
performed separately for each source surface group to reduce unwanted bridges
between LV/RV epi, endo, and annulus points. For each group it:

1. Computes a median nearest-neighbor spacing.
2. Estimates normals with a K-nearest-neighbor search.
3. Orients the normals consistently with the local tangent geometry.
4. Creates ball radii from the spacing and configured multipliers.
5. Runs Open3D Ball Pivoting.
6. Converts the reconstructed triangles into a VTK surface.

The current command-line defaults are:

```text
--step6-ball-pivot-normal-k 30
--step6-ball-pivot-radii-multipliers 0.75,1.5,3.0
```

The reconstructed surface is saved as:

`data/subject_1/volmesh_pipline/step_06_reconstruct_annulus/subject_1_ED_LVRV_surf.vtk`

The reconstruction summary is saved as:

`subject_1_ED_LVRV_surface_ball_pivoting_summary.json`

For Subject 1 ED, the current reconstruction produced 42,188 surface points
and 82,980 triangles. The LV group contributed 30,844 triangles and the RV
groups contributed 52,136 triangles. The LV annulus group is a one-dimensional
ring and produced no Ball Pivoting triangles by itself. The result still has
3,393 free edges, so this output should be treated as an experimental surface
reconstruction rather than a validated closed surface for TetGen.

## Step 7: remove RV points close to LVepi

Step 7 uses the Step 6 surfaces and the Step 6 LVepi as the fixed reference.
It removes points independently from RVendo and RVepi when their distance to
LVepi is within the configured tolerance. It also applies the same point
filter to the RV annulus before creating the merged RV point cloud.

The current default tolerance is:

```text
--step7-lvepi-point-tolerance 0.05
```

The outputs are point-cloud VTKs, with one `VTK_VERTEX` cell per retained
point. The custom writer stores the requested `elemTag` cell data and can
also store point data such as source-surface identifiers.

The output folder is:

`data/<subject>/volmesh_pipline/step_07_whole_points_surf/`

The current files are:

```text
<subject>_<frame>_RVendo_clean_surf.vtk
<subject>_<frame>_RVepi_clean_surf.vtk
<subject>_<frame>_LVRV_surf.vtk
```

For Subject 1 ED, the latest validation reported:

- RVendo points removed: `49`
- RVepi points removed: `80`
- RV annulus points removed: `0`
- merged retained RV points checked: `27,006`
- remaining RV-to-LVepi point intersections: `0`

## Current limitations and status

- The active path stops after Step 7 point-cloud generation.
- The current Open3D reconstruction is grouped by source surface, but it is
  not yet a guaranteed watertight single surface.
- The current Ball Pivoting output has free edges and therefore needs further
  topology validation before it is used as a TetGen input.
- The Step 7 outputs are point clouds by design; they are not reconstructed
  triangle surfaces.
- No old moved-RVendo files are used by the current Step 5 path.
- All modifications described above are contained in
  `generate_open_volmesh_stable.py`.

