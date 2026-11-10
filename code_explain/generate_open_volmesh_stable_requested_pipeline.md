# generate_open_volmesh_stable.py Requested Pipeline

This note documents the current pipeline implemented in `../model_vol_septum/generate_open_volmesh_stable.py`. It focuses on the path from four input surface meshes to the step-wise open-base surfaces, LV/RV myocardial shells, and optional TetGen volume mesh.

## Inputs

For one subject and one frame, the script reads these files from `SUBJECT_DIR/vtks` by default:

| File | Meaning | Role in this script |
|---|---|---|
| `LVendo_FRAME.vtk` | LV endocardium / LV blood-pool boundary | Inner LV surface for LV myocardium shell |
| `LVepi_FRAME.vtk` | LV epicardium / LV outer wall | Outer LV surface and fixed anatomical reference for RV/LV intersection cleanup |
| `RV_FRAME.vtk` | RV endocardium / RV blood-pool boundary | Inner RV surface, later clipped/moved to enforce thickness |
| `RVepi_FRAME.vtk` | RV epicardium / RV outer wall | Outer RV surface, later clipped against LVepi |

Example for subject 1 ED:

```bash
conda run --no-capture-output -n morphinet python -u model_vol_septum/generate_open_volmesh_stable.py   --subject-dir data/subject_1   --frame ED   --stop-after-merge-surface
```

## Tag Rule

All saved VTKs use the custom ASCII legacy unstructured-grid style used by `custom_vtk.py`.

| `elemTag` | Meaning |
|---:|---|
| `1` | LV myocardium or LV-related surface cells |
| `0` | RV myocardium or RV-related surface cells |

Step 10 now saves two separate TetGen volume meshes in `step10tetgen_LVRV`: LV cells are written with `elemTag=1`, and RV cells are written with `elemTag=0`.

## Output Folders

Stage outputs are saved under:

```text
SUBJECT_DIR/volmesh_generate_pipline/
```

The main subfolders are:

| Folder | Purpose |
|---|---|
| `step3open_base` | Immediate LV/RV open-base surfaces and shells before RV/LV intersection cleanup |
| `step5remove_intersection` | RVepi/RVendo after intersection-point removal against LVepi |
| `step7moveRVendo` | RVendo after enforcing minimum RV wall thickness and updated RV open-base annulus |
| `step8fix_intersection_new_surf` | Final fixed LV/RV surfaces and rebuilt LV/RV open-base annuli |
| `step9merge_surface` | Final LV/RV myocardial shell inspection files |
| `step10tetgen_LVRV` | Separate LV and RV TetGen surfaces and volume meshes |

If step 11 is run, the final tagged tetrahedral mesh is saved under `--output-root`, defaulting to:

```text
SUBJECT_DIR/volmesh_tetgen/
```

## Pipeline With Detailed Stage Meaning

```text
LVendo/LVepi/RV/RVepi input surfaces
  -> base-side detection and z-area profiles
  -> LV open-base shell
  -> initial RV open-base shell
  -> remove RVepi-LVepi intersection points
  -> remove RVendo-LVepi intersection points
  -> move RVendo to satisfy RVendo-RVepi distance >= 3 mm
  -> rebuild fixed LV/RV surfaces and open-base annuli
  -> fill RV septal gaps with LVepi patch and stitch annuli
  -> save LV and RV myocardial shells
  -> optional separate LV/RV TetGen volume meshes
  -> optional final merge of LV/RV tagged volume meshes
```

## Step 1: Load Surfaces

The script loads the four VTK files and converts them to triangle surface meshes. LV surfaces are treated as fixed references in later intersection-removal steps. RV surfaces are allowed to be clipped and RVendo can be moved to satisfy wall-thickness constraints.

The code path is:

```text
load_surfaces -> base.load_surface -> extract_surface/triangulate/clean
```

## Step 2: Detect Subject Base Side And Write Z-Area Profiles

The base side is detected from the original subject surfaces. This is important because later clipped surfaces may have missing regions and should not drive basal-side detection.

The z-area profiles are written so the basal decision can be checked manually. The profile records how cross-sectional area changes along z and determines whether the base is on the low-z or high-z side.

Outputs are written into the normal output folder, usually `volmesh_tetgen`:

```text
subject_X_FRAME_LV_z_area_profile.csv
subject_X_FRAME_RV_z_area_profile.csv
```

## Step 3: Build The Open-Base LV Myocardial Shell

LVendo and LVepi are clipped at the detected basal side. The open LVendo boundary and open LVepi boundary are connected by a generated annulus.

The immediate LV shell is:

```text
LV_openbase_myo_shell = LVepi_open + LVendo_open + LV_openbase_annulus
```

Saved files in `step3open_base` include:

```text
subject_X_FRAME_stage03_LV_openbase_myo_shell.vtk
subject_X_FRAME_stage03_LV_openbase_epi.vtk
subject_X_FRAME_stage03_LV_openbase_endo.vtk
subject_X_FRAME_stage03_LV_openbase_annulus.vtk
```

These files should only show the basal opening operation. They should not include RV/LV intersection edits.

## Step 4: Build The Initial Open-Base RV Myocardial Shell

RVendo and RVepi are opened by the same basal rule. This stage also saves the RV shell before any LVepi intersection removal, so it can be compared against later clipped RV outputs.

The immediate RV shell is:

```text
RV_initial_openbase_myo_shell = RVepi_open + RVendo_open + RV_openbase_annulus
```

Saved files in `step3open_base` include:

```text
subject_X_FRAME_stage04_RV_initial_openbase_myo_shell.vtk
subject_X_FRAME_stage04_RV_initial_openbase_epi.vtk
subject_X_FRAME_stage04_RV_initial_openbase_endo.vtk
subject_X_FRAME_stage04_RV_initial_openbase_annulus.vtk
```

## Step 5: Remove RVepi Points Intersecting LVepi

The closed original LVepi surface is used as the fixed reference. RVepi points that are inside or very near LVepi are treated as intersecting points.

The operation is not a simple vertex deletion. The code handles triangles as follows:

| Triangle condition | Action |
|---|---|
| No intersecting vertices | Keep the original triangle |
| All vertices intersect LVepi | Delete the triangle |
| Mixed valid/intersecting vertices | Clip and reconstruct the triangle polygon |

For mixed triangles, new cut vertices are computed along the triangle edge and projected onto LVepi using a VTK closest-point locator. This preserves the local boundary better than deleting the entire mixed triangle.

The result is an RVepi surface that is open at the base and also open along the LVepi contact region. That second loop is the septal gap loop used later.

Saved file:

```text
step5remove_intersection/subject_X_FRAME_stage05_RVepi_after_LVepi_intersection_removal_surface.vtk
```

## Step 6: Remove RVendo Points Intersecting LVepi

The same clipped-triangle reconstruction is applied to RVendo. The result is an RVendo surface with:

- one basal opening loop
- one LVepi-contact gap loop

Saved file:

```text
step5remove_intersection/subject_X_FRAME_stage06_RVendo_after_LVepi_intersection_removal_surface.vtk
```

## Step 7: Move RVendo To Enforce Minimum RV Wall Thickness

RVepi is fixed. RVendo is moved inward where the RVendo-to-RVepi distance is below the requested minimum, currently 3 mm by default.

Main rule:

```text
RVendo-to-RVepi distance >= 3 mm
```

The basal RVendo opening ring is not kept rigid. Instead, it is updated adaptively from the displacement of neighboring points and smoothed. This avoids a bad visualization where the surrounding RVendo moves but the opening ring remains frozen.

After RVendo is moved, the RV open-base annulus is rebuilt from:

```text
fixed RVepi basal ring + adjusted RVendo basal ring
```

Saved files in `step7moveRVendo`:

```text
subject_X_FRAME_stage07_RVepi_fixed_surface.vtk
subject_X_FRAME_stage07_RVendo_after_3mm_distance_surface.vtk
subject_X_FRAME_stage07_RV_openbase_annulus_after_adaptive_RVendo_ring_surface.vtk
```

Point-data arrays may include displacement/debug fields such as `movedByStep7`, `distanceToRVepi`, `belowRequested3mm`, `adaptiveOpeningBoundary`, and `openingTransitionBand`.

## Step 8: Re-Fix RV/LVepi Intersections And Rebuild Annuli

After RVendo has moved, the script checks RVepi and RVendo again against the fixed LVepi reference. LVendo and LVepi are kept fixed. RVepi and RVendo may be clipped again if overlap remains.

Then the open-base annuli are rebuilt directly from current boundary loops:

```text
LV_openbase_annulus = connect current LVepi basal loop to current LVendo basal loop
RV_openbase_annulus = connect current RVepi basal loop to current RVendo basal loop
```

Saved files in `step8fix_intersection_new_surf`:

```text
subject_X_FRAME_stage08_LVendo_openbase_surface.vtk
subject_X_FRAME_stage08_LVepi_openbase_surface.vtk
subject_X_FRAME_stage08_RVendo_fixed_openbase_surface.vtk
subject_X_FRAME_stage08_RVepi_fixed_openbase_surface.vtk
subject_X_FRAME_stage08_LV_openbase_annulus_surface.vtk
subject_X_FRAME_stage08_RV_openbase_annulus_surface.vtk
```

These are the final surface pieces used by step 9.

## Step 9: Fill RV Septal Gaps And Build Myocardial Shells

After steps 5 and 6, RVepi and RVendo are not closed surfaces. They are open at the base and also have non-basal gap loops where intersection with LVepi was removed.

Step 9 fills the RV septal gap before constructing the RV myocardial shell.

The current strategy is:

```text
1. Detect all RVepi boundary loops.
2. Use base-side rules to identify the basal opening loop.
3. Treat the remaining RVepi boundary loop as the LVepi-contact gap loop.
4. Repeat the same loop classification for RVendo.
5. Extract a patch from the existing LVepi surface:
     LVepi face centers inside original RVepi and outside original RVendo.
6. Match the LVepi patch boundary loops to the RVepi and RVendo gap loops.
7. Build narrow stitch annuli between:
     RVepi gap loop -> matching LVepi patch loop
     RVendo gap loop -> matching LVepi patch loop
8. Merge RVepi, RVendo, RV open-base annulus, LVepi patch, and stitch annuli.
9. Merge LVepi, LVendo, and LV open-base annulus separately.
```

This gives two chamber shells:

```text
LV myocardial shell = LVepi + LVendo + LV_openbase_annulus
RV myocardial shell = RVepi + RVendo + RV_openbase_annulus
                    + LVepi septal gap-fill patch
                    + RV-to-LVepi-patch stitch annuli
```

Saved files in `step9merge_surface`:

```text
subject_X_FRAME_stage09_LV_openbase_myo_shell.vtk
subject_X_FRAME_stage09_RV_openbase_myo_shell_filled_by_LVepi.vtk
subject_X_FRAME_stage09_RV_LVepi_gap_fill_patch.vtk
subject_X_FRAME_stage09_RV_LVepi_gap_stitch_annuli.vtk
subject_X_FRAME_stage09_whole_epi_surface.vtk
subject_X_FRAME_stage09_whole_endo_surface.vtk
subject_X_FRAME_stage09_whole_epi_endo_surface.vtk
```

Expected check for the two shell files:

```text
LV myo shell free_edges = 0
RV myo shell free_edges = 0
```

For the latest subject_1 ED run, the shell checks were:

```text
LV myo shell: 31400 triangles, free_edges=0, elemTag={1: 31400}
RV myo shell: 49040 triangles, free_edges=0, elemTag={0: 49040}
```

## Step 10: Separate LV/RV TetGen Volume Meshes

The script can stop after separate LV/RV TetGen by using:

```bash
--stop-after-tetgen
```

The step-10 outputs are saved in:

```text
step10tetgen_LVRV/subject_X_FRAME_stage10_LV_tetgen_surface_used.vtk
step10tetgen_LVRV/subject_X_FRAME_stage10_RV_tetgen_surface_used.vtk
step10tetgen_LVRV/subject_X_FRAME_stage10_LV_tetgen_volmesh.vtk
step10tetgen_LVRV/subject_X_FRAME_stage10_RV_tetgen_volmesh.vtk
step10tetgen_LVRV/subject_X_FRAME_stage10_LVRV_tetgen_summary.json
```

The LV volume mesh is tetrahedralized from the closed step-9 LV myocardial shell and is written with `elemTag=1` for all tetrahedra. The RV volume mesh is tetrahedralized from the closed step-9 RV myocardial shell and is written with `elemTag=0` for all tetrahedra.

TetGen still requires each chamber shell to be valid:

```text
closed surface shell
no free boundary edges
no self-intersections
no segment/facet or facet/facet intersections
consistent orientation
```

For subject_1 ED, raw LV TetGen failed because TetGen detected three self-intersecting input triangles. The current default `--tetgen-input-mode raw_then_repair` succeeded by using the repaired LV/RV shell candidates. The generated Step 10 meshes were:

```text
LV: 32485 points, 154079 tetrahedra, elemTag={1: 154079}
RV: 35070 points, 153375 tetrahedra, elemTag={0: 153375}
```

Repair mode can move or simplify surfaces, so `*_stage10_LV_tetgen_surface_used.vtk` and `*_stage10_RV_tetgen_surface_used.vtk` should be checked together with the volume meshes in ParaView.

## Step 11: Final Tagged Volume Mesh

If the script runs past step 10, it performs final post-processing:

```text
1. Merge the separate LV and RV TetGen outputs.
2. Keep constant chamber tags from Step 10.
3. Remove tiny disconnected tet components.
4. Write final custom VTK with elemTag.
5. Write summary JSON and elemTag legend CSV.
```

The final intended tag rule is:

```text
elemTag = 1 -> LV myocardium
elemTag = 0 -> RV myocardium
```

This step is intentionally separated from step 10 so the chamber-specific TetGen meshes can be inspected before the optional merged final mesh is used.

## Current Important Implementation Points

- LV surfaces are kept fixed during RV/LV intersection cleanup.
- RVepi is fixed during the 3 mm RV wall-thickness enforcement step.
- RVendo can move, but its basal ring is adaptively updated and smoothed.
- Step 9 now explicitly fills RV non-basal septal gaps before saving RV myocardial shell.
- The RV gap fill uses existing LVepi geometry as the main transition patch.
- Narrow stitch annuli are added because the LVepi patch boundary and RV clipped gap loops are spatially close but not necessarily identical vertex loops.
- The saved LV and RV myocardial shell files are the best inspection files for checking whether the chamber shells are closed before TetGen.
- Step 10 now tetrahedralizes LV and RV separately instead of using a single whole biventricular PLC.
