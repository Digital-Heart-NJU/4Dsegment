# Cardiac-axis and short-axis modification work log

Date: 2026-10-10

This file records today's commands, code modifications, test results, and the
recommended replacement method for Step 2 base-side detection.

## Main source file

`/home/mengao/Code/4Dsegment/model_vol_septum/generate_open_volmesh_stable.py`

The pipeline currently runs through Step 7. The whole surface generated in
Step 6 uses a Manifold3D Boolean union. Later TetGen and volume-tagging stages
remain disabled in the active path.

## Commands executed today

### Syntax verification

```bash
python -m py_compile model_vol_septum/generate_open_volmesh_stable.py
```

### Subject and frame tests

```bash
conda run -n morphinet python model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_1 \
  --frames ED,ES

conda run -n morphinet python model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_2 \
  --frames ED,ES

conda run -n morphinet python model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_3 \
  --frame ES
```

Subject 3 ED was also tested with the normal pipeline:

```bash
conda run -n morphinet python model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_3 \
  --frame ED
```

Additional diagnostic tests for Subject 3 ED were run with:

```bash
conda run -n morphinet python model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_3 \
  --frame ED \
  --stop-after-open-shells \
  --boundary-smooth-iterations 0 \
  --boundary-blend-rings 0 \
  --boundary-blend-relaxation 0

conda run -n morphinet python model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_3 \
  --frame ED \
  --stop-after-open-shells \
  --opening-strategy max_interface

conda run -n morphinet python model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_3 \
  --frame ED \
  --stop-after-open-shells \
  --subject-base-side low_z
```

### Output validation

The generated Step 6 surfaces were loaded with PyVista and checked for:

- number of points;
- number of triangles;
- free edges;
- manifold status;
- LV/RV `elemTag` counts.

The validation covered all available files matching:

`data/subject_[1-3]/volmesh_pipline/step_06_reconstruct_annulus/subject_*_*_LVRV_surf.vtk`

## Code modifications made today

### 1. Replaced Ball Pivoting

The previous Step 6 method merged six point clouds and reconstructed triangles
with Open3D Ball Pivoting. This was removed from the active path because it
could create bridges, lose the source triangle topology, and introduce
uncontrolled connections between LV and RV surfaces.

The active method is now:

```text
LVepi + LVendo + LV annulus -> closed LV shell
RVepi + RVendo + RV annulus -> closed RV shell
closed LV shell + closed RV shell -> Manifold3D Boolean union
```

New functions added to the source file:

- `_prepare_manifold_shell`
- `save_step6_manifold_union_reconstruction`

The active pipeline now calls:

`save_step6_manifold_union_reconstruction`

### 2. Manifold3D shell preparation

For each chamber, the code:

1. loads the three Step 6 triangle surfaces;
2. merges coincident points within the chamber;
3. checks that the shell has no free edges;
4. checks that the shell is manifold;
5. consistently orients the shell triangles;
6. converts the shell to `manifold3d.Mesh`;
7. validates the result as a Manifold3D solid.

### 3. Boolean union and exact intersection handling

The LV/RV union uses:

```python
whole_mesh = lv_manifold + rv_manifold
```

Manifold3D computes the shell intersection, creates exact intersection
vertices, splits affected faces, removes overlapping internal faces, and
returns a valid output manifold.

The output triangle source IDs are preserved. LV-derived triangles receive
`elemTag=1`, while RV-derived triangles receive `elemTag=0`.

### 4. Increased VTK coordinate precision

The custom triangle-surface writer now accepts a `point_precision` argument.
The Boolean-union output is written with nine decimal places instead of five:

```python
write_custom_surface_vtk(..., point_precision=9)
```

This prevents small seam gaps caused by coordinate rounding. Subject 2 ED
initially showed five free edges after five-decimal writing; after this change,
it regenerated with zero free edges.

## Successful whole-surface outputs

All successful files are saved in each subject's:

`volmesh_pipline/step_06_reconstruct_annulus/`

| Subject/frame | Points | Triangles | Free edges | Manifold |
|---|---:|---:|---:|---|
| subject_1 ED | 38,141 | 76,278 | 0 | Yes |
| subject_1 ES | 32,508 | 65,012 | 0 | Yes |
| subject_2 ED | 39,879 | 79,754 | 0 | Yes |
| subject_2 ES | 34,982 | 69,960 | 0 | Yes |
| subject_3 ES | 33,941 | 67,874 | 0 | Yes |

The generated filename format is:

`subject_x_<frame>_LVRV_surf.vtk`

The outputs contain LV/RV tags. For example, the current tag counts are:

```text
subject_1 ED: tag 0 = 45,560; tag 1 = 30,718
subject_1 ES: tag 0 = 41,690; tag 1 = 23,322
subject_2 ED: tag 0 = 46,835; tag 1 = 32,919
subject_2 ES: tag 0 = 40,731; tag 1 = 29,229
subject_3 ES: tag 0 = 36,690; tag 1 = 31,184
```

## Subject 3 ED failure

Subject 3 ED does not currently reach Step 6. It fails during Step 3 LV
open-base shell construction:

```text
RuntimeError: LV: shell has 52 free edges after repair.
```

The subject-level detector selects `high_z` for Subject 3, supported by both
ED/ES LVendo and RV endocardial profiles. Therefore, this failure is not
caused by an obvious global base-direction reversal.

The following tests did not resolve the failure:

- disabling boundary smoothing;
- changing the opening strategy to `max_interface`;
- forcing `low_z` as the subject base side.

The current evidence points to a Subject 3 ED LV shell or clipping topology
problem during Step 3, before Manifold3D is called.

## Step 2 analysis and proposed modification

### Current Step 2 behavior

The current detector:

1. uses the global mesh Z direction;
2. creates Z slabs;
3. projects slab points onto the global XY plane;
4. calculates convex-hull area per slab;
5. compares the low-Z and high-Z end areas;
6. votes across ED/ES LVendo and RV surfaces;
7. applies one subject-level `low_z` or `high_z` decision.

This can fail when the heart is rotated relative to the global Z axis. A fixed
Z slab can cut the heart obliquely, and the XY convex hull can overestimate or
merge cross-sectional regions.

### Recommended replacement

Use a subject-level cardiac long axis followed by true short-axis sections:

```text
Original LV/RV surfaces from ED and ES
        -> robust cardiac long-axis estimation
        -> planes perpendicular to that axis
        -> smoothed short-axis area profile
        -> sustained basal plateau detection
        -> subject-level base/apex decision
        -> open-base processing along the cardiac axis
```

The proposed method is:

1. Combine the original LVendo, LVepi, RVendo, and RVepi points from ED and
   ES.
2. Remove extreme outliers and estimate the first principal component as a
   candidate cardiac long axis.
3. Resolve the axis sign by comparing the two ends of the combined ventricular
   shape.
4. Create planes perpendicular to the axis rather than fixed XY planes.
5. Intersect the LV/RV surfaces with those planes.
6. Calculate the 2D cross-sectional area in each short-axis plane.
7. Smooth the profile and evaluate a basal interval, not one maximum slice.
8. Select as base the end with the larger sustained area plateau and clearer
   taper toward the opposite apex.
9. Use the same subject-level axis and base direction for ED and ES.

### Required downstream generalization

Changing only the Step 2 label is not sufficient. The opening code must also
replace global-Z operations with axis-coordinate operations:

```text
global z coordinate       -> s = dot(point - axis_origin, cardiac_axis)
[0, 0, +/-1] direction   -> +/- cardiac_axis
Z basal distance          -> distance along cardiac_axis
XY slab area              -> area of a plane perpendicular to cardiac_axis
```

The future implementation should save the cardiac axis, axis origin, short-
axis profile, base/apex positions, and confidence scores for inspection.

## Current status

- Manifold3D whole-surface reconstruction works for five of six requested
  subject/frame combinations.
- All five successful outputs are closed and manifold after high-precision
  writing.
- Subject 3 ED still requires a Step 3 LV shell/topology fix.
- The cardiac-axis short-axis method is a recommended Step 2 redesign only;
  it has not yet been implemented in the code.

