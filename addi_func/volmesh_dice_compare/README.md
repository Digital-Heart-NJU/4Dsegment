# Volumetric mesh Dice comparison

## Manual interactive alignment

Open a desktop window with subject0081 fixed (blue) and subject 11 moving (orange):

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/manual_align_meshes.py
```

The default inputs are in `volmesh_dice_compare/5`. The script works from any
working directory. Override them using `--fixed PATH --moving PATH`. A working
desktop display is required; PyVista and VTK are sufficient, with no Qt dependency.
Translation, rotation and mirror reflection are available. The moving mesh keeps
its size, distances and topology; mirroring reverses its handedness. The entire
moving mesh is transformed as one object, while the template stays fixed.

- Drag colored arrows to translate and colored arcs to rotate around the current
  moving-volume center. The handles recenter after each drag. Drag the background
  to rotate the camera; scroll to zoom. The template cannot be dragged.
- Arrow keys translate along world X/Y. PageDown/PageUp translate along world Z.
- `a`/`d`, `w`/`s`, `z`/`x` rotate negatively/positively about world X, Y, Z.
- `5`/`6`/`7` mirror along world X/Y/Z, respectively, through the current moving
  center. For example, X maps `x` to `2*center_x - x`. The mirror plane follows
  world axes, even after rotating the object. Press the same key twice without
  other adjustments to restore the previous pose. Mirror changes support undo,
  reset, saving and resuming, and the status shows whether the pose is reflected.
- `f` toggles coarse/fine increments: 1 mm/1 degree or 0.1 mm/0.1 degree by default.
  Set coarse increments using `--translation-step` and `--rotation-step`.
- `u` undoes a pose change (up to 200 retained changes); `r` restores original
  coordinates; `c` matches the bounding-box centers while keeping current rotation.
- `1`/`2`/`3` select views along X/Y/Z; `4` selects isometric; `v` fits the view.
- Sliders adjust opacity. `t`/`o` toggle template/object visibility; `g` toggles
  transform handles; `p` switches parallel/perspective projection.
- `k` saves the full moving volume, transform JSON and screenshot. `q` closes
  without saving further changes. Pose changes are marked UNSAVED until saved.

Outputs default to `5/manual_alignment/`: `subject_11_volmesh_aligned.vtk`,
`transform.json`, and `alignment_preview.png`. Each new session chooses a numbered
directory if that directory already contains files. Repeated saves within the same
session update that session's outputs. Inputs are never overwritten. Saving stages
and reloads the full mesh to check coordinates, connectivity, arrays and signed
tetrahedral volumes before committing it. Scalar arrays are preserved; active
VTK vectors, normals and tensors transform with the geometry. For a reflected
pose, exporting swaps the first two local vertex slots in every tetrahedron to
preserve its original signed orientation. Point order, cell order, cell labels
and each tetrahedron's four vertex IDs are preserved. The JSON records the net
reflection and this orientation correction.

Resume a saved pose with:

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/manual_align_meshes.py \
  --transform-in volmesh_dice_compare/5/manual_alignment/transform.json
```

Resume with the **original moving mesh**, since the JSON maps its original
coordinates to the template frame: `p_fixed = A @ p_original_moving + t`, where
`A` is an orthogonal matrix with determinant +1 or -1. For a pose with determinant
+1, the JSON's `source_to_reference` format also works with the Dice utility's
`--transform-in`; specify subject 11 as `--source`, subject0081 as `--reference`,
and `--no-trim` when comparing these full volumes. The current Dice utility accepts
only proper rigid transforms, so it rejects a mirrored JSON (determinant -1).
The manual alignment tool can resume either kind of transform. Units default to mm;
`--units` changes the display label only, never the coordinates.

For a single off-screen preview without opening a window or exporting a volume:

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/manual_align_meshes.py --preview-only /tmp/manual_alignment.png
```

Run its geometry, export and off-screen controller tests with:

```bash
conda run --no-capture-output -n cme-dt-env python -m unittest discover \
  -s tests -p test_manual_align_meshes.py -v
```

## Fast movable cutting plane and Dice

Run from `addi_func` (the existing filename/command is retained):

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/interactive_coords_z_dice.py
```

Defaults are the template `5/subject0081_mesh_all.vtk` and the complete aligned
volume `5/manual_alignment_002/subject_11_volmesh_aligned.vtk`. COORDS_Z files
are not needed. Both original files are read only. Only the template is clipped;
the entire aligned subject 11 volume stays in the Dice comparison.

The interface now uses **one movable cutting plane** with a lightweight preview:

- Blue is retained template, red is discarded template, and orange is subject 11.
- Move the **Plane position** slider to slide the plane along its current normal.
  Drag the plane center to reposition it, or drag the normal arrow to tilt it.
- **Flip Kept Side** (`f`) swaps blue/red sides. The normal arrow points toward
  the positive side; the default retains the negative side.
- `x`/`y`/`z` orient the plane perpendicular to a world axis.
- **Reset Plane** (`r`) restores the initial plane; `u` undoes completed edits.
- `g` toggles the plane, `t` toggles red discarded tissue, and `o` toggles subject 11.
- `1`/`2`/`3` select X/Y/Z camera views, `4` isometric, `v` fits, and `q` closes.
- **Confirm & Calculate Dice** (`c`) confirms the current plane and starts the
  full-resolution calculation in the background. Opening or adjusting the plane
  never starts a volume calculation.

At startup, boundary surfaces are extracted and simplified once (target 30,000
triangles per mesh). Plane movement only updates the renderer's clipping planes.
No tetrahedral clipping, surface rebuilding, merging or volume statistics run
while you adjust the plane. The exterior preview is approximate and open at the
cut; it is not the tetrahedral mesh used for Dice. Adjust `--preview-faces` if
needed (for example `--preview-faces 10000` for a lighter display).

After confirmation, the original tetrahedral template is clipped once at the
chosen plane. Crossing tetrahedra are geometrically split, and retained plus
removed volumes are checked against the original volume for both tags and overall.
The aligned mesh receives no additional transformation or clipping. Its tissue
outside the retained template still contributes to its Dice denominator.

Volume Dice is reported for tag 0, tag 1 and their union. Tag 0 is tissue, with
background encoded separately. The overall score is not the average of tag scores.
Both volumes are sampled by actual tetrahedral containment on one physical grid
covering their union. Defaults are 1.0 and 0.5 mm spacing, with a resolution
sensitivity check at tolerance 0.005. Set `--spacing` and
`--convergence-tolerance` to change these. Empty/empty tag scores are undefined.

Results appear in the window and are saved to `5/dice_manual_plane/`, with numbered
folders for subsequent confirmed calculations. Moving the plane afterward marks
scores as previous results; confirm again to recalculate. Plane edits are locked
during a calculation. Closing the window lets a confirmed calculation finish.

Saved files are `selection_preview.png`, `template_selected.vtu`,
`template_removed.vtu` (when nonempty), `selection.json`, `dice_scores.csv`, and
`results.json`. The selection records the plane origin in mm, unit normal and
kept side. The completed report includes full-resolution native volumes, grid
settings, convergence, and checks that input SHA-256 hashes are unchanged.
`results.json` is written only after successful calculation.

Restore a saved plane with:

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/interactive_coords_z_dice.py \
  --selection-in volmesh_dice_compare/5/dice_manual_plane/selection.json
```

Use `--normal NX NY NZ --origin X Y Z` for an exact initial plane, and optionally
`--keep-side positive` (default negative). Negative retains
`dot(normal, point-origin) <= 0`; positive retains `>= 0`. The initial GUI default
is world Z through the template center, for manual adjustment rather than an
inferred anatomical boundary. `--template`, `--aligned` and `--output-dir` override
paths. The saved plane has no dependency on the simplified preview resolution.

For a preview without calculating Dice:

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/interactive_coords_z_dice.py \
  --normal 1 0 0 --origin 50 0 0 --preview-only /tmp/plane_preview.png
```

For noninteractive calculation, add `--headless` with explicit `--normal` and
`--origin`, or with `--selection-in PATH`. This counts as confirmation. The maximum
is 80 million voxel centers per spacing by default (`--max-grid-points`). Finer
grids need more time and memory. Old box/COORDS_Z selection files are not plane
selections and are rejected by `--selection-in`.

Run focused geometry, preview-side consistency, deferred-clipping and interface tests:

```bash
conda run --no-capture-output -n cme-dt-env python -m unittest discover \
  -s tests -p test_interactive_coords_z_dice.py -v
```

## Automated Dice comparison

`calculate_volmesh_dice.py` aligns a tetrahedral mesh with extra basal/valve anatomy
to a reference mesh, clips both using a shared anatomical boundary, and measures
volume overlap. Input VTK files are read only. Coordinates are assumed to be mm.

Run from `addi_func`:

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/calculate_volmesh_dice.py \
  --case-dir volmesh_dice_compare/1
```

The defaults are source `subject0081_mesh_all.vtk`, reference
`subject_11_volmesh.vtk`, output `1/dice_results`, and voxel spacings 1.0 and 0.5 mm.
Use `--source`, `--reference`, and `--output-dir` for other paths. Existing output
directories require `--overwrite`. Dependencies are NumPy, SciPy, PyVista/VTK,
and Matplotlib, available in `cme-dt-env`; a GUI is not required.

## Registration using anatomical landmarks with a fixed LV apex

For folder 4, use:

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/calculate_volmesh_dice.py \
  --case-dir volmesh_dice_compare/4 --registration-method landmarks --no-trim
```

In this mode the moving volume defaults to `subject_11_volmesh.vtk`, the fixed
volume defaults to `subject0081_mesh_all.vtk`, and outputs default to
`4/dice_results_landmarks`. The existing ICP mode keeps its previous direction and
defaults. Override volume paths with `--source` and `--reference` if necessary;
`--landmarks-dir` selects the sidecar directory (default: `--case-dir`).

The following files define three corresponding landmarks in each volume's native
coordinate frame:

| Landmark | Fixed subject0081 | Moving subject 11 |
|---|---|---|
| LV endo apex | Referenced vertex of `Mesh_LV_endo_timeframe001.vtk` maximizing its minimum distance to the aorta-valve triangles | Coordinate in `subject_11_ED_apex_point.vtk` |
| LV mitral point | Area-weighted surface centroid of `Mesh_aorta_valve_timeframe001.vtk`, as explicitly requested | Length-weighted curve centroid of `subject_11_ED_LV_opening_rim.vtk` |
| RV tricuspid point | Area-weighted surface centroid of `Mesh_tricuspid_valve_timeframe001.vtk` | Length-weighted curve centroid of `subject_11_ED_RV_opening_rim.vtk` |

The fixed endocardial and valve files share a 5,810-point table containing unused
points. Only vertices referenced by the named surface's faces participate in apex
selection or centroid computation. Triangle-interior distances, rather than
nearest-vertex or centroid distances, define the fixed apex.

The entire moving volume is transformed by
`T(x) = fixed_apex + R @ (x - moving_apex)`. A proper rotation `R` minimizes the
sum of squared mitral and tricuspid errors with equal weights. The SVD fit uses
the two apex-relative valve vectors without centering them. Apex coincidence is
a hard constraint (tolerance 1e-6 mm); scaling and subsequent ICP are excluded.
Different landmark-triangle dimensions can leave nonzero valve errors, which are
reported individually along with the objective and pairwise landmark distances.

`landmarks.json` associates each reference point with its full volume, recording
nearby boundary points, original volume-cell IDs, and volume-vertex IDs. These
nearest locations are diagnostic: geometric centroids retain their definitions
and are not snapped to arbitrary wall points. Landmark VTP files and residual
lines can be displayed together with the aligned volume in ParaView. The preview
marks apex (A), mitral (M), and tricuspid (T), before and after registration.

Dice still uses **both complete tetrahedral volumes** on the common grid; the six
sidecars are used only to estimate registration. Landmark mode defaults to no
trimming unless explicit `--plane` boundaries are supplied. `source_aligned.vtu`
is the full subject 11 volume in subject0081's coordinate system. The report checks
native-volume and connectivity preservation and verifies input-file SHA-256 hashes
before/after execution. Input meshes and sidecars are read only.

## Alignment and trimming

Boundary triangles are sampled by area. The script tests 25 rigid initial
alignments and refines the strongest candidates using symmetric, tag-constrained,
trimmed ICP and robust point-to-plane fitting. Refinement checks normal agreement
to reduce matches between opposing inner/outer walls. The source surface used for
refinement is restricted to the reference basal region plus a 2 mm margin.
Scaling, reflections, and nonrigid deformation are excluded.

Automatic trimming requires one dominant, planar, supporting cap on the reference.
It fits a plane and retains the halfspace containing the ventricular body. This
is a geometric inference, not valve segmentation: review `comparison.png` and
`basal_planes.vtp`. Several comparable flat patches cause an error rather than an
arbitrary anatomical choice. An uncertain or nonplanar base needs explicit planes.

Supply one or more reference-frame planes with `--plane NX NY NZ D`, retaining
`NX*x + NY*y + NZ*z <= D`. For example, `--plane 0 0 1 30` retains `z <= 30`.
Normals and offsets are normalized together. Multiple planes intersect their
retained halfspaces. `--no-trim` calculates aligned full-volume scores only.

Trimming never follows the intersection of the two volumes, a nearest-distance
threshold, or the Dice score. Lateral wall disagreements remain in the comparison.
Clipped meshes split boundary elements, retain cell labels, and are tetrahedralized;
the script checks that retained plus removed volumes equal each input volume.

## Labels and Dice

The script compares numeric `elemTag` values by default. Confirm their anatomical
meanings before describing results as LV or RV. If needed, map source tags explicitly:

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/calculate_volmesh_dice.py \
  --case-dir volmesh_dice_compare/1 --label-map 0:1 --label-map 1:0 \
  --output-dir volmesh_dice_compare/1/dice_results_remapped
```

This example demonstrates syntax; it does not recommend swapping these files' tags.
**Input tag 0 is tissue.** Saved mask codes start at 1, reserving code 0 for
background. `results.json` records the mapping.

Voxel centers are tested for containment in actual tetrahedra with VTK probing.
Both volumes use the same grid covering their full union. No cavity filling or
nearest-vertex occupancy approximation is used. For the trimmed result, both masks
are restricted to the same basal halfspace(s). Dice is twice the intersection
volume divided by the sum of volumes. Empty/empty regions are reported as undefined.

The report compares the two finest requested spacings. The default acceptable
absolute Dice change is 0.005; this is a numerical convergence check, not an
anatomical validation or registration acceptance criterion. After alignment,
different subjects or cardiac phases can still have low Dice.

## Outputs and refinement

- `dice_scores.csv`, `results.json`: full aligned and basal-trimmed scores for all
  tissue and each tag, voxel volumes, native volumes, removed fractions, convergence,
  transformation, plane information, and interpretation limits.
- `transform.json`, `clipping_planes.json`: reusable rigid matrix and trimming rule.
- `source_aligned.vtu`, `source_retained.vtu`, `reference_retained.vtu`: comparison
  meshes containing geometry, `elemTag`, and original cell IDs. Other input arrays,
  such as fibers, are not copied to these comparison artifacts.
- `source_removed.vtu`, `reference_removed.vtu`: excluded volume, when nonempty.
- `basal_planes.vtp`, `comparison.png`: clipping plane and three orthogonal views
  before/after clipping (reference blue, source orange, removed source red).
- `comparison_masks.vti`: finest-grid cell labels and comparison-region mask for
  inspection in ParaView or PyVista.

Use the saved transform to check finer spacing without repeating registration:

```bash
conda run --no-capture-output -n cme-dt-env python \
  volmesh_dice_compare/calculate_volmesh_dice.py \
  --case-dir volmesh_dice_compare/1 \
  --transform-in volmesh_dice_compare/1/dice_results/transform.json \
  --spacing 0.5 0.25 --output-dir volmesh_dice_compare/1/dice_results_finer
```

`--plane-offset-mm` shifts the boundary outward along its unit normal; negative
values clip deeper. Reuse the transform and use separate output directories to
assess boundary sensitivity. A finer grid may need a larger `--max-grid-points`
(default 80 million); memory use grows with voxel count.

Run the analytical geometry tests:

```bash
conda run --no-capture-output -n cme-dt-env python -m unittest discover \
  -s tests -p test_calculate_volmesh_dice.py -v
```
