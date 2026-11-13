# Volumetric mesh Dice comparison

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
