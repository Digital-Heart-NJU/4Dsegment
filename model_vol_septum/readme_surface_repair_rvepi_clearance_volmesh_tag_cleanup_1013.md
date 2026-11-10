# Surface repair, RVepi clearance, volume generation, and LV/RV tag cleanup

Requested record date suffix: **1013**. Written on **2026-10-14**, Asia/Shanghai.

Main pipeline: [generate_open_volmesh_stable.py](generate_open_volmesh_stable.py).

## Scope and reliability of this record

This records the recent approximately 12-hour development session, spanning the
afternoon/evening of October 13 through shortly after midnight on October 14.
The available file timestamps include hole-filling work at approximately 14:34
on October 13 and final tag-cleanup validation at approximately 00:26 on October 14.
The requested `1013` suffix is retained even though the session crossed midnight.

User requests below are quoted from the available conversation, in order.
Earlier assistant replies are **summaries reconstructed from the retained session
record, current source, and saved validation reports**, not a verbatim transcript.
The available context does not contain every earlier assistant sentence or an
exact timestamp for each message. This document therefore records all recoverable
requests, decisions, implementation outcomes, and abandoned approaches without
inventing missing dialogue. Filesystem modification times are supporting evidence,
not exact times of user messages or proof of when every code line was introduced.

This README is a historical record and current implementation description.
Creating it did not rerun meshing or change pipeline behavior.

## 1. User commands, assistant answers, and resulting decisions

### 1.1 Code access and checking surface holes

User:

> can you read code now?

> now pls tell me whether current pipline which named generate_open_volmesh_stable.py operate surface holes check in step 01?

Assistant answer/outcome summary: inspected the pipeline's surface-loading path.
The subsequent agreed change was to make hole checking and conditional filling an
explicit part of input preparation before Step 01 output and downstream processing.
The exact original inspection answer is not retained here.

User:

> Pls integrate the fill_step01_surface_holes.py into the current pipeline. Before step 01, perform a hole check on the surface mesh in the VTKS. When holes are found, use the function to fill them and then load. If no holes are found, directly load the original mesh. Do you understand what i want to do, tell me for confirmation

> yes, pls modify the code

Assistant answer/outcome summary: integrated the hole-filling helper into the input
preparation path. The implementation reads a surface into memory to inspect its
topology, fills holes when required, and passes the checked surface onward. It
does not overwrite the original `vtks` files. Closed surfaces bypass filling.

### 1.2 Testing subjects 1–3 at ED and ES

User:

> now pls use pipline test on ED and ES of subject_1, subject_2 and subject_3

Assistant answer/outcome summary: ran the then-active pipeline on all six cases.
The initial report recorded all six completing and the original VTKS remaining
unchanged. At that time, stage numbers and the final active stage differed from
the current ten-stage pipeline.

Evidence: [initial run summary](template_trash/pipeline_test_20261013_211542/run_summary.json)
and [initial validation](template_trash/pipeline_test_20261013_211542/validation_summary.json).

### 1.3 Diagnosing incomplete RVepi movement in the former Step 05

User:

> now i check results in paraview and find the results (RV_openbase_epi_moved.vtk) in step 05 have little issues, pls explain the methods of step 05 current pipline used

> From my own observation, I found that some points of the open-base RVepi did not move completely. What could be the possible reasons for this?

> Is it because of the movement based on points that caused this situation? Would it be possible to solve this problem by changing to triangle based movement?

Assistant answer/outcome summary: discussed the limitations of a vertex-distance
method. A nearest sampled vertex is not necessarily the closest location on a
triangle surface; vertex checks can miss insufficient separation inside faces.
An unsigned distance also does not establish that RVepi is outside RVendo.
Opening-ring restrictions, smoothing, direction choices, and existing geometry
defects can contribute. These were possible causes, not proof that every observed
point had the same cause. Recommended surface-distance checks and connected,
validated deformation rather than independently translating triangles.

User:

> pls try your better approach

Assistant answer/outcome summary: implemented and tested a triangle-distance trial
in separate output directories. Its strict geometry checks rejected all six trial
cases because of reported input/diagnostic-closure intersections. It was not
accepted as the final default solver. Later intersection confirmation and the
signed solver replaced this trial as the validated approach.

Evidence: [triangle trial summary](template_trash/step05_triangle_trial_20261013_215206/trial_summary.json).
The intersection messages in that report are historical diagnostics; they should
not be confused with the later confirmed-intersection validation results.

### 1.4 Adding MeshFix as the new Step 02

User:

> ok, i understand, so pls add a new step, i call it step 02, which i need to use meshfix to clean the self-intersections of the surface meshes input from step_01, the cleaned outputs need to save into folder "step_02_surf_clean", corresponding file save using old name,  and the old step_02_** should be named step_03_**, and so on, pls understand what i want to do and ask me for confirmation

> yes, pls modify the code

Assistant answer/outcome summary: inserted MeshFix after Step 01, kept each surface
basename, routed downstream stages through the cleaned copies, and shifted the
later active stage numbers by one. RVepi movement consequently became Step 06.
The confirmed validation covered 24 input surfaces across six subject/frame cases.

Evidence: [confirmed Step 02 validation](template_trash/step02_meshfix_validation_20261013_220835_confirmed/validation_summary.json).
This historical report used the vertex solver, which was still the default then.

### 1.5 Fixing RVepi vertices inside RVendo in Step 06

User:

> ok, now pls re-analyse the current step_06 mov RVepi, i observe that some vertex on RVepi inside the RVendo, what may cause this issue? And by your advice, recommand change move RVepi methods or add fucntion to fix them? give a plan

> make this plan easy to understand

> ok, pls try this plan

Assistant answer/outcome summary: recommended changing the movement method itself:
determine the correct side of RVendo, move violating parts outward as a connected
surface, include the opening ring, and check the entire surface after movement.
Implemented the signed triangle-distance solver with a cleaned, closed Step 02
RVendo as the inside/outside reference. All six saved Step 06 results passed the
clearance, containment, and self-intersection checks. This became the default.

Evidence: [signed solver runs](template_trash/step06_signed_validation/run_summary.json) and
[saved-mesh validation](template_trash/step06_signed_validation/saved_mesh_validation.json).

### 1.6 Disabling the old Step 08 point filter

User:

> now, the current step 08 is not need, pls disable related code or remove them, i will add new step 08

Assistant answer/outcome summary: removed the old point-filter stage from active
execution. The old `step_08_whole_points_surf` helper and related options remain
as legacy code; they are not called by the active pipeline.

### 1.7 Adding TetGen volume generation, initially called Step 08

User:

> Now i need to add a new step 8, which i need to used LVRV_surf.vtk from step 7 as input, using tetgen to generate volumetric mesh, save vol mesh named subject_x_{timeframe}_volmesh_notag.vtk under folder "step_08_volmesh_generate", give me a plan

> yes, pls modify the code

Assistant answer/outcome summary: added untagged TetGen volume generation using the
saved combined surface, with checks on input validity, tetrahedra, preserved
boundary, and saved output. Double precision was used through the Boolean surface
and export path to preserve small intersection triangles. Direct generation still
encountered genuine surface intersections in subject 3 ES; four offending triangle
pairs were recorded. This motivated the following explicit surface-cleaning stage.

Evidence: [subject 3 ES defect record](template_trash/step08_tetgen_validation/subject_3_ES_input_defect.json).

### 1.8 Making Step 08 clean the combined surface; moving TetGen to Step 09

User:

> i found a issue, when operate the volumetric mesh generation, the quality of LVRV_surf.vtk is important, so make a new step 08, using Meshfix to clean the intersections on the LVRV_surf.vtk, save output under folder "step_08_LVRV_surf_clean" and use the cleaned vtk to generate vol mesh, the old "step_08_volmesh_generate" changed to be "step_09_volmesh_generate"

Assistant answer/outcome summary: inserted MeshFix for the saved Step 07 combined
surface, saved the repaired copy under the requested Step 08 folder, and made
Step 09 read that exact saved output. The Step 07 source remained unchanged.
All six cleaned surfaces and generated volumes passed validation; subject 3 ES
went from four confirmed intersection pairs to zero.

Evidence: [Step 08/09 validation](template_trash/step08_meshfix_step09_validation/validation_summary.json).

### 1.9 Adding initial LV/RV tags in Step 10

User:

> now i want to add step 10, i need to seperate tags for subject_x_{timeframe}_volmesh_notag.vtk in step 9, LV tag with 1, and RV tag with 0, the final mesh need to save into folder "step_10_final_mesh", pls give me a plan

> ok, pls add the step 10

Assistant answer/outcome summary: added integer cell array `elemTag`, with LV=1 and
RV=0. A tetrahedron initially receives LV if its centroid or at least three of its
four vertices are inside the cleaned, closed Step 02 LVepi; otherwise it receives
RV. Mesh coordinates, tetrahedron connectivity, and order are preserved.
The first implementation reported disconnected groups without changing them.
All six tagging cases passed, with 24 active regression tests passing at that point.

Evidence: [initial tagging results](template_trash/step10_tagging_validation/run_summary.json)
and [initial tagging audit](template_trash/step10_tagging_validation/validation_summary.json).

### 1.10 Adding neighborhood cleanup after initial tagging

User:

> From my observation of the results, I noticed that in the final mesh of some subjects, there are small disconnected areas in the left and right tags. Therefore, please add an additional step 10 for me. In this step, the neighborhood analysis method is employed to eliminate the disconnected erroneous tags. pls give me a plan

Assistant plan summary:

1. Add cleanup inside existing Step 10, after initial tagging; do not renumber stages.
2. Treat tetrahedra as neighbors only when they share an entire triangular face.
3. Find connected LV and RV groups and distinguish their main regions from islands.
4. Correct small islands using surrounding-region support, measured by shared-face area.
5. Change one group at a time, recompute connectivity, and repeat until no eligible
   islands remain; retain and report large or unsupported groups.
6. Save the initial mesh for comparison and add `tagChanged` to the final mesh.
7. Validate all six cases and verify geometry, connectivity, and element order.

User:

> yes, pls modify the code

Assistant final answer, as delivered:

> Added neighborhood cleanup to Step 10, enabled by default. It corrects small disconnected tag groups using shared triangular faces and preserves mesh geometry.
>
> Tested subjects 1–3 at ED and ES: **all six now have one connected LV region and one connected RV region**. All 29 active pipeline tests passed.
>
> Outputs are in each subject’s `volmesh_pipline_signed_clearance/step_10_final_mesh/`:
>
> - `subject_x_{timeframe}_volmesh.vtk` — cleaned tags.
> - `subject_x_{timeframe}_volmesh_before_tag_cleanup.vtk` — comparison copy.
>
> In ParaView, color by **`tagChanged`** to highlight corrected cells (`1`). LV remains `elemTag=1`; RV remains `elemTag=0`.

### 1.11 Requesting this modification record

User first requested a README of the modification, interrupted the initial
documentation attempt, and then clarified the intended scope:

> ok, pls generate a readme_{key info of modification}_1013.md, which record my command and your answer and your final used modification, what you point but finally not used, all record in this file

> ok, pls generate a readme_{key info of modification}_1013.md, which record my command and your answer and your final used modification in recent 12h, what you point but finally not used, all record in this file

Assistant response: create this single README for the recent development session,
cross-check it against current source and saved reports, and distinguish exact
quoted requests from reconstructed earlier answer summaries.

## 2. Final active pipeline and numbering

The current entry point uses `process_frame_new_pipeline`. The older
`process_frame` and various numbered legacy helpers still exist; their names do
not define the active workflow.

| Step | Folder | Final purpose |
| --- | --- | --- |
| 01 | `step_01_load_surf` | Check/fill input holes in memory and save four checked surface copies. |
| 02 | `step_02_surf_clean` | MeshFix the four Step 01 surfaces; preserve basenames. |
| 03 | `step_03_base_detection` | Detect the subject's base using checked and cleaned inputs. |
| 04 | `step_04_LV_openbase` | Construct LV open-base surfaces. |
| 05 | `step_05_RV_openbase` | Construct initial RV open-base surfaces. |
| 06 | `step_06_move_RVepi` | Signed, connected RVepi displacement relative to fixed RVendo. |
| 07 | `step_07_reconstruct_annulus` | Reconstruct annuli and form the combined surface with Manifold3D union. |
| 08 | `step_08_LVRV_surf_clean` | MeshFix the saved Step 07 `LVRV_surf.vtk`. |
| 09 | `step_09_volmesh_generate` | TetGen from the saved Step 08 surface; save untagged tetrahedra. |
| 10 | `step_10_final_mesh` | Initial LV/RV classification followed by small-island tag cleanup. |

Numbering history:

- Inserting Step 02 shifted old Steps 02 onward by one; old Step 05 movement is
  now Step 06.
- The old point-filter Step 08 was disabled.
- TetGen was briefly the replacement Step 08, then became Step 09 after combined
  surface cleaning was inserted as Step 08.
- Neighborhood cleanup is a sub-operation of Step 10, not a new Step 11.

## 3. Final methods actually used

### 3.1 Input hole checking and filling

- `load_checked_surfaces` applies `check_and_fill_loaded_surface` to LVendo,
  LVepi, RV (the RV endocardial input), and RVepi.
- Boundary edges identify open holes. Closed surfaces do not need filling.
- `fill_step01_surface_holes.fill_surface` uses `vtkFillHolesFilter` and validates
  the added caps. Original vertex coordinates/order and original triangles are
  checked for preservation during filling.
- The checked surfaces are used before downstream base detection and processing.
- Original `vtks` are read-only inputs to this workflow; Step 01 stores copies.
- Hole filling is not a general self-intersection repair. That is Step 02's role.

### 3.2 MeshFix in Steps 02 and 08

- Shared helper: `clean_step02_surfaces.clean_surface_meshfix`.
- Main call: `repair(joincomp=False, remove_smallest_components=False)`.
- If confirmed intersections remain, run additional cleaning, intersection
  removal, and degeneracy removal, followed by validation.
- Require no free edges, nonmanifold edges, degenerate triangles, inconsistent
  orientation edges, or confirmed self-intersections in accepted closed outputs.
- Open3D supplies candidate intersection pairs; FCL double-precision triangle
  distance confirms them. Candidate pairs alone are not the final verdict.
- Step 02 assigns each repaired individual surface its chamber tag.
- Step 08 clears inherited surface arrays because MeshFix can change connectivity
  and invalidate the old per-face labels. Volume tags are assigned later.
- MeshFix may change positions and connectivity. The exact preservation guarantee
  of Step 10 must not be applied to these repair stages.

### 3.3 RVepi movement in Step 06

Default option: `--step6-clearance-method signed`.

1. Keep the actual open RVendo fixed.
2. Use cleaned closed Step 02 RVendo for inside/outside classification.
3. Measure clearance against real open RVendo triangles, not an artificial cap.
4. Compute signed-distance corrections at vertices and triangle probes, including
   centroids and edge midpoints; use triangle-level clearance checks.
5. Spread corrections through a connected smoothed vector field, including the
   opening ring. This still updates shared mesh vertices; triangles are not
   translated independently.
6. Limit each displacement and backtrack if a candidate flips triangles or creates
   self-intersections. Basal boundary smoothing also has geometry guards.
7. Require whole-surface clearance of at least the requested 3 mm, correct exterior
   placement, and no RVepi self-intersections. Revalidate the saved mesh.

The default padding is 0.05 mm, so the correction target is normally 3.05 mm;
acceptance uses the requested 3 mm minimum. Surface connectivity and point IDs
are preserved by this movement stage. Diagnostics include displacement,
signed distance, inside flags, and opening/transition-band membership.

### 3.4 Combined surface and volume generation

- Step 07 retains the point-only `LVRV.vtk` diagnostic output, but the meshing
  input is the triangle surface `LVRV_surf.vtk`.
- The Boolean union uses Manifold3D `Mesh64` / `to_mesh64`. Relevant surface
  exports use double coordinates and 17-digit precision to preserve tiny faces.
- Step 08 writes a separate repaired binary VTK surface and validates readback.
- Step 09 reads that saved Step 08 file, not an unrecorded repaired object.
- TetGen uses a linear tetrahedral PLC mesh: `plc=True`, `quality=True`,
  `minratio=1.2`, `mindihedral=5.0`, `nobisect=True`, `regionattrib=False`, `order=1`.
  An optional positive maximum volume enables the corresponding size constraint.
- Check positive tetrahedron volumes, valid connectivity, duplicate cells,
  face topology, boundary agreement with Step 08, and total volume agreement.
- Save `subject_x_{timeframe}_volmesh_notag.vtk` without LV/RV tag arrays.

### 3.5 Step 10 initial classification

For each Step 09 tetrahedron, using cleaned closed Step 02 LVepi:

```text
if centroid is inside LVepi OR at least 3 of 4 vertices are inside LVepi:
    elemTag = 1  # LV
else:
    elemTag = 0  # RV
```

The current active rule fixes the LV/RV values and the three-vertex threshold.
LV takes priority under this rule. It is a geometric classification, not a claim
of a separately measured anatomical septal interface. Centroid/vertex
disagreements and mixed vertex membership are reported for inspection.

### 3.6 Step 10 final neighborhood cleanup

Implemented in `tag_step10_volume.cleanup_tag_islands`:

1. Build tetrahedron adjacency from identical complete triangular faces. A shared
   vertex or edge is insufficient. Reject faces with more than two owners.
2. Compute same-tag connected components. Protect the initially largest LV and RV
   components using stable seed cells; their growing connected regions stay protected.
3. Fix a size threshold for each tag from its initial largest component:

   ```text
   threshold = max(min_cells, ceil(initial_largest_component_cells * min_fraction))
   defaults: min_cells = 32; min_fraction = 0.0005
   ```

4. A non-main component is eligible only if its size is **strictly less** than
   that threshold and it shares positive face area with the opposite protected
   main region. The threshold is not a fixed 32 cells for these large meshes.
5. Accumulate the supporting shared-face area. With only two tags, the possible
   replacement label is the opposite tag; there is no multi-label majority vote.
6. Process an eligible component, starting with the smallest, change all its
   cells together, then recompute connectivity before selecting the next group.
7. Every accepted change merges connected components and reduces their total
   count. Cells already in protected main regions are never removed. This avoids
   adjacent islands repeatedly swapping labels.
8. Stop when no eligible groups remain. Large groups and groups without a shared
   face to the opposite main region remain unchanged and are listed with reasons.

The implementation is stricter than an unrestricted neighborhood vote: it requires
support from the opposite **main** region. Shared-face area measures that support;
it does not rank competing labels in this binary case. No rule forces all future
subjects to have exactly one component per tag. That outcome was achieved on the
six validated cases; unsupported or large islands can remain on other inputs.

Runtime options:

| Option | Effect in current Step 10 |
| --- | --- |
| `--tag-island-cleanup` | Enabled by default. |
| `--no-tag-island-cleanup` | Keep initial tags and report groups; comparison output is still saved. |
| `--tag-island-min-cells` | Absolute part of the size threshold; default 32. |
| `--tag-island-min-fraction` | Fractional part; default 0.0005. |
| `--tag-island-cleanup-iterations` | Legacy only; active cleanup runs until no eligible groups remain. |

Step 10 writes both files only after validation of the staged saved meshes:

```text
step_10_final_mesh/
  subject_x_{timeframe}_volmesh_before_tag_cleanup.vtk
  subject_x_{timeframe}_volmesh.vtk
  subject_x_{timeframe}_step10_final_mesh_summary.json
```

The comparison file preserves initial `elemTag`; the final file contains cleaned
`elemTag` and cell array `tagChanged` (1=changed, 0=unchanged). The JSON records
components before/after, changed cell IDs, supporting areas, thresholds, unresolved
groups, input hashes, and serialization checks. Coordinates, tetrahedron
connectivity, cell types, and element order exactly match Step 09 in both files.

## 4. Discussed, tried, superseded, or not used in the final path

| Approach | Final status and reason |
| --- | --- |
| Nearest-vertex movement as the default | Superseded by signed triangle clearance. Still selectable as legacy `vertex`; it does not supply the final solver's whole-surface checks. |
| Earlier normal-only triangle trial | Retained as optional `triangle`, not the default. The historical six-case trial stopped at geometry checks. |
| Artificial basal closure as the final side reference | Used by the older trial; replaced in the signed solver by cleaned closed Step 02 RVendo. |
| Independent movement of each triangle | Not used. Final triangle-based distance checks drive shared vertices through one connected field, preserving mesh connectivity. |
| A frozen opening rim during correction | Not used by the final signed field; the ring participates in displacement. |
| Relying only on a later patch to hide remaining inside points | Not the final strategy. Signed corrections and final validation are integrated into movement. |
| Treating every Open3D intersection candidate as confirmed | Superseded by FCL double-precision confirmation to avoid false-positive conclusions. |
| Removing small surface components automatically during MeshFix | Not enabled: `remove_smallest_components=False`; component joining also remains disabled. |
| Old Step 08 RV-point filtering / whole point-cloud stage | Disabled in the active path at the user's request; helper code and old outputs may remain. |
| TetGen directly from raw Step 07 as the final workflow | Superseded by explicit Step 08 MeshFix, followed by Step 09 TetGen. |
| `step_08_volmesh_generate` as the final volume folder | Superseded by `step_09_volmesh_generate`; historical outputs were not globally deleted. |
| Float32 Boolean/export path | Replaced in the relevant path with double precision to preserve small triangles. |
| Copying repaired combined-surface face tags into the volume | Not used. Repair can change faces; Step 09 is untagged and Step 10 classifies volume cells. |
| Reporting tag islands without changing them | Used in the first Step 10 version, superseded by the subsequently requested cleanup. |
| Legacy simultaneous neighbor-count island relabeling | Not used in active Step 10. Initial classification explicitly disables it; the new saved-volume helper performs sequential cleanup. |
| Fixed two-pass cleanup | Not used by the new cleanup. Connectivity is recomputed after each change until no eligible group remains. |
| Vertex/edge-contact adjacency | Excluded; tetrahedra must share a triangular face. |
| Relabeling every disconnected group regardless of size/support | Not used. Large or unsupported groups remain reported for review. |
| A new Step 11 for neighborhood cleanup | Not used; the approved plan keeps it inside existing Step 10. |
| Separate LV/RV tetrahedralization described in older documentation | Not the current active path. Current workflow meshes the combined solid once, then tags its tetrahedra. |

No unrecorded experiment is implied by this table: some entries are historical
trials, some are superseded implementations, and others clarify how the final
method differs from possibilities raised in the discussion.

## 5. Validation results and limits

### 5.1 Final six-case results

| Subject | Frame | Tetrahedra | LV components before → after | RV components before → after | Cells relabeled |
| --- | --- | ---: | --- | --- | ---: |
| 1 | ED | 365,437 | 1 → 1 | 91 → 1 | 151 |
| 1 | ES | 332,201 | 2 → 1 | 74 → 1 | 124 |
| 2 | ED | 398,484 | 3 → 1 | 90 → 1 | 171 |
| 2 | ES | 363,515 | 3 → 1 | 99 → 1 | 213 |
| 3 | ED | 460,819 | 5 → 1 | 111 → 1 | 277 |
| 3 | ES | 364,054 | 1 → 1 | 76 → 1 | 151 |

All six final cleanup audits passed. There were no unresolved tag components in
these cases. Both tags remained present; saved `elemTag` was int32; `tagChanged`
exactly matched the difference from the saved initial tags. Both tagged files
preserved Step 09 points, cells, cell types, and ordering. Input hashes were unchanged.

Evidence: [cleanup audit](template_trash/step10_cleanup_validation/validation_summary.json)
and [cleanup run log](template_trash/step10_cleanup_validation/run.log).

### 5.2 Earlier validation milestones

- Initial six-case pipeline test: all six completed; original VTKS unchanged.
- Confirmed Step 02 test: all 24 repaired surfaces validated; 14 tests passed at
  that milestone. Original VTKS and Step 01 copies remained unchanged.
- Signed Step 06: all six saved RVepi meshes entirely outside RVendo, no final
  self-intersections, fixed RVendo, preserved topology/point IDs, and certified
  global triangle-to-triangle clearance at least 3 mm.
- Step 08/09: all six cases passed, including repaired subject 3 ES; 21 active
  regression tests passed at that milestone. Saved volume boundaries matched
  cleaned Step 08 surfaces, with positive tetrahedra and volume agreement.
- Initial Step 10: all six passed; 24 active regression tests passed. Islands were
  reported at this stage and were the basis for the later cleanup request.
- Current Step 10: all six passed; **29 active regression tests passed**.

The latest cleanup validation reran Step 10 against existing validated Step 09
meshes; it did not repeat the entire surface-to-volume pipeline. Earlier stage
claims come from their saved reports.

The active regression selection covers hole filling, Step 02 repair, RVepi
clearance, Step 08 cleaning, Step 09 volume generation, and Step 10 tagging/cleanup.
An older `test_step8_annulus_repair.py` suite has nine previously identified errors
referencing unavailable legacy APIs. It was not part of the 29-test active suite;
this record does not claim unrestricted discovery of every historical test passes.

Connectivity cleanup is a topology-based correction of small label islands. It
does not independently establish anatomical accuracy of the LV/RV interface or
improve tetrahedron shape quality. Large/unsupported groups are deliberately
reported rather than silently forced into a label.

## 6. Source files involved

| File | Role in this session's final workflow |
| --- | --- |
| [generate_open_volmesh_stable.py](generate_open_volmesh_stable.py) | Active stage orchestration, numbering, input preparation, movement selection, double-precision export, Steps 08–10 integration and options. |
| [fill_step01_surface_holes.py](fill_step01_surface_holes.py) | Conditional hole-filling helper and topology checks. |
| [clean_step02_surfaces.py](clean_step02_surfaces.py) | Shared MeshFix repair and validation for Steps 02/08. |
| [mesh_intersections.py](mesh_intersections.py) | Confirm candidate intersections using FCL double precision. |
| [rvepi_outward_clearance.py](rvepi_outward_clearance.py) | Signed connected deformation, geometric guards, clearance validation; older trial retained. |
| [open-base helper](../mesh_vol/generate_subject_open_base_max_interface_smooth_boundary_custom_tag_volmesh.py) | Geometry-safe basal boundary smoothing. |
| [generate_step09_volume.py](generate_step09_volume.py) | TetGen generation, volume/boundary checks, validated saved untagged output. |
| [tag_step10_volume.py](tag_step10_volume.py) | Face connectivity, sequential tag cleanup, before/after reports, validated tagged outputs. |
| [Step 10 tests](tests/test_step10_tagging.py) | Classification, geometry preservation, shared-face adjacency, sequential convergence, thresholds, unsupported groups, saved flags, failure preservation. |
| [cleanup validation runner](template_trash/step10_cleanup_validation/run_validation.py) | Six-case Step 10 rerun and saved-artifact audit. |

Other active regression files are listed in the command below. This is a role
inventory, not a claim that all files were newly created in the latest cleanup edit.
Some changes precede the currently visible Git diff; untracked new files also
need to be considered when reviewing the development session.

## 7. Execution commands and output locations

Run commands from `/home/mengao/Code/4Dsegment`. The mesh environment used was
`/home/mengao/anaconda3/envs/morphinet/bin/python`. Thread limits avoid excessive
parallelism in the numerical libraries.

### 7.1 Current full-pipeline rerun command

The following loop expresses the three subject commands recorded for the signed
pipeline in a compact form. Running it **now** executes the current Steps 01–10;
historical executions stopped at the stages active at their respective times.

```bash
for subject in 1 2 3; do
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/mengao/anaconda3/envs/morphinet/bin/python -B \
    model_vol_septum/generate_open_volmesh_stable.py \
    --subject-dir "data/subject_${subject}" \
    --frames ED,ES \
    --pipeline-output-root "data/subject_${subject}/volmesh_pipline_signed_clearance"
done
```

Default output root when no override is given is `data/subject_x/volmesh_pipline`.
The latest six-case validated outputs are under **`volmesh_pipline_signed_clearance`**.
Do not confuse older default-root or triangle-trial outputs with these files.

### 7.2 Latest Step 10-only regeneration and audit

This command was executed for the final cleanup validation:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/mengao/anaconda3/envs/morphinet/bin/python -B \
  model_vol_septum/step10_cleanup_validation/run_validation.py \
  > model_vol_septum/step10_cleanup_validation/run.log 2>&1
```

The runner reads existing Step 09 volumes and Step 02 LVepi references for all six
cases, regenerates Step 10, reads both saved files back, checks invariants, and
writes `validation_summary.json`.

### 7.3 Latest active regression suite

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/mengao/anaconda3/envs/morphinet/bin/python -B -m unittest \
  model_vol_septum.tests.test_step1_hole_filling \
  model_vol_septum.tests.test_step2_surface_cleaning \
  model_vol_septum.tests.test_step5_outward_clearance \
  model_vol_septum.tests.test_step8_lvrv_cleaning \
  model_vol_septum.tests.test_step9_volume_generation \
  model_vol_septum.tests.test_step10_tagging -v \
  > model_vol_septum/step10_cleanup_validation/regression_tests.log 2>&1
```

Result: `Ran 29 tests ... OK`. The historical `test_step5_outward_clearance`
filename remains even though movement is now Step 06.

During cleanup development, an adjacency test initially expected ordered cell
pairs even though face neighbors are undirected. The test was corrected to compare
unordered pairs. It then passed, together with the additional repeated-merge test.
This was a test expectation issue, not a failure of the six saved meshes.

### 7.4 Historical command provenance

Exact recorded subject/frame arguments and exit codes are available in:

- [initial six-case commands](template_trash/pipeline_test_20261013_211542/run_summary.json);
- [triangle trial commands](template_trash/step05_triangle_trial_20261013_215206/trial_summary.json),
  using `--step5-clearance-method triangle` and `volmesh_pipline_triangle_trial`;
- [confirmed MeshFix runs](template_trash/step02_meshfix_validation_20261013_220835_confirmed/run_summary.json);
- [signed movement commands](template_trash/step06_signed_validation/run_summary.json);
- [Step 08/09 runs](template_trash/step08_meshfix_step09_validation/run_summary.json).

This file records the substantive commands and points to their detailed logs;
it is not a reconstructed shell transcript of every source inspection command.

## 8. Inspecting the final result in ParaView

For each subject and frame, open:

```text
data/subject_x/volmesh_pipline_signed_clearance/step_10_final_mesh/
  subject_x_{timeframe}_volmesh.vtk
```

- Color by **cell** array `elemTag`: LV=1, RV=0.
- Color or threshold by **cell** array `tagChanged`: 1 identifies corrected cells.
- Open `subject_x_{timeframe}_volmesh_before_tag_cleanup.vtk` alongside it to compare
  initial labels with cleaned labels on identical geometry.
- Consult the corresponding Step 10 summary for changed cell IDs and any unresolved
  groups. The six validated cases have none.

Older experiment directories and disabled-stage outputs may still exist. Their
presence does not mean those stages or methods are part of the current pipeline.
