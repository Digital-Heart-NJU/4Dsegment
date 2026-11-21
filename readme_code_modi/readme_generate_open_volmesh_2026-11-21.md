# generate_open_volmesh.py: Conversation and Modification Record

Date: 2026-11-21 (Asia/Shanghai)

Project: `/home/mengao/Code/4Dsegment`

This is a structured record of today's discussion, decisions, changes, commands,
and verification concerning `generate_open_volmesh.py`. It is not a verbatim
transcript or a claim that every proposed improvement has been implemented.
The documentation task itself did not modify Python code or regenerate meshes.

## 1. Files and Scope

- Main pipeline: [generate_open_volmesh.py](../model_vol_septum/generate_open_volmesh.py).
- Batch entry point: [run_pipline.py](../run_pipline.py).
- Runtime: Conda environment `cme-dt-env`.
- Subject outputs: `data/subject_x/volmesh_pipline/`.
- Relevant tests: `model_vol_septum/tests/test_new_pipeline_step08.py`,
  `test_new_pipeline_step09.py`, and `test_run_pipline_uvc.py`.
- The older `generate_open_volmesh_stable.py` is not the volumetric runner target.

Current overall workflow:

```text
vtks/{LVendo,LVepi,RV,RVepi}_{ED,ES}.vtk
  -> Step 01: load/copy original surfaces
  -> Step 02: clean and validate closed surfaces
  -> Step 03: move LVepi and RVepi outward with endocardia fixed
  -> Step 04: construct combined outer envelope and complete inner surfaces
  -> Step 05: clean the combined myocardial surface
  -> Step 06: TetGen volume mesh and initial LV/RV tags
  -> Step 07: subject-level base detection
  -> Step 08: basal opening shell, TetGen, spatial tag transfer
  -> Step 09: tag 1 growth by 10 mm, then tag 0 growth by 12 mm
  -> final_volmesh.vtk
```

LV is `elemTag=1`; RV is `elemTag=0`. The `RV` input filename denotes RVendo.

## 2. Main Requests and Outcomes

| Request or question | Agreed behavior / outcome |
| --- | --- |
| Save Step 03 intermediate results when movement fails. | Diagnostic-only last-accepted and rejected surfaces, fixed endocardium, and a failure JSON are supported. These are not valid final outputs. |
| Explain 18 backtracking trials versus 600 iterations. | Up to 600 movement updates; each update can try 18 step sizes. An unrecoverable update can stop the run much earlier. |
| Enable optional local repair. | `--epi-local-repair=true` enables local displacement damping/smoothing after rejected backtracking or too-small progress. |
| Do not accept tiny movement indefinitely. | Default `--epi-min-accepted-step=0.01`; smaller updates are allowed only if they pass final convergence checks. |
| Publish successful Step 08 frames independently. | ED and ES are isolated and published independently; native TetGen crashes do not kill the retry controller. |
| Handle thin triangles before TetGen. | Tolerance-bounded Manifold3D simplification is available, with topology and geometry checks. |
| Try larger collapse tolerances first. | Original shell first; after an eligible TetGen failure, try `0.01`, `0.001`, `0.0001`, stopping at the first success. |
| Check `STAGE_ARGS['volumetric']`. | CLI option names and values must be separate strings; Python-list syntax inside one string is invalid. |
| Replace percentage-based Step 09 growth. | Exactly two shared-face distance-growth passes: tag 1 by 10 mm, then surviving tag 0 by 12 mm. |
| Regenerate subject_24 Step 09. | ED and ES generated; final and intermediate tags checked against the supplied reference algorithm. |
| Analyze subject_23 failure without edits. | Confirmed Step 03 near-contact/self-intersection during RVepi movement. No collision-aware repair was implemented after that analysis. |

## 3. Step 03: Movement, Backtracking, and Local Repair

### 3.1 Existing movement requirements

`move_closed_epi()` moves a closed epicardium while keeping its corresponding
endocardium fixed. It checks outward displacement, surface clearance, containment,
closed topology, and absence of detected self-intersections. The default required
clearance is 3 mm, with a 0.1 mm movement target padding.

There is no open basal ring at this stage. The Step 03 folder retains its old
name, `step_03_move_RVepi`, although the stage processes RVepi and LVepi separately.

### 3.2 Meaning of the iteration and movement parameters

```python
max_iterations = 600
target = minimum + clearance_padding
max_step = min(1.0, max(0.5 * median_original_edge_length, 1e-3))
```

- `max_step` is a per-update displacement cap in millimetres, not total movement.
- Vertices are updated simultaneously, but their directions and amounts differ.
- The first backtracking trial uses the full proposed displacement.
- The 18 trials use factors `1, 1/2, ..., 1/2**17`.
- The maximum accepted movement can decrease because the complete proposed
  displacement is halved to avoid folded or intersecting triangles.
- The algorithm does not promise to reach iteration 600: an update with no
  acceptable movement can terminate earlier.

With `--epi-min-accepted-step=0.01`, the threshold applies to the maximum vertex
movement in an update. It does not require every vertex to move at least 0.01 mm.
An intersection-free update below the threshold is accepted only if it satisfies
the final clearance and geometry requirements.

### 3.3 Optional local repair

```bash
--epi-local-repair=true --epi-min-accepted-step=0.01
```

The function `_repair_closed_epi_displacement()` identifies problematic faces,
selects their vertices and local neighbors, and damps/smooths their proposed
displacements. It allows up to six repair attempts per event. It retains point
numbering and triangle connectivity.

Each candidate is checked for folded triangles, self-intersections, outward
movement, and sufficient progress. It does not perform MeshFix remeshing,
triangle reconnection, or explicit collision-separation optimization.

The reported global maximum step may remain unchanged during local repair:
vertices outside the repaired patch can still move by the original maximum.

### 3.4 Failure diagnostics

Failure artifacts are saved below:

```text
step_03_move_RVepi/failed_attempts/subject_x_FRAME_CHAMBEREPI/TIMESTAMP/
  RVepi_last_accepted.vtk
  RVepi_rejected_trial.vtk
  RVendo_fixed.vtk
  failure_summary.json
```

Names vary with chamber. The surfaces carry diagnostic fields including
displacement and signed endocardial distance. A last-accepted surface can be
self-intersection-free while still failing the required 3 mm clearance.

The independent publication/skip policy added for Step 08 does not make Step 03
independent per frame. In the current Step 03 code, an ED exception is re-raised,
so ES is not reached in that subject's invocation. The runner then continues to
the next subject, if any.

## 4. Step 08: Thin-Triangle Cleanup and Automatic Fallback

### 4.1 Opening behavior retained

Step 08 uses the saved Step 07 apex-to-base direction. A full endocardial
translation defines reference opening rims, which are mapped back to split the
original endocardia. Basal patches are extended and subtracted from the original
myocardial solid. The resulting shell is tetrahedralized with TetGen.

This is remeshing, not preservation of original tetrahedral connectivity.
Step 06 tags are transferred spatially using containing/nearest source cells;
`referenceCellId` records that source-cell lookup.

### 4.2 Automatic attempt sequence

```text
Build and cache original open-base shell
  -> try original shell with TetGen, without collapse
  -> if successful: publish that frame; stop retrying
  -> if TetGen failed AND original shell has triangles below 0.1 degree:
       -> collapse original shell with relative tolerance 0.01
       -> validate geometry/topology, then TetGen
       -> if unsuccessful, retry original shell with 0.001
       -> if unsuccessful, retry original shell with 0.0001
       -> stop immediately at the first validated meshing success
  -> if all eligible attempts fail: retain existing final outputs and skip frame
```

- Each retry starts from the same cached original shell, not a previous cleanup.
- Tolerances must be positive and finite; the list is deduplicated and sorted
  largest first.
- Missing/stale inputs, shell-construction failures, or post-meshing output
  failures do not trigger this thin-triangle fallback merely because they failed.
- Each attempt runs in a child process, including the original attempt.
- A native TetGen abort is recorded as a failed attempt; the parent can continue.
- ED is published before ES starts. ES failure does not discard successful ED.
- When Step 09 follows Step 08 in the same invocation, only successful frames
  are forwarded. A standalone Step 09 invocation uses the saved files requested.
- If all requested Step 08 frames fail, the main command returns an error.

### 4.3 What the tolerance means

```text
tolerance_mm = relative_tolerance * original_shell_median_edge_mm
```

`collapse_open_shell_short_edges()` uses Manifold3D tolerance-bounded
simplification. This is not unconditional deletion of thin triangles and does
not guarantee that every short edge or sliver disappears.

Checks include closed clean topology, detected self-intersections, component
count, Euler characteristic, retained anatomical source roles, and displacement
of both opening rims. Surface deviation is sampled bidirectionally using vertices
and triangle centroids. The acceptance bound is:

```python
max(tolerance_mm * 1.01, original_shell.length * 1e-10)
```

This is a sampled geometric guard, not a proof of an exact continuous Hausdorff
bound. A larger tolerance can remove more triangles but can still be rejected
before TetGen if the resulting geometry exceeds the bound.

TetGen settings were retained: `plc=True`, `quality=True`, `nobisect=True`,
`order=1`, `minratio=1.2`, `mindihedral=5`, with no maximum volume unless supplied.

### 4.4 subject_24 results

| Frame | Attempt outcomes | Points | Tetrahedra | Step 08 tag 0 / tag 1 |
| --- | --- | ---: | ---: | ---: |
| ED | Original succeeds; no collapse | 62,031 | 292,959 | 118,837 / 174,122 |
| ES | Original aborts; 0.01 rejected; 0.001 succeeds; 0.0001 not tried | 39,597 | 188,030 | 64,262 / 123,768 |

For ES, the original shell had 19,891 points and 39,778 triangles. The successful
0.001 cleanup produced 19,032 points and 38,060 triangles. Triangles with minimum
angle below 0.1 degree decreased from 361 to 247; they were not all eliminated.

- Median original edge: 1.3824995468 mm.
- Actual 0.001 tolerance: 0.0013824995 mm.
- Maximum sampled surface deviation: 0.0012769095 mm.
- LV opening-rim deviation: approximately 7.16e-15 mm.
- RV opening-rim deviation: approximately 5.31e-10 mm.
- Both generated volumes were checked for positive tetrahedral volumes and one
  overall shared-face-connected component. Tiny elements remain; this is not a
  comprehensive element-quality certification.

The ES original attempt failed inside native TetGen with an abort. The 0.01
candidate was rejected by the surface/opening-rim displacement guard before
TetGen, not because a higher tolerance is syntactically invalid.

### 4.5 Outputs and evidence

Each successful frame retains ten main VTK files:

```text
subject_x_FRAME_openbase_volmesh.vtk
subject_x_FRAME_removed_myocardium_raw_diagnostic.vtk
subject_x_FRAME_LV_endo_basal_patch.vtk
subject_x_FRAME_LV_endo_basal_patch_displaced.vtk
subject_x_FRAME_LV_mapped_endo_rim.vtk
subject_x_FRAME_LV_opening_rim.vtk
subject_x_FRAME_RV_endo_basal_patch.vtk
subject_x_FRAME_RV_endo_basal_patch_displaced.vtk
subject_x_FRAME_RV_mapped_endo_rim.vtk
subject_x_FRAME_RV_opening_rim.vtk
```

The removed-myocardium diagnostic is a raw Boolean triangle surface, not the
tetrahedral input. Additional logs/cache/diagnostics use subfolders `run_logs`,
`shell_diagnostics`, `failed_attempts`, and `previous_outputs` when applicable.
Existing frame outputs are archived only when a successful replacement is ready.

The original 22:16 run discussed in the conversation was subsequently replaced
by another run. At documentation time, the current saved success reports are:

- [ED Step 08 report](../data/subject_24/volmesh_pipline/step_08_LV_RV_openbase/run_logs/subject_24_ED_20261121_224321_260335_result.json).
- [ES Step 08 report](../data/subject_24/volmesh_pipline/step_08_LV_RV_openbase/run_logs/subject_24_ES_20261121_224329_813603_result.json).

They confirm the same attempt outcomes and tetrahedron counts recorded above.

## 5. Step 09: Replace Percentage Rules with Distance Growth

### 5.1 Removed behavior

The previous Step 09 absorbed small connected tag components using a percentage
of the original largest same-tag component's volume. It protected both initial
main regions and could repeat passes until no more eligible islands remained.

That active Step 09 behavior has been replaced. `--island-volume-percent` is no
longer the CLI parameter for this stage. No protected-main-region or percentage
criterion is applied by the new method.

### 5.2 Implemented distance rule

1. Load the original saved Step 08 tetrahedral mesh and its `elemTag` array.
2. Connect tetrahedra only across a complete shared triangular face.
3. Weight each graph edge by Euclidean distance between the two cell centres.
4. Seed all current tag-1 cells simultaneously. Use multi-source Dijkstra to
   relabel reachable tag-0 cells within 10 mm as tag 1.
5. Seed all tag-0 cells remaining after pass 1. Start a fresh Dijkstra distance
   budget and relabel reachable tag-1 cells within 12 mm as tag 0.
6. Stop after those two passes. Do not repeat cycles or apply extra cleanup.

Coordinates are assumed to be millimetres. The distance is a graph shortest path
through tetrahedron centres, not an exact continuous interface displacement.
Edge-only and vertex-only contacts do not permit growth.

All current matching cells are seeds, not just the largest component. A tag-0
region erased by pass 1 cannot independently seed pass 2. If no tag-0 cells
remain at all, pass 2 makes no changes and a warning is recorded. A zero distance
disables its pass.

Functions added/replaced in `generate_open_volmesh.py`:

- `_validate_growth_distances()`.
- `_tag_distance_graph()`.
- `_grow_tag_distance()`.
- `_tag_distance_statistics()`.
- `correct_openbase_tags(opened, tag1_mm=10., tag0_mm=12.)`.
- `step_09_final_mesh(..., tag1_mm=10., tag0_mm=12.)`.

The graph reuses the existing `shared_face_neighbors()` helper from
`tag_step10_volume.py`. SciPy supplies sparse adjacency and Dijkstra; no new
external meshing script was introduced for this change.

### 5.3 Geometry, format, and diagnostic arrays

Step 09 does not move points, remove tetrahedra, reorder cells, or remesh.
Existing arrays are retained. Both meshes use the existing lossless ASCII
`_write_step09_mesh()` writer, and saved geometry/arrays are checked after reload.

| Array | Meaning |
| --- | --- |
| `elemTag` | Current labels; final labels in the final mesh. |
| `elemTag_step08` | Original Step 08 labels. |
| `elemTag_after_tag1_growth` | Labels after the first pass, stored in the final mesh. |
| `grewTag1` | Cell changed from 0 to 1 in pass 1, even if later reverted. |
| `grewTag0` | Cell changed from 1 to 0 in pass 2. |
| `tagChanged` | Final label differs from its input label. |
| `growthPass` | Last modifying pass: 0 untouched, 1 tag-1 growth, 2 tag-0 growth. |

A cell can have `growthPass=2` but `tagChanged=0` when pass 2 restores its original
tag. This is expected and differs from the old no-reversal component strategy.

### 5.4 subject_24 generated results

| Measurement | ED | ES |
| --- | ---: | ---: |
| Tetrahedra, unchanged | 292,959 | 188,030 |
| Pass 1: 0 to 1 changes | 30,258 | 20,708 |
| Pass 2: 1 to 0 changes | 43,074 | 30,721 |
| Changed during both passes | 26,904 | 19,921 |
| Final changes relative to Step 08 | 19,524 | 11,587 |
| Final 0 to 1 changes | 3,354 | 787 |
| Final 1 to 0 changes | 16,170 | 10,800 |
| Final tag-0 cells | 131,653 | 74,275 |
| Final tag-1 cells | 161,306 | 113,755 |
| Final tag-0 face-connected components | 1 | 1 |
| Final tag-1 face-connected components | 14 | 13 |

The requested distance rule does not eliminate every separate label component.
After pass 1, tag 1 had one component in both frames; pass 2 left 14 and 13
tag-1 components respectively. No unrequested third pass was added.

Independent verification reproduced the user's reference face-graph/Dijkstra
algorithm and matched both saved tag arrays. It also checked exact geometry,
original arrays, diagnostic arrays, and unchanged Step 08 input hashes.
At documentation time, the current Step 08 files still match the source hashes
recorded in both Step 09 summaries, and both final files match their output hashes.

Outputs are under `data/subject_24/volmesh_pipline/step_09_final_mesh/`:

```text
subject_24_ED_final_volmesh.vtk
subject_24_ED_final_volmesh_after_tag1.vtk
subject_24_ED_step09_tag_correction_summary.json
subject_24_ES_final_volmesh.vtk
subject_24_ES_final_volmesh_after_tag1.vtk
subject_24_ES_step09_tag_correction_summary.json
```

- [ED final mesh](../data/subject_24/volmesh_pipline/step_09_final_mesh/subject_24_ED_final_volmesh.vtk).
- [ES final mesh](../data/subject_24/volmesh_pipline/step_09_final_mesh/subject_24_ES_final_volmesh.vtk).
- [ED tag report](../data/subject_24/volmesh_pipline/step_09_final_mesh/subject_24_ED_step09_tag_correction_summary.json).
- [ES tag report](../data/subject_24/volmesh_pipline/step_09_final_mesh/subject_24_ES_step09_tag_correction_summary.json).

## 6. Commands and Runner Configuration

Run commands from the project root:

```bash
cd /home/mengao/Code/4Dsegment
conda activate cme-dt-env
```

### 6.1 Step 08 only: existing Step 07 inputs

```bash
python model_vol_septum/generate_open_volmesh.py \
  --subject-dir data/subject_24 \
  --frames ED,ES \
  --start-step 8 --stop-after 8 \
  --step8-short-edge-collapse=true \
  --step8-collapse-tolerances 0.01 0.001 0.0001 \
  --overwrite
```

The default retry list already contains these values. Set
`--step8-short-edge-collapse=false` to disable fallback. The legacy
`--step8-collapse-relative-tolerance VALUE` remains a single-candidate override;
it is mutually exclusive with `--step8-collapse-tolerances` and still follows
the original-shell-first behavior.

### 6.2 Step 09 only: existing Step 08 inputs

```bash
python model_vol_septum/generate_open_volmesh.py \
  --subject-dir data/subject_24 \
  --frames ED,ES \
  --start-step 9 --stop-after 9 \
  --tag1-mm 10 --tag0-mm 12 \
  --overwrite
```

The actual generation also used this equivalent Conda launcher:

```bash
conda run --no-capture-output -n cme-dt-env python -B -u \
  model_vol_septum/generate_open_volmesh.py \
  --subject-dir data/subject_24 --frames ED,ES \
  --start-step 9 --stop-after 9 --tag1-mm 10 --tag0-mm 12 --overwrite
```

### 6.3 Correct `STAGE_ARGS['volumetric']`

```python
"volumetric": [
    "--frames", "ED,ES",
    "--overwrite",
    "--epi-local-repair=true",
    "--epi-min-accepted-step=0.01",
    "--step8-short-edge-collapse=true",
    "--step8-collapse-tolerances", "0.01", "0.001", "0.0001",
    "--tag1-mm", "10",
    "--tag0-mm", "12",
],
```

The invalid form discussed was:

```python
"--step8-collapse-tolerances [.01, .001, .0001]"
```

`subprocess.run()` receives a list of argument tokens, not a shell command. Each
option/value must be supplied as its own string as shown above. The invalid form
was confirmed to produce an argparse error; the corrected form was accepted
with actual geometry operations mocked during that check.

```bash
python run_pipline.py --request volumetric --subject 24
```

The configured runner invokes Steps 01-09 for ED and ES, not Step 09 alone.
For a Step-09-only runner invocation, temporarily add these tokens to the
volumetric list:

```python
"--start-step", "9", "--stop-after", "9",
```

The runner reads stage-specific settings from `STAGE_ARGS`; arbitrary generator
flags are not automatically accepted by `run_pipline.py` itself. Its example
multi-subject syntax is `--subject 11-18`.

## 7. subject_23 ED Failure: Analysis Only

### 7.1 Reported failure

```text
Step03 RVepi iteration 7
accepted maximum step 0.00133129 mm is below 0.01 mm before convergence
local repair also failed after 6 attempts
no final output published
```

The saved JSON records six completed updates, a 3 mm required clearance,
3.1 mm movement target, and nine backtracking halvings:

```text
0.6816200069 / 2**9 = 0.0013312891 mm
```

The first intersection-free backtracked update was too small under the current
progress rule. Repair candidates then retained the same self-intersection pairs:

```text
[12850, 12857]
[12850, 12858]
```

All six repair attempts reported no folded triangles and outward movement,
but these face intersections remained. Thus the failure was not a 600-iteration
limit or a TetGen failure. Step 08 tolerance changes and Step 09 tag distances
cannot correct a failure that occurs before those stages are reached.

### 7.2 Measurements from the saved surfaces

| Surface state | Minimum RVepi-to-RVendo surface gap | Gap between the problematic RVepi triangles |
| --- | ---: | ---: |
| Original Step 02 RVepi | 0 mm | 0.0060203392 mm |
| Last accepted RVepi | 1.7339150873 mm | approximately 6.68e-7 mm |
| Rejected repair trial | 1.8145489252 mm | 0 mm; detected intersection |

The last accepted surface had no detected self-intersections but had not reached
3 mm clearance. Two nonadjacent patches were almost touching. Their outward
movement brought them into collision before adequate wall clearance was obtained.

The original Step 02 RVepi passed validation: 14,054 points, 28,104 triangles,
zero free edges, nonmanifold edges, orientation inconsistencies, degenerate
triangles, or detected self-intersections. However, validity before movement
does not guarantee collision-free outward deformation.

The three implicated original faces had minimum angles approximately 32.4,
34.2, and 15.8 degrees. Although the whole input contains 327 triangles below
1 degree, these particular colliding faces are not extreme slivers. Thin
triangles elsewhere are not established as the direct cause of this event.

### 7.3 Explanation and unimplemented recommendation

Local normal-outward motion does not guarantee that distinct surface patches
separate from each other. The existing local repair changes the next movement
field only; it does not undo accumulated deformation or explicitly resolve
collisions. Global backtracking also slows unaffected regions when a small patch
limits the entire update.

The 0.01 mm floor detects the stall; it is not the original collision mechanism.
Lowering it might postpone failure. Increasing the outer iteration limit does
not directly address a rejected movement update.

Collision-aware local displacement adjustment was proposed as a future
improvement: handle nearly colliding patches explicitly while allowing safe
regions to progress, retaining endocardial fixation and final clearance checks.
This was an analysis/recommendation only. The user requested no modification,
and no such algorithm was added or tested for subject_23.

Evidence folder:

```text
data/subject_23/volmesh_pipline/step_03_move_RVepi/failed_attempts/
  subject_23_ED_RVepi/20261121_223943_243429/
    RVepi_last_accepted.vtk
    RVepi_rejected_trial.vtk
    RVendo_fixed.vtk
    failure_summary.json
```

## 8. Verification and Remaining Limitations

Focused tests run during today's implementation:

| Test file | Passing tests |
| --- | ---: |
| `test_new_pipeline_step08.py` | 26 |
| `test_new_pipeline_step09.py` | 21 |
| `test_run_pipline_uvc.py` | 3 |
| Total | 50 |

Commands:

```bash
conda run --no-capture-output -n cme-dt-env python -B -m unittest discover \
  -s model_vol_septum/tests -p test_new_pipeline_step08.py -v

conda run --no-capture-output -n cme-dt-env python -B -m unittest discover \
  -s model_vol_septum/tests -p test_new_pipeline_step09.py -v

conda run --no-capture-output -n cme-dt-env python -B -m unittest discover \
  -s model_vol_septum/tests -p test_run_pipline_uvc.py -v
```

Step 08 coverage includes descending fallback, early stopping, restarting from
the cached original shell, native-process failure isolation, and independent
frame publication. Step 09 coverage includes physical distance budgets, seed
selection, disconnected/edge-only contact, reverted cells, immutable geometry,
serialization, output backups, and CLI routing. Runner tests verify the new
growth arguments and separate collapse-list tokens.

`git diff --check` passed after the implementation. These are focused results,
not a claim that every unrelated project test or every subject succeeds.

Remaining limitations:

- subject_23 ED Step 03 collision remains unresolved.
- Step 08 simplification can leave thin surface triangles and tiny tetrahedra.
- Geometric simplification checks are sampled and do not certify every mesh
  quality criterion.
- Distance growth is mesh-dependent, changes the main LV/RV interface as well
  as islands, and does not guarantee one connected region for each label.
- The final subject_24 tag-1 components still number 14 for ED and 13 for ES.
- No collision-aware movement changes, extra region-growing cycles, or
  unrequested geometry modifications were made following the final analysis.

