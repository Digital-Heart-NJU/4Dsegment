# 4Dsegment: From Cine MRI to LV/RV Surface Meshes

This document explains the **active implementation** in this repository's
`code/` directory and the pretrained checkpoint in `model/`. It was checked
against the Python sources, checkpoint graph, installed Docker tools, and
existing Subject 211 outputs on 2026-11-05.

The main question is how an input image becomes these four surfaces, for
each of ED and ES:

```text
LVendo_ED.vtk / LVendo_ES.vtk   LV endocardium
LVepi_ED.vtk  / LVepi_ES.vtk    LV epicardium
RV_ED.vtk     / RV_ES.vtk       RV endocardium
RVepi_ED.vtk  / RVepi_ES.vtk    RV epicardium
```

**The CNN does not directly predict a mesh.** It predicts voxel labels.
Those labels guide atlas registration and label fusion. The refined labels
are converted to triangle surfaces by marching cubes. Template fitting then
creates additional surface files with different prefixes and purposes.

`RV_*.vtk` is the actual endocardial filename in this extraction pipeline;
it is not called `RVendo_*.vtk`. Files such as `/vtks/2/RVendo_ED.vtk` belong
to the motion template, not to this subject-level naming convention.

The later open-base, intersection-repair, TetGen, LV/RV volume-tagging, UVC,
and fiber pipelines are **not** part of `DMACS.py` and are outside this document.

## 1. Overall Pipeline

```text
data/<subject>/example.nii, a 4D cine MRI [X, Y, Z, T]
    |
    v
imagePreprocessing(): reset spatial header, temporal alignment, autocontrast
    |
    +--> lvsa_.nii.gz, preprocessed full cine
    |
    v
cardiacphasedetection: extract ED and ES images
    |
    +--> lvsa_ED.nii.gz and lvsa_ES.nii.gz
    |
    v
Pretrained 2D region-and-edge CNN, applied slice by slice
    |
    +--> segs/seg_lvsa_ED.nii.gz and segs/seg_lvsa_ES.nii.gz
    |    Initial five-class voxel segmentations, not surface meshes
    |
    v
Compare candidate shape atlases; select the best three for each phase
    |
    v
Resample image grid to 1.25 x 1.25 x 2; warp selected atlas labels onto it
    |
    v
Vote-based label fusion; blur/threshold masks; rebuild anatomical labels
    |
    +--> PHsegmentation_ED.gipl and PHsegmentation_ES.gipl
    |    Refined segmentations actually used for surface extraction
    |
    v
Extract binary masks for LVendo, LVepi, RVendo, RVepi, and LVmyo
    |
    v
mcubes: blur each mask, then extract its isosurface at threshold 120
    |
    +--> vtks/LVendo_<phase>.vtk
    +--> vtks/LVepi_<phase>.vtk
    +--> vtks/RV_<phase>.vtk
    +--> vtks/RVepi_<phase>.vtk
    +--> vtks/LVmyo_<phase>.vtk
    |    Unprefixed, segmentation-derived triangle surfaces
    |
    v
Rigid alignment, affine/nonrigid registration, and template surface mapping
    |
    +--> vtks/N_*.vtk, F_*.vtk, C_*.vtk, S_*.vtk, W_*.vtk, FC_*.vtk
    |    Additional fitted or measurement-related surfaces
    |
    v
Full-cine motion estimation and propagation of an RV template surface
    |
    +--> motion/RV_fr00.vtk, RV_fr01.vtk, ... and corresponding .txt files
    |
    v
Decimate the first valid subject's reference RV motion mesh;
save reduced meshes and point correspondences
```

Here `<phase>` is `ED` or `ES`. The same image-to-surface operations run for
both phases; the neural network is not applied independently to every cine
frame by the default pipeline.

## 2. What DMACS.py Actually Calls

Entry point: [DMACS.py](DMACS.py#L37).

| Order | Active call | Implementation | Main role |
| --- | --- | --- | --- |
| 1 | `deeplearningseg(...)` | [deepseg.py](deepseg.py#L7) | Preprocess cine, extract phases, run CNN segmentation. |
| 2 | `multiatlasreg3D(...)` | [p2processing.py](p2processing.py#L114) | Refine CNN labels using registered 3D shape atlases. |
| 3 | `meshCoregstration(...)` | [meshfitting.py](meshfitting.py#L450) | Extract surfaces and perform template fitting. The spelling is as in the code. |
| 4 | `motionTracking(...)` | [motionEstimation.py](motionEstimation.py#L140) | Estimate cine motion and generate time-resolved RV surfaces. |
| 5 | `decimate(...)` | [decimation.py](decimation.py#L159) | Generate a reduced reference RV surface and correspondences. |

Important distinctions between imported code and executed code:

- `multiatlasreg2D(...)` from [p1processing.py](p1processing.py#L95) is
  imported but its call in `DMACS.py` is commented out. It does not refine
  the initial CNN segmentation in a normal run.
- `clearBaseManbrance(...)` and its adjacent repeat-refinement call in
  `p2processing.py` are commented out. This is not the later open-base method.
- `removeSegsAboveBase(...)` exists in `image_utils.py` but is not called by
  the active path described here.
- `prepare_dmacs_input.py` is an optional input-preparation script, not an
  automatic step of `DMACS.py`.
- The `track_cine` used by `motionTracking` is the function in
  `motionEstimation.py`. A similarly named function in `decimation.py` is
  not invoked by the default `decimate` call.

### Parallelism

`coreNo` defaults to 8. Atlas refinement, mesh fitting, and motion tracking
create multiprocessing pools **across subject directory entries**. Within
one subject, ED/ES and the atlas-comparison loops are processed sequentially.
The CNN also iterates through slices sequentially. Consequently, `--coreNo 8`
does not mean that eight atlases or eight CNN slices of one subject are
necessarily processed simultaneously.

## 3. Inputs, Resources, and Runtime Paths

The defaults are declared in [DMACS.py](DMACS.py#L26).

| Argument | Default inside Docker | Meaning |
| --- | --- | --- |
| `--test_dir` | `/data` | Parent dataset directory containing subject subdirectories. |
| `--model_path` | `/model/vgg_RE_network.ckpt-50000` | Explicit TensorFlow checkpoint prefix. |
| `--atlas_dir` | `/refs` | Temporal reference and candidate labeled shape atlases. |
| `--param_dir` | `/par` | Registration configuration files. |
| `--template_dir` | `/vtks/1` | ED/ES template surfaces and masks for fitting. |
| `--template_PH` | `/vtks/2` | RV template for motion tracking. |
| `--coreNo` | `8` | Subject-level worker count. |
| `--irtk` | `True` | Forwarded to the atlas-refinement function; see backend caveat below. |

Only mounting `code/`, `model/`, and an input image into an arbitrary Python
container is insufficient. `/refs`, `/par`, `/vtks/1`, `/vtks/2`, and the
compiled image/surface tools are also required. These resources are present
in the locally installed `jinmingduan/segmentationmeshmotion:latest` image.

The inspected image contains Python 2.7.15rc1 and TensorFlow 1.4.0. The source
uses TensorFlow 1 APIs such as `tf.Session`, `tf.app.flags`, and
`tf.train.import_meta_graph`, plus the legacy NiBabel `get_data()` API.
Running the same code unchanged in a recent TensorFlow 2/NiBabel environment
is not equivalent to using the supplied legacy environment.

### Input discovery and assumptions

For each directory under `test_dir`, [deepseg.py](deepseg.py#L44) selects
the first matching `*.nii`; if none exist, it selects the first `*.nii.gz`.
The code does not specifically require the basename `example.nii`, although
that is the convention used here. The glob results are not explicitly sorted.
Keep one intended input cine in each subject directory to avoid ambiguity.

The expected input is a short-axis cardiac cine with meaningful geometry and
intensities. The phase images returned by preprocessing are expected to have
shape `[X, Y, Z, 1]`: the CNN code removes their final singleton dimension
with `np.squeeze(image, axis=-1)`.

The core pipeline does not automatically validate anatomical orientation,
crop the heart, enforce 512 x 512 images, or resample the input in-plane
spacing to 0.9375. It pads CNN inputs to multiples of 16 later. Those are
different operations.

### Backend caveat

The flag name `irtk` is misleading if interpreted as a global switch:

- `DMACS.py` passes its value into the parameter named `mirtk` in
  `p2processing.py`. With the default `True`, atlas deformation uses
  `mirtk register` and `mirtk transform-image`; `False` selects the older
  `nreg`/`transformation` branch there.
- `meshCoregstration(..., True, False)` is called with a hard-coded last
  argument. Its active `meshGeneration` implementation invokes the listed
  IRTK-style tools directly; it does not dynamically switch those commands.
- Motion estimation explicitly invokes MIRTK as well as surface tools.

Thus, the normal run uses a mixture of the bundled tools.

## 4. Stage A: Preprocessing and ED/ES Extraction

Implementation: [imagePreprocessing](image_utils.py#L16), called from
`deeplearningseg` before neural inference for each subject.

| Operation | Actual command | Output and interpretation |
| --- | --- | --- |
| Header preparation | `headertool INPUT SUBJECT/lvsa_.nii.gz -reset` | Writes a working cine; resets origin and axis orientation to the tool's defaults. |
| Temporal alignment | `temporalalign /refs/temporal.gipl SUBJECT/lvsa_.nii.gz SUBJECT/lvsa_.nii.gz -St1 0 -St2 0 -Et1 1 -Et2 1` | Aligns the working cine to the temporal reference. |
| Contrast processing | `autocontrast SUBJECT/lvsa_.nii.gz SUBJECT/lvsa_.nii.gz` | Changes intensities in the working cine. |
| Phase extraction | `cardiacphasedetection SUBJECT/lvsa_.nii.gz SUBJECT/lvsa_ED.nii.gz SUBJECT/lvsa_ES.nii.gz` | Produces ED and ES phase images. |

`SUBJECT` denotes the subject directory inside the container, for example
`/data/subject_211`.

Important consequences:

1. `headertool -reset` is **not anatomical image reorientation** or a learned
   alignment procedure. Resetting header geometry is not the same as rotating
   voxel data into a canonical orientation.
2. Later segmentations and raw meshes use the processed image geometry.
   Do not assume their coordinates are the unchanged scanner-world coordinates
   of the original input. Compare with the processed images first.
3. Temporal alignment can change the number of frames. The Python source does
   not hard-code 20 here; that behavior depends on the temporal reference/tool.
4. ED/ES selection is performed by the external `cardiacphasedetection` binary,
   not by an explicit Python LV-volume minimum/maximum calculation. Its full
   detection algorithm is not implemented in these Python files. Frame indices
   reported after temporal alignment need not equal original cine indices.
5. The original `example.nii` is not the destination of these preprocessing
   commands; the derived `lvsa_` working images are.

## 5. Stage B: The Segmentation Model

### Checkpoint files and loading

The active prefix is `model/vgg_RE_network.ckpt-50000`, comprising:

| File | Purpose |
| --- | --- |
| `vgg_RE_network.ckpt-50000.meta` | Serialized computation graph and TensorFlow metadata. |
| `vgg_RE_network.ckpt-50000.index` | Index of the stored checkpoint tensors. |
| `vgg_RE_network.ckpt-50000.data-00000-of-00001` | Stored tensor values, including trained weights. |
| `checkpoint` | Historical checkpoint-list text; not how this inference call selects its model. |

`deeplearningseg` creates a session, imports `MODEL_PREFIX.meta`, and calls
`saver.restore(sess, MODEL_PREFIX)` explicitly. It does not call
`latest_checkpoint`. The local `checkpoint` text contains old author-machine
paths ending in `49500`; those entries do not override the explicit `50000`
prefix used here.

The graph is loaded once around the subject loop. This is inference with
existing weights, not training or fine-tuning.

### Architecture confirmed from the saved graph

The checkpoint contains a **2D, VGG-like, fully convolutional region-and-edge
network**. This description is based on the graph, not just the filename:

- The image placeholder is `[batch, X, Y, 1]`.
- Its encoder has 13 convolution kernels of size 3 x 3, with channel blocks
  `16, 32, 64, 128, 256` and block depths `2, 2, 3, 3, 3`.
- Four stride-2 convolutions reduce spatial resolution. The inspected graph
  does not use `MaxPool` operations for those reductions.
- Multiscale features are projected and upsampled with transpose-convolution
  strides `2, 4, 8, 16`, then concatenated with a full-resolution branch.
- The concatenated representation has 192 channels, followed by convolutional
  fusion and a 1 x 1 output convolution with six channels.
- Five output channels form the region-classification branch, and one forms
  the edge branch. Region probabilities use softmax; edge probability uses
  sigmoid. The graph also contains batch-normalization and ReLU operations.

This is not a 3D CNN, a cine-wide temporal network, or a model that outputs
surface vertices. Its region-and-edge graph should not be described as an
unmodified off-the-shelf VGG classifier or assumed to be a standard U-Net.
The original Python training/network-construction source is not included in
`code/`; the serialized graph supplies the executable architecture.

| Tensor | Shape | Role in this run |
| --- | --- | --- |
| `image:0` | `[N, X, Y, 1]` | Single-channel image batch; `N=1` for each slice. |
| `training:0` | Scalar | Set to `False` for inference. |
| `probR:0` | `[N, X, Y, 5]` | Five-class region probabilities in the graph. |
| `predR:0` | `[N, X, Y]` | Region labels derived from the class argmax; these become the saved segmentation. |
| `probE:0` | `[N, X, Y]` | Edge probability; fetched but not saved or used downstream by this Python path. |
| `predE:0` | `[N, X, Y]` | Edge prediction exists in the graph but is not fetched by `deepseg.py`. |

### Per-slice inference

Implementation: [deepseg.py](deepseg.py#L91) and
[rescale_intensity](image_utils.py#L6).

For both ED and ES:

```text
phase image [X, Y, Z, 1]
  -> remove singleton time dimension; cast to int16
  -> take one XY slice
  -> compute its 1st and 99th intensity percentiles
  -> clip intensities and rescale to approximately [0, 1]
  -> center-pad X and Y to the next multiples of 16
  -> reshape to [1, X_padded, Y_padded, 1], float32
  -> run probE:0 and predR:0, with training:0=False
  -> crop away the temporary padding
  -> place predicted int16 labels into the corresponding Z slice
  -> save the assembled 3D segmentation with the phase image affine
```

The padding dimensions are:

```text
X_padded = ceil(X / 16) * 16
Y_padded = ceil(Y / 16) * 16
```

For Subject 211, `151 x 121` is therefore temporarily padded to `160 x 128`,
not stretched to that size. The output labels are cropped back to `151 x 121`.
The model is not explicitly told the physical voxel spacing, so matching a
tensor shape alone does not guarantee matching the training distribution.

### Anatomical label meanings

These meanings are confirmed by the mask-extraction rules in `meshfitting.py`:

| Voxel label | Meaning |
| --- | --- |
| `0` | Background. |
| `1` | LV blood pool / cavity. |
| `2` | LV myocardium. |
| `3` | RV myocardium. |
| `4` | RV blood pool / cavity. |

Initial CNN outputs are:

```text
segs/seg_lvsa_ED.nii.gz
segs/seg_lvsa_ES.nii.gz
```

They are useful for checking the neural prediction, but they are **not the
files directly passed to marching cubes** by the active mesh extraction path.

## 6. Stage C: 3D Atlas-Based Segmentation Refinement

Implementations: [p2processing.py](p2processing.py#L55),
[allAtlasShapeSelection](image_utils.py#L369), and
[topSimilarAtlasShapeSelection](image_utils.py#L406).

### 6.1 Candidate atlases and ranking

The code considers up to the first 400 atlas directory entries in sorted
order under `/refs`, separately for ED and ES. Each candidate supplies a
`PHsegmentation_<phase>.nii.gz` label image. Although landmark paths are
assembled, `landmarks=False` in the active selection function.

For each candidate:

1. Register its label image to the subject CNN segmentation using `areg` and
   `/par/segareg.txt`.
2. Save the transformation as `dofs/shapelandmarks_<index>.dof.gz`. Despite
   the basename, this active branch estimates it from label images, not from
   manually matched landmark points.
3. Run `cardiacimageevaluation` with that transformation and 64 histogram bins.
4. Read `tmps/shapenmi_<index>.txt`, take `similarities[3]`, and rank in
   descending order.
5. Select the best three candidates for the current phase.

**Ranking detail:** the Python variable is called `nmi`, but the installed
evaluation tool exports four values in this order:

```text
index 0: SSD
index 1: CC
index 2: NMI
index 3: label consistency
```

The actual ranking therefore uses **label consistency**, not the exported
NMI value. This ordering was checked against the installed executable's
console and text output. It is an atlas-selection comparison, not a
ground-truth accuracy evaluation of the patient's segmentation.

The candidate-discovery condition checks whether the atlas image **or** its
landmarks exist, rather than requiring both. Incomplete atlas resources can
therefore cause downstream command failures instead of clean exclusion.

### 6.2 High-resolution target grid

[formHighResolutionImg](image_utils.py#L160) runs:

```text
resample SUBJECT/lvsa_<phase>.nii.gz \
         SUBJECT/lvsa_SR_<phase>.nii.gz -size 1.25 1.25 2
```

These are physical voxel spacings in the tools' image geometry, conventionally
millimeters. The inputs must use consistent geometry/spacing conventions.

`SR` is a filename convention here: this is **resampling**, not a learned
super-resolution network. In particular, interpolating thick Z slices onto a
2-unit grid does not recover genuinely measured fine-slice detail.

### 6.3 Deform selected atlas labels and fuse them

[output3DRefinement](p2processing.py#L7) registers each of the three selected
atlases using the affine initialization. In the default branch it uses:

```text
mirtk register ... -parin /par/ffd_label_1.cfg -dofin ...
    -> dofs/shapeffd_<selected_rank>_<phase>.dof.gz

mirtk transform-image ATLAS_LABEL OUTPUT_LABEL \
    -dofin ... -target SUBJECT/lvsa_SR_<phase>.nii.gz -interp NN
    -> tmps/seg_lvsa_SR_<selected_rank>_<phase>.nii.gz
```

Nearest-neighbor interpolation preserves discrete label IDs. `combineLabels`
then combines these aligned labels into `SUBJECT/seg_lvsa_SR_<phase>.nii.gz`.
The bundled tool uses per-voxel modal voting; ties are resolved randomly.
The Python wrapper does not supply a random seed for this external operation.

### 6.4 Smooth masks and rebuild the anatomical labels

[refineFusionResults](image_utils.py#L261) is called with `alfa=2`:

| Temporary mask | Labels selected | Foreground / background | Blur parameter | Threshold |
| --- | --- | --- | --- | --- |
| `tmps/hrt.nii.gz` | `1..4` | `255 / 0` | `2` | `130` |
| `tmps/rvendo.nii.gz` | `4` | `255 / 0` | `2` | `130` |
| `tmps/lvepi.nii.gz` | `1..2` | `255 / 0` | `2` | `115` |
| `tmps/lvendo.nii.gz` | `1` | `255 / 0` | `2` | `130` |

The subsequent `padding` commands rebuild the label map by layering these
masks: the whole-heart foreground supplies label 3, RV cavity supplies label
4, the filled LV region supplies label 2, and LV cavity supplies label 1.
Here `padding` means replacing values according to another mask; it is not
the tensor-size padding performed before CNN inference.

This step is another reason the final surface contour can differ from the
initial neural segmentation. It changes voxel-region boundaries before any
triangles are generated.

### 6.5 Final refined labels and archives

[convertImageSegment](image_utils.py#L176) converts the refined label image
to `.gipl` and copies the resampled MRI. [moveVolumes](image_utils.py#L222)
archives several intermediates and removes their temporary root-level copies.

| File after refinement | Meaning |
| --- | --- |
| `PHsegmentation_ED.gipl`, `PHsegmentation_ES.gipl` | Refined labels used for surface extraction. |
| `lvsa_ED_enlarged_SR.nii.gz`, `lvsa_ES_enlarged_SR.nii.gz` | Resampled phase MRI. The enlargement call is commented out; the basename does not prove additional Z padding happened. |
| `sizes/lvsa_ED.nii.gz`, `sizes/lvsa_ES.nii.gz` | Archived preprocessed phase images. |
| `sizes/lvsa_ED_SR.nii.gz`, `sizes/lvsa_ES_SR.nii.gz` | Archived resampled phase images. Note the phase/SR ordering differs from the temporary filename. |
| `sizes/3D_segmentation_ED.nii.gz`, `sizes/3D_segmentation_ES.nii.gz` | NIfTI archives of the refined segmentation corresponding to the GIPL labels. |
| `segs/seg_lvsa_ED.nii.gz`, `segs/seg_lvsa_ES.nii.gz` | Original CNN predictions, retained separately. |

There is a path inconsistency in the attempted `sizes/2D_segmentation_*.nii.gz`
backup: `moveVolumes` looks for `SUBJECT/seg_lvsa_<phase>.nii.gz`, while the
active CNN writes it under `SUBJECT/segs/`. Consequently, that backup may be
missing, with `cp` warnings. Do not confuse this with loss of the actual
`segs/` predictions.

`outputVolumes` also uses `cardiacvolumecount` to write
`<test_dir>/<subject>_<phase>.txt`. Labels 2 and 3 use the explicit scale
factor `1.05`; labels 1 and 4 do not. These are segmentation-based volume
summaries, not tetrahedral mesh files.

## 7. Stage D: Refined Labels to the Four Requested Surface Meshes

Implementation: [meshGeneration](meshfitting.py#L6), called through
`meshCoregstration` and its subject worker.

### 7.1 Build the binary anatomical masks

For each phase, the input is `SUBJECT/PHsegmentation_<phase>.gipl`.

| Requested structure | Selected label set | Temporary binary mask | Direct surface output |
| --- | --- | --- | --- |
| LV endocardium | `{1}` | `tmps/vtk_LVendo_<phase>.nii.gz` | `vtks/LVendo_<phase>.vtk` |
| LV epicardium | `{1, 2}` | `tmps/vtk_LVepi_<phase>.nii.gz` | `vtks/LVepi_<phase>.vtk` |
| RV endocardium | `{4}` | `tmps/vtk_RV_<phase>.nii.gz` | `vtks/RV_<phase>.vtk` |
| RV epicardium | `{3, 4}` | `tmps/vtk_RVepi_<phase>.nii.gz` | `vtks/RVepi_<phase>.vtk` |
| LV myocardium boundary | `{2}` | `tmps/vtk_LVmyo_<phase>.nii.gz` | `vtks/LVmyo_<phase>.vtk` |

For the masks in this table, `binarize` writes foreground `255` and background
`0`.

**Why include cavity labels in an epicardial mask?** The epicardial surface
is the outer boundary of the filled ventricle, including blood pool and wall.
Using only myocardium would also expose its cavity-facing boundary. Thus,
LVepi is extracted from labels 1+2, while RVepi is extracted from labels 3+4.

The separate `LVmyo` mask uses label 2 alone. Its extracted boundary can
include both inner and outer wall surfaces; it is still triangle polydata,
not a filled tetrahedral myocardial volume.

The code also constructs `tmps/vtk_LV_<phase>.nii.gz` by replacing labels
4 and 3 with zero. Unlike the individual 255-valued binary masks above, this
auxiliary image retains the LV labels and is used during subsequent registration.

### 7.2 Marching cubes is where image voxels become mesh vertices/faces

Every direct extraction uses this pattern:

```bash
mcubes BINARY_MASK OUTPUT_SURFACE.vtk 120 -blur 2
```

For example:

```bash
binarize SUBJECT/PHsegmentation_ED.gipl \
         SUBJECT/tmps/vtk_LVendo_ED.nii.gz 1 1 255 0
mcubes SUBJECT/tmps/vtk_LVendo_ED.nii.gz \
       SUBJECT/vtks/LVendo_ED.vtk 120 -blur 2
```

The tool blurs the scalar mask using its `-blur 2` option, then locates the
isosurface where the blurred value equals 120. Marching cubes interpolates
surface points and generates triangular faces. Therefore:

- Mesh points are not simply copies of voxel centers, and a surface need
  not lie exactly on the unsmoothed binary boundary.
- Image geometry is respected unless `-ignoreGeom` is requested; that option
  is not passed here. Remember that the working image header was reset earlier.
- The active call does not request the optional `-smooth`, `-decimate`,
  `-isotropic`, or `-close` operations. Mask blurring must not be described
  as an explicit post-extraction Laplacian surface-smoothing step.
- The tool defaults to binary VTK output unless `-ascii` is specified.
  Binary versus ASCII is a storage choice, not a change of anatomical labels.

The resulting unprefixed files are already the four surfaces requested in
the question. They come from **atlas-refined segmentation**, not directly
from the neural-network prediction files.

### 7.3 Closure, intersection, and labels

These surfaces are not deliberately cut open at the cardiac base. They
represent boundaries of segmented regions, often enclosing the cavities.
Nevertheless, closure is not guaranteed: a foreground mask reaching an image
edge can leave holes, and the extraction command does not request `-close`.

This original pipeline does not certify no self-intersections, no LV/RV
surface overlaps, a fixed wall thickness, or valid downstream tetrahedral
meshing. Independent mask processing, interpolation, and deformation can
introduce geometrical issues. Those are separate from whether a VTK is readable.

The five voxel labels are also not the later `elemTag=1` for LV and
`elemTag=0` for RV used in the open-base volume pipeline. `DMACS.py` does not
perform that tetrahedral tagging operation.

## 8. Stage E: Registration and Template-Fitted Surfaces

After extracting the unprefixed surfaces, `meshGeneration` continues.
The following operations explain the additional VTK files; they do not
overwrite the four unprefixed surface outputs above.

### 8.1 Rigid alignment and N_ surfaces

[meshfitting.py](meshfitting.py#L189) first invokes:

1. `msrreg 3` on RV, LVendo, and LVepi together against `/vtks/1` surfaces.
2. `msrreg 2` for the LV pair and `srreg` for RV, with the initial transform.
3. `ptransformation` to write `N_RV`, `N_RVepi`, `N_LVendo`, `N_LVepi`,
   and `N_LVmyo` surfaces.
4. Image `transformation` calls to create corresponding `N_vtk_*` masks.

`N_` files are transformed working surfaces used in template alignment and
fitting. They should not be overlaid on original subject MRI merely because
they have the same anatomical suffix; check the applied transforms and the
appropriate image coordinate system.

One literal implementation detail worth retaining when reproducing results:
the `N_vtk_RV` image transformation uses `lv_<phase>_rreg.dof.gz`, whereas
the `N_RV` surface transformation uses `rv_<phase>_rreg.dof.gz`. This document
describes the implemented commands rather than silently replacing either.

### 8.2 Affine and nonrigid registration

[meshfitting.py](meshfitting.py#L264) performs:

- `areg` for RV and LV label images with `/par/segareg.txt`.
- `nreg` with `/par/segreg.txt` for a deformable initialization.
- `snreg` for the RV surface, using `-ds 8 -symmetric`.
- `msnreg 2` for the LV endo/epi pair, using `-ds 4 -symmetric`.

These are image- and surface-registration stages, not additional CNNs.
The `-ds` values are surface-deformation configuration values, not the image
voxel spacing and not a myocardial-thickness requirement.

### 8.3 Surface correspondence and measurements

[meshfitting.py](meshfitting.py#L314) then uses `cardiacsurfacemap` to map
template point positions to the subject-derived working surfaces with the
estimated deformation. Its role is correspondence-controlled surface mapping,
as indicated by the code's "same number of points" step.

| File family | How it is produced / intended role |
| --- | --- |
| `F_LVendo_<phase>.vtk`, `F_LVepi_<phase>.vtk` | Template-mapped LV endo/epi surfaces from `cardiacsurfacemap`. |
| `F_LVmyo_<phase>.vtk` | Template LV myocardial surface transformed with the final LV deformation. |
| `C_RV_<phase>.vtk` | Template-mapped RV endocardial surface, subsequently used for curvature-related processing. There is no matching `F_RV` output in this implementation. |
| `S_LV*`, `S_RV*` | Copies used for template-relative distance measurements. |
| `W_LVmyo*`, `W_RV*` | Copies used for wall-thickness measurements. |
| `C_LVmyo*`, `FC_LVmyo*`, `FC_RV*` | Curvature-processing surfaces/intermediates. |

`cardiacwallthickness`, `cardiacenlargedistance`, `sevaluation`,
`DiscreteCurvatureEstimator`, and `cardiaccurvature -smooth 64` produce
measurement-related surfaces or scalar fields. The Python code also exports
these quantities to `rv_ed_*`, `rv_es_*`, `lv_myoed_*`, and `lv_myoes_*`
text files through `vtk2txt`. Curvature-field processing is not a guarantee
that every mesh has globally smooth, valid geometry.

**Which family should be inspected?** Use unprefixed LVendo/LVepi/RV/RVepi
to inspect the direct segmentation-derived boundaries. Inspect fitted files
when studying template correspondences or the associated measurements, taking
their coordinate system into account. They are not interchangeable outputs
of a single geometry-preserving operation.

## 9. Stage F: Motion Tracking and Decimation

Implementation: [motionEstimation.py](motionEstimation.py#L8).

The active motion path:

1. Splits the preprocessed `lvsa_.nii.gz` into `motion/lvsa_00.nii.gz`, etc.
2. Registers consecutive frames with MIRTK using `/par/ffd_motion.cfg`.
3. Composes successive transformations into frame-00-to-frame-N estimates.
4. Refines those estimates against frame 00 using `/par/ffd_refine.cfg`.
5. Fits `/vtks/2/RVendo_ED.vtk` to the subject's ED RV surface using surface
   registration, producing `motion/RV_fr00.vtk`.
6. Transforms this reference RV mesh through the cine and writes
   `motion/RV_frNN.vtk` and `motion/RV_frNN.txt`.

This keeps the reference RV mesh's point identity through time instead of
rerunning marching cubes independently for every frame. This function outputs
an RV motion sequence; it does not generate matching LV endo/epi and RVepi
sequences in this active implementation. It uses frame 00 as its reference
and treats that reference as ED; verify that convention for new cine data.

[decimation.py](decimation.py#L159) then runs the non-parallel branch because
`DMACS.py` passes `parallel=False`. It processes **only the first valid
subject directory** due to its `i == 0` guard, generating:

```text
motion/RV_fr00_new.vtk
motion/transformation.vtk
motion/RV_fr00_new_new.vtk
<test_dir>/matchedpointsnew.txt
```

The decimator requests `-reduceby 98.9 -preservetopology on` on the reference
RV motion mesh, then performs registration and point matching. It is not a
decimation pass over all four original ED/ES ventricular surfaces.

## 10. Output Directory Guide

```text
data/<subject>/
|-- example.nii                         original cine input
|-- lvsa_.nii.gz                        preprocessed full cine
|-- PHsegmentation_ED.gipl              refined labels used for ED extraction
|-- PHsegmentation_ES.gipl              refined labels used for ES extraction
|-- lvsa_ED_enlarged_SR.nii.gz           resampled ED image
|-- lvsa_ES_enlarged_SR.nii.gz           resampled ES image
|-- segs/
|   |-- seg_lvsa_ED.nii.gz               initial CNN labels
|   `-- seg_lvsa_ES.nii.gz               initial CNN labels
|-- sizes/                              archived phase MRI and refined labels
|-- dofs/                               atlas deformation files
|-- tmps/                               masks, fitting transforms, intermediates
|-- vtks/
|   |-- LVendo_ED.vtk / LVendo_ES.vtk    direct LV endocardial surfaces
|   |-- LVepi_ED.vtk  / LVepi_ES.vtk     direct LV epicardial surfaces
|   |-- RV_ED.vtk     / RV_ES.vtk        direct RV endocardial surfaces
|   |-- RVepi_ED.vtk  / RVepi_ES.vtk     direct RV epicardial surfaces
|   |-- LVmyo_ED.vtk  / LVmyo_ES.vtk     LV wall boundary surfaces
|   `-- N_*, F_*, C_*, FC_*, S_*, W_*    fitted/measurement surface families
|-- motion/                             full-cine RV meshes and transforms
`-- *_curvature.txt, *_wallthickness.txt, *_signeddistances.txt
```

`tmps/` is reused and cleared at the beginning of mesh generation. Atlas
ranking files are also overwritten between phase selections. Not every
intermediate described above remains available after the full run.

The volume summary text files, `subjnames.txt`, and the default non-parallel
decimation's `matchedpointsnew.txt` are written under `test_dir`, not necessarily
inside the subject directory. A wrapper can preserve them alongside subject
outputs, but this placement is not the default Python implementation.

### Verified Subject 211 example: dimensions are not hard-coded

Existing saved ED outputs were inspected without rerunning the model:

| Image | Observed shape | Spatial spacing values |
| --- | --- | --- |
| `example.nii` | `(151, 121, 12, 31)` | `(0.9375, 0.9375, approximately 8)` |
| `lvsa_.nii.gz` | `(151, 121, 12, 20)` | Same spatial spacing. |
| `sizes/lvsa_ED.nii.gz` | `(151, 121, 12, 1)` | Same spatial spacing. |
| `segs/seg_lvsa_ED.nii.gz` | `(151, 121, 12)` | Same spatial spacing. |
| `sizes/lvsa_ED_SR.nii.gz` | `(113, 90, 48, 1)` | `(1.25, 1.25, 2)` |
| `sizes/3D_segmentation_ED.nii.gz` | `(113, 90, 48, 1)` | `(1.25, 1.25, 2)` |

The 20-frame cine and resampled dimensions in this table describe this saved
example. They are not general fixed dimensions for every subject. Check NIfTI
time units and temporal metadata separately rather than assuming every fourth
spacing value is expressed in seconds.

## 11. Running the Original Pipeline

**Rerun warning:** `deepseg.py` removes previous generated segmentation,
mesh, motion, and temporary directories when they exist. `meshGeneration`
also removes subject-root `*.txt` files and clears `vtks/` and `tmps/`.
Back up results before rerunning. The commands below are instructions, not
commands that were executed while preparing this documentation.

### Single subject, with persistent dataset-level summaries

Run this in Bash, adjusting `PROJECT` and `SUBJECT` as needed:

```bash
PROJECT=/home/mengao/Code/4Dsegment
SUBJECT=subject_211
RUN_ROOT="$PROJECT/run_DMACS_$SUBJECT"

# /data must be a writable parent containing only the intended subject.
mkdir -p "$RUN_ROOT/$SUBJECT"
set -o pipefail

docker run --rm --name "dmacs_$SUBJECT" \
  --user "$(id -u):$(id -g)" \
  -e LD_LIBRARY_PATH=/lib64 \
  -e OMP_NUM_THREADS=8 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "$RUN_ROOT:/data" \
  -v "$PROJECT/data/$SUBJECT:/data/$SUBJECT" \
  -v "$PROJECT/code:/code:ro" \
  -v "$PROJECT/model:/model:ro" \
  -w /code \
  jinmingduan/segmentationmeshmotion:latest \
  python -u DMACS.py --test_dir /data --coreNo 8 --irtk True \
  2>&1 | tee "$RUN_ROOT/DMACS.log"
```

The nested subject bind mount writes the main results into
`$PROJECT/data/$SUBJECT`. The separate writable parent keeps dataset-level
summaries, `subjnames.txt`, and `matchedpointsnew.txt` in `RUN_ROOT`, which
survives container removal. Keep that run root dedicated to this subject;
the program scans all directory entries, not a `subject_` name pattern.

This is a full ED/ES and motion run. `DMACS.py` has no built-in `--subject`,
`--frame`, or stop-after-surface flag. `--test_dir` must point to the **parent
dataset directory**, not directly to a populated subject folder.

To run all subject folders instead, mount the complete local `data/` as
`/data` rather than using this isolated parent/subject pair. That causes every
subject directory under it to be processed and its previous derived outputs
to be replaced. It also retains the first-subject-only decimation limitation.

### Optional preparation is a separate operation

[prepare_dmacs_input.py](prepare_dmacs_input.py#L127) can prepare
`example.nii` from an original `example.nii.gz`: it resamples X/Y to 0.9375,
preserves the Z stack, orientation, and frames, and robustly scales positive
intensities to int16 values up to 4095 using their 99.5th percentile.
It does not center-pad to 512 x 512. It is not imported or called by DMACS.
The separate input-cropping script under `additional_function/` is likewise
not an automatic model step.

## 12. How to Inspect Results and Diagnose Differences

Use these checkpoints in order:

1. **Processed MRI:** inspect `lvsa_.nii.gz` and archived ED/ES phase images
   for phase selection, stack direction, field of view, and contrast. A header
   reset does not fix an incorrectly oriented input array.
2. **Initial prediction:** overlay `segs/seg_lvsa_<phase>.nii.gz` on the
   corresponding preprocessed phase image to assess what the CNN predicted.
3. **Atlas-refined labels:** compare `sizes/3D_segmentation_<phase>.nii.gz`
   with the corresponding resampled MRI. Large changes appearing here arise
   after CNN inference: atlas selection, deformation, fusion, and mask cleanup.
4. **Direct surface:** inspect the unprefixed VTKs against the refined labels.
   Further contour changes can come from mask blurring, isovalue interpolation,
   image-edge truncation, and different sampling grids.
5. **Fitted surfaces:** compare fitted and working surface families in their
   appropriate coordinates. Registration and correspondence mapping can change
   geometry independently of the segmentation and raw marching-cubes output.
6. **Geometric validation:** check boundary edges, degenerate triangles,
   self-intersections, LV/RV overlaps, and wall thickness before using these
   files for downstream volume meshing. The original runner does not certify
   those properties.

Many external commands are launched through `os.system` without checking their
return codes. A final timing message or a top-level zero exit code therefore
does not prove every intermediate operation succeeded. Inspect the log and
verify required files. In particular, non-invertible-transform warnings from
registration should not be dismissed merely because surfaces were written.

The default code does not calculate Dice or Hausdorff distance against manual
ground-truth segmentations. Atlas comparison and template-distance measurements
serve other purposes and must not be presented as ground-truth validation.

## 13. Source Map and Verification Limits

| Source | Relevant entry points |
| --- | --- |
| [DMACS.py](DMACS.py) | Runtime flags and five active top-level calls. |
| [deepseg.py](deepseg.py) | Input discovery, graph restoration, phase/slice loops, saved CNN labels. |
| [image_utils.py](image_utils.py) | Preprocessing commands, normalization, atlas ranking, resampling, fusion cleanup, format conversion, archives. |
| [p2processing.py](p2processing.py) | Active 3D atlas refinement and per-subject dispatch. |
| [meshfitting.py](meshfitting.py) | Masks, marching cubes, surface/image registration, mapping, measurements. |
| [motionEstimation.py](motionEstimation.py) | Active cine motion and RV mesh propagation. |
| [decimation.py](decimation.py) | Active non-parallel reference-mesh reduction and point matching. |
| [p1processing.py](p1processing.py) | Optional 2D atlas refinement, disabled in the default runner. |
| [prepare_dmacs_input.py](prepare_dmacs_input.py) | Optional standalone preprocessing, not part of the active call chain. |
| [../model/](../model/) | Inference graph, checkpoint tensors, and historical checkpoint-list text. |

No source code, input image, or existing mesh was modified to produce this
document. No new neural inference, segmentation, registration pipeline, or
mesh-generation run was performed. Verification used source inspection,
read-only checkpoint metadata, saved-image headers, executable usage text,
and an isolated temporary evaluation of existing label images to identify
the similarity-file column order. That evaluation was not an accuracy test.

Detailed implementations of the compiled tools are not supplied by the
Python files. Their verified command-line behavior is distinguished above
from internals that cannot be established just from this directory, such as
the exact ED/ES detection heuristic. For a future code or Docker image change,
re-check those assumptions against the new executable versions.
