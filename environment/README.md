# Recreate the 4Dsegment runtime on another host

This directory records the installed `cme-dt-env` environment on 2026-11-18.
It does not change the segmentation, mesh, UVC, or fiber algorithms.

## 1. What is included

`cme-dt-env.yml` is the complete version-pinned Conda/pip specification: 217
Conda dependencies and 115 pip dependencies, including notebook and visualization
packages already installed in the source environment. It is not a minimal set.
The original absolute environment prefix and Conda build strings are omitted.
No local package paths or editable installations are present in the export.

Use Conda to install this YAML, not `pip install -r`: it contains native Conda
libraries as well as pip packages. Conda installs the pip section automatically.
See the official [Conda environment creation documentation](https://docs.conda.io/projects/conda/en/stable/commands/env/create.html).

| Part of the workflow | Runtime |
| --- | --- |
| Host Python, nnUNet, geometry processing, UVC and fiber orchestration | `cme-dt-env`, Python 3.9.23 |
| Legacy DMACS preprocessing, atlas refinement and raw surface extraction | `jinmingduan/segmentationmeshmotion:latest` Docker image |
| UVC mesh conversion and PHO Laplace solution | Native openCARP tools in `/opt/openCARP/usr/bin` |
| nnUNet GPU inference | Host NVIDIA driver compatible with the installed PyTorch CUDA build |

Some important versions are NumPy 1.26.4, SciPy 1.13.1, PyVista 0.46.3,
VTK 9.5.1, PyMeshFix 0.17.2, TetGen 0.6.4, Open3D 0.19.0,
python-fcl 0.7.0.11, Manifold3D 3.5.4, PyTorch 2.8.0 and nnUNet 2.5.2.
The source PyTorch build uses CUDA 12.8. In particular, do not substitute the
different PyMeshFix requirement from the older partial `requirements_step5.txt`
when reproducing this environment.

The YAML does not contain Docker, openCARP, an NVIDIA driver, model weights,
or patient data. These are separate requirements below.

## 2. Supported migration target

The source host is Ubuntu 24.04.3, Linux x86-64. Use Ubuntu 24.04 x86-64 on
the new host for the closest match. The Docker image is also Linux amd64.
This is not a Windows, macOS, ARM, or CPU-only environment specification.
Windows users should perform the Linux setup inside a suitable WSL2 Ubuntu
installation with working Docker integration; that target has not been tested
here. Do not expect the native Linux environment to install in Windows Python.

Removing build strings allows Conda to select builds for the destination, but
this YAML is not an exact binary lockfile. A new installation still requires
network access, available package versions, compatible system libraries, and
validation on the target host. It retains CUDA packages even on a machine where
inference will be run on CPU. Allow sufficient disk space for these packages,
the approximately 8.8 GB Docker image, model weights, and generated data.

## 3. Create the Python environment on the new host

Transfer this project, including `environment/`, to the new host. The repository
does not have to remain at `/home/mengao/Code/4Dsegment`; use its new location.
Install Conda if it is not already installed, following the official
[Miniconda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install).

From the repository root on the new host:

```bash
cd /path/to/4Dsegment
conda env create --file environment/cme-dt-env.yml
conda activate cme-dt-env
python --version
python -m pip check
```

The expected Python version is 3.9.23. If `conda activate` is unavailable, run
`conda init bash` and open a new shell first. If `cme-dt-env` already exists,
do not remove it blindly; use a fresh name for a trial installation:

```bash
conda env create --name cme-dt-env-test --file environment/cme-dt-env.yml
conda activate cme-dt-env-test
```

`run_pipline.py` explicitly selects the name `cme-dt-env`, so use that name
for the final deployment or deliberately update its `CONDA_ENV` setting.
Read and handle any Anaconda channel-terms prompts according to your usage;
do not work around repository or licensing errors by disabling TLS verification.

Check the main imports without running any subject processing:

```bash
python -c "import numpy, pandas, scipy, pyvista, vtk, pymeshfix, tetgen, fcl, open3d, manifold3d, nibabel, shapely, trimesh; print('Geometry imports OK')"
python -c "from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor; print('nnUNet import OK')"
python -c "from model_vol_septum import generate_open_volmesh, generate_UVC_fiber; print('Pipeline imports OK')"
```

The standalone nnUNet import may print notices about undefined `nnUNet_raw`,
`nnUNet_preprocessed`, and `nnUNet_results`. The project launcher sets these
variables before inference, so those notices alone do not mean the import failed.

If an import reports a missing system OpenGL/OpenMP library on a minimal Ubuntu
installation, install the library named in the error. Common prerequisites are:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libgomp1
```

For GPU inference, install a suitable NVIDIA driver on the host, then check:

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

The normal nnUNet launcher defaults to CUDA and requires `CUDA available: True`.
For a CPU-only test, call DMACS directly with `--device cpu`, as shown in
Section 7. The current launcher runs PyTorch on the host, not inside the legacy
Docker container; its Docker commands do not request GPU passthrough.

## 4. Configure Docker on the new host

### 4.1 Install Docker Engine on a clean Ubuntu host

These commands target Ubuntu with systemd. For existing Docker installations,
first review conflicting packages in the
[official Docker Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/).
Do not run host installation commands inside the DMACS container.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

. /etc/os-release
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and log back in after the last command. Then verify access from the
same non-root user who will run the pipeline:

```bash
docker run --rm hello-world
```

The Python launcher invokes `docker` without `sudo`, so this must work.
Docker-group membership grants root-level privileges. Do not make the Docker
socket world-writable. See Docker's
[post-installation and security notes](https://docs.docker.com/engine/install/linux-postinstall/).

### 4.2 Obtain the exact DMACS image used on the source host

The source image has these identifiers:

```text
Tag:          jinmingduan/segmentationmeshmotion:latest
Platform:     linux/amd64
Image ID:     sha256:7b1d1b316a1b4e13653a1b036354875f55c01925b99567fd71f5dd7adbb186ff
Repo digest:  sha256:ff46d8c49197f561120a6a109a6fbdef46f9d92175e8145dffed9d5d18ce9f30
```

On the new host, pull the recorded digest and give it the tag expected by the
existing launcher:

```bash
IMAGE=jinmingduan/segmentationmeshmotion
DIGEST=sha256:ff46d8c49197f561120a6a109a6fbdef46f9d92175e8145dffed9d5d18ce9f30
docker pull --platform linux/amd64 "$IMAGE@$DIGEST"
docker tag "$IMAGE@$DIGEST" "$IMAGE:latest"
docker image inspect "$IMAGE:latest" --format '{{.Id}} {{.Architecture}}'
docker run --rm --network none -e LD_LIBRARY_PATH=/lib64 "$IMAGE:latest" python -V
```

The recorded image reports Python 2.7.15rc1. This legacy Python is separate
from Python 3.9.23 in the host Conda environment.

If that digest is no longer available remotely, migrate the existing local
image instead. This also avoids changes hidden behind the `latest` tag.

On the source host, in a directory with sufficient free space:

```bash
docker save --output dmacs_docker.tar jinmingduan/segmentationmeshmotion:latest
```

Transfer the archive and, on the new host, run:

```bash
docker load --input dmacs_docker.tar
```

Do not export a running container: use `docker save`/`docker load` for the image.
The archive is not included in this repository. A plain
`docker pull jinmingduan/segmentationmeshmotion:latest` is another option, but
it does not guarantee the same image if that tag changes.

No permanent DMACS container needs to be activated manually. The modern
DMACS launcher creates temporary containers with `docker run --rm`, mounts
`code_by_stage` at `/code` read-only and the selected subject at `/data`, and
sets `LD_LIBRARY_PATH=/lib64` inside the container. Activate Conda on the host
and let the launcher manage these mounts. Do not install the old TensorFlow
runtime into `cme-dt-env`.

## 5. Restore openCARP for UVC/fiber generation

Conda and the DMACS image do not supply the native openCARP installation used
by this project. For the closest migration, archive the complete directory on
the source host, preserving its bundled libraries and symbolic links:

```bash
tar -C /opt -czf openCARP.tar.gz openCARP
```

Transfer the archive to the new host. If `/opt/openCARP` already exists there,
inspect it first and do not overwrite an unrelated installation. On a fresh
host, restore and check it with:

```bash
sudo tar -C /opt -xzf openCARP.tar.gz
test -x /opt/openCARP/usr/bin/openCARP
test -x /opt/openCARP/usr/bin/meshtool
test -x /opt/openCARP/usr/bin/mpiexec
test -x /opt/openCARP/usr/bin/igbextract
ldd /opt/openCARP/usr/bin/openCARP
ldd /opt/openCARP/usr/bin/meshtool
/opt/openCARP/usr/bin/meshtool help
```

No required library in the `ldd` output should say `not found`. Do not move only
the four executables: they rely on the rest of the `/opt/openCARP` bundle.
The archive is not included here. Another installation is possible using the
[official openCARP installation guide](https://opencarp.org/download/installation),
but new versions and different install locations need their own validation.

`run_pipline.py` currently passes `--carp-bin-dir /opt/openCARP/usr/bin`.
If openCARP is installed elsewhere, update that argument in `STAGE_ARGS`,
or pass the actual directory when calling `generate_UVC_fiber.py` directly.
Keep the existing `.par` files and `model_vol_septum/amg_cg_opts` with the source.

## 6. Transfer model weights and data separately

Preserve the project layout, particularly `code_by_stage`, `model_vol_septum`,
`run_pipline.py`, and any already-generated stage outputs needed for resuming.
For the default nnUNet model, retain this directory and its contents:

```text
model_nnunet/nnunet_workspace/nnUNet_results/
  Dataset100_UKBB_Petersen_SAX/nnUNetTrainer__nnUNetPlans__2d/
    dataset.json
    plans.json
    fold_0/checkpoint_final.pth
    fold_1/checkpoint_final.pth
    fold_2/checkpoint_final.pth
    fold_3/checkpoint_final.pth
    fold_4/checkpoint_final.pth
```

The default inference uses all five folds. The launcher sets the nnUNet
workspace environment variables itself. Preserve `model/` too if you intend
to use the original TensorFlow backend; it is not the modern nnUNet model.
Input images belong in `data/subject_<ID>/example.nii`. Patient data, weights,
Docker storage, and openCARP are not packaged into `cme-dt-env.yml`.

## 7. Run after verification

From the project root on the new host:

```bash
conda activate cme-dt-env
python run_pipline.py --request segment --subject 11
python run_pipline.py --request raw_surface --subject 11
python run_pipline.py --request volumetric --subject 11
python run_pipline.py --request uvc_fiber --subject 11
```

Run these in order for a new subject, or resume only a stage whose inputs
already exist. `--request all` runs the same ordered workflow. The runner's
segmentation, raw-surface, and volumetric stages replace generated outputs;
use a test subject or backup first. Environment verification does not require
running these commands or processing patient data.

A CPU segmentation example, bypassing the runner's default CUDA selection:

```bash
python code_by_stage/DMACS.py --segmentation_backend=nnunet \
  --stage01_seg_only=true --subject subject_11 --device cpu
```

## 8. Validation scope

On the source host, the YAML was parsed and compared with the live
`conda env export -n cme-dt-env --no-builds` result, excluding only `prefix`.
All 217 Conda and 115 pip entries match. `python -m pip check` reports
`No broken requirements found.` A Conda dependency-solver dry run passed:

```bash
conda env create --name cme-dt-env-portability-check \
  --file environment/cme-dt-env.yml --dry-run
```

The dry run does not install or verify the pip section. The geometry, nnUNet,
and pipeline import checks passed in the existing source environment, and
all 15 Bash command blocks in this guide passed `bash -n` syntax checks.
The recorded Docker image was started only to check `python -V`; no subject
processing was run. The Docker tag, digest, architecture, and the four
openCARP executable paths were inspected locally.

These are source-host checks, not a claim that a fresh host has been tested.
Always run the import, GPU (if used), Docker, and native-library checks above
on the destination. Do not alter pipeline algorithms to work around a missing
runtime dependency.
