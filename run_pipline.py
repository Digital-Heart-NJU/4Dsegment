"""Run one stage or all configured stages for a subject or inclusive range:

python run_pipline.py --request segment --subject 11
python run_pipline.py --request raw_surface --subject 11
python run_pipline.py --request volumetric --subject 11
python run_pipline.py --request volumetric --subject 23 --epi-clearance-method constrained
python run_pipline.py --request volumetric --subject 23 --epi-clearance-method normal --epi-local-repair-method pymeshfix
python run_pipline.py --request uvc_fiber --subject 11

python run_pipline.py --request all --subject 11-18

Requires Conda and the project-local ./cme-dt-env; DMACS stages also require running Docker.
Batch jobs run sequentially.
Segmentation, raw-surface, and volumetric stages replace generated outputs.
uvc_fiber resumes completed coordinate/fiber stages and writes UVC_fiber_Outputs.
With --request all, it regenerates them after volumetric meshing.
raw_surface also clears subject-root *.txt files through the original Stage 02
routine. Inputs/manual files are retained.

actual_command in DMACS.py:
conda run --no-capture-output --prefix ./cme-dt-env \
  python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_11 \
  --stage02_surf_only=true --seg_manual=false

actual_command in generate_open_volmesh.py:
conda run --no-capture-output --prefix ./cme-dt-env python -B -u \
  model_vol_septum/generate_open_volmesh.py \
  --subject-dir data/subject_11 \
  --frames ED,ES \
  --overwrite

actual_command in generate_UVC_fiber.py:
conda run --no-capture-output --prefix ./cme-dt-env python -B -u \
  model_vol_septum/generate_UVC_fiber.py \
  --data-dir data --subject subject_11 --frames ED,ES \
  --carp-bin-dir /opt/openCARP/usr/bin --uvc-fiber
"""

import argparse
import math
from pathlib import Path
import shutil
import subprocess


# Environment directory next to this script, independent of named Conda envs.
CONDA_ENV = "cme-dt-env"

# Edit these arguments to adjust each stage's settings.
STAGE_ARGS = {
    "segment": ["--stage01_seg_only=true",
                "--overwrite"],
    "raw_surface": ["--stage02_surf_only=true",
                    "--seg_manual=false"],
    "volumetric": ["--frames", "ED,ES",
                   "--overwrite",
                   # "--epi-clearance-method=sdf",
                   "--epi-clearance-method=auto",
                   "--epi-local-repair=true", "--epi-min-accepted-step=0.01",
                   "--step8-short-edge-collapse=true", "--step8-collapse-tolerances", "0.01", "0.001", "0.0001",
                   "--tag1-mm", "10", "--tag0-mm", "12"],
    "uvc_fiber": ["--uvc-fiber",
                  "--frames", "ED,ES",
                  "--carp-bin-dir", "/opt/openCARP/usr/bin"],
}

# --request all runs these stages in this order for each subject.
REQUEST_ORDER = ["segment", "raw_surface", "volumetric", "uvc_fiber"]


def subject_range(value):
    try:
        first, separator, last = value.partition("-")
        first = int(first)
        last = int(last) if separator else first
        if first < 1 or last < first:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Use a positive subject ID or ascending range, e.g. 11 or 11-18.")
    return range(first, last + 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request", required=True, choices=list(STAGE_ARGS) + ["all"])
    parser.add_argument("--subject", required=True, type=subject_range, help="Subject ID or inclusive range: 11 or 11-18.")
    parser.add_argument("--epi-clearance-method", choices=("auto", "normal", "sdf", "constrained"),
                        help="Override the volumetric Step 03 method configured in STAGE_ARGS.")
    parser.add_argument("--epi-local-repair-method", choices=("damping", "pymeshfix", "constrained"),
                        help="Override normal-mode Step 03 repair; pymeshfix or constrained enables connectivity-preserving recovery.")
    parser.add_argument("--epi-min-accepted-step", type=float,
                        help="Override the normal-mode minimum accepted step in mm; must be positive and finite.")
    parser.add_argument("--epi-freeze-satisfied", choices=("true", "false"), nargs="?", const="true",
                        help="Lock clearance-satisfying vertices at optimizer/local repair entry; default false.")
    args = parser.parse_args(argv)
    if args.epi_min_accepted_step is not None and (not math.isfinite(args.epi_min_accepted_step) or args.epi_min_accepted_step <= 0):
        parser.error("--epi-min-accepted-step must be positive and finite.")
    requests = REQUEST_ORDER if args.request == "all" else [args.request]

    root = Path(__file__).resolve().parent
    dmacs = root / "code_by_stage" / "DMACS.py"
    scripts = {
        "segment": dmacs,
        "raw_surface": dmacs,
        "volumetric": root / "model_vol_septum" / "generate_open_volmesh.py",
        "uvc_fiber": root / "model_vol_septum" / "generate_UVC_fiber.py",
    }
    output_folders = {"segment": "segs", "raw_surface": "vtks", "volumetric": "volmesh_pipline",
                      "uvc_fiber": "UVC_fiber_Outputs"}
    conda = shutil.which("conda")
    if not conda:
        raise SystemExit("Conda was not found. Run this from a terminal with Conda available.")
    for request in requests:
        if not scripts[request].is_file():
            raise SystemExit("Cannot find: " + str(scripts[request]))

    environment_prefix = root / CONDA_ENV
    if not (environment_prefix / "conda-meta" / "history").is_file() or not (environment_prefix / "bin" / "python").is_file():
        raise SystemExit("Project Conda environment is missing or incomplete: " + str(environment_prefix) +
                         ". Create it using environment/cme-dt-env.yml and --prefix " + str(environment_prefix))
    python_command = [conda, "run", "--no-capture-output", "--prefix", str(environment_prefix),
                      "python", "-B", "-u"]
    print("Python environment: " + str(environment_prefix), flush=True)

    completed, failed = [], []
    for number in args.subject:
        subject = "subject_" + str(number)
        subject_dir = root / "data" / subject
        if not subject_dir.is_dir():
            print(subject + ": folder is missing; skipped.", flush=True)
            failed.append(subject)
            continue
        if "segment" in requests and not (subject_dir / "example.nii").is_file():
            print(subject + ": example.nii is missing; skipped.", flush=True)
            failed.append(subject)
            continue

        try:
            for request in requests:
                command = python_command + [str(scripts[request])]
                if request == "volumetric":
                    command += ["--subject-dir", str(subject_dir)]
                elif request == "uvc_fiber":
                    command += ["--data-dir", str(root / "data"), "--subject", subject]
                    if "volumetric" in requests:
                        command += ["--force"]
                else:
                    command += ["--segmentation_backend=nnunet",
                                "--data-dir", str(root / "data"), "--subject", subject]
                print("\n" + subject + ": starting " + request, flush=True)
                stage_args = list(STAGE_ARGS[request])
                if request == "volumetric" and args.epi_clearance_method:
                    stage_args += ["--epi-clearance-method", args.epi_clearance_method]
                if request == "volumetric" and args.epi_local_repair_method:
                    stage_args += ["--epi-local-repair-method", args.epi_local_repair_method]
                if request == "volumetric" and args.epi_min_accepted_step is not None:
                    stage_args += ["--epi-min-accepted-step", str(args.epi_min_accepted_step)]
                if request == "volumetric" and args.epi_freeze_satisfied is not None:
                    stage_args += ["--epi-freeze-satisfied", args.epi_freeze_satisfied]
                subprocess.run(command + stage_args, cwd=root, check=True)
                output = subject_dir / output_folders[request]
                print(subject + ": " + request + " finished. Outputs: " + str(output), flush=True)
        except subprocess.CalledProcessError as error:
            print(subject + ": " + request + " failed (exit " + str(error.returncode) + "); continuing.", flush=True)
            failed.append(subject)
            continue
        completed.append(subject)

    print("\nCompleted: " + (", ".join(completed) or "none"), flush=True)
    print("Failed/skipped: " + (", ".join(failed) or "none"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
