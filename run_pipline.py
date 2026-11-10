"""Run one stage or all configured stages for a subject or inclusive range:

python run_pipline.py --request segment --subject 11
python run_pipline.py --request raw_surface --subject 11
python run_pipline.py --request volumetric --subject 11

python run_pipline.py --request all --subject 11-18

Requires Conda and cme-dt-env; DMACS stages also require running Docker.
Batch jobs run sequentially.
Existing generated outputs are replaced. raw_surface also clears subject-root
*.txt files through the original Stage 02 routine. Inputs/manual files are retained.

actual_command in DMACS.py:
conda run --no-capture-output -n cme-dt-env \
  python -B -u code_by_stage/DMACS.py \
  --segmentation_backend=nnunet --subject subject_11 \
  --stage02_surf_only=true --seg_manual=false

actual_command in generate_open_volmesh_stable.py:
conda run --no-capture-output -n cme-dt-env python -B -u \
  model_vol_septum/generate_open_volmesh_stable.py \
  --subject-dir data/subject_11 \
  --frames ED,ES \
  --step6-clearance-method signed
"""

import argparse
from pathlib import Path
import shutil
import subprocess


CONDA_ENV = "cme-dt-env"

# Edit these arguments to adjust each stage's settings.
STAGE_ARGS = {
    "segment": ["--stage01_seg_only=true", "--overwrite"],
    "raw_surface": ["--stage02_surf_only=true", "--seg_manual=false"],
    "volumetric": ["--frames", "ED,ES", "--step6-clearance-method", "signed"],
}

# --request all runs these stages in this order for each subject.
REQUEST_ORDER = ["segment", "raw_surface", "volumetric"]


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
    args = parser.parse_args(argv)
    requests = REQUEST_ORDER if args.request == "all" else [args.request]

    root = Path(__file__).resolve().parent
    dmacs = root / "code_by_stage" / "DMACS.py"
    scripts = {
        "segment": dmacs,
        "raw_surface": dmacs,
        "volumetric": root / "model_vol_septum" / "generate_open_volmesh_stable.py",
    }
    output_folders = {"segment": "segs", "raw_surface": "vtks", "volumetric": "volmesh_pipline"}
    conda = shutil.which("conda")
    if not conda:
        raise SystemExit("Conda was not found. Run this from a terminal with Conda available.")
    for request in requests:
        if not scripts[request].is_file():
            raise SystemExit("Cannot find: " + str(scripts[request]))

    python_command = [conda, "run", "--no-capture-output", "-n", CONDA_ENV,
                      "python", "-B", "-u"]

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
                else:
                    command += ["--segmentation_backend=nnunet",
                                "--data-dir", str(root / "data"), "--subject", subject]
                print("\n" + subject + ": starting " + request, flush=True)
                subprocess.run(command + STAGE_ARGS[request], cwd=root, check=True)
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
