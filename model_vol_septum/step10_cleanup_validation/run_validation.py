"""Rerun Step 10 on the six existing Step 09 meshes and audit saved artifacts."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model_vol_septum import generate_open_volmesh_stable as pipeline
from model_vol_septum.tag_step10_volume import shared_face_neighbors, tag_components
from model_vol_septum.generate_step09_volume import signed_volumes


def main():
    with patch.object(sys, "argv", ["pipeline"]):
        args = pipeline.parse_args()
    report = []
    for subject in ("subject_1", "subject_2", "subject_3"):
        root = ROOT / "data" / subject / "volmesh_pipline_signed_clearance"
        for phase in ("ED", "ES"):
            prefix = f"{subject}_{phase}"
            source = root / "step_09_volmesh_generate" / f"{prefix}_volmesh_notag.vtk"
            reference = root / "step_02_surf_clean" / f"{prefix}_LVepi.vtk"
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, reference)}
            print(f"Running {prefix}", flush=True)
            paths, summary = pipeline.save_step10_final_mesh(
                root / "step_10_final_mesh", source, reference, subject, phase, args)
            original, before, final = (pv.read(path) for path in
                                       (source, paths["before_tag_cleanup"], paths["volmesh"]))
            for saved in (before, final):
                for field in ("points", "cells", "celltypes"):
                    np.testing.assert_array_equal(getattr(saved, field), getattr(original, field))
                assert saved["elemTag"].dtype == np.int32
                assert set(np.unique(saved["elemTag"])) == {0, 1}
            np.testing.assert_array_equal(final["tagChanged"], before["elemTag"] != final["elemTag"])
            cells = original.cells.reshape(-1, 5)[:, 1:]
            volumes = signed_volumes(original.points, cells)
            assert np.all(volumes > 0)
            neighbors = shared_face_neighbors(cells)
            measured_before = tag_components(cells, before["elemTag"], volumes, neighbors)
            measured_after = tag_components(cells, final["elemTag"], volumes, neighbors)
            assert measured_before == summary["regions_before_cleanup"]
            assert measured_after == summary["regions"]
            for name in ("LV", "RV"):
                assert measured_after[name]["component_count"] == 1
            for path, digest in hashes.items():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            changed = int(np.count_nonzero(final["tagChanged"]))
            assert changed == summary["tag_cleanup"]["relabeled_cells"]
            record = {
                "subject": subject, "phase": phase, "status": "passed", "paths": paths,
                "changed_cells": changed,
                "components_before": {name: region["component_count"] for name, region in measured_before.items()},
                "components_after": {name: region["component_count"] for name, region in measured_after.items()},
                "unresolved_components": summary["tag_cleanup"]["unresolved_components"],
                "geometry_connectivity_order_preserved": True, "inputs_unchanged": True,
                "tag_changed_flags_verified": True,
            }
            report.append(record)
            (Path(__file__).parent / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"PASS {prefix}: {record['components_before']} -> {record['components_after']}; "
                  f"changed cells={changed}", flush=True)


if __name__ == "__main__":
    main()
