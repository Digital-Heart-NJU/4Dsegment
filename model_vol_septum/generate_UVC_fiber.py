'''
python run.py --data-dir /home/mengao/4t/Dicemesh --carp-bin-dir /opt/openCARP/usr/bin --uvc-fiber --subject subject001
python run.py --data-dir /home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/UKB_example_4s --carp-bin-dir /opt/openCARP/usr/bin --uvc-fiber --subject subject004
'''

import argparse
import configparser
import numpy as np
import pandas as pd
import pyvista as pv
import os
import itertools
import logging
import re
import shutil
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "model_vol_septum"

from . import compute_UVC_fiber as UVC
from . import meshIO
from . import meshtool_func as meshtool
from . import py_atrial_fibres as paf
from . import v_mesh_generation as vmesh

logger = logging.getLogger('volumetric')

CARP_BIN = None
OPTS = None
INPUT_PREFIX = None


def get_timeframe_n(filename):
    return int(re.search(r'timeframe(\d+)', filename).group(1))


def timeframes(subject_id, input_dir):
    vtk_files = [f for f in os.listdir(input_dir) if re.search(r'timeframe\d+\.vtk$', f)]

    # sort by timeframe number then alphabetically
    vtk_files.sort(key=lambda s: (get_timeframe_n(s), s))

    for n, vtk_files_n in itertools.groupby(vtk_files, get_timeframe_n):
        mesh_names = [filename[:-4] for filename in vtk_files_n]
        # print("line 42")
        print("mesh_names:", mesh_names)

        # ---- 仍然期望 8 个表面网格文件（输入数据未变） ----
        if len(mesh_names) != 8:
            logger.error('subject {}, {}: incorrect number of surface meshes'.format(subject_id, n))
            continue

        yield n, mesh_names


PHO_OUTPUT_NAMES = (
    "PHO.case",
    "PHO.dat",
    "PHO.fiber.ens",
    "PHO.geo",
    "PHO.tags.ens",
    "PHO_PHO.ens",
)

PHI_OUTPUT_NAMES = (
    "PHI.case",
    "PHI.dat",
    "PHI.fiber.ens",
    "PHI.geo",
    "PHI.tags.ens",
    "PHI_PHI.ens",
)

COORDS_Z_OUTPUT_NAMES = (
    "COORDS_Z.case",
    "COORDS_Z.dat",
    "COORDS_Z.fiber.ens",
    "COORDS_Z.geo",
    "COORDS_Z.tags.ens",
    "COORDS_Z_COORDS_Z.ens",
)

BIV_OUTPUT_NAMES = ("biv.dat",)

FIBER_OUTPUT_NAMES = ("subject_mesh_all_fiber.vtk",)


def pipeline_frame_paths(subject_dir, frame):
    """Resolve saved pipeline inputs without copying or renaming source files."""
    subject_dir = Path(subject_dir).resolve()
    prefix = f"{INPUT_PREFIX or subject_dir.name}_{frame}"
    root = subject_dir / "volmesh_pipline"
    surfaces = root / "step_07_reconstruct_annulus"
    return {
        "volume": root / "step_10_final_mesh" / f"{prefix}_volmesh.vtk",
        "LVendo": surfaces / f"{prefix}_LV_openbase_endo.vtk",
        "LVepi": surfaces / f"{prefix}_LV_openbase_epi.vtk",
        "LVannulus": surfaces / f"{prefix}_LV_openbase_annulus_recon.vtk",
        "RVendo": surfaces / f"{prefix}_RV_openbase_endo.vtk",
        "RVepi": surfaces / f"{prefix}_RV_openbase_epi_moved.vtk",
        "RVannulus": surfaces / f"{prefix}_RV_openbase_annulus_recon.vtk",
        "interface": surfaces / f"{prefix}_septum.vtk",
        "epi": root / "step_08_LVRV_surf_clean" / f"{prefix}_LVRV_surf.vtk",
    }


def require_frame_paths(paths, names):
    missing = [str(paths[name]) for name in names
               if not paths[name].is_file() or not paths[name].stat().st_size]
    if missing:
        raise FileNotFoundError(f"Required pipeline inputs do not exist or are empty: {missing}")


def prepare_working_mesh(source_volume, frame_output, meshname, meshtoolloc, force=False):
    """Convert an ASCII working copy; legacy meshtool misreads binary doubles."""
    working_mesh = frame_output / meshname
    working_vtk = working_mesh.with_suffix(".vtk")
    source = pv.read(source_volume)
    if not isinstance(source, pv.UnstructuredGrid) or not np.all(np.isfinite(source.points)):
        raise ValueError(f"Invalid UVC volume mesh: {source_volume}")
    carp_paths = [working_mesh.with_suffix(suffix) for suffix in (".elem", ".pts", ".lon")]
    converted = force or not all(path.is_file() and path.stat().st_size for path in carp_paths)
    if not converted:
        points = np.asarray(meshIO.read_pts(basename=str(working_mesh)), dtype=float)
        converted = points.shape != source.points.shape or not np.allclose(
            points, source.points, rtol=0.0, atol=1e-4)
    if converted or not working_vtk.is_file():
        source.points = np.asarray(source.points, dtype=np.float64)
        source.save(working_vtk, binary=False)
    if converted:
        if not meshtool.convert_mesh(meshtoolloc, str(working_mesh), str(working_mesh),
                                     ifmt="vtk", ofmt="carp_txt"):
            raise RuntimeError(f"meshtool failed to convert the volume mesh: {source_volume}")
    missing = [str(path) for path in carp_paths if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError(f"Missing CARP files after VTK conversion: {missing}")
    points = np.asarray(meshIO.read_pts(basename=str(working_mesh)), dtype=float)
    if points.shape != source.points.shape or not np.allclose(points, source.points, rtol=0.0, atol=1e-4):
        raise RuntimeError("Converted CARP coordinates do not match the source volume mesh.")
    return working_mesh, converted


def frame_names(subject_dir, frame=None, frames=None):
    if frame and frames:
        raise ValueError("Use either --frame or --frames, not both.")
    if frame:
        requested = [frame.upper()]
    elif frames:
        requested = [value.strip().upper() for value in frames.split(',') if value.strip()]
    else:
        requested = [name for name in ("ED", "ES")
                     if pipeline_frame_paths(subject_dir, name)["volume"].is_file()]
    if not requested:
        raise RuntimeError(f"No ED/ES tagged meshes found under {subject_dir}/volmesh_pipline/step_10_final_mesh.")
    invalid = [name for name in requested if name not in ("ED", "ES")]
    if invalid:
        raise ValueError(f"Unsupported frame names: {invalid}. Expected ED and/or ES.")
    return list(dict.fromkeys(requested))


def run_frame_coords_z(subject_id, frame, data_dir, force=False):
    subject_dir = Path(data_dir).resolve() / subject_id
    inputs = pipeline_frame_paths(subject_dir, frame)
    require_frame_paths(inputs, ("volume", "LVannulus", "RVannulus"))
    volume_vtk, lv_annulus, rv_annulus = (inputs[name] for name in ("volume", "LVannulus", "RVannulus"))

    frame_output = subject_dir / "UVC_fiber_Outputs" / frame
    uvc_dir = frame_output / "UVC_i"
    frame_output.mkdir(parents=True, exist_ok=True)
    uvc_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [uvc_dir / name for name in COORDS_Z_OUTPUT_NAMES]
    if all(path.is_file() for path in expected_outputs) and not force:
        logger.info("subject %s, frame %s: skipping COORDS_Z because all outputs exist", subject_id, frame)
        return {"skipped": True, "outputs": [str(path) for path in expected_outputs]}

    meshtoolloc = os.path.join(CARP_BIN, "meshtool")
    if not os.path.isfile(meshtoolloc):
        raise FileNotFoundError(f"meshtool executable does not exist: {meshtoolloc}")

    meshname = subject_id + "_mesh_all"
    working_mesh, converted = prepare_working_mesh(volume_vtk, frame_output, meshname, meshtoolloc, force)

    elements = meshIO.read_elems(basename=str(working_mesh))
    tags = set(np.asarray(elements.iloc[:, 4], dtype=np.int64).tolist())
    if not {0, 1}.issubset(tags):
        raise RuntimeError(f"Converted volume mesh must contain RV tag 0 and LV tag 1; found {sorted(tags)}.")

    coords_z_path, details = UVC.coord_z_openbase_annuli(
        str(frame_output),
        meshtoolloc,
        meshname,
        str(volume_vtk),
        str(lv_annulus),
        str(rv_annulus),
        "surfaces",
        str(uvc_dir),
    )
    missing_outputs = [str(path) for path in expected_outputs if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f"COORDS_Z generation did not create the required files: {missing_outputs}")

    summary = {
        "subject": subject_id,
        "frame": frame,
        "converted_vtk_to_carp": converted,
        "working_mesh": str(working_mesh),
        "COORDS_Z_dat": coords_z_path,
        "outputs": [str(path.resolve()) for path in expected_outputs],
        "details": details,
    }
    summary_path = frame_output / f"{subject_id}_{frame}_COORDS_Z_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("subject %s, frame %s: generated COORDS_Z", subject_id, frame)
    return summary


def run_frame_biv(subject_id, frame, data_dir, force=False):
    """Generate the nodal LV/RV biventricular coordinate for one frame."""
    subject_dir = Path(data_dir).resolve() / subject_id
    inputs = pipeline_frame_paths(subject_dir, frame)
    require_frame_paths(inputs, ("volume",))
    source_volume = inputs["volume"]

    frame_output = subject_dir / "UVC_fiber_Outputs" / frame
    uvc_dir = frame_output / "UVC_i"
    frame_output.mkdir(parents=True, exist_ok=True)
    uvc_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [uvc_dir / name for name in BIV_OUTPUT_NAMES]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected_outputs) and not force:
        logger.info("subject %s, frame %s: skipping biv because output exists", subject_id, frame)
        return {"skipped": True, "outputs": [str(path) for path in expected_outputs]}

    meshtoolloc = os.path.join(CARP_BIN, "meshtool")
    if not os.path.isfile(meshtoolloc):
        raise FileNotFoundError(f"meshtool executable does not exist: {meshtoolloc}")

    meshname = subject_id + "_mesh_all"
    working_mesh, converted = prepare_working_mesh(source_volume, frame_output, meshname, meshtoolloc, force)

    elements = meshIO.read_elems(basename=str(working_mesh))
    tags = set(np.asarray(elements.iloc[:, 4], dtype=np.int64).tolist())
    if not {0, 1}.issubset(tags):
        raise RuntimeError(
            f"Biventricular mesh must contain RV tag 0 and LV tag 1; found {sorted(tags)}."
        )

    biv_path = UVC.bivlabel(
        str(frame_output), meshname, str(uvc_dir), source_meshname=meshname
    )
    values = np.loadtxt(biv_path, dtype=np.int64, ndmin=1)
    point_count = len(meshIO.read_pts(basename=str(working_mesh)))
    if len(values) != point_count:
        raise RuntimeError(f"biv.dat length does not match the mesh point count: {biv_path}")
    unique_values = set(np.unique(values).tolist())
    if not unique_values.issubset({-1, 1}):
        raise RuntimeError(f"biv.dat contains unexpected labels: {sorted(unique_values)}")
    if not {-1, 1}.issubset(unique_values):
        raise RuntimeError(f"biv.dat does not contain both LV and RV labels: {biv_path}")

    logger.info(
        "subject %s, frame %s: generated biv.dat with LV=%d, RV=%d nodes",
        subject_id,
        frame,
        int(np.count_nonzero(values == -1)),
        int(np.count_nonzero(values == 1)),
    )
    return {
        "subject": subject_id,
        "frame": frame,
        "converted_vtk_to_carp": converted,
        "working_mesh": str(working_mesh),
        "biv_dat": str(Path(biv_path).resolve()),
        "lv_nodes": int(np.count_nonzero(values == -1)),
        "rv_nodes": int(np.count_nonzero(values == 1)),
        "outputs": [str(path.resolve()) for path in expected_outputs],
    }


def run_frame_fiber(subject_id, frame, data_dir, force=False):
    """Generate LV/RV fibers and merge them into the complete ED/ES mesh."""
    subject_dir = Path(data_dir).resolve() / subject_id
    inputs = pipeline_frame_paths(subject_dir, frame)
    require_frame_paths(inputs, ("volume",))
    source_volume = inputs["volume"]

    frame_output = subject_dir / "UVC_fiber_Outputs" / frame
    uvc_dir = frame_output / "UVC_i"
    fiber_work = frame_output / "FIBER_work"
    fiber_folder = frame_output / "fiberfolder"
    frame_output.mkdir(parents=True, exist_ok=True)
    uvc_dir.mkdir(parents=True, exist_ok=True)
    fiber_work.mkdir(parents=True, exist_ok=True)
    fiber_folder.mkdir(parents=True, exist_ok=True)

    meshname = subject_id + "_mesh_all"
    final_vtk = fiber_folder / f"{meshname}_fiber.vtk"
    if final_vtk.is_file() and final_vtk.stat().st_size > 0 and not force:
        logger.info("subject %s, frame %s: skipping fibers because output exists", subject_id, frame)
        return {"skipped": True, "outputs": [str(final_vtk.resolve())]}

    meshtoolloc = os.path.join(CARP_BIN, "meshtool")
    if not os.path.isfile(meshtoolloc):
        raise FileNotFoundError(f"meshtool executable does not exist: {meshtoolloc}")

    working_mesh, converted = prepare_working_mesh(source_volume, frame_output, meshname, meshtoolloc, force)

    coords_z_path = uvc_dir / "COORDS_Z.dat"
    pho_path = uvc_dir / "PHO.dat"
    missing_fields = [str(path) for path in (coords_z_path, pho_path) if not path.is_file() or path.stat().st_size == 0]
    if missing_fields:
        raise RuntimeError(f"Fiber generation requires existing coordinate fields: {missing_fields}")

    # PHI preparation already creates the chamber-local meshes, base loops,
    # and apex landmarks needed by compute_fiber_new.
    chamber_inputs = {
        "LV": ("LVmyo_i", "LV_base", "LV_apex.vtx"),
        "RV": ("RVmyo_i", "RV_base", "RV_apex.vtx"),
    }
    required_work = []
    for mesh_prefix, base_prefix, apex_name in chamber_inputs.values():
        required_work.extend(
            fiber_work / name
            for name in (
                f"{mesh_prefix}.elem", f"{mesh_prefix}.pts", f"{mesh_prefix}.lon",
                f"{mesh_prefix}.eidx", f"{base_prefix}.surf", apex_name,
            )
        )
    if force or not all(path.is_file() and path.stat().st_size > 0 for path in required_work):
        phi_work = frame_output / "PHI_work"
        if not all(path.is_file() and path.stat().st_size > 0 for path in (
            phi_work / name for mesh_prefix, base_prefix, apex_name in chamber_inputs.values()
            for name in (f"{mesh_prefix}.elem", f"{mesh_prefix}.pts", f"{mesh_prefix}.lon", f"{mesh_prefix}.eidx", f"{base_prefix}.surf", apex_name)
        )):
            run_frame_phi(subject_id, frame, data_dir, force=True)
        fiber_work.mkdir(parents=True, exist_ok=True)
        for mesh_prefix, base_prefix, apex_name in chamber_inputs.values():
            for name in (
                f"{mesh_prefix}.elem", f"{mesh_prefix}.pts", f"{mesh_prefix}.lon",
                f"{mesh_prefix}.eidx", f"{mesh_prefix}.nod", f"{base_prefix}.surf", apex_name,
            ):
                source = phi_work / name
                if source.is_file():
                    shutil.copy2(source, fiber_work / name)

    missing_work = [str(path) for path in required_work if not path.is_file() or path.stat().st_size == 0]
    if missing_work:
        raise RuntimeError(f"Missing chamber fiber inputs after PHI preparation: {missing_work}")

    alpha_epi = -np.pi / 3
    alpha_endo = np.pi / 3
    beta_epi = 2 * np.pi * 25 / 360
    beta_endo = -2 * np.pi * 65 / 360
    chamber_fiber_bases = {}
    for chamber, (mesh_prefix, base_prefix, apex_name) in chamber_inputs.items():
        result = UVC.compute_fiber_new(
            str(fiber_work),
            meshtoolloc,
            mesh_prefix,
            apex_name,
            base_prefix,
            str(pho_path.resolve()),
            str(coords_z_path.resolve()),
            alpha_epi,
            alpha_endo,
            beta_epi,
            beta_endo,
        )
        if result is None:
            raise RuntimeError(f"{chamber} fiber generation failed.")
        chamber_fiber_bases[chamber] = fiber_work / f"{mesh_prefix}_fiber"

    full_elements = meshIO.read_elems(basename=str(working_mesh))
    element_count = len(full_elements)
    full_lon = np.zeros((element_count, 8), dtype=float)
    assigned = np.zeros(element_count, dtype=bool)
    for chamber, (mesh_prefix, _, _) in chamber_inputs.items():
        eidx_path = fiber_work / f"{mesh_prefix}.eidx"
        eidx = np.fromfile(eidx_path, dtype=np.int64)
        local_lon = np.asarray(
            meshIO.read_fibres(basename=str(chamber_fiber_bases[chamber])), dtype=float
        )
        if len(eidx) != len(local_lon):
            raise RuntimeError(
                f"{chamber} fiber mapping has {len(eidx)} element indices but "
                f"{len(local_lon)} fiber rows."
            )
        if len(eidx) and (eidx.min() < 0 or eidx.max() >= element_count):
            raise RuntimeError(f"{chamber} fiber element mapping is outside the full mesh range.")
        if np.any(assigned[eidx]):
            raise RuntimeError(f"LV/RV fiber element mappings overlap for {chamber}.")
        if not np.all(np.isfinite(local_lon)):
            raise RuntimeError(f"{chamber} fiber vectors contain non-finite values.")
        full_lon[eidx] = local_lon
        assigned[eidx] = True

    if not np.all(assigned):
        raise RuntimeError(f"Fiber merge left {int(np.count_nonzero(~assigned))} elements without fibers.")
    longitudinal_norms = np.linalg.norm(full_lon[:, 0:3], axis=1)
    sheet_norms = np.linalg.norm(full_lon[:, 5:8], axis=1)
    if np.any(longitudinal_norms <= 1e-8) or np.any(sheet_norms <= 1e-8):
        raise RuntimeError("Merged fibers contain zero-length longitudinal or sheet vectors.")

    final_base = fiber_folder / f"{meshname}_fiber"
    shutil.copy2(working_mesh.with_suffix(".elem"), final_base.with_suffix(".elem"))
    shutil.copy2(working_mesh.with_suffix(".pts"), final_base.with_suffix(".pts"))
    meshIO.write_lon_includesheet(str(final_base), pd.DataFrame(full_lon))
    if not meshtool.convert_mesh(
        meshtoolloc, str(final_base), str(final_base), ifmt="carp_txt", ofmt="vtk"
    ):
        raise RuntimeError("meshtool failed to convert the merged fiber mesh to VTK.")
    if not final_vtk.is_file() or final_vtk.stat().st_size == 0:
        raise RuntimeError(f"Merged fiber VTK was not created: {final_vtk}")

    source_grid = pv.read(str(source_volume))
    final_grid = pv.read(str(final_vtk))
    if final_grid.n_points != source_grid.n_points or final_grid.n_cells != source_grid.n_cells:
        raise RuntimeError(
            f"Merged fiber mesh topology differs from the source: "
            f"source=({source_grid.n_points}, {source_grid.n_cells}), "
            f"fiber=({final_grid.n_points}, {final_grid.n_cells})."
        )

    summary = {
        "subject": subject_id,
        "frame": frame,
        "source_volume": str(source_volume.resolve()),
        "converted_vtk_to_carp": converted,
        "fiber_mesh": str(final_vtk.resolve()),
        "element_count": int(element_count),
        "LV_element_count": int(np.count_nonzero(full_elements.iloc[:, 4].astype(int).to_numpy() == 1)),
        "RV_element_count": int(np.count_nonzero(full_elements.iloc[:, 4].astype(int).to_numpy() == 0)),
        "longitudinal_norm_range": [float(np.min(longitudinal_norms)), float(np.max(longitudinal_norms))],
        "sheet_norm_range": [float(np.min(sheet_norms)), float(np.max(sheet_norms))],
        "units_preserved": True,
        "outputs": [str(final_vtk.resolve()), str(final_base.with_suffix(".lon").resolve())],
    }
    summary_path = frame_output / f"{subject_id}_{frame}_FIBER_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("subject %s, frame %s: generated and merged complete fiber mesh", subject_id, frame)
    return summary


def run_frame_landmarks(subject_id, frame, data_dir, force=False):
    """Use the Step 07 interface curve with the existing ANT/INF selection rule."""
    subject_dir = Path(data_dir).resolve() / subject_id
    inputs = pipeline_frame_paths(subject_dir, frame)
    require_frame_paths(inputs, ("volume", "LVannulus", "RVannulus", "interface"))
    output = subject_dir / "UVC_fiber_Outputs" / frame
    output.mkdir(parents=True, exist_ok=True)
    ant = output / f"{subject_id}_{frame}_RV_INSERT_ANT.vtk"
    inf = output / f"{subject_id}_{frame}_RV_INSERT_INF.vtk"
    if not force and all(path.is_file() and path.stat().st_size for path in (ant, inf)):
        return {"skipped": True, "outputs": [str(ant), str(inf)]}
    coords = output / "UVC_i" / "COORDS_Z.dat"
    if not coords.is_file():
        run_frame_coords_z(subject_id, frame, data_dir, force=force)
    details = UVC.find_two_RV_insertion_pts_openbase(
        str(inputs["volume"]), str(coords), str(inputs["LVannulus"]),
        str(inputs["RVannulus"]), str(inputs["interface"]), str(ant), str(inf),
    )
    for name in ("LV", "RV"):
        pv.PolyData(np.asarray(details[f"{name.lower()}_centroid"]).reshape(1, 3)).save(
            output / f"{subject_id}_{frame}_{name}_annulus_centroid.vtk", binary=False,
        )
    summary = {"subject": subject_id, "frame": frame,
               "interface_curve": str(inputs["interface"]), "details": details,
               "outputs": [str(ant), str(inf)]}
    (output / f"{subject_id}_{frame}_landmarks_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_frame_phi(subject_id, frame, data_dir, force=False):
    """Generate PHI for an open-base ED/ES mesh using the current surface inputs."""
    subject_dir = Path(data_dir).resolve() / subject_id
    inputs = pipeline_frame_paths(subject_dir, frame)
    require_frame_paths(inputs, ("volume", "LVannulus", "RVannulus"))
    volume_vtk, lv_annulus, rv_annulus = (inputs[name] for name in ("volume", "LVannulus", "RVannulus"))
    ant_landmark = subject_dir / "UVC_fiber_Outputs" / frame / f"{subject_id}_{frame}_RV_INSERT_ANT.vtk"
    inf_landmark = subject_dir / "UVC_fiber_Outputs" / frame / f"{subject_id}_{frame}_RV_INSERT_INF.vtk"
    if not all(path.is_file() for path in (ant_landmark, inf_landmark)):
        run_frame_landmarks(subject_id, frame, data_dir, force=force)
    required = (volume_vtk, lv_annulus, rv_annulus, ant_landmark, inf_landmark)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required PHI input does not exist: {missing}")

    frame_output = subject_dir / "UVC_fiber_Outputs" / frame
    uvc_dir = frame_output / "UVC_i"
    frame_output.mkdir(parents=True, exist_ok=True)
    uvc_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [uvc_dir / name for name in PHI_OUTPUT_NAMES]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected_outputs) and not force:
        logger.info("subject %s, frame %s: skipping PHI because all outputs exist", subject_id, frame)
        return {"skipped": True, "outputs": [str(path) for path in expected_outputs]}

    meshtoolloc = os.path.join(CARP_BIN, "meshtool")
    if not os.path.isfile(meshtoolloc):
        raise FileNotFoundError(f"meshtool executable does not exist: {meshtoolloc}")
    meshname = subject_id + "_mesh_all"
    working_mesh, converted = prepare_working_mesh(volume_vtk, frame_output, meshname, meshtoolloc, force)

    coords_z_path = uvc_dir / "COORDS_Z.dat"
    if not coords_z_path.is_file() or coords_z_path.stat().st_size == 0:
        run_frame_coords_z(subject_id, frame, data_dir, force=force)
    if not coords_z_path.is_file() or coords_z_path.stat().st_size == 0:
        raise RuntimeError(f"COORDS_Z is not available for PHI: {coords_z_path}")

    phi_path, details = UVC.compute_phi_openbase_annuli(
        str(frame_output), meshtoolloc, meshname, str(volume_vtk), str(coords_z_path),
        str(lv_annulus), str(rv_annulus), str(ant_landmark), str(inf_landmark), str(uvc_dir)
    )
    missing_outputs = [str(path) for path in expected_outputs if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f"PHI generation did not create valid files: {missing_outputs}")
    summary = {
        "subject": subject_id,
        "frame": frame,
        "converted_vtk_to_carp": converted,
        "working_mesh": str(working_mesh),
        "PHI_dat": phi_path,
        "outputs": [str(path.resolve()) for path in expected_outputs],
        "details": details,
    }
    summary_path = frame_output / f"{subject_id}_{frame}_PHI_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("subject %s, frame %s: generated PHI", subject_id, frame)
    return summary



def run_frame_pho(subject_id, frame, data_dir, force=False):
    """Generate PHO from the open-base LV/RV surfaces for one ED/ES frame."""
    subject_dir = Path(data_dir).resolve() / subject_id
    inputs = pipeline_frame_paths(subject_dir, frame)
    source_volume = inputs["volume"]
    surface_paths = {name: inputs[name] for name in
                     ("LVendo", "LVannulus", "RVendo", "RVannulus", "RVepi", "LVepi", "epi")}
    missing = [str(path) for path in (source_volume, *surface_paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required PHO input does not exist: {missing}")

    frame_output = subject_dir / "UVC_fiber_Outputs" / frame
    uvc_dir = frame_output / "UVC_i"
    frame_output.mkdir(parents=True, exist_ok=True)
    uvc_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [uvc_dir / name for name in PHO_OUTPUT_NAMES]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected_outputs) and not force:
        logger.info("subject %s, frame %s: skipping PHO because all outputs exist", subject_id, frame)
        return {"skipped": True, "outputs": [str(path) for path in expected_outputs]}

    meshtoolloc = os.path.join(CARP_BIN, "meshtool")
    carp = os.path.join(CARP_BIN, "openCARP")
    mpi = os.path.join(CARP_BIN, "mpiexec")
    igbextract = os.path.join(CARP_BIN, "igbextract")
    for executable in (meshtoolloc, carp, mpi, igbextract):
        if not os.path.isfile(executable):
            raise FileNotFoundError(f"Required CARP executable does not exist: {executable}")

    meshname = subject_id + "_mesh_all"
    working_mesh, converted = prepare_working_mesh(source_volume, frame_output, meshname, meshtoolloc, force)
    working_vtk = working_mesh.with_suffix(".vtk")

    par_file = str(Path(__file__).resolve().with_name("transmural.par"))
    pho_path, details = UVC.transmural_openbase(
        str(frame_output),
        meshtoolloc,
        meshname,
        str(working_vtk),
        {name: str(path) for name, path in surface_paths.items()},
        mpi,
        OPTS,
        carp,
        par_file,
        igbextract,
        str(uvc_dir),
    )
    missing_outputs = [str(path) for path in expected_outputs if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f"PHO generation did not create valid files: {missing_outputs}")
    summary = {
        "subject": subject_id,
        "frame": frame,
        "converted_vtk_to_carp": converted,
        "working_mesh": str(working_mesh),
        "PHO_dat": pho_path,
        "outputs": [str(path.resolve()) for path in expected_outputs],
        "details": details,
    }
    summary_path = frame_output / f"{subject_id}_{frame}_PHO_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("subject %s, frame %s: generated PHO", subject_id, frame)
    return summary


def run_frame_all(subject_id, frame, data_dir, force=False):
    """Run the ED/ES coordinate and fiber stages in dependency order."""
    subject_dir = Path(data_dir).resolve() / subject_id
    inputs = pipeline_frame_paths(subject_dir, frame)
    require_frame_paths(inputs, tuple(inputs))
    initial_directory = Path.cwd()
    results = {}
    try:
        for name, operation in (
            ("COORDS_Z", run_frame_coords_z), ("landmarks", run_frame_landmarks),
            ("PHI", run_frame_phi), ("PHO", run_frame_pho),
            ("biv", run_frame_biv), ("fiber", run_frame_fiber),
        ):
            logger.info("subject %s, frame %s: starting %s", subject_id, frame, name)
            results[name] = operation(subject_id, frame, data_dir, force=force)
        summary = {"subject": subject_id, "frame": frame,
                   "inputs": {name: str(path) for name, path in inputs.items()},
                   "stages": results}
        output = subject_dir / "UVC_fiber_Outputs" / frame
        (output / f"{subject_id}_{frame}_UVC_fiber_summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        os.chdir(initial_directory)


def run_timeframe_uvc(subject_id, timeframe_num, input_dir, mesh_names, par_file, output_dir, output_dir_uvc):
    logger.debug('subject {}, timeframe {}: started uvc'.format(subject_id, timeframe_num))

    meshtoolloc = os.path.join(CARP_BIN, 'meshtool')

    CARP = os.path.join(CARP_BIN, 'openCARP')
    IGBEXTRACT = os.path.join(CARP_BIN, 'igbextract')
    MPIEXEC = os.path.join(CARP_BIN, 'mpiexec')

    meshdir = os.path.join(output_dir, subject_id + '_mesh_all') #volumetric_mesh_outputs/timeframe/subject_id_mesh_all
    meshname = subject_id + "_mesh_all"
    surfloc = "surfaces"

    if not os.path.exists(meshdir):
        logger.error('subject {}, timeframe {}: no volumetric mesh found'.format(subject_id, timeframe_num))
        return

    finalmeshdir = output_dir_uvc   #UVC_fiber_Outputs/timeframe

    os.chdir(finalmeshdir)
    if not os.path.exists(meshdir + '/' + meshname + ".elem"):  # volumetric_mesh_outputs/timeframe/subject_id_mesh_all
        logger.error('subject {}, timeframe {}: no volumetric mesh found'.format(subject_id, timeframe_num))
        return

    shutil.copyfile(meshdir + '/' + meshname + ".elem", meshname + ".elem")
    shutil.copyfile(meshdir + '/' + meshname + ".pts", meshname + ".pts")
    shutil.copyfile(meshdir + '/' + meshname + ".lon", meshname + ".lon")
    shutil.copyfile(meshdir + '/' + meshname + ".vtk", meshname + ".vtk")

    UVCfolder = finalmeshdir + '/UVC_i'
    if not os.path.exists(UVCfolder):
        os.mkdir(UVCfolder)

    # UVC.coord z generation
    try:
        # (finalmeshdir:UVC_fiber_Outputs/timeframe,
        # meshtoolloc:meshtool,
        # meshname:subject_mesh_all,
        # input_dir:Mesh_Outputs
        # surfloc:surfaces,
        # UVCfolder:UVC_fiber_Outputs/timeframe/UVC_i)
        coordszname = UVC.coord_z(finalmeshdir, meshtoolloc, meshname, input_dir, mesh_names, surfloc, UVCfolder)

    except Exception:
        logger.error('subject {}, timeframe {}: failed to generate coord z'.format(subject_id, timeframe_num))
        return
    if coordszname == None:
        logger.error('subject {}, timeframe {}: failed to generate coord z'.format(subject_id, timeframe_num))
        return

    # UVC.rotational coords generation:LV
    # (finalmeshdir:UVC_fiber_Outputs/timeframe,
    # meshname:subject_mesh_all,
    # surfloc:surfaces,
    # meshtoolloc:meshtool,
    # coordszname:UVC_fiber_Outputs/timeframe/UVC_i/COORDS_Z.dat)
    # coordszname = UVCfolder + "/COORDS_Z.dat"
    if not UVC.find_two_RV_insertion_pts(finalmeshdir, meshtoolloc, meshname, input_dir, mesh_names, surfloc, coordszname):
        logger.error('subject {}, timeframe {}: failed to find RV insertion points'.format(subject_id, timeframe_num))
        return
    predata = UVC.extract_surf_forLV(finalmeshdir, meshtoolloc, meshname, input_dir, mesh_names, surfloc)
    if predata == None:
        logger.error('subject {}, timeframe {}: failed to extract surface for LV'.format(subject_id, timeframe_num))
        return
    phiLV = 'Phi_LV.dat'
    PhiLV = UVC.compute_uvc_rotational(finalmeshdir, meshtoolloc, 'LVmyo_i', predata[0], predata[1], predata[2], predata[3], phiLV)

    # UVC.rotational coords generation:RV
    predataRV = UVC.extract_surf_forRV(finalmeshdir, meshtoolloc, meshname, input_dir, mesh_names, surfloc, coordszname)
    if predataRV == None:
        logger.error('subject {}, timeframe {}: failed to extract surface for RV'.format(subject_id, timeframe_num))
        return
    phiRV = 'Phi_RV.dat'
    PhiRV = UVC.compute_uvc_rotationalRV(finalmeshdir, meshtoolloc, 'RVmyo_i', predataRV[0], predataRV[1], predataRV[2], predataRV[3], phiRV)
    # merge UVC.rotational coords (PHI) files of LV and RV
    if UVC.merge_PHI(finalmeshdir, meshtoolloc, meshname, phiLV, phiRV, surfloc, UVCfolder) == None:
        logger.error('subject {}, timeframe {}: failed to merge phi'.format(subject_id, timeframe_num))
        return

    # UVC.transmural coords generation
    SIMID = "transmural_results"
    meshtool.convert_mesh(meshtoolloc, f'{input_dir}/Mesh_aorta_valve_timeframe{timeframe_num:03}', f'Mesh_aorta_valve_timeframe{timeframe_num:03}', 'vtk')
    pts5810 = meshIO.read_pts(basename=f'Mesh_aorta_valve_timeframe{timeframe_num:03}')
    PHOname = UVC.transmural(finalmeshdir, meshtoolloc, meshname, pts5810, input_dir, mesh_names, surfloc, MPIEXEC, OPTS, CARP, par_file,SIMID, IGBEXTRACT, UVCfolder)
    if PHOname == None:
        logger.error('subject {}, timeframe {}: failed to generate transmural coordinates'.format(subject_id, timeframe_num))
        return

    # UVC.biventricular coords (LV:-1 and RV:1)
    UVC.bivlabel(finalmeshdir, meshname, UVCfolder)

    # #########################################
    # fiber generation
    alpha_epi = -np.pi / 3
    alpha_endo = np.pi / 3

    beta_epi = 2 * np.pi * 25 / 360
    beta_endo = -2 * np.pi * 65 / 360

    # for LV
    apexvtx = predata[1]
    base_surf = predata[0]
    LV_fiber = UVC.compute_fiber_new(finalmeshdir, meshtoolloc, "LVmyo_i", apexvtx, base_surf, PHOname, coordszname,
                                     alpha_epi, alpha_endo, beta_epi, beta_endo)
    if LV_fiber == None:
        logger.error('subject {}, timeframe {}: failed to generate LV fiber'.format(subject_id, timeframe_num))
        return
    # for RV
    apexvtx = predataRV[1]
    base_surf = predataRV[0]
    RV_fiber = UVC.compute_fiber_new(finalmeshdir, meshtoolloc, "RVmyo_i", apexvtx, base_surf, PHOname, coordszname,
                                     alpha_epi, alpha_endo, beta_epi, beta_endo)
    if RV_fiber == None:
        logger.error('subject {}, timeframe {}: failed to generate RV fiber'.format(subject_id, timeframe_num))
        return

    # merge fiber files and save to folder 'fiberfolder'
    fiberfolder = finalmeshdir + '/fiberfolder'
    if not os.path.exists(fiberfolder):
        os.mkdir(fiberfolder)
    if not UVC.merge_fibertomyoonly(finalmeshdir, meshtoolloc, meshname, fiberfolder):
        logger.error('subject {}, timeframe {}: failed to merge fiber'.format(subject_id, timeframe_num))
        return

    logger.debug('subject {}, timeframe {}: finished uvc'.format(subject_id, timeframe_num))

def main(do_mesh=False, do_uvc=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--profile', '-p', action='store', default='default', help='config profile to be used')
    parser.add_argument('--job', '-j', action='store', default='default', help='job identifier')
    parser.add_argument('--force', '-f', action='store_true', help='force overwrite')

    parser.add_argument('--data-dir', '-d', action='store', help='path to data directory')
    parser.add_argument('--input-dir', '-I', action='store', default='Mesh_Outputs', help='name of input directories')
    parser.add_argument('--carp-bin-dir', '-c', action='store', help='path to carp bin directory')

    parser.add_argument('--instance', '-i', type=int, action='store', default=2, help='instance to be processed')

    parser.add_argument('--all', '-a', action='store_true', help='process all subjects')
    parser.add_argument('--subject', '-s', action='store', help='subject id to be processed')
    parser.add_argument('--input-prefix', help='Explicit input filename subject prefix for copied pipeline folders; outputs still use --subject.')
    parser.add_argument('--start', '-S', action='store', type=int, help='index of first subject id to be processed')
    parser.add_argument('--number', '-n', action='store', type=int, help='number of subjects to be processed from first subject id')
    parser.add_argument('--allowlist', '-l', action='store', help='path to subject allowlist')

    parser.add_argument('--timeframe', '-t', type=int, action='store', help='timeframe to be processed')
    parser.add_argument('--frame', choices=['ED', 'ES', 'ed', 'es'], help='ED/ES frame for the new COORDS_Z workflow')
    parser.add_argument('--frames', help='comma-separated ED/ES frames for the new COORDS_Z workflow')

    parser.add_argument('--mesh', '-m', action='store_true', help='generate mesh')
    parser.add_argument('--uvc', '-u', action='store_true', help='generate uvc and fiber')
    parser.add_argument('--uvc-meshes', '-z', action='store_true', help='generate uvc meshes')
    parser.add_argument('--quality-control', '-q', action='store_true', help='output quality control screenshots')
    parser.add_argument('--coords-z', action='store_true', help='generate only COORDS_Z from open-base LV/RV annulus surfaces')
    parser.add_argument('--phi', action='store_true', help='generate only PHI from open-base LV/RV annulus surfaces')
    parser.add_argument('--pho', action='store_true', help='generate only PHO from open-base LV/RV boundary surfaces')
    parser.add_argument('--biv', action='store_true', help='generate only the nodal LV/RV biv.dat field')
    parser.add_argument('--fiber', action='store_true', help='generate and merge frame-based LV/RV fibers')
    parser.add_argument('--uvc-fiber', action='store_true', help='run all ED/ES stages directly from volmesh_pipline')

    args, _ = parser.parse_known_args()

    cfg = configparser.ConfigParser()
    cfg.read('config.ini')

    global CARP_BIN
    CARP_BIN = str(Path(args.carp_bin_dir if args.carp_bin_dir else cfg[args.profile]['CarpBinDir']).resolve())
    global INPUT_PREFIX
    INPUT_PREFIX = args.input_prefix

    global OPTS
    OPTS = os.path.abspath('amg_cg_opts')
    if not os.path.isfile(OPTS):
        OPTS = str(Path(__file__).resolve().with_name('amg_cg_opts'))

    data_dir = str(Path(args.data_dir if args.data_dir else cfg[args.profile]['DataDir']).resolve())
    par_file = os.path.abspath('LVendo_RVseptum_RVendo.par')
    uvc_par_file = os.path.abspath('transmural.par')

    if args.allowlist and os.path.exists(args.allowlist):
        with open(args.allowlist) as f:
            sids = [n for n in f.read().splitlines() if os.path.isdir(os.path.join(data_dir, n))]
    else:
        sids = [n for n in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, n))]

    if args.all:
        subject_ids = sorted(sids)
    elif args.subject:
        sid = args.subject
        if sid in sids:
            subject_ids = [sid]
        else:
            subject_ids = []
    elif args.start is not None and args.start >= 0 and args.start < len(sids):
        if args.number is not None and args.number > 0:
            end = args.start + args.number - 1
            subject_ids = sorted(sids)[args.start:end + 1]
        else:
            subject_ids = sorted(sids)[args.start:]
    else:
        subject_ids = []

    log_filename = os.path.join(data_dir, f'volumetric-{args.job}.log')
    formatter = logging.Formatter(fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    np.seterr(all='warn')

    if do_uvc or args.uvc_fiber:
        results, failures = [], []
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            for frame in frame_names(subject_dir, args.frame, args.frames):
                try:
                    results.append(run_frame_all(subject_id, frame, data_dir, args.force))
                except Exception as error:
                    logger.exception("subject %s, frame %s: UVC/fiber generation failed", subject_id, frame)
                    failures.append({"subject": subject_id, "frame": frame, "error": str(error)})
        if failures:
            raise RuntimeError(f"UVC/fiber generation failed for: {failures}")
        return results

    if args.coords_z:
        results = []
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            for frame in frame_names(subject_dir, args.frame, args.frames):
                results.append(run_frame_coords_z(subject_id, frame, data_dir, args.force))
        return results

    if args.phi:
        results = []
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            for frame in frame_names(subject_dir, args.frame, args.frames):
                results.append(run_frame_phi(subject_id, frame, data_dir, args.force))
        return results

    if args.pho:
        results = []
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            for frame in frame_names(subject_dir, args.frame, args.frames):
                results.append(run_frame_pho(subject_id, frame, data_dir, args.force))
        return results

    if args.biv:
        results = []
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            for frame in frame_names(subject_dir, args.frame, args.frames):
                results.append(run_frame_biv(subject_id, frame, data_dir, args.force))
        return results

    if args.fiber:
        results = []
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            for frame in frame_names(subject_dir, args.frame, args.frames):
                results.append(run_frame_fiber(subject_id, frame, data_dir, args.force))
        return results

    for subject_id in subject_ids:
        if not os.path.isdir(os.path.join(data_dir, subject_id)):
            continue

        input_dir = os.path.join(data_dir, subject_id, f'Instance_{args.instance}', f'{args.input_dir}') # input_dir为Mesh_Outputs文件夹
        print("line 512:", input_dir)
        output_dir = os.path.join(data_dir, subject_id, f'Instance_{args.instance}', 'volumetric_mesh_outputs')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir_uvc = os.path.join(data_dir, subject_id, f'Instance_{args.instance}', 'UVC_fiber_Outputs')
        if not os.path.exists(output_dir_uvc):
            os.mkdir(output_dir_uvc)

        for tf_n, mesh_names in timeframes(subject_id, input_dir):
            if (args.timeframe is not None and args.timeframe != tf_n):
                continue

            if do_uvc or args.uvc:
                uftf_dir = os.path.join(output_dir_uvc, 'timeframe{}'.format(tf_n)) #UVC_fiber_Outputs/timeframe
                if not os.path.exists(uftf_dir):
                    os.mkdir(uftf_dir)

                if os.path.exists(
                        os.path.join(uftf_dir, 'fiberfolder', subject_id + '_mesh_all_fiber.vtk')) and not args.force:
                    logger.debug('subject {}, timeframe {}: skipping, fiber mesh exists'.format(subject_id, tf_n))
                    continue

                vmtf_dir = os.path.join(output_dir, 'timeframe{}'.format(tf_n)) #volumetric_mesh_outputs/timeframe

                run_timeframe_uvc(subject_id, tf_n, input_dir, mesh_names, uvc_par_file, vmtf_dir, uftf_dir)

# if __name__ == '__main__':
#     main()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    main()

# command:
'''
python generate_UVC_fiber.py --data-dir /home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/UKB_example_4s --carp-bin-dir /opt/openCARP/usr/bin --uvc-fiber --subject subject0081
'''
