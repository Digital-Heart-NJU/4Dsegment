"""Generate an untagged tetrahedral mesh from the saved Step 08 boundary."""
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import tetgen

from model_vol_septum.fill_step01_surface_holes import topology, triangles
from model_vol_septum.mesh_intersections import confirmed_self_intersections


def load_boundary(path: Path):
    """Read triangle cells without changing geometry or carrying surface tags."""
    source = pv.read(path)
    if isinstance(source, pv.UnstructuredGrid):
        if not np.all(source.celltypes == pv.CellType.TRIANGLE):
            raise RuntimeError(f"Step 09 input must contain only triangles: {path}")
        mesh = pv.PolyData(np.asarray(source.points, dtype=float), source.cells)
    elif isinstance(source, pv.PolyData) and source.is_all_triangles:
        mesh = pv.PolyData(np.asarray(source.points, dtype=float), source.faces)
    else:
        raise RuntimeError(f"Step 09 requires a triangle surface: {path}")
    if not mesh.n_cells or not np.all(np.isfinite(mesh.points)):
        raise RuntimeError("Step 09 input is empty or contains non-finite coordinates.")
    if len(np.unique(np.sort(triangles(mesh), axis=1), axis=0)) != mesh.n_cells:
        raise RuntimeError("Step 09 input contains duplicate triangles.")
    stats = topology(mesh)
    for key in ("free_edges", "nonmanifold_edges", "inconsistent_orientation_edges", "degenerate_triangles"):
        if stats[key]:
            raise RuntimeError(f"Step 09 input {path.name}: {key}={stats[key]}; correct Step 08 first.")
    pairs = confirmed_self_intersections(mesh.points, triangles(mesh))
    if len(pairs):
        raise RuntimeError(f"Step 09 input {path.name}: {len(pairs)} self-intersections; triangle pairs {pairs[:10].tolist()}.")
    stats["self_intersection_pairs"] = 0
    return mesh, stats


def signed_volumes(points, cells):
    xyz = points[cells]
    return np.einsum("ij,ij->i", np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]),
                     xyz[:, 3] - xyz[:, 0]) / 6.


def validate_volume(grid: pv.UnstructuredGrid, surface: pv.PolyData):
    """Check tetrahedra, face connectivity, the full boundary and enclosed volume."""
    if not isinstance(grid, pv.UnstructuredGrid) or not grid.n_cells or not np.all(grid.celltypes == pv.CellType.TETRA):
        raise RuntimeError("Step 09 output must be a nonempty tetrahedron-only grid.")
    if len(grid.cell_data) or len(grid.point_data) or len(grid.field_data):
        raise RuntimeError("The untagged Step 09 mesh must not contain data arrays.")
    points = np.asarray(grid.points, dtype=float)
    cells = np.asarray(grid.cells).reshape(-1, 5)[:, 1:]
    if not np.all(np.isfinite(points)) or cells.min() < 0 or cells.max() >= len(points):
        raise RuntimeError("Step 09 output contains invalid coordinates or connectivity.")
    sorted_cells = np.sort(cells, axis=1)
    if np.any(np.diff(sorted_cells, axis=1) == 0) or len(np.unique(sorted_cells, axis=0)) != len(cells):
        raise RuntimeError("Step 09 output contains repeated vertices or duplicate tetrahedra.")
    volumes = signed_volumes(points, cells)
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise RuntimeError("Step 09 output contains inverted or degenerate tetrahedra.")

    all_faces = np.sort(np.vstack([cells[:, ids] for ids in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]), axis=1)
    faces, inverse, counts = np.unique(all_faces, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError("Step 09 tetrahedra have nonmanifold face connectivity.")
    boundary = faces[counts == 1]
    if not len(boundary):
        raise RuntimeError("Step 09 output has no exterior boundary.")
    boundary_ids = np.unique(boundary)
    distances, source_ids = cKDTree(surface.points).query(points[boundary_ids])
    tolerance = max(float(surface.length), 1.) * 1e-10
    if np.max(distances) > tolerance:
        raise RuntimeError("TetGen changed the saved Step 08 boundary geometry.")
    mapping = np.full(len(points), -1, dtype=np.int64)
    mapping[boundary_ids] = source_ids
    mapped_faces = np.sort(mapping[boundary], axis=1)
    expected = np.sort(triangles(surface), axis=1)
    if len(mapped_faces) != len(expected) or not np.array_equal(
            np.unique(mapped_faces, axis=0), np.unique(expected, axis=0)):
        raise RuntimeError("TetGen output boundary does not match the full Step 08 surface.")

    xyz = np.asarray(surface.points, dtype=float)[triangles(surface)] - np.mean(surface.points, axis=0)
    enclosed_volume = abs(float(np.einsum("ij,ij->i", xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum() / 6.))
    total_volume = float(volumes.sum())
    relative_error = abs(total_volume - enclosed_volume) / max(enclosed_volume, np.finfo(float).tiny)
    if relative_error > 1e-6:
        raise RuntimeError(f"TetGen volume differs from the Step 08 solid by {relative_error:.3g}; check cavity preservation.")

    # Shared triangle faces define volume connectivity, not mere vertex contact.
    owners = np.tile(np.arange(len(cells)), 4)
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
    a, b = owners[order[offsets]], owners[order[offsets + 1]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(cells), len(cells)))
    components = int(connected_components(graph, directed=False, return_labels=False))
    return {
        "points": int(len(points)), "tetrahedra": int(len(cells)),
        "component_count": components, "nonpositive_tetrahedra": 0,
        "minimum_tetrahedron_volume_mm3": float(volumes.min()),
        "maximum_tetrahedron_volume_mm3": float(volumes.max()),
        "total_tetrahedron_volume_mm3": total_volume,
        "input_solid_volume_mm3": enclosed_volume, "relative_volume_error": relative_error,
        "boundary_triangles": int(len(boundary)), "boundary_matches_step08": True,
        "maximum_boundary_point_error_mm": float(np.max(distances)),
        "point_data": [], "cell_data": [], "field_data": [],
    }


def generate_volume(input_path: Path, output_path: Path, *, minratio=1.2,
                    mindihedral=5., max_volume=0., verbose=False):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Step 09 output must differ from the Step 08 input.")
    if not np.isfinite([minratio, mindihedral, max_volume]).all() or minratio <= 0 or not 0 <= mindihedral < 180 or max_volume < 0:
        raise ValueError("Invalid TetGen quality or volume settings.")
    checksum = hashlib.sha256(input_path.read_bytes()).hexdigest()
    surface, input_validation = load_boundary(input_path)
    settings = dict(plc=True, quality=True, minratio=float(minratio), mindihedral=float(mindihedral),
                    nobisect=True, regionattrib=False, order=1,
                    quiet=not verbose, verbose=1 if verbose else 0)
    if max_volume > 0:
        settings.update(fixedvolume=True, maxvolume=float(max_volume))
    print(f"[Step09 TetGen] {input_path.name}: {surface.n_points} points, {surface.n_cells} triangles", flush=True)
    result = tetgen.TetGen(surface).tetrahedralize(**settings)
    points = np.asarray(result[0], dtype=float)
    cells = np.asarray(result[1], dtype=np.int64).copy()
    if cells.ndim != 2 or cells.shape[1] != 4 or not len(cells):
        raise RuntimeError("TetGen did not return linear tetrahedral elements.")
    if not np.all(np.isfinite(points)) or cells.min() < 0 or cells.max() >= len(points):
        raise RuntimeError("TetGen returned invalid points or element indices.")
    reverse = signed_volumes(points, cells) < 0
    cells[reverse, :2] = cells[reverse, 1::-1]
    grid = pv.UnstructuredGrid(np.column_stack((np.full(len(cells), 4), cells)).ravel(),
                               np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8), points)
    validation = validate_volume(grid, surface)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".step09_validate_", dir=output_path.parent) as directory:
        staged = Path(directory) / output_path.name
        grid.save(staged, binary=True)
        saved = pv.read(staged)
        if not np.array_equal(saved.points, grid.points) or not np.array_equal(saved.cells, grid.cells):
            raise RuntimeError("Saved Step 09 mesh changed coordinates or tetrahedra.")
        saved_validation = validate_volume(saved, surface)
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != checksum:
            raise RuntimeError("Step 08 input changed during Step 09.")
        staged.replace(output_path)
    return {
        "input_path": str(input_path.resolve()), "output_path": str(output_path.resolve()),
        "input_sha256": checksum, "step08_input_unchanged": True,
        "tetgen_version": tetgen.__version__, "tetgen_settings": settings,
        "input_validation": input_validation, "volume_validation": validation,
        "saved_volume_validation": saved_validation,
        "negative_tetrahedra_reoriented": int(np.count_nonzero(reverse)),
        "tagging_performed": False, "surface_repair_performed": False,
    }
