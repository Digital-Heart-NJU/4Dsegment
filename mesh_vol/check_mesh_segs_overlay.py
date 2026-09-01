#!/usr/bin/env python3
"""Create a 3D HTML overlay of segmentation contours on surface and volume meshes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv
from skimage import measure
import trimesh

LABEL_NAMES = {1: "LV", 2: "MYO", 3: "RV"}
LABEL_COLORS = {1: "#33a02c", 2: "#ff7f00", 3: "#1f78b4"}
AXIS_NAMES = {0: "axis0", 1: "axis1", 2: "axis2"}


def vertices_to_grid(vertices: np.ndarray, grid_size: int = 128) -> np.ndarray:
    """Map MorphiNet normalized mesh coordinates to the processed label grid."""
    grid = (np.asarray(vertices, dtype=np.float32) + 1.0) * 0.5 * float(grid_size - 1)
    return grid[:, [2, 1, 0]]


def load_label(path: Path) -> np.ndarray:
    label = nib.load(str(path)).get_fdata().astype(np.int16)
    if label.ndim != 3:
        raise ValueError(f"Expected a 3D label volume, got shape {label.shape}")
    return label


def load_obj_mesh(path: Path, grid_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    vertices = vertices_to_grid(mesh.vertices, grid_size=grid_size)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    info = {
        "path": str(path),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "source_bounds": np.asarray(mesh.bounds).tolist(),
        "grid_bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "is_watertight": bool(mesh.is_watertight),
    }
    return vertices, faces, info


def pyvista_faces_to_triangles(poly: pv.PolyData) -> np.ndarray:
    poly = poly.triangulate()
    faces = np.asarray(poly.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected triangulated PyVista PolyData faces")
    return faces[:, 1:4]


def load_volume_mesh(path: Path, grid_size: int, max_edges: int) -> tuple[np.ndarray, np.ndarray, dict, tuple[np.ndarray, np.ndarray]]:
    grid = pv.read(path)
    if not hasattr(grid, "celltypes"):
        raise ValueError("Expected an unstructured grid with tetrahedral cells")
    celltypes = np.asarray(grid.celltypes)
    if not np.all(celltypes == 10):
        unique, counts = np.unique(celltypes, return_counts=True)
        summary = {int(k): int(v) for k, v in zip(unique, counts)}
        raise ValueError(f"Expected only tetra cells (VTK type 10), got {summary}")

    boundary = grid.extract_surface().triangulate()
    boundary_vertices = vertices_to_grid(np.asarray(boundary.points), grid_size=grid_size)
    boundary_faces = pyvista_faces_to_triangles(boundary)

    all_vertices = vertices_to_grid(np.asarray(grid.points), grid_size=grid_size)
    cells = np.asarray(grid.cells, dtype=np.int64).reshape(-1, 5)
    tetra = cells[:, 1:5]
    edges = np.concatenate(
        [
            tetra[:, [0, 1]],
            tetra[:, [0, 2]],
            tetra[:, [0, 3]],
            tetra[:, [1, 2]],
            tetra[:, [1, 3]],
            tetra[:, [2, 3]],
        ],
        axis=0,
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    total_edges = int(len(edges))
    if max_edges > 0 and total_edges > max_edges:
        step = int(math.ceil(total_edges / max_edges))
        edges = edges[::step]

    info = {
        "path": str(path),
        "points": int(grid.n_points),
        "tetrahedra": int(grid.n_cells),
        "boundary_points": int(boundary.n_points),
        "boundary_faces": int(boundary.n_cells),
        "unique_tetra_edges": total_edges,
        "displayed_tetra_edges": int(len(edges)),
        "source_bounds": np.asarray(grid.bounds).reshape(3, 2).tolist(),
        "grid_bounds": [boundary_vertices.min(axis=0).tolist(), boundary_vertices.max(axis=0).tolist()],
    }
    return boundary_vertices, boundary_faces, info, (all_vertices, edges)


def select_slices(label: np.ndarray, axis: int, max_slices: int) -> list[int]:
    other_axes = tuple(i for i in range(3) if i != axis)
    areas = np.sum(label > 0, axis=other_axes)
    indices = np.where(areas > 0)[0]
    if indices.size == 0:
        return []
    if indices.size <= max_slices:
        return [int(i) for i in indices]
    positions = np.linspace(0, indices.size - 1, max_slices)
    picks = set(indices[np.round(positions).astype(int)].tolist())
    picks.add(int(indices[np.argmax(areas[indices])]))
    return sorted(int(i) for i in picks)


def contours_for_label_axis(label: np.ndarray, label_value: int, axis: int, slices: list[int]) -> tuple[list[float], list[float], list[float], int]:
    keep_axes = [i for i in range(3) if i != axis]
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    contour_count = 0
    for index in slices:
        mask2d = np.take(label == label_value, index, axis=axis).astype(np.uint8)
        if not mask2d.any():
            continue
        for contour in measure.find_contours(mask2d, 0.5):
            if len(contour) < 3:
                continue
            points = np.zeros((len(contour), 3), dtype=np.float32)
            points[:, axis] = float(index)
            points[:, keep_axes[0]] = contour[:, 0]
            points[:, keep_axes[1]] = contour[:, 1]
            xs.extend(points[:, 0].tolist() + [None])
            ys.extend(points[:, 1].tolist() + [None])
            zs.extend(points[:, 2].tolist() + [None])
            contour_count += 1
    return xs, ys, zs, contour_count


def add_mesh3d(fig, vertices, faces, name, color, opacity, row, col):
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=opacity,
            flatshading=False,
            lighting=dict(ambient=0.65, diffuse=0.75, specular=0.12),
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def edge_trace(edge_vertices: np.ndarray, edges: np.ndarray, name: str):
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    for a, b in edges:
        p0 = edge_vertices[a]
        p1 = edge_vertices[b]
        x.extend([float(p0[0]), float(p1[0]), None])
        y.extend([float(p0[1]), float(p1[1]), None])
        z.extend([float(p0[2]), float(p1[2]), None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line=dict(color="rgba(40,40,40,0.18)", width=1),
        hoverinfo="skip",
        visible="legendonly",
    )


def contour_trace(label_name: str, axis_name: str, color: str, x, y, z):
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=f"{label_name} contours {axis_name}",
        legendgroup=f"{label_name}-{axis_name}",
        line=dict(color=color, width=4),
        hoverinfo="skip",
    )


def build_figure(label, surface, volume, volume_edges, selected_slices):
    surface_vertices, surface_faces, _surface_info = surface
    volume_vertices, volume_faces, _volume_info = volume
    edge_vertices, edges = volume_edges

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Surface mesh OBJ + segmentation contours", "Volumetric tetra mesh boundary + segmentation contours"),
        horizontal_spacing=0.02,
    )
    add_mesh3d(fig, surface_vertices, surface_faces, "Surface mesh OBJ", "rgb(90,140,235)", 0.28, 1, 1)
    add_mesh3d(fig, volume_vertices, volume_faces, "Volume mesh boundary", "rgb(160,160,170)", 0.28, 1, 2)
    fig.add_trace(edge_trace(edge_vertices, edges, "Sampled tetra edges"), row=1, col=2)

    contour_summary = {}
    for label_value, label_name in LABEL_NAMES.items():
        contour_summary[label_name] = {}
        for axis, slices in selected_slices.items():
            x, y, z, contour_count = contours_for_label_axis(label, label_value, axis, slices)
            contour_summary[label_name][AXIS_NAMES[axis]] = {
                "selected_slices": slices,
                "contour_polylines": contour_count,
                "plot_points_including_breaks": len(x),
            }
            if contour_count == 0:
                continue
            trace = contour_trace(label_name, AXIS_NAMES[axis], LABEL_COLORS[label_value], x, y, z)
            fig.add_trace(trace, row=1, col=1)
            trace2 = contour_trace(label_name, AXIS_NAMES[axis], LABEL_COLORS[label_value], x, y, z)
            trace2.showlegend = False
            fig.add_trace(trace2, row=1, col=2)

    scene_common = dict(
        xaxis=dict(title="axis 0", range=[0, 127]),
        yaxis=dict(title="axis 1", range=[0, 127]),
        zaxis=dict(title="axis 2", range=[0, 127]),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.6, y=1.5, z=1.15)),
    )
    fig.update_layout(
        title=(
            "3D projection of processed segmentation contours on MorphiNet surface and volumetric meshes"
            "<br><sup>Contours are stacked slice contours in the processed 128^3 grid; meshes are mapped from normalized MorphiNet coordinates.</sup>"
        ),
        scene=scene_common,
        scene2=scene_common,
        template="plotly_white",
        legend=dict(itemsizing="constant"),
        width=1700,
        height=900,
        margin=dict(l=0, r=0, t=90, b=0),
    )
    return fig, contour_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="results/results_acdc_single_label_resnet_metrics/acdc_anchor/acdc_anchor_gt_decoder_grid.nii.gz")
    parser.add_argument("--surface-obj", default="results/results_acdc_single_label_meshonly/acdc/MorphiNet/myo/f0/acdc_anchor.obj")
    parser.add_argument("--volume-vtk", default="results/results_acdc_single_label_meshonly/volume_mesh/acdc_anchor/FINAL_acdc_anchor_myo_volume.vtk")
    parser.add_argument("--output-dir", default="results/acdc_anchor_3d_contour_projection")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--max-slices", type=int, default=12)
    parser.add_argument("--max-tetra-edges", type=int, default=12000)
    args = parser.parse_args()

    label_path = Path(args.label)
    surface_path = Path(args.surface_obj)
    volume_path = Path(args.volume_vtk)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = load_label(label_path)
    selected_slices = {axis: select_slices(label, axis, args.max_slices) for axis in range(3)}
    surface = load_obj_mesh(surface_path, args.grid_size)
    volume_boundary_vertices, volume_boundary_faces, volume_info, volume_edges = load_volume_mesh(
        volume_path, args.grid_size, args.max_tetra_edges
    )
    volume = (volume_boundary_vertices, volume_boundary_faces, volume_info)

    fig, contour_summary = build_figure(label, surface, volume, volume_edges, selected_slices)
    html_path = output_dir / "acdc_anchor_3d_segmentation_contours_on_surface_and_volume_mesh.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)

    label_counts = {str(int(v)): int(np.sum(label == v)) for v in sorted(np.unique(label).tolist())}
    summary = {
        "label_path": str(label_path),
        "label_shape": list(label.shape),
        "label_counts": label_counts,
        "coordinate_mapping": "grid_vertices=((mesh_vertices + 1) * 0.5 * 127)[:, [2, 1, 0]]",
        "selected_slices": {AXIS_NAMES[k]: v for k, v in selected_slices.items()},
        "surface_mesh": surface[2],
        "volumetric_mesh": volume_info,
        "contours": contour_summary,
        "outputs": {"html": str(html_path)},
    }
    summary_path = output_dir / "acdc_anchor_3d_segmentation_contours_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_html={html_path}")
    print(f"saved_summary={summary_path}")
    print(f"label_shape={label.shape}")
    print(f"surface_vertices_faces={surface[2]['vertices']}/{surface[2]['faces']}")
    print(f"volume_boundary_vertices_faces={volume_info['boundary_points']}/{volume_info['boundary_faces']}")
    print(f"displayed_tetra_edges={volume_info['displayed_tetra_edges']} of {volume_info['unique_tetra_edges']}")


if __name__ == "__main__":
    main()