import numpy as np
from volumetric import meshIO
from scipy.spatial import cKDTree
import os
import numpy as np
import os
import pandas as pd
import pyvista as pv
from scipy.spatial import KDTree

# 找寻左心室单独结构的瓣环位置
def find_intersection_base(valve_files, mesh, basesurf_path):
    annulus_coords = []
    for pts_path, topo_path in valve_files:
        # 1. 读取节点坐标
        coords = np.array(meshIO.read_pts(basename=pts_path.replace('.pts', '')))

        # 2. 【修复点】精准解析 CARP .elem 格式
        # 跳过第一行（面片数量），读取为字符串，提取第2~4列作为节点索引
        raw_tris = np.loadtxt(topo_path, dtype=str, skiprows=1)
        tris = raw_tris[:, 1:4].astype(int)

        # 3. 提取边界边（仅被1个三角形使用的边 = 瓣环）
        edges = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        boundary_verts = np.unique(boundary_edges.flatten())
        annulus_coords.append(coords[boundary_verts])

    # 4. 合并四个瓣膜的瓣环坐标
    valve_annulus = np.vstack(annulus_coords)
    print(f"✅ 提取的瓣膜瓣环边界点数: {len(valve_annulus)}")

    # 5. 读取纯心肌网格(mesh)的节点坐标
    pts_biv = np.array(meshIO.read_pts(basename=mesh))

    # 6. 空间匹配：仅匹配瓣环边界
    tree = cKDTree(valve_annulus)
    dists, _ = tree.query(pts_biv)
    threshold = 1.5  # 收紧阈值，匹配1D环状边界
    base_node_indices = np.where(dists < threshold)[0]
    print(f"✅ 匹配到的心肌基底节点数: {len(base_node_indices)}")

    # 7. 写入底座节点文件
    os.makedirs(os.path.dirname(basesurf_path), exist_ok=True)
    # 1. 写入 .vtx 文件（节点索引，供可视化或 -evtx 使用）
    with open(basesurf_path + ".surf.vtx", 'w') as f_vtx:
        f_vtx.write(f"{len(base_node_indices)}\nextra\n")
        for idx in base_node_indices:
            f_vtx.write(f"{int(idx)}\n")
    # 2. 生成 .surf 文件（供 meshtool -esurf 读取）
    # 标准 CARP  workaround：将每个点写成退化三角形 Tr i i i 0
    # meshtool 解析时会自动提取唯一节点作为距离场边界
    with open(basesurf_path + ".surf", 'w') as f_surf:
        f_surf.write(f"{len(base_node_indices)}\n")
        for idx in base_node_indices:
            f_surf.write(f"Tr {idx} {idx} {idx} \n")

    print(f"✅ 已生成: {basesurf_path}.vtx")
    print(f"✅ 已生成: {basesurf_path}.surf")
    return True

def save_pts_for_vis(mesh, basesurf_path,save_name):
    # 1. 加载 mesh 的点和底座索引
    pts = np.array(meshIO.read_pts(basename=mesh))
    idx = np.loadtxt(basesurf_path + ".vtx", skiprows=2, dtype=int)  # 如果存的是1-based，这里减1
    # 2. 提取底座坐标
    base_pts = pts[idx]
    # 3. 保存为临时 .pts 供 ParaView 查看
    np.savetxt(surfloc + f"/{save_name}.pts", base_pts, fmt="%.6f")
    return True

def read_surface_from_vtk(vtk_path):
    """从 VTK Polydata 中提取表面网格实际使用的顶点坐标"""
    with open(vtk_path, 'r') as f:
        # 过滤空行和注释，简化解析
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]

    # 1. 定位 POINTS 并读取全部坐标
    pts_line = next(i for i, l in enumerate(lines) if l.startswith('POINTS'))
    num_pts = int(lines[pts_line].split()[1])

    coord_vals = []
    idx = pts_line + 1
    while len(coord_vals) < 3 * num_pts:
        coord_vals.extend(lines[idx].split())
        idx += 1
    all_coords = np.array(coord_vals[:3 * num_pts], dtype=np.float32).reshape(-1, 3)

    # 2. 定位 POLYGONS 并提取三角形顶点索引
    poly_line = next(i for i, l in enumerate(lines[idx:], start=idx) if l.startswith('POLYGONS'))
    surface_indices = []
    for line in lines[poly_line + 1:]:
        parts = list(map(int, line.split()))
        if len(parts) >= 4 and parts[0] == 3:  # VTK格式: 3 v1 v2 v3
            surface_indices.extend(parts[1:4])

    # ⚠️ VTK 索引始终为 0-based，直接映射
    unique_idx = np.unique(surface_indices)
    surface_coords = pd.DataFrame(all_coords[unique_idx])
    return surface_coords


def extract_fine_by_distance(coarse_file, fine_file, output_prefix, min_dist=0.0, max_dist=5.0, max_angle_deg=45.0, mode='within'):
    """
    Extract fine mesh surface based on distance AND normal angle relative to coarse mesh.
    Outputs: .vtp, .nod, .surf, .surf.vtx
    """
    coarse = pv.read(coarse_file)
    fine = pv.read(fine_file)

    # Helper: ensure mesh is a surface (PolyData)
    def ensure_polydata(mesh):
        if isinstance(mesh, pv.UnstructuredGrid):
            print(f"  -> Converting {mesh.__class__.__name__} to surface...")
            return mesh.extract_surface()
        return mesh

    coarse = ensure_polydata(coarse)
    fine = ensure_polydata(fine)

    # Ensure pure triangles
    if not coarse.is_all_triangles: coarse = coarse.triangulate()
    if not fine.is_all_triangles: fine = fine.triangulate()

    # 1️⃣ Compute point normals
    print("Computing surface normals...")
    fine_normals = fine.compute_normals(point_normals=True, cell_normals=False)['Normals']
    coarse_normals = coarse.compute_normals(point_normals=True, cell_normals=False)['Normals']

    # 2️⃣ Compute distance to coarse surface
    print("Computing distances...")
    temp = fine.compute_implicit_distance(coarse)
    dist_keys = [k for k in temp.point_data.keys() if 'distance' in k.lower()]
    if not dist_keys:
        raise RuntimeError("❌ Could not find distance array. PyVista version may be unsupported.")
    distances = np.abs(temp.point_data[dist_keys[0]])

    # 3️⃣ Compute normal angles (fast KDTree nearest-neighbor matching)
    print("Computing normal angles...")
    tree = cKDTree(coarse.points)
    _, idx = tree.query(fine.points)
    coarse_n_at_fine = coarse_normals[idx]

    # Dot product -> angle (use abs() to ignore normal direction flips)
    dot = np.einsum('ij,ij->i', fine_normals, coarse_n_at_fine)
    angles = np.degrees(np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))

    # 4️⃣ Vectorized face filtering
    print("Filtering elements...")
    faces = fine.faces.reshape(-1, 4)[:, 1:]  # (N_faces, 3)
    dist_per_face = distances[faces]
    angle_per_face = angles[faces]

    dist_ok = (dist_per_face >= min_dist) & (dist_per_face <= max_dist)
    angle_ok = angle_per_face <= max_angle_deg
    vertex_ok = dist_ok & angle_ok

    if mode == 'within':
        keep_mask = np.all(vertex_ok, axis=1)
    elif mode == 'outside':
        keep_mask = np.any(~vertex_ok, axis=1)
    else:
        raise ValueError("mode must be 'within' or 'outside'")

    keep = np.where(keep_mask)[0]
    n_faces = len(faces)
    print(f"✅ Kept {len(keep)} / {n_faces} elements ({100 * len(keep) / n_faces:.1f}%)")

    if len(keep) == 0:
        print("⚠️  Warning: No elements matched the criteria!")
        return

    # 5️⃣ Build & save output
    selected_faces = faces[keep]
    used_nodes = np.unique(selected_faces)

    # VTP for ParaView
    vtk_faces = np.column_stack([np.full(len(selected_faces), 3), selected_faces]).flatten()
    result = pv.PolyData(fine.points, vtk_faces)
    result.point_data['distance_to_coarse'] = distances
    result.point_data['normal_angle_deg'] = angles
    result.save(f"{output_prefix}.vtp")

    # ✅ DIRECT .PTS OUTPUT (matches .vtp exactly)
    extracted_pts = fine.points[used_nodes]
    np.savetxt(f"{output_prefix}.pts", extracted_pts, fmt="%.6f")
    return True


def extract_surface_by_pts(vol_file, pts_file, out_prefix, tol=1e-5, one_based=False):
    """
    从体积网格中提取与 .pts 坐标匹配的表面三角片，并按指定格式输出 .surf 和 .vtx
    """
    print(f"📂 加载体积网格: {vol_file}")
    vol = pv.read(vol_file)

    print(f"📂 加载目标点集: {pts_file}")
    target_pts = np.loadtxt(pts_file)
    if target_pts.ndim == 1:
        target_pts = target_pts.reshape(1, -1)

    # 1. 空间匹配：找到 .pts 在体积网格中对应的全局节点索引
    tree_vol = KDTree(vol.points)
    dists, matched_idxs = tree_vol.query(target_pts, k=1)
    valid_mask = dists < tol
    matched_global_ids = np.unique(matched_idxs[valid_mask].astype(int))
    print(f"✅ 空间匹配成功: {len(matched_global_ids)} 个节点 (容差: {tol})")

    # 2. 提取边界表面并强制三角化
    surf = vol.extract_surface().triangulate()

    # 3. 将表面网格节点映射回原始 all.vtk 的全局索引
    _, surf_to_vol = tree_vol.query(surf.points, k=1)
    surf_orig_ids = surf_to_vol.astype(int)

    # 4. 提取三角面拓扑 (PyVista faces 格式: [3, v1, v2, v3, ...])
    faces = surf.faces.reshape(-1, 4)[:, 1:]  # (N_tri, 3)
    tri_global_ids = surf_orig_ids[faces]

    # 5. 过滤：仅保留三个顶点均属于目标结构的三角面
    is_target = np.zeros(len(vol.points), dtype=bool)
    is_target[matched_global_ids] = True
    keep_mask = np.all(is_target[tri_global_ids], axis=1)
    filtered_tris = tri_global_ids[keep_mask]
    print(f"✅ 表面三角面过滤后: {len(filtered_tris)} 个")

    # 索引偏移处理 (0-based / 1-based)
    idx_offset = 1 if one_based else 0
    final_tris = filtered_tris + idx_offset

    # 提取实际参与构成表面的唯一节点ID（保证 .vtx 与 .surf 严格对应）
    final_nodes = np.unique(final_tris.flatten())
    final_nodes.sort()

    # 6. 写入 .vtx 文件 (严格遵循用户格式)
    vtx_path = f"{out_prefix}.surf.vtx"
    with open(vtx_path, "w") as f:
        f.write(f"{len(final_nodes)}\n")
        f.write("extra\n")
        for nid in final_nodes:
            f.write(f"{nid}\n")

    # 7. 写入 .surf 文件 (严格遵循用户格式)
    surf_path = f"{out_prefix}.surf"
    with open(surf_path, "w") as f:
        f.write(f"{len(final_tris)}\n")
        for tri in final_tris:
            f.write(f"Tr {tri[0]} {tri[1]} {tri[2]}\n")

    print(f"💾 输出完成:\n   📄 {vtx_path} ({len(final_nodes)} 节点)\n   📄 {surf_path} ({len(final_tris)} 三角面)")
    return vtx_path, surf_path

if __name__ == "__main__":
    valve_files = [
        ("/home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/UKB_example_4s/subject0071/Instance_2/Mesh_Outputs/Mesh_mitral_valve_timeframe001.pts",
         "/home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/UKB_example_4s/subject0071/Instance_2/Mesh_Outputs/Mesh_mitral_valve_timeframe001.elem"),
    ]
    mesh = "/home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/UKB_example_4s/subject0071/Instance_2/UVC_fiber_Outputs/timeframe1/LV_all"
    surfloc = "/home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/UKB_example_4s/subject0071/Instance_2/UVC_fiber_Outputs/timeframe1"
    basesurf_path = f"{surfloc}/biv_i/lv_base_i"
    find_intersection_base(valve_files, mesh, basesurf_path)

