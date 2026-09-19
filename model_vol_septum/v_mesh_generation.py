#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:31:59 2022

@author: sq20
"""

import pandas as pd
import numpy as np
import os
import glob
import math
import random
import shutil
import logging
import pyvista as pv
import numpy as np
from collections import deque
from scipy.spatial import cKDTree
from scipy.stats import mode
from . import carpfunc
from . import meshIO
from . import meshtool_func as meshtool
from . import mesh_clean

logger = logging.getLogger(__name__)

def merge_resample(meshtoolloc, input_dir, folder, LV_endo, RV_FW, RV_septum, epi, min, max, outmsh):
    if not os.path.exists(outmsh):
        os.mkdir(outmsh)
    os.chdir(outmsh)
    # merge all myo surfaces and generate mesh
    if not meshtool.merge_mesh(meshtoolloc, input_dir + "/" + LV_endo, input_dir + "/" + RV_FW, "M1", ifmt="vtk", ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc, "M1", input_dir + "/" + RV_septum, "M2", ifmt="vtk", ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc, "M2", input_dir + "/" + epi, "M3", ifmt="vtk", ofmt="vtk"):
        return False

    # -------- M3.vtk 后处理修复(不修复的M3.vtk无法通过TETGEN的generate_mesh要求) --------
    logger.debug("Repairing M3.vtk with pymeshfix ...")
    mesh_clean.mesh_fix(
        input_vtk="M3.vtk",
        output_dir=".",
        output_name="M3_fixed.vtk"       # 直接覆盖原文件，供下一步 generate_mesh 使用
    )
    logger.debug("M3.vtk repaired successfully.")
    # -------- 修复结束 --------

    if not meshtool.generate_mesh(meshtoolloc, "M3_fixed", "M3_volume", ifmt="vtk", ofmt="vtk"):
        return False
    if not meshtool.convert_mesh(meshtoolloc, "M3_volume", "M3_volume", ifmt="vtk"):
        return False

    # resample mesh directly from M3_volume (no valve merging) and save edge lengths in "edgeinfo.txt"
    if not meshtool.resample_mesh(meshtoolloc, "M3_volume", min, max, outmsh, ifmt="vtk", ofmt="vtk"):
        return False
    if not meshtool.convert_mesh(meshtoolloc, outmsh, outmsh, ifmt="vtk"):
        return False
    if not meshtool.query_edge(meshtoolloc, outmsh, file="edgeinfo"):
        return False

    if not meshtool.extract_mesh(meshtoolloc, outmsh, "0", outmsh + "_i"):
        return False

    # rewrite the lon file to (1,0,0) or the Laplace solver wont run
    lon = meshIO.read_fibres(basename=outmsh + "_i", file_lon=None)
    lon.insert(0, 'x_axis', 1)
    # print (elem.head)
    lonnew = lon.drop(2, axis=1)  # axis 1 drops columns, 0 will drop rows that match index value in labels
    meshIO.write_lon(outmsh + "_i", pd.DataFrame(lonnew))
    if not meshtool.convert_mesh(meshtoolloc, outmsh + "_i", outmsh + "_i", ofmt="vtk"):
        return False

    os.chdir(folder)

    return True


def extract_surfacenolabel(meshtoolloc, mesh, pts5810, surfmesh, input_dir, RV_septum):
    # input mesh with labels as RV and LV:0 (no valve labels)
    # input the original pts file for finding coords in LV_endo, RV_endo and RV_septum
    # output is 3 surf: LV_endo, RV_endo and RV_septum

    print('line 187 *')
    if not os.path.exists(surfmesh):
        os.mkdir(surfmesh)
    os.chdir(surfmesh)

    et_vertex_start_end = np.array(
        [[0, 1499], [1500, 2164], [2165, 3223], [3224, 5581],
         [5582, 5630], [5631, 5655], [5656, 5696], [5697, 5729],
         [5730, 5809]])
    '''Class constant, surface index limits for vertices `et_pos`. 
        Surfaces are defined in the following order:

        LV_ENDOCARDIAL = 0 
        RV_SEPTUM = 1
        RV_FREEWALL = 2
        EPICARDIAL =3
        MITRAL_VALVE =4
        AORTA_VALVE = 5
        TRICUSPID_VALVE = 6
        PULMONARY_VALVE = 7
        RV_INSERT = 8
        '''

    LVendo = et_vertex_start_end[0][0]
    LVendocoord = str(pts5810.iloc[LVendo, 0]) + "," + str(pts5810.iloc[LVendo, 1]) + "," + str(pts5810.iloc[LVendo, 2])
    RVendo = et_vertex_start_end[2][0]
    RVendocoord = str(pts5810.iloc[RVendo, 0]) + "," + str(pts5810.iloc[RVendo, 1]) + "," + str(pts5810.iloc[RVendo, 2])

    # extract LV_endo and RV_endo. For Rv_endo, both carp txt and .surf are written out
    # No valve regions, so extract surface of tag 0 directly
    meshtool.extract_surf(meshtoolloc, mesh, "LV_endo", "0-4,5,6,7", "-coord=" + LVendocoord, '-ofmt=carp_txt') # get LV_endo.neubc/surf/surfmesh.pts/Ion
    meshtool.extract_surf(meshtoolloc, mesh, "RV_endo", "0-4,5,6,7", "-coord=" + RVendocoord, '-ofmt=carp_txt')  # get RV_endo.neubc/surf/surfmesh.pts/Ion

    # RV_septum mapping from low res to high res. Credit to Martin Bishop.
    # create RV_septum from original surface mesh in vtk to carp_txt
    meshtool.convert_mesh(meshtoolloc, input_dir + '/' + RV_septum, RV_septum, ifmt='vtk')

    # generate a dummy element data file of ones for this septal surface mesh
    elems = meshIO.read_elem(RV_septum)
    data = np.ones([len(elems), 1])
    np.savetxt('RV_septum_elemdata.dat', data)

    # interpolates elemdata across to high res mesh
    omsh = 'RV_endo.surfmesh'
    meshtool.interpolate_elemdata(meshtoolloc, RV_septum, 'RV_septum_elemdata.dat', omsh, 'rv_endo_septdata.dat')

    # #thresholds interpolated datafield to define septal surface on highres mesh
    # extract surface_fromdata

    surf = meshIO.read_surf('RV_endo')

    data = np.loadtxt('rv_endo_septdata.dat')
    thr = 0.9
    Septum_surf_hre = surf[data > thr]
    meshIO.write_surf('RV_septumfinal', Septum_surf_hre)
    surf_vtx = np.unique(Septum_surf_hre)
    meshIO.write_vtx_File(vtxFilename='RV_septumfinal.surf', vtx=pd.DataFrame(surf_vtx))

    # #extract RV free wall as RV endo - RV septum
    meshtool.extract_surf(meshtoolloc, mesh, "RV_endo_FW", "RV_endo-RV_septumfinal")

    biv_surf_i = "surf_i"
    if not os.path.exists(biv_surf_i):
        os.mkdir(biv_surf_i)
    if not meshtool.map_file(meshtoolloc, mesh + "_i", "*.surf", biv_surf_i):
        return None
    if not meshtool.map_file(meshtoolloc, mesh + "_i", "*.vtx", biv_surf_i):
        return None
    biv_surf_i = "/surf_i/"
    logger.debug('line 258 ', surfmesh + biv_surf_i + "/LV_endo", surfmesh + biv_surf_i + "/RV_septumfinal", surfmesh + biv_surf_i + "/RV_endo_FW")
    print('line 258 *')
    return [surfmesh + biv_surf_i + "/LV_endo", surfmesh + biv_surf_i + "/RV_septumfinal", surfmesh + biv_surf_i + "/RV_endo_FW"]


def split_RVLV(meshtoolloc, MPIEXEC, OPTS, CARP, parfile, mesh_nolabel, SIMID, surf_endo, IGBEXTRACT, thres, outmsh):
    # input SIMID:result from Laplace
    # mesh_nolabel: resampled mesh dir plus the name
    # thres: define LV and RV boundary, typically 0.7
    # pathtopar: the location of the .par for Laplace solver
    # pathtosurface: the location of the surfaces in the resampled mesh
    # output the new labeled .elem file.
    # Labels: RV:0; LV:1

    args = "-stimulus[0].vtx_file" + " " + surf_endo[0] + ".surf -stimulus[1].vtx_file" + " " + surf_endo[1] + ".surf -stimulus[2].vtx_file" + " " + surf_endo[2] + ".surf"
    carpfunc.Laplace_solver(MPIEXEC, OPTS, CARP, parfile, mesh_nolabel, SIMID, args)

    gradMag = carpfunc.igb_todata(IGBEXTRACT, SIMID)

    elem = meshIO.read_elems(basename=mesh_nolabel)
    # thres=0.51
    print("Start spliting...")
    split = elem[[1, 2, 3, 4]].applymap(lambda x: x < len(gradMag) and gradMag[x] < thres) #old
    # # # 自动取势场中位数作为分割点，避免固定 thres 因网格分辨率变化失效
    # # thres = np.median(gradMag)
    # split = elem[[1, 2, 3, 4]].applymap(lambda x: x < len(gradMag) and gradMag[x] > thres) #new
    elem[5] = split.any(axis=1).map(int)
    # elem.loc[elem[5]==0,5]=0     #RV:0
    outmsh = mesh_nolabel + "_split"
    meshIO.write_elem(elemFilename=outmsh, elem=elem, shapes="Tt")
    print("End spliting!")
    shutil.copyfile(mesh_nolabel + ".lon", outmsh + ".lon")
    shutil.copyfile(mesh_nolabel + ".pts", outmsh + ".pts")
    shutil.copyfile(mesh_nolabel + ".nod", outmsh + ".nod")
    shutil.copyfile(mesh_nolabel + ".eidx", outmsh + ".eidx")

    if not meshtool.convert_mesh(meshtoolloc, outmsh, outmsh, ofmt="vtk"):
        return None

    print("Split LV and RV mesh is" + outmsh)

    return outmsh

def refine_tags_via_connectivity(cells_conn, cell_tags, min_size_ratio=0.001):
    """
    基于单元邻接关系的连通域过滤：
    1. 构建四面体面-面邻接图
    2. 按标签寻找连通区域
    3. 将小于阈值的小连通域重标为相邻区域的众数标签
    """
    n_cells = len(cell_tags)
    min_size = max(30, int(n_cells * min_size_ratio))  # 动态最小尺寸，防止极小网格误杀

    print(f"   🔍 构建单元邻接图 (共 {n_cells} 个单元)...")
    # 1. 提取所有内部面并建立 面->单元 映射
    face_to_cells = {}
    for cid, tet in enumerate(cells_conn):
        # 四面体的4个面，每个面由3个顶点组成（排序保证唯一性）
        for i in range(4):
            face = tuple(sorted(np.delete(tet, i)))
            if face not in face_to_cells:
                face_to_cells[face] = []
            face_to_cells[face].append(cid)

    # 2. 构建单元邻接表 (仅共享面的内部邻接)
    adj = [[] for _ in range(n_cells)]
    for cells in face_to_cells.values():
        if len(cells) == 2:
            c1, c2 = cells
            adj[c1].append(c2)
            adj[c2].append(c1)

    # 3. 连通域搜索与重标定
    print(f"   🔍 执行连通域过滤 (阈值: {min_size} 单元)...")
    new_tags = cell_tags.copy()
    visited = np.zeros(n_cells, dtype=bool)
    unique_tags = np.unique(cell_tags)

    # 预计算最大标签值用于 bincount
    max_tag = int(unique_tags.max()) if len(unique_tags) > 0 else 8

    for tag in unique_tags:
        tag_mask = (cell_tags == tag)
        tag_indices = np.where(tag_mask)[0]

        for start in tag_indices:
            if visited[start]:
                continue
            # BFS 查找连通域
            comp = []
            stack = [start]
            visited[start] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if tag_mask[nb] and not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)

            # 若为孤立小区域，则重标
            if len(comp) < min_size:
                neighbor_tags = []
                for c in comp:
                    for nb in adj[c]:
                        neighbor_tags.append(new_tags[nb])

                if neighbor_tags:
                    # 使用 bincount 替代 scipy.mode，更快且无版本兼容问题
                    counts = np.bincount(neighbor_tags, minlength=max_tag + 1)
                    majority_tag = np.argmax(counts)
                    new_tags[comp] = majority_tag

    # 统计修正结果
    fixed_count = np.sum(new_tags != cell_tags)
    print(f"   ✅ 修正了 {fixed_count} 个单元的孤立标签")
    return new_tags

def refine_tags_via_region_growing(cells_conn, cell_tags, min_size_ratio=0.001):
    """
    基于多源区域生长（Graph Voronoi）的标签平滑与孤立区域消除：
    1. 构建四面体面-面邻接图
    2. 识别大尺寸连通域作为“种子”
    3. 多源同步拓扑生长，按距离覆盖小区域/噪声
    4. 保证区域边界清晰，彻底消除孤立杂散标签
    """
    n_cells = len(cell_tags)
    min_size = max(30, int(n_cells * min_size_ratio))
    print(f"   🌱 执行区域生长过滤 (阈值: {min_size} 单元)...")

    # 1. 构建单元邻接表 (共享面的单元互为邻居)
    face_to_cells = {}
    for cid, tet in enumerate(cells_conn):
        for i in range(4):
            # 排序保证面的唯一性
            face = tuple(sorted(np.delete(tet, i)))
            if face not in face_to_cells:
                face_to_cells[face] = []
            face_to_cells[face].append(cid)

    adj = [[] for _ in range(n_cells)]
    for cells in face_to_cells.values():
        if len(cells) == 2:
            c1, c2 = cells
            adj[c1].append(c2)
            adj[c2].append(c1)

    # 2. 筛选种子区域 (尺寸 >= min_size 的连通域)
    final_tags = -np.ones(n_cells, dtype=np.int32)  # -1 表示待分配
    visited = np.zeros(n_cells, dtype=bool)
    queue = deque()

    unique_tags = np.unique(cell_tags)
    for tag in unique_tags:
        tag_mask = (cell_tags == tag)
        tag_indices = np.where(tag_mask)[0]

        for start in tag_indices:
            if visited[start]:
                continue
            # DFS 查找当前标签的连通域
            comp = []
            stack = [start]
            visited[start] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if tag_mask[nb] and not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)

            if len(comp) >= min_size:
                # 达标区域作为生长种子
                final_tags[comp] = tag
                queue.extend(comp)
            # 未达标的小区域保持 final_tags == -1，等待被生长覆盖

    # 3. 多源区域生长 (Graph Voronoi 划分)
    while queue:
        cell = queue.popleft()
        current_tag = final_tags[cell]
        for nb in adj[cell]:
            if final_tags[nb] == -1:
                final_tags[nb] = current_tag
                queue.append(nb)

    # 4. 安全兜底 (防止极个别因网格断裂未被覆盖的单元)
    unassigned = np.where(final_tags == -1)[0]
    if len(unassigned) > 0:
        print(f"   ⚠️ 发现 {len(unassigned)} 个孤立单元，按最近邻居填充...")
        for idx in unassigned:
            nb_tags = [final_tags[nb] for nb in adj[idx] if final_tags[nb] != -1]
            if nb_tags:
                counts = np.bincount(nb_tags)
                final_tags[idx] = np.argmax(counts)
            else:
                final_tags[idx] = cell_tags[idx]  # 极端情况回退原始值

    fixed_count = np.sum(final_tags != cell_tags)
    print(f"   ✅ 区域生长完成，修正了 {fixed_count} 个单元")
    return final_tags

# def process_and_export_vtk(coarse_vtk_path, fine_vtk_path, output_vtk_path):
def process_and_export_vtk(coarse_vtk_path, fine_vtk_path, output_vtk_path, min_size_ratio=0.001):

    print("📥 1. 加载粗网格并初始化区域标签...")
    coarse = pv.read(coarse_vtk_path)
    if coarse.n_points != 5810:
        raise ValueError(f"❌ 粗网格点数应为 5810，实际为 {coarse.n_points}")

    pts = coarse.points
    coarse_tags = np.zeros(5810, dtype=np.int32)
    LV_RV_Only=True
    # 严格按索引区间赋值 (Python切片左闭右开)
    if not LV_RV_Only:
        coarse_tags[0:1500] = 0  # LV_ENDOCARDIAL
        coarse_tags[1500:2165] = 1  # RV_SEPTUM
        coarse_tags[2165:3224] = 2  # RV_FREEWALL
        coarse_tags[3224:5582] = 3  # EPICARDIAL (待细分)
        coarse_tags[5582:5631] = 4  # MITRAL_VALVE
        coarse_tags[5631:5656] = 5  # AORTA_VALVE
        coarse_tags[5656:5697] = 6  # TRICUSPID_VALVE
        coarse_tags[5697:5730] = 7  # PULMONARY_VALVE
        coarse_tags[5730:5810] = 8  # RV_INSERT
    else:
        coarse_tags[0:1500] = 1  # LV_ENDOCARDIAL
        coarse_tags[1500:2165] = 1  # RV_SEPTUM
        coarse_tags[2165:3224] = 0  # RV_FREEWALL
        coarse_tags[3224:5582] = 3  # EPICARDIAL (待细分)
        coarse_tags[5582:5631] = 1  # MITRAL_VALVE
        coarse_tags[5631:5656] = 1  # AORTA_VALVE
        coarse_tags[5656:5697] = 0  # TRICUSPID_VALVE
        coarse_tags[5697:5730] = 0  # PULMONARY_VALVE
        coarse_tags[5730:5810] = 0  # RV_INSERT

    print("📐 2. 细分心外膜区域 [3224, 5581]...")
    epi_idx = np.arange(3224, 5582)
    epi_pts = pts[epi_idx]

    # 构建 KD-Tree 加速距离查询
    tree_lv = cKDTree(pts[0:1500])
    tree_rv = cKDTree(np.vstack([pts[1500:2165], pts[2165:3224]]))

    d_lv, _ = tree_lv.query(epi_pts, k=1)
    d_rv, _ = tree_rv.query(epi_pts, k=1)

    # 靠近(或等距) LV_ENDO -> tag 1，靠近 RV -> tag 0 (严格按需求)
    coarse_tags[epi_idx] = np.where(d_lv <= d_rv, 1, 0)

    print("📥 3. 加载细网格并映射标签...")
    fine = pv.read(fine_vtk_path)
    n_cells = fine.n_cells
    print(f"   细网格: {fine.n_points} 点, {n_cells} 单元")

    # 提取四面体连接表 (移除每行开头的 '4')
    tet_conn = fine.cells.reshape(-1, 5)[:, 1:]  # shape: (N, 4)

    # 批量查找细网格顶点最近的粗网格点
    tree_coarse = cKDTree(pts)
    _, nearest_idx = tree_coarse.query(fine.points[tet_conn.flatten()], k=1)
    nearest_idx = nearest_idx.reshape(-1, 4)

    # 获取顶点标签并执行众数投票
    vertex_tags = coarse_tags[nearest_idx]
    elem_tags = mode(vertex_tags, axis=1, keepdims=True).mode.ravel()

    # 🆕 新增：连通域拓扑过滤
    # elem_tags = refine_tags_via_connectivity(tet_conn, elem_tags, min_size_ratio=min_size_ratio)
    elem_tags = refine_tags_via_region_growing(tet_conn, elem_tags, min_size_ratio=min_size_ratio)

    # 提取需保留的单元数据 (如 fiber)
    fiber_data = fine.cell_data.get("fiber", None)
    if fiber_data is None:
        print("⚠️ 未在原网格中找到 'fiber' 数据，输出文件将仅包含 elemTag")

    print("💾 4. 按指定格式写入 ASCII VTK...")
    write_custom_vtk(output_vtk_path, fine.points, tet_conn, elem_tags, fiber_data)
    print(f"✅ 完成！结果已保存至: {output_vtk_path}")
    return True


def write_custom_vtk(filename, points, cells_conn, elem_tags, fiber_data=None):
    """自定义写入器：严格匹配用户提供的 VTK Legacy ASCII 格式"""
    n_pts = len(points)
    n_cells = len(elem_tags)

    with open(filename, 'w') as f:
        # 文件头
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")

        # POINTS
        f.write(f"POINTS {n_pts} float\n")
        np.savetxt(f, points, fmt='%.5f', delimiter=' ')
        f.write("\n")

        # CELL_TYPES (全为四面体 VTK Type 10)
        f.write(f"CELL_TYPES {n_cells}\n")
        np.savetxt(f, np.full(n_cells, 10, dtype=np.int32), fmt='%d')
        f.write("\n")

        # CELLS
        n_vals = 5 * n_cells  # 每个四面体 5 个整数 (4 + 4个顶点索引)
        f.write(f"CELLS {n_cells} {n_vals}\n")
        cell_lines = np.column_stack([np.full(n_cells, 4, dtype=np.int32), cells_conn])
        np.savetxt(f, cell_lines, fmt='%d')
        f.write("\n")

        # CELL_DATA
        f.write(f"CELL_DATA {n_cells}\n")

        # SCALARS elemTag
        f.write("SCALARS elemTag int 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, elem_tags.reshape(-1, 1), fmt='%d')
        f.write("\n")

        # VECTORS fiber (若存在)
        if fiber_data is not None:
            f.write("VECTORS fiber float\n")
            np.savetxt(f, fiber_data, fmt='%.5f')
            f.write("\n")

