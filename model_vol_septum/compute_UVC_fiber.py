#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:36:35 2022

@author: sq20
"""
import subprocess

import pandas as pd
import numpy as np
import os
import glob
import math
import random
import shutil
import vtk
import pyvista as pv
import logging

from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

from . import meshIO
from . import meshtool_func as meshtool
from . import carpfunc
from . import geometrical
from . import Coord_UVC_function

'''
LV_ENDOCARDIAL = 0
RV_SEPTUM = 1
RV_FREEWALL = 2
EPICARDIAL = 3
MITRAL_VALVE = 4
AORTA_VALVE = 5
TRICUSPID_VALVE = 6
PULMONARY_VALVE = 7
RV_INSERT = 8
'''

def coord_z(folder, meshtoolloc, mesh, input_dir, meshnames, surfloc, UVCfolder):
    # "folder" is where the labeled mesh stored,"mesh" is the mesh name
    # input_dir is where surface mesh stored
    # "surfloc" is where to store surfaces
    # find apex point in LV only mesh
    # output is the x,y,z coord of the apex point
    # mesh_names: ['Mesh_LV_endo_timeframe001', 'Mesh_RV_FW_timeframe001', 'Mesh_RV_septum_timeframe001',
    #              'Mesh_aorta_valve_timeframe001', 'Mesh_epi_timeframe001', 'Mesh_mitral_valve_timeframe001',
    #              'Mesh_pulmonary_valve_timeframe001', 'Mesh_tricuspid_valve_timeframe001']
    os.chdir(folder)

    if not meshtool.extract_mesh(meshtoolloc, mesh, "1", "LV_all"): # 只提取LV_all部分,此部分的tag为1
        return None

    Mesh_mitral_valve_surf = meshnames[5]

    if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_mitral_valve_surf}", input_dir+f"/{Mesh_mitral_valve_surf}", "vtk", "carp_txt"):
        return None

    if not os.path.exists(surfloc):
        os.mkdir(surfloc)

    # extract all surface in biv mesh (only LV and RV) to store in folder surfloc/biv_i
    if not os.path.exists(surfloc + "/biv_i/"):
        os.mkdir(surfloc + "/biv_i/")

    mitral_valve_files = [
        (input_dir + f"/{Mesh_mitral_valve_surf}" + ".pts",
        input_dir + f"/{Mesh_mitral_valve_surf}" + ".elem")
    ]

    if not Coord_UVC_function.find_intersection_base(mitral_valve_files, "LV_all", surfloc + "/biv_i/lv_base"):
        return None
    # if not meshtool.extract_surf(meshtoolloc, "LV_all", surfloc + "/mitral", "4"):
    #     return None
    #1待修改:此处先convert Mesh_mitral_valve_timeframe001的vtk为carp_txt
    #2待修改:python find_intersection_lv_base.py 即通过二尖瓣的.pts/.elem寻找瓣膜环lv_base_i

    d_from_mitral = "d_mitraltoLVall.dat"
    # if not meshtool.generate_dfield(meshtoolloc, "LV_all", surfloc + "/mitral", d_from_mitral):
    #     return None
    #3待修改:if not meshtool.generate_dfield(meshtoolloc, "LV_all", "lv_base_i", d_from_mitral):
    #     return None

    if not meshtool.generate_dfield(meshtoolloc, "LV_all", surfloc + "/biv_i/lv_base", d_from_mitral):
        return None

    d_LVtomitral = np.genfromtxt(d_from_mitral)
    # find the index of the furthest point from the mitral valve as apex point
    index = np.where(d_LVtomitral == np.amax(d_LVtomitral))[0]

    if not meshtool.extract_mesh(meshtoolloc, mesh, "0,1", mesh + "_i"):  # 此处有valve的mesh进行左右心室提取,无valve的情况下提取就是本身(可复制替代本操作)
        return None

    pts_LV = np.array(meshIO.read_pts(basename="LV_all"))
    pts_biv = np.array(meshIO.read_pts(basename=mesh + "_i"))

    # find the apex point's index in biv mesh
    ptsindex_biv = np.where(np.all(pts_biv == pts_LV[index], axis=1))[0]

    if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surfloc + "/biv_i/myoall", "0,1"):
        return None

    surf = meshIO.read_surf(basename=surfloc + "/biv_i/myoall")
    # find the surface including the apex point and write to mesh+"_i_apex" in both .surf and .vtx
    surfnew = surf.loc[
        (surf[1] == int(ptsindex_biv)) | (surf[2] == int(ptsindex_biv)) | (surf[3] == int(ptsindex_biv)), :]
    surf_vtx = np.unique(surfnew.loc[:, 1:3])

    Filename = surfloc + "/biv_i/apex_i"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)

    # extract base surf in myo only mesh which is the intersection region of valves with myo
    # first extract the "biv_i_all-biv_allvalves" in the whole mesh and map it to biv myo only mesh
    # if not meshtool.extract_surf(meshtoolloc, mesh, surfloc + "/intersection", "0,1:4,5,6,7"):
    #     return None
    #4待修改:此处先convert Mesh_mitral_valve(之前已完成)/Mesh_tricuspid_valve/Mesh_aorta_valve/Mesh_pulmonary_valve的vtk为carp_txt

    #5待修改:python find_intersection.py 即通过多个瓣膜的.pts/.elem寻找瓣膜环的整体base_i
    # if not meshtool.map_file(meshtoolloc, mesh + "_i", surfloc + "/intersection.surf", surfloc + "/biv_i"):
    #     return None
    # if not meshtool.map_file(meshtoolloc, mesh + "_i", surfloc + "/intersection.surf.vtx", surfloc + "/biv_i"):
    #     return None

    # Mesh_mitral_valve_surf = meshnames[5] # 上面已经完成
    Mesh_tricuspid_valve = meshnames[7]
    Mesh_aorta_valve = meshnames[3]
    Mesh_pulmonary_valve = meshnames[6]

    if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_tricuspid_valve}", input_dir+f"/{Mesh_tricuspid_valve}", "vtk", "carp_txt"):
        return None
    if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_aorta_valve}", input_dir+f"/{Mesh_aorta_valve}", "vtk", "carp_txt"):
        return None
    if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_pulmonary_valve}", input_dir+f"/{Mesh_pulmonary_valve}", "vtk", "carp_txt"):
        return None

    all_valve_files = [
        (input_dir + f"/{Mesh_mitral_valve_surf}" + ".pts",
        input_dir + f"/{Mesh_mitral_valve_surf}" + ".elem"),
        (input_dir + f"/{Mesh_tricuspid_valve}" + ".pts",
         input_dir + f"/{Mesh_tricuspid_valve}" + ".elem"),
        (input_dir + f"/{Mesh_aorta_valve}" + ".pts",
         input_dir + f"/{Mesh_aorta_valve}" + ".elem"),
        (input_dir + f"/{Mesh_pulmonary_valve}" + ".pts",
         input_dir + f"/{Mesh_pulmonary_valve}" + ".elem"),
    ]

    if not Coord_UVC_function.find_intersection_base(all_valve_files, mesh + "_i", surfloc + "/biv_i/base_i"):
        return None

    # calculate the distance from base to apex and store the results in
    apexsurf = surfloc + "/biv_i/apex_i"
    # basesurf = surfloc + "/biv_i/intersection"
    #6待修改:basesurf = surfloc + "/biv_i/base_i"
    basesurf = surfloc + "/biv_i/base_i"
    outputFilename = UVCfolder + "/COORDS_Z.dat"

    if not meshtool.generate_dfield(meshtoolloc, mesh + "_i", apexsurf, outputFilename, "-esurf=" + basesurf):
        return None

    if not meshtool.collect_nodal(meshtoolloc, mesh + "_i", "carp_txt", UVCfolder + "/COORDS_Z", '',UVCfolder + "/COORDS_Z.dat"):
        return None
    #7待修改:可视化COORDS_Z.dat可通过collect -imsh=subject0071_mesh_all -omsh=result -nod=COORDS_Z.dat
    # 产物: result.geo / result.case / result.tags.ens / result.fiber.ens / result_COORDS_Z.ens
    return outputFilename

def transmural(folder, meshtoolloc, mesh, pts5810, input_dir, meshnames, surfloc, MPIEXEC, OPTS, CARP, parfile, SIMID, IGBEXTRACT, UVCfolder):
    #
    os.chdir(folder)

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
    # LVendo = et_vertex_start_end[0][0]
    # LVendocoord = str(pts5810.iloc[LVendo, 0]) + "," + str(pts5810.iloc[LVendo, 1]) + "," + str(pts5810.iloc[LVendo, 2])
    # RVendo = et_vertex_start_end[2][0]
    # RVendocoord = str(pts5810.iloc[RVendo, 0]) + "," + str(pts5810.iloc[RVendo, 1]) + "," + str(pts5810.iloc[RVendo, 2])
    # RVseptum = et_vertex_start_end[1][0]
    # RVseptumcoord = str(pts5810.iloc[RVseptum, 0]) + "," + str(pts5810.iloc[RVseptum, 1]) + "," + str(pts5810.iloc[RVseptum, 2])
    # epi = et_vertex_start_end[3][0]
    # epicoord = str(pts5810.iloc[epi, 0]) + "," + str(pts5810.iloc[epi, 1]) + "," + str(pts5810.iloc[epi, 2])


    # mesh_names: ['Mesh_LV_endo_timeframe001', 'Mesh_RV_FW_timeframe001', 'Mesh_RV_septum_timeframe001',
    #              'Mesh_aorta_valve_timeframe001', 'Mesh_epi_timeframe001', 'Mesh_mitral_valve_timeframe001',
    #              'Mesh_pulmonary_valve_timeframe001', 'Mesh_tricuspid_valve_timeframe001']
    Mesh_LV_endo_surf = meshnames[0]
    Mesh_RV_septum_surf = meshnames[2]
    Mesh_RV_endo_surf = meshnames[1]
    Mesh_epi_surf = meshnames[4]

    LV_endo_coarse_file = input_dir+f"/{Mesh_LV_endo_surf}" + ".vtk"
    RV_septum_coarse_file = input_dir+f"/{Mesh_RV_septum_surf}" + ".vtk"
    RV_endo_coarse_file = input_dir+f"/{Mesh_RV_endo_surf}" + ".vtk"
    epi_coarse_file = input_dir+f"/{Mesh_epi_surf}" + ".vtk"
    fine_file = mesh + ".vtk"
    output_prefix = surfloc + "/biv_i"
    # if not os.path.exists(output_prefix):
    #     os.mkdir(output_prefix)
    Coord_UVC_function.extract_fine_by_distance(LV_endo_coarse_file,fine_file,output_prefix + "/LV_endo",0.0, 0.3, 90)
    Coord_UVC_function.extract_fine_by_distance(RV_septum_coarse_file,fine_file,output_prefix + "/RV_septum",0.0, 0.3, 90)
    Coord_UVC_function.extract_fine_by_distance(RV_endo_coarse_file,fine_file,output_prefix + "/RV_endo",0.0, 0.3, 90)
    Coord_UVC_function.extract_fine_by_distance(epi_coarse_file,fine_file,output_prefix + "/epi",0.0, 0.3, 90)

    Coord_UVC_function.extract_surface_by_pts(fine_file, output_prefix + "/LV_endo" + ".pts", output_prefix + "/LV_endo", 1e-3, False)
    Coord_UVC_function.extract_surface_by_pts(fine_file, output_prefix + "/RV_septum" + ".pts", output_prefix + "/RV_septum", 1e-3, False)
    Coord_UVC_function.extract_surface_by_pts(fine_file, output_prefix + "/RV_endo" + ".pts", output_prefix + "/RV_endo", 1e-3, False)
    Coord_UVC_function.extract_surface_by_pts(fine_file, output_prefix + "/epi" + ".pts", output_prefix + "/epi", 1e-3, False)

    # surf_iloc = surfloc + "/Surface_pts/"
    surf_iloc = surfloc + "/biv_i/"
    # op = surf_iloc + "myoall-" + surf_iloc + "intersection"
    op = surf_iloc + "myoall" + "-" + surf_iloc + "base_i"
    #待修改：intersection实际就是coord z计算中生成的base

    LV_endo = surf_iloc + 'LV_endo'
    RV_septum = surf_iloc + 'RV_septum'
    RV_endo = surf_iloc + 'RV_endo'
    epi = surf_iloc + 'epi'

    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "LV_endo", op, "-coord=" + LVendocoord):
    #     return None
    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "RV_septum", surf_iloc + 'myoall_insertion',"-coord=" + RVseptumcoord, "-edge=10"):
    #     return None
    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "RV_endo1", op, "-coord=" + RVendocoord,"-edge=10"):
    #     return None
    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "RV_endo",surf_iloc + "RV_endo1-" + surf_iloc + 'RV_septum', "-coord=" + RVendocoord):
    #     return None
    # if not meshtool.extract_surf(meshtoolloc, mesh + "_i", surf_iloc + "epi", op, "-coord=" + epicoord):
    #     return None

    args = "-stimulus[0].vtx_file " + LV_endo + ".surf "
    args = args + " -stimulus[1].vtx_file " + RV_endo + ".surf "
    args = args + " -stimulus[2].vtx_file " + epi + ".surf "
    args = args + " -stimulus[3].vtx_file " + RV_septum + ".surf "
    # LV,RV_septum, RV_endo and epi Laplace solver
    if not carpfunc.Laplace_solver(MPIEXEC, OPTS, CARP, parfile, mesh + "_i", SIMID, args):
        return None

    gradMag = carpfunc.igb_todata(IGBEXTRACT, SIMID)
    file = UVCfolder + "/PHO.dat"
    meshIO.write_data(dataFilename=file, data=gradMag)
    # print (max(gradMag))
    # print (min(gradMag))
    if not meshtool.collect_nodal(meshtoolloc, mesh, "carp_txt", UVCfolder + "/PHO", '', UVCfolder + "/PHO.dat"):
        return None
    return file

def find_two_RV_insertion_pts(folder, meshtoolloc, meshname, input_dir, meshnames, surfloc,  coordzdir):
    # this fuc is to find two pts in RV_insertion by
    # (1)finding the "anterior point" which is the closest point to the pulmonary valve
    # (2)finding the "interior point" which is the furtherest point to ANT in RV insertion surf and also have same z
    # ANT and INF is wrong. it needs to switch for later steps!!!
    os.chdir(folder)
    LVmyosurf = surfloc + "/LVmyo"
    if not os.path.exists(LVmyosurf):
        os.mkdir(LVmyosurf)

    LV = surfloc + '/LV'
    if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/LV", "1"):
        return False

    RVinsertion = surfloc + '/RVinsertion'
    if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/RVinsertion", "0:1"):
        return False
    if not meshtool.extract_surf(meshtoolloc, meshname + '_i', surfloc + "/biv_i/RVinsertion", "0:1"): #这两处无valve时是一样的
        return False

    # if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/RVinsertion.surf",surfloc+"/LVmyo"):
    #     return False
    # if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/RVinsertion.surf.vtx",surfloc+"/LVmyo"):
    #     return False

    LVvtx = meshIO.read_vtx_File(surfloc + "/LV.surf.vtx")
    LVsurf = meshIO.read_surf(surfloc + "/LV")
    pts = meshIO.read_pts(meshname)
    coord_LV = np.zeros((len(LVvtx), 3))
    for i in range(len(LVvtx)):
        coord_LV[i] = pts.loc[int(LVvtx[i])]
    #######################
    # find the centre point of Pulmonary valve

    # if not meshtool.extract_mesh(meshtoolloc, meshname, "7", "pulmonarymesh"):
    #     return False
    #8待修改:此处先用pulmonarymesh_pts_read.py读取surface mesh的点坐标

    # pulmonary_pts = meshIO.read_pts(folder + "/pulmonarymesh")
    #9待修改:pulmonary_pts应该改为读取上述pulmonarymesh_pts_read.py的输出
    Mesh_pulmonary_valve_surf = meshnames[6]
    Mesh_pulmonary_valve_surf_pts = input_dir + f"/{Mesh_pulmonary_valve_surf}" + ".vtk"
    pulmonary_pts = Coord_UVC_function.read_surface_from_vtk(Mesh_pulmonary_valve_surf_pts)

    pulmonary_centre = np.array(np.mean(pulmonary_pts, axis=0))
    ####################

    # 1. 提取 RV insertion 顶点坐标（与 LV 坐标同属 pts 空间）
    RVinsertionvtx = meshIO.read_vtx_File(surfloc + "/RVinsertion.surf.vtx")
    coord_RVins = pts.loc[RVinsertionvtx.astype(int)].values

    # 2. 使用 KDTree 高效计算 LV 表面各点到 RV insertion 的最近距离
    tree_rv = cKDTree(coord_RVins)
    dist_LV_to_RVins, _ = tree_rv.query(coord_LV)

    # 3. 自适应阈值：仅保留距离 RV insertion 最近的 5% LV 顶点作为候选
    # 百分比策略对网格尺度/分辨率不敏感，比固定绝对值更鲁棒
    tol_rv = np.percentile(dist_LV_to_RVins, 5)

    # 防呆回退：若阈值过严导致无候选点，自动放宽至 10%
    mask_near_RV = dist_LV_to_RVins <= tol_rv
    if not np.any(mask_near_RV):
        print("[Warning] RV proximity filter too strict. Relaxing to top 10%.")
        tol_rv = np.percentile(dist_LV_to_RVins, 10)
        mask_near_RV = dist_LV_to_RVins <= tol_rv

    # 4. 【核心修正】仅对候选点集合计算到肺动脉中心的距离
    coord_candidates = coord_LV[mask_near_RV]  # 形状: (N_candidates, 3)
    dist_candidates_to_pul = np.linalg.norm(coord_candidates - pulmonary_centre, axis=1)  # 形状: (N_candidates,)

    # 5. 在候选集中找距离肺动脉最近的点
    best_idx_in_candidates = np.argmin(dist_candidates_to_pul)

    # ⚠️ 关键：将候选集内的局部索引 映射回 原始 LV 顶点索引
    original_indices = np.where(mask_near_RV)[0]  # 所有候选点在 coord_LV 中的真实位置
    best_idx_original = original_indices[best_idx_in_candidates]

    NodeANT = int(LVvtx[best_idx_original])
    print(f"[Info] ANT selected from {len(coord_candidates)} LV candidates near RV insertion. "
          f"Min dist to Pul: {dist_candidates_to_pul[best_idx_in_candidates]:.4f}")

    # distances = np.linalg.norm(coord_LV - pulmonary_centre, axis=1)
    # NodeANT = int(LVvtx[np.argmin(distances)])
    ######################################
    # print this out for visualizing but not necesseary

    surfnew = LVsurf.loc[(LVsurf[1] == NodeANT) | (LVsurf[2] == NodeANT) | (LVsurf[3] == NodeANT), :]
    surf_vtx = np.unique(surfnew.loc[:, 1:3])
    Filename = surfloc + "/Node_ANT"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
    ######################################

    #################################
    # Node_INF is identified by
    # 1) find all surfaces in RV insertion having similar coordz as Node_ANT
    # 2) In those surfaces, find the furtherest point from the Node_ANT
    # note coordz.dat is for biv_i mesh, so suface need to be mapped

    # map to biv_i mesh
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/Node_ANT.vtx", surfloc + "/biv_i"):
        return False
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/Node_ANT.surf", surfloc + "/biv_i"):
        return False

    # RVinsertion = surfloc + '/RVinsertion'
    # if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/RVinsertion", "0:1"):
    #     return False
    #10待修改:与本function前面的RVinsertion赋值冗余,可做删除

    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/RVinsertion.surf", surfloc + "/biv_i"):
        return False
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/RVinsertion.surf.vtx", surfloc + "/biv_i"):
        return False

    coordz = np.genfromtxt(coordzdir)

    ANTvtx = meshIO.read_vtx_File(surfloc + "/biv_i/Node_ANT.vtx")
    ANTsurf = meshIO.read_surf(surfloc + "/biv_i/Node_ANT")

    ANT_coordz = coordz[int(ANTvtx[0])]

    RVinsertionvtx = meshIO.read_vtx_File(surfloc + "/biv_i/RVinsertion.surf.vtx")
    RVinsertion_coordz = np.zeros(len(RVinsertionvtx))
    vtx_similarz = np.empty((0, 1), int)
    print("RV insertion:", len(RVinsertionvtx))
    for i in range(len(RVinsertionvtx)):
        if coordz[int(RVinsertionvtx[i])] < (ANT_coordz + 0.005) and coordz[int(RVinsertionvtx[i])] > (ANT_coordz - 0.005): #Z值与ANT相差 <0.005 的顶点索引数组
        # if coordz[int(RVinsertionvtx[i])] < (ANT_coordz + 0.1) and coordz[int(RVinsertionvtx[i])] > (ANT_coordz - 0.1): #Z值与ANT相差 <0.005 的顶点索引数组
            vtx_similarz = np.append(vtx_similarz, [RVinsertionvtx[i]])
    print("line 359")
    print('vtx_similarz: {}'.format(vtx_similarz))
    RVinsertionsurf = meshIO.read_surf(surfloc + "/biv_i/RVinsertion")
    pts = meshIO.read_pts(meshname + "_i")
    coordz_ANT = np.array(pts.loc[ANTvtx[0]])
    coord_similarz = np.zeros((len(vtx_similarz), 3))
    for i in range(len(vtx_similarz)):
        coord_similarz[i] = pts.loc[int(vtx_similarz[i])]

    distances = np.linalg.norm(coord_similarz - coordz_ANT, axis=1)
    if len(distances) == 0:
        return False
    NodeINF = int(vtx_similarz[np.argmax(distances)]) #定位INF点（同Z平面内距ANT最远的点）
    print('Node INF: {}'.format(NodeINF))

    ######################################
    # print this out for visualizing but not necesseary
    surfnew = RVinsertionsurf.loc[(RVinsertionsurf[1] == NodeINF) | (RVinsertionsurf[2] == NodeINF) | (RVinsertionsurf[3] == NodeINF), :]
    surf_vtx = np.unique(surfnew.loc[:, 1:3])
    Filename = surfloc + "/biv_i/Node_INF"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
    ######################################

    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/biv_i/Node_INF.surf", surfloc, "s2m"):
        return False
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/biv_i/Node_INF.surf.vtx", surfloc, "s2m"):
        return False

    return True


#########################################
# Function to preprocessing data for rotational fuc for LV
#########################################
def extract_surf_forLV(folder, meshtoolloc, meshname, input_dir, meshnames, surfloc):
    # extract surface: apex,base(mitral_intersect), Node_ANT, Node_INF
    os.chdir(folder)

    LVmyosurf = surfloc + "/LVmyo"
    if not os.path.exists(LVmyosurf):
        os.mkdir(LVmyosurf)

    if not meshtool.extract_mesh(meshtoolloc, meshname + "_i", "1", "LVmyo_i"):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/apex_i.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/apex_i.vtx", LVmyosurf):
        return None

    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/RVinsertion.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/RVinsertion.vtx", LVmyosurf):
        return None

    if not meshtool.extract_surf(meshtoolloc, "LVmyo_i", LVmyosurf + '/myoall_insertion',
                                 '1-' + LVmyosurf + '/RVinsertion'):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", LVmyosurf + '/myoall_insertion.surf', surfloc + "/biv_i",
                             mode='s2m'):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", LVmyosurf + '/myoall_insertion.vtx', surfloc + "/biv_i",
                             mode='s2m'):
        return None

    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/Node_ANT.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/Node_ANT.vtx", LVmyosurf):
        return None

    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/Node_INF.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo_i", surfloc + "/biv_i/Node_INF.vtx", LVmyosurf):
        return None

    # map the base surf from whole mesh as the mitral valve intersection with the LVmyo
    if not meshtool.extract_mesh(meshtoolloc, meshname, "1", "LVmyo"):
        return None
    # mitral_intersect = surfloc + '/mitral_intersect'

    Mesh_mitral_valve_surf = meshnames[5]

    mitral_valve_files = [
        (input_dir + f"/{Mesh_mitral_valve_surf}" + ".pts",
         input_dir + f"/{Mesh_mitral_valve_surf}" + ".elem")
    ]

    if not Coord_UVC_function.find_intersection_base(mitral_valve_files, "LVmyo", surfloc + "/mitral_intersect"):
        return None

    # if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/mitral_intersect", "1:4"):
    #     return None
    if not meshtool.map_file(meshtoolloc, "LVmyo", surfloc + "/mitral_intersect.surf", LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc, "LVmyo", surfloc + "/mitral_intersect.surf.vtx", LVmyosurf):
        return None
    #11修改:使用find_intersection_lv_base生成mitral_intersect.surf/.surf.vtx
    
    Node_ANT = meshIO.read_vtx_File(LVmyosurf + "/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        return None
    NodeANTvtx = int(Node_ANT[0])

    Node_INF = meshIO.read_vtx_File(LVmyosurf + "/Node_INF.vtx")
    if len(Node_INF) == 0:
        return None
    NodeINFvtx = int(Node_INF[0])
    return [LVmyosurf + "/mitral_intersect", LVmyosurf + "/apex_i.vtx", NodeINFvtx, NodeANTvtx]


#########################################
# Function to preprocessing data for rotational fuc for RV
#########################################
def extract_surf_forRV(folder, meshtoolloc, meshname, input_dir, meshnames, surfloc, coordzdir):
    os.chdir(folder)
    RVmyosurf = surfloc + "/RVmyo"
    if not os.path.exists(RVmyosurf):
        os.mkdir(RVmyosurf)

    if not meshtool.extract_mesh(meshtoolloc, meshname + "_i", "0", "RVmyo_i"):
        return None

    RV_endo = surfloc + '/RV_all'
    if not meshtool.extract_surf(meshtoolloc, meshname + "_i", surfloc + "/RV_all", "0"):
        return None

    coordz = np.genfromtxt(coordzdir)
    # Node_RVapex=np.argmin(coordz)

    RVvtx = meshIO.read_vtx_File(surfloc + "/RV_all.surf.vtx")
    RVsurf = meshIO.read_surf(surfloc + "/RV_all")

    coordz_RV = np.zeros((len(RVvtx), 1))
    for i in range(len(RVvtx)):
        coordz_RV[i] = coordz[int(RVvtx[i])]

    Node_RVapex = int(RVvtx[np.argmin(coordz_RV)])
    ######################################
    # print this out for visualizing but not necesseary
    surfnew = RVsurf.loc[(RVsurf[1] == Node_RVapex) | (RVsurf[2] == Node_RVapex) | (RVsurf[3] == Node_RVapex), :]
    surf_vtx = np.unique(surfnew.loc[:, 1:3])
    Filename = surfloc + "/Node_RV_apex"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)

    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/Node_RV_apex.surf", surfloc + "/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/Node_RV_apex.vtx", surfloc + "/RVmyo"):
        return None

    # tricuspid_intersect = surfloc + '/tricuspid_intersect'
    # if not meshtool.extract_surf(meshtoolloc, meshname, surfloc + "/tricuspid_intersect", "0:6"):
    #     return None
    #11待修改,python find_intersection_tricuspid.py
    Mesh_tricuspid_valve_surf = meshnames[7]

    tricuspid_valve_files = [
        (input_dir + f"/{Mesh_tricuspid_valve_surf}" + ".pts",
         input_dir + f"/{Mesh_tricuspid_valve_surf}" + ".elem")
    ]

    if not Coord_UVC_function.find_intersection_base(tricuspid_valve_files, meshname, surfloc + "/tricuspid_intersect"):
        return None

    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/tricuspid_intersect.surf", surfloc + "/biv_i"):
        return None
    if not meshtool.map_file(meshtoolloc, meshname + "_i", surfloc + "/tricuspid_intersect.surf.vtx", surfloc + "/biv_i"):
        return None
    # 12待修改,surfloc + "/tricuspid_intersect.vtx"

    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/tricuspid_intersect.surf", surfloc + "/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/tricuspid_intersect.surf.vtx", surfloc + "/RVmyo"):
        return None
    # 13待修改,surfloc + "/tricuspid_intersect.vtx"

    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/Node_INF.surf", surfloc + "/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/Node_INF.vtx", surfloc + "/RVmyo"):
        return None

    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/Node_ANT.surf", surfloc + "/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc, "RVmyo_i", surfloc + "/biv_i/Node_ANT.vtx", surfloc + "/RVmyo"):
        return None

    RVsurfdir = surfloc + "/RVmyo"
    Node_ANT = meshIO.read_vtx_File(RVsurfdir + "/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        ########################
        # find the closest point in RVmyo_i to ANT node in biv_i mesh
        ptsbiv = meshIO.read_pts(meshname + '_i')
        bivvtx = meshIO.read_vtx_File(surfloc + "/biv_i/Node_ANT.vtx")
        ptscoordANT = ptsbiv.loc[int(bivvtx[0])]

        if not meshtool.extract_surf(meshtoolloc, "RVmyo_i", surfloc + "/RVmyo/myoall", "0"):
            return None

        ptsRV = meshIO.read_pts('RVmyo_i')
        RVvtx = meshIO.read_vtx_File(surfloc + "/RVmyo/myoall.surf.vtx")
        RVsurf = meshIO.read_surf(surfloc + "/RVmyo/myoall")
        ptscoordRV = ptsRV.loc[RVvtx]

        distances = np.linalg.norm(ptscoordRV - ptscoordANT, axis=1)
        NodeANT = int(RVvtx[np.argmin(distances)])

        ######################################
        # print this out for visualizing but not necesseary
        surfnew = RVsurf.loc[(RVsurf[1] == NodeANT) | (RVsurf[2] == NodeANT) | (RVsurf[3] == NodeANT), :]
        surf_vtx = np.unique(surfnew.loc[:, 1:3])
        Filename = surfloc + "/RVmyo/Node_ANT"
        meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
        meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
        RVsurfdir = surfloc + "/RVmyo"
        Node_ANT = meshIO.read_vtx_File(RVsurfdir + "/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        return None
    NodeANTvtx = int(Node_ANT[0])

    Node_INF = meshIO.read_vtx_File(RVsurfdir + "/Node_INF.vtx")
    if len(Node_INF) == 0:
        return None
    NodeINFvtx = int(Node_INF[0])

    return [surfloc + "/RVmyo/tricuspid_intersect", surfloc + "/RVmyo/Node_RV_apex.vtx", NodeINFvtx, NodeANTvtx]


#########################################
# Function to compute rotationall UVC coordinate by Martin
#########################################
# PS: function(find_two_RV_insertion_pts) gives NodeINFvtx and NodeANTvtx by turn
# but in function(compute_uvc_rotational) read NodeINFvtx and NodeANTvtx represent RVinsertionNodeANT and RVinsertionNodeINF
# The index of RVinsertionNodeANT and RVinsertionNodeINF actually flipped 20260418
def compute_uvc_rotational(folder, meshtoolloc, meshname, base_surfFilename, apex_pointFilename, RVinsertionNodeANT,
                           RVinsertionNodeINF, phiname):
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apex_pointFilename)
    apex_vtx = int(apex_vtx[0])
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]

    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    centroids = meshIO.create_centroids(elems, pts)

    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    base_surf = meshIO.read_surf(base_surfFilename)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(), pts.loc[base_nodes][1].mean(), pts.loc[base_nodes][2].mean()]

    # Computes long axis between apex and CoM
    longAxis = (apex_coord - CoM_coord) / np.linalg.norm(apex_coord - CoM_coord)

    # Translates mesh to have x,y point aligned with apex
    pts_t = pts - apex_coord

    # Rotates mesh to align long axis with z [001] direction
    z_axis = [0, 0, 1]
    R = geometrical.computeRotationMatrix(z_axis, longAxis)
    pts_r = np.transpose(np.matmul(R, np.transpose(pts_t)))

    #########################################
    # Computes phi based on relative angles
    #########################################
    # Defines 2 RV insertion sites and coords
    RVinsPtANT_coord = pts_r.iloc[RVinsertionNodeANT]
    RVinsPtINF_coord = pts_r.iloc[RVinsertionNodeINF]

    # Picks out the xy coord (projecting it into xy plane)
    zero_phase_RVANT = [RVinsPtANT_coord[0], RVinsPtANT_coord[1]] / np.linalg.norm(
        [RVinsPtANT_coord[0], RVinsPtANT_coord[1]])
    zero_phase_RVINF = [RVinsPtINF_coord[0], RVinsPtINF_coord[1]] / np.linalg.norm(
        [RVinsPtINF_coord[0], RVinsPtINF_coord[1]])

    # find singularity
    sing_phase = -0.5 * (zero_phase_RVANT + zero_phase_RVINF)
    sing_phase = sing_phase / np.linalg.norm(sing_phase)

    # Projects all points in mesh into xy plane
    pts_proj = pts_r.loc[:, 0:1]
    mag_pts = np.linalg.norm(pts_proj, axis=1)

    # computes the angles between every point in the mesh and...
    # ... anterior RV insertion site
    cosThetas_RVANT = np.dot(pts_proj, zero_phase_RVANT) / mag_pts
    phi_RVANT = np.arccos(cosThetas_RVANT)
    # ... Interior RV insertion site
    cosThetas_RVINF = np.dot(pts_proj, zero_phase_RVINF) / mag_pts
    phi_RVINF = np.arccos(cosThetas_RVINF)
    # ... centre of free wall
    cosThetas_SING = np.dot(pts_proj, sing_phase) / mag_pts
    phi_SING = np.arccos(cosThetas_SING)

    # Defines array to store phi coords
    phi_coord = np.zeros(len(pts))

    # defines anterior nodes and interior nodes
    ant_LV_nodes = np.where(phi_RVANT < phi_RVINF)
    inf_LV_nodes = np.where(phi_RVANT >= phi_RVINF)

    phi_coord[ant_LV_nodes] = -np.pi / 2 - (np.pi / 2) * phi_RVANT[ant_LV_nodes] / (
                phi_RVANT[ant_LV_nodes] + phi_SING[ant_LV_nodes])
    phi_coord[inf_LV_nodes] = np.pi / 2 + (np.pi / 2) * phi_RVINF[inf_LV_nodes] / (
                phi_RVINF[inf_LV_nodes] + phi_SING[inf_LV_nodes])

    #############################
    # Computes phi for septum
    #############################
    septal_angle = np.arccos(np.dot(zero_phase_RVANT, zero_phase_RVINF))
    septal_nodes = np.intersect1d(
        np.intersect1d(np.where(phi_RVANT < septal_angle), np.where(phi_RVINF < septal_angle)),
        np.where(phi_SING > np.pi / 2))
    phi_coord[septal_nodes] = np.pi / 2 - np.pi * (
                phi_RVINF[septal_nodes] / (phi_RVANT[septal_nodes] + phi_RVINF[septal_nodes]))
    meshIO.write_data(dataFilename=phiname, data=phi_coord)

    if not meshtool.collect_nodal(meshtoolloc, meshname, "carp_txt", folder + "/Phi_LV", '', folder + "/" + phiname):
        return None

    return folder + "/" + phiname


########################################
# Function to compute rotationall UVC coordinate by Martin
#########################################
def compute_uvc_rotationalRV(folder, meshtoolloc, meshname, base_surfFilename, apex_pointFilename, RVinsertionNodeANT,
                             RVinsertionNodeINF, phiname):
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apex_pointFilename)
    apex_vtx = int(apex_vtx[0])
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]

    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    centroids = meshIO.create_centroids(elems, pts)

    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    base_surf = meshIO.read_surf(base_surfFilename)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(), pts.loc[base_nodes][1].mean(), pts.loc[base_nodes][2].mean()]

    # Computes long axis between apex and CoM
    longAxis = (apex_coord - CoM_coord) / np.linalg.norm(apex_coord - CoM_coord)

    # Translates mesh to have x,y point aligned with apex
    pts_t = pts - apex_coord

    # Rotates mesh to align long axis with z [001] direction
    z_axis = [0, 0, 1]
    R = geometrical.computeRotationMatrix(z_axis, longAxis)
    pts_r = np.transpose(np.matmul(R, np.transpose(pts_t)))

    #########################################
    # Computes phi based on relative angles
    #########################################
    # Defines 2 RV insertion sites and coords
    RVinsPtANT_coord = pts_r.iloc[RVinsertionNodeANT]
    RVinsPtINF_coord = pts_r.iloc[RVinsertionNodeINF]

    # Picks out the xy coord (projecting it into xy plane)
    zero_phase_RVANT = [RVinsPtANT_coord[0], RVinsPtANT_coord[1]] / np.linalg.norm(
        [RVinsPtANT_coord[0], RVinsPtANT_coord[1]])
    zero_phase_RVINF = [RVinsPtINF_coord[0], RVinsPtINF_coord[1]] / np.linalg.norm(
        [RVinsPtINF_coord[0], RVinsPtINF_coord[1]])

    # find singularity
    sing_phase = -0.5 * (zero_phase_RVANT + zero_phase_RVINF)
    sing_phase = sing_phase / np.linalg.norm(sing_phase)

    # Projects all points in mesh into xy plane
    pts_proj = pts_r.loc[:, 0:1]
    mag_pts = np.linalg.norm(pts_proj, axis=1)

    # computes the angles between every point in the mesh and...
    # ... anterior RV insertion site
    cosThetas_RVANT = np.dot(pts_proj, zero_phase_RVANT) / mag_pts
    phi_RVANT = np.arccos(cosThetas_RVANT)
    # ... Interior RV insertion site
    cosThetas_RVINF = np.dot(pts_proj, zero_phase_RVINF) / mag_pts
    phi_RVINF = np.arccos(cosThetas_RVINF)

    # Defines array to store phi coords
    phi_coord = np.zeros(len(pts))

    # defines anterior nodes and interior nodes
    ant_LV_nodes = np.where(phi_RVANT < phi_RVINF)
    inf_LV_nodes = np.where(phi_RVANT >= phi_RVINF)

    phi_coord = np.pi / 2 - np.pi * (phi_RVINF / (phi_RVANT + phi_RVINF))
    meshIO.write_data(dataFilename=phiname, data=phi_coord)

    if not meshtool.collect_nodal(meshtoolloc, meshname, "carp_txt", folder + "/Phi_RV", '', folder + "/" + phiname):
        return None
    return folder + "/" + phiname


def merge_PHI(folder, meshtoolloc, meshname, phiLV, phiRV, surfloc, UVCfolder):
    os.chdir(folder)
    if not meshtool.insert_data(meshtoolloc, 'LVmyo_i', phiLV, meshname + '_i', 'Phi_insertLV.dat', '0'):
        return None
    if not meshtool.insert_data(meshtoolloc, 'RVmyo_i', phiRV, meshname + '_i', 'Phi_insertRV.dat', '0'):
        return None
    Phi_insertLV = np.genfromtxt('Phi_insertLV.dat')
    Phi_insertRV = np.genfromtxt('Phi_insertRV.dat')
    if not meshtool.extract_surf(meshtoolloc, meshname + '_i', surfloc + "/biv_i/RVinsertion", "0:1"):
        return None
    RVinsertionvtx = meshIO.read_vtx_File(surfloc + "/biv_i/RVinsertion.surf.vtx")
    RVinsertionvtx = RVinsertionvtx.astype(int)
    Phi = Phi_insertLV + Phi_insertRV
    Phi[RVinsertionvtx] = Phi_insertLV[RVinsertionvtx]
    meshIO.write_data(dataFilename=UVCfolder + '/PHI.dat', data=Phi)
    if not meshtool.collect_nodal(meshtoolloc, meshname, "carp_txt", UVCfolder + "/PHI", '', UVCfolder + '/PHI.dat'):
        return None
    return UVCfolder + '/PHI.dat'


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# This function is a modified version of the original by Martin Bishop
def compute_fiber_new(folder, meshtoolloc, meshname, apexvtx, base_surf, PHOname, coordszname, alpha_epi, alpha_endo,
                      beta_epi, beta_endo):
    '''
    return the mesh with longitudinal fiber according to alpha_epi and alpha_endo (usually:pi/3 and -pi/3)
    the 'meshname' is only LV or RV
    '''
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apexvtx)
    apex_vtx = int(apex_vtx[0])
    # meshname="LVmyo"
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]

    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    # centroids = meshIO.create_centroids(elems, pts)

    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    # base_surf=predata[0]
    base_surf = meshIO.read_surf(base_surf)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(), pts.loc[base_nodes][1].mean(), pts.loc[base_nodes][2].mean()]

    # Computes long axis between apex and CoM
    longAxis = (apex_coord - CoM_coord) / np.linalg.norm(apex_coord - CoM_coord)
    # extract the PHO and COORDS_Z for the specific mesh (LV or RV) to a new .dat
    if not meshtool.extract_data(meshtoolloc, meshname, PHOname, 'PHO_' + meshname + '.dat', "0"):
        return None
    if not meshtool.extract_data(meshtoolloc, meshname, coordszname, 'COORDS_Z' + meshname + '.dat', "0"):
        return None
    # calculate the gradient of the PHO and COORDS_Z

    if not meshtool.extract_gradient(meshtoolloc, meshname, 'COORDS_Z' + meshname + '.dat',
                                     'COORDS_Z_' + meshname + '_gradient', '1'):
        return None
    if not meshtool.extract_gradient(meshtoolloc, meshname, 'PHO_' + meshname + '.dat', 'PHO_' + meshname + '_gradient',
                                     '1'):
        return None

    transmuralFile = 'PHO_' + meshname + '.dat'
    coordz_vec = np.genfromtxt('COORDS_Z_' + meshname + '_gradient.grad.vec')
    transmural_vec = np.genfromtxt('PHO_' + meshname + '_gradient.grad.vec')
    print('write in all three files!')
    # calculate the fiber along the myo surface (circumferential direction)
    circumferential = np.cross(transmural_vec, -coordz_vec)

    # calculate the PHO for each element by averaging the PHO for four pts
    transmural = np.genfromtxt(transmuralFile)
    trans_n0 = transmural[elems.iloc[:, 0]]
    trans_n1 = transmural[elems.iloc[:, 1]]
    trans_n2 = transmural[elems.iloc[:, 2]]
    trans_n3 = transmural[elems.iloc[:, 3]]
    mean_trans = (trans_n0 + trans_n1 + trans_n2 + trans_n3) * 0.25

    alpha_trans = np.zeros(len(circumferential))
    beta_trans = np.zeros(len(circumferential))

    lon_longitudinal = circumferential  # to initialize the sheet lon vector
    lonsheet = circumferential  # to initialize the sheet lon vector

    lonall = np.zeros((len(circumferential), 8))
    lonall[:] = np.nan
    for i in range(len(circumferential)):
        alpha_trans[i] = -((alpha_endo - alpha_epi)) * mean_trans[i] + alpha_endo
        # rotate the fiber vectors by alpha_trans (in radian) along transmural_vec
        lon_longitudinal[i] = np.dot(rotation_matrix(transmural_vec[i], alpha_trans[i]), circumferential[i])
        lon_longitudinal[i] = lon_longitudinal[i] / np.linalg.norm(lon_longitudinal[i])
        lonall[i, 0:3] = lon_longitudinal[i]

    for i in range(len(circumferential)):
        beta_trans[i] = -((beta_endo - beta_epi)) * mean_trans[i] + beta_endo
        lonsheet[i] = np.dot(rotation_matrix(circumferential[i], beta_trans[i]), transmural_vec[i])
        lonsheet[i] = lonsheet[i] / np.linalg.norm(lonsheet[i])
        lonall[i, 5:8] = lonsheet[i]

    meshIO.write_lon_includesheet(lonFilename=meshname + '_fiber', lon=pd.DataFrame(lonall))
    meshIO.write_lon(lonFilename=meshname + '_longitudinal_fiber', lon=pd.DataFrame(lon_longitudinal))
    meshIO.write_lon(lonFilename=meshname + '_sheet_fiber', lon=pd.DataFrame(lonsheet))
    # meshIO.write_lon(lonFilename=meshname+'_fiber', lon=pd.DataFrame(lon_longitudinal))

    shutil.copyfile(meshname + ".elem", meshname + "_fiber.elem")
    shutil.copyfile(meshname + ".pts", meshname + "_fiber.pts")
    if not meshtool.convert_mesh(meshtoolloc, meshname + "_fiber", meshname + "_fiber", ofmt="vtk"):
        return None
    return meshname + "_fiber"


def merge_fibertomyoonly(folder, meshtoolloc, meshname, fiberfolder):
    os.chdir(folder)
    meshall_fiber = meshIO.read_fibres(basename=meshname + '_i')
    lonall = np.zeros((len(meshall_fiber.index), 8))
    meshIO.write_lon_includesheet(lonFilename=meshname + '_i', lon=pd.DataFrame(lonall))
    submesh = 'LVmyo_i'
    shutil.copyfile(submesh + '.eidx', submesh + '_fiber.eidx')
    shutil.copyfile(submesh + '.nod', submesh + '_fiber.nod')
    if not meshtool.insert_submesh(meshtoolloc, submesh + '_fiber', meshname + '_i', meshname + '_insertLV'):
        return False
    if not meshtool.convert_mesh(meshtoolloc, meshname + '_insertLV', meshname + '_insertLV', ofmt="vtk"):
        return False
    submesh = 'RVmyo_i'
    shutil.copyfile(submesh + '.eidx', submesh + '_fiber.eidx')
    shutil.copyfile(submesh + '.nod', submesh + '_fiber.nod')
    finalmesh_i = meshname + '_i_fiber'
    if not meshtool.insert_submesh(meshtoolloc, submesh + '_fiber', meshname + '_insertLV', finalmesh_i):
        return False
    if not meshtool.convert_mesh(meshtoolloc, finalmesh_i, finalmesh_i, ofmt="vtk"):
        return False

    submesh = meshname + '_i'
    shutil.copyfile(submesh + '.eidx', finalmesh_i + '.eidx')
    shutil.copyfile(submesh + '.nod', finalmesh_i + '.nod')

    meshall = meshIO.read_fibres(basename=meshname)
    lonall1 = np.zeros((len(meshall.index), 8))
    meshIO.write_lon_includesheet(lonFilename=meshname, lon=pd.DataFrame(lonall1))

    finalmesh = fiberfolder + '/' + meshname + '_fiber'
    if not meshtool.insert_submesh(meshtoolloc, finalmesh_i, meshname, finalmesh):
        return False
    meshall_pts = meshIO.read_pts(basename=meshname)
    # change pts unit from 'mm' to 'um' to fit for carp simulations
    meshall_pts = meshIO.write_pts(ptsFilename=finalmesh, pts=meshall_pts * 1000)
    if not meshtool.convert_mesh(meshtoolloc, finalmesh, finalmesh, ofmt="vtk"):
        return False

    return True


def bivlabel(folder, meshname, UVCfolder):
    # make a dat file to seperate LV and RV
    # LV: -1, RV:1
    os.chdir(folder)
    pts = meshIO.read_pts(basename=meshname + '_i')
    elem = meshIO.read_elems(basename=meshname + '_i')

    bivcoord = np.ones(len(pts))
    for i in range(len(elem)):
        if elem.loc[i, 5] == 1:
            bivcoord[elem.loc[i, 1]] = -1
            bivcoord[elem.loc[i, 2]] = -1
            bivcoord[elem.loc[i, 3]] = -1
            bivcoord[elem.loc[i, 4]] = -1
    meshIO.write_list(dataFilename=UVCfolder + '/biv.dat', data=bivcoord)
    return UVCfolder + '/biv.dat'
