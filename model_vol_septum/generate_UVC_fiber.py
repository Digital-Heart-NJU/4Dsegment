'''
python run.py --data-dir /home/mengao/4t/Dicemesh --carp-bin-dir /opt/openCARP/usr/bin --uvc-fiber --subject subject001
python run.py --data-dir /home/mengao/Code/test_1/biv_volumetric_meshing/src/segmentation/data/UKB_example_4s --carp-bin-dir /opt/openCARP/usr/bin --uvc-fiber --subject subject004
'''

import argparse
import configparser
import numpy as np
import pyvista as pv
import os
import itertools
import logging
import re
import shutil

from . import compute_UVC_fiber as UVC
from . import meshIO
from . import meshtool_func as meshtool
from . import py_atrial_fibres as paf
from . import v_mesh_generation as vmesh

# logger = logging.getLogger('volumetric')

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

    # if not meshtool.extract_mesh(meshtoolloc, mesh, "1", "LV_all"): # 只提取LV_all部分,此部分的tag为1
    #     return None
    meshtool.extract_mesh(meshtoolloc, mesh, "1", "LV_all") # 只提取LV_all部分,此部分的tag为1
    print("提取左心室体网格")
    return
    # Mesh_mitral_valve_surf = meshnames[5]

    # if not meshtool.convert_mesh(meshtoolloc, input_dir+f"/{Mesh_mitral_valve_surf}", input_dir+f"/{Mesh_mitral_valve_surf}", "vtk", "carp_txt"):
    #     return None

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

    # find the apex point' index in biv mesh
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


CARP_BIN = None
OPTS = None

# def run_timeframe_uvc(subject_id, timeframe, input_dir, mesh_names, par_file, output_dir, output_dir_uvc):
subject_id = "subject_1"
timeframe = "ED"

print('subject {}, timeframe {}: started uvc'.format(subject_id, timeframe))

meshtoolloc = os.path.join(CARP_BIN, 'meshtool')

CARP = os.path.join(CARP_BIN, 'openCARP')
IGBEXTRACT = os.path.join(CARP_BIN, 'igbextract')
MPIEXEC = os.path.join(CARP_BIN, 'mpiexec')

output_dir = "/home/mengao/Code/4Dsegment/data/subject_1/volmesh_generate_pipline (Copy 4)"
meshdir = os.path.join(output_dir, 'step11volmesh_LVRV') #volmesh_generate_pipline (Copy 4)/step11volmesh_LVRV
meshname = subject_id + "_{}".format_map(timeframe) + "_stage11_final_elemTag_volmesh"

#???
surfloc = "surfaces"
# ???

if not os.path.exists(meshdir):
    print('subject {}, timeframe {}: no volumetric mesh found'.format(subject_id, timeframe))
#     return

output_dir_uvc = os.path.join(output_dir, 'step12volmesh_fiber')
finalmeshdir = output_dir_uvc   #volmesh_generate_pipline (Copy 4)/step12volmesh_fiber

os.chdir(finalmeshdir)

if not meshtool.convert_mesh(meshtoolloc, meshdir + '/' + meshname, meshdir + '/' + meshname, ifmt="vtk", ofmt="carp_txt"):
    print('failed to convert mesh')

if not os.path.exists(meshdir + '/' + meshname + ".elem"):  # volumetric_mesh_outputs/timeframe/subject_id_mesh_all
    print('subject {}, timeframe {}: no volumetric mesh found'.format(subject_id, timeframe))

shutil.copyfile(meshdir + '/' + meshname + ".elem", meshname + ".elem")
shutil.copyfile(meshdir + '/' + meshname + ".pts", meshname + ".pts")
shutil.copyfile(meshdir + '/' + meshname + ".lon", meshname + ".lon")
shutil.copyfile(meshdir + '/' + meshname + ".vtk", meshname + ".vtk")

UVCfolder = finalmeshdir + '/UVC_i'
if not os.path.exists(UVCfolder):
    os.mkdir(UVCfolder)

input_dir = "/home/mengao/Code/4Dsegment/data/subject_1"
mesh_names = subject_id + "_{}".format_map(timeframe)
# UVC.coord z generation
try:
    # (finalmeshdir:UVC_fiber_Outputs/timeframe,
    # meshtoolloc:meshtool,
    # meshname:subject_mesh_all,
    # input_dir:Mesh_Outputs
    # surfloc:surfaces,
    # UVCfolder:UVC_fiber_Outputs/timeframe/UVC_i)
    # coordszname = UVC.coord_z(finalmeshdir, meshtoolloc, meshname, input_dir, mesh_names, surfloc, UVCfolder)
    coordszname = coord_z(finalmeshdir, meshtoolloc, meshname, input_dir, mesh_names, surfloc, UVCfolder)

except Exception:
#     # logger.error('subject {}, timeframe {}: failed to generate coord z'.format(subject_id, timeframe_num))
    print('subject {}, timeframe {}: failed to generate coord z'.format(subject_id, timeframe))
#     return
if coordszname == None:
    # logger.error('subject {}, timeframe {}: failed to generate coord z'.format(subject_id, timeframe_num))
    print('subject {}, timeframe {}: failed to generate coord z'.format(subject_id, timeframe))
    # return

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
    parser.add_argument('--start', '-S', action='store', type=int, help='index of first subject id to be processed')
    parser.add_argument('--number', '-n', action='store', type=int, help='number of subjects to be processed from first subject id')
    parser.add_argument('--allowlist', '-l', action='store', help='path to subject allowlist')

    parser.add_argument('--timeframe', '-t', type=int, action='store', help='timeframe to be processed')

    parser.add_argument('--mesh', '-m', action='store_true', help='generate mesh')
    parser.add_argument('--uvc', '-u', action='store_true', help='generate uvc and fiber')
    parser.add_argument('--uvc-meshes', '-z', action='store_true', help='generate uvc meshes')
    parser.add_argument('--quality-control', '-q', action='store_true', help='output quality control screenshots')

    args, _ = parser.parse_known_args()

    cfg = configparser.ConfigParser()
    cfg.read('config.ini')

    global CARP_BIN
    CARP_BIN = args.carp_bin_dir if args.carp_bin_dir else cfg[args.profile]['CarpBinDir']

    global OPTS
    OPTS = os.path.abspath('amg_cg_opts')

    data_dir = args.data_dir if args.data_dir else cfg[args.profile]['DataDir']
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

            if do_mesh or args.mesh:
                vmtf_dir = os.path.join(output_dir, 'timeframe{}'.format(tf_n))
                if not os.path.exists(vmtf_dir):
                    os.mkdir(vmtf_dir)

                if os.path.exists(os.path.join(vmtf_dir, subject_id + '_mesh_all', subject_id + '_mesh_all.vtk')) and not args.force:
                    logger.debug('subject {}, timeframe {}: skipping, volumetric mesh exists'.format(subject_id, tf_n))
                    continue

                run_timeframe(subject_id, tf_n, input_dir, mesh_names, par_file, vmtf_dir)

                for filename in os.listdir(vmtf_dir):
                    filepath = os.path.join(vmtf_dir, filename)
                    if os.path.isdir(filepath):
                        if filename != f'{subject_id}_mesh_all':
                            pass  # shutil.rmtree(filepath)
                    else:
                        pass  # os.remove(filepath)

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

                for filename in os.listdir(uftf_dir):
                    filepath = os.path.join(uftf_dir, filename)
                    if os.path.isdir(filepath):
                        if filename not in ['fiberfolder', 'surfaces', 'UVC_i']:
                            pass  # shutil.rmtree(filepath)
                    else:
                        pass  # os.remove(filepath)

if __name__ == '__main__':
    main()


