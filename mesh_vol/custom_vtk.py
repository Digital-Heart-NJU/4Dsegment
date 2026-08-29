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
