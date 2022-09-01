import numpy as np
import os, math, salome, GEOM, killSalome, time
from salome.geom import geomBuilder
from os import path
main_path = r"C:/Users/James/OneDrive - University College London/Activation of Metasurface/Ben Motors/SSM_assembly/static/static_lenses/focus_m12_n12_f=6.0lam_to_14.0lam_step=2.0lam"
# main_path = r"E:/OneDrive - University College London/Activation of Metasurface/Ben Motors/SSM_assembly/static/static_lenses/focus_m12_n12_f=6.0lam_to_14.0lam_step=2.0lam"
geompy = geomBuilder.New() # Creation of a geomBuilder instance which provides GEOM operations (used for geometry)

c = 343 # m/s
v = 40000 # Hz
lam = c/v # m
elem_side = (lam/2)
buffer = lam/4

filename = "seg_heightmap_ID=5"
heightmap = np.load(main_path + "/heightmaps/" + filename + ".npy")
m, n = heightmap.shape

fuse_type = "merge"
fuse_list = []

for cols in range(m) : 
    for rows in range(n) : 
        elem_height = heightmap[cols, rows]
        elem = geompy.MakeBoxDXDYDZ(elem_side, elem_side, elem_height+buffer, "Square") 
        elem = geompy.MakeTranslation(elem, (elem_side)*cols, (elem_side)*rows, 0) 
        study = geompy.addToStudy(elem, "Square_" + str(cols) + "_" + str(rows))
        fuse_list.append(elem) 

if fuse_type == "partition":
    fused_obj = geompy.MakePartition(fuse_list, [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
    study = geompy.addToStudy(fused_obj, "partitioned_lens")
    geompy.ExportSTEP(fused_obj, main_path + "/step/" + filename + ".step", GEOM.LU_MILLIMETER)
    geompy.ExportSTL(fused_obj, main_path+"/stl/" + filename + ".stl", True, 0.001, True)

elif fuse_type == "merge":
    fused_obj = geompy.MakeFuseList(fuse_list, rmExtraEdges=True)
    study = geompy.addToStudy(fused_obj, "partitioned_lens")
    geompy.ExportSTEP(fused_obj, main_path + "/step/" + filename + ".step", GEOM.LU_MILLIMETER)
    geompy.ExportSTL(fused_obj, main_path+"/stl/" + filename + ".stl", True, 0.001, True)

else:
    print("Please specify a valid fuse type ('partition' or 'merge').")
