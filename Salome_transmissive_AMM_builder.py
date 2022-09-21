import numpy as np, os, math, salome, GEOM, killSalome, time
from salome.geom import geomBuilder
import math
import SALOMEDS
salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
# main_path = r"E:/OneDrive - University College London/constrained_GS/fabrication"
main_path = r"C:/Users/James/OneDrive - University College London/constrained_GS/fabrication"
sys.path.insert(0, main_path)
geompy = geomBuilder.New()

# https://stackoverflow.com/questions/13266480/running-salome-script-without-graphics
# https://docs.salome-platform.org/7/gui/KERNEL/running_salome_page.html
# https://docs.salome-platform.org/6/gui/GEOM/tui_test_others_page.html

# ---> physics params <---
lam = 8.66/1000  # [m]
dx = lam/2 # cell spacing [m]

# ---> brick params <---
brick_model = "metabricks_2022"
file_type = "step"
brick_filename = "22_brick"

# ---> brickmap params <---
m, n = 12, 30

AMM_filename = "twin_trap-A0=3.2921-A1=3.0411-AMM_dist=12lam-board_dist=0.2-m=12-n=30-tran_m=16-tran_n=16-shift=0.5lam"
brickmap = np.load(main_path+"/static_AMMs/brickmaps/"+AMM_filename+".npy")

fuse_list = []
fuse_type = "merge"    

for col in range(n): 
    for row in range(m):
        print(brickmap[row][col])
        [Solid, Generic_SOLID] = geompy.ImportSTEP(main_path+"/"+brick_model+"/"+file_type+"/"+brick_filename+str(brickmap[row][col])+".stp", False, True)
        translation = geompy.MakeTranslation(Solid, dx*col, 0, dx*row)
        geompy.addToStudy(translation, 'translation_'+str(col)+"_"+str(row))
        fuse_list.append(translation) # The element is added to the list

if fuse_type == "partition":       
    fused_object = geompy.MakePartition(fuse_list, [], [], [], geompy.ShapeType["SOLID"], 0, [], 0) # Merging each element in the list into one object

elif fuse_type == "merge": 
    fused_object = geompy.MakeFuseList(fuse_list, rmExtraEdges=True)
    
geompy.ExportSTEP(fused_object, main_path+"/static_AMMs/step/"+AMM_filename+".step", GEOM.LU_MILLIMETER) # Exporting the segment as a .step file, using millimeters as length units.
fused_object = geompy.Scale(fused_object, thePoint=None, theFactor=1000) # must scale the model up by 1000x before exporting as stl cos Salome is dumb.
geompy.ExportSTL(fused_object, main_path+"/static_AMMs/stl/"+AMM_filename+".stl", True, 0.001, True) # Exporting the segment as a .stl file, using millimeters as length units.
