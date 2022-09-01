import numpy as np, os, math, salome, GEOM, killSalome, time
from salome.geom import geomBuilder
import math
import SALOMEDS
salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
main_path = r"E:/OneDrive - University College London/constrained_GS/fabrication"
sys.path.insert(0, main_path)
geompy = geomBuilder.New()

# https://stackoverflow.com/questions/13266480/running-salome-script-without-graphics
# https://docs.salome-platform.org/7/gui/KERNEL/running_salome_page.html
# https://docs.salome-platform.org/6/gui/GEOM/tui_test_others_page.html

# ---> physics params <---
lam = 8.661/1000  # [m]
dx = lam/2 # cell spacing [m]

# ---> brick params <---
brick_model = "metabricks_2022"
file_type = "step"
brick_filename = "22_brick"

# ---> brickmap params <---
m, n = 12, 30

AMM_filename = "twin_trap-A0=3.9711-A1=2.9911-AMM_dist=12lam-board_dist=0.2-m=12-n=30-tran_m=16-tran_n=16-shift=0.5lam"
brickmap = np.load(main_path+"/static_AMMs/brickmaps/"+AMM_filename+".npy")

merge_list = []

for col in range(n): 
    for row in range(m):
        print(brickmap[row][col])
        [Solid, Generic_SOLID] = geompy.ImportSTEP(main_path+"/"+brick_model+"/"+file_type+"/"+brick_filename+str(brickmap[row][col])+".stp", False, True)
        translation = geompy.MakeTranslation(Solid, dx*col, 0, dx*row)
        geompy.addToStudy(translation, 'translation_'+str(col)+"_"+str(row))
        merge_list.append(translation) # The element is added to the list
        
merged_object = geompy.MakePartition(merge_list, [], [], [], geompy.ShapeType["SOLID"], 0, [], 0) # Merging each element in the list into one object

geompy.ExportSTEP(merged_object, main_path+"/static_AMMs/step/"+AMM_filename+".step", GEOM.LU_MILLIMETER) # Exporting the segment as a .step file, using millimeters as length units.
merged_object = geompy.Scale(merged_object, thePoint=None, theFactor=1000) # must scale the model up by 1000x before exporting as stl cos Salome is dumb.
geompy.ExportSTL(merged_object, main_path+"/static_AMMs/stl/"+AMM_filename+".stl", True, 0.001, True) # Exporting the segment as a .stl file, using millimeters as length units.
