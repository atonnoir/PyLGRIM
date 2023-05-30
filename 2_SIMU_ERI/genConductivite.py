#!/usr/bin/python3.4
# -*-coding:Utf-8 -*

# by A. Tonnoir

import numpy as np
from scipy import linalg
import copy

# Added "modules"
import sys
sys.path.append("../Lib/")
import mesh
import dataERI
import finiteElem  


    
#########################################################
## Lecture maillage et construction matrices EF :
# Lecture du maillage
monMaillage = "../Data/meshDEM.msh"
print("Reading mesh 3D")
[ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D] = mesh.readMsh3(monMaillage)
print("Fin lecture maillage 3D")


# Construction des conductivites
sigma = []

print("For each mesh")
for ite,m in enumerate(m3D):
    pt1 = ptsM[m[0]]
    pt2 = ptsM[m[1]]
    pt3 = ptsM[m[2]]
    pt4 = ptsM[m[3]]
    conduc = 1.0
    conduc = 1.0/40.0 # wet sand
    bary = (pt1 + pt2 + pt3 + pt4)/4.0
    centerBall1 = np.array([0.65,-0.2,-0.0])
    centerBall2 = np.array([0.2,0.35,-0.25])
    if(bary[2] <= -0.25):
        conduc = 1.0/100.0 # dry sand
    if ( np.dot(bary-centerBall1,bary-centerBall1)  <= 0.15*0.15 ):
        conduc = 1.0/5.0
    if ( np.dot(bary-centerBall2,bary-centerBall2)  <= 0.1*0.1 ):
        conduc = 1.0/5.0
    
    # Homogene :
    conduc = 1.0
        
        
    sigma.append(conduc)    
#########################################################
    
    
#########################################################
np.savetxt('conductivite.data',sigma)
sigma = np.array(sigma)
mesh.writeCellVTK('resistivity.vtk',1./sigma,ptsM,m3D)
#########################################################





print("End of the program")