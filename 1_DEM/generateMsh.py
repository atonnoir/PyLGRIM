#!/usr/bin/python3.4
# -*-coding:Utf-8 -*

# by A. Tonnoir

import numpy as np
import copy

from scipy import spatial
import matplotlib.pyplot as plt

import sys
sys.path.append("../Lib/")
import mesh
import utils

import os
    

# Find altitude of the closest neighbourg     
def topoDEM(x,y,listCoordXYZ,tree): # listCoord
    [dist,ind] = tree.query([x,y])
    return listCoordXYZ[ind,2]

    
# Altitude defined by a given function  
def topoF(x,y):
    return 0.0
    # Step like topography:
    if(x<=0.3): 
        return 0.0
    if(x>=0.3 and x<=0.6):
        return (x-0.3)/(0.6-0.3)*0.5
    if(x>=0.6):
        return 0.5        



print("Start of the program")
print("STEP 1 : Construction of the .geo file.")
#########################################################
## Choice of the case: 
# 1. DEM from data file
# 2. DEM from the function topoF above
print("To create DEM mesh file, do you want to use :")
print("1. an (x,y,z) points from files in Data repertory")
print("2. a defined function")
print("3. a .geo file with an (x,y,z) points from files in Data repertory")
choice = input("Answer: ")

DEMfile = ""
ListElectrodeLoc = []
if (choice == "1" or choice == "3"):
    DEMfile = "../Data/MNT_2016-04-18_VN_JL - Cloud.txt";
    #ListElectrodeLoc = ["../Data/localisation_TV1_BAS_A_UTILISER.txt"];#"../Data/localisation_P5.txt", "../Data/localisation_TV1_BAS_A_UTILISER.txt", "../Data/localisation_P2.txt" ,"../Data/localisation_P1.txt" 
    ListElectrodeLoc = ["../Data/localisation_TV1_BAS_A_UTILISER.txt","../Data/localisation_TV2.txt","../Data/localisation_P5.txt","../Data/localisation_P2.txt","../Data/localisation_P1.txt"]; # ,"../Data/localisation_P5.txt","../Data/localisation_P2.txt" 
    
    # Test Dalle beton :
    #DEMfile = "../Data/dalle_beton_PVC_Cloud.txt";
    #ListElectrodeLoc = ["../Data/positions_electrodes_ok.txt"];
    
    # Test Dalle beton :
    #DEMfile = "../Data/Chapelle/model_chapelle_pymeri_ok.txt";
    #ListElectrodeLoc = ["../Data/Chapelle/coord_electrodes1m.txt"];
    
    # Test Terril :
    #DEMfile = "../Data/Terril/MNTL93_728_7028_MNT_terril_Cloud.txt";
    #ListElectrodeLoc = ["../Data/Terril/positions_electrodes_P3_ok.txt","../Data/Terril/positions_electrodes_P1_ok.txt","../Data/Terril/positions_electrodes_P2_ok.txt"]; # ,"../Data/Terril/positions_electrodes_P3_ok.txt"
    #ListElectrodeLoc = ["../Data/Terril/positions_electrodes_P3_ok.txt","../Data/Terril/positions_electrodes_P1_ok.txt"]; # ,"../Data/Terril/positions_electrodes_P2_ok.txt"
    
if (choice == "3"):    
    geoFile = "../Data/meshDEMFlat.geo"
#########################################################

#########################################################
# Parameter for discretization aspect :
# h : the smaller h is, the finer are the mesh around the electrode
# coeff : this parameter controle the fact that the mesh become larger far from the electrodes 
coeff = 10.0
h = 1.0

# Parameter to enlarge the domain of interest (w.r.t. x and y):
Lx = 20.
Ly = 20.0
# Deepness of the mesh:
minZ = -40.0
#########################################################

#########################################################
## Reading points :
xmin,xmax,ymin,ymax,zmin,zmax = 0.0,0.0,0.0,0.0,0.0,0.0
Lz = 0.0
ptsDEM = []
ptsElec = []
listPts = []
resRecur = True
ptsElec = []
indElecSE = [0]
    
# Case of a DEM given by a points cloud:
if(choice == "1" or choice == "3"):
    # Recuperation of the DEM data
    ptsDEM = np.genfromtxt(DEMfile, delimiter=" ")
    # Restriction to coordinates (x,y,z) of the DEM
    ptsDEM = ptsDEM[:,0:3]
    
    # Number of electrode lines
    nbLig = len(ListElectrodeLoc)
    
    # Recuperation of electrode positions
    # ptsElec = []
    # indElecSE = [0]
    for j,name in enumerate(ListElectrodeLoc):
        if(j==0):
            ptsElec = np.genfromtxt(name, delimiter=" ")
        else:
            ptsElectmp = np.genfromtxt(name, delimiter=" ")
            # if(j==1):
            #     ptsElectmp = ptsElectmp[::-1] # On renverse l'ordre
            indElecSE.append(len(ptsElec[:,0]))
            ptsElecAdd = []
            for pttmp in ptsElectmp:
                inList = 0
                for pt in ptsElec:
                    dist = np.sqrt(np.dot(pt-pttmp,pt-pttmp))
                    if(dist <= 1e-3):
                        inList = 1
                        break
                if(not(inList)):
                    ptsElecAdd.append(pttmp)
                    
            ptsElecAdd = np.array(ptsElecAdd)        
            ptsElec = np.concatenate((ptsElec,ptsElecAdd))
            
    indElecSE.append(len(ptsElec[:,0]))     
    
    # for i,pt in enumerate(ptsElec):
#         plt.plot(pt[0],pt[1],'x')
#
#     plt.show()   
        

# Case of a topography given by a function f:
if(choice == "2"):
    # Number of electrode lines :
    nbLig = 1
    
    p0 = np.array([0,0])
    p1 = np.array([1,1])
    nbElec = 1
    ptsElec = np.array(ptsElec)
    for i in range(nbLig):
        # Line i:
        # Parameters for line i=0
        if(i==0):
            # Starting point (in 2D)
            p0 = np.array([0.1,-0.0])
            # End point (in 2D)
            p1 = np.array([0.9,-0.0])
            # Number of electrode
            nbElec = 64
        # Parameters for line i=1
        if(i==1):
            # Starting point (in 2D)
            p0 = np.array([0.10,0.4])
            # End point (in 2D)
            p1 = np.array([0.75,-0.3])
            # Number of electrode
            nbElec = 32
                
        hElec = 1.0 / (nbElec-1)
        tt = np.arange(0,1.0+hElec*0.5,hElec)
        ptsElecAdd = []
        for t in tt:
            coord2D = p0 + (p1-p0)*t
            x = coord2D[0]
            y = coord2D[1]
            z = topoF(x,y)
            ptsElecAdd.append([x,y,z])
        
        ptsElecAdd = np.array(ptsElecAdd)   
        if(i==0): 
            ptsElec = ptsElecAdd*1.0
        else:
            ptsElec = np.concatenate((ptsElec,ptsElecAdd))    
            
        indElecSE.append(len(ptsElec[:,0]))    
#########################################################


#########################################################
# Path construction between each electrode lignes :        
# Size of the domain of interest:

xmin = min(ptsElec[:,0])
xmax = max(ptsElec[:,0])
xmoy = np.sum(ptsElec[:,0]) / len(ptsElec[:,0])
ymin = min(ptsElec[:,1])
ymax = max(ptsElec[:,1])
ymoy = np.sum(ptsElec[:,1]) / len(ptsElec[:,1])
zmin = min(ptsElec[:,2])
zmax = max(ptsElec[:,2])

xmin = xmin-Lx
xmax = xmax+Lx
ymin = ymin-Ly
ymax = ymax+Ly

Lz = max(xmax-xmin,ymax-ymin)
    
if(choice == "1" or choice =="2"):

    # Shift :
    ptsElec[:,0] = ptsElec[:,0] - xmoy
    ptsElec[:,1] = ptsElec[:,1] - ymoy
    xmin = xmin - xmoy
    xmax = xmax - xmoy
    ymin = ymin - ymoy
    ymax = ymax - ymoy

    Lz = max(xmax-xmin,ymax-ymin)

  


    resRecur = False 
    # 1 line case:
    if(nbLig == 1):
        pt1 = ptsElec[0,0:2]
        ptEnd = ptsElec[-1,0:2]
        ptStart = np.array([xmin,ymax])
        dist1 = np.dot(pt1 - ptStart,pt1 - ptStart)
        distEnd = np.dot(ptEnd - ptStart,ptEnd - ptStart)
        if(distEnd < dist1):
            # Flip the list order:
            ptsElec = np.flip(ptsElec,0)
        resRecur = True    

    # >= 2 lines case:   
    if(nbLig >= 2 and nbLig <= 7): 
        resRecur = False 
        listLine = []
        for i in range(nbLig):
            line = ptsElec[indElecSE[i]:indElecSE[i+1],:]
            listLine.append(line)
    
        listSeg = []
    
        # Case of crossing lines:
        # Split line1 and line2 into four lines that are not 
        newlistLine = []
    
        for i,lineI in enumerate(listLine):
            splitLineI = []
            vProj = lineI[-1,0:2] - lineI[0,0:2]
            projectI = np.dot(lineI[:,0:2]-lineI[0,0:2],vProj)
            indDecoup = [0]
            for j, lineJ in enumerate(listLine):
                if(j != i):
                    [doCross,CrossPt,s1,s2] = mesh.checkIntersection(lineI,lineJ)
                    if(doCross == True):
                        coordProj = np.dot(CrossPt-lineI[0,0:2],vProj)
                        indDecoup.append(len( (np.where(projectI <coordProj))[0] ))
            indDecoup.append(len(lineI))
            indDecoup.sort()
        
            for j in range(len(indDecoup)-1):
                newlistLine.append(lineI[indDecoup[j]:indDecoup[j+1],:])            
            
        listLine = newlistLine                
    
        # Case of // lines (not crossing lines)
        
        endPt = np.array([xmax,ymin])
        res = [np.array([xmin,ymax])]
    
        for i,line in enumerate(listLine):
            seg = np.array([line[0,0:2],line[-1,0:2]])
            listSeg.append(seg)

        #resRecur = mesh.computeLineSeqRecurs(listSeg,res,endPt)
        print("Start recursion :")
        resRecur = mesh.computeLineSeqRecurs(listLine,res,endPt)
    
        print("ResRecurs : ",resRecur)
        #print("Res :",res)
        plt.ion()
        for i,e in enumerate(res):
            if(i < len(res)-1):
                plt.plot([e[0],res[i+1][0]],[e[1],res[i+1][1]],'k-')
            plt.show()
            plt.pause(0.01)
        plt.plot([res[-1][0],xmax],[res[-1][1],ymin],'k-')    
        plt.ioff()
        input("Enter 1 to continue :")
    
        if(resRecur == True):
            res.pop(0)
            ptsElec = np.array(res)
            #print(ptsElec)

    if(resRecur == False):
        ptsElec = ptsElec[ptsElec[:,0].argsort()]  
                
    # Shift :
    ptsElec[:,0] = ptsElec[:,0] + xmoy
    ptsElec[:,1] = ptsElec[:,1] + ymoy
    xmin = xmin + xmoy
    xmax = xmax + xmoy
    ymin = ymin + ymoy
    ymax = ymax + ymoy
#########################################################


listPts = []
if(len(ptsDEM) == 0):
    listPts = ptsElec*1.0
else:
    listPts = np.concatenate((ptsDEM,ptsElec))
listPts = np.array(listPts)


# New dimension:
print("Dimension :")
print(xmin,xmax,ymin,ymax,zmin,zmax)
# print(Lz)
# print(minZ)
# input()
#########################################################




if(choice == "1" or choice =="2"):
    #########################################################
    ## Construction of the .geo file:

    listPtsNew = []

    listPtsNew.append(np.array([xmin,ymin])) # 1
    listPtsNew.append(np.array([xmin,ymax])) # 2
    listPtsNew.append(np.array([xmax,ymax])) # 3
    listPtsNew.append(np.array([xmax,ymin])) # 4
    for pt in ptsElec:
        listPtsNew.append(np.array([pt[0],pt[1]]))
    
    listPtsNew = np.array(listPtsNew)

    # Parameters h, coeff correspoond to the precision of the mesh around the electrodes, and around the corner of the domain
    if (resRecur == True):
        #mesh.writeMeshGeoNew2(listPtsNew,Lz,"../Data/meshDEMFlat.geo",h,coeff)
        mesh.writeMeshGeoNew2(listPtsNew,Lx,Ly,Lz,"../Data/meshDEMFlat.geo",h,coeff,3.0)
    else:
        mesh.writeMeshGeo(listPtsNew,Lx,Ly,Lz,"../Data/meshDEMFlat.geo",h,coeff)
    #########################################################

        
if(choice == "3"):
    #########################################################        
    print("Reconstruction of geo file:")
    mesh.reWriteGeo(geoFile,"../Data/meshDEMFlat.geo",Lx,Ly,Lz,h,coeff,xmax,xmin,ymax,ymin,3.0)
    #os.system("/Applications/Gmsh.app/Contents/MacOS/gmsh "+str(geoFile)+" -3 -o ../Data/meshDEMFlat.msh")
    #########################################################
        
        
#########################################################        
print("Construction of msh file :")
# To be modified depending on the path de GMSH
os.system("/Applications/Gmsh.app/Contents/MacOS/gmsh ../Data/meshDEMFlat.geo -3 -o ../Data/meshDEMFlat.msh")
#########################################################    

#########################################################
## Reading Mesh file :
[ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D] = mesh.readMsh("../Data/meshDEMFlat.msh")
#########################################################


#########################################################
## Deformation of the flat mesh:
print("STEP 2 : Deformation of the mesh.")



maxZ = zmax
## Deepness of the DEM mesh:


newpoints = []
## Creation du 2D tree to get nearest neighbourg
listCoordXY = []
tree = []
print("Construction of the kd-tree:")
if (choice == "1" or choice == "3"):
    listCoordXY = listPts[:,0:2]
    tree = spatial.KDTree(listCoordXY)

cpt = 0
nbPts = len(ptsM)

# Deformation of the flat mesh:
print("Deformation of the mesh:")
for pp in ptsM:
    cpt = cpt + 1
    # if (cpt % 10 == 0):
    #     print(str(cpt)+" / "+str(nbPts))
    utils.update_progress((cpt*1.0/nbPts))    
        
    x = pp[0]
    y = pp[1]
    z = pp[2]
    znew = 0.0
    if(choice == "1" or choice == "3"):
        znew = topoDEM(x,y,listPts,tree)*z/Lz + minZ*(Lz-z)/Lz
    if(choice == "2"):
        znew = topoF(x,y)*z/Lz + minZ*(Lz-z)/Lz
    newpoints.append([x,y,znew])
#########################################################




#########################################################
## Writing deformed mesh:
mesh.writeMesh3("../Data/meshDEM.msh",newpoints,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D) 
#########################################################

print("End of the program")