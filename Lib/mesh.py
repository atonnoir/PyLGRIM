#!/usr/bin/python3.4
# -*-coding:Utf-8 -*

# by A. Tonnoir

import numpy as np
import copy

import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

import utils


#########################################################
## Read gmsh file (version 4.0) :
def readMsh(nomFichier):
    fid = open(nomFichier,"r");

    contenu = fid.read();
    contenu = contenu.split("\n");

    points = []
    meshes3D = []
    meshes2D = []
    meshes1D = []
    meshes0D = []
    meshesTag3D = []
    meshesTag2D = []
    meshesTag1D = []
    meshesTag0D = []
    
    it = 0
    ele = contenu[it]
    while (ele != "$Nodes"):
        it = it + 1
        ele = contenu[it]
    
    it = it + 1
    ele = contenu[it]
    res = ele.split(" ")
    nBlock = int(res[0])
    nNodes = int(res[1])
    minTag = int(res[2])
    maxTag = int(res[3])
    
    print("Nb nodes : ",nNodes)
    
    cpt = 0
    for b in range(nBlock):
        it = it + 1
        ele = contenu[it]
        res = ele.split(" ")
        [entDim,tagEnt,param,nNodeB] = [int(res[0]),int(res[1]),int(res[2]),int(res[3])]
        
        for i in range(nNodeB):
            it = it + 1
            ele = contenu[it]
        
        for i in range(nNodeB):
            it = it + 1
            ele = contenu[it]
            res = ele.split(" ")
            points.append([float(res[0]), float(res[1]), float(res[2])])

    points = np.array(points)
    #print(len(points))

    while (ele != "$Elements"):
        it = it + 1
        ele = contenu[it]
    
    it = it + 1
    ele = contenu[it]
    res = ele.split(" ")
    
    [nElemB,nElems,minElemTag,maxElemTag] = [int(res[0]),int(res[1]),int(res[2]),int(res[3])]
    
    print("Number of Elements (Tot): ",nElems)
    
    for b in range(nElemB): 
        it = it + 1
        ele = contenu[it]
        res = ele.split(" ")
        [entDim,entTag,entTyp,nbEleEnt] = [int(res[0]),int(res[1]),int(res[2]),int(res[3])]
        typeEle = int(entTyp)
        tag = int(entTag)
        for i in range(nbEleEnt):
            it = it + 1
            ele = contenu[it]
            res = ele.split(" ")
            if (typeEle == 4): # Tetrahedron
                meshes3D.append([int(res[1])-1, int(res[2])-1, int(res[3])-1, int(res[4])-1])    
                meshesTag3D.append(tag)
            if (typeEle == 2): # Triangle
                meshes2D.append([int(res[1])-1, int(res[2])-1, int(res[3])-1])    
                meshesTag2D.append(tag)  
            if (typeEle == 1): # Line
                meshes1D.append([int(res[1])-1, int(res[2])-1])    
                meshesTag1D.append(tag)      
            if (typeEle == 15): # Point
                meshes0D.append(int(res[1])-1)    
                meshesTag0D.append(tag)      

        
    meshes3D = np.array(meshes3D)
    meshes2D = np.array(meshes2D)
    meshes1D = np.array(meshes1D)
    meshes0D = np.array(meshes0D)
    
    fid.close()
    return [points,meshes3D,meshes2D,meshes1D,meshes0D,meshesTag3D,meshesTag2D,meshesTag1D,meshesTag0D]
#########################################################

#########################################################
## Read gmsh file (version 3.0) :
def readMsh3(nomFichier):
    fid = open(nomFichier,"r");

    contenu = fid.read();
    contenu = contenu.split("\n");

    points = []
    meshes3D = []
    meshes2D = []
    meshes1D = []
    meshes0D = []
    meshesTag3D = []
    meshesTag2D = []
    meshesTag1D = []
    meshesTag0D = []
    
    it = 0
    ele = contenu[it]
    while (ele != "$Nodes"):
        it = it + 1
        ele = contenu[it]

    it = it + 1    
    nbNodes = int(contenu[it])
    print("Nb nodes : ",nbNodes)
    for i in range(nbNodes):
        it = it + 1
        ele = contenu[it]
        res = ele.split(" ")
        points.append([float(res[1]), float(res[2]), float(res[3])])

    points = np.array(points)
    print(len(points))

    while (ele != "$Elements"):
        it = it + 1
        ele = contenu[it]
    
    it = it + 1    
    nbElements = int(contenu[it])
    print("Nb elements : ",nbElements)
    for i in range(nbElements):
        it = it + 1
        ele = contenu[it]
        res = ele.split(" ")
        if (float(res[1]) == 4): # Tetrahedron
            meshes3D.append([int(res[5])-1, int(res[6])-1, int(res[7])-1, int(res[8])-1])    
            meshesTag3D.append(res[3])
        if (float(res[1]) == 2): # Triangle
            meshes2D.append([int(res[5])-1, int(res[6])-1, int(res[7])-1])    
            meshesTag2D.append(res[3])
        if (float(res[1]) == 1): # Line
            meshes1D.append([int(res[5])-1, int(res[6])-1])    
            meshesTag1D.append(res[3])
        if (float(res[1]) == 15): # Point
            meshes0D.append(int(res[5])-1)    
            meshesTag0D.append(res[3])    
            
    meshes3D = np.array(meshes3D)
    meshes2D = np.array(meshes2D)
    meshes1D = np.array(meshes1D)
    meshes0D = np.array(meshes0D)
    
    
    xmin = min(points[:,0])
    xmax = max(points[:,0])
    ymin = min(points[:,1])
    ymax = max(points[:,1])
    zmin = min(points[:,2])
    zmax = max(points[:,2])
    
    # Detection of boundary positions :
    # Setting 0D tag (corresponds to corner or injection position)
    for j,m in enumerate(meshes0D):
        pt = points[m]
        meshesTag0D[j] = "0"
        if(pt[0] == xmin and pt[1] == ymin and pt[2] == zmin):
            meshesTag0D[j] = "3"
        if(pt[0] == xmax and pt[1] == ymin and pt[2] == zmin):
            meshesTag0D[j] = "3"
        if(pt[0] == xmin and pt[1] == ymax and pt[2] == zmin):
            meshesTag0D[j] = "3"
        if(pt[0] == xmax and pt[1] == ymax and pt[2] == zmin):
            meshesTag0D[j] = "3"
        if(pt[0] != xmin and pt[0] != xmax and pt[1] != ymin and pt[1] != ymax):
            meshesTag0D[j] = "4"     
        
             
    # Line boundaries
    for j,m in enumerate(meshes1D):
        pt1 = points[m[0]]
        pt2 = points[m[1]]
        meshesTag1D[j] = "0"
        if(pt1[0] == pt2[0] == xmin and pt1[1] == pt2[1] == ymin):
            meshesTag1D[j] = "2"
        if(pt1[0] == pt2[0] == xmax and pt1[1] == pt2[1] == ymin):
            meshesTag1D[j] = "2"
        if(pt1[0] == pt2[0] == xmin and pt1[1] == pt2[1] == ymax):
            meshesTag1D[j] = "2"    
        if(pt1[0] == pt2[0] == xmax and pt1[1] == pt2[1] == ymax):
            meshesTag1D[j] = "2"
        
        if(pt1[0] == pt2[0] == xmin and pt1[2] == pt2[2] == zmin):
            meshesTag1D[j] = "2"
        if(pt1[0] == pt2[0] == xmax and pt1[2] == pt2[2] == zmin):
            meshesTag1D[j] = "2"

        if(pt1[1] == pt2[1] == ymin and pt1[2] == pt2[2] == zmin):
            meshesTag1D[j] = "2"
        if(pt1[1] == pt2[1] == ymax and pt1[2] == pt2[2] == zmin):
            meshesTag1D[j] = "2"
        
    
    # Faces boundary
    for j,m in enumerate(meshes2D):
        pt1 = points[m[0]]
        pt2 = points[m[1]]
        pt3 = points[m[2]]
        meshesTag2D[j] = "0"
        if(pt1[0] == pt2[0] == pt3[0] == xmin or pt1[0] == pt2[0] == pt3[0] == xmax):
            meshesTag2D[j] = "1"
        if(pt1[1] == pt2[1] == pt3[1] == ymin or pt1[1] == pt2[1] == pt3[1] == ymax):
            meshesTag2D[j] = "1"
        if(pt1[2] == pt2[2] == pt3[2] == zmin):
            meshesTag2D[j] = "1"
    
    
    
    fid.close()
    return [points,meshes3D,meshes2D,meshes1D,meshes0D,meshesTag3D,meshesTag2D,meshesTag1D,meshesTag0D]
#########################################################


#########################################################
## Write vtk file :
def writeVTK(nomFichier,Sol,Solex,err,pts,m3D):
    fid = open(nomFichier,"w")

    # Ecriture de la solution :
    fid.write("# vtk DataFile Version 3.6\n")
    fid.write("PolyDATA\n")
    fid.write("ASCII\n")
    fid.write("DATASET UNSTRUCTURED_GRID\n")

    # Ecriture points du maillage :
    fid.write(str("POINTS   "+str(len(pts))+"    float\n"))

    for p in pts:
        fid.write(str(str(p[0])+" "+str(p[1])+"    "+str(p[2])+"\n"))

    # Ecriture mailles :
    fid.write(str("CELLS    "+str(len(m3D))+"    "+str(len(m3D)*5)+"\n"))

    for m in m3D:
        fid.write("4    "+str(m[0])+"    "+str(m[1])+"    "+str(m[2])+"    "+str(m[3])+"\n")

    fid.write("CELL_TYPES    "+str(len(m3D))+"\n")
    for m in m3D:
        fid.write("10\n")

    # Ecriture valeur solution aux noeuds du maillage :
    fid.write("POINT_DATA   "+str(len(pts))+"\n")
    fid.write("SCALARS scalars float    3\n");
    fid.write("LOOKUP_TABLE default\n");

    for ite,e in enumerate(Sol):
        fid.write(str(e)+"  "+str(Solex[ite])+" "+str(err[ite])+"\n")
#########################################################


#########################################################
##Write vtk cell :
def writeCellVTK(nomFichier,conduc,pts,m3D):
    fid = open(nomFichier,"w")

    # Ecriture de la solution :
    fid.write("# vtk DataFile Version 3.6\n")
    fid.write("PolyDATA\n")
    fid.write("ASCII\n")
    fid.write("DATASET UNSTRUCTURED_GRID\n")

    # Ecriture points du maillage :
    fid.write(str("POINTS   "+str(len(pts))+"    float\n"))

    for p in pts:
        fid.write(str(str(p[0])+" "+str(p[1])+"    "+str(p[2])+"\n"))

    # Ecriture mailles :
    fid.write(str("CELLS    "+str(len(m3D))+"    "+str(len(m3D)*5)+"\n"))

    for m in m3D:
        fid.write("4    "+str(m[0])+"    "+str(m[1])+"    "+str(m[2])+"    "+str(m[3])+"\n")

    fid.write("CELL_TYPES    "+str(len(m3D))+"\n")
    for m in m3D:
        fid.write("10\n")
        
    # Ecriture valeur solution aux noeuds du maillage :
    fid.write("CELL_DATA   "+str(len(conduc))+"\n")
    fid.write("SCALARS cell_scalars float    1\n");
    fid.write("LOOKUP_TABLE default\n");

    for ite,e in enumerate(conduc):
        fid.write(str(e)+"\n")    


#########################################################
## Ecriture solution fichier vtk (version 3.0) :
def writeMesh3(nomFichier,ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D):
    fid = open(nomFichier,"w")

    # Ecriture du maillage :
    fid.write("$MeshFormat\n")
    fid.write("2.2 0 8\n")
    fid.write("$EndMeshFormat\n")
    
    # Ecriture points
    fid.write("$Nodes\n")
    fid.write(str(len(ptsM))+"\n")

    for i,p in enumerate(ptsM):
        fid.write(str(str(i+1)+" "+str(p[0])+" "+str(p[1])+" "+str(p[2])+"\n"))

    fid.write("$EndNodes\n")
    
        
    # Ecriture mailles :
    fid.write("$Elements\n")
    fid.write(str(len(m3D)+len(m2D)+len(m1D)+len(m0D))+"\n")
    cpt = 1
    for i,m in enumerate(m0D):
        fid.write(str(cpt)+" 15 2 "+str(mt0D[i])+" "+str(mt0D[i])+" "+str(m+1)+"\n")
        cpt = cpt+1
        
    for i,m in enumerate(m1D):
        fid.write(str(cpt)+" 1 2 "+str(mt1D[i])+" "+str(mt1D[i])+" "+str(m[0]+1)+" "+str(m[1]+1)+"\n")
        cpt = cpt+1
        
    for i,m in enumerate(m2D):
        fid.write(str(cpt)+" 2 2 "+str(mt2D[i])+" "+str(mt2D[i])+" "+str(m[0]+1)+" "+str(m[1]+1)+" "+str(m[2]+1)+"\n")
        cpt = cpt+1
   
    for i,m in enumerate(m3D):
        fid.write(str(cpt)+" 4 2 "+str(mt3D[i])+" "+str(mt3D[i])+" "+str(m[0]+1)+" "+str(m[1]+1)+" "+str(m[2]+1)+" "+str(m[3]+1)+"\n")
        cpt = cpt+1
   
    fid.write("$EndElements\n")
#########################################################



#########################################################
## Writing GMSH geo file:
def writeMeshGeo(points,Lx,Ly,Lz,nomFichier,h,coeff):
    fid = open(nomFichier,"w")

    fid.write("h="+str(h)+";\n")
    fid.write("coeff="+str(coeff)+";\n")
    fid.write("Lx="+str(Lx)+";\n")
    fid.write("Ly="+str(Ly)+";\n")
    fid.write("Lz="+str(Lz)+";\n")
    
    fid.write("// Points :\n")
    for j,pt in enumerate(points):
        if(j<=3):
            #fid.write("Point("+str(j)+") = {"+str(pt[0])+","+str(pt[1])+","+str(Lz)+","+str(h*coeff)+"};\n")
            if(j==0):
                fid.write("Point("+str(j)+") = {"+str(pt[0]+Lx)+"-Lx,"+str(pt[1]+Ly)+"-Ly,Lz,h*coeff};\n")
            if(j==1):
                fid.write("Point("+str(j)+") = {"+str(pt[0]+Lx)+"-Lx,"+str(pt[1]-Ly)+"+Ly,Lz,h*coeff};\n")   
            if(j==2):
                fid.write("Point("+str(j)+") = {"+str(pt[0]-Lx)+"+Lx,"+str(pt[1]-Ly)+"+Ly,Lz,h*coeff};\n")    
            if(j==3):
                fid.write("Point("+str(j)+") = {"+str(pt[0]-Lx)+"+Lx,"+str(pt[1]+Ly)+"-Ly,Lz,h*coeff};\n")     
        else:
            #fid.write("Point("+str(j)+") = {"+str(pt[0])+","+str(pt[1])+","+str(Lz)+","+str(h)+"};\n")
            fid.write("Point("+str(j)+") = {"+str(pt[0])+","+str(pt[1])+",Lz,h};\n")
    ydiff = points[1][1] - points[0][1]
    xdiff = points[2][0] - points[2][0]
    coeffBis = int(Lz / max(xdiff,ydiff))
    if(coeffBis == 0):
        coeffBis = 1   
    for i in range(4):
        pt = points[i]
        #fid.write("Point("+str(len(points)+i)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+","+str(h*coeff*2.0)+"};\n")
        #fid.write("Point("+str(len(points)+i)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+",h*coeff*2.0};\n") 
        if(i==0):
            fid.write("Point("+str(len(points)+i)+") = {"+str(pt[0]+Lx)+"-Lx,"+str(pt[1]+Ly)+"-Ly,"+str(0)+",h*coeff*2.0};\n") 
        if(i==1):
            fid.write("Point("+str(len(points)+i)+") = {"+str(pt[0]+Lx)+"-Lx,"+str(pt[1]-Ly)+"+Ly,"+str(0)+",h*coeff*2.0};\n") 
        if(i==2):
            fid.write("Point("+str(len(points)+i)+") = {"+str(pt[0]-Lx)+"+Lx,"+str(pt[1]-Ly)+"+Ly,"+str(0)+",h*coeff*2.0};\n")  
        if(i==3):
            fid.write("Point("+str(len(points)+i)+") = {"+str(pt[0]-Lx)+"+Lx,"+str(pt[1]-Ly)+"+Ly,"+str(0)+",h*coeff*2.0};\n")  
    
    fid.write("// Lines +surface (top surface):\n")
    fid.write("Line("+str(0)+") = {"+str(0)+","+str(1)+"};\n")
    fid.write("Line("+str(1)+") = {"+str(1)+","+str(2)+"};\n")
    fid.write("Line("+str(2)+") = {"+str(2)+","+str(3)+"};\n")
    fid.write("Line("+str(3)+") = {"+str(3)+","+str(0)+"};\n")
    
    fid.write("Line("+str(4)+") = {"+str(0)+","+str(4)+"};\n")
    
    nblig = len(points)-4
    cpt = 5
    for i in range(nblig-1):
        fid.write("Line("+str(cpt)+") = {"+str(i+4)+","+str(i+5)+"};\n")
        cpt += 1    
        
    fid.write("Line("+str(cpt)+") = {"+str(nblig+3)+","+str(2)+"};\n")
    fid.write("Line Loop("+str(1)+") = {0,1,"+str(-4)+":"+str(-cpt)+"};\n")
    fid.write("Plane Surface("+str(1)+") = {"+str(1)+"};\n")
    
    fid.write("Line Loop("+str(2)+") = {"+str(4)+":"+str(cpt)+","+str(2)+","+str(3)+"};\n")
    fid.write("Plane Surface("+str(2)+") = {"+str(2)+"};\n")
    
    
    fid.write("// Lines + surface (bottom surface):\n")
    cpt += 1
    fid.write("Line("+str(cpt)+") = {"+str(len(points)+0)+","+str(len(points)+1)+"};\n")
    fid.write("Line("+str(cpt+1)+") = {"+str(len(points)+1)+","+str(len(points)+2)+"};\n")
    fid.write("Line("+str(cpt+2)+") = {"+str(len(points)+2)+","+str(len(points)+3)+"};\n")
    fid.write("Line("+str(cpt+3)+") = {"+str(len(points)+3)+","+str(len(points)+0)+"};\n")
    fid.write("Line Loop("+str(3)+") = {"+str(cpt)+":"+str(cpt+3)+"};\n")
    fid.write("Plane Surface("+str(3)+") = {"+str(3)+"};\n")
    
    
    fid.write("// Lines + surface (laterals):\n")
    
    fid.write("Line("+str(cpt+4)+") = {"+str(0)+","+str(len(points)+0)+"};\n")
    fid.write("Line("+str(cpt+5)+") = {"+str(1)+","+str(len(points)+1)+"};\n")
    fid.write("Line("+str(cpt+6)+") = {"+str(2)+","+str(len(points)+2)+"};\n")
    fid.write("Line("+str(cpt+7)+") = {"+str(3)+","+str(len(points)+3)+"};\n")
    
    #fid.write("Line Loop("+str(4)+") = {"+str(cpt+4)+","+str(cpt)+","+str(cpt+1)+","+str(-(cpt+5))+","+str(0)+"};\n")
    fid.write("Line Loop("+str(4)+") = {"+str(0)+","+str(cpt+5)+","+str(-cpt)+","+str(-(cpt+4))+"};\n")
    fid.write("Plane Surface("+str(4)+") = {"+str(4)+"};\n")
    
    fid.write("Line Loop("+str(5)+") = {"+str(1)+","+str(cpt+6)+","+str(-(cpt+1))+","+str(-(cpt+5))+"};\n")
    fid.write("Plane Surface("+str(5)+") = {"+str(5)+"};\n")
    
    fid.write("Line Loop("+str(6)+") = {"+str(2)+","+str(cpt+7)+","+str(-(cpt+2))+","+str(-(cpt+6))+"};\n")
    fid.write("Plane Surface("+str(6)+") = {"+str(6)+"};\n")
    
    fid.write("Line Loop("+str(7)+") = {"+str(3)+","+str(cpt+4)+","+str(-(cpt+3))+","+str(-(cpt+7))+"};\n")
    fid.write("Plane Surface("+str(7)+") = {"+str(7)+"};\n")
    
    
    fid.write("Surface Loop(1)={1,2,3,4,5,6,7};\n")
    fid.write("Volume(1)={1};\n")
    
    fid.write("Physical Surface(\"ArtBound\") = {3,4,5,6,7};\n")
    fid.write("Physical Line(\"ArtBound\") = {"+str(cpt)+":"+str(cpt+7)+"};\n")
    fid.write("Physical Point(\"ArtBound\") = {"+str(len(points))+":"+str(len(points)+3)+"};\n")
    
    fid.write("Physical Point(\"Injection\") = {4:"+str(len(points))+"};\n")
    
    fid.write("Physical Surface(\"Groud\") = {1,2} ;\n")
    
    fid.write("Physical Volume(\"Volume\") = {1};\n")
    # fid.write("Line Loop("+str(3)+") = {"+str(cpt)+":"+str(cpt+3)+"};\n")
#     fid.write("Plane Surface("+str(3)+") = {"+str(3)+"};\n")
    
#########################################################


#########################################################
## Writing GMSH geo file:
def writeMeshGeoNew(points,Lz,nomFichier,h,coeff):
    fid = open(nomFichier,"w")

    
    fid.write("// Points :\n")
    for j,pt in enumerate(points):
        if(j<=3):
            fid.write("Point("+str(j)+") = {"+str(pt[0])+","+str(pt[1])+","+str(Lz)+","+str(h*coeff*2)+"};\n")
        else:
            fid.write("Point("+str(j)+") = {"+str(pt[0])+","+str(pt[1])+","+str(Lz)+","+str(h)+"};\n")
    ydiff = points[1][1] - points[0][1]
    xdiff = points[2][0] - points[2][0]
    coeffBis = int(Lz / max(xdiff,ydiff))
    
    
    for i in range(4):
        pt = points[i]
        fid.write("Point("+str(len(points)+i)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+","+str(h*coeff*6)+"};\n")
        
    pt = points[4]
    fid.write("Point("+str(len(points)+4)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+","+str(h*coeff*4)+"};\n")
    pt = points[-1]
    fid.write("Point("+str(len(points)+5)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+","+str(h*coeff*4)+"};\n")
    
    
    fid.write("// Lines +surface (top surface):\n")
    fid.write("Line("+str(0)+") = {"+str(0)+","+str(1)+"};\n")
    fid.write("Line("+str(1)+") = {"+str(1)+","+str(2)+"};\n")
    fid.write("Line("+str(2)+") = {"+str(2)+","+str(3)+"};\n")
    fid.write("Line("+str(3)+") = {"+str(3)+","+str(0)+"};\n")
    
    fid.write("Line("+str(4)+") = {"+str(0)+","+str(4)+"};\n")
    
    nblig = len(points)-4
    cpt = 5
    for i in range(nblig-1):
        fid.write("Line("+str(cpt)+") = {"+str(i+4)+","+str(i+5)+"};\n")
        cpt += 1    
        
    fid.write("Line("+str(cpt)+") = {"+str(nblig+3)+","+str(2)+"};\n")
    fid.write("Curve Loop("+str(1)+") = {0,1,"+str(-4)+":"+str(-cpt)+"};\n")
    fid.write("Surface("+str(1)+") = {"+str(1)+"};\n")
    
    fid.write("Curve Loop("+str(2)+") = {"+str(4)+":"+str(cpt)+","+str(2)+","+str(3)+"};\n")
    fid.write("Surface("+str(2)+") = {"+str(2)+"};\n")
    
    
    fid.write("// Lines + surface (bottom surface):\n")
    cpt += 1
    
    debcptBot = cpt
    
    fid.write("Line("+str(cpt)+") = {"+str(len(points)+0)+","+str(len(points)+1)+"};\n")
    fid.write("Line("+str(cpt+1)+") = {"+str(len(points)+1)+","+str(len(points)+2)+"};\n")
    fid.write("Line("+str(cpt+2)+") = {"+str(len(points)+2)+","+str(len(points)+3)+"};\n")
    fid.write("Line("+str(cpt+3)+") = {"+str(len(points)+3)+","+str(len(points)+0)+"};\n")
    
    fid.write("Line("+str(cpt+4)+") = {"+str(len(points)+0)+","+str(len(points)+4)+"};\n")
    fid.write("Line("+str(cpt+5)+") = {"+str(len(points)+4)+","+str(len(points)+5)+"};\n")
    fid.write("Line("+str(cpt+6)+") = {"+str(len(points)+5)+","+str(len(points)+2)+"};\n")
    

 
    fid.write("Curve Loop("+str(3)+") = {"+str(debcptBot)+","+str(debcptBot+1)+","+str(-(cpt+4))+":"+str(-(cpt+6))+"};\n")
    fid.write("Surface("+str(3)+") = {"+str(3)+"};\n")
    
    fid.write("Curve Loop("+str(4)+") = {"+str(4+debcptBot)+":"+str(6+debcptBot)+","+str(2+debcptBot)+","+str(3+debcptBot)+"};\n")
    fid.write("Surface("+str(4)+") = {"+str(4)+"};\n")
    
    
    fid.write("// Lines + surface (laterals):\n")
    cpt = cpt+3
    
    fid.write("Line("+str(cpt+4)+") = {"+str(0)+","+str(len(points)+0)+"};\n")
    fid.write("Line("+str(cpt+5)+") = {"+str(1)+","+str(len(points)+1)+"};\n")
    fid.write("Line("+str(cpt+6)+") = {"+str(2)+","+str(len(points)+2)+"};\n")
    fid.write("Line("+str(cpt+7)+") = {"+str(3)+","+str(len(points)+3)+"};\n")
    
    fid.write("Line("+str(cpt+8)+") = {"+str(len(points)+4)+","+str(4)+"};\n")
    fid.write("Line("+str(cpt+9)+") = {"+str(len(points)+5)+","+str(len(points)-1)+"};\n")
    
    
    #fid.write("Line Loop("+str(4)+") = {"+str(cpt+4)+","+str(cpt)+","+str(cpt+1)+","+str(-(cpt+5))+","+str(0)+"};\n")
    fid.write("Curve Loop("+str(5)+") = {"+str(0)+","+str(cpt+5)+","+str(-debcptBot)+","+str(-(cpt+4))+"};\n")
    fid.write("Surface("+str(5)+") = {"+str(5)+"};\n")
    
    fid.write("Curve Loop("+str(6)+") = {"+str(1)+","+str(cpt+6)+","+str(-(debcptBot+1))+","+str(-(cpt+5))+"};\n")
    fid.write("Surface("+str(6)+") = {"+str(6)+"};\n")
    
    fid.write("Curve Loop("+str(7)+") = {"+str(2)+","+str(cpt+7)+","+str(-(debcptBot+2))+","+str(-(cpt+6))+"};\n")
    fid.write("Surface("+str(7)+") = {"+str(7)+"};\n")
    
    fid.write("Curve Loop("+str(8)+") = {"+str(3)+","+str(cpt+4)+","+str(-(debcptBot+3))+","+str(-(cpt+7))+"};\n")
    fid.write("Surface("+str(8)+") = {"+str(8)+"};\n")
    
    fid.write("Curve Loop("+str(9)+") = {"+str(debcptBot+4)+","+str(cpt+8)+","+str(-(4))+","+str(cpt+4)+"};\n")
    fid.write("Surface("+str(9)+") = {"+str(9)+"};\n")
    
    fid.write("Curve Loop("+str(10)+") = {"+str(debcptBot+5)+","+str(cpt+9)+","+str(-(nblig+3))+":"+str(-5)+","+str(-(cpt+8))+"};\n")
    fid.write("Surface("+str(10)+") = {"+str(10)+"};\n")
    
    fid.write("Curve Loop("+str(11)+") = {"+str((debcptBot+6))+","+str(-(cpt+6))+","+str(-(nblig+4))+","+str(-(cpt+9))+"};\n")
    fid.write("Surface("+str(11)+") = {"+str(11)+"};\n")
    
    
    
    
    fid.write("Surface Loop(1)={1,9,10,11,3,5,6};\n")
    fid.write("Surface Loop(2)={2,9,10,11,4,7,8};\n")
    fid.write("Volume(1)={1};\n")
    fid.write("Volume(2)={2};\n")
    
    fid.write("Physical Surface(\"ArtBound\") = {3,4,5,6,7};\n")
    fid.write("Physical Line(\"ArtBound\") = {"+str(cpt)+":"+str(cpt+7)+"};\n")
    fid.write("Physical Point(\"ArtBound\") = {"+str(len(points))+":"+str(len(points)+3)+"};\n")
    
    fid.write("Physical Point(\"Injection\") = {4:"+str(len(points))+"};\n")
    
    fid.write("Physical Surface(\"Ground\") = {1,2} ;\n")
    
    fid.write("Physical Volume(\"Volume\") = {1,2};\n")
    # fid.write("Line Loop("+str(3)+") = {"+str(cpt)+":"+str(cpt+3)+"};\n")
#     fid.write("Plane Surface("+str(3)+") = {"+str(3)+"};\n")
    
#########################################################


#########################################################
## Writing GMSH geo file:
def writeMeshGeoNew2(points,Lx,Ly,Lz,nomFichier,h,coeff,coeffBot=-1):
    fid = open(nomFichier,"w")
    
    ydiff = points[1][1] - points[0][1]
    xdiff = points[2][0] - points[2][0]
    coeffBis = int(Lz / max(xdiff,ydiff))
    if(coeffBis <= 1):
        coeffBis = 1
    if(coeffBis >=4):
        coeffBis = 4
        
    if(coeffBot != -1):
        print(coeffBot)
        coeffBis = coeffBot    

    fid.write("h="+str(h)+";\n")
    fid.write("coeff="+str(coeff)+";\n")
    fid.write("coeffBis="+str(coeffBis)+";\n")
    fid.write("Lx="+str(Lx)+";\n")
    fid.write("Ly="+str(Ly)+";\n")
    fid.write("Lz="+str(Lz)+";\n")
    
    fid.write("// Points :\n")
    for j,pt in enumerate(points):
        if(j<=3):
            #fid.write("Point("+str(j+1)+") = {"+str(pt[0])+","+str(pt[1])+","+str(Lz)+","+str(h*coeff)+"};\n")
            #fid.write("Point("+str(j+1)+") = {"+str(pt[0])+","+str(pt[1])+",Lz,h*coeff};\n")
            if(j==0):
                fid.write("Point("+str(j+1)+") = {"+str(pt[0]+Lx)+"-Lx,"+str(pt[1]+Ly)+"-Ly,Lz,h*coeff};\n")
            if(j==1):
                fid.write("Point("+str(j+1)+") = {"+str(pt[0]+Lx)+"-Lx,"+str(pt[1]-Ly)+"+Ly,Lz,h*coeff};\n")   
            if(j==2):
                fid.write("Point("+str(j+1)+") = {"+str(pt[0]-Lx)+"+Lx,"+str(pt[1]-Ly)+"+Ly,Lz,h*coeff};\n")    
            if(j==3):
                fid.write("Point("+str(j+1)+") = {"+str(pt[0]-Lx)+"+Lx,"+str(pt[1]+Ly)+"-Ly,Lz,h*coeff};\n")
        else:
            #fid.write("Point("+str(j+1)+") = {"+str(pt[0])+","+str(pt[1])+","+str(Lz)+","+str(h)+"};\n")
            fid.write("Point("+str(j+1)+") = {"+str(pt[0])+","+str(pt[1])+",Lz,h};\n")
       
    
    distElec = np.sqrt(np.dot(points[5]-points[4],points[5]-points[4]))
    # Shift index for points on bottom
    sBot = len(points)
    for j,pt in enumerate(points):
        if(j<=3):
            #fid.write("Point("+str(j+1+sBot)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+","+str(h*coeff*coeffBis)+"};\n")
            #fid.write("Point("+str(j+1+sBot)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+",h*coeff*coeffBis};\n")
            if(j==0):
                fid.write("Point("+str(j+1+sBot)+") = {"+str(pt[0]+Lx)+"-Lx,"+str(pt[1]+Ly)+"-Ly,0,h*coeff*coeffBis};\n")
            if(j==1):
                fid.write("Point("+str(j+1+sBot)+") = {"+str(pt[0]+Lx)+"-Lx,"+str(pt[1]-Ly)+"+Ly,0,h*coeff*coeffBis};\n")   
            if(j==2):
                fid.write("Point("+str(j+1+sBot)+") = {"+str(pt[0]-Lx)+"+Lx,"+str(pt[1]-Ly)+"+Ly,0,h*coeff*coeffBis};\n")    
            if(j==3):
                fid.write("Point("+str(j+1+sBot)+") = {"+str(pt[0]-Lx)+"+Lx,"+str(pt[1]+Ly)+"-Ly,0,h*coeff*coeffBis};\n")
        else:
            #fid.write("Point("+str(j+1+sBot)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+","+str(h*coeff*coeffBis)+"};\n")
            fid.write("Point("+str(j+1+sBot)+") = {"+str(pt[0])+","+str(pt[1])+","+str(0)+",h*coeff*coeffBis};\n")
            
    
    # Top Surface
    fid.write("// Lines + surface (Top surface):\n")
    # Top square
    fid.write("Line("+str(1)+") = {"+str(1)+","+str(2)+"};\n")
    fid.write("Line("+str(2)+") = {"+str(2)+","+str(3)+"};\n")
    fid.write("Line("+str(3)+") = {"+str(3)+","+str(4)+"};\n")
    fid.write("Line("+str(4)+") = {"+str(4)+","+str(1)+"};\n")
    
    # Electrode lines top :
    # Starting line
    fid.write("Line("+str(5)+") = {"+str(2)+","+str(5)+"};\n") # AICI !!
    nbElec = len(points)-4
    nbLigE = nbElec - 1
    # lines between electrodes
    for i in range(nbLigE):
        fid.write("Line("+str(i+6)+") = {"+str(i+5)+","+str(i+6)+"};\n")
    # last line    
    fid.write("Line("+str(nbLigE+6)+") = {"+str(nbLigE+5)+","+str(4)+"};\n") # AICI !!
    # Electrode Loop
    fid.write("Line Loop("+str(1)+") = {1,5:"+str(6+nbLigE)+",4};\n")
    fid.write("Plane Surface("+str(1)+") = {"+str(1)+"};\n")
    
    fid.write("Line Loop("+str(2)+") = {5:"+str(6+nbLigE)+",-3,-2};\n")
    fid.write("Plane Surface("+str(2)+") = {"+str(2)+"};\n")
    
    
    
    # Bottom Surface
    # Shift index for bottom lines
    sBotL = 6 + nbLigE
    fid.write("// Lines + surface (Bottom surface):\n")
    # Top square
    fid.write("Line("+str(sBotL+1)+") = {"+str(1+sBot)+","+str(2+sBot)+"};\n")
    fid.write("Line("+str(sBotL+2)+") = {"+str(2+sBot)+","+str(3+sBot)+"};\n")
    fid.write("Line("+str(sBotL+3)+") = {"+str(3+sBot)+","+str(4+sBot)+"};\n")
    fid.write("Line("+str(sBotL+4)+") = {"+str(4+sBot)+","+str(1+sBot)+"};\n")
    
    # Electrode lines top :
    # Starting line
    fid.write("Line("+str(sBotL+5)+") = {"+str(2+sBot)+","+str(5+sBot)+"};\n") # AICI !!
    # lines between electrodes
    for i in range(nbLigE):
        fid.write("Line("+str(sBotL+i+6)+") = {"+str(i+5+sBot)+","+str(i+6+sBot)+"};\n")
    # last line    
    fid.write("Line("+str(nbLigE+6+sBotL)+") = {"+str(nbLigE+5+sBot)+","+str(4+sBot)+"};\n") # AICI !!
    # Electrode Loop
    fid.write("Line Loop("+str(3)+") = {"+str(1+sBotL)+","+str(5+sBotL)+":"+str(6+nbLigE+sBotL)+","+str(sBotL+4)+"};\n")
    fid.write("Plane Surface("+str(3)+") = {"+str(3)+"};\n")
    
    fid.write("Line Loop("+str(4)+") = {"+str(5+sBotL)+":"+str(6+nbLigE+sBotL)+","+str(-3-sBotL)+","+str(-2-sBotL)+"};\n")
    fid.write("Plane Surface("+str(4)+") = {"+str(4)+"};\n")
    
    
    
    
    
    # Vertical lines & surfaces
    # Shift index vertical lines :
    sVertL = sBotL+6 + nbLigE
    
    # Vertical lines:
    for i in range(5+nbLigE):
        fid.write("Line("+str(sVertL+i+1)+") = {"+str(i+1)+","+str(i+1+sBot)+"};\n")
    
    # Vertical surfraces    
    # Exterior faces
    fid.write("Line Loop("+str(5)+") = {"+str(1)+","+str(sVertL+2)+","+str(-1-sBotL)+","+str(-1-sVertL)+"};\n")
    fid.write("Plane Surface("+str(5)+") = {"+str(5)+"};\n")
    fid.write("Line Loop("+str(6)+") = {"+str(2)+","+str(sVertL+3)+","+str(-2-sBotL)+","+str(-2-sVertL)+"};\n")
    fid.write("Plane Surface("+str(6)+") = {"+str(6)+"};\n")
    fid.write("Line Loop("+str(7)+") = {"+str(3)+","+str(sVertL+4)+","+str(-3-sBotL)+","+str(-3-sVertL)+"};\n")
    fid.write("Plane Surface("+str(7)+") = {"+str(7)+"};\n")
    fid.write("Line Loop("+str(8)+") = {"+str(4)+","+str(sVertL+1)+","+str(-4-sBotL)+","+str(-4-sVertL)+"};\n")
    fid.write("Plane Surface("+str(8)+") = {"+str(8)+"};\n")   
    
    # Line electrode faces :
    fid.write("Line Loop("+str(9)+") = {"+str(5)+","+str(sVertL+5)+","+str(-5-sBotL)+","+str(-2-sVertL)+"};\n")
    fid.write("Plane Surface("+str(9)+") = {"+str(9)+"};\n") 
        
    for i in range(nbLigE):
        fid.write("Line Loop("+str(10+i)+") = {"+str(6+i)+","+str(sVertL+6+i)+","+str(-6-i-sBotL)+","+str(-5-i-sVertL)+"};\n")
        fid.write("Plane Surface("+str(10+i)+") = {"+str(10+i)+"};\n")
        
    fid.write("Line Loop("+str(10+nbLigE)+") = {"+str(6+nbLigE)+","+str(sVertL+4)+","+str(-6-nbLigE-sBotL)+","+str(-5-nbLigE-sVertL)+"};\n")
    fid.write("Plane Surface("+str(10+nbLigE)+") = {"+str(10+nbLigE)+"};\n")    
    
    
    # Surface Lopp
    fid.write("Surface Loop(1)={9:"+str(10+nbLigE)+",8,5,1,3};\n")
    fid.write("Surface Loop(2)={9:"+str(10+nbLigE)+",7,6,2,4};\n")
    fid.write("Volume(1)={1};\n")
    fid.write("Volume(2)={2};\n")
    
    fid.write("Physical Surface(\"ArtBound\") = {3,4,5,6,7,8};\n")
    fid.write("Physical Line(\"ArtBound\") = {"+str(sBotL+1)+":"+str(sBotL+4)+","+str(sVertL+1)+":"+str(sVertL+4)+"};\n")
    fid.write("Physical Point(\"ArtBound\") = {"+str(len(points)+1)+":"+str(len(points)+4)+"};\n")
    
    fid.write("Physical Point(\"Injection\") = {5:"+str(len(points)+1)+"};\n")
    
    fid.write("Physical Surface(\"Ground\") = {1,2} ;\n")
    
    fid.write("Physical Volume(\"Volume\") = {1,2};\n")
    # fid.write("Line Loop("+str(3)+") = {"+str(cpt)+":"+str(cpt+3)+"};\n")
#     fid.write("Plane Surface("+str(3)+") = {"+str(3)+"};\n")
    
#########################################################


#########################################################
def getLinkm3Dm2Dm1Dm0D(m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D):
    # For boundary BC
    linkSig2Dto3D = []
    
    m3Dsort = np.sort(m3D,axis=1)
    
    for ite,m in enumerate(m2D):
        linkSig2Dto3D.append(-1)
        if(mt2D[ite] == "1"): 
            
            # Je n'aime pas cette facon car on fait plein de calculs inutiles... mais parce que la manipulation des tableaux est rapide en python, c'est plus efficace qu'une boucle....
            [n1,n2,n3] = np.sort(m)
            mm = np.array([n1,n2,n3,-1])
            m3Dtmp = (m3Dsort - mm)[:,[0,1,2]]
            m3Dtmp = np.sum(m3Dtmp,1)
            list_pos = np.where(m3Dtmp==0)[0]
            if(len(list_pos) > 0):
                linkSig2Dto3D[ite] = list_pos[0]
                
            mm = np.array([n1,n2,-1,n3])
            m3Dtmp = (m3Dsort - mm)[:,[0,1,3]]
            m3Dtmp = np.sum(m3Dtmp,1)
            list_pos = np.where(m3Dtmp==0)[0]
            if(len(list_pos) > 0):
                linkSig2Dto3D[ite] = list_pos[0]
                
            mm = np.array([n1,-1,n2,n3])
            m3Dtmp = (m3Dsort - mm)[:,[0,2,3]]
            m3Dtmp = np.sum(m3Dtmp,1)
            list_pos = np.where(m3Dtmp==0)[0]
            if(len(list_pos) > 0):
                linkSig2Dto3D[ite] = list_pos[0]
                
            mm = np.array([-1,n1,n2,n3])
            m3Dtmp = (m3Dsort - mm)[:,[1,2,3]]
            m3Dtmp = np.sum(m3Dtmp,1)
            list_pos = np.where(m3Dtmp==0)[0]
            if(len(list_pos) > 0):
                linkSig2Dto3D[ite] = list_pos[0]            
            
            # for ite3,mm3 in enumerate(m3D):
            #     [m1,m2,m3,m4] = np.sort(mm3)
            #     if(n1 == m1 and n2 == m2 and n3 == m3):
            #         linkSig2Dto3D[ite] = ite3
            #         break
            #     if(n1 == m1 and n2 == m2 and n3 == m4):
            #         linkSig2Dto3D[ite] = ite3
            #         break
            #     if(n1 == m1 and n2 == m3 and n3 == m4):
            #         linkSig2Dto3D[ite] = ite3
            #         break
            #     if(n1 == m2 and n2 == m3 and n3 == m4):
            #         linkSig2Dto3D[ite] = ite3
            #         break
    
    linkSig1Dto3D = []                
    for ite,m in enumerate(m1D):
        linkSig1Dto3D.append(-1)
        if(mt1D[ite] == "2"): 
            [n1,n2] = np.sort(m)
            
            for ite2,mm2 in enumerate(m2D):
                [m1,m2,m3] = np.sort(mm2)
                if(n1 == m1 and n2 == m2):
                    linkSig1Dto3D[ite] = linkSig2Dto3D[ite2]
                    break
                if(n1 == m2 and n2 == m3):
                    linkSig1Dto3D[ite] = linkSig2Dto3D[ite2]
                    break           
                    
    linkSig0Dto3D = []                
    for ite,m in enumerate(m0D):
        linkSig0Dto3D.append(-1)
        if(mt0D[ite] == "3"): 
            
            for ite1,mm1 in enumerate(m1D):
                [m1,m2] = np.sort(mm1)
                if(m == m1 or m == m2):
                    linkSig0Dto3D[ite] = linkSig1Dto3D[ite1]
                    break
                    
                    
    return [linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D]    
    
    
#########################################################
# Convertion from P1 mesh to P2 mesh.    
def convertP1toP2(nomFichierP1, nomFichierP2):
    
    # Récupération des données du maillage d'ordre 1 :
    [points,meshes3D,meshes2D,meshes1D,meshes0D,meshesTag3D,meshesTag2D,meshesTag1D,meshesTag0D] = readMsh3(nomFichierP1)
    nbPoints = len(points)
    nbTetra = len(meshes3D)
    nbTri = len(meshes2D)
    nbLi = len(meshes1D)
    nbNoeuds = len(meshes0D)
    points2 = points
    meshes3D2 = np.zeros((len(meshes3D[:,0]),len(meshes3D[0,:]) + 6),dtype = int) # ok
    meshes2D2 = np.zeros((len(meshes2D[:,0]),len(meshes2D[0,:]) + 3),dtype = int)
    meshes1D2 = np.zeros((len(meshes1D[:,0]),len(meshes1D[0,:]) + 1),dtype = int)
    meshes0D2 = meshes0D
    meshesTag3D2 = meshesTag3D
    meshesTag2D2 = meshesTag2D
    meshesTag1D2 = meshesTag1D
    meshesTag0D2 = meshesTag0D
    fid = open("tmp","w")
    print("ecriture .msh")
    # Ecriture du maillage :    
    # Ecriture des noeuds, on en rajoute sur les milieux de chaques arêtes
    indP = 0
    for pt in points:
        indP = indP + 1
        fid.write(str(indP) + " " + str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n")
    
    #  tout en créant les points milieux, on crée un tableau d'arètes :
    
    MatriceAr = np.zeros((nbPoints,nbPoints),dtype=int)
    print("aretes tetraedres")
    for i in range(nbTetra):
        utils.update_progress(((i+1.0)/nbTetra))
        [indS1,indS2,indS3,indS4] = meshes3D[i]
        Coord = [points[indS1,:],points[indS2,:],points[indS3,:],points[indS4,:]]
        CoordMilieu = [(Coord[0] + Coord[1])/2,(Coord[0] + Coord[2])/2,(Coord[0] + Coord[3])/2,(Coord[2] + Coord[1])/2,(Coord[3] + Coord[1])/2,(Coord[2] + Coord[3])/2]
        if (MatriceAr[indS1,indS2] == 0):
            indP = indP + 1
            MatriceAr[indS1,indS2] = indP
            MatriceAr[indS2,indS1] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[0][0]) + " " + str(CoordMilieu[0][1]) + " " + str(CoordMilieu[0][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[0][0],CoordMilieu[0][1],CoordMilieu[0][2]]], axis = 0)
        if (MatriceAr[indS1,indS3] == 0):
            indP = indP + 1
            MatriceAr[indS1,indS3] = indP
            MatriceAr[indS3,indS1] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[1][0]) + " " + str(CoordMilieu[1][1]) + " " + str(CoordMilieu[1][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[1][0],CoordMilieu[1][1],CoordMilieu[1][2]]], axis = 0)
        if (MatriceAr[indS1,indS4] == 0):
            indP = indP + 1
            MatriceAr[indS1,indS4] = indP
            MatriceAr[indS4,indS1] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[2][0]) + " " + str(CoordMilieu[2][1]) + " " + str(CoordMilieu[2][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[2][0],CoordMilieu[2][1],CoordMilieu[2][2]]], axis = 0)
        if (MatriceAr[indS2,indS3] == 0):
            indP = indP + 1
            MatriceAr[indS2,indS3] = indP
            MatriceAr[indS3,indS2] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[3][0]) + " " + str(CoordMilieu[3][1]) + " " + str(CoordMilieu[3][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[3][0],CoordMilieu[3][1],CoordMilieu[3][2]]], axis = 0)
        if (MatriceAr[indS2,indS4] == 0):
            indP = indP + 1
            MatriceAr[indS2,indS4] = indP
            MatriceAr[indS4,indS2] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[4][0]) + " " + str(CoordMilieu[4][1]) + " " + str(CoordMilieu[4][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[4][0],CoordMilieu[4][1],CoordMilieu[4][2]]], axis = 0)
        if (MatriceAr[indS3,indS4] == 0):
            indP = indP + 1
            MatriceAr[indS3,indS4] = indP
            MatriceAr[indS4,indS3] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[5][0]) + " " + str(CoordMilieu[5][1]) + " " + str(CoordMilieu[5][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[5][0],CoordMilieu[5][1],CoordMilieu[5][2]]], axis = 0)
    print("aretes triangles")
    for i in range(nbTri):
        utils.update_progress(((i+1.0)/nbTri))
        [indS1,indS2,indS3] = meshes2D[i]
        Coord = [points[indS1,:],points[indS2,:],points[indS3,:]]
        CoordMilieu = [(Coord[0] + Coord[1])/2,(Coord[0] + Coord[2])/2,(Coord[2] + Coord[1])/2]
        if (MatriceAr[indS1,indS2] == 0):
            indP = indP + 1
            MatriceAr[indS1,indS2] = indP
            MatriceAr[indS2,indS1] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[0][0]) + " " + str(CoordMilieu[0][1]) + " " + str(CoordMilieu[0][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[0][0],CoordMilieu[0][1],CoordMilieu[0][2]]], axis = 0)
        if (MatriceAr[indS1,indS3] == 0):
            indP = indP + 1
            MatriceAr[indS1,indS3] = indP
            MatriceAr[indS3,indS1] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[1][0]) + " " + str(CoordMilieu[1][1]) + " " + str(CoordMilieu[1][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[1][0],CoordMilieu[1][1],CoordMilieu[1][2]]], axis = 0)
        if (MatriceAr[indS2,indS3] == 0):
            indP = indP + 1
            MatriceAr[indS2,indS3] = indP
            MatriceAr[indS3,indS2] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[2][0]) + " " + str(CoordMilieu[2][1]) + " " + str(CoordMilieu[2][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[2][0],CoordMilieu[2][1],CoordMilieu[2][2]]], axis = 0)
    print("lignes")
    for i in range(nbLi):
        utils.update_progress(((i+1.0)/nbLi))
        [indS1,indS2] = meshes1D[i]
        Coord = [points[indS1,:],points[indS2,:]]
        CoordMilieu = [(Coord[0] + Coord[1])/2]
        if (MatriceAr[indS1,indS2] == 0):
            indP = indP + 1
            MatriceAr[indS1,indS2] = indP
            MatriceAr[indS2,indS1] = indP
            fid.write(str(indP) + " " + str(CoordMilieu[0][0]) + " " + str(CoordMilieu[0][1]) + " " + str(CoordMilieu[0][2]) + "\n")
            points2 = np.append(points2,[[CoordMilieu[0][0],CoordMilieu[0][1],CoordMilieu[0][2]]], axis = 0)
    fid.write("$EndNodes\n")
    MatriceAr = csc_matrix(MatriceAr)
    
    #on écrit les elements, ils changent de types
    fid.write("$Elements\n")
    fid.write(str(nbNoeuds + nbLi + nbTri + nbTetra) + "\n") #numberOfElements
    #points
    indE = 1
    for i in range(nbNoeuds):
        Tag = meshesTag0D[i]
        fid.write(str(indE) + " 15 2 " + str(Tag) + " " + str(Tag) + " " + str(meshes0D[i]+1) + "\n")
        indE = indE + 1
    #lignes
    for i in range(nbLi):
        Tag = meshesTag1D[i]
        ligneP1 = meshes1D[i,:]
        fid.write(str(indE) + " 8 2 " + str(Tag) + " " + str(Tag) + " " + str(ligneP1[0]+1) + " " + str(ligneP1[1]+1) + " " + str(MatriceAr[ligneP1[0],ligneP1[1]]) + "\n")
        meshes1D2[i,:] = [ligneP1[0],ligneP1[1],MatriceAr[ligneP1[0],ligneP1[1]] - 1]
        indE = indE + 1
    #triangles
    for i in range(nbTri):
        Tag = meshesTag2D[i]
        tri = meshes2D[i,:]
        fid.write(str(indE) + " 9 2 " + str(Tag) + " " + str(Tag) + " " + str(tri[0]+1) + " " + str(tri[1]+1) + " " + str(tri[2]+1) + " " + str(MatriceAr[tri[0],tri[1]]) + " " + str(MatriceAr[tri[2],tri[1]]) + " " + str(MatriceAr[tri[0],tri[2]]) + "\n")
        meshes2D2[i,:] = [tri[0],tri[1],tri[2],MatriceAr[tri[0],tri[1]] - 1,MatriceAr[tri[2],tri[1]] - 1,MatriceAr[tri[0],tri[2]] - 1]
        indE = indE + 1
    #tretraedrons
    for i in range(nbTetra):
        Tag = meshesTag3D[i]
        tetra = meshes3D[i,:]
        fid.write(str(indE) + " 11 2 " + str(Tag) + " " + str(Tag) + " " + str(tetra[0]+1) + " " + str(tetra[1]+1) + " " + str(tetra[2]+1) + " " + str(tetra[3]+1) + " " + str(MatriceAr[tetra[0],tetra[1]]) + " " + str(MatriceAr[tetra[2],tetra[1]]) + " " + str(MatriceAr[tetra[0],tetra[2]]) + " " + str(MatriceAr[tetra[0],tetra[3]]) + " " + str(MatriceAr[tetra[2],tetra[3]]) + " " + str(MatriceAr[tetra[3],tetra[1]]) + "\n")
        meshes3D2[i,:] = [tetra[0],tetra[1],tetra[2],tetra[3],MatriceAr[tetra[0],tetra[1]] - 1,MatriceAr[tetra[2],tetra[1]] - 1,MatriceAr[tetra[0],tetra[2]] - 1,MatriceAr[tetra[0],tetra[3]] - 1,MatriceAr[tetra[2],tetra[3]] - 1,MatriceAr[tetra[3],tetra[1]] - 1]
        indE = indE + 1
    fid.write("$EndElements\n")
    fid.close()
    # On a besoin de procéder ainsi car on ne peut pas savoir, au moment de l'écriture des points de tmp, le nombre total de noeuds pour éviter les doublons
    fidTmp = open("tmp","r")
    fidRes = open(nomFichierP2,"w")
    fidRes.write("$MeshFormat\n")
    fidRes.write("2.2 0 8\n")
    fidRes.write("$EndMeshFormat\n")
    fidRes.write("$Nodes\n")
    fidRes.write(str(indP) + "\n")
    for ligne in fidTmp.readlines():
        fidRes.write(ligne)
    fidRes.close()
    fidTmp.close()
    return [points2,meshes3D2,meshes2D2,meshes1D2,meshes0D2,meshesTag3D2,meshesTag2D2,meshesTag1D2,meshesTag0D2]
    
#########################################################
# Conversion from P2 mesh to P1 mesh (only for writing solution)    
def getEqP1(m3D,order):
    m3DP1 = [] 
    
    for m in m3D:
        if(order == 1):
            m3DP1.append([m[0],m[1],m[2],m[3]])
        if(order == 2):
            m3DP1.append([m[3],m[7],m[8],m[9]])
            m3DP1.append([m[0],m[4],m[6],m[7]])
            m3DP1.append([m[4],m[6],m[7],m[8]])
            m3DP1.append([m[4],m[7],m[9],m[8]])
            m3DP1.append([m[1],m[4],m[5],m[9]])
            m3DP1.append([m[4],m[5],m[9],m[6]])
            m3DP1.append([m[5],m[6],m[9],m[2]])
            m3DP1.append([m[2],m[6],m[8],m[9]])

    return np.array(m3DP1)   
#########################################################


#########################################################
# Check if two lines intersects:
def checkIntersection(seg1,seg2):
    coeff1 = np.polynomial.polynomial.polyfit(seg1[:,0],seg1[:,1],1)
    coeff2 = np.polynomial.polynomial.polyfit(seg2[:,0],seg2[:,1],1)
    
    [s1,o1] = [coeff1[1],coeff1[0]]
    [s2,o2] = [coeff2[1],coeff2[0]]
    
    # Check if the lines cross in the computational domain.
    doCross = False
    CrossPt = np.array([0,0])
    
    # print(s1,o1)
    # print(s2,o2)
    eps = 1e-10
    if(np.abs(s1-s2) > 1e-10):
        x = (o2 - o1)/(s1 - s2)
        y = s1 * x + o1
        
        min1X = min(seg1[0,0],seg1[-1,0])
        max1X = max(seg1[0,0],seg1[-1,0])
        min1Y = min(seg1[0,1],seg1[-1,1])
        max1Y = max(seg1[0,1],seg1[-1,1])
        
        min2X = min(seg2[0,0],seg2[-1,0])
        max2X = max(seg2[0,0],seg2[-1,0])
        min2Y = min(seg2[0,1],seg2[-1,1])
        max2Y = max(seg2[0,1],seg2[-1,1])
        
        if(x >= min1X-eps and x <= max1X+eps and y >= min1Y-eps and y <= max1Y+eps):
            if(x >= min2X-eps and x <= max2X+eps and y >= min2Y-eps and y <= max2Y+eps):
                CrossPt = np.array([x,y])
                doCross = True
                
    # if(np.abs(s1-s2) < 1e-3 and np.abs(o1-o2) < 1e-3):
#         doCross = True
    return [doCross,CrossPt,s1,s2]     
    
    
def computeLineSeqRecurs(listeSeg,res,endPt,segAdd=[]):
    # print(len(res))
#     print('listeSeg',listeSeg)
#     print('res',res)
#     print('segAdd',segAdd)
#     print()
    if (len(segAdd) != 0):
        for i in range(len(res)-1):
            segTmp = np.array([res[i][0:2],res[i+1][0:2]])
            
            for j in range(len(segAdd)-1):
                ssegAdd = np.array([segAdd[j,0:2],segAdd[j+1,0:2]])
                [doCross,CrossPt,s1,s2] = checkIntersection(ssegAdd,segTmp)
                if (doCross == True): 
                    return False
        
        # distPt = (np.dot(res[len(res)-1] - segAdd[1], res[len(res)-1] - segAdd[1]))
#         lenSeg = (np.dot(segAdd[1] - segAdd[0],segAdd[1] - segAdd[0]))
#         if(distPt/lenSeg  < 0.5):
#             return False
        
        segAddbis = np.array([res[len(res)-1][0:2],segAdd[0,0:2]])
        for i in range(len(res)-2):
            segTmp = np.array([res[i][0:2],res[i+1][0:2]])
                            
            [doCross,CrossPt,s1,s2] = checkIntersection(segAddbis,segTmp)
            if (doCross == True):
                return False
                
        for i in range(len(segAdd)-2):   
            ssegAdd = np.array([segAdd[i+1,0:2],segAdd[i+2,0:2]])
            [doCross,CrossPt,s1,s2] = checkIntersection(ssegAdd,segAddbis)
            if (doCross == True): 
                return False     
                
        # distPt = np.dot(res[len(res)-2] - segAdd[0], res[len(res)-2] - segAdd[0])
#         lenSeg = np.dot(res[len(res)-1] - segAdd[0],res[len(res)-1] - segAdd[0])
#         if(distPt / lenSeg < 0.5):
#             return False
        
        for s in segAdd:
            res.append(s)
            
        if (len(listeSeg) == 0):
            # plt.ion()
#             for i,e in enumerate(res):
#                 if(i < len(res)-1):
#                     if(i % 2 == 0):
#                         plt.plot([e[0],res[i+1][0]],[e[1],res[i+1][1]],'k-')
#                     else:
#                         plt.plot([e[0],res[i+1][0]],[e[1],res[i+1][1]],'r-')
#                 plt.show()
#                 plt.pause(1)
#             plt.ioff()
#             input()
            
            segEnd = np.array([res[-1][0:2],endPt[0:2]])
            for i in range(len(res)-2):
                segTmp = np.array([res[i][0:2],res[i+1][0:2]])
                                
                [doCross,CrossPt,s1,s2] = checkIntersection(segEnd,segTmp)
                if (doCross == True):
                    for j in range(len(segAdd)):
                        res.pop()
                    return False
            return True
            
    lastPos = res[-1][0:2]  
    listDist = []
    listSign = []   
    for i,seg in enumerate(listeSeg):
        pt1 = seg[0,0:2]
        pt2 = seg[-1,0:2]
        #print(pt1,lastPos)
        dist1 = np.dot(lastPos-pt1,lastPos-pt1)
        dist2 = np.dot(lastPos-pt2,lastPos-pt2)
        if(dist1 <= dist2):
            listDist.append(dist1)
            listSign.append(1)
        else:
            listDist.append(dist2)
            listSign.append(-1)
                
    listDist = np.array(listDist)   
    ordering = listDist.argsort()
    #listeSeg = listeSeg[listDist.argsort()] 
            
    for i,seg in enumerate(listeSeg):
        ind = ordering[i]
        segTobeAdd = listeSeg[ind]
        if(listSign[i] == -1):
            segTobeAdd = np.flip(segTobeAdd,0)
        newListSeg = [segNew for j,segNew in enumerate(listeSeg) if j != ind]
        #copyList = (copy.deepcopy(listeSeg))
        #newListSeg = copyList.pop(i)
        #print(newListSeg)
        resRecurs = computeLineSeqRecurs(newListSeg,res,endPt,segTobeAdd)
        if(resRecurs == True):
            return True
        resRecurs = computeLineSeqRecurs(newListSeg,res,endPt,np.flip(segTobeAdd,0))
        if(resRecurs == True):
            return True
    
    if(len(segAdd) != 0):
        for j in range(len(segAdd)):
            res.pop()
        return False
    return False                 
                
                
#Function for VTK reader:
def splitEsp(Listchar):
    res = []
    tmp = Listchar.split(" ")
    for terme in tmp :
        if (terme != ''):
            res.append(terme)
    return res
#Reading VTK file for a priori resistivity
def readVtk(nomFichier):
    fid = open(nomFichier,"r");

    contenu = fid.read();
    contenu = contenu.split("\n");

    points = []
    meshes3D = [] # à l'ordre 1
    resis = [] # même longueur que la liste des tétraèdres, la ieme composante de la liste contient la valeur approchée de la résistivité dans le tétraèdre i, cad stocké dans la ieme ligne de meshes3D
    i = 0
    while(contenu[i][:6] != "POINTS"):
        i = i + 1
        print(i)
    nbPoints = int(splitEsp(contenu[i])[1])
    print("nbPoints = " + str(nbPoints))
    for k in range(nbPoints):
        i = i + 1
        points.append(splitEsp(contenu[i]))
    i = i + 1
    nbCells = int(splitEsp(contenu[i])[1])
    print("nbCells = " + str(nbCells))
    for k in range(nbCells):
        i = i + 1
        meshes3D.append(splitEsp(contenu[i])[1:])
    i = i + 4 + nbCells
    for k in range(nbCells):
        i = i + 1
        resis.append(float(contenu[i]))
    fid.close()
    return [points,meshes3D,resis]                                      
#########################################################   


#########################################################
## Read gmsh file (version 3.0) :
def reWriteGeo(nomFichierR,nomFichierW,Lx,Ly,Lz,h,coeff,xmax,xmin,ymax,ymin,coeffBot=-1):
    fidR = open(nomFichierR,"r");

    contenu = fidR.read();
    contenu = contenu.split("\n");

    
    fid = open(nomFichierW,"w")
    
    ydiff = ymax-ymin
    xdiff = xmax-xmin
    coeffBis = int(Lz / max(xdiff,ydiff))
    if(coeffBis <= 1):
        coeffBis = 1
    if(coeffBis >=4):
        coeffBis = 4
        
    if(coeffBot != -1):
        print(coeffBot)
        coeffBis = coeffBot    

    fid.write("h="+str(h)+";\n")
    fid.write("coeff="+str(coeff)+";\n")
    fid.write("coeffBis="+str(coeffBis)+";\n")
    fid.write("Lx="+str(Lx)+";\n")
    fid.write("Ly="+str(Ly)+";\n")
    fid.write("Lz="+str(Lz)+";\n")
    
    
    it = 0
    for ele in contenu:
        if(it > 5):
            fid.write(ele+'\n')
            #print(it,ele)
        it += 1
#########################################################                      