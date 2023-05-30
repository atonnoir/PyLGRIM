#!/usr/bin/python3.4
# -*-coding:Utf-8 -*

# by A. Tonnoir

import numpy as np
from scipy import linalg
import scipy.sparse.linalg
import copy

import matplotlib.pyplot as plt


# Added "modules"
import sys
sys.path.append("../Lib/")
import mesh
import dataERI
import finiteElem    
    
    
print("Start of the program")
#########################################################
print("Choose finite element order (1 or 2) :")
order = int(input("Order :"))


print("Choose BC :")
print("1. Infinite Element")
print("2. Mixed BC")
print("3. Dirichlet BC")    
choiceBC = input()    

saveSolVTK = 1
#########################################################
## Reading mesh DEM and building FE matrices:
# Reading mesh:
monMaillage = "../Data/meshDEM.msh"
print("Reading 3D mesh DEM.")
[ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D] = mesh.readMsh3(monMaillage)
# linkSig2Dto3D[i] gives the index of the 3D mesh associated to the boundary 2D mesh i.
# linkSig1Dto3D[i] gives an index of a 3D mesh with the (line) boundary 1D mesh i.
# linkSig0Dto3D[i] gives an index of a 3D mesh associated to the (corner) boundary 0D mesh i.
print("Get link between m3D, m2D, m1D and m0D")
[linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D] = mesh.getLinkm3Dm2Dm1Dm0D(m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D) 

if(order == 2):
    [ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D] = mesh.convertP1toP2(monMaillage,"../Data/meshDEMO2.msh")
    print("Nb nodes new:",len(ptsM))


xmin = min(ptsM[:,0])
xmax = max(ptsM[:,0])
ymin = min(ptsM[:,1])
ymax = max(ptsM[:,1])
zmin = min(ptsM[:,2])
zmax = max(ptsM[:,2])

# Index in the list ptsM (points of the mesh) of the electrode positions: 
nodesInj = [ele for i,ele in enumerate(m0D) if(mt0D[i] == "4")]


# Barycenter of electrode positions:  (necessary for Mixed BC)      
posM = np.array([0.0,0.0,0.0])
for i,inj in enumerate(nodesInj):
    posM += ptsM[inj]
posM = posM / len(nodesInj)

Lmin = max(xmax-xmin,ymax-ymin)
#########################################################

#########################################################
## Initialization 
# of the conductivity:
print("Initializing conductivity vector")
sigma = []
for m in m3D:
    sigma.append(1.0)
# If non constant sigma:    
sigma = np.loadtxt('conductivite.data')     
sigma = np.array(sigma)
sigmaHom = sigma*0.0 + 1.0

# and of the source vectors and results.
FF = np.zeros( (len(ptsM),len(nodesInj)) )
VV = np.zeros( (len(ptsM),len(nodesInj)) )

Vmes = np.zeros( (len(nodesInj),len(nodesInj) ) )
VmesEx = np.zeros( (len(nodesInj),len(nodesInj) ) )

VVhom = np.zeros( (len(ptsM),len(nodesInj)) )
for ite,inj in enumerate(nodesInj):
    FF[inj,ite] = 1.0
    
# Getting list of local rigidity matrix on each cell:
# print("Construction of matrix:")
# KKint = finiteElem.rigidityDirect(ptsM,m3D,order,sigma,1)
# KKhomint = finiteElem.rigidityDirect(ptsM,m3D,order,sigmaHom,1)

# Getting list of local rigidity matrix on each cell:
print("Building split matrix:")
listKEle = finiteElem.rigidityHomogenSplit(ptsM,m3D,order)  
#########################################################


#########################################################
# Mesures simualtion
# Building rigidity matrix:
# Case of variable sigma
print("Building matrix:")
#KK = finiteElem.buildMatrixDirect(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,sigma,KKint,order,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D)    
KK = finiteElem.buildMatrix(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,sigma,listKEle,order,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D)    
# Case of constant sigma = 1 (sigmaHom):
print("Building matrix (homogeneous case):")
#KKhom = finiteElem.buildMatrixDirect(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,sigmaHom,KKhomint,order,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D)
KKhom = finiteElem.buildMatrix(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,sigmaHom,listKEle,order,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D)   
print("Resolution of the linear systeme for each injection:")
VV = scipy.sparse.linalg.spsolve(KK,FF)
print("Resolution of the linear systeme for each injection (homogeneous problem):")
VVhom = scipy.sparse.linalg.spsolve(KKhom,FF)      
print("End of the resolution.")
#########################################################

#########################################################
print("Saving solution.")

m3DeqP1 = mesh.getEqP1(m3D,order)
for i,inj in enumerate(nodesInj):
    print("Writting solution:",i+1," / ",len(nodesInj))

    Sol = VV[:,i]
    #########################################################
    ## Computation of the exact solution in case of a halfsphere:
    Solex = Sol * 0.0
    for ite,pts in enumerate(ptsM):
        val = 0
        if(ite != inj):
            # Completer ICI (q. 2.c) ) :
            ptEl = ptsM[inj]*1.0
            vdist = pts - ptEl
            #vdist[2] = 0.0
            val = 1.0 / (2.0 * np.pi * np.sqrt( np.dot(vdist,vdist) ) )
        Solex[ite] = val
    #########################################################

    #########################################################    
    ## Ecriture de la solution vtk    
    if(saveSolVTK == 1):
        mesh.writeVTK("ResVTK/sol"+str(i+1)+".vtk",Sol,Solex,np.abs(Sol-Solex),ptsM,m3DeqP1)
    #########################################################

    #########################################################
    ## Matrice de mesures :
    Vmes[i,:] = [Sol[ind] for ind in nodesInj]  
    Vmes[i,i] = 0
    VmesEx[i,:] = [Solex[ind] for ind in nodesInj]   
    VmesEx[i,i] = 0
    #########################################################
#########################################################


#########################################################
# Writting ER file
print("Writing save file")
coordXYZinj = []
for inj in nodesInj:
    coordXYZinj.append(ptsM[inj])
np.savetxt('../Data/nodesInjLoc.txt',coordXYZinj)
dataERI.writeERFile('../Data/resistivityMes.txt',VV,VVhom,nodesInj,ptsM)

#########################################################

#########################################################
## Sauvergarde resultat et affichage erreur
np.savetxt('VmesExp.data',Vmes)
np.savetxt('VmesExa.data',VmesEx)

# Rho Apparent :
RhoA = Vmes / VmesEx
np.savetxt('RhoApp.data',RhoA)

#plt.show()
c = plt.pcolor(100*np.abs(Vmes-VmesEx)/VmesEx,cmap='jet',vmin=0.0,vmax=20)
plt.colorbar(c)
plt.savefig('Error_potential.pdf')
plt.show()



ll = []
nbT = int(len(nodesInj)/2)
for i in range(nbT):
    #print(i,len(nodesInj)-1-i,RhoA[i,len(nodesInj)-1-i])
    ll.append(RhoA[i,len(nodesInj)-1-i])

ll.reverse()    
plt.plot(ll,'x-')
plt.show()    

#########################################################



mesh.writeCellVTK('conductivite.vtk',sigma,ptsM,m3D)

print("Fin du programme")
