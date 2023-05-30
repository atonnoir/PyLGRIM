#!/usr/bin/python3.4
# -*-coding:Utf-8 -*

# by A. Tonnoir


# Standard library
import numpy as np
from scipy import linalg
import scipy.sparse.linalg
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix, identity

# Added "modules"
import sys
sys.path.append("../Lib/")
import mesh
import dataERI
import finiteElem
import utils

from time import sleep

# 1:9.815440124573982, 2:0.38162052064662444 (o1)
# 1:0.12435048946759508, 2:0.01788009782083559 (o2)
    
    
print("Start of the program:")    
#########################################################
# Parameter for the inverse problem resolution
useLog = 1 # Use the minimisation with respect to the log of the apparent resistivity (0- no, 1- yes)
order = 1 # Finite element order (1 or 2 for the moment)
choiceBC = "" # Choice of the Boundary Condition (Infinie elemnent, mixte BC or homogeneous Dirichlet)
choiceReg = "" # Choice of the regularization term (ticho, gradient) 
choiceMetho = "" # Choice of the minimization algorithm (Gauss_Newton or stepest descent) 

FichierVtk = "./Inv1Lig_o1_synth_coarsed/resistivity5.vtk" # A priori file
FichierVtk = "./Inv3LigVNCLarge/resistivity3.vtk" # A priori file
#########################################################


#########################################################
## Reading experimental transfert resistance data:
## ERI File:
# 1 Lig
#listERIFile = ['../Data/TV1_BAS_ALL64_Mardi19Avril2016.dat']
# 2 Lig
#listERIFile = ['../Data/P5_ALL64_1-2017-04-04-114315.dat','../Data/P2_ALL64_2-Mercredi20Avril2016.dat']
# 3 Lig
listERIFile = ['../Data/TV1_BAS_ALL64_Mardi19Avril2016.dat','../Data/TV2_ALL64_1-2016-04-20.dat','../Data/P5_ALL64_1-2017-04-04-114315.dat','../Data/P2_ALL64_2-Mercredi20Avril2016.dat','../Data/P1_ALL64_1-Mercredi20Avril2016.dat']# ,'../Data/P5_ALL64_1-2017-04-04-114315.dat','../Data/P2_ALL64_2-Mercredi20Avril2016.dat'

# Terril:
#listERIFile = ['../Data/Terril/P3_WEN64_14-2014-07-30-181203.dat','../Data/Terril/P1_WEN64_5-2014-07-30-180232.dat','../Data/Terril/P2_WEN64_9-2014-07-30-181301.dat']

# Synthetic data (STEP)
#listERIFile = ['../Data/resistivityMesStep1.txt']#,'../Data/resistivityMesStep2.txt'

# Exemple of files:
# '../Data/P1_ALL64_1-Mercredi20Avril2016.dat', '../Data/P2_ALL64_2-Mercredi20Avril2016.dat', '../Data/P5_ALL64_1-2017-04-04-114315.dat', 
# '../Data/TV1_BAS_ALL64_Mardi19Avril2016.dat',
# '../Data/resistivityMes1.txt','../Data/resistivityMes2.txt'
# Church: '../Data/Chapelle/2020-03-10-CHAP-EST_DDP64_Vin_7ms_ec6_1.dat'



#[seqABMN, resistyExp] = dataERI.readERFile('./../Data/TV1_BAS_ALL64_Mardi19Avril2016.dat')
#ListElectrodeLoc = ["../Data/localisation_TV1_BAS_A_UTILISER.txt"];
#ListElectrodeLoc = ["../Data/localisation_TV1_BAS_A_UTILISER.txt"]; #"../Data/localisation_TV1_BAS_A_UTILISER.txt" ,"../Data/localisation_P1.txt" ,"../Data/localisation_P5.txt"
#ListElectrodeLoc = ["../Data/nodesInjLoc.txt"]; ,"../Data/localisation_P5.txt","../Data/localisation_P2.txt"

## Electrode positions:
# !! Must be in the same order (electrode positions and ERI data). !!
# 1 Lig :
#ListElectrodeLoc = ["../Data/localisation_TV1_BAS_A_UTILISER.txt"];
# 2 Lig :
#ListElectrodeLoc = ["../Data/localisation_P5.txt","../Data/localisation_P2.txt"]
# 3 Lig :
ListElectrodeLoc = ["../Data/localisation_TV1_BAS_A_UTILISER.txt","../Data/localisation_TV2.txt","../Data/localisation_P5.txt","../Data/localisation_P2.txt","../Data/localisation_P1.txt"] # ,"../Data/localisation_P5.txt","../Data/localisation_P2.txt"

# Terril:
#ListElectrodeLoc = ["../Data/Terril/positions_electrodes_P3_ok.txt","../Data/Terril/positions_electrodes_P1_ok.txt","../Data/Terril/positions_electrodes_P2_ok.txt"]; # "../Data/Terril/positions_electrodes_P2_ok.txt",

# Synthetic data (STEP):
#ListElectrodeLoc = ["../Data/nodesInjLocStep1.txt"]; #,"../Data/nodesInjLocStep2.txt"

# Exemple of files:
# "../Data/localisation_P1.txt", "../Data/localisation_P2.txt", "../Data/localisation_P5.txt"
# "../Data/localisation_TV1_BAS_A_UTILISER.txt"
# "../Data/nodesInjLoc.txt"
# Church: "../Data/Chapelle/coord_electrodes1m.txt"


#########################################################


#########################################################
# Treatment of electrode positions:
ptsElec = []
nbElec = [0]
for j,name in enumerate(ListElectrodeLoc):
    if(j==0):
        ptsElec = np.genfromtxt(name, delimiter=" ")
        nbElec.append(len(ptsElec))
    else:
        ptsElectmp = np.genfromtxt(name, delimiter=" ")
        ptsElec = np.concatenate((ptsElec,ptsElectmp))  
        nbElec.append(len(ptsElectmp))

# Reading list of electrodes experimental data:        
seqABMN = []
resistyExp = []
shift = 0
for j,name in enumerate(listERIFile):
    [seqABMNtmp, resistyExptmp] = dataERI.readERFile(name)
    shift += nbElec[j] #len(seqABMN)
    for k,sq in enumerate(seqABMNtmp):
        for l in range(4):
            seqABMNtmp[k][l] += shift # Shift of the index number of the electrode
        seqABMN.append(seqABMNtmp[k])
        
        resistyExp.append(resistyExptmp[k])  
        
resistyExp = np.array(resistyExp)   

# print('nbElec : ',len(ptsElec))
# print('nbSeqABMN : ',len(seqABMN))
# input()        
#########################################################

    
#########################################################
## Reading mesh DEM and building FE matrices:
# Reading mesh:
print("Choose finite element order (1 or 2) :")
order = int(input("Order :"))

monMaillage = "../Data/meshDEM.msh"
print("Reading DEM mesh.")
[ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D] = mesh.readMsh3(monMaillage)
# linkSig2Dto3D[i] gives the index of the 3D mesh associated to the boundary 2D mesh i.
# linkSig1Dto3D[i] gives an index of a 3D mesh with the (line) boundary 1D mesh i.
# linkSig0Dto3D[i] gives an index of a 3D mesh associated to the (corner) boundary 0D mesh i.      
print("Get link between m3D, m2D, m1D and m0D")
[linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D] = mesh.getLinkm3Dm2Dm1Dm0D(m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D)

if(order == 2):
    print("Conversion mesh P1 to P2 :")
    [ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D] = mesh.convertP1toP2(monMaillage,"../Data/meshDEMO2.msh")    
    print("Nb pts new : ",len(ptsM))
# Index in the list ptsM (points of the mesh) of the electrode positions: 
nodesInj = [ele for i,ele in enumerate(m0D) if(mt0D[i] == "4")]

# Matching electrode position from 3D mesh and from file.
print("Ordering electrode positions list.")
for i,ptElec in enumerate(ptsElec):
    find = 0
    utils.update_progress((i+1.0)/len(ptsElec)) 
    for j,inj in enumerate(nodesInj):#nodesInj
        ptNode = ptsM[inj]
        diffPts = ptNode - ptElec
        if(np.dot(diffPts,diffPts) < 1e-10):
            tmp = nodesInj[i]
            nodesInj[i] = inj
            nodesInj[j] = tmp
            find = 1
            break
    if(find == 0):
        print("Matching problem")
        input()

# Barycenter of electrode positions:  (necessary for Mixte BC)      
posM = np.array([0.0,0.0,0.0])
for i,inj in enumerate(nodesInj):
    posM += ptsM[inj]
posM = posM / len(nodesInj)
        
        
xmin = min(ptsM[:,0])
xmax = max(ptsM[:,0])
ymin = min(ptsM[:,1])
ymax = max(ptsM[:,1])
zmin = min(ptsM[:,2])
zmax = max(ptsM[:,2])


Lmin = max(xmax-xmin,ymax-ymin)

# Getting list of local rigidity matrix on each cell:
print("Building split matrix:")
listKEle = finiteElem.rigidityHomogenSplit(ptsM,m3D,order)
#########################################################

#########################################################
# Choice of the BC:
print("Choose BC :")
print("1. Infinite Element")
print("2. Mixte BC")
print("3. Dirichlet BC")    
choiceBC = input()    
#########################################################

#########################################################
# Initialisation for the Inverse problem resolution:
iteGN = 0
errL2 = 10.0
tol = 1e-10
IteMax = 100
epsReg = 1.0e0
step = 1.0
errL2Old = 1e5
diffErr = 1.0

# Initialization of the conductivity / resistivity:
sigma = []
sigmaRef = []
resistivity = []
for m in m3D:
    sigma.append(1.0)
    resistivity.append(1.0)
    sigmaRef.append(1.0) 
# If a priori guess:    
print("A priori guess :")
print("1. no")
print("2. yes")
choiceApriori = input()
#sigma = np.loadtxt('conductivite.data') 
if(choiceApriori == "2"):
    #resistivity = np.loadtxt('resistivityApriori.txt')    # A amleriorer avec un read VTK...
    [ptsTmp,tetraTmp,resistivity] = mesh.readVtk(FichierVtk)
sigma = np.array(sigma)
resistivity = np.array(resistivity)
resistivityOld = resistivity*1.0



FF = np.zeros( (len(ptsM),len(nodesInj)) )
VV = np.zeros( (len(ptsM),len(nodesInj)) )
VVhom = np.zeros( (len(ptsM),len(nodesInj)) )
for ite,inj in enumerate(nodesInj):
    FF[inj,ite] = 1.0
        
    
mesh.writeCellVTK('ResVTK/conductivite'+str(iteGN)+'.vtk',sigma,ptsM,m3D)
mesh.writeCellVTK('ResVTK/resistivity'+str(iteGN)+'.vtk',resistivity,ptsM,m3D)     
#########################################################


#########################################################
# Construction of the regularization matrix
print("Regularisation term: ")
print("1. Tichonov")
print("2. Gradient")
print("3. GradStrat")
choiceReg = input()

choiceRegustr = ''
if(choiceReg == "1"):
    choiceRegustr = 'Ticho'
if(choiceReg == "2"):
    choiceRegustr = 'Grad'
if(choiceReg == "3"):    
    choiceRegustr = 'GradStrat'
#########################################################    


#########################################################
# Choice of the minimization method:
print("Minimisation method: ")
print("1. Gauss Newton")
print("2. Gradient descend")
choiceMetho = input()
#########################################################

#########################################################
# Building regularisation matrix:
print("Construction of the regularisation matrix:")
matRegu = finiteElem.buildRegu(ptsM,m3D,choiceRegustr)
#########################################################

#########################################################
# Starting minimization process:

SensitivityMesResis = np.zeros((len(seqABMN),len(sigma)))
matReguCsr = csr_matrix((matRegu[2], (matRegu[0], matRegu[1])),shape=(len(sigma),len(sigma))) 


nbIteDiffErr = 1.0
matRegu[0] = matRegu[0] + len(seqABMN)
dVhom = np.zeros(len(seqABMN))
resistyComp = np.zeros(len(seqABMN))
 
while(errL2 > tol and iteGN < IteMax and nbIteDiffErr <= 1e4):
    iteGN = iteGN + 1
    
    #########################################################
    # Generation of the computational measures:
    print("Building matrix:")
    #KK = finiteElem.buildMatrix(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,sigma,listKEle,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D)
    KK = finiteElem.buildMatrix(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,1.0/resistivity,listKEle,order,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D)     
    print("Resolution of the linear system for all electrode position injection : ")
    #VV = linalg.solve(KK,FF)
    VV = scipy.sparse.linalg.spsolve(KK,FF)
    
    if(iteGN == 1): # First iteration we have sigma = 1 everywhere
        KKhom = finiteElem.buildMatrix(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,sigmaRef,listKEle,order,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D)     
        VVhom = scipy.sparse.linalg.spsolve(KKhom,FF)
        
           
    #########################################################
    
    #########################################################
    print("Computation of ABMN sequence (error computation):")
    ErrResi = np.zeros(len(seqABMN))       

    useEql = []
    for ind,seq in enumerate(seqABMN):
        utils.update_progress((ind+1.0)/len(seqABMN)) 
        
        injA = seq[0]
        injB = seq[1]
        injM = seq[2]
        injN = seq[3]
        
        #print(injA,injB,injM,injN)
        indMesM = nodesInj[seq[2]]
        indMesN = nodesInj[seq[3]]
        
        
        if(iteGN == 1):
            dVhom[ind] = (VVhom[indMesM,injA] - VVhom[indMesM,injB] - VVhom[indMesN,injA] + VVhom[indMesN,injB])
            resistyExp[ind] = resistyExp[ind] / dVhom[ind] 
        
        # Filter to avoid to consider non relevant equations...
        useEq = 1
        
        if(useLog == 1):
        	if(np.abs(dVhom[ind])<1.0/25000):
        		useEq = 0
                
        resistyComp[ind] = (VV[indMesM,injA] - VV[indMesM,injB] - VV[indMesN,injA] + VV[indMesN,injB]) / (dVhom[ind]) 
        if(useLog == 1):   
            if(resistyComp[ind] < 0.01):
            	useEq = 0
            if(resistyComp[ind] > 1e5):
                useEq = 0
                
        if(useLog == 1): 	       
       		if(resistyExp[ind] < 0.01 or resistyExp[ind] > 1e5): 
       			useEq = 0
        
        if(useLog == 1):    
        	if(useEq == 1):
	            ErrResi[ind] = np.log(resistyExp[ind]) - np.log(resistyComp[ind])
	        else:
	         	ErrResi[ind] = 0.0
        else:
            ErrResi[ind] = resistyExp[ind] - resistyComp[ind]
            
        useEql.append(useEq)   
    #########################################################    
         
        
    #########################################################
    # Error computation:    
    print("Computation of the discrepancy.")            
    ErrMes = np.zeros(len(seqABMN) + len(sigma) )
    ErrMes[0:len(seqABMN)] = ErrResi
    
    errL2 = 0.0
    if(useLog == 1):  
        errL2 = (np.dot(ErrResi,ErrResi)  ) / len(nodesInj) # / np.dot(np.log(resistyExp),np.log(resistyExp))
    else:
        errL2 = np.dot(ErrResi,ErrResi) / np.dot(resistyExp,resistyExp)
            
    print("Ite: ",iteGN,", L2 err: ",np.sqrt(errL2)," vs err at previous iteration: ",np.sqrt(errL2Old)," nbIteDifferr:",nbIteDiffErr)        
    #########################################################    

    #########################################################
    print("Computation of ABMN sequences  (for Sensitivity matrix computation):")
    SensitivityMes = {}
    for ind,seq in enumerate(seqABMN):
        utils.update_progress((ind+1.0)/len(seqABMN)) 
        
        useEq = useEql[ind]
        
        injA = seq[0]
        injB = seq[1]
        injM = seq[2]
        injN = seq[3]
        
        #print(injA,injB,injM,injN)
        indMesM = nodesInj[seq[2]]
        indMesN = nodesInj[seq[3]]
                
        # We get only the potential computed for injection at A,B,M and N
        #VABMN = VV[:,[injA,injB,injM,injN]]
        vA = VV[:,injA]
        vB = VV[:,injB]
        vM = VV[:,injM]
        vN = VV[:,injN]
        
        SensitivityMesAM = np.zeros(len(m3D))
        SensitivityMesAN = np.zeros(len(m3D))
        SensitivityMesBM = np.zeros(len(m3D))
        SensitivityMesBN = np.zeros(len(m3D))
        
        if(useEq == 1):
            minAM = min(injA,injM)
            maxAM = max(injA,injM)
            if ((minAM,maxAM) in SensitivityMes):
                SensitivityMesAM = SensitivityMes[(minAM,maxAM)]
            else:
                sizeKel = 16
                if(order == 2):
                    sizeKel = 100
                for iteM,m in enumerate(m3D):
                    #utils.update_progress((iteM+1.0)/len(m3D))
                    # tmpAM = 0.0
#                     for ii,il in enumerate(m):
#                         for jj,ic in enumerate(m):
#                             tmpAM += vM[il] * listKEle[iteM*sizeKel+ii*len(m)+jj] * vA[ic]
                    #KKtmp = finiteElem.rigidityDirect(ptsM,[m3D[iteM]],order,[1.0],0) 
                    KKtmp = listKEle[iteM]
                    tmpAM = np.dot(vM[m],np.dot(KKtmp, vA[m]))
                    SensitivityMesAM[iteM] = tmpAM
                    #del KKtmp
                SensitivityMes[(minAM,maxAM)] = SensitivityMesAM    
        
        
            minAN = min(injA,injN)
            maxAN = max(injA,injN)
            if ((minAN,maxAN) in SensitivityMes):
                SensitivityMesAN = SensitivityMes[(minAN,maxAN)]
            else:
                for iteM,m in enumerate(m3D):
                    #utils.update_progress((iteM+1.0)/len(m3D))
                    # tmpAN = 0.0
#                     for ii,il in enumerate(m):
#                         for jj,ic in enumerate(m):
#                             tmpAN += vN[il] * listKEle[iteM*sizeKel+ii*len(m)+jj] * vA[ic]
                    #KKtmp = finiteElem.rigidityDirect(ptsM,[m3D[iteM]],order,[1.0],0)
                    KKtmp = listKEle[iteM]
                    tmpAN = np.dot(vN[m],np.dot(KKtmp, vA[m]))
                    SensitivityMesAN[iteM] = tmpAN
                    #del KKtmp
                SensitivityMes[(minAN,maxAN)] = SensitivityMesAN    
                
        
            minBM = min(injB,injM)
            maxBM = max(injB,injM)
            if ((minBM,maxBM) in SensitivityMes):
                SensitivityMesBM = SensitivityMes[(minBM,maxBM)]
            else:
                for iteM,m in enumerate(m3D):
                    #utils.update_progress((iteM+1.0)/len(m3D))
                    # tmpBM = 0.0
#                     for ii,il in enumerate(m):
#                         for jj,ic in enumerate(m):
#                             tmpBM += vM[il] * listKEle[iteM*sizeKel+ii*len(m)+jj] * vB[ic]
                    #KKtmp = finiteElem.rigidityDirect(ptsM,[m3D[iteM]],order,[1.0],0)
                    KKtmp = listKEle[iteM]
                    tmpBM = np.dot(vM[m],np.dot(KKtmp, vB[m]))
                    SensitivityMesBM[iteM] = tmpBM
                    #del KKtmp
                SensitivityMes[(minBM,maxBM)] = SensitivityMesBM
            
        
            minBN = min(injB,injN)
            maxBN = max(injB,injN)
            if ((minBN,maxBN) in SensitivityMes):
                SensitivityMesBN = SensitivityMes[(minBN,maxBN)]
            else:
                for iteM,m in enumerate(m3D):
                    #utils.update_progress((iteM+1.0)/len(m3D))
                    # tmpBN = 0.0
#                     for ii,il in enumerate(m):
#                         for jj,ic in enumerate(m):
#                             tmpBN += vN[il] * listKEle[iteM*sizeKel+ii*len(m)+jj] * vB[ic]
                    #KKtmp = finiteElem.rigidityDirect(ptsM,[m3D[iteM]],order,[1.0],0)
                    KKtmp = listKEle[iteM]
                    tmpBN = np.dot(vN[m],np.dot(KKtmp, vB[m]))
                    SensitivityMesBN[iteM] = tmpBN
                    #del KKtmp
                SensitivityMes[(minBN,maxBN)] = SensitivityMesBN
        
            
        indAM = len(nodesInj)*injA+injM
        indBM = len(nodesInj)*injB+injM
        indAN = len(nodesInj)*injA+injN
        indBN = len(nodesInj)*injB+injN
        
        #SensitivityMesResis[ind,:] = (SensitivityMes[indAM,:] - SensitivityMes[indBM,:] - SensitivityMes[indAN,:] + SensitivityMes[indBN,:]) / (dVhom)
        SensitivityMesResis[ind,:] = (SensitivityMesAM - SensitivityMesBM - SensitivityMesAN + SensitivityMesBN) / (dVhom[ind])
        
        if(useLog == 1):
            SensitivityMesResis[ind,:] = useEq*(SensitivityMesResis[ind,:] / resistivity) / (resistyComp[ind]) #* (-1) 
        else:
            SensitivityMesResis[ind,:] = (SensitivityMesResis[ind,:] / resistivity) 
        # We divide by resistyComp because we minimize the misfit between log(rho_a_exp) and log(rho_a_comp) (not "simply" rho_a_comp and rho_a_exp).
    del SensitivityMes    

    print("Filtering sensitivity matrix:")
    maxSensi = np.max(SensitivityMesResis)
    minSensi = np.min(SensitivityMesResis)
    
    maxAbs = max(np.abs(minSensi),np.abs(maxSensi))
    SensitivityMesResis = np.where(np.abs(SensitivityMesResis) < maxAbs*1e-8,0.0,SensitivityMesResis)
    
    #########################################################

    #########################################################
    print("Construction of the Sensitivity matrix:")            
    if(choiceMetho == "1"): # Gauss-Newton
        # matRegu[2] = matRegu[2] * epsReg / (nbIteDiffErr*nbIteDiffErr)
#         indL = []
#         indC = []
#         val = []
#         for i in range(len(seqABMN)):
#             utils.update_progress((i+1.0)/len(seqABMN))
#             indL.extend(i*np.ones(len(sigma)))
#             indC.extend(np.arange(len(sigma)))
#             val.extend(SensitivityMesResis[i,:])
#         indL.extend(matRegu[0])
#         indL = np.array(indL,dtype='int32')
#         indC.extend(matRegu[1])
#         indC = np.array(indC,dtype='int32')
#         val.extend(matRegu[2])
#
#         # Sparse matrix format:
#         SensitivityMesResisCsr = csr_matrix((val, (indL, indC)),shape=(len(sigma)+len(seqABMN),len(sigma)))
#
#         del indL,indC,val
        #SensitivityMesResis[len(seqABMN):,:] = matRegu * epsReg / (nbIteDiffErr*nbIteDiffErr) #/ (iteGN*iteGN)
        
        
        # CG with optimized matrix product to solve the normal equation.
        print("Computation of the correction term (leastsquare problem resolution).")
        SensitivityMesResisT = SensitivityMesResis.T
        #vTmp = (matReguCsr * ErrMes) # ErrResi
        # A^t b
        rhs = np.dot(SensitivityMesResisT,ErrResi) #+ (epsReg / (nbIteDiffErr*nbIteDiffErr)) * vTmp
        # x0 = 0
        r = -rhs
        w = r * 1.0
        sol = rhs*0.0
        residusIni = np.dot(rhs,rhs)
        residusCurrent = residusIni*1.0
        iteMaxCG = len(sigma)
        iteCG = -1
        print("CG:")
        while(residusCurrent >= 1e-10*residusIni and iteMaxCG > iteCG):
            iteCG += 1
            utils.update_progressWithError((iteCG+1.0)/iteMaxCG,residusCurrent/residusIni)
            
            # Restart : (pas sur que ce soit une bonne idee..)
            # if(iteCG % 2000 == 0):
 #                print('Restart CG: ',iteCG)
 #                w = r*1.0
                
            # Matrix produc with w:
            vTmp = (matReguCsr * w)
            S2w = (matReguCsr * vTmp)
            vTmp = np.dot(SensitivityMesResis,w)
            SMR2w = np.dot(SensitivityMesResisT,vTmp)
            
            Aw = SMR2w + ((epsReg / (nbIteDiffErr*nbIteDiffErr))**2) * S2w
            
            #Aw = np.dot(A,w)
            z = np.dot(r,w) /  np.dot(Aw,w)
            sol = sol - z*w
            r = r - z*Aw
            l = np.dot(r,Aw) / np.dot(Aw,w)
            w = r - l*w
            residusCurrent = np.dot(r,r)

            
        
        print("Ite : ",iteCG," / ",iteMaxCG) 
        print("Norme residu : ",residusCurrent/residusIni)
        
        deltaSigma = sol*1.0
        
        del vTmp, rhs, r, w, SensitivityMesResisT, sol
        
        resistivity = resistivity * np.exp(deltaSigma)
        if(useLog == 1):    
            resistivity[np.where(resistivity<0.01)]=0.01
            resistivity[np.where(resistivity>1e6)]=1e6
        
    if(choiceMetho == "2"): # Stepest descent (to be fully reviewed !)
        SensitivityMesResis[len(seqABMN):,:] = matRegu * epsReg #/ (np.sqrt(iteGN))
    #########################################################
        
    
    
    #########################################################
    # print("Computation of the correction term (leastsquare problem resolution).")
#     ## Gauss-Newton
#     if(choiceMetho == "1"):
#         # Lst square method
#         deltaSigma = scipy.sparse.linalg.lsmr(SensitivityMesResisCsr,ErrMes)[0]
#
#         resistivity = resistivity * np.exp(deltaSigma)
#         if(useLog == 1):
#             resistivity[np.where(resistivity<0.01)]=0.01
#             resistivity[np.where(resistivity>1e6)]=1e6
    
    ## Gradient descent with optimal step (not working at the moment...)
    if(choiceMetho == "2"):
        grad = np.dot(SensitivityMesResis.T,ErrMes)
        SenGrad = np.dot(SensitivityMesResis,grad)
        denom = np.dot(SenGrad,SenGrad)
        step = 1.0
        if (np.abs(denom) > 1e-10):
            step = np.dot(grad,grad) / denom
        else:
            print("Denom null")
        # Update of the conductivity:  
        sigma = sigma*np.exp(-grad * step)
        
    #del SensitivityMesResisCsr    
    #########################################################    
    
    #########################################################
    # Writing update:
    sigma = 1.0 / resistivity 
    mesh.writeCellVTK('ResVTK/conductivite'+str(iteGN)+'.vtk',sigma,ptsM,m3D)
    mesh.writeCellVTK('ResVTK/resistivity'+str(iteGN)+'.vtk',resistivity,ptsM,m3D)
    diffErr = np.abs(errL2 - errL2Old)
    
    if(diffErr < 1e-1 and errL2 < errL2Old):
        nbIteDiffErr = nbIteDiffErr*2.0
    if(errL2 > errL2Old):
        nbIteDiffErr = nbIteDiffErr/1.5
        resistivity = resistivityOld*1.0
        
    resistivityOld = resistivity*1.0     
    
    errL2Old = errL2
    #########################################################
    
#########################################################

print("End of the program.")



