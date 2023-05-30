#!/usr/bin/python3.4
# -*-coding:Utf-8 -*

# by A. Tonnoir

import numpy as np
import copy




#########################################################
## Read ERI file (must be improved !) :
def readERFile(nomFichier):
    fid = open(nomFichier,"r");

    contenu = fid.read();
    contenu = contenu.split("\n");
    

    seqABMN = []
    resExp = []
    
    it = 0
    ele = contenu[it]
    it = it + 1
    ele = contenu[it]
    spacing = float(ele)
    
    while (ele != "Type of measurement (0=app.resistivity,1=resistance)"):
        it = it + 1
        ele = contenu[it]
    
    it = it + 2
    ele = contenu[it]
    nbMes = int(ele)
    it = it + 1
    
    it = it + 1
    
    
    for i in range(nbMes):
        it = it + 1
        ele = contenu[it]
        res = ele.split(" ")
        xA = float(res[1])
        xB = float(res[3])
        xM = float(res[5])
        xN = float(res[7])
        resis = float(res[9])
        na = int(xA/spacing + spacing*0.05)
        nb = int(xB/spacing + spacing*0.05)
        nm = int(xM/spacing + spacing*0.05)
        nn = int(xN/spacing + spacing*0.05)
        if(nm == nn):
            print(xA,xB,xM,xN)
            print(na,nb,nm,nn)
            print(i)
            input()
        seqABMN.append([na,nb,nm,nn])
        resExp.append(resis)
    
    # print("Seq ABMN:")
    # print(seqABMN)
    # print("\n")
    # print("Resistivity Exp:")
    # print(resExp)
    
    
    fid.close()
    return [seqABMN,resExp]
#########################################################


#########################################################
## Read ERI file (must be improved !) :
def writeERFile(nomFichier,VV,VVhom,nodesInj,ptsM):
    nbInj = len(nodesInj)
    [x0,y0,z0] = ptsM[nodesInj[0]]
    [x1,y1,z1] = ptsM[nodesInj[1]]
    distElec = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
    fid = open(nomFichier,"w");
    
    fid.write("Numerical data\n")
    fid.write(str(distElec)+"\n")
    fid.write("11\n")
    fid.write("Type of measurement (0=app.resistivity,1=resistance)\n")
    fid.write("1\n")
    
    # Pas ideal, a revoir pour faire mieux ! Fonctionne tres mal...
    # a = 0
#     b = 1
#     tab = []
#     for m in range(2,nbInj):
#         for n in range(m+1,nbInj):
#             dVhom = (VVhom[nodesInj[m],a] - VVhom[nodesInj[m],b] - VVhom[nodesInj[n],a] + VVhom[nodesInj[n],b])
#             resistivity = (VV[nodesInj[m],a] - VV[nodesInj[m],b] - VV[nodesInj[n],a] + VV[nodesInj[n],b]) / dVhom
#
#             tab.append("4 "+str(distElec*a)+" 0.0 "+str(distElec*b)+" 0.0 "+str(distElec*m)+" 0.0 "+str(distElec*n)+" 0.0 "+str(resistivity)+"\n")
    
    # First test :
    #nbSeq = int(np.log2(nbInj))
    # tab = []
 #    step = 1
 #    #for seq in range(nbSeq):
 #    while(nbInj-3*step > 0):
 #        for a in range(nbInj-3*step):
 #
 #            b = a + step*3
 #            m = a + step
 #            n = a + step*2
 #            # for m in range(a+1,b-1):
 #            #     for n in range(m+1,b-1):
 #
 #            dVhom = (VVhom[nodesInj[m],a] - VVhom[nodesInj[m],b] - VVhom[nodesInj[n],a] + VVhom[nodesInj[n],b])
 #            resistivity = (VV[nodesInj[m],a] - VV[nodesInj[m],b] - VV[nodesInj[n],a] + VV[nodesInj[n],b]) / dVhom
 #
 #            tab.append("4 "+str(distElec*a)+" 0.0 "+str(distElec*b)+" 0.0 "+str(distElec*m)+" 0.0 "+str(distElec*n)+" 0.0 "+str(resistivity)+"\n")
 #
 #        step = step+1
        
    # New test :    
    tab = []
    step = 1
    nnn = 1
    while(nbInj-2*nnn-1 > 0):
        for a in range(nbInj-2*nnn-1):

            b = a + 1 + nnn*2
            m = a + nnn
            n = a + nnn+1
            # for m in range(a+1,b-1):
            #     for n in range(m+1,b-1):

            dVhom = (VVhom[nodesInj[m],a] - VVhom[nodesInj[m],b] - VVhom[nodesInj[n],a] + VVhom[nodesInj[n],b])
            resistivity = (VV[nodesInj[m],a] - VV[nodesInj[m],b] - VV[nodesInj[n],a] + VV[nodesInj[n],b]) #/ dVhom # A revoir !

            tab.append("4 "+str(distElec*a)+" 0.0 "+str(distElec*b)+" 0.0 "+str(distElec*m)+" 0.0 "+str(distElec*n)+" 0.0 "+str(resistivity)+"\n")

        nnn = nnn+1    
            
    fid.write(str( len(tab) )+"\n")
    fid.write("2\n")
    fid.write("0\n")
    
    for tt in tab:
        fid.write(tt)
    
            
    
    fid.close()
#########################################################











