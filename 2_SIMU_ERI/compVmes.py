#!/usr/bin/python3.4
# -*-coding:Utf-8 -*

# by A. Tonnoir

import numpy as np
from scipy import linalg
import copy

import matplotlib.pyplot as plt
#import maillage as mesh


#########################################################
# BC :
VmesSmall = np.genfromtxt('ComparaisonVNC/VmesExpIELarge.data', delimiter=" ")
VmesLarge = np.genfromtxt('ComparaisonVNC/VmesExpMBCLarge.data', delimiter=" ")

ErrV = 100*np.abs((VmesLarge-VmesSmall)/VmesLarge)
(nl,nc) = ErrV.shape
errInf = 0
errL2 = 0
for i in range(nl):
    for j in range(nc):
        if(i != j):
            errL2 += ErrV[i,j]**2
            if(errInf < ErrV[i,j]):
                errInf = ErrV[i,j]

print('Err Inf : ',errInf)
print('Err L2 : ',np.sqrt(errL2/(64*63.)))

plt.show()
c = plt.pcolor(100*np.abs((VmesLarge-VmesSmall)/VmesLarge),cmap='jet',vmin=0.0,vmax=100)
plt.colorbar(c)
#plt.savefig('DiffSmallLargeMBC.pdf')
plt.show()


#
#
# Vmes = np.genfromtxt('VmesExpIE.data',delimiter=" ")
# VmesEx = np.genfromtxt('VmesExa.data',delimiter=" ")
#
# ErrV = np.abs(Vmes-VmesEx)/VmesEx
# (nl,nc) = ErrV.shape
#
# errInf = 0
# errL2 = 0
# for i in range(nl):
#     for j in range(nc):
#         if(i != j):
#             errL2 += ErrV[i,j]**2
#             if(errInf < ErrV[i,j]):
#                 errInf = ErrV[i,j]
#
# print('Err Inf : ',errInf)
# print('Err L2 : ',np.sqrt(errL2/(64*63.)))             


# Rhoa = Vmes / VmesEx
#
# ll = []
# for i in range(32):
#     print(i,64-i,Rhoa[i,64-i])
#     ll.append(Rhoa[i,64-i])
#
# ll.reverse()
# plt.plot(ll,'x-')
# plt.show()
#########################################################





print("Fin du programme")
