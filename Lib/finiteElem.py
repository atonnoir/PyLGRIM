import numpy as np
from scipy.sparse import lil_matrix, coo_matrix, csc_matrix, csr_matrix, identity

import utils


#########################################################
def quadrature(f): # utile uniquement pour la fonction ci dessous pour l'ordre 2, approche l'intégrale de f de R3 dans R sur le tétraèdre (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    # Formule a 4 points exactes sur P2
    a = 0.5854101966249680
    b = 0.1381966011250110
    
    listPtsQuad = [np.array([a,b,b]),np.array([b,a,b]),np.array([b,b,a]),np.array([b,b,b])]
    listPdsQuad = np.array([0.25,0.25,0.25,0.25])
    
    # listPtsQuad = [np.array([0.25,0.25,0.25])]
#     listPdsQuad = [1.0/6]
    
    I = 0
    for j,pts in enumerate(listPtsQuad):
        pds = listPdsQuad[j]
        I += f(pts[0],pts[1],pts[2]) * pds 
        
    del listPtsQuad,listPdsQuad
    return I
#########################################################    
    
#########################################################   
def rigidityHomogenSplit(ptsM,m3D,order):
    # Order 1
    if (order == 1):
        GGref = np.array([[-1.0,-1.0,-1.0], [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
        listKEle = []
        for ite,m in enumerate(m3D):
            utils.update_progress((ite+1.0)/len(m3D) )
            
            pt1 = ptsM[m[0]]
            pt2 = ptsM[m[1]]
            pt3 = ptsM[m[2]]
            pt4 = ptsM[m[3]]
            
            T2 = pt2 - pt1
            T3 = pt3 - pt1
            T4 = pt4 - pt1
            
            TT = np.array([ [T2[0],T3[0],T4[0]], [T2[1],T3[1],T4[1]], [T2[2],T3[2],T4[2]] ])
            vol = np.abs( np.linalg.det(TT)/6.0 )
            
            invTT = np.linalg.inv(TT)
            GG = np.dot(invTT.T,GGref.T)
            Kele = np.dot(GG.T,GG) * vol
            
            # indL = [m[0],m[0],m[0],m[0],m[1],m[1],m[1],m[1],m[2],m[2],m[2],m[2],m[3],m[3],m[3],m[3]]
            # indC = [m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3]]
            # listKEle.append(csc_matrix((Kele.flatten(),(indL,indC)),shape=(len(ptsM),len(ptsM)) ) )
            
            listKEle.append(Kele*1.0)
            
            del pt1,pt2,pt3,pt4,T2,T3,T4,TT,invTT,GG#,indL,indC,Kele
        del GGref    
        return np.array(listKEle)
    
    # Order 2
    if (order == 2):
        listKEle = []
        lenm3D = len(m3D[:,0])
        # Gradient fonctions bases ? A valider.
        f1 = lambda x,y,z : (-3 + 4*(x + y + z))*np.ones(3) # Ok
        f2 = lambda x,y,z : np.array([4*x -1,0,0]) # ok 
        f3 = lambda x,y,z : np.array([0,4*y -1,0]) # ok
        f4 = lambda x,y,z : np.array([0,0,4*z -1]) # ok
        #f5 = lambda x,y,z : np.array([1-2*x-y-z,-x,-x]) # Pas sur...
        f5 = lambda x,y,z : np.array([4.0 -4.0*z -4.0*y-8.0*x,-4*x,-4*x]) # Correction.
        
        #f6 = lambda x,y,z : np.array([-y,1-2*y-x-z,-y]) # Pas sur...
        f6 = lambda x,y,z : np.array([4*y,4*x,0]) # Correction.
        
        #f7 = lambda x,y,z : np.array([-z,-z,1-2*z-x-y]) # Pas sur...
        f7 = lambda x,y,z : np.array([-4*y,4-4*z-8*y-4*x,-4*y]) # Correction
        
        #f8 = lambda x,y,z : 4*np.array([y,x,0]) # Pas sur...
        f8 = lambda x,y,z : np.array([-4*z,-4*z,4-8*z-4*y-4*x]) # Correction
        
        #f9 = lambda x,y,z : 4*np.array([z,0,x]) # Pas sur...
        f9 = lambda x,y,z : np.array([0,4*z,4*y]) # Correction
        
        #f10 = lambda x,y,z : 4*np.array([0,z,y]) # Pas sur...
        f10 = lambda x,y,z :np.array([4*z,0,4*x]) # Correction
        
        
        listMatriceRef = []
        ff = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
        for lk in range(3):
            for ck in range(3):
                matTr = np.zeros((3,3))
                matTr[lk,ck] = 1.0
                
                matRef = np.zeros((10,10))
                for i in range(10):
                    for j in range(10):
                        matRef[i,j] = (quadrature(lambda x,y,z : np.dot(ff[i](x,y,z),np.dot(matTr,ff[j](x,y,z)))))
                listMatriceRef.append(matRef)   
                
                del matTr
                        
        
        for ite,m in enumerate(m3D):
            utils.update_progress((ite+1.0)/len(m3D))
            #print(str(ite) + "/" + str(lenm3D))
            pt1 = ptsM[m[0]]
            pt2 = ptsM[m[1]]
            pt3 = ptsM[m[2]]
            pt4 = ptsM[m[3]]
                        
            T2 = pt2 - pt1
            T3 = pt3 - pt1
            T4 = pt4 - pt1
            TT = np.array([ [T2[0],T3[0],T4[0]], [T2[1],T3[1],T4[1]], [T2[2],T3[2],T4[2]] ])
            vol = np.abs( np.linalg.det(TT)/6.0 )
            invTT = np.linalg.inv(TT)
            invTTTT = np.dot(invTT,invTT.T)
            
            cpt = 0
            resMat = np.zeros((10,10))
            for lk in range(3):
                for ck in range(3):
                    resMat += invTTTT[lk,ck] * listMatriceRef[cpt] * vol
                    cpt += 1
                    
            # data = resMat.flatten()
#             indL = []
#             indC = []
#             for i in range(10):
#                 for j in range(10):
#                     #data.append(vol*quadrature(lambda x,y,z : np.dot(ff[i](x,y,z),np.dot(invTTTT,ff[j](x,y,z)))))
#                     indL.append(m[i])
#                     indC.append(m[j]) 
                               
            
            # interpolation
            # for i in range(10):
#                 for j in range(10):
#                     data.append(vol*quadrature(lambda x,y,z : np.dot(ff[i](x,y,z),np.dot(invTTTT,ff[j](x,y,z)))))
#                     indL.append(m[i])
#                     indC.append(m[j])
            #listKEle.append(csc_matrix((data,(indL,indC)),shape=(len(ptsM),len(ptsM)) ) )
            listKEle.append(resMat*1.0)
            
            del pt1,pt2,pt3,pt4,T2,T3,T4,TT,invTT,invTTTT,resMat#,indL,indC,data
            
        del f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
        del ff,listMatriceRef
        return np.array(listKEle)
#########################################################  


#########################################################   
def rigidityHomogenSplitNew(ptsM,m3D,order):
    # Order 1
    if (order == 1):
        GGref = np.array([[-1.0,-1.0,-1.0], [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
        listKEle = np.zeros(len(m3D)*16)
        for ite,m in enumerate(m3D):
            utils.update_progress((ite+1.0)/len(m3D) )
            
            pt1 = ptsM[m[0]]
            pt2 = ptsM[m[1]]
            pt3 = ptsM[m[2]]
            pt4 = ptsM[m[3]]
            
            T2 = pt2 - pt1
            T3 = pt3 - pt1
            T4 = pt4 - pt1
            
            TT = np.array([ [T2[0],T3[0],T4[0]], [T2[1],T3[1],T4[1]], [T2[2],T3[2],T4[2]] ])
            vol = np.abs( np.linalg.det(TT)/6.0 )
            
            invTT = np.linalg.inv(TT)
            GG = np.dot(invTT.T,GGref.T)
            Kele = np.dot(GG.T,GG) * vol
            
            
            
            listKEle[ite*16:(ite+1)*16] = Kele.flatten()
            
            # indL = [m[0],m[0],m[0],m[0],m[1],m[1],m[1],m[1],m[2],m[2],m[2],m[2],m[3],m[3],m[3],m[3]]
            # indC = [m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3]]
            # listKEle.append(csc_matrix((Kele.flatten(),(indL,indC)),shape=(len(ptsM),len(ptsM)) ) )
            
            del pt1,pt2,pt3,pt4,T2,T3,T4,TT,invTT,GG#,Kele,indL,indC
        del GGref    
        return listKEle
    
    # Order 2
    if (order == 2):
        listKEle = np.zeros(len(m3D)*100)
        lenm3D = len(m3D[:,0])
        # Gradient fonctions bases ? A valider.
        f1 = lambda x,y,z : (-3 + 4*(x + y + z))*np.ones(3) # Ok
        f2 = lambda x,y,z : np.array([4*x -1,0,0]) # ok 
        f3 = lambda x,y,z : np.array([0,4*y -1,0]) # ok
        f4 = lambda x,y,z : np.array([0,0,4*z -1]) # ok
        #f5 = lambda x,y,z : np.array([1-2*x-y-z,-x,-x]) # Pas sur...
        f5 = lambda x,y,z : np.array([4.0 -4.0*z -4.0*y-8.0*x,-4*x,-4*x]) # Correction.
        
        #f6 = lambda x,y,z : np.array([-y,1-2*y-x-z,-y]) # Pas sur...
        f6 = lambda x,y,z : np.array([4*y,4*x,0]) # Correction.
        
        #f7 = lambda x,y,z : np.array([-z,-z,1-2*z-x-y]) # Pas sur...
        f7 = lambda x,y,z : np.array([-4*y,4-4*z-8*y-4*x,-4*y]) # Correction
        
        #f8 = lambda x,y,z : 4*np.array([y,x,0]) # Pas sur...
        f8 = lambda x,y,z : np.array([-4*z,-4*z,4-8*z-4*y-4*x]) # Correction
        
        #f9 = lambda x,y,z : 4*np.array([z,0,x]) # Pas sur...
        f9 = lambda x,y,z : np.array([0,4*z,4*y]) # Correction
        
        #f10 = lambda x,y,z : 4*np.array([0,z,y]) # Pas sur...
        f10 = lambda x,y,z :np.array([4*z,0,4*x]) # Correction
        
        
        listMatriceRef = []
        ff = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
        for lk in range(3):
            for ck in range(3):
                matTr = np.zeros((3,3))
                matTr[lk,ck] = 1.0
                
                matRef = np.zeros((10,10))
                for i in range(10):
                    for j in range(10):
                        matRef[i,j] = (quadrature(lambda x,y,z : np.dot(ff[i](x,y,z),np.dot(matTr,ff[j](x,y,z)))))
                listMatriceRef.append(matRef)   
                
                del matTr
                        
        
        for ite,m in enumerate(m3D):
            utils.update_progress((ite+1.0)/len(m3D))
            #print(str(ite) + "/" + str(lenm3D))
            pt1 = ptsM[m[0]]
            pt2 = ptsM[m[1]]
            pt3 = ptsM[m[2]]
            pt4 = ptsM[m[3]]
                        
            T2 = pt2 - pt1
            T3 = pt3 - pt1
            T4 = pt4 - pt1
            TT = np.array([ [T2[0],T3[0],T4[0]], [T2[1],T3[1],T4[1]], [T2[2],T3[2],T4[2]] ])
            vol = np.abs( np.linalg.det(TT)/6.0 )
            invTT = np.linalg.inv(TT)
            invTTTT = np.dot(invTT,invTT.T)
            
            cpt = 0
            resMat = np.zeros((10,10))
            for lk in range(3):
                for ck in range(3):
                    resMat += invTTTT[lk,ck] * listMatriceRef[cpt] * vol
                    cpt += 1
                    
            data = resMat.flatten()
            listKEle[ite*100:(ite+1)*100] = data
            
            del pt1,pt2,pt3,pt4,T2,T3,T4,TT,invTT,invTTTT,resMat#,indL,indC,data
            
        del f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
        del ff,listMatriceRef
        return listKEle
#########################################################           
        
    
#########################################################       
def buildRegu(ptsM,m3D,choix):
    # Nb mailles :
    nm = len(m3D)
    # Tichonov :
    if(choix == 'Ticho'):
        return np.eye(nm)
        
    if(choix == 'Grad'):
        
        indL = []
        indC = []
        val = []
        
        #res=np.zeros((nm,nm))
        m3DTronc = m3D[:,0:4]
        for i in range(0, nm):
            utils.update_progress(((i+1.0)/nm))
            
            v1 = (m3DTronc == m3DTronc[i][0]) + 0
            v2 = (m3DTronc == m3DTronc[i][1]) + 0
            v3 = (m3DTronc == m3DTronc[i][2]) + 0
            v4 = (m3DTronc == m3DTronc[i][3]) + 0
            
            v=np.sum(v1+v2+v3+v4,1)
            
            Voisin= np.where(v == 3)[0]
            nbVoisin = len(Voisin)
            
            indL.extend(i*np.ones(nbVoisin+1))
            indC.extend([i])
            indC.extend(Voisin)
            val.extend([1.0])
            val.extend(((-1.0)*np.ones(nbVoisin))/nbVoisin)
            
            del v1,v2,v3,v4,v
            del Voisin
            
        del m3DTronc
        return [np.array(indL),np.array(indC),np.array(val)]
        
    if(choix == 'GradStrat'):
        indL = []
        indC = []
        val = []
        
        #res=np.zeros((nm,nm))
        m3DTronc = m3D[:,0:4]
        listBary = []
        for i in range(0, nm):
            i1 = m3DTronc[i,0]
            i2 = m3DTronc[i,1]
            i3 = m3DTronc[i,2]
            i4 = m3DTronc[i,3]
            
            bary = 0.25*(ptsM[i1]+ptsM[i2]+ptsM[i3]+ptsM[i4])
            listBary.append(bary)
            

        for i in range(0, nm):
            utils.update_progress(((i+1.0)/nm))
            
            v1 = (m3DTronc == m3DTronc[i][0]) + 0
            v2 = (m3DTronc == m3DTronc[i][1]) + 0
            v3 = (m3DTronc == m3DTronc[i][2]) + 0
            v4 = (m3DTronc == m3DTronc[i][3]) + 0
            
            v=np.sum(v1+v2+v3+v4,1)
            
            Voisin= np.where(v == 3)[0]
            nbVoisin = len(Voisin)
            
            indL.extend(i*np.ones(nbVoisin+1))
            indC.extend([i])    
            val.extend([1.0])
            
            indC.extend(Voisin)
            
            baryCM = listBary[i]
            summ = 0
            listAngle = []
            for vv in Voisin:
                baryV = listBary[vv]
                diffBary = baryCM - baryV
                ndiffBary = np.sqrt(np.dot(diffBary,diffBary))
                # With [0,0,1]
                #prodVec = np.array([diffBary[2],-difBary[1],0])
                #nprodVec = np.sqrt(np.dot(prodVec,prodVec))
                #nprodVec = np.sqrt(diffBary[2]*diffBary[2] + diffBary[1]*diffBary[1])
                depXY = np.sqrt(diffBary[0]*diffBary[0]+diffBary[1]*diffBary[1])
                coeffAngle = depXY / ndiffBary
                # Formula 1:
                #coeffAngle = (coeffAngle+0.0)/1.0
                # Formula 2:
                coeffAngle = coeffAngle**4
                #coeffAngle = 1.0 # normalement comme l'autre terme de regu...
                summ += coeffAngle
                listAngle.append(coeffAngle)
                
                
            
            for j,vv in enumerate(Voisin):   
                coeffAngle = listAngle[j] 
                val.extend([-coeffAngle/summ])
            del listAngle    
                
            
            # val.extend([1.0])
            # val.extend(((-1.0)*np.ones(nbVoisin))/nbVoisin)
            
            del v1,v2,v3,v4,v
            del Voisin
            
        del m3DTronc,listBary
        return [np.array(indL),np.array(indC),np.array(val)]
    
#########################################################       
def buildMatrix(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,sigma,listKEle,order,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D):
    
    KKindL = []
    KKindC = []
    KKdata = []
    # Rigidity part:
    #KK = np.zeros( (len(ptsM),len(ptsM) ) )
    nm3D = len(m3D)
    for ite,m in enumerate(m3D):
        utils.update_progress(((ite+1.0)/nm3D))
        
        Kele = listKEle[ite]
        
        if(order == 1):
            for i in range(4):
                for j in range(4):
                    #KK[m[i],m[j]] += Kele[i,j] *sigma[ite]    
                    #KK[m[i],m[j]] += Kele[m[i],m[j]] *sigma[ite]
                    KKindL.append(m[i])
                    KKindC.append(m[j])
                    #KKdata.append(Kele[m[i],m[j]]*sigma[ite])
                    KKdata.append(Kele[i,j]*sigma[ite])
                        
        if(order == 2):
            for i in range(10):
                for j in range(10):
                    #KK[m[i],m[j]] += Kele[i,j] *sigma[ite]    
                    #KK[m[i],m[j]] += Kele[m[i],m[j]] *sigma[ite]
                    KKindL.append(m[i])
                    KKindC.append(m[j])
                    #KKdata.append(Kele[m[i],m[j]]*sigma[ite])
                    KKdata.append(Kele[i,j]*sigma[ite])
                    
        
                    
    
                    
    # ordre 1
    GGref2D = np.array([[-1.0,-1.0], [1.0,0.0], [0.0,1.0] ])
    lMatRefGrad =  []
    MatRefMass = np.array([ [1.0/6.0, 1.0/12.0, 1.0/12.0], [1.0/12.0, 1.0/6.0, 1.0/12.0], [1.0/12.0, 1.0/12.0, 1.0/6.0] ])
    if(order == 1):
        for i in range(2):
            for j in range(2):
                matTr = np.zeros((2,2))
                matTr[i,j] = 1.0
                matRes = np.dot(GGref2D,np.dot(matTr,GGref2D.T))
                lMatRefGrad.append(matRes)
                
                del matTr
                
        
    # Order 2           
    if(order == 2):
        lptsQuad = [np.array([0.659027622374092,0.231933368553031]), np.array([0.659027622374092,0.109039009072877]), np.array([0.231933368553031,0.659027622374092]), np.array([0.231933368553031,0.109039009072877]), np.array([0.109039009072877,0.659027622374092]), np.array([0.109039009072877,0.231933368553031])]
        lpdsQuad = [1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6]
        
        # lptsQuad = [np.array([0.5,0]),np.array([0.0,0.5]),np.array([0.5,0.5])]
#         lpdsQuad = [1.0/3,1.0/3,1.0/3]
        
        
        gf1 = lambda x,y : np.array([4*(x+y)-3,4*(x+y)-3])
        gf2 = lambda x,y : np.array([4*x-1,0])
        gf3 = lambda x,y : np.array([0,4*y-1])
        gf4 = lambda x,y : np.array([4-8*x-4*y,-4*x])
        gf5 = lambda x,y : np.array([4*y,4*x])
        gf6 = lambda x,y : np.array([-4*y,4-8*y-4*x])
        
        f1 = lambda x,y : 2*(x+y-0.5)*(x+y-1)
        f2 = lambda x,y : 2*x*x - x
        f3 = lambda x,y : 2*y*y-y
        f4 = lambda x,y : 4*x*(1-x)-4*x*y
        f5 = lambda x,y : 4*x*y
        f6 = lambda x,y : 4*y*(1-y)-4*x*y
        
        gf = [gf1,gf2,gf3,gf4,gf5,gf6]
        ff = [f1,f2,f3,f4,f5,f6]
        for i in range(2):
            for j in range(2):
                matTr = np.zeros((2,2))
                matTr[i,j] = 1.0
                matRes = np.zeros((6,6))
                for l in range(6):
                    for c in range(6):
                        for k,ptq in enumerate(lptsQuad):
                            matRes[l,c] += np.dot(gf[l](ptq[0],ptq[1]),np.dot(matTr,gf[c](ptq[0],ptq[1]))) * lpdsQuad[k]
                            
                lMatRefGrad.append(matRes)            
        
        matRes = np.zeros((6,6))
        for l in range(6):
            for c in range(6):
                for k,ptq in enumerate(lptsQuad):
                    matRes[l,c] += ff[l](ptq[0],ptq[1]) * ff[c](ptq[0],ptq[1]) * lpdsQuad[k]                    
            
        MatRefMass = matRes              
        
                              
        
    if(choiceBC == "1"):
        # Taking into account the BC:
        # coeff1D = 0.5785273170649641 * Lmin
        # dcoeff1D = 0.5106484442933402 / Lmin
        #Lmin = 1.0
        coeff1D = 0.53 * Lmin
        dcoeff1D = 0.53 / Lmin
        coeff2D = coeff1D*coeff1D
        dcoeff2D = dcoeff1D * dcoeff1D * Lmin * Lmin
        dcoeff3D = dcoeff1D*dcoeff1D*dcoeff1D * Lmin * Lmin * Lmin * Lmin
        
        
        ## Test a revoir avec ABC o1
        # coeff1D = 0.5
 #        dcoeff1D = 0.5
 #        coeff2D = 0.25
 #        dcoeff2D = 0.5 # Ce coefficient est à revoir !!!!
 #        dcoeff3D = 3.0/8.0
 #
        # x0 = 1.0
        # a0 = 2.0
        # coeff1D = a0/(x0+a0)
        # dcoeff1D = x0 / (x0+a0)
        #
        # coeff2D = coeff1D * coeff1D
        # dcoeff2D = coeff1D * 2.0*dcoeff1D
        #
        # dcoeff3D = 3.0*dcoeff1D*coeff2D
        
        
        # 2D artificial boundaries:
        for ite,m in enumerate(m2D):
            if(mt2D[ite] == "1"): 
                n1 = m[0]
                n2 = m[1]
                n3 = m[2]
        
                sig = sigma[linkSig2Dto3D[ite]]
        
                pt1 = ptsM[n1]
                pt2 = ptsM[n2]
                pt3 = ptsM[n3]
    
                T2 = pt2 - pt1
                T3 = pt3 - pt1
                TT = []
        
                if( (T2[0] == T3[0]) and (T2[0] == 0) ):              
                    TT = np.array([ [T2[1],T3[1]], [T2[2],T3[2]] ])
                if( (T2[1] == T3[1]) and (T2[1] == 0) ):
                    TT = np.array([ [T2[0],T3[0]], [T2[2],T3[2]] ])
                if( (T2[2] == T3[2]) and (T2[2] == 0) ):              
                    TT = np.array([ [T2[0],T3[0]], [T2[1],T3[1]] ])
            
                vol = np.abs( np.linalg.det(TT) * 0.5 )
        
                invTT = np.linalg.inv(TT)
                invTTT = np.dot(invTT,invTT.T)
                
                Kele = MatRefMass*0.0
                Mele = MatRefMass*vol
                
                cpt = 0
                for i in range(2):
                    for j in range(2):
                        Kele += invTTT[i,j] * lMatRefGrad[cpt] * vol
                        cpt += 1
                        
                if (order == 1):        
                    for i in range(3):
                        for j in range(3):
                            #KK[m[i],m[j]] += (Kele[i,j] * coeff1D  + Mele[i,j] * dcoeff1D) * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append((Kele[i,j] * coeff1D + Mele[i,j]*dcoeff1D)*sig)
                if(order == 2):
                    for i in range(6):
                        for j in range(6):
                            #KK[m[i],m[j]] += (Kele[i,j] * coeff1D  + Mele[i,j] * dcoeff1D) * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append((Kele[i,j] * coeff1D + Mele[i,j]*dcoeff1D)*sig)
                            
                # GG = np.dot(invTT.T,GGref2D.T)
#                 Kele = np.dot(GG.T,GG)
#
#                 Mele = np.array([ [vol/6.0, vol/12.0, vol/12.0], [vol/12.0, vol/6.0, vol/12.0], [vol/12.0, vol/12.0, vol/6.0] ])
#
#                 for i in range(3):
#                     for j in range(3):
#                         # Completer ICI (q. 3.c) ) :
#                         KK[m[i],m[j]] += (Kele[i,j] * coeff1D * vol  + Mele[i,j] * dcoeff1D) * sig
        

        
  
        # 1D artificial boundaries:
        MatRefMass = np.array([ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ])
        MatRefGrad = np.array([ [1.0, -1.0], [-1.0, 1.0] ])
        
        if(order == 2):
            MatRefMass = np.array([ [1.0/6.0, -1.0/30,2.0/30.0], [-1.0/30,1.0/6.0, 2.0/30], [2.0/30, 2.0/30,0.5+1.0/30 ]])
            MatRefGrad = np.array([ [2+1.0/3, 1.0/3,-2.0-2.0/3], [1.0/3,2+1.0/3.0, -2.0-2.0/3], [-2.0-2.0/3, -2.0-2.0/3, 5 + 1.0/3] ])
        
        for ite,m in enumerate(m1D):
            if(mt1D[ite] == "2"):
                n1 = m[0]
                n2 = m[1]
                pt1 = ptsM[n1]
                pt2 = ptsM[n2]

                sig = sigma[linkSig1Dto3D[ite]]

                T2 = pt2 - pt1
                vol = np.sqrt(np.dot(T2,T2))

                Kele = MatRefGrad / vol #np.array([[1.0/vol,-1.0/vol],[-1.0/vol,1.0/vol]])
                Mele = MatRefMass * vol #np.array([[vol/3.0, vol/6.0],[vol/6.0, vol/3.0]])
                
                if(order == 1):
                    for i in range(2):
                        for j in range(2):
                            #KK[m[i],m[j]] += (Kele[i,j] * coeff2D + Mele[i,j] * dcoeff2D) * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append((Kele[i,j] * coeff2D + Mele[i,j]*dcoeff2D)*sig)
                if(order == 2):
                    for i in range(3):
                        for j in range(3):
                            #KK[m[i],m[j]] += (Kele[i,j] * coeff2D + Mele[i,j] * dcoeff2D) * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append((Kele[i,j] * coeff2D + Mele[i,j]*dcoeff2D)*sig)

        # 0D (corner) articificial boundaries:
        for ite,m in enumerate(m0D):
            if(mt0D[ite] == "3"):
                sig = sigma[linkSig0Dto3D[ite]]
                
                #KK[m,m] += dcoeff3D*sig
                KKindL.append(m)
                KKindC.append(m)
                KKdata.append(dcoeff3D*sig)
                
                
                
                
    if(choiceBC == "2"): # Mixte BC
    
        GGref2D = np.array([[-1.0,-1.0], [1.0,0.0], [0.0,1.0] ])
        for ite,m in enumerate(m2D):
            if(mt2D[ite] == "1"): 
                n1 = m[0]
                n2 = m[1]
                n3 = m[2]
                
                sig = sigma[linkSig2Dto3D[ite]]
                
                pt1 = ptsM[n1]
                pt2 = ptsM[n2]
                pt3 = ptsM[n3]
    
                T2 = pt2 - pt1
                T3 = pt3 - pt1
                TT = []
        
                normal = np.array([1.0,0.0,0.0])
                if( (T2[0] == T3[0]) and (T2[0] == 0) ):              
                    TT = np.array([ [T2[1],T3[1]], [T2[2],T3[2]] ])
                if( (T2[1] == T3[1]) and (T2[1] == 0) ):
                    TT = np.array([ [T2[0],T3[0]], [T2[2],T3[2]] ])
                    normal = np.array([0.0,1.0,0.0])
                if( (T2[2] == T3[2]) and (T2[2] == 0) ):              
                    TT = np.array([ [T2[0],T3[0]], [T2[1],T3[1]] ])
                    normal = np.array([0.0,0.0,1.0])
            
                vol = np.abs( np.linalg.det(TT) * 0.5 )
        
                invTT = np.linalg.inv(TT)
                invTTT = np.dot(invTT,invTT.T)

                Bary = (pt1 + pt2 + pt3)/3.0
                rr = Bary - posM

                alpha = np.abs(np.dot(normal,rr)) / np.dot(rr,rr)

                Mele = MatRefMass*vol


                if (order == 1):
                    for i in range(3):
                        for j in range(3):
                            #KK[m[i],m[j]] += Mele[i,j] * alpha * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append(Mele[i,j] * alpha *sig)
                            
                if(order == 2):
                    for i in range(6):
                        for j in range(6):
                            #KK[m[i],m[j]] += Mele[i,j] * alpha * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append(Mele[i,j] * alpha *sig)
                
                
                
                # GG = np.dot(invTT.T,GGref2D.T)
#                 Kele = np.dot(GG.T,GG)
#
#                 Mele = np.array([ [vol/6.0, vol/12.0, vol/12.0], [vol/12.0, vol/6.0, vol/12.0], [vol/12.0, vol/12.0, vol/6.0] ])
#
#                 Bary = (pt1 + pt2 + pt3)/3.0
#                 rr = Bary - posM
#
#                 alpha = np.abs(np.dot(normal,rr)) / np.dot(rr,rr)
#
#                 for i in range(3):
#                     for j in range(3):
#                         # Completer ICI (q. 3.c) ) :
#                         KK[m[i],m[j]] += Mele[i,j] * alpha
                    
                    
    if(choiceBC == "3"): # Dirichlet BC
        for ite,m in enumerate(m2D):
            if(mt2D[ite] == "1"): 
                n1 = m[0]
                n2 = m[1]
                n3 = m[2]
        
                tmptab = []
                for i in m:
                    tmptab.append(KK[i,i])
                for i in m:
                    KK[i,:] = KK[i,:]*0.0
                    KK[:,i] = KK[:,i]*0.0   
                for j,i in enumerate(m):
                    KK[i,i] = tmptab[j]     
                     
                # tmpn1 = KK[n1,n1]
#                 tmpn2 = KK[n2,n2]
#                 tmpn3 = KK[n3,n3]
#
#                 KK[n1,:] = KK[n1,:]*0.0
#                 KK[:,n1] = KK[:,n1]*0.0
#                 KK[n2,:] = KK[n2,:]*0.0
#                 KK[:,n2] = KK[:,n2]*0.0
#                 KK[n3,:] = KK[n3,:]*0.0
#                 KK[:,n3] = KK[:,n3]*0.0
#
#                 KK[n1,n1] = tmpn1
#                 KK[n2,n2] = tmpn2
#                 KK[n3,n3] = tmpn3        
    
    KK = csc_matrix((KKdata,(KKindL,KKindC)))        
    return KK     
#########################################################



#########################################################   
def rigidityDirect(ptsM,m3D,order,sigma,affich):
    
    KK = lil_matrix((len(ptsM), len(ptsM)))
    # Order 1
    if (order == 1):
        GGref = np.array([[-1.0,-1.0,-1.0], [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
        for ite,m in enumerate(m3D):
            if(affich == 1):
                utils.update_progress((ite+1.0)/len(m3D) )
            
            pt1 = ptsM[m[0]]
            pt2 = ptsM[m[1]]
            pt3 = ptsM[m[2]]
            pt4 = ptsM[m[3]]
            
            T2 = pt2 - pt1
            T3 = pt3 - pt1
            T4 = pt4 - pt1
            
            TT = np.array([ [T2[0],T3[0],T4[0]], [T2[1],T3[1],T4[1]], [T2[2],T3[2],T4[2]] ])
            vol = np.abs( np.linalg.det(TT)/6.0 )
            
            invTT = np.linalg.inv(TT)
            GG = np.dot(invTT.T,GGref.T)
            Kele = np.dot(GG.T,GG) * vol
            
            
            for i in range(4):
                for j in range(4):
                    KK[m[i],m[j]] += Kele[i,j] * sigma[ite]
            
            # indL = [m[0],m[0],m[0],m[0],m[1],m[1],m[1],m[1],m[2],m[2],m[2],m[2],m[3],m[3],m[3],m[3]]
            # indC = [m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3],m[0],m[1],m[2],m[3]]
            # listKEle.append(csc_matrix((Kele.flatten(),(indL,indC)),shape=(len(ptsM),len(ptsM)) ) )
            
            del pt1,pt2,pt3,pt4,T2,T3,T4,TT,invTT,GG,Kele
        del GGref    
        #return listKEle
        return KK.tocsc()
    
    # Order 2
    if (order == 2):
        lenm3D = len(m3D[:,0])
        # Gradient fonctions bases ? A valider.
        f1 = lambda x,y,z : (-3 + 4*(x + y + z))*np.ones(3) # Ok
        f2 = lambda x,y,z : np.array([4*x -1,0,0]) # ok 
        f3 = lambda x,y,z : np.array([0,4*y -1,0]) # ok
        f4 = lambda x,y,z : np.array([0,0,4*z -1]) # ok
        #f5 = lambda x,y,z : np.array([1-2*x-y-z,-x,-x]) # Pas sur...
        f5 = lambda x,y,z : np.array([4.0 -4.0*z -4.0*y-8.0*x,-4*x,-4*x]) # Correction.
        
        #f6 = lambda x,y,z : np.array([-y,1-2*y-x-z,-y]) # Pas sur...
        f6 = lambda x,y,z : np.array([4*y,4*x,0]) # Correction.
        
        #f7 = lambda x,y,z : np.array([-z,-z,1-2*z-x-y]) # Pas sur...
        f7 = lambda x,y,z : np.array([-4*y,4-4*z-8*y-4*x,-4*y]) # Correction
        
        #f8 = lambda x,y,z : 4*np.array([y,x,0]) # Pas sur...
        f8 = lambda x,y,z : np.array([-4*z,-4*z,4-8*z-4*y-4*x]) # Correction
        
        #f9 = lambda x,y,z : 4*np.array([z,0,x]) # Pas sur...
        f9 = lambda x,y,z : np.array([0,4*z,4*y]) # Correction
        
        #f10 = lambda x,y,z : 4*np.array([0,z,y]) # Pas sur...
        f10 = lambda x,y,z :np.array([4*z,0,4*x]) # Correction
        
        
        listMatriceRef = []
        ff = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
        for lk in range(3):
            for ck in range(3):
                matTr = np.zeros((3,3))
                matTr[lk,ck] = 1.0
                
                matRef = np.zeros((10,10))
                for i in range(10):
                    for j in range(10):
                        matRef[i,j] = (quadrature(lambda x,y,z : np.dot(ff[i](x,y,z),np.dot(matTr,ff[j](x,y,z)))))
                listMatriceRef.append(matRef)   
                
                del matTr
                        
        
        for ite,m in enumerate(m3D):
            if(affich == 1):
                utils.update_progress((ite+1.0)/len(m3D))
            #print(str(ite) + "/" + str(lenm3D))
            pt1 = ptsM[m[0]]
            pt2 = ptsM[m[1]]
            pt3 = ptsM[m[2]]
            pt4 = ptsM[m[3]]
                        
            T2 = pt2 - pt1
            T3 = pt3 - pt1
            T4 = pt4 - pt1
            TT = np.array([ [T2[0],T3[0],T4[0]], [T2[1],T3[1],T4[1]], [T2[2],T3[2],T4[2]] ])
            vol = np.abs( np.linalg.det(TT)/6.0 )
            invTT = np.linalg.inv(TT)
            invTTTT = np.dot(invTT,invTT.T)
            
            cpt = 0
            resMat = np.zeros((10,10))
            for lk in range(3):
                for ck in range(3):
                    resMat += invTTTT[lk,ck] * listMatriceRef[cpt] * vol
                    cpt += 1
                    
            #data = resMat.flatten()
            #indL = []
            #indC = []        
            for i in range(10):
                for j in range(10):
                    #data.append(vol*quadrature(lambda x,y,z : np.dot(ff[i](x,y,z),np.dot(invTTTT,ff[j](x,y,z)))))
                    #indL.append(m[i])
                    #indC.append(m[j])
                    KK[m[i],m[j]] += resMat[i,j]     
                               
            
            # interpolation
            # for i in range(10):
#                 for j in range(10):
#                     data.append(vol*quadrature(lambda x,y,z : np.dot(ff[i](x,y,z),np.dot(invTTTT,ff[j](x,y,z)))))
#                     indL.append(m[i])
#                     indC.append(m[j])
            #listKEle.append(csc_matrix((data,(indL,indC)),shape=(len(ptsM),len(ptsM)) ) )
            del pt1,pt2,pt3,pt4,T2,T3,T4,TT,invTT,invTTTT,resMat
            
        del f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
        del ff,listMatriceRef
        return KK.tocsc()
#########################################################   


#########################################################       
def buildMatrixDirect(ptsM,m3D,m2D,m1D,m0D,mt3D,mt2D,mt1D,mt0D,sigma,KKneumann,order,choiceBC,Lmin,posM,linkSig2Dto3D,linkSig1Dto3D,linkSig0Dto3D):
    
    KKindL = []
    KKindC = []
    KKdata = []
    
    nm3D = len(m3D)
                    
        
                    
    
                    
    # ordre 1
    GGref2D = np.array([[-1.0,-1.0], [1.0,0.0], [0.0,1.0] ])
    lMatRefGrad =  []
    MatRefMass = np.array([ [1.0/6.0, 1.0/12.0, 1.0/12.0], [1.0/12.0, 1.0/6.0, 1.0/12.0], [1.0/12.0, 1.0/12.0, 1.0/6.0] ])
    if(order == 1):
        for i in range(2):
            for j in range(2):
                matTr = np.zeros((2,2))
                matTr[i,j] = 1.0
                matRes = np.dot(GGref2D,np.dot(matTr,GGref2D.T))
                lMatRefGrad.append(matRes)
                
                del matTr
                
        
    # Order 2           
    if(order == 2):
        lptsQuad = [np.array([0.659027622374092,0.231933368553031]), np.array([0.659027622374092,0.109039009072877]), np.array([0.231933368553031,0.659027622374092]), np.array([0.231933368553031,0.109039009072877]), np.array([0.109039009072877,0.659027622374092]), np.array([0.109039009072877,0.231933368553031])]
        lpdsQuad = [1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6]
        
        # lptsQuad = [np.array([0.5,0]),np.array([0.0,0.5]),np.array([0.5,0.5])]
#         lpdsQuad = [1.0/3,1.0/3,1.0/3]
        
        
        gf1 = lambda x,y : np.array([4*(x+y)-3,4*(x+y)-3])
        gf2 = lambda x,y : np.array([4*x-1,0])
        gf3 = lambda x,y : np.array([0,4*y-1])
        gf4 = lambda x,y : np.array([4-8*x-4*y,-4*x])
        gf5 = lambda x,y : np.array([4*y,4*x])
        gf6 = lambda x,y : np.array([-4*y,4-8*y-4*x])
        
        f1 = lambda x,y : 2*(x+y-0.5)*(x+y-1)
        f2 = lambda x,y : 2*x*x - x
        f3 = lambda x,y : 2*y*y-y
        f4 = lambda x,y : 4*x*(1-x)-4*x*y
        f5 = lambda x,y : 4*x*y
        f6 = lambda x,y : 4*y*(1-y)-4*x*y
        
        gf = [gf1,gf2,gf3,gf4,gf5,gf6]
        ff = [f1,f2,f3,f4,f5,f6]
        for i in range(2):
            for j in range(2):
                matTr = np.zeros((2,2))
                matTr[i,j] = 1.0
                matRes = np.zeros((6,6))
                for l in range(6):
                    for c in range(6):
                        for k,ptq in enumerate(lptsQuad):
                            matRes[l,c] += np.dot(gf[l](ptq[0],ptq[1]),np.dot(matTr,gf[c](ptq[0],ptq[1]))) * lpdsQuad[k]
                            
                lMatRefGrad.append(matRes)            
        
        matRes = np.zeros((6,6))
        for l in range(6):
            for c in range(6):
                for k,ptq in enumerate(lptsQuad):
                    matRes[l,c] += ff[l](ptq[0],ptq[1]) * ff[c](ptq[0],ptq[1]) * lpdsQuad[k]                    
            
        MatRefMass = matRes              
        
                              
        
    if(choiceBC == "1"):
        # Taking into account the BC:
        # coeff1D = 0.5785273170649641 * Lmin
        # dcoeff1D = 0.5106484442933402 / Lmin
        coeff1D = 0.53 * Lmin
        dcoeff1D = 0.53 / Lmin
        coeff2D = coeff1D*coeff1D
        dcoeff2D = dcoeff1D * dcoeff1D * Lmin * Lmin
        dcoeff3D = dcoeff1D*dcoeff1D*dcoeff1D * Lmin * Lmin * Lmin * Lmin
        
        # 2D artificial boundaries:
        for ite,m in enumerate(m2D):
            if(mt2D[ite] == "1"): 
                n1 = m[0]
                n2 = m[1]
                n3 = m[2]
        
                sig = sigma[linkSig2Dto3D[ite]]
        
                pt1 = ptsM[n1]
                pt2 = ptsM[n2]
                pt3 = ptsM[n3]
    
                T2 = pt2 - pt1
                T3 = pt3 - pt1
                TT = []
        
                if( (T2[0] == T3[0]) and (T2[0] == 0) ):              
                    TT = np.array([ [T2[1],T3[1]], [T2[2],T3[2]] ])
                if( (T2[1] == T3[1]) and (T2[1] == 0) ):
                    TT = np.array([ [T2[0],T3[0]], [T2[2],T3[2]] ])
                if( (T2[2] == T3[2]) and (T2[2] == 0) ):              
                    TT = np.array([ [T2[0],T3[0]], [T2[1],T3[1]] ])
            
                vol = np.abs( np.linalg.det(TT) * 0.5 )
        
                invTT = np.linalg.inv(TT)
                invTTT = np.dot(invTT,invTT.T)
                
                Kele = MatRefMass*0.0
                Mele = MatRefMass*vol
                
                cpt = 0
                for i in range(2):
                    for j in range(2):
                        Kele += invTTT[i,j] * lMatRefGrad[cpt] * vol
                        cpt += 1
                        
                if (order == 1):        
                    for i in range(3):
                        for j in range(3):
                            #KK[m[i],m[j]] += (Kele[i,j] * coeff1D  + Mele[i,j] * dcoeff1D) * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append((Kele[i,j] * coeff1D + Mele[i,j]*dcoeff1D)*sig)
                if(order == 2):
                    for i in range(6):
                        for j in range(6):
                            #KK[m[i],m[j]] += (Kele[i,j] * coeff1D  + Mele[i,j] * dcoeff1D) * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append((Kele[i,j] * coeff1D + Mele[i,j]*dcoeff1D)*sig)
                            
                del pt1,pt2,pt3,invTT,invTTT,Kele,Mele            
                            
                # GG = np.dot(invTT.T,GGref2D.T)
#                 Kele = np.dot(GG.T,GG)
#
#                 Mele = np.array([ [vol/6.0, vol/12.0, vol/12.0], [vol/12.0, vol/6.0, vol/12.0], [vol/12.0, vol/12.0, vol/6.0] ])
#
#                 for i in range(3):
#                     for j in range(3):
#                         # Completer ICI (q. 3.c) ) :
#                         KK[m[i],m[j]] += (Kele[i,j] * coeff1D * vol  + Mele[i,j] * dcoeff1D) * sig
        

        
  
        # 1D artificial boundaries:
        MatRefMass = np.array([ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ])
        MatRefGrad = np.array([ [1.0, -1.0], [-1.0, 1.0] ])
        
        if(order == 2):
            MatRefMass = np.array([ [1.0/6.0, -1.0/30,2.0/30.0], [-1.0/30,1.0/6.0, 2.0/30], [2.0/30, 2.0/30,0.5+1.0/30 ]])
            MatRefGrad = np.array([ [2+1.0/3, 1.0/3,-2.0-2.0/3], [1.0/3,2+1.0/3.0, -2.0-2.0/3], [-2.0-2.0/3, -2.0-2.0/3, 5 + 1.0/3] ])
        
        for ite,m in enumerate(m1D):
            if(mt1D[ite] == "2"):
                n1 = m[0]
                n2 = m[1]
                pt1 = ptsM[n1]
                pt2 = ptsM[n2]

                sig = sigma[linkSig1Dto3D[ite]]

                T2 = pt2 - pt1
                vol = np.sqrt(np.dot(T2,T2))

                Kele = MatRefGrad / vol #np.array([[1.0/vol,-1.0/vol],[-1.0/vol,1.0/vol]])
                Mele = MatRefMass * vol #np.array([[vol/3.0, vol/6.0],[vol/6.0, vol/3.0]])
                
                if(order == 1):
                    for i in range(2):
                        for j in range(2):
                            #KK[m[i],m[j]] += (Kele[i,j] * coeff2D + Mele[i,j] * dcoeff2D) * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append((Kele[i,j] * coeff2D + Mele[i,j]*dcoeff2D)*sig)
                if(order == 2):
                    for i in range(3):
                        for j in range(3):
                            #KK[m[i],m[j]] += (Kele[i,j] * coeff2D + Mele[i,j] * dcoeff2D) * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append((Kele[i,j] * coeff2D + Mele[i,j]*dcoeff2D)*sig)
                            
                del pt1,pt2,Kele,Mele            

        # 0D (corner) articificial boundaries:
        for ite,m in enumerate(m0D):
            if(mt0D[ite] == "3"):
                sig = sigma[linkSig0Dto3D[ite]]
                
                #KK[m,m] += dcoeff3D*sig
                KKindL.append(m)
                KKindC.append(m)
                KKdata.append(dcoeff3D*sig)
                
                
                
                
    if(choiceBC == "2"): # Mixte BC
    
        GGref2D = np.array([[-1.0,-1.0], [1.0,0.0], [0.0,1.0] ])
        for ite,m in enumerate(m2D):
            if(mt2D[ite] == "1"): 
                n1 = m[0]
                n2 = m[1]
                n3 = m[2]
                
                sig = sigma[linkSig2Dto3D[ite]]
                
                pt1 = ptsM[n1]
                pt2 = ptsM[n2]
                pt3 = ptsM[n3]
    
                T2 = pt2 - pt1
                T3 = pt3 - pt1
                TT = []
        
                normal = np.array([1.0,0.0,0.0])
                if( (T2[0] == T3[0]) and (T2[0] == 0) ):              
                    TT = np.array([ [T2[1],T3[1]], [T2[2],T3[2]] ])
                if( (T2[1] == T3[1]) and (T2[1] == 0) ):
                    TT = np.array([ [T2[0],T3[0]], [T2[2],T3[2]] ])
                    normal = np.array([0.0,1.0,0.0])
                if( (T2[2] == T3[2]) and (T2[2] == 0) ):              
                    TT = np.array([ [T2[0],T3[0]], [T2[1],T3[1]] ])
                    normal = np.array([0.0,0.0,1.0])
            
                vol = np.abs( np.linalg.det(TT) * 0.5 )
        
                invTT = np.linalg.inv(TT)
                invTTT = np.dot(invTT,invTT.T)

                Bary = (pt1 + pt2 + pt3)/3.0
                rr = Bary - posM

                alpha = np.abs(np.dot(normal,rr)) / np.dot(rr,rr)

                Mele = MatRefMass*vol


                if (order == 1):
                    for i in range(3):
                        for j in range(3):
                            #KK[m[i],m[j]] += Mele[i,j] * alpha * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append(Mele[i,j] * alpha *sig)
                            
                if(order == 2):
                    for i in range(6):
                        for j in range(6):
                            #KK[m[i],m[j]] += Mele[i,j] * alpha * sig
                            KKindL.append(m[i])
                            KKindC.append(m[j])
                            KKdata.append(Mele[i,j] * alpha *sig)
                
                
                
                # GG = np.dot(invTT.T,GGref2D.T)
#                 Kele = np.dot(GG.T,GG)
#
#                 Mele = np.array([ [vol/6.0, vol/12.0, vol/12.0], [vol/12.0, vol/6.0, vol/12.0], [vol/12.0, vol/12.0, vol/6.0] ])
#
#                 Bary = (pt1 + pt2 + pt3)/3.0
#                 rr = Bary - posM
#
#                 alpha = np.abs(np.dot(normal,rr)) / np.dot(rr,rr)
#
#                 for i in range(3):
#                     for j in range(3):
#                         # Completer ICI (q. 3.c) ) :
#                         KK[m[i],m[j]] += Mele[i,j] * alpha
                    
                    
    if(choiceBC == "3"): # Dirichlet BC
        for ite,m in enumerate(m2D):
            if(mt2D[ite] == "1"): 
                n1 = m[0]
                n2 = m[1]
                n3 = m[2]
        
                tmptab = []
                for i in m:
                    tmptab.append(KK[i,i])
                for i in m:
                    KK[i,:] = KK[i,:]*0.0
                    KK[:,i] = KK[:,i]*0.0   
                for j,i in enumerate(m):
                    KK[i,i] = tmptab[j]     
                     
                # tmpn1 = KK[n1,n1]
#                 tmpn2 = KK[n2,n2]
#                 tmpn3 = KK[n3,n3]
#
#                 KK[n1,:] = KK[n1,:]*0.0
#                 KK[:,n1] = KK[:,n1]*0.0
#                 KK[n2,:] = KK[n2,:]*0.0
#                 KK[:,n2] = KK[:,n2]*0.0
#                 KK[n3,:] = KK[n3,:]*0.0
#                 KK[:,n3] = KK[:,n3]*0.0
#
#                 KK[n1,n1] = tmpn1
#                 KK[n2,n2] = tmpn2
#                 KK[n3,n3] = tmpn3        
    
    KK = csc_matrix((KKdata,(KKindL,KKindC)),shape=(len(ptsM),len(ptsM))) + KKneumann     
    return KK     
#########################################################

    
#########################################################    
def genSigma(sigC1,sigC2,sigC3,m3D,ptsM):
    sigma = np.zeros(len(m3D))
    
    for ite,m in enumerate(m3D):
        bary = (ptsM[m[0]] + ptsM[m[1]] + ptsM[m[2]] + ptsM[m[3]])*0.25
        if(bary[2] >= 0.6666):
            sigma[ite] = sigC1
        if(bary[2] >= 0.3333 and bary[2] <= 0.6666):
            sigma[ite] = sigC2
        if(bary[2] >= 0.0 and bary[2] <= 0.33333):
            sigma[ite] = sigC3
            
    return sigma
#########################################################    

#########################################################    
def genMesures(VV,nodesInj):
    Vmes = np.zeros(len(nodesInj)*len(nodesInj))
    for i,inj in enumerate(nodesInj):
        Vmesi = VV[nodesInj,i]
        Vmesi[i] = 0
        Vmes[len(nodesInj)*i:(len(nodesInj)*(i+1))] = Vmesi
    
    return Vmes
#########################################################    