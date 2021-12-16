# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 07:23:37 2020

@author: Ambroise M
"""
from sfm_start import start_sfm
import numpy as np
import collections




def compute_sfm(image,init,it):


    results = start_sfm(image,init)
    
    init, phi, label, Lz = results[0], results[1], results[2], results[3]
    Ln1, Lp1, Ln2, Lp2   = results[4], results[5], results[6], results[7]
    img = results[8]
    
    Sp1, Sn1, Sn2, Sp2, Sz = collections.deque(), collections.deque(), collections.deque(), collections.deque(), collections.deque()    
    
    F = np.zeros((phi.shape[0], phi.shape[1]))
    
    for it in range(it):
    #------------------------------------------------------------------------------
        
        upts = np.flatnonzero(phi <= 0)  # interior points
        vpts = np.flatnonzero(phi > 0)  # exterior points
        u = np.sum(img.flat[upts]) / (len(upts))  # interior mean
        v = np.sum(img.flat[vpts]) / (len(vpts))  # exterior mean
    
    
        idx = np.argwhere(np.logical_and(phi <= 0.5, phi >= -0.5))
        
        
        for k in idx:
            x,y = k[0], k[1]
            F[x][y] = (img[x][y] - u)**2 - (img[x][y] - v)**2
        F /= np.max(np.absolute(F))*0.4
    
        
        for i in list(Lz):
            point = i
            x,y = point[0], point[1]
            phi[x][y] += F[x][y]
    
            if phi[x][y] >0.5:
                Lz.remove(point)
                if (point not in Sp1): Sp1.append(point)
                
            if phi[x][y] < -0.5:
                Lz.remove(point)
                if (point not in Sn1): Sn1.append(point)
                    
        #------------------------------------------------------------------------------
        for i in list(Ln1):
            point = i
            x,y = point[0], point[1]
                    
            pmax = -3
            
            if ((x== 0) or (x == init.shape[0]) or  (x-1< 0) or  (x+1 >= init.shape[0]) or (y == 0) or (y == init.shape[1]) or (y+1 >= init.shape[1]) or (y-1 <0)):
                pass
            else:
            
                if ((label[x-1][y]>=0) and (phi[x-1][y]>pmax)):pmax = phi[x-1][y]
                if ((label[x+1][y]>=0) and (phi[x+1][y]>pmax)):pmax = phi[x+1][y]
                if ((label[x][y-1]>=0) and ((phi[x][y-1])>pmax)):pmax = phi[x][y-1]
                if ((label[x][y+1]>=0) and ((phi[x][y+1])>pmax)):pmax = phi[x][y+1]
                
                if pmax >= - 0.5:
                    
                    phi[x][y] = pmax - 1
                    if phi[x][y] >= - 0.5:
                        Ln1.remove(point)
                        if point not in Sz: Sz.append(point)
                            
                    elif phi[x][y] < - 1.5:
                        Ln1.remove(point)
                        if point not in Sn2: Sn2.append(point)
                        
                else: 
                    Ln1.remove(point)
                    if point not in Sn2: Sn2.append(point)
                    
        #---------------------------------------------
        for i in list(Lp1):
            point = i
            x,y = point[0], point[1]
            pmin = 3
            
            if ((x== 0) or (x == init.shape[0]) or  (x-1< 0) or  (x+1 > init.shape[0]) or (y == 0) or (y == init.shape[1]) or (y+1 > init.shape[1]) or (y-1 <0)):
                pass
            else:
    
                if ((label[x-1][y]<=0) and (phi[x-1][y]<pmin)):pmin = phi[x-1][y]
                if ((label[x+1][y]<=0) and (phi[x+1][y]<pmin)):pmin = phi[x+1][y]
                if ((label[x][y-1]<=0) and ((phi[x][y-1])<pmin)):pmin = phi[x][y-1]
                if ((label[x][y+1]<=0) and ((phi[x][y+1])<pmin)):pmin = phi[x][y+1]
            
                if pmin <= 0.5:
                
                    phi[x][y] = pmin + 1
                    if phi[x][y] <=  0.5:
                        Lp1.remove(point)
                        if point not in Sz: Sz.append(point)
                            
                    elif phi[x][y] > 1.5:
                        Lp1.remove(point)
                        if point not in Sp2: Sp2.append(point)
                        
                else: 
                    Lp1.remove(point)
                    if point not in Sp2: Sp2.append(point)
                    
        #------------------------------------------------------------------------------
        for i in list(Ln2):
            point = i
            x,y = point[0], point[1]
            pmax = -3
            
            if ((x== 0) or (x == init.shape[0]) or  (x-1< 0) or  (x+1 >= init.shape[0]) or (y == 0) or (y == init.shape[1]) or (y+1 >= init.shape[1]) or (y-1 <0)):
                pass
            else:
                        
                if ((label[x-1][y]>=-1) and (phi[x-1][y]>pmax))  :pmax = phi[x-1][y]
                if ((label[x+1][y]>=-1) and (phi[x+1][y]>pmax))  :pmax = phi[x+1][y]
                if ((label[x][y-1]>=-1) and ((phi[x][y-1])>pmax)):pmax = phi[x][y-1]
                if ((label[x][y+1]>=-1) and ((phi[x][y+1])>pmax)):pmax = phi[x][y+1]
                
                if pmax >= -1.5:
                    phi[x][y] = pmax - 1
                    if phi[x][y] >= - 1.5:
                        Ln2.remove(point)
                        if point not in Sn1: Sn1.append(point)
                            
                    elif phi[x][y] < - 2.5:
                        Ln2.remove(point)
                        label[x][y] = -3
                        phi[x][y] = -3
                        
                else:
                    Ln2.remove(point)
                    label[x][y] = -3
                    phi[x][y] = -3
            
        #------------------------------------------
        for i in list(Lp2):
            point = i
            x,y = point[0], point[1]
            pmin = 3
            
            if ((x== 0) or (x == init.shape[0]) or  (x-1< 0) or  (x+1 >= init.shape[0]) or (y == 0) or (y == init.shape[1]) or (y+1 >= init.shape[1]) or (y-1 <0)):
                pass
            else:
                        
        
                if ((label[x-1][y]<=1) and (phi[x-1][y]<pmin)):pmin = phi[x-1][y]
                if ((label[x+1][y]<=1) and (phi[x+1][y]<pmin)):pmin = phi[x+1][y]
                if ((label[x][y-1]<=1) and ((phi[x][y-1])<pmin)):pmin = phi[x][y-1]
                if ((label[x][y+1]<=1) and ((phi[x][y+1])<pmin)):pmin = phi[x][y+1]
            
                if pmin <= 1.5:
                    phi[x][y] = pmin + 1
                    if phi[x][y] <=  1.5:
                        Lp2.remove(point)
                        if point not in Sp1: Sp1.append(point)
                            
                    if phi[x][y] > 2.5:
                        Lp2.remove(point)
                        label[x][y] = 3
                        phi[x][y] = 3
                else:
                    Lp2.remove(point)
                    label[x][y] = 3
                    phi[x][y] = 3
                           
                    
                    
        #==============================================================================         
                                        # PROCEDURE 3 
        #==============================================================================
            
        #Move points into zero level set. ---------------------------------------------
        
        
        for i in list(Sz):
        
            point,x,y = i,point[0],point[1]
            label[x][y]=0
            
            if (point not in Lz):Lz.append(point)
            Sz.remove(point)
        
         
        #Move points into -1 and +1 level sets ----------------------------------------
        #and ensure -2, +2 neighbors --------------------------------------------------
            
        for i in list(Sn1):
        
            point,x,y = i,point[0],point[1]
            label[x][y]=-1
            
            if (point not in Ln1):Ln1.append(point)
            Sn1.remove(point)
                
            if (phi[x-1][y]==-3):
                phi[x-1][y]=phi[x][y]-1
                if (point not in Sn2):Sn2.append((x-1,y))
                              
            if phi[x+1][y]==-3:
                phi[x+1][y]=phi[x][y]-1
                if (point not in Sn2):Sn2.append((x+1,y))
                          
            if phi[x][y-1]==-3:
                phi[x][y-1]=phi[x][y]-1
                if (point not in Sn2):Sn2.append((x,y-1))
                          
            if label[x][y+1]==-3:
                phi[x][y+1]=phi[x][y]-1
                if (point not in Sn2):Sn2.append((x,y+1))
        
        #--------------------------------------------
        for i in list(Sp1):
        
            point,x,y = i,point[0],point[1]
            label[x][y]=1
            
            if (point not in Lp1):Lp1.append(point)
            Sp1.remove(point)
            
            if (phi[x-1][y]==3):
                phi[x-1][y]=phi[x][y]+1
                if (point not in Sp2):Sp2.append((x-1,y))
                  
            if phi[x+1][y]==3:
                phi[x+1][y]=phi[x][y]+1
                if (point not in Sp2):Sp2.append((x+1,y))
                  
            if phi[x][y-1]==3:
                phi[x][y-1]=phi[x][y]+1
                if (point not in Sp2):Sp2.append((x,y-1))
        
                  
            if label[x][y+1]==3:
                phi[x][y+1]=phi[x][y]+1
                if (point not in Sp2):Sp2.append((x,y+1))       
                
                
        #Move points into -2 and +2 level sets ----------------------------------------
        for i in list(Sn2):
        
            point,x,y = i,point[0],point[1]
            label[x][y]= -2
            
            if (point not in Ln2):Ln2.append(point)
            Sn2.remove(point)
           
        #--------------------------------------------
        for i in list(Sp2):
        
            point,x,y = i,point[0],point[1]
            label[x][y]= 2
            
            if (point not in Lp2):Lp2.append(point)
            Sp2.remove(point)
            
    return phi
    
            