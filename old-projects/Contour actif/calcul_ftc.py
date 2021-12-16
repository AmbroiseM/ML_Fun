# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:41:18 2020

@author: Ambroise M
"""

from debut_ftc import start_ftc
import numpy as np
import cv2



def Fin(x,y,new_img,liste):
    if liste == "Lin":
        if (new_img[x][y]<1/2): return -1
        else: return 0
        
    if liste == "Lout":
        if (new_img[x][y]>1/2): return 1
        else: return 0
        


def compute_ftc(image,init,Na,Ns):
        
    results = start_ftc(image,init)
    
    img  = results[0]
    phi  = results[1]
    Lin  = results[2]
    Lout = results[3]
    
    
    Fd = np.zeros((phi.shape[0],phi.shape[1]))    
    
       
    for it in range(Na):
        
        upts = np.flatnonzero(phi <= -1)  # interior points
        vpts = np.flatnonzero(phi >= 1)  # exterior points
        u = np.sum(img.flat[upts]) / (len(upts))  # interior mean
        v = np.sum(img.flat[vpts]) / (len(vpts))  # exterior mean
        
        for k in Lin:
            x,y = k[0], k[1]
            Fd[x][y] = (img[x][y] - u)**2 - (img[x][y] - v)**2
            
        for k in Lout:
            x,y = k[0], k[1]
            Fd[x][y] = (img[x][y] - u)**2 - (img[x][y] - v)**2
                        
        Fd[Fd>=0]=1
        Fd[Fd<0]=-1
    
    #------------------------------------------------------------------------------
        for i in Lout:
            point = i
            x, y = point[0],point[1]
            
    #        if Fd[x][y]>0:
            if Fd[x][y]<0:
                Lout.remove(point)
                
                if point not in Lin : Lin.append(point)
                if ((x== 0) or (x == phi.shape[0]) or  (x-1< 0) or  (x+1 >= phi.shape[0]) or (y == 0) or (y == phi.shape[1]) or (y+1 >= phi.shape[1]) or (y-1 <0)):
                    pass
                else:    
                    if phi[x-1][y] == 3: Lout.append((x-1,y))
                    if phi[x+1][y] == 3: Lout.append((x+1,y))
                    if phi[x][y-1] == 3: Lout.append((x,y-1))
                    if phi[x][y+1] == 3: Lout.append((x,y+1))
                        
                phi[x][y] = -1
                
    #------------------------------------------------------------------------------
        for i in Lin:
            point = i
            x, y = point[0],point[1]
            
            if ((x== 0) or (x == phi.shape[0]) or  (x-1< 0) or  (x+1 >= phi.shape[0]) or (y == 0) or (y == phi.shape[1]) or (y+1 >= phi.shape[1]) or (y-1 <0)):
                    pass
            else:
                if ((phi[x-1][y]<0) and (phi[x+1][y]<0) and (phi[x][y-1]<0) and (phi[x][y+1]<0)):
                    Lin.remove(point)
                    phi[x][y] = -3
                    
                    
    #------------------------------------------------------------------------------
                    
        for i in Lin:
            point = i
            x, y = point[0],point[1]
            
    #        if Fd[x][y]<0:
            if Fd[x][y]>0:
                Lin.remove(point)
                
                if point not in Lout : Lout.append(point)
                
                if ((x== 0) or (x == phi.shape[0]) or  (x-1< 0) or  (x+1 >= phi.shape[0]) or (y == 0) or (y == phi.shape[1]) or (y+1 >= phi.shape[1]) or (y-1 <0)):
                    pass
                else:
                    if phi[x-1][y] == -3: Lin.append((x-1,y))
                    if phi[x+1][y] == -3: Lin.append((x+1,y))
                    if phi[x][y-1] == -3: Lin.append((x,y-1))
                    if phi[x][y+1] == -3: Lin.append((x,y+1))
                    
                phi[x][y] = 1
                
    #--------------------------------------------------------------------------
        for i in Lout:
            point = i
            x, y = point[0],point[1]
            
            if ((x== 0) or (x == phi.shape[0]) or  (x-1< 0) or  (x+1 >= phi.shape[0]) or (y == 0) or (y == phi.shape[1]) or (y+1 >= phi.shape[1]) or (y-1 <0)):
                    pass
            else:
                if ((phi[x-1][y]>0) and (phi[x+1][y]>0) and (phi[x][y-1]>0) and (phi[x][y+1]>0)):
                    Lout.remove(point)
                    phi[x][y] = 3
                    
                
    #==========================================================================
                                    #SMOOTHING
    #========================================================================== 
    
    new_img = cv2.GaussianBlur(phi, (7, 7), 1)
    for Ns in range(Ns):
        
      for i in Lout:
              point = i
              x, y = point[0],point[1]
              
              if Fin(x,y,new_img,"Lin")<0:
                  Lout.remove(point)
                  
                  if point not in Lin :
                      Lin.append(point)
                  
                  if ((x== 0) or (x == phi.shape[0]) or  (x-1< 0) or  (x+1 >= phi.shape[0]) or (y == 0) or (y == phi.shape[1]) or (y+1 >= phi.shape[1]) or (y-1 <0)):
                      pass
                  else:
                      if phi[x-1][y] == 3: 
                          Lout.append((x-1,y))
                          
                      if phi[x+1][y] == 3: 
                          Lout.append((x+1,y))
                          
                      if phi[x][y-1] == 3:
                          Lout.append((x,y-1))
                          
                      if phi[x][y+1] == 3:
                          Lout.append((x,y+1))
                          
                  phi[x][y] = -1
    
    #--------------------------------------------------------------------------
      for i in Lin:
              point = i
              x, y = point[0],point[1]
              
              if ((x== 0) or (x == phi.shape[0]) or  (x-1< 0) or  (x+1 >= phi.shape[0]) or (y == 0) or (y == phi.shape[1]) or (y+1 >= phi.shape[1]) or (y-1 <0)):
                      pass
              else:
                  if ((phi[x-1][y]<0) and (phi[x+1][y]<0) and (phi[x][y-1]<0) and (phi[x][y+1]<0)):
                      Lin.remove(point)
                      phi[x][y] = -3
    
    #--------------------------------------------------------------------------
      for i in Lin:
              point = i
              x, y = point[0],point[1]
              
              if Fin(x,y,new_img,"Lout")>0:
                  Lin.remove(point)
                  
                  if point not in Lout : Lout.append(point)
                  
                  if ((x== 0) or (x == phi.shape[0]) or  (x-1< 0) or  (x+1 >= phi.shape[0]) or (y == 0) or (y == phi.shape[1]) or (y+1 >= phi.shape[1]) or (y-1 <0)):
                      pass
                  else:
                  
                      if phi[x-1][y] == -3: Lin.append((x-1,y))
                          
                      if phi[x+1][y] == -3: Lin.append((x+1,y))
                          
                      if phi[x][y-1] == -3: Lin.append((x,y-1))
                          
                      if phi[x][y+1] == -3: Lin.append((x,y+1))
                  
                  phi[x][y] = 1
                  
    #--------------------------------------------------------------------------
      for i in Lout:
              point = i
              x, y = point[0],point[1]
              
              if ((x== 0) or (x == phi.shape[0]) or  (x-1< 0) or  (x+1 >= phi.shape[0]) or (y == 0) or (y == phi.shape[1]) or (y+1 >= phi.shape[1]) or (y-1 <0)):
                      pass
              else:
                  if ((phi[x-1][y]>0) and (phi[x+1][y]>0) and (phi[x][y-1]>0) and (phi[x][y+1]>0)):
                      Lout.remove(point)
                      phi[x][y] = 3
               
                
    return phi