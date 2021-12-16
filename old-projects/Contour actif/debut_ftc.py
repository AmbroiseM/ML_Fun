# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:17:21 2020

@author: Ambroise M
"""


import numpy as np


def start_ftc(img, init):
    '''init = binary mask
    img = 2D image
    '''

    
    phi = np.zeros((img.shape[0], img.shape[1]))
    Lin, Lout = [],[]
    
    #Pre-condition labelmap and phi
    for i in range(init.shape[0]):
      for j in range(init.shape[1]):
        if init[i][j]==0:
          phi[i][j]=3
          
        if ((init[i][j]==1)):
          phi[i][j]=-3
    
    #Find the zero-level set
    for i in range(init.shape[0]):
      for j in range(init.shape[1]):
        if (init[i][j]==1):
          if ((i == 0) or (i == init.shape[0]) or (i+1 >= init.shape[0]) or (j == 0) or (j == init.shape[1]) or (j+1 >= init.shape[1])):
            pass
          else:
            if ((init[i-1][j]==0) or (init[i+1][j]==0) or (init[i][j-1]==0) or (init[i][j+1]==0)):
              Lin.append((i,j))
              phi[i][j]=-1
              
        
    #Find the +1 and -1 level set
    for i in range(len(Lin)):
      point = Lin[i]   #Lin = [(x,y),(x2,y2),...].
      x = point[0]
      y = point[1]
      
      if ((x == 0) or (x == init.shape[0]) or (x+1 >= init.shape[0]) or (y == 0) or (y == init.shape[1]) or (y+1 >= init.shape[1])):
          pass
      else:
         #voisin 1======================================================   (4-neighborhood)
          if (phi[x-1][y]==3):
            Lout.append((x-1,y))
            phi[x-1][y] = 1
            
        #voisin 2=======================================================
          if (phi[x+1][y]==3):
            Lout.append((x+1,y))
            phi[x+1][y] = 1
        
        #voisin3==========================================================
          if (phi[x][y-1]==3):
            Lout.append((x,y-1))
            phi[x][y-1] = 1
            
        #voisin4==========================================================
          if (phi[x][y+1]==3):
            Lout.append((x,y+1))
            phi[x][y+1] = 1
            
    return img, phi, Lin, Lout