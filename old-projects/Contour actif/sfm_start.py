# -*- coding: utf-8 -*-
import numpy as np



def start_sfm(img,init):
    '''init = binary mask
    img = 2D image
    '''
    
    phi = np.zeros((img.shape[0], img.shape[1]),float)
    label = np.zeros((img.shape[0], img.shape[1]))
    Lz, Ln1, Ln2, Lp1, Lp2 = [], [], [], [], []
    
    #Pre-condition labelmap and phi
    for i in range(init.shape[0]):
      for j in range(init.shape[1]):
        if init[i][j]==0:
          phi[i][j]=3.0
          label[i][j]=3
          
        if init[i][j]==1:
          phi[i][j]=-3.0
          label[i][j]=-3

    
    #Find the zero-level set
    for i in range(init.shape[0]):
      for j in range(init.shape[1]):
        if (init[i][j]==1):
          if ((i == 0) or (i == init.shape[0]) or  (i-1< 0) or  (i+1 >= init.shape[0]) or (j == 0) or (j == init.shape[1]) or (j+1 >= init.shape[1]) or (j-1 <0)):
            pass
          else:
            if ((init[i-1][j]==0) or (init[i+1][j]==0) or (init[i][j-1]==0) or (init[i][j+1]==0)):
              Lz.append((i,j))
              label[i][j]=0
              phi[i][j]=0
              
    
    #Find the +1 and -1 level set
    for i in range(len(Lz)):
      point = Lz[i]   #Lz = [(x,y),(x2,y2),...]
      x = point[0]
      y = point[1]
      
      if ((x == 0) or  (x-1< 0) or (x == init.shape[0]) or  (x+1 > init.shape[0]) or (y == 0) or  (y-1< 0) or (y == init.shape[1]) or (y+1 > init.shape[1])):
          pass
      else:
         #voisin 1====================================================== (4-neighborhood)
          if (label[x-1][y]==-3):
            Ln1.append((x-1,y))
            label[x-1][y] = -1
            phi[x-1][y] = -1
            
          if (label[x-1][y]==3):
            Lp1.append((x-1,y))
            label[x-1][y] = 1
            phi[x-1][y] = 1
            
        #voisin 2=======================================================
          if (label[x+1][y]==-3):
            Ln1.append((x+1,y))
            label[x+1][y] = -1
            phi[x+1][y] = -1
            
          if (label[x+1][y]==3):
            Lp1.append((x+1,y))
            label[x+1][y] = 1
            phi[x+1][y] = 1
        
        #voisin3==========================================================
          if (label[x][y-1]==-3):
            Ln1.append((x,y-1))
            label[x][y-1] = -1
            phi[x][y-1] = -1
            
          if (label[x][y-1]==3):
            Lp1.append((x,y-1))
            label[x][y-1] = 1
            phi[x][y-1] = 1
            
        #voisin4==========================================================
          if (label[x][y+1]==-3):
            Ln1.append((x,y+1))
            label[x][y+1] = -1
            phi[x][y+1] = -1
            
          if (label[x][y+1]==3):
            Lp1.append((x,y+1))
            label[x][y+1] = 1
            phi[x][y+1] = 1
    
    #Find the +2 and -2 level set
    for i in range(len(Ln1)):
      point = Ln1[i]
      x,y = point[0], point[1]
      
      #gestion des coins
      if ((x == 0) or  (x-1< 0) or (x == init.shape[0]) or  (x+1 >= init.shape[0]) or (y == 0) or  (y-1< 0) or (y == init.shape[1]) or (y+1 >= init.shape[1])):
          pass
      
      else:
         #voisin 1======================================================   
          if (label[x-1][y]==-3):
            Ln2.append((x-1,y))
            label[x-1][y] = -2
            phi[x-1][y] = -2
            
        #voisin 2=======================================================
          if (label[x+1][y]==-3):
            Ln2.append((x+1,y))
            label[x+1][y] = -2
            phi[x+1][y] = -2
            
        #voisin3==========================================================
          if (label[x][y-1]==-3):
            Ln2.append((x,y-1))
            label[x][y-1] = -2
            phi[x][y-1] = -2
            
        #voisin4==========================================================
          if (label[x][y+1]==-3):
            Ln2.append((x,y+1))
            label[x][y+1] = -2
            phi[x][y+1] = -2
            
    
    for i in range(len(Lp1)):
      point = Lp1[i]
      x, y = point[0], point[1]
      
      if ((x == 0) or  (x-1< 0) or (x == init.shape[0]) or  (x+1 > init.shape[0]) or (y == 0) or  (y-1< 0) or (y == init.shape[1]) or (y+1 > init.shape[1])):
          pass
      else:
          
     #voisin 1======================================================   
          if (label[x-1][y]==3):
            Lp2.append((x-1,y))
            label[x-1][y] = 2
            phi[x-1][y] = 2
            
        #voisin 2=======================================================
          if (label[x+1][y]==3):
            Lp2.append((x+1,y))
            label[x+1][y] = 2
            phi[x+1][y] = 2
            
        #voisin3==========================================================
          if (label[x][y-1]==3):
            Lp2.append((x,y-1))
            label[x][y-1] = 2
            phi[x][y-1] = 2
            
        #voisin4==========================================================
          if (label[x][y+1]==3):
            Lp2.append((x,y+1))
            label[x][y+1] = 2
            phi[x][y+1] = 2
        
        
    return init,phi,label,Lz,Ln1,Lp1,Ln2,Lp2, img