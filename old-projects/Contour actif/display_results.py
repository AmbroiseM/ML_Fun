# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:57:04 2020

@author: Ambroise M
"""

from calcul_sfm import compute_sfm
from calcul_ftc import compute_ftc
import  cv2, time
import numpy as np
import matplotlib.pyplot as plt



def display_sfm(img,init,it):
    return compute_sfm(img, init,it)


def display_ftc(img,init,Na,Ns):
    return compute_ftc(img, init,Na,Ns)
    
    




#=image & mask ================================================================
    
img = "test0.jpg"
img = cv2.imread(img)
img = cv2.resize(img,None,fx=0.5,fy=0.5)

if len(img.shape)==2:pass
else:img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
init = np.zeros((img.shape[0],img.shape[1]))
init[20:120,20:110]=1


#compute ======================================================================

start_sfm = time.time()
seg_sfm =(display_sfm(img,init,60))
end_sfm = time.time()
print("sfm: ",end_sfm - start_sfm, "secondes")


start_ftc = time.time()
seg_ftc=(display_ftc(img,init,10,5))
end_ftc = time.time()
print("ftc:",end_ftc - start_ftc, "secondes")

#plot results==================================================================

plt.subplot(2,2,1)
plt.figsize=(20, 15)
plt.imshow(img)
plt.title("Original image")

plt.subplot(2,2,2)
plt.figsize=(20, 15)
plt.imshow(init)
plt.title("Mask")

plt.tight_layout(pad=1.9)

plt.subplot(2,2,3)
plt.figsize=(20, 15)
plt.imshow(seg_sfm)
plt.title("SFM")

plt.subplot(2,2,4)
plt.figsize=(20, 15)
plt.imshow(seg_ftc)
plt.title("FTC")

    

        