# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:33:17 2021

@author: Ambroise M
"""

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

import imageio
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

img_path = r'C:/Users/Ambroise M/Desktop/archive/images/images/'
mask_path = r'C:/Users/Ambroise M/Desktop/archive/annotations/annotations/trimaps/'


class DataGen(tf.keras.utils.Sequence):
    
    def __init__(self, img_size, img_path, mask_path, batch_size=32): 
        
        self.img_size = img_size
        
        self.imgs = [f for f in listdir(img_path) 
                          if (isfile(join(img_path,f)) and f.endswith(".jpg"))]
        
        self.masks = [f for f in listdir(mask_path) 
                          if (isfile(join(mask_path,f)) and f.endswith(".png") and not f.startswith("."))]
        
        self.mask_path = mask_path
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.img_path)//self.batch_size
    
    def __getitem__(self, index):
        img_batch = self.imgs[ index*self.batch_size:(index + 1)*self.batch_size]
        mask_batch = self.masks[ index*self.batch_size:(index + 1)*self.batch_size]
        
        # x =
        # y = 
        for image, mask in zip(img_batch, mask_batch):
           
            img = imageio.imread(img_path + '/' + image)
            img = np.resize(img,(self.img_size))
            
            msk = imageio.imread(mask_path + '/' + mask)
            msk = np.resize(img,self.mask_size)# print(tf.shape(img))
            print(img.shape)
            

unet_data = DataGen((100,100,3), img_path, mask_path)

unet_data.__getitem__(1)