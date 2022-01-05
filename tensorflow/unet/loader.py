# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from PIL import Image

import imageio
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


class DataGen(tf.keras.utils.Sequence):
    
    def __init__(self, img_size, mask_size, imgs, masks, batch_size=16): 
        
        self.img_size = img_size
        self.mask_size = mask_size
        self.imgs = imgs         
        self.masks = masks
        self.batch_size = batch_size
        
                
    def __len__(self):
        return len(self.imgs)//self.batch_size
    
    def __getitem__(self, index):

        img_batch = self.imgs[ index*self.batch_size:(index + 1)*self.batch_size]
        mask_batch = self.masks[ index*self.batch_size:(index + 1)*self.batch_size]
        
        #x, y =[], []
        #for image, mask in zip(img_batch, mask_batch):

        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(img_batch):
          img = tf.keras.preprocessing.image.load_img(img_path + '/'+path, target_size=self.img_size)
          x[j] = img

        y = np.zeros((self.batch_size,) + self.mask_size, dtype="uint8")
        for j, path in enumerate(mask_batch):
          img = tf.keras.preprocessing.image.load_img(mask_path + '/'+path, target_size=self.img_size, color_mode="grayscale")
          y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
          y[j] -= 1
        
        return x, y
            
