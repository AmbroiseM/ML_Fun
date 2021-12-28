# -*- coding: utf-8 -*-



"""
@author: Ambroise M
"""

import torch
import os
from os import listdir
from os.path import isfile, join
import albumentations as A
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


def preprocess_mask(mask):
    mask = np.array(mask, dtype = 'float32')
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask



class MyDataset(Dataset):
    def __init__(self, images_path, masks_path, imgs_filenames, transform=None):
        
        self.transform = transform
        self.images_path = images_path 
        self.mask_path = masks_path 
        self.filenames = imgs_filenames
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        
        img_filename = self.filenames[index]
        img = Image.open((os.path.join(self.images_path, img_filename))).convert('RGB')
                         
        mask = Image.open(os.path.join(self.mask_path, img_filename.replace(".jpg", ".png")))
        img = np.array(img)

        mask = preprocess_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
            
        return img, mask