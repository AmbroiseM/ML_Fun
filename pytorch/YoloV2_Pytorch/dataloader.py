# -*- coding: utf-8 -*-
"""
@author: mopju
"""

import torch 
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou




class VOC(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=13, B=5, C=20):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        self.S = S
        self.B = B
        self.C = C
        
        self.anchor =  (torch.tensor(
                [
                    [0, 0, 1.3221, 1.73145],
                    [0, 0, 3.19275, 4.00944],
                    [0, 0, 5.05587, 8.09892],
                    [0, 0, 9.47112, 4.84053],
                    [0, 0, 11.2364, 10.0071],
                ])/ self.S)
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        
        transform = transforms.Resize((416, 416))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2,0,1)
        image =  transform(image)/255.0

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.B, 1 + self.C + 4))
        
        for box in boxes:
            
            anchor = torch.clone(self.anchor)
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            
            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (width * self.S, height * self.S)


            xmin = x - width / 2
            ymin = y - height / 2
            xmax = xmin + width
            ymax = ymin + height
            
            anchor[:, 0] = xmin
            anchor[:, 1] = ymin
            anchor[:, 2] = xmin + anchor[:, 2] / 2
            anchor[:, 3] = ymin + anchor[:, 3] / 2
            
            ious = box_iou(anchor, torch.tensor([[xmin, ymin, xmax, ymax]]))
            max_index = (torch.argmax(ious))
            
                         
            label_matrix[i, j, max_index, :4] =  torch.tensor([x_cell, y_cell, width_cell, height_cell])

            label_matrix[i, j, max_index, 4] = 1 #confidence
    
            label_matrix[i,j, max_index, 5 + class_label] = 1 # one hot encoding 
            
            return image, label_matrix
            
            


