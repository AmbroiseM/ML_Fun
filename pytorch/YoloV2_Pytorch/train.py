# -*- coding: utf-8 -*-
"""
@author: mopju
"""

import torch 
from torch import nn
import numpy as np
from loss import yolov2loss
from dataloader import VOC
from yolov2 import *
from torch.utils.data import DataLoader


device = 'cuda'

def train_yolo(model, optimizer, train_data, val_data, epochs):
    
    
    #train 
    model.train()
    loss = yolov2loss()
    
    for epoch in range(epochs):
        
        train_loss, val_loss = [], []
        for i, data in enumerate(train_data,0):
            
            images, target = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            predictions = model(images.float())
            yolo_loss = loss(predictions, target)
            
            train_loss.append(yolo_loss.item())
            
            yolo_loss.backward()
            optimizer.step()
            

            
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_data, 0):
                
                images, target = data[0].to(device), data[1].to(device)
                predictions = model(images.float())
                yolo_loss = loss(predictions,target)
                val_loss.append(yolo_loss.item())
                
            
        print("Epoch ", epoch +1,  "| Train Loss: ", np.array(train_loss).mean(),  "| Val Loss: ", np.array(val_loss).mean())
            
            



if __name__ == '__main__':
    
   yolo =  yolov2().to(device)
   optimizer = torch.optim.Adam(yolo.parameters(),lr=2e-5)
   
   voc = VOC(csv_file=r"C:\Users\mopju\Desktop\voc\100examples.csv", 
                     img_dir=r"C:\Users\mopju\Desktop\voc\images/",
                     label_dir=r"C:\Users\mopju\Desktop\voc\labels/")
   
   train_size = int(0.8 * len(voc))
   test_size = len(voc) - train_size
   train_dataset, test_dataset = torch.utils.data.random_split(voc, [train_size, test_size])
   
   
   train_dataset = DataLoader(train_dataset, batch_size=8, shuffle=True)
   test_dataset = DataLoader(test_dataset, batch_size=8, shuffle=True)

   train_yolo(yolo, optimizer, train_dataset, test_dataset, 5)