# -*- coding: utf-8 -*-
"""
@author: Ambroise M
"""

import torch 
from torch import nn 
from yololoss import *
from dataloader import VOCDataset
from yolo_model import Yolo
from torch.utils.data import DataLoader
import numpy as np
from utils import *





device = "cuda:0" if torch.cuda.is_available else "cpu"





def train_yolo(model, optimizer, trainloader, val_loader, epochs):
    
    model.train()
    loss = YoloLoss()
    
    for epoch in range(epochs):
        
        train_loss, val_loss = [], []
        
        for i, data in enumerate(trainloader, 0):
            
            images, target = data[0].to(device), data[1].to(device)
                  
            optimizer.zero_grad()
            
            predictions = model(images)
            
            yolo_loss = loss(predictions,target)
            
            train_loss.append(yolo_loss.item())
            
            yolo_loss.backward()
    
            optimizer.step()   
            
                
                
        #Eval    
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                
                images, target = data[0].to(device), data[1].to(device)
                
                predictions = model(images)
                
                yolo_loss = loss(predictions,target)
                
                val_loss.append(yolo_loss.item())
                
                
                
            # for j in range(predictions.shape[0]):   
            #     bbox = decode_yolo_output(predictions[j])
            #     plt.imshow(display_image(images[j].cpu().permute(1,2,0), bbox))
            #     plt.show()
            
        print("Epoch ", epoch +1,  "| Train Loss: ", np.array(train_loss).mean(),  "| Val Loss: ", np.array(val_loss).mean())
            


     


if __name__ == '__main__':
    
    yolo =  Yolo().to(device)

    optimizer = torch.optim.Adam(yolo.parameters(), lr=2e-5)

    voc = VOCDataset(csv_file=r"C:\Users\mopju\Desktop\voc\train.csv", 
                      img_dir=r"C:\Users\mopju\Desktop\voc\images/",
                      label_dir=r"C:\Users\mopju\Desktop\voc\labels/")
    

    train_size = int(0.8 * len(voc))
    test_size = len(voc) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(voc, [train_size, test_size])
    
    
    train_dataset = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=8, shuffle=True)

    train_yolo(yolo, optimizer, train_dataset, test_dataset, 50)
