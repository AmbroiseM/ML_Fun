# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 21:13:40 2022

@author: Ambroise M
"""
import torch 
from torch import nn 




    
def compute_iou(box1, box2):
    
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[2]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[2]
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0
    
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I # Union = Total Area - I
    
    return I / U
    


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss,self).__init__()
        
        self.S, self.B, self.C = S, B, C
        
        self.lambda_noobj = 0.5
        
        self.lambda_coord = 5
        
        
    def forward(self, predictions, target):
        
        #5 losses
        xy_loss, wh_loss = 0, 0
        
        no_object_loss, object_loss = 0, 0
        
        class_loss = 0
        
        for n in range(predictions.shape[0]):# predictions.shape[0] = bacth_size
        
            for i in range(self.S):
                
                for j in range(self.S):
                    
                    #(from the paper) "if object appears in cell i" ==> if an object is detected
                    if target[n,i,j,4]==1:
                        
                        bbox_target = target[n,i,j,0:4]
                        
                        bb1_pred = predictions[n,i,j,0:4] 
                        
                        bb2_pred = predictions[n,i,j,5:9]
                        
                        iou1, iou2 = compute_iou(bb1_pred, bbox_target), compute_iou(bb2_pred, bbox_target)
                            
                        #class_loss
                        class_loss += torch.sum(torch.pow(predictions[n,i,j,10:] - target[n,i,j,10:] , 2))
                        
                        #we'll then look for the bounding box "responsible" of that prediction
                        if iou1 > iou2:
                            
                            #xy loss
                            xy_loss += self.lambda_coord * torch.sum( torch.pow((predictions[n,i,j,0:2] - target[n,i,j,0:2]),2) )
                            
                            #w loss
                            wh_loss +=  self.lambda_coord * torch.sum((torch.pow(torch.abs(predictions[n,i,j,2:4]).sqrt() - target[n,i,j,2:4].sqrt(),2)))
                            
                            #no_object_loss (hmm...easy shorcut !!)
                            no_object_loss  += self.lambda_noobj * torch.sum( torch.pow(0 - predictions[n,i,j,9], 2) )
                            
                            #object_loss 
                            object_loss += torch.sum( torch.pow(1 - predictions[n,i,j,4], 2) )
                            
                        else: 
                            
                            #xy loss
                            xy_loss += self.lambda_coord *  torch.sum( torch.pow(predictions[n,i,j,5:7] - target[n,i,j,0:2], 2) )
                            
                            #wh loss
                            wh_loss +=  self.lambda_coord *  torch.sum( torch.pow(torch.abs(predictions[n,i,j,7:9]).sqrt() - target[n,i,j,2:4].sqrt(), 2) )
                            # print(wh_loss.item())
                          
                            #no object loss (humm...easy shortcut !!)
                            no_object_loss += self.lambda_noobj * torch.sum(torch.pow(0 - predictions[n,i,j,4],2) )
                            
                            #object loss
                            object_loss += torch.sum(torch.pow(1 - predictions[n,i,j,9], 2))
                         
                    #no object detected
                    else:
                        no_object_loss += self.lambda_noobj * torch.sum( torch.pow(predictions[n,i,j,[4,9]] - 0,2)) 
                        
                        
        return xy_loss + wh_loss + no_object_loss + object_loss + class_loss
                        
                            
                            
                            
        
        
        