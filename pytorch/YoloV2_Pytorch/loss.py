# -*- coding: utf-8 -*-
"""
@author: mopju
"""

import torch 
from torch import nn


device = 'cuda'
    
def compute_iou( boxes_preds, boxes_labels):
    
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
   
    




class yolov2loss(nn.Module):
    def __init__(self, S=13, B=5, C=20):
        super(yolov2loss,self).__init__()
        
        self.S, self.B, self.C = S, B, C
        
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.mse = nn.MSELoss(reduction="sum")
        self.anchor =  (torch.tensor(
                [
                    [0, 0, 1.3221, 1.73145],
                    [0, 0, 3.19275, 4.00944],
                    [0, 0, 5.05587, 8.09892],
                    [0, 0, 9.47112, 4.84053],
                    [0, 0, 11.2364, 10.0071],
                ])/ self.S).to(device)
        
        
        
    def forward(self, predictions, target):
        
        exist = target[:,:,:,:,4:5]
        exist_boxes = exist * predictions
        
        localization_loss, obj_loss, no_obj_loss, class_loss = 0, 0, 0, 0
        
        #cell index
        cell = torch.arange(13).to(device)
        
        #sigmoid(tx) + cx
        bx =  exist * torch.sigmoid(predictions[:,:,:,:,0:1]) + exist * cell.view(1,-1,1,1,1)

        #sigmoidty) + cy 
        by =  exist * torch.sigmoid(predictions[:,:,:,:,1:2]) + exist * cell.view(1,-1,1,1,1)
        
        #pw * exp(tw)
        bw = self.anchor[...,2].view(1,1,1,-1,1) * exist * torch.exp( predictions[:,:,:,:,2:3])
        
        #py * exp(ty)
        bh = self.anchor[...,3].view(1,1,1,-1,1) * exist * torch.exp( predictions[:,:,:,:,3:4])
    
    
        ious = compute_iou(torch.cat([bx,by,bw,bh],dim=-1), target[...,:4])
        # print(torch.max(ious))
        
        xy_loss = self.mse(torch.cat([bx, by], dim=-1), target[..., :2])
        
        wh_loss = self.mse(torch.sqrt(torch.abs(torch.cat([bw, bh], dim=-1)) +1e-10 ), torch.sqrt(torch.abs(target[..., 2:4])+1e-10))
        
        localization_loss = 5*xy_loss + 5*wh_loss
        
        # this part is taken from https://github.com/Vijayabhaskar96/Object-Detection-Algorithms/blob/main/Yolo_V2/YoloV2_model.py  = = = = = = = =
        obj_loss = self.mse(exist, exist * ious * torch.sigmoid(exist_boxes[..., 4:5]))
        
        no_obj_loss = self.mse(
            (1 - exist),
            (
                ((1 - exist) * (1 - torch.sigmoid(predictions[..., 4:5])))
                * ((ious.max(-1)[0] < 0.6).int().unsqueeze(-1))
            ),
        )
        
        
        class_loss = nn.functional.nll_loss((exist * nn.functional.log_softmax(predictions[..., 5:], dim=-1)).flatten(end_dim=-2),
            target[..., 5:].flatten(end_dim=-2).argmax(-1))
        #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        
        return localization_loss + obj_loss + no_obj_loss + class_loss
