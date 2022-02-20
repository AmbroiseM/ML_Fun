# -*- coding: utf-8 -*-
"""

@author: Ambroise M
"""

import torch
import matplotlib.pyplot as plt
import cv2



class_names = {0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle", 
              5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow", 
              10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person", 
              15: "plant", 16: "sheep", 17: "sofa", 18: "train", 19: "monitor"}


voc_color = {'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 0, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (255, 0, 0),
    'plant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'monitor': (0, 64, 128)}





def decode_yolo_output(prediction):
    
    bbox = torch.zeros((7,7,25))
    
    for i in range(bbox.size(0)):
        for j in range(bbox.size(1)):
            
            if prediction[i,j,4] > prediction[i,j,9]: #confidence 1 > confidence 2
                
                bbox[i,j,4] = prediction[i,j,4] #confidence
                bbox[i,j,:4] = prediction[i,j,:4]  #coordinates
                bbox[i,j,5:] = prediction[i,j,10:] #classes prob
                
            else: 
                
                bbox[i,j,4] = prediction[i,j,9]
                bbox[i,j,:4] = prediction[i,j,5:9]
                bbox[i,j,5:] = prediction[i,j,10:]
                
    bbox = torch.reshape(bbox,(7*7,25))
    
    #nms
    return nms(bbox)
                
    


    
def compute_iou(box1, box2):
    
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[2]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[2]
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    
    if w_intersection <= 0 or h_intersection <= 0: 
        return 0
    
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I # 
    
    return I / U




#code from https://github.com/ivanwhaf/yolov1-pytorch
def nms(bboxs, num_classes=20, conf_thresh=0.5, iou_thresh=0.5):
    # Non-Maximum Suppression, bboxs is a 98*15 tensor
    bbox_prob = bboxs[:, 5:].clone().detach()  # 98*10
    bbox_conf = bboxs[:, 4].clone().detach().unsqueeze(1).expand_as(bbox_prob)  # 98*10
    bbox_cls_spec_conf = bbox_conf * bbox_prob  # 98*10
    bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0

    # for each class, sort the cls-spec-conf score
    for c in range(num_classes):
        rank = torch.sort(bbox_cls_spec_conf[:, c], descending=True).indices  # sort conf
        # for each bbox
        for i in range(bboxs.shape[0]):
            if bbox_cls_spec_conf[rank[i], c] == 0:
                continue
            for j in range(i + 1, bboxs.shape[0]):
                if bbox_cls_spec_conf[rank[j], c] != 0:
                    iou = compute_iou(bboxs[rank[i], 0:4], bboxs[rank[j], 0:4])
                    if iou > iou_thresh:
                        bbox_cls_spec_conf[rank[j], c] = 0

    # exclude cls-specific confidence score=0
    bboxs = bboxs[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    ret = torch.ones((bboxs.size()[0], 6))

    # return null
    if bboxs.size()[0] == 0:
        return torch.tensor([])

    # bbox coord
    ret[:, 0:4] = bboxs[:, 0:4]
    # bbox class-specific confidence scores
    ret[:, 4] = torch.max(bbox_cls_spec_conf, dim=1).values
    # bbox class
    ret[:, 5] = torch.argmax(bboxs[:, 5:], dim=1).int()
    
    return ret






def display_image(img, bbox):
    
    n = bbox.size()[0]
    bbox = bbox.detach().numpy()
    height, width = img.shape[0], img.shape[1]
    img = cv2.UMat(img.cpu().numpy())
    
    for i in range(n):
        p1 = (int((bbox[i, 0] - bbox[i, 2] / 2) * width), int((bbox[i, 1] - bbox[i, 3] / 2) * height))
        p2 = (int((bbox[i, 0] + bbox[i, 2] / 2) * width), int((bbox[i, 1] + bbox[i, 3] / 2) * height))
        class_name = class_names[int(bbox[i, 5])] 
        
        cv2.rectangle(img, (p1[0]+10,p1[1]-25), p2, color = voc_color[class_name], thickness=4)
        
        cv2.rectangle(img, (p1[0]+10,p1[1]-25), (p1[0]+100,p1[1]+10), color = voc_color[class_name], thickness=-1)
        cv2.putText(img, class_name, (p1[0]+10,p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,  color = (255,255,255), thickness=2)


    return cv2.UMat.get(img)