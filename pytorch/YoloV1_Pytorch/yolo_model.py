# -*- coding: utf-8 -*-
"""
@author: Ambroise M
"""


from yolo_layers import *

import torch 
from torch import nn
from torchsummary import summary

class Yolo(nn.Module):
    def __init__(self, feature_size=7, n_boxes=2, n_classes=20):
        super(Yolo, self).__init__()
        
        #feactures extractor
        self.features = Features()
        
        #double conv block
        self.cnn_bloc = CNN_Block()
        
        #Fully connected part
        self.fully_connected = FC(S=feature_size, B=n_boxes, C=n_classes)
        
    def forward(self, x):
        
        x = x.float()
        x = self.features(x)
        x = self.cnn_bloc(x)
        x = self.fully_connected(x) #[n x S * S * (5 * B + C) ]
        
        #reshape --> [n x 7 X 7 X 30 ]
        n = x.shape[0]
        x = torch.reshape(x, (n,7,7,30))#  --> [n x 7 X 7 X 30]
        
        return x
    
    








