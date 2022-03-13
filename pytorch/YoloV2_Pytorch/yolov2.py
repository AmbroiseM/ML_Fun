# -*- coding: utf-8 -*-
"""
@author: mopju
"""

import torch 
from torch import nn
from darknet import *


class yolov2(nn.Module):
    def __init__(self):
        super(yolov2,self).__init__()
        
        
        self.darknet = darknet19()
        
        #until shortcut
        self.features = self.darknet.features
        
        #reorganize layer
        self.reorganize = Reorganize()
        
        #straight path
        self.features2 = self.darknet.features2
        
        #final conv output
        self.conv_out = nn.Sequential(
            nn.Conv2d(3072, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 25 * 5, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(25*5),
            nn.LeakyReLU(0.1, inplace=True))
        
        
    def forward(self, x):
        
        x1 = self.reorganize(self.features(x))
        x2 = self.features2(self.features(x))    
        x3 = torch.cat([x1, x2], dim=1)
        
        x3 = self.conv_out(x3).permute(0,3,2,1)

        return x3.view(x3.shape[0],x3.shape[1],x3.shape[1],5,25)
    
  
    
  
if __name__ == "__main__":
    
    x = torch.randn(1,3,416,416)
    yolo = yolov2()
    print(yolo(x).shape)
    