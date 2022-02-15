# -*- coding: utf-8 -*-
"""
@author: Ambroise M
"""

import torch 
from torch import nn



class Features(nn.Module):
    def __init__(self)  :
        super(Features, self).__init__()
        
        self.extract = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(2,stride=2),
            #---------------------------------
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2,stride=2),
            #---------------------------------
            nn.Conv2d(192, 128, 1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.MaxPool2d(2,stride=2),
            #---------------------------------
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.MaxPool2d(2,stride=2),
            #---------------------------------
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(512, 1024, 3, padding=1)
            )
        
    def forward(self,x): 
        return self.extract(x)
        
        
    
        
class CNN_Block(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024):
        super(CNN_Block, self).__init__()
        
        self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),

                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
            )
        
    def forward(self,x):
        return self.conv_block(x)
    
    
    
    
    
class FC(nn.Module):
    def __init__(self, S, B, C):
        super(FC,self).__init__()
        
        self.fully = nn.Sequential(
        nn.Flatten(),
        nn.Linear(7 * 7 * 1024, 1024),
        nn.Linear(1024, S * S * (5 * B + C))
        )
        
    def forward(self,x): 
        return self.fully(x)
        
        

        
        

    

    