# -*- coding: utf-8 -*-
"""
@author: mopju
"""



import torch
from torch import nn




class Reorganize(nn.Module):
    def __init__(self, stride=2):
        super(Reorganize, self).__init__()
        
        self.stride = stride
        
    def forward(self, x):
        
        batch_size, num_channel, height, width = x.size()
        x = x.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(batch_size, -1, int(height / 2), int(width / 2))
        return x        
   



class darknet19(nn.Module):
    def __init__(self):
        super(darknet19, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3,stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32, 64, 3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, 3,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1,stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, 3,stride=1, padding= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True))
        
        
            
        self.features2 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Conv2d(1024,1000,1,1, padding=1),
            nn.AvgPool2d(13),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        
        x = self.classifier(self.features2(self.features(x)))
        x = x.view(x.shape[0],x.shape[1])
       
        return x
    
    
    
if __name__ == "__main__":
    
    x = torch.randn(1,3,416,416)
    darknet = darknet19()
    print(darknet(x).size()) #[1,1000]
    