# -*- coding: utf-8 -*-


"""
@author: Ambroise M
"""


import torch
from torch import  nn



class double_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
             nn.ReLU(),
             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
              nn.ReLU()
             )
    
    def forward(self, x):
        return self.double_conv(x)
    
    
     
class mini_encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.double_conv = double_conv(in_channel, out_channel)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self,x):
        
        conv_output = self.double_conv(x)
        return conv_output, self.pool(conv_output)
    
    
       
class mini_decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.double_conv = double_conv(in_channel, out_channel)
        
    def forward(self,input_decoder, previous_conv):
        
        deconv = self.deconv(input_decoder)
        concat = torch.cat([deconv, previous_conv],axis=1)

        return self.double_conv(concat)
    
