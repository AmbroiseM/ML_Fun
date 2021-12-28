# -*- coding: utf-8 -*-



"""
@author: Ambroise M
"""


from unet_pytorch_layers import * 
import torch
from torch import nn
    
class Unet(nn.Module):
    def __init__(self, n_classes, n_channels):
        super().__init__()
        
        self.n_classes = n_classes
        self.encoder_1 = mini_encoder(n_channels, 64)
        self.encoder_2 = mini_encoder(64, 128)
        self.encoder_3 = mini_encoder(128, 256)
        self.encoder_4 = mini_encoder(256, 512)
        
        self.bottleneck = mini_encoder(512, 1024)
        
        self.decoder_1 = mini_decoder(1024, 512)
        self.decoder_2 = mini_decoder(512, 256)
        self.decoder_3 = mini_decoder(256, 128)
        self.decoder_4 = mini_decoder(128, 64)
        
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        
        c1, p1 = self.encoder_1(x)
        c2, p2 = self.encoder_2(p1)
        c3, p3 = self.encoder_3(p2)
        c4, p4 = self.encoder_4(p3)
        
        bn, _ = self.bottleneck(p4)
        
        up1 = self.decoder_1(bn,c4)
        up2 = self.decoder_2(up1,c3)
        up3 = self.decoder_3(up2,c2)
        up4 = self.decoder_4(up3,c1)
        
        out = self.output(up4)
                
        return out
        
        