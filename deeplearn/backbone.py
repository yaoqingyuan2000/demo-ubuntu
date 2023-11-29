
import sys

from registry.registry import *

import torch
import torch.nn as nn




@MODELS.register_module(name="ResNet")
class ResNet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ResNet, self).__init__()

        print("backbone:", out_channels)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels) 
        torch.nn.init.kaiming_normal_(self.conv.weight,nonlinearity='relu') 
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):

        feats = self.conv(x)
        feats = self.batch_norm(feats)
        feats = torch.relu(feats)
        return feats + x

