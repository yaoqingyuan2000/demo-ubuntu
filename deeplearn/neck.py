import sys

from registry.registry import *

import torch
import torch.nn as nn


@MODELS.register_module(name="GlobalAveragePooling")
class GlobalAveragePooling(nn.Module):

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()

        print("neck")

        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        
        outs = self.gap(inputs)
        outs = outs.view(inputs.size(0), -1)

        return outs