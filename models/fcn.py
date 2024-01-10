import math
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from models.utils.misc import initialize_weights
from models.aspp import ASPP
from models.ResNet_50 import ResNet_50



"""de"""
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class Mynet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(Mynet, self).__init__()
        self.cnn_branch = ResNet_50(in_channels, num_classes, pretrained=True)
     
        self.cbr0 = ConvBNReLU(2048, 256, 1, 1)
        self.classifier = nn.Sequential(ConvBNReLU(256, 256),
                                               nn.Dropout2d(p=0.1, inplace=True),
                                               Conv(256, num_classes, kernel_size=1))

        initialize_weights(self.classifier)

    def forward(self, x):
        x_size = x.size()

        x1, x2, x3, x4 = self.cnn_branch(x)   #encoder
        x2_size = x2.size()

        x4 = self.cbr0(x4)
        out = self.classifier(x4)
        out = F.interpolate(out, x2_size[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)

        return out