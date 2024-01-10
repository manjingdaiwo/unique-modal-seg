from cmd import IDENTCHARS
import math
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from models.utils.misc import initialize_weights
# from models.ResNet_50 import ResNet_50
from models.mobilenet import MobileNetV2
from models.my_aspp import ASPP
from models.attention import cam



class EOB(nn.Module):
    def __init__(self, in_cha=768):
        super(EOB, self).__init__()
        # mobilenet
        # self.conv1 = nn.Conv2d(in_channels=320, out_channels=32, padding = 1, kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1)

        #resnet_50
        self.conv1 = nn.Conv2d(in_channels=in_cha, out_channels=256, padding = 1, kernel_size=3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64,momentum=0.95),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
            )



    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)


        y1 = torch.argmax(y, dim=1)
        y1 = torch.unsqueeze(y1, 1)

        y_out = y1.expand_as(x) * x
        y_out = y_out + x
        return  y_out, y
