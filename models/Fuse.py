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



class Fuse(nn.Module):
    def __init__(self, indim1, indim3):
        super(Fuse, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=indim1, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=indim3, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=indim3, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=indim3, out_channels=64, kernel_size=1)
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.cam = cam(k_size=3)
        # self.pool1 = nn.AvgPool2d(2)
        # self.pool2 = nn.AvgPool2d(4)
        # self.pool3 = nn.AvgPool2d(6)
        inplanes = 64

        dilation = [2, 4, 6]
        self.atrous_conv1 = nn.Sequential(nn.Conv2d(inplanes, inplanes , kernel_size=3, stride=1, padding=dilation[0], dilation=dilation[0], bias=False),
                                          nn.BatchNorm2d(inplanes),
                                          nn.ReLU())
        self.atrous_conv2 = nn.Sequential(nn.Conv2d(inplanes, inplanes , kernel_size=3, stride=1, padding=dilation[1], dilation=dilation[1], bias=False),
                                          nn.BatchNorm2d(inplanes),
                                          nn.ReLU())
        self.atrous_conv3 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=dilation[2], dilation=dilation[2], bias=False),
                                          nn.BatchNorm2d(inplanes),
                                          nn.ReLU())
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f_low, f_high, f_st):

        f_cnn = torch.cat((f_low, f_high), dim=1)

        f_cnn = self.cam(f_cnn)

        f_s = self.conv1(f_st)

        f_c1 = self.conv2(f_cnn)
        f_c2 = self.conv3(f_cnn)

        b, c, h, w = f_s.size()

        # f_s = f_s.view(b, -1, w * h).permute(0, 2, 1)
        # f_c1 = f_c1.view(b, -1, w * h)
        # f_lh = torch.bmm(f_s, f_c1)#hw*hw

        f_lh = f_s * f_c1

        
        f_lh = self.softmax(f_lh)

        # f_c2 = f_c2.view(b, -1, w*h)
        # f_lh = torch.bmm(f_c2, f_lh)
        # f_lh = f_lh.view(b, c, h, w)

        f_lh = f_c2 * f_lh

        f_cnn = self.conv4(f_cnn)

        f_out = f_lh + f_cnn

        w_x = f_out

        w_x_2 = self.atrous_conv1(w_x)
        w_x_4 = self.atrous_conv2(w_x)
        w_x_8 = self.atrous_conv3(w_x)

        # w_x_2 = self.pool1(w_x)
        # w_x_4 = self.pool2(w_x)
        # w_x_8 = self.pool3(w_x)
        #
        # w_x_2 = F.interpolate(w_x_2, w_x.size()[2:], mode='bilinear', align_corners=True)
        # w_x_4 = F.interpolate(w_x_4, w_x.size()[2:], mode='bilinear', align_corners=True)
        # w_x_8 = F.interpolate(w_x_8, w_x.size()[2:], mode='bilinear', align_corners=True)

        m_w_x = torch.cat((w_x_2, w_x_4, w_x_8, w_x), dim=1)
        # m_w_x = self.conv5(m_w_x)

        return  m_w_x
