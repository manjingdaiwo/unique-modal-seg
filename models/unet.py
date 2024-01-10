import math
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from models.utils.misc import initialize_weights
from models.mobilenet import MobileNetV2
from models.aspp import ASPP
from models.ResNet_50 import ResNet_50
from models.attention import WindowAttention
from models.attention import SEBlock


"""33"""
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class Mynet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(Mynet, self).__init__()
        self.cnn_branch = ResNet_50(in_channels, num_classes, pretrained=True)
        # self.cnn_branch = MobileNetV2(16, nn.BatchNorm2d)
        self.pool =nn.AvgPool2d(2)

        BatchNorm = nn.BatchNorm2d
        self.aspp = ASPP(BatchNorm, 2048)

        self.cbr1 = ConvBNReLU(1024, 256, 1, 1)
        self.cbr0 = ConvBNReLU(2048, 256, 1, 1)
        self.cbr2 = ConvBNReLU(512, 256, 1, 1)    

        self.conv1 = ConvBNReLU(512, 256, 1, 1)        
        self.conv2 = ConvBNReLU(512, 256, 1, 1)        
        self.conv3 = ConvBNReLU(512, 256, 1, 1)


        # num_heads = [3, 6, 12, 24]
        # self.wmsa1 = WindowAttention(256, 8, num_heads[3])        
        # self.wmsa2 = WindowAttention(256, 8, num_heads[2])        
        # self.wmsa3 = WindowAttention(256, 8, num_heads[1])        
        # self.wmsa4 = WindowAttention(256, 8, num_heads[0])        

        self.classifier = nn.Sequential(ConvBNReLU(256, 256),
                                               nn.Dropout2d(p=0.1, inplace=True),
                                               Conv(256, num_classes, kernel_size=1))


        initialize_weights(self.classifier)

    def forward(self, x):
        x_size = x.size()


        x1, x2, x3, x4 = self.cnn_branch(x)   #encoder

        """resnet
        x0, 1/2, 64/16
        x1, 1/4, 256/24
        x2, 1/8, 512/32
        x3, 1/16, 1024/320
        """

        x1_size = x1.size()
        x2_size = x2.size()
        x3_size = x3.size()

        """decoder"""
        f_high4 = self.cbr0(x4) #1/32, 256
        f_high4 = F.interpolate(f_high4, x3_size[2:], mode='bilinear', align_corners=True)  #1/16, 256

        h4 = F.interpolate(f_high4, x_size[2:], mode='bilinear', align_corners=True)  #1/4, 256

        f_high3 = self.cbr1(x3)  #1/16, 256 
        f_high3 = torch.cat((f_high4, f_high3), dim=1)    #1/16, 512
        f_high3 = self.conv1(f_high3)    #1/16, 256

        h3 = F.interpolate(f_high3, x_size[2:], mode='bilinear', align_corners=True)  #1/4, 256

        f_high2 = self.cbr2(x2)  #1/8, 256 
        f_high3 = F.interpolate(f_high3, x2_size[2:], mode='bilinear', align_corners=True)  #1/8, 256
        f_high2 = torch.cat((f_high2, f_high3), dim=1)    #1/8, 512
        f_high2 = self.conv2(f_high2)    #1/8, 256

        h2 = F.interpolate(f_high2, x_size[2:], mode='bilinear', align_corners=True)  #1/16, 256

        f_high2 = F.interpolate(f_high2, x1_size[2:], mode='bilinear', align_corners=True)  #1/4, 256
        f_high1 = torch.cat((f_high2, x1), dim=1)    #1/4, 512
        f_high1 = self.conv3(f_high1)    #1/4, 256

        out = self.classifier(f_high1)  # 1/4, num_class
        
        
        
        out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
        
        
        return out