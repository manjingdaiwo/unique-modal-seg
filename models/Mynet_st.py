import math
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from models.utils.misc import initialize_weights
from models.ResNet_50 import ResNet_50
from models.mobilenet import MobileNetV2
from models.my_aspp import ASPP
from models.st.swin_transformer import SwinTransformer
from models.attention import SEBlock

def conv1x1_0(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)

def conv1x1_1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)

class Mynet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(Mynet, self).__init__()
        self.cnn_branch = ResNet_50(in_channels, num_classes, pretrained=True)
        # self.cnn_branch = MobileNetV2(16, nn.BatchNorm2d)
        self.pool =nn.AvgPool2d(2)

        BatchNorm = nn.BatchNorm2d
        self.aspp = ASPP(BatchNorm, 768)

        self.conv1 = nn.Sequential(nn.Conv2d(640, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(448, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(352, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

        self.se1 = SEBlock(640)
        self.se2 = SEBlock(448)
        self.se3 = SEBlock(352)

        # transformer
        embed_dim = 96
        depths = [2, 2, 6, 2]
        self.num_layers = len(depths)
        num_heads = [3, 6, 12, 24]
        window_size = 8
        mlp_ratio = 4.
        drop_rate = 0.
        attn_drop_rate = 0.
        use_checkpoint = False

        self.st = SwinTransformer(
            pretrain_img_size=512, 
            patch_size=4, 
            in_chans=3, 
            embed_dim=embed_dim, 
            depths=depths, 
            num_heads=num_heads,
            window_size=window_size, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=True, 
            qk_scale=None,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm, 
            ape=False, 
            patch_norm=True,
            out_indices=(0,1,2,3),
            frozen_stages=-1,
            use_checkpoint=use_checkpoint)

        self.classifier = nn.Sequential(conv1x1_0(256, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1_1(64, num_classes))
        self.classifier1 = nn.Sequential(conv1x1_0(256, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1_1(64, num_classes))
        initialize_weights(self.classifier)

    def forward(self, x):
        x_size = x.size()

        ST_x = self.st(x)  # swin_transformer features
        st_x1, st_x2, st_x3, st_x4 = ST_x[0], ST_x[1], ST_x[2], ST_x[3]

        """swin
        st_x1,1/4, 96
        st_x2,1/8, 192
        st_x3,1/16, 384
        st_x4,1/32, 768

        """
        x2_size = st_x2.size()
        x3_size = st_x3.size()
        x1_size = st_x1.size()

        high_4 = self.aspp(st_x4)  # 1/32, 256

        high_3 = F.interpolate(high_4, x3_size[2:], mode='bilinear', align_corners=True)  # 1/16, 256
        high_3 = torch.cat((st_x3, high_3), dim=1)   #1/16, 640
        high_3 = self.se1(high_3)
        high_3 = self.conv1(high_3) #1/16, 256

        high_2 = F.interpolate(high_3, x2_size[2:], mode='bilinear', align_corners=True)  # 1/16, 256
        high_2 = torch.cat((st_x2, high_2), dim=1)   #1/16, 1280
        high_2 = self.se2(high_2)
        high_2 = self.conv2(high_2) #1/8, 256

        high_1 = F.interpolate(high_2, x1_size[2:], mode='bilinear', align_corners=True)  # 1/16, 256
        high_1 = torch.cat((st_x1, high_1), dim=1)   #1/16, 1280
        high_1 = self.se3(high_1)
        high_1 = self.conv3(high_1) #1/4, 256

        out = self.classifier(high_1)  # 1/4, 7
        out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)

        return out