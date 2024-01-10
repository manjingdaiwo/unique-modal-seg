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

def fuse(x, st_x):
    """x:c2, st_x:c1"""
    _,st_c, _,_ = st_x.size()
    _,c, _,_ = x.size()
    weight = nn.Sequential(

        nn.Conv2d(st_c, st_c // 4, 1),
        nn.BatchNorm2d(st_c // 4),
        nn.ReLU(),
        nn.Conv2d(st_c//4, 1, 1),
        nn.BatchNorm2d(1),
        nn.ReLU()).cuda()(st_x)
    weight = nn.Sigmoid()(weight)
        
    weight_x = weight * x
    out = weight_x + x
    return out

class Mynet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(Mynet, self).__init__()
        self.cnn_branch = ResNet_50(in_channels, num_classes, pretrained=True)
        # self.cnn_branch = MobileNetV2(16, nn.BatchNorm2d)
        self.pool =nn.AvgPool2d(2)

        BatchNorm = nn.BatchNorm2d
        self.aspp = ASPP(BatchNorm, 2048)

        self.conv1 = nn.Sequential(nn.Conv2d(1280, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())


        self.se1 = SEBlock(1280)
        self.se2 = SEBlock(768)
        self.se3 = SEBlock(512)

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
        x1 , x2, x3, x4 = self.cnn_branch(x)  #resnet_50 features

        ST_x = self.st(x)  # swin_transformer features
        st_x1, st_x2, st_x3, st_x4 = ST_x[0], ST_x[1], ST_x[2], ST_x[3]

        """resnet
        x1, 1/4, 256/16
        x2, 1/8, 512/24
        x3, 1/16, 1024/32
        x4, 1/32, 2048/320
        """

        """swin
        st_x1,1/4, 96
        st_x2,1/8, 192
        st_x3,1/16, 384
        st_x4,1/32, 768

        """
        x2_size = x2.size()
        x3_size = x3.size()
        x1_size = x1.size()

        fuse3 = fuse(x4, st_x4)  #1/32, 2048
        fuse3 = self.aspp(fuse3)  # 1/32, 256

        f_high2 = F.interpolate(fuse3, x3_size[2:], mode='bilinear', align_corners=True)  # 1/16, 256
        fuse2 = fuse(x3, st_x3)    #1/16, 1024
        fuse2 = torch.cat((fuse2, f_high2), dim=1)   #1/16, 1280
        fuse2 = self.se1(fuse2)
        fuse2 = self.conv1(fuse2) #1/16, 256

        f_high1 = F.interpolate(fuse2, x2_size[2:], mode='bilinear', align_corners=True)  # 1/8, 256
        fuse1 = fuse(x2, st_x2)    #1/8, 512
        fuse1 = torch.cat((f_high1, fuse1), dim=1)   #1/4, 768
        fuse1 = self.se2(fuse1)
        fuse1 = self.conv2(fuse1)  #1/4, 256

        out1 = self.classifier1(fuse1)   #1/4, 7
        out1 = F.interpolate(out1, x_size[2:], mode='bilinear', align_corners=True)

        f_high0 = F.interpolate(fuse1, x1_size[2:], mode='bilinear', align_corners=True)  # 1/4, 256
        fuse0 = fuse(x1, st_x1)  # 1/4
        fuse0 = torch.cat((f_high0, fuse0), dim=1)  #1/4, 320
        fuse0 = self.se3(fuse0)
        fuse0 = self.conv3(fuse0)

        out = self.classifier(fuse0)  # 1/2, 7
        out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)

        return out+out1