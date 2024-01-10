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
from models.Fuse import Fuse
from models.swin_transformer import SwinTransformer
from models.attention import SEBlock

class Mynet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(Mynet, self).__init__()
        self.cnn_branch = ResNet_50(in_channels, num_classes, pretrained=True)
        # self.cnn_branch = MobileNetV2(16, nn.BatchNorm2d)
        self.pool =nn.AvgPool2d(2)

        BatchNorm = nn.BatchNorm2d
        self.aspp = ASPP(BatchNorm, 2048)

        embed_dim = 96
        depths = [2, 2, 6, 2]
        self.num_layers = len(depths)
        num_heads = [3, 6, 12, 24]
        window_size = 8
        mlp_ratio = 4.
        drop_rate = 0.
        attn_drop_rate = 0.
        use_checkpoint = False
        self.st = SwinTransformer(img_size=512, patch_size=4, in_chans=3, num_classes=6,
                 embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=use_checkpoint)



        # self.fuse1 = Fuse(392, 648)
        # self.fuse2 = Fuse(72, 328)
        # self.fuse3 = Fuse(40, 296)
        # self.fuse4 = Fuse(16, 272)

        self.se = SEBlock(24)

        self.fuse1 = Fuse(3840,3392)
        self.fuse2 = Fuse(1792,1088)
        self.fuse3 = Fuse(768,576)
        self.fuse4 = Fuse(256,320)

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=0.95),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        self.classifier1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256, momentum=0.95),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=0.95),
            nn.ReLU(),
            # nn.Dropout2d(0.5),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        self.classifier2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=0.95),
            nn.ReLU(),
            # nn.Dropout2d(0.5),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        self.classifier3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=0.95),
            nn.ReLU(),
            # nn.Dropout2d(0.5),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        self.classifier_out = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(24, num_classes, kernel_size=1)
        )

        initialize_weights(self.classifier, self.classifier1, self.classifier2, self.classifier3)

    def forward(self, x):
        x_size = x.size()

        x0, x1 , x2, x3 = self.cnn_branch(x)
        x0_size = x0.size()
        x1_size = x1.size()
        x2_size = x2.size()
        x3_size = x3.size()

        ST_x = self.st.forward_trans(x)  # swin_transformer features
        st_x0, st_x1, st_x2, st_x3 = ST_x[0], ST_x[1], ST_x[2], ST_x[3]

        """resnet
        x0, 1/4, 256
        x1, 1/8, 512
        x2, 1/16, 1024
        x3, 1/32, 2048

        """
        x0_size = x0.size()
        x1_size = x1.size()
        x2_size = x2.size()

        down_x0 = self.pool(x0) #1/4, 16
        down_st_x0 = self.pool(st_x0) #1/4, 16

        f_low0 = x0
        f_st0 = st_x0

        f_low1 = torch.cat((down_x0, x1), dim=1)
        f_st1 = torch.cat((down_st_x0, st_x1), dim=1)

        mq2 = self.pool(f_low1)
        f_low2 = torch.cat((x2, mq2), dim=1)
        nq2 = self.pool(f_st1)
        f_st2 = torch.cat((st_x2, nq2), dim=1)

        mq3 = self.pool(f_low2)
        f_low3 = torch.cat((x3, mq3), dim=1)
        nq3 = self.pool(f_st2)
        f_st3 = torch.cat((st_x3, nq3), dim=1)

        f_high3_1 = self.aspp(x3) # 1/16, 256
        f_high3_2 = self.aspp(st_x3) # 1/16, 256
        f_high3 = torch.cat((f_high3_1, f_high3_2), dim=1) # 1/16, 256


        f_high2 = self.fuse1(f_low3, f_high3, f_st3) #1/16, 256
        f_high2 = F.interpolate(f_high2, x2_size[2:], mode='bilinear', align_corners=True)  # 1/8, 256

        f_high1 = self.fuse2(f_high2, f_low2, f_st2)#1/8, 256
        f_high1 = F.interpolate(f_high1, x1_size[2:], mode='bilinear', align_corners=True)  # 1/4, 256

        f_high0 = self.fuse3(f_high1, f_low1, f_st1)#1/4, 256
        f_high0 = F.interpolate(f_high0, x0_size[2:], mode='bilinear', align_corners=True)  # 1/2, 256

        out = self.fuse4(f_high0, f_low0, f_st0)#1/2, 256


        head_out = self.classifier(out)  # 1/2,num_class
        w_x3_out = self.classifier1(f_high3)  # 1/16,num_class
        w_x2_out = self.classifier2(f_high2)  # 1/8,num_class
        w_x1_out = self.classifier3(f_high1)  # 1/4,num_class


        head_out = F.interpolate(head_out, x_size[2:], mode='bilinear', align_corners=True)
        w_x1_out = F.interpolate(w_x1_out, x_size[2:], mode='bilinear', align_corners=True)
        w_x2_out = F.interpolate(w_x2_out, x_size[2:], mode='bilinear', align_corners=True)
        w_x3_out = F.interpolate(w_x3_out, x_size[2:], mode='bilinear', align_corners=True)

        f_out = torch.cat((head_out, w_x1_out, w_x2_out, w_x3_out), dim=1)
        f_out = self.se(f_out)
        f_out = self.classifier_out(f_out)

        # return head_out+w_x1_out+w_x2_out+w_x3_out
        return(f_out)