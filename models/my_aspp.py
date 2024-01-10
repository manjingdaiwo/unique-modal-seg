import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, BatchNorm, inplanes):
        super(ASPP, self).__init__()
        dilations = [1, 3, 6, 9]

        inplanes = inplanes
        self.aspp0 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.aspp1 = _ASPPModule(inplanes, 256, 3, padding=dilations[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x0 = self.aspp0(x) #1x1
        x1 = self.aspp1(x) #rate=1
        x2 = self.aspp2(x) #rate=3
        x3 = self.aspp3(x) #rate=6
        x4 = self.aspp4(x) #rate=9
        x5 = self.global_avg_pool(x) #GAP
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)



        x1_3 = x1 + x2
        x1_3_6 = x1_3 + x3
        x1_3_6_9 = x1_3_6 + x4

        x_aspp = x1 + x1_3 +x1_3_6 + x1_3_6_9


        x_aspp_1x1 = torch.mul(x0, x_aspp)

        x = x5 + x_aspp_1x1


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(BatchNorm):
    return ASPP(BatchNorm)




# class ASPP(nn.Module):
#     def __init__(self, BatchNorm):
#         super(ASPP, self).__init__()
#         dilations = [2, 3]
#
#         inplanes = 2048
#         self.aspp0 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
#         self.aspp1 = _ASPPModule(inplanes, 256, 3, padding=dilations[0], dilation=dilations[0], BatchNorm=BatchNorm)
#         self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
#
#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
#                                              BatchNorm(256),
#                                              nn.ReLU())
#
#         self.conv1 = nn.Conv2d(256, 256, 1, bias=False)
#         self.bn1 = BatchNorm(256)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self._init_weight()
#
#     def forward(self, x):
#         x0 = self.aspp0(x) #1x1
#         x1 = self.aspp1(x) #rate=1
#         x2 = self.aspp2(x) #rate=3
#         x5 = self.global_avg_pool(x) #GAP
#         x5 = F.interpolate(x5, size=x0.size()[2:], mode='bilinear', align_corners=True)
#
#
#
#         x2_3 = x1 + x2
#
#
#         x_aspp = x1 + x2_3
#
#
#         x_aspp_1x1 = torch.mul(x0, x_aspp)
#
#         x = x5 + x_aspp_1x1
#
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         return self.dropout(x)
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
# def build_aspp(BatchNorm):
#     return ASPP(BatchNorm)