import torch.nn as nn
from torchvision import models


class ResNet_50(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(ResNet_50, self).__init__()
        resnet = models.resnet50(pretrained)
        # jkdfjvbbmdgbhmkgfbhrt    stride 2-1
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer2.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)



    def forward(self, x):
        x0 = self.layer0(x)   # 1/2, 64
        x0 = self.maxpool(x0)  # 1/4, 256
        x1 = self.layer1(x0)   # 1/4, 256
        x2 = self.layer2(x1)   # 1/8, 512
        x3 = self.layer3(x2)    # 1/16, 1024
        x4 = self.layer4(x3)    # 1/32, 2048

        return x1, x2, x3, x4
