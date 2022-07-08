#
# This source code is from UNet/FCN PyTorch
#   https://github.com/usuyama/pytorch-unet
#
# Copyright (usuyama) Naoto Usuyama
#
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
#

import torch
from torch import nn
from torchvision import models
import pdb

class FCN(nn.Module):
    """Fully-convolutional neural network:

        code from https://github.com/usuyama/pytorch-unet with MIT license
    """

    def __init__(self):
        super(FCN, self).__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*layers[:5]) # size=(N, 64, x.H/2, x.W/2)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.layer3 = layers[6]  # size=(N, 256, x.H/8, x.W/8)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.layer4 = layers[7]  # size=(N, 512, x.H/16, x.W/16)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, 3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)

        merge = torch.cat([up1, up2, up3, up4], dim=1)
        out = self.conv1k(merge)
        
        return out


class FCNSmall(nn.Module):
    """Fully-convolutional neural network, a smaller variant:

        code based on https://github.com/usuyama/pytorch-unet with MIT license
    """

    def __init__(self):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*layers[:5]) # size=(N, 64, x.H/2, x.W/2)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        self.conv1k = nn.Conv2d(64 + 128, 3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)

        merge = torch.cat([up1, up2], dim=1)
        out = self.conv1k(merge)
        
        return out


class FCNSmall3(nn.Module):
    """Fully-convolutional neural network, a smaller variant:

        code based on https://github.com/usuyama/pytorch-unet with MIT license
    """

    def __init__(self):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*layers[:5]) # size=(N, 64, x.H/2, x.W/2)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        self.conv1k = nn.Conv2d(64 + 128, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)

        merge = torch.cat([up1, up2], dim=1)
        out = self.conv1k(merge)
        
        return out
