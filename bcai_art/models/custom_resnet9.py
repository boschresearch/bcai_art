#
# A custom ResNet9 class fine-tuned to be trained quickly on CIFAR10 or CIFAR100.
# It is largely a reimplementation of the code from https://github.com/lambdal/cifar10-fast
# with a few classes / functions (e.g., Flatten) copied verbatim.
#
# Copyright (c) 
#    2018 davidcpage
#    2022+ Robert Bosch GmbH
#
# cifar10-fast source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
#
#
import torch.nn as nn

CUSTOM_RESNET9_WEIGHT = 0.125

class Flatten(nn.Module):
    def forward(self, x): 
        #pdb.set_trace()
        #return x.view(x.size(0), x.size(1))
        return x.view(x.size(0), -1)

def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
        
    return m

class ConvBatchNorm(nn.Module):
    def __init__(self, c_in, c_out, bn_weight_init=1.0, **kw):
        super(ConvBatchNorm, self).__init__()
        self.conv= nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = batch_norm(c_out, bn_weight_init=bn_weight_init, **kw) 
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x) 


class Residual(nn.Module):
    def __init__(self, c, **kw):
        super(Residual, self).__init__()
        self.res1 = ConvBatchNorm(c, c, **kw)
        self.res2 = ConvBatchNorm(c, c, **kw)

    def forward(self, x):
        return x + self.res2(self.res1(x))

class CustomResnet9(nn.Module):
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(CustomResnet9, self).__init__()

        assert inner_model is None, "inner_model is not supported!"
        assert inner_model is None, "inner_model is not supported!"
        kw = {}
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}

        self.prep = ConvBatchNorm(3, channels['prep'], **kw)
        self.layer1 = ConvBatchNorm(channels['prep'], channels['layer1'], **kw)
        self.layer1_pool = nn.MaxPool2d(2)
        self.layer1_res = Residual(channels['layer1'])
        self.layer2 = ConvBatchNorm(channels['layer1'], channels['layer2'], **kw)
        self.layer2_pool = nn.MaxPool2d(2)
        self.layer3 = ConvBatchNorm(channels['layer2'], channels['layer3'], **kw)
        self.layer3_pool = nn.MaxPool2d(2)
        self.layer3_res = Residual(channels['layer3'])

        self.pool = nn.MaxPool2d(4)
        self.flatten = Flatten()
        self.linear = nn.Linear(channels['layer3'], num_classes, bias=False)
        self.weight = CUSTOM_RESNET9_WEIGHT

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer1_pool(x)
        x = self.layer1_res(x)
        x = self.layer2(x)
        x = self.layer2_pool(x)
        x = self.layer3(x)
        x = self.layer3_pool(x)
        x = self.layer3_res(x)
        
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.weight * x

    
class CustomResnet9Small(nn.Module):
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(CustomResnet9Small, self).__init__()
        assert inner_model is None, "inner_model is not supported!"
        kw = {}
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}

        self.prep = ConvBatchNorm(3, channels['prep'], **kw)
        self.layer1 = ConvBatchNorm(channels['prep'], channels['layer1'], **kw)
        self.layer1_pool = nn.MaxPool2d(2)
        self.layer1_res = Residual(channels['layer1'])
        self.layer2 = ConvBatchNorm(channels['layer1'], channels['layer2'], **kw)
        self.layer2_pool = nn.MaxPool2d(2)
        self.layer3 = ConvBatchNorm(channels['layer2'], channels['layer3'], **kw)
        self.layer3_pool = nn.MaxPool2d(2)
        self.layer3_res = Residual(channels['layer3'])
        
        self.pool = nn.MaxPool2d(2)
        self.flatten = Flatten()
        self.linear = nn.Linear(channels['layer3'], num_classes, bias=False)
        self.weight = CUSTOM_RESNET9_WEIGHT

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer1_pool(x)
        x = self.layer1_res(x)
        x = self.layer2(x)
        x = self.layer2_pool(x)
        x = self.layer3(x)
        x = self.layer3_pool(x)
        x = self.layer3_res(x)
        #x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.weight * x

