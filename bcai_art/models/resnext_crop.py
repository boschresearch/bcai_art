#
# This source code is from MARS:
#   https://github.com/craston/MARS
#
# Copyright (c) 2019 craston
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
#
from bcai_art.utils_patch import gen_crops
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

#__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, bias=False,padding=1,stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
       # out = self.relu(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 input_channels=3,
                 output_layers=[], avpoolbool=True,  position_encoding =False,
                 crops_width=0, crops_height=0,num_crops = 0, encode_emb = True):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        
        self.pe = None
        self.encode_emb = True
        if position_encoding:
            self.encode_emb = encode_emb
            self.num_crops = num_crops
            if self.encode_emb:
                self.pe = nn.Parameter(torch.randn(self.inplanes,int(sample_size/2),int(sample_size/2)))
                self.crops_height = int(crops_height/2)
                self.crops_width = int(crops_width/2)          
                last_size_w = max(int(math.ceil(crops_width / 16))-1,1)
                last_size_h = max(int(math.ceil(crops_height / 16))-1,1)
            else:
                self.pe = nn.Parameter(torch.randn(input_channels,int(sample_size),int(sample_size)))
                self.crops_height = int(crops_height)
                self.crops_width = int(crops_width)
                last_size_w = max(int(math.ceil(crops_width / 32))-1,1)
                last_size_h = max(int(math.ceil(crops_height / 32))-1,1)
        else:
            last_size_w = max(int(math.ceil(crops_width / 32))-1,1)
            last_size_h = max(int(math.ceil(crops_height / 32))-1,1)

        self.sample_duration = sample_duration
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size_w, last_size_h), stride=1)
        self.avpoolbool = avpoolbool
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        #self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        # layer to output on forward pass
        self.output_layers = output_layers

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def sample_add_pe(self,x):
        # _, _,frame_len,_,_ = x.shape
        x, pe_crops = gen_crops(imgTensor=x,  num_crops=self.num_crops, crops_size_x=self.crops_width, crops_size_y=self.crops_height, position_embedding=self.pe)
        #pe_crops is (batch*num_crops)xemb_lenxcrop_widthxcrop_height
        x = x + torch.unsqueeze(pe_crops,2).repeat(1,1,self.sample_duration,1,1,)

        return x

    def forward(self, x):
        # pdb.set_trace()
        if (self.pe is not None) and (not self.encode_emb):
           x = self.sample_add_pe(x)

        x = self.conv1(x)
        if self.pe is not None and self.encode_emb:
           x = self.sample_add_pe(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.avpoolbool:
            x5 = self.avgpool(x4)
        else:
            x5 = x4

        x6 = x5.view(x5.size(0), -1)
        x7 = self.fc(x6)

        if len(self.output_layers) == 0:
            return x7
        else:
            out = []
            out.append(x7)
            for i in self.output_layers:
                if i == 'avgpool':
                    out.append(x6)
                if i == 'layer4':
                    out.append(x4)
                if i == 'layer3':
                    out.append(x3)

        return out

    def freeze_batch_norm(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m,
                                                           nn.BatchNorm3d):  # PHIL: i Think we can write just  "if isinstance(m, nn._BatchNorm)
                m.eval()  # use mean/variance from the training
                m.weight.requires_grad = False
                m.bias.requires_grad = False


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    print("Layers to finetune : ", ft_module_names)

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def resnet18_nopool(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNeXt(ResNeXtBasicBlock, [2, 2, 2, 2], avpoolbool=False, **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNeXt(ResNeXtBasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNeXt(ResNeXtBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50_nopool(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], avpoolbool=False, **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
