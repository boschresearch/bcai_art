#
# BCAI ART : Bosch Center for AI Adversarial Robustness Toolkit
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import torch.nn as nn
import torch
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerPatch(nn.Module):
    def __init__(self, dim, depth, depth_mlp=3, patch_size=7, in_chans=3, num_classes=1000, input_size=224, activation=nn.GELU, mlp_only =False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )
        self.padding = None
        padding_total = 0
        if not input_size % patch_size == 0:
            padding_total = patch_size- input_size % patch_size
            self.padding = (int(padding_total/2), padding_total - int(padding_total/2),int(padding_total/2), padding_total - int(padding_total/2))


        n_patch = int((input_size+padding_total)/patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, n_patch, n_patch))
        self.blocks = nn.Sequential(
            *[
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2d(dim)
                )
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ) for i in range(depth_mlp)]
        )

        self.mlp_only = mlp_only

    def get_classifier(self):
        return self.head

    def feature_upper_lower(self):
        upper = None
        lower = None
        for name, layer in self.pooling.named_modules():
            if isinstance(layer, nn.Tanh):
                upper = 1
                lower = -1
            elif isinstance(layer, nn.Sigmoid):
                upper = 1
                lower = 0
        
        if upper == None or lower == None:
            return None

        upper = upper * torch.ones(1,self.num_features)
        lower = lower * torch.ones(1,self.num_features)

        return (upper, lower)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if self.padding is not None:
            x = F.pad(x, self.padding, "constant",0)
        x = self.stem(x)
        x += self.pos_embedding
        x = self.blocks(x)
        x = self.pooling(x)
        
        return x

    def forward(self, x):

        if not self.mlp_only:
            x = self.forward_features(x)
            
        x = self.mlp(x)
        x = self.head(x)

        return x
