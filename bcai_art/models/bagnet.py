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
import torch.nn.functional as F

import torch




class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class BagNetPatch(nn.Module):
    def __init__(self, dim_per_group, depth, depth_mlp=3, patch_size=7, kernel_size=3, in_chans=3, num_classes=1000, activation=nn.GELU, input_size=224, dim=0,mlp_only=False, **kwargs):
        super().__init__()
        assert kernel_size <= patch_size

        self.padding = None
        padding_total = 0
        if not input_size % patch_size == 0:
            padding_total = patch_size - input_size % patch_size
            self.padding = (int(padding_total/2), padding_total - int(padding_total/2),int(padding_total/2), padding_total - int(padding_total/2))

            

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = int((input_size + padding_total) / patch_size)**2

        if dim == 0:
            dim = dim_per_group * self.num_patches
        self.num_features = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, patch_size, patch_size))

        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans*self.num_patches, dim, kernel_size, groups=self.num_patches, padding=int((kernel_size-1)/2)),
            activation(),
            nn.BatchNorm2d(dim)
        )

        self.blocks = nn.Sequential(
            *[Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=self.num_patches, padding=int((kernel_size-1)/2)),
                activation(),
                nn.BatchNorm2d(dim)
                )) 
                for i in range(depth)
            ]
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

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


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

    def forward_features(self, x):
        if self.padding is not None:
            x = F.pad(x, self.padding, "constant",0)
        x = F.unfold(x, kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size).transpose(-2, -1)
        x = x.reshape(x.size(0), -1, self.patch_size, self.patch_size)
        x = self.stem(x)
        x += self.pos_embedding

        x = self.blocks(x)
        x = self.pooling(x)
        # x = x.reshape(x.size(0), -1, self.num_patches)
        
        return x

    def forward(self, x):
        if not self.mlp_only:
            x = self.forward_features(x)
        x = self.mlp(x)
        x = self.head(x)

        return x



def bagnet_32_patch_4_20_3_7(model_arch, num_classes, use_pretrained=False, add_args=None, inner_model=None, **kwargs):
    model = BagNetPatch(dim_per_group=4, depth=20, depth_mlp=3, patch_size=7, num_classes=num_classes, input_size=32, **kwargs)
    return model


def bagnet_32_patch_1024_6_2_7(model_arch, num_classes, use_pretrained=False, add_args=None, inner_model=None, **kwargs):
    model = BagNetPatch(dim=1024, depth=6, depth_mlp=2, patch_size=7, num_classes=num_classes, input_size=32 **kwargs)
    return model
