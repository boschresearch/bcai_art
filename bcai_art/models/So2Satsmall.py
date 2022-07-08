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
from bcai_art.models.custom_resnet9 import ConvBatchNorm, Residual, CUSTOM_RESNET9_WEIGHT


class So2SatSubNetSmall(nn.Module):
    """A So2SatNet sub-module."""
    def __init__(self, in_channels):
        """Constructor.

        :param in_channels:  the number of input channels.
        """
        super().__init__()
        channels = {"prep":18, "out":36}
        self.conv1 = ConvBatchNorm(c_in=in_channels, c_out=channels['prep'])
        self.layer1_pool = nn.MaxPool2d(2)
        self.layer1_res = Residual(channels['prep'])
        self.conv2 = ConvBatchNorm(c_in=channels['prep'], c_out=channels['out'])
        self.layer2_pool = nn.MaxPool2d(2)
        self.layer2_res = Residual(channels['out'])
        """
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        """

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1_pool(x)
        x = self.layer1_res(x)
        #x = self.dropout1(x)
        x = self.conv2(x)
        x = self.layer2_pool(x)
        x = self.layer2_res(x)
        #x = self.dropout2(x)
        return x


class So2SatNetSmall(nn.Module):
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(So2SatNetSmall, self).__init__()
        assert inner_model is None, "inner_model is not supported!"

        self.SAR_model = So2SatSubNetSmall(in_channels=4)

        self.EO_model = So2SatSubNetSmall(in_channels=10)

        self.conv_fusion =  ConvBatchNorm(c_in=72, c_out=120)
        #(in_channels=72, out_channels=120, kernel_size=(4, 4))
        #torch.nn.init.xavier_uniform_(self.conv_fusion.weight)

        self.weight = CUSTOM_RESNET9_WEIGHT
        self.fc = nn.Linear(480, 17)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.dropout = nn.Dropout(0.5)
        


    def forward(self, x):
        x_sar = 128* x[:,0:4,:,:]
        x_eo = 4 * x[:, 4:, :, :]

        y_sar = self.SAR_model(x_sar)
        y_eo = self.EO_model(x_eo)

        y_cat = torch.cat((y_sar, y_eo), dim=1)

        y_cat = torch.flatten(self.conv_fusion(y_cat), start_dim=1)

        y_cat = self.dropout(y_cat)

        return self.fc(y_cat)

class So2SatSubNetSmallSeparate(nn.Module):
    """A So2SatNet sub-module."""
    def __init__(self, in_channels):
        """Constructor.

        :param in_channels:  the number of input channels.
        """
        super().__init__()
        channels = {"prep":36,"out":60}
        self.prep_network = So2SatSubNetSmall(in_channels=in_channels)
        self.conv_fusion =  ConvBatchNorm(c_in=channels["prep"], c_out=channels['out'])
        self.fusion_res = Residual(channels['out'])
        self.weight = CUSTOM_RESNET9_WEIGHT
        self.fc = nn.Linear(240, 17)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.dropout = nn.Dropout(0.5)
        """
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        """

    def forward(self, x):
        x = self.prep_network(x)
        x = self.conv_fusion(x)
        x = self.fusion_res(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)

        return self.fc(x)
        


class So2SatNetSeparate(nn.Module):
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(So2SatNetSeparate, self).__init__()
        assert inner_model is None, "inner_model is not supported!"

        self.SAR_model = So2SatSubNetSmallSeparate(in_channels=4)

        self.EO_model = So2SatSubNetSmallSeparate(in_channels=10)


    def forward(self, x):
        x_sar = 128* x[:,0:4,:,:]
        x_eo = 4 * x[:, 4:, :, :]

        y_sar = self.SAR_model(x_sar)
        y_eo = self.EO_model(x_eo)


        return torch.cat((y_sar, y_eo), dim=0)
