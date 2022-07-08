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

from torch import nn
from torch.autograd import Variable
import torch

    
class VAE(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_downsample=4, n_res = 3):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_downsample, n_res, dim, input_dim)
        self.decoder = Decoder(n_downsample, n_res, self.encoder.output_dim, input_dim)
        self.training = False
        
    def forward(self, x):
        
        hiddens = self.encoder(x)
        if self.training == True:
            #we assume latent spaces are uncorrelated with unit variance
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            x_recon = self.decoder(hiddens + noise)
        else:
            x_recon = self.decoder(hiddens)
            
        return x_recon
    
class Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, dim, input_dim):
        super(Encoder, self).__init__()
        self.model = []
        self.model += [ConvBatchNorm(input_dim, dim, 7, 1, 3)]
        # downsample and conv
        for ii in range(n_downsample):
            self.model += [ConvBatchNorm(dim, 2 * dim, 4, 2, 1)]
            dim *= 2
            
        # residual blocks
        for jj in range(n_res):
            self.model += [ResBlock(dim)]
            
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim):
        super(Decoder, self).__init__()

        self.model = []
        # residual blocks
        for jj in range(n_res):
            self.model += [ResBlock(dim)]
        # upsampling blocks
        for ii in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           ConvBatchNorm(dim, dim // 2, 5, 1, 2)]
            dim //= 2

        self.model += [ConvBatchNorm(dim, output_dim, 7, 1, 3, activation='tanh', bn=False)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

"""
basic blocks
"""

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
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=0, activation='relu', bn=True, bn_weight_init=1.0, **kw):
        super(ConvBatchNorm, self).__init__()
        self.conv= nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)
        self.batchnorm = batch_norm(c_out, bn_weight_init=bn_weight_init, **kw) 
        self.bn = bn
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            # Tanh causes weird problems sometimes, failing with a weird error (it has no in-place computation!):
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
            # [torch.cuda.FloatTensor [48, 3, 112, 112]], which is output 0 of TanhBackward, is at version 1; expected version 0 instead.
            # Hint: enable anomaly detection to find the operation that failed to compute its gradient,
            # with torch.autograd.set_detect_anomaly(True).
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise Exception("unsupported activation")

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batchnorm(x)
        if self.activation:
            return self.activation(x) 
        
        return x
    
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        model = []
        model += [ConvBatchNorm(dim ,dim, 3, 1, 1)]
        model += [ConvBatchNorm(dim ,dim, 3, 1, 1, activation='none')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
