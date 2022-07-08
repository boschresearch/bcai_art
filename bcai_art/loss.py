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

import torch
import pdb

RANDOM_CROP_CLASSVOTE_LOSS = "randomized_crop_hardvote_class"
RANDOM_CROP_SOFTMAXVOTE_LOSS = "randomized_crop_hardvote_softmax"


def get_loss(loss_class):
    if loss_class is None:
        return torch.nn.CrossEntropyLoss
    
    if loss_class == RANDOM_CROP_CLASSVOTE_LOSS:
        return Randomized_crop_hardvote_class_loss
    
    if loss_class == RANDOM_CROP_SOFTMAXVOTE_LOSS:
        return torch.nn.NLLLoss

    
class Randomized_crop_hardvote_class_loss(torch.nn.modules.loss._Loss):
    def __init__(self, weight = None, size_average=None, reduce=None, reduction= 'mean'):
        super(Randomized_crop_hardvote_class_loss, self).__init__()
        self.orig_loss = torch.nn.CrossEntropyLoss(weight=weight,size_average= size_average, reduce=reduce, reduction=reduction)
        
        
    def forward(self, X, y):
        
        num_crops = int(X.shape[0]/y.shape[0])
        y_repeated = torch.repeat_interleave(y, num_crops, dim = 0)
        return self.orig_loss(X, y_repeated)
