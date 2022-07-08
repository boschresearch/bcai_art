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

"""
Gradient estimation functions. This file contains some finite difference based gradient 
estimation methods for blackbox attacks.
"""

# from bcai_art.eval import PREDICTION_OUTPUT
# from bcai_art.utils_misc import args_to_paramdict, get_norm_code, calc_correct_qty
# from bcai_art.utils_tensor import project, get_start_delta, START_RANDOM, START_ZERO, apply_func, assert_property, \
#                                     get_frame_shape, get_batched, DATASET_TYPE_VIDEO_FIX_SIZE
# from bcai_art.utils_patch import *
# from bcai_art.conf_attr import ATTACK_EPS_ATTR, ATTACK_NORM_ATTR

import torch
import numpy as np

def estimate_gradient(train_env, X, y, loss, sampling_num, grad_step=1.0):
    """Estimate gradients using random sampling.
    Inspired from https://arxiv.org/abs/1712.09491
    Code modified from https://github.com/sunblaze-ucb/blackbox-attacks.
    The paper presents gradient estimation methods that use finite differences. However, as finite
    differences require O(d) samples, they present random sampling and PCA based sampling of directional derivatives.

    We support random sampling based gradient estimation. 

    Note: Loss needs to be calculated per sample and not batchwise. 

    :param train_env: a training environment 
    :param X: input image (bs x ch x sx x sy)
    :param y: target label or None
    :param loss: Loss function object (should take as input trainenv, X, y). Required to be samplewise loss function (no reduction).
    :param sampling_num: number of directions to sample in per group
    :param grad_step: step size for gradient estimation (proposed to be 1.0 for x-entropy and 0.01 for CW type losses)
    :return: gradient estimate (bs x ch x sx x sy) 
    """
    grad_est = torch.zeros_like(X)
    assert len(X.shape) == 4, "Currently only supports image inputs, but input is of size: " + str(len(X.shape))
    
    bs, ch, sx, sy = X.shape
    # print(X.shape)
    dim = int(sx * sy)
    random_indices = np.random.permutation(dim)
    num_groups = dim // sampling_num
    rows = random_indices // sx
    cols = random_indices % sy
    j_list = range(num_groups)

    for j in j_list:
        curr_indices_row = rows[j*sampling_num:(j+1)*sampling_num]
        curr_indices_col = cols[j*sampling_num:(j+1)*sampling_num]
        v = torch.zeros_like(X)
        v[:,:,curr_indices_row, curr_indices_col] = 1.0
        x_plus = X + grad_step * v
        x_minus = X - grad_step * v
        with torch.no_grad():
            # Since we are estimating gradient with finite differences, we do not need to compute gradients for backprop.
            #print(self.loss(train_env, x_plus, y, reduction='none').shape)
            grad_update = (loss(train_env, x_plus, y) -  loss(train_env, x_minus, y))/(2*grad_step*sampling_num)
            for i in range(bs):
                #TODO: Find a smarter way for this matrix substitution
                grad_est[i,:,curr_indices_row, curr_indices_col] =  grad_update[i]
    return grad_est
