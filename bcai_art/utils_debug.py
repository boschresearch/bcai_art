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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb

def l2_norm_or_zero_4none(x):
    return torch.norm(x, p=2) if x is not None else 0


def print_model_grad_l2(model):
    """Print all mode gradients L2-norms (or None)"""
    print('Model param GRAD L2 norms')
    for name, param in model.named_parameters():
        print(name, '%.7f' % l2_norm_or_zero_4none(param.grad))


def print_reqgrads(model):
    """Print all parameter gropu names and whether they require gradidents"""
    for name, param in model.named_parameters():
        print(name, param.requires_grad)


def visualize_attack(X, y, X_perturbed, preds):
    """
    Plot and show original and perturbed samples for debugging purposes:
    The batch size must be 64.
    """
    assert X.size(0) == 64
    originals = X.permute(0,2,3,1).data.cpu().numpy()
    perturbed = X_perturbed.permute(0,2,3,1).data.cpu().numpy()
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(1, 2, wspace=0.2)
    for i in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(8, 8, subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        for j in range(64):
            ax = plt.Subplot(fig, inner[j])
            #t.set_ha('center')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                t = ax.text(0.5,0.5, '%d' % y[j])
                ax.imshow(np.squeeze(originals[j]))
            else:
                t = ax.text(0.5,0.5, '%d' % preds[j])
                ax.imshow(np.squeeze(perturbed[j]))
            fig.add_subplot(ax)

    plt.show()
