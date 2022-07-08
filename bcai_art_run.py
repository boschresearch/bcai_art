#!/usr/bin/env python
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

import argparse

from bcai_art.runner import main, CONFIG_PARAM, OVERRIDES_JSON, EPOCH_QTY_PARAM
from bcai_art.utils_misc import enable_spawn

if __name__ == '__main__':
    #All multiprocessing with CUDA must call this function in the main module.

    enable_spawn()

    # Important notes:
    # 1. Except the number of epochs,
    #    the names of the optional parameters should exactly match the
    # object structure of the JSON config file. For example,
    # the parameter --dataset.root is supposed to override
    # the config_obj.dataset.root variable, where config_obj
    # is the loaded and parsed config.
    #
    # 2. All non-positional parameters must be optional with the default value None
    #
    # 3. The number of epochs affects all the trainer objects

    parser = argparse.ArgumentParser('trainer and evaluator')

    parser.add_argument(CONFIG_PARAM, type=str, metavar='JSON config',
                        help='configuration json file')
    parser.add_argument('--dataset.root', type=str, metavar='data set root',
                        help='data set root ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--model.architecture', type=str, metavar='model arch',
                        help='model architecture/type ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--general.device.device_name', type=str, metavar='device name',
                        help='device name, e.g., cpu, cuda:1 ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--general.dist_backend', type=str, metavar='distributed backend',
                        help='a distributed backend type ' + OVERRIDES_JSON, default='gloo')
    parser.add_argument('--general.base_seed', type=int, metavar='base random seed',
                        help='base random seed ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--general.master_port', type=int, metavar='master port',
                        help='master port (for || processing) ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--general.device.device_qty', type=int, metavar='# of CUDA',
                        help='Number of CUDA devices to use ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--model.weights_file', type=str, metavar='model weights',
                        help='load these model weights ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--model.pretrained', type=int, metavar='use pretrained flag (0/1)',
                        help='use pretrained flag (0/1) ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--optim.init_lr', type=float, metavar='init. learn. rate',
                        help='initial learning rate ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--' + EPOCH_QTY_PARAM, type=int, metavar='# of epochs (all trainers)',
                        help='can be used to limit the # of epochs in all trainers', default=None)
    parser.add_argument('--eval_fract', type=float, default=None, metavar='eval. this fract of data',
                        help='eval. this fract of data')
    parser.add_argument('--train_fract', type=float, default=None,
                        metavar='use this fract of data for training',
                        help='a fraction of data that we should use for training')
    parser.add_argument('--eval_only', action='store_true', help='no training: eval. only')

    main(parser.parse_args())

