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
from bcai_art.config import load_settings
from autoattack import AutoAttack
from bcai_art.runner import CONFIG_PARAM, OVERRIDES_JSON
from bcai_art.utils_misc import *
from bcai_art.conf_attr import *
from bcai_art.runner import DEFAULT_CUDA_ID
from bcai_art.models_main import *
from bcai_art.datasets_main import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser('auto-attack evaluator for images')

    parser.add_argument(CONFIG_PARAM, type=str, metavar='JSON config',
                        help='configuration json file')
    parser.add_argument('--dataset.root', type=str, metavar='data set root',
                        help='data set root ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--model.architecture', type=str, metavar='model arch',
                        help='model architecture/type ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--general.device.device_name', type=str, metavar='device name',
                        help='device name, e.g., cpu, cuda:1 ' + OVERRIDES_JSON, default=None)
    parser.add_argument('--eval_fract', type=float, default=1.0, metavar='eval. this fract of data',
                        help='eval. this fract of data')
    parser.add_argument('--metric', default="Linf", choices =  ['Linf', 'L2'],
                         metavar='auto-attack metric type',
                        help='metric type')
    parser.add_argument('--eps', type=float, default=8./255, metavar='attack radius',
                        help='the radius of attack (epsilon value)')

    input_args = parser.parse_args()
    input_args_vars=vars(input_args)

    args = load_settings(input_args.config)

    # Some command line parameters should override JSON-config settings
    # TODO, this is a copy-paste from the runner.py (with a minor modification),
    #       we can, in principle, avoid it.
    for attr_path, attr_val in input_args_vars.items():
        # CONFIG_PARAM is a positional argument
        # We override the value only if the argument is not None
        if attr_path != CONFIG_PARAM and attr_val is not None:
            set_nested_attr(args, attr_path, attr_val)


    eval_fract = min(1.0, max(0.0, args.eval_fract))
    metric = args.metric
    eps = args.eps
    print('Testing using the following fraction of the data: %g' % eval_fract)
    print(f'Metric {metric} eps: %g' % eps)

    # Print arguments after modifying
    print(args)

    device_name = getattr(args.general.device, DEVICE_NAME_ATTR, f'cuda:{DEFAULT_CUDA_ID}')
    print('CUDA device:' + device_name)

    _, test_set, dataset_info = create_dataset(args.dataset)
    
    class_qty = dataset_info[DATASET_NUM_CLASSES_PARAM]

    mu = get_shaped_tensor(dataset_info, DATASET_MEAN_PARAM)
    std = get_shaped_tensor(dataset_info, DATASET_STD_PARAM)

    upper_limit = get_shaped_tensor(dataset_info, DATASET_UPPER_LIMIT_PARAM)
    lower_limit = get_shaped_tensor(dataset_info, DATASET_LOWER_LIMIT_PARAM)

    model = create_toplevel_model(class_qty,
                                  mu, std,
                                  args.model)

    model.to(device_name)
    model.eval()
    
    if hasattr(args.model, WEIGHTS_FILE_ATTR) and args.model.weights_file is not None:
        model_file_name = get_snapshot_path(args.model.weights_file, MODEL_SNAPSHOT_PREFIX, SNAPSHOT_SUFFIX)

        print('Loading model from:', model_file_name)
        # beware: a model is actually a wrapper it doesn't fully mimic Pytorch model interface
        
        saved_state = torch.load(model_file_name, map_location='cpu')
        # We always save model in a wrapper dictionary, but this is not the case
        # when we load some externally pre-trained models
        if SAVE_MODEL_KEY in saved_state.keys():
            model.load_state_dict(saved_state[SAVE_MODEL_KEY])
        else:
            model.orig_model.load_state_dict(saved_state)


    tot_qty = len(test_set)
    sample_qty = int(eval_fract * tot_qty)

    np.random.seed(0)
    sel_idx = np.random.choice(np.arange(tot_qty), replace=False, size=sample_qty)
    sel_idx.sort()
    #print(sel_idx[0:50])

    c, h, w = test_set[0][0].shape

    images= torch.zeros((sample_qty, c, h, w), dtype=torch.float32)
    labels= torch.zeros((sample_qty), dtype=torch.int64)

    for i in tqdm(range(sample_qty), 'Copying images'):
        images[i,:,:,:], labels[i]= test_set[sel_idx[i]]

    print('Total # of test images:', tot_qty, ' sampled: ', len(images))
    batch_size = args.evaluation.eval_batch_size
    print('Batch size: ', batch_size)

    adversary = AutoAttack(model.forward_no_loss_comp, norm=metric, eps=eps, version='standard', seed=0)
    x_adv = adversary.run_standard_evaluation(images, labels, bs=batch_size)
