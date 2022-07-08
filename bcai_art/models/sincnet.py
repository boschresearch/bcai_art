#
# This source code is an adapted SincNet model:
#   https://github.com/mravanelli/SincNet
# our version is based on a TwoSix Armory repo
#   https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/sincnet_full.py
#
# Copyright (c) 2019  Mirco Ravanelli, TwoSix Labs
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
#
import torch
from torch import nn

from bcai_art.models.sincnet_dnn_models import SincWrapper
from bcai_art.datasets.twosix_librispeech import WINDOW_LENGTH, WINDOW_SAMPLE_RATE

def create_sincnet(class_qty, weights_file=None):
    pretrained = weights_file is not None
    if pretrained:
        model_params = torch.load(weights_file, map_location='cpu')
        print('Loaded pre-trained weights from', weights_file)
    else:
        model_params = {}
    CNN_params = model_params.get("CNN_model_par")
    DNN1_params = model_params.get("DNN1_model_par")
    DNN2_params = model_params.get("DNN2_model_par")

    # from SincNet/cfg/SincNet_dev_LibriSpeech.cfg
    cnn_N_filt = [80, 60, 60]
    cnn_len_filt = [251, 5, 5]
    cnn_max_pool_len = [3, 3, 3]
    cnn_use_laynorm_inp = True
    cnn_use_batchnorm_inp = False
    cnn_use_laynorm = [True, True, True]
    cnn_use_batchnorm = [False, False, False]
    cnn_act = ["relu", "relu", "relu"]
    cnn_drop = [0.0, 0.0, 0.0]

    fc_lay = [2048, 2048, 2048]
    fc_drop = [0.0, 0.0, 0.0]
    fc_use_laynorm_inp = True
    fc_use_batchnorm_inp = False
    fc_use_batchnorm = [True, True, True]
    fc_use_laynorm = [False, False, False]
    fc_act = ["leaky_relu", "linear", "leaky_relu"]

    class_lay = [class_qty]
    class_drop = [0.0, 0.0]
    class_use_laynorm_inp = True
    class_use_batchnorm_inp = False
    class_use_batchnorm = [False]
    class_use_laynorm = [False]
    class_act = ["softmax"]

    CNN_options = {
        "input_dim": WINDOW_LENGTH,
        "fs": WINDOW_SAMPLE_RATE,
        "cnn_N_filt": cnn_N_filt,
        "cnn_len_filt": cnn_len_filt,
        "cnn_max_pool_len": cnn_max_pool_len,
        "cnn_use_laynorm_inp": cnn_use_laynorm_inp,
        "cnn_use_batchnorm_inp": cnn_use_batchnorm_inp,
        "cnn_use_laynorm": cnn_use_laynorm,
        "cnn_use_batchnorm": cnn_use_batchnorm,
        "cnn_act": cnn_act,
        "cnn_drop": cnn_drop,
        "pretrained": pretrained,
        "model_params": CNN_params,
    }

    DNN1_options = {
        "fc_lay": fc_lay,
        "fc_drop": fc_drop,
        "fc_use_batchnorm": fc_use_batchnorm,
        "fc_use_laynorm": fc_use_laynorm,
        "fc_use_laynorm_inp": fc_use_laynorm_inp,
        "fc_use_batchnorm_inp": fc_use_batchnorm_inp,
        "fc_act": fc_act,
        "pretrained": pretrained,
        "model_params": DNN1_params,
    }

    DNN2_options = {
        "input_dim": fc_lay[-1],
        "fc_lay": class_lay,
        "fc_drop": class_drop,
        "fc_use_batchnorm": class_use_batchnorm,
        "fc_use_laynorm": class_use_laynorm,
        "fc_use_laynorm_inp": class_use_laynorm_inp,
        "fc_use_batchnorm_inp": class_use_batchnorm_inp,
        "fc_act": class_act,
    }

    sincNet = SincWrapper(DNN2_options, DNN1_options, CNN_options)

    if pretrained:
        sincNet.eval()
        sincNet.load_state_dict(DNN2_params)

    else:
        sincNet.train()

    return sincNet
