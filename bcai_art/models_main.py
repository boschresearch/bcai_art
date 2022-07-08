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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time

from bcai_art.models.custom_resnet9 import CustomResnet9, CustomResnet9Small
from bcai_art.models.sincnet import create_sincnet
from bcai_art.models.vae import VAE
from bcai_art.models.fcn import FCN, FCNSmall, FCNSmall3
from bcai_art.models.unet import UNet
from bcai_art.models.dncnn import DnCNN
from bcai_art.utils_tensor import apply_func_squeeze, apply_func, \
                                    get_frame_channel_height_width, get_frame_shape, DATASET_TYPE_IMAGE

from bcai_art.utils_misc import args_to_paramdict, delete_selected_args, get_trainable_param_count

from bcai_art.conf_attr import FREEZE_FEATURES_MODEL_ATTR, FREEZE_INNER_MODEL_ATTR, LOSS_ATTR, LOSS_NATIVE, \
                                ADD_ARG_ATTR, USE_PRETRAINED_ATTR, \
                                INNER_MODEL_ATTR, MODEL_ARCH_ATTR, MODEL_ARGS_KEY

from bcai_art.loss import get_loss
from bcai_art.utils_misc import EmptyClass

from bcai_art.datasets_main import IMAGENET_CLASS_QTY, SO2SAT_CHANNEL_QTY
from bcai_art.datasets.ucf101 import UCF101_SAMPLE_DURATION, UCF101_SAMPLE_SIZE, UCF101_NORM_VALUE
from bcai_art.models.resnext import resnet101
from bcai_art.models.resnext_crop import resnet18_nopool
from bcai_art.models.wideresnet import WideResNet
from bcai_art.models.So2Satsmall import So2SatNetSmall, So2SatNetSeparate
from bcai_art.models.bagnet import BagNetPatch
from bcai_art.models.convmixer import ConvMixerPatch

from bcai_art.utils_patch import gen_crops


LOSS_OUTPUT = "loss"
PREDICTION_OUTPUT = "prediction"

NUM_INPUT_CHANNELS_ATTR = "num_input_channels"
SCALE_FACTOR = "scale_factor"
WIDERESNET_SCALE_FACTOR = "wideresnet_scale_factor"
CIFAR_MODEL_SUFFIX = "_cifar"
WIDE_RESNET_PREFIX = "wideresnet"

ROBUSTIFY_FCN = "fcn"
ROBUSTIFY_FCN_SMALL = "fcn_small"
ROBUSTIFY_FCN_SMALL3 = "fcn_small3"
ROBUSTIFY_VAE = "vae"
ROBUSTIFY_UNET = "unet"
ROBUSTIFY_DNCNN = "dncnn"

ROBUSTIFY_LAYERS = [ROBUSTIFY_FCN, ROBUSTIFY_FCN_SMALL, ROBUSTIFY_VAE, ROBUSTIFY_UNET, ROBUSTIFY_DNCNN]
DOWNSAMPLE_KEY = "n_downsample"
RES_KEY = "n_res"
DIM_FEATURE_KEY = "dim"
DEPTH_KEY = "depth"
UPSAMPLE_FACTOR = "upsample_factor"
OBJECT_DETECTION_MAXSIZE_KEY = "max_imagesize"
ROBUSTIFIER_WEIGHT_KEY = "robustifier_weights_file"

OBJECT_DETECTOR_ARGS = [OBJECT_DETECTION_MAXSIZE_KEY]

RANDOM_CROP_VOTE_HARD_CLASS = "hard_class"
RANDOM_CROP_VOTE_HARD_SOFTMAX = "hard_softmax"
RANDOM_CROP_VOTE_SOFT = "soft"
RANDOM_CROP_VOTE_TYPE = [RANDOM_CROP_VOTE_HARD_CLASS, RANDOM_CROP_VOTE_HARD_SOFTMAX, RANDOM_CROP_VOTE_SOFT]


PE_2D = "2D"
PE_1D = "1D"
PE_learn = "learn"
PE_TYPE = [PE_2D,PE_1D,PE_learn]


def get_xavier_init_linear(in_features, out_features, bias=True):
    """A wrapepr function that creates a Xavier initialized linear layer

    :param in_features:   size of each input sample
    :param out_features:  size of each output sample
    :param bias:          enable/disable bias

    :return:  an initialized layer
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    torch.nn.init.xavier_uniform_(layer.weight)

    return layer




def downscale_objectdetction(x, y, imagepixel_max):
    """Downscale an image for object detection."""

    for ii in range(len(x.inner_list)):
        img = x.inner_list[ii]
        _, img_w, img_h = get_frame_channel_height_width(img)
        num_pixels = img_w * img_h
        if num_pixels > imagepixel_max:
            scale_factor = (imagepixel_max/num_pixels)**(1/2)
            img_scaled = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, align_corners=True,
                      mode='bilinear')

            x.inner_list[ii] = img_scaled.squeeze(0)

            num_obj, _ = y.inner_list[ii]['boxes'].shape
            for jj in range(num_obj):
                y.inner_list[ii]['boxes'][jj] = (y.inner_list[ii]['boxes'][jj]*scale_factor)

    return x,y


class SincNetWrapper(nn.Module):
    """
    A SincNet wrapper/creator. Or rather a wrapper for a wrapper :-)
    It loads pre-trained weights in a non-standard fashion.

    | Original github link: https://github.com/mravanelli/SincNet
    """
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(SincNetWrapper, self).__init__()
        assert inner_model is None, "inner_model is not supported!"
        add_args_dict = args_to_paramdict(add_args, [])
        weights_file = None
        # This attribute can, in principle, be eliminated and loading of some initial
        # weights can be done trough the StandardWrapperModel class. This, however,
        # requires some slightly non-trivial modification of create_sincnet function.
        if use_pretrained and 'orig_pretrained_weights_file' in add_args_dict:
            weights_file = add_args_dict['orig_pretrained_weights_file']
        self.model = create_sincnet(num_classes, weights_file)

    def to(self, device):
        return self.model.to(device)

    def forward(self, x):
        return self.model(x)


class StandardWrapperModel(nn.Module):
    """A special standard wrapper class. See the comment for load_state_dict."""
    def __init__(self):
        super().__init__()

    def load_state_dict(self, state_dict, strict=True):
        """Loads a state dictionary either for the current model,
           for for the wrapped model stored in the attribute inner_model.
           The rational is that in some cases we only need to initialize
           one of the wrapped/inner models and not the complete model.
           We want the function that can do this for us automatically
           without specifying the loading mode manually.
        """
        assert strict, "Only strict key-matching loading is supported!"
        if set(self.state_dict().keys()) == set(state_dict.keys()):
            # This invokes the original Pytorch function directly
            torch.nn.Module.load_state_dict(self, state_dict, strict=True)
        else:
            self.inner_model.load_state_dict(state_dict)

class Bagnet_ConvmixerPatchWrapper(StandardWrapperModel):
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(Bagnet_ConvmixerPatchWrapper,self).__init__()

        setattr(self, FREEZE_FEATURES_MODEL_ATTR, getattr(add_args, FREEZE_FEATURES_MODEL_ATTR, False))

        if architecture == "bagnet_patch":
            self.model = BagNetPatch(dim_per_group=add_args.dim_per_group, depth=add_args.depth, depth_mlp=add_args.depth_mlp, 
        patch_size=add_args.patch_size, num_classes=num_classes, input_size = add_args.input_size, in_chans=add_args.in_chans, mlp_only=add_args.mlp_only)
        elif architecture == "convmixer_patch":
            self.model = ConvMixerPatch(dim=add_args.dim, depth=add_args.depth, depth_mlp=add_args.depth_mlp, 
        patch_size=add_args.patch_size, num_classes=num_classes, input_size = add_args.input_size, in_chans=add_args.in_chans, mlp_only=add_args.mlp_only)

    def to(self, device):
        return self.model.to(device)

    def forward(self, x):
        return self.model(x)

class RandomCropsWrapper(StandardWrapperModel):
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        """
        Constructor.
        num_crops: number of random crops to take from the image
        crops_width: width of each crop in pixels
        crops_height: height of each crop in pixels
        combine: If true, all the crops are stacked on top of each other to form a num_crops*3 channels input
        num_samples: perform random sampling num_samples time
        """
        super(RandomCropsWrapper, self).__init__()
        assert inner_model is not None, "wrapper requires inner_model!"
        self.inner_model = inner_model
        
        self.num_crops = add_args.num_crops
        self.crops_width = add_args.crops_width
        self.crops_height = add_args.crops_height

        
        self.combine = add_args.combine
        if self.combine:
            self.num_samples = add_args.num_samples
        else:
            self.num_samples = 1
        assert add_args.voting_type in RANDOM_CROP_VOTE_TYPE
        self.voting_type = add_args.voting_type
        self.softmax = nn.LogSoftmax(dim=1)
        self.resize = add_args.resize
        if self.resize:
            self.crops_width_resize = add_args.crops_width_resize
            self.crops_height_resize = add_args.crops_height_resize
        self.crop_time = 0.0
        self.inference_time = 0.0
        self.postproc_time = 0.0
        
        
        try:
            self.position_encoding = add_args.position_encoding
        except:
            self.position_encoding = False

    
    def forward(self, x):
        
        if self.position_encoding:
            outputs = self.inner_model.forward(x)
        else:
            ex_x = gen_crops(x, self.num_crops, self.crops_width, self.crops_height, combine=self.combine, num_samples=self.num_samples)

            if (not self.training) and len(x.shape) == 5:
                #data is video in eval mode
                output_0 = self.inner_model.forward(torch.unsqueeze(ex_x[0],0))
                outputs = output_0.repeat(self.num_crops,1)
                for ii in range(self.num_crops - 1):
                    outputs[ii+1] = self.inner_model.forward(torch.unsqueeze(ex_x[ii+1],0))
            else:
                outputs = self.inner_model.forward(ex_x)
        
        self.inference_time += time.time()
        
        dim = self.num_samples if self.combine else self.num_samples * self.num_crops
        
        
        if self.voting_type == RANDOM_CROP_VOTE_SOFT:
            return outputs.view(outputs.size(0)//dim, dim, outputs.size(1)).mean(1)
        
        if self.voting_type == RANDOM_CROP_VOTE_HARD_SOFTMAX:     
            outputs = self.softmax(outputs)
            return outputs.view(outputs.size(0)//dim, dim, outputs.size(1)).mean(1)
                
        if self.voting_type == RANDOM_CROP_VOTE_HARD_CLASS:
            #for evaluating class-based hard voting, return output as count of class / # of crops
            if not self.training:
                final_output = self.post_inference(x, outputs)

                return final_output
            
            return outputs

    def post_inference(self,x,outputs):
        use_np = True
        batch_size = x.shape[0]
        num_class = outputs.shape[-1]
        dim = self.num_samples if self.combine else self.num_samples * self.num_crops

        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.view(batch_size,-1)
        
        if not use_np:
            output_voted = torch.ones(batch_size, num_class)* 0.00001
            tmpt = torch.unbind(predicted)

            all_class_count = [torch.unique(t,return_counts=True) for t in tmpt]
            img_id = 0
            for classes, counts in all_class_count:
                majority_1_num, majority_1_class = torch.max(counts, dim=0)
                output_voted[img_id][classes[majority_1_class]] = float(majority_1_num)/float(dim)

                counts[majority_1_class] = 0
                majority_2_num, majority_2_class = torch.max(counts, dim=0)
                output_voted[img_id][classes[majority_2_class]] = float(majority_2_num)/float(dim)
                img_id += 1
        else:
            
            predicted = predicted.cpu().numpy()
            output_voted = np.zeros((batch_size, num_class))
            for img_id in range(batch_size):
                
                bincount = np.bincount(predicted[img_id])
                
                output_voted[img_id][:len(bincount)] = bincount
            
            output_voted = output_voted.astype(float)/ self.num_crops
            output_voted = torch.from_numpy(output_voted)
        
        if torch.cuda.device_count() > 0:
            output_voted = output_voted.cuda()
        return output_voted
        


class RobustifyWrapper(StandardWrapperModel):
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(RobustifyWrapper, self).__init__()
        assert inner_model is not None, "wrapper requires inner_model!"

        self.imagepixel_max = getattr(add_args, OBJECT_DETECTION_MAXSIZE_KEY, None)
        self.inner_model = inner_model
        self.robustifier = add_args.robustifier
        
        assert self.robustifier in ROBUSTIFY_LAYERS
        self.get_robust_layers(self.robustifier, add_args)

        # Freezing the inner model by default
        setattr(self, FREEZE_INNER_MODEL_ATTR, getattr(add_args, FREEZE_INNER_MODEL_ATTR, True))
        print('RobustifyWrapper freezing the inner model?: ', getattr(self, FREEZE_INNER_MODEL_ATTR))

        ufactor = getattr(add_args, UPSAMPLE_FACTOR, None)
        self.upsample = None
        if ufactor is not None:
            self.upsample =  torch.nn.Upsample(scale_factor=(ufactor, ufactor), mode='bilinear', align_corners=False)
        
        robustifier_weights = getattr(add_args, ROBUSTIFIER_WEIGHT_KEY, None)
        if robustifier_weights is not None:
            robustifier_state_dict = torch.load(robustifier_weights)
            print("loading robustifier weights from: " + robustifier_weights)
            self.robust_layers.load_state_dict(robustifier_state_dict['state_dict'], strict=True)
        
        print("number of parameters in robustifier:" + str(get_trainable_param_count(self.robust_layers)))

    def get_robust_layers(self, robustifier, add_args):
        if robustifier == ROBUSTIFY_FCN:
            self.robust_layers = FCN()
        elif robustifier == ROBUSTIFY_FCN_SMALL:
            self.robust_layers = FCNSmall()
        elif robustifier == ROBUSTIFY_FCN_SMALL3:
            self.robust_layers = FCNSmall3()
        elif robustifier == ROBUSTIFY_VAE:
            self.n_downsample = getattr(add_args, DOWNSAMPLE_KEY, 3)
            self.n_res = getattr(add_args, RES_KEY, 4)
            self.dim = getattr(add_args, DIM_FEATURE_KEY, 64)
            self.robust_layers = VAE(dim=self.dim, n_downsample=self.n_downsample, n_res = self.n_res)
        elif robustifier == ROBUSTIFY_UNET:
            self.n_channels = getattr(add_args, NUM_INPUT_CHANNELS_ATTR, 3)
            self.scale = getattr(add_args, SCALE_FACTOR, 8)
            self.robust_layers = UNet(n_channels=self.n_channels, scale=self.scale)
        elif robustifier == ROBUSTIFY_DNCNN:
            self.dim = getattr(add_args, DIM_FEATURE_KEY, 64)
            self.depth = getattr(add_args, DEPTH_KEY, 17)
            self.robust_layers = DnCNN(depth=self.depth, n_channels=self.dim)
        else:
            raise Exception('Unsupported robustifier model: ' + robustifier)

    def robustify(self, x, y=None):
        shlen = len(x.shape)

        if self.upsample is not None:
            x = self.upsample(x)

        if self.imagepixel_max is not None:
            x, y = downscale_objectdetction(x, y, self.imagepixel_max)

        assert shlen >= 4 and shlen <= 5, "expecting either 4d or 5d input!"

        if shlen == 5:
            # If we have a time dimension, it needs to be combined with the batch
            # for the purpose of "robustification"
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)

        assert len(x.shape) == 4

        x = apply_func_squeeze(self.robust_layers.forward, x)

        assert len(x.shape) == 4

        if shlen == 5:
            # but then we need to reshape the tensor back
            x = x.view(b, t, c, h, w)

        return x, y

    def forward(self, x, y=None):
        x, y = self.robustify(x, y)
        
        if y is None:
            output = self.inner_model(x)
        else:
            output = self.inner_model(x, y)
        
        return output


class MNISTNet(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(MNISTNet, self).__init__()
        assert inner_model is None, "inner_model is not supported!"
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(128, num_classes)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class So2SatSubNet(nn.Module):
    """A So2SatNet sub-module."""
    def __init__(self, in_channels):
        """Constructor.

        :param in_channels:  the number of input channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=18, kernel_size=(9, 9))
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=36, kernel_size=(5, 5))
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), kernel_size=(2, 2))
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x), kernel_size=(2, 2))
        x = self.dropout2(x)
        return x


class So2SatNet(nn.Module):
    """A baseline model for SO2SAT collection.
       Modelled after the following Keras code:
       https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/keras/so2sat.py
    """
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(So2SatNet, self).__init__()
        assert inner_model is None, "inner_model is not supported!"

        self.SAR_model = So2SatSubNet(in_channels=4)

        self.EO_model = So2SatSubNet(in_channels=10)

        self.conv_fusion = nn.Conv2d(in_channels=72, out_channels=120, kernel_size=(4, 4))
        torch.nn.init.xavier_uniform_(self.conv_fusion.weight)

        self.fc = nn.Linear(120, 17)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x_sar = 128* x[:,0:4,:,:]
        x_eo = 4 * x[:, 4:, :, :]

        y_sar = self.SAR_model(x_sar)
        y_eo = self.EO_model(x_eo)

        y_cat = torch.cat((y_sar, y_eo), dim=1)

        y_cat = torch.flatten(F.relu(self.conv_fusion(y_cat)), start_dim=1)

        y_cat = self.dropout(y_cat)

        return self.fc(y_cat)



class CIFARWideResnet(StandardWrapperModel):
    """A wrapper for CIFAR-tailored WideResnets"""
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(CIFARWideResnet, self).__init__()
        assert inner_model is None, "inner_model is not supported!"

        if not architecture.endswith(CIFAR_MODEL_SUFFIX):
            raise ValueError(f"unsupported CIFARWideResnet architecture: must end with {CIFAR_MODEL_SUFFIX}")
        architecture = architecture[:-len(CIFAR_MODEL_SUFFIX)]

        if not architecture.startswith(WIDE_RESNET_PREFIX):
            raise ValueError(f"unsupported CIFARWideResnet architecture: must start with {WIDE_RESNET_PREFIX}")
        architecture = architecture[len(WIDE_RESNET_PREFIX):]
        depth, widen_factor = architecture.split('_')
        depth = int(depth)
        widen_factor = int(widen_factor)

        scale_factor = getattr(add_args, WIDERESNET_SCALE_FACTOR, 1)

        print(f'CIFARWideResnet depth: {depth} widen factor: {widen_factor} scale factor: {scale_factor}')

        num_input_channels = getattr(add_args, NUM_INPUT_CHANNELS_ATTR, 3)

        self.inner_model = WideResNet(num_channels=num_input_channels, depth=depth, num_classes=num_classes,
                                      widen_factor=widen_factor, scale_factor=scale_factor,
                                      dropout=0.1)

    def forward(self, x):
        return self.inner_model.forward(x)


class TorchCIFARVisionNets(StandardWrapperModel):
    """Torchvision models modified specifically for CIFAR datasets.
       ASSUMPTION: all model names end with CIFAR_MODEL_SUFFIX.
       In particular, we make resnets compatabile with https://github.com/huyvnphan/PyTorch_CIFAR10.
    """
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(TorchCIFARVisionNets, self).__init__()
        assert inner_model is None, "inner_model is not supported!"

        if not architecture.endswith(CIFAR_MODEL_SUFFIX):
            raise ValueError(f"unsupported TorchVision CIFAR architecture: must end with {CIFAR_MODEL_SUFFIX}")
        architecture = architecture[:-len(CIFAR_MODEL_SUFFIX)]

        change_fc = num_classes != IMAGENET_CLASS_QTY
        num_input_channels = getattr(add_args, NUM_INPUT_CHANNELS_ATTR, 3)
        print('The number of model input channels:', num_input_channels)

        self.inner_model = models.__dict__[architecture](pretrained=use_pretrained)
        if architecture.startswith('resnet'):
            in_feats = self.inner_model.fc.in_features
            if change_fc:
                self.inner_model.fc = nn.Linear(in_feats, num_classes)
                torch.nn.init.xavier_uniform_(self.inner_model.fc.weight)

            self.inner_model.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            torch.nn.init.xavier_uniform_(self.inner_model.conv1.weight)
        else:
            raise ValueError("unsupported TorchVision CIFAR architecture")

    def forward(self, x):
        return self.inner_model.forward(x)



class TorchVisionNets(StandardWrapperModel):
    """A wrapper for TorchVision models. The following replacements are done:
       1. If the number of channels is not three, we replace the first convolutional layer.
       2. If the number of classes differes from that in ImageNet, we replace the fully-connected
          classification layer.
    """
    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super(TorchVisionNets, self).__init__()
        assert inner_model is None, "inner_model is not supported!"

        change_fc = num_classes != IMAGENET_CLASS_QTY
        num_input_channels = getattr(add_args, NUM_INPUT_CHANNELS_ATTR, 3)
        self.inner_model = models.__dict__[architecture](pretrained=use_pretrained)
        if architecture.startswith('resnet'):
            in_feats = self.inner_model.fc.in_features

            if change_fc:
                self.inner_model.fc = get_xavier_init_linear(in_feats, num_classes)

            if num_input_channels != 3:
                self.inner_model.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                torch.nn.init.xavier_uniform_(self.inner_model.conv1.weight)

        elif architecture.startswith('inception'):
            in_feats = self.inner_model.fc.in_features

            if change_fc:
                self.inner_model.fc = get_xavier_init_linear(in_feats, num_classes)

            if num_input_channels != 3:
                raise NotImplementedError

        elif architecture.startswith('densenet'):
            in_feats = self.inner_model.classifier.in_features

            if change_fc:
                self.inner_model.classifier = get_xavier_init_linear(in_feats, num_classes)

            if num_input_channels != 3:
                self.inner_model.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                torch.nn.init.xavier_uniform_(self.inner_model.conv1.weight)

        elif architecture.startswith('vgg'):
            in_feats = self.inner_model.classifier._modules['6'].in_features
            if change_fc:
                self.inner_model.classifier._modules['6'] = get_xavier_init_linear(in_feats, num_classes)

            if num_input_channels != 3:
                self.inner_model.features[0] = nn.Conv2d(num_input_channels, self.inner_model.features[0].out_channels,
                                                         kernel_size=3, padding=1)
                torch.nn.init.xavier_uniform_(self.inner_model.conv1.weight)

        elif architecture.startswith('alexnet'):
            in_feats = self.inner_model.classifier._modules['6'].in_features
            if change_fc:
                self.inner_model.classifier._modules['6'] = get_xavier_init_linear(in_feats, num_classes)

            if num_input_channels != 3:
                self.inner_model.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=11, stride=4, padding=2)
                torch.nn.init.xavier_uniform_(self.inner_model.conv1.weight)
        else:
            raise ValueError("unsupported TorchVision architecture")

    def forward(self, x):
        return self.inner_model.forward(x)


class ResNeXt101UCF101TestAvgWrapper(StandardWrapperModel):
    """A ResNet101 class wrapper specifically designed to work with UCF101 video data,
        which performs differently at train and test time.
        At train time, it processes each data batch as usual.
        However, at test time, it expects batches of the size one,
        where each element is a video. The video is chunked into several
        pieces each of which is processed by the model. Then,
        we return the average of the produced logits.
    """

    def __init__(self, architecture, num_classes, use_pretrained, add_args, inner_model):
        super().__init__()
        assert inner_model is None, "inner_model is not supported!"

        self.sample_duration = UCF101_SAMPLE_DURATION
        self.sample_size = UCF101_SAMPLE_SIZE
        self.channel_qty = 3

        try:
            self.position_encoding = add_args.position_encoding
            self.pe = None
            if self.position_encoding == True:
            #currently pe only works for ucf101 model
            #all ucf101 model's first embedding output dimension is the same as inpu
            # with 64 channels
                #emb_dim = 64
                #self.pe = nn.Parameter(torch.randn(emb_dim,self.sample_size,self.sample_size))
                self.num_crops = add_args.num_crops
                self.crops_width = add_args.crops_width
                self.crops_height = add_args.crops_height
                self.encode_emb = add_args.encode_emb
                self.sample_width = UCF101_SAMPLE_SIZE
                self.sample_height = UCF101_SAMPLE_SIZE
            else:
                self.num_crops = 0
                self.crops_width = 0
                self.crops_height = 0
                self.encode_emb = True
                self.sample_width = add_args.crops_width
                self.sample_height = add_args.crops_height
        except:
            self.position_encoding = False
            self.num_crops = 0
            self.crops_width = 0
            self.crops_height = 0
            self.encode_emb = True
            self.sample_width = UCF101_SAMPLE_SIZE
            self.sample_height = UCF101_SAMPLE_SIZE

        if architecture.startswith("resnext18_nopool"): 
            self.inner_model = resnet18_nopool(
                num_classes=num_classes,
                shortcut_type='B',
                cardinality=32,
                sample_size=self.sample_size,
                sample_duration=self.sample_duration,
                input_channels=self.channel_qty,
                output_layers=[],
                position_encoding = self.position_encoding,
                crops_height = self.crops_height,
                crops_width = self.crops_width,
                num_crops = self.num_crops,
                encode_emb = self.encode_emb) # This one must be empty
        elif architecture.startswith("resnext101"):
            self.inner_model = resnet101(
                        num_classes=num_classes,
                        shortcut_type='B',
                        cardinality=32,
                        sample_size=self.sample_size,
                        sample_duration=self.sample_duration,
                        input_channels=self.channel_qty,
                        output_layers=[]) # This one must be empty
        else:
            Exception("unsupported architecture: " + architecture)

    def forward(self, x):
        shape = x.shape
        assert len(shape) == 5, "Video tensor should have rank 5!"
        # The model gets the tensor of the shape B x T x C x H x W,
        # but it needs B x C x T x H x W
        x = x.transpose(1, 2)

        # The model was trained to run on unscaled values from 0 to 255
        # Note that the input was scaled as (STD is all ones):
        # (x_orig / UCF101_NORM_VALUE - UNSCALED_MEAN / UCF101_NORM_VALUE) / 1.0
        # but the model needs x_orig - UNSCALED_MEAN
        x *= UCF101_NORM_VALUE

        if self.training and self.num_crops == 0:
            return self.inner_model(x)
        else:
            assert shape[0] == 1, "In test mode, the model processes one video at a time!"
            x = x.squeeze(dim=0)  # b/c batch size is 1
            sample_qty = x.shape[1] # first one is the channel
            clip_qty = int(sample_qty / self.sample_duration)

            inputs = torch.Tensor(clip_qty,
                                  self.channel_qty, self.sample_duration,
                                  self.sample_size, self.sample_size).to(x.device)

            inputs_qty = inputs.shape[0]
            for k in range(inputs_qty):
                inputs[k, :, :, :, :] = x[:, k * self.sample_duration:(k + 1) * self.sample_duration, :, :]

            outputs = self.inner_model(inputs)

            if self.position_encoding == True:
                order = torch.from_numpy(np.linspace(0,self.num_crops-1, self.num_crops))
                order = order.repeat(inputs_qty)

                outputs_averaged = torch.zeros((self.num_crops,outputs.shape[-1]))
                for jj in range(self.num_crops):
                    outputs_averaged[jj] = torch.mean(outputs[order == jj,:],dim=0, keepdim=True)
                
                return outputs_averaged

            return torch.mean(outputs, dim=0, keepdim=True)


# Let's use only lower-case names
MODEL_DICT = {
    'mnistnet' : MNISTNet,
    'so2satnet': So2SatNet,
    'so2satnetsmall': So2SatNetSmall,
    'so2satnetsmallseparate': So2SatNetSeparate,

    # Can be compared against the more standard ResNet9 if needed
    f'resnet9{CIFAR_MODEL_SUFFIX}' : CustomResnet9,
    f'resnet9_small{CIFAR_MODEL_SUFFIX}' : CustomResnet9Small,
    f'resnet18{CIFAR_MODEL_SUFFIX}': TorchCIFARVisionNets,
    f'resnet34{CIFAR_MODEL_SUFFIX}': TorchCIFARVisionNets,
    f'resnet50{CIFAR_MODEL_SUFFIX}': TorchCIFARVisionNets,
    f'resnet101{CIFAR_MODEL_SUFFIX}': TorchCIFARVisionNets,
    f'resnet152{CIFAR_MODEL_SUFFIX}': TorchCIFARVisionNets,

    # see also for refs. https://github.com/meliketoy/wide-resnet.pytorch
    f'{WIDE_RESNET_PREFIX}70_16{CIFAR_MODEL_SUFFIX}' : CIFARWideResnet,
    f'{WIDE_RESNET_PREFIX}34_20{CIFAR_MODEL_SUFFIX}' : CIFARWideResnet,
    f'{WIDE_RESNET_PREFIX}34_10{CIFAR_MODEL_SUFFIX}' : CIFARWideResnet,
    f'{WIDE_RESNET_PREFIX}28_20{CIFAR_MODEL_SUFFIX}': CIFARWideResnet,
    f'{WIDE_RESNET_PREFIX}28_10{CIFAR_MODEL_SUFFIX}': CIFARWideResnet,

    'densenet121': TorchVisionNets,
    'resnet18': TorchVisionNets,
    'resnet34': TorchVisionNets,
    'resnet50': TorchVisionNets,
    'resnet101': TorchVisionNets,
    'resnet152': TorchVisionNets,
    'vgg16' : TorchVisionNets,
    'alexnet': TorchVisionNets,

    'sincnet' : SincNetWrapper,

    'resnext101_ucf101' : ResNeXt101UCF101TestAvgWrapper,
    'resnext18_nopool_ucf101' : ResNeXt101UCF101TestAvgWrapper,

    'bagnet_patch': Bagnet_ConvmixerPatchWrapper,
    'convmixer_patch': Bagnet_ConvmixerPatchWrapper,


    # General-purpose front-ends that can work with a variety of models
    'robustifier': RobustifyWrapper,
    'randomized_crop':RandomCropsWrapper
}


class TopLevelModelWrapper(torch.nn.Module):
    """A top-level wrapper class that does some standardization, in particular, it:

        1. Applies data normalization before calling the actual forward function of a model.
        2. Standardization of the loss computation.
            2.1 Some classes compute the loss on their own. In this case, we simply return the
            loss computed by such a model.
            2.2 However, in most cases, the loss is computed by the training code.
            To provide a uniform interface, we do this computation inside this wrapper.
    """
    def __init__(self, model, model_arg_obj, loss_class, num_classes, mean, std):
        super().__init__()
        self.orig_model = model
        setattr(self, MODEL_ARGS_KEY, model_arg_obj)
        self.mean = mean
        self.num_classes = num_classes
        self.std = std
        self.inv_std = 1.0 / std
        if loss_class is not None:
            self.loss = loss_class(reduction='none')
        else:
            self.loss = None
        
        #print the number of trainable parameters
        print("number of trainable parameters:", str(get_trainable_param_count(self.orig_model)))

    def to(self, device):
        """Move the model to a given device."""
        self.orig_model.to(device)
        self.mean = self.mean.to(device)
        self.inv_std = self.inv_std.to(device)
        return self

    def cpu(self):
        """move the model to CPU"""
        return self.to('cpu')

    def load_state_dict(self, state_dict, strict=True):
        """load the state dictionary of the original/inner model"""
        self.orig_model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """returns a state dictionary of the original/inner model"""
        return self.orig_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def forward(self, x, y):
        """The model's forward function that normalizes input data and computes the loss:

           If self.loss is not None we assume that the original/inner model computes loss
           on its own (e.g. detection models do this).

            :param x:   input
            :param y:   labels
            :return: a dictionary that contains prediction and loss

        """
        if self.loss is not None:
            outputs = self.orig_model( apply_func(self.normalize, x) )
            outputs_loss = self.loss(outputs, y)
            return {PREDICTION_OUTPUT: outputs, LOSS_OUTPUT: outputs_loss}
        
        losses, outputs = self.orig_model(apply_func(self.normalize, x) , y)
        return {PREDICTION_OUTPUT: outputs, LOSS_OUTPUT: losses}

    def forward_no_loss_comp(self, x):
        """The forward function that normalizes the data, but does not compute the loss.

            :param x: input
            :return: the output of the inner model forward function.
        """
        return self.orig_model( apply_func(self.normalize, x) )
    
    def orig_forward(self, x):
        """This is just a proxy for the inner model forward function."""
        return self.orig_model(x)
    
    def normalize(self, x):
        return (x - self.mean) * self.inv_std


def create_wrapped_model(model_arg_obj, num_classes, top_level):
    """Create a to-be-wrapped model (potentially recursively)

    :param model_arg_obj: a reference to the model argument object
    :param num_classes:   a number of classes
    :param top_level:     should be True only for the top-level call of this function:
                          subsequent recursive calls should set this parameter to False
    :return:
    """

    model_arch = getattr(model_arg_obj, MODEL_ARCH_ATTR)
    add_args = getattr(model_arg_obj, ADD_ARG_ATTR, EmptyClass)
    use_pretrained = getattr(model_arg_obj, USE_PRETRAINED_ATTR, False)
    loss_attr = getattr(model_arg_obj, LOSS_ATTR, None)
    if top_level:
        if loss_attr == LOSS_NATIVE:
            loss = None
        else:
            loss = get_loss(loss_attr)
    else:
        if loss_attr is not None:
            raise  Exception('Loss attribute can be used only at a top-level model!')
        loss = None

    inner_model_args = getattr(model_arg_obj, INNER_MODEL_ATTR, None)

    if inner_model_args is not None:
        model_arch_inner, inner_model, _ = create_wrapped_model(inner_model_args, num_classes, False)
        model_arch_full = f'_{model_arch_inner}'
    else:
        inner_model = None
        model_arch_full = ""

    model_arch = model_arch.lower()
    model_arch_full = model_arch + model_arch_full
    if model_arch in MODEL_DICT:
        # add_args is not used now, it's a plug for the future extensions
        model = MODEL_DICT[model_arch](model_arch,
                                       num_classes,
                                       use_pretrained,
                                       add_args,
                                       inner_model)
    else:
        raise Exception(f'Unsupported model type: {model_arch}')

    return model_arch_full, model, loss


def create_toplevel_model(num_classes,
                          mean, std,
                          model_arg_obj):
    """Create a top-level wrapper model. The wrapped model is created recursively,
       because it can be wrapper itself.

    :param num_classes:   a number of classes
    :param mean:          a mean value tensor, e.g., computed with get_shaped_tensor (for data normalization)
    :param std:           a standard deviation tensor, e.g., computed with get_shaped_tensor  (for data normalization)
    :param model_arg_obj:

    :return: a model object reference.
    """

    model_arch, model, loss_class = create_wrapped_model(model_arg_obj, num_classes, True)

    return TopLevelModelWrapper(model, model_arg_obj, loss_class, num_classes, mean, std)



