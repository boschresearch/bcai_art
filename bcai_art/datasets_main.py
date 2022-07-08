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

import os
import copy
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
#from timm.data.auto_augment import rand_augment_transform


from bcai_art.datasets.cifar_split import create_cifar_dataset_3way, CIFAR_SPLIT_TRAIN, CIFAR_SPLIT_VAL, CIFAR_SPLIT_TEST
from bcai_art.datasets.twosix_librispeech import *
from bcai_art.datasets.armory.datasets import so2sat_tfrecs, SO2SAT_CHANNEL_QTY, SO2SAT_CLASS_QTY
from bcai_art.datasets.ucf101 import UCF101DataSetNPZ, UCF101DataSetJPG, UCF101_MEAN, UCF101_STD, \
    UCF101_CLASS_QTY, UCF101_SAMPLE_SIZE, UCF101_SAMPLE_DURATION, UCF101_SPLIT
from bcai_art.conf_attr import ADD_ARG_ATTR, UCF101_SAMPLE_SIZE_ATTR, UCF101_MAX_TEST_SAMPLE_QTY_ATTR, DATASET_REG_FILE
from bcai_art.utils_misc import DATASET_TYPE_IMAGE, DATASET_TYPE_AUDIO_FIX_SIZE, DATASET_TYPE_VIDEO_FIX_SIZE
from bcai_art.utils_misc import mscoco_collate_fn, mscoco_target_transform, EmptyClass

from torchvision.datasets.folder import IMG_EXTENSIONS

DATASET_NAME_PARAM = "name"
DATASET_ROOT_PARAM = "root"
DATASET_NUM_CHANNELS_PARAM = "num_channels"
DATASET_STD_PARAM = "std"
DATASET_MEAN_PARAM = "mean"
DATASET_NUM_CHANNELS_PARAM = "num_channels"
DATASET_NUM_CLASSES_PARAM = "num_classes"
DATASET_UPPER_LIMIT_PARAM = "upper_limit"
DATASET_LOWER_LIMIT_PARAM = "lower_limit"
DATASET_CREATOR_PARAM = "creator"
DATASET_TYPE = "dataset_type",
DATASET_COLLATE_FN = "collate_fn"
DATASET_ANNOT_FILE = "annotation_file"

IMAGENET_CLASS_QTY=1000

KEY_AUTO_AUG="autoaug_config"

# This limits the number of consequtive chunks to be used for UCF101 tests
UCF101_MAX_TEST_SAMPLE_QTY=8

class DataSetNPZ(Dataset):
    """A simple wrapper for a dataset converted to NPZ format."""

    def __init__(self,
                 train,
                 dataset_root,
                 torch_transform=None,
                 timm_transform = None):
        """
        :param  train : 1 for training and 0 for testing.
        :param  dataset_root: a dataset root directory.
        :param  torch_transform: an optional transform object

        """
        self.train = bool(train)
        self.root_dir = dataset_root
        self.torch_transform = torch_transform
        self.timm_transform = timm_transform

        split_subdir = os.path.join(dataset_root, 'train' if train else 'test')
        self.data_list = torch.load(os.path.join(split_subdir, DATASET_REG_FILE))

    def __len__(self):
        """
        :return: number of dataset items
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        comp_np_file, label = self.data_list[idx]

        with open(os.path.join(self.root_dir, comp_np_file), 'rb') as f:
            data = torch.from_numpy(np.load(f)['arr_0'])

        if self.torch_transform is not None:
            data = self.torch_transform(data)

        if self.timm_transform is not None:
            data_PILImage = transforms.ToPILImage()(data)
            data_PILImage = self.timm_transform(data_PILImage)
            data = transforms.ToTensor()(data_PILImage)

        return data, label


class CocoDetection(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

        This class is borrowed from the Torchvision repo (BSD 3 license).

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        print("Start loading COCO format annotation file")
        with open(annFile,"rt") as f:
            annotations= json.load(f)
        self.root= root
        self.target_transform= target_transform
        self.transform= transform
        self.ids = sorted([x['id'] for x in annotations['images']])
        self.filenames = {x['id']:x['file_name'] for x in annotations['images']}
        self.ann = {}
        for a in annotations["annotations"]:
            if a["image_id"] in self.ann:
                self.ann[a["image_id"]].append(a)
            else:
                self.ann[a["image_id"]]= [a]
        print("Loading is done.")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.ann[img_id] if img_id in self.ann else []
        path = self.filenames[img_id]

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img), self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


def create_MNIST(dataset_root, add_args):
    transform_augm1 = [transforms.RandomCrop(size=28, padding=1)]

    transform_train = transforms.Compose(transform_augm1 + [transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform_train)
    test = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform_test)

    return train, test


def create_ImageNet(dataset_root, add_args):
    # Based on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L204
    #can add randomizedcrop
    transforms_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

    train = torchvision.datasets.ImageNet(dataset_root, split='train', transform=transforms_train)
    test = torchvision.datasets.ImageNet(dataset_root, split='val', transform=transforms_test)
    return train, test


def create_CIFAR10_train_only(dataset_root, add_args):
    # This is basically the standard CIFAR10 dataset, but instead of the
    # official test set, it will use only the training set and carve out
    # a validation set out of it, i.e., no use of the official test!
    # It will not download data on its own and this is by design!
    #
    # Transforms are similar to those from https://github.com/lambdal/cifar10-fast
    # but the order and the probabilities are different
    #
    transform_base = [transforms.ToTensor(), ]
    transform_augm1 = [transforms.RandomCrop(size=32, padding=4),
                      transforms.RandomHorizontalFlip(p=0.5)]
    transform_augm2 = [transforms.RandomErasing(scale=(0.1, 0.25), ratio=(0.5,2), p=1, value=0)]

    transform_train = transforms.Compose(transform_augm1 + transform_base + transform_augm2)
    transform_test = transforms.Compose(transform_base)

    train = create_cifar_dataset_3way(root=dataset_root,
                                      data_part=CIFAR_SPLIT_TRAIN,
                                      transform=transform_train, is_cifar10=True)
    test = create_cifar_dataset_3way(root=dataset_root,
                                      data_part=CIFAR_SPLIT_VAL,
                                      transform=transform_test, is_cifar10=True)

    return train, test


def create_CIFAR10_test_only(dataset_root, add_args):
    # This is only the test part of the CIFAR10 dataset: The train part is None.
    # It will not download data on its own and this is by design!
    #
    # Transforms are similar to those from https://github.com/lambdal/cifar10-fast
    # but the order and the probabilities are different
    #
    transform_base = [transforms.ToTensor(), ]
    transform_test = transforms.Compose(transform_base)

    test = create_cifar_dataset_3way(root=dataset_root,
                                      data_part=CIFAR_SPLIT_TEST,
                                      transform=transform_test, is_cifar10=True)

    return None, test


def create_CIFAR10(dataset_root, add_args):
    # Transforms are similar to those from https://github.com/lambdal/cifar10-fast
    # but the order and the probabilities are different
    transform_base = [transforms.ToTensor(), ]
    transform_augm1 = [transforms.RandomCrop(size=32, padding=4),
                      transforms.RandomHorizontalFlip(p=0.5)]
    transform_augm2 = [transforms.RandomErasing(scale=(0.1, 0.25), ratio=(0.5,2), p=1, value=0)]

    transform_train = transforms.Compose(transform_augm1 + transform_base + transform_augm2)
    transform_test = transforms.Compose(transform_base)

    train = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True,
                                         transform=transform_train)
    test = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True,
                                        transform=transform_test)

    return train, test

def create_CIFAR100(dataset_root, add_args):
    # Transforms are similar to those from https://github.com/lambdal/cifar10-fast
    # but the order and the probabilities are different
    transform_base = [transforms.ToTensor(), ]
    transform_augm1 = [transforms.RandomCrop(size=32, padding=4),
                      transforms.RandomHorizontalFlip(p=0.5)]
    transform_augm2 = [transforms.RandomErasing(scale=(0.1, 0.25), ratio=(0.5,2), p=1, value=0)]

    transform_train = transforms.Compose(transform_augm1 + transform_base + transform_augm2)
    transform_test = transforms.Compose(transform_base)

    train = torchvision.datasets.CIFAR100(root=dataset_root, train=True, download=True,
                                         transform=transform_train)
    test = torchvision.datasets.CIFAR100(root=dataset_root, train=False, download=True,
                                        transform=transform_test)

    return train, test


def get_resisc45_filenum_and_ext(file_name):
    """A simple function to extract the file number
    and extension from a RESISC-45 file name,
    e.g., it returns 262, 'jpg' if an input file is
    island_262.jpg. If the file does
    not match the pattern, None is returned.

    :param   file_name:   input file name
    :return  a tuple: file number, lower-cased extension with preceding dot or None
    """
    s1 = file_name.rfind('_')
    s2 = file_name.rfind('.')
    if s1 < 0 or s2 < 0 or s2 - s1 <=0:
        return None

    try:
        num = int(file_name[s1 + 1: s2])
        return num, file_name[s2:].lower()
    except:
        return None


def is_resisc_valid(file_name, min_num, max_num):
    """Check if
        1) the file name a valid RESISC-45 file name with a proper file number.
        2) Its number is in the range

        :param file_name:  a file name to check
        :param min_num:    the minimum file number (inclusive)
        :param max_num:    the maximum file number (exclusive)

        :return True if the valid file name with the number in a give range
    """
    ret = get_resisc45_filenum_and_ext(file_name)
    if ret is None:
        return False
    num, ext = ret
    if ext not in IMG_EXTENSIONS:
        return False
    return num >= min_num and num < max_num

# This split is proposed by TWOSIX
def is_resisc_valid_train(file_name):
    return is_resisc_valid(file_name, 0, 600)

def is_resisc_valid_test(file_name):
    return is_resisc_valid(file_name, 600, float('inf'))


def create_RESISC45(dataset_root, add_args):
    # Adjusting croping & cutting proportions of the CIFAR10 dataset
    # to the larger 256x256 RESISC45.
    transform_augm0 = [transforms.RandomCrop(size=256, padding=32),
                       transforms.RandomHorizontalFlip(p=0.5)]
    # Random cut-out must be applied after the tensorifcation & normalization
    transform_augm1 = [transforms.RandomErasing(scale=(0.1, 0.25), ratio=(0.5,2), p=1, value=0)]
    transform_base = [transforms.Resize(224), transforms.ToTensor()]
    transform_train = transforms.Compose(transform_augm0 + transform_base + transform_augm1)
    transform_test =  transforms.Compose(transform_base)

    train = torchvision.datasets.ImageFolder(dataset_root,
                                             transform = transform_train,
                                             is_valid_file=is_resisc_valid_train)
    test = torchvision.datasets.ImageFolder(dataset_root,
                                            transform=transform_test,
                                            is_valid_file=is_resisc_valid_test)
    
    return train, test

def create_UCF101_JPG(dataset_root, add_args):
    sample_size = getattr(add_args, UCF101_SAMPLE_SIZE_ATTR, UCF101_SAMPLE_SIZE)
    max_sample_qty = getattr(add_args, UCF101_MAX_TEST_SAMPLE_QTY_ATTR, UCF101_MAX_TEST_SAMPLE_QTY)

    train = UCF101DataSetJPG(train=1,
                          frame_dir=os.path.join(dataset_root, 'frames'),
                          annotation_path=os.path.join(dataset_root, 'ucfTrainTestlist'),
                          sample_duration=UCF101_SAMPLE_DURATION,
                          max_test_frame_qty=UCF101_SAMPLE_DURATION * max_sample_qty,
                          sample_size=sample_size,
                          split=UCF101_SPLIT)

    test = UCF101DataSetJPG(train=0,
                          frame_dir=os.path.join(dataset_root, 'frames'),
                          annotation_path=os.path.join(dataset_root, 'ucfTrainTestlist'),
                          sample_duration=UCF101_SAMPLE_DURATION,
                          max_test_frame_qty=UCF101_SAMPLE_DURATION * max_sample_qty,
                          sample_size=sample_size,
                          split=UCF101_SPLIT)

    return train, test


def create_UCF101_NPZ(dataset_root, add_args):
    sample_size = getattr(add_args, UCF101_SAMPLE_SIZE_ATTR, UCF101_SAMPLE_SIZE)
    max_sample_qty = getattr(add_args, UCF101_MAX_TEST_SAMPLE_QTY_ATTR, UCF101_MAX_TEST_SAMPLE_QTY)

    train = UCF101DataSetNPZ(train=1,
                          root_dir=dataset_root,
                          sample_duration=UCF101_SAMPLE_DURATION,
                          max_test_frame_qty=UCF101_SAMPLE_DURATION * max_sample_qty,
                          sample_size=sample_size)

    test = UCF101DataSetNPZ(train=0,
                          root_dir=dataset_root,
                          sample_duration=UCF101_SAMPLE_DURATION,
                          max_test_frame_qty=UCF101_SAMPLE_DURATION * max_sample_qty,
                          sample_size=sample_size)

    return train, test


def create_MSCOCO(dataset_root, add_args, annot_file):
    transform_base = [transforms.ToTensor()]
    base_class = CocoDetection
    if hasattr(add_args, "non_coco") and add_args.non_coco:
        base_class= torchvision.datasets.CocoDetection
    train = base_class(os.path.join(dataset_root, annot_file[0][0]),
                       os.path.join(dataset_root, annot_file[0][1]),
                       transform=transforms.Compose(transform_base),
                       target_transform=mscoco_target_transform)
    test = base_class(os.path.join(dataset_root, annot_file[1][0]),
                      os.path.join(dataset_root, annot_file[1][1]),
                      transform=transforms.Compose(transform_base),
                      target_transform=mscoco_target_transform)

    return train, test


def create_TWOSIX_LIBRISPEECH(dataset_root, add_args):
    train = LibriSpeechTrainingSet(dataset_root)

    test = create_librispeech_armory_subset(root=dataset_root,
                                            window_length=WINDOW_LENGTH,
                                            chunk_qty=None, # Use all the chunks\n",
                                            chunk_gen_func=seq_chunk_generator,
                                            is_train=False,
                                            sub_dirs = ['test'])

    return train, test


def create_SO2SAT_TFRECS(dataset_root, add_args):
    """Create a SO2SAT dataset directly from the Tensorflow dataset (TFREC format)"""
    transform_train = transforms.Compose([transforms.RandomErasing(scale=(0.1, 0.25), ratio=(0.5,2), p=1, value=0)])

    train = so2sat_tfrecs(split='train', dataset_dir=dataset_root, torch_transform=transform_train)
    test = so2sat_tfrecs(split='validation', dataset_dir=dataset_root)

    return train, test

def create_SO2SAT_NPZ(dataset_root, add_args):
    """A SO2SAT dataset created by converting data from a Tensorflow TFREC format to compressed numpy.
       Conversion utility: convert_so2sat_tfrecs_to_numpy.py
    """
    transform_train = transforms.Compose([transforms.RandomErasing(scale=(0.1, 0.25), ratio=(0.5,2), p=1, value=0)])
    
    # args_dict = vars(add_args)
    # if KEY_AUTO_AUG in args_dict:
    #     transform_timm_train = rand_augment_transform(config_str=args_dict[KEY_AUTO_AUG],hparams={'translate_const': 16})

    train = DataSetNPZ(train=True, dataset_root=dataset_root, torch_transform=transform_train)
    test = DataSetNPZ(train=False, dataset_root=dataset_root)

    return train, test


# Notes:
# 1. use a lowercase ony here
# 2. mean, std, lower_limit, upper_limit must all be tuples
#    if you have a single-number tuple, don't forget to insert comma (see, e.g., mnist)
DATASET_CREATORS = {
    'imagenet' : {DATASET_CREATOR_PARAM: create_ImageNet,
                 DATASET_TYPE: DATASET_TYPE_IMAGE,
                 DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
                 DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
                 # From https://pytorch.org/hub/pytorch_vision_resnet/
                 DATASET_MEAN_PARAM: (0.485, 0.456, 0.406),
                 DATASET_STD_PARAM: (0.229, 0.224, 0.225),
                 DATASET_NUM_CLASSES_PARAM: IMAGENET_CLASS_QTY,
                 DATASET_NUM_CHANNELS_PARAM: 3},
    'cifar10' : {DATASET_CREATOR_PARAM: create_CIFAR10,
                 DATASET_TYPE: DATASET_TYPE_IMAGE,
                 DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
                 DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
                 #DATASET_MEAN_PARAM: (0.5, 0.5, 0.5),
                 #DATASET_STD_PARAM: (0.5, 0.5, 0.5),
                 # More accurate CIFAR10 numbers can be found at a link below.
                 # Yet, you can typically train any model with these
                 # simplified MU/STD value to the same accuracy
                 # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L34
                 DATASET_MEAN_PARAM: (0.4914, 0.4822, 0.4465),
                 DATASET_STD_PARAM: (0.2023, 0.1994, 0.2010),
                 DATASET_NUM_CLASSES_PARAM: 10,
                 DATASET_NUM_CHANNELS_PARAM: 3},
    'cifar10_test_only' : {DATASET_CREATOR_PARAM: create_CIFAR10_test_only,
                 DATASET_TYPE: DATASET_TYPE_IMAGE,
                 DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
                 DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
                 DATASET_MEAN_PARAM: (0.4914, 0.4822, 0.4465),
                 DATASET_STD_PARAM: (0.2023, 0.1994, 0.2010),
                 DATASET_NUM_CLASSES_PARAM: 10,
                 DATASET_NUM_CHANNELS_PARAM: 3},
    'cifar10_train_only' : {DATASET_CREATOR_PARAM: create_CIFAR10_train_only,
                 DATASET_TYPE: DATASET_TYPE_IMAGE,
                 DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
                 DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
                 DATASET_MEAN_PARAM: (0.4914, 0.4822, 0.4465),
                 DATASET_STD_PARAM: (0.2023, 0.1994, 0.2010),
                 DATASET_NUM_CLASSES_PARAM: 10,
                 DATASET_NUM_CHANNELS_PARAM: 3},
    'cifar100': {DATASET_CREATOR_PARAM: create_CIFAR100,
                DATASET_TYPE: DATASET_TYPE_IMAGE,
                DATASET_LOWER_LIMIT_PARAM: (0, 0, 0),
                DATASET_UPPER_LIMIT_PARAM: (1, 1, 1),
                DATASET_MEAN_PARAM: (0.5, 0.5, 0.5),
                DATASET_STD_PARAM: (0.5, 0.5, 0.5),
                 # More accurate CIFAR100 numbers can be found at a link below.
                 # Yet, you can typically train any model with these
                 # simplified MU/STD value to the same accuracy
                 # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                 DATASET_MEAN_PARAM: (0.5071, 0.4867, 0.4408),
                 DATASET_STD_PARAM: (0.2675, 0.2565, 0.2761),
                DATASET_NUM_CLASSES_PARAM: 100,
                DATASET_NUM_CHANNELS_PARAM: 3},
    'mnist' : {DATASET_CREATOR_PARAM: create_MNIST,
               DATASET_TYPE: DATASET_TYPE_IMAGE,
               DATASET_LOWER_LIMIT_PARAM: (0,),
               DATASET_UPPER_LIMIT_PARAM: (1,),
               DATASET_MEAN_PARAM: (0.1307,),
               DATASET_STD_PARAM: (0.3081,),
               DATASET_NUM_CLASSES_PARAM: 10,
               DATASET_NUM_CHANNELS_PARAM: 1},
    'resisc45' : {DATASET_CREATOR_PARAM: create_RESISC45,
                 DATASET_TYPE: DATASET_TYPE_IMAGE,
                 DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
                 DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
                 DATASET_MEAN_PARAM: (0.3680, 0.3810, 0.3436),
                 DATASET_STD_PARAM: (0.1454, 0.1356, 0.1320),
                 DATASET_NUM_CLASSES_PARAM: 45,
                 DATASET_NUM_CHANNELS_PARAM: 3},
    'carla' : {DATASET_CREATOR_PARAM: create_MSCOCO,
                DATASET_TYPE: DATASET_TYPE_IMAGE,
                DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
                DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
                # This includes the background class
                DATASET_NUM_CLASSES_PARAM: 4, 
                DATASET_NUM_CHANNELS_PARAM: 3,
                DATASET_MEAN_PARAM: (0.485, 0.456, 0.406),
                DATASET_STD_PARAM: (0.229, 0.224, 0.225),
                DATASET_ANNOT_FILE: [["train", "train/annotations/coco_annotations_proc.json"], ["dev","dev/annotations/coco_annotations_proc.json"]],
                DATASET_COLLATE_FN: mscoco_collate_fn},
    'mscoco' : {DATASET_CREATOR_PARAM: create_MSCOCO,
                DATASET_TYPE: DATASET_TYPE_IMAGE,
                DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
                DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
                DATASET_NUM_CLASSES_PARAM: 91,
                DATASET_NUM_CHANNELS_PARAM: 3,
                DATASET_MEAN_PARAM: (0.0, 0.0, 0.0),
                DATASET_STD_PARAM: (1.0, 1.0, 1.0),
                DATASET_ANNOT_FILE: [["train2017","annotations/instances_train2017.json"], ["val2017","annotations/instances_val2017.json"]],
                DATASET_COLLATE_FN: mscoco_collate_fn},
    'apricot' : {DATASET_CREATOR_PARAM: create_MSCOCO,
                 DATASET_TYPE: DATASET_TYPE_IMAGE,
                 DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
                 DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
                 DATASET_NUM_CLASSES_PARAM: 91,
                 DATASET_NUM_CHANNELS_PARAM: 3,
                 DATASET_MEAN_PARAM: (0.0, 0.0, 0.0),
                 DATASET_STD_PARAM: (1.0, 1.0, 1.0),
                 DATASET_ANNOT_FILE: [["Images/dev","Annotations/apricot_dev_all_annotations.json"],
                     ["Images/test","Annotations/apricot_test_all_annotations.json"]],
                 DATASET_COLLATE_FN: mscoco_collate_fn},
    'xview' : {DATASET_CREATOR_PARAM: create_MSCOCO,
               DATASET_TYPE: DATASET_TYPE_IMAGE,
               DATASET_LOWER_LIMIT_PARAM:(0, 0, 0),
               DATASET_UPPER_LIMIT_PARAM:(1, 1, 1),
               DATASET_NUM_CLASSES_PARAM: 63,
               DATASET_NUM_CHANNELS_PARAM: 3,
               DATASET_MEAN_PARAM: (0.0, 0.0, 0.0),
               DATASET_STD_PARAM: (1.0, 1.0, 1.0),
               DATASET_ANNOT_FILE: [["train_images","train_images.json"],
                     ["test_images","test_images.json"]],
               DATASET_COLLATE_FN: mscoco_collate_fn},
    # These two versions of the UCF101 datasets have differences.
    'ucf101_npz': {DATASET_CREATOR_PARAM: create_UCF101_NPZ,
               DATASET_TYPE: DATASET_TYPE_VIDEO_FIX_SIZE,
               DATASET_LOWER_LIMIT_PARAM: (0, 0, 0),
               DATASET_UPPER_LIMIT_PARAM: (1, 1, 1),
               DATASET_MEAN_PARAM: UCF101_MEAN,
               DATASET_STD_PARAM: UCF101_STD,
               DATASET_NUM_CLASSES_PARAM: UCF101_CLASS_QTY,
               DATASET_NUM_CHANNELS_PARAM: 3,
               },
    'ucf101_jpg': {DATASET_CREATOR_PARAM: create_UCF101_JPG,
               DATASET_TYPE: DATASET_TYPE_VIDEO_FIX_SIZE,
               DATASET_LOWER_LIMIT_PARAM: (0, 0, 0),
               DATASET_UPPER_LIMIT_PARAM: (1, 1, 1),
               DATASET_MEAN_PARAM: UCF101_MEAN,
               DATASET_STD_PARAM: UCF101_STD,
               DATASET_NUM_CLASSES_PARAM: UCF101_CLASS_QTY,
               DATASET_NUM_CHANNELS_PARAM: 3,
               },

    'so2sat_npz' : {DATASET_CREATOR_PARAM: create_SO2SAT_NPZ,
                DATASET_TYPE: DATASET_TYPE_IMAGE,
                DATASET_LOWER_LIMIT_PARAM: (0,) *4  +(-1,) * (SO2SAT_CHANNEL_QTY - 4),
                DATASET_UPPER_LIMIT_PARAM: (1,) * SO2SAT_CHANNEL_QTY,
                # Computed by torch.mean(X, dim=(0,1,2)) on the training set
                DATASET_MEAN_PARAM: (-2.8057e-07, -5.9780e-08,  4.6430e-07,  1.9653e-07,  4.1942e-03,
         4.9777e-03,  5.8702e-03,  5.8612e-03,  5.9701e-03,  6.3296e-03,
         6.2185e-03,  6.7153e-03,  6.8393e-03,  6.0386e-03),
                # As computed by torch.std(X, dim=(0,1,2)) on the training set
                DATASET_STD_PARAM:  (0.0014, 0.0014, 0.0036, 0.0036, 0.0099, 0.0119, 0.0166, 0.0159, 0.0194,
        0.0228, 0.0230, 0.0254, 0.0250, 0.0220),
                DATASET_NUM_CLASSES_PARAM: SO2SAT_CLASS_QTY,
                DATASET_NUM_CHANNELS_PARAM: SO2SAT_CHANNEL_QTY
                },
    'twosix_librispeech': {DATASET_CREATOR_PARAM: create_TWOSIX_LIBRISPEECH,
                           DATASET_TYPE: DATASET_TYPE_AUDIO_FIX_SIZE,
                           DATASET_LOWER_LIMIT_PARAM: (-100,),
                           DATASET_UPPER_LIMIT_PARAM: (100,),
                           DATASET_MEAN_PARAM: (0,),
                           DATASET_STD_PARAM: (1,),
                           DATASET_NUM_CLASSES_PARAM: len(LIBRI_SPEECH_LABELS),
                           DATASET_NUM_CHANNELS_PARAM: 1}

}


def get_shaped_tensor(dataset_info, const_name, tensor_type=torch.FloatTensor):
    """This function creates a properly shaped tensor from
       a normalization or a lower/upper bound constant.

        :param dataset_info:    a data set information dictionry
        :param const_name:      a name of the constants tuples from the dataset info
        :param tensor_type:     a tensor type to use (float by default)
    """
    dtype = dataset_info[DATASET_TYPE]
    channel_qty = dataset_info[DATASET_NUM_CHANNELS_PARAM]

    if const_name not in dataset_info:
        raise Exception('Unsupported constant type:' + const_name)

    x = tensor_type(dataset_info[const_name])

    # The number of values in a norm/std tuple should match
    # the number of channels in the data!
    if dtype == DATASET_TYPE_IMAGE:
        return x.view(channel_qty, 1, 1)
    elif dtype == DATASET_TYPE_AUDIO_FIX_SIZE:
        return x.view(channel_qty, 1)
    elif dtype == DATASET_TYPE_VIDEO_FIX_SIZE:
        return x.view(channel_qty, 1, 1)
    else:
        raise Exception('Unsupported data type:' + dtype)


def create_dataset(dataset_arg_obj):
    """Create training and testing data sets. Raise an exception
    if the dataset is not supported or required arguments are missing.

    :param dataset_arg_obj:  an argument object containing at least two
                             parameters: name and dataset_root

    :return: a triple: train, test, dataset information dictionary
    """
    name = getattr(dataset_arg_obj, DATASET_NAME_PARAM, None)
    if name is None:
        raise Exception(f'Missing dataset parameter {DATASET_NAME_PARAM}')
    dataset_root = getattr(dataset_arg_obj, DATASET_ROOT_PARAM, None)
    if dataset_root is None:
        raise Exception(f'Missing dataset parameter {DATASET_ROOT_PARAM}')
    mean = getattr(dataset_arg_obj, DATASET_MEAN_PARAM, None)
    std = getattr(dataset_arg_obj, DATASET_STD_PARAM, None)
    annot_file = getattr(dataset_arg_obj, DATASET_ANNOT_FILE, None)
    add_args = getattr(dataset_arg_obj, ADD_ARG_ATTR, EmptyClass)

    name = name.lower()

    if name in DATASET_CREATORS:
        dataset_info = copy.copy(DATASET_CREATORS[name])
        if annot_file is not None:
            dataset_info[DATASET_ANNOT_FILE] = annot_file
        
        if DATASET_ANNOT_FILE in dataset_info:
            train, test = dataset_info[DATASET_CREATOR_PARAM](dataset_root,
                                                              add_args=add_args,
                                                              annot_file=dataset_info[DATASET_ANNOT_FILE])
        else:
            train, test = dataset_info[DATASET_CREATOR_PARAM](dataset_root,
                                                              add_args=add_args)

        if mean is not None:
            mean = tuple(mean)
            if len(mean) != dataset_info[DATASET_NUM_CHANNELS_PARAM]:
                raise Exception('Wrong # of elements for the mean parameter')
            dataset_info[DATASET_MEAN_PARAM] = mean

        if std is not None:
            std = tuple(std)
            if len(std) != dataset_info[DATASET_NUM_CHANNELS_PARAM]:
                raise Exception('Wrong # of elements for the std parameter')
            dataset_info[DATASET_STD_PARAM] = std

        return train, test, dataset_info

    raise Exception(f'Unsupported data set {name}')
