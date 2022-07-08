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

#
# This source code is from Torch-Vision
#   https://github.com/pytorch/vision
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# This source code is licensed under the BSD 3-Clause "New" or "Revised" License found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
#

from PIL import Image
import os
import os.path
import numpy as np
import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity

from bcai_art.utils_dataset import DataPartitioner

CIFAR_SPLIT_TRAIN = 'cifar_split_train'
CIFAR_SPLIT_VAL = 'cifar_split_val'
CIFAR_SPLIT_TEST = 'cifar_split_test'
CIFAR_VAL_FRACT = 0.02 # 50K * 0.02 == 1K or 1/10th the number of the actual test entries in CIFAR10

#
# This file has special modifications of the CIFAR10/100 data sets that support the following:
#
# 1. Loading *ONLY* training or testing data, which permits storing these in separate directories.
# 2. Unlike the original variant it *WILL NOT* download CIFAR10/100.
#
#    This is done on purpose to prevent accidental download of the complete CIFAR10/100 dataset
#    to the folder, which needs to store only training data.
#    This, in turn, can help ensuring we never accidentally train on test data.
#
# 3. It splits collection into train, validation, split by carving
#    out a portion of the training data.
#
# The code is a modified official CIFAR10 PyTorch loader from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
#


class CIFAR10Split(VisionDataset):
    """
        CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root,
            train,
            transform
    ):
        """
        :param root:   Root directory of dataset where directory
                       ``cifar-10-batches-py`` exists. This class only cares
                       about the part specified by the part flag:
                       if we ask for train data, it will not try to check whether
                       the test part is present and vice versa.

        :param train:  true for training data and false for test data.
        :param transform: an image transformation object
        """

        super().__init__(root, transform=transform, target_transform=None)

        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int):
        """

        :param index:  index
        :return:       tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        if self.train:
            check_list = self.train_list
        else:
            check_list = self.test_list
        for fentry in check_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100Split(CIFAR10Split):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10Split` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


# 2% of the training set is 1K images
def create_cifar_dataset_3way(root, data_part, transform, is_cifar10, val_fract=CIFAR_VAL_FRACT, val_seed=0):
    """Supports 3-way splitting of CIFAR10/100 data sets as opposed to the
       original 2-way splitting.
       The data will not be downloaded automatically and the split is NOT enforced to be
       class balanced.

    :param root:
    :param data_part:     type of the data
    :param transform:     A function/transform that takes in an PIL image
                          and returns a transformed version. E.g, ``transforms.RandomCrop``
    :param is_cifar10:    True for CIFAR10 and False for CIFAR100
    :param val_fract:     a fraction of the training data to carve out for validation.
    :param val_seed:      a random seed to randomly select validation data.

    :return:              a CIFAR10/100 data (sub)set
    """

    if data_part == CIFAR_SPLIT_TRAIN:
        train = True
    elif data_part == CIFAR_SPLIT_VAL:
        train = True
    elif data_part == CIFAR_SPLIT_TEST:
        train = False
    else:
        raise Exception(f'Invalid CIFAR set part: {data_part}')

    if is_cifar10:
        data_all = CIFAR10Split(root=root, train=train, transform=transform)
    else:
        data_all = CIFAR100Split(root=root, train=train, transform=transform)

    # We never resplit the official test data, only the
    if data_part in [CIFAR_SPLIT_TRAIN, CIFAR_SPLIT_VAL] and val_fract is not None and val_fract > 0:
        assert val_fract > 0 and val_fract < 1
        partition_obj = DataPartitioner(data_all, size_fracs=[val_fract, 1-val_fract], seed=val_seed)
        return partition_obj.use(int(data_part != CIFAR_SPLIT_VAL))
    else:
        return data_all
