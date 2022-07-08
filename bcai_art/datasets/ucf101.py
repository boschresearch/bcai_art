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
# The content of this file is a modified (both simplified and extended)
# data-processing code the from the MARS-model repository:
# https://github.com/craston/MARS
# Unlike original code, which processes images directly, our
# code uses data extracted from the Tensorflow datasets.
#
import os
import torch
import random
import numpy as np
import numbers
import collections
import glob

from bcai_art.conf_attr import DATASET_REG_FILE

from torch.utils.data import Dataset
from PIL import Image

# Tensorflow datasets use this split too:
# see https://www.tensorflow.org/datasets/catalog/ucf101#ucf101ucf101_1_256_default_config
UCF101_SPLIT=1
UCF101_CLASS_QTY=101
UCF101_SAMPLE_DURATION=16
UCF101_SAMPLE_SIZE=112
UCF101_NORM_VALUE=255
UCF101_MEAN=(114.7748/UCF101_NORM_VALUE, 107.7354/UCF101_NORM_VALUE, 99.4750/UCF101_NORM_VALUE)
UCF101_STD=(1, 1, 1)

scale_choice = [1, 1/2**0.25, 1/2**0.5, 1/2**0.75, 0.5]
crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

UCF101_TFDS_DATA_KEY = 'video'
UCF101_TFDS_LABEL_KEY = 'label'

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255/norm_value].
    """

    def __init__(self, norm_value):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(self.norm_value)

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output sizeself.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self. will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, scale, size, crop_position, interpolation=Image.BILINEAR):
        self.scale = scale
        self.size = size
        self.interpolation = interpolation
        self.crop_position = crop_position

    def __call__(self, img):

        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def scale_crop_rgb(clip, train, sample_size):
    """Preprocess list(frames) based on train/test and modality.
       This is modified version of the scale_crop function from
       the MARS repository preprocess_data.py. It supports only
       the RGB model. It does not normalize frames.
       Furthermore, it returns the tensor in a different format
       than the MARS model needs: Specifically, we transpose
       the time and channel dimensions to be compatible
       with frame-level universal attacks that operate on the data
       generated by Torchvision's ToTensor, which
       produces tensor in the format (C, H, W).
       This also makes us compatible with Tensorboard that expects
       video tensor to be in the (B, T, C, H, W) format.

    Training:
        - Multiscale corner crop
        - Random Horizonatal Flip (change direction of Flow accordingly)
        - Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor
    Testing/ Validation:
        - Scale frame
        - Center crop
        - Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor

    :param  clip:        list of RGB/Flow frames
    :param  train:       1 for train, 0 for test
    :param  sample_size: Height and width of inputs (originally a parameter from opts.py file)

    :return: Tensor(frames) of shape T x C x H x W
    """
    processed_clip = torch.Tensor(len(clip), 3, sample_size, sample_size)

    flip_prob = random.random()
    scale_factor = scale_choice[random.randint(0, len(scale_choice) - 1)]
    crop_position = crop_positions[random.randint(0, len(crop_positions) - 1)]

    if train == 1:
        for i, I in enumerate(clip):
            I = MultiScaleCornerCrop(scale=scale_factor, size=sample_size, crop_position=crop_position)(I)
            I = RandomHorizontalFlip(p=flip_prob)(I)
            I = ToTensor(UCF101_NORM_VALUE)(I)
            processed_clip[i, :, :, :] = I
    else:
        for i, I in enumerate(clip):
            I = Scale(sample_size)(I)
            I = CenterCrop(sample_size)(I)
            I = ToTensor(UCF101_NORM_VALUE)(I)

            processed_clip[i, :, :, :] = I

    return (processed_clip)


def get_train_video(sample_duration, frame_path, total_frames):
    """Chooses a random clip from a video for training/validation.

    :param sample_duration:   temporal duration of inputs (originally a parameter from opts.py file)
    :param frame_path:        a path to the directory with video frames
    :param total_frames:      number of frames in the video

    :return:  random clip (list of frames of length sample_duration) from a video for training/ validation
    """

    clip = []
    i = 0
    loop = 0

    # choosing a random frame
    if total_frames <= sample_duration:
        loop = 1
        start_frame = np.random.randint(0, total_frames)
    else:
        start_frame = np.random.randint(0, total_frames - sample_duration)


    while len(clip) < sample_duration:
        try:
            im = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i + 1)))
            clip.append(im.copy())
            im.close()
        except:
            pass
        i += 1

        if loop == 1 and i == total_frames:
            i = 0

    return clip


def get_test_video(sample_duration, frame_path, total_frames):
    """Read test-video frames.

    :param sample_duration:   temporal duration of inputs, which can possibly include all the frames.
    :param frame_path:        a path to the directory with video frames
    :param total_frames:      number of frames in the video

    :return:  list of video frames whose numbers doesn't exceed the sample size (and the total # of frames)
    """

    clip = []
    i = 0
    loop = 0
    if total_frames < sample_duration: loop = 1

    while len(clip) < max(sample_duration, total_frames):
        try:
            im = Image.open(os.path.join(frame_path, '%05d.jpg' % (i + 1)))
            clip.append(im.copy())
            im.close()
        except:
            pass
        i += 1

        if loop == 1 and i == total_frames:
            i = 0

    return clip


def tfds_numpy_to_imgs(video_np, max_frames=None):
    """Convert a numpy entry (from a Tensorflow UCF101 dataset) to a list of PIL Images.

    :param video_np:      a numpy array storing the video of the shape 1 x T x H x W x C
    :param max_frames:  a max number of frames

    :return: a list of PIL Images
    """
    img_lst = []
    assert len(video_np.shape) == 5, "Video shape is supposed to have 5 elements"
    b, total_frames, h, w, c = video_np.shape
    assert b == 1, "Video batch size should be one!"

    if max_frames is None:
        max_frames = total_frames

    for fid in range(min(total_frames, max_frames)):
        img_lst.append(Image.fromarray(video_np[0, fid]))

    return img_lst


class UCF101DataSetNPZ(Dataset):
    """UCF101 dataset, which uses tensors converted from a UCF101 tensorflow tfrec format to compressed numpy files (NPZ).
       The conversion utility: convert_ucf101_tfrecs_to_numpy.py
    """

    def __init__(self,
                 train,
                 root_dir,
                 sample_duration,
                 max_test_frame_qty,
                 sample_size):
        """
        :param  train : 1 for training and 0 for testing.
        :param  root_dir: a dataset root directory.
        :param  sample_duration:   temporal duration of inputs (originally a parameter from opts.py file)
        :param  max_test_frame_qty: a maximum number of frames used for testing.
        :param  sample_size: Height and width of inputs (originally a parameter from opts.py file)

        """
        self.train = bool(train)
        self.root_dir = root_dir
        self.max_test_frame_qty = max_test_frame_qty
        self.sample_duration = sample_duration
        self.sample_size = sample_size

        split_subdir = os.path.join(root_dir, 'train' if train else 'test')
        self.data_list = torch.load(os.path.join(split_subdir, DATASET_REG_FILE))

    def __len__(self):
        """
        :return: number of dataset items
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        comp_np_file, label = self.data_list[idx]

        with open(os.path.join(self.root_dir, comp_np_file), 'rb') as f:
            video_np = np.load(f)['arr_0']

        assert len(video_np.shape) == 5, "Video shape is supposed to have 5 elements"
        b, total_frames, h, w, c = video_np.shape
        assert b == 1, "Video batch size should be one!"

        if self.train:
            clip = tfds_numpy_to_imgs(video_np, total_frames)
            assert total_frames >= self.sample_duration, \
                f'Video in {fn} is too short. Need: {self.sample_duration}, but got {total_frames}'
            start_frame = np.random.randint(0, total_frames - self.sample_duration)
            clip = clip[start_frame : start_frame + self.sample_duration]
        else:
            clip = tfds_numpy_to_imgs(video_np, min(total_frames, self.max_test_frame_qty))

        return scale_crop_rgb(clip, self.train, self.sample_size), label


class UCF101DataSetJPG(Dataset):
    """UCF101 dataset, which is created by following frame-extraction instructions from
       https://github.com/craston/MARS. Extracted frames are stored as JPG files.
    """

    def __init__(self,
                 train,
                 frame_dir,
                 annotation_path,
                 sample_duration,
                 max_test_frame_qty,
                 sample_size,
                 split):
        """
        :param  train : 1 for training and 0 for testing.
        :param  frame_dir: a path to JPG files
        :param  annotation_path: a path to the labels' file
        :param  sample_duration:   temporal duration of inputs (originally a parameter from opts.py file)
        :param  max_test_frame_qty: a maximum number of frames used for testing.
        :param  sample_size: Height and width of inputs (originally a parameter from opts.py file)
        :param  split : 1,2,3
        """

        self.is_train = train
        self.max_test_frame_qty = max_test_frame_qty
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.frame_dir = frame_dir

        with open(os.path.join(annotation_path, "classInd.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]

        with open(os.path.join(annotation_path, "classInd.txt")) as lab_file:
            index = [int(line.strip('\n').split(' ')[0]) for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 101

        self.class_idx = dict(zip(self.lab_names, index))  # Each label is mappped to a number
        self.idx_class = dict(zip(index, self.lab_names))  # Each number is mappped to a label

        # indexes for training/test set
        split_lab_filenames = sorted(
            [file for file in os.listdir(annotation_path) if file.strip('.txt')[-1] == str(split)])

        if self.is_train == 1:
            split_lab_filenames = [f for f in split_lab_filenames if 'train' in f]
        else:
            split_lab_filenames = [f for f in split_lab_filenames if 'test' in f]

        SPLIT_FILE_DESC = 'defining the data set split (into train/test)'
        if not split_lab_filenames:
            raise Exception('Cannot find the file ' + SPLIT_FILE_DESC)

        assert len(split_lab_filenames) == 1, 'There should be exactly one ' + SPLIT_FILE_DESC

        self.data = []  # (filename , lab_id)

        split_file_name = os.path.join(annotation_path, split_lab_filenames[0])

        f = open(split_file_name, 'r')
        for line in f:
            video_fn = line.strip().split()[0]
            video_no_avi_suff = video_fn[:-4]
            frame_sub_dir = os.path.join(frame_dir, video_no_avi_suff)
            class_id = self.class_idx.get(video_fn.split('/')[0]) - 1
            if os.path.exists(frame_sub_dir) == True:
                self.data.append((frame_sub_dir, class_id))

        if not self.data:
            raise Exception(f'No data files found (possibly a bug), split file name: {split_file_name} frame directory: {frame_dir}')

        f.close()

    def __len__(self):
        """
        :return: number of dataset items
        """
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        frame_path = os.path.join(self.frame_dir, self.idx_class.get(label_id + 1), video[0])

        total_frames = len(glob.glob(glob.escape(frame_path) + '/0*.jpg'))

        if self.is_train == 0:
            clip = get_test_video(self.sample_duration, frame_path, min(self.max_test_frame_qty, total_frames))
        else:
            clip = get_train_video(self.sample_duration, frame_path, total_frames)

        return scale_crop_rgb(clip, self.is_train, self.sample_size), label_id

