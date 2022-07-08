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

# This file contains helper functions to create fixed-audio-chunk data
# from twosix subset of librispeech. This subset is intended to be used
# for speaker-identification only
import re
import os
import math

import numpy as np

from tqdm import tqdm
from torch.utils.data import TensorDataset

import torch
import torchaudio

from bcai_art.utils_misc import sync_out, get_filelist_recursive
from bcai_art.utils_tensor import normalize_idx

LIBRISPEECH_SAMPLE_RATE=16000

LIBRI_SPEECH_RAND_SCALE_MIN=0.8
LIBRI_SPEECH_RAND_SCALE_MAX=1.2

# These numbers come from twosix armory's repo sincent_full.py file:
# NOTE: Underlying dataset sample rate is 16 kHz. SincNet uses this SAMPLE_RATE to
# determine internal filter high cutoff frequency.
WINDOW_SAMPLE_RATE = 8000
WINDOW_STEP_SIZE = 375
# This makes a WINDOW_LENGTH of the size 3000, which is
# approximately 0.2 sec (b/c the "window" sample rate is 8Khz)
WINDOW_LENGTH = int(WINDOW_SAMPLE_RATE * WINDOW_STEP_SIZE / 1000)

# A number of random samples from each audio when creating a training set.
# It's better to use a large number of samples, b/c training is relatively
# fast compared to pauses between epochs (apparently due to saving model snapshots)
TWOSIX_LIBRISPEECH_SAMPLE_QTY=500

#
# This is a subset of labels used Armory librispeech dataset
# Some constants in this file are from Armory as well.
#
#
# https://github.com/twosixlabs/armory/blob/master/armory/data/librispeech/librispeech_dev_clean_split.py
# Copyright (c) 2019 Two Six Labs, licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tre
#
LIBRI_SPEECH_LABELS = [
    "84",
    "174",
    "251",
    "422",
    "652",
    "777",
    "1272",
    "1462",
    "1673",
    "1919",
    "1988",
    "1993",
    "2035",
    "2078",
    "2086",
    "2277",
    "2412",
    "2428",
    "2803",
    "2902",
    "3000",
    "3081",
    "3170",
    "3536",
    "3576",
    "3752",
    "3853",
    "5338",
    "5536",
    "5694",
    "5895",
    "6241",
    "6295",
    "6313",
    "6319",
    "6345",
    "7850",
    "7976",
    "8297",
    "8842",
]


LIBRI_SPEECH_LABELS_CLASS_MAP = { LIBRI_SPEECH_LABELS[k] : k for k in range(len(LIBRI_SPEECH_LABELS))}


def frequency_cutout(chunk, freq_cutout_qty, n_fft):
    """A data augmentation technique that zeros out randomly selected
       adjacent frequency band. It operates on raw audio by converting
       the raw audio to the frequency domain, zeroing out part of the spectrum,
       and converting the result back to the raw audio domain.

    :param chunk:             a raw audio input array (it should be a single-channel plain array)
    :param freq_cutout_qty:   a number of adjacent frequency band to zero out
    :param n_fft:             size of Fourier transform

    :return: modified raw audio
    """

    chunk_spectr = torch.stft(chunk, n_fft=n_fft)

    assert len(chunk_spectr.shape) == 3
    band_qty = chunk_spectr.shape[0]
    zero_band = torch.randint(low=0, high=band_qty, size=(1,)).item()

    if False:
        print(freq_cutout_qty, n_fft)
        print(chunk_spectr.shape, zero_band, freq_cutout_qty, zero_band - freq_cutout_qty, zero_band + freq_cutout_qty)

    chunk_spectr[zero_band - freq_cutout_qty: zero_band + freq_cutout_qty, :, :] = 0 # zero the band

    res = torchaudio.functional.istft(chunk_spectr, n_fft=n_fft)

    # The result is somehow shorter than original, let's fill it with zeros
    res = torch.cat( [res, torch.zeros(len(chunk) - len(res))] )

    return res


class LibriSpeechTrainingSet:
    def __init__(self, root,
                 rand_scale_min=LIBRI_SPEECH_RAND_SCALE_MIN,
                 rand_scale_max=LIBRI_SPEECH_RAND_SCALE_MAX,
                 window_length=WINDOW_LENGTH,
                 dir_subsets=['train', 'val'],
                 sample_qty=TWOSIX_LIBRISPEECH_SAMPLE_QTY,
                 cutout_prob=None,
                 cutout_n_fft=512,
                 cutout_freq_qty=16):
        """Constructor of random sampling and rescaling training set.

        :param  root:             the data set root with train, and validation sub-folders.
        :param  rand_scale_min:   a minimum random scaling factor
        :param  rand_scale_max:   a maximum random scaling factor
        :param  window_length:    the length of each sample
        :param  sample_qty:       a number of samples per audio

        :param  cutout_prob:      a probability of applying a cutout data augmentation (or None)
        :param  cutout_n_fft:     Fourier transform size (for the cutout augmentation)
        :param  cutout_freq_qty:  a number of frequencies to zero out
        """
        self.rand_scale_min = rand_scale_min
        self.rand_scale_max = rand_scale_max
        self.window_length = window_length
        self.sample_qty = sample_qty

        self.cutout_prob = cutout_prob
        self.cutout_n_fft = cutout_n_fft
        self.cutout_freq_qty = cutout_freq_qty

        file_list = []
        for d in dir_subsets:
            file_list.extend(get_flac_audio_files(os.path.join(root, d)))

        print('# of training files', len(file_list))

        self.raw_audio = []
        self.y = []

        sync_out()

        for fn in tqdm(file_list):
            signal, sample_rate = torchaudio.load(fn)
            assert sample_rate == LIBRISPEECH_SAMPLE_RATE
            class_code = os.path.basename(os.path.dirname(os.path.dirname(fn)))
            assert class_code in LIBRI_SPEECH_LABELS_CLASS_MAP
            class_id = LIBRI_SPEECH_LABELS_CLASS_MAP[class_code]

            assert len(signal.shape) == 2
            assert signal.shape[0] == 1

            signal = normalize_librispeech_signal(signal.view(-1).numpy())

            self.raw_audio.append(signal)
            self.y.append(class_id)

        sync_out()
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.y) * self.sample_qty

    def __getitem__(self, idx):
        idx = normalize_idx(idx)
        # Now we have only two cases:
        # 1. idx is an integer
        # 2. A list of integers
        is_singleton = False
        if not isinstance(idx, list):
            idx = [idx]
            is_singleton = True

        data = []

        qty = len(self.y)

        # these are not actual indices, so we need to take module the actual # of audios
        for k in range(len(idx)):
            idx[k] = idx[k] % qty

        for i in idx:
            signal = self.raw_audio[i]
            signal_length = len(signal)
            assert signal_length > self.window_length
            start = np.random.randint(0, signal_length - self.window_length)

            scale = self.rand_scale_min + (self.rand_scale_max - self.rand_scale_min) * np.random.random()
            # print('scale', scale, 'start', start, 'window_length', self.window_length, 'len', signal.shape)

            chunk = torch.FloatTensor(signal[start: start + self.window_length])
            if self.cutout_prob is not None:
                if np.random.random() < self.cutout_prob:
                    chunk = frequency_cutout(chunk, freq_cutout_qty=self.cutout_freq_qty, n_fft=self.cutout_n_fft)

            data.append(chunk)

        labels = self.y[idx]

        assert len(labels) == len(data)

        if is_singleton:
            x = data[0]
            labels = labels[0]
        else:
            x = torch.stack(data)

        return [x, labels]


def get_flac_audio_files(root):
    """Retrieve all flac audio files given the current root (search recursively)

    :param  root:  a root folder
    :return a list of file names
    """
    return get_filelist_recursive(root, re.compile('[.]flac$'))


def normalize_librispeech_signal(x):
    """Just mimicking Armory's approach to normalize the signal."""
    return x / np.max(np.abs(x))


def random_chunk_generator(signal_length, window_length, qty):
    """A generator of random audio chunks.
       Assumption: signal_length > window_length and qty is not None.

        param:   signal_length:   the length of the audio
        param:   window_length:   the length of the sample
        param:   qty:             the number of samples per audio
    """
    assert (signal_length > window_length)
    assert (qty is not None)

    starts = np.random.randint(0, signal_length - window_length, qty)
    return list(zip(list(starts), list(starts + window_length)))


def seq_chunk_generator(signal_length, window_length, qty):
    """A generator of consequitive audio chunks. The last incomplete
       chunk is dropped. If the signal isn't long enough,
       fewer than qty: chunks are generated. If qty: is None,
       all available chunks are retrieved.

       Assumption: signal_length > window_length.

        param:   signal_length:   the length of the audio
        param:   window_length:   the length of the sample
        param:   qty:             the number of samples per audio,
                                  if None, all the chunks are retrieved
    """
    assert (signal_length > window_length)

    res = []

    max_chunk_qty = int(math.floor(signal_length / window_length))

    if qty is None:
        chunk_qty = max_chunk_qty
    else:
        chunk_qty = min(max_chunk_qty, qty)

    for k in range(chunk_qty):
        start = k * window_length
        res.append((start, start + window_length))

    return res

def create_librispeech_armory_subset(root,
                                     window_length,
                                     chunk_qty,
                                     chunk_gen_func,
                                     is_train,
                                     sub_dirs,
                                     seed=0):
    """Create a training set from Armory's librispeech training or testing subset.

    :param root:            the data set root with train, test, and validation sub-folders.
    :param window_length:   the length of the sample
    :param chunk_qty:       the number of samples per audio
    :param chunk_gen_func:  a chunk
    :param sub_dirs:        a list of sub-directories
    :param is_train:        true for the training set
    :param seed:            a random seed

    """

    file_list = []
    for d in sub_dirs:
        file_list.extend(get_flac_audio_files(os.path.join(root, d)))

    set_desc = 'training' if is_train else 'testing'
    print(f'# of {set_desc} files:', len(file_list))

    np.random.seed(seed)

    x = []
    y = []

    sync_out()

    for fn in tqdm(file_list):
        signal, sample_rate = torchaudio.load(fn)
        assert sample_rate == LIBRISPEECH_SAMPLE_RATE
        class_code = os.path.basename(os.path.dirname(os.path.dirname(fn)))
        assert class_code in LIBRI_SPEECH_LABELS_CLASS_MAP
        class_id = LIBRI_SPEECH_LABELS_CLASS_MAP[class_code]

        assert len(signal.shape) == 2
        assert signal.shape[0] == 1

        signal = normalize_librispeech_signal(signal.view(-1).numpy())
        signal_length = len(signal)

        assert (signal_length > window_length)

        for start, end in chunk_gen_func(signal_length, window_length, chunk_qty):
            x.append(signal[start: end])
            y.append(class_id)

    sync_out()

    x_t = torch.FloatTensor(np.stack(x))
    y_t = torch.LongTensor(np.stack(y))

    return TensorDataset(x_t, y_t)










