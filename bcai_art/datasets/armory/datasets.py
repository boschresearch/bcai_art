#
# This is based on the code from Armory:
# 
# https://github.com/twosixlabs/armory/blob/master/armory/data/datasets.py
# 
# Copyright (c) 2019 Two Six Labs
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
# 
# The code was simplified: 
#    1. librispeech, xview, and digits were excluded.
#    2. no variable-length input data or lable data is supported.
#    3. no shuffling of data files and no multiple-epoch data pre-generation.
#

import logging
import json
import os
import re
import torch

from typing import Callable, Union

from tqdm import tqdm

import numpy as np

# import torch before tensorflow to ensure torch.utils.data.DataLoader can utilize
#     all CPU resources when num_workers > 1
try:
    import torch  # noqa: F401
except ImportError:
    pass

import tensorflow as tf
import tensorflow_datasets as tfds

SO2SAT_CLASS_QTY = 17
SO2SAT_CHANNEL_QTY = 14

SO2SAT_TFDS_REG_FILE = ''

from bcai_art.datasets.armory.ucf101 import ucf101_clean as uc  # noqa: F401

os.environ["KMP_WARNINGS"] = "0"

logger = logging.getLogger(__name__)


def _parse_token(token: str):
    """
    Token from parse_split index

    Return parsed token
    """
    if not token:
        raise ValueError("empty token found")

    left = token.find("[")
    if left == -1:
        return token

    right = token.rfind("]")
    if right != len(token) - 1:
        raise ValueError(f"could not parse token {token} - mismatched brackets")
    name = token[:left]
    index = token[left + 1 : right]  # remove brackets
    if not name:
        raise ValueError(f"empty split name: {token}")
    if not index:
        raise ValueError(f"empty index found: {token}")
    if index.count(":") == 2:
        raise NotImplementedError(f"slice 'step' not enabled: {token}")
    elif index == "[]":
        raise ValueError(f"empty list index found: {token}")

    if re.match(r"^\d+$", index):
        # single index
        i = int(index)
        return f"{name}[{i}:{i+1}]"
    elif re.match(r"^\[\d+(\s*,\s*\d+)*\]$", index):
        # list index of nonnegative integer indices
        # out-of-order and duplicate indices are allowed
        index_list = json.loads(index)
        token_list = [f"{name}[{i}:{i+1}]" for i in index_list]
        if not token_list:
            raise ValueError
        return "+".join(token_list)

    return token


class ArmoryDataGenerator:
    """
    Returns batches of numpy data.

    variable_length - if True, returns a 1D object array of arrays for x.
    """

    def __init__(
        self,
        generator,
        size,
        epochs,
        batch_size,
        preprocessing_fn=None,
        label_preprocessing_fn=None,
        variable_length=False,
        variable_y=False,
        context=None,
    ):
        self.preprocessing_fn = preprocessing_fn
        self.label_preprocessing_fn = label_preprocessing_fn
        self.generator = generator

        self.epochs = epochs
        self.samples_per_epoch = size

        # drop_remainder is False
        self.batches_per_epoch = self.samples_per_epoch // batch_size + bool(
            self.samples_per_epoch % batch_size
        )

        self.variable_length = variable_length
        self.variable_y = variable_y
        if self.variable_length:
            self.current = 0
        elif self.variable_y:
            raise NotImplementedError("variable_y=True requires variable_length=True")

        self.context = context

    @staticmethod
    def np_1D_object_array(x_list):
        """
        Take a list of single-element batches and return as a numpy 1D object array

        Similar to np.stack, but designed to handle variable-length elements
        """
        x = np.empty((len(x_list),), dtype=object)
        for i in range(len(x_list)):
            x[i] = x_list[i][0]
        return x

    def get_batch(self) -> (np.ndarray, np.ndarray):
        if self.variable_length:
            # build the batch
            x_list, y_list = [], []
            for i in range(self.batch_size):
                x_i, y_i = next(self.generator)
                x_list.append(x_i)
                y_list.append(y_i)
                self.current += 1
                # handle end of epoch partial batches
                if self.current == self.samples_per_epoch:
                    self.current = 0
                    break

            if isinstance(x_list[0], dict):
                # Translate a list of dicts into a dict of arrays
                x = {}
                for k in x_list[0].keys():
                    x[k] = self.np_1D_object_array([x_i[k] for x_i in x_list])
            elif isinstance(x_list[0], tuple):
                # Translate a list of tuples into a tuple of arrays
                x = tuple(self.np_1D_object_array(i) for i in zip(*x_list))
            else:
                x = self.np_1D_object_array(x_list)

            if self.variable_y:
                if isinstance(y_list[0], dict):
                    # Store y as a list of dicts
                    y = y_list
                elif isinstance(y_list[0], tuple):
                    # Translate a list of tuples into a tuple of arrays
                    y = tuple(self.np_1D_object_array(i) for i in zip(*y_list))
                else:
                    y = self.np_1D_object_array(y_list)
            else:
                if isinstance(y_list[0], dict):
                    y = {}
                    for k in y_list[0].keys():
                        y[k] = np.hstack([y_i[k] for y_i in y_list])
                elif isinstance(y_list[0], tuple):
                    y = tuple(np.hstack(i) for i in zip(*y_list))
                else:
                    y = np.hstack(y_list)
        else:
            x, y = next(self.generator)

        if self.label_preprocessing_fn:
            y = self.label_preprocessing_fn(x, y)

        if self.preprocessing_fn:
            # Apply preprocessing to multiple inputs as needed
            if isinstance(x, dict):
                x = {k: self.preprocessing_fn(v) for (k, v) in x.items()}
            elif isinstance(x, tuple):
                x = tuple(self.preprocessing_fn(i) for i in x)
            else:
                x = self.preprocessing_fn(x)
        return x, y

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def __len__(self):
        return self.batches_per_epoch * self.epochs


def parse_split_index(split: str):
    """
    Take a TFDS split argument and rewrite index arguments such as:
        test[10] --> test[10:11]
        test[[1, 5, 7]] --> test[1:2]+test[5:6]+test[7:8]
    """
    if not isinstance(split, str):
        raise ValueError(f"split must be str, not {type(split)}")
    if not split.strip():
        raise ValueError("split cannot be empty")

    tokens = split.split("+")
    tokens = [x.strip() for x in tokens]
    output_tokens = [_parse_token(x) for x in tokens]
    return "+".join(output_tokens)


def filter_by_index(dataset: "tf.data.Dataset", index: list, dataset_size: int):
    """
    index must be a list or iterable of integer values

    returns the dataset and the indexed size
    """
    logger.info(f"Filtering dataset to the following indices: {index}")
    dataset_size = int(dataset_size)
    if len(index) == 0:
        raise ValueError(
            "The specified dataset 'index' param must have at least one value"
        )
    valid_indices = sorted([int(x) for x in set(index) if int(x) < dataset_size])
    num_valid_indices = len(valid_indices)
    if num_valid_indices == 0:
        raise ValueError(
            f"The specified dataset 'index' param values all exceed dataset size of {dataset_size}"
        )
    elif index[0] < 0:
        raise ValueError("The specified dataset 'index' values must be nonnegative")
    elif num_valid_indices != len(set(index)):
        logger.warning(
            f"All dataset 'index' values exceeding dataset size of {dataset_size} are being ignored"
        )

    index_tensor = tf.constant(index, dtype=tf.int64)

    def enum_index(i, x):
        i = tf.expand_dims(i, 0)
        out, _ = tf.raw_ops.ListDiff(x=i, y=index_tensor, out_idx=tf.int64)
        return tf.equal(tf.size(out), 0)

    return dataset.enumerate().filter(enum_index).map(lambda i, x: x), num_valid_indices


def filter_by_class(dataset: "tf.data.Dataset", class_ids: Union[list, int]):
    """
    class_ids must be an int or list of ints

    returns the dataset filtered by class id, keeping elements with label in class_ids
    """
    logger.info(f"Filtering dataset to the following class IDs: {class_ids}")
    if len(class_ids) == 0:
        raise ValueError(
            "The specified dataset 'class_ids' param must have at least one value"
        )

    def _filter_by_class(x, y, classes_to_keep=tf.constant(class_ids, dtype=tf.int64)):
        isallowed_array = tf.equal(classes_to_keep, tf.cast(y, tf.int64))
        isallowed = tf.reduce_sum(tf.cast(isallowed_array, tf.int64))
        return tf.greater(isallowed, tf.constant(0, dtype=tf.int64))

    filtered_ds = dataset.filter(_filter_by_class)

    if tf.executing_eagerly():
        filtered_ds_size = int(filtered_ds.reduce(0, lambda x, _: x + 1).numpy())
    else:
        filtered_ds_size = len(list(tfds.as_numpy(filtered_ds)))

    if filtered_ds_size == 0:
        raise ValueError(
            "All elements of dataset were removed. Please ensure the specified class_ids appear in the dataset"
        )

    return filtered_ds, filtered_ds_size


def parse_str_slice(index: str):
    """
    Parse simple slice from string
    """
    index = (
        index.strip().lstrip("[").rstrip("]").strip()
    )  # remove brackets and white space
    tokens = index.split(":")
    if len(tokens) != 2:
        raise ValueError("Slice needs a single ':' character. No fancy slicing.")

    lower, upper = [int(x.strip()) if x.strip() else None for x in tokens]
    if lower is not None and lower < 0:
        raise ValueError(f"slice lower {lower} must be nonnegative")
    if upper is not None and lower is not None and upper <= lower:
        raise ValueError(
            f"slice upper {upper} must be strictly greater than lower {lower}"
        )
    return lower, upper


def filter_by_str_slice(dataset: "tf.data.Dataset", index: str, dataset_size: int):
    """
    returns the dataset and the indexed size
    """
    lower, upper = parse_str_slice(index)
    if lower is None:
        lower = 0
    if upper is None:
        upper = dataset_size
    if lower >= dataset_size:
        raise ValueError(f"lower {lower} must be less than dataset_size {dataset_size}")
    if upper > dataset_size:
        upper = dataset_size
    indexed_size = upper - lower

    def slice_index(i, x):
        return (i >= lower) & (i < upper)

    return dataset.enumerate().filter(slice_index).map(lambda i, x: x), indexed_size


def _generator_from_tfds(
    dataset_name: str,
    split: str,
    dataset_dir: str,
    # preprocessing_fn is used by the Armory code and it's applied to numpy arrays
    preprocessing_fn: Callable,
    # in contrast torch_transform is applied after Tensors are generated, but
    #  *AFTER* the tensors are permuted
    torch_transform: Callable=None,
    label_preprocessing_fn: Callable = None,
    as_supervised: bool = True,
    supervised_xy_keys=None,
    download_and_prepare_kwargs=None,
    framework: str = "numpy",
    lambda_map: Callable = None,
    context=None,
    class_ids=None,
    index=None
):
    """
    If as_supervised=False, must designate keys as a tuple in supervised_xy_keys:
        supervised_xy_keys=('video', 'label')  # ucf101 dataset
        supervised_xy_keys=('speech', 'text')  # librispeech-dev-clean with ASR
    lambda_map - if not None, mapping function to apply to dataset elements
    """

    permute_tensor = None
    if context is not None:
        permute_tensor = context.permute_tensor

    default_graph = tf.compat.v1.keras.backend.get_session().graph

    if not isinstance(split, str):
        raise ValueError(f"split must be str, not {type(split)}")

    try:
        ds, ds_info = tfds.load(
            dataset_name,
            split=split,
            as_supervised=as_supervised,
            data_dir=dataset_dir,
            with_info=True,
            download_and_prepare_kwargs=download_and_prepare_kwargs,
            shuffle_files=False,
        )
    except AssertionError as e:
        if not str(e).startswith("Unrecognized instruction format: "):
            raise
        logger.warning(f"Caught AssertionError in TFDS load split argument: {e}")
        logger.warning(f"Attempting to parse split {split}")
        split = parse_split_index(split)
        logger.warning(f"Replacing split with {split}")
        ds, ds_info = tfds.load(
            dataset_name,
            split=split,
            as_supervised=as_supervised,
            data_dir=dataset_dir,
            with_info=True,
            download_and_prepare_kwargs=download_and_prepare_kwargs,
            shuffle_files=False,
        )

    if not as_supervised:
        try:
            x_key, y_key = supervised_xy_keys
        except (TypeError, ValueError):
            raise ValueError(
                f"When as_supervised=False, supervised_xy_keys must be a (x_key, y_key)"
                f" tuple, not {supervised_xy_keys}"
            )
        for key in [x_key, y_key]:
            if not (isinstance(key, str) or isinstance(key, tuple)):
                raise ValueError(
                    f"supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    f" not {type(x_key), type(y_key)}"
                )
        if isinstance(x_key, tuple):
            if isinstance(y_key, tuple):
                raise ValueError(
                    "Only one of (x_key, y_key) can be a tuple while the other must be a string."
                )
            for k in x_key:
                if not (isinstance(k, str)):
                    raise ValueError(
                        "supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    )
            ds = ds.map(lambda x: (tuple(x[k] for k in x_key), x[y_key]))
        elif isinstance(y_key, tuple):
            for k in y_key:
                if not (isinstance(k, str)):
                    raise ValueError(
                        "supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    )
            ds = ds.map(lambda x: (x[x_key], tuple(x[k] for k in y_key)))
        else:
            ds = ds.map(lambda x: (x[x_key], x[y_key]))
    if lambda_map is not None:
        ds = ds.map(lambda_map)

    dataset_size = ds_info.splits[split].num_examples

    # Add class-based filtering
    if class_ids is not None:
        if split == "train":
            logger.warning(
                "Filtering by class entails iterating over the whole dataset and thus "
                "can be very slow if using the 'train' split"
            )
        if isinstance(class_ids, list):
            ds, dataset_size = filter_by_class(ds, class_ids=class_ids)
        elif isinstance(class_ids, int):
            ds, dataset_size = filter_by_class(ds, class_ids=[class_ids])
        else:
            raise ValueError(
                f"class_ids must be a list, int, or None, not {type(class_ids)}"
            )

    # Add index-based filtering
    if isinstance(index, list):
        ds, dataset_size = filter_by_index(ds, index, dataset_size)
    elif isinstance(index, str):
        ds, dataset_size = filter_by_str_slice(ds, index, dataset_size)
    elif index is not None:
        raise ValueError(f"index must be a list, str, or None, not {type(index)}")

    #ds = ds.repeat(epochs)
    ds = ds.batch(1, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    if framework != "numpy" and (
        preprocessing_fn is not None or label_preprocessing_fn is not None
    ):
        raise ValueError(
            f"Data/label preprocessing functions only supported for numpy framework.  Selected {framework} framework"
        )

    if framework == "numpy":
        ds = tfds.as_numpy(ds, graph=default_graph)
        generator = ArmoryDataGenerator(
            ds,
            size=dataset_size,
            batch_size=1,
            epochs=1,
            preprocessing_fn=preprocessing_fn,
            label_preprocessing_fn=label_preprocessing_fn,
            variable_length=False,
            variable_y=False,
            context=context,
        )
        return ArmoryToTorchGenerator(generator, permute_tensor=permute_tensor, torch_transform=torch_transform)

    elif framework == "pytorch":
        return TFToTorchGenerator(ds, permute_tensor=permute_tensor, torch_transform=torch_transform)

    raise ValueError(f"`framework` must be one of ['tf', 'pytorch', 'numpy']. Found {framework}")



def preprocessing_chain(*args):
    """
    Wraps and returns a sequence of functions
    """
    functions = [x for x in args if x is not None]
    if not functions:
        return None

    def wrapped(x):
        for function in functions:
            x = function(x)
        return x

    return wrapped


def check_shapes(actual, target):
    """
    Ensure that shapes match, ignoring None values

    actual and target should be tuples
        actual should not have None values
    """
    if type(actual) != tuple:
        raise ValueError(f"actual shape {actual} is not a tuple")
    if type(target) != tuple:
        raise ValueError(f"target shape {target} is not a tuple")
    if None in actual:
        raise ValueError(f"None should not be in actual shape {actual}")
    if len(actual) != len(target):
        raise ValueError(f"len(actual) {len(actual)} != len(target) {len(target)}")
    for a, t in zip(actual, target):
        if a != t and t is not None:
            raise ValueError(f"shape {actual} does not match shape {target}")


class ImageContext:
    def __init__(self, x_shape, permute_tensor=(2, 0, 1)):
        self.x_shape = x_shape
        self.permute_tensor = permute_tensor
        self.input_type = np.uint8
        self.input_min = 0
        self.input_max = 255

        self.quantization = 255

        self.output_type = np.float32
        self.output_min = 0.0
        self.output_max = 1.0


class VideoContext(ImageContext):
    def __init__(self, x_shape, frame_rate, permute_tensor=(0, 3, 1, 2)):
        super().__init__(x_shape)
        self.frame_rate = frame_rate
        self.permute_tensor = permute_tensor


def canonical_image_preprocess(context, batch):
    check_shapes(batch.shape, (None,) + context.x_shape)
    if batch.dtype != context.input_type:
        raise ValueError("input batch dtype {batch.dtype} != {context.input_type}")
    assert batch.min() >= context.input_min
    assert batch.max() <= context.input_max

    batch = batch.astype(context.output_type) / context.quantization

    if batch.dtype != context.output_type:
        raise ValueError("output batch dtype {batch.dtype} != {context.output_type}")
    assert batch.min() >= context.output_min
    assert batch.max() <= context.output_max

    return batch


def canonical_variable_image_preprocess(context, batch):
    """
    Preprocessing when images are of variable size
    """
    if batch.dtype == np.object:
        for x in batch:
            check_shapes(x.shape, context.x_shape)
            assert x.dtype == context.input_type
            assert x.min() >= context.input_min
            assert x.max() <= context.input_max

        quantized_batch = np.zeros_like(batch, dtype=np.object)
        for i in range(len(batch)):
            quantized_batch[i] = (
                batch[i].astype(context.output_type) / context.quantization
            )
        batch = quantized_batch
    elif batch.dtype == context.input_type:
        check_shapes(batch.shape, (None,) + context.x_shape)
        assert batch.dtype == context.input_type
        assert batch.min() >= context.input_min
        assert batch.max() <= context.input_max

        batch = batch.astype(context.output_type) / context.quantization
    else:
        raise ValueError(
            f"input dtype {batch.dtype} not in ({context.input_type}, 'O')"
        )

    for x in batch:
        assert x.dtype == context.output_type
        assert x.min() >= context.output_min
        assert x.max() <= context.output_max

    return batch


mnist_context = ImageContext(x_shape=(28, 28, 1))
cifar10_context = ImageContext(x_shape=(32, 32, 3))
gtsrb_context = ImageContext(x_shape=(None, None, 3))
resisc45_context = ImageContext(x_shape=(256, 256, 3))
imagenette_context = ImageContext(x_shape=(None, None, 3))
xview_context = ImageContext(x_shape=(None, None, 3))
ucf101_context = VideoContext(x_shape=(None, None, None, 3), frame_rate=25)


def mnist_canonical_preprocessing(batch):
    return canonical_image_preprocess(mnist_context, batch)


def cifar10_canonical_preprocessing(batch):
    return canonical_image_preprocess(cifar10_context, batch)


def gtsrb_canonical_preprocessing(batch):
    return canonical_variable_image_preprocess(gtsrb_context, batch)


def resisc45_canonical_preprocessing(batch):
    return canonical_image_preprocess(resisc45_context, batch)


def imagenette_canonical_preprocessing(batch):
    return canonical_variable_image_preprocess(imagenette_context, batch)


def xview_canonical_preprocessing(batch):
    return canonical_variable_image_preprocess(xview_context, batch)


def ucf101_canonical_preprocessing(batch):
    return canonical_variable_image_preprocess(ucf101_context, batch)


def canonical_audio_preprocess(context, batch):
    if batch.dtype == np.object:
        for x in batch:
            check_shapes(x.shape, context.x_shape)
            assert x.dtype == context.input_type
            assert x.min() >= context.input_min
            assert x.max() <= context.input_max

        batch = np.array(
            [x.astype(context.output_type) / context.quantization for x in batch],
            dtype=object,
        )

        for x in batch:
            assert x.dtype == context.output_type
            assert x.min() >= context.output_min
            assert x.max() <= context.output_max
        return batch

    check_shapes(batch.shape, (None,) + context.x_shape)
    assert batch.dtype == context.input_type
    assert batch.min() >= context.input_min
    assert batch.max() <= context.input_max

    batch = batch.astype(context.output_type) / context.quantization  # 2**15

    assert batch.dtype == context.output_type
    assert batch.min() >= context.output_min
    assert batch.max() <= context.output_max

    return batch


def mnist(
    split: str = "train",
    dataset_dir: str = None,
    preprocessing_fn: Callable = mnist_canonical_preprocessing,
    torch_transform : Callable = None,
    fit_preprocessing_fn: Callable = None,
    framework: str = "numpy",
    **kwargs,
):
    """
    Handwritten digits dataset:
        http://yann.lecun.com/exdb/mnist/
    """
    preprocessing_fn = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)

    return _generator_from_tfds(
        "mnist:3.0.1",
        split=split,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        torch_transform=torch_transform,
        framework=framework,
        context=mnist_context,
        **kwargs,
    )


def cifar10(
    split: str = "train",
    dataset_dir: str = None,
    preprocessing_fn: Callable = cifar10_canonical_preprocessing,
    torch_transform : Callable = None,
    fit_preprocessing_fn: Callable = None,
    framework: str = "numpy",
    **kwargs,
):
    """
    Ten class image dataset:
        https://www.cs.toronto.edu/~kriz/cifar.html
    """
    preprocessing_fn = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)

    return _generator_from_tfds(
        "cifar10:3.0.2",
        split=split,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        torch_transform=torch_transform,
        framework=framework,
        context=cifar10_context,
        **kwargs,
    )


def imagenette(
    split: str = "train",
    dataset_dir: str = None,
    preprocessing_fn: Callable = imagenette_canonical_preprocessing,
    torch_transform : Callable = None,
    fit_preprocessing_fn: Callable = None,
    framework: str = "numpy",
    **kwargs,
):
    """
    Smaller subset of 10 classes of Imagenet
        https://github.com/fastai/imagenette
    """
    preprocessing_fn = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)

    return _generator_from_tfds(
        "imagenette/full-size:0.1.0",
        split=split,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        torch_transform=torch_transform,
        framework=framework,
        context=imagenette_context,
        **kwargs,
    )


def german_traffic_sign(
    split: str = "train",
    preprocessing_fn: Callable = gtsrb_canonical_preprocessing,
    torch_transform : Callable = None,
    dataset_dir: str = None,
    framework: str = "numpy",
    **kwargs,
):
    """
    German traffic sign dataset with 43 classes and over 50,000 images.
    """
    return _generator_from_tfds(
        "german_traffic_sign:3.0.0",
        split=split,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        torch_transform=torch_transform,
        framework=framework,
        context=gtsrb_context,
        **kwargs,
    )


def resisc45(
    split: str = "train",
    dataset_dir: str = None,
    preprocessing_fn: Callable = resisc45_canonical_preprocessing,
    torch_transform : Callable = None,
    fit_preprocessing_fn: Callable = None,
    framework: str = "numpy",
    **kwargs,
):
    """
    REmote Sensing Image Scene Classification (RESISC) dataset
        http://http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html

    Contains 31,500 images covering 45 scene classes with 700 images per class

    Uses TFDS:
        https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image/resisc45.py

    Dimensions of X: (31500, 256, 256, 3) of uint8, ~ 5.8 GB in memory
        Each sample is a 256 x 256 3-color (RGB) image
    Dimensions of y: (31500,) of int, with values in range(45)

    split - one of ("train", "validation", "test")
    """
    preprocessing_fn = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)

    return _generator_from_tfds(
        "resisc45_split:3.0.0",
        split=split,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        torch_transform=torch_transform,
        framework=framework,
        context=resisc45_context,
        **kwargs,
    )


def ucf101(
    split: str = "train",
    dataset_dir: str = None,
    preprocessing_fn: Callable = ucf101_canonical_preprocessing,
    torch_transform : Callable = None,
    fit_preprocessing_fn: Callable = None,
    framework: str = "numpy",
    **kwargs,
):
    """
    UCF 101 Action Recognition Dataset
        https://www.crcv.ucf.edu/data/UCF101.php
    """
    preprocessing_fn = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)

    return _generator_from_tfds(
        "ucf101/ucf101_1:2.0.0",
        split=split,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        torch_transform=torch_transform,
        as_supervised=False,
        supervised_xy_keys=("video", "label"),
        framework=framework,
        context=ucf101_context,
        **kwargs,
    )


def ucf101_clean(
    split: str = "train",
    dataset_dir: str = None,
    preprocessing_fn: Callable = ucf101_canonical_preprocessing,
    fit_preprocessing_fn: Callable = None,
    torch_transform : Callable = None,
    framework: str = "numpy",
    **kwargs,
):
    """
    UCF 101 Action Recognition Dataset with high quality MPEG extraction
        https://www.crcv.ucf.edu/data/UCF101.php
    """
    preprocessing_fn = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)

    return _generator_from_tfds(
        "ucf101_clean/ucf101_1:2.0.0",
        split=split,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        torch_transform=torch_transform,
        as_supervised=False,
        supervised_xy_keys=("video", "label"),
        framework=framework,
        context=ucf101_context,
        **kwargs,
    )


def xview_label_preprocessing(x, y):
    """
    Converts boxes from TF format to PyTorch format
    TF format: [y1/height, x1/width, y2/height, x2/width]
    PyTorch format: [x1, y1, x2, y2] (unnormalized)

    Additionally, if batch_size is 1, this function converts the single y dictionary
    to a list of length 1.
    """
    y_preprocessed = []
    # This will be true only when batch_size is 1
    if isinstance(y, dict):
        y = [y]
    for i, label_dict in enumerate(y):
        orig_boxes = label_dict["boxes"].reshape((-1, 4))
        converted_boxes = orig_boxes[:, [1, 0, 3, 2]]
        height, width = x[i].shape[:2]
        converted_boxes *= [width, height, width, height]
        label_dict["boxes"] = converted_boxes
        y_preprocessed.append(label_dict)
    return y_preprocessed


class So2SatContext:
    def __init__(self, permute_tensor=(2, 0, 1)):
        self.default_type = np.float32
        self.default_float = np.float32

        self.permute_tensor = permute_tensor

        self.x_dimensions = (
            None,
            32,
            32,
            14,
        )
        self.quantization = np.concatenate(
            (
                128 * np.ones((1, 1, 1, 4), dtype=np.float32),
                4 * np.ones((1, 1, 1, 10), dtype=np.float32),
            ),
            axis=-1,
        )


so2sat_context = So2SatContext()


def so2sat_canonical_preprocessing(batch):
    if batch.ndim != len(so2sat_context.x_dimensions):
        raise ValueError(
            f"input batch dim {batch.ndim} != {len(so2sat_context.x_dimensions)}"
        )
    for dim, (source, target) in enumerate(
        zip(batch.shape, so2sat_context.x_dimensions)
    ):
        pass
    assert batch.dtype == so2sat_context.default_type
    assert batch.shape[1:] == so2sat_context.x_dimensions[1:]

    batch = batch.astype(so2sat_context.default_float) / so2sat_context.quantization
    assert batch.dtype == so2sat_context.default_float
    assert batch.max() <= 1.0
    assert batch.min() >= -1.0

    return batch


def so2sat_tfrecs(
    split: str = "train",
    dataset_dir: str = None,
    preprocessing_fn: Callable = so2sat_canonical_preprocessing,
    fit_preprocessing_fn: Callable = None,
    torch_transform : Callable = None,
    framework: str = "numpy",
    **kwargs,
):
    """
    Multimodal SAR / EO image dataset
    """
    preprocessing_fn = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)

    return _generator_from_tfds(
        "so2sat/all:2.1.0",
        split=split,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        torch_transform=torch_transform,
        as_supervised=False,
        supervised_xy_keys=(("sentinel1", "sentinel2"), "label"),
        lambda_map=so2sat_concat_map,
        framework=framework,
        context=so2sat_context,
        **kwargs,
    )


def so2sat_concat_map(x, y):
    try:
        x1, x2 = x
    except (ValueError, TypeError):
        raise ValueError(
            "so2 dataset intermediate format corrupted. Should be in format (sentinel1,sentinel2),label"
        )
    return tf.concat([x1[..., :4], x2], -1), y


def _parse_dataset_name(dataset_name: str):
    try:
        name_config, version = dataset_name.split(":")
        splits = name_config.split("/")
        if len(splits) > 2:
            raise ValueError
        name = splits[0]
        config = splits[1:]
        subpath = os.path.join(*config + [version])
    except ValueError:
        raise ValueError(
            f'Dataset name "{dataset_name}" not properly formatted.\n'
            'Should be formatted "<name>[/<config>]:<version>", '
            'where "[]" indicates "/<config>" is optional.'
        )
    return name, subpath


class ArmoryToTorchGenerator(torch.utils.data.Dataset):
    def __init__(self, armory_generator, permute_tensor, torch_transform):
        super().__init__()

        x_list = []
        y_list = []

        for x, y in tqdm(armory_generator, 'converting NUMPY records to PyTorch tensors'):
            x_torch = torch.from_numpy(x).squeeze(dim=0) # Removing singleton batch dimension
            if permute_tensor is not None:
                x_torch = x_torch.permute(permute_tensor)
            if torch_transform is not None:
                x_torch = torch_transform(x_torch)

            x_list.append(x_torch)
            y_list.append(torch.LongTensor(torch.from_numpy(y)))

        self.len = len(x_list)

        # We stack x, becuse it doesn't have the batch dimension, but
        # concatenate y, which does have a singleton batch dimension
        self.data_x = torch.stack(x_list)
        self.data_y = torch.cat(y_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]


class TFToTorchGenerator(torch.utils.data.Dataset):
    def __init__(self, tf_dataset, permute_tensor, torch_transform):
        super().__init__()
        self.tf_dataset = tf_dataset

        self.len = tf.data.experimental.cardinality(tf_dataset)
        if self.len == tf.data.experimental.UNKNOWN_CARDINALITY or \
            self.len == tf.data.experimental.INFINITE_CARDINALITY:
            raise Exception('Data set has unknown/infinite cardinlity!')
        self.len == int(self.len)

        x_list = []
        y_list = []

        for ex in tqdm(self.tf_dataset.take(-1), 'converting TF records to PyTorch tensors'):
            x, y = ex
            # separately handle benign/adversarial data formats
            if isinstance(x, tuple):
                x_torch = (
                    torch.from_numpy(x[0].numpy()),
                    torch.from_numpy(x[1].numpy()),
                )
            else:
                x_torch = torch.from_numpy(x.numpy())

            if isinstance(y, dict):
                y_torch = {}
                for k, v in y.items():
                    if isinstance(v, tf.Tensor):
                        y_torch[k] = torch.from_numpy(v.numpy())
                    else:
                        raise ValueError(
                            f"Expected all values to be of type tf.Tensor, but value at key {k} is of type {type(v)}"
                        )
            else:
                y_torch = torch.from_numpy(y.numpy())

            x_torch = x_torch.squeeze(dim=0) # Removing singleton batch dimension
            if permute_tensor is not None:
                x_torch = x_torch.permute(permute_tensor)
            if torch_transform is not None:
                x_torch = torch_transform(x_torch)

            x_list.append(x_torch)
            y_list.append(torch.LongTensor(y_torch))

        assert len(x_list) == self.len
        # We stack x, becuse it doesn't have the batch dimension, but
        # concatenate y, which does have a singleton batch dimension
        self.data_x = torch.stack(x_list)
        self.data_y = torch.cat(y_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]



