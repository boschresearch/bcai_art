### Section `dataset`

The dataset section defines the following key parameters:

1. `name` : a name of one of the supported datasets 
from the file [datasets_main](bcai_art/datases_main.py)

2. `root` : a dataset location

3. `mean` : a mean value used for input data mean/variance normalization

4. `std`  : a variance value used for input data mean/variance normalization

5. `add_arguments` : a nested definition of dataset-specific arguments (usually we have none)


### Dataset list

Image classification datasets:

1. `mnist`   MNIST
2. `cifar10` CIFAR-10
3. `cifar100` CIFAR-100
4. `imagenet` ImageNet
5. `resisc45` [RESISC:45 Remote Sensing Image Scene Classification: Benchmark and State of the Art
Gong Cheng, Junwei Han, Xiaoqiang Lu](https://arxiv.org/abs/1703.00121v1)

Video classificaion datasets:

1. `ucf101_npz` is a Tensorflow-datasets UCF-101 versions. 
It is downloaded and converted to a (compressed)
numpy format using `data_scripts/convert_ucf101_tfrecs_to_numpy.py`

Misc dataset:

1. `so2sat` [SO2SAT](https://www.tensorflow.org/datasets/catalog/so2sat)
It is downloaded and converted to a (compressed)
numpy format using `data_scripts/convert_so2sat_tfrecs_to_numpy.py`
2. `twosix_librispeech` a [TWOSIX Librispeech dev dataset for speaker identification](https://github.com/twosixlabs/armory/blob/master/docs/datasets.md) of the LibriSpeech dataset.