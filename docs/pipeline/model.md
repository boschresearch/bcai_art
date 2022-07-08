### Section `model`

A definition of the model has the following key parameters:

1. `architecture`   a name of one of the supported models 
from the file [models_main](bcai_art/models_main.py)

2. `pretrained` for a number of models, e.g., coming from `Torchvision`,
one can download a pre-trained version by setting this option value to 'True'

3. `weights_file` is an optional file name storing a previously saved model.

4. `add_arguments` a nested definition of model-specific arguments

5. `inner_model` a nested defintion of the model when the model uses a frontend.

One most common frontend type is `robustifier`. It has the following parameters
specified via `add_arguments`.

1. `max_imagesize` maximum image size, if input image is larger than that, 
it will be downscaled.

2. `freeze_inner_model` if True, we freeze the model after the frontend (including
normalization layers).

3. `upsample_factor` upsample/downsample the image using this upsampling factor
(it can be < 1).

4. `robustifier_weights_file` robustifier weights file.


### Model list

CIFAR classificaiton models:

1. `resnet9_cifar`
2. `resnet9_small_cifar`
3. `resnet18_cifar`
4. `resnet34_cifar`
5. `resnet50_cifar`
6. `wideresnet70_16_cifar`
7. `wideresnet34_20_cifar`
8. `wideresnet34_10_cifar`
9. `wideresnet28_20_cifar`
10. `wideresnet28_10_cifar`

Imagenet classification models:

1. `densenet121`
2. `resnet18`
3. `resnet34`
4. `resnet50`
5. `resnet101`
6. `resnet152`
7. `vgg16`
8. `alexnet`

Video classification models:

1. resnext101_ucf101

Audio classification models:

1. `sincnet`

Frontend models:

1. `robustifier`


Misc. models:

1. `So2SatNet` a special model for the `so2sat` dataset.
