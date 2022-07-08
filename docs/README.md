# BCAI ART : Bosch Center for AI Adversarial Robustness Toolkit

## Introduction

BCAI ART is a framework that provides core capabilities for 
training and testing models robust to a number of adversarial attacks.
It also comes with wrappers for a number of deep models.
We focus on fully differentiable architectures and simple modalities:
We support image, video, and audio classification. In that, 
most of our model wrappers are for image classification.

## Installation

The toolkit is organized as a Python package. For local development, one can install a 
fake package (pointing to the folder where the repository code is checked out):
```
python setup.py develop
```
In this case, when a developer changes repository code on disk, the package content 
changes accordingly.

A more standard installation procedure installs an immutable package snapshot:
```
python setup.py install
```

## Architecture overview

BCAI ART has several core abstractions:

1. `TopLevelModelWrapper` is a top-level model wrapper that standardizes
   data normalization and computation of the loss. 

2. `AttackBase` is a base class for an attack.
3. `BaseTrainer` is a base class for a model/attack trainer.
4. `BaseEvaluator` is a base class for an evaluator object.
2. `TrainEvalEnviron` is a wrapper class for training and evaluation environment that
       encapsulates a model, an optional adversarial perturbation,
       and an optimizer. It also provides a few convenience functions to simplify training,
       including a multi-device/multi-processing training.
       
Both trainer and evaluator can use an attack object to train a robust model
and to evaluate it under a given attack. The attack can be stateless,
e.g., a PGD attack, or it can have a mutable attack object such as a 
patch or a universal perturbation. 

## Training/evaluation pipeline overview

It is possible to use the framework in a programmatic fashion without a 
predefined training and evaluation order. An illustrated example of doing
so for the MNIST dataset can be found in ["Annotated MNIST" notebook](annotated_mnist.ipynb).
 
Generally, we carry out several steps where we first create:

1. a dataset
2. a model
3. a training environment
4. a training and test loaders
5. an optimizer with an optional scheduler
6. a trainer object with an optional attack object.
7. an evaluator object with an optional attack object

Then we can run training and evaluation procedures. If the procedure
involves training an attack, we run a training procedure twice:
the first time we train both the model and the attack. In the second loop,
we fix the model and only train the attack object.


## Detailed description of training pipeline parameters

A number of sample configuration files can be found in the [`sample_configs`](../sample_configs)
directory. Consider, e.g., a pipeline to evaluate an (sample_configs/mnist/mnist_pgd.json )[adversarially trained model on MNIST].
Sample scripts using these configuration files can be found in the [`test_scripts`](../test_scripts) directory. 


The JSON descriptor of a training/evaluation pipeline has the following sections, which we describe 
in separate documents. The structure of the configuration file is defined by
the  [JSON schema file](../bcai_art/config_schema.json).

1. [general](pipeline/general.md)
2. [dataset](pipeline/dataset.md)
3. [model](pipeline/model.md)
4. [training](pipeline/training.md)
5. [evaluator](pipeline/evaluator.md)


## Running the pipeline and carrying out an autoattack

To run the pipeline one simply runs the script `bcai_art_run.py` as follows:
```
./bcai_art_run.py <JSON configuration>
```
This scripts has several parameters which can override values in the JSON file.

The autoattack is carried in the same fashion, but we use the script `bcai_art_run_autoattack.py`
```
./bcai_art_run_autoattack.py <simplified JSON configuration>
```
The configuration file has the same structure as a regular pipeline, but it does not 
need to define trainers (`training` array should be empty) or evaluators (`evaluators` should be empty).
One should specify a model, a dataset, the logging directory (for evaluation), and the 
evaluation batch size. Autoattack should be used only for **image-classification**
datasets.



