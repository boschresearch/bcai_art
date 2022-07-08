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
import numpy as np
import os
import math
import threading as thread

from time import time
from tqdm import tqdm

import torch.optim as optim
import torch.distributed as dist

from bcai_art.datasets_main import DATASET_TYPE, DATASET_LOWER_LIMIT_PARAM, DATASET_UPPER_LIMIT_PARAM, \
                                    DATASET_MEAN_PARAM, DATASET_STD_PARAM,  DATASET_NUM_CLASSES_PARAM

from bcai_art.utils_misc import args_to_paramdict, sync_out, \
                                NORM_INF, get_norm_code, \
                                DATASET_TYPE_AUDIO_FIX_SIZE, DATASET_TYPE_IMAGE, DATASET_TYPE_VIDEO_FIX_SIZE
from bcai_art.utils_tensor import project, clamp, clear_device_cache, apply_func

from bcai_art.utils_misc import BATCH_TIME_KEY, ACCURACY_KEY, LOSS_KEY, LR_KEY, \
                                ADV_PERTURB_KEY,\
                                model_validity, set_reqgrads, get_trainable_param_count
from bcai_art.conf_attr import FREEZE_INNER_MODEL_ATTR

from bcai_art.utils_patch import derandomized_ablate_batch

from bcai_art.models_main import LOSS_OUTPUT, PREDICTION_OUTPUT, RobustifyWrapper
from bcai_art.metrics_main import create_metric

from bcai_art.conf_attr import \
    MODEL_ARGS_KEY, SAVE_MODEL_EPOCH, SAVE_MODEL_OPTIM, SAVE_MODEL_KEY, INP_SHAPE_KEY, MODEL_ARGS_KEY

REPORT_QTY=4

# five minutes should be enough
BARRIER_WAIT_TIMEOUT=60*5

STEP_SIZE_UP_PARAM = 'step_size_up'
STEP_SIZE_DOWN_PARAM = 'step_size_down'

PCT_START_PARAM = 'pct_start'
CYCLE_QTY_PARAM = 'cycle_qty'
TOTAL_STEPS_PARAM = 'total_steps'
MAX_DECAY_PARAM = 'max_decay'

TRAIN_PARALLEL_INDEPENDENT = 'independent'
TRAIN_PARALLEL_DATA = 'data'
TRAIN_PARALLEL_ATTACK = 'attack'

TRAINER_ADV_FREE = 'adv_free'
TRAINER_ADV_PLUG_ATTACK = 'adv_plug_attack'
TRAINER_NORMAL = 'normal'

DEBUG_PRINT = False


def compute_lr_steps(epoch_qty, train_set, batch_size):
    """A simple function to compute/estimate the number of optimizer/scheduler steps.

    :param epoch_qty:   #of epochs
    :param train_set:   training set object
    :param batch_size:  an (adjusted) size of the batch
    :return: the number of steps
    """
    return epoch_qty * int(math.floor(len(train_set) / batch_size))


def create_optimizer(optim_type, model, lr, add_args):
    """Create an optimizer.

    :param optim_type:   an optimizer type/name (case insensitive).
    :param model:        a model to optimize
    :param lr:           a learning rate
    :param add_args:     an object with additional parameters
    :return:
    """
    add_args_dict = args_to_paramdict(add_args, [])
    optim_type = optim_type.lower()
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, **add_args_dict)
    elif optim_type == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr, **add_args_dict)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, **add_args_dict)
    else:
        raise Exception('Unsupported optimizere:' + optim_type)
    return optimizer


def create_scheduler(opt, scheduler_type, lr_steps, add_args):
    add_args_dict = args_to_paramdict(add_args, [TOTAL_STEPS_PARAM])
    if scheduler_type == 'cyclic':
        # These parameters aren't supported by PyTorch scheduler,
        # but they are, in fact, super useful. However, they
        # are incompatible with manual settings for the step sizes
        if PCT_START_PARAM in add_args_dict or CYCLE_QTY_PARAM in add_args_dict:
            if STEP_SIZE_DOWN_PARAM in add_args_dict or STEP_SIZE_DOWN_PARAM in add_args_dict:
                raise Exception('Step size parameters should not be used with settings' +
                                f' {PCT_START_PARAM} and {CYCLE_QTY_PARAM}/1')
            pct_start = 0.5
            cycle_qty = 1
            if PCT_START_PARAM in add_args_dict:
                pct_start = add_args_dict[PCT_START_PARAM]
            if CYCLE_QTY_PARAM in add_args_dict:
                cycle_qty = add_args_dict[CYCLE_QTY_PARAM]

            add_args_dict[STEP_SIZE_UP_PARAM] = int(pct_start * lr_steps / cycle_qty)
            add_args_dict[STEP_SIZE_DOWN_PARAM] = int((1 - pct_start) * lr_steps / cycle_qty)
            # These parameters are ours, they aren't natively supported by CyclicLR
            del(add_args_dict[CYCLE_QTY_PARAM])
            del (add_args_dict[PCT_START_PARAM])

        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, **add_args_dict)
    elif scheduler_type == 'one_cycle':
        # A "super-convergence" LR
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                        total_steps = lr_steps,
                                                        **add_args_dict)
    elif scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                         gamma=0.1,
                                                         **add_args_dict)
    elif scheduler_type == 'exponential':
        if MAX_DECAY_PARAM not in add_args_dict:
            raise Exception('Exponential scheduler requires parameter: ' + MAX_DECAY_PARAM)
        max_decay = add_args_dict[MAX_DECAY_PARAM]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: max_decay ** (step/lr_steps))
    else:
        raise Exception('Unsupported scheduler:' + scheduler_type)

    return scheduler


def create_trainer(trainer_type, train_env, trainer_attack, trainer_args):
    """Create a trainer object

    :param trainer_type:    a trainer type
    :param train_env:       a trainer trainer environment
    :param trainer_attack:  a trainer attack object (which can be None for some trainers)
    :param trainer_args:    additional trainer arguments
    :return:
    """
    trainer_args_dict = args_to_paramdict(trainer_args, [])
    if trainer_type == TRAINER_NORMAL:
        trainer = NormalTrainer(train_env, **trainer_args_dict)
    elif trainer_type == TRAINER_ADV_PLUG_ATTACK:
        if trainer_attack is None:
            raise Exception('The attack needs to be defined for the trainer: ' + trainer_type)
        trainer = AdversarialTrainer(train_env, trainer_attack, **trainer_args_dict)
    elif trainer_type == TRAINER_ADV_FREE:
        trainer = FreeTrainer(train_env, **trainer_args_dict)
    else:
        raise Exception(f'Unsupported trainer {trainer_type}')

    return trainer


def average_params(model):
    """Average all model parameters accross all CUDA devices."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM)
        param.data /= size


def sum_gradients(model):
    """Sum all model gradients accross all CUDA devices."""
    for param in model.parameters():
        dist.all_reduce(param.grad.data,
                        op=torch.distributed.ReduceOp.SUM)

class TrainEvalEnviron:
    """A thin wrapper class for training and evaluation environment that
       encapsulates a model, an optional adversarial perturbation,
       and an optimizer. It also provides a few convenience functions to simplify training,
       including a multi-device/multi-processing training."""

    def __init__(self,
                 device_name,
                 data_set,
                 dataset_type,
                 batch_size,
                 device_qty, para_type,
                 model, train_model,
                 adv_perturb, train_attack,
                 optimizer,
                 lower_limit, upper_limit,
                 log_writer,
                 use_amp):
        """Constructor.

        :param device_name: the name of the device to use.
        :param data_set:    the current (train or test data set)
        :param dataset_type: a type of the dataset
        :param batch_size:  the size of the batch
        :param device_qty:  total # of CUDA devices
        :param para_type:   parallelization approach
        :param model:       model
        :param train_model: train the model if and only if this flag is true
        :param adv_perturb: a pre-trained adversarial perturbation (or None)
        :param train_attack: train attack if and only if this flag is true
        :param lower_limit: the lower limit for clamping data (not perturbations)
        :param upper_limit: the upper limit for clamping data (not perturbations)
        :param optimizer:   optimizer
        :param log_writer:  Tensorboard writer
        :param use_amp:     if True, we use automatic mixed precision
        """

        self.data_set = data_set
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.device_qty = device_qty
        self.para_type = para_type
        self.use_amp = use_amp
        self.model = model
        assert model is not None, "Bug: model should always be specified/loaded!"
        self.train_model = train_model
        self.train_attack = train_attack
        self.adv_perturb = adv_perturb
        self.optimizer = optimizer
        self.log_writer = log_writer
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.to(device_name)
        if use_amp:
            from apex import amp

    def is_matr(self):
        """

        :return: True if the data is 2dim and False otherwise (for 1dim data)
        """
        return self.dataset_type in [DATASET_TYPE_IMAGE, DATASET_TYPE_VIDEO_FIX_SIZE]

    def to(self, device_name):
        """Just move the mode and lower/upper limit to the corresponding device"""
        self.device_name = device_name
        self.model.to(device_name)
        if self.adv_perturb is not None:
            self.adv_perturb = self.adv_perturb.to(device_name)
        self.lower_limit = self.lower_limit.to(device_name)
        self.upper_limit = self.upper_limit.to(device_name)

    def clamp(self, X):
        return apply_func(clamp, X, self.lower_limit, self.upper_limit)

    def optim_zero_grad(self):
        """Zero gradients unless model updates are disabled."""
        if self.train_model:
            self.optimizer.zero_grad()

    def optim_step(self):
        """Run an optimization step: if the model update flag is not set,
           the function won't do anything.
        """
        if self.train_model:
            self.optimizer.step()
            # if model_validity(self.model, grad_mode=True):
            #     self.optimizer.step()

    def set_model_reqgrads(self, flag):
        """Enables/disables gradient computation for model parameters:
           Importantly if the inner model is frozen, we set require_grad
           for its parameters to False."""

        set_reqgrads(self.model, flag)

        freeze_inner = getattr(self.model.orig_model, FREEZE_INNER_MODEL_ATTR, None)

        if freeze_inner:
            assert isinstance(self.model.orig_model, RobustifyWrapper), "This attack works only with RobustifyWrapper!"
            set_reqgrads(self.model.orig_model.inner_model, False)
            # Don't set eval on the whole model, there are models that check if they are
            # in the training or testing modes and change their behavior!
            #self.model.orig_model.inner_model.eval()
            for module in self.model.orig_model.inner_model.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()


    def train(self, compute_grad=True):
        """Enable training mode. Enable gradient computation, unless compute_grad: is false

        :param  compute_grad:  compute gradients if and only if this flag is True.
        """
        self.model.train()
        self.set_model_reqgrads(compute_grad)
        
    def eval(self):
        """Disable training mode/enable eval mode and disable gradient computation.
           Also disable gradients for the adversarial perturbation tensor (if applicable)."""
        self.model.eval()
        self.set_model_reqgrads(False)
        if self.adv_perturb is not None:
            self.adv_perturb.requires_grad = False

    def comp_loss(self, X, y, reduce_by_sum=True):
        """obtain model outputs,
           compute the value of the loss,
           and the gradients.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :reduce_by_sum carry out loss reduction if True
        :return total loss value, # of correct predictions
        """
        outputs_all = self.model(X, y)
        
        outputs = outputs_all[PREDICTION_OUTPUT]
        loss_value = outputs_all[LOSS_OUTPUT]
        
        assert isinstance(loss_value, torch.Tensor), "Loss value must be a tensor"

        if reduce_by_sum:
            loss_value = torch.sum(loss_value)

        return outputs, loss_value

    def comp_loss_and_backprop(self, X, y):
        """obtain model outputs,
           compute the value of the loss,
           and the gradients. Finally, run a backprop step.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return total loss value, # of correct predictions
        """
        outputs, loss_value = self.comp_loss(X, y)
        self.loss_backward(loss_value)
        
        return outputs, loss_value

    def clamp_comp_loss(self, X, y):
        """Clamp X, obtain model outputs,
           compute the value of the loss,
           and the gradients.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return total loss value, # of correct predictions
        """
        return self.comp_loss(self.clamp(X), y)


    def clamp_comp_loss_and_backprop(self, X, y):
        """Clamp X, obtain model outputs,
           compute the value of the loss,
           and the gradients. Finally, run a backprop step.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return total loss value, # of correct predictions
        """
        return self.comp_loss_and_backprop(self.clamp(X), y)

    def loss_backward(self, loss_val):
        """Do a back-propagation step. This wrapper enabling transparent use of Apex AMP,
        but only when AMP is enabled and installed."""
        # backprop even if we don't train a model or an attack, b/c we need
        # gradients for evaluation as well.
        if self.use_amp:
            #from apex import amp
            with amp.scale_loss(loss_val, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_val.backward()


class BaseTrainer:
    """
    Base class for a training procedure.
    """
    def __init__(self, train_env):
        """Base constructor.

        :param   train_env: a training environment object
        :type    train_env: `bcai_art.TrainEvalEnviron`
        """
        self.train_env = train_env

    def before_train(self):
        """An event function called before training begins."""
        pass

    def before_epoch(selfs):
        """An event function called before each epoch"""
        pass

    def after_epoch(self):
        """An event function called after each epoch"""
        pass

    def get_writer(self):
        return self.train_env.log_writer

    def get_adv_perturb(self):
        return self.train_env.adv_perturb


class NormalTrainer(BaseTrainer):
    """
    A standard, non-adversarial, training procedure.
    """

    def __init__(self, train_env, metric=None):
        """Base constructor.

        :param   train_env: a training environment object
        :type    train_env: `bcai_art.TrainEvalEnviron`
        """
        super().__init__(train_env)
        self.train_env = train_env
        self.metric = create_metric(train_env, args_to_paramdict(metric) if metric else None)

    def fit_batch(self, X, y):
        """
        Train the model on a given batch
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return total loss value, # of correct predictions
        """
        if self.train_env.train_model:
            self.train_env.train(compute_grad=True)
            self.train_env.optim_zero_grad()
            outputs, loss_value = self.train_env.clamp_comp_loss_and_backprop(X, y)
            self.train_env.optim_step()
        else:
            outputs, loss_value = self.train_env.clamp_comp_loss(X, y)

        # A special condition for randomized crop models
        if torch.is_tensor(y) and y.shape[0] < outputs.shape[0]:
            repeat = int(outputs.shape[0]/ y.shape[0]) 

            y = y.repeat_interleave(repeat)
            corret_qty = self.metric.update(outputs, y)
            return loss_value.item(), corret_qty/repeat


        return loss_value.item(), self.metric.update(outputs, y)
    
    def after_epoch(self):
        self.metric.finalize()


class AdversarialTrainer(NormalTrainer):
    """
    The base class for an adverserial trainer
    """
    def __init__(self, train_env, attack, metric=None):
        """Constructor.

        :param   train_env: a training environment object
        :type    train_env: `bcai_art.TrainEvalEnviron`
        """
        super().__init__(train_env, metric)
        self.attack = attack

    def before_train(self):
        """An event function called before training begins.
        """
        self.attack.before_train_eval(self.train_env)

    def fit_batch(self, X, y):
        """
        Generate adversarial/perturbed samples and train on them.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return total loss value, # of correct predictions
        """

        # All attacked models need to be in the training mode!
        # Not computing gradients has an efficiency benefit (about 30% in some cases).
        self.train_env.train(compute_grad=False)

        if self.train_env.para_type in [TRAIN_PARALLEL_DATA, TRAIN_PARALLEL_INDEPENDENT] or self.train_env.device_qty <= 1:
            # data-parallel, or single-device training (single-device should include independent parallelization mode)
            perturbations = self.attack.generate(self.train_env, X, y)
            X_perturbed = self.attack.apply(self.train_env, X, perturbations)

            return super().fit_batch(X_perturbed, y)
        else:
            assert self.train_env.para_type == TRAIN_PARALLEL_ATTACK
            # attack-parallel implementation
            batch_size = len(y)
            rank = dist.get_rank()
            assert(batch_size % self.train_env.device_qty == 0)
            subbatch_size = int(batch_size/self.train_env.device_qty)
            start = subbatch_size * rank
            end = start + subbatch_size
            X_subbatch = X[start:end]
            y_subbatch = y[start:end]
            pert_subbatch = self.attack.generate(self.train_env, X_subbatch, y_subbatch)
            x_pert_subbatch = self.attack.apply(self.train_env, X_subbatch, pert_subbatch)

            x_pert_subbatch_list = []
            # It's super-important that each element in the list is a unique tensor!
            for k in range(self.train_env.device_qty):
                x_pert_subbatch_list.append(torch.ones_like(x_pert_subbatch))
            dist.all_gather(x_pert_subbatch_list, x_pert_subbatch)
            self.train_env.train(compute_grad=True)  # unfreezes the model
            loss_tot = 0
            acc_total = 0
            for rank in range(self.train_env.device_qty):
                start = subbatch_size * rank
                end = start + subbatch_size
                loss_value, acc = super().fit_batch(x_pert_subbatch_list[rank], y[start:end])
                loss_tot += loss_value
                acc_total += acc

            return loss_tot, acc_total


class FreeTrainer(NormalTrainer):
    """
    Implementation of the adversarial training for free:

    Shafahi, Ali, et al.
    "Adversarial training for free!." Advances in Neural Information Processing Systems. 2019.
    """
    def __init__(self, train_env,
                 epsilon=0.2, restarts=10, norm=NORM_INF, metric=None):
        """

        :param   train_env: a training environment object
        :type    train_env: `bcai_art.TrainEvalEnviron`

        :param epsilon:   both the attack step size and the point update step
        :param restarts:  a number of restarts
        :param norm:      a norm
        """
        super().__init__(train_env, metric)
        self.restarts = restarts
        self.norm = get_norm_code(norm)
        self.epsilon = epsilon
        self.metric = create_metric(train_env, args_to_paramdict(metric) if metric else None)

    def fit_batch(self, X, y):
        """
        Train a batch of data using the free adversarial training procedure.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return loss value, accuracy
        """
        self.train_env.train()
        delta = torch.zeros_like(X, requires_grad=True)

        for r in range(self.restarts):
            self.train_env.optim_zero_grad()
            outputs, loss_value = self.train_env.clamp_comp_loss_and_backprop(self.train_env.clamp(X + delta), y)
            delta.data = project(delta + self.epsilon * delta.grad.detach().sign(),
                                 self.epsilon, self.norm,
                                 is_matr=self.train_env.is_matr())
            self.train_env.optim_step()
            # The code above zero gradients through the optimizer.
            # However, the optimizer "knows" only about the model
            # parameters and it will not zero data gradients.
            # This is why we do it manually in the following line:
            delta.grad.zero_()

        return loss_value.item(), self.metric.update(outputs.data, y.data)


def save_model(model, dataset_info, inp_shape, epoch, train_env, out_file_name):
    """A convenience wrapper to save the model.

    :param model:           a model object, should be our ModelWrapper.
    :param dataset_info:    a dictionary of dataset properties.
    :param inp_shape:       input data shape.
    :param epoch:           epoch ID.
    :param train_env:       training environment.
    :param out_file_name:   an output file name.
    :return:
    """

    assert hasattr(model, MODEL_ARGS_KEY), \
           f"The {MODEL_ARGS_KEY} attribute is missing, do you use our standard model wrapper object?"

    out_dict = {SAVE_MODEL_EPOCH: epoch,
                SAVE_MODEL_OPTIM: train_env.optimizer,
                SAVE_MODEL_KEY: model.state_dict(),
                INP_SHAPE_KEY: inp_shape,
                MODEL_ARGS_KEY : getattr(model, MODEL_ARGS_KEY),
                DATASET_LOWER_LIMIT_PARAM: train_env.lower_limit.cpu(),
                DATASET_UPPER_LIMIT_PARAM: train_env.upper_limit.cpu(),
                DATASET_MEAN_PARAM: model.mean.cpu(),
                DATASET_STD_PARAM: model.std.cpu()}

    for key in [DATASET_TYPE, DATASET_NUM_CLASSES_PARAM]:
        assert key in dataset_info
        out_dict[key] = dataset_info[key]

    torch.save(out_dict, out_file_name)


def train(data_loader, trainer,
          batch_sync_step,
          num_epochs,
          seed, is_master_proc,
          dataset_info,
          batch_sync_barrier=None, sync_qty_target=None,
          train_fract=None,
          scheduler=None,
          snapshot_dir=None,
          print_train_stat=False):
    """Train a model.

    :param data_loader:    data loader
    :param trainer:        a training object
    :type  trainer:        `bcai_art.BaseTrainer`
    :param batch_sync_step: how frequently we synchronize model params in the case of distributed training
    :param sync_qty_target:     a number of model sync. points to carry out in each process
    :param num_epochs:     a number of epochs
    :param is_master_proc: indicates a "master" process when training is distributed.
    :param dataset_info:   a dictionary of dataset properties.
    :param batch_sync_barrier:  a batch synchronization barrier (or None)
    :param train_fract:    a fraction of training data to use (or None).
    :param seed:           a random seed
    :param scheduler:      an optional optimizer scheduler
    :param snapshot_dir:   an optional directory to save snapshots for models and adversarial perturbations:
                           the function assumes the directory exists.
    :param print_train_stat:  print training statistics even if when the progress bar isn't displayed.
    """

    dataset_len = len(data_loader)
    assert dataset_len is not None

    start_train_time = time()

    train_env = trainer.train_env
    device_name = train_env.device_name

    clear_device_cache(device_name)

    writer = trainer.get_writer()

    model = train_env.model
    optimizer = train_env.optimizer

    torch.manual_seed(seed)
    np.random.seed(seed)

    # For DATA-parallel processing we can sync. gradients after  a few batches are processed.
    # No need to do so for ATTACK-parallel processing.
    sync_model_params = (train_env.para_type == TRAIN_PARALLEL_DATA and
                         train_env.train_model and
                         trainer.train_env.device_qty > 1)

    if sync_model_params:
        assert batch_sync_barrier is not None, \
               "Misssing batch_sync_barrier despite we sync model parameters accross processes"
        assert sync_qty_target is not None, \
                "Misssing sync_qty_target despite we sync model parameters accross processes"

    # Need to extract one batch before training,
    # to get the shape even if the training will have zero batches (when train_fract <= 0)
    X = None
    for X, _ in data_loader:
        break
    assert X is not None, "Dataset is empty or too small (to have enough entries for every training process)"
    inp_shape = tuple(X.shape)

    total_steps = 0
    trainer.before_train()


    for epoch in range(num_epochs):
        trainer.before_epoch()

        running_loss = 0.0
        running_acc = 0.0
        running_time = 0.0
        total = 0
        bqty = 0

        tqdm_desc_template = 'avg. loss: %.4f avg acc: %.4f'

        if is_master_proc:
            pbar = tqdm(data_loader, desc=tqdm_desc_template % (0, 0))
        else:
            pbar = data_loader

        sync_out()

        # This should be reset before every epoch
        sync_qty = 0

        for batch_id, (X,y) in enumerate(pbar):
            # data set length is measured in the number of batches, not individual entries
            if train_fract is not None and batch_id >= int(train_fract * dataset_len):
                print('Training finished, because we processed %f fraction of the data' % train_fract)
                break

            start_time = time()

            X = X.to(device_name)
            y = y.to(device_name)

            loss, acc = trainer.fit_batch(X, y)
            running_loss += loss
            running_acc += acc
            bqty += 1
            total += y.size(0)
            running_time += time() - start_time

            if sync_model_params and ((batch_id + 1) % batch_sync_step == 0):
                if sync_qty < sync_qty_target:
                    try:
                        batch_sync_barrier.wait(BARRIER_WAIT_TIMEOUT)
                    except thread.BrokenBarrierError as e:
                        raise Exception('A model parameter synchronization timeout!')
                    sync_qty += 1
                    average_params(model)

            # Scheduler must make a step in each batch! *AFTER* the optimizer makes an update!
            if scheduler is not None:
                scheduler.step()

            loss_avg = running_loss / total
            acc_avg = running_acc / total
            batch_time_avg = running_time / bqty

            lr = None
            if optimizer is not None:
                # We assume all parameters have the same learning rate
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']

            if writer is not None:
                if lr is not None:
                    writer.add_scalar(f'train/{LR_KEY}', lr, total_steps + bqty)
                writer.add_scalar(f'train/{LOSS_KEY}', loss_avg, total_steps + bqty)
                writer.add_scalar(f'train/{ACCURACY_KEY}', acc_avg, total_steps + bqty)
                writer.add_scalar(f'train/{BATCH_TIME_KEY}', batch_time_avg, total_steps + bqty)

            if is_master_proc or print_train_stat:
                if lr is None:
                    lr_str = ''
                else:
                    lr_str = 'lr: %.6f' % lr

                if is_master_proc:
                    pbar.set_description((lr_str + ' ' + tqdm_desc_template) % (loss_avg, acc_avg))


                if print_train_stat and batch_id % REPORT_QTY == 0:
                    sync_out()
                    print(device_name, lr_str,
                          'epoch: %d\t Avg. Loss: %.4f\t Avg. Accuracy: %.4f\t Avg. Batch Time: %.3f'
                          % (epoch, loss_avg, acc_avg, batch_time_avg))
                    sync_out()

        total_steps += bqty
        
        sync_out()

        if not model_validity(train_env.model, grad_mode=False):
            if snapshot_dir is not None:
                save_model(model, dataset_info, inp_shape, epoch, train_env,
                           os.path.join(snapshot_dir, 'model_snapshot_%i_invalid.pth' % epoch))

            # This must be an exception, otherwise we quit training in one process but other processes,
            # will continue to run.
            raise Exception("Model has a NAN parameter, quiting training process!")

        # In the end of *EACH* epoch average models on all CUDA devices (if needed)
        if sync_model_params:
            # This ensures we go through the barrier and averaging parameters exactly the same number of time in each process
            # sync_qty_target + 1 is to ensure we make at least one more final sync after the end of the epoch
            while sync_qty < sync_qty_target + 1:
                if DEBUG_PRINT:
                    print('Averaging model params')

                try:
                    batch_sync_barrier.wait(BARRIER_WAIT_TIMEOUT)
                except thread.BrokenBarrierError as e:
                    raise Exception('A model parameter synchronization timeout!')
                sync_qty += 1
                average_params(model)

        if is_master_proc:
            print('epoch: %d\t Avg. Loss: %.4f\t Avg. Accuracy: %.4f\t Avg. Batch Time: %.3f'
              % (epoch, running_loss / total, running_acc / total, running_time / bqty))
            print('total train time: %.4f minutes' % ((time() - start_train_time) / 60))

        trainer.after_epoch()

        assert inp_shape is not None, "The training set should not be empty!"
        if is_master_proc:
            if snapshot_dir is not None:
                if train_env.train_model:
                    if writer is not None:
                        writer.flush()

                    save_model(model, dataset_info, inp_shape, epoch, train_env,
                               os.path.join(snapshot_dir, 'model_snapshot_%i.pth' % epoch))

        adv_perturb = train_env.adv_perturb
        if train_env.train_attack:
            if adv_perturb is not None:
                if writer is not None:
                    dtype = dataset_info[DATASET_TYPE]
                    if dtype == DATASET_TYPE_IMAGE:
                        writer.add_image(f'train/{ADV_PERTURB_KEY}', adv_perturb, epoch)
                    elif dtype == DATASET_TYPE_AUDIO_FIX_SIZE:
                        writer.add_audio(f'train/{ADV_PERTURB_KEY}', adv_perturb, epoch)
                    writer.flush()
                if snapshot_dir is not None:
                    torch.save(adv_perturb.cpu(),
                           os.path.join(snapshot_dir, 'adv_perturb_snapshot_%i.pth' % epoch))

        clear_device_cache(device_name)

    # This also sets all require_grad parameters to false
    train_env.eval()





