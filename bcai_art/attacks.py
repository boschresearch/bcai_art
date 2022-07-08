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

import numpy as np
from tqdm import tqdm
from copy import deepcopy

from bcai_art.train import TrainEvalEnviron
from bcai_art.utils_misc import args_to_paramdict, get_norm_code, calc_correct_qty, EMPTY_CLASS_OBJ
from bcai_art.utils_tensor import project, get_start_delta, START_RANDOM, START_ZERO, apply_func, assert_property, \
                                    get_frame_shape, get_batched, DATASET_TYPE_VIDEO_FIX_SIZE, \
                                    get_abs_max_batched, get_max_norm_batched
from bcai_art.utils_patch import *
from bcai_art.conf_attr import ATTACK_EPS_ATTR, ATTACK_NORM_ATTR, ADD_ARG_ATTR
from bcai_art.models_main import RobustifyWrapper, LOSS_OUTPUT, PREDICTION_OUTPUT

ATTACK_TARG_PARAM = 'targeted'

ATTACK_PATCHSCALE_MAX_PARAM = 'scale_max'
ATTACK_PATCHSCALE_MIN_PARAM = 'scale_min'
ATTACK_PATCHROT_PARAM = 'rotation'

ATTACK_DUMMY = 'dummy'
ATTACK_FGSM = 'fgsm'
ATTACK_PGD = 'pgd'
ATTACK_PATCH = 'patch'
ATTACK_UNIVERSAL_PERTURB = 'univ_perturb'
ATTACK_MASK_PGD = 'mask_pgd'
ATTACK_FRAME_SALIENCY = 'frame_saliency'
ATTACK_FRONT_END_REVERSAL_SIMPLE = 'front_rev_simple'

FRAME_SALIENCY_ONE_SHOT = 'one_shot'
FRAME_SALIENCY_ITER = 'iter'
FRAME_SALIENCY_ITER_REFRESH_GRAD = 'iter_rg'


ATTACK_LIST = [ATTACK_FGSM, ATTACK_PGD, ATTACK_PATCH]


def create_attack(attack_name, inner_attack, target, epsilon, norm_name, add_args, silent=False):
    """A function to create an attack class.

    :param target: an attack target class
    :param inner_attack: a base attack that the current attack uses to construct a more complex one
    :param attack_name: a name/code of the attack
    :param epsilon: an attack strength parameter, should be None for patch attack
    :param norm_name: a norm name, l1, l2, linf, should be None for patch attack
    :param add_args: an object with additional arguments

    :return:
    """
    
    add_args_dict = args_to_paramdict(add_args, [ATTACK_EPS_ATTR, ATTACK_NORM_ATTR])
    norm = get_norm_code(norm_name)

    if not silent:
        print(f'Creating attack: {attack_name} target: {target} epsilon: {epsilon} norm: {norm_name} add args:',
              add_args_dict)

    if not attack_name in [ATTACK_PATCH, ATTACK_DUMMY]:
        if epsilon is None:
            raise Exception(f'Attack {attack_name} needs parameter {ATTACK_EPS_ATTR}')
        if norm_name is None:
            raise Exception(f'Attack {attack_name} needs parameter {ATTACK_NORM_ATTR}')

    # Every attack needs to get a target class argument.
    if attack_name == ATTACK_PGD:
        return PGDAttack(target=target, norm=norm, epsilon=epsilon, **add_args_dict)
    elif attack_name == ATTACK_FGSM:
        return FGSMAttack(target=target, norm=norm, epsilon=epsilon, **add_args_dict)
    elif attack_name == ATTACK_PATCH:
        return PatchAttack(target=target, **add_args_dict)
    elif attack_name == ATTACK_UNIVERSAL_PERTURB:
        return UniversalPerturbAttack(target=target, norm=norm, epsilon=epsilon, **add_args_dict)
    elif attack_name == ATTACK_FRAME_SALIENCY:
        return FrameSaliencyAttack(target=target, norm=norm, epsilon=epsilon, **add_args_dict)
    elif attack_name == ATTACK_FRONT_END_REVERSAL_SIMPLE:
        assert inner_attack is not None, f'Attack {ATTACK_FRONT_END_REVERSAL_SIMPLE} requires specifying the "inner" attack'
        return FrontEndReversalSimple(norm=norm, epsilon=epsilon, inner_attack=inner_attack, **add_args_dict)
    elif attack_name == ATTACK_DUMMY:
        return DummyAttack(target=target)
    else:
        raise Exception('Unsupported attack:' + attack_name)


class AttackBase:
    """
    Base class for Attacks. Curently it has
    no functionality, but we may need to add some in the future.
    """

    def __init__(self, target):
        """Base constructor. Target is a class ID for a targeted attack. To
        make use of the target in a seamless fashion, all child classes have
        to call the parent functions clamp_comp_loss_and_backprop and loss.

        For untargeted attacks these functions compute the loss with respect to the
        batch true labels. Thus, an attack would maximize the batch overall loss.

        However, for a targeted
        attack, clamp_comp_loss_and_backprop and loss compute the loss for a given target
        and flip its sign. Thus, an attack would minimize the loss with respect
        to a given attack target.

        :param target: an attack target class (or None)
        """
        self.target = target

    def get_target_batched(self, train_env):
        return torch.LongTensor([self.target] * train_env.batch_size).to(train_env.device_name)

    def clamp_comp_loss_and_backprop(self, train_env, X, y_true):

        if self.target is None:
            return train_env.clamp_comp_loss_and_backprop(X, y_true)
        else:
            outputs, loss_value = train_env.clamp_comp_loss_and_backprop(X, self.get_target_batched(train_env))
            return outputs, -loss_value

    def loss(self, train_env, X, y_true, reduce_by_sum=True):

        if self.target is None:
            outputs, loss_value = train_env.comp_loss(X, y_true, reduce_by_sum=reduce_by_sum)
        else:
            outputs, loss_value = train_env.comp_loss(X, self.get_target_batched(train_env), reduce_by_sum)
            loss_value = -loss_value
        
        return loss_value

    def generate(self, train_env, X, y):
        """
        To be implemented by sub classes.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data. Should be non-None even for a targeted attack.
        :type  y: `torch.tensor`

        :return generated data.
        """
        raise NotImplementedError

    def apply(self, train_env, X, perturbations):
        """
        A default method of applying pertubations to input tensor X. It simply
        adds perturbations and restricts the result to be in a domain of interest.
        However, more complicated approaches are possible. See, e.g.,
        a patch attack.
        """
        #TODO for variable length audio:
        # need to first check length of X and pert, then intert pert into X with random start timestamps
        return train_env.clamp(X+perturbations)

    def before_train_eval(self, train_env):
        """An event function called before training or evaluation begins.

        :param train_env: a training environment object
        """
        pass


class DummyAttack(AttackBase):
    """A dummy attack class that just passes through the original training data."""
    def __init__(self, target):
        super().__init__(target)

    def generate(self, train_env, X, y):
        return torch.zeros_like(X)

    def apply(self, train_env, X, perturbations):
        return X


class SimpleModelWrapper(torch.nn.Module):
    """
    A simple model wrapper that will replace the top-level model to generate an attack.
    """
    def __init__(self, model, loss):
        super().__init__()
        assert loss is not None
        self.orig_model = model
        self.loss = loss

    def forward_no_loss_comp(self, x):
        outputs = self.orig_model(x)
        return outputs

    def forward(self, x, y=None, context="normal", add_clean=False):
        assert y is not None
        outputs = self.orig_model(x)
        outputs_loss = self.loss(outputs, y)
        return {PREDICTION_OUTPUT: outputs, LOSS_OUTPUT: outputs_loss}


class FrontEndReversalSimple(AttackBase):
    """
        A simple (works sometimes, but not universally),
        non-gradient based attempt to reverse the front end. Limitations & caveats:
        1. Attacks should be stateless and cheap to create:
            We will not train the attack and/or load its weights.
        2. Multiple restarts only make sense if the inner attack has a random starting point,
           but we currently don't enforce it (it's difficult to do in a unified fashion,
           b/c different attacks may or may not support this option)
        3. No support for a targeted attack.
    """
    def __init__(self, norm, epsilon,
                    inner_attack, auto_scale_add_params,
                    scale_sample_qty=1024,
                    restarts=1,
                    reduce_scale_iter_qty=3,
                    back_scale_step_qty=10,
                    debug=False
                 ):
        """Constructor.

        :param   epsilon:                   a maximum perturbation size (for a given norm)
        :param   norm:                      a norm
        :param   inner_attack:              an attack to use
        :param   auto_scale_add_params:     a list of additional parameters to scale automatically
        :param   scale_sample_qty:          a number of samples to estimate scaling of input (approximate) by a robustifier
        :param   restarts:                  a number of restarts
        :param   start:                     a type of the starting point in each restart.
        :param   reduce_scale_iter_qty:     a number of iterations when we reduce epsilon (and other parameters)
                                            and generate an adversarial example within a smaller ball (but with a
                                            different starting point)
        :param   back_scale_step_qty:       number of (a linear-step) iterations in trying to reverse
                                            an after-the-front-end perturbation (this reversal is naive and
                                            linear).
        """
        self.auto_scale_add_params_list = list(set(auto_scale_add_params))

        assert scale_sample_qty > 0

        self.scale_sample_qty = scale_sample_qty

        self.norm = norm
        self.epsilon = epsilon

        self.back_scale_step_qty = back_scale_step_qty

        self.scale_estim = None

        self.debug = debug

        self.restarts = restarts
        self.reduce_scale_iter_qty = reduce_scale_iter_qty

        self.inner_attack_type = inner_attack.attack_type
        self.inner_attack_epsilon = getattr(inner_attack, ATTACK_EPS_ATTR, None)
        self.inner_attack_norm_name = getattr(inner_attack, ATTACK_NORM_ATTR, None)
        self.inner_attack_add_args = getattr(inner_attack, ADD_ARG_ATTR, EMPTY_CLASS_OBJ)

    def before_train_eval(self, train_env):
        """An event function called before training or evaluation begins.

        :param train_env: a training environment object
        """
        scale_vals = []
        data_set = train_env.data_set
        model = train_env.model
        orig_model = model.orig_model
        assert isinstance(orig_model, RobustifyWrapper), "This attack works only with RobustifyWrapper!"
        dqty = len(data_set)
        dids = torch.LongTensor(np.random.choice(np.arange(dqty),
                                                 size=min(dqty, self.scale_sample_qty),
                                                 replace=False))

        data_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data_set, dids),
                                                    pin_memory=True,
                                                    batch_size=train_env.batch_size,
                                                    shuffle=False)
        with torch.no_grad():
            for X, _ in tqdm(data_loader, 'Estimating front-end distortion'):
                X = X.to(train_env.device_name)
                perturb = self.epsilon * (2 * torch.rand_like(X) - 1)
                X_robust, _ = model.orig_model.robustify(model.normalize(X), y=None)
                X_perturb_robust, _ = model.orig_model.robustify(model.normalize(X + perturb), y=None)
                scale_vals.extend((get_abs_max_batched(X_perturb_robust - X_robust) /
                                   (1e-10 + get_abs_max_batched(perturb))).tolist())

        self.scale_estim = np.mean(scale_vals)
        print('Robustifier scale-change estimate:', self.scale_estim)


    def generate(self, train_env, X, y):
        """Generates and returns adversarial perturbations.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data. Should be non-None even for a targeted attack.
        :type  y: `torch.tensor`
        """
        assert self.scale_estim is not None

        # Because attack happens after the front-end, it must use a different
        # training environment obj (different limits) as well as a different
        # model wrapper that uses only the inner model and neither the robustifier not the normalizer.
        model = train_env.model
        orig_model = model.orig_model
        inner_model = orig_model.inner_model

        assert isinstance(orig_model, RobustifyWrapper), "This attack works only with RobustifyWrapper!"

        # It's quite cheap to create both the model wrapper and the environment object
        attack_model = SimpleModelWrapper(inner_model, model.loss)

        attack_train_env = TrainEvalEnviron(device_name=train_env.device_name,
                                             data_set=train_env.data_set,
                                             dataset_type=train_env.dataset_type,
                                             batch_size=train_env.batch_size,
                                             device_qty=1,
                                             para_type=train_env.para_type,
                                             model=attack_model, train_model=False,
                                             adv_perturb=None, train_attack=False,
                                             optimizer=None,
                                             lower_limit=torch.full_like(train_env.lower_limit, -np.inf),
                                             upper_limit=torch.full_like(train_env.lower_limit, np.inf),
                                             log_writer=None,
                                             use_amp=train_env.use_amp)

        # TODO I use clone(), because for some datasets and models (e.g., UCF101) comp_loss
        #      changes input, which seems to be a bug, but I haven't fixed it yet
        B = X.shape[0]

        X_adv = X.clone()

        with torch.no_grad():
            normal_outputs, _ = train_env.comp_loss(X.clone(), y)

        norm_corr_qty = calc_correct_qty(normal_outputs.data, y.data)
        if self.debug:
            print(f'Clean data, batch size {B} correct: {norm_corr_qty}')

        for restart_id in range(self.restarts):
            X_start = X
            if self.debug:
                print(f'Restart id: {restart_id}')

            for reduce_scale_iter_id in range(self.reduce_scale_iter_qty):
                curr_dev_from_start = torch.max(get_max_norm_batched(X - X_start,
                                                                     self.norm,
                                                                     train_env.is_matr)).item()
                # Given the current deviation from the original point, we compute
                # the maximum radius of the ball R that would guarantee that attack
                # with the eps = R will stay within the limits for the original attack epsilon.
                curr_epsilon = self.epsilon - curr_dev_from_start
                assert curr_epsilon > 0

                if self.debug:
                    print(f'restart {restart_id}, current epsilon: {curr_epsilon}')

                eps_reduct = curr_epsilon / self.epsilon
                # Full attack scale should account for changing the scale
                # after passing data through the robustifier
                curr_attack_scale = eps_reduct * self.scale_estim

                # Scale the epsilon and other parameters as necessary
                attack_eps_scaled = curr_attack_scale * self.inner_attack_epsilon

                inner_attack_scaled_args = deepcopy(self.inner_attack_add_args)

                for par_name in self.auto_scale_add_params_list:
                    par_val = getattr(self.inner_attack_add_args, par_name, None)
                    if par_val is not None:
                        setattr(inner_attack_scaled_args, par_name, curr_attack_scale * par_val)

                # It's quite cheap to create a simple-attack object
                inner_attack = create_attack(self.inner_attack_type,
                                                  inner_attack=None,
                                                  target=None,
                                                  epsilon=attack_eps_scaled,
                                                  norm_name=self.inner_attack_norm_name,
                                                  add_args=inner_attack_scaled_args,
                                                  silent=not self.debug)

                # Generating perturbations after the front-end
                X_robust, _ = model.orig_model.robustify(model.normalize(X_start), y=None)
                perturb_after_robustifier = inner_attack.generate(attack_train_env, X_robust, y)
                adv_outputs = inner_model(X_robust + perturb_after_robustifier)

                if self.debug:
                    print(f'Adv. data (after frontend), restart {restart_id}, batch size {B} correct: ' +
                          str(calc_correct_qty(adv_outputs.data, y.data)))

                # This scaling is approximate, because the robustifier is typically non-linear
                perturb_back_scaled = perturb_after_robustifier / self.scale_estim

                for backscale_iter_id in range(1, self.back_scale_step_qty + 1):
                    alpha = float(backscale_iter_id) / self.back_scale_step_qty
                    X_pert = X_start + alpha * perturb_back_scaled
                    # This needs to be projected, b/c there's no guarantee it's within the specified ball
                    X_adv_rev = X_start + project(X_pert - X_start,
                                            # Note that we project to the original ball, regardless the scale of attack
                                            self.epsilon, self.norm,
                                            is_matr=train_env.is_matr())

                    with torch.no_grad():
                        adv_outputs, _ = train_env.comp_loss(X_adv_rev, y)
                        predicted_adv = torch.argmax(adv_outputs.data, 1)
                        wrong_indx = predicted_adv != y
                        X_adv[wrong_indx] = X_adv_rev[wrong_indx]

                        if self.debug:
                            print(f'Adv. data (full model), restart {restart_id}, epsilon {curr_epsilon} alpha {alpha} batch size {B}, correct: ' +
                                  str(calc_correct_qty(adv_outputs.data, y.data)))

                if self.debug:
                    with torch.no_grad():
                        adv_outputs, _ = train_env.comp_loss(X_adv, y)

                    print(f'Adv. data (full model), restart {restart_id}, epsilon {curr_epsilon} batch size {B}, correct: ' +
                          str(calc_correct_qty(adv_outputs.data, y.data)))

                # Finally we need to compute a new starting point that is closer to the ball boundaries
                X_pert = X_start + 0.5 * perturb_back_scaled
                # This needs to be projected, b/c there's no guarantee it's within the specified ball
                X_start = X_start + project(X_pert - X_start,
                                      # Note that we project to the original ball, regardless the scale of attack
                                        self.epsilon, self.norm,
                                        is_matr=train_env.is_matr())

        # Project the perturbation just in case
        return project(X_adv - X, self.epsilon, self.norm, is_matr=train_env.is_matr())


class FrameSaliencyAttack(AttackBase):
    """
    A frame-saliency attack (only for videos). https://arxiv.org/abs/1811.11875
    describes only a single-step L-INF attack, which is basically an FGSM attack.
    We support frame-specific PGD-like attacks as well as other norms.
    """
    def __init__(self,
                 attack_subtype,
                 target,
                 epsilon,
                 norm=np.inf,
                 alpha=0.01, num_iters=5, restarts=1,
                 start=START_RANDOM
                 ):
        """Constructor.

        :param attack_subtype:  attack type (one shot, iterative, iterative + refresh gradient)
        :param alpha: PGD step size (can be seen as a learning rate)
        :param num_iters: a number of iterations
        :param restarts: a number of restarts
        :param target: an attack target class (or None)
        :param epsilon:  a maximum perturbation size (for a given norm)
        :param norm: a norm
        """
        super().__init__(target)

        self.alpha = alpha
        self.num_iters = num_iters
        self.restarts = restarts
        self.start = start
        self.epsilon = epsilon
        self.norm = norm
        self.attack_subtype = attack_subtype

    def compute_grads(self, train_env, X, y):
        """A helper function to compute (detached) gradients

        :param train_env: a training environment object
        :param X:   data
        :param y:   labels
        :return: a triple: loss value, gradients, the averaged mean value of abs. grad values.

        """
        delta = torch.zeros_like(X)
        delta.requires_grad = True

        _, loss_val = self.clamp_comp_loss_and_backprop(train_env, X + delta, y)
        grads = delta.grad.detach()
        # Although the paper isn't very clear about summing over channels
        # it seems a very reasonable thing to
        grads_abs_mean = torch.mean(grads.abs(), dim=(-3, -2, -1))

        return loss_val, grads, grads_abs_mean

    def generate(self, train_env, X, y):
        """Generates and returns adversarial perturbations.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data. Should be non-None even for a targeted attack.
        :type  y: `torch.tensor`
        """

        assert len(X.shape) == 5, "Video is expected to have shape size 5, but the actual size is: " + str(len(X.shape))
        bzs, frame_qty, _, _, _ = X.shape

        # These gradients are detached
        _, curr_grads, curr_grads_abs_avg = self.compute_grads(train_env, X, y)

        assert curr_grads_abs_avg.shape == (bzs, frame_qty)

        if self.attack_subtype == FRAME_SALIENCY_ONE_SHOT:
            # This is basically FGSM and here we use only epsilon, no alpha!
            return project(self.epsilon * curr_grads.sign(),
                            self.epsilon, self.norm,
                            is_matr=train_env.is_matr())

        # We use it to prevent selection of the same frame
        used_frame_mask = torch.full( (bzs, frame_qty), np.inf)

        perturb = torch.zeros_like(X)

        # It will loop until we successfully attack all the frames
        for frame_id in range(frame_qty):
            # Although it's not impossible to do everything in the GPU memory, but it's tricky
            # and probably not worth the trouble, b/c we move only very small tensors (size order of bsz x frame_qty )
            # from/to GPU to/from CPU. This should be fast compared to computation of gradients and
            # evaluating the model outcomes.

            # Masking sets gradients of already used frames to -inf
            curr_grads_abs_avg_masked = torch.min(curr_grads_abs_avg, used_frame_mask.to(train_env.device_name))
            # Frames "masked" by minus infinity will be in the tail
            grad_sort_obj = torch.sort(curr_grads_abs_avg_masked, descending=True)

            best_grad_indx = grad_sort_obj.indices[:, 0].cpu()
            best_grad_values = grad_sort_obj.values[:, 0].cpu().numpy()
            # no -infinity values should be in the first position!
            assert sum(np.isinf(best_grad_values)) == 0

            salient_grad_mask = torch.zeros(bzs, frame_qty)
            for bid in range(bzs):
                # Setting the mask value
                salient_grad_mask[bid, best_grad_indx[bid]] = 1
                # Make sure we mask out the used frame to never use it again
                used_frame_mask[bid, best_grad_indx[bid]] = -np.inf

            # A perturbation will be updated using only salient frames
            curr_frame_mask = salient_grad_mask.to(train_env.device_name).view(bzs, frame_qty, 1, 1, 1).to(train_env.device_name)

            # An FGSM style step is only one line:
            #perturb += self.alpha * (curr_grads.sign() * curr_frame_mask)

            max_loss = torch.zeros(y.shape[0]).to(y.device)
            max_delta = apply_func(torch.zeros_like, X)

            for iter_id in range(self.restarts):
                delta = apply_func(get_start_delta, X,
                                   self.start, self.epsilon, self.norm,
                                   is_matr=train_env.is_matr(),
                                   requires_grad=False) * curr_frame_mask

                for step_id in range(self.num_iters):
                    # If we start from zero and the attack refreshes gradients at each step,
                    # there's no need to recompute gradients in the first iteration.
                    # This can save a lot of compute for FGSM-like training
                    if step_id == 0 and \
                            self.start == START_ZERO and \
                            self.attack_subtype == FRAME_SALIENCY_ITER_REFRESH_GRAD:
                        # as mentioned aboved curr_grads are detached
                        delta_grad = curr_grads
                    else:
                        delta.requires_grad = True
                        self.clamp_comp_loss_and_backprop(train_env, X + perturb + delta, y)
                        delta_grad = delta.grad.detach()
                        delta = delta.detach()

                    # So far, perturbations are applied only to other frames, so the projection
                    # needs to be done only for delta
                    delta = project(delta + self.alpha * (delta_grad.sign() * curr_frame_mask),
                                    self.epsilon, self.norm, is_matr=train_env.is_matr())

                all_loss = self.loss(train_env,
                                     X + perturb + delta, y,
                                     reduce_by_sum=False)
                # Select maximum-loss perturbations
                max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
                max_loss = torch.max(max_loss, all_loss)

            perturb += max_delta
            perturb = project(perturb, self.epsilon, self.norm, is_matr=train_env.is_matr())

            if frame_id + 1 < frame_qty:
                X_perturb = train_env.clamp(X + perturb)
                # Before doing extra work let's check if any frame is classified correctly
                with torch.no_grad():
                    outputs = train_env.model.forward_no_loss_comp(X_perturb)
                    corr_qty = calc_correct_qty(outputs, y)
                    if corr_qty == 0:
                        break

                if self.attack_subtype == FRAME_SALIENCY_ITER_REFRESH_GRAD:
                    # Refresh gradients, but use perturb frames: not the original ones!
                    _, curr_grads, curr_grads_abs_avg = self.compute_grads(train_env, X_perturb, y)

        return perturb


class FGSMAttack(AttackBase):
    """
    FGSM Attack.

    | Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, target,
                 epsilon=0.1, norm=np.inf,
                 start=START_RANDOM):
        """Constructor.

        :param target: an attack target class (or None)
        :param epsilon:  a maximum perturbation size (for a given norm)
        :param norm: a norm
        :param start: a type of the starting point
        """
        super().__init__(target)

        self.epsilon = epsilon
        self.norm = norm
        self.start = start

    def generate(self, train_env, X, y):
        """Generates and returns adversarial perturbations.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data. Should be non-None even for a targeted attack.
        :type  y: `torch.tensor`

        :return generated perturbations.
        """
        delta = apply_func(get_start_delta, X, self.start, self.epsilon, self.norm,
                           is_matr=train_env.is_matr(),
                           requires_grad=False)

        delta.requires_grad = True
        self.clamp_comp_loss_and_backprop(train_env, X + delta, y)

        return project(self.epsilon * delta.grad.detach().sign(),
                       self.epsilon, self.norm,
                       is_matr=train_env.is_matr())


class PGDAttack(AttackBase):
    """
    A projected (multi-step) gradient descent attack.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    def __init__(self, target,
                 epsilon=0.1,
                 norm=np.inf,
                 alpha=0.01, num_iters=5, restarts=1,
                 start=START_RANDOM,
                 patch_width=0,
                 debug=False):
        """Constructor.

        :param target: an attack target class (or None)
        :param epsilon:  a maximum perturbation size (for a given norm)
        :param norm: a norm
        :param alpha: PGD step size (can be seen as a learning rate)
        :param num_iters: a number of PGD steps
        :param restarts: a number of PGD restarts
        :param start: a type of the starting point
        :param patch_width: a width of the patch for the masked PGD. if specified,
                            a PGD attack is applied only to pixels in a random square patch
                            whose side is specified by this parameter.
        """
        super().__init__(target)

        self.epsilon = epsilon
        self.norm = norm
        self.alpha = alpha
        self.num_iters = num_iters
        self.start = start
        # assert number of restarts is 1 in case of starting from the given data points
        if self.start == 'zero':
            if restarts != 1:
                raise Exception('For the non-ramdom (zero) starting point, the # of restarts must be one!')
        self.restarts = restarts
        
        self.masked = not (patch_width == 0)
        if debug:
            print("masked PGD: "+ str(self.masked))
        
        if self.masked:
            self.patch_width = patch_width
        
    def generate(self, train_env, X, y):
        """ 
        Generate and return PGD adversarial perturbations on the samples X, with random restarts.
        Among restarts, we select perturbations with the maximal loss.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data. Should be non-None even for a targeted attack.
        :type  y: `torch.tensor`

        :return generated perturbations.
        """
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = apply_func(torch.zeros_like,X)
        
        if self.masked:
            masks = get_rect_mask_random((self.patch_width,self.patch_width),X)
            masks = masks.to(train_env.device_name)
            masks = masks.unsqueeze(1)


        for i in range(self.restarts):
            delta = apply_func(get_start_delta, X, self.start, self.epsilon, self.norm,
                                    is_matr=train_env.is_matr(),
                                    requires_grad=False)
            if self.masked:
                delta.data = delta.data * masks

            for t in range(self.num_iters):
                delta.requires_grad=True
                self.clamp_comp_loss_and_backprop(train_env, X + delta, y)
                delta_grad = delta.grad.detach()
                delta = delta.detach()
                assert_property(delta, "requires_grad", True)
                assert_property(delta_grad, "requires_grad", True)
                if self.masked:
                    delta_update = self.alpha * (delta_grad.sign()*masks)
                else:
                    delta_update = self.alpha * delta_grad.sign()
                    
                delta = apply_func(project, delta + delta_update,
                                self.epsilon, self.norm,
                                is_matr=train_env.is_matr())

            all_loss = self.loss(train_env,
                                 X+delta, y,
                                 reduce_by_sum=False)

            # Select maximum-loss perturbations
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return max_delta


class UniversalPerturbAttack(AttackBase):
    """
    Universal adversarial perturbation. Based *LOOSELY* on the paper by Moosavi-Dezfooli et al.
    However, the paper is unclear about the details of the  a loss-maximizing step.
    Here, we follow an approach proposed by Anit Kumar Sahu.


    | Paper link: https://arxiv.org/abs/1610.08401
    """
    def __init__(self,
                 target,
                 epsilon, norm,
                 alpha=0.01,
                 start=START_RANDOM):
        """Constructor.

        :param target: an attack target class (or None)
        :param epsilon: a ball radius
        :param norm:    a norm
        :param alpha:   a step size (can be seen as a learning rate)
        :param start:   a type of the starting point
        """
        super().__init__(target)

        self.epsilon = epsilon
        self.norm = norm
        self.alpha = alpha
        self.start = start

    def before_train_eval(self, train_env):
        """An event function called before training or evaluation begins:
           we use it to initialize the patch. The function assumes
           that the data set is not empty.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        """
        #TODO for variable length audio: need an extra input as pert size

        assert train_env.device_name is not None,\
            'Possibly a bug: calling before_train_eval when the device name is not set!'

        data_set = train_env.data_set
        assert len(data_set) > 0, ""

        #TODO for variable length audio: need to use input pert size for X
        X, _ = data_set[0]

        if train_env.adv_perturb is None:
            train_env.adv_perturb = get_start_delta(X,
                                                self.start,
                                                self.epsilon, self.norm,
                                                requires_grad=False,
                                                is_matr=train_env.is_matr()).to(train_env.device_name)

    def generate(self, train_env, X, y):
        """
        Updates and returns adversarial perturbations.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data. Should be non-None even for a targeted attack.
        :type  y: `torch.tensor`
        :return generated patch.
        """
        batch_size = X.size(0)

        curr_perturb = get_batched(train_env.adv_perturb.to(train_env.device_name),
                                   batch_size)

        assert(not curr_perturb.requires_grad)

        if train_env.train_attack:
            curr_perturb.requires_grad = True

            #TODO for variable length audio: (actually apply to all cases) -- use apply function
            self.clamp_comp_loss_and_backprop(train_env, X + curr_perturb, y)

            delta = self.alpha * (torch.sum(curr_perturb.grad.detach(), dim=0, keepdim=False).sign())

            train_env.adv_perturb = project(train_env.adv_perturb + delta,
                                            self.epsilon, self.norm,
                                            is_matr=train_env.is_matr())

        res = get_batched(train_env.adv_perturb, batch_size)
        if train_env.dataset_type == DATASET_TYPE_VIDEO_FIX_SIZE:
            res = res.unsqueeze(dim=1) # adding time dimension

        return res


class PatchAttack(AttackBase):
    """
    Patch Attack.

    | Paper link: https://arxiv.org/abs/1712.09665
    """

    def __init__(self,
                 target,
                 alpha=0.01,
                 mask_kind= MASK_CIRCLE,
                 scale_max=0.25,
                 scale_min=0.1,
                 aspect_ratio= 1,
                 rotation=5.0,
                 start=START_RANDOM):
        """Constructor.

        :param target: an attack target class (or None)
        :param alpha: an attack step size (can be seen as a learning rate)
        :param mask_kind: a type of the mask (circle, rectangle)
        :param scale_min: a minimum scaling factor of the patch
        :param scale_max: a maximum scaling factor of the patch
        :param rotation: maximum rotation (in degrees) of the patch
        :param aspect_ratio: an aspect ratio for rectangular patches
        :param start: a type of the starting point
        """
        super().__init__(target)

        self.alpha = alpha
        self.mask_kind = mask_kind
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.aspect_ratio = aspect_ratio
        self.rotation = rotation
        self.random = (start == START_RANDOM)
        self.patch_width = None

    def generate(self, train_env, X, y):
        """
        Updates and returns adversarial perturbations.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.  Ignored in the case of a targeted attack.
        :type  y: `torch.tensor`
        :return generated patch.
        """
        batch_size = X.size(0)
        self.update_patch(train_env, X, y)

        return get_batched(train_env.adv_perturb, batch_size)

    def before_train_eval(self, train_env):
        """An event function called before training or evaluation begins:
           we use it to initialize the patch. The function assumes
           that the data set is not empty.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        """

        assert train_env.device_name is not None,\
                'Possibly a bug: calling before_train_eval when the device name is not set!'

        data_set = train_env.data_set
        assert len(data_set) > 0, ""

        X, _ = data_set[0]
        X = X.unsqueeze(dim=0)  # fake batch dimension for get_frame_shape

        _, channel, h, w = get_frame_shape(train_env.dataset_type, X)

        assert h is not None
        assert w is not None

        self.patch_width = min(h, w)
        patch_shape = (channel, self.patch_width, self.patch_width)
        self.mask = torch.Tensor(get_mask((channel, self.patch_width, self.patch_width),
                                          kind=self.mask_kind))
        self.mask = self.mask.to(train_env.device_name)

        if not train_env.train_attack:
            if train_env.adv_perturb is None:
                raise Exception('The patch training is disabled: You have to specify a pretrained patch!')
            # We have a pre-train patch: do nothing
            return

        if self.random:
            patch = torch.rand(patch_shape)  # random values in [0,1]

            train_env_upper_limit = train_env.upper_limit.cpu()
            train_env_lower_limit = train_env.lower_limit.cpu()

            # We expect the limits to not include a time and/or batch dimesnion
            # If this doesn't hold, we cannot use '*' below
            assert len(patch.shape) == len(train_env_upper_limit.shape)
            assert len(patch.shape) == len(train_env_lower_limit.shape)

            upper_limit = train_env_upper_limit * torch.ones_like(patch[0])
            lower_limit = train_env_lower_limit * torch.ones_like(patch[0])

            patch = patch * (upper_limit - lower_limit) + lower_limit
        else:
            patch = torch.zeros(patch_shape)

        patch = patch.to(train_env.device_name)
        patch.requires_grad=False
        train_env.adv_perturb = patch

    def update_patch(self, train_env, X, y):
        """
        Update an existing patch. The patch is retrieved from/saved to
        the training environment. It has the shape (channel x patch_w x patch_h).

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        """
        patch = train_env.adv_perturb
        assert patch is not None, "Bug: we should have an initialized patch here!"

        if not train_env.train_attack:
            return

        _, h, w = get_frame_channel_height_width(patch)
        if h != w:
            raise Exception('The last two patch dimensions should be the same: ' +
                            'did you load a pre-rained patch that is not square?')

        # 1. apply the patch (enable gradient computations and zero grads)
        # 2. compute gradients with respect to the data:
        #    because the patch transform is differentiable we backpropagate all the way back to the patch
        # 3. update the patch and compute gradients
        # 4. update the pach using the gradients
        assert not patch.requires_grad
        patch.requires_grad = True

        patch_batched = get_batched(patch, train_env.batch_size)

        X_patched = self.apply(train_env=train_env, X=X,
                               perturbations=patch_batched)
        # compute gradients with respect to input
        self.clamp_comp_loss_and_backprop(train_env, X_patched, y)
        patch_grad = patch.grad.detach()
        delta = self.alpha * patch_grad.sign()
        patch = patch.detach()
        train_env.adv_perturb = train_env.clamp(patch + delta)

    def apply(self, train_env, X, perturbations):
        # Here height and width returned by get_frame_shape
        # can be None if X is a TensorList
        batch_size, channel, _, _ = get_frame_shape(train_env.dataset_type, X)

        assert batch_size is not None

        batched_mask = get_batched(self.mask,
                                   batch_size).to(train_env.device_name)
        batched_patch = perturbations

        # apply random affine transformations to each member of the batch
        masks_tform, patches_tform, _ = random_transform(batched_mask, batched_patch,
                                                         self.scale_min, self.scale_max,
                                                         self.rotation, self.aspect_ratio,
                                                         X)

        # This is a questionable hack, but some functions, e.g.,
        # random_transform won't work if there's an extra time dimension
        # Furthermore, it makes sense to apply the same patch for frames within a vdeo anyways.
        # Thus, we make the patch have only CxHxW dimensions and add a fake dimensions at this point.
        if train_env.dataset_type == DATASET_TYPE_VIDEO_FIX_SIZE:
            masks_tform = masks_tform.unsqueeze(dim=1)  # adding time dimension
            patches_tform = patches_tform.unsqueeze(dim=1)  # adding time dimension

        # This check makes sense even if X is a TensorList:
        # In this case, the last two shape values are None,
        # but the sizes of the shape values must still match
        assert len(masks_tform.shape) == len(X.shape)
        assert len(patches_tform.shape) == len(X.shape)

        X_patched = (1 - masks_tform) * X + masks_tform * patches_tform
        
        X_patched = apply_func(train_env.clamp, X_patched)

        return X_patched

