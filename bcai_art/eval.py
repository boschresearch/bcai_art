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
from time import time
from tqdm import tqdm

from bcai_art.utils_tensor import clear_device_cache
from bcai_art.utils_misc import calc_correct_qty, sync_out, args_to_paramdict, save_sample_and_pred_info
from bcai_art.utils_misc import BATCH_TIME_KEY, ACCURACY_KEY, LOSS_KEY, ATTACK_SUCCESS_KEY, CERTIFICATION_KEY, CERTIFICATION_ACCURACY_KEY
from bcai_art.utils_patch import derandomized_ablate_batch
from bcai_art.utils_patch_certification import CertificationProb
from bcai_art.models_main import LOSS_OUTPUT, PREDICTION_OUTPUT 
from bcai_art.metrics_main import create_metric

EVALUATOR_NORMAL = 'normal'
EVALUATOR_ATTACK = 'attack'
EVALUATOR_DEFENSE = 'defense'
EVALUATOR_DERANDOMIZED_SMOOTHING = 'derandomized_smoothing'
EVALUATOR_RANDOM_CROPS_CERT = 'random_crops_certification'


def create_evaluator(evaluator_type,
                     eval_train_env,
                     add_args,
                     eval_attack_obj=None):
    """Create an evaluator.

    :param evaluator_type:   evaluator type
    :param eval_train_env:   evaluator training environement
    :param eval_attack_obj:  an optional (but mandatory for some evaluators) attack object
    :param add_args:         additional evaluator arguments (object)
    :return:
    """
    add_args_dict = args_to_paramdict(add_args, [])
    print(add_args_dict)
    if evaluator_type == EVALUATOR_NORMAL:
        evaluator = NormalEvaluator(eval_train_env, **add_args_dict)
    elif evaluator_type == EVALUATOR_ATTACK:
        if eval_attack_obj is None:
            raise Exception('The attack needs to be defined for the evaluator: ' + evaluator_type)
        evaluator = AttackEvaluator(eval_train_env, eval_attack_obj, **add_args_dict)
    elif evaluator_type == EVALUATOR_DEFENSE:
        if eval_attack_obj is None:
            raise Exception('The attack needs to be defined for the evaluator' + evaluator_type)
        evaluator = DefenseEvaluator(eval_train_env, eval_attack_obj,  **add_args_dict)
    elif evaluator_type == EVALUATOR_DERANDOMIZED_SMOOTHING:
        evaluator = DerandomizedSmoothingEvaluator(eval_train_env, eval_attack_obj, **add_args_dict)
    elif evaluator_type == EVALUATOR_RANDOM_CROPS_CERT:
        evaluator = RandomizedCropCertificationEvaluator(eval_train_env, eval_attack_obj, **add_args_dict)
    else: 
        raise Exception(f'Unsupported evaluator {evaluator_type}')
    return evaluator


class BaseEvaluator:
    """
    Base class for an evaluation procedure.
    """
    def __init__(self, train_env, metric=None):
        """Base constructor.

        :param   train_env: a training environment object
        :type    train_env: `bcai_art.TrainEvalEnviron`
        """
        self.train_env = train_env
        self.metric = create_metric(train_env, args_to_paramdict(metric) if metric else None)

    def get_model(self):
        return self.train_env.model
    
    def get_writer(self):
        return self.train_env.log_writer

    def before_eval(self):
        """An event function called before evaluation begins."""
        pass

    def after_eval(self, eval_name, log_writer):
        """An event function called after evaluation end.

        :param log_writer: a tensorboard writer
        :param eval_name:  evaluator name
        """
        pass

    def eval_batch(self, X, y):
        """
        Evaluate a model on a given batch: The metric reduction
        function is assumed to be sum.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return a dictionary of metric values, which should at least
                include the accuracy and the loss. We expect to
                return metric values summed up over all batch points.
        """
        raise NotImplementedError


class NormalEvaluator(BaseEvaluator):
    """
    Base class for a standard, non-adversarial evaluation.
    """

    def __init__(self, train_env, metric=None):
        """
        Constructor.

        :param   train_env: a training environment object
        :type    train_env: `bcai_art.TrainEvalEnviron`
        """
        super().__init__(train_env, metric)
        

    def eval_batch(self, X, y):
        """
        Evaluate a model on a given batch.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return loss value, accuracy
        """
        self.train_env.eval()

        with torch.no_grad():
            outputs, loss_value = self.train_env.comp_loss(X,y)

        return {LOSS_KEY : loss_value.item(),
                ACCURACY_KEY : self.metric.update(outputs, y)}
    
    def after_eval(self, eval_name, log_writer):
        print('Final evaluation results:', self.metric.finalize())


class EvaluatorWithAttackBase(NormalEvaluator):
    """
        The base class for an evaluator that has an attack
        object and needs to save samples of original
        and perturbed objects. It inherits from NormalEvaluator
        to reuse the function eval_batch (in DefenseEvaluator at least)!
    """
    def __init__(self, train_env, attack,
                 sample_qty=None, metric=None):
        """
        Constructor.

        :param   train_env:  a training environment object
        :type    train_env:  `bcai_art.TrainEvalEnviron`
        :param   attack:     an attack class
        :param   sample_qty: a number of samples to make

        """
        super().__init__(train_env, metric)
        self.attack = attack

        self.sample_qty = sample_qty
        self.proc_qty = 0

        self.X_lst = []
        self.y_true_lst = []
        self.X_perturbed_lst = []
        self.y_pred_lst = []

    def before_eval(self):
        """An event function called before evaluation begins."""
        self.attack.before_train_eval(self.train_env)

        if self.sample_qty is not None:
            data_set = self.train_env.data_set
            data_qty = len(data_set)
            self.sample_flags = np.zeros(data_qty, dtype=np.bool)
            sel_idx = np.random.choice(np.arange(data_qty), self.sample_qty, replace=False)
            self.sample_flags[sel_idx] = True
        else:
            self.sample_flags = None


    def sample_batch(self, X, y, X_adv, model_adv_out):
        """Sample adversarial outputs and model predictions.

        :param X:           original data
        :param y:           true values
        :param X_adv:       adversarially modified data
        :param model_adv_out: output of the model on adversarial data
        :return:
        """
        if self.sample_flags is not None:
            _, adv_predicted = torch.max(model_adv_out.data, 1)
            for k in range(len(X)):
                if self.sample_flags[self.proc_qty + k]:
                    self.X_lst.append(X[k])
                    self.y_true_lst.append(y[k])
                    self.X_perturbed_lst.append(X_adv[k])
                    self.y_pred_lst.append(adv_predicted[k])

        self.proc_qty += len(X)

    def after_eval(self, eval_name, log_writer):
        """An event function called after evaluation end.

        :param log_writer: a tensorboard writer.
        :param eval_name:  evaluator name
        """
        if self.sample_qty is not None and log_writer is not None:
            save_sample_and_pred_info(self.train_env.dataset_type,
                                      self.X_lst, self.y_true_lst, self.X_perturbed_lst, self.y_pred_lst,
                                      eval_name, log_writer)
        super().after_eval(eval_name, log_writer)


class AttackEvaluator(EvaluatorWithAttackBase):
    """
    This class evaluates success of a given attack on a model, i.e.,
    we first evaluate the model on original data. Then, we perturbed
    the data and see how much performance deteriorated.
    """
    def __init__(self,
                 train_env, attack,
                 sample_qty=None, metric=None
                 ):
        """
        Constructor.

        :param   train_env:  a training environment object
        :param   attack:     an attack class
        :param   sample_qty: a number of samples to make
        """
        super().__init__(train_env, attack, sample_qty, metric)

    def eval_batch(self, X, y):
        """
        Evaluate an attack on a given model and batch of data and measures what percent of attacks were successful
        Success means the prediction of the model is changed.

        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return loss value, success rate
        """
        self.train_env.eval()
        perturbations = self.attack.generate(self.train_env, X, y)

        with torch.no_grad():
            # The attack is expected to clamp the result, but let's do it again to be on the safe side
            X_adv = self.train_env.clamp(self.attack.apply(self.train_env, X, perturbations))
            normal_outputs, loss_value = self.train_env.comp_loss(X, y)
            adv_outputs, adv_loss_value = self.train_env.comp_loss(X_adv, y)

            self.sample_batch(X, y, X_adv, adv_outputs)

        attack_success = (torch.argmax(normal_outputs, 1) != torch.argmax(adv_outputs, 1)).sum().item()

        return {LOSS_KEY : adv_loss_value,
                ACCURACY_KEY: calc_correct_qty(adv_outputs.data, y.data),
                ATTACK_SUCCESS_KEY : attack_success}


class DerandomizedSmoothingEvaluator(EvaluatorWithAttackBase):
    """
    Random smoothing evaluation.
    Paper link: https://arxiv.org/abs/2002.10733
    """
    def __init__(self, train_env, attack, reuse_noise, block_size_w, block_size_h, treshhold, size_to_certify, metric=None):
        """
        Constructor.

        :param   train_env:  a training environment object
        :type    train_env:  `bcai_art.TrainEvalEnviron`
        :param   attack:     an attack class
        :param   reuse_noise: whether to use the same smoothing for entire batch or not
        :type    reuse_noise: boolean
        :param   block_size_w: ablation block width
        :type    block_size_w: int
        :param   block_size_h: ablation block height
        :type    block_size_h: int
        :param   treshhold: certification treshhold theta
        :type    treshhold: float
        :param   size_to_certify: size of the attack patch to certify
        :type    size_to_certify: int
        """
        super().__init__(train_env, attack)
        self.train_env = train_env
        self.reuse_noise = reuse_noise
        self.size_to_certify = size_to_certify
        self.treshhold = treshhold
        self.block_size_w = block_size_w
        self.block_size_h = block_size_h


    def eval_batch(self, X, y):
        """
        Evaluate an attack on a given model and batch of data and measures what percent of attacks were successful
        Success means the prediction of the model is changed.

        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return loss value, success rate
        """
        self.train_env.eval()
        perturbations = self.attack.generate(self.train_env, X, y)

        with torch.no_grad():
            # The attack is expected to clamp the result, but let's do it again to be on the safe side
            X_adv = self.train_env.clamp(self.attack.apply(self.train_env, X, perturbations))
            counts = torch.zeros(X_adv.size(0), self.train_env.model.num_classes).type(torch.int).to(X_adv.device)
            X_adv = self.train_env.model.normalize(X_adv)
            for xcorner in range(X.shape[2] if X.shape[2] < self.block_size_w else 1):
                for ycorner in range(X.shape[3] if X.shape[3] < self.block_size_h else 1):
                    ex_X = derandomized_ablate_batch(self.train_env.clamp(X_adv), (self.block_size_w, self.block_size_h), pos= (xcorner, ycorner), reuse_noise = self.reuse_noise)
                    logits = self.train_env.model.orig_forward(ex_X)
                    softmx = torch.nn.functional.softmax(logits, dim=1)
                    counts += (softmx >= self.treshhold).type(torch.int).to(X_adv.device)
            counts_np = counts.cpu().numpy()
            idxsort = np.argsort(-counts_np, axis = 1, kind = 'stable')
            valsort = -np.sort(-counts_np, axis = 1, kind = 'stable')
            val = valsort[:, 0] 
            idx = idxsort[:, 0]
            valsecond =  valsort[:, 1]
            idxsecond =  idxsort[:, 1] 
            affected_blocks= (self.size_to_certify + self.block_size_w -1)*(self.size_to_certify + self.block_size_h - 1)
            cert = torch.tensor(((val - valsecond > 2 * affected_blocks) | ((val - valsecond == 2 * affected_blocks) & (idx < idxsecond)))).cuda()
            acc = (torch.tensor(idx).cuda() == y).sum().item()
            cert_acc = (torch.tensor(idx).cuda()[cert] == y[cert]).sum().item()
            return {LOSS_KEY : self.train_env.loss(logits, y.data),
                    ACCURACY_KEY: acc,
                    ATTACK_SUCCESS_KEY : 0, "certified_acc": cert_acc}


class DefenseEvaluator(EvaluatorWithAttackBase):
    """
    This class evaluates a "defended" model on a given attack, i.e.,
    we carry out an attack and evaluate the model performance on
    perturbed data.
    """
    def __init__(self, train_env, attack,
                 sample_qty=None, metric=None):
        """
        Constructor.

        :param   train_env:  a training environment object
        :type    train_env:  `bcai_art.TrainEvalEnviron`
        :param   attack:     an attack class
        :param   sample_qty: a number of samples to make
        """
        super().__init__(train_env, attack, sample_qty, metric)

    def eval_batch(self, X, y):
        """
        Run perturbed samples on a defended model.

        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return  loss value, accuracy
        """
        self.train_env.eval()
        perturbations = self.attack.generate(self.train_env, X, y)

        with torch.no_grad():
            # The attack is expected to clamp the result, but let's do it again to be on the safe side
            X_adv = self.train_env.clamp(self.attack.apply(self.train_env, X, perturbations))

            if self.sample_qty is not None:
                output_dict = self.train_env.model(X, y)
                adv_outputs, loss_value = output_dict[PREDICTION_OUTPUT], output_dict[LOSS_OUTPUT]
                
                self.sample_batch(X, y, X_adv, adv_outputs)

        return super().eval_batch(X_adv, y)


class RandomizedCropCertificationEvaluator(BaseEvaluator):
    """
    Given a crop-based classifier and its corresponding crop size, patch_scale (as in percentage of image area), threshold for likelihood of certification, and if the patch is placed at random location or the worst location, return the percentage of evaluation data that can be certifiably robust under randomized crop
    """
    def __init__(self, train_env, attack, certification_threshold,patch_scale, random_location=True, metric=None):
        """
        Constructor.

        :param   train_env: a training environment object
        :type    train_env: `bcai_art.TrainEvalEnviron`
        """
        super().__init__(train_env)
        
        assert patch_scale >= 0.0 and patch_scale < 1.0
        self.cert_prob_calculater = CertificationProb(patch_scale, train_env.model.orig_model.crops_width, train_env.model.orig_model.crops_height, num_crops=self.train_env.model.orig_model.num_crops, random_location=random_location)
        
        assert certification_threshold > 0.0 and certification_threshold < 1.0
        self.cert_prob_threshold = certification_threshold
        
        
        
        self.attack = attack

    def eval_batch(self, X, y):
        """
        Evaluate a model on a given batch.

        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data.
        :type  y: `torch.tensor`
        :return loss value, accuracy
        """
        self.train_env.eval()
        
        if self.attack is not None:
            perturbations = self.attack.generate(self.train_env, X, y)
            X = self.train_env.clamp(self.attack.apply(self.train_env, X, perturbations))

        with torch.no_grad():
            output_dict = self.train_env.model(X, y)
        outputs, loss_value = output_dict[PREDICTION_OUTPUT], output_dict[LOSS_OUTPUT]
        cert_dict = self.certification(X, y, outputs)
        cert_dict.update({LOSS_KEY : loss_value.item(),
                ACCURACY_KEY : calc_correct_qty(outputs.data, y.data)})
        
        return cert_dict
    
    def certification(self, X, y, outputs):
        
        x_dim =  X.shape
        if len(x_dim) == 4:
            batch_size, channel, img_width, img_height = X.shape
        elif len(x_dim) == 5:
            batch_size, frame_len, channel, img_width, img_height = X.shape
        
        num_cert = 0
        num_cert_accurate = 0
        self.cert_prob_calculater.set_img_sizes(img_width, img_height)
        _, predicted = torch.max(outputs.data, 1)
        
        class_diff = 0
        for ii in range(batch_size):
            cert_prob, class_diff_s = self.cert_prob_calculater.get_certification_prob(outputs.data[ii])
            
            if predicted[ii] == y[ii]:
                class_diff += class_diff_s
            
            if cert_prob < self.cert_prob_threshold:
                continue
                
            num_cert += 1
            
            #check if it is accurate
            if predicted[ii] == y[ii]:
                num_cert_accurate += 1

        return {CERTIFICATION_KEY:num_cert, CERTIFICATION_ACCURACY_KEY:num_cert_accurate, "class_diff":class_diff}
        

def evaluate(data_loader,
             evaluator_name,
             evaluator):
    """Evaluate model using a given, e.g., adversarial, evaluation procedure.

    :param data_loader:     data loader
    :param evaluator_name:  an evaluator name. it needs to be unique so
                            that tensorboard makes unique entries for each evaluator.
    :param evaluator:       an evaluator object
    :return summary statistics
    """
    run_metr_dict = {}
    running_time = 0.0
    total = 0
    bqty = 0

    evaluator.before_eval()
    device_name = evaluator.train_env.device_name

    clear_device_cache(device_name)


    sync_out()

    log_writer = evaluator.get_writer()

    with tqdm(data_loader) as pbar:
        for i, (X, y) in enumerate(pbar):
            start_time = time()
            X = X.to(device_name)
            y = y.to(device_name)
            
            res_dict = evaluator.eval_batch(X, y)
            assert ACCURACY_KEY in res_dict, "Bug: evaluator should always compute accuracy"
            assert LOSS_KEY in res_dict, "Bug: evaluator should always compute loss"

            total += y.size(0)
            bqty += 1

            running_time += time() - start_time
            for k, v in res_dict.items():
                if not k in run_metr_dict:
                    run_metr_dict[k] = 0
                run_metr_dict[k] += v
                if log_writer is not None:
                    log_writer.add_scalar(f'test/{evaluator_name}/{k}', run_metr_dict[k] / total, bqty)

            if log_writer is not None:
                log_writer.add_scalar(f'test/{evaluator_name}/{BATCH_TIME_KEY}', running_time/bqty, bqty)
            pbar.set_description('avg. loss: %.4f avg acc: %.4f' %
                                 (run_metr_dict[LOSS_KEY] / total, run_metr_dict[ACCURACY_KEY] / total))


    sync_out()

    evaluator.after_eval(evaluator_name, log_writer)

    avg_batch_time = running_time / bqty
    out_res = {BATCH_TIME_KEY : avg_batch_time}

    print('Average batch time: %.3f' % (avg_batch_time))
    # All metric values except the batch size are supposed to be averaged (by the number of data points),
    # i.e., batch-level reduction function is sum.

    for k, v in run_metr_dict.items():
        avg_val = v/total
        if isinstance(avg_val, torch.Tensor):
            avg_val = avg_val.item()
        print('%s %.4f' % (k, avg_val))
        out_res[k] = avg_val
        
        if k == "class_diff":
            torch.save(v, "class_diff_epoch100.pth")
    clear_device_cache(device_name)


    return out_res


