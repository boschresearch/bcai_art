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

from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process
from multiprocessing import Barrier

import shutil
from time import sleep

from bcai_art.config import *
from bcai_art.train import *
from bcai_art.eval import *
from bcai_art.attacks import *
from bcai_art.models_main import *
from bcai_art.datasets_main import *
from bcai_art.utils_tensor import *
from bcai_art.utils_misc import *
from bcai_art.conf_attr import *
from bcai_art.loss import *
from bcai_art.utils_dataset import DataPartitioner

OVERRIDES_JSON = '(overrides JSON config)'
CONFIG_PARAM='config'

EPOCH_QTY_PARAM="epoch_qty"


INSTANCE_SUB_DIR_TEMPLATE = 'run%d'

PYTORCH_DISTR_BACKEND='gloo'
DEFAULT_CUDA_ID=0

LOSS_CLASS = nn.CrossEntropyLoss

START_DELAY=0.25


def use_distr_train(device_qty, para_type):
    """
        :return true if we need initialize and destroy Pytorch multi-processing environment object.
    """
    return para_type != TRAIN_PARALLEL_INDEPENDENT and device_qty > 1


def run_train_process(rank, is_master_proc,
                      device_name, device_qty,
                      dataset_info,
                      para_type,
                      model, train_model, adv_perturb, train_attack,
                      optim_type,
                      batch_sync_step, batch_sync_barrier, sync_qty_target,
                      init_lr, optim_add_args,
                      scheduler_type, scheduler_add_args,
                      use_amp,
                      trainer_type, trainer_args, trainer_attack_obj,
                      lower_limit, upper_limit,
                      train_set,
                      adj_batch_size,
                      epoch_qty,
                      snapshot_dir,
                      log_dir,
                      num_data_workers,
                      random_seed,
                      train_fract,
                      port,
                      dist_backend):
    """The main training procedure that runs in its own process.

    :param rank:                a process rank (zero for the master process)
    :param is_master_proc:      True for the "master" process.
    :param device_name:         a device name: it is ignored when the process runs on multiple GPUs
    :param device_qty:          a number of devices
    :param dataset_info:        a dictionary of dataset properties.
    :param para_type:           an approach to parallelization
    :param model:               a model object
    :param train_model:         train the model if and only if this flag is True
    :param adv_perturb:         an optional perturbation object
    :param train_attack:        train an attack if and only if this flag is True
    :param optim_type:          a type of the optimizer
    :param batch_sync_step:     how frequently we synchronize model params in the case of distributed training
    :param batch_sync_barrier:  a batch synchronization barrier
    :param sync_qty_target:     a number of model sync. points to carry out in each process
    :param init_lr:             initial learning rate
    :param optim_add_args:      additional optimizer arguments
    :param scheduler_type:      a type of the scheduler
    :param scheduler_add_args:  scheduler additional arguments
    :param use_amp:             if True, we use the mixed precision.
    :param trainer_type:        a type of the trainer, i.e., a training procedure type
    :param trainer_args:        trainer arguments
    :param trainer_attack_obj:  an optional attack object used by the trainer
    :param lower_limit:         data clipping lower bound
    :param upper_limit:         data clippiog upper bound
    :param train_set:           a training set
    :param adj_batch_size:      an (adjusted) batch size
    :param epoch_qty:           a number of epochs
    :param snapshot_dir:        a directory to store model snapshots
    :param log_dir:             Tensorboard log directory
    :param num_data_workers:    number of workers in a data loader
    :param random_seed:         a process-specific random seed
    :param train_fract:         a fraction of training data to use (or None).
    :param port:                a master port for distributed Pytorch computation.
    :param dist_backend:        a type of the distributed-processing backend
    """
    # Initialize the distributed environment if needed
    if use_distr_train(device_qty, para_type):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(port)
        if dist_backend not in ['gloo', 'nccl', 'mpi']:
            raise Exception('Unknown distributed processing backend')
        dist.init_process_group(dist_backend, rank=rank, world_size=device_qty)

    if device_qty > 1:
        device_name = f'cuda:{rank}'

    # We want each process to have its own random seed
    # (see also the DataLoader comment below)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print(f'Process rank {rank} adjusted batch size: {adj_batch_size} training set size: %d' % (len(train_set)),
          f'device: {device_name} random seed {random_seed}')

    # The otpimizer and the training environment need to
    # be initialized in each process separately
    if train_model:
        optimizer = create_optimizer(optim_type, model, init_lr, optim_add_args)

        if scheduler_type is not None:
            lr_steps = compute_lr_steps(epoch_qty=epoch_qty,
                                        train_set=train_set,
                                        batch_size=adj_batch_size)
            if lr_steps > 0:
                scheduler = create_scheduler(optimizer, scheduler_type, lr_steps, scheduler_add_args)
            else:
                print('Not creating the scheduler, because the number of steps is not positive!')
                scheduler = None
        else:
            scheduler = None
    else:
        # If we don't train the model, we don't need an optimizer
        optimizer = None
        scheduler = None

    if is_master_proc or (para_type == TRAIN_PARALLEL_INDEPENDENT):
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    if use_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    train_env = TrainEvalEnviron(device_name,
                                 train_set,
                                 dataset_info[DATASET_TYPE],
                                 adj_batch_size,
                                 device_qty, para_type,
                                 model, train_model,
                                 adv_perturb, train_attack,
                                 optimizer,
                                 lower_limit, upper_limit,
                                 writer, use_amp)

    # It is crucial to
    # 1. drop incomplete the last incomplete batch (training code assumes that the size of the batch is constant)
    # 2. do *NOT* shuffle but only if we use attack-|| training
    do_shuffle = para_type != TRAIN_PARALLEL_ATTACK

    print('Use dataloader shuffling:', do_shuffle)

    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                              batch_size=adj_batch_size,
                                              shuffle=do_shuffle,
                                              drop_last=True,
                                              num_workers=num_data_workers,
                                              collate_fn=dataset_info.get('collate_fn', None))

    sync_out()
    # A silly hack to let print finish its work (when we have multiple processes)
    # before tqdm in train starts working
    # TODO a better way to use a condition (and proper logging, not just prints).
    sleep(START_DELAY)

    trainer = create_trainer(trainer_type, train_env, trainer_attack_obj, trainer_args)

    # For independent training processes there's no progress bar,
    # but it's nice to see statistics on the screen
    print_train_stat = (para_type == TRAIN_PARALLEL_INDEPENDENT)

    train(train_loader,
          trainer=trainer,
          num_epochs=epoch_qty,
          batch_sync_step=batch_sync_step,
          batch_sync_barrier=batch_sync_barrier,
          sync_qty_target=sync_qty_target,
          scheduler= scheduler,
          snapshot_dir=snapshot_dir,
          seed=random_seed,
          train_fract=train_fract,
          is_master_proc=is_master_proc,
          print_train_stat=print_train_stat,
          dataset_info=dataset_info)

    if writer is not None:
        writer.close()


def create_attack_from_args_obj(args, attack_id=0, target_id=None):
    """Create a trainer or an evaluator attack.

    :param args:      evaluator or trainer top-level argument object
    :param attack_id: if the list of attack is provided, we chose one of the list.
                      If the list is not large enough, we wrap around by taking the
                      attack id modulo attack list length.
    :param target_id: when multiple attacks are created the attack id
                      can be used to select different targets.
                      if this parameters is None, we create an untargeted attack.

    :return a tuple: attack object, attack weights file (if applicable)
    """
    curr_attack = None
    curr_attack_obj = None
    weights_file = None

    if hasattr(args, ATTACK_ATTR):
        curr_attack = getattr(args, ATTACK_ATTR)
    elif hasattr(args, ATTACK_LIST_ATTR):
        attack_arr = getattr(args, ATTACK_LIST_ATTR)
        assert type(attack_arr) == list
        qty = len(attack_arr)
        curr_attack = attack_arr[attack_id % qty]

    if curr_attack is not None:
        inner_attack = getattr(curr_attack, INNER_ATTACK_ATTR, None)
        target_arr = getattr(curr_attack, TARGETS_ATTR, None)
        target = None
        if target_id is not None and target_arr is not None:
            if target_id >= len(target_arr):
                raise Exception("Target array %s is too short, need at least %d items" % (str(target_arr), target_id + 1))
            target = target_arr[target_id]
            print('Using attack target', target)

        weights_file = getattr(curr_attack, WEIGHTS_FILE_ATTR, None)

        curr_attack_obj = create_attack(curr_attack.attack_type,
                                        inner_attack=inner_attack,
                                        target=target,
                                        epsilon=getattr(curr_attack, ATTACK_EPS_ATTR, None),
                                        norm_name=getattr(curr_attack, ATTACK_NORM_ATTR, None),
                                        add_args=getattr(curr_attack, ADD_ARG_ATTR, EMPTY_CLASS_OBJ))

    return curr_attack_obj, weights_file


def load_attack(weights_file_name):
    """Load attack weights.

    :param weights_file_name: a weights file name or directory to search for the latest snapshot.
                              if the input parameter is None, the function does nothing and returns None.
    :return a loaded weights object or None.
    """
    if weights_file_name is not None:
        weights_file_name = get_snapshot_path(weights_file_name,
                                              ADV_PERTURB_SNAPSHOT_PREFIX,
                                              SNAPSHOT_SUFFIX)
        print('Loading adversarial perturbation from:', weights_file_name)
        return torch.load(weights_file_name, map_location='cpu')
    else:
        return None


def run_trainer(all_args, trainer_args,
                train_set,
                dataset_info,
                lower_limit, upper_limit,
                model,
                model_file_name,
                device_name,
                random_seed,
                train_fract):
    """A wrapper function to run the training process (or processes) corresponding to a single trainer.

    :param all_args:        all parsed configuration parameters.
    :param trainer_args:    the trainer arguments.
    :param train_set:       the training set.
    :param dataset_info:    a dictionary of dataset properties.
    :param lower_limit:     the lower limit for clamping data.
    :param upper_limit:     the upper limit for clamping data.
    :param model:           a model object.
    :param model_file_name: the name of the serialized model file.
    :param device_name:     the name of the device (for single-processing only)
    :param random_seed:     a base random seed.
    :param train_fract:     a fraction of training data to use (or None).
    """

    train_model = getattr(trainer_args, TRAIN_MODEL_ATTR, False)
    train_attack = getattr(trainer_args, TRAIN_ATTACK_ATTR, False)

    optim_type = None
    init_lr = None
    optim_add_args = None

    scheduler_type = None
    scheduler_add_args = None

    if hasattr(trainer_args, OPTIMIZER_ATTR):
        optim_args = getattr(trainer_args, OPTIMIZER_ATTR)
        optim_type = optim_args.algorithm
        init_lr = optim_args.init_lr
        optim_add_args = getattr(optim_args, ADD_ARG_ATTR, EMPTY_CLASS_OBJ)

        if hasattr(optim_args, SCHEDULER_ATTR):
            scheduler_args = getattr(optim_args, SCHEDULER_ATTR)
            scheduler_type = scheduler_args.scheduler_type
            scheduler_add_args = getattr(scheduler_args, ADD_ARG_ATTR, EMPTY_CLASS_OBJ)

    else:
        if train_model:
            raise Exception('Each model training sub-config needs a definition of an optimizer!')

    general_args = all_args.general
    device_qty = general_args.device.device_qty

    log_dir = getattr(trainer_args, LOG_DIR_ATTR, None)
    # Not forgetting to clean-up the log dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    snapshot_dir = getattr(trainer_args, SNAPSHOT_DIR_ATTR, None)

    batch_sync_barrier = Barrier(device_qty)

    if (train_model or train_attack):

        if snapshot_dir is not None:
            # Check that this file doesn't come from the same directory as the snapshot_dir.
            # We will remove the contents of the snapshot directory, so we need
            # to prevent an accidental deletion of seed model/perturbation weights
            if model_file_name is not None and os.path.abspath(snapshot_dir) == \
                    os.path.abspath(os.path.dirname(model_file_name)):
                raise Exception(f'The model weights should not be placed into the snapshot directory {snapshot_dir}!')

            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)
            os.makedirs(snapshot_dir)
            print('Re-created snapshot directory: ', snapshot_dir)



        train_batch_size = trainer_args.train_batch_size
        adj_batch_size = train_batch_size
        data_split_qty = 1
        port=getattr(general_args, MASTER_PORT_ATTR, 0)

        if device_qty > 1:
            if port is None or port < 1:
                raise Exception('Specify the master port for distributed training!')

            if trainer_args.para_type is None:
                raise Exception('Specify a parallelization method for distributed training!')

            if trainer_args.para_type == TRAIN_PARALLEL_INDEPENDENT:
                if train_model:
                    raise Exception(f'Model cannot be trained when the parallelization mode is {TRAIN_PARALLEL_INDEPENDENT}')
                if not train_attack:
                    raise Exception(f'An attack should be trained when the parallelization mode is {TRAIN_PARALLEL_INDEPENDENT}')

            if trainer_args.para_type in [TRAIN_PARALLEL_ATTACK, TRAIN_PARALLEL_DATA]:
                if not train_model:
                    raise Exception(f'Model training flag should be set when ' +
                                    f'parallelization mode is {TRAIN_PARALLEL_ATTACK} or {TRAIN_PARALLEL_DATA}' +
                                    f' or change the parallelization mode to {TRAIN_PARALLEL_INDEPENDENT}')

            if trainer_args.para_type == TRAIN_PARALLEL_DATA:
                adj_batch_size=int(math.ceil(train_batch_size/device_qty))
                data_split_qty = device_qty
            else:
                assert trainer_args.para_type in [TRAIN_PARALLEL_ATTACK, TRAIN_PARALLEL_INDEPENDENT]
                if trainer_args.para_type == TRAIN_PARALLEL_ATTACK and train_batch_size % device_qty > 0:
                    raise Exception(f'For parallelization type {TRAIN_PARALLEL_ATTACK} the batch size must be divisible by the # of devices')
                adj_batch_size = train_batch_size
                data_split_qty = 1

        # We use the data partition class even for device_qty == 1,
        # because it also shuffles data (except for attack-parallel computation mode).
        # Note that we cannot shuffle data by a data
        # loader b/c attack-parallel training needs to read the same
        # set of data in every process, process-specific shuffling (in the function run_group_process)
        # leads to a violation of this assumption
        train_data_partition = DataPartitioner(train_set,
                                         size_fracs=np.full(data_split_qty, 1.0/data_split_qty),
                                         seed=random_seed)

        # This number (of sync. poins) must be the same in each process
        batch_sync_step = getattr(trainer_args, "batch_sync_step", 1)
        sync_qty_target = int(len(train_set) / (device_qty * batch_sync_step * adj_batch_size) )

        param_dict_template = \
            {'dataset_info': dataset_info,
              'device_name': device_name, 'device_qty': device_qty,
              'para_type': trainer_args.para_type,
              'dist_backend': general_args.dist_backend,
              'model': model, 'train_model': train_model,
              'adv_perturb': None, 'train_attack': train_attack,
              'optim_type': optim_type,
              'batch_sync_step': batch_sync_step,
              'batch_sync_barrier': batch_sync_barrier,
              'init_lr': init_lr,
              'optim_add_args': optim_add_args,
              'scheduler_type': scheduler_type, 'scheduler_add_args': scheduler_add_args,
              'use_amp': general_args.use_amp,
              'trainer_type': trainer_args.trainer_type,
              'trainer_args': getattr(trainer_args, ADD_ARG_ATTR, EMPTY_CLASS_OBJ),
              'trainer_attack_obj': None,
              'lower_limit': lower_limit, 'upper_limit': upper_limit,
              'adj_batch_size': adj_batch_size,
              'epoch_qty': trainer_args.epoch_qty,
              'train_fract': train_fract,
              'log_dir': log_dir,
              'num_data_workers': general_args.num_data_workers,
              'port': port,
              'sync_qty_target': sync_qty_target,
              'dist_backend': general_args.dist_backend}


        if trainer_args.para_type != TRAIN_PARALLEL_INDEPENDENT:
            processes = []


            for rank in range(device_qty - 1, -1, -1):
                # If device_qty == 1 or we split at the attack level,
                # this will generate the SHUFFLED version of the complete data set.
                if data_split_qty > 1:
                    data_set = train_data_partition.use(rank)
                else:
                    data_set = train_data_partition.use(0)

                is_master_proc = rank == 0

                trainer_attack_obj, trainer_attack_file_name = create_attack_from_args_obj(trainer_args,
                                                                                           attack_id=rank,
                                                                                           target_id=None)
                if snapshot_dir is not None:
                    if trainer_attack_file_name is not None and os.path.abspath(snapshot_dir) == \
                        os.path.abspath(os.path.dirname(trainer_attack_file_name)):
                        raise Exception(
                            f'The perturbation weights should not be placed into the snapshot directory {snapshot_dir}!')
                adv_perturb = load_attack(trainer_attack_file_name)

                param_dict = copy.copy(param_dict_template)

                param_dict['rank'] = rank
                param_dict['is_master_proc'] = is_master_proc
                param_dict['train_set'] = data_set
                param_dict['snapshot_dir'] = snapshot_dir if is_master_proc else None

                param_dict['adv_perturb'] = adv_perturb
                param_dict['trainer_attack_obj'] = trainer_attack_obj

                # We use a different seed for each process
                # to make sure different processes generate different results when the work
                # on the same subset of data (e.g., in the case of a patch attack)
                param_dict['random_seed'] = general_args.base_seed + rank
                
                if not is_master_proc:
                    p = Process(target=run_train_process, kwargs=param_dict)
                    p.start()
                    processes.append(p)
                else:
                    # The first device is going to train in this process:
                    # no new process is going to be created. In this way,
                    # we can eval on cuda:0 without copying model
                    # from another process or without reloading it from the disk
                    run_train_process(**param_dict)

            for p in processes:
                join_check_exit_stat(p)

        else:
            assert trainer_args.para_type == TRAIN_PARALLEL_INDEPENDENT
            assert data_split_qty == 1, f'Parallelization type {TRAIN_PARALLEL_INDEPENDENT} uses unsplit dataset'

            instance_qty = getattr(trainer_args, INSTANCE_QTY_ATTR, None)
            if instance_qty is None:
                raise Exception(
                    f'Parallelization type {TRAIN_PARALLEL_INDEPENDENT} requires setting {INSTANCE_QTY_ATTR}')

            param_dict_template['train_set'] = train_data_partition.use(0)

            # It would be nice to use Pool here. However, Pools start *NON*-daemonic processes,
            # which, in turn, cannot start other processes. Hence, the data loader cannot use
            # multiprocessing, which is a bigger limitation compared to not using Pool.
            # After all, there shouldn't be big variance in the finishing times (no stragglers) as
            # processes are working on tasks of very similar complexity using the same data and model.
            instance_id = 0

            while instance_id < instance_qty:

                processes = []

                for rank in range(min(device_qty, instance_qty - instance_id)):
                    param_dict = copy.copy(param_dict_template)

                    param_dict['rank'] = rank

                    param_dict['is_master_proc'] = False  # not master proces here

                    snapshot_subdir = os.path.join(snapshot_dir, INSTANCE_SUB_DIR_TEMPLATE % instance_id)
                    os.makedirs(snapshot_subdir)
                    print(f'(re)-creating a snapshot sub-directory: {snapshot_subdir}')
                    log_subdir = os.path.join(log_dir, INSTANCE_SUB_DIR_TEMPLATE % instance_id)
                    os.makedirs(log_subdir)
                    print(f'(re)-creating a snapshot sub-directory: {log_subdir}')

                    param_dict['snapshot_dir'] = snapshot_subdir
                    param_dict['log_dir'] = log_subdir

                    # We use a different seed for each process
                    # to make sure different processes generate different results when the work
                    # on the same subset of data (e.g., in the case of a patch attack)
                    param_dict['random_seed'] = random_seed + instance_id

                    trainer_attack_obj, trainer_attack_file_name = create_attack_from_args_obj(trainer_args,
                                                                                               attack_id=0,
                                                                                               target_id=instance_id)
                    if snapshot_dir is not None:
                        if trainer_attack_file_name is not None and os.path.abspath(snapshot_dir) == \
                            os.path.abspath(os.path.dirname(trainer_attack_file_name)):
                            raise Exception(
                                f'The perturbation weights should not be placed into the snapshot directory {snapshot_dir}!')
                    adv_perturb = load_attack(trainer_attack_file_name)

                    param_dict['adv_perturb'] = adv_perturb
                    param_dict['trainer_attack_obj'] = trainer_attack_obj

                    p = Process(target=run_train_process, kwargs=param_dict)
                    p.start()
                    processes.append(p)
                    instance_id += 1

                for p in processes:
                    join_check_exit_stat(p)


    if use_distr_train(device_qty, trainer_args.para_type):
        dist.destroy_process_group()



def main(input_args):
    """
    The main workhorse function that runs a training and/or evaluation procedure.

    :param input_args:     parsed command line arguments.
    """
    input_args_vars=vars(input_args)
    args = load_settings(input_args.config)

    # Some command line parameters should override JSON-config settings
    for attr_path, attr_val in input_args_vars.items():
        # CONFIG_PARAM is a positional argument
        # We override the value only if the argument is not None
        if attr_path != CONFIG_PARAM and attr_path != EPOCH_QTY_PARAM and attr_val is not None:
            set_nested_attr(args, attr_path, attr_val)

    epoch_qty_arg = input_args_vars[EPOCH_QTY_PARAM]
    if epoch_qty_arg is not None:
        print('Limiting the number of epochs (everywhere) to %d ' % epoch_qty_arg)
        for trainer in args.training:
            trainer.epoch_qty = epoch_qty_arg

    random_seed = args.general.base_seed


    if hasattr(args, EVALUATION_ATTR):
        used_names = set()
        for eval_add_args in args.evaluation.evaluators:
            evaluator_name = eval_add_args.evaluator_name
            if evaluator_name in used_names:
                raise Exception(f'Repeating evaluator name: {evaluator_name}, all eval. names must be unique!')
            used_names.add(evaluator_name)

    # Print arguments after modifying
    print(args)
    device_qty = args.general.device.device_qty

    device_name = getattr(args.general.device, DEVICE_NAME_ATTR, f'cuda:{DEFAULT_CUDA_ID}')
    if device_qty <= 1:
        if device_name not in get_device_list():
            raise Exception('Invalid device name: ' + device_name)
        print('CUDA device:' + device_name)

    train_set, test_set, dataset_info = create_dataset(args.dataset)
    
    class_qty = dataset_info[DATASET_NUM_CLASSES_PARAM]

    mu = get_shaped_tensor(dataset_info, DATASET_MEAN_PARAM)
    std = get_shaped_tensor(dataset_info, DATASET_STD_PARAM)

    upper_limit = get_shaped_tensor(dataset_info, DATASET_UPPER_LIMIT_PARAM)
    lower_limit = get_shaped_tensor(dataset_info, DATASET_LOWER_LIMIT_PARAM)

    # Training processing outline
    # 1. There's always one model shared by all trainers and evaluators.
    # 2. In the beginning, we either create the model from scratch or load weights from disk.
    # 3. Multiple trainer objects can update this model.
    # 4. The trained model weights are always present in the master process. However,
    #    in the case of the multi-device training, this model is replicated in several
    #    processes.
    # 5. If these processes train the model, the results can be aggregated in two ways:
    #    i) The attack-parallel training is useful for PGD and it assigns a separate
    #       sub-batch to compute adversarial examples to a separate GPU.
    #       Then each GPU merges all the generated examples and update its own copy
    #       of the model. These copies are supposed to be nearly identical.
    #    ii) In the data-parallel training, the data is divided among processes each of
    #       which has a separate version of the model. These version are synchronized every
    #       batch_sync_step batches, where batch_sync_step is a (jSON-config) parameter.
    # 6. In all cases, when we have multiple process training a single model, model weights
    #    are always averaged across processes in the end of the epoch.
    # 6. In the case of a trainable attack, the model can be parallelized using only the
    #    data-parallel approach. In that, each process trains its own variant of the attack,
    #    but we save only the attack trained in the master process.
    # 7. To train multiple attacks with a fixed model, one needs a separate trainer/object
    #    config, where:
    #    i) parallelization set to 'independent'
    #    ii) the attribute replica_qty is set to some integer >= 1.
    # 8. In the end, we can run one or more evaluation procedures. They use the previously trained
    #    (or loaded model).

    model = create_toplevel_model(class_qty, mu, std, args.model)

    if hasattr(args.model, WEIGHTS_FILE_ATTR) and args.model.weights_file is not None:
        model_file_name = get_snapshot_path(args.model.weights_file, MODEL_SNAPSHOT_PREFIX, SNAPSHOT_SUFFIX)

        print('Loading model from:', model_file_name)
        # beware: a model is actually a wrapper it doesn't fully mimic Pytorch model interface
        
        saved_state = torch.load(model_file_name, map_location='cpu')
        # We always save model in a wrapper dictionary, but this is not the case
        # when we load some externally pre-trained models
        if SAVE_MODEL_KEY in saved_state.keys():
            model.load_state_dict(saved_state[SAVE_MODEL_KEY])
        else:
            model.orig_model.load_state_dict(saved_state)

    else:
        model_file_name = None

    train_fract = input_args.train_fract
    if train_fract is not None:
        if train_fract < 0 or train_fract > 1:
            raise Exception('Invalid --train_fract parameter!')

    if not args.eval_only:
        for trainer_args in getattr(args, TRAINING_ATTR, []):

            run_trainer(all_args=args,
                        trainer_args=trainer_args,
                        dataset_info=dataset_info,
                        train_set=train_set,
                        model=model,
                        lower_limit=lower_limit,
                        upper_limit=upper_limit,
                        model_file_name=model_file_name,
                        device_name=device_name,
                        random_seed=random_seed,
                        train_fract=train_fract)

    if hasattr(args, EVALUATION_ATTR):
        log_dir = getattr(args.evaluation, LOG_DIR_ATTR, None)
        # Not forgetting to clean-up the log dir

        result_file = None
        if log_dir is not None:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)

            result_file = os.path.join(log_dir, 'summary.json')

        writer = SummaryWriter(log_dir=log_dir)
        eval_batch_size = args.evaluation.eval_batch_size

        eval_fract = input_args.eval_fract
        if eval_fract is not None:
            if eval_fract < 0 or eval_fract > 1:
                raise Exception('Invalid --eval_fract parameter!')
            eval_data_partition = DataPartitioner(test_set,
                                                   size_fracs=[eval_fract],
                                                   seed=random_seed)
            test_set = eval_data_partition.use(0) # We have only one partition defined

        summary_stat = []

        for evaluator in args.evaluation.evaluators:
            evaluator_type = evaluator.evaluator_type
            evaluator_name = evaluator.evaluator_name
            eval_add_args = getattr(evaluator, ADD_ARG_ATTR, EMPTY_CLASS_OBJ)

            print('Evaluator: ' + evaluator_type + '/' + evaluator_name)
            testloader = torch.utils.data.DataLoader(test_set,  pin_memory=True,
                                                     batch_size=eval_batch_size,
                                                     shuffle=False,
                                                     num_workers=args.general.num_data_workers,
                                                     collate_fn=dataset_info.get("collate_fn", None))

            eval_attack_obj, eval_attack_weights_file = create_attack_from_args_obj(evaluator,
                                                                                    attack_id=0,
                                                                                    target_id=None)
            eval_adv_perturb = load_attack(eval_attack_weights_file)

            eval_train_env = TrainEvalEnviron(device_name,
                                              test_set,
                                              dataset_info[DATASET_TYPE],
                                              eval_batch_size,
                                              1, None,
                                              model, False,
                                              eval_adv_perturb, False,
                                              None, 
                                              lower_limit, upper_limit,
                                              writer, args.general.use_amp)

            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            
            evaluator_obj = create_evaluator(evaluator_type, eval_train_env, eval_add_args, eval_attack_obj)
            stat = evaluate(testloader, evaluator_name, evaluator_obj)
            summary_stat.append({'name' : evaluator_name, 'stat': stat})

        if result_file is not None:
            write_json(summary_stat, result_file)

        writer.close()
