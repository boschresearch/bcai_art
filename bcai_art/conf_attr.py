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

# We need a separate file to keep all commonly used attributes/constants

# Only for datasets converted to numpy
DATASET_REG_FILE = 'registry_file.pth'

LOG_DIR_ATTR="log_dir"
SCHEDULER_ATTR = "scheduler"
TRAINING_ATTR = "training"
OPTIMIZER_ATTR = "optimizer"
TRAIN_ATTACK_ATTR = "train_attack"
TRAIN_MODEL_ATTR = "train_model"
WEIGHTS_FILE_ATTR = "weights_file"
ATTACK_EPS_ATTR = "eps"
ATTACK_NORM_ATTR = "norm"
SNAPSHOT_DIR_ATTR = "snapshot_dir"
DEVICE_NAME_ATTR = "device_name"
EVALUATION_ATTR = "evaluation"
INNER_ATTACK_ATTR = "inner_attack"
TARGETS_ATTR = "targets"
ATTACK_ATTR = "attack"
ATTACK_LIST_ATTR = "attack_list"
INSTANCE_QTY_ATTR = "instance_qty"
MASTER_PORT_ATTR = "master_port"

SAMPLE_QTY= "sample_qty"
SIGMA="sigma"
UCF101_SAMPLE_SIZE_ATTR = "ucf101_sample_size"
UCF101_MAX_TEST_SAMPLE_QTY_ATTR = "ucf101_max_test_sample_qty"

FREEZE_INNER_MODEL_ATTR = "freeze_inner_model"
FREEZE_FEATURES_MODEL_ATTR = "freeze_feature_extractor"

LOSS_ATTR = "loss"
LOSS_NATIVE = "native"
MODEL_ARCH_ATTR = "architecture"
ADD_ARG_ATTR = "add_arguments"
INNER_MODEL_ATTR = "inner_model"
USE_PRETRAINED_ATTR = "pretrained"


SAVE_MODEL_KEY='model'
SAVE_MODEL_EPOCH='epoch'
SAVE_MODEL_OPTIM='otimizer'
MODEL_ARGS_KEY='model_args'
INP_SHAPE_KEY='input_shape'
