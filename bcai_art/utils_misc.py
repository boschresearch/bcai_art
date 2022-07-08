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

import sys
import os
import json
import torch
import copy
import numpy as np
from PIL import Image, ImageDraw
import math

from torch.multiprocessing import set_start_method

from torchvision.utils import make_grid

from bcai_art.utils_tensor import TensorList, DictTensorList, \
    DATASET_TYPE_VIDEO_FIX_SIZE, DATASET_TYPE_AUDIO_FIX_SIZE, DATASET_TYPE_IMAGE

BATCH_TIME_KEY = 'batch_time'
ACCURACY_KEY = 'accuracy'
LOSS_KEY = 'loss'
LR_KEY = 'learning_rate'
ATTACK_SUCCESS_KEY = 'attack_success'
ADV_PERTURB_KEY = 'adv_perturb'

MODEL_SNAPSHOT_PREFIX='model_snapshot_'
ADV_PERTURB_SNAPSHOT_PREFIX='adv_perturb_snapshot_'
SNAPSHOT_SUFFIX='.pth'

CERTIFICATION_KEY = 'certified'
CERTIFICATION_ACCURACY_KEY = 'certified_accuracy'

NORM_INF = "linf"


class EmptyClass:
    pass


EMPTY_CLASS_OBJ = EmptyClass()


def get_trainable_param_count(model): 

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    return params


def model_validity(model, grad_mode = True):
    """Check if any model gradients or parameters are None.

    :param model: a model to check.
    :param grad_mode: if True check gradients. Otherwise, check parameters.

    :return: True if the check succeeded.
    """
    if grad_mode:        
        for p in model.parameters():
            if p.grad is not None and not torch.sum(torch.isnan(p.grad)) == 0:
                return False
            
        return True
    
    return sum(torch.sum(torch.isnan(p)) for p in model.parameters()) == 0


def set_reqgrads(model, flag):
    """Enables/disables gradient computation for *ALL* model parameters

    :param model: input model
    :param flag:  set requres_grad of every parameter to this value.
    """
    for param in model.parameters():
        param.requires_grad = flag


def get_filelist_recursive(root, reg_exp):
    """Find files recursively that match a given glob pattern.
        Unfortunately, none of the standard Python
       functions seem to walk directories recursively proeprly.
       :param   root:     a root directory
       :param   reg_exp:  a regular expression to match the file
    """
    res = []

    for file_name in os.listdir(root):
        file_path_name = os.path.join(root, file_name)
        if not file_name in ['.', '..'] and os.path.isdir(file_path_name):
            res.extend(get_filelist_recursive(file_path_name, reg_exp))
        elif reg_exp.search(file_name):
            res.append(file_path_name)
    return res


def create_dir_if_doesnot_exist(file_name):
    dn = os.path.dirname(file_name)
    if not os.path.exists(dn):
        os.makedirs(dn)


def read_json(fn):
    with open(fn, 'r') as f:
        dt = json.load(f)
    return dt


def write_json(data, fn):
    create_dir_if_doesnot_exist(fn)
    with open(fn, 'w') as f:
        json.dump(data, f, indent=4)


def sync_out():
    """Just flush all stdin and stderr to make streams go in sync.
    It is extremely useful to prevent tqdm interfering with other output.
    """
    sys.stderr.flush()
    sys.stdout.flush()


def join_check_exit_stat(proc):
    """Join the process and raise an exception when a sub-process terminated
       with an error."""
    proc.join()
    if proc.exitcode != 0:
        raise Exception('A process exited abnormally with code:' + str(proc.exitcode))


def enable_spawn():
    """All multiprocessing with CUDA must call this function in the main module.
    The main module code must be guarded by:

    if __name__ == '__main__':

        ...

        enable_spawn()

        ...

    """
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass


class ObjectFromDict(object):
    """Make an object from a dictionary"""
    def __init__(self, d):
        """Constructor.
        :param d: input dictionary.
        """
        self.__dict__ = d


def delete_selected_args(src_args, args_to_del):
    """Generate a dictionary of arguments (from an existing one),
       but exclude selected attributes.
    """
    res_args = copy.deepcopy(src_args)

    for arg in args_to_del:
        try:
            delattr(res_args, arg)
        except:
            continue

    return res_args


def paramdict_to_args(inp_dict):
    """Convert a dictionary into an object recursively.
       The recursion processes only values that are dictionaries,
       all other types including arrays are not converted.
       Be careful, it does not check for possible (infinite) loops.

       :param inp_dict:  input dictionry.

       :return an object of the type ObjectFromDict whose
               attributes are defined by the input dictionary.
    """
    tmp_dict = {}

    for k, v in inp_dict.items():
        if isinstance(v, dict):
            tmp_dict[k] = paramdict_to_args(v)
        else:
            tmp_dict[k] = v

    return ObjectFromDict(tmp_dict)


def args_to_paramdict(arg_obj, exclude_params=[]):
    """Create a parameter dictionary from an arguments object *AND*
       optionally exclude parameters from a given list. The resulting
       dictionary can be later used with a double-asterisk notation
       to pass a variable number of named arguments to a function.

    :param arg_obj:          an argument object
    :param exclude_params:   an array of parameters to exlucde
    :return: a dictionary where keys are parameter names and values are parameter values:
             all parameters from a given list are excluded.
    """
    if arg_obj is None:
        return {}

    ret = vars(arg_obj)
    for key in exclude_params:
        if key in ret:
            del(ret[key])

    return ret


def get_snapshot_path(file_or_dir, file_prefix, file_suffix):
    """Determine a snapshot path: if the path is an exact file name, we load this file name.
       If the path specifies the directory only, we load  the latest snapshot.

    :param file_or_dir: model snapshot directory or exact file name
    :param file_prefix: a prefix of the model file (snapshot number excluded)
    :param file_suffix: a suffix of the model file (snapshot number excluded)
    :return: model file location
    """

    if os.path.isfile(file_or_dir):
        return file_or_dir
    if not os.path.isdir(file_or_dir):
        raise Exception(file_or_dir + ' is not a proper file/directory!')
    file_list = []
    for orig_fn in os.listdir(file_or_dir):
        fn = orig_fn
        if fn.startswith(file_prefix):
            fn = fn[len(file_prefix):]
            if fn.endswith(file_suffix):
                fn = fn[0:-len(file_suffix):]
                if not fn.isnumeric():
                    raise Exception('Invalid file name: ' + orig_fn)
                file_list.append((int(fn), orig_fn))

    file_list.sort(reverse=True)
    if not file_list:
        raise Exception(
            f'No snapshot files are found in {file_or_dir}, expected prefix/suffix: {file_prefix}/{file_suffix}')
    return os.path.join(file_or_dir, file_list[0][1])


def set_nested_attr(obj, attr_path, attr_val):
    """Set a nested attribute value. For example, if we have
       an attribute a that has an attribute b for the object
       instance x, we can do: set_nested_attr(x, 'a.b', some_value).
       All *BUT THE LAST* attributes in the sequence must exist.

       :param      obj: an object whose attribute we need to change
       :param      attr_path:  a an attribute 'path', i.e., dot-separated
                               names of nested attributes (see the example above).
       :param      attr_val:   a new attribute value


    """
    curr_obj = obj
    prop_names = attr_path.split('.')
    for k in range(len(prop_names)):
        curr_prop = prop_names[k]

        if k + 1 < len(prop_names):
            if hasattr(curr_obj, curr_prop):
                curr_obj = getattr(curr_obj, curr_prop)
            else:
                raise Exception(f'Missing attribute {curr_prop} in the sequence {attr_path}')
        else:
            setattr(curr_obj, curr_prop, attr_val)


def get_norm_code(norm_name):
    """Convert an LP norm name to a code. If name
       is None, None is returned.

    :param norm_name: one of the norm names
    :return: an LP-norm p-number or np.inf for LINF
    """
    if norm_name is None:
        return None

    norm_name = norm_name.lower()
    if norm_name == NORM_INF:
        return np.inf

    if norm_name.startswith('l'):
        try:
            p = int(norm_name[1:])
            if p >= 1:
                return p
        except:
            pass

        raise Exception('Invalid LP-norm code: ' + norm_name)

    raise Exception('Unsupported norm: ' + norm_name)


def calc_correct_qty(outputs, labels):
    """
    Calculate # of correct outputs given network outputs and true labels
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct


def remove_state_dict_prefix(state_dict, prefix='module.'):
    """Use of wrappers, in particular, DataParallel leads to addition of silly prefixes such as
      module, which prevent the model from being loaded later (if the model is used without the wrapper).

    :param   state_dict: initial state dictionary
    :prefix  prefix: a prefix to get rid of

    :return: a new state dictionary with updated keys (prefix removed).
    """
    res = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        if k in res:
            raise Exception('Repeating key!' + k)
        res[k] = v
    return res


def save_sample_and_pred_info(dataset_type,
                              X_lst, y_true_lst, X_perturbed_lst, y_pred_lst,
                              log_name, log_writer):
    """Save information about original and perturbed samples as well
       as about model predictions.

    :param dataset_type:        a type of the dataset
    :param X_lst:               a list of original images
    :param y_true_lst:          a list of original image labels
    :param X_perturbed_lst:     a list of perturbted images
    :param y_pred_lst:          a list of model predictions
    :param log_name:            some name or id of the writing process
    :param log_writer:          Tensorbaord log writer
    """

    assert log_writer is not None

    tlst = []
    qty = len(X_lst)
    desc = []

    for i in range(qty):
        tlst.append(X_lst[i])
        tlst.append(X_perturbed_lst[i])
        desc.append({'y_true' : y_true_lst[i].cpu().item(), 'y_pred' : y_pred_lst[i].cpu().item()})

    pred_json = json.dumps(desc, indent=4)
    log_writer.add_text(f'{log_name}/SAMPLE_PRED', pred_json)

    if tlst:
        if dataset_type == DATASET_TYPE_IMAGE:
            grd = make_grid(tlst, nrow=2)
            log_writer.add_image(f'{log_name}/SAMPLES', grd)
        elif dataset_type == DATASET_TYPE_AUDIO_FIX_SIZE:
            for i in range(qty):
                log_writer.add_audio(f'{log_name}/SAMPLE_X_{i}', X_lst[i])
                log_writer.add_audio(f'{log_name}/SAMPLE_X_pert_{i}', X_perturbed_lst[i])
        elif dataset_type == DATASET_TYPE_VIDEO_FIX_SIZE:
            for i in range(qty):
                assert len(X_lst[i].shape) == 4, "Expecting video tensor with the shape T x C x H x W (without batch)"
                assert len(X_perturbed_lst[i].shape) == 4, "Expecting video tensor with the shape T x C x H x W (without batch)"
                log_writer.add_video(f'{log_name}/SAMPLE_X_{i}', X_lst[i].unsqueeze(dim=0))
                log_writer.add_video(f'{log_name}/SAMPLE_X_pert_{i}', X_perturbed_lst[i].unsqueeze(dim=0))
    else:
        print('Warning: no samples to write! It is a configuration issue or a bug.')


def get_dfom(clean_trained_clean_data_acc, defended_attacked_data_acc, clean_trained_attacked_data_acc):
    """Compute DFOM effectiveness metric.

    :param clean_trained_clean_data_acc:     accuracy of the standard model on clean data (N in GARD terms)
    :param defended_attacked_data_acc:       accuracy of a defended model under attack (M in GARD terms)
    :param clean_trained_attacked_data_acc:  accuracy of the clean model under attack (P in GARD terms)
    :return: DFOM value
    """

    return (defended_attacked_data_acc - clean_trained_attacked_data_acc) / \
           (clean_trained_clean_data_acc - clean_trained_attacked_data_acc)


def mscoco_collate_fn(batch):
    """Collate function for datasets stored in MSCOCO format"""
    converted = tuple(zip(*batch))
    return (TensorList(converted[0]), DictTensorList(converted[1]))


def mscoco_target_transform(coco_annotation):
    """Label-transformation function for datasets stored in MSCOCO format"""
    num_objs = len(coco_annotation)

    # Bounding boxes for objects
    # In coco format, bbox = [xmin, ymin, width, height]
    # In pytorch, the input should be [xmin, ymin, xmax, ymax]
    boxes = []
    labels = []
    areas = []
    iscrowds = []
    image_ids= []
    patch_ids= []
    for i in range(num_objs):
        xmin = coco_annotation[i]['bbox'][0]
        ymin = coco_annotation[i]['bbox'][1]
        xmax = xmin + coco_annotation[i]['bbox'][2]
        ymax = ymin + coco_annotation[i]['bbox'][3]
        if xmin<xmax and ymin<ymax:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
            try:
                areas.append(coco_annotation[i]['area'])
            except:
                areas.append(coco_annotation[i]['areas'])
            try:
                iscrowds.append(coco_annotation[i]['iscrowd'])
            except:
                iscrowds.append(coco_annotation[i]['is_crowd'])
            image_ids.append(coco_annotation[i]['image_id'])
            if 'patch_id' in coco_annotation[i]:
                patch_ids.append(coco_annotation[i]['patch_id'])
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    # Labels (In my case, I only one class: target class or background)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    # Tensorise img_id
    img_id = torch.as_tensor(image_ids[0] if len(image_ids)>0 else image_ids, dtype=torch.int64)
    # Size of bbox (Rectangular)
    areas = torch.as_tensor(areas, dtype=torch.float32)
    # Iscrowd
    iscrowd = torch.as_tensor(iscrowds, dtype=torch.int64)
    if len(patch_ids)>0:
        patch_id = torch.as_tensor(patch_ids[0], dtype=torch.int64)
    # Annotation is in dictionary format
    my_annotation = {}
    my_annotation["boxes"] = boxes
    my_annotation["labels"] = labels
    my_annotation["image_id"] = img_id
    my_annotation["area"] = areas
    my_annotation["iscrowd"] = iscrowd
    if len(patch_ids)>0:
        my_annotation["patch_id"] = patch_id

    return my_annotation


## from https://github.com/pytorch/vision/blob/fc34ccb29229a9651bde932069786ed5330dd986/torchvision/utils.py
def draw_bounding_boxes(
    image,boxes,labels,label_names,colors=None,draw_labels=True,width = 5
):

    """
    Draws bounding boxes on given image.
    Args:
        image (Tensor): Tensor of shape (C x H x W) or (1 x C x H x W)
        bboxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
        labels (Tensor): Tensor of size (N) Labels for each bounding boxes.
        label_names (List): List containing labels excluding background.
        colors (dict): Dict with key as label id and value as color name.
        draw_labels (bool): If True (default) draws label names on bounding boxes.
        width (int): Width of bounding box.
    """

    # Code co-contributed by sumanthratna

    if not (torch.is_tensor(image)):
        raise TypeError(f'tensor expected, got {type(image)}')

    if(image.dim() == 4):
        if(image.shape[0] == 1):
            image = image.squeeze(0)
        else:
            raise ValueError("Batch size > 1 is not supported. Pass images with batch size 1 only")

    if label_names is not None:
        # Since for our detection models class 0 is background
        label_names.insert(0, "__background__")

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # Neceassary check to remove grad if present
    if(boxes.requires_grad):
        boxes = boxes.detach()

    boxes = boxes.to('cpu').numpy().astype('int').tolist()
    labels = labels.to('cpu').numpy().astype('int').tolist()

    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)

    for bbox, label in zip(boxes, labels):
        if colors is None:
            draw.rectangle(bbox, width=width)
        else:
            draw.rectangle(bbox, width=width, outline=colors[label])

        if label_names is None:
            draw.text((bbox[0], bbox[1]), str(label))
        elif draw_labels is True:
            draw.text((bbox[0], bbox[1]), label_names[int(label)])

    return img_to_draw


def create_colormap(num_classes):
    """
    given num_classes, return a list of tuples of 3. Each tuple represent RGB color of the class
    """
    per_channel = math.ceil(num_classes** (1. / 3))
    channels = np.ones(3)*per_channel
    channels=channels.astype(int)

    colormap = []
    step = np.floor(255/channels).astype(int)
    
    for rr in range(channels[0]):
        for gg in range(channels[1]):
            for bb in range(channels[2]):
                colormap.append((step[0]*rr,step[1]*gg, step[2]*bb))

    return colormap[:num_classes]
