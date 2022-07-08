#
# This is based on the code from Armory:
# 
# https://github.com/twosixlabs/armory
# 
# Copyright (c) 2019 Two Six Labs
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
#
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import inspect
import numpy as np
from collections import Counter
from tqdm import tqdm
from bcai_art.utils_misc import calc_correct_qty, args_to_paramdict

import pdb


class Meter(object):
    """parent class for any metric. By default always returns 0"""
    def __init__(self, preds, labels):
        """Constructor"""
        pass

    def update(self, preds, labels):
        """This functions gets called on every batch"""
        return 0.0

    def finalize(self):
        """This functions gets called once a training epoch is done or when the evaluation ends"""
        pass


class SimpleAccuracy(Meter):
    """Computes simple classification accuracy """
    def __init__(self,trainenv):
        pass
    def update(self, preds, labels):
        return calc_correct_qty(preds, labels)


class mAPMeter(Meter):
    """Computes mAP for object detection models.
       It simply store results on every batch and then calculates mAP at the end.
    """
    def __init__(self, train_env, iou_type="bbox"):
        self.all_preds=[]
        self.iou_type = iou_type
        self.train_env= train_env

    def convert_to_xywh(self,boxes):
        """
        Make bbox coordinates compatible with COCO api
        """
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def prepare_for_coco_detection(self, predictions):
        """
        Put the model outputs into the right format for the COCO api.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = self.convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def update(self, preds , ground_truth):
        """
        Store predictions and return 0.0
        """
        if len(ground_truth)>0:
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in preds]
            res = {}
            for target, output in zip(ground_truth, outputs):
                if target["image_id"].nelement()==1:
                    res = {**{target["image_id"].item(): output }, **res}
            self.all_preds.extend(self.prepare_for_coco_detection(res))
        return 0.0

    def finalize(self):
        """
        Actual mAP calculation is get done here.
        """
        try:
            cocoGt = self.train_env.data_set.coco
        except:
            cocoGt = self.train_env.data_set.dataset.coco
            
        if len(self.all_preds) < 1:
            print("no predictions found over the whole set")
            return
        cocoDt = cocoGt.loadRes(self.all_preds)
        cocoEval = COCOeval(cocoGt, cocoDt, self.iou_type)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        self.all_preds=[]


def object_detection_class_precision(y, y_pred, ADV_PATCH_MAGIC_NUMBER_LABEL_ID, score_threshold=0.5):
    _check_object_detection_input(y, y_pred)
    num_tps, num_fps, num_fns = _object_detection_get_tp_fp_fn(
        y, y_pred[0], ADV_PATCH_MAGIC_NUMBER_LABEL_ID, score_threshold=score_threshold
    )
    if num_tps + num_fps > 0:
        return [num_tps / (num_tps + num_fps)]
    else:
        return [0]


def object_detection_class_recall(y, y_pred, ADV_PATCH_MAGIC_NUMBER_LABEL_ID, score_threshold=0.5):
    _check_object_detection_input(y, y_pred)
    num_tps, num_fps, num_fns = _object_detection_get_tp_fp_fn(
        y, y_pred[0], ADV_PATCH_MAGIC_NUMBER_LABEL_ID, score_threshold=score_threshold
    )
    if num_tps + num_fns > 0:
        return [num_tps / (num_tps + num_fns)]
    else:
        return [0]


def _object_detection_get_tp_fp_fn(y, y_pred, ADV_PATCH_MAGIC_NUMBER_LABEL_ID, score_threshold=0.5):
    """
    Helper function to compute the number of true positives, false positives, and false
    negatives given a set of of object detection labels and predictions
    """
    ground_truth_set_of_classes = set(
        y["labels"][np.where(y["labels"] != ADV_PATCH_MAGIC_NUMBER_LABEL_ID)]
        .flatten()
        .tolist()
    )
    predicted_set_of_classes = set(
        y_pred["labels"][np.where(y_pred["scores"] > score_threshold)].tolist()
    )

    num_true_positives = len(
        predicted_set_of_classes.intersection(ground_truth_set_of_classes)
    )
    num_false_positives = len(
        [c for c in predicted_set_of_classes if c not in ground_truth_set_of_classes]
    )
    num_false_negatives = len(
        [c for c in ground_truth_set_of_classes if c not in predicted_set_of_classes]
    )

    return num_true_positives, num_false_positives, num_false_negatives


def _check_object_detection_input(y, y_pred):
    """
    Helper function to check that the object detection labels and predictions are in
    the expected format and contain the expected fields
    """
    if not isinstance(y, dict):
        raise TypeError("Expected y to be a dictionary")

    if not isinstance(y_pred, list):
        raise TypeError("Expected y_pred to be a list")

    # Current object detection pipeline only supports batch_size of 1
    if len(y_pred) != 1:
        raise ValueError(
            f"Expected y_pred to be a list of length 1, found length of {len(y_pred)}"
        )

    y_pred = y_pred[0]

    REQUIRED_LABEL_KEYS = ["labels", "boxes"]
    REQUIRED_PRED_KEYS = REQUIRED_LABEL_KEYS + ["scores"]

    if not all(key in y for key in REQUIRED_LABEL_KEYS):
        raise ValueError(
            f"y must contain the following keys: {REQUIRED_LABEL_KEYS}. The following keys were found: {y.keys()}"
        )

    if not all(key in y_pred for key in REQUIRED_PRED_KEYS):
        raise ValueError(
            f"y_pred must contain the following keys: {REQUIRED_PRED_KEYS}. The following keys were found: {y_pred.keys()}"
        )


def _intersection_over_union(box_1, box_2):
    """
    Assumes each input has shape (4,) and format [y1, x1, y2, x2] or [x1, y1, x2, y2]
    """
    assert box_1[2] >= box_1[0]
    assert box_2[2] >= box_2[0]
    assert box_1[3] >= box_1[1]
    assert box_2[3] >= box_2[1]

    if all(i <= 1.0 for i in box_1[np.where(box_1 > 0)]) ^ all(
        i <= 1.0 for i in box_2[np.where(box_2 > 0)]
    ):
        print(
            "One set of boxes appears to be normalized while the other is not"
        )

    # Determine coordinates of intersection box
    x_left = max(box_1[1], box_2[1])
    x_right = min(box_1[3], box_2[3])
    y_top = max(box_1[0], box_2[0])
    y_bottom = min(box_1[2], box_2[2])

    intersect_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    if intersect_area == 0:
        return 0

    box_1_area = (box_1[3] - box_1[1]) * (box_1[2] - box_1[0])
    box_2_area = (box_2[3] - box_2[1]) * (box_2[2] - box_2[0])

    iou = intersect_area / (box_1_area + box_2_area - intersect_area)
    assert iou >= 0
    assert iou <= 1
    return iou


def object_detection_AP_per_class(
        y_list,
        y_pred_list,
        iou_threshold=0.5,
        class_list=None,
        targets=None,
        adv_patch_magic_number_label_id=-10
):
    """
    Mean average precision for object detection. This function returns a dictionary
    mapping each class to the average precision (AP) for the class. The mAP can be computed
    by taking the mean of the AP's across all classes.
    This metric is computed over all evaluation samples, rather than on a per-sample basis.
    """

    # Precision will be computed at recall points of 0, 0.1, 0.2, ..., 1
    RECALL_POINTS = np.linspace(0, 1, 11)

    # Converting all boxes to a list of dicts (a list for predicted boxes, and a
    # separate list for ground truth boxes), where each dict corresponds to a box and
    # has the following keys "img_idx", "label", "box", as well as "score" for predicted boxes
    pred_boxes_list = []
    gt_boxes_list = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        img_labels = y["labels"].flatten()
        img_boxes = y["boxes"].reshape((-1, 4))
        for gt_box_idx in range(img_labels.flatten().shape[0]):
            label = img_labels[gt_box_idx].item()
            box = img_boxes[gt_box_idx]
            gt_box_dict = {"img_idx": img_idx, "label": label, "box": box}
            gt_boxes_list.append(gt_box_dict)

        for pred_box_idx in range(y_pred["labels"].flatten().shape[0]):
            pred_label = (y_pred["labels"][pred_box_idx]).item()
            pred_box = y_pred["boxes"][pred_box_idx]
            pred_score = y_pred["scores"][pred_box_idx]
            pred_box_dict = {
                "img_idx": img_idx,
                "label": pred_label,
                "box": pred_box,
                "score": pred_score,
            }
            pred_boxes_list.append(pred_box_dict)

    # Union of (1) the set of all true classes and (2) the set of all predicted classes
    set_of_class_ids = set([i["label"] for i in gt_boxes_list]) | set(
        [i["label"] for i in pred_boxes_list]
    )

    if class_list:
        # Filter out classes not in class_list
        set_of_class_ids = set(i for i in set_of_class_ids if i in class_list)

    # Remove the class ID that corresponds to a physical adversarial patch in APRICOT
    # dataset, if present
    set_of_class_ids.discard(adv_patch_magic_number_label_id)

    # Initialize dict that will store AP for each class
    average_precisions_by_class = {}

    # Compute AP for each class
    for class_id in tqdm(set_of_class_ids):

        # Build lists that contain all the predicted/ground-truth boxes with a
        # label of class_id
        class_predicted_boxes = []
        class_gt_boxes = []
        for pred_box in pred_boxes_list:
            if pred_box["label"] == class_id:
                class_predicted_boxes.append(pred_box)
        for gt_box in gt_boxes_list:
            if gt_box["label"] == class_id:
                class_gt_boxes.append(gt_box)

        # Determine how many gt boxes (of class_id) there are in each image
        num_gt_boxes_per_img = Counter([gt["img_idx"] for gt in class_gt_boxes])

        # Initialize dict where we'll keep track of whether a gt box has been matched to a
        # prediction yet. This is necessary because if multiple predicted boxes of class_id
        # overlap with a single gt box, only one of the predicted boxes can be considered a
        # true positive
        img_idx_to_gtboxismatched_array = {}
        for img_idx, num_gt_boxes in num_gt_boxes_per_img.items():
            img_idx_to_gtboxismatched_array[img_idx] = np.zeros(num_gt_boxes)

        # Sort all predicted boxes (of class_id) by descending confidence
        class_predicted_boxes.sort(key=lambda x: x["score"], reverse=True)

        # Initialize arrays. Once filled in, true_positives[i] indicates (with a 1 or 0)
        # whether the ith predicted box (of class_id) is a true positive. Likewise for
        # false_positives array
        true_positives = np.zeros(len(class_predicted_boxes))
        false_positives = np.zeros(len(class_predicted_boxes))

        # Iterating over all predicted boxes of class_id
        for pred_idx, pred_box in enumerate(class_predicted_boxes):
            # Only compare gt boxes from the same image as the predicted box
            gt_boxes_from_same_img = [
                gt_box
                for gt_box in class_gt_boxes
                if gt_box["img_idx"] == pred_box["img_idx"]
            ]

            # If there are no gt boxes in the predicted box's image that have the predicted class
            if len(gt_boxes_from_same_img) == 0:
                false_positives[pred_idx] = 1
                continue

            # Iterate over all gt boxes (of class_id) from the same image as the predicted box,
            # determining which gt box has the highest iou with the predicted box
            highest_iou = 0
            for gt_idx, gt_box in enumerate(gt_boxes_from_same_img):
                iou = _intersection_over_union(pred_box["box"], gt_box["box"])
                if iou >= highest_iou:
                    highest_iou = iou
                    highest_iou_gt_idx = gt_idx

            if highest_iou > iou_threshold:
                # If the gt box has not yet been covered
                if (
                    img_idx_to_gtboxismatched_array[pred_box["img_idx"]][
                        highest_iou_gt_idx
                    ]
                    == 0
                ):
                    true_positives[pred_idx] = 1

                    # Record that we've now covered this gt box. Any subsequent
                    # pred boxes that overlap with it are considered false positives
                    img_idx_to_gtboxismatched_array[pred_box["img_idx"]][
                        highest_iou_gt_idx
                    ] = 1
                else:
                    # This gt box was already covered previously (i.e a different predicted
                    # box was deemed a true positive after overlapping with this gt box)
                    false_positives[pred_idx] = 1
            else:
                false_positives[pred_idx] = 1

        # Cumulative sums of false/true positives across all predictions which were sorted by
        # descending confidence
        tp_cumulative_sum = np.cumsum(true_positives)
        fp_cumulative_sum = np.cumsum(false_positives)

        # Total number of gt boxes with a label of class_id
        total_gt_boxes = len(class_gt_boxes)

        recalls = tp_cumulative_sum / (total_gt_boxes + 1e-6)
        precisions = tp_cumulative_sum / (tp_cumulative_sum + fp_cumulative_sum + 1e-6)

        interpolated_precisions = np.zeros(len(RECALL_POINTS))
        # Interpolate the precision at each recall level by taking the max precision for which
        # the corresponding recall exceeds the recall point
        # See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf
        for i, recall_point in enumerate(RECALL_POINTS):
            precisions_points = precisions[np.where(recalls >= recall_point)]
            # If there's no cutoff at which the recall > recall_point
            if len(precisions_points) == 0:
                interpolated_precisions[i] = 0
            else:
                interpolated_precisions[i] = max(precisions_points)

        # Compute mean precision across the different recall levels
        average_precision = interpolated_precisions.mean()
        average_precisions_by_class[int(class_id)] = np.around(
            average_precision, decimals=2
        )

    print(average_precisions_by_class)

    if targets is not None:
        print("printing mAP for only targeted class:")
        av_map = 0.0
        num_target_class = 0

        for class_id, ap in average_precisions_by_class.items():
            if class_id in targets:
                print("class " + str(class_id) + " : " + str(ap))
                av_map += ap
                num_target_class += 1
        print("averaged target map: " + str(av_map))

    return average_precisions_by_class


def object_detection_mAP(
        y_list, y_pred_list, iou_threshold=0.5, class_list=None, targets=None, adv_patch_magic_number_label_id=-10
):
    ap_per_class = object_detection_AP_per_class(
        y_list, y_pred_list, class_list, targets, adv_patch_magic_number_label_id
    )
    return np.fromiter(ap_per_class.values(), dtype=float).mean()


def _object_detection_get_tpr_mr_dr_hr(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):

    true_positive_rate_per_img = []
    misclassification_rate_per_img = []
    disappearance_rate_per_img = []
    hallucinations_per_img = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        if class_list:
            # Filter out ground-truth classes with labels not in class_list
            indices_to_keep = np.where(np.isin(y["labels"], class_list))
            gt_boxes = y["boxes"][indices_to_keep]
            gt_labels = y["labels"][indices_to_keep]
        else:
            gt_boxes = y["boxes"]
            gt_labels = y["labels"]

        # initialize count of hallucinations
        num_hallucinations = 0
        num_gt_boxes = len(gt_boxes)

        # Initialize arrays that will indicate whether each respective ground-truth
        # box is a true positive or misclassified
        true_positive_array = np.zeros((num_gt_boxes,))
        misclassification_array = np.zeros((num_gt_boxes,))

        # Only consider the model's confident predictions
        conf_pred_indices = np.where(y_pred["scores"] > score_threshold)[0]
        if class_list:
            # Filter out predictions from classes not in class_list kwarg
            conf_pred_indices = conf_pred_indices[
                np.isin(y_pred["labels"][conf_pred_indices], class_list)
            ]

        # For each confident prediction
        for y_pred_idx in conf_pred_indices:
            y_pred_box = y_pred["boxes"][y_pred_idx]

            # Compute the iou between the predicted box and the ground-truth boxes
            ious = np.array([_intersection_over_union(y_pred_box, a) for a in gt_boxes])

            # Determine which ground-truth boxes, if any, the predicted box overlaps with
            overlap_indices = np.where(ious > iou_threshold)[0]

            # If the predicted box doesn't overlap with any ground-truth boxes, increment
            # the hallucination counter and move on to the next predicted box
            if len(overlap_indices) == 0:
                num_hallucinations += 1
                continue

            # For each ground-truth box that the prediction overlaps with
            for y_idx in overlap_indices:
                # If the predicted label is correct, mark that the ground-truth
                # box has a true positive prediction
                if gt_labels[y_idx] == y_pred["labels"][y_pred_idx]:
                    true_positive_array[y_idx] = 1
                else:
                    # Otherwise mark that the ground-truth box has a misclassification
                    misclassification_array[y_idx] = 1

        # Convert these arrays to binary to avoid double-counting (i.e. when multiple
        # predicted boxes overlap with a single ground-truth box)
        true_positive_rate = (true_positive_array > 0).mean()
        misclassification_rate = (misclassification_array > 0).mean()

        # Any ground-truth box that had no overlapping predicted box is considered a
        # disappearance
        disappearance_rate = 1 - true_positive_rate - misclassification_rate

        true_positive_rate_per_img.append(true_positive_rate)
        misclassification_rate_per_img.append(misclassification_rate)
        disappearance_rate_per_img.append(disappearance_rate)
        hallucinations_per_img.append(num_hallucinations)

    return (
        true_positive_rate_per_img,
        misclassification_rate_per_img,
        disappearance_rate_per_img,
        hallucinations_per_img,
    )


def object_detection_true_positive_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):

    true_positive_rate_per_img, _, _, _ = _object_detection_get_tpr_mr_dr_hr(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )
    return true_positive_rate_per_img


def object_detection_misclassification_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    _, misclassification_rate_per_image, _, _ = _object_detection_get_tpr_mr_dr_hr(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )
    return misclassification_rate_per_image


def object_detection_disappearance_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    _, _, disappearance_rate_per_img, _ = _object_detection_get_tpr_mr_dr_hr(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )
    return disappearance_rate_per_img


def object_detection_hallucinations_per_image(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    _, _, _, hallucinations_per_image = _object_detection_get_tpr_mr_dr_hr(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )
    return hallucinations_per_image
          

APRICOT_PATCHES = {
    0: {
        "adv_model": 0,
        "adv_target": 53,
        "id": 0,
        "is_circle": True,
        "is_square": False,
        "name": "frc1",
    },
    1: {
        "adv_model": 0,
        "adv_target": 27,
        "id": 1,
        "is_circle": True,
        "is_square": False,
        "name": "frc2",
    },
    2: {
        "adv_model": 0,
        "adv_target": 44,
        "id": 2,
        "is_circle": True,
        "is_square": False,
        "name": "frc3",
    },
    3: {
        "adv_model": 0,
        "adv_target": 17,
        "id": 3,
        "is_circle": True,
        "is_square": False,
        "name": "frc4",
    },
    4: {
        "adv_model": 0,
        "adv_target": 85,
        "id": 4,
        "is_circle": True,
        "is_square": False,
        "name": "frc5",
    },
    5: {
        "adv_model": 0,
        "adv_target": 73,
        "id": 5,
        "is_circle": True,
        "is_square": False,
        "name": "frc6",
    },
    6: {
        "adv_model": 0,
        "adv_target": 78,
        "id": 6,
        "is_circle": True,
        "is_square": False,
        "name": "frc7",
    },
    7: {
        "adv_model": 0,
        "adv_target": 1,
        "id": 7,
        "is_circle": True,
        "is_square": False,
        "name": "frc8",
    },
    8: {
        "adv_model": 0,
        "adv_target": 64,
        "id": 8,
        "is_circle": True,
        "is_square": False,
        "name": "frc9",
    },
    9: {
        "adv_model": 0,
        "adv_target": 33,
        "id": 9,
        "is_circle": True,
        "is_square": False,
        "name": "frc10",
    },
    10: {
        "adv_model": 0,
        "adv_target": 53,
        "id": 10,
        "is_circle": False,
        "is_square": True,
        "name": "frs1",
    },
    11: {
        "adv_model": 0,
        "adv_target": 27,
        "id": 11,
        "is_circle": False,
        "is_square": True,
        "name": "frs2",
    },
    12: {
        "adv_model": 0,
        "adv_target": 44,
        "id": 12,
        "is_circle": False,
        "is_square": True,
        "name": "frs3",
    },
    13: {
        "adv_model": 0,
        "adv_target": 17,
        "id": 13,
        "is_circle": False,
        "is_square": True,
        "name": "frs4",
    },
    14: {
        "adv_model": 0,
        "adv_target": 85,
        "id": 14,
        "is_circle": False,
        "is_square": True,
        "name": "frs5",
    },
    15: {
        "adv_model": 0,
        "adv_target": 73,
        "id": 15,
        "is_circle": False,
        "is_square": True,
        "name": "frs6",
    },
    16: {
        "adv_model": 0,
        "adv_target": 78,
        "id": 16,
        "is_circle": False,
        "is_square": True,
        "name": "frs7",
    },
    17: {
        "adv_model": 0,
        "adv_target": 1,
        "id": 17,
        "is_circle": False,
        "is_square": True,
        "name": "frs8",
    },
    18: {
        "adv_model": 0,
        "adv_target": 64,
        "id": 18,
        "is_circle": False,
        "is_square": True,
        "name": "frs9",
    },
    19: {
        "adv_model": 0,
        "adv_target": 33,
        "id": 19,
        "is_circle": False,
        "is_square": True,
        "name": "frs10",
    },
    20: {
        "adv_model": 1,
        "adv_target": 53,
        "id": 20,
        "is_circle": True,
        "is_square": False,
        "name": "rrc1",
    },
    21: {
        "adv_model": 1,
        "adv_target": 27,
        "id": 21,
        "is_circle": True,
        "is_square": False,
        "name": "rrc2",
    },
    22: {
        "adv_model": 1,
        "adv_target": 44,
        "id": 22,
        "is_circle": True,
        "is_square": False,
        "name": "rrc3",
    },
    23: {
        "adv_model": 1,
        "adv_target": 17,
        "id": 23,
        "is_circle": True,
        "is_square": False,
        "name": "rrc4",
    },
    24: {
        "adv_model": 1,
        "adv_target": 85,
        "id": 24,
        "is_circle": True,
        "is_square": False,
        "name": "rrc5",
    },
    25: {
        "adv_model": 1,
        "adv_target": 73,
        "id": 25,
        "is_circle": True,
        "is_square": False,
        "name": "rrc6",
    },
    26: {
        "adv_model": 1,
        "adv_target": 78,
        "id": 26,
        "is_circle": True,
        "is_square": False,
        "name": "rrc7",
    },
    27: {
        "adv_model": 1,
        "adv_target": 1,
        "id": 27,
        "is_circle": True,
        "is_square": False,
        "name": "rrc8",
    },
    28: {
        "adv_model": 1,
        "adv_target": 64,
        "id": 28,
        "is_circle": True,
        "is_square": False,
        "name": "rrc9",
    },
    29: {
        "adv_model": 1,
        "adv_target": 33,
        "id": 29,
        "is_circle": True,
        "is_square": False,
        "name": "rrc10",
    },
    30: {
        "adv_model": 1,
        "adv_target": 53,
        "id": 30,
        "is_circle": False,
        "is_square": True,
        "name": "rrs1",
    },
    31: {
        "adv_model": 1,
        "adv_target": 27,
        "id": 31,
        "is_circle": False,
        "is_square": True,
        "name": "rrs2",
    },
    32: {
        "adv_model": 1,
        "adv_target": 44,
        "id": 32,
        "is_circle": False,
        "is_square": True,
        "name": "rrs3",
    },
    33: {
        "adv_model": 1,
        "adv_target": 17,
        "id": 33,
        "is_circle": False,
        "is_square": True,
        "name": "rrs4",
    },
    34: {
        "adv_model": 1,
        "adv_target": 85,
        "id": 34,
        "is_circle": False,
        "is_square": True,
        "name": "rrs5",
    },
    35: {
        "adv_model": 1,
        "adv_target": 73,
        "id": 35,
        "is_circle": False,
        "is_square": True,
        "name": "rrs6",
    },
    36: {
        "adv_model": 1,
        "adv_target": 78,
        "id": 36,
        "is_circle": False,
        "is_square": True,
        "name": "rrs7",
    },
    37: {
        "adv_model": 1,
        "adv_target": 1,
        "id": 37,
        "is_circle": False,
        "is_square": True,
        "name": "rrs8",
    },
    38: {
        "adv_model": 1,
        "adv_target": 64,
        "id": 38,
        "is_circle": False,
        "is_square": True,
        "name": "rrs9",
    },
    39: {
        "adv_model": 1,
        "adv_target": 33,
        "id": 39,
        "is_circle": False,
        "is_square": True,
        "name": "rrs10",
    },
    40: {
        "adv_model": 2,
        "adv_target": 53,
        "id": 40,
        "is_circle": True,
        "is_square": False,
        "name": "smc1",
    },
    41: {
        "adv_model": 2,
        "adv_target": 27,
        "id": 41,
        "is_circle": True,
        "is_square": False,
        "name": "smc2",
    },
    42: {
        "adv_model": 2,
        "adv_target": 44,
        "id": 42,
        "is_circle": True,
        "is_square": False,
        "name": "smc3",
    },
    43: {
        "adv_model": 2,
        "adv_target": 17,
        "id": 43,
        "is_circle": True,
        "is_square": False,
        "name": "smc4",
    },
    44: {
        "adv_model": 2,
        "adv_target": 85,
        "id": 44,
        "is_circle": True,
        "is_square": False,
        "name": "smc5",
    },
    45: {
        "adv_model": 2,
        "adv_target": 73,
        "id": 45,
        "is_circle": True,
        "is_square": False,
        "name": "smc6",
    },
    46: {
        "adv_model": 2,
        "adv_target": 78,
        "id": 46,
        "is_circle": True,
        "is_square": False,
        "name": "smc7",
    },
    47: {
        "adv_model": 2,
        "adv_target": 1,
        "id": 47,
        "is_circle": True,
        "is_square": False,
        "name": "smc8",
    },
    48: {
        "adv_model": 2,
        "adv_target": 64,
        "id": 48,
        "is_circle": True,
        "is_square": False,
        "name": "smc9",
    },
    49: {
        "adv_model": 2,
        "adv_target": 33,
        "id": 49,
        "is_circle": True,
        "is_square": False,
        "name": "smc10",
    },
    50: {
        "adv_model": 2,
        "adv_target": 53,
        "id": 50,
        "is_circle": False,
        "is_square": True,
        "name": "sms1",
    },
    51: {
        "adv_model": 2,
        "adv_target": 27,
        "id": 51,
        "is_circle": False,
        "is_square": True,
        "name": "sms2",
    },
    52: {
        "adv_model": 2,
        "adv_target": 44,
        "id": 52,
        "is_circle": False,
        "is_square": True,
        "name": "sms3",
    },
    53: {
        "adv_model": 2,
        "adv_target": 17,
        "id": 53,
        "is_circle": False,
        "is_square": True,
        "name": "sms4",
    },
    54: {
        "adv_model": 2,
        "adv_target": 85,
        "id": 54,
        "is_circle": False,
        "is_square": True,
        "name": "sms5",
    },
    55: {
        "adv_model": 2,
        "adv_target": 73,
        "id": 55,
        "is_circle": False,
        "is_square": True,
        "name": "sms6",
    },
    56: {
        "adv_model": 2,
        "adv_target": 78,
        "id": 56,
        "is_circle": False,
        "is_square": True,
        "name": "sms7",
    },
    57: {
        "adv_model": 2,
        "adv_target": 1,
        "id": 57,
        "is_circle": False,
        "is_square": True,
        "name": "sms8",
    },
    58: {
        "adv_model": 2,
        "adv_target": 64,
        "id": 58,
        "is_circle": False,
        "is_square": True,
        "name": "sms9",
    },
    59: {
        "adv_model": 2,
        "adv_target": 33,
        "id": 59,
        "is_circle": False,
        "is_square": True,
        "name": "sms10",
    },
}

def apricot_target_classes(apricot_patches):
    targets = []
    for patch_num, info in apricot_patches.items():
        target = info["adv_target"]
        if target in targets:
            continue
            
        targets.append(target)
        
    return targets
    

def apricot_patch_targeted_AP_per_class(y_list, y_pred_list, ADV_PATCH_MAGIC_NUMBER_LABEL_ID=12, class_list=None):
    """
    Average precision indicating how successfully the APRICOT patch causes the detector
    to predict the targeted class of the patch at the location of the patch. A higher
    value for this metric implies a more successful patch.
    The box associated with the patch is assigned the label of the patch's targeted class.
    Thus, a true positive is the case where the detector predicts the patch's targeted
    class (at a location overlapping the patch). A false positive is the case where the
    detector predicts a non-targeted class at a location overlapping the patch. If the
    detector predicts multiple instances of the target class (that overlap with the patch),
    one of the predictions is considered a true positive and the others are ignored.
    This metric is computed over all evaluation samples, rather than on a per-sample basis.
    It returns a dictionary mapping each class to the average precision (AP) for the class.
    The only classes with potentially nonzero AP's are the classes targeted by the patches
    (see above paragraph).
    """
    
    targets = apricot_target_classes(APRICOT_PATCHES)
    # From https://arxiv.org/abs/1912.08166: use a low IOU since "the patches will sometimes
    # generate many small, overlapping predictions in the region of the attack"
    IOU_THRESHOLD = 0.1

    # Precision will be computed at recall points of 0, 0.1, 0.2, ..., 1
    RECALL_POINTS = np.linspace(0, 1, 11)

    # Converting boxes to a list of dicts (a list for predicted boxes that overlap with the patch,
    # and a separate list for ground truth patch boxes), where each dict corresponds to a box and
    # has the following keys "img_idx", "label", "box", as well as "score" for predicted boxes
    patch_boxes_list = []
    overlappping_pred_boxes_list = []

    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        idx_of_patch = np.where(y["labels"].flatten() == ADV_PATCH_MAGIC_NUMBER_LABEL_ID)[0]
        patch_box = y["boxes"].reshape((-1, 4))[idx_of_patch].flatten()
        patch_id = int(y["patch_id"].flatten()[idx_of_patch])
        patch_target_label = APRICOT_PATCHES[patch_id]["adv_target"]
        patch_box_dict = {
            "img_idx": img_idx,
            "label": patch_target_label,
            "box": patch_box,
        }
        patch_boxes_list.append(patch_box_dict)

        for pred_box_idx in range(y_pred["labels"].size(0)):
            box = y_pred["boxes"][pred_box_idx]
            if _intersection_over_union(box, patch_box) > IOU_THRESHOLD:
                label = y_pred["labels"][pred_box_idx].item()
                score = y_pred["scores"][pred_box_idx]
                pred_box_dict = {
                    "img_idx": img_idx,
                    "label": label,
                    "box": box,
                    "score": score,
                }
                overlappping_pred_boxes_list.append(pred_box_dict)

    # Union of (1) the set of classes targeted by patches and (2) the set of all classes
    # predicted at a location that overlaps the patch in the image
    set_of_class_ids = set([i["label"] for i in patch_boxes_list]) | set(
        [i["label"] for i in overlappping_pred_boxes_list]
    )

    # Initialize dict that will store AP for each class
    average_precisions_by_class = {}

    # Compute AP for each class
    for class_id in tqdm(set_of_class_ids):
        # Build lists that contain all the predicted and patch boxes with a
        # label of class_id
        class_predicted_boxes = []
        class_patch_boxes = []
        for pred_box in overlappping_pred_boxes_list:
            if pred_box["label"] == class_id:
                class_predicted_boxes.append(pred_box)
        for patch_box in patch_boxes_list:
            if patch_box["label"] == class_id:
                class_patch_boxes.append(patch_box)

        # Determine how many patch boxes (of class_id) there are in each image
        num_patch_boxes_per_img = Counter([gt["img_idx"] for gt in class_patch_boxes])

        # Initialize dict where we'll keep track of whether a patch box has been matched to a
        # prediction yet. This is necessary because if multiple predicted boxes of class_id
        # overlap with a patch box, only one of the predicted boxes can be considered a
        # true positive. The rest will be ignored
        img_idx_to_patchboxismatched_array = {}
        for img_idx, num_patch_boxes in num_patch_boxes_per_img.items():
            img_idx_to_patchboxismatched_array[img_idx] = np.zeros(num_patch_boxes)

        # Sort all predicted boxes (of class_id) by descending confidence
        class_predicted_boxes.sort(key=lambda x: x["score"], reverse=True)

        # Initialize list. Once filled in, true_positives[i] indicates (with a 1 or 0)
        # whether the ith predicted box (of class_id) is a true positive or false positive
        is_true_positive = []

        # Iterating over all predicted boxes of class_id
        for pred_idx, pred_box in enumerate(class_predicted_boxes):
            # Only compare patch boxes from the same image as the predicted box
            patch_boxes_from_same_img = [
                patch_box
                for patch_box in class_patch_boxes
                if patch_box["img_idx"] == pred_box["img_idx"]
            ]

            # If there are no patch boxes in the predicted box's image that target the predicted class
            if len(patch_boxes_from_same_img) == 0:
                is_true_positive.append(0)
                continue

            # Iterate over all patch boxes (of class_id) from the same image as the predicted box,
            # determining which patch box has the highest iou with the predicted box.
            highest_iou = 0
            for patch_idx, patch_box in enumerate(patch_boxes_from_same_img):
                iou = _intersection_over_union(pred_box["box"], patch_box["box"])
                if iou >= highest_iou:
                    highest_iou = iou
                    highest_iou_patch_idx = patch_idx

            # If the patch box has not yet been covered
            if (
                img_idx_to_patchboxismatched_array[pred_box["img_idx"]][
                    highest_iou_patch_idx
                ]
                == 0
            ):
                is_true_positive.append(1)

                # Record that we've now covered this patch box. Any subsequent
                # pred boxes that overlap with it are ignored
                img_idx_to_patchboxismatched_array[pred_box["img_idx"]][
                    highest_iou_patch_idx
                ] = 1
            else:
                # This patch box was already covered previously (i.e a different predicted
                # box was deemed a true positive after overlapping with this patch box).
                # The predicted box is thus ignored.
                continue

        # Cumulative sums of false/true positives across all predictions which were sorted by
        # descending confidence
        tp_cumulative_sum = np.cumsum(is_true_positive)
        fp_cumulative_sum = np.cumsum([not i for i in is_true_positive])

        # Total number of patch boxes with a label of class_id
        total_patch_boxes = len(class_patch_boxes)

        recalls = tp_cumulative_sum / (total_patch_boxes + 1e-6)
        precisions = tp_cumulative_sum / (tp_cumulative_sum + fp_cumulative_sum + 1e-6)

        interpolated_precisions = np.zeros(len(RECALL_POINTS))
        # Interpolate the precision at each recall level by taking the max precision for which
        # the corresponding recall exceeds the recall point
        # See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf
        for i, recall_point in enumerate(RECALL_POINTS):
            precisions_points = precisions[np.where(recalls >= recall_point)]
            # If there's no cutoff at which the recall > recall_point
            if len(precisions_points) == 0:
                interpolated_precisions[i] = 0
            else:
                interpolated_precisions[i] = max(precisions_points)

        # Compute mean precision across the different recall levels
        average_precision = interpolated_precisions.mean()
        average_precisions_by_class[int(class_id)] = np.around(
            average_precision, decimals=2
        )

    print(average_precisions_by_class)

    print("printing mAP for only targeted class:")
    av_map = 0.0
    num_target_class = 0

    for class_id, ap in average_precisions_by_class.items():
        if class_id in targets:
            print("class " + str(class_id) + " : " + str(ap))
            av_map += ap
            num_target_class += 1
    print("averaged target map: " + str(av_map))


class ComputeMetrics(Meter):
    def __init__(self, train_env, metrics_info):
        self.metrics_info = metrics_info
        self.y_list = []
        self.y_pred_list = []

    def update(self, preds, ground_truth):
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in preds]
        self.y_pred_list.extend(outputs)
        self.y_list.extend(ground_truth.to("cpu"))
        return 0.0

    def finalize(self):
        metrics_result = {}
        for metric_name, metric_params in self.metrics_info.items():
            # CARLA object detection datasets contains class labels 1-4, with class 4 representing
            # the green screen/patch itself, which should not be treated as an object class.
            class_list = [1, 2, 3] if "carla" in metric_name else None
            metric_func = METRIC_DICT[metric_name]
            metric_args = inspect.signature(metric_func).parameters.keys()
            if 'class_list' in metric_args:
                metrics_result[metric_name] = metric_func(self.y_list, self.y_pred_list, 
                                                          class_list=class_list, **args_to_paramdict(metric_params))
            else:
                metrics_result[metric_name] = metric_func(self.y_list, self.y_pred_list, 
                                                          **args_to_paramdict(metric_params))
        return metrics_result


def create_metric(train_env, info):
    """
    Creates a metric with a name and a dict of arguments
    """
    if info is None:
        return SimpleAccuracy(train_env)
            
    for metric_name, metric_params in info.items():
        assert metric_name in METRIC_DICT, "Undefined metric: " + metric_name
        
    return ComputeMetrics(train_env, info)


METRIC_DICT = {
    'skip': Meter,
    'accuracy': SimpleAccuracy,
    'mean_average_precision': mAPMeter,

    'armory_APRICOT_per_class_average_precision': apricot_patch_targeted_AP_per_class,
    'armory_per_class_average_precision': object_detection_AP_per_class,
    'armory_mean_average_precision': object_detection_mAP,

    'armory_object_detection_true_positive_rate': object_detection_true_positive_rate,
    'armory_object_detection_misclassification_rate': object_detection_misclassification_rate,
    'armory_object_detection_disappearance_rate': object_detection_disappearance_rate,
    'armory_object_detection_hallucinations_per_image': object_detection_hallucinations_per_image,
}

METRIC_DICT['armory_carla_od_mAP'] = METRIC_DICT["armory_mean_average_precision"]
METRIC_DICT['armory_carla_od_AP_per_class'] = METRIC_DICT["armory_per_class_average_precision"]
METRIC_DICT['armory_carla_od_true_positive_rate'] = METRIC_DICT["armory_object_detection_true_positive_rate"]
METRIC_DICT['armory_carla_od_misclassification_rate'] = METRIC_DICT["armory_object_detection_misclassification_rate"]
METRIC_DICT['armory_carla_od_disappearance_rate'] = METRIC_DICT["armory_object_detection_disappearance_rate"]
METRIC_DICT['armory_carla_od_hallucinations_per_image'] = METRIC_DICT["armory_object_detection_hallucinations_per_image"]
