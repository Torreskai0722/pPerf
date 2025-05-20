from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData
from pyquaternion import Quaternion
import numpy as np
from typing import List
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.algo import calc_ap
import os
import json

json_path = '/mmdetection3d_ros2/data/nuscenes/v1.0-mini/sample_data.json'
classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

dist_ths = [0.25, 0.5, 0.75, 1.0]


def convert_det_to_nusc_format(det, token):
    """
    Convert full detection result to list of NuScenes-format dicts.
    Expects:
        det['predictions'][0]['bboxes_3d']: List of 9D boxes [x, y, z, dx, dy, dz, yaw, vx, vy]
        det['predictions'][0]['scores_3d']: Score list
        det['predictions'][0]['labels_3d']: Label list
    Returns:
        List of dicts in NuScenes format with translation, size, rotation, velocity, detection_name, detection_score.
    """
    preds = det['predictions'][0]
    bboxes = preds['bboxes_3d']
    scores = preds['scores_3d']
    labels = preds['labels_3d']

    results = []
    for i in range(len(bboxes)):
        x, y, z, dx, dy, dz, yaw, vx, vy = bboxes[i]
        quat = Quaternion(axis=[0, 0, 1], radians=yaw)
        result = {
            "sample_token": token,
            "translation": [x, y, z],
            "size": [dy, dx, dz],  # NuScenes expects [w, l, h]
            "rotation": [quat.w, quat.x, quat.y, quat.z],
            "velocity": [vx, vy],
            "detection_name": labels[i],
            "detection_score": scores[i],
            "attribute_name": ""
        }
        results.append(result)

    return results

def get_token_from_filename(json_path, target_filename):
    """
    Given the path to a sample_data JSON and a file name, returns the token corresponding to that file.
    
    Args:
        json_path (str): Path to the sample_data JSON file.
        target_filename (str): File name to search for, e.g., 'n015-2018-08-01-15-16-36+0800__LIDAR_TOP__1533113235880478.bin'
    
    Returns:
        str: The corresponding token, or None if not found.
    """
    with open(json_path, 'r') as f:
        sample_data = json.load(f)
    
    # Match only the base file name
    target_filename = os.path.basename(target_filename)

    for item in sample_data:
        if os.path.basename(item['filename']) == target_filename:
            return item['token']
    
    return None  # If not found

def interpolate_gt(nusc, sample_data_token: str, box_cls=DetectionBox) -> List[DetectionBox]:
    """
    Generate interpolated or fallback ground truth boxes for a given sample_data_token.
    Falls back to prev or next keyframe if interpolation is not possible.
    """
    sd = nusc.get('sample_data', sample_data_token)
    timestamp = sd['timestamp']

    if sd['is_key_frame']:
        sample = nusc.get('sample', sd['sample_token'])
        annos = [nusc.get('sample_annotation', tok) for tok in sample['anns']]

        return [box_cls(
            sample_token=sample_data_token,
            translation=a['translation'],
            size=a['size'],
            rotation=a['rotation'],
            velocity=a.get('velocity', [0.0, 0.0]),
            detection_name=category_to_detection_name(a['category_name']),
            attribute_name=''  # optional
        ) for a in annos]

    # Walk backward to find previous keyframe
    prev_sd_token = sd['prev']
    prev_keyframe = None
    while prev_sd_token:
        prev_sd = nusc.get('sample_data', prev_sd_token)
        if prev_sd['is_key_frame']:
            prev_keyframe = prev_sd
            break
        prev_sd_token = prev_sd['prev']

    # Walk forward to find next keyframe
    next_sd_token = sd['next']
    next_keyframe = None
    while next_sd_token:
        next_sd = nusc.get('sample_data', next_sd_token)
        if next_sd['is_key_frame']:
            next_keyframe = next_sd
            break
        next_sd_token = next_sd['next']

    if prev_keyframe and next_keyframe:
        # Interpolation case
        t0, t1 = prev_keyframe['timestamp'], next_keyframe['timestamp']
        alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

        prev_sample = nusc.get('sample', prev_keyframe['sample_token'])
        next_sample = nusc.get('sample', next_keyframe['sample_token'])

        prev_annos = [nusc.get('sample_annotation', tok) for tok in prev_sample['anns']]
        next_annos = [nusc.get('sample_annotation', tok) for tok in next_sample['anns']]

        prev_map = {a['instance_token']: a for a in prev_annos}
        next_map = {a['instance_token']: a for a in next_annos}

        common_instances = set(prev_map.keys()) & set(next_map.keys())
        interpolated_boxes = []

        for inst in common_instances:
            a0, a1 = prev_map[inst], next_map[inst]

            t0 = np.array(a0['translation'])
            t1 = np.array(a1['translation'])
            center = (1 - alpha) * t0 + alpha * t1

            s0 = np.array(a0['size'])
            s1 = np.array(a1['size'])
            size = (1 - alpha) * s0 + alpha * s1

            q0 = Quaternion(a0['rotation'])
            q1 = Quaternion(a1['rotation'])
            rotation = Quaternion.slerp(q0, q1, amount=alpha)

            v0 = np.array(a0.get('velocity', [0, 0]))
            v1 = np.array(a1.get('velocity', [0, 0]))
            velocity = (1 - alpha) * v0 + alpha * v1

            if(category_to_detection_name(a0['category_name']) == None):
                continue

            interpolated_boxes.append(box_cls(
                sample_token=sample_data_token,
                translation=center.tolist(),
                size=size.tolist(),
                rotation=rotation.elements.tolist(),
                velocity=velocity.tolist(),
                detection_name=category_to_detection_name(a0['category_name']),
                attribute_name=''  # optional
            ))

        return interpolated_boxes

    # Fallback to one keyframe
    fallback_frame = prev_keyframe or next_keyframe
    fallback_sample = nusc.get('sample', fallback_frame['sample_token'])
    annos = [nusc.get('sample_annotation', tok) for tok in fallback_sample['anns']]
    fallback_boxes = []

    for a in annos:
        fallback_boxes.append(box_cls(
            sample_token=sample_data_token,
            translation=a['translation'],
            size=a['size'],
            rotation=a['rotation'],
            velocity=a.get('velocity', [0.0, 0.0]),
            detection_name=category_to_detection_name(a['category_name']),
            attribute_name=''  # optional
        ))

    return fallback_boxes



def accumulate(nusc,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:

    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    # Sort by confidence
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp, fp, conf = [], [], []
    match_data = {k: [] for k in ['trans_err', 'vel_err', 'scale_err', 'orient_err', 'attr_err', 'conf']}
    taken = set()
    interpolated_cache = {}

    # Precompute interpolated GTs per sample_data_token
    for token in pred_boxes.sample_tokens:
        interpolated_cache[token] = interpolate_gt(nusc, token)  # Your interpolate function must be per token

    # Compute total GTs for this class (avoid double counting)
    npos = sum(
        sum(1 for box in boxes if box.detection_name == class_name)
        for boxes in interpolated_cache.values()
    )

    if verbose:
        print(f"GT count for class {class_name}: {npos}")

    # Matching
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        token = pred_box.sample_token
        interpolated_gts = interpolated_cache[token]

        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(interpolated_gts):
            if gt_box.detection_name == class_name and (token, gt_idx) not in taken:
                this_dist = dist_fcn(gt_box, pred_box)
                if this_dist < min_dist:
                    min_dist = this_dist
                    match_gt_idx = gt_idx

        is_match = min_dist < dist_th

        if is_match:
            taken.add((token, match_gt_idx))
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            gt_box_match = interpolated_gts[match_gt_idx]
            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period))
            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)
        else:
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    for key in match_data:
        if key == 'conf':
            continue
        tmp = cummean(np.array(match_data[key]))
        match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    return DetectionMetricData(
        recall=rec,
        precision=prec,
        confidence=conf,
        trans_err=match_data['trans_err'],
        vel_err=match_data['vel_err'],
        scale_err=match_data['scale_err'],
        orient_err=match_data['orient_err'],
        attr_err=match_data['attr_err']
    )


def load_prediction_of_sample_tokens(result_path: str,
                                     sample_tokens, all, verbose: bool = False, box_cls = DetectionBox):
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)

    assert isinstance(data['results'], dict), 'Error: results must be a dict.'

     # Filter by sample tokens.
    if all:
        results_of_split = data['results']
    else:
        results_of_split = {sample_token: data['results'][sample_token] for sample_token in sample_tokens}

    # Deserialize results and get meta data.
    boxes_of_split = EvalBoxes.deserialize(results_of_split, box_cls)
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(boxes_of_split.sample_tokens)))

    return boxes_of_split

prediction_path = '/mmdetection3d_ros2/perf_ws/results_nusc.json'
pred_boxes = load_prediction_of_sample_tokens(prediction_path, [], True, verbose=True)

metrics = {}
metric_data_list = DetectionMetricDataList()
for class_name in classes:
    for dist_th in dist_ths:
        md = accumulate(nusc, pred_boxes, class_name, center_distance, dist_th)
        metric_data_list.set(class_name, dist_th, md)
        ap = calc_ap(md, 0.1, 0.1)
        metrics[(class_name, dist_th)] = ap

print(metrics)

