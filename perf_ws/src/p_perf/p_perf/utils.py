import os
from contextlib import redirect_stdout
import random
from pyquaternion import Quaternion
import numpy as np
from nuscenes.utils.data_classes import Box as NuScenesBox
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet.datasets.coco import CocoDataset

from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.utils import category_to_detection_name

from p_perf.config.constant import classes

VIDEO_DICT = {
        'rainy_night_city': ['02d478d1-e6811391', '024dd592-94359ff1', '00a04f65-8c891f94'],
        'rainy_daytime_city': ['020cc8f8-8b679d0b', 'b2036451-aa924fd1', '03112119-0aafd3ad'],
        'clear_night_city': ['0001542f-ec815219', '000d35d3-41990aa4', '00134776-9123d227'],
        'clear_day_city': ['00067cfb-5443fe39', '00067cfb-f1b91e3c', '000e0252-8523a4a9'],
        'rainy_night_highway': ['028584e7-6a14163e', '035268c2-5cf95581', 'b20eae11-18cd8ca2'],
        'rainy_daytime_highway': ['b1e1a7b8-b397c445', 'b1e1a7b8-a7426a97', '012fdff1-9d1d0d1d'],
        'clear_night_highway': ['00268999-0b20ef00', '0059f17f-f0882eef', '007b11e5-c22ddae8'],
        'clear_daytime_highway': ['002d290d-89f4e5c0', '004071a4-4e8a363a', '0049e5b8-725e21a0']
    }    

# GENERAL HELPER FUNCTION
def list_filenames(directory, ending):
    """List all files in a directory with a specific file extension, sorted alphabetically."""
    file_names = []
    for entry in sorted(os.listdir(directory)):  # 🔧 sort added
        entry_path = os.path.join(directory, entry)
        if os.path.isfile(entry_path) and entry.endswith(f'.{ending}'):
            file_names.append(entry_path)
    return file_names
    
def suppress_function_output(func, *args, **kwargs):
    """Suppress the output of a function that prints to stdout."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):  # Redirect stdout to devnull (suppresses print output)
            result = func(*args, **kwargs)
    return result


# HELPER FUNCTION FOR EVALUATION PIPELINE ==> LIDAR

def lidar_output_to_nusc_box(
        detection, token, score_thresh=0.5):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (Det3DDataSample): Detection results.

            - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], np.ndarray or None]: List of standard
        NuScenesBoxes and attribute labels.
    """
    bbox3d = detection.bboxes_3d.to('cpu')
    scores = detection.scores_3d.cpu().numpy()
    labels = detection.labels_3d.cpu().numpy()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if isinstance(bbox3d, LiDARInstance3DBoxes):
        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            # filter out bbox with low confidence score
            if scores[i] < score_thresh:
                continue
            quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity,
                token=token)
            box_list.append(box)
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes '
            'to standard NuScenesBoxes.')

    return box_list

def lidar_nusc_box_to_global(
        nusc,
        sample_data_token: str,
        boxes):
    """
    Convert predicted NuScenesBoxes from LiDAR to global coordinates using sample_data_token.

    Args:
        nusc: NuScene instance
        sample_data_token (str): Token for the sample_data (e.g. from file name).
        boxes (List[NuScenesBox]): Predicted bounding boxes in LiDAR coordinates.

    Returns:
        List[NuScenesBox]: Boxes transformed into global coordinates and filtered.
    """
    # Step 1: Get sensor calibration and ego poses
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Construct transformation matrices
    lidar2ego_rot = Quaternion(cs_record['rotation']).rotation_matrix
    lidar2ego_trans = np.array(cs_record['translation'])

    ego2global_rot = Quaternion(pose_record['rotation']).rotation_matrix
    ego2global_trans = np.array(pose_record['translation'])

    # Transform boxes
    box_list = []
    for box in boxes:
        # LiDAR -> Ego
        box.rotate(Quaternion(matrix=lidar2ego_rot))
        box.translate(lidar2ego_trans)

        # Ego -> Global
        box.rotate(Quaternion(matrix=ego2global_rot))
        box.translate(ego2global_trans)

        box_list.append(box)

    return box_list


# HELPER_FUNCTION FOR IMAGE EVALUATION PIPELINE
def image_output_to_coco(detection, image_id, score_thresh=0.5):
    """
    Convert image-based detection results to COCO-style format.

    Args:
        pred_instance (InstanceData): Contains `bboxes`, `labels`, and `scores`.
        image_id (int or str): Unique image ID in the COCO dataset.
        category_id_map (dict): Mapping from internal label IDs to COCO category IDs.
        score_thresh (float): Minimum score threshold to keep predictions.

    Returns:
        List[dict]: List of COCO-style prediction dicts for this image.
    """
    bboxes = detection.bboxes.cpu().numpy()
    labels = detection.labels.cpu().numpy()
    labels = mmdet_to_nusc_labels(labels)
    scores = detection.scores.cpu().numpy()

    coco_results = []

    for bbox, label, score in zip(bboxes, labels, scores):
        # Apply score threshold
        if score < score_thresh:
            continue
        # Convert [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = bbox
        coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        result = {
            "image_id": image_id,
            "category_id": int(label),
            "bbox": coco_bbox,
            "score": float(score)
        }
        coco_results.append(result)

    return coco_results


def mmdet_to_nusc_labels(mmdet_labels):
    # MMDetection COCO-style class names
    mmdet_classes = CocoDataset.METAINFO['classes']

    # Direct class name mappings from COCO to NuScenes (no approximation)
    coco_to_nusc = {
        'car': 'car',
        'truck': 'truck',
        'bus': 'bus',
        'bicycle': 'bicycle',
        'motorcycle': 'motorcycle',
        'person': 'pedestrian',
    }

    # Convert: MMDet label index → NuScenes class index
    result = []
    for label_id in mmdet_labels:
        class_name = mmdet_classes[label_id]
        nusc_name = coco_to_nusc.get(class_name)
        if nusc_name in classes:
            nusc_index = classes.index(nusc_name)
            result.append(nusc_index)

    return np.array(result)


# HELPER FUNCTION USED BY BOTH LIDAR AND IMAGE EVALUATION PIPELINE

def interpolate_gt(nusc, sample_data_token: str, box_cls=DetectionBox):
        """
        Generate interpolated or fallback ground truth boxes for a given sample_data_token.
        Falls back to prev or next keyframe if interpolation is not possible.
        """
        sd = nusc.get('sample_data', sample_data_token)
        timestamp = sd['timestamp']

        box_list = []
        instance_tokens = []

        if sd['is_key_frame']:
            sample = nusc.get('sample', sd['sample_token'])
            annos = [nusc.get('sample_annotation', tok) for tok in sample['anns']]

            for a in annos:
                detection_name = category_to_detection_name(a['category_name'])
                if detection_name is None:
                    continue

                box_list.append(box_cls(
                    sample_token=sample_data_token,
                    translation=a['translation'],
                    size=a['size'],
                    rotation=a['rotation'],
                    velocity=a.get('velocity', [0.0, 0.0]),
                    detection_name=detection_name,
                    attribute_name=''  # optional
                ))
                instance_tokens.append(a['instance_token'])

            return box_list, instance_tokens

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

                detection_name = category_to_detection_name(a0['category_name'])
                if detection_name is None:
                    continue

                box_list.append(box_cls(
                    sample_token=sample_data_token,
                    translation=center.tolist(),
                    size=size.tolist(),
                    rotation=rotation.elements.tolist(),
                    velocity=velocity.tolist(),
                    detection_name=detection_name,
                    attribute_name=''
                ))
                instance_tokens.append(inst)

            return box_list, instance_tokens

        # Fallback case
        fallback_frame = prev_keyframe or next_keyframe
        fallback_sample = nusc.get('sample', fallback_frame['sample_token'])
        annos = [nusc.get('sample_annotation', tok) for tok in fallback_sample['anns']]

        for a in annos:
            detection_name = category_to_detection_name(a['category_name'])
            if detection_name is None:
                continue

            box_list.append(box_cls(
                sample_token=sample_data_token,
                translation=a['translation'],
                size=a['size'],
                rotation=a['rotation'],
                velocity=a.get('velocity', [0.0, 0.0]),
                detection_name=detection_name,
                attribute_name=''
            ))
            instance_tokens.append(a['instance_token'])

        return box_list, instance_tokens






