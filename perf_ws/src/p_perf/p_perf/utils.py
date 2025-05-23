import os
from contextlib import redirect_stdout
from pyquaternion import Quaternion
import numpy as np


from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.utils import category_to_detection_name
import cv2
from p_perf.config.constant import lidar_classes, image_classes

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


# HELPER FUNCTION USED BY BOTH LIDAR AND IMAGE EVALUATION PIPELINE

def interpolate_gt(nusc, sample_data_token: str, visibility: bool, visibilities):
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
        if visibility:
            annos = [anno for anno in annos if (anno['visibility_token'] in visibilities)]

        for a in annos:
            detection_name = category_to_detection_name(a['category_name'])
            if detection_name is None:
                continue

            box_list.append(DetectionBox(
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
        if visibility:
            prev_annos = [prev_anno for prev_anno in prev_annos if (prev_anno['visibility_token'] in visibilities)]
            next_annos = [next_anno for next_anno in next_annos if (next_anno['visibility_token'] in visibilities)]

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

            box_list.append(DetectionBox(
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
    if visibility:
        annos = [anno for anno in annos if (anno['visibility_token'] in visibilities)]

    for a in annos:
        detection_name = category_to_detection_name(a['category_name'])
        if detection_name is None:
            continue

        box_list.append(DetectionBox(
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







def draw_coco_bboxes_from_path(image_path, coco_bboxes, labels=None, scores=None, class_names=None, color=(0, 255, 0), thickness=2):
    """
    Load an image from file and draw COCO-format bounding boxes on it.

    Args:
        image_path (str): Path to the image file.
        coco_bboxes (List[List[float]]): List of [x, y, w, h] boxes.
        labels (List[int], optional): COCO category IDs.
        scores (List[float], optional): Confidence scores.
        class_names (Dict[int, str], optional): COCO category_id to name mapping.
        color (Tuple[int, int, int], optional): Box color in BGR.
        thickness (int): Box line thickness.

    Returns:
        np.ndarray: Image with boxes drawn.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for i, box in enumerate(coco_bboxes):
        x, y, w, h = map(int, box)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

        label_text = ""
        if labels is not None:
            class_id = labels[i]
            label_text += class_names[class_id] if class_names else f"class {class_id}"
        if scores is not None:
            label_text += f" {scores[i]:.2f}"

        if label_text:
            cv2.putText(image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Show image before returning
    cv2.imshow("COCO Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






