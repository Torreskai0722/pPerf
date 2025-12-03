import os
from contextlib import redirect_stdout
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import json
import os
import cv2
import torch

from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.utils import category_to_detection_name
import cv2
from p_perf.config.constant import nus_lidar_classes, image_classes

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

def interpolate_gt(nusc, sd_token: str, sd_offset_token: str, visibility: bool, visibilities):
    """
    Generate interpolated or fallback ground truth boxes for a given sample_data_token.
    Falls back to prev or next keyframe if interpolation is not possible.

    Args:
        streaming_gt: if set to true
    """
    sd = nusc.get('sample_data', sd_offset_token)
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
                sample_token=sd_token,
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
                sample_token=sd_token,
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
            sample_token=sd_token,
            translation=a['translation'],
            size=a['size'],
            rotation=a['rotation'],
            velocity=a.get('velocity', [0.0, 0.0]),
            detection_name=detection_name,
            attribute_name=''
        ))
        instance_tokens.append(a['instance_token'])

    return box_list, instance_tokens



def get_offset_sd_token(nusc, start_token: str, model_name: str, model_type: str, delay_csv_path: str) -> str:
    """
    Given a sample_data token, sensor type, and delay CSV, compute the nearest sample token
    based on the processing time from the CSV.

    Args:
        nusc: NuScenes instance
        start_token: Current sample_data token (e.g. from a frame just processed)
        sensor_type: 'image' or 'lidar'
        delay_csv_path: Path to the CSV file containing process_time column

    Returns:
        A future sample_data token that is closest to when processing finishes
    """

    # Load delay CSV
    df = pd.read_csv(delay_csv_path)

    # Find the matching row
    row = df[(df['input_token'] == start_token) & (df['model_name'] == model_name)]
    if row.empty:
        raise ValueError(f"No matching row for token {start_token} and sensor type {model_name}")

    # Extract process_time in seconds
    process_time = float(row['e2e_delay'].values[0])

    # Determine frame interval
    freq = 12 if model_type == 'image' else 20
    frame_interval = 1.0 / freq

    # Compute how many frames ahead
    frame_offset = round(process_time / frame_interval)

    # Walk forward through the sample_data chain
    token = start_token
    for _ in range(frame_offset):
        sd = nusc.get('sample_data', token)
        if not sd['next']:
            break
        token = sd['next']

    return token


def load_sweep_sd(nusc, scene, sensor_channel='CAM_FRONT'):
    """
    Get all sample_data tokens (including intermediate sweep frames) for a given channel from a scene.

    Args:
        nusc: NuScenes instance
        scene: scene dictionary from nusc.scene
        sensor_channel: e.g., 'CAM_FRONT'

    Returns:
        List of sample_data tokens (str) ordered by time, including all sweeps.
    """
    # Get the first sample token
    first_sample_token = scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    
    # Get the initial sample_data token for the desired sensor channel
    current_sd_token = None
    for sd_token in first_sample['data'].values():
        sd = nusc.get('sample_data', sd_token)
        if sd['channel'] == sensor_channel:
            current_sd_token = sd_token
            break

    if current_sd_token is None:
        raise ValueError(f"No sample_data found for channel {sensor_channel} in first sample")

    # Traverse through sample_data's 'next' field to get all sweep frames
    sweep_tokens = []
    while current_sd_token:
        sd = nusc.get('sample_data', current_sd_token)
        if sd['channel'] == sensor_channel:
            sweep_tokens.append(current_sd_token)
        current_sd_token = sd['next'] if sd['next'] else None

    return sweep_tokens

def get_paths_from_sd(nusc, sd_tokens):
    """
    Given a list of sample_data tokens, return the corresponding filenames.
    """
    filenames = []
    for token in sd_tokens:
        sd = nusc.get('sample_data', token)
        filenames.append(os.path.join(nusc.dataroot, sd['filename']))
    return filenames


def convert_to_kitti_ros(points_array):
    """
    Convert points from nuScenes format to KITTI format.
    
    Args:
        points_array: numpy structured array with fields (x, y, z, intensity, ring)
        
    Returns:
        numpy structured array in KITTI format with 4 channels (x, y, z, intensity)
    """
    # Create output array directly with transformed coordinates
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32)
    ])
    
    kitti_points = np.empty(len(points_array), dtype=dtype)
    
    # Transform coordinates and normalize intensity in one go
    kitti_points['x'] = points_array['y']  # y_nusc -> x_kitti
    kitti_points['y'] = -points_array['x']  # -x_nusc -> y_kitti
    kitti_points['z'] = points_array['z']  # z_nusc -> z_kitti
    
    # Normalize intensity
    intensity = points_array['intensity']
    intensity_min = intensity.min()
    intensity_range = max(1e-5, intensity.ptp())
    kitti_points['intensity'] = (intensity - intensity_min) / intensity_range
    
    return kitti_points


def convert_to_kitti(nusc, lidar_token):
    # Load metadata
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = os.path.join(nusc.dataroot, lidar_data["filename"])
    scan = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))[:, :4]  # [x, y, z, intensity]
    
    # Normalize intensity
    scan[:, 3] = (scan[:, 3] - scan[:, 3].min()) / max(1e-5, scan[:, 3].ptp())
    
    # Transform from nuScenes to KITTI
    x_nusc = scan[:, 0]
    y_nusc = scan[:, 1]
    z_nusc = scan[:, 2]
    intensity = scan[:, 3]

    x_kitti = y_nusc
    y_kitti = -x_nusc
    z_kitti = z_nusc

    scan_kitti = np.stack((x_kitti, y_kitti, z_kitti, intensity), axis=1)

    # KITTI format uses an extra dummy column (e.g., reflectivity or ring index)
    zeros_col = np.zeros((scan_kitti.shape[0], 1), dtype=np.float32)
    scan_kitti = np.hstack((scan_kitti, zeros_col))
    return scan_kitti


import json
import os
import cv2

def visualize_coco_predictions(nusc_token, pred_json_path, gt_json_path, image_dir, index, score_thresh=0.3):
    # Load prediction and GT files
    with open(pred_json_path) as f:
        preds = json.load(f)

    with open(gt_json_path) as f:
        gt = json.load(f)

    # Build mapping from token → image_id
    token_to_image = {img['token']: img for img in gt['images']}
    if nusc_token not in token_to_image:
        raise ValueError(f"Token {nusc_token} not found in ground truth JSON.")

    image_info = token_to_image[nusc_token]
    image_id = image_info['id']
    image_path = os.path.join(image_dir, image_info['file_name'])

    # Map category ID to name
    id_to_name = {cat['id']: cat['name'] for cat in gt['categories']}

    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Draw ground truth boxes (green)
    for ann in gt['annotations']:
        if ann['image_id'] != image_id:
            continue
        x, y, w, h = map(int, ann['bbox'])
        label = id_to_name.get(ann['category_id'], str(ann['category_id']))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"GT: {label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw predicted boxes (red)
    for pred in preds:
        if pred['image_id'] != image_id or pred['score'] < score_thresh:
            continue
        x, y, w, h = map(int, pred['bbox'])
        label = id_to_name.get(pred['category_id'], str(pred['category_id']))
        score = pred['score']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, f"Pred: {label} {score:.2f}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show image
    cv2.imwrite(f'{index}.png', image)



def visualize_lidar_predictions(nusc, token, interpolate, pred_json_path, delay_path):
    import json
    import numpy as np
    import open3d as o3d
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    def create_open3d_box(box, color=[1, 0, 0]):
        corners = box.corners().T
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        colors = [color for _ in lines]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    # Load point cloud
    sd = nusc.get('sample_data', token)
    lidar_path = nusc.get_sample_data_path(token)
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]

    # Get transforms
    cs_record = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])

    lidar2ego_trans = np.array(cs_record['translation'])
    lidar2ego_rot = Quaternion(cs_record['rotation']).rotation_matrix

    ego2global_trans = np.array(ego_pose['translation'])
    ego2global_rot = Quaternion(ego_pose['rotation']).rotation_matrix

    # Ground Truth Boxes (in global)
    sd_offset = get_offset_sd_token(nusc, token, 'lidar', delay_path)
    if interpolate:
        gt_boxes_raw, _ = interpolate_gt(nusc, token, sd_offset, False, [])
    else:
        gt_boxes_raw = []
        sample = nusc.get('sample', sd['sample_token'])
        annos = [nusc.get('sample_annotation', tok) for tok in sample['anns']]

        for a in annos:
            detection_name = category_to_detection_name(a['category_name'])
            if detection_name is None:
                continue

            gt_boxes_raw.append(DetectionBox(
                sample_token=token,
                translation=a['translation'],
                size=a['size'],
                rotation=a['rotation'],
                velocity=a.get('velocity', [0.0, 0.0]),
                detection_name=detection_name,
                attribute_name=''  # optional
            ))

    gt_boxes = []

    for det_box in gt_boxes_raw:
        box = Box(
            center=det_box.translation,
            size=det_box.size,
            orientation=Quaternion(det_box.rotation),
            name=det_box.detection_name,
            score=det_box.detection_score
        )

        # Global → Ego
        box.translate(-ego2global_trans)
        box.rotate(Quaternion(matrix=ego2global_rot).inverse)

        # Ego → LiDAR
        box.translate(-lidar2ego_trans)
        box.rotate(Quaternion(matrix=lidar2ego_rot).inverse)

        gt_boxes.append(box)

    # Load prediction boxes
    with open(pred_json_path) as f:
        pred_json = json.load(f)

    pred_boxes = []
    if token in pred_json["results"]:
        for item in pred_json["results"][token]:
            center = item['translation']
            size = item['size']
            rot = item['rotation']  # [w, x, y, z]
            box = Box(center=center, size=size, orientation=Quaternion(rot))

            # Global -> Ego -> LiDAR
            box.translate(-ego2global_trans)
            box.rotate(Quaternion(matrix=ego2global_rot).inverse)
            box.translate(-lidar2ego_trans)
            box.rotate(Quaternion(matrix=lidar2ego_rot).inverse)

            pred_boxes.append(box)

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    vis.add_geometry(pcd)

    # GT = red, Pred = green
    for box in gt_boxes:
        vis.add_geometry(create_open3d_box(box, color=[1, 0, 0]))
    for box in pred_boxes:
        vis.add_geometry(create_open3d_box(box, color=[0, 1, 0]))

    vis.run()
    vis.destroy_window()


def build_channel_timestamp_token_map(nusc, scene_token: str, sensor_channel: str) -> dict:
    """
    Build a dictionary mapping timestamp (in seconds) to sample_data tokens for a given sensor channel.

    Args:
        nusc (NuScenes): Initialized NuScenes instance.
        scene_token (str): Token of the scene.
        sensor_channel (str): e.g., 'CAM_FRONT', 'LIDAR_TOP'

    Returns:
        dict: {timestamp_sec: sample_data_token}
    """
    ts_token_map = {}

    scene = nusc.get('scene', scene_token)
    first_sample_token = scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)

    current_sd_token = first_sample['data'].get(sensor_channel)
    if current_sd_token is None:
        raise ValueError(f"No sample_data found for channel '{sensor_channel}' in first sample")

    # Traverse the sample_data chain and build the dictionary
    while current_sd_token:
        sd = nusc.get('sample_data', current_sd_token)
        if sd['channel'] == sensor_channel:
            timestamp_sec = sd['timestamp'] / 1e6  # microseconds to seconds
            ts_token_map[timestamp_sec] = current_sd_token
        current_sd_token = sd['next']

    return ts_token_map



def get_closest_token_from_timestamp(timestamp: float, ts_token_map: dict) -> str:
    """
    Given a timestamp (in seconds) and a timestamp→token map, return the closest sample_data token.

    Args:
        timestamp (float): Input timestamp (e.g. ROS message time).
        ts_token_map (dict): Dictionary from build_timestamp_token_map().

    Returns:
        str: Closest matching sample_data token.
    """
    closest_ts = min(ts_token_map.keys(), key=lambda t: abs(t - timestamp))
    return ts_token_map[closest_ts]


def visualize_segmentation(result, output_path=None, alpha=0.5, palette='bdd'):
    """
    Visualize semantic segmentation results by overlaying colored masks on the original image.
    
    Args:
        result: Segmentation result (DetDataSample or dict-like) containing:
                - 'img' or result.img: Original image (H, W, 3) in BGR format
                - 'pred_sem_seg' or result.pred_sem_seg: Predicted segmentation mask
        output_path (str, optional): Path to save the visualization. If None, display using cv2.imshow()
        alpha (float): Transparency of the segmentation overlay (0.0 = transparent, 1.0 = opaque)
        palette (str): Color palette to use ('bdd', 'cityscapes', 'ade20k', or 'default')
    
    Returns:
        numpy.ndarray: Visualization image in BGR format
    """
    import torch
    
    # Extract image
    if hasattr(result, 'img'):
        img = result.img
    elif isinstance(result, dict) and 'img' in result:
        img = result['img']
    else:
        raise ValueError("Result must contain 'img' field")
    
    # Extract segmentation mask
    if hasattr(result, 'pred_sem_seg'):
        seg_mask = result.pred_sem_seg.data
    elif isinstance(result, dict) and 'pred_sem_seg' in result:
        seg_mask = result['pred_sem_seg']['data'] if isinstance(result['pred_sem_seg'], dict) else result['pred_sem_seg']
    else:
        raise ValueError("Result must contain 'pred_sem_seg' field")
    
    # Convert to numpy if needed
    if torch.is_tensor(seg_mask):
        seg_mask = seg_mask.cpu().numpy()
    
    # Remove batch dimension if present
    if len(seg_mask.shape) == 3 and seg_mask.shape[0] == 1:
        seg_mask = seg_mask[0]
    
    # Ensure img is numpy array
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    
    # Get color palette
    color_map = get_segmentation_palette(palette)
    
    # Create colored segmentation mask
    h, w = seg_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in np.unique(seg_mask):
        class_id = int(class_id)
        if class_id < len(color_map):
            mask = seg_mask == class_id
            colored_mask[mask] = color_map[class_id]
    
    # Resize colored mask to match original image size if needed
    if colored_mask.shape[:2] != img.shape[:2]:
        colored_mask = cv2.resize(colored_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Blend the image with the colored mask
    vis_img = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)
    
    # Optionally save or display
    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"Saved segmentation visualization to: {output_path}")
    
    return vis_img


def get_segmentation_palette(palette='bdd'):
    """
    Get color palette for segmentation visualization.
    
    Args:
        palette (str): Name of the palette ('bdd', 'cityscapes', 'ade20k', 'default')
    
    Returns:
        list: List of (B, G, R) color tuples for each class
    """
    if palette == 'bdd':
        # BDD100K semantic segmentation classes
        # 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole, 
        # 6: traffic light, 7: traffic sign, 8: vegetation, 9: terrain, 
        # 10: sky, 11: person, 12: rider, 13: car, 14: truck, 15: bus,
        # 16: train, 17: motorcycle, 18: bicycle
        colors = [
            (128, 64, 128),   # 0: road - purple
            (244, 35, 232),   # 1: sidewalk - pink
            (70, 70, 70),     # 2: building - dark gray
            (102, 102, 156),  # 3: wall - gray-blue
            (190, 153, 153),  # 4: fence - light gray
            (153, 153, 153),  # 5: pole - gray
            (250, 170, 30),   # 6: traffic light - orange
            (220, 220, 0),    # 7: traffic sign - yellow
            (107, 142, 35),   # 8: vegetation - green
            (152, 251, 152),  # 9: terrain - light green
            (70, 130, 180),   # 10: sky - blue
            (220, 20, 60),    # 11: person - red
            (255, 0, 0),      # 12: rider - bright red
            (0, 0, 142),      # 13: car - dark blue
            (0, 0, 70),       # 14: truck - darker blue
            (0, 60, 100),     # 15: bus - blue
            (0, 80, 100),     # 16: train - blue
            (0, 0, 230),      # 17: motorcycle - bright blue
            (119, 11, 32),    # 18: bicycle - brown-red
        ]
    elif palette == 'cityscapes':
        # Cityscapes color palette
        colors = [
            (128, 64, 128),   # road
            (244, 35, 232),   # sidewalk
            (70, 70, 70),     # building
            (102, 102, 156),  # wall
            (190, 153, 153),  # fence
            (153, 153, 153),  # pole
            (250, 170, 30),   # traffic light
            (220, 220, 0),    # traffic sign
            (107, 142, 35),   # vegetation
            (152, 251, 152),  # terrain
            (70, 130, 180),   # sky
            (220, 20, 60),    # person
            (255, 0, 0),      # rider
            (0, 0, 142),      # car
            (0, 0, 70),       # truck
            (0, 60, 100),     # bus
            (0, 80, 100),     # train
            (0, 0, 230),      # motorcycle
            (119, 11, 32),    # bicycle
        ]
    else:  # default palette
        # Generate distinct colors for up to 256 classes
        np.random.seed(42)
        colors = [(int(r), int(g), int(b)) 
                  for r, g, b in np.random.randint(0, 255, size=(256, 3))]
        # Make background (class 0) black
        colors[0] = (0, 0, 0)
    
    return colors


def visualize_segmentation_with_legend(result, output_path=None, alpha=0.5, palette='bdd', class_names=None):
    """
    Visualize semantic segmentation with a legend showing class names and colors.
    
    Args:
        result: Segmentation result containing image and predictions
        output_path (str, optional): Path to save the visualization
        alpha (float): Transparency of the segmentation overlay
        palette (str): Color palette to use
        class_names (list, optional): List of class names. If None, uses indices.
    
    Returns:
        numpy.ndarray: Visualization image with legend
    """
    # Get base visualization
    vis_img = visualize_segmentation(result, output_path=None, alpha=alpha, palette=palette)
    
    # Extract segmentation mask for legend
    if hasattr(result, 'pred_sem_seg'):
        seg_mask = result.pred_sem_seg.data
    else:
        seg_mask = result['pred_sem_seg']['data'] if isinstance(result['pred_sem_seg'], dict) else result['pred_sem_seg']
    
    if torch.is_tensor(seg_mask):
        seg_mask = seg_mask.cpu().numpy()
    
    if len(seg_mask.shape) == 3 and seg_mask.shape[0] == 1:
        seg_mask = seg_mask[0]
    
    # Get unique classes present in the image
    unique_classes = np.unique(seg_mask)
    
    # Get color palette
    color_map = get_segmentation_palette(palette)
    
    # Create legend
    legend_height = 30 * len(unique_classes) + 40
    legend_width = 200
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
    
    y_offset = 20
    for class_id in sorted(unique_classes):
        class_id = int(class_id)
        
        # Draw color box
        color = color_map[class_id] if class_id < len(color_map) else (128, 128, 128)
        cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 10), color, -1)
        cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 10), (0, 0, 0), 1)
        
        # Draw class name
        if class_names and class_id < len(class_names):
            text = class_names[class_id]
        else:
            text = f"Class {class_id}"
        
        cv2.putText(legend, text, (40, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        y_offset += 30
    
    # Combine visualization and legend
    h, w = vis_img.shape[:2]
    combined = np.zeros((h, w + legend_width, 3), dtype=np.uint8)
    combined[:h, :w] = vis_img
    combined[:min(h, legend_height), w:] = legend[:min(h, legend_height)]
    
    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"Saved segmentation visualization with legend to: {output_path}")
    
    return combined

    