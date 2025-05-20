import numpy as np
from mmdet3d.apis import LidarDet3DInferencer, inference_multi_modality_detector, init_model
from mmdet.apis import DetInferencer
from utils import list_filenames
import mmcv
import matplotlib.pyplot as plt
from mmdet.visualization import DetLocalVisualizer
import os
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage
)
import threading
import cv2
import torch
import functools
import inspect
import time
from pyquaternion import Quaternion
import json

WARM_PCD = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'

class AutoProfiler:
    def __init__(self, model_name, inferencer, depth, monitor_interval=0.1):
        self.model_name = model_name
        self.method_timings = {}  # {method_id: (start, end, tag)}
        self.filtered_methods = []  # methods selected after filtering
        self.module_method_map = {}  # for looking up method handles
        self.target_depth = depth

        # Model inferencing
        self.inferencer = inferencer

        # GPU monitoring
        self.gpu_stats = []
        self.gpu_monitor_thread = None
        self.monitoring = False
        self.monitor_interval = monitor_interval  # seconds
        self.warming = False

    def _trace_wrapper(self, fn, tag, method_id):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.time()
            self.method_timings[method_id] = (start, end, tag)
            return result
        return wrapped

    def trace_and_record_times(self, warmup_data):
        """
        Wraps all model methods, runs inference, and records time per method.
        Uses full module names like 'backbone.middle_encoder.conv1'.
        """
        print("[AutoProfiler] Wrapping all model methods to record timing...")

        # Build name -> module map (like model.backbone.middle_encoder)
        module_to_name = {}
        model = self.inferencer.model
        for name, module in model.named_modules():
            module_to_name[id(module)] = name  # id ensures match with method_id

        for module in model.modules():
            module_id = id(module)
            qualified_name = module_to_name.get(module_id, module.__class__.__name__)
            for name, method in inspect.getmembers(module, predicate=inspect.ismethod):
                method_id = (module_id, name)
                tag = f"{self.model_name}.{qualified_name}.{name}"
                marker_name = f"_original_{name}"

                if not hasattr(module, marker_name):
                    setattr(module, marker_name, method)
                    wrapped = self._trace_wrapper(method, tag, method_id)
                    setattr(module, name, wrapped)
                    self.module_method_map[method_id] = (module, name, tag)

        print("[AutoProfiler] Running input to measure execution time...")
        self.inferencer(warmup_data)
        print(f"[AutoProfiler] Recorded {len(self.method_timings)} method timings.")
        

    def filter_nested_ranges(self, model, tolerance_ms=0.5):
        """
        Filters overlapping methods (within tolerance_ms),
        and keeps only methods belonging to model's top-level children and their descendants.
        """
        print(f"[AutoProfiler] Filtering methods within {tolerance_ms} ms...")

        # Step 1: Get all valid modules = children + their descendants
        valid_modules = set()
        for _, child in model.named_children():
            for submodule in child.modules():
                valid_modules.add(id(submodule))

        # Step 2: Prepare timing ranges
        ranges = [
            (method_id, start, end, tag)
            for method_id, (start, end, tag) in self.method_timings.items()
        ]
        keep_set = set(method_id for method_id, *_ in ranges)

        # Step 3: Time-based filtering
        for i in range(len(ranges)):
            id_i, start_i, end_i, _ = ranges[i]
            for j in range(i + 1, len(ranges)):
                id_j, start_j, end_j, _ = ranges[j]

                start_diff = abs((start_i - start_j) * 1000)
                end_diff = abs((end_i - end_j) * 1000)

                if start_diff < tolerance_ms and end_diff < tolerance_ms:
                    if start_i <= start_j and end_i >= end_j:
                        keep_set.discard(id_j)
                    elif start_j <= start_i and end_j >= end_i:
                        keep_set.discard(id_i)

        # Step 4: Source-based filtering
        filtered = []
        for method_id in keep_set:
            module, _, _ = self.module_method_map[method_id]
            if id(module) in valid_modules:
                filtered.append(method_id)

        # Step 5: Compute nesting depth using only filtered methods
        timings = {
            method_id: (start, end, tag)
            for method_id, (start, end, tag) in self.method_timings.items()
            if method_id in filtered  # use filtered, not keep_set
        }

        # Sort filtered methods by start time
        sorted_timings = sorted(timings.items(), key=lambda x: x[1][0])

        # Compute depth using a nesting stack
        self.method_depths = {}  # Maps method_id -> depth
        active_stack = []

        for method_id, (start, end, _) in sorted_timings:
            while active_stack and active_stack[-1][1] <= start:
                active_stack.pop()

            depth = len(active_stack)
            self.method_depths[method_id] = depth
            active_stack.append((method_id, end))

        self.filtered_methods = filtered
        print(f"[AutoProfiler] Kept {len(filtered)} methods after time + module filtering with depth info.")

    def _nvtx_wrapper(self, fn, tag):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(tag)
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return result
        return wrapped

    def wrap_filtered_methods_with_nvtx(self):
        """
        Wraps only filtered methods at the target depth with NVTX markers.
        """
        count = 0
        for method_id in self.filtered_methods:
            if self.method_depths[method_id] != self.target_depth:
                continue
            module, name, tag = self.module_method_map[method_id]
            original = getattr(module, name)
            setattr(module, name, self._nvtx_wrapper(original, tag))
            count += 1


    def summary(self):
        print("\n[AutoProfiler] Final NVTX-annotated methods by depth:")
        max_depth = max(self.method_depths.values(), default=-1)
        for d in range(max_depth + 1):
            print(f"\n  --- Depth {d} ---")
            for method_id in self.filtered_methods:
                if self.method_depths[method_id] == d:
                    _, _, tag = self.module_method_map[method_id]
                    print(f"{tag}")

    # ACTUAL METHODS USED IN INFERENCER
    # During inferencing, the pipeline should be 
    # warm_up ==> register_hooks ==> run_inference
    
    def warm_up(self, warm_data, num_warmups=10):
        for _ in range(num_warmups):
            self.inferencer(warm_data)

    def register_hooks(self, warm_data):
        self.trace_and_record_times(warm_data)
        self.filter_nested_ranges(self.inferencer.model)
        self.wrap_filtered_methods_with_nvtx()
        print("MAX_DEPTH is: ", max(self.method_depths.values(), default=-1))
    
    def run_inference(self, data, input_name):
        torch.cuda.nvtx.range_push(f'{input_name}.{self.model_name}.e2e')
        result = self.inferencer(data, return_datasamples=True)
        torch.cuda.nvtx.range_pop()
        return result
    
    # GPU Profiling
    def start_gpu_monitoring(self):
        # If self monitoring is false, skip both start and stop
        if self.monitoring == False:
            return
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        def monitor():
            while self.monitoring:
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0

                self.gpu_stats.append({
                    'time': time.time(),
                    'gpu_util': util.gpu,
                    'mem_util': util.memory,
                    'mem_used_MB': mem.used / (1024 ** 2),
                    'power_W': power
                })
                time.sleep(self.monitor_interval)

        self.monitoring = True
        self.gpu_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.gpu_monitor_thread.start()

    def stop_gpu_monitoring(self):
        if self.monitoring == False:
            return
        self.monitoring = False
        if self.gpu_monitor_thread is not None:
            self.gpu_monitor_thread.join()
        nvmlShutdown()


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

lidar_paths = list_filenames('/mmdetection3d_ros2/data/nuscenes/sweeps/LIDAR_TOP', 'bin')
inferencer = LidarDet3DInferencer('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d')
points = np.fromfile(lidar_paths[0], dtype=np.float32).reshape(-1, 5)
input_tensor = dict(points=np.array(points, dtype=np.float32))  # or your actual test input

profiler = AutoProfiler("pointpillar", inferencer, 0)
profiler.warm_up(input_tensor)
profiler.register_hooks(input_tensor)
# # Optional: View summary
# profiler.summary()

# dets = []
# results = []
# for path in lidar_paths[:200]:
#     input_name = os.path.basename(path).split('.')[0]
#     points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
#     input_tensor = dict(points=np.array(points, dtype=np.float32)) 
#     det = profiler.run_inference(input_tensor, input_name)
#     dets.append((os.path.basename(path), det['predictions'][0].pred_instances_3d))

# for det in dets:
#     token = get_token_from_filename(json_path, det[0])
#     results.extend(convert_det_to_nusc_format(det[1], token))




# img_paths = list_filenames('/mmdetection3d_ros2/data/nuscenes/sweeps/CAM_FRONT', 'jpg')
# image_inferencer = DetInferencer('faster-rcnn_r50_fpn_1x_coco')
# input_tensor = cv2.imread(img_paths[0])

# profiler = AutoProfiler("faster_rcnn", image_inferencer, 0)
# profiler.warm_up(input_tensor)
# profiler.register_hooks(input_tensor)

# # # Optional: View summary
# profiler.summary()

# # Step 4: Run actual profiling
# for path in img_paths[1:5]:
#     input_name = os.path.basename(path).split('.')[0]
#     image = cv2.imread(path)
#     profiler.run_inference(image, input_name)

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)

def output_to_nusc_box(
        detection, token):
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

nusc = NuScenes(
    version='v1.0-mini',            # or 'v1.0-mini', 'v1.0-test', etc.
    dataroot='/mmdetection3d_ros2/data/nuscenes'  # replace with the actual path
)

nusc_annos = {}
token = ''
json_path = '/mmdetection3d_ros2/data/nuscenes/v1.0-mini/sample_data.json'
classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

class_range = {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "construction_vehicle": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "traffic_cone": 30,
    "barrier": 30
  },

dist_ths = [0.5, 1, 2, 4]

# for det in dets:
#     token = get_token_from_filename(json_path, det[0])
#     boxes = output_to_nusc_box(det[1], token)
#     boxes = lidar_nusc_box_to_global(nusc, token, boxes)

#     annos = []
#     for box in boxes:
#         name = classes[box.label]
#         nusc_anno = dict(
#             sample_token=token,
#             translation=box.center.tolist(),
#             size=box.wlh.tolist(),
#             rotation=box.orientation.elements.tolist(),
#             velocity=box.velocity[:2].tolist(),
#             detection_name=name,
#             detection_score=box.score,
#             attribute_name='')
#         annos.append(nusc_anno)
#     nusc_annos[token] = annos

# import mmengine
# import os

# output_path = 'results_nusc.json'

# # Wrap in NuScenes evaluation format
# nusc_submission = {
#     'meta': {
#         'use_camera': False,
#         'use_lidar': True,
#         'use_radar': False,
#         'use_map': False,
#         'use_external': False
#     },
#     'results': nusc_annos
# }

# # Write to file
# mmengine.dump(nusc_submission, output_path)
# print(f"Results written to {output_path}")


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
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
)
from collections import defaultdict

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


def accumulate(nusc,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn,
               dist_th: float,
               verbose: bool = False
              ):
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    # Sort by confidence
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp, fp, conf = [], [], []
    match_data = {k: [] for k in ['trans_err', 'vel_err', 'scale_err', 'orient_err', 'attr_err', 'conf']}
    taken = set()
    interpolated_cache = {}
    instance_hits = {}

    # Precompute interpolated GTs per sample_data_token
    for token in pred_boxes.sample_tokens:
        boxes, instance_tokens = interpolate_gt(nusc, token)
        interpolated_cache[token] = (boxes, instance_tokens)

    # Compute total GTs for this class (avoid double counting)
    npos = sum(
        sum(1 for box in boxes if box.detection_name == class_name)
        for boxes, _ in interpolated_cache.values()
    )

    if verbose:
        print(f"GT count for class {class_name}: {npos}")

    # Matching
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        token = pred_box.sample_token
        interpolated_gts, instance_tokens = interpolated_cache[token]

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
            instance_token = instance_tokens[match_gt_idx]

            # Track instance hit
            if instance_token is not None:
                instance_hits[instance_token] = instance_hits.get(instance_token, 0) + 1

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
        return DetectionMetricData.no_predictions(), instance_hits

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
    ), instance_hits



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
all_instance_hits = {dist_th: defaultdict(int) for dist_th in dist_ths}

for class_name in classes:
    for dist_th in dist_ths:
        md, instance_hits = accumulate(nusc, pred_boxes, class_name, center_distance, dist_th)
        
        # Record AP
        metric_data_list.set(class_name, dist_th, md)
        ap = calc_ap(md, 0.1, 0.1)
        metrics[(class_name, dist_th)] = ap

        # Aggregate instance hits per threshold
        for inst_token, count in instance_hits.items():
            all_instance_hits[dist_th][inst_token] += count

print(metrics)

# Optional: Show instance hit stats for each distance threshold
for dist_th, hits_dict in all_instance_hits.items():
    print(f"\n[Threshold {dist_th}] Unique instances matched: {len(hits_dict)}")
    top_hits = sorted(hits_dict.items(), key=lambda x: -x[1])[:5]
    print("Top 5 most matched instances:")
    for token, count in top_hits:
        print(f"  Instance {token}: matched {count} times")


    file_path = f"dist_{dist_th}.json"
    with open(file_path, 'w') as json_file:
        json.dump(hits_dict, json_file, indent=4)