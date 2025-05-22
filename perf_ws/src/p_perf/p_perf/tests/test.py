# import numpy as np
# from mmdet3d.apis import LidarDet3DInferencer, inference_multi_modality_detector, init_model
# from mmdet.apis import DetInferencer
# from p_perf.utils import list_filenames
# import mmcv
# import matplotlib.pyplot as plt
# from mmdet.visualization import DetLocalVisualizer
# import os
# from pynvml import (
#     nvmlInit,
#     nvmlShutdown,
#     nvmlDeviceGetHandleByIndex,
#     nvmlDeviceGetUtilizationRates,
#     nvmlDeviceGetMemoryInfo,
#     nvmlDeviceGetPowerUsage
# )
# import threading
# import cv2
# import torch
# import functools
# import inspect
# import time
# from pyquaternion import Quaternion
# import json
# from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
#                                 bbox3d2result, xywhr2xyxyr)
# from p_perf.utils import image_output_to_coco

# WARM_PCD = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'

# class AutoProfiler:
#     def __init__(self, model_name, inferencer, depth, monitor_interval=0.1):
#         self.model_name = model_name
#         self.method_timings = {}  # {method_id: (start, end, tag)}
#         self.filtered_methods = []  # methods selected after filtering
#         self.module_method_map = {}  # for looking up method handles
#         self.target_depth = depth

#         # Model inferencing
#         self.inferencer = inferencer

#         # GPU monitoring
#         self.gpu_stats = []
#         self.gpu_monitor_thread = None
#         self.monitoring = False
#         self.monitor_interval = monitor_interval  # seconds
#         self.warming = False

#     def _trace_wrapper(self, fn, tag, method_id):
#         @functools.wraps(fn)
#         def wrapped(*args, **kwargs):
#             torch.cuda.synchronize()
#             start = time.time()
#             result = fn(*args, **kwargs)
#             torch.cuda.synchronize()
#             end = time.time()
#             self.method_timings[method_id] = (start, end, tag)
#             return result
#         return wrapped

#     def trace_and_record_times(self, warmup_data):
#         """
#         Wraps all model methods, runs inference, and records time per method.
#         Uses full module names like 'backbone.middle_encoder.conv1'.
#         """
#         print("[AutoProfiler] Wrapping all model methods to record timing...")

#         # Build name -> module map (like model.backbone.middle_encoder)
#         module_to_name = {}
#         model = self.inferencer.model
#         for name, module in model.named_modules():
#             module_to_name[id(module)] = name  # id ensures match with method_id

#         for module in model.modules():
#             module_id = id(module)
#             qualified_name = module_to_name.get(module_id, module.__class__.__name__)
#             for name, method in inspect.getmembers(module, predicate=inspect.ismethod):
#                 method_id = (module_id, name)
#                 tag = f"{self.model_name}.{qualified_name}.{name}"
#                 marker_name = f"_original_{name}"

#                 if not hasattr(module, marker_name):
#                     setattr(module, marker_name, method)
#                     wrapped = self._trace_wrapper(method, tag, method_id)
#                     setattr(module, name, wrapped)
#                     self.module_method_map[method_id] = (module, name, tag)

#         print("[AutoProfiler] Running input to measure execution time...")
#         self.inferencer(warmup_data)
#         print(f"[AutoProfiler] Recorded {len(self.method_timings)} method timings.")
        

#     def filter_nested_ranges(self, model, tolerance_ms=0.5):
#         """
#         Filters overlapping methods (within tolerance_ms),
#         and keeps only methods belonging to model's top-level children and their descendants.
#         """
#         print(f"[AutoProfiler] Filtering methods within {tolerance_ms} ms...")

#         # Step 1: Get all valid modules = children + their descendants
#         valid_modules = set()
#         for _, child in model.named_children():
#             for submodule in child.modules():
#                 valid_modules.add(id(submodule))

#         # Step 2: Prepare timing ranges
#         ranges = [
#             (method_id, start, end, tag)
#             for method_id, (start, end, tag) in self.method_timings.items()
#         ]
#         keep_set = set(method_id for method_id, *_ in ranges)

#         # Step 3: Time-based filtering
#         for i in range(len(ranges)):
#             id_i, start_i, end_i, _ = ranges[i]
#             for j in range(i + 1, len(ranges)):
#                 id_j, start_j, end_j, _ = ranges[j]

#                 start_diff = abs((start_i - start_j) * 1000)
#                 end_diff = abs((end_i - end_j) * 1000)

#                 if start_diff < tolerance_ms and end_diff < tolerance_ms:
#                     if start_i <= start_j and end_i >= end_j:
#                         keep_set.discard(id_j)
#                     elif start_j <= start_i and end_j >= end_i:
#                         keep_set.discard(id_i)

#         # Step 4: Source-based filtering
#         filtered = []
#         for method_id in keep_set:
#             module, _, _ = self.module_method_map[method_id]
#             if id(module) in valid_modules:
#                 filtered.append(method_id)

#         # Step 5: Compute nesting depth using only filtered methods
#         timings = {
#             method_id: (start, end, tag)
#             for method_id, (start, end, tag) in self.method_timings.items()
#             if method_id in filtered  # use filtered, not keep_set
#         }

#         # Sort filtered methods by start time
#         sorted_timings = sorted(timings.items(), key=lambda x: x[1][0])

#         # Compute depth using a nesting stack
#         self.method_depths = {}  # Maps method_id -> depth
#         active_stack = []

#         for method_id, (start, end, _) in sorted_timings:
#             while active_stack and active_stack[-1][1] <= start:
#                 active_stack.pop()

#             depth = len(active_stack)
#             self.method_depths[method_id] = depth
#             active_stack.append((method_id, end))

#         self.filtered_methods = filtered
#         print(f"[AutoProfiler] Kept {len(filtered)} methods after time + module filtering with depth info.")

#     def _nvtx_wrapper(self, fn, tag):
#         @functools.wraps(fn)
#         def wrapped(*args, **kwargs):
#             torch.cuda.synchronize()
#             torch.cuda.nvtx.range_push(tag)
#             result = fn(*args, **kwargs)
#             torch.cuda.synchronize()
#             torch.cuda.nvtx.range_pop()
#             return result
#         return wrapped

#     def wrap_filtered_methods_with_nvtx(self):
#         """
#         Wraps only filtered methods at the target depth with NVTX markers.
#         """
#         count = 0
#         for method_id in self.filtered_methods:
#             if self.method_depths[method_id] != self.target_depth:
#                 continue
#             module, name, tag = self.module_method_map[method_id]
#             original = getattr(module, name)
#             setattr(module, name, self._nvtx_wrapper(original, tag))
#             count += 1


#     def summary(self):
#         print("\n[AutoProfiler] Final NVTX-annotated methods by depth:")
#         max_depth = max(self.method_depths.values(), default=-1)
#         for d in range(max_depth + 1):
#             print(f"\n  --- Depth {d} ---")
#             for method_id in self.filtered_methods:
#                 if self.method_depths[method_id] == d:
#                     _, _, tag = self.module_method_map[method_id]
#                     print(f"{tag}")

#     # ACTUAL METHODS USED IN INFERENCER
#     # During inferencing, the pipeline should be 
#     # warm_up ==> register_hooks ==> run_inference
    
#     def warm_up(self, warm_data, num_warmups=10):
#         for _ in range(num_warmups):
#             self.inferencer(warm_data)

#     def register_hooks(self, warm_data):
#         self.trace_and_record_times(warm_data)
#         self.filter_nested_ranges(self.inferencer.model)
#         self.wrap_filtered_methods_with_nvtx()
#         print("MAX_DEPTH is: ", max(self.method_depths.values(), default=-1))
    
#     def run_inference(self, data, input_name):
#         torch.cuda.nvtx.range_push(f'{input_name}.{self.model_name}.e2e')
#         result = self.inferencer(data, return_datasamples=True, pred_score_thr=0.5)
#         torch.cuda.nvtx.range_pop()
#         return result
    
#     # GPU Profiling
#     def start_gpu_monitoring(self):
#         # If self monitoring is false, skip both start and stop
#         if self.monitoring == False:
#             return
#         nvmlInit()
#         handle = nvmlDeviceGetHandleByIndex(0)

#         def monitor():
#             while self.monitoring:
#                 util = nvmlDeviceGetUtilizationRates(handle)
#                 mem = nvmlDeviceGetMemoryInfo(handle)
#                 power = nvmlDeviceGetPowerUsage(handle) / 1000.0

#                 self.gpu_stats.append({
#                     'time': time.time(),
#                     'gpu_util': util.gpu,
#                     'mem_util': util.memory,
#                     'mem_used_MB': mem.used / (1024 ** 2),
#                     'power_W': power
#                 })
#                 time.sleep(self.monitor_interval)

#         self.monitoring = True
#         self.gpu_monitor_thread = threading.Thread(target=monitor, daemon=True)
#         self.gpu_monitor_thread.start()

#     def stop_gpu_monitoring(self):
#         if self.monitoring == False:
#             return
#         self.monitoring = False
#         if self.gpu_monitor_thread is not None:
#             self.gpu_monitor_thread.join()
#         nvmlShutdown()


from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox

data_base = '/home/mg/pPerf/data/nuscenes'      # '/mmdetection3d_ros2/data/nuscenes'

nusc = NuScenes(
    version='v1.0-mini',            # or 'v1.0-mini', 'v1.0-test', etc.
    # dataroot='/mmdetection3d_ros2/data/nuscenes'  # replace with the actual path
    dataroot='/home/mg/pPerf/data/nuscenes'
)

def get_token_from_filename(nusc, filename):
    """
    Given a filename (e.g., 'n015-2018-08-01-11-03-03+0800__LIDAR_TOP__1533091988890184.bin'),
    return the corresponding sample_data token.
    """
    for sd in nusc.sample_data:
        if filename in sd['filename']:  # or use os.path.basename(sd['filename']) == filename
            return sd['token']
    return None

# lidar_paths = list_filenames('/mmdetection3d_ros2/data/nuscenes/sweeps/LIDAR_TOP', 'bin')
# inferencer = LidarDet3DInferencer('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d')
# points = np.fromfile(lidar_paths[0], dtype=np.float32).reshape(-1, 5)
# input_tensor = dict(points=np.array(points, dtype=np.float32))  # or your actual test input

# profiler = AutoProfiler("pointpillar", inferencer, 0)
# profiler.warm_up(input_tensor)
# profiler.register_hooks(input_tensor)
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




# img_paths = list_filenames('/mmdetection3d_ros2/data/nuscenes/sweeps/CAM_FRONT', 'jpg')
# image_inferencer = DetInferencer('faster-rcnn_r50_fpn_1x_coco')
# input_tensor = cv2.imread(img_paths[0])

# profiler = AutoProfiler("faster_rcnn", image_inferencer, 0)
# profiler.warm_up(input_tensor)
# profiler.register_hooks(input_tensor)

# # Optional: View summary
# profiler.summary()

# dets = []
# # Step 4: Run actual profiling
# for path in img_paths[1:5]:
#     input_name = os.path.basename(path).split('.')[0]
#     image = cv2.imread(path)
#     det = profiler.run_inference(input_tensor, input_name)
#     dets.append((os.path.basename(path), det['predictions'][0].pred_instances))


# image_dets = []
# for det in dets:    
#     coco_result = image_output_to_coco(det[1], det[0])
#     image_dets.extend(coco_result)

# with open('coco_predictions.json', 'w') as f:
#     json.dump(image_dets, f)

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import json





import cv2
import json
import os
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :param filename:The corresponding image file where the annotation is present.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    return repro_rec


def get_2d_boxes(sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'

    s_rec = nusc.get('sample', sd_rec['sample_token'])
    print(s_rec, '\n')

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs

def xyxy_to_coco(bboxes_xyxy):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to COCO format [x, y, width, height].

    Args:
        bboxes_xyxy (np.ndarray): Array of shape (N, 4) in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Array of shape (N, 4) in [x, y, width, height] format.
    """
    bboxes_xyxy = np.asarray(bboxes_xyxy)
    x1 = bboxes_xyxy[:, 0]
    y1 = bboxes_xyxy[:, 1]
    x2 = bboxes_xyxy[:, 2]
    y2 = bboxes_xyxy[:, 3]

    w = x2 - x1
    h = y2 - y1
    return np.stack([x1, y1, w, h], axis=1)



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


results = get_2d_boxes("f97f711fff2b43fab0328cf4db040608", ["3", "4"])

bboxes = []
labels = []
for result in results:
    bboxes.append(result['bbox_corners'])
    labels.append(result['category_name'])

print(bboxes)
bboxes = xyxy_to_coco(bboxes)

file_name = f"{data_base}/samples/CAM_BACK/n008-2018-08-30-15-16-55-0400__CAM_BACK__1535657124637558.jpg"
draw_coco_bboxes_from_path(file_name, bboxes, labels)

nusc.render_sample("73ccc9c10a3547849b8ff130d50bac98")











