import numpy as np
from mmdet3d.apis import LidarDet3DInferencer, inference_multi_modality_detector, init_model
from mmdet.apis import DetInferencer
from p_perf.utils import list_filenames
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
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)
from nuscenes import NuScenes
import os

from p_perf.post_process.lidar_eval import lidar_output_to_nusc_box, lidar_nusc_box_to_global


# from p_perf.pPerf import pPerf

WARM_PCD = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'

# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import Box as NuScenesBox

# # data_base = '/home/mg/pPerf/data/nuscenes'      
# data_base = '/mmdetection3d_ros2/data/nuscenes'

# nusc = NuScenes(
#     version='v1.0-mini',            # or 'v1.0-mini', 'v1.0-test', etc.
#     dataroot=data_base
# )

# nusc = NuScenes(version='v1.0-mini', dataroot='/mmdetection3d_ros2/data/nuscenes', verbose=True)

# token = 'd296220b51f44f90b57f22ec9b83b42d'
# # Step 1: Get the sample_data record
# sd = nusc.get('sample_data', token)
# img_path = os.path.join(nusc.dataroot, sd['filename'])

# inferencer = LidarDet3DInferencer('hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d')
# points = np.fromfile(img_path, dtype=np.float32).reshape(-1, 5)
# input_tensor = dict(points=np.array(points, dtype=np.float32))  # or your actual test input

# det = inferencer(input_tensor, show=False, return_datasamples=True)


# print(det['predictions'][0].pred_instances_3d)
# boxes = lidar_output_to_nusc_box(det['predictions'][0].pred_instances_3d, token)

# print(boxes)
# boxes = lidar_nusc_box_to_global(nusc, token, boxes)

# print(boxes)

# output_base = "/mmdetection3d_ros2/outputs/test"
# lidar_pred_file = f"{output_base}/lidar_pred_0.json"

# with open(lidar_pred_file) as f:
#     data = json.load(f)

# num_cars = sum(1 for d in data['results'] if d.get("detection_name") == "car")
# print(num_cars)



# profiler = pPerf("pointpillar", inferencer, 0)
# profiler.warm_up(input_tensor)
# profiler.register_hooks(input_tensor)
# # Optional: View summary
# profiler.summary()

# dets = []
# results = []
# for path in lidar_paths[:5]:
#     input_name = os.path.basename(path).split('.')[0]
#     points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
#     input_tensor = dict(points=np.array(points, dtype=np.float32)) 
#     det = profiler.run_inference(input_tensor, input_name)
#     dets.append((os.path.basename(path), det['predictions'][0].pred_instances_3d))




# # # img_paths = list_filenames('/mmdetection3d_ros2/data/nuscenes/sweeps/CAM_FRONT', 'jpg')
# # # image_inferencer = DetInferencer('faster-rcnn_r50_fpn_1x_coco')
# # # input_tensor = cv2.imread(img_paths[0])

# # # profiler = AutoProfiler("faster_rcnn", image_inferencer, 0)
# # # profiler.warm_up(input_tensor)
# # # profiler.register_hooks(input_tensor)

# # # # Optional: View summary
# # # profiler.summary()

# # # dets = []
# # # # Step 4: Run actual profiling
# # # for path in img_paths[1:5]:
# # #     input_name = os.path.basename(path).split('.')[0]
# # #     image = cv2.imread(path)
# # #     det = profiler.run_inference(input_tensor, input_name)
# # #     dets.append((os.path.basename(path), det['predictions'][0].pred_instances))


# # # image_dets = []
# # # for det in dets:    
# # #     coco_result = image_output_to_coco(det[1], det[0])
# # #     image_dets.extend(coco_result)

# # # with open('coco_predictions.json', 'w') as f:
# # #     json.dump(image_dets, f)

# # # from pycocotools.coco import COCO
# # # from pycocotools.cocoeval import COCOeval
# # # import json


# # bboxes, labels = get_2d_boxes("f97f711fff2b43fab0328cf4db040608", ["3", "4"])

# # print(bboxes)
# # print(labels)
# # file_name = f"{data_base}/samples/CAM_BACK/n008-2018-08-30-15-16-55-0400__CAM_BACK__1535657124637558.jpg"
# # draw_coco_bboxes_from_path(file_name, bboxes, labels)
# # # nusc.render_sample("73ccc9c10a3547849b8ff130d50bac98")





# from p_perf.utils import visualize_coco_predictions, load_sweep_sd
# from p_perf.post_process.image_eval import generate_coco_gt
# from p_perf.post_process.pseudo_gt import generate_pseudo_coco_gt
# import json
# from nuscenes import NuScenes

# output_base = "/mmdetection3d_ros2/outputs/test"
# image_pred_file = f"{output_base}/image_pred_0.json"
# image_gt_file = f"{output_base}/image_gt_0.json"

# nusc = NuScenes(version='v1.0-mini', dataroot='/mmdetection3d_ros2/data/nuscenes', verbose=True)


# delay_csv = f"{output_base}/delays_0.csv"
# with open(image_gt_file, 'r') as f:
#     data = json.load(f)

# tokens = [img['token'] for img in data.get('images', []) if 'token' in img]
# # generate_coco_gt(tokens, 'test.json', delay_csv, iob_thresh=0.2, visibilities=['2', '3', '4'])
# generate_pseudo_coco_gt(nusc, tokens, None, None, None, delay_csv, 'test.json')

# sample_tokens = load_sweep_sd(nusc, nusc.scene[0])
# for i, token in enumerate(sample_tokens[:40]):
#     if token in tokens:
#         visualize_coco_predictions(token, image_pred_file, 'test.json', '/mmdetection3d_ros2/data/nuscenes', i)


import pandas as pd

# Path to your feather file
feather_path = '/mmdetection3d_ros2/data/Argoverse/sensor/train/00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/annotations.feather'
csv_path = '/mmdetection3d_ros2/data/Argoverse/sensor/train/00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/annotation.csv'

# Read the feather file
df = pd.read_feather(feather_path)

# Save to CSV
df.to_csv(csv_path, index=False)

print(f"Saved CSV to {csv_path}")