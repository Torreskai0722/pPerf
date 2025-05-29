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

nusc = NuScenes(version='v1.0-mini', dataroot='/mmdetection3d_ros2/data/nuscenes', verbose=True)

token = 'd296220b51f44f90b57f22ec9b83b42d'
# Step 1: Get the sample_data record
sd = nusc.get('sample_data', token)
img_path = os.path.join(nusc.dataroot, sd['filename'])

inferencer = LidarDet3DInferencer('hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d')
points = np.fromfile(img_path, dtype=np.float32).reshape(-1, 5)
input_tensor = dict(points=np.array(points, dtype=np.float32))  # or your actual test input

det = inferencer(input_tensor, show=False, return_datasamples=True)


print(det['predictions'][0].pred_instances_3d)
boxes = lidar_output_to_nusc_box(det['predictions'][0].pred_instances_3d, token)

print(boxes)
boxes = lidar_nusc_box_to_global(nusc, token, boxes)

print(boxes)

output_base = "/mmdetection3d_ros2/outputs/test"
lidar_pred_file = f"{output_base}/lidar_pred_0.json"

with open(lidar_pred_file) as f:
    data = json.load(f)

num_cars = sum(1 for d in data['results'] if d.get("detection_name") == "car")
print(num_cars)



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


[0.1328, 0.1078, 0.1010, 0.1004, 0.0986, 0.0973, 0.0957, 0.0964, 0.0929,
                0.4355, 0.3819, 0.2991, 0.2882, 0.2875, 0.2748, 0.2475, 0.2315, 0.2093,
                0.1967, 0.1959, 0.1927, 0.1871, 0.1823, 0.1778, 0.1762, 0.1747, 0.1677,
                0.1644, 0.1609, 0.1438, 0.1338, 0.1311, 0.1270, 0.1251, 0.1248, 0.1247,
                0.1212, 0.1207, 0.1204, 0.1179, 0.1161, 0.1155, 0.1140, 0.1140, 0.1125,
                0.1125, 0.1115, 0.1102, 0.1075, 0.1059, 0.1053, 0.1031, 0.1015, 0.1007,
                0.0982, 0.0975, 0.0975, 0.0939, 0.0925, 0.0922, 0.0920, 0.0907, 0.0907,
                0.0906, 0.0902, 0.1669, 0.1322, 0.1249, 0.0922, 0.0888, 0.3427, 0.3198,
                0.3157, 0.3139, 0.2832, 0.2505, 0.2401, 0.2174, 0.2164, 0.2147, 0.2089,
                0.2057, 0.2045, 0.2004, 0.1973, 0.1960, 0.1763, 0.1752, 0.1727, 0.1717,
                0.1704, 0.1689, 0.1662, 0.1596, 0.1574, 0.1496, 0.1339, 0.1333, 0.1301,
                0.1296, 0.1290, 0.1280, 0.1260, 0.1226, 0.1215, 0.1197, 0.1180, 0.1179,
                0.1157, 0.1144, 0.1134, 0.1124, 0.1111, 0.1101, 0.1093, 0.1090, 0.1084,
                0.1051, 0.1040, 0.1037, 0.1028, 0.1011, 0.1005, 0.0993, 0.0992, 0.0985,
                0.0984, 0.0979, 0.0959, 0.0954, 0.0951, 0.0934, 0.0927, 0.0914, 0.0897,
                0.4676, 0.4293, 0.3837, 0.3491, 0.3465, 0.3465, 0.3032, 0.2931, 0.2827,
                0.2623, 0.2485, 0.2462, 0.2194, 0.2166, 0.2122, 0.2109, 0.2105, 0.2085,
                0.2028, 0.1990, 0.1932, 0.1896, 0.1889, 0.1878, 0.1768, 0.1765, 0.1755,
                0.1746, 0.1639, 0.1589, 0.1529, 0.1454, 0.1293, 0.1283, 0.1242, 0.1219,
                0.1217, 0.1160, 0.1159, 0.1159, 0.1152, 0.1149, 0.1143, 0.1125, 0.1124,
                0.1121, 0.1097, 0.1094, 0.1065, 0.1045, 0.1045, 0.1030, 0.1026, 0.1010,
                0.1002, 0.0995, 0.0993, 0.0987, 0.0983, 0.0982, 0.0975, 0.0971, 0.0943,
                0.0941, 0.0932, 0.0931, 0.0913, 0.0892, 0.0891, 0.0891, 0.2891, 0.2196,
                0.2038, 0.1878, 0.1814, 0.1573, 0.1537, 0.1522, 0.1503, 0.1367, 0.1364,
                0.1311, 0.1298, 0.1267, 0.1239, 0.1225, 0.1105, 0.1081, 0.1051, 0.1037,
                0.1036, 0.1004, 0.1002, 0.0989, 0.0906, 0.0896, 0.1129, 0.1122, 0.0986,
                0.1757, 0.1658, 0.1317, 0.1054, 0.0891, 0.3021, 0.1549, 0.1159, 0.1102,
                0.1093, 0.0969, 0.0936]