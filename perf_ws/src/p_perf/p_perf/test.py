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
# from nuscenes import NuScenes
# import os

# from p_perf.post_process.lidar_eval import lidar_output_to_nusc_box, lidar_nusc_box_to_global
# from p_perf.pre_process.data_converter import convert_to_kitti


# # from p_perf.pPerf import pPerf

# WARM_PCD = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'

# # from nuscenes.nuscenes import NuScenes
# # from nuscenes.utils.data_classes import Box as NuScenesBox

# # # data_base = '/home/mg/pPerf/data/nuscenes'      
# # data_base = '/mmdetection3d_ros2/data/nuscenes'

# # nusc = NuScenes(
# #     version='v1.0-mini',            # or 'v1.0-mini', 'v1.0-test', etc.
# #     dataroot=data_base
# # )

# # nusc = NuScenes(version='v1.0-mini', dataroot='/mmdetection3d_ros2/data/nuscenes', verbose=True)

# # token = 'd296220b51f44f90b57f22ec9b83b42d'
# # # Step 1: Get the sample_data record
# # sd = nusc.get('sample_data', token)
# # img_path = os.path.join(nusc.dataroot, sd['filename'])

# # inferencer = LidarDet3DInferencer('hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d')
# # points = np.fromfile(img_path, dtype=np.float32).reshape(-1, 5)
# # input_tensor = dict(points=np.array(points, dtype=np.float32))  # or your actual test input

# # det = inferencer(input_tensor, show=False, return_datasamples=True)


# # print(det['predictions'][0].pred_instances_3d)
# # boxes = lidar_output_to_nusc_box(det['predictions'][0].pred_instances_3d, token)

# # print(boxes)
# # boxes = lidar_nusc_box_to_global(nusc, token, boxes)

# # print(boxes)

# # output_base = "/mmdetection3d_ros2/outputs/test"
# # lidar_pred_file = f"{output_base}/lidar_pred_0.json"

# # with open(lidar_pred_file) as f:
# #     data = json.load(f)

# # num_cars = sum(1 for d in data['results'] if d.get("detection_name") == "car")
# # print(num_cars)



# # profiler = pPerf("pointpillar", inferencer, 0)
# # profiler.warm_up(input_tensor)
# # profiler.register_hooks(input_tensor)
# # # Optional: View summary
# # profiler.summary()

# # dets = []
# # results = []
# # for path in lidar_paths[:5]:
# #     input_name = os.path.basename(path).split('.')[0]
# #     points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
# #     input_tensor = dict(points=np.array(points, dtype=np.float32)) 
# #     det = profiler.run_inference(input_tensor, input_name)
# #     dets.append((os.path.basename(path), det['predictions'][0].pred_instances_3d))




# # # # img_paths = list_filenames('/mmdetection3d_ros2/data/nuscenes/sweeps/CAM_FRONT', 'jpg')
# # # # image_inferencer = DetInferencer('faster-rcnn_r50_fpn_1x_coco')
# # # # input_tensor = cv2.imread(img_paths[0])

# # # # profiler = AutoProfiler("faster_rcnn", image_inferencer, 0)
# # # # profiler.warm_up(input_tensor)
# # # # profiler.register_hooks(input_tensor)

# # # # # Optional: View summary
# # # # profiler.summary()

# # # # dets = []
# # # # # Step 4: Run actual profiling
# # # # for path in img_paths[1:5]:
# # # #     input_name = os.path.basename(path).split('.')[0]
# # # #     image = cv2.imread(path)
# # # #     det = profiler.run_inference(input_tensor, input_name)
# # # #     dets.append((os.path.basename(path), det['predictions'][0].pred_instances))


# # # # image_dets = []
# # # # for det in dets:    
# # # #     coco_result = image_output_to_coco(det[1], det[0])
# # # #     image_dets.extend(coco_result)

# # # # with open('coco_predictions.json', 'w') as f:
# # # #     json.dump(image_dets, f)

# # # # from pycocotools.coco import COCO
# # # # from pycocotools.cocoeval import COCOeval
# # # # import json


# # # bboxes, labels = get_2d_boxes("f97f711fff2b43fab0328cf4db040608", ["3", "4"])

# # # print(bboxes)
# # # print(labels)
# # # file_name = f"{data_base}/samples/CAM_BACK/n008-2018-08-30-15-16-55-0400__CAM_BACK__1535657124637558.jpg"
# # # draw_coco_bboxes_from_path(file_name, bboxes, labels)
# # # # nusc.render_sample("73ccc9c10a3547849b8ff130d50bac98")





# # from p_perf.utils import visualize_coco_predictions, load_sweep_sd
# # from p_perf.post_process.image_eval import generate_coco_gt
# # from p_perf.post_process.pseudo_gt import generate_pseudo_coco_gt
# # import json
# # from nuscenes import NuScenes

# # output_base = "/mmdetection3d_ros2/outputs/test"
# # image_pred_file = f"{output_base}/image_pred_0.json"
# # image_gt_file = f"{output_base}/image_gt_0.json"

# nusc = NuScenes(version='v1.0-mini', dataroot='/mmdetection3d_ros2/data/nuscenes', verbose=True)


# # delay_csv = f"{output_base}/delays_0.csv"
# # with open(image_gt_file, 'r') as f:
# #     data = json.load(f)

# # tokens = [img['token'] for img in data.get('images', []) if 'token' in img]
# # # generate_coco_gt(tokens, 'test.json', delay_csv, iob_thresh=0.2, visibilities=['2', '3', '4'])
# # generate_pseudo_coco_gt(nusc, tokens, None, None, None, delay_csv, 'test.json')

# # sample_tokens = load_sweep_sd(nusc, nusc.scene[0])
# # for i, token in enumerate(sample_tokens[:40]):
# #     if token in tokens:
# #         visualize_coco_predictions(token, image_pred_file, 'test.json', '/mmdetection3d_ros2/data/nuscenes', i)


# import pandas as pd
# from p_perf.utils import visualize_lidar_predictions
# from p_perf.post_process.lidar_eval import lidar_output_to_nusc_box, lidar_nusc_box_to_global
# import mmengine

# output_base = "/mmdetection3d_ros2/outputs/nusc_scene0"
# pred_json_path = f"{output_base}/lidar_pred_2.json"
# delay_path = f"{output_base}/delays_2.csv"
# # token = '4f792c8da81e4cb7aca1790654da1c27'

# with open(pred_json_path) as f:
#     pred_json = json.load(f)

# tokens = pred_json['results'].keys()
# for token in tokens:
#     sd = nusc.get('sample_data', token)
#     if sd['is_key_frame']:
#         img_path = os.path.join(nusc.dataroot, sd['filename'])
#         inferencer = LidarDet3DInferencer('pv_rcnn_8xb2-80e_kitti-3d-3class')
#         points = convert_to_kitti(nusc, token)
#         input_tensor = dict(points=np.array(points, dtype=np.float32))  # or your actual test input
#         det = inferencer(input_tensor, show=True, return_datasamples=True)

#         # print(det['predictions'][0].pred_instances_3d)
#         # boxes = lidar_output_to_nusc_box(det['predictions'][0].pred_instances_3d, token, 0.1, 'kitti')
#         # boxes = lidar_nusc_box_to_global(nusc, token, boxes)


#         # nusc_annos = {}
#         # annos = []
#         # for box in boxes:
#         #     nusc_anno = dict(
#         #         sample_token=token,
#         #         translation=box.center.tolist(),
#         #         size=box.wlh.tolist(),
#         #         rotation=box.orientation.elements.tolist(),
#         #         detection_score=box.score,
#         #         attribute_name='')
#         #     annos.append(nusc_anno)
#         # nusc_annos[token] = annos
        
#         # nusc_submission = {
#         #     'results': nusc_annos
#         # }        
#         # mmengine.dump(nusc_submission, 'test.json')

#         visualize_lidar_predictions(nusc, token, False, pred_json_path, delay_path)


# import pandas as pd
# import ast  # to safely parse the string tuple for lidar_model

# # Load parameter mapping
# param_df = pd.read_csv("/home/mg/pdnn/pPerf/outputs/nusc_centerpoint/param_mapping.csv")

# # Loop over all run indices
# for run_index in range(45):
#     # Load corresponding profiling CSV
#     csv_path = f"/home/mg/pdnn/pPerf/outputs/nusc_centerpoint/layer_timings_{run_index}.csv"
#     df = pd.read_csv(csv_path)

#     # Get image and lidar model names from param_mapping
#     image_model = param_df.loc[param_df["run_index"] == run_index, "image_model"].values[0]
#     lidar_model_raw = param_df.loc[param_df["run_index"] == run_index, "lidar_model"].values[0]
    
#     # Parse lidar_model if stored as a stringified tuple
#     try:
#         lidar_model = ast.literal_eval(lidar_model_raw)[0]
#     except:
#         lidar_model = lidar_model_raw  # fallback if already clean string

#     # Layers to modify
#     target_layers = ['lidar_encode', 'image_encode', 'lidar_decode', 'image_decode']

#     # Update rows accordingly
#     for idx, row in df.iterrows():
#         if row['Layer'] in target_layers:
#             original_model = row['Model']
#             if 'lidar' in row['Layer']:
#                 df.at[idx, 'Model'] = lidar_model
#             elif 'image' in row['Layer']:
#                 df.at[idx, 'Model'] = image_model
#             df.at[idx, 'Input'] = original_model

#     # Save the updated CSV
#     df.to_csv(csv_path, index=False)






import os.path as osp
import numpy as np
import torch
from copy import deepcopy
from typing import List, Dict
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from mmdet3d.apis import init_model
from mmengine.dataset import Compose
from mmdet3d.structures import get_box_type
from mmdet3d.datasets.transforms import LoadPointsFromFile
from mmengine.dataset import pseudo_collate

from p_perf.pPerf import pPerf


class BEVFormerInferencer:
    def __init__(self, config: str, checkpoint: str):
        self.model = init_model(config, checkpoint, device='cuda:0')
        self.cfg = self.model.cfg
        self.pipeline = Compose(deepcopy(self.cfg.test_dataloader.dataset.pipeline))
        self.box_type_3d, self.box_mode_3d = get_box_type(self.cfg.test_dataloader.dataset.box_type_3d)

        DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
        self.nusc = NuScenes(
                    version='v1.0-mini',
                    dataroot=DATA_ROOT,
                    verbose=False
                )


    def load_all_camera_sweeps(self, s_token):
        """
        Load all camera sweeps for all 6 camera channels in the given scene.

        Args:
            s_token (str): Scene token from nuScenes.

        Returns:
            Dict[str, Dict[str, dict]]:
                Outer dict keys = camera channels (e.g., 'CAM_FRONT'),
                Inner dict keys = sample_data tokens,
                Values = dicts with cam info fields (img_path, cam2img, etc.)
        """
        def transform_matrix(translation, rotation):
            from pyquaternion import Quaternion
            q = Quaternion(rotation)
            T = np.eye(4)
            T[:3, :3] = q.rotation_matrix
            T[:3, 3] = translation
            return T

        cam_keys = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        cam_sweeps = {cam: {} for cam in cam_keys}

        scene = self.nusc.get('scene', s_token)
        sample = self.nusc.get('sample', scene['first_sample_token'])

        # Get initial sample_data tokens for each camera
        initial_sd_tokens = {cam: sample['data'][cam] for cam in cam_keys}

        for cam in cam_keys:
            sd_token = initial_sd_tokens[cam]
            while sd_token:
                sd = self.nusc.get('sample_data', sd_token)
                cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                ep = self.nusc.get('ego_pose', sd['ego_pose_token'])

                cam2img = np.array(cs['camera_intrinsic'])
                cam2ego = transform_matrix(cs['translation'], cs['rotation'])
                ego2global_cam = transform_matrix(ep['translation'], ep['rotation'])
                global2ego = np.linalg.inv(ego2global_cam)
                ego2cam = np.linalg.inv(cam2ego)

                # Use the sample_token to get synchronized LiDAR pose
                lidar_sample = self.nusc.get('sample', sd['sample_token'])
                lidar_sd = self.nusc.get('sample_data', lidar_sample['data']['LIDAR_TOP'])
                lidar_cs = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
                lidar_ep = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])

                lidar2ego = transform_matrix(lidar_cs['translation'], lidar_cs['rotation'])
                ego2global_lidar = transform_matrix(lidar_ep['translation'], lidar_ep['rotation'])
                lidar2global = ego2global_lidar @ lidar2ego

                lidar2cam = ego2cam @ global2ego @ lidar2global

                cam_sweeps[cam][sd_token] = {
                    'img_path': osp.join(self.nusc.dataroot, sd['filename']),
                    'cam2img': cam2img.tolist(),
                    'sample_data_token': sd_token,
                    'timestamp': sd['timestamp'],
                    'cam2ego': cam2ego.tolist(),
                    'lidar2cam': lidar2cam.tolist()
                }

                if not sd['next']:
                    break
                sd_token = sd['next']

        return cam_sweeps

    def infer_single_sample(self, lidar_token: str, cam_tokens: dict, cam_sweeps):
        """
        Run inference using a LiDAR sample_data token and 6 image sample_data tokens.

        Args:
            lidar_token (str): NuScenes sample_data token for the LiDAR point cloud.
            cam_tokens (dict): Dict of camera type → sample_data token.

        Returns:
            Inference result.
        """
        # Resolve the LiDAR path using NuScenes API
        lidar_sd = self.nusc.get('sample_data', lidar_token)
        lidar_path = osp.join(self.nusc.dataroot, lidar_sd['filename'])

        # Build images_info['images'] dictionary
        images_info = {'images': {}}
        for cam_type, token in cam_tokens.items():
            assert cam_type in cam_sweeps, f"Unknown camera: {cam_type}"
            assert token in cam_sweeps[cam_type], f"Token not found for {cam_type}: {token}"

            sweep = cam_sweeps[cam_type][token]
            images_info['images'][cam_type] = {
                'img_path': sweep['img_path'],
                'cam2img': sweep['cam2img'],
                'sample_data_token': sweep['sample_data_token'],
                'timestamp': sweep['timestamp'],
                'cam2ego': sweep['cam2ego'],
                'lidar2cam': sweep['lidar2cam']
            }

        # Prepare the model input
        main_cam = 'CAM_FRONT'
        main_token = cam_tokens[main_cam]
        main_timestamp = cam_sweeps[main_cam][main_token]['timestamp']

        data = dict(
            lidar_points=dict(lidar_path=lidar_path),
            images={cam: cam_sweeps[cam][cam_tokens[cam]] for cam in cam_tokens},
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            timestamp=main_timestamp  # ✅ Add this line to fix the KeyError
        )

        data = self.pipeline(data)
        collated = pseudo_collate([data])

        with torch.no_grad():
            result = self.model.test_step(collated)[0]

        return result
    
    def __call__(self, lidar_token: str, cam_tokens: dict, cam_sweeps):
        """
        Makes the inferencer callable.

        Args:
            lidar_token (str): Sample_data token of the LiDAR point cloud.
            cam_tokens (dict): Dict of camera type → sample_data token.

        Returns:
            Inference result.
        """
        return self.infer_single_sample(lidar_token, cam_tokens, cam_sweeps)
        







config = '/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
chkpt = '/mmdetection3d_ros2/perf_ws/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'

inferencer = BEVFormerInferencer(config, chkpt)
cam_sweeps = inferencer.load_all_camera_sweeps('c5224b9b454b4ded9b5d2d2634bbda8a')

scene_token = 'c5224b9b454b4ded9b5d2d2634bbda8a'
scene = inferencer.nusc.get('scene', scene_token)
first_sample_token = scene['first_sample_token']
sample = inferencer.nusc.get('sample', first_sample_token)

# Step 2: Get camera and LiDAR sample_data tokens
cam_keys = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

cam_tokens = {cam: sample['data'][cam] for cam in cam_keys}
lidar_token = sample['data']['LIDAR_TOP']

result = inferencer(lidar_token, cam_tokens, cam_sweeps)

print(result.pred_instances_3d)

# profiler = pPerf('bev_former', inferencer, 0)
# profiler.register_hooks()


from nuscenes.nuscenes import NuScenes


DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
SCENE_TOKEN = 'cc8c0bf57f984915a77078b10eb33198'  # replace if needed

nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=False)

# 1. Load scene
scene = nusc.get('scene', SCENE_TOKEN)

# 2. Get first sample token
sample_token = scene['first_sample_token']
sample = nusc.get('sample', sample_token)

# 3. Get sample_data tokens
lidar_token = sample['data']['LIDAR_TOP']
cam_tokens = {
    cam: sample['data'][cam]
    for cam in [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
}

print(lidar_token)
print(cam_tokens)