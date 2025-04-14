import numpy as np
from mmdet3d.apis import LidarDet3DInferencer, inference_multi_modality_detector, init_model
from mmdet.apis import DetInferencer
from utils import list_filenames
import mmcv
import matplotlib.pyplot as plt
from mmdet.visualization import DetLocalVisualizer

lidar_paths = list_filenames('/mmdetection3d_ros2/dataset/nuscenes_mini/sweeps/LIDAR_TOP', 'bin')
img_paths = list_filenames('/mmdetection3d_ros2/dataset/nuscenes_mini/sweeps/CAM_FRONT', 'jpg')

# my_config = '/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
# my_checkpoint = '/mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'

# model = init_model(my_config, my_checkpoint)
# ann = "/mmdetection3d_ros2/dataset/nuscenes_mini/nuscenes_dbinfos_train.pkl"
# cam_type = 'CAM_FRONT'

# for lidar_path in lidar_paths[:5]:
#     for img_path in img_paths[:5]:
#         lidar_data = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
#         points = dict(points=lidar_data)
#         image_data = mmcv.imread(img_path)
#         result, data = inference_multi_modality_detector(model, lidar_data, img_path,
#                                                         ann, cam_type)
#         print(result)
#         continue


model = LidarDet3DInferencer('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d')

for path in lidar_paths[1:2]:
    lidar_data = np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    lidar_data = dict(points=path)
    result = model(lidar_data, show=True)

    print(result)
