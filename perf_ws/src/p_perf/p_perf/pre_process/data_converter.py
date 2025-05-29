import numpy as np
import os
from pyquaternion import Quaternion
import nuscenes.utils.geometry_utils as geoutils
from nuscenes.nuscenes import NuScenes
from mmdet3d.apis import LidarDet3DInferencer

def convert_lidar_token_to_kitti_array(nusc, lidar_token):
    # Load metadata
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = os.path.join(nusc.dataroot, lidar_data["filename"])
    scan = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))[:, :4]  # [x, y, z, intensity]
    scan[:, 3] = (scan[:, 3] - scan[:, 3].min()) / max(1e-5, scan[:, 3].ptp())
    scan[:, 1] *= -1  # Flip y-axis to match KITTI convention

    return scan


token = '197ffacc1b2440679975bf5e54d40b0a'
nusc = NuScenes(version='v1.0-mini', dataroot='/mmdetection3d_ros2/data/nuscenes', verbose=True)

points = convert_lidar_token_to_kitti_array(nusc, token)

inferencer_1 = LidarDet3DInferencer('pv_rcnn_8xb2-80e_kitti-3d-3class')

result = inferencer_1(dict(points=np.array(points, dtype=np.float32)), return_datasamples=True)
print(result['predictions'][0].pred_instances_3d.bboxes_3d)

sd = nusc.get('sample_data', token)
img_path = os.path.join(nusc.dataroot, sd['filename'])
points = np.fromfile(img_path, dtype=np.float32).reshape(-1, 5)
input_tensor = dict(points=np.array(points, dtype=np.float32))
inferencer_2 = LidarDet3DInferencer('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d')

result = inferencer_2(input_tensor)
print(result['predictions'][0])