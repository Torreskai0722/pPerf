import numpy as np
import os
from pyquaternion import Quaternion
import nuscenes.utils.geometry_utils as geoutils
from nuscenes.nuscenes import NuScenes
from mmdet3d.apis import LidarDet3DInferencer

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

    return scan_kitti



# token = '197ffacc1b2440679975bf5e54d40b0a'
# nusc = NuScenes(version='v1.0-mini', dataroot='/mmdetection3d_ros2/data/nuscenes', verbose=True)

# points = convert_lidar_token_to_kitti_array(nusc, token)

# inferencer_1 = LidarDet3DInferencer('3dssd_4x4_kitti-3d-car')

# result = inferencer_1(dict(points=np.array(points, dtype=np.float32)), return_datasamples=True, show=True)
# print(result['predictions'][0])

# sd = nusc.get('sample_data', token)
# img_path = os.path.join(nusc.dataroot, sd['filename'])
# points = np.fromfile(img_path, dtype=np.float32).reshape(-1, 5)
# input_tensor = dict(points=np.array(points, dtype=np.float32))
# inferencer_2 = LidarDet3DInferencer('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d')

# result = inferencer_2(input_tensor, show=True)
# print(result['predictions'][0])