# import open3d as o3d
# import numpy as np
# from pyquaternion import Quaternion
# from nuscenes.utils.data_classes import LidarPointCloud
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import Box as NuScenesBox
# from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
#                                 bbox3d2result, xywhr2xyxyr)
# from nuscenes.eval.common.data_classes import EvalBoxes
# from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
# from nuscenes.eval.detection.data_classes import DetectionMetricData
# from pyquaternion import Quaternion
# import numpy as np
# from typing import List
# from nuscenes.eval.detection.data_classes import (
#     DetectionBox,
#     DetectionConfig,
#     DetectionMetricDataList,
#     DetectionMetrics,
# )
# from nuscenes.eval.detection.utils import category_to_detection_name
# from nuscenes.eval.detection.algo import calc_ap





# def interpolate_gt(nusc, sample_data_token: str, box_cls=DetectionBox):
#     """
#     Generate interpolated or fallback ground truth boxes for a given sample_data_token.
#     Falls back to prev or next keyframe if interpolation is not possible.
#     """
#     sd = nusc.get('sample_data', sample_data_token)
#     timestamp = sd['timestamp']

#     box_list = []
#     instance_tokens = []

#     if sd['is_key_frame']:
#         sample = nusc.get('sample', sd['sample_token'])
#         annos = [nusc.get('sample_annotation', tok) for tok in sample['anns']]

#         for a in annos:
#             detection_name = category_to_detection_name(a['category_name'])
#             if detection_name is None:
#                 continue

#             box_list.append(box_cls(
#                 sample_token=sample_data_token,
#                 translation=a['translation'],
#                 size=a['size'],
#                 rotation=a['rotation'],
#                 velocity=a.get('velocity', [0.0, 0.0]),
#                 detection_name=detection_name,
#                 attribute_name=''  # optional
#             ))
#             instance_tokens.append(a['instance_token'])

#         return box_list, instance_tokens

#     # Walk backward to find previous keyframe
#     prev_sd_token = sd['prev']
#     prev_keyframe = None
#     while prev_sd_token:
#         prev_sd = nusc.get('sample_data', prev_sd_token)
#         if prev_sd['is_key_frame']:
#             prev_keyframe = prev_sd
#             break
#         prev_sd_token = prev_sd['prev']

#     # Walk forward to find next keyframe
#     next_sd_token = sd['next']
#     next_keyframe = None
#     while next_sd_token:
#         next_sd = nusc.get('sample_data', next_sd_token)
#         if next_sd['is_key_frame']:
#             next_keyframe = next_sd
#             break
#         next_sd_token = next_sd['next']

#     if prev_keyframe and next_keyframe:
#         # Interpolation case
#         t0, t1 = prev_keyframe['timestamp'], next_keyframe['timestamp']
#         alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

#         prev_sample = nusc.get('sample', prev_keyframe['sample_token'])
#         next_sample = nusc.get('sample', next_keyframe['sample_token'])

#         prev_annos = [nusc.get('sample_annotation', tok) for tok in prev_sample['anns']]
#         next_annos = [nusc.get('sample_annotation', tok) for tok in next_sample['anns']]

#         prev_map = {a['instance_token']: a for a in prev_annos}
#         next_map = {a['instance_token']: a for a in next_annos}

#         common_instances = set(prev_map.keys()) & set(next_map.keys())

#         for inst in common_instances:
#             a0, a1 = prev_map[inst], next_map[inst]

#             t0 = np.array(a0['translation'])
#             t1 = np.array(a1['translation'])
#             center = (1 - alpha) * t0 + alpha * t1

#             s0 = np.array(a0['size'])
#             s1 = np.array(a1['size'])
#             size = (1 - alpha) * s0 + alpha * s1

#             q0 = Quaternion(a0['rotation'])
#             q1 = Quaternion(a1['rotation'])
#             rotation = Quaternion.slerp(q0, q1, amount=alpha)

#             v0 = np.array(a0.get('velocity', [0, 0]))
#             v1 = np.array(a1.get('velocity', [0, 0]))
#             velocity = (1 - alpha) * v0 + alpha * v1

#             detection_name = category_to_detection_name(a0['category_name'])
#             if detection_name is None:
#                 continue

#             box_list.append(box_cls(
#                 sample_token=sample_data_token,
#                 translation=center.tolist(),
#                 size=size.tolist(),
#                 rotation=rotation.elements.tolist(),
#                 velocity=velocity.tolist(),
#                 detection_name=detection_name,
#                 attribute_name=''
#             ))
#             instance_tokens.append(inst)

#         return box_list, instance_tokens

#     # Fallback case
#     fallback_frame = prev_keyframe or next_keyframe
#     fallback_sample = nusc.get('sample', fallback_frame['sample_token'])
#     annos = [nusc.get('sample_annotation', tok) for tok in fallback_sample['anns']]

#     for a in annos:
#         detection_name = category_to_detection_name(a['category_name'])
#         if detection_name is None:
#             continue

#         box_list.append(box_cls(
#             sample_token=sample_data_token,
#             translation=a['translation'],
#             size=a['size'],
#             rotation=a['rotation'],
#             velocity=a.get('velocity', [0.0, 0.0]),
#             detection_name=detection_name,
#             attribute_name=''
#         ))
#         instance_tokens.append(a['instance_token'])

#     return box_list, instance_tokens



# def create_o3d_bbox(center, size, rotation, color=[1, 0, 0]):
#     """Create Open3D wireframe box given center, size, and quaternion rotation."""
#     bbox = o3d.geometry.OrientedBoundingBox()
#     bbox.center = center
#     bbox.extent = size
#     bbox.R = Quaternion(rotation).rotation_matrix
#     bbox.color = color
#     return bbox

# def transform_box_global_to_lidar(nusc, box, sample_data_token):
#     """
#     Transforms DetectionBox from global frame to lidar frame and fixes orientation misalignment.
#     """
#     import copy
#     transformed_box = copy.deepcopy(box)

#     sd = nusc.get('sample_data', sample_data_token)
#     cs_record = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
#     pose_record = nusc.get('ego_pose', sd['ego_pose_token'])

#     # Get quaternions
#     global_to_ego_rot = Quaternion(pose_record['rotation']).inverse
#     ego_to_sensor_rot = Quaternion(cs_record['rotation']).inverse

#     # --- Translate ---
#     box_center = np.array(box.translation)
#     box_center = global_to_ego_rot.rotate(box_center - np.array(pose_record['translation']))
#     box_center = ego_to_sensor_rot.rotate(box_center - np.array(cs_record['translation']))
#     transformed_box.translation = box_center.tolist()

#     # --- Rotate ---
#     box_rot = Quaternion(box.rotation)
#     box_rot = global_to_ego_rot * box_rot
#     box_rot = ego_to_sensor_rot * box_rot

#     # ✅ Apply fixed Z correction to align with LiDAR heading
#     z_correction = Quaternion(axis=[0, 0, 1], radians=np.pi / 2)
#     box_rot = z_correction * box_rot

#     transformed_box.rotation = box_rot.elements.tolist()

#     return transformed_box



# def visualize_sample(nusc, sample_data_token: str, boxes: List, title: str = "Scene"):
#     """Visualize point cloud and bounding boxes for a sample_data_token."""
#     for i in range(len(boxes)):
#         boxes[i] = transform_box_global_to_lidar(nusc, boxes[i], sample_data_token)
#     sd = nusc.get('sample_data', sample_data_token)
#     pc_path = nusc.get_sample_data_path(sample_data_token)
#     pc = LidarPointCloud.from_file(pc_path).points[:3].T

#     # Point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc)
#     pcd.paint_uniform_color([0.5, 0.5, 0.5])

#     # Bounding boxes
#     bbox_list = []
#     for box in boxes:
#         bbox = create_o3d_bbox(
#             center=np.array(box.translation),
#             size=np.array(box.size),
#             rotation=box.rotation
#         )
#         bbox_list.append(bbox)

#     # Visualize
#     o3d.visualization.draw_geometries([pcd] + bbox_list, window_name=title)

# nusc = NuScenes(
#     version='v1.0-mini',            # or 'v1.0-mini', 'v1.0-test', etc.
#     dataroot='/mmdetection3d_ros2/data/nuscenes',  # replace with the actual path
#     verbose=True                        # optional, shows loading progress
# )







# sample_data_token = '60a854350d974579b303262662ed1928'
# # Current frame
# interpolated_boxes, _ = interpolate_gt(nusc, sample_data_token)
# visualize_sample(nusc, sample_data_token, interpolated_boxes, title="Interpolated GT Boxes")

# # Previous keyframe
# prev_sd_token = nusc.get('sample_data', sample_data_token)['prev']
# while prev_sd_token:
#     prev_sd = nusc.get('sample_data', prev_sd_token)
#     if prev_sd['is_key_frame']:
#         break
#     prev_sd_token = prev_sd['prev']

# if prev_sd_token:
#     prev_sample = nusc.get('sample', prev_sd['sample_token'])
#     prev_annos = [nusc.get('sample_annotation', tok) for tok in prev_sample['anns']]
#     prev_boxes = [DetectionBox(
#         sample_token=sample_data_token,
#         translation=a['translation'],
#         size=a['size'],
#         rotation=a['rotation'],
#         velocity=a.get('velocity', [0.0, 0.0]),
#         detection_name=category_to_detection_name(a['category_name']),
#         attribute_name=''
#     ) for a in prev_annos if category_to_detection_name(a['category_name']) is not None]
    
#     visualize_sample(nusc, prev_sd_token, prev_boxes, title="Previous Keyframe Boxes")

# # Next keyframe
# next_sd_token = nusc.get('sample_data', sample_data_token)['next']
# while next_sd_token:
#     next_sd = nusc.get('sample_data', next_sd_token)
#     if next_sd['is_key_frame']:
#         break
#     next_sd_token = next_sd['next']

# if next_sd_token:
#     next_sample = nusc.get('sample', next_sd['sample_token'])
#     next_annos = [nusc.get('sample_annotation', tok) for tok in next_sample['anns']]
#     next_boxes = [DetectionBox(
#         sample_token=sample_data_token,
#         translation=a['translation'],
#         size=a['size'],
#         rotation=a['rotation'],
#         velocity=a.get('velocity', [0.0, 0.0]),
#         detection_name=category_to_detection_name(a['category_name']),
#         attribute_name=''
#     ) for a in next_annos if category_to_detection_name(a['category_name']) is not None]

#     visualize_sample(nusc, next_sd_token, next_boxes, title="Next Keyframe Boxes")


from nuscenes.nuscenes import NuScenes
from collections import defaultdict
import numpy as np
import json

nusc = NuScenes(
    version='v1.0-mini',            # or 'v1.0-mini', 'v1.0-test', etc.
    dataroot='/home/mg/pdnn/pPerf/data/nuscenes'  # replace with the actual path
)

instance_stats = []

for instance in nusc.instance:
    ann_token = instance['first_annotation_token']
    timestamps = []

    # Grab category once from the first annotation
    first_ann = nusc.get('sample_annotation', ann_token)

    while ann_token:
        ann = nusc.get('sample_annotation', ann_token)
        sample = nusc.get('sample', ann['sample_token'])
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])  # Get timestamp via lidar
        timestamps.append(sd['timestamp'])
        ann_token = ann['next']

    timestamps.sort()
    duration = (timestamps[-1] - timestamps[0]) / 1e6 if len(timestamps) > 1 else 0.0

    instance_stats.append({
        'instance_token': instance['token'],
        'num_samples': len(timestamps),
        'duration_sec': duration
    })

instance_stats.sort(key=lambda x: x['duration_sec'], reverse=True)

instance_count = "/home/mg/pdnn/pPerf/perf_ws/dist_0.5.json"
with open(instance_count, 'r') as file:
    data = json.load(file)

# Show some results
for stat in instance_stats:
    if stat['duration_sec'] > 10 and stat['instance_token'] in data.keys():
        print(stat)
