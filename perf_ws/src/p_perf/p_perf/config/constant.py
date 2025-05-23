from nuscenes.nuscenes import NuScenes

# Lidar evaluations constants
lidar_classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
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
  }

dist_ths = [0.5, 1, 2, 4]

DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
nusc = NuScenes(
            version='v1.0-mini',
            dataroot=DATA_ROOT 
        )

# Image evaluation constants
coco_to_nusc = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'bicycle': 'bicycle',
    'motorcycle': 'motorcycle',
    'person': 'pedestrian',
}

image_classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'pedestrian']



