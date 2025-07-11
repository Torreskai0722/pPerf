from nuscenes.nuscenes import NuScenes

# Lidar evaluations constants
nus_lidar_classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
kitti_lidar_classes = ['pedestrian', 'bicycle', 'car']
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

dist_ths = [0.5, 1, 2, 4, 6]


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







image_models = [    
    'faster-rcnn_r50_fpn_1x_coco',      # TWO STAGE
    'yolov3_d53_mstrain-608_273e_coco', # ONE STAGE ANCHOR
    'yolox_x_8x8_300e_coco',            # ONE STAGE ANCHOR FREE
    'centernet_r18-dcnv2_8xb16-crop512-140e_coco',   # ONE STAGE ANCHOR FREE
    'detr_r50_8xb2-150e_coco'           # TRANSFORMER
]

lidar_models = [
    ('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'nus', 0.5),    # VOXEL BASED
    ('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'nus', 0.5),        # VOXEL BASED
    ('hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'nus', 0.5),      # VOXEL BASED
    ('point-rcnn_8xb2_kitti-3d-3class', 'kitti-3class', 0.2),            # POINT BASED
    ('3dssd_4x4_kitti-3d-car', 'kitti-car', 0.2),                       # POINT BASED
    ('pv_rcnn_8xb2-80e_kitti-3d-3class', 'kitti-3class', 0.2),             # POINT + VOXEL (Hybrid)
]

# Model name mappings for shorter display names
model_name_mappings = {
    # Image models
    'faster-rcnn_r50_fpn_1x_coco': 'FasterRCNN',
    'yolov3_d53_mstrain-608_273e_coco': 'YOLOv3',
    'yolox_x_8x8_300e_coco': 'YOLOX',
    'centernet_r18-dcnv2_8xb16-crop512-140e_coco': 'CenterNet',
    'detr_r50_8xb2-150e_coco': 'DETR',
    
    # Lidar models
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d': 'PointPillars',
    'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d': 'CenterPoint',
    'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d': 'HV-SSN',
    'point-rcnn_8xb2_kitti-3d-3class': 'PointRCNN',
    '3dssd_4x4_kitti-3d-car': '3DSSD',
    'pv_rcnn_8xb2-80e_kitti-3d-3class': 'PV-RCNN'
}



