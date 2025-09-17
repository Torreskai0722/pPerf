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
    # 'yolov3_d53_mstrain-608_273e_coco', # ONE STAGE ANCHOR
    # 'yolox_x_8x8_300e_coco',            # ONE STAGE ANCHOR FREE
    # 'detr_r50_8xb2-150e_coco'           # TRANSFORMER
]

lidar_models = [
    # ('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'nus'),    # VOXEL BASED
    ('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'nus'),        # VOXEL BASED
    # ('hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'nus'),      # VOXEL BASED
    ('3dssd_4x4_kitti-3d-car', 'kitti-car'),                       # POINT BASED
    # ('pv_rcnn_8xb2-80e_kitti-3d-3class', 'kitti-3class'),             # POINT + VOXEL (Hybrid)
]

seg_models = [
    ('cascade_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k', 'ins_seg'),  # Instance segmentation: CNN based
    # ('gcnet_mask_rcnn_r50_fpn_3x_ins_seg_bdd100k', 'ins_seg'),    # Instance segmentation: CNN based
    # ('mask_rcnn_hrnetv2p_w18_1x_ins_seg_bdd100k', 'ins_seg'),    # Instance segmentation: CNN based
    # ('panoptic_fpn_r50_fpn_1x_pan_seg_bdd100k', 'pan_seg'),       # Panoptic segmentation
    # ('upernet_swin-t_512x1024_40k_sem_seg_bdd100k', 'sem_seg'),    # Semantic segmentation: transformer based
    # ('fcn_hr48_512x1024_80k_sem_seg_bdd100k', 'sem_seg'),    # Semantic segmentation : CNN based
    # ('deeplabv3+_r50-d8_512x1024_40k_sem_seg_bdd100k', 'sem_seg'),    # Semantic segmentation: CNN based
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
    'pv_rcnn_8xb2-80e_kitti-3d-3class': 'PV-RCNN',

    # Segmentation models
    'cascade_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k': 'CascadeMaskRCNN',
    'panoptic_fpn_r50_fpn_1x_pan_seg_bdd100k': 'PanopticFPN',
    'gcnet_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k': 'GCNet',
    'mask_rcnn_hrnetv2p_w18_1x_ins_seg_bdd100k': 'HRNet',
    'upernet_swin-t_512x1024_40k_sem_seg_bdd100k': 'SwinTransformer',
    'fcn_hr48_512x1024_80k_sem_seg_bdd100k': 'FCN',
    'deeplabv3+_r50-d8_512x1024_40k_sem_seg_bdd100k': 'DeepLabV3+',
    'upernet_convnext-t_fp16_512x1024_40k_sem_seg_bdd100k': 'ConvNeXt'
}


scenes = [
    '2f0e54af35964a3fb347359836bec035',  # ENVIRONMENT: Crowded city (scene 429)
    '3a1850241080418b88dcee97c7d17ed7',  # ENVIRONMENT: Crowded city (scene 27)
    
    'bcb05cffb5814973a6cff4fbdca2b99b',  # ENVIRONMENT: Highspeed highway (scene 194)
    '74d03e3b673f4a72a3f76e1b0c7840bd',  # ENVIRONMENT: Highspeed highway (scene 612)
    
    '32185f91e68f4069ab3cdd2f4f1a4ff1',  # ENVIRONMENT: Suburban (scene 709)
    '93032908bafd484ea10a39b0d5d3fff6',  # ENVIRONMENT: Suburban (scene 143)
    
    # '7ca5e90766dc4b7bb6a9b3b0d95c99da',  # WEATHER: Clear (scene 201)
    # 'bed8426a524d45afab05b19cf02386b2',  # WEATHER: Clear (scene 97)
    
    # '5fc7c5dfc56b4971b14005bc53f69908',  # WEATHER: Rainy (scene 389)
    # '0ced08ea43754420a23b2fbec667a763',  # WEATHER: Rainy (scene 364)
    
    # 'b519ec833e23450a8bd3340b67f2516b',  # LIGHTING: Day (scene 314)
    # '8edbc31083ab4fb187626e5b3c0411f7',  # LIGHTING: Day (scene 37)
    
    # 'f634de95cc7043b8b38ceaac67d472cf',  # LIGHTING: Night (scene 784)
    # 'b2119bd765a94b0cb705424800307bbf'   # LIGHTING: Night (scene 768)
]



