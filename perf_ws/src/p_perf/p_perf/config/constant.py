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
    # 'yolox_x_8x8_300e_coco',            # ONE STAGE ANCHOR FREE
    'detr_r50_8xb2-150e_coco',           # TRANSFORMER
    'dino-4scale_r50_8xb2-12e_coco',
]

lidar_models = [
    ('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'nus'),    # VOXEL BASED
    ('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'nus'),        # VOXEL BASED
    # ('hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'nus'),      # VOXEL BASED
    # ('3dssd_4x4_kitti-3d-car', 'kitti-car'),                       # POINT BASED
    # ('pv_rcnn_8xb2-80e_kitti-3d-3class', 'kitti-3class'),             # POINT + VOXEL (Hybrid)
]

# Segmentation: use MMSeg model-zoo aliases. Config and weights are loaded from the
# MMSeg model zoo automatically (no ETH URLs or local BDD100K paths).
seg_models = [
    # ('cascade_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k', 'ins_seg'),  # Instance segmentation: CNN based
    # ('gcnet_mask_rcnn_r50_fpn_3x_ins_seg_bdd100k', 'ins_seg'),    # Instance segmentation: CNN based
    ('mask-rcnn_r50_fpn_1x_coco', 'ins_seg'),    # Instance segmentation: CNN based
    # ('panoptic_fpn_r50_fpn_1x_pan_seg_bdd100k', 'pan_seg'),       # Panoptic segmentation
    # ('upernet_swin-t_512x1024_40k_sem_seg_bdd100k', 'sem_seg'),    # Semantic segmentation: transformer based
    ('fcn_hr18s_4xb2-40k_cityscapes-512x1024', 'sem_seg'),    # Semantic segmentation : CNN based
    ('deeplabv3plus_r50-d8_4xb2-300k_mapillay_v1_65-1280x1280', 'sem_seg'),
    ('vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512', 'sem_seg'),
    ('swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512', 'sem_seg'),
]

# Model name mappings for shorter display names
model_name_mappings = {
    # Image models
    'faster-rcnn_r50_fpn_1x_coco': 'FasterRCNN',
    'yolov3_d53_mstrain-608_273e_coco': 'YOLOv3',
    'yolox_x_8x8_300e_coco': 'YOLOX',
    'centernet_r18-dcnv2_8xb16-crop512-140e_coco': 'CenterNet',
    'detr_r50_8xb2-150e_coco': 'DETR',
    'dino-4scale_r50_8xb2-12e_coco': 'DINO',
    # Lidar models
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d': 'PointPillars',
    'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d': 'CenterPoint',
    'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d': 'HV-SSN',
    'point-rcnn_8xb2_kitti-3d-3class': 'PointRCNN',
    '3dssd_4x4_kitti-3d-car': '3DSSD',
    'pv_rcnn_8xb2-80e_kitti-3d-3class': 'PV-RCNN',

    # Segmentation models (MMSeg model-zoo aliases; config/weights from zoo)
    'cascade_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k': 'CascadeMaskRCNN',
    'panoptic_fpn_r50_fpn_1x_pan_seg_bdd100k': 'PanopticFPN',
    'gcnet_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k': 'GCNet',
    'mask-rcnn_r50_fpn_1x_coco': 'MaskRCNN',
    'upernet_swin-t_512x1024_40k_sem_seg_bdd100k': 'SwinTransformer',
    'fcn_hr18s_4xb2-40k_cityscapes-512x1024': 'FCN',
    'deeplabv3plus_r50-d8_4xb2-300k_mapillay_v1_65-1280x1280': 'DeepLabV3+',
    'upernet_convnext-t_fp16_512x1024_40k_sem_seg_bdd100k': 'ConvNeXt',
    'vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512': 'ViT',
    'swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512': 'SwinTransformer',
}


scenes = [
    # '2f0e54af35964a3fb347359836bec035',  # ENVIRONMENT: Crowded city (scene 429)
    # '3a1850241080418b88dcee97c7d17ed7',  # ENVIRONMENT: Crowded city (scene 27)
    
    # 'bcb05cffb5814973a6cff4fbdca2b99b',  # ENVIRONMENT: Highspeed highway (scene 194)
    # '74d03e3b673f4a72a3f76e1b0c7840bd',  # ENVIRONMENT: Highspeed highway (scene 612)
    
    # '32185f91e68f4069ab3cdd2f4f1a4ff1',  # ENVIRONMENT: Suburban (scene 709)
    # '93032908bafd484ea10a39b0d5d3fff6',  # ENVIRONMENT: Suburban (scene 143)
    
    # '7ca5e90766dc4b7bb6a9b3b0d95c99da',  # WEATHER: Clear (scene 201)
    # 'bed8426a524d45afab05b19cf02386b2',  # WEATHER: Clear (scene 97)
    
    # '5fc7c5dfc56b4971b14005bc53f69908',  # WEATHER: Rainy (scene 389)
    # '0ced08ea43754420a23b2fbec667a763',  # WEATHER: Rainy (scene 364)
    
    # 'b519ec833e23450a8bd3340b67f2516b',  # LIGHTING: Day (scene 314)
    # '8edbc31083ab4fb187626e5b3c0411f7',  # LIGHTING: Day (scene 37)
    
    # 'f634de95cc7043b8b38ceaac67d472cf',  # LIGHTING: Night (scene 784)
    # 'b2119bd765a94b0cb705424800307bbf'   # LIGHTING: Night (scene 768)
  "a7d073bc435b4356a0a9a5ebfb61f229",
  "c164a8e8e8b8489f964f711f472789be",
  "938479858b4849dd8ceddc11c154082b",
  "e083627747d8499a8c4e94c7abf941e2",
  "fd4ef697de684d0a8e016a8f8ae61193",
  "9068766ee9374872a380fe75fcfb299e",
  "2422d2652f4f42298a389a6e0dfef307",
  "17302a41218442ffbb0b094adb0669ab",
  "82024d9f5f7f46dba436cd93c27173c7",
  "3045ed93c2534ec2a5cabea89b186bd9",
  "078e6d7a3d4a429bba404bf12c7a98f2",
  "2131d9fd2c48418081875019e35cf6a2",
  "d3b86ca0a17840109e9e049b3dd40037",
  "a2b64d02c5fa4b9bab671a97561b6b33",
  "d3d94f2ce3dc4db4b3ba6f4aa81c3987",
  "380ff00ec86447e3b986edc8e82ffba7",
  "e15260c4ed5b4505900d497ef92ddd2d",
  "d033fb0d79ea42b9a356fa5162172598",
  "7e8ff24069ff4023ac699669b2c920de",
  "a2b005c4dd654af48194ada18662c8ca",
  # "68e79a88244f447f993a72da444b29ba",
  # "ba76f4185aa34ed3bff467ac640da902",
  # "6746ba640c0e45c7961efbe6af51757e",
  # "0e7ede02718341558414865d5c604745",
  # "98fc76cc0e364efdaa7e86a36b666dda",
  # "221a67d95acf448ba863917ec4d32ed0",
  # "c5224b9b454b4ded9b5d2d2634bbda8a",
  # "c5b36e5275cc4a2285d095e87347c49a",
  # "6447a9c30b0140d783871b08104ae7b3",
  # "782f9486adbd4546a492e69b6697ef70",
    
  "c8e6213038884d05907b1edd7e55552e",
  "f8ef029224b84d14843db98a76a7f4a1",
  "e0a212aafd574781b122a6ba66599a1e",
  "eac3102e4cc24d4b95532bcc711a902f",
  "8857cf15fa7049a6b000490835d3b9fc",
  "0d2cc345342a460e94ff54748338ac22",
  "f9e460f092c94466b1211704b5a8859d",
  "2ce2df158372461fa91ce77455656f81",
  "fabc1800b9ed432d96aee5ecf0156831",
  "81227a12571240e0b9d537165f57e177",
  "6207269cb9474c14b01f9610b2ee30aa",
  "2d8e61ea27fd419ba9b708488e3f2f10",
  "7463b2d752f0464f803b07a945ee7c5a",
  "28d385e6db0e495da3a606b58e2432f0",
  "47cfdbcf14fe47d2a3b9e7e5b46d8a36",
  "49eb09ab4c4142268131125d6e619a0c",
  "aedcd3cf7c4a49d7a4a43ab7443a9eb1",
  "21a7ba093614493b83838b9656b3558d",
  "4bb9626f13184da7a67ba1b241c19558",
  "e60ef590e3614187b7800db3e5284e1a",
  # "e7ef871f77f44331aefdebc24ec034b7",
  # "de943e246dad4ad686de98008a634ecf",
  # "0c601ff2bf004fccafec366b08bf29e2",
  # "1d4db80d13f342aba4881b38099bc4b7",
  # "9047b53fd41540649dce014a128cbe1b",
  # "6e81ee0f64274490a403bbd6482c2bf9",
  # "7ad94b6e54f641208bf59cba6bb55220",
  # "7365495b74464629813b41eacdb711af",
  # "ce2d6bdc33084dc1a2780f41f6740e06",
  # "fcb40dc8514444c49b1fdc06f80a123b",
]


scenes_mapping = {
  # "2f0e54af35964a3fb347359836bec035": "TEST",
  # Low complexity scenes
  "c8e6213038884d05907b1edd7e55552e": "Low_1",
  "f8ef029224b84d14843db98a76a7f4a1": "Low_2",
  "e0a212aafd574781b122a6ba66599a1e": "Low_3",
  "eac3102e4cc24d4b95532bcc711a902f": "Low_4",
  "8857cf15fa7049a6b000490835d3b9fc": "Low_5",
  "0d2cc345342a460e94ff54748338ac22": "Low_6",
  "f9e460f092c94466b1211704b5a8859d": "Low_7",
  "2ce2df158372461fa91ce77455656f81": "Low_8",
  "fabc1800b9ed432d96aee5ecf0156831": "Low_9",
  "81227a12571240e0b9d537165f57e177": "Low_10",
  "6207269cb9474c14b01f9610b2ee30aa": "Low_11",
  "2d8e61ea27fd419ba9b708488e3f2f10": "Low_12",
  "7463b2d752f0464f803b07a945ee7c5a": "Low_13",
  "28d385e6db0e495da3a606b58e2432f0": "Low_14",
  "47cfdbcf14fe47d2a3b9e7e5b46d8a36": "Low_15",
  "49eb09ab4c4142268131125d6e619a0c": "Low_16",
  "aedcd3cf7c4a49d7a4a43ab7443a9eb1": "Low_17",
  "21a7ba093614493b83838b9656b3558d": "Low_18",
  "4bb9626f13184da7a67ba1b241c19558": "Low_19",
  "e60ef590e3614187b7800db3e5284e1a": "Low_20",
  "e7ef871f77f44331aefdebc24ec034b7": "Low_21",
  "de943e246dad4ad686de98008a634ecf": "Low_22",
  "0c601ff2bf004fccafec366b08bf29e2": "Low_23",
  "1d4db80d13f342aba4881b38099bc4b7": "Low_24",
  "9047b53fd41540649dce014a128cbe1b": "Low_25",
  "6e81ee0f64274490a403bbd6482c2bf9": "Low_26",
  "7ad94b6e54f641208bf59cba6bb55220": "Low_27",
  "7365495b74464629813b41eacdb711af": "Low_28",
  "ce2d6bdc33084dc1a2780f41f6740e06": "Low_29",
  "fcb40dc8514444c49b1fdc06f80a123b": "Low_30",

  # High (Extreme) complexity scenes
  "a7d073bc435b4356a0a9a5ebfb61f229": "High_1",
  "c164a8e8e8b8489f964f711f472789be": "High_2",
  "938479858b4849dd8ceddc11c154082b": "High_3",
  "e083627747d8499a8c4e94c7abf941e2": "High_4",
  "fd4ef697de684d0a8e016a8f8ae61193": "High_5",
  "9068766ee9374872a380fe75fcfb299e": "High_6",
  "2422d2652f4f42298a389a6e0dfef307": "High_7",
  "17302a41218442ffbb0b094adb0669ab": "High_8",
  "82024d9f5f7f46dba436cd93c27173c7": "High_9",
  "3045ed93c2534ec2a5cabea89b186bd9": "High_10",
  "078e6d7a3d4a429bba404bf12c7a98f2": "High_11",
  "2131d9fd2c48418081875019e35cf6a2": "High_12",
  "d3b86ca0a17840109e9e049b3dd40037": "High_13",
  "a2b64d02c5fa4b9bab671a97561b6b33": "High_14",
  "d3d94f2ce3dc4db4b3ba6f4aa81c3987": "High_15",
  "380ff00ec86447e3b986edc8e82ffba7": "High_16",
  "e15260c4ed5b4505900d497ef92ddd2d": "High_17",
  "d033fb0d79ea42b9a356fa5162172598": "High_18",
  "7e8ff24069ff4023ac699669b2c920de": "High_19",
  "a2b005c4dd654af48194ada18662c8ca": "High_20",
  "68e79a88244f447f993a72da444b29ba": "High_21",
  "ba76f4185aa34ed3bff467ac640da902": "High_22",
  "6746ba640c0e45c7961efbe6af51757e": "High_23",
  "0e7ede02718341558414865d5c604745": "High_24",
  "98fc76cc0e364efdaa7e86a36b666dda": "High_25",
  "221a67d95acf448ba863917ec4d32ed0": "High_26",
  "c5224b9b454b4ded9b5d2d2634bbda8a": "High_27",
  "c5b36e5275cc4a2285d095e87347c49a": "High_28",
  "6447a9c30b0140d783871b08104ae7b3": "High_29",
  "782f9486adbd4546a492e69b6697ef70": "High_30",
}



def generate_image_lidar_combinations():
    """
    Generate all combinations of image and lidar models.
    
    Returns:
        list: List of tuples (image_model, lidar_model) where:
            - image_model: str (image model name)
            - lidar_model: tuple (lidar_model_name, lidar_mode)
    """
    from itertools import product
    return list(product(image_models, lidar_models))


def generate_image_lidar_seg_combinations():
    """
    Generate all combinations of image, lidar, and segmentation models.
    
    Returns:
        list: List of tuples (image_model, lidar_model, seg_model) where:
            - image_model: str (image model name)
            - lidar_model: tuple (lidar_model_name, lidar_mode)
            - seg_model: tuple (seg_model_name, seg_mode)
    """
    from itertools import product
    return list(product(image_models, lidar_models, seg_models))


def get_abbreviated_name(model_name):
    """
    Get abbreviated name for a model from model_name_mappings.
    
    Args:
        model_name: Full model name or tuple (model_name, mode)
        
    Returns:
        str: Abbreviated name if found in mappings, otherwise returns the original name
    """
    # Handle tuple input (for lidar/seg models)
    if isinstance(model_name, tuple):
        model_name = model_name[0]
    
    return model_name_mappings.get(model_name, model_name)


def get_abbreviated_scene(scene_token):
    """
    Get abbreviated scene name from scenes_mapping.
    
    Args:
        scene_token: Full scene token string
        
    Returns:
        str: Abbreviated scene name if found, otherwise returns the original token
    """
    return scenes_mapping.get(scene_token, scene_token)


def get_full_model_name(abbreviated_name):
    """
    Get full model name from abbreviated name (reverse lookup).
    
    Args:
        abbreviated_name: Abbreviated model name
        
    Returns:
        str: Full model name if found, otherwise returns the abbreviated name
    """
    # Create reverse mapping
    reverse_mapping = {v: k for k, v in model_name_mappings.items()}
    return reverse_mapping.get(abbreviated_name, abbreviated_name)


def get_full_scene_token(abbreviated_scene):
    """
    Get full scene token from abbreviated scene name (reverse lookup).
    
    Args:
        abbreviated_scene: Abbreviated scene name
        
    Returns:
        str: Full scene token if found, otherwise returns the abbreviated name
    """
    # Create reverse mapping
    reverse_mapping = {v: k for k, v in scenes_mapping.items()}
    return reverse_mapping.get(abbreviated_scene, abbreviated_scene)


