import os
from contextlib import redirect_stdout
import random
from pyquaternion import Quaternion
import numpy as np
from nuscenes.utils.data_classes import Box as NuScenesBox
from mmdet3d.structures import LiDARInstance3DBoxes

from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.utils import category_to_detection_name

LABELS = {0:1, 2:3, 7:4, 5:5, 6:6, 3:7, 1:8, 9:9}   # label id in coco : lable id in bdd100k 

# MODEL_LIST = ['atss_r50_fpn_1x_coco', 'atss_r101_fpn_1x_coco', 'autoassign_r50_fpn_8x2_1x_coco', 'faster_rcnn_r50_fpn_carafe_1x_coco', 'mask_rcnn_r50_fpn_carafe_1x_coco', 'cascade_rcnn_r50_caffe_fpn_1x_coco', 'cascade_rcnn_r50_fpn_1x_coco', 'cascade_rcnn_r50_fpn_20e_coco', 'cascade_rcnn_r101_caffe_fpn_1x_coco', 'cascade_rcnn_r101_fpn_1x_coco', 'cascade_rcnn_r101_fpn_20e_coco', 'cascade_rcnn_x101_32x4d_fpn_1x_coco', 'cascade_rcnn_x101_32x4d_fpn_20e_coco', 'cascade_rcnn_x101_64x4d_fpn_1x_coco', 'cascade_rcnn_x101_64x4d_fpn_20e_coco', 'cascade_mask_rcnn_r50_caffe_fpn_1x_coco', 'cascade_mask_rcnn_r50_fpn_1x_coco', 'cascade_mask_rcnn_r50_fpn_20e_coco', 'cascade_mask_rcnn_r101_caffe_fpn_1x_coco', 'cascade_mask_rcnn_r101_fpn_1x_coco', 'cascade_mask_rcnn_r101_fpn_20e_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_1x_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_20e_coco', 'cascade_mask_rcnn_x101_64x4d_fpn_1x_coco', 'cascade_mask_rcnn_x101_64x4d_fpn_20e_coco', 'cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_r50_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_r101_caffe_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_r101_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_x101_32x8d_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco', 'crpn_fast_rcnn_r50_caffe_fpn_1x_coco', 'crpn_faster_rcnn_r50_caffe_fpn_1x_coco', 'centernet_resnet18_dcnv2_140e_coco', 'centernet_resnet18_140e_coco', 'centripetalnet_hourglass104_mstest_16x6_210e_coco', 'cornernet_hourglass104_mstest_10x5_210e_coco', 'cornernet_hourglass104_mstest_8x6_210e_coco', 'cornernet_hourglass104_mstest_32x3_210e_coco', 'faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco', 'faster_rcnn_r50_fpn_dpool_1x_coco', 'faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco', 'faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco', 'mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco', 'mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco', 
#                     'mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco', 'cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco', 'cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco', 'cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco', 'cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco', 'faster_rcnn_r50_fpn_mdpool_1x_coco', 'mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco', 'mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco', 'deformable_detr_r50_16x2_50e_coco', 'deformable_detr_refine_r50_16x2_50e_coco', 'deformable_detr_twostage_refine_r50_16x2_50e_coco', 'cascade_rcnn_r50_rfp_1x_coco', 'cascade_rcnn_r50_sac_1x_coco', 'detectors_cascade_rcnn_r50_1x_coco', 'htc_r50_rfp_1x_coco', 'htc_r50_sac_1x_coco', 'detectors_htc_r50_1x_coco', 'detr_r50_8x2_150e_coco', 'dh_faster_rcnn_r50_fpn_1x_coco', 'atss_r50_caffe_fpn_dyhead_1x_coco', 'atss_r50_fpn_dyhead_1x_coco', 'dynamic_rcnn_r50_fpn_1x_coco', 'retinanet_effb3_fpn_crop896_8x4_1x_coco', 'faster_rcnn_r50_fpn_attention_1111_1x_coco', 'faster_rcnn_r50_fpn_attention_0010_1x_coco', 'faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco', 'faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco', 'faster_rcnn_r50_caffe_c4_1x_coco', 'faster_rcnn_r50_caffe_c4_mstrain_1x_coco', 'faster_rcnn_r50_caffe_dc5_1x_coco', 'faster_rcnn_r50_caffe_fpn_1x_coco', 'faster_rcnn_r50_fpn_1x_coco', 'faster_rcnn_r50_fpn_fp16_1x_coco', 'faster_rcnn_r50_fpn_2x_coco', 'faster_rcnn_r101_caffe_fpn_1x_coco', 'faster_rcnn_r101_fpn_1x_coco', 'faster_rcnn_r101_fpn_2x_coco', 'faster_rcnn_x101_32x4d_fpn_1x_coco', 'faster_rcnn_x101_32x4d_fpn_2x_coco', 'faster_rcnn_x101_64x4d_fpn_1x_coco', 'faster_rcnn_x101_64x4d_fpn_2x_coco', 'faster_rcnn_r50_fpn_iou_1x_coco', 'faster_rcnn_r50_fpn_giou_1x_coco', 'faster_rcnn_r50_fpn_bounded_iou_1x_coco', 'faster_rcnn_r50_caffe_dc5_mstrain_1x_coco', 'faster_rcnn_r50_caffe_dc5_mstrain_3x_coco', 'faster_rcnn_r50_caffe_fpn_mstrain_2x_coco', 'faster_rcnn_r50_caffe_fpn_mstrain_3x_coco', 
#                     'faster_rcnn_r50_fpn_mstrain_3x_coco', 'faster_rcnn_r101_caffe_fpn_mstrain_3x_coco', 'faster_rcnn_r101_fpn_mstrain_3x_coco', 'faster_rcnn_x101_32x4d_fpn_mstrain_3x_coco', 'faster_rcnn_x101_32x8d_fpn_mstrain_3x_coco', 'faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco', 'faster_rcnn_r50_fpn_tnr-pretrain_1x_coco', 'fcos_r50_caffe_fpn_gn-head_1x_coco', 'fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco', 'fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco', 'fcos_r101_caffe_fpn_gn-head_1x_coco', 'fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco', 'fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco', 'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco', 
#                     'fovea_r50_fpn_4x4_1x_coco', 'fovea_r50_fpn_4x4_2x_coco', 'fovea_align_r50_fpn_gn-head_4x4_2x_coco', 'fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco', 'fovea_r101_fpn_4x4_1x_coco', 'fovea_r101_fpn_4x4_2x_coco', 'fovea_align_r101_fpn_gn-head_4x4_2x_coco', 'fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco', 'faster_rcnn_r50_fpg_crop640_50e_coco', 'faster_rcnn_r50_fpg-chn128_crop640_50e_coco', 'mask_rcnn_r50_fpg_crop640_50e_coco', 'mask_rcnn_r50_fpg-chn128_crop640_50e_coco', 'retinanet_r50_fpg_crop640_50e_coco', 'retinanet_r50_fpg-chn128_crop640_50e_coco', 'retinanet_free_anchor_r50_fpn_1x_coco', 'retinanet_free_anchor_r101_fpn_1x_coco', 'retinanet_free_anchor_x101_32x4d_fpn_1x_coco', 'fsaf_r50_fpn_1x_coco', 'fsaf_r101_fpn_1x_coco', 'fsaf_x101_64x4d_fpn_1x_coco', 'mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco', 'mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco', 'mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco', 'mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco', 'mask_rcnn_r50_fpn_syncbn-backbone_1x_coco', 'mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco', 'mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco', 'mask_rcnn_r101_fpn_syncbn-backbone_1x_coco', 'mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco', 'mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco', 'mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco', 'mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco', 'mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco', 'cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco', 
#                     'cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco', 'gfl_r50_fpn_1x_coco', 'gfl_r50_fpn_mstrain_2x_coco', 'gfl_r101_fpn_mstrain_2x_coco', 'gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco', 'gfl_x101_32x4d_fpn_mstrain_2x_coco', 'gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco', 'retinanet_ghm_r50_fpn_1x_coco', 'retinanet_ghm_r101_fpn_1x_coco', 'retinanet_ghm_x101_32x4d_fpn_1x_coco', 'retinanet_ghm_x101_64x4d_fpn_1x_coco', 'mask_rcnn_r50_fpn_gn-all_2x_coco', 'mask_rcnn_r50_fpn_gn-all_3x_coco', 'mask_rcnn_r101_fpn_gn-all_2x_coco', 'mask_rcnn_r101_fpn_gn-all_3x_coco', 'mask_rcnn_r50_fpn_gn-all_contrib_2x_coco', 'mask_rcnn_r50_fpn_gn-all_contrib_3x_coco', 'faster_rcnn_r50_fpn_gn_ws-all_1x_coco', 'faster_rcnn_r101_fpn_gn_ws-all_1x_coco', 'faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco', 'faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco', 'mask_rcnn_r50_fpn_gn_ws-all_2x_coco', 'mask_rcnn_r101_fpn_gn_ws-all_2x_coco', 'mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco', 'mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco', 'mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco', 'mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco', 'mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco', 'mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco', 'grid_rcnn_r50_fpn_gn-head_2x_coco', 'grid_rcnn_r101_fpn_gn-head_2x_coco', 'grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco', 'grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco', 'faster_rcnn_r50_fpn_groie_1x_coco', 'grid_rcnn_r50_fpn_gn-head_groie_1x_coco', 'mask_rcnn_r50_fpn_groie_1x_coco', 'mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco', 'mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco', 'ga_rpn_r50_caffe_fpn_1x_coco', 'ga_rpn_r101_caffe_fpn_1x_coco.py', 'ga_rpn_x101_32x4d_fpn_1x_coco.py', 
#                     'ga_rpn_x101_64x4d_fpn_1x_coco.py.py', 'ga_faster_r50_caffe_fpn_1x_coco', 'ga_faster_r101_caffe_fpn_1x_coco', 'ga_faster_x101_32x4d_fpn_1x_coco', 'ga_faster_x101_64x4d_fpn_1x_coco', 'ga_retinanet_r50_caffe_fpn_1x_coco', 'ga_retinanet_r101_caffe_fpn_1x_coco', 'ga_retinanet_x101_32x4d_fpn_1x_coco', 'ga_retinanet_x101_64x4d_fpn_1x_coco', 'faster_rcnn_hrnetv2p_w18_1x_coco', 'faster_rcnn_hrnetv2p_w18_2x_coco', 'faster_rcnn_hrnetv2p_w32_1x_coco', 'faster_rcnn_hrnetv2p_w32_2x_coco', 'faster_rcnn_hrnetv2p_w40_1x_coco', 'faster_rcnn_hrnetv2p_w40_2x_coco', 'mask_rcnn_hrnetv2p_w18_1x_coco', 'mask_rcnn_hrnetv2p_w18_2x_coco', 'mask_rcnn_hrnetv2p_w32_1x_coco', 'mask_rcnn_hrnetv2p_w32_2x_coco', 'mask_rcnn_hrnetv2p_w40_1x_coco', 'mask_rcnn_hrnetv2p_w40_2x_coco', 'cascade_rcnn_hrnetv2p_w18_20e_coco', 'cascade_rcnn_hrnetv2p_w32_20e_coco', 'cascade_rcnn_hrnetv2p_w40_20e_coco', 'cascade_mask_rcnn_hrnetv2p_w18_20e_coco', 'cascade_mask_rcnn_hrnetv2p_w32_20e_coco', 'cascade_mask_rcnn_hrnetv2p_w40_20e_coco', 'htc_hrnetv2p_w18_20e_coco', 'htc_hrnetv2p_w32_20e_coco', 'htc_hrnetv2p_w40_20e_coco', 'fcos_hrnetv2p_w18_gn-head_4x4_1x_coco', 'fcos_hrnetv2p_w18_gn-head_4x4_2x_coco', 'fcos_hrnetv2p_w32_gn-head_4x4_1x_coco', 'fcos_hrnetv2p_w32_gn-head_4x4_2x_coco', 'fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco', 'fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco', 'fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco', 'htc_r50_fpn_1x_coco', 'htc_r50_fpn_20e_coco', 'htc_r101_fpn_20e_coco', 'htc_x101_32x4d_fpn_16x1_20e_coco', 'htc_x101_64x4d_fpn_16x1_20e_coco', 'htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco', 'mask_rcnn_r50_fpn_instaboost_4x_coco', 'mask_rcnn_r101_fpn_instaboost_4x_coco', 'mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco', 'cascade_mask_rcnn_r50_fpn_instaboost_4x_coco', 'lad_r50_paa_r101_fpn_coco_1x', 'lad_r101_paa_r50_fpn_coco_1x', 
#                     'ld_r18_gflv1_r101_fpn_coco_1x', 'ld_r34_gflv1_r101_fpn_coco_1x', 'ld_r50_gflv1_r101_fpn_coco_1x', 'ld_r101_gflv1_r101dcn_fpn_coco_1x', 'libra_faster_rcnn_r50_fpn_1x_coco', 'libra_faster_rcnn_r101_fpn_1x_coco', 'libra_faster_rcnn_x101_64x4d_fpn_1x_coco', 'libra_retinanet_r50_fpn_1x_coco', 'mask_rcnn_r50_caffe_fpn_1x_coco', 'mask_rcnn_r50_fpn_1x_coco', 'mask_rcnn_r50_fpn_fp16_1x_coco', 'mask_rcnn_r50_fpn_2x_coco', 'mask_rcnn_r101_caffe_fpn_1x_coco', 'mask_rcnn_r101_fpn_1x_coco', 'mask_rcnn_r101_fpn_2x_coco', 'mask_rcnn_x101_32x4d_fpn_1x_coco', 'mask_rcnn_x101_32x4d_fpn_2x_coco', 'mask_rcnn_x101_64x4d_fpn_1x_coco', 'mask_rcnn_x101_64x4d_fpn_2x_coco', 'mask_rcnn_x101_32x8d_fpn_1x_coco', 'mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco', 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco', 'mask_rcnn_r50_fpn_mstrain-poly_3x_coco', 'mask_rcnn_r101_fpn_mstrain-poly_3x_coco', 'mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco', 'mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_coco', 'mask_rcnn_x101_32x8d_fpn_mstrain-poly_1x_coco', 'mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco', 'mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco', 'ms_rcnn_r50_caffe_fpn_1x_coco', 'ms_rcnn_r50_caffe_fpn_2x_coco', 'ms_rcnn_r101_caffe_fpn_1x_coco', 'ms_rcnn_r101_caffe_fpn_2x_coco', 'ms_rcnn_x101_32x4d_fpn_1x_coco', 'ms_rcnn_x101_64x4d_fpn_1x_coco', 'ms_rcnn_x101_64x4d_fpn_2x_coco', 'nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco', 'nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco', 'retinanet_r50_fpn_crop640_50e_coco', 'retinanet_r50_nasfpn_crop640_50e_coco', 'faster_rcnn_r50_fpn_32x2_1x_openimages', 'retinanet_r50_fpn_32x2_1x_openimages', 'ssd300_32x8_36e_openimages', 'faster_rcnn_r50_fpn_32x2_1x_openimages_challenge', 'faster_rcnn_r50_fpn_32x2_cas_1x_openimages', 'faster_rcnn_r50_fpn_32x2_cas_1x_openimages_challenge', 'paa_r50_fpn_1x_coco', 'paa_r50_fpn_1.5x_coco', 'paa_r50_fpn_2x_coco', 
#                     'paa_r50_fpn_mstrain_3x_coco', 'paa_r101_fpn_1x_coco', 'paa_r101_fpn_2x_coco', 'paa_r101_fpn_mstrain_3x_coco', 'faster_rcnn_r50_pafpn_1x_coco', 'panoptic_fpn_r50_fpn_1x_coco', 'panoptic_fpn_r50_fpn_mstrain_3x_coco', 'panoptic_fpn_r101_fpn_1x_coco', 'panoptic_fpn_r101_fpn_mstrain_3x_coco', 'retinanet_pvt-t_fpn_1x_coco', 'retinanet_pvt-s_fpn_1x_coco', 'retinanet_pvt-m_fpn_1x_coco', 'retinanet_pvtv2-b0_fpn_1x_coco', 'retinanet_pvtv2-b1_fpn_1x_coco', 'retinanet_pvtv2-b2_fpn_1x_coco', 'retinanet_pvtv2-b3_fpn_1x_coco', 'retinanet_pvtv2-b4_fpn_1x_coco', 'retinanet_pvtv2-b5_fpn_1x_coco', 'pisa_faster_rcnn_r50_fpn_1x_coco', 'pisa_faster_rcnn_x101_32x4d_fpn_1x_coco', 'pisa_mask_rcnn_r50_fpn_1x_coco', 'pisa_retinanet_r50_fpn_1x_coco', 'pisa_retinanet_x101_32x4d_fpn_1x_coco', 'pisa_ssd300_coco', 'pisa_ssd512_coco', 'point_rend_r50_caffe_fpn_mstrain_1x_coco', 'point_rend_r50_caffe_fpn_mstrain_3x_coco', 'queryinst_r50_fpn_1x_coco', 'queryinst_r50_fpn_mstrain_480-800_3x_coco', 'queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco', 'queryinst_r101_fpn_mstrain_480-800_3x_coco', 'queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco', 'mask_rcnn_regnetx-3.2GF_fpn_1x_coco', 'mask_rcnn_regnetx-4GF_fpn_1x_coco', 'mask_rcnn_regnetx-6.4GF_fpn_1x_coco', 'mask_rcnn_regnetx-8GF_fpn_1x_coco', 'mask_rcnn_regnetx-12GF_fpn_1x_coco', 'mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco', 'faster_rcnn_regnetx-3.2GF_fpn_1x_coco', 'faster_rcnn_regnetx-3.2GF_fpn_2x_coco', 'retinanet_regnetx-800MF_fpn_1x_coco', 'retinanet_regnetx-1.6GF_fpn_1x_coco', 'retinanet_regnetx-3.2GF_fpn_1x_coco', 'faster_rcnn_regnetx-400MF_fpn_mstrain_3x_coco', 'faster_rcnn_regnetx-800MF_fpn_mstrain_3x_coco', 'faster_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco', 'faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco', 'faster_rcnn_regnetx-4GF_fpn_mstrain_3x_coco', 'mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco', 
#                     'mask_rcnn_regnetx-400MF_fpn_mstrain-poly_3x_coco', 'mask_rcnn_regnetx-800MF_fpn_mstrain-poly_3x_coco', 'mask_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco', 'mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco', 'cascade_mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco', 'bbox_r50_grid_fpn_gn-neck+head_1x_coco', 'reppoints_moment_r50_fpn_1x_coco', 'reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco', 'reppoints_moment_r50_fpn_gn-neck+head_2x_coco', 
#                     'reppoints_moment_r101_fpn_gn-neck+head_2x_coco', 'reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck+head_2x_coco', 'reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck+head_2x_coco', 'faster_rcnn_r2_101_fpn_2x_coco', 'mask_rcnn_r2_101_fpn_2x_coco', 'cascade_rcnn_r2_101_fpn_20e_coco', 'cascade_mask_rcnn_r2_101_fpn_20e_coco', 'htc_r2_101_fpn_20e_coco', 'faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco', 'faster_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco', 'mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco', 'mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco', 'cascade_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco', 'cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco', 'cascade_mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco', 'cascade_mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco', 'retinanet_r18_fpn_1x_coco', 'retinanet_r18_fpn_1x8_1x_coco', 'retinanet_r50_caffe_fpn_1x_coco', 'retinanet_r50_fpn_1x_coco', 'retinanet_r50_fpn_fp16_1x_coco', 'retinanet_r50_fpn_2x_coco', 'retinanet_r50_fpn_mstrain_3x_coco', 'retinanet_r101_caffe_fpn_1x_coco', 'retinanet_r101_caffe_fpn_mstrain_3x_coco', 'retinanet_r101_fpn_1x_coco', 'retinanet_r101_fpn_2x_coco', 'retinanet_r101_fpn_mstrain_3x_coco', 'retinanet_x101_32x4d_fpn_1x_coco', 'retinanet_x101_32x4d_fpn_2x_coco', 'retinanet_x101_64x4d_fpn_1x_coco', 'retinanet_x101_64x4d_fpn_2x_coco', 'retinanet_x101_64x4d_fpn_mstrain_3x_coco', 'sabl_faster_rcnn_r50_fpn_1x_coco', 'sabl_faster_rcnn_r101_fpn_1x_coco', 'sabl_cascade_rcnn_r50_fpn_1x_coco', 'sabl_cascade_rcnn_r101_fpn_1x_coco', 'sabl_retinanet_r50_fpn_1x_coco', 'sabl_retinanet_r50_fpn_gn_1x_coco', 'sabl_retinanet_r101_fpn_1x_coco', 'sabl_retinanet_r101_fpn_gn_1x_coco', 'sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco', 
#                     'sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco', 'scnet_r50_fpn_1x_coco', 'scnet_r50_fpn_20e_coco', 'scnet_r101_fpn_20e_coco', 'scnet_x101_64x4d_fpn_20e_coco', 'faster_rcnn_r50_fpn_gn-all_scratch_6x_coco', 'mask_rcnn_r50_fpn_gn-all_scratch_6x_coco', 'mask_rcnn_r50_fpn_random_seesaw_loss_mstrain_2x_lvis_v1', 'mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1', 'mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1', 'mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1', 'mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1', 'mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1', 'mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1', 'mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1', 'cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1', 'cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1', 'cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1', 'cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1', 'sparse_rcnn_r50_fpn_1x_coco', 'sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco', 'sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco', 'sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco', 'sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco', 'decoupled_solo_r50_fpn_1x_coco', 'decoupled_solo_r50_fpn_3x_coco', 'decoupled_solo_light_r50_fpn_3x_coco', 'solo_r50_fpn_3x_coco', 'solo_r50_fpn_1x_coco', 'ssd300_coco', 'ssd512_coco', 'ssdlite_mobilenetv2_scratch_600e_coco', 'mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco', 'mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco', 'mask_rcnn_swin-t-p4-w7_fpn_1x_coco', 'mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco', 'tridentnet_r50_caffe_1x_coco', 'tridentnet_r50_caffe_mstrain_1x_coco', 'tridentnet_r50_caffe_mstrain_3x_coco', 
#                     'tood_r101_fpn_mstrain_2x_coco', 'tood_x101_64x4d_fpn_mstrain_2x_coco', 'tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco', 'tood_r50_fpn_anchor_based_1x_coco', 'tood_r50_fpn_1x_coco', 'tood_r50_fpn_mstrain_2x_coco', 'vfnet_r50_fpn_1x_coco', 'vfnet_r50_fpn_mstrain_2x_coco', 'vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco', 'vfnet_r101_fpn_1x_coco', 'vfnet_r101_fpn_mstrain_2x_coco', 'vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco', 'vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco', 'vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco', 'yolact_r50_1x8_coco', 'yolact_r50_8x8_coco', 'yolact_r101_1x8_coco', 'yolov3_d53_320_273e_coco', 'yolov3_d53_mstrain-416_273e_coco', 'yolov3_d53_mstrain-608_273e_coco', 'yolov3_d53_fp16_mstrain-608_273e_coco', 'yolov3_mobilenetv2_320_300e_coco', 'yolov3_mobilenetv2_mstrain-416_300e_coco', 'yolof_r50_c5_8x8_1x_coco', 'yolox_s_8x8_300e_coco', 'yolox_l_8x8_300e_coco', 'yolox_x_8x8_300e_coco', 'yolox_tiny_8x8_300e_coco']

VIDEO_DICT = {
        'rainy_night_city': ['02d478d1-e6811391', '024dd592-94359ff1', '00a04f65-8c891f94'],
        'rainy_daytime_city': ['020cc8f8-8b679d0b', 'b2036451-aa924fd1', '03112119-0aafd3ad'],
        'clear_night_city': ['0001542f-ec815219', '000d35d3-41990aa4', '00134776-9123d227'],
        'clear_day_city': ['00067cfb-5443fe39', '00067cfb-f1b91e3c', '000e0252-8523a4a9'],
        'rainy_night_highway': ['028584e7-6a14163e', '035268c2-5cf95581', 'b20eae11-18cd8ca2'],
        'rainy_daytime_highway': ['b1e1a7b8-b397c445', 'b1e1a7b8-a7426a97', '012fdff1-9d1d0d1d'],
        'clear_night_highway': ['00268999-0b20ef00', '0059f17f-f0882eef', '007b11e5-c22ddae8'],
        'clear_daytime_highway': ['002d290d-89f4e5c0', '004071a4-4e8a363a', '0049e5b8-725e21a0']
    }    

MODE = 'proposals'      # mode: proposals   s_overhead
PROPOSALS = 2000        # 1000, 2000, 500
ALL = False             # True will be running all the scenes
RESULT_DIR = f'{MODE}_{PROPOSALS}_test_2'

MODEL_LIST = ['faster_rcnn_r50_caffe_dc5_1x_coco', 'faster_rcnn_r50_fpn_1x_coco', 'faster_rcnn_r101_fpn_1x_coco']
SCENE_TYPES = ['rainy_night_city'] if not ALL else list(VIDEO_DICT.keys())

random.seed(30)

# GENERAL HELPER FUNCTION
def list_filenames(directory, ending):
    """List all files in a directory with a specific file extension, sorted alphabetically."""
    file_names = []
    for entry in sorted(os.listdir(directory)):  # 🔧 sort added
        entry_path = os.path.join(directory, entry)
        if os.path.isfile(entry_path) and entry.endswith(f'.{ending}'):
            file_names.append(entry_path)
    return file_names
    
def suppress_function_output(func, *args, **kwargs):
    """Suppress the output of a function that prints to stdout."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):  # Redirect stdout to devnull (suppresses print output)
            result = func(*args, **kwargs)
    return result


# HELPER FUNCTION FOR EVALUATION PIPELINE ==> LIDAR

def lidar_output_to_nusc_box(
        detection, token, score_thresh=0.5):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (Det3DDataSample): Detection results.

            - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], np.ndarray or None]: List of standard
        NuScenesBoxes and attribute labels.
    """
    bbox3d = detection.bboxes_3d.to('cpu')
    scores = detection.scores_3d.cpu().numpy()
    labels = detection.labels_3d.cpu().numpy()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if isinstance(bbox3d, LiDARInstance3DBoxes):
        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            # filter out bbox with low confidence score
            if scores[i] < score_thresh:
                continue
            quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity,
                token=token)
            box_list.append(box)
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes '
            'to standard NuScenesBoxes.')

    return box_list

def lidar_nusc_box_to_global(
        nusc,
        sample_data_token: str,
        boxes):
    """
    Convert predicted NuScenesBoxes from LiDAR to global coordinates using sample_data_token.

    Args:
        nusc: NuScene instance
        sample_data_token (str): Token for the sample_data (e.g. from file name).
        boxes (List[NuScenesBox]): Predicted bounding boxes in LiDAR coordinates.

    Returns:
        List[NuScenesBox]: Boxes transformed into global coordinates and filtered.
    """
    # Step 1: Get sensor calibration and ego poses
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Construct transformation matrices
    lidar2ego_rot = Quaternion(cs_record['rotation']).rotation_matrix
    lidar2ego_trans = np.array(cs_record['translation'])

    ego2global_rot = Quaternion(pose_record['rotation']).rotation_matrix
    ego2global_trans = np.array(pose_record['translation'])

    # Transform boxes
    box_list = []
    for box in boxes:
        # LiDAR -> Ego
        box.rotate(Quaternion(matrix=lidar2ego_rot))
        box.translate(lidar2ego_trans)

        # Ego -> Global
        box.rotate(Quaternion(matrix=ego2global_rot))
        box.translate(ego2global_trans)

        box_list.append(box)

    return box_list


# HELPER_FUNCTION FOR IMAGE EVALUATION PIPELINE
def image_output_to_coco(pred_instance, image_id, category_id_map, score_thresh=0.5):
    """
    Convert image-based detection results to COCO-style format.

    Args:
        pred_instance (InstanceData): Contains `bboxes`, `labels`, and `scores`.
        image_id (int or str): Unique image ID in the COCO dataset.
        category_id_map (dict): Mapping from internal label IDs to COCO category IDs.
        score_thresh (float): Minimum score threshold to keep predictions.

    Returns:
        List[dict]: List of COCO-style prediction dicts for this image.
    """
    bboxes = pred_instance.bboxes.cpu().numpy()
    labels = pred_instance.labels.cpu().numpy()
    scores = pred_instance.scores.cpu().numpy()

    coco_results = []

    for bbox, label, score in zip(bboxes, labels, scores):
        # Apply score threshold
        if score < score_thresh:
            continue
        # Convert [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = bbox
        coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        result = {
            "image_id": image_id,
            "category_id": int(category_id_map[label]),
            "bbox": coco_bbox,
            "score": float(score)
        }
        coco_results.append(result)

    return coco_results

# HELPER FUNCTION USED BY BOTH LIDAR AND IMAGE EVALUATION PIPELINE

def interpolate_gt(nusc, sample_data_token: str, box_cls=DetectionBox):
        """
        Generate interpolated or fallback ground truth boxes for a given sample_data_token.
        Falls back to prev or next keyframe if interpolation is not possible.
        """
        sd = nusc.get('sample_data', sample_data_token)
        timestamp = sd['timestamp']

        box_list = []
        instance_tokens = []

        if sd['is_key_frame']:
            sample = nusc.get('sample', sd['sample_token'])
            annos = [nusc.get('sample_annotation', tok) for tok in sample['anns']]

            for a in annos:
                detection_name = category_to_detection_name(a['category_name'])
                if detection_name is None:
                    continue

                box_list.append(box_cls(
                    sample_token=sample_data_token,
                    translation=a['translation'],
                    size=a['size'],
                    rotation=a['rotation'],
                    velocity=a.get('velocity', [0.0, 0.0]),
                    detection_name=detection_name,
                    attribute_name=''  # optional
                ))
                instance_tokens.append(a['instance_token'])

            return box_list, instance_tokens

        # Walk backward to find previous keyframe
        prev_sd_token = sd['prev']
        prev_keyframe = None
        while prev_sd_token:
            prev_sd = nusc.get('sample_data', prev_sd_token)
            if prev_sd['is_key_frame']:
                prev_keyframe = prev_sd
                break
            prev_sd_token = prev_sd['prev']

        # Walk forward to find next keyframe
        next_sd_token = sd['next']
        next_keyframe = None
        while next_sd_token:
            next_sd = nusc.get('sample_data', next_sd_token)
            if next_sd['is_key_frame']:
                next_keyframe = next_sd
                break
            next_sd_token = next_sd['next']

        if prev_keyframe and next_keyframe:
            # Interpolation case
            t0, t1 = prev_keyframe['timestamp'], next_keyframe['timestamp']
            alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

            prev_sample = nusc.get('sample', prev_keyframe['sample_token'])
            next_sample = nusc.get('sample', next_keyframe['sample_token'])

            prev_annos = [nusc.get('sample_annotation', tok) for tok in prev_sample['anns']]
            next_annos = [nusc.get('sample_annotation', tok) for tok in next_sample['anns']]

            prev_map = {a['instance_token']: a for a in prev_annos}
            next_map = {a['instance_token']: a for a in next_annos}

            common_instances = set(prev_map.keys()) & set(next_map.keys())

            for inst in common_instances:
                a0, a1 = prev_map[inst], next_map[inst]

                t0 = np.array(a0['translation'])
                t1 = np.array(a1['translation'])
                center = (1 - alpha) * t0 + alpha * t1

                s0 = np.array(a0['size'])
                s1 = np.array(a1['size'])
                size = (1 - alpha) * s0 + alpha * s1

                q0 = Quaternion(a0['rotation'])
                q1 = Quaternion(a1['rotation'])
                rotation = Quaternion.slerp(q0, q1, amount=alpha)

                v0 = np.array(a0.get('velocity', [0, 0]))
                v1 = np.array(a1.get('velocity', [0, 0]))
                velocity = (1 - alpha) * v0 + alpha * v1

                detection_name = category_to_detection_name(a0['category_name'])
                if detection_name is None:
                    continue

                box_list.append(box_cls(
                    sample_token=sample_data_token,
                    translation=center.tolist(),
                    size=size.tolist(),
                    rotation=rotation.elements.tolist(),
                    velocity=velocity.tolist(),
                    detection_name=detection_name,
                    attribute_name=''
                ))
                instance_tokens.append(inst)

            return box_list, instance_tokens

        # Fallback case
        fallback_frame = prev_keyframe or next_keyframe
        fallback_sample = nusc.get('sample', fallback_frame['sample_token'])
        annos = [nusc.get('sample_annotation', tok) for tok in fallback_sample['anns']]

        for a in annos:
            detection_name = category_to_detection_name(a['category_name'])
            if detection_name is None:
                continue

            box_list.append(box_cls(
                sample_token=sample_data_token,
                translation=a['translation'],
                size=a['size'],
                rotation=a['rotation'],
                velocity=a.get('velocity', [0.0, 0.0]),
                detection_name=detection_name,
                attribute_name=''
            ))
            instance_tokens.append(a['instance_token'])

        return box_list, instance_tokens




