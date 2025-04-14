import os
from contextlib import redirect_stdout
import random


from collections import defaultdict

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

# def generate_timing_tree(inference_file, model_name, output_dir):
#     with open(inference_file, 'r') as file:
#         reader = csv.reader(file)
#         header = next(reader)  # Read the first row as header

#     # Filter out unwanted columns
#     header = [ele for ele in header[2:] if ele != '' and ele != 'time_preprocessing']

#     # Build the tree structure
#     root = Node("root")
#     nodes = {}

#     for path in header:
#         parts = path.split('.')
#         parent = root
#         full_path = ""
#         for part in parts:
#             full_path = f"{full_path}.{part}" if full_path else part
#             if full_path not in nodes:
#                 nodes[full_path] = Node(part, parent=parent)
#             parent = nodes[full_path]

#     # Visualize the tree in a 'tree' command-like format and write to a file
#     def visualize_tree_to_file(root_node, output_path):
#         with open(output_path, 'w') as file:
#             def print_node(node, prefix=""):
#                 connector = "├── " if node.is_leaf else "└── "
#                 file.write(f"{prefix}{connector}{node.name}\n")
#                 children = list(node.children)
#                 for i, child in enumerate(children):
#                     extension = "│   " if i < len(children) - 1 else "    "
#                     print_node(child, prefix + extension)
            
#             print_node(root_node)

#     output_path = f'{output_dir}/{model_name}.txt'
#     visualize_tree_to_file(root, output_path)

# # Thie function returned a preordered list that has file_name image_id pair
# def label_pre_processing(annotation_path='../labels/det_val_2000.json'):
#     with open(annotation_path, 'r') as f:
#         data = json.load(f)

#     file_id_dict = {image['file_name']: image['id'] for image in data['images']}
#     return file_id_dict

# '''This function reads an annotation file and return a list of list of ROI format bbox (x1, y1, x2, y2, label_cat)'''
# def reading_ann_file(coco_file_path):
#     # Load the COCO annotation file
#     with open(coco_file_path, 'r') as f:
#         coco_data = json.load(f)
    
#     # Extract the annotations
#     annotations = coco_data.get('annotations', [])
    
#     # Group bounding boxes by image_id
#     grouped_bboxes = defaultdict(list)
#     for annotation in annotations:
#         image_id = annotation['image_id']
#         x_min, y_min, width, height = annotation['bbox']
#         category_id = annotation['category_id']
#         # Convert to [x1, y1, x2, y2]
#         x1, y1 = x_min, y_min
#         x2, y2 = x_min + width, y_min + height
#         grouped_bboxes[image_id].append([x1, y1, x2, y2, category_id])
    
#     return grouped_bboxes

# '''Calculate IoU between two box with x1, y1, x2, y2'''
# def calculate_iou(bbox1, bbox2):
#     # Unpack the bounding boxes
#     x1_1, y1_1, x2_1, y2_1 = bbox1
#     x1_2, y1_2, x2_2, y2_2 = bbox2

#     assert x1_1 < x2_1 and y1_1 < y2_1, "bbox1 has invalid coordinates"
#     assert x1_2 < x2_2 and y1_2 < y2_2, "bbox2 has invalid coordinates"
    
#     # Calculate the coordinates of the intersection rectangle
#     inter_x1 = max(x1_1, x1_2)
#     inter_y1 = max(y1_1, y1_2)
#     inter_x2 = min(x2_1, x2_2)
#     inter_y2 = min(y2_1, y2_2)
    
#     # Calculate the area of the intersection rectangle
#     inter_width = max(0, inter_x2 - inter_x1)
#     inter_height = max(0, inter_y2 - inter_y1)
#     inter_area = inter_width * inter_height
    
#     # Calculate the area of both bounding boxes
#     bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
#     bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
#     # Calculate the union area
#     union_area = bbox1_area + bbox2_area - inter_area
    
#     # Compute IoU
#     if union_area == 0:
#         return 0.0  # Avoid division by zero
#     iou = inter_area / union_area
#     return iou

# def plot_results(
#     video_dict, root_directory, target_scene, target_layer, mode, 
#     models_to_include, plot_type="line", fontsize=23, ann_dir='/home/mg/pdnn/Characterization-PDNN/track_labels', scr_threshold=0.5
# ):

#     handles, labels = [], []
#     # Iterate over the video dictionary
#     for scene_type in target_scene:
#         video_ids = video_dict[scene_type]
#         data = []
#         proposal_data = []
#         TP_proposals = {}
#         for video_id in video_ids:
#             subdir_path = f'{root_directory}/{scene_type}_{video_id}'
#             ann_path = f'{ann_dir}/{video_id}.json'
#             gt_bboxs = reading_ann_file(ann_path)

#             for model in models_to_include:
#                 if mode == "inference":
#                     file_path = f'{subdir_path}/{model}_inference.csv'
#                     df = pd.read_csv(file_path)

#                     for index, row in df.iterrows():
#                         data.append([index, model, row[target_layer], video_id, scene_type])

#                 if mode == 'proposal':
#                     ROI_path = f'{subdir_path}/{model}_ROI_proposals.pt'
#                     RPN_path = f'/home/mg/pdnn/Characterization-PDNN/uni_model_exp/proposals_2000_test/{scene_type}_{video_id}/{model}_RPN_proposals.pt'

#                     ROI_proposals = torch.load(ROI_path)
#                     RPN_proposals = torch.load(RPN_path)

#                     image_paths = list_filenames(f'/home/mg/pdnn/bdd100k-data/images/track/{video_id}', 'jpg')

#                     for image in image_paths:
#                         image_id_match = re.search(r'0*(\d+)?', os.path.basename(image).split('-')[-1].split('.')[0])
#                         if image_id_match:
#                             image_id = image_id_match.group(1) or '0'
#                         else:
#                             image_id = None

#                         GT_proposal = gt_bboxs[int(image_id)]
#                         ROI_proposal = ROI_proposals[int(image_id) - 1]

#                         if scene_type not in TP_proposals:
#                             TP_proposals[scene_type] = {}
#                         if video_id not in TP_proposals[scene_type]:
#                             TP_proposals[scene_type][video_id] = []

#                         TP_count = 0
#                         TP_proposal = []
#                         FP_proposal = []

#                         for proposal in ROI_proposal:
#                             dt_x1, dt_y1, dt_x2, dt_y2, _, dt_label, index = proposal
#                             found = False
#                             for gt in GT_proposal:
#                                 gt_x1, gt_y1, gt_x2, gt_y2, gt_label = gt
#                                 if dt_label not in list(LABELS.keys()):
#                                     continue
#                                 dt_label = LABELS[int(dt_label)]

#                                 gt_bbox = [float(gt_x1), float(gt_y1), float(gt_x2), float(gt_y2)]
#                                 pred_bbox = [float(dt_x1), float(dt_y1), float(dt_x2), float(dt_y2)]

#                                 IoU = calculate_iou(gt_bbox, pred_bbox)

#                                 if IoU > scr_threshold and gt_label == dt_label:
#                                     pred_bbox.append(int(dt_label))
#                                     TP_count = TP_count + 1
#                                     TP_proposal.append(pred_bbox)
#                                     found =True
#                                     break
#                             if found == False:
#                                 FP_proposal.append(RPN_proposals[int(image_id) - 1][int(index)//80].tolist())

#                         RPN_count = len(RPN_proposals[int(image_id) - 1]) * 80
#                         FP_count = RPN_count - TP_count

#                         TP_proposals[scene_type][video_id].append({
#                             "Model": model,
#                             "Image_ID": image_id,
#                             "TP_count": TP_count,
#                             "FP_count": FP_count,
#                             "TP_proposals": TP_proposal,
#                             "FP_proposals": FP_proposal
#                         })


#                         proposal_data.append([image_id, model[12:-8], TP_count, FP_count, RPN_count, video_id, scene_type])
    
#         output_json_path = f"{root_directory}/{scene_type}_proposals.json"
#         with open(output_json_path, "w") as json_file:
#             json.dump(TP_proposals, json_file, indent=4)
                
#         # # Define column names based on the mode
#         # if mode == "inference":
#         #     df = pd.DataFrame(data, columns=["ImageIndex", "Model", target_layer, "VideoID", "Scene_type"])
#         #     y_label = target_layer
#         # elif mode == "proposal":
#         #     df = pd.DataFrame(proposal_data, columns=["ImageIndex", "Model", "TP_count", "FP_count", "RPN_count", "VideoID", "Scene_type"])
#         #     y_label = "TP_count"

#         # # Generate subplots side by side for each video_id
#         # video_ids = video_dict[scene_type]
#         # num_videos = len(video_ids)
#         # fig, axes = plt.subplots(1, num_videos, figsize=(6 * num_videos, 6), sharey=True)

#         # for ax, video_id in zip(axes, video_ids):
#         #     video_df = df[df["VideoID"] == video_id]
#         #     if plot_type == "line":
#         #         sns.lineplot(x="ImageIndex", y=y_label, data=video_df, hue="Model", ax=ax, linestyle='-')
#         #         ax.set_ylabel(y_label, fontsize=fontsize)
#         #         ax.set_xlabel(None, fontsize=fontsize)
#         #         subplot_handles, subplot_labels = ax.get_legend_handles_labels()
#         #         if not handles:  # Add only once
#         #             handles, labels = subplot_handles, subplot_labels
#         #     elif plot_type == "boxplot":
#         #         sns.boxplot(x="Model", y=y_label, data=video_df, ax=ax)
#         #         ax.set_ylabel(y_label, fontsize=fontsize)
#         #         ax.set_xlabel(None, fontsize=fontsize)

#         # for ax in axes:
#         #     ax.legend().remove()
#         #     ax.tick_params(axis='x', labelsize=18)  # X-axis font size
#         #     ax.tick_params(axis='y', labelsize=18)  # Y-axis font size

#         # # fig.legend(
#         # #     handles, 
#         # #     labels, 
#         # #     loc='lower right', 
#         # #     ncol=1,  # Number of columns in the legend
#         # #     fontsize=fontsize
#         # # )

#         # plt.tight_layout()
#         # plt.savefig(f'{root_directory}/{scene_type}_TP_box.png')
#         # plt.close()

# def plot_results_model(
#     video_dict, root_directory, target_scene=None, target_layer=None, mode="inference", 
#     models_to_include=None, models_to_exclude=None, plot_type="line", fontsize=23
# ):
#     data = []

#     # Validate input
#     if not isinstance(target_layer, list):
#         raise ValueError("`target_layer` must be a list of layer names.")

#     # Iterate over the video dictionary
#     for scene_type, video_ids in video_dict.items():
#         if scene_type == target_scene or target_scene is None:  # Focus on the specific scene
#             for video_id in video_ids:
#                 subdir_path = f'{root_directory}/{scene_type}_{video_id}'
#                 if os.path.exists(subdir_path):
#                     for file in os.listdir(subdir_path):
#                         if mode == "inference" and file.endswith("_inference.csv"):
#                             file_path = os.path.join(subdir_path, file)
#                             print(file_path)
#                             df = pd.read_csv(file_path)
#                             model_name = file.replace("_inference.csv", "")

#                             # Apply model filters
#                             if models_to_include and model_name not in models_to_include:
#                                 continue
#                             if models_to_exclude and model_name in models_to_exclude:
#                                 continue

#                             # Collect values with image_id, scene, and target layers
#                             for index, row in df.iterrows():
#                                 for layer in target_layer:  # Collect multiple target layers
#                                     if layer in row:
#                                         data.append([index, model_name, layer, row[layer], video_id, scene_type])

#     # Convert data to DataFrame
#     if mode == "inference":
#         df = pd.DataFrame(data, columns=["ImageIndex", "Model", "TargetLayer", "Value", "VideoID", "Scene_type"])
#         y_label = "Inference Time (s)"
#         title_metric = "TargetLayer"
#         df.to_csv('test.csv')

#     video_ids = video_dict.get(target_scene, [])
#     num_videos = len(video_ids)
    
#     handles, labels = [], []
#     # Iterate through each model
#     for model_name in df["Model"].unique():
        
#         fig, axes = plt.subplots(1, num_videos, figsize=(6 * num_videos, 6), sharey=True)
#         model_df = df[df["Model"] == model_name]  # Filter data for the current model
#         model_df.to_csv(f'{model_name}.csv')
#         # Create subplots for each video ID
#         for ax, video_id in zip(axes, video_ids):
#             video_df = model_df[model_df["VideoID"] == video_id]  # Filter data for the current video ID
            
#             # Line plot: Different line for each target layer
#             if plot_type == "line":
#                 sns.lineplot(
#                     x="ImageIndex", 
#                     y="Value", 
#                     data=video_df, 
#                     hue="TargetLayer",  # Hue is now based on target layers
#                     ax=ax, 
#                     linestyle='-',
#                     linewidth=2
#                 )
#                 ax.set_ylabel(y_label, fontsize=fontsize)
#                 ax.set_xlabel(None, fontsize=fontsize)
#                 subplot_handles, subplot_labels = ax.get_legend_handles_labels()
#                 if not handles:  # Add only once
#                     handles, labels = subplot_handles, subplot_labels
            
#             # Boxplot: Target layers on the x-axis
#             elif plot_type == "boxplot":
#                 sns.boxplot(x="TargetLayer", y="Value", data=video_df, ax=ax)
#                 ax.set_ylabel(y_label, fontsize=fontsize)
#                 ax.set_xlabel("Target Layer", fontsize=fontsize)
        
#         for ax in axes:
#             ax.legend().remove()
#             ax.tick_params(axis='x', labelsize=18)  # X-axis font size
#             ax.tick_params(axis='y', labelsize=18)  # Y-axis font size

#         # fig.legend(
#         #     handles, 
#         #     labels, 
#         #     loc='lower right', 
#         #     bbox_to_anchor=(1, 0.65),  # Adjust position as needed
#         #     ncol=1,  # Number of columns in the legend
#         #     fontsize=fontsize
#         # )

#         plt.tight_layout()
#         plt.tight_layout(rect=[0, 0, 0.95, 0.95])
#         plt.savefig(f'{root_directory}/{model_name}.png')

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

# def visualize_all_images(models, image_dir, annotation_file, output_dir):
#     """
#     Visualize inference results and ground truth bounding boxes for all images in a directory.

#     Args:
#         models (list): List of tuples (config_file, checkpoint_file).
#         image_dir (str): Directory containing images.
#         annotation_file (str): COCO format annotation file.

#     Returns:
#         None: Displays the images with bounding boxes.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     # Load COCO annotations
#     coco = COCO(annotation_file)
    
#     # Load all models once
#     loaded_models = [
#         (model_name, init_detector(config_file, checkpoint_file, device='cuda:0'))
#         for [model_name, config_file, checkpoint_file] in models
#     ]
    
#     # Get all image IDs
#     image_ids = coco.getImgIds()

#     bright_colors = [
#         (255, 0, 0),    # Red
#         (0, 0, 255),    # Blue
#         (255, 255, 0),  # Yellow
#         (255, 0, 255),  # Magenta
#         (0, 255, 255),  # Cyan
#     ]
    
#     # Iterate over each image
#     for image_id in image_ids:
#         # Get image info
#         img_info = coco.loadImgs(image_id)[0]
#         img_path = os.path.join(image_dir, img_info['file_name'])
#         original_img = cv2.imread(img_path)
#         original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
#         # Draw ground truth bounding boxes
#         gt_img = original_img.copy()
#         ann_ids = coco.getAnnIds(imgIds=image_id)
#         annotations = coco.loadAnns(ann_ids)
#         gt_color = (0, 255, 0)  # Green for ground truth
        
#         for ann in annotations:
#             x, y, w, h = map(int, ann['bbox'])  # [x, y, w, h]
#             cat_name = coco.loadCats(ann['category_id'])[0]['name']
#             cv2.rectangle(gt_img, (x, y), (x + w, y + h), gt_color, 2)
#             cv2.putText(gt_img, f"GT: {cat_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1)
        
#         # Save ground truth image
#         plt.figure(figsize=(12, 8))
#         plt.imshow(gt_img)
#         plt.title(f"Ground Truth - {img_info['file_name']}")
#         plt.axis("off")
#         plt.savefig(f"{output_dir}/{img_info['file_name']}_gt.png")
#         plt.close()
        
#         # Draw predicted bounding boxes for each model
#         for idx, (model_name, model) in enumerate(loaded_models):
#             pred_img = original_img.copy()
#             results = inference_detector(model, img_path)
#             color = bright_colors[idx % len(bright_colors)]  # Cycle through bright colors
            
#             for i, result in enumerate(results):
                
#                 if isinstance(result, tuple):
#                     bboxes, _ = result
#                 else:
#                     bboxes = result
                
#                 for bbox in bboxes:
#                     x1, y1, x2, y2, score = bbox
#                     cv2.rectangle(pred_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                     label = f"{model.CLASSES[i]} {score:.2f}"
#                     cv2.putText(pred_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
#             # Save predicted image
#             plt.figure(figsize=(12, 8))
#             plt.imshow(pred_img)
#             plt.title(f"{model_name} - {img_info['file_name']}")
#             plt.axis("off")
#             plt.savefig(f"{output_dir}/{model_name}_{img_info['file_name']}.png")
#             plt.close()

# def model_downloads(model_list, output_dir):
#     count = 0
#     failed_count = 0
#     for model in sorted(model_list):
#         try:
#             download('mmdet', [model], dest_root=output_dir)
#             count += 1
#             print('processed: ', count, 'failed: ', failed_count)
#         except Exception as e:
#             failed_count += 1
#             print('processed: ', count, 'failed: ', failed_count)
#             print(e)
#             continue

# def visualize_from_pred(model_name, root_dir, scene):
#     full_path = None
#     for dirpath, dirnames, _ in os.walk(root_dir):
#         for dirname in dirnames:
#             # Check if the directory name ends with the specified substring
#             if dirname.endswith(scene):
#                 full_path = os.path.join(dirpath, dirname)
#                 print(f"Directory found: {full_path}")
    
#     if full_path!= None:
#         prediction = f'{full_path}/{model_name}_e2e_pred.json'
#         img_dir = f'/home/mg/pdnn/bdd100k-data/images/track/{scene}'

#         with open(prediction, "r") as f:
#             predictions = json.load(f)

#         # Group predictions by image_id
#         grouped_predictions = {}
#         for pred in predictions:
#             image_id = pred["image_id"]
#             if image_id not in grouped_predictions:
#                 grouped_predictions[image_id] = []
#             grouped_predictions[image_id].append(pred)

#         # Iterate through the grouped predictions
#         for image_id, annotations in grouped_predictions.items():
#             # Define the path to the image
#             image_path = Path(img_dir) / f"{scene}-{image_id:07d}.jpg"  # Adjust extension if needed
#             if not image_path.exists():
#                 print(f"Image not found: {image_path}")
#                 continue
#             plot_bounding_boxes(image_path, annotations, scr_threshold=0.5)

# def plot_bounding_boxes(image_path, annotations, scr_threshold=0.5):
#     # Load the image
#     image = cv2.imread(str(image_path))
#     if image is None:
#         print(f"Image not found: {image_path}")
#         return
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     for ann in annotations:
#         bbox = ann["bbox"]  # COCO format: [x, y, width, height]
#         category_id = ann["category_id"]
#         score = ann["score"]
#         if score < scr_threshold:
#             continue
#         # Extract coordinates
#         x, y, w, h = bbox
#         # Draw the bounding box
#         cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(255, 0, 0), thickness=2)
#         # Add category label
#         cv2.putText(image, f"Cat: {category_id}, Score: {score:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
#     # Plot the image with Matplotlib
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     plt.axis("off")
#     plt.savefig(f'/home/mg/pdnn/Characterization-PDNN/uni_model_exp/proposals_2000_test_2/{os.path.basename(image_path)}')
#     plt.close()

# def plot_cat_mAP(root_dir, exp):
#     # Load your CSV file
#     file_path = f"{root_dir}/{exp}_mAP_cat_size.csv"  # Replace with your file path
#     df = pd.read_csv(file_path)

#     # Extract unique categories
#     categories = df['category'].unique()

#     # Dictionary to store the statistics
#     statistics = {}

#     # Iterate through each category
#     for category in categories:
#         category_data = df[df['category'] == category]
#         models = category_data['model'].unique()
        
#         # Create a figure with 3 subplots
#         fig, axs = plt.subplots(3, 1, figsize=(20, 20), sharex=False)
#         fig.suptitle(f"mAP Comparison for Category: {category}", fontsize=20)
        
#         # Iterate through each model and plot on the same subplots
#         handles = []  # To collect legend handles
#         labels = []   # To collect legend labels

#         category_stats = {}
#         for model_idx, model in enumerate(models):
#             model_data = category_data[category_data['model'] == model]
#             indices = range(1, len(model_data['small']) + 1)

#             def calc_stats(data):
#                 mean = data[data != -1].mean()
#                 std = data[data != -1].std()
#                 cv = std / mean if mean != 0 else 0
#                 return mean, std, cv

#             # Calculate statistics before filtering
#             small_mean, small_variation, small_cv = calc_stats(model_data['small'])
#             medium_mean, medium_variation, medium_cv = calc_stats(model_data['medium'])
#             large_mean, large_variation, large_cv = calc_stats(model_data['large'])

#             # Save statistics for this model
#             category_stats[model] = {
#                 'small_mean': small_mean,
#                 'small_variation': small_variation,
#                 'small_cv': small_cv,
#                 'medium_mean': medium_mean,
#                 'medium_variation': medium_variation,
#                 'medium_cv': medium_cv,
#                 'large_mean': large_mean,
#                 'large_variation': large_variation,
#                 'large_mean': large_cv
#             }

#             # Replace -1 with 0 in 'small' column
#             model_data['small'] = model_data['small'].apply(lambda x: 0 if x == -1 else x)
#             model_data['medium'] = model_data['medium'].apply(lambda x: 0 if x == -1 else x)
#             model_data['large'] = model_data['large'].apply(lambda x: 0 if x == -1 else x)

#             # Plot for small
#             handle, = axs[0].plot(indices, model_data['small'], marker='o')
#             axs[0].set_title('Small', fontsize=20)
#             axs[0].set_ylabel('mAP', fontsize=20)
#             axs[0].grid(True)

#             # Plot for medium
#             axs[1].plot(indices, model_data['medium'], marker='o')
#             axs[1].set_title('Medium', fontsize=20)
#             axs[1].set_ylabel('mAP', fontsize=20)
#             axs[1].grid(True)

#             # Plot for large
#             axs[2].plot(indices, model_data['large'], marker='o')
#             axs[2].set_title('Large', fontsize=20)
#             axs[2].set_xlabel('Video ID', fontsize=20)
#             axs[2].set_ylabel('mAP', fontsize=20)
#             axs[2].grid(True)

#             handles.append(handle)
#             labels.append(model)

#             # Add statistics beside each subplot with dynamic positioning
#             # vertical_offset = 0.9 - (model_idx * 0.1)  # Adjust text position dynamically
#             # axs[0].text(1.05, vertical_offset, f"{model}: Mean={small_mean:.2f}, Std={small_variation:.2f}, Cv={small_cv:.2f}",
#             #             transform=axs[0].transAxes, fontsize=22, color='red', ha='left', va='center')
#             # axs[1].text(1.05, vertical_offset, f"{model}: Mean={medium_mean:.2f}, Std={medium_variation:.2f}, Cv={medium_cv:.2f}",
#             #             transform=axs[1].transAxes, fontsize=22, color='red', ha='left', va='center')
#             # axs[2].text(1.05, vertical_offset, f"{model}: Mean={large_mean:.2f}, Std={large_variation:.2f}, Cv={large_cv:.2f}",
#             #             transform=axs[2].transAxes, fontsize=22, color='red', ha='left', va='center')

#         statistics[category] = category_stats

#         for ax in axs:
#             ax.tick_params(axis='x', labelsize=16)  # X-axis font size
#             ax.tick_params(axis='y', labelsize=16)  # Y-axis font size

#         fig.legend(handles=handles, labels=labels, fontsize=20, loc='center right')
        
#         plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Adjust layout to make room for text
#         os.makedirs(f"/home/mg/pdnn/Characterization-PDNN/uni_model_exp/plots/cat", exist_ok=True)
#         plt.savefig(f'{root_directory}/{category}_{exp}.png') 
#         plt.close()

#     # Save statistics to a file
#     stats_df = pd.DataFrame.from_dict({(cat, mod): stats 
#                                     for cat, models in statistics.items() 
#                                     for mod, stats in models.items()}, 
#                                     orient='index')
#     stats_df.index.names = ['Category', 'Model']
#     stats_df.reset_index(inplace=True)
#     stats_df.to_csv(f'{root_dir}/model_cat_mAP_{exp}.csv', index=False)

#     print("Statistics saved to 'model_statistics.csv'")

# def plot_mAP(root_dir, exp):
#     file_path = f'{root_dir}/{exp}_mAP_size.csv'
#     df = pd.read_csv(file_path)

#     unique_sizes = df['size'].unique()
#     for size in unique_sizes:
#         size_df = df[df['size'] == size]
#         indices = range(1, 24 + 1)
#         plt.figure(figsize=(8, 6))
#         for model in size_df['model'].unique():
#             model_df = size_df[size_df['model'] == model]
#             plt.plot(model_df['video_id'], model_df['mAP'], marker='o', label=model)

#             mean = model_df['mAP'].mean()
#             std = model_df['mAP'].std()
#             cv = std / mean

#             print(f'Model: {model}  size: {size}    mean: {mean}  std: {std}    cov: {cv}')

#         plt.title(f"mAP for size: {size}")
#         plt.xlabel("Video ID")

#         plt.ylabel("mAP")
#         plt.legend(title="Model")
#         plt.grid(True)
#         os.makedirs(f"/home/mg/pdnn/Characterization-PDNN/uni_model_exp/plots/size", exist_ok=True)
#         plt.savefig(f'{root_directory}/{size}_{exp}.png') 
#         plt.close()

# '''
# images, list of image file paths
# '''
# def plot_gt_TP_FP(scene_type, video_id, model, root_dir, outdir):
#     # Create output directories for GT, TP, and FP
#     gt_outdir = os.path.join(outdir, 'GT')
#     tp_outdir = os.path.join(outdir, 'TP')
#     fp_outdir = os.path.join(outdir, 'FP')

#     os.makedirs(gt_outdir, exist_ok=True)
#     os.makedirs(tp_outdir, exist_ok=True)
#     os.makedirs(fp_outdir, exist_ok=True)

#     # Load image paths and ground truth bounding boxes
#     image_paths = list_filenames(f'/home/mg/pdnn/bdd100k-data/images/track/{video_id}', 'jpg')
#     ann_path = f'/home/mg/pdnn/Characterization-PDNN/track_labels/{video_id}.json'
#     gt_bbox = reading_ann_file(ann_path)

#     # Load proposals from JSON
#     with open(f'{root_dir}/{scene_type}_proposals.json') as file:
#         data = json.load(file)

#     for image in image_paths:
#         # Extract image ID
#         image_id_match = re.search(r'0*(\d+)?', os.path.basename(image).split('-')[-1].split('.')[0])
#         if image_id_match:
#             image_id = image_id_match.group(1) or '0'
#         else:
#             image_id = None
#         original_img = cv2.imread(image)
#         original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

#         # Plot and save GT bounding boxes
#         GT_proposals = gt_bbox[int(image_id)]
#         gt_img = original_img.copy()
#         for gt in GT_proposals:
#             x1, y1, x2, y2, gt_label = gt
#             cv2.rectangle(gt_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(gt_img, str(gt_label), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                         0.5, (0, 255, 0), 2)
#         gt_output_path = os.path.join(gt_outdir, f"{video_id}_{image_id}_GT.jpg")
#         plt.figure(figsize=(10, 8))
#         plt.imshow(gt_img)
#         plt.axis('off')
#         plt.savefig(gt_output_path)
#         plt.close()

#         for entry in data[scene_type][video_id]:
#             if int(entry['Image_ID']) == int(image_id) and entry['Model'] == model:
#                 TP_proposals = entry['TP_proposals']
#                 FP_proposals = entry['FP_proposals']

#                 # Plot and save TP bounding boxes
#                 tp_img = original_img.copy()
#                 for tp in TP_proposals:
#                     x1, y1, x2, y2, label = tp
#                     cv2.rectangle(tp_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                     cv2.putText(tp_img, str(label), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                                 0.5, (0, 255, 0), 2)
#                 tp_output_path = os.path.join(tp_outdir, f"{scene_type}_{video_id}_{image_id}_{model}_TP.jpg")
#                 plt.figure(figsize=(10, 8))
#                 plt.imshow(tp_img)
#                 plt.axis('off')
#                 plt.savefig(tp_output_path)
#                 plt.close()

#                 # # Plot and save FP bounding boxes
#                 # fp_img = original_img.copy()
#                 # for fp in FP_proposals:
#                 #     x1, y1, x2, y2, _ = fp  # Assuming FP proposals include a score or similar
#                 #     cv2.rectangle(fp_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#                 # fp_output_path = os.path.join(fp_outdir, f"{scene_type}_{video_id}_{image_id}_{model}_FP.jpg")
#                 # plt.figure(figsize=(10, 8))
#                 # plt.imshow(fp_img)
#                 # plt.axis('off')
#                 # plt.savefig(fp_output_path)
#                 # plt.close()

                


# def plot_fp_density_heatmap(scene_type, video_id, model, root_dir, outdir, grid_size=20, font_scale=3):
#     os.makedirs(outdir, exist_ok=True)

#     image_paths = list_filenames(f'/home/mg/pdnn/bdd100k-data/images/track/{video_id}', 'jpg')

#     with open(f'{root_dir}/{scene_type}_proposals.json') as file:
#         data = json.load(file)

#     for image in image_paths:
#         image_id_match = re.search(r'0*(\d+)?', os.path.basename(image).split('-')[-1].split('.')[0])
#         if image_id_match:
#             image_id = int(image_id_match.group(1)) if image_id_match.group(1) else 0
#         else:
#             image_id = None

#         for entry in data[scene_type][video_id]:
#             if int(entry['Image_ID']) == image_id and entry['Model'] == model:
#                 FP_proposals = entry['FP_proposals']

#                 # Load the original image
#                 original_img = cv2.imread(image)
#                 height, width, _ = original_img.shape

#                 # Create a grid for the heatmap
#                 heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

#                 # Map FPs to the grid
#                 for fp in FP_proposals:
#                     x1, y1, x2, y2, _ = fp  

#                     # Calculate the center of the FP box
#                     center_x = (x1 + x2) / 2
#                     center_y = (y1 + y2) / 2

#                     # Map the center to a grid cell and clamp indices within bounds
#                     grid_x = min(max(int(center_x / width * grid_size), 0), grid_size - 1)
#                     grid_y = min(max(int(center_y / height * grid_size), 0), grid_size - 1)

#                     # Increment the density at the grid cell
#                     heatmap[grid_y, grid_x] += 1

#                 # Plot the heatmap
#                 plt.figure(figsize=(10, 8))
#                 sns.heatmap(heatmap, cmap="Reds", cbar=True)
#                 sns.set_theme(font_scale=3)
#                 plt.axis('off')

#                 # Save the heatmap
#                 heatmap_output_path = os.path.join(outdir, f"{scene_type}_{video_id}_{image_id}_{model}_FP_heatmap.jpg")
#                 plt.savefig(heatmap_output_path)
#                 plt.close()

# def visualize_point_cloud(points):
#     """
#     Visualize a point cloud using Open3D.
    
#     Args:
#     points (np.ndarray): N x 3 or N x 4 array representing the point cloud.
#     """
#     # Ensure the input is N x 3
#     xyz = points[:, :3]  # Use only x, y, z
#     intensity = points[:, 3]

#     # Create an Open3D point cloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)

#     intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
#     pcd.colors = o3d.utility.Vector3dVector(np.tile(intensity_normalized[:, None], (1, 3)))

#     # Visualize
#     o3d.visualization.draw_geometries([pcd])

# if __name__ == '__main__':
#     # models set up
#     model_dir = '/home/mg/pdnn/Characterization-PDNN/mmdet_checkpoints/faster_rcnn_yolo'
#     model_list = list_filenames(model_dir, 'py')
#     model_list = [os.path.basename(config).split('.')[0] for config in model_list]

#     models = []    # key: model_name from model_list   # value[0] config value[1] checkpoint
#     for model in model_list:
#         model_pair = [model, None, None]  
#         # Iterate over each file in the directory
#         for file_name in os.listdir(model_dir):
#             if file_name.startswith(model):
#                 file_path = os.path.join(model_dir, file_name)
#                 # config file at 0 index, checkpoint file at 1 index
#                 if file_name.endswith('.py'):
#                     model_pair[1] = file_path
#                 elif file_name.endswith('.pth'):
#                     model_pair[2] = file_path
#         if model_pair == [model, None, None]:
#             continue  
#         models.append(model_pair)

#     # global variables used all the time

#     root_directory = f"/home/mg/pdnn/Characterization-PDNN/uni_model_exp/{RESULT_DIR}"
#     target_layers = ['e2e_time', 'backbone', 'rpn_head', 'roi_head', 'time_preprocessing', 'neck']


#     '''
#     model downloading
#     '''
#     # model_list = ['centernet_resnet18_140e_coco', 'detr_r50_8x2_150e_coco', 
#     #                 'faster_rcnn_r50_caffe_c4_1x_coco', 'faster_rcnn_r50_caffe_dc5_1x_coco',
#     #                 'pisa_ssd300_coco','retinanet_r50_fpg_crop640_50e_coco', 'ssd300_coco', 'ssdlite_mobilenetv2_scratch_600e_coco', 
#     #                 'yolov3_d53_fp16_mstrain-608_273e_coco', 'yolov3_mobilenetv2_mstrain-416_300e_coco', 'faster_rcnn_r50_fpn_1x_coco']
#     # model_downloads(['faster_rcnn_r50_caffe_c4_1x_coco'], '/home/mg/pdnn/Characterization-PDNN/mmdet_checkpoints/faster_rcnn_yolo')


#     '''
#     generate timing tree
#     '''
#     # result_dir = f'/home/mg/pdnn/Characterization-PDNN/uni_model_exp/faster_rcnn_s_overhead_1/clear_day_city_000e0252-8523a4a9'
#     # for model in models_to_include:
#     #     generate_timing_tree(f'{result_dir}/{model}_inference.csv', model, '/home/mg/pdnn/Characterization-PDNN/uni_model_exp')

#     '''
#     generate line or boxplot for inference time
#     '''
#     # for target_scene in SCENE_TYPES:
#     #     plot_results_model(VIDEO_DICT, root_directory, target_scene, target_layers, 
#     #                 models_to_exclude=None, models_to_include=MODEL_LIST, plot_type='line', mode='inference')

#     '''
#     ability to plot inference and proposals'''
#     plot_results(VIDEO_DICT, root_directory, SCENE_TYPES, target_layers, 'proposal', MODEL_LIST, 'line')

#     '''
#     visualize from predicion json
#     '''
#     # scene_ids = ['02d478d1-e6811391']
#     # for scene_id in scene_ids:
#     #     for model in MODEL_LIST:
#     #         visualize_from_pred(model, root_directory, scene_id)

#     '''
#     plot mAP based on category and bbox size
#     '''
#     # plot_cat_mAP(root_directory, 'rpn')

#     # plot_mAP(root_directory, 'rpn')

#     # target_scenes = ['rainy_night_city']
#     # for target_scene in target_scenes:
#     #     plot_results(video_dict, root_directory, output_directory, target_scene, target_layers, 
#     #             models_to_exclude=models_to_exclude, models_to_include=[model], plot_type='line', mode='proposal')

#     '''
#     visualization of specific video'''
#     # video_id = '02d478d1-e6811391'
#     # for model in MODEL_LIST:
#     #     plot_gt_TP_FP('rainy_night_city', video_id, model, root_directory, f'{root_directory}/{video_id}')
#     #     plot_fp_density_heatmap('rainy_night_city', video_id, model, root_directory, f'{root_directory}/{video_id}')
