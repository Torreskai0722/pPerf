#!/usr/bin/env python3
"""
Test script for the create_experiment_config method of pPerfConfigManager
"""

import sys
import os
from pathlib import Path
from p_perf.pPerf_manager import pPerfConfigManager

def test_create_experiment_config():
    manager = pPerfConfigManager("/mmdetection3d_ros2/perf_ws/test_manager.yaml")
    manager.update_sensor_replayer()
    manager.add_det_inferencer(inferencer_index='lidar_det', mode='lidar')
    manager.add_seg_inferencer(inferencer_index='image_seg', mode='sem_seg')
    manager.list_inferencers()
    manager.update_det_inferencer(inferencer_index='lidar_det', model_name='pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', index=0)
    manager.update_seg_inferencer(inferencer_index='image_seg', model_name='yolov3_d53_320_273e_coco',index=0)
    manager.list_inferencers()
    manager.update_det_inferencer(inferencer_index='lidar_det', model_name='pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',index=1)
    manager.update_seg_inferencer(inferencer_index='image_seg', model_name='yolov3_d53_320_273e_coco',index=1)
    manager.list_inferencers()
    # manager.remove_det_inferencer(inferencer_index='lidar_det')
    # manager.remove_seg_inferencer(inferencer_index='image_seg')
    # manager.list_inferencers()
    manager.save_config()

if __name__ == "__main__":
    test_create_experiment_config() 