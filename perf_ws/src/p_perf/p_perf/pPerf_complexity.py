#! /usr/bin/env python3

from mmdet3d.apis import LidarDet3DInferencer
from mmdet.apis import DetInferencer
import torch.cuda.nvtx as nvtx
import numpy as np
from p_perf.utils import load_sweep_sd, convert_to_kitti
from p_perf.nuscenes_instance import get_nuscenes_instance
import argparse
import logging
import sys

import warnings
warnings.filterwarnings("ignore")

# Configure logging to write to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def process_scene(nusc, scene_token, model_name, sensor_type='lidar', lidar_model_mode='nus'):
    # Initialize the model based on mode
    if sensor_type == 'lidar':
        inferencer = LidarDet3DInferencer(model_name)
        sensor_mode = 'LIDAR_TOP'
        frame_skip = 5  # Skip every 5 frames for lidar
    else:  # image mode
        inferencer = DetInferencer(model_name)
        sensor_mode = 'CAM_FRONT'  # default camera
        frame_skip = 3  # Skip every 3 frames for image
    
    inferencer.show_progress = False

    # Get the scene
    scene = nusc.get('scene', scene_token)
    
    # Get all sample data tokens for the specified sensor mode
    sd_tokens = load_sweep_sd(nusc, scene, sensor_channel=sensor_mode)

    # Process each frame
    for frame_idx, sd_token in enumerate(sd_tokens):
        # Skip frames based on sensor type
        if frame_idx % frame_skip != 0:
            continue
            
        # Get the sample data
        sd = nusc.get('sample_data', sd_token)
        data_path = f"{nusc.dataroot}/{sd['filename']}"
        
        # Create NVTX range for inference
        nvtx.range_push(f"Frame{frame_idx}")
        
        if sensor_type == 'lidar':
            # Load and convert point cloud data based on mode
            if lidar_model_mode == 'kitti':
                points = convert_to_kitti(nusc, sd_token)[:, :4] 
            else:  # nuscenes mode
                points = np.fromfile(data_path, dtype=np.float32).reshape(-1, 5)
            
            # Run inference for lidar
            results = inferencer(dict(points=points))
        else:
            # Run inference for image
            results = inferencer(data_path)
            
        nvtx.range_pop()  # End inference range
    
        logger.info(f"Frame {frame_idx} done")


def parse_args():
    parser = argparse.ArgumentParser(description='Process NuScenes data with specified model')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to use')
    parser.add_argument('--sensor_type', type=str, choices=['lidar', 'image'], default='lidar',
                      help='Mode of the model (lidar or image)')
    parser.add_argument('--lidar_model_mode', type=str, choices=['nus', 'kitti'], default='nus',
                      help='Format of lidar data (only used when model_mode is lidar)')
    parser.add_argument('--scene_token', type=str, required=True,
                      help='Token of the scene to process')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    nusc = get_nuscenes_instance()
    logger.info(f"Processing scene {args.scene_token} with model {args.model_name} and sensor type {args.sensor_type}") 
    process_scene(
        nusc=nusc,
        scene_token=args.scene_token,
        model_name=args.model_name,
        sensor_type=args.sensor_type,
        lidar_model_mode=args.lidar_model_mode
    )