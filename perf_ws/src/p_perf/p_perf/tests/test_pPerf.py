#!/usr/bin/python3

import os
import cv2
import numpy as np
import torch
import pandas as pd
import time
from nuscenes.nuscenes import NuScenes

from p_perf.pPerf import pPerf
from p_perf.pre_process.pPerf_inferencer import pPerf3dDetInferencer, pPerf2dDetInferencer
from p_perf.general_utils import load_sweep_sd, get_paths_from_sd, convert_to_kitti

# Configuration
TEST_MODE = 'image'  # 'lidar' or 'image'
SCENE_TOKEN = 'cc8c0bf57f984915a77078b10eb33198'  # Default scene
DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
LIDAR_MODEL_MODE = 'nus'  # 'nus' or 'kitti'

# Initialize NuScenes
nusc = NuScenes(
    version='v1.0-mini',
    dataroot=DATA_ROOT,
    verbose=True
)

# Camera keys (same as sensor_publisher)
cam_keys = [
    'CAM_FRONT', 
]

def load_scene_data():
    """Load all data for the scene, similar to sensor_publisher's preload_all_data method"""
    print("Loading scene data...")
    
    # Get scene
    scene = nusc.get('scene', SCENE_TOKEN)
    print(f"Scene: {scene['name']}")
    
    # Load LIDAR data
    print("Loading LIDAR data...")
    lidar_tokens = load_sweep_sd(nusc, scene, 'LIDAR_TOP')
    lidar_files = get_paths_from_sd(nusc, lidar_tokens)
    
    lidar_data = []
    for i in range(len(lidar_files)):
        try:
            path = lidar_files[i]
            token = lidar_tokens[i]
            if 'kitti' in LIDAR_MODEL_MODE:
                points = convert_to_kitti(nusc, token)
            elif LIDAR_MODEL_MODE == 'nus':
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
            else:
                raise ValueError("Unsupported lidar_model_mode")
            
            lidar_data.append((token, points))
            print(f"Loaded LIDAR {i+1}/{len(lidar_files)}: {token}")
            
        except Exception as e:
            print(f"Failed to load LIDAR {path}: {e}")
    
    # Load IMAGE data
    print("Loading IMAGE data...")
    image_data = {cam: [] for cam in cam_keys}
    
    for cam in cam_keys:
        image_tokens = load_sweep_sd(nusc, scene, cam)
        image_files = get_paths_from_sd(nusc, image_tokens)
        
        for i in range(len(image_files)):
            try:
                path = image_files[i]
                token = image_tokens[i]
                img = cv2.imread(path)
                if img is None:
                    raise ValueError("cv2.imread returned None")
                
                image_data[cam].append((token, img))
                print(f"Loaded IMAGE {i+1}/{len(image_files)} from {cam}: {token}")
                
            except Exception as e:
                print(f"Failed to load IMAGE {path} from {cam}: {e}")
    
    return lidar_data, image_data

def test_lidar_mode(lidar_data):
    """Test LIDAR mode using pPerf3dDetInferencer"""
    print("\n=== Testing LIDAR Mode ===")
    
    # Initialize model
    model_name = 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d'
    print(f"Initializing {model_name}...")
    
    inferencer = pPerf3dDetInferencer(model_name)
    inferencer.show_progress = False
    inferencer.model.eval()
    
    # Initialize profiler
    profiler = pPerf(model_name, inferencer, depth=0, mode='lidar')
    
    # Warm up with first data point
    if lidar_data:
        warm_token, warm_points = lidar_data[0]
        print(f"Warming up with token: {warm_token}")
        profiler.warm_up(dict(points=warm_points))
        profiler.register_hooks(dict(points=warm_points))
    
    # Process all LIDAR data
    start_time = time.time()
    
    for i, (token, points) in enumerate(lidar_data):
        
        # Run inference with NVTX annotation
        torch.cuda.nvtx.range_push(f"{token}.lidar_inference")
        try:
            result = profiler.run_inference(dict(points=points), token)
        except Exception as e:
            print(f"  Error processing {token}: {e}")
        finally:
            torch.cuda.nvtx.range_pop()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"LIDAR processing completed in {total_time:.2f} seconds")
    print(f"Average time per frame: {total_time/len(lidar_data):.3f} seconds")
    
    # Generate summary
    profiler.summary()
    
    return profiler

def test_image_mode(image_data):
    """Test IMAGE mode using pPerf2dDetInferencer"""
    print("\n=== Testing IMAGE Mode ===")
    
    # Initialize model
    model_name = 'faster-rcnn_r50_fpn_1x_coco'
    print(f"Initializing {model_name}...")
    
    inferencer = pPerf2dDetInferencer(model_name)
    inferencer.show_progress = False
    inferencer.model.eval()
    
    # Initialize profiler
    profiler = pPerf(model_name, inferencer, depth=0, mode='image')
    
    # Warm up with first data point
    if image_data and image_data['CAM_FRONT']:
        warm_token, warm_img = image_data['CAM_FRONT'][0]
        print(f"Warming up with token: {warm_token}")
        profiler.warm_up(dict(img=warm_img))
        profiler.register_hooks(dict(img=warm_img))
    
    # Process all IMAGE data
    total_frames = sum(len(frames) for frames in image_data.values())
    print(f"Processing {total_frames} IMAGE frames...")
    start_time = time.time()
    
    for cam in cam_keys:
        if not image_data[cam]:
            continue
            
        for i, (token, img) in enumerate(image_data[cam]):
            
            # Run inference with NVTX annotation
            torch.cuda.nvtx.range_push(f"{token}.{cam}.image_inference")
            try:
                result = profiler.run_inference(dict(img=img), token)
            except Exception as e:
                print(f"  Error processing {token}: {e}")
            finally:
                torch.cuda.nvtx.range_pop()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"IMAGE processing completed in {total_time:.2f} seconds")
    print(f"Average time per frame: {total_time/total_frames:.3f} seconds")
    
    # Generate summary
    profiler.summary()
    
    return profiler

def main():
    """Main function to run the test"""
    print("=== pPerf Test with Sensor Publisher Data Loading Simulation ===")
    print(f"Test Mode: {TEST_MODE}")
    print(f"Scene Token: {SCENE_TOKEN}")
    print(f"LIDAR Model Mode: {LIDAR_MODEL_MODE}")
    
    # Load all scene data
    lidar_data, image_data = load_scene_data()
    
    print(f"\nData Summary:")
    print(f"  LIDAR frames: {len(lidar_data)}")
    for cam in cam_keys:
        print(f"  {cam} frames: {len(image_data[cam])}")
    
    # Run appropriate test mode
    if TEST_MODE == 'lidar':
        profiler = test_lidar_mode(lidar_data)
    elif TEST_MODE == 'image':
        profiler = test_image_mode(image_data)
    else:
        print(f"Unknown test mode: {TEST_MODE}")
        return
    
    print("\n=== Test Completed ===")

if __name__ == '__main__':
    main()
