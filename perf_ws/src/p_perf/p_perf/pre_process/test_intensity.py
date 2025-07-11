#!/usr/bin/env python3
"""
Test script for intensity modification and pPerf inference.

This script reads a LiDAR PCD file, modifies the intensity values in different ways:
- Uniform intensity
- Random intensity
- Gaussian noise added to original intensity
- Zero intensity
- Maximum intensity

Then runs inference using pPerf to analyze performance differences.
"""

import numpy as np
import os
import time
from typing import Dict, List, Tuple
from pathlib import Path

# Set environment variables to avoid Qt/display issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':99'

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import pPerf components
from p_perf.pPerf import pPerf
from mmdet3d.apis import LidarDet3DInferencer

def convert_to_kitti(scan: np.ndarray) -> np.ndarray:

    # # Normalize intensity
    # scan[:, 3] = (scan[:, 3] - scan[:, 3].min()) / max(1e-5, scan[:, 3].ptp())
    
    # Transform from nuScenes to KITTI
    x_nusc = scan[:, 0]
    y_nusc = scan[:, 1]
    z_nusc = scan[:, 2]
    intensity = scan[:, 3]

    x_kitti = y_nusc
    y_kitti = -x_nusc
    z_kitti = z_nusc

    scan_kitti = np.stack((x_kitti, y_kitti, z_kitti, intensity), axis=1)

    # KITTI format uses an extra dummy column (e.g., reflectivity or ring index)
    # zeros_col = np.zeros((scan_kitti.shape[0], 1), dtype=np.float32)
    # scan_kitti = np.hstack((scan_kitti, zeros_col))
    return scan_kitti


def load_pcd_file(file_path: str, kitti: bool = False) -> np.ndarray:
    """
    Load a PCD file and return point cloud data.
    
    Args:
        file_path: Path to the PCD file
        kitti: Whether to convert to KITTI format
        
    Returns:
        Point cloud array with shape (N, 4) or (N, 5) [x, y, z, intensity, ring]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PCD file not found: {file_path}")
    
    # Load binary PCD data
    points = np.fromfile(file_path, dtype=np.float32)
    points_reshaped = points.reshape(-1, 5)  # [x, y, z, intensity, ring]
    
    if kitti:
        points_reshaped = convert_to_kitti(points_reshaped)
    
    return points_reshaped


def modify_intensity_uniform(points: np.ndarray, value: float = 0.5) -> np.ndarray:
    """
    Set all intensity values to a uniform value.
    
    Args:
        points: Point cloud array (N, 4) or (N, 5)
        value: Uniform intensity value (default: 0.5)
        
    Returns:
        Modified point cloud array
    """
    modified_points = points.copy()
    modified_points[:, 3] = value  # Intensity is always at index 3
    return modified_points


def modify_intensity_random(points: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Set intensity values to random values within a range.
    
    Args:
        points: Point cloud array (N, 4) or (N, 5)
        min_val: Minimum intensity value
        max_val: Maximum intensity value
        
    Returns:
        Modified point cloud array
    """
    modified_points = points.copy()
    modified_points[:, 3] = np.random.uniform(min_val, max_val, len(points))  # Intensity is always at index 3
    return modified_points


def modify_intensity_gaussian_noise(points: np.ndarray, std: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to original intensity values.
    
    Args:
        points: Point cloud array (N, 4) or (N, 5)
        std: Standard deviation of Gaussian noise
        
    Returns:
        Modified point cloud array
    """
    modified_points = points.copy()
    # noise = np.random.normal(0.5, std, len(points))
    modified_points[:, 3] = np.clip(modified_points[:, 3], 0, 1)

    return modified_points


def modify_intensity_zero(points: np.ndarray) -> np.ndarray:
    """
    Set all intensity values to zero.
    
    Args:
        points: Point cloud array (N, 4) or (N, 5)
        
    Returns:
        Modified point cloud array
    """
    modified_points = points.copy()
    modified_points[:, 3] = 0.0  # Intensity is always at index 3
    return modified_points


def modify_intensity_max(points: np.ndarray) -> np.ndarray:
    """
    Set all intensity values to maximum (1.0).
    
    Args:
        points: Point cloud array (N, 4) or (N, 5)
        
    Returns:
        Modified point cloud array
    """
    modified_points = points.copy()
    modified_points[:, 3] = 1.0  # Intensity is always at index 3
    return modified_points


def modify_intensity_inverse(points: np.ndarray) -> np.ndarray:
    """
    Invert intensity values (1 - original_intensity).
    
    Args:
        points: Point cloud array (N, 4) or (N, 5)
        
    Returns:
        Modified point cloud array
    """
    modified_points = points.copy()
    modified_points[:, 3] = 1.0 - modified_points[:, 3]  # Intensity is always at index 3
    return modified_points


def run_inference_with_pPerf(profiler, points: np.ndarray, modification_type: str = "original") -> Dict:
    """
    Run inference using pPerf and return performance metrics.
    
    Args:
        profiler: Pre-initialized pPerf profiler
        points: Point cloud data
        modification_type: Type of intensity modification applied
        
    Returns:
        Dictionary containing performance metrics
    """
    # Prepare data in the format expected by the inferencer
    data = dict(points=points.astype(np.float32))
    
    # Run inference with profiling
    start_time = time.time()
    
    # Register hooks and run inference
    result = profiler.run_inference(data, modification_type)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time


def main():
    """Main function to run the intensity modification tests."""
    # Configuration parameters
    pcd_file = '/mmdetection3d_ros2/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489308396618.pcd.bin'
    model_name = '3dssd_4x4_kitti-3d-car'
    depth = 0
    gaussian_std = 0.15
    random_min = 0.0
    random_max = 1.0
    
    # Load original point cloud
    original_points = load_pcd_file(pcd_file, kitti=True)
    
    # Define intensity modifications
    modifications = {
        'original': lambda p: p.copy(),
        'random': lambda p: modify_intensity_random(p, random_min, random_max),
        'gaussian_noise': lambda p: modify_intensity_gaussian_noise(p, gaussian_std),
        'zero': modify_intensity_zero,
        'max': modify_intensity_max,
        'inverse': modify_intensity_inverse
    }
    
    # Preload all 100 Gaussian noise point clouds
    gaussian_point_clouds = []
    for run_idx in range(1000):
        np.random.seed(run_idx)
        modified_points = modify_intensity_gaussian_noise(original_points, gaussian_std)
        gaussian_point_clouds.append(modified_points)
    
    # Initialize inferencer and profiler once
    inferencer = LidarDet3DInferencer(model_name)
    inferencer.show_progress = False
    profiler = pPerf(model_name, inferencer, depth, mode='lidar')
    
    # Warm up the profiler once
    warmup_data = dict(points=original_points.astype(np.float32))
    profiler.warm_up(warmup_data, num_warmups=10)
    profiler.register_hooks(warmup_data)

    # Run tests
    for mod_name, mod_func in modifications.items():
        # Apply modification
        modified_points = mod_func(original_points)
        
        # Run inference 3 times
        for run_idx in range(3):
            try:
                run_inference_with_pPerf(profiler, modified_points, mod_name)
            except Exception:
                pass
    
    timing_results = []
    # Run inference with preloaded Gaussian noise point clouds
    for run_idx in range(1000):
        try:
            timing_results.append(run_inference_with_pPerf(profiler, gaussian_point_clouds[run_idx], f"gaussian_{run_idx}"))
        except Exception:
            pass

    print(timing_results)
    print(np.mean(timing_results))
    print(np.std(timing_results))
    print(np.min(timing_results))
    print(np.max(timing_results))
    
    # Create simple bar plot of inference time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(timing_results, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Inference Time (seconds)')
    plt.ylabel('Count')
    plt.title('Distribution of Inference Times (300 Gaussian Noise Runs)')
    plt.grid(True, alpha=0.3)
    plt.savefig('inference_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
