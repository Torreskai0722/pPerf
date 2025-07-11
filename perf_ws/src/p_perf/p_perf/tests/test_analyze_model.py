#!/usr/bin/env python3

import os
from nuscenes.nuscenes import NuScenes
from p_perf.model_analyzer import ModelAnalyzer
from p_perf.nuscenes_instance import get_nuscenes_instance

def test_analyze_model():
    # Test parameters
    model_name = '/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py' 

    model_mode = "lidar"
    lidar_model_mode = "nus"
    scene_token = '2f0e54af35964a3fb347359836bec035'  # Example scene token
    output_dir = "/mmdetection3d_ros2/outputs/lidar_base"
    
    # Initialize NuScenes
    nusc = get_nuscenes_instance()
    
    # Create analyzer
    analyzer = ModelAnalyzer(
        nusc=nusc,
        scene_token=scene_token,
        model_name=model_name,
        model_mode=model_mode,
        lidar_model_mode=lidar_model_mode,
        output_dir=output_dir
    )
    
    # Run analysis
    results = analyzer.analyze_scene()
    
    # Print summary
    print(f"\nAnalysis Results for {model_name} ({model_mode}):")
    print(f"Total MACs: {results['total_macs']:,.0f}")
    print(f"Total FLOPS: {results['total_flops']:,.0f}")
    print(f"Number of frames analyzed: {results['frame_count']}")
    
    # Print per-token statistics
    print("\nPer-token statistics:")
    for token, stats in results['per_token_stats'].items():
        print(f"\nToken: {token}")
        print(f"Total MACs: {stats['total_macs']:,.0f}")
        print(f"Total FLOPS: {stats['total_flops']:,.0f}")
        print("Top 5 modules by MACs:")
        sorted_modules = sorted(stats['module_macs'].items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:5]
        for name, macs in sorted_modules:
            print(f"  {name}: {macs:,.0f} MACs")
    
    # Print overall top modules
    print("\nOverall top 5 modules by MACs:")
    sorted_modules = sorted(results['module_macs'].items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:5]
    for name, macs in sorted_modules:
        print(f"{name}: {macs:,.0f} MACs")

if __name__ == '__main__':
    test_analyze_model() 