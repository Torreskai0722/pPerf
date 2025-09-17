#!/usr/bin/env python3
"""
Test script to verify pipeline profiling with LidarDet3DInferencer.
This tests the actual mmdet3d inferencer to ensure pipeline hooking works correctly.
"""

import numpy as np
import os
import sys
import time

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
WARM_PCD_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_PCD = np.fromfile(WARM_PCD_PATH, dtype=np.float32).reshape(-1, 5)

from p_perf.pPerf import pPerf
from mmdet3d.apis import LidarDet3DInferencer

def test_lidar_inferencer_pipeline():
    """
    Test pipeline profiling with LidarDet3DInferencer.
    """
    print("=== Testing LidarDet3DInferencer Pipeline Profiling ===")
    
    # Test with a simple model that should have a pipeline
    model_name = 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d'  # This should have a pipeline
    
    try:
        # Initialize LidarDet3DInferencer
        print(f"Initializing LidarDet3DInferencer with model: {model_name}")
        inferencer = LidarDet3DInferencer(model_name)
        inferencer.show_progress = False
        
        # Check if pipeline exists
        if hasattr(inferencer, 'pipeline'):
            print(f"✓ Pipeline found with {len(inferencer.pipeline.transforms)} transforms")
            
            # Print pipeline transform types
            for i, transform in enumerate(inferencer.pipeline.transforms):
                print(f"  Transform {i}: {transform.__class__.__name__}")
        else:
            print("✗ No pipeline found in inferencer")
            return
        
        # Create pPerf instance
        print("Creating pPerf instance...")
        profiler = pPerf(
            model_name=model_name,
            inferencer=inferencer,
            depth=0,
            mode='lidar'
        )
        
        # Create test data (simulate point cloud)
        print("Creating test data...")
        test_data = dict(points=WARM_PCD)
        
        # Warm up
        print("Warming up...")
        profiler.warm_up(test_data, num_warmups=2)
        
        # Register hooks
        print("Registering hooks...")
        profiler.register_hooks(test_data)
        
        # Check pipeline profiling results
        print(f"\n=== Pipeline Profiling Results ===")
        print(f"Pipeline transforms traced: {len(profiler.pipeline_timings)}")
        print(f"Pipeline transforms called: {len(profiler.pipeline_transform_called)}")
        
        # Show pipeline transform details
        if profiler.pipeline_timings:
            print(f"\nPipeline transform timings:")
            for transform_id, (start, end, tag) in profiler.pipeline_timings.items():
                duration = (end - start) * 1000  # Convert to ms
                print(f"  {tag}: {duration:.2f} ms")
        
        # Run inference
        print("\nRunning inference...")
        result = profiler.run_inference(test_data, "test_lidar")
        
        print("\n✓ LidarDet3DInferencer pipeline profiling test completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_pipeline_structure():
    """
    Test to understand the pipeline structure of LidarDet3DInferencer.
    """
    print("\n=== Testing Pipeline Structure ===")
    
    model_name = 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d'
    
    try:
        inferencer = LidarDet3DInferencer(model_name)
        
        if hasattr(inferencer, 'pipeline'):
            print(f"Pipeline type: {type(inferencer.pipeline)}")
            print(f"Number of transforms: {len(inferencer.pipeline.transforms)}")
            
            print("\nTransform details:")
            for i, transform in enumerate(inferencer.pipeline.transforms):
                print(f"  {i}: {transform.__class__.__name__}")
                print(f"      Type: {type(transform)}")
                print(f"      Has __call__: {hasattr(transform, '__call__')}")
                if hasattr(transform, '__call__'):
                    print(f"      __call__ signature: {transform.__call__.__code__.co_argcount} args")
                print()
        
        # Test if we can wrap a transform
        if hasattr(inferencer, 'pipeline') and len(inferencer.pipeline.transforms) > 0:
            original_transform = inferencer.pipeline.transforms[0]
            print(f"Original transform: {original_transform}")
            
            # Test wrapping
            def test_wrapper(fn):
                def wrapped(*args, **kwargs):
                    print(f"Transform called with {len(args)} args")
                    return fn(*args, **kwargs)
                return wrapped
            
            wrapped_transform = test_wrapper(original_transform)
            inferencer.pipeline.transforms[0] = wrapped_transform
            
            print("✓ Transform wrapping test successful")
            
            # Restore original
            inferencer.pipeline.transforms[0] = original_transform
            
    except Exception as e:
        print(f"✗ Structure test failed: {e}")

if __name__ == "__main__":
    # Test pipeline structure first
    test_pipeline_structure()
    
    # Test full pipeline profiling
    test_lidar_inferencer_pipeline() 