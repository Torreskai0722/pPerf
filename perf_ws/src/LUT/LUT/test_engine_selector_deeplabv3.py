#!/usr/bin/env python3
"""
Test engine_selector with DeepLabV3+ operator 63 using different engine configs
"""

import torch
import sys
import numpy as np

from p_perf.pPerf_inferencer import bddSegInferencer
from p_perf.pPerf import pPerf
from LUT.engine_selector import EngineSelector

print("=" * 80)
print("ENGINE_SELECTOR TEST - DeepLabV3+ Operator 63")
print("=" * 80)
print()

# Load warm-up image
WARM_IMAGE_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'
try:
    from PIL import Image
    WARM_IMAGE = np.array(Image.open(WARM_IMAGE_PATH))
    print(f"✓ Loaded test image: {WARM_IMAGE.shape}\n")
except Exception as e:
    print(f"✗ Failed to load image: {e}")
    sys.exit(1)

# Load model
model_name = 'deeplabv3+_r50-d8_512x1024_40k_sem_seg_bdd100k'
print(f"Loading model: {model_name}")
inferencer = bddSegInferencer(model_name, mode='sem_seg')
inferencer.model.eval()
print("✓ Model loaded\n")

# Target operator 63 (64th convolution, since counting starts at 1)
target_op = 64

# Prepare data
warm_data = dict(img=WARM_IMAGE)

# Test baseline
print("=" * 80)
print("BASELINE (no engine selector)")
print("=" * 80)

profiler_baseline = pPerf(
    model_name=model_name,
    inferencer=inferencer,
    depth=0,
    mode='sem_seg',
    engine_selector=None
)
profiler_baseline.warm_up(warm_data)
profiler_baseline.register_hooks(warm_data)


baseline_result = profiler_baseline.run_inference(warm_data, input_name='baseline')
print("✓ Baseline complete\n")

# Test different engine configs
print("=" * 80)
print("TESTING DIFFERENT ENGINE CONFIGS FOR OPERATOR 63")
print("=" * 80)
print()

# Test engine configs 0-9
test_engines = list(range(10))
results = {}

for engine_idx in test_engines:
    print(f"Engine {engine_idx}: ", end='', flush=True)
    
    try:
        # Create engine selector: map operator 63 to this engine
        engine_map = {str(target_op): engine_idx}
        selector = EngineSelector(engine_map)
        
        # Create pPerf profiler with engine selector
        profiler = pPerf(
            model_name=model_name,
            inferencer=inferencer,
            depth=0,
            mode='sem_seg',
            engine_selector=selector
        )
        
        result = profiler.run_inference(warm_data, input_name=f'engine_{engine_idx}')
        
        results[engine_idx] = {'passed': True, 'result': result}
        print("✓ PASS")
        
    except Exception as e:
        results[engine_idx] = {'passed': False, 'error': str(e)}
        print(f"✗ FAIL - {e}")

print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

passed = [idx for idx, res in results.items() if res['passed']]
failed = [idx for idx, res in results.items() if not res['passed']]

print(f"Operator: {target_op}")
print(f"Tested: {len(test_engines)} engine configs")
print(f"Passed: {len(passed)} - {passed}")
print(f"Failed: {len(failed)} - {failed}")
print()

if failed:
    print("Failed engine details:")
    for idx in failed:
        print(f"  Engine {idx}: {results[idx]['error']}")
    print()

print("=" * 80)

sys.exit(0 if not failed else 1)
