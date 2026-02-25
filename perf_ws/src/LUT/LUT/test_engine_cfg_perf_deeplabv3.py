#!/usr/bin/env python3
"""
Real-world test for engine_cfg_perf.cpp using DeepLabV3+ model

Tests the custom convolution implementation against PyTorch's native 
convolution using actual convolution operations from DeepLabV3+.
"""

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import sys
from typing import List, Dict

# Import the inferencer and utilities
from p_perf.pPerf_inferencer import bddSegInferencer
from LUT.ops_recorder import OpRecorder
from LUT.utils import move_to_device

print("=" * 70)
print("DEEPLABV3+ ENGINE_CFG_PERF TEST")
print("=" * 70)
print()

# Load the extension
print("Loading engine_cfg_perf extension...")
try:
    load(
        name="my_cudnn_prof_ext",
        sources=["/mmdetection3d_ros2/perf_ws/src/LUT/LUT/engine_cfg_perf.cpp"],
        with_cuda=True,
        extra_cflags=["-O0", "-g", "-fno-omit-frame-pointer"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
        is_python_module=False,
    )
    print("✓ Extension loaded successfully\n")
except Exception as e:
    print(f"✗ Failed to load extension: {e}")
    sys.exit(1)

# Load warm-up image
WARM_IMAGE_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'
try:
    import numpy as np
    from PIL import Image
    WARM_IMAGE = np.array(Image.open(WARM_IMAGE_PATH))
    print(f"✓ Loaded test image: {WARM_IMAGE_PATH}")
    print(f"  Image shape: {WARM_IMAGE.shape}\n")
except Exception as e:
    print(f"✗ Failed to load image: {e}")
    sys.exit(1)

# Load model
model_name = 'deeplabv3+_r50-d8_512x1024_40k_sem_seg_bdd100k'
print(f"Loading model: {model_name}")
try:
    inferencer = bddSegInferencer(model_name, mode='sem_seg')
    inferencer.model.eval()
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Record operations
print("Recording convolution operations...")
rec = OpRecorder(record_caller=True)

with torch.no_grad(), rec:
    result = inferencer(dict(img=WARM_IMAGE))
    
print(f"✓ Recorded {len(rec.events)} operations")

# Filter convolution events
conv_events = [e for e in rec.events if e["op_name"].startswith("convolution")]
print(f"✓ Found {len(conv_events)} convolution operations\n")

if len(conv_events) == 0:
    print("✗ No convolution operations found!")
    sys.exit(1)

# Helper functions
def _as_pair(v):
    """conv.stride/padding/dilation may be int or tuple"""
    if isinstance(v, (list, tuple)):
        return [int(v[0]), int(v[1])]
    return [int(v), int(v)]

def test_conv_operation(event: Dict, test_idx: int, tolerance: float = 1e-4) -> bool:
    """
    Test a single convolution operation.
    
    Returns:
        True if test passed, False otherwise
    """
    device = "cuda:0"
    
    # Move args back to device
    args = move_to_device(event["args"], device)
    
    # Extract convolution parameters
    input_tensor = args[0]
    weight = args[1]
    bias = args[2]
    stride = args[3]
    padding = args[4]
    dilation = args[5]
    groups = args[8] if len(args) > 8 else 1
    
    # Prepare tensors
    x = input_tensor.contiguous().float()
    w = weight.contiguous().float()
    if bias is None:
        b_pytorch = None
        b_custom = torch.zeros((w.shape[0],), device=w.device, dtype=torch.float32)
    else:
        b_pytorch = bias.contiguous().float()
        b_custom = b_pytorch
    
    op_id = event.get("op_id", f"conv_{test_idx}")
    caller_info = event.get("caller", {})
    
    print(f"Test {test_idx}: op_id={op_id}")
    if caller_info:
        print(f"  Caller: {caller_info.get('module', 'unknown')}.{caller_info.get('function', 'unknown')}")
    print(f"  Input: {tuple(x.shape)}, Weight: {tuple(w.shape)}")
    print(f"  stride={stride}, padding={padding}, dilation={dilation}, groups={groups}")
    
    try:
        # Run PyTorch native convolution
        with torch.no_grad():
            y_pytorch = F.conv2d(
                x, w, b_pytorch,
                stride=tuple(stride),
                padding=tuple(padding),
                dilation=tuple(dilation),
                groups=groups
            )
        
        # Run custom engine_cfg_perf (use first engine config only)
        with torch.no_grad():
            y_custom, engine_table = torch.ops.LUT_perf.profile_conv2d(
                x, w, b_custom,
                _as_pair(stride),
                _as_pair(padding),
                _as_pair(dilation),
                groups,
                1,  # max_engine_variants - use first only
                1,  # iters
                str(op_id)
            )
        
        # Check shapes match
        shape_match = y_pytorch.shape == y_custom.shape
        
        # Compare output values
        abs_diff = torch.abs(y_pytorch - y_custom)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        
        # Relative error
        rel_diff = abs_diff / (torch.abs(y_pytorch) + 1e-8)
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        # Check if test passed
        passed = shape_match and max_abs_diff < tolerance
        
        # Print results
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}")
        print(f"  Output shapes - PyTorch: {tuple(y_pytorch.shape)}, Custom: {tuple(y_custom.shape)}")
        print(f"  Shape match: {'✓' if shape_match else '✗'}")
        
        print(f"  PyTorch output - mean: {y_pytorch.mean().item():.6f}, std: {y_pytorch.std().item():.6f}, "
              f"min: {y_pytorch.min().item():.6f}, max: {y_pytorch.max().item():.6f}")
        print(f"  Custom output  - mean: {y_custom.mean().item():.6f}, std: {y_custom.std().item():.6f}, "
              f"min: {y_custom.min().item():.6f}, max: {y_custom.max().item():.6f}")
        
        print(f"  PyTorch first 5: [{', '.join(f'{v:.6f}' for v in y_pytorch.flatten()[:5].tolist())}]")
        print(f"  Custom first 5:  [{', '.join(f'{v:.6f}' for v in y_custom.flatten()[:5].tolist())}]")
        
        print(f"  Max abs diff: {max_abs_diff:.2e}, Mean abs diff: {mean_abs_diff:.2e}")
        print(f"  Max rel diff: {max_rel_diff:.2e}, Mean rel diff: {mean_rel_diff:.2e}")
        
        if shape_match:
            max_diff_idx = abs_diff.argmax()
            max_diff_loc = torch.unravel_index(max_diff_idx, y_pytorch.shape)
            print(f"  Max diff at: {tuple(int(i) for i in max_diff_loc)} "
                  f"(PyTorch: {y_pytorch[max_diff_loc].item():.6f}, "
                  f"Custom: {y_custom[max_diff_loc].item():.6f})")
        
        print(f"  Engine variants: {engine_table.shape[0]} (using first only)")
        if engine_table.shape[0] > 0:
            print(f"  First engine timing: {engine_table[0, 1].item():.6f} ms")
        
        if not passed:
            if not shape_match:
                print(f"  ⚠ Shape mismatch!")
            if max_abs_diff >= tolerance:
                print(f"  ⚠ Value tolerance exceeded! (threshold: {tolerance:.2e})")
        
        print()
        return passed
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        print()
        return False

# Test all convolution operations (or a subset)
print("=" * 70)
print("TESTING CONVOLUTION OPERATIONS")
print("=" * 70)
print()

# Test first 10 convolutions (or all if fewer)
num_tests = min(10, len(conv_events))
print(f"Testing first {num_tests} convolution operations (out of {len(conv_events)} total)\n")

results = []
for idx in range(num_tests):
    passed = test_conv_operation(conv_events[idx], idx + 1)
    results.append(passed)

# Summary
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
total = len(results)
passed_count = sum(results)
failed_count = total - passed_count

print(f"Total tests:  {total}")
print(f"Passed:       {passed_count} ({100*passed_count/total:.1f}%)")
print(f"Failed:       {failed_count} ({100*failed_count/total:.1f}%)")
print("=" * 70)

if failed_count > 0:
    print(f"\n⚠ {failed_count} test(s) failed!")
    exit_code = 1
else:
    print("\n✓ All tests passed!")
    exit_code = 0

sys.exit(exit_code)

