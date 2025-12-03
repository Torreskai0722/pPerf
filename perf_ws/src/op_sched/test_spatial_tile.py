#!/usr/bin/env python3
"""
Test script to verify SpatiallyTiledModule functionality
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/mg/pdnn/pPerf/perf_ws/src/op_sched')

from op_sched.spatial_tile import SpatiallyTiledModule


def print_diff_histogram(diff_tensor, title="Difference Histogram"):
    """Print a simple ASCII histogram of difference magnitudes"""
    diffs = diff_tensor.flatten().cpu().numpy()
    
    # Create bins
    bins = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    bin_labels = ['0', '1e-10', '1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3+']
    
    import numpy as np
    hist, _ = np.histogram(diffs, bins=bins + [np.inf])
    
    print(f"\n{title}:")
    max_count = max(hist) if max(hist) > 0 else 1
    for i, (label, count) in enumerate(zip(bin_labels, hist)):
        bar_length = int(40 * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_length
        pct = 100 * count / len(diffs)
        print(f"  [{label:>8}]: {bar:<40} {count:>8} ({pct:>5.2f}%)")


def test_basic_functionality():
    """Test basic tiling with a simple Conv2d module"""
    print("Test 1: Basic functionality with Conv2d")
    print("-" * 50)
    
    # Create a simple Conv2d module
    conv = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    conv.eval()
    
    # Create tiled version (2x2 tiling)
    tiled_conv = SpatiallyTiledModule(conv, splits_h=2, splits_w=2, halo=1)
    
    # Create test input
    x = torch.randn(1, 16, 64, 64)
    
    # Run through both
    with torch.no_grad():
        y_normal = conv(x)
        y_tiled = tiled_conv(x)
    
    # Check if outputs match
    diff_tensor = torch.abs(y_normal - y_tiled)
    max_diff = diff_tensor.max().item()
    mean_diff = diff_tensor.mean().item()
    std_diff = diff_tensor.std().item()
    num_nonzero = (diff_tensor > 1e-8).sum().item()
    total_elements = diff_tensor.numel()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (normal): {y_normal.shape}")
    print(f"Output shape (tiled): {y_tiled.shape}")
    print(f"\nElement-wise Difference Analysis:")
    print(f"  Max difference:      {max_diff:.6e}")
    print(f"  Mean difference:     {mean_diff:.6e}")
    print(f"  Std difference:      {std_diff:.6e}")
    print(f"  Non-zero diffs:      {num_nonzero}/{total_elements} ({100*num_nonzero/total_elements:.2f}%)")
    
    print_diff_histogram(diff_tensor, "Distribution of element-wise differences")
    
    if max_diff < 1e-6:
        print("✅ PASS: Outputs match!")
    else:
        print("❌ FAIL: Outputs don't match!")
    
    return max_diff < 1e-6


def test_no_tiling():
    """Test that no tiling (1x1) produces identical results"""
    print("\nTest 2: No tiling (1x1 splits)")
    print("-" * 50)
    
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    conv.eval()
    
    # No tiling
    tiled_conv = SpatiallyTiledModule(conv, splits_h=1, splits_w=1, halo=0)
    
    x = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        y_normal = conv(x)
        y_tiled = tiled_conv(x)
    
    diff = torch.abs(y_normal - y_tiled).max().item()
    print(f"Max difference: {diff:.6e}")
    
    if diff == 0:
        print("✅ PASS: No tiling produces identical output!")
    else:
        print("❌ FAIL: No tiling should produce identical output!")
    
    return diff == 0


def test_uneven_splits():
    """Test with non-divisible dimensions"""
    print("\nTest 3: Non-divisible dimensions (65x65 with 2x2 splits)")
    print("-" * 50)
    
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    conv.eval()
    
    # 2x2 tiling on odd dimensions
    tiled_conv = SpatiallyTiledModule(conv, splits_h=2, splits_w=2, halo=2)
    
    x = torch.randn(1, 3, 65, 65)
    
    with torch.no_grad():
        y_normal = conv(x)
        y_tiled = tiled_conv(x)
    
    diff_tensor = torch.abs(y_normal - y_tiled)
    max_diff = diff_tensor.max().item()
    mean_diff = diff_tensor.mean().item()
    std_diff = diff_tensor.std().item()
    num_nonzero = (diff_tensor > 1e-8).sum().item()
    total_elements = diff_tensor.numel()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (normal): {y_normal.shape}")
    print(f"Output shape (tiled): {y_tiled.shape}")
    print(f"\nElement-wise Difference Analysis:")
    print(f"  Max difference:      {max_diff:.6e}")
    print(f"  Mean difference:     {mean_diff:.6e}")
    print(f"  Std difference:      {std_diff:.6e}")
    print(f"  Non-zero diffs:      {num_nonzero}/{total_elements} ({100*num_nonzero/total_elements:.2f}%)")
    
    print_diff_histogram(diff_tensor, "Distribution of element-wise differences")
    
    if max_diff < 1e-6:
        print("✅ PASS: Uneven splits work correctly!")
    else:
        print("❌ FAIL: Uneven splits failed!")
    
    return max_diff < 1e-6


def test_different_split_patterns():
    """Test various split patterns"""
    print("\nTest 4: Various split patterns")
    print("-" * 50)
    
    conv = nn.Conv2d(3, 8, kernel_size=5, padding=2)
    conv.eval()
    
    x = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        y_normal = conv(x)
    
    patterns = [
        (1, 2, 2),  # 1x2 splits
        (2, 1, 2),  # 2x1 splits
        (4, 4, 3),  # 4x4 splits
        (3, 3, 2),  # 3x3 splits
    ]
    
    all_pass = True
    for splits_h, splits_w, halo in patterns:
        tiled_conv = SpatiallyTiledModule(conv, splits_h=splits_h, splits_w=splits_w, halo=halo)
        
        with torch.no_grad():
            y_tiled = tiled_conv(x)
        
        diff_tensor = torch.abs(y_normal - y_tiled)
        max_diff = diff_tensor.max().item()
        mean_diff = diff_tensor.mean().item()
        num_nonzero = (diff_tensor > 1e-8).sum().item()
        total_elements = diff_tensor.numel()
        
        status = "✅" if max_diff < 1e-6 else "❌"
        print(f"{status} {splits_h}x{splits_w} splits, halo={halo}:")
        print(f"     max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, non-zero={num_nonzero}/{total_elements} ({100*num_nonzero/total_elements:.2f}%)")
        
        if max_diff >= 1e-6:
            all_pass = False
    
    if all_pass:
        print("✅ PASS: All split patterns work correctly!")
    else:
        print("❌ FAIL: Some split patterns failed!")
    
    return all_pass


def test_with_complex_module():
    """Test with a more complex module (small network)"""
    print("\nTest 5: Complex module (Sequential network)")
    print("-" * 50)
    
    # Create a small sequential network
    module = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 3, 3, padding=1),
    )
    module.eval()
    
    tiled_module = SpatiallyTiledModule(module, splits_h=2, splits_w=2, halo=4)
    
    x = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        y_normal = module(x)
        y_tiled = tiled_module(x)
    
    diff_tensor = torch.abs(y_normal - y_tiled)
    max_diff = diff_tensor.max().item()
    mean_diff = diff_tensor.mean().item()
    std_diff = diff_tensor.std().item()
    num_nonzero = (diff_tensor > 1e-8).sum().item()
    total_elements = diff_tensor.numel()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_normal.shape}")
    print(f"\nElement-wise Difference Analysis:")
    print(f"  Max difference:      {max_diff:.6e}")
    print(f"  Mean difference:     {mean_diff:.6e}")
    print(f"  Std difference:      {std_diff:.6e}")
    print(f"  Non-zero diffs:      {num_nonzero}/{total_elements} ({100*num_nonzero/total_elements:.2f}%)")
    
    print_diff_histogram(diff_tensor, "Distribution of element-wise differences")
    
    if max_diff < 1e-6:
        print("✅ PASS: Complex module works correctly!")
    else:
        print("❌ FAIL: Complex module failed!")
    
    return max_diff < 1e-6


def main():
    print("="*60)
    print("Testing SpatiallyTiledModule")
    print("="*60)
    
    results = []
    
    results.append(test_basic_functionality())
    results.append(test_no_tiling())
    results.append(test_uneven_splits())
    results.append(test_different_split_patterns())
    results.append(test_with_complex_module())
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ ALL TESTS PASSED - SpatiallyTiledModule is workable!")
    else:
        print("❌ SOME TESTS FAILED - There may be issues!")
    
    return all(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

