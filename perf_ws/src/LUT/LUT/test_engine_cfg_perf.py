#!/usr/bin/env python3
"""
Test script for engine_cfg_perf.cpp

Validates that the custom CUDNN backend convolution implementation produces
identical results to PyTorch's native convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import sys
from typing import List, Dict, Tuple, Optional

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


class ConvTester:
    """Test harness for comparing engine_cfg_perf against PyTorch native convolution."""
    
    def __init__(self, device: str = "cuda", tolerance: float = 1e-4):
        """
        Initialize the tester.
        
        Args:
            device: Device to run tests on (default: "cuda")
            tolerance: Numerical tolerance for comparison (default: 1e-4)
        """
        self.device = device
        self.tolerance = tolerance
        self.test_results: List[Dict] = []
        
    def _prepare_inputs(
        self,
        batch_size: int,
        in_channels: int,
        height: int,
        width: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        has_bias: bool = True,
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Generate random input, weight, and bias tensors."""
        torch.manual_seed(seed)
        
        # Input tensor: N x C_in x H x W
        x = torch.randn(batch_size, in_channels, height, width, device=self.device, dtype=torch.float32)
        
        # Weight tensor: C_out x C_in x kH x kW
        kH, kW = kernel_size
        w = torch.randn(out_channels, in_channels, kH, kW, device=self.device, dtype=torch.float32)
        
        # Bias tensor: C_out
        if has_bias:
            b = torch.randn(out_channels, device=self.device, dtype=torch.float32)
        else:
            b = None
            
        return x, w, b
    
    def _prepare_grouped_inputs(
        self,
        batch_size: int,
        in_channels: int,
        height: int,
        width: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        groups: int,
        has_bias: bool = True,
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Generate random input, weight, and bias tensors for grouped convolution."""
        torch.manual_seed(seed)
        
        # Input tensor: N x C_in x H x W
        x = torch.randn(batch_size, in_channels, height, width, device=self.device, dtype=torch.float32)
        
        # Weight tensor for grouped conv: C_out x (C_in / groups) x kH x kW
        kH, kW = kernel_size
        w = torch.randn(out_channels, in_channels // groups, kH, kW, device=self.device, dtype=torch.float32)
        
        # Bias tensor: C_out
        if has_bias:
            b = torch.randn(out_channels, device=self.device, dtype=torch.float32)
        else:
            b = None
            
        return x, w, b
    
    def run_test(
        self,
        test_name: str,
        batch_size: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        has_bias: bool = True,
        max_engine_variants: int = 1,
        seed: int = 42
    ) -> bool:
        """
        Run a single test case comparing engine_cfg_perf with PyTorch native conv.
        
        Returns:
            True if test passed, False otherwise
        """
        try:
            # Prepare inputs
            if groups > 1:
                x, w, b = self._prepare_grouped_inputs(
                    batch_size, in_channels, height, width, out_channels, 
                    kernel_size, groups, has_bias, seed
                )
            else:
                x, w, b = self._prepare_inputs(
                    batch_size, in_channels, height, width, out_channels,
                    kernel_size, has_bias, seed
                )
            
            # Make inputs contiguous
            x = x.contiguous()
            w = w.contiguous()
            if b is not None:
                b = b.contiguous()
            else:
                # engine_cfg_perf expects a bias tensor (can be zeros)
                b_for_custom = torch.zeros(out_channels, device=self.device, dtype=torch.float32)
            
            # Run PyTorch native convolution
            with torch.no_grad():
                y_pytorch = F.conv2d(
                    x, w, 
                    bias=b if has_bias else None,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups
                )
            
            # Run custom engine_cfg_perf
            with torch.no_grad():
                y_custom, engine_table = torch.ops.LUT_perf.profile_conv2d(
                    x, w, 
                    b if has_bias else b_for_custom,
                    list(stride),
                    list(padding),
                    list(dilation),
                    groups,
                    max_engine_variants,
                    1,  # iters
                    test_name
                )
            
            # Check shapes match
            shape_match = y_pytorch.shape == y_custom.shape
            if not shape_match:
                print(f"  ⚠ Shape mismatch! PyTorch: {tuple(y_pytorch.shape)}, Custom: {tuple(y_custom.shape)}")
            
            # Compare output values
            abs_diff = torch.abs(y_pytorch - y_custom)
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            # Relative error
            rel_diff = abs_diff / (torch.abs(y_pytorch) + 1e-8)
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            # Check if test passed (both shape and values must match)
            passed = shape_match and max_abs_diff < self.tolerance
            
            # Store results
            result = {
                'test_name': test_name,
                'passed': passed,
                'shape_match': shape_match,
                'config': {
                    'batch_size': batch_size,
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'input_size': (height, width),
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                    'dilation': dilation,
                    'groups': groups,
                    'has_bias': has_bias,
                },
                'metrics': {
                    'max_abs_diff': max_abs_diff,
                    'mean_abs_diff': mean_abs_diff,
                    'max_rel_diff': max_rel_diff,
                    'mean_rel_diff': mean_rel_diff,
                },
                'pytorch_shape': tuple(y_pytorch.shape),
                'custom_shape': tuple(y_custom.shape),
                'num_engine_variants': engine_table.shape[0]
            }
            
            self.test_results.append(result)
            
            # Print result
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {test_name}")
            print(f"  Input: {tuple(x.shape)}, Weight: {tuple(w.shape)}")
            print(f"  Output shapes - PyTorch: {tuple(y_pytorch.shape)}, Custom: {tuple(y_custom.shape)}")
            print(f"  Shape match: {'✓' if shape_match else '✗'}")
            
            # Print sample output statistics
            print(f"  PyTorch output - mean: {y_pytorch.mean().item():.6f}, std: {y_pytorch.std().item():.6f}, "
                  f"min: {y_pytorch.min().item():.6f}, max: {y_pytorch.max().item():.6f}")
            print(f"  Custom output  - mean: {y_custom.mean().item():.6f}, std: {y_custom.std().item():.6f}, "
                  f"min: {y_custom.min().item():.6f}, max: {y_custom.max().item():.6f}")
            
            # Print first few values for comparison
            print(f"  PyTorch first 5: [{', '.join(f'{v:.6f}' for v in y_pytorch.flatten()[:5].tolist())}]")
            print(f"  Custom first 5:  [{', '.join(f'{v:.6f}' for v in y_custom.flatten()[:5].tolist())}]")
            
            print(f"  Max abs diff: {max_abs_diff:.2e}, Mean abs diff: {mean_abs_diff:.2e}")
            print(f"  Max rel diff: {max_rel_diff:.2e}, Mean rel diff: {mean_rel_diff:.2e}")
            
            if not passed:
                if not shape_match:
                    print(f"  ⚠ Shape mismatch!")
                if max_abs_diff >= self.tolerance:
                    print(f"  ⚠ Value tolerance exceeded! (threshold: {self.tolerance:.2e})")
            print()
            
            return passed
            
        except Exception as e:
            print(f"✗ ERROR | {test_name}")
            print(f"  Exception: {e}")
            print()
            
            self.test_results.append({
                'test_name': test_name,
                'passed': False,
                'error': str(e)
            })
            
            return False
    
    def print_summary(self):
        """Print summary of all test results."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = total - passed
        
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total tests:  {total}")
        print(f"Passed:       {passed} ({100*passed/total:.1f}%)" if total > 0 else "Passed:       0")
        print(f"Failed:       {failed} ({100*failed/total:.1f}%)" if total > 0 else "Failed:       0")
        print("=" * 70)
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.test_results:
                if not r['passed']:
                    print(f"  - {r['test_name']}")
                    if 'error' in r:
                        print(f"    Error: {r['error']}")
                    elif 'shape_match' in r:
                        if not r['shape_match']:
                            print(f"    Shape mismatch: PyTorch {r['pytorch_shape']} vs Custom {r['custom_shape']}")
                        if 'metrics' in r and r['metrics']['max_abs_diff'] >= self.tolerance:
                            print(f"    Value error: max_abs_diff={r['metrics']['max_abs_diff']:.2e} (threshold={self.tolerance:.2e})")
        
        return passed == total


def run_all_tests():
    """Run comprehensive test suite."""
    print("=" * 70)
    print("ENGINE_CFG_PERF TEST SUITE")
    print("=" * 70)
    print()
    
    tester = ConvTester(device="cuda", tolerance=1e-4)
    
    # Test 1: Basic small convolution
    tester.run_test(
        test_name="basic_3x3_conv",
        batch_size=2,
        in_channels=64,
        out_channels=128,
        height=32,
        width=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        has_bias=True
    )
    
    # Test 2: 1x1 convolution (common in ResNet)
    tester.run_test(
        test_name="1x1_conv",
        batch_size=4,
        in_channels=256,
        out_channels=512,
        height=28,
        width=28,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        has_bias=True
    )
    
    # Test 3: Strided convolution
    tester.run_test(
        test_name="strided_conv",
        batch_size=2,
        in_channels=128,
        out_channels=256,
        height=56,
        width=56,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        has_bias=True
    )
    
    # Test 4: Large kernel (7x7)
    tester.run_test(
        test_name="large_kernel_7x7",
        batch_size=1,
        in_channels=3,
        out_channels=64,
        height=224,
        width=224,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        has_bias=True
    )
    
    # Test 5: No bias
    tester.run_test(
        test_name="no_bias",
        batch_size=2,
        in_channels=64,
        out_channels=64,
        height=32,
        width=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        has_bias=False
    )
    
    # Test 6: Dilated convolution
    tester.run_test(
        test_name="dilated_conv",
        batch_size=2,
        in_channels=128,
        out_channels=128,
        height=64,
        width=64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(2, 2),
        dilation=(2, 2),
        has_bias=True
    )
    
    # Test 7: Grouped convolution (depthwise separable style)
    tester.run_test(
        test_name="grouped_conv",
        batch_size=2,
        in_channels=128,
        out_channels=128,
        height=32,
        width=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        groups=4,
        has_bias=True
    )
    
    # Test 8: Depthwise convolution
    tester.run_test(
        test_name="depthwise_conv",
        batch_size=2,
        in_channels=256,
        out_channels=256,
        height=28,
        width=28,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        groups=256,
        has_bias=True
    )
    
    # Test 9: Small input (edge case)
    tester.run_test(
        test_name="small_input",
        batch_size=1,
        in_channels=32,
        out_channels=64,
        height=7,
        width=7,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        has_bias=True
    )
    
    # Test 10: Asymmetric kernel
    tester.run_test(
        test_name="asymmetric_kernel",
        batch_size=2,
        in_channels=64,
        out_channels=128,
        height=32,
        width=32,
        kernel_size=(3, 5),
        stride=(1, 1),
        padding=(1, 2),
        has_bias=True
    )
    
    # Test 11: Asymmetric stride
    tester.run_test(
        test_name="asymmetric_stride",
        batch_size=2,
        in_channels=128,
        out_channels=256,
        height=64,
        width=64,
        kernel_size=(3, 3),
        stride=(2, 1),
        padding=(1, 1),
        has_bias=True
    )
    
    # Test 12: Large batch
    tester.run_test(
        test_name="large_batch",
        batch_size=16,
        in_channels=64,
        out_channels=64,
        height=28,
        width=28,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        has_bias=True
    )
    
    # Test 13: High resolution input
    tester.run_test(
        test_name="high_resolution",
        batch_size=1,
        in_channels=32,
        out_channels=32,
        height=512,
        width=512,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        has_bias=True
    )
    
    # Test 14: Many channels
    tester.run_test(
        test_name="many_channels",
        batch_size=1,
        in_channels=512,
        out_channels=512,
        height=14,
        width=14,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        has_bias=True
    )
    
    # Test 15: No padding
    tester.run_test(
        test_name="no_padding",
        batch_size=2,
        in_channels=64,
        out_channels=128,
        height=32,
        width=32,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(0, 0),
        has_bias=True
    )
    
    # Print summary
    print()
    all_passed = tester.print_summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

