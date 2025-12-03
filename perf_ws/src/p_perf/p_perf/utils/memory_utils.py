#!/usr/bin/env python3
"""
Simple memory cleanup utilities.
"""

import gc
import torch


def clear_gpu_memory(device: int = None):
    """
    Clear GPU memory cache.
    
    Args:
        device: Specific GPU device to clear. If None, clears all devices.
    """
    if not torch.cuda.is_available():
        return
    
    if device is not None:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    print("✓ GPU memory cleared")


def clear_python_memory():
    """Run Python garbage collection."""
    gc.collect()
    print("✓ Python memory cleaned")


def clear_all_memory(device: int = None):
    """
    Clear both GPU and Python memory.
    
    Args:
        device: Specific GPU device to clear. If None, clears all devices.
    """
    # Python cleanup
    gc.collect()
    
    # GPU cleanup
    if torch.cuda.is_available():
        if device is not None:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    print("✓ Memory cleaned")


def force_memory_cleanup(device: int = None):
    """
    Aggressive memory cleanup with multiple rounds.
    
    Args:
        device: Specific GPU device to clear. If None, clears all devices.
    """
    # 3 rounds of cleanup
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            if device is not None:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            else:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
    print("✓ Aggressive memory cleanup completed")

