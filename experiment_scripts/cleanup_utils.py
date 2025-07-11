#!/usr/bin/env python3
"""
Cleanup utilities for experiment scripts.
Provides comprehensive cleanup functions for memory, cache, and GPU memory.
"""

import os
import time
import subprocess
import psutil
import torch
import gc
from typing import Optional

def clear_gpu():
    """Clear GPU memory and wait for it to be ready"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)  # Give some time for GPU to stabilize

def clear_system_cache():
    """Clear system cache using sync and drop_caches"""
    try:
        # Sync file system
        os.sync()
        
        # Clear page cache, dentries and inodes
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('3')
        
        print("System cache cleared successfully")
    except (PermissionError, FileNotFoundError) as e:
        print(f"Could not clear system cache (requires root): {e}")

def clear_gpu_memory():
    """Clear GPU memory using nvidia-smi if available"""
    try:
        # Reset GPU memory
        subprocess.run(['nvidia-smi', '--gpu-reset'], 
                      capture_output=True, check=True, timeout=30)
        print("GPU memory reset successfully")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Could not reset GPU memory: {e}")
        # Fallback to PyTorch GPU cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU cache cleared using PyTorch")

def kill_ros2_processes():
    """Kill any remaining ROS2 processes"""
    try:
        # Kill ROS2 launch processes
        subprocess.run(['pkill', '-f', 'ros2 launch p_perf'], 
                      capture_output=True, timeout=10)
        
        # Kill other ROS2 processes
        subprocess.run(['pkill', '-f', 'ros2'], 
                      capture_output=True, timeout=10)
        
        print("ROS2 processes killed successfully")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Could not kill all ROS2 processes: {e}")

def kill_python_experiment_processes():
    """Kill Python experiment processes"""
    try:
        # Kill bag_test.py and ms_test.py processes
        subprocess.run(['pkill', '-f', 'bag_test.py'], 
                      capture_output=True, timeout=10)
        subprocess.run(['pkill', '-f', 'ms_test.py'], 
                      capture_output=True, timeout=10)
        
        print("Python experiment processes killed successfully")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Could not kill all Python experiment processes: {e}")

def force_kill_processes():
    """Force kill stubborn processes"""
    try:
        # Force kill ROS2 processes
        subprocess.run(['pkill', '-9', '-f', 'ros2 launch p_perf'], 
                      capture_output=True, timeout=10)
        subprocess.run(['pkill', '-9', '-f', 'ros2'], 
                      capture_output=True, timeout=10)
        
        # Force kill Python experiment processes
        subprocess.run(['pkill', '-9', '-f', 'bag_test.py'], 
                      capture_output=True, timeout=10)
        subprocess.run(['pkill', '-9', '-f', 'ms_test.py'], 
                      capture_output=True, timeout=10)
        
        print("Force killed stubborn processes")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Could not force kill all processes: {e}")

def clear_python_memory():
    """Clear Python memory using garbage collection"""
    # Force garbage collection
    gc.collect()
    
    # Clear any remaining references
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Python memory cleared successfully")

def get_memory_usage():
    """Get current memory usage information"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_cache = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"GPU memory allocated: {gpu_memory:.2f} MB")
        print(f"GPU memory cached: {gpu_cache:.2f} MB")

def comprehensive_cleanup(wait_time: int = 10, verbose: bool = True):
    """
    Perform comprehensive cleanup between experiment runs.
    
    Args:
        wait_time (int): Time to wait after cleanup (seconds)
        verbose (bool): Whether to print verbose output
    """
    if verbose:
        print(f"\n{'='*60}")
        print("STARTING COMPREHENSIVE CLEANUP")
        print(f"{'='*60}")
    
    # Step 1: Kill processes
    if verbose:
        print("\n1. Killing processes...")
    kill_ros2_processes()
    kill_python_experiment_processes()
    
    # Step 2: Wait for processes to terminate
    if verbose:
        print("\n2. Waiting for processes to terminate...")
    time.sleep(5)
    
    # Step 3: Force kill if needed
    if verbose:
        print("\n3. Force killing stubborn processes...")
    force_kill_processes()
    time.sleep(2)
    
    # Step 4: Clear GPU memory
    if verbose:
        print("\n4. Clearing GPU memory...")
    clear_gpu_memory()
    
    # Step 5: Clear Python memory
    if verbose:
        print("\n5. Clearing Python memory...")
    clear_python_memory()
    
    # Step 6: Clear system cache (requires root)
    if verbose:
        print("\n6. Clearing system cache...")
    clear_system_cache()
    
    # Step 7: Wait for system to stabilize
    if verbose:
        print(f"\n7. Waiting {wait_time} seconds for system to stabilize...")
    time.sleep(wait_time)
    
    # Step 8: Report memory usage
    if verbose:
        print("\n8. Current memory usage:")
        get_memory_usage()
        print(f"\n{'='*60}")
        print("CLEANUP COMPLETED")
        print(f"{'='*60}\n")

def quick_cleanup(wait_time: int = 5, verbose: bool = True):
    """
    Perform quick cleanup between experiment runs.
    
    Args:
        wait_time (int): Time to wait after cleanup (seconds)
        verbose (bool): Whether to print verbose output
    """
    if verbose:
        print(f"\n{'='*40}")
        print("QUICK CLEANUP")
        print(f"{'='*40}")
    
    # Kill processes
    kill_ros2_processes()
    kill_python_experiment_processes()
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Clear Python memory
    clear_python_memory()
    
    # Wait
    if verbose:
        print(f"Waiting {wait_time} seconds...")
    time.sleep(wait_time)
    
    if verbose:
        print("Quick cleanup completed\n")

def cleanup_between_runs(run_index: int, total_runs: int, 
                        comprehensive: bool = True, 
                        wait_time: Optional[int] = None,
                        verbose: bool = True):
    """
    Cleanup between experiment runs with automatic wait time calculation.
    
    Args:
        run_index (int): Current run index
        total_runs (int): Total number of runs
        comprehensive (bool): Whether to do comprehensive cleanup
        wait_time (int, optional): Custom wait time, auto-calculated if None
        verbose (bool): Whether to print verbose output
    """
    if run_index >= total_runs - 1:
        # Last run, no cleanup needed
        return
    
    # Auto-calculate wait time based on run progress
    if wait_time is None:
        if comprehensive:
            # Longer wait for comprehensive cleanup
            wait_time = max(10, min(30, total_runs // 10))
        else:
            # Shorter wait for quick cleanup
            wait_time = max(5, min(15, total_runs // 20))
    
    if verbose:
        print(f"\nCleaning up between runs ({run_index + 1}/{total_runs})...")
    
    if comprehensive:
        comprehensive_cleanup(wait_time=wait_time, verbose=verbose)
    else:
        quick_cleanup(wait_time=wait_time, verbose=verbose) 