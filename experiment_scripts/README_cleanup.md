# Cleanup System for Experiment Scripts

This document explains the cleanup system implemented for `ms_test.py` and `bag_test.py` to ensure proper memory, cache, and GPU memory cleanup between experiment runs.

## Overview

The cleanup system provides comprehensive memory and process management between experiment runs to prevent:
- Memory leaks
- GPU memory accumulation
- Orphaned processes
- System cache buildup
- Performance degradation over multiple runs

## Files

### `cleanup_utils.py`
Main cleanup utilities module containing all cleanup functions.

### `ms_test.py` and `bag_test.py`
Updated experiment scripts that use the cleanup utilities.

### `test_cleanup.py`
Test script to verify cleanup utilities work correctly.

## Cleanup Functions

### Individual Functions

1. **`clear_gpu()`**
   - Clears PyTorch GPU cache
   - Synchronizes GPU operations
   - Waits 2 seconds for stabilization

2. **`clear_gpu_memory()`**
   - Uses `nvidia-smi --gpu-reset` if available
   - Falls back to PyTorch cache clearing
   - More aggressive GPU memory clearing

3. **`clear_python_memory()`**
   - Forces garbage collection
   - Clears PyTorch GPU cache
   - Removes Python object references

4. **`clear_system_cache()`**
   - Syncs file system
   - Clears page cache, dentries, and inodes
   - Requires root privileges

5. **`kill_ros2_processes()`**
   - Kills ROS2 launch processes
   - Kills other ROS2 processes
   - Uses `pkill` with timeout

6. **`kill_python_experiment_processes()`**
   - Kills `bag_test.py` and `ms_test.py` processes
   - Uses `pkill` with timeout

7. **`force_kill_processes()`**
   - Force kills stubborn processes with `-9` signal
   - Used as fallback when normal kill fails

8. **`get_memory_usage()`**
   - Reports current memory usage
   - Shows RAM and GPU memory usage

### High-Level Functions

1. **`comprehensive_cleanup(wait_time=10, verbose=True)`**
   - Performs all cleanup steps in sequence
   - Includes process killing, memory clearing, and system cache clearing
   - Waits specified time for system stabilization
   - Reports memory usage after cleanup

2. **`quick_cleanup(wait_time=5, verbose=True)`**
   - Faster version of comprehensive cleanup
   - Skips system cache clearing (requires root)
   - Shorter wait times

3. **`cleanup_between_runs(run_index, total_runs, comprehensive=True, wait_time=None, verbose=True)`**
   - Smart cleanup function for experiment loops
   - Auto-calculates wait time based on total runs
   - Skips cleanup after the last run
   - Chooses between comprehensive and quick cleanup

## Usage in Experiment Scripts

### Automatic Integration

Both `ms_test.py` and `bag_test.py` now automatically call cleanup between runs:

```python
# Cleanup between runs
cleanup_between_runs(
    run_index=i, 
    total_runs=len(df), 
    comprehensive=True,  # Use comprehensive cleanup
    wait_time=None,  # Auto-calculate wait time
    verbose=True
)
```

### Wait Time Calculation

The system automatically calculates appropriate wait times:
- **Comprehensive cleanup**: 10-30 seconds (based on total runs)
- **Quick cleanup**: 5-15 seconds (based on total runs)

### Verbose Output

The cleanup process provides detailed output showing:
- Each cleanup step
- Success/failure of operations
- Memory usage before and after
- Process killing results

## Configuration Options

### Comprehensive vs Quick Cleanup

- **Comprehensive**: Full cleanup including system cache (requires root)
- **Quick**: Faster cleanup without system cache clearing

### Wait Time

- **Auto-calculated**: Based on total number of runs
- **Custom**: Specify exact wait time in seconds

### Verbose Mode

- **True**: Detailed output for debugging
- **False**: Minimal output for production

## Testing

Run the test script to verify cleanup utilities work:

```bash
cd experiment_scripts
python3 test_cleanup.py
```

## Requirements

### System Requirements
- Linux system with `/proc/sys/vm/drop_caches` (for system cache clearing)
- `nvidia-smi` (for GPU memory reset)
- `pkill` command (for process killing)

### Python Requirements
- `torch` (for GPU operations)
- `psutil` (for memory monitoring)
- `subprocess` (for system commands)

## Troubleshooting

### Permission Errors
- System cache clearing requires root privileges
- Some operations may fail without sudo

### Process Killing Failures
- Some processes may be protected
- Force kill is used as fallback
- Check process list manually if needed

### GPU Memory Issues
- `nvidia-smi --gpu-reset` may fail if GPU is in use
- PyTorch cache clearing is used as fallback
- Check GPU status with `nvidia-smi`

## Performance Impact

### Benefits
- Prevents memory leaks across runs
- Ensures consistent performance
- Reduces system resource accumulation
- Improves experiment reliability

### Overhead
- Adds 10-30 seconds between runs
- Minimal impact on overall experiment time
- Worthwhile trade-off for stability

## Customization

You can customize the cleanup behavior by:

1. **Modifying wait times**:
   ```python
   cleanup_between_runs(wait_time=15)  # Custom wait time
   ```

2. **Choosing cleanup type**:
   ```python
   cleanup_between_runs(comprehensive=False)  # Quick cleanup
   ```

3. **Adjusting verbosity**:
   ```python
   cleanup_between_runs(verbose=False)  # Silent mode
   ```

4. **Adding custom cleanup steps**:
   ```python
   # Add custom cleanup before/after
   custom_cleanup_function()
   cleanup_between_runs(...)
   custom_cleanup_function()
   ``` 