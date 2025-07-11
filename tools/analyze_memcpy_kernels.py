#!/usr/bin/env python3
"""
Script to analyze memcpy kernels from image and lidar models.
Extracts statistics about host2device and device2host operations.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_kernel_timing_files(base_dir: str) -> Dict[str, List[str]]:
    """
    Find all kernel timing CSV files organized by model type.
    
    Args:
        base_dir: Base directory containing model outputs
        
    Returns:
        Dictionary mapping model types to lists of file paths
    """
    models = {}
    
    # Look for different model directories
    model_dirs = {
        'image': ['image_base'],
        'lidar': ['lidar_base']
    }
    
    for model_type, dir_names in model_dirs.items():
        model_files = []
        for dir_name in dir_names:
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.exists(dir_path):
                # Find all kernel_timings_*.csv files
                pattern = os.path.join(dir_path, 'kernel_timings_*.csv')
                files = glob.glob(pattern)
                model_files.extend(files)
        
        if model_files:
            models[model_type] = sorted(model_files)
            logger.info(f"Found {len(model_files)} files for {model_type} model")
    
    return models

def analyze_memcpy_kernels(file_path: str) -> Dict:
    """
    Analyze memcpy kernels in a single kernel timing file.
    
    Args:
        file_path: Path to kernel timing CSV file
        
    Returns:
        Dictionary containing memcpy statistics
    """
    logger.info(f"Analyzing {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get model name from the first row
        model_name = df['Model'].iloc[0] if not df.empty else 'unknown'
        
        # Get unique inputs
        unique_inputs = df['Input'].unique()
        if len(unique_inputs) == 0:
            logger.warning(f"No unique inputs found in {file_path}")
            return {
                'model_name': model_name,
                'total_host2device': 0,
                'total_device2host': 0,
                'mean_interval_host2device': 0,
                'mean_interval_device2host': 0,
                'mean_interval_all_memcpy': 0,
                'total_memcpy_operations': 0
            }
        
        # Randomly select a unique input
        selected_input = random.choice(unique_inputs)
        logger.info(f"Selected input: {selected_input}")
        
        # Create sub-df that only includes rows whose input data is this token, and its kernel name is either device2host or host2device
        sub_df = df[(df['Input'] == selected_input) & 
                    (df['Kernel Name'].isin(['host2device', 'device2host'])) & 
                    (df['Layer'] == 'e2e')]
        
        if sub_df.empty:
            logger.warning(f"No memcpy kernels found for input {selected_input} in {file_path}")
            return {
                'model_name': model_name,
                'total_host2device': 0,
                'total_device2host': 0,
                'mean_interval_host2device': 0,
                'mean_interval_device2host': 0,
                'mean_interval_all_memcpy': 0,
                'total_memcpy_operations': 0
            }
        
        # Sort them based on the start timestamp
        sub_df = sub_df.sort_values('Start Timestamp')
        
        print(sub_df)
        # Calculate counts
        host2device_count = len(sub_df[sub_df['Kernel Name'] == 'host2device'])
        device2host_count = len(sub_df[sub_df['Kernel Name'] == 'device2host'])
        total_memcpy_operations = len(sub_df)
        
        # Calculate intervals from this sorted data
        intervals = []
        if len(sub_df) > 1:
            for i in range(len(sub_df) - 1):
                current_start = sub_df.iloc[i]['Start Timestamp']
                next_start = sub_df.iloc[i + 1]['Start Timestamp']
                interval = (next_start - current_start) / 1000000.0  # Convert to microseconds
                intervals.append(interval)
                print(f"Interval {i}: {interval} microseconds")
        
        mean_interval_all_memcpy = np.mean(intervals) if intervals else 0
        
        # Calculate 25th to 75th percentile mean for all memcpy operations
        if len(intervals) >= 4:  # Need at least 4 intervals for meaningful percentiles
            sorted_intervals = sorted(intervals)
            q25_idx = int(0.25 * len(sorted_intervals))
            q75_idx = int(0.75 * len(sorted_intervals))
            middle_50_intervals = sorted_intervals[q25_idx:q75_idx]
            mean_interval_all_memcpy = np.mean(middle_50_intervals) if middle_50_intervals else 0
        
        # Calculate intervals by operation type
        host2device_intervals = []
        device2host_intervals = []
        
        for op_type in ['host2device', 'device2host']:
            op_kernels = sub_df[sub_df['Kernel Name'] == op_type]
            if len(op_kernels) > 1:
                op_kernels = op_kernels.sort_values('Start Timestamp')
                for i in range(len(op_kernels) - 1):
                    current_start = op_kernels.iloc[i]['Start Timestamp']
                    next_start = op_kernels.iloc[i + 1]['Start Timestamp']
                    interval = (next_start - current_start) / 1000000.0  # Convert to microseconds
                    if op_type == 'host2device':
                        host2device_intervals.append(interval)
                    else:
                        device2host_intervals.append(interval)
        
        # Calculate 25th to 75th percentile mean for each operation type
        mean_interval_host2device = 0
        mean_interval_device2host = 0
        
        if len(host2device_intervals) >= 4:
            sorted_host2device = sorted(host2device_intervals)
            q25_idx = int(0.25 * len(sorted_host2device))
            q75_idx = int(0.75 * len(sorted_host2device))
            middle_50_host2device = sorted_host2device[q25_idx:q75_idx]
            mean_interval_host2device = np.mean(middle_50_host2device) if middle_50_host2device else 0
        elif host2device_intervals:
            mean_interval_host2device = np.mean(host2device_intervals)
        
        if len(device2host_intervals) >= 4:
            sorted_device2host = sorted(device2host_intervals)
            q25_idx = int(0.25 * len(sorted_device2host))
            q75_idx = int(0.75 * len(sorted_device2host))
            middle_50_device2host = sorted_device2host[q25_idx:q75_idx]
            mean_interval_device2host = np.mean(middle_50_device2host) if middle_50_device2host else 0
        elif device2host_intervals:
            mean_interval_device2host = np.mean(device2host_intervals)
        
        return {
            'model_name': model_name,
            'total_host2device': host2device_count,
            'total_device2host': device2host_count,
            'mean_interval_host2device': mean_interval_host2device,
            'mean_interval_device2host': mean_interval_device2host,
            'mean_interval_all_memcpy': mean_interval_all_memcpy,
            'total_memcpy_operations': total_memcpy_operations
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {
            'model_name': 'unknown',
            'total_host2device': 0,
            'total_device2host': 0,
            'mean_interval_host2device': 0,
            'mean_interval_device2host': 0,
            'mean_interval_all_memcpy': 0,
            'total_memcpy_operations': 0,
            'error': str(e)
        }

def analyze_model_type(model_files: List[str], model_type: str) -> pd.DataFrame:
    """
    Analyze all files for a specific model type.
    
    Args:
        model_files: List of kernel timing file paths
        model_type: Type of model (image, lidar)
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    for file_path in model_files:
        result = analyze_memcpy_kernels(file_path)
        result['model_type'] = model_type
        results.append(result)
    
    return pd.DataFrame(results)

def main():
    # Configuration
    base_dir = 'outputs'
    output_file = 'memcpy_analysis.csv'
    
    # Find kernel timing files
    logger.info(f"Searching for kernel timing files in {base_dir}")
    models = find_kernel_timing_files(base_dir)
    
    if not models:
        logger.error("No kernel timing files found!")
        return
    
    # Analyze each model type
    all_results = []
    
    for model_type, files in models.items():
        logger.info(f"Analyzing {model_type} model with {len(files)} files")
        df = analyze_model_type(files, model_type)
        all_results.append(df)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save results
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print("\n=== MEMCPY KERNEL ANALYSIS SUMMARY ===")
        for model_type in combined_df['model_type'].unique():
            model_data = combined_df[combined_df['model_type'] == model_type]
            print(f"\n{model_type} model:")
            print(f"  Files analyzed: {len(model_data)}")
            print(f"  Average host2device operations per frame: {model_data['total_host2device'].mean():.1f}")
            print(f"  Average device2host operations per frame: {model_data['total_device2host'].mean():.1f}")
            print(f"  Average mean interval between host2device: {model_data['mean_interval_host2device'].mean():.2f} microseconds")
            print(f"  Average mean interval between device2host: {model_data['mean_interval_device2host'].mean():.2f} microseconds")
            print(f"  Average mean interval between all memcpy operations: {model_data['mean_interval_all_memcpy'].mean():.2f} microseconds")
    
    else:
        logger.error("No results to save!")

if __name__ == "__main__":
    main() 