#!/usr/bin/env python3
"""
Simple script to analyze kernel_timing.csv files and count kernels from e2e layers
within a given time range.
"""

import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

def extract_e2e_timing_ranges(csv_file: str, model_name: str) -> List[Dict]:
    """
    Extract input tokens and their corresponding e2e timing ranges for a given model.
    
    Args:
        csv_file: Path to the kernel timing CSV file
        model_name: Name of the model to filter by
    
    Returns:
        List of dictionaries containing input token and timing range information
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Filter for e2e layers and specified model
        e2e_df = df[(df['Layer'] == 'e2e') & (df['Model'] == model_name)].copy()
        
        if e2e_df.empty:
            return []
        
        # Group by input token to find timing ranges for each input
        timing_ranges = []
        
        for input_token in e2e_df['Input'].unique():
            input_kernels = e2e_df[e2e_df['Input'] == input_token]
            
            if len(input_kernels) > 0:
                # Find the latest end time for this input
                end_timestamp = input_kernels['End Timestamp'].max()
                
                # Get the GPU Turnaround Time for this input (should be the same for all kernels of same input)
                gpu_turnaround_time = input_kernels['GPU Turnaround Time'].iloc[0]  # Use first kernel's value
                
                # Calculate start timestamp as end_timestamp - GPU Turnaround Time
                start_timestamp = end_timestamp - (gpu_turnaround_time * 1e9)  # Convert to nanoseconds
                
                timing_ranges.append({
                    'input_token': input_token,
                    'model': model_name,
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp,
                    'gpu_turnaround_time': gpu_turnaround_time,
                })
        
        # Sort by start timestamp
        timing_ranges.sort(key=lambda x: x['start_timestamp'])
        
        return timing_ranges
        
    except Exception as e:
        print(f"Error extracting timing ranges: {str(e)}")
        return []

def analyze_e2e_kernels_in_timerange(csv_file: str, start_timestamp: int, end_timestamp: int, model_name: str, primary_model: str = None) -> Dict:
    """
    Analyze a kernel timing CSV file and count e2e kernels from the specified model within the specified timestamp range.
    
    Args:
        csv_file: Path to the kernel timing CSV file
        start_timestamp: Start timestamp (absolute timestamp from CSV)
        end_timestamp: End timestamp (absolute timestamp from CSV)
        model_name: Name of the model to filter by
        primary_model: Name of the primary model to get memcpy kernel timestamps for comparison
    
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Filter for e2e layers and specified model
        e2e_df = df[(df['Layer'] == 'e2e') & (df['Model'] == model_name)].copy()
        
        if e2e_df.empty:
            return {
                'model': model_name,
                'kernels_in_range': 0,
                'sum_e2e_time_in_range': 0.0,
                'memcpy_kernels_in_range': 0,
                'unique_kernel_types': 0,
                'kernel_types': [],
                'error': f'No e2e layers found for model: {model_name}'
            }
        
        # Filter kernels that overlap with the specified timestamp range
        # A kernel overlaps if:
        # - kernel starts before range ends AND kernel ends after range starts
        kernels_in_range = e2e_df[
            (e2e_df['Start Timestamp'] < end_timestamp) & 
            (e2e_df['End Timestamp'] > start_timestamp)
        ]
        
        kernels_in_range_count = len(kernels_in_range)
        
        # Calculate total sum of execution times for e2e kernels in the specified range
        sum_e2e_time_in_range = kernels_in_range['Elapsed Time'].sum()
        
        # Count memcpy kernels (host2device and device2host) in the range
        memcpy_kernels_in_range = kernels_in_range[
            kernels_in_range['Kernel Name'].isin(['host2device', 'device2host'])
        ]
        
        # If primary_model is provided, filter memcpy kernels based on proximity to primary model memcpy kernels
        if primary_model is not None:
            # Get primary model memcpy kernels in the same time range
            primary_e2e_df = df[(df['Layer'] == 'e2e') & (df['Model'] == primary_model)].copy()
            primary_memcpy_kernels = primary_e2e_df[
                (primary_e2e_df['Kernel Name'].isin(['host2device', 'device2host'])) &
                (primary_e2e_df['Start Timestamp'] < end_timestamp) & 
                (primary_e2e_df['End Timestamp'] > start_timestamp)
            ]
            
            # Calculate timestamps for primary model memcpy kernels (start+end)/2
            primary_memcpy_timestamps = []
            for _, kernel in primary_memcpy_kernels.iterrows():
                timestamp = (kernel['Start Timestamp'] + kernel['End Timestamp']) / 2
                primary_memcpy_timestamps.append(timestamp)
            
            # Filter contention model memcpy kernels based on proximity to primary model memcpy kernels
            filtered_memcpy_kernels = []
            for _, kernel in memcpy_kernels_in_range.iterrows():
                kernel_timestamp = (kernel['Start Timestamp'] + kernel['End Timestamp']) / 2
                
                # Check if this kernel is within ±0.004 seconds of any primary model memcpy kernel
                is_near_primary = False
                for primary_timestamp in primary_memcpy_timestamps:
                    if abs(kernel_timestamp - primary_timestamp) <= 0.005 * 1e9:  # Convert 0.004s to nanoseconds
                        is_near_primary = True
                        break
                
                if is_near_primary:
                    filtered_memcpy_kernels.append(kernel)
            
            memcpy_count = len(filtered_memcpy_kernels)
        else:
            # Original logic: count all memcpy kernels in range
            memcpy_count = len(memcpy_kernels_in_range)
        
        # Get unique kernel names in the time range
        unique_kernels = kernels_in_range['Kernel Name'].unique().tolist()
        
        return {
            'model': model_name,
            'kernels_in_range': kernels_in_range_count,
            'sum_e2e_time_in_range': sum_e2e_time_in_range,
            'memcpy_kernels_in_range': memcpy_count,
            'unique_kernel_types': len(unique_kernels),
            'kernel_types': unique_kernels,
            'error': None
        }
        
    except Exception as e:
        return {
            'model': model_name,
            'kernels_in_range': 0,
            'sum_e2e_time_in_range': 0.0,
            'memcpy_kernels_in_range': 0,
            'unique_kernel_types': 0,
            'kernel_types': [],
            'error': f'Error processing file: {str(e)}'
        }

def analyze_model_kernel_stats(csv_file: str, model_name: str) -> Dict:
    """
    Analyze kernel statistics for a specified model.
    
    Args:
        csv_file: Path to the kernel timing CSV file
        model_name: Name of the model to analyze
    
    Returns:
        Dictionary containing kernel statistics for the model
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        # Filter for the specified model
        model_df = df[df['Model'] == model_name].copy()
        
        if model_df.empty:
            return {
                'model': model_name,
                'total_kernels': 0,
                'avg_kernel_time': 0.0,
                'total_execution_time': 0.0,
                'unique_kernel_types': 0,
                'kernel_types': [],
                'error': f'No kernels found for model: {model_name}'
            }
        
        # Calculate statistics
        total_kernels = len(model_df)
        total_execution_time = model_df['Elapsed Time'].sum()
        avg_kernel_time = total_execution_time / total_kernels if total_kernels > 0 else 0.0
        unique_kernels = model_df['Kernel Name'].unique().tolist()
        
        return {
            'model': model_name,
            'total_kernels': total_kernels,
            'avg_kernel_time': avg_kernel_time,
            'total_execution_time': total_execution_time,
            'unique_kernel_types': len(unique_kernels),
            'kernel_types': unique_kernels,
            'error': None
        }
        
    except Exception as e:
        return {
            'model': model_name,
            'total_kernels': 0,
            'avg_kernel_time': 0.0,
            'total_execution_time': 0.0,
            'unique_kernel_types': 0,
            'kernel_types': [],
            'error': f'Error processing file: {str(e)}'
        }

def create_plots(results_df: pd.DataFrame, output_dir: str = "analysis_outputs", mode='MP'):
    """
    Create line plots showing relationships between primary model e2e time and contention model metrics.
    
    Args:
        results_df: DataFrame containing the analysis results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out rows with errors
    valid_results = results_df[results_df['error'].isna()].copy()
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    # Convert e2e time to seconds for better readability
    valid_results['primary_e2e_time_sec'] = valid_results['primary_e2e_time'] / 1e9
    valid_results['contention_sum_e2e_time_sec'] = valid_results['sum_e2e_time_in_range'] / 1e3
    
    # Calculate ratio of memcpy kernels to contention sum e2e time
    valid_results['memcpy_to_contention_ratio'] = valid_results['memcpy_kernels_in_range'] / valid_results['contention_sum_e2e_time_sec']
    
    # Remove max and min primary e2e time points
    min_idx = valid_results['primary_e2e_time_sec'].idxmin()
    max_idx = valid_results['primary_e2e_time_sec'].idxmax()
    valid_results = valid_results.drop([min_idx, max_idx])
    
    # Select only 60% of the points (middle 60% by removing outliers)
    n_points = len(valid_results)
    n_to_keep = int(0.5 * n_points)
    n_to_remove_each_end = (n_points - n_to_keep) // 2
    
    # Sort by primary e2e time and remove outliers from both ends
    valid_results_sorted = valid_results.sort_values('primary_e2e_time_sec')
    valid_results = valid_results_sorted.iloc[n_to_remove_each_end:n_to_remove_each_end + n_to_keep]
    
    # Sort by the ratio for better line plot visualization
    valid_results = valid_results.sort_values('memcpy_to_contention_ratio')
    
    # Plot 1: Primary model e2e time vs Memcpy/Contention ratio
    plt.figure(figsize=(15, 6))
    plt.plot(valid_results['memcpy_to_contention_ratio'], valid_results['primary_e2e_time_sec'], 'o-', linewidth=2, markersize=3)
    plt.xlabel('Memcpy Ratio', fontsize=27)
    plt.ylabel('Primary Model E2E (s)', fontsize=27)
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=27)
    # Format y-axis to show clean values
    y_min, y_max = plt.ylim()
    y_ticks = np.arange(np.floor(y_min * 100) / 100, np.ceil(y_max * 100) / 100 + 0.01, 0.01)
    plt.yticks(y_ticks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'primary_vs_memcpy_contention_ratio_{mode}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    valid_results = valid_results.sort_values('contention_sum_e2e_time_sec')
    # Plot 2: Primary model e2e time vs Contention sum e2e time (line plot)
    plt.figure(figsize=(15, 6))
    plt.plot(valid_results['contention_sum_e2e_time_sec'], valid_results['primary_e2e_time_sec'], 'o-', linewidth=2, markersize=3)
    plt.xlabel('Contention Model Sum E2E Time (seconds)', fontsize=27)
    plt.ylabel('Primary Model E2E (s)', fontsize=27)
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=27)
    # Format y-axis to show clean values
    y_min, y_max = plt.ylim()
    y_ticks = np.arange(np.floor(y_min * 100) / 100, np.ceil(y_max * 100) / 100 + 0.01, 0.01)
    plt.yticks(y_ticks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'primary_vs_contention_e2e_time_line_{mode}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

def main():
    """
    Main function to demonstrate the analysis.
    """
    # Example parameters
    primary_model = "faster-rcnn_r50_fpn_1x_coco"
    contention_model = "pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d"
    mode = "MP"
    
    # Specify the single kernel timing file to analyze
    kernel_file = "outputs/Image_Lidar_full/kernel_timings_0.csv"  # Change this to your specific file
    layer_file = "outputs/Image_Lidar_full/layer_timings_0.csv"

    
    # Check if file exists
    if not os.path.exists(kernel_file):
        print(f"Error: File '{kernel_file}' not found!")
        return
    
    # Save kernel statistics for both models to CSV files
    output_dir = "tools/kernel_stats"
    os.makedirs(output_dir, exist_ok=True)
    save_model_kernel_stats_to_csv(kernel_file, primary_model, f"{output_dir}/{primary_model}_kernel_stats.csv")
    save_model_kernel_stats_to_csv(kernel_file, contention_model, f"{output_dir}/{contention_model}_kernel_stats.csv")
    
    # Extract e2e timing ranges for all inputs
    print("\nEXTRACTING E2E TIMING RANGES")
    print("=" * 80)
    primary_timing_ranges = extract_e2e_timing_ranges(layer_file, primary_model)
    
    if primary_timing_ranges:
        pass
    else:
        print(f"No e2e timing ranges found for model: {primary_model}")
        return
    
    # Analyze the single file for the specified time range
    print("ANALYZING SPECIFIED TIME RANGE")
    print("=" * 80)
    
    # Collect results for all timing ranges
    results = []
    
    for timing_range in tqdm(primary_timing_ranges, desc="Analyzing timing ranges"):
        start_timestamp = timing_range['start_timestamp']
        end_timestamp = timing_range['end_timestamp']
        result = analyze_e2e_kernels_in_timerange(kernel_file, start_timestamp, end_timestamp, contention_model, primary_model)
        
        # Add primary model information to the result
        result['input_token'] = timing_range['input_token']
        result['primary_e2e_time'] = end_timestamp - start_timestamp
        result['primary_start_timestamp'] = start_timestamp
        result['primary_end_timestamp'] = end_timestamp
        
        results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    output_file = "e2e_analysis_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Create plots
    create_plots(results_df, mode=mode)
    
    print()
    print("SUMMARY")
    print("=" * 80)
    print(f"Primary model: {primary_model}")
    print(f"Contention model: {contention_model}")
    print(f"File processed: {kernel_file}")
    print(f"Total timing ranges analyzed: {len(results)}")
    print(f"Results saved to: {output_file}")

def get_model_kernel_stats(csv_file: str, model_name: str) -> None:
    """
    Standalone function to get and print kernel statistics for a specified model.
    
    Args:
        csv_file: Path to the kernel timing CSV file
        model_name: Name of the model to analyze
    """
    print(f"\nAnalyzing kernel statistics for model: {model_name}")
    print(f"File: {csv_file}")
    print("-" * 60)
    
    stats = analyze_model_kernel_stats(csv_file, model_name)
    
    if stats['error']:
        print(f"Error: {stats['error']}")
        return
    
    print(f"Total kernels: {stats['total_kernels']}")
    print(f"Average kernel time: {stats['avg_kernel_time']:.2f} ms")
    print(f"Total execution time: {stats['total_execution_time']:.2f} ms")
    print(f"Unique kernel types: {stats['unique_kernel_types']}")
    
    if stats['kernel_types']:
        print(f"Kernel types: {', '.join(stats['kernel_types'][:10])}")  # Show first 10 types
        if len(stats['kernel_types']) > 10:
            print(f"... and {len(stats['kernel_types']) - 10} more types")

def save_model_kernel_stats_to_csv(csv_file: str, model_name: str, output_csv: str) -> None:
    """
    Save kernel statistics for a specified model to a CSV file.
    Args:
        csv_file: Path to the kernel timing CSV file
        model_name: Name of the model to analyze
        output_csv: Path to the output CSV file
    """
    stats = analyze_model_kernel_stats(csv_file, model_name)
    # Prepare a dict for DataFrame (flatten kernel_types to a string)
    stats_to_save = stats.copy()
    stats_to_save['kernel_types'] = ','.join(stats_to_save['kernel_types']) if stats_to_save['kernel_types'] else ''
    # Save as single-row DataFrame
    pd.DataFrame([stats_to_save]).to_csv(output_csv, index=False)

if __name__ == "__main__":
    main() 