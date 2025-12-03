#!/usr/bin/env python3
"""
Script to analyze the density of LIDAR_TOP data under a scene in nuScenes dataset.
This script provides density statistics about point cloud data including:
- Point density statistics
- Distance and intensity analysis
- Global and per-frame density characteristics

CONFIGURATION:
Edit the parameters below to customize the analysis:
"""

import os
import sys

# Set environment variables to prevent Qt display issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':99'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from nuscenes import NuScenes
from scipy.spatial import cKDTree

sys.path.append('/mmdetection3d_ros2/perf_ws/src/p_perf')
from p_perf.config.constant import (
    image_models,
    lidar_models,
    seg_models,
    scenes,
    get_abbreviated_name,
    get_abbreviated_scene,
    get_full_model_name,
    get_full_scene_token
)

def load_sweep_sd(nusc, scene, sensor_channel='CAM_FRONT'):
    """
    Get all sample_data tokens (including intermediate sweep frames) for a given channel from a scene.

    Args:
        nusc: NuScenes instance
        scene: scene dictionary from nusc.scene
        sensor_channel: e.g., 'CAM_FRONT'

    Returns:
        List of sample_data tokens (str) ordered by time, including all sweeps.
    """
    print(f"DEBUG: Loading {sensor_channel} data for scene: {scene['name']} (token: {scene['token']})")
    
    # Get the first sample token
    first_sample_token = scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    
    print(f"DEBUG: First sample token: {first_sample_token}")
    print(f"DEBUG: Available channels in first sample: {list(first_sample['data'].keys())}")
    
    # Get the initial sample_data token for the desired sensor channel
    current_sd_token = None
    for sd_token in first_sample['data'].values():
        sd = nusc.get('sample_data', sd_token)
        print(f"DEBUG: Checking sample_data token {sd_token} -> channel: {sd['channel']}")
        if sd['channel'] == sensor_channel:
            current_sd_token = sd_token
            print(f"DEBUG: Found {sensor_channel} token: {sd_token}")
            break

    if current_sd_token is None:
        raise ValueError(f"No sample_data found for channel {sensor_channel} in first sample")

    # Traverse through sample_data's 'next' field to get all sweep frames
    sweep_tokens = []
    while current_sd_token:
        sd = nusc.get('sample_data', current_sd_token)
        if sd['channel'] == sensor_channel:
            sweep_tokens.append(current_sd_token)
        current_sd_token = sd['next'] if sd['next'] else None

    return sweep_tokens

# =============================================================================
# CONFIGURATION - Edit these parameters as needed
# =============================================================================

# Point cloud range filter [x_min, y_min, z_min, x_max, y_max, z_max]
POINT_CLOUD_RANGE = [-50, -50, -5, 50, 50, 3]

# Intensity filter - points with intensity below this threshold will be filtered out
MIN_INTENSITY_THRESHOLD = 0.0

# Local neighborhood density - radius for counting neighbors around each point
LOCAL_DENSITY_RADIUS = 0.5  # meters

# Frame limit - maximum number of frames to process per scene (set to None for all frames)
MAX_FRAMES_PER_SCENE = 30

# Output settings
SAVE_RESULTS = False  # Set to True to save detailed results to JSON

# =============================================================================


class LidarRangeAnalyzer:
    """Analyzer for LIDAR_TOP range-based point counts in nuScenes scenes."""
    
    def __init__(self, nusc, scene_token: str):
        """
        Initialize the analyzer.
        
        Args:
            nusc: NuScenes instance
            scene_token: Token of the scene to analyze
        """
        self.nusc = nusc
        self.scene_token = scene_token
        self.scene = nusc.get('scene', scene_token)
        all_lidar_tokens = load_sweep_sd(nusc, self.scene, sensor_channel='LIDAR_TOP')
        
        # Limit the number of frames if specified
        if MAX_FRAMES_PER_SCENE is not None:
            self.lidar_tokens = all_lidar_tokens[:MAX_FRAMES_PER_SCENE]
            print(f"Processing first {len(self.lidar_tokens)} frames (limited from {len(all_lidar_tokens)} total)")
        else:
            self.lidar_tokens = all_lidar_tokens
            print(f"Processing all {len(self.lidar_tokens)} frames")
            
        self.analysis_results = {}
        
    def analyze_point_density(self) -> Dict:
        """
        Analyze point counts within specific distance ranges.
        Applies spatial range filtering and intensity filtering before analysis.
        
        Returns:
            Dictionary containing range-based point count analysis
        """
        print("Analyzing point counts within distance ranges...")
        print(f"Applying filters - Spatial: {POINT_CLOUD_RANGE}, Intensity: >= {MIN_INTENSITY_THRESHOLD}")
        
        # Define distance ranges to analyze - these should match the annular ranges
        distance_ranges = [0.5, 1, 2.5, 5, 10, 25, 50]
        
        frame_stats = []
        all_distances = []
        all_intensities = []
        range_counts_all_frames = {f"range_{r}m": [] for r in distance_ranges}
        
        total_original_points = 0
        total_after_spatial_filter = 0
        total_after_intensity_filter = 0
        
        for token in self.lidar_tokens:
            sd = self.nusc.get('sample_data', token)
            file_path = os.path.join(self.nusc.dataroot, sd['filename'])
            
            if not os.path.exists(file_path):
                continue
                
            try:
                # Load point cloud data
                points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
                original_count = len(points)
                total_original_points += original_count
                
                # Apply point cloud range filter
                mask = (points[:, 0] >= POINT_CLOUD_RANGE[0]) & (points[:, 0] <= POINT_CLOUD_RANGE[3]) & \
                       (points[:, 1] >= POINT_CLOUD_RANGE[1]) & (points[:, 1] <= POINT_CLOUD_RANGE[4]) & \
                       (points[:, 2] >= POINT_CLOUD_RANGE[2]) & (points[:, 2] <= POINT_CLOUD_RANGE[5])
                points_spatial_filtered = points[mask]
                spatial_count = len(points_spatial_filtered)
                total_after_spatial_filter += spatial_count
                
                # Apply intensity filter
                intensity_mask = points_spatial_filtered[:, 3] >= MIN_INTENSITY_THRESHOLD
                points_filtered = points_spatial_filtered[intensity_mask]
                final_count = len(points_filtered)
                total_after_intensity_filter += final_count
                
                # Check if any points remain after filtering
                if final_count == 0:
                    print(f"Warning: No points remain after spatial and intensity filtering for token {token}")
                    continue
                
                # Calculate distances from origin
                distances = np.sqrt(points_filtered[:, 0]**2 + points_filtered[:, 1]**2 + points_filtered[:, 2]**2)
                
                # Count points within each distance range
                range_counts = {}
                range_percentages = {}
                total_points = len(points_filtered)
                
                for range_limit in distance_ranges:
                    count = np.sum(distances <= range_limit)
                    percentage = (count / total_points) * 100 if total_points > 0 else 0
                    range_counts[f"range_{range_limit}m"] = count
                    range_percentages[f"range_{range_limit}m_pct"] = percentage
                    range_counts_all_frames[f"range_{range_limit}m"].append(count)
                
                # Intensity statistics
                intensities = points_filtered[:, 3]
                
                # Calculate distance-stratified local neighborhood density
                print(f"  Calculating distance-stratified local density for {final_count} points...")
                distance_density_stats = self.calculate_distance_stratified_density(points_filtered, distance_ranges)
                
                # Calculate global density for all points in this frame
                print(f"  Calculating global density for all {final_count} points...")
                global_density_stats = self.calculate_local_density(points_filtered, LOCAL_DENSITY_RADIUS)
                
                # Flatten distance density stats for DataFrame storage
                flattened_density_stats = {}
                for range_name, density_data in distance_density_stats.items():
                    for key, value in density_data.items():
                        if key != 'neighbor_counts':  # Skip the full array
                            flattened_density_stats[f"{range_name}_{key}"] = value
                
                # Add global density stats
                for key, value in global_density_stats.items():
                    if key != 'neighbor_counts':  # Skip the full array
                        flattened_density_stats[f"global_{key}"] = value
                
                # Store all distances and intensities for global analysis
                all_distances.extend(distances.tolist())
                all_intensities.extend(intensities.tolist())
                
                frame_stats.append({
                    'num_points': len(points_filtered),
                    'mean_distance': np.mean(distances),
                    'max_distance': np.max(distances),
                    'min_distance': np.min(distances),
                    'std_distance': np.std(distances),
                    'mean_intensity': np.mean(intensities),
                    'std_intensity': np.std(intensities),
                    'min_intensity': np.min(intensities),
                    'max_intensity': np.max(intensities),
                    'median_intensity': np.median(intensities),
                    **range_counts,  # Add range counts to frame stats
                    **range_percentages,  # Add range percentages to frame stats
                    **flattened_density_stats # Add distance density stats to frame stats
                })
                
            except Exception as e:
                print(f"Error analyzing points for token {token}: {e}")
        
        if frame_stats:
            # Convert to DataFrame for easier analysis
            frame_df = pd.DataFrame(frame_stats)
            
            # Convert to numpy arrays for analysis
            all_distances = np.array(all_distances)
            all_intensities = np.array(all_intensities)
            
            # Calculate percentiles for distances and intensities
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            distance_percentiles = {}
            intensity_percentiles = {}
            
            for p in percentiles:
                distance_percentiles[f'p{p}'] = np.percentile(all_distances, p)
                intensity_percentiles[f'p{p}'] = np.percentile(all_intensities, p)
            
            # Calculate statistics for each distance range
            range_statistics = {}
            for range_limit in distance_ranges:
                range_col = f"range_{range_limit}m"
                range_pct_col = f"range_{range_limit}m_pct"
                range_statistics[range_col] = {
                    'mean': frame_df[range_col].mean(),
                    'std': frame_df[range_col].std(),
                    'min': frame_df[range_col].min(),
                    'max': frame_df[range_col].max(),
                    'median': frame_df[range_col].median(),
                    'percentiles': {
                        f'p{p}': np.percentile(frame_df[range_col], p) for p in percentiles
                    }
                }
                
                # Add percentage statistics
                range_statistics[range_pct_col] = {
                    'mean': frame_df[range_pct_col].mean(),
                    'std': frame_df[range_pct_col].std(),
                    'min': frame_df[range_pct_col].min(),
                    'max': frame_df[range_pct_col].max(),
                    'median': frame_df[range_pct_col].median(),
                    'percentiles': {
                        f'p{p}': np.percentile(frame_df[range_pct_col], p) for p in percentiles
                    }
                }
            
            # Calculate distance-stratified density statistics
            density_statistics = {}
            
            # Define the annular ranges that match our density calculation
            annular_ranges = [
                (0, 0.5, 0.25),      # [0, 0.5m] with 0.25m local radius
                (0.5, 1, 0.25),      # [0.5, 1m] with 0.25m local radius  
                (1, 2.5, 0.5),       # [1, 2.5m] with 0.5m local radius
                (2.5, 5, 0.75),      # [2.5, 5m] with 0.75m local radius
                (5, 10, 1.0),        # [5, 10m] with 1.0m local radius
                (10, 25, 1.5),       # [10, 25m] with 1.5m local radius
                (25, 50, 2.0),       # [25, 50m] with 2.0m local radius
            ]
            
            for min_dist, max_dist, local_radius in annular_ranges:
                # Skip ranges not in our distance_ranges list
                if max_dist not in distance_ranges:
                    continue
                    
                range_name = f"{min_dist}_to_{max_dist}m"
                
                # Check if this density range exists in the frame data
                mean_col = f"{range_name}_mean_neighbors"
                if mean_col in frame_df.columns:
                    density_statistics[range_name] = {
                        'mean_neighbors_avg': frame_df[mean_col].mean(),
                        'mean_neighbors_std': frame_df[mean_col].std(),
                        'mean_neighbors_min': frame_df[mean_col].min(),
                        'mean_neighbors_max': frame_df[mean_col].max(),
                        'local_radius': local_radius
                    }
            
            # Calculate global density statistics
            global_density_statistics = {}
            if 'global_mean_neighbors' in frame_df.columns:
                global_density_statistics = {
                    'mean_neighbors_avg': frame_df['global_mean_neighbors'].mean(),
                    'mean_neighbors_std': frame_df['global_mean_neighbors'].std(),
                    'mean_neighbors_min': frame_df['global_mean_neighbors'].min(),
                    'mean_neighbors_max': frame_df['global_mean_neighbors'].max(),
                    'local_radius': LOCAL_DENSITY_RADIUS
                }
            
            # Calculate frame-level variation statistics
            distance_variation = {
                'min_mean_distance': frame_df['mean_distance'].min(),
                'max_mean_distance': frame_df['mean_distance'].max(),
                'min_max_distance': frame_df['max_distance'].min(),
                'max_max_distance': frame_df['max_distance'].max(),
                'distance_range_mean': frame_df['max_distance'].mean() - frame_df['min_distance'].mean(),
                'distance_range_std': (frame_df['max_distance'] - frame_df['min_distance']).std(),
            }
            
            intensity_variation = {
                'min_mean_intensity': frame_df['mean_intensity'].min(),
                'max_mean_intensity': frame_df['mean_intensity'].max(),
                'min_max_intensity': frame_df['max_intensity'].min(),
                'max_max_intensity': frame_df['max_intensity'].max(),
                'intensity_range_mean': frame_df['max_intensity'].mean() - frame_df['min_intensity'].mean(),
                'intensity_range_std': (frame_df['max_intensity'] - frame_df['min_intensity']).std(),
            }
            
            range_analysis = {
                'num_frames': len(self.lidar_tokens),
                'total_points': sum(stat['num_points'] for stat in frame_stats),
                'distance_ranges': distance_ranges,
                'range_statistics': range_statistics,
                'density_statistics': density_statistics,
                'global_density_statistics': global_density_statistics,
                'filtering_stats': {
                    'original_points': total_original_points,
                    'after_spatial_filter': total_after_spatial_filter,
                    'after_intensity_filter': total_after_intensity_filter,
                    'spatial_filter_retention': (total_after_spatial_filter / total_original_points * 100) if total_original_points > 0 else 0,
                    'intensity_filter_retention': (total_after_intensity_filter / total_after_spatial_filter * 100) if total_after_spatial_filter > 0 else 0,
                    'total_retention': (total_after_intensity_filter / total_original_points * 100) if total_original_points > 0 else 0
                },
                'mean_mean_distance': frame_df['mean_distance'].mean(),
                'std_mean_distance': frame_df['mean_distance'].std(),
                'mean_max_distance': frame_df['max_distance'].mean(),
                'std_max_distance': frame_df['max_distance'].std(),
                'mean_intensity': frame_df['mean_intensity'].mean(),
                'std_intensity': frame_df['std_intensity'].mean(),
                'intensity_range': frame_df['max_intensity'].mean() - frame_df['min_intensity'].mean(),
                
                # Global statistics across all points
                'global_distance_stats': {
                    'mean': np.mean(all_distances),
                    'std': np.std(all_distances),
                    'min': np.min(all_distances),
                    'max': np.max(all_distances),
                    'median': np.median(all_distances),
                    'percentiles': distance_percentiles
                },
                'global_intensity_stats': {
                    'mean': np.mean(all_intensities),
                    'std': np.std(all_intensities),
                    'min': np.min(all_intensities),
                    'max': np.max(all_intensities),
                    'median': np.median(all_intensities),
                    'percentiles': intensity_percentiles
                },
                
                # Frame-level variation
                'distance_variation': distance_variation,
                'intensity_variation': intensity_variation,
                
                # Original per-frame data
                'frame_stats': frame_stats
            }
        else:
            range_analysis = {}
        
        self.analysis_results['range_analysis'] = range_analysis
        return range_analysis
    
    def print_summary(self, save_to_file=True, output_dir="."):
        """Print a comprehensive summary of the range-based point count analysis and optionally save to file."""
        if 'range_analysis' not in self.analysis_results:
            self.analyze_point_density()
        
        analysis = self.analysis_results['range_analysis']
        
        # Prepare summary text
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append(f"LIDAR_TOP RANGE ANALYSIS SUMMARY FOR SCENE: {self.scene_token}")
        summary_lines.append("="*80)
        
        summary_lines.append(f"\nScene Information:")
        summary_lines.append(f"  Scene Name: {self.scene['name']}")
        summary_lines.append(f"  Description: {self.scene['description']}")
        summary_lines.append(f"  Number of Frames Processed: {analysis['num_frames']}")
        if MAX_FRAMES_PER_SCENE is not None:
            summary_lines.append(f"  Frame Limit Applied: First {MAX_FRAMES_PER_SCENE} frames only")
        summary_lines.append(f"  Total Points (filtered): {analysis['total_points']:,}")
        summary_lines.append(f"  Point Cloud Range Filter: {POINT_CLOUD_RANGE}")
        summary_lines.append(f"    X: [{POINT_CLOUD_RANGE[0]}, {POINT_CLOUD_RANGE[3]}] m")
        summary_lines.append(f"    Y: [{POINT_CLOUD_RANGE[1]}, {POINT_CLOUD_RANGE[4]}] m") 
        summary_lines.append(f"    Z: [{POINT_CLOUD_RANGE[2]}, {POINT_CLOUD_RANGE[5]}] m")
        summary_lines.append(f"  Intensity Filter: >= {MIN_INTENSITY_THRESHOLD}")
        summary_lines.append(f"  Local Density Radius: {LOCAL_DENSITY_RADIUS}m")
        
        # Add range-based point counts
        if analysis:
            summary_lines.append(f"\nPoint Percentages by Distance Range:")
            for r in analysis['distance_ranges']:
                range_pct_col = f"range_{r}m_pct"
                if range_pct_col in analysis['range_statistics']:
                    range_stats = analysis['range_statistics'][range_pct_col]
                    summary_lines.append(f"  Points within {r}m:")
                    summary_lines.append(f"    Mean Percentage: {range_stats['mean']:.2f}%")
                    summary_lines.append(f"    Std Dev: {range_stats['std']:.2f}%")
                    summary_lines.append(f"    Min Percentage: {range_stats['min']:.2f}%")
                    summary_lines.append(f"    Max Percentage: {range_stats['max']:.2f}%")
                    summary_lines.append(f"    Median Percentage: {range_stats['median']:.2f}%")
                    summary_lines.append("")
            
            summary_lines.append(f"Distance Statistics:")
            summary_lines.append(f"  Mean Distance: {analysis.get('mean_mean_distance', 0):.2f} ± {analysis.get('std_mean_distance', 0):.2f} m")
            summary_lines.append(f"  Max Distance: {analysis.get('mean_max_distance', 0):.2f} ± {analysis.get('std_max_distance', 0):.2f} m")
            summary_lines.append(f"  Mean Intensity: {analysis.get('mean_intensity', 0):.2f} ± {analysis.get('std_intensity', 0):.2f}")
            summary_lines.append(f"  Intensity Range: {analysis.get('intensity_range', 0):.2f}")
            
            # Add global distance statistics
            if 'global_distance_stats' in analysis:
                global_dist = analysis['global_distance_stats']
                summary_lines.append(f"\nGlobal Distance Statistics (All Points):")
                summary_lines.append(f"  Mean Distance: {global_dist['mean']:.2f} ± {global_dist['std']:.2f} m")
                summary_lines.append(f"  Min Distance: {global_dist['min']:.2f} m")
                summary_lines.append(f"  Max Distance: {global_dist['max']:.2f} m")
                summary_lines.append(f"  Median Distance: {global_dist['median']:.2f} m")
                
                # Add distance percentiles
                summary_lines.append(f"  Distance Percentiles:")
                for p in [10, 25, 50, 75, 90, 95, 99]:
                    summary_lines.append(f"    {p}th percentile: {global_dist['percentiles'][f'p{p}']:.2f} m")
            
            # Add global intensity statistics
            if 'global_intensity_stats' in analysis:
                global_int = analysis['global_intensity_stats']
                summary_lines.append(f"\nGlobal Intensity Statistics (All Points):")
                summary_lines.append(f"  Mean Intensity: {global_int['mean']:.2f} ± {global_int['std']:.2f}")
                summary_lines.append(f"  Min Intensity: {global_int['min']:.2f}")
                summary_lines.append(f"  Max Intensity: {global_int['max']:.2f}")
                summary_lines.append(f"  Median Intensity: {global_int['median']:.2f}")
                
                # Add intensity percentiles
                summary_lines.append(f"  Intensity Percentiles:")
                for p in [10, 25, 50, 75, 90, 95, 99]:
                    summary_lines.append(f"    {p}th percentile: {global_int['percentiles'][f'p{p}']:.2f}")
            
            # Add frame-level variation statistics
            if 'distance_variation' in analysis:
                dist_var = analysis['distance_variation']
                summary_lines.append(f"\nDistance Variation Across Frames:")
                summary_lines.append(f"  Mean Distance Range: {dist_var['min_mean_distance']:.2f} - {dist_var['max_mean_distance']:.2f} m")
                summary_lines.append(f"  Max Distance Range: {dist_var['min_max_distance']:.2f} - {dist_var['max_max_distance']:.2f} m")
                summary_lines.append(f"  Distance Range (Mean): {dist_var['distance_range_mean']:.2f} ± {dist_var['distance_range_std']:.2f} m")
            
            if 'intensity_variation' in analysis:
                int_var = analysis['intensity_variation']
                summary_lines.append(f"\nIntensity Variation Across Frames:")
                summary_lines.append(f"  Mean Intensity Range: {int_var['min_mean_intensity']:.2f} - {int_var['max_mean_intensity']:.2f}")
                summary_lines.append(f"  Max Intensity Range: {int_var['min_max_intensity']:.2f} - {int_var['max_max_intensity']:.2f}")
                summary_lines.append(f"  Intensity Range (Mean): {int_var['intensity_range_mean']:.2f} ± {int_var['intensity_range_std']:.2f}")

            # Add filtering statistics
            if 'filtering_stats' in analysis:
                filtering_stats = analysis['filtering_stats']
                summary_lines.append(f"\nFiltering Statistics:")
                summary_lines.append(f"  Original Points: {filtering_stats['original_points']:,}")
                summary_lines.append(f"  After Spatial Filter: {filtering_stats['after_spatial_filter']:,}")
                summary_lines.append(f"  After Intensity Filter: {filtering_stats['after_intensity_filter']:,}")
                summary_lines.append(f"  Spatial Filter Retention: {filtering_stats['spatial_filter_retention']:.2f}%")
                summary_lines.append(f"  Intensity Filter Retention: {filtering_stats['intensity_filter_retention']:.2f}%")
                summary_lines.append(f"  Total Retention: {filtering_stats['total_retention']:.2f}%")
        
            # Add local density statistics
            if 'density_statistics' in analysis:
                summary_lines.append(f"\nDistance-Stratified Local Density:")
                for range_name, density_stats in analysis['density_statistics'].items():
                    local_radius = density_stats.get('local_radius', 'N/A')
                    summary_lines.append(f"  {range_name} (radius: {local_radius}m):")
                    summary_lines.append(f"    Mean Neighbors per Point: {density_stats['mean_neighbors_avg']:.2f}")
                    summary_lines.append(f"    Std Dev: {density_stats['mean_neighbors_std']:.2f}")
                    summary_lines.append(f"    Range: {density_stats['mean_neighbors_min']:.2f} - {density_stats['mean_neighbors_max']:.2f}")
                    summary_lines.append("")
            
            # Add global density statistics
            if 'global_density_statistics' in analysis:
                global_stats = analysis['global_density_statistics']
                if global_stats:  # Check if not empty
                    summary_lines.append(f"\nGlobal Local Density (radius: {global_stats.get('local_radius', LOCAL_DENSITY_RADIUS)}m):")
                    summary_lines.append(f"  Mean Neighbors per Point: {global_stats['mean_neighbors_avg']:.2f}")
                    summary_lines.append(f"  Std Dev: {global_stats['mean_neighbors_std']:.2f}")
                    summary_lines.append(f"  Range: {global_stats['mean_neighbors_min']:.2f} - {global_stats['mean_neighbors_max']:.2f}")
                    summary_lines.append("")
        
        summary_lines.append("="*80)
        
        # Print to console
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        
        # Save to file if requested
        if save_to_file:
            # Create a safe filename from scene token
            safe_scene_name = self.scene_token.replace('/', '_').replace('\\', '_')
            summary_file = os.path.join(output_dir, f"lidar_range_summary_{safe_scene_name}.txt")
            
            with open(summary_file, 'w') as f:
                f.write(summary_text)
            
            print(f"\nSummary saved to: {summary_file}")
        
        return summary_text
    
    def save_results(self, output_file: str = "lidar_range_results.json"):
        """Save analysis results to a JSON file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.analysis_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        serializable_results[key][subkey] = subvalue.tolist()
                    elif isinstance(subvalue, np.integer):
                        serializable_results[key][subkey] = int(subvalue)
                    elif isinstance(subvalue, np.floating):
                        serializable_results[key][subkey] = float(subvalue)
                    elif isinstance(subvalue, np.bool_):
                        serializable_results[key][subkey] = bool(subvalue)
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def get_csv_data(self) -> List[Dict]:
        """
        Extract CSV-ready data from analysis results.
        
        Returns:
            List of dictionaries with scene_name, distance, mean_percentage, std_dev_percentage, 
            distance-specific local density metrics, and global density metrics
        """
        if 'range_analysis' not in self.analysis_results:
            self.analyze_point_density()
        
        analysis = self.analysis_results['range_analysis']
        csv_data = []
        
        scene_name = self.scene['name']
        
        # Get global density stats for the scene
        global_density_stats = analysis.get('global_density_statistics', {})
        global_density_avg = global_density_stats.get('mean_neighbors_avg', 0)
        global_density_std = global_density_stats.get('mean_neighbors_std', 0)
        
        for i, distance in enumerate(analysis['distance_ranges']):
            range_pct_col = f"range_{distance}m_pct"
            if range_pct_col in analysis['range_statistics']:
                range_stats = analysis['range_statistics'][range_pct_col]
                
                # Define the annular ranges to match our density calculation
                annular_ranges = [
                    (0, 0.5), (0.5, 1), (1, 2.5), (2.5, 5), (5, 10), (10, 25), (25, 50)
                ]
                
                # Find the corresponding density range for this distance
                density_range_name = None
                for min_dist, max_dist in annular_ranges:
                    if max_dist == distance:
                        density_range_name = f"{min_dist}_to_{max_dist}m"
                        break
                
                # Get density stats for this specific distance range
                density_stats = analysis.get('density_statistics', {}).get(density_range_name, {}) if density_range_name else {}
                local_density_avg = density_stats.get('mean_neighbors_avg', 0)
                local_density_std = density_stats.get('mean_neighbors_std', 0)
                
                csv_data.append({
                    'scene_name': scene_name,
                    'distance': distance,
                    'mean_percentage': round(range_stats['mean'], 2),
                    'std_dev_percentage': round(range_stats['std'], 2),
                    'local_density_avg': round(local_density_avg, 2),
                    'local_density_std': round(local_density_std, 2),
                    'global_density_avg': round(global_density_avg, 2),
                    'global_density_std': round(global_density_std, 2)
                })
        
        return csv_data

    def calculate_local_density(self, points: np.ndarray, radius: float = LOCAL_DENSITY_RADIUS) -> Dict:
        """
        Calculate local neighborhood density for each point.
        
        Args:
            points: Point cloud array (N x 5) with [x, y, z, intensity, ring]
            radius: Radius for neighbor search in meters
            
        Returns:
            Dictionary with local density statistics
        """
        if len(points) == 0:
            return {
                'mean_neighbors': 0,
                'std_neighbors': 0,
                'max_neighbors': 0,
                'min_neighbors': 0,
                'median_neighbors': 0,
                'neighbor_counts': []
            }
        
        # Extract 3D coordinates
        coords = points[:, :3]  # x, y, z
        
        # Build KD-tree for efficient neighbor search
        tree = cKDTree(coords)
        
        # Count neighbors within radius for each point
        neighbor_counts = []
        
        # Use query_ball_tree for efficiency - returns indices within radius
        neighbor_indices = tree.query_ball_tree(tree, radius)
        
        for i, neighbors in enumerate(neighbor_indices):
            # Subtract 1 to exclude the point itself
            neighbor_count = len(neighbors) - 1
            neighbor_counts.append(neighbor_count)
        
        neighbor_counts = np.array(neighbor_counts)
        
        return {
            'mean_neighbors': np.mean(neighbor_counts),
            'std_neighbors': np.std(neighbor_counts),
            'max_neighbors': np.max(neighbor_counts),
            'min_neighbors': np.min(neighbor_counts),
            'median_neighbors': np.median(neighbor_counts),
            'neighbor_counts': neighbor_counts
        }
    
    def calculate_distance_stratified_density(self, points: np.ndarray, distance_ranges: List[float]) -> Dict:
        """
        Calculate local density for points within each distance range.
        Uses annular regions: [0, 0.5], [0.5, 1], [1, 2.5], [2.5, 5], [5, 10], etc.
        
        Args:
            points: Point cloud array (N x 5) with [x, y, z, intensity, ring]
            distance_ranges: List of distance thresholds
            
        Returns:
            Dictionary with distance-stratified density statistics
        """
        if len(points) == 0:
            return {}
        
        # Calculate distances from origin for all points
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        
        density_by_range = {}
        
        # Define specific annular ranges with appropriate local density radius
        annular_ranges = [
            (0, 0.5, 0.25),      # [0, 0.5m] with 0.25m local radius
            (0.5, 1, 0.25),      # [0.5, 1m] with 0.25m local radius  
            (1, 2.5, 0.5),       # [1, 2.5m] with 0.5m local radius
            (2.5, 5, 0.75),      # [2.5, 5m] with 0.75m local radius
            (5, 10, 1.0),        # [5, 10m] with 1.0m local radius
            (10, 25, 1.5),       # [10, 25m] with 1.5m local radius
            (25, 50, 2.0),       # [25, 50m] with 2.0m local radius
        ]
        
        for min_dist, max_dist, local_radius in annular_ranges:
            # Skip ranges not in our distance_ranges list
            if max_dist not in distance_ranges:
                continue
                
            # Get points in this annular range
            mask = (distances >= min_dist) & (distances < max_dist)
            range_points = points[mask]
            
            range_name = f"{min_dist}_to_{max_dist}m"
            
            if len(range_points) > 0:
                # Calculate local density for points in this range with appropriate radius
                density_stats = self.calculate_local_density(range_points, local_radius)
                density_by_range[range_name] = density_stats
                print(f"    {range_name}: {len(range_points)} points, radius={local_radius}m, avg_neighbors={density_stats['mean_neighbors']:.2f}")
            else:
                # No points in this range
                density_by_range[range_name] = {
                    'mean_neighbors': 0,
                    'std_neighbors': 0,
                    'max_neighbors': 0,
                    'min_neighbors': 0,
                    'median_neighbors': 0,
                    'neighbor_counts': []
                }
                print(f"    {range_name}: 0 points")
        
        return density_by_range


def main():
    """Main function to run the range analysis based on configuration parameters."""
    OUTPUT_DIR = "post_processing/lidar_analysis_output"  # Directory for results
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filtered_scenes = scenes[:10]
    
    # Collect CSV data from all scenes
    all_csv_data = []
    
    try:
        # Initialize nuScenes
        print("Initializing nuScenes...")
        nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/nas/Nuscenes', verbose=True)
        
        # Process each scene
        for scene_token in filtered_scenes:
            print(f"\n{'='*60}")
            print(f"Processing scene: {scene_token}")
            print(f"{'='*60}")
            
            # Verify scene exists and get scene info
            try:
                scene_info = nusc.get('scene', scene_token)
                print(f"Scene name: {scene_info['name']}")
                print(f"Scene description: {scene_info['description']}")
                print(f"Scene token: {scene_info['token']}")
            except Exception as e:
                print(f"ERROR: Could not load scene {scene_token}: {e}")
                continue
            
            # Create scene-specific output directory
            safe_scene_name = scene_token.replace('/', '_').replace('\\', '_')
            scene_output_dir = os.path.join(OUTPUT_DIR, safe_scene_name)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            try:
                # Create analyzer
                analyzer = LidarRangeAnalyzer(nusc, scene_token)
                
                # Run range analysis
                print(f"Starting range analysis for scene: {scene_token}")
                range_stats = analyzer.analyze_point_density()
                
                # Print summary and save to file
                analyzer.print_summary(save_to_file=True, output_dir=scene_output_dir)
                
                # Collect CSV data for this scene
                scene_csv_data = analyzer.get_csv_data()
                all_csv_data.extend(scene_csv_data)
                
                # Save detailed results if requested
                if SAVE_RESULTS:
                    result_file = os.path.join(scene_output_dir, f"range_results_{safe_scene_name}.json")
                    analyzer.save_results(result_file)
                
                print(f"Range analysis complete for scene: {scene_token}")
                print(f"Results saved to: {scene_output_dir}")
                
            except Exception as e:
                print(f"Error analyzing scene {scene_token}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate consolidated CSV file
        if all_csv_data:
            csv_file_path = os.path.join(OUTPUT_DIR, "distance_analysis.csv")
            
            with open(csv_file_path, 'w', newline='') as csvfile:
                fieldnames = ['scene_name', 'distance', 'mean_percentage', 'std_dev_percentage', 'local_density_avg', 'local_density_std', 'global_density_avg', 'global_density_std']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write data
                for row in all_csv_data:
                    writer.writerow(row)
            
            print(f"\n{'='*60}")
            print(f"CSV file generated: {csv_file_path}")
            print(f"Total rows: {len(all_csv_data)} (excluding header)")
            print(f"Scenes processed: {len(set(row['scene_name'] for row in all_csv_data))}")
            print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print("All scene range analysis complete!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 