#!/usr/bin/env python3
"""
Script to analyze the size of LIDAR_TOP data under a scene in nuScenes dataset.
This script provides comprehensive statistics about point cloud data including:
- Number of points per frame
- File sizes
- Point density statistics
- Temporal analysis

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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid displaying plots
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from nuscenes import NuScenes


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

# Scene to analyze (required)

# Output settings

GENERATE_PLOTS = True  # Set to False to skip plot generation
SAVE_RESULTS = False  # Set to True to save detailed results to JSON

# =============================================================================


class LidarSceneAnalyzer:
    """Analyzer for LIDAR_TOP data in nuScenes scenes."""
    
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
        self.lidar_tokens = load_sweep_sd(nusc, self.scene, sensor_channel='LIDAR_TOP')
        self.analysis_results = {}
        
    def analyze_point_cloud_sizes(self) -> Dict:
        """
        Analyze the number of points in each LIDAR_TOP frame.
        
        Returns:
            Dictionary containing point count statistics
        """
        print(f"Analyzing {len(self.lidar_tokens)} LIDAR_TOP frames...")
        
        point_counts = []
        file_sizes = []
        timestamps = []
        filenames = []
        
        for token in self.lidar_tokens:
            # Get sample data
            sd = self.nusc.get('sample_data', token)
            
            # Get file path and size
            file_path = os.path.join(self.nusc.dataroot, sd['filename'])
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                file_sizes.append(file_size)
            else:
                print(f"Warning: File not found: {file_path}")
                file_sizes.append(0)
            
            # Load point cloud data
            try:
                points = np.fromfile(file_path, dtype=np.float32)
                # Reshape to get the correct number of points
                points_reshaped = points.reshape(-1, 5)  # Each point has 5 values (x, y, z, intensity, ring)
                num_points = points_reshaped.shape[0]  # Get number of points using shape
                point_counts.append(num_points)
                
                # Debug: Print first few files to see if they're different
                if len(point_counts) <= 5:
                    print(f"  Debug: File {len(point_counts)}: {sd['filename']}")
                    print(f"  Debug: Points: {num_points}, File size: {file_size} bytes")
                    print(f"  Debug: Raw array shape: {points.shape}, Reshaped: {points_reshaped.shape}")
                
                # Store timestamp and filename
                timestamps.append(sd['timestamp'] / 1e6)  # Convert to seconds
                filenames.append(sd['filename'])
                
            except Exception as e:
                print(f"Error loading point cloud for token {token}: {e}")
                point_counts.append(0)
                timestamps.append(sd['timestamp'] / 1e6)
                filenames.append(sd['filename'])
        
        # Calculate statistics
        point_counts = np.array(point_counts)
        file_sizes = np.array(file_sizes)
        
        # Debug: Check for duplicate filenames
        unique_filenames = set(filenames)
        print(f"  Debug: Total files processed: {len(filenames)}")
        print(f"  Debug: Unique filenames: {len(unique_filenames)}")
        if len(filenames) != len(unique_filenames):
            print(f"  Warning: Found {len(filenames) - len(unique_filenames)} duplicate files!")
            # Show some duplicates
            filename_counts = {}
            for filename in filenames:
                filename_counts[filename] = filename_counts.get(filename, 0) + 1
            duplicates = {k: v for k, v in filename_counts.items() if v > 1}
            print(f"  Debug: Duplicate files: {list(duplicates.keys())[:5]}")  # Show first 5
        
        # Debug: Show actual statistics for this scene
        print(f"  DEBUG SCENE STATS:")
        print(f"    Scene: {self.scene['name']}")
        print(f"    Total points: {np.sum(point_counts):,}")
        print(f"    Mean points: {np.mean(point_counts):.0f}")
        print(f"    Min points: {np.min(point_counts):,}")
        print(f"    Max points: {np.max(point_counts):,}")
        print(f"    Total file size: {np.sum(file_sizes) / (1024*1024):.2f} MB")
        print(f"    First 5 point counts: {point_counts[:5]}")
        print(f"    First 5 file sizes (MB): {file_sizes[:5] / (1024*1024)}")
        
        # Check if all values are the same (which would indicate a problem)
        if len(point_counts) > 1:
            if np.all(point_counts == point_counts[0]):
                print(f"  WARNING: All point counts are identical ({point_counts[0]}) - this suggests a problem!")
            if np.all(file_sizes == file_sizes[0]):
                print(f"  WARNING: All file sizes are identical ({file_sizes[0]}) - this suggests a problem!")
        
        stats = {
            'point_counts': point_counts,
            'file_sizes': file_sizes,
            'timestamps': timestamps,
            'filenames': filenames,
            'num_frames': len(self.lidar_tokens),
            'total_points': np.sum(point_counts),
            'mean_points': np.mean(point_counts),
            'std_points': np.std(point_counts),
            'min_points': np.min(point_counts),
            'max_points': np.max(point_counts),
            'median_points': np.median(point_counts),
            'total_file_size_mb': np.sum(file_sizes) / (1024 * 1024),
            'mean_file_size_mb': np.mean(file_sizes) / (1024 * 1024),
            'std_file_size_mb': np.std(file_sizes) / (1024 * 1024),
            'min_file_size_mb': np.min(file_sizes) / (1024 * 1024),
            'max_file_size_mb': np.max(file_sizes) / (1024 * 1024),
            'median_file_size_mb': np.median(file_sizes) / (1024 * 1024),
        }
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'points_p{p}'] = np.percentile(point_counts, p)
            stats[f'file_size_p{p}_mb'] = np.percentile(file_sizes, p) / (1024 * 1024)
        
        self.analysis_results['point_cloud_stats'] = stats
        return stats
    
    def analyze_point_density(self) -> Dict:
        """
        Analyze point density and distribution characteristics.
        
        Returns:
            Dictionary containing density analysis
        """
        print("Analyzing point density characteristics...")
        
        density_stats = []
        intensity_stats = []
        all_distances = []
        all_intensities = []
        
        for token in self.lidar_tokens:
            sd = self.nusc.get('sample_data', token)
            file_path = os.path.join(self.nusc.dataroot, sd['filename'])
            
            if not os.path.exists(file_path):
                continue
                
            try:
                # Load point cloud data
                points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
                
                # Calculate distances from origin
                distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
                
                # Calculate point density (points per unit volume)
                # Using a simple spherical shell approach
                max_dist = np.max(distances)
                if max_dist > 0:
                    # Approximate density as points per cubic meter
                    volume = (4/3) * np.pi * max_dist**3
                    density = len(points) / volume if volume > 0 else 0
                else:
                    density = 0
                
                # Intensity statistics
                intensities = points[:, 3]
                
                # Store all distances and intensities for global analysis
                all_distances.extend(distances.tolist())
                all_intensities.extend(intensities.tolist())
                
                density_stats.append({
                    'mean_distance': np.mean(distances),
                    'max_distance': max_dist,
                    'min_distance': np.min(distances),
                    'std_distance': np.std(distances),
                    'density': density,
                    'num_points': len(points)
                })
                
                intensity_stats.append({
                    'mean_intensity': np.mean(intensities),
                    'std_intensity': np.std(intensities),
                    'min_intensity': np.min(intensities),
                    'max_intensity': np.max(intensities),
                    'median_intensity': np.median(intensities)
                })
                
            except Exception as e:
                print(f"Error analyzing density for token {token}: {e}")
        
        if density_stats:
            # Aggregate statistics
            density_df = pd.DataFrame(density_stats)
            intensity_df = pd.DataFrame(intensity_stats)
            
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
            
            # Calculate frame-level variation statistics
            distance_variation = {
                'min_mean_distance': density_df['mean_distance'].min(),
                'max_mean_distance': density_df['mean_distance'].max(),
                'min_max_distance': density_df['max_distance'].min(),
                'max_max_distance': density_df['max_distance'].max(),
                'distance_range_mean': density_df['max_distance'].mean() - density_df['min_distance'].mean(),
                'distance_range_std': (density_df['max_distance'] - density_df['min_distance']).std(),
            }
            
            intensity_variation = {
                'min_mean_intensity': intensity_df['mean_intensity'].min(),
                'max_mean_intensity': intensity_df['mean_intensity'].max(),
                'min_max_intensity': intensity_df['max_intensity'].min(),
                'max_max_intensity': intensity_df['max_intensity'].max(),
                'intensity_range_mean': intensity_df['max_intensity'].mean() - intensity_df['min_intensity'].mean(),
                'intensity_range_std': (intensity_df['max_intensity'] - intensity_df['min_intensity']).std(),
            }
            
            density_analysis = {
                'mean_density': density_df['density'].mean(),
                'std_density': density_df['density'].std(),
                'mean_mean_distance': density_df['mean_distance'].mean(),
                'std_mean_distance': density_df['mean_distance'].std(),
                'mean_max_distance': density_df['max_distance'].mean(),
                'std_max_distance': density_df['max_distance'].std(),
                'mean_intensity': intensity_df['mean_intensity'].mean(),
                'std_intensity': intensity_df['std_intensity'].mean(),
                'intensity_range': intensity_df['max_intensity'].mean() - intensity_df['min_intensity'].mean(),
                
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
                'density_stats': density_stats,
                'intensity_stats': intensity_stats
            }
        else:
            density_analysis = {}
        
        self.analysis_results['density_analysis'] = density_analysis
        return density_analysis
    
    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze temporal patterns in the data.
        
        Returns:
            Dictionary containing temporal analysis
        """
        print("Analyzing temporal patterns...")
        
        if 'point_cloud_stats' not in self.analysis_results:
            self.analyze_point_cloud_sizes()
        
        timestamps = self.analysis_results['point_cloud_stats']['timestamps']
        point_counts = self.analysis_results['point_cloud_stats']['point_counts']
        
        # Calculate time intervals
        time_intervals = np.diff(timestamps)
        
        # Calculate frame rate statistics
        frame_rates = 1.0 / time_intervals if len(time_intervals) > 0 else []
        
        temporal_stats = {
            'scene_duration_seconds': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'mean_frame_interval': np.mean(time_intervals) if len(time_intervals) > 0 else 0,
            'std_frame_interval': np.std(time_intervals) if len(time_intervals) > 0 else 0,
            'mean_frame_rate': np.mean(frame_rates) if len(frame_rates) > 0 else 0,
            'std_frame_rate': np.std(frame_rates) if len(frame_rates) > 0 else 0,
            'min_frame_interval': np.min(time_intervals) if len(time_intervals) > 0 else 0,
            'max_frame_interval': np.max(time_intervals) if len(time_intervals) > 0 else 0,
            'total_frames': len(timestamps),
            'time_intervals': time_intervals.tolist(),
            'frame_rates': frame_rates,
            'timestamps': timestamps
        }
        
        # Analyze point count trends over time
        if len(point_counts) > 1:
            # Simple linear trend
            time_normalized = np.array(timestamps) - timestamps[0]
            coeffs = np.polyfit(time_normalized, point_counts, 1)
            temporal_stats['point_count_trend_slope'] = coeffs[0]
            temporal_stats['point_count_trend_intercept'] = coeffs[1]
            
            # Correlation between time and point count
            correlation = np.corrcoef(time_normalized, point_counts)[0, 1]
            temporal_stats['time_point_count_correlation'] = correlation
        
        self.analysis_results['temporal_analysis'] = temporal_stats
        return temporal_stats
    
    def generate_plots(self, output_dir: str = "lidar_analysis_plots"):
        """
        Generate visualization plots for the analysis.
        
        Args:
            output_dir: Directory to save the plots
        """
        if 'point_cloud_stats' not in self.analysis_results:
            self.analyze_point_cloud_sizes()
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Point count distribution
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'LIDAR_TOP Analysis for Scene: {self.scene_token}', fontsize=16)
            
            point_counts = self.analysis_results['point_cloud_stats']['point_counts']
            file_sizes = self.analysis_results['point_cloud_stats']['file_sizes'] / (1024 * 1024)  # Convert to MB
            
            # Point count histogram
            axes[0, 0].hist(point_counts, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Number of Points')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Point Count Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # File size histogram
            axes[0, 1].hist(file_sizes, bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[0, 1].set_xlabel('File Size (MB)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('File Size Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Point count over time
            timestamps = self.analysis_results['point_cloud_stats']['timestamps']
            axes[1, 0].plot(timestamps, point_counts, 'b-', alpha=0.7, linewidth=1)
            axes[1, 0].scatter(timestamps, point_counts, c='red', s=20, alpha=0.6)
            axes[1, 0].set_xlabel('Timestamp (seconds)')
            axes[1, 0].set_ylabel('Number of Points')
            axes[1, 0].set_title('Point Count Over Time')
            axes[1, 0].grid(True, alpha=0.3)
            
            # File size vs point count scatter
            axes[1, 1].scatter(point_counts, file_sizes, alpha=0.6, c='green')
            axes[1, 1].set_xlabel('Number of Points')
            axes[1, 1].set_ylabel('File Size (MB)')
            axes[1, 1].set_title('File Size vs Point Count')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'lidar_analysis_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Temporal analysis plots
            if 'temporal_analysis' in self.analysis_results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Temporal Analysis for Scene: {self.scene_token}', fontsize=16)
                
                temporal_stats = self.analysis_results['temporal_analysis']
                
                # Frame intervals
                if temporal_stats['time_intervals']:
                    axes[0, 0].hist(temporal_stats['time_intervals'], bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 0].set_xlabel('Frame Interval (seconds)')
                    axes[0, 0].set_ylabel('Frequency')
                    axes[0, 0].set_title('Frame Interval Distribution')
                    axes[0, 0].grid(True, alpha=0.3)
                
                # Frame rates
                if temporal_stats['frame_rates']:
                    axes[0, 1].hist(temporal_stats['frame_rates'], bins=20, alpha=0.7, edgecolor='black', color='orange')
                    axes[0, 1].set_xlabel('Frame Rate (Hz)')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].set_title('Frame Rate Distribution')
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Point count trend
                if len(point_counts) > 1:
                    time_normalized = np.array(timestamps) - timestamps[0]
                    axes[1, 0].scatter(time_normalized, point_counts, alpha=0.6, c='blue')
                    
                    # Plot trend line
                    if 'point_count_trend_slope' in temporal_stats:
                        trend_line = temporal_stats['point_count_trend_slope'] * time_normalized + temporal_stats['point_count_trend_intercept']
                        axes[1, 0].plot(time_normalized, trend_line, 'r-', linewidth=2, label=f'Trend (slope: {temporal_stats["point_count_trend_slope"]:.2f})')
                        axes[1, 0].legend()
                    
                    axes[1, 0].set_xlabel('Time (seconds)')
                    axes[1, 0].set_ylabel('Number of Points')
                    axes[1, 0].set_title('Point Count Trend Over Time')
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Correlation heatmap
                if len(point_counts) > 1 and len(file_sizes) > 1:
                    try:
                        correlation_data = np.corrcoef([point_counts, file_sizes, time_normalized])[:2, :2]
                        im = axes[1, 1].imshow(correlation_data, cmap='coolwarm', vmin=-1, vmax=1)
                        axes[1, 1].set_xticks([0, 1])
                        axes[1, 1].set_yticks([0, 1])
                        axes[1, 1].set_xticklabels(['Points', 'File Size'])
                        axes[1, 1].set_yticklabels(['Points', 'File Size'])
                        axes[1, 1].set_title('Correlation Matrix')
                        
                        # Add correlation values as text
                        for i in range(2):
                            for j in range(2):
                                text = axes[1, 1].text(j, i, f'{correlation_data[i, j]:.3f}',
                                                     ha="center", va="center", color="black", fontweight='bold')
                        
                        plt.colorbar(im, ax=axes[1, 1])
                    except Exception as e:
                        print(f"Warning: Could not create correlation heatmap: {e}")
                        axes[1, 1].text(0.5, 0.5, 'Correlation\nData\nUnavailable', 
                                       ha='center', va='center', transform=axes[1, 1].transAxes)
                        axes[1, 1].set_title('Correlation Matrix')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'temporal_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Plots saved to {output_dir}/")
            
        except Exception as e:
            print(f"Warning: Could not generate plots due to error: {e}")
            print("Continuing with analysis without plots...")
            # Continue without plots if there's an error
    
    def print_summary(self, save_to_file=True, output_dir="."):
        """Print a comprehensive summary of the analysis and optionally save to file."""
        if 'point_cloud_stats' not in self.analysis_results:
            self.analyze_point_cloud_sizes()
        
        if 'density_analysis' not in self.analysis_results:
            self.analyze_point_density()
        
        if 'temporal_analysis' not in self.analysis_results:
            self.analyze_temporal_patterns()
        
        stats = self.analysis_results['point_cloud_stats']
        density = self.analysis_results['density_analysis']
        temporal = self.analysis_results['temporal_analysis']
        
        # Prepare summary text
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append(f"LIDAR_TOP ANALYSIS SUMMARY FOR SCENE: {self.scene_token}")
        summary_lines.append("="*80)
        
        summary_lines.append(f"\nScene Information:")
        summary_lines.append(f"  Scene Name: {self.scene['name']}")
        summary_lines.append(f"  Description: {self.scene['description']}")
        summary_lines.append(f"  Number of Frames: {stats['num_frames']}")
        summary_lines.append(f"  Scene Duration: {temporal['scene_duration_seconds']:.2f} seconds")
        
        summary_lines.append(f"\nPoint Cloud Statistics:")
        summary_lines.append(f"  Total Points: {stats['total_points']:,}")
        summary_lines.append(f"  Mean Points per Frame: {stats['mean_points']:.0f} ± {stats['std_points']:.0f}")
        summary_lines.append(f"  Median Points per Frame: {stats['median_points']:.0f}")
        summary_lines.append(f"  Min Points per Frame: {stats['min_points']:,}")
        summary_lines.append(f"  Max Points per Frame: {stats['max_points']:,}")
        summary_lines.append(f"  Point Count Range: {stats['max_points'] - stats['min_points']:,}")
        
        summary_lines.append(f"\nFile Size Statistics:")
        summary_lines.append(f"  Total Data Size: {stats['total_file_size_mb']:.2f} MB")
        summary_lines.append(f"  Mean File Size: {stats['mean_file_size_mb']:.2f} ± {stats['std_file_size_mb']:.2f} MB")
        summary_lines.append(f"  Median File Size: {stats['median_file_size_mb']:.2f} MB")
        summary_lines.append(f"  Min File Size: {stats['min_file_size_mb']:.2f} MB")
        summary_lines.append(f"  Max File Size: {stats['max_file_size_mb']:.2f} MB")
        
        # Add density analysis if available
        if density:
            summary_lines.append(f"\nPoint Density Statistics:")
            summary_lines.append(f"  Mean Point Density: {density.get('mean_density', 0):.2f} points/m³")
            summary_lines.append(f"  Mean Distance: {density.get('mean_mean_distance', 0):.2f} ± {density.get('std_mean_distance', 0):.2f} m")
            summary_lines.append(f"  Max Distance: {density.get('mean_max_distance', 0):.2f} ± {density.get('std_max_distance', 0):.2f} m")
            summary_lines.append(f"  Mean Intensity: {density.get('mean_intensity', 0):.2f} ± {density.get('std_intensity', 0):.2f}")
            summary_lines.append(f"  Intensity Range: {density.get('intensity_range', 0):.2f}")
            
            # Add global distance statistics
            if 'global_distance_stats' in density:
                global_dist = density['global_distance_stats']
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
            if 'global_intensity_stats' in density:
                global_int = density['global_intensity_stats']
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
            if 'distance_variation' in density:
                dist_var = density['distance_variation']
                summary_lines.append(f"\nDistance Variation Across Frames:")
                summary_lines.append(f"  Mean Distance Range: {dist_var['min_mean_distance']:.2f} - {dist_var['max_mean_distance']:.2f} m")
                summary_lines.append(f"  Max Distance Range: {dist_var['min_max_distance']:.2f} - {dist_var['max_max_distance']:.2f} m")
                summary_lines.append(f"  Distance Range (Mean): {dist_var['distance_range_mean']:.2f} ± {dist_var['distance_range_std']:.2f} m")
            
            if 'intensity_variation' in density:
                int_var = density['intensity_variation']
                summary_lines.append(f"\nIntensity Variation Across Frames:")
                summary_lines.append(f"  Mean Intensity Range: {int_var['min_mean_intensity']:.2f} - {int_var['max_mean_intensity']:.2f}")
                summary_lines.append(f"  Max Intensity Range: {int_var['min_max_intensity']:.2f} - {int_var['max_max_intensity']:.2f}")
                summary_lines.append(f"  Intensity Range (Mean): {int_var['intensity_range_mean']:.2f} ± {int_var['intensity_range_std']:.2f}")
        
        summary_lines.append(f"\nTemporal Statistics:")
        summary_lines.append(f"  Mean Frame Rate: {temporal['mean_frame_rate']:.2f} ± {temporal['std_frame_rate']:.2f} Hz")
        summary_lines.append(f"  Mean Frame Interval: {temporal['mean_frame_interval']:.3f} ± {temporal['std_frame_interval']:.3f} seconds")
        summary_lines.append(f"  Frame Interval Range: {temporal['min_frame_interval']:.3f} - {temporal['max_frame_interval']:.3f} seconds")
        
        if 'point_count_trend_slope' in temporal:
            summary_lines.append(f"  Point Count Trend: {temporal['point_count_trend_slope']:.2f} points/second")
            summary_lines.append(f"  Time-Point Count Correlation: {temporal['time_point_count_correlation']:.3f}")
        
        summary_lines.append(f"\nPercentiles (Point Counts):")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            summary_lines.append(f"  {p}th percentile: {stats[f'points_p{p}']:.0f} points")
        
        summary_lines.append(f"\nPercentiles (File Sizes):")
        for p in percentiles:
            summary_lines.append(f"  {p}th percentile: {stats[f'file_size_p{p}_mb']:.2f} MB")
        
        summary_lines.append("="*80)
        
        # Print to console
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        
        # Save to file if requested
        if save_to_file:
            # Create a safe filename from scene token
            safe_scene_name = self.scene_token.replace('/', '_').replace('\\', '_')
            summary_file = os.path.join(output_dir, f"lidar_summary_{safe_scene_name}.txt")
            
            with open(summary_file, 'w') as f:
                f.write(summary_text)
            
            print(f"\nSummary saved to: {summary_file}")
        
        return summary_text
    
    def save_results(self, output_file: str = "lidar_analysis_results.json"):
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


def main():
    """Main function to run the analysis based on configuration parameters."""
    OUTPUT_DIR = "post_processing/lidar_analysis_output"  # Directory for results and plots
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenes = ['2f0e54af35964a3fb347359836bec035', 
              '2f0e54af35964a3fb347359836bec035_rainrate25',
              '2f0e54af35964a3fb347359836bec035_rainrate50',
              '2f0e54af35964a3fb347359836bec035_rainrate100',
              'bcb05cffb5814973a6cff4fbdca2b99b', # Highway
              '32185f91e68f4069ab3cdd2f4f1a4ff1',]
    
    try:
        # Initialize nuScenes
        print("Initializing nuScenes...")
        nusc = NuScenes(version='v1.0-trainval-rain', dataroot='/mnt/nas/Nuscenes', verbose=True)
        
        # Process each scene
        for scene_token in scenes:
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
                analyzer = LidarSceneAnalyzer(nusc, scene_token)
                
                # Run analysis
                print(f"Starting analysis for scene: {scene_token}")
                
                # Point cloud size analysis
                point_stats = analyzer.analyze_point_cloud_sizes()
                
                # Density analysis
                density_stats = analyzer.analyze_point_density()
                
                # Temporal analysis
                temporal_stats = analyzer.analyze_temporal_patterns()
                
                # Print summary and save to file
                analyzer.print_summary(save_to_file=True, output_dir=scene_output_dir)
                
                # Generate plots
                if GENERATE_PLOTS:
                    analyzer.generate_plots(scene_output_dir)
                
                print(f"Analysis complete for scene: {scene_token}")
                print(f"Results saved to: {scene_output_dir}")
                
            except Exception as e:
                print(f"Error analyzing scene {scene_token}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*60}")
        print("All scene analysis complete!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 