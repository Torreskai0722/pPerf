#!/usr/bin/env python3
"""
Script to visualize LIDAR_TOP point clouds from nuScenes scenes.
This script provides simple point cloud visualization for analysis and debugging.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, List
import argparse

# Set environment variables to prevent Qt display issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':99'

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: open3d not available. Will use matplotlib for visualization.")
    OPEN3D_AVAILABLE = False

from nuscenes import NuScenes


def load_sweep_sd(nusc, scene, sensor_channel='LIDAR_TOP'):
    """
    Get all sample_data tokens for a given channel from a scene.
    
    Args:
        nusc: NuScenes instance
        scene: scene dictionary from nusc.scene
        sensor_channel: e.g., 'LIDAR_TOP'
    
    Returns:
        List of sample_data tokens (str) ordered by time
    """
    # Get the first sample token
    first_sample_token = scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    
    # Get the initial sample_data token for the desired sensor channel
    current_sd_token = None
    for sd_token in first_sample['data'].values():
        sd = nusc.get('sample_data', sd_token)
        if sd['channel'] == sensor_channel:
            current_sd_token = sd_token
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


def load_point_cloud(nusc, token: str) -> np.ndarray:
    """
    Load point cloud data from a sample_data token.
    
    Args:
        nusc: NuScenes instance
        token: sample_data token
    
    Returns:
        Point cloud as numpy array with shape (N, 5) [x, y, z, intensity, ring]
    """
    sd = nusc.get('sample_data', token)
    file_path = os.path.join(nusc.dataroot, sd['filename'])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    
    # Load point cloud data
    points = np.fromfile(file_path, dtype=np.float32)
    points_reshaped = points.reshape(-1, 5)  # [x, y, z, intensity, ring]
    
    return points_reshaped


def visualize_point_cloud_matplotlib(points: np.ndarray, title: str = "Point Cloud", 
                                   max_points: int = 10000, save_path: Optional[str] = None):
    """
    Visualize point cloud using matplotlib.
    
    Args:
        points: Point cloud array with shape (N, 5) [x, y, z, intensity, ring]
        title: Plot title
        max_points: Maximum number of points to plot (for performance)
        save_path: Path to save the plot
    """
    # Subsample points if too many
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_subset = points[indices]
    else:
        points_subset = points
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by intensity
    intensities = points_subset[:, 3]
    scatter = ax.scatter(points_subset[:, 0], points_subset[:, 1], points_subset[:, 2], 
                        c=intensities, cmap='viridis', s=0.5, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Intensity')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"{title}\n{len(points_subset):,} points (of {len(points):,} total)")
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_point_cloud_open3d(points: np.ndarray, title: str = "Point Cloud", 
                                max_points: int = 50000, save_path: Optional[str] = None):
    """
    Visualize point cloud using Open3D.
    
    Args:
        points: Point cloud array with shape (N, 5) [x, y, z, intensity, ring]
        title: Window title
        max_points: Maximum number of points to visualize (for performance)
        save_path: Path to save the visualization
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, falling back to matplotlib")
        visualize_point_cloud_matplotlib(points, title, max_points, save_path)
        return
    
    # Subsample points if too many
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_subset = points[indices]
    else:
        points_subset = points
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_subset[:, :3])  # x, y, z
    
    # Color points by intensity
    intensities = points_subset[:, 3]
    # Normalize intensities to [0, 1] for coloring
    intensities_normalized = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
    
    # Create colors based on intensity (using a colormap)
    colors = plt.cm.viridis(intensities_normalized)[:, :3]  # RGB values
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    if save_path:
        # Save as image
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=800)
        vis.add_geometry(pcd)
        
        # Set view
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.1)
        
        # Render and save
        vis.run()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        print(f"Visualization saved to: {save_path}")
    else:
        # Interactive visualization
        o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_scene_overview(nusc, scene_token: str, output_dir: str = "lidar_visualizations"):
    """
    Create an overview visualization of a scene with multiple frames.
    
    Args:
        nusc: NuScenes instance
        scene_token: Token of the scene to visualize
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get scene info
    scene = nusc.get('scene', scene_token)
    lidar_tokens = load_sweep_sd(nusc, scene, sensor_channel='LIDAR_TOP')
    
    print(f"Visualizing scene: {scene['name']}")
    print(f"Found {len(lidar_tokens)} LIDAR frames")
    
    # Create overview plot with multiple frames
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
    fig.suptitle(f"Scene Overview: {scene['name']}\n{scene['description']}", fontsize=16)
    
    # Select frames to visualize (first, middle, last, and some in between)
    frame_indices = [0, len(lidar_tokens)//4, len(lidar_tokens)//2, 
                    3*len(lidar_tokens)//4, len(lidar_tokens)-1]
    
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(lidar_tokens):
            continue
            
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        try:
            # Load point cloud
            points = load_point_cloud(nusc, lidar_tokens[frame_idx])
            
            # Subsample for visualization
            if len(points) > 5000:
                indices = np.random.choice(len(points), 5000, replace=False)
                points_subset = points[indices]
            else:
                points_subset = points
            
            # Color by intensity
            intensities = points_subset[:, 3]
            scatter = ax.scatter(points_subset[:, 0], points_subset[:, 1], points_subset[:, 2], 
                               c=intensities, cmap='viridis', s=1, alpha=0.6)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f"Frame {frame_idx}\n{len(points):,} points")
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\nframe {frame_idx}\n{e}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Frame {frame_idx} - Error")
    
    # Hide unused subplots
    for i in range(len(frame_indices), 6):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save overview
    overview_path = os.path.join(output_dir, f"scene_overview_{scene_token}.png")
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scene overview saved to: {overview_path}")
    
    return overview_path


def main():
    """Main function to run point cloud visualization."""
    parser = argparse.ArgumentParser(description="Visualize LIDAR point clouds from nuScenes scenes")
    parser.add_argument("--scene_token", type=str, required=True,
                       help="Scene token to visualize")
    parser.add_argument("--frame_idx", type=int, default=0,
                       help="Frame index to visualize (default: 0)")
    parser.add_argument("--output_dir", type=str, default="lidar_visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--method", type=str, choices=["matplotlib", "open3d"], default="matplotlib",
                       help="Visualization method (default: matplotlib)")
    parser.add_argument("--max_points", type=int, default=10000,
                       help="Maximum number of points to visualize (default: 10000)")
    parser.add_argument("--overview", action="store_true",
                       help="Create scene overview with multiple frames")
    parser.add_argument("--save", action="store_true",
                       help="Save visualization instead of showing interactively")
    
    args = parser.parse_args()
    
    try:
        # Initialize nuScenes
        print("Initializing nuScenes...")
        nusc = NuScenes(version='v1.0-trainval-rain', dataroot='/mnt/nas/Nuscenes', verbose=False)
        
        # Get scene info
        scene = nusc.get('scene', args.scene_token)
        lidar_tokens = load_sweep_sd(nusc, scene, sensor_channel='LIDAR_TOP')
        
        print(f"Scene: {scene['name']}")
        print(f"Description: {scene['description']}")
        print(f"Number of LIDAR frames: {len(lidar_tokens)}")
        
        if args.overview:
            # Create scene overview
            visualize_scene_overview(nusc, args.scene_token, args.output_dir)
        else:
            # Visualize specific frame
            if args.frame_idx >= len(lidar_tokens):
                print(f"Error: Frame index {args.frame_idx} out of range (0-{len(lidar_tokens)-1})")
                return
            
            # Load point cloud
            print(f"Loading frame {args.frame_idx}...")
            points = load_point_cloud(nusc, lidar_tokens[args.frame_idx])
            print(f"Loaded {len(points):,} points")
            
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Visualize
            title = f"Scene: {scene['name']} - Frame {args.frame_idx}"
            
            if args.save:
                save_path = os.path.join(args.output_dir, 
                                       f"lidar_frame_{args.scene_token}_{args.frame_idx}.png")
            else:
                save_path = None
            
            if args.method == "open3d":
                visualize_point_cloud_open3d(points, title, args.max_points, save_path)
            else:
                visualize_point_cloud_matplotlib(points, title, args.max_points, save_path)
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 