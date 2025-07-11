import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

from pylisa.lisa import Lisa
import open3d as o3d
import matplotlib.pyplot as plt
import os


def visualize_pointcloud(points: np.ndarray, title: str = "Point Cloud", point_size: float = 2.0):
    """
    Visualize point cloud data using Open3D.
    
    Args:
        points (np.ndarray): Point cloud array with shape (N, 5) [x, y, z, intensity, ring]
        title (str): Window title for the visualization
        point_size (float): Size of points in the visualization
    """
    print(f"Visualizing {len(points)} points...")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z
    
    # Color points by intensity
    intensities = points[:, 3]
    # Normalize intensities to [0, 1] for coloring
    intensities_normalized = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
    
    # Create colors based on intensity (using a colormap)
    colors = plt.cm.viridis(intensities_normalized)[:, :3]  # RGB values
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    print("Opening Open3D visualization window...")
    print("Press 'q' to close the window")
    
    # Create visualizer with custom point size
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)
    
    # Set point size
    opt = vis.get_render_option()
    opt.point_size = point_size
    
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def visualize_changes(original_points: np.ndarray, augmented_points: np.ndarray, title: str = "Point Cloud Changes", point_size: float = 2.0):
    """
    Visualize which points have changed between original and augmented point clouds.
    
    Args:
        original_points (np.ndarray): Original point cloud (N, 5)
        augmented_points (np.ndarray): Augmented point cloud (N, 5)
        title (str): Window title
        point_size (float): Size of points
    """
    print(f"Analyzing changes between {len(original_points)} points...")
    
    # Calculate changes
    position_changes = np.linalg.norm(original_points[:, :3] - augmented_points[:, :3], axis=1)
    intensity_changes = np.abs(original_points[:, 3] - augmented_points[:, 3])
    
    # Identify changed points
    position_threshold = 0.1  # meters
    intensity_threshold = 0.01
    
    position_changed = position_changes > position_threshold
    intensity_changed = intensity_changes > intensity_threshold
    any_changed = position_changed | intensity_changed
    
    print(f"Points with position changes > {position_threshold}m: {np.sum(position_changed)}")
    print(f"Points with intensity changes > {intensity_threshold}: {np.sum(intensity_changed)}")
    print(f"Total points changed: {np.sum(any_changed)}")
    
    # Create visualization with color coding
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(augmented_points[:, :3])
    
    # Color coding: unchanged=blue, position_changed=red, intensity_changed=green, both=purple
    colors = np.zeros((len(augmented_points), 3))
    colors[~any_changed] = [0, 0, 1]  # Blue for unchanged
    colors[position_changed & ~intensity_changed] = [1, 0, 0]  # Red for position only
    colors[~position_changed & intensity_changed] = [0, 1, 0]  # Green for intensity only
    colors[position_changed & intensity_changed] = [1, 0, 1]  # Purple for both
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    print("Opening Open3D visualization window...")
    print("Press 'q' to close the window")
    print("Color legend:")
    print("  Blue: Unchanged points")
    print("  Red: Position changed")
    print("  Green: Intensity changed")
    print("  Purple: Both position and intensity changed")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)
    opt = vis.get_render_option()
    opt.point_size = point_size
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
    return {
        'position_changed': position_changed,
        'intensity_changed': intensity_changed,
        'any_changed': any_changed,
        'position_changes': position_changes,
        'intensity_changes': intensity_changes
    }


def test_lisa_parameters(original_points: np.ndarray, test_configs: list):
    """
    Test different LISA parameter configurations and compare their effects.
    
    Args:
        original_points (np.ndarray): Original point cloud
        test_configs (list): List of parameter configurations to test
    """
    print("\n=== Testing Different LISA Parameters ===")
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test {i+1}: {config['name']} ---")
        print(f"Parameters: {config['params']}")
        
        # Create LISA instance with custom parameters
        lisa = Lisa(**config['params'])
        
        # Prepare point cloud for LISA (N, 4 format)
        point_cloud = original_points[:, :4]
        
        # Run augmentation
        aug_pc_rr = lisa.augment(point_cloud, config['rain_rate'])[:, :4].astype(np.float32)
        
        # Add zeros column to match original format
        zeros_col = np.zeros((aug_pc_rr.shape[0], 1), dtype=np.float32)
        augmented_points = np.hstack([aug_pc_rr, zeros_col])
        
        # Analyze changes
        changes = visualize_changes(original_points, augmented_points, 
                                  f"Changes: {config['name']}", point_size=2.0)
        
        # Store results
        result = {
            'config': config,
            'changes': changes,
            'augmented_points': augmented_points,
            'position_changed_pct': 100 * np.sum(changes['position_changed']) / len(original_points),
            'intensity_changed_pct': 100 * np.sum(changes['intensity_changed']) / len(original_points),
            'total_changed_pct': 100 * np.sum(changes['any_changed']) / len(original_points)
        }
        results.append(result)
        
        print(f"Position changed: {result['position_changed_pct']:.2f}%")
        print(f"Intensity changed: {result['intensity_changed_pct']:.2f}%")
        print(f"Total changed: {result['total_changed_pct']:.2f}%")
    
    # Print summary
    print(f"\n=== Parameter Test Summary ===")
    for result in results:
        config = result['config']
        print(f"{config['name']:20s} | Pos: {result['position_changed_pct']:5.2f}% | Int: {result['intensity_changed_pct']:5.2f}% | Total: {result['total_changed_pct']:5.2f}%")
    
    return results


def aug_pcd_rain_snow(weather, rain_drop, bin_file):
    lisa = Lisa(atm_model=weather)
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)[:, :4]
    aug_pc_rr = lisa.augment(point_cloud, rain_drop)[:, :4].astype(np.float32)
    
    # Add a column of zeros to make shape (N, 5)
    # zeros_col = np.zeros((aug_pc_rr.shape[0], 1), dtype=np.float32)
    # aug_pc_rr = np.hstack([aug_pc_rr, zeros_col])

    return aug_pc_rr


def main():
    """
    Main function to test rain augmentation with visualization.
    """
    # Test bin file path - you can change this to any valid point cloud file
    test_bin_file = "/home/mg/pdnn/pPerf/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489308396618.pcd.bin"
    
    # Check if test file exists
    if not os.path.exists(test_bin_file):
        print(f"Test file not found: {test_bin_file}")
        print("Please provide a valid path to a point cloud bin file.")
        return
    
    print("=== Testing Rain Augmentation with Parameter Tuning ===")
    print(f"Input file: {test_bin_file}")
    
    # Load original point cloud
    print("\n1. Loading original point cloud...")
    original_points = np.fromfile(test_bin_file, dtype=np.float32).reshape(-1, 5)
    print(f"Original point cloud: {len(original_points)} points")
    
    # Visualize original
    visualize_pointcloud(original_points, "Original Point Cloud", point_size=2.0)
    
    # Define test configurations for different parameters
    test_configs = [
        {
            'name': 'Default Snow 35mm/h',
            'params': {'atm_model': 'rain', 'mode': 'strongest'},
            'rain_rate': 35
        },
        {
            'name': 'High Snow Rate 100mm/h',
            'params': {'atm_model': 'rain', 'mode': 'strongest'},
            'rain_rate': 100
        },
        {
            'name': 'High Range Uncertainty',
            'params': {'atm_model': 'rain', 'mode': 'strongest', 'dR': 0.5},
            'rain_rate': 35
        },
        {
            'name': 'Small Droplets',
            'params': {'atm_model': 'rain', 'mode': 'strongest', 'dst': 0.01},
            'rain_rate': 35
        },
        {
            'name': 'Wide Beam',
            'params': {'atm_model': 'rain', 'mode': 'strongest', 'bdiv': 1e-2},
            'rain_rate': 35
        },
        {
            'name': 'Last Mode',
            'params': {'atm_model': 'rain', 'mode': 'last'},
            'rain_rate': 35
        },
        {
            'name': 'Combined High Effect',
            'params': {'atm_model': 'rain', 'mode': 'strongest', 'dR': 0.3, 'dst': 0.02, 'bdiv': 5e-3},
            'rain_rate': 100
        }
    ]
    
    # Test different parameter configurations
    results = test_lisa_parameters(original_points, test_configs)
    
    print(f"\n=== Recommendations for Maximum Effects ===")
    print("1. Increase rain rate (Rr): 100-200 mm/h")
    print("2. Increase range uncertainty (dR): 0.3-1.0")
    print("3. Decrease minimum droplet size (dst): 0.01-0.02")
    print("4. Increase beam divergence (bdiv): 5e-3 to 1e-2")
    print("5. Try 'last' mode instead of 'strongest'")
    print("6. Combine multiple parameters for maximum effect")


if __name__ == "__main__":
    main()