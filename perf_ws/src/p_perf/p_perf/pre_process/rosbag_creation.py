#!/usr/bin/env python3
"""
ROS Bag Creation Script using NuScenes to MCAP conversion functionality.

This script provides an interface to create ROS bags (MCAP format) from NuScenes data
by leveraging the existing write_scene_to_mcap function.
"""

import sys
from pathlib import Path

# Add the nuscenes_to_ros2bag directory to Python path
nuscenes_converter_dir = "/mmdetection3d_ros2/nuscenes_to_ros2bag/nuscenes2bag"
sys.path.insert(0, str(nuscenes_converter_dir))

try:
    from convert_to_bag import write_scene
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure that:")
    print("1. NuScenes devkit is installed: pip install nuscenes-devkit")
    print("2. The nuscenes_to_ros2bag directory contains all required files")
    print("3. All dependencies from convert_to_bag.py are available")
    sys.exit(1)


def create_rosbag_from_scene(
    data_dir: Path,
    output_dir: Path,
    scene_name: str,
    dataset_version: str = "v1.0-trainval-rain"
):
    """
    Create a ROS bag (MCAP format) from a specific NuScenes scene.
    
    Args:
        data_dir: Path to NuScenes data directory
        output_dir: Path to write MCAP files
        scene_name: Name of the scene to convert (e.g., "scene-0001")
        dataset_version: NuScenes dataset version (e.g., "v1.0-mini", "v1.0-trainval")
    """
    print(f"Loading NuScenes dataset: {dataset_version}")
    nusc = NuScenes(version=dataset_version, dataroot=str(data_dir), verbose=True)
    nusc_can = NuScenesCanBus(dataroot=str(data_dir))
    
    # Find the scene by name
    target_scene = None
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            target_scene = scene
            break
    
    if not target_scene:
        raise ValueError(f"Scene with name '{scene_name}' not found")
    
    # Create output filename (ROS bag directory)
    bag_dirname = f"NuScenes-{dataset_version}-{target_scene['name']}"
    output_path = output_dir / bag_dirname
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting scene '{target_scene['name']}' to ROS bag format...")
    print(f"Output directory: {output_path}")
    
    # Call the conversion function
    write_scene(nusc, nusc_can, target_scene, output_path)
    
    print(f"Successfully created ROS bag: {output_path}")



def main():
    """
    Simple main function for direct usage.
    Modify the values below to use with your specific scene.
    """
    # Configure these values for your use case
    data_dir = Path("/mnt/nas/Nuscenes")  # Update this path
    output_dir = Path("/mmdetection3d_ros2/data/bag")
    scene_names = ["scene-0539_rainrate65", "scene-0539_rainrate90"]
    dataset_version = "v1.0-trainval-rain"
    
    print(f"Creating ROS bag for scene: {scene_names}")
    
    try:
        for scene_name in scene_names:
            create_rosbag_from_scene(
                data_dir=data_dir,
                output_dir=output_dir,
                scene_name=scene_name,
                dataset_version=dataset_version
            )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
