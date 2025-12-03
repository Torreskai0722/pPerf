#!/usr/bin/env python3
"""
Utilities for rosbag management and validation.
"""

import os
import json
from pathlib import Path
from glob import glob
from typing import List


def get_scene_name_from_token(scene_token: str, dataset_version: str = "v1.0-trainval") -> str:
    """
    Get scene name from scene token without loading entire NuScenes dataset.
    
    Args:
        scene_token: Scene token to look up
        dataset_version: NuScenes dataset version
        
    Returns:
        str: Scene name
        
    Raises:
        ValueError: If scene token not found
    """
    scene_file = f'/mnt/nas/Nuscenes/{dataset_version}/scene.json'
    
    try:
        with open(scene_file, 'r') as f:
            scenes = json.load(f)
            for scene in scenes:
                if scene['token'] == scene_token:
                    return scene['name']
    except FileNotFoundError:
        raise FileNotFoundError(f"Scene file not found: {scene_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in scene file: {e}")
    
    raise ValueError(f"Scene token {scene_token} not found in {scene_file}")


def check_bag_exists(bag_dir: str, scene_token: str, dataset_version: str = "v1.0-trainval") -> bool:
    """
    Check if a rosbag exists for a given scene token.
    
    Args:
        bag_dir: Directory to search for bags
        scene_token: Scene token to check
        dataset_version: NuScenes dataset version
        
    Returns:
        bool: True if bag exists, False otherwise
    """
    try:
        scene_name = get_scene_name_from_token(scene_token, dataset_version)
    except (ValueError, FileNotFoundError) as e:
        print(f"Warning: Could not get scene name: {e}")
        return False
    
    # Search for .mcap files matching the scene
    all_bags = glob(os.path.join(bag_dir, '**', "*.mcap"), recursive=True)
    matching_bags = [f for f in all_bags if scene_name in os.path.basename(f)]
    
    if matching_bags:
        print(f"✓ Found {len(matching_bags)} bag(s) for scene '{scene_name}' ({scene_token})")
        return True
    else:
        print(f"✗ No bags found for scene '{scene_name}' ({scene_token})")
        return False


def ensure_bags_exist(
    scenes: List[str],
    bag_dir: str,
    nuscenes_data_dir: Path,
    dataset_version: str = "v1.0-trainval",
    failure_log: str = None
) -> List[str]:
    """
    Check and create rosbags for all scenes that don't have bags yet.
    
    Args:
        scenes: List of scene tokens
        bag_dir: Directory where bags should be stored
        nuscenes_data_dir: Path to NuScenes dataset
        dataset_version: NuScenes dataset version
        failure_log: Optional path to failure log file
        
    Returns:
        List[str]: List of scene tokens that have bags available (successfully created or already existed)
    """
    from p_perf.pre_process.rosbag_creation import create_rosbag_from_scene
    
    print("\n" + "="*60)
    print("ROSBAG PREPARATION PHASE")
    print("="*60)
    
    bag_output_dir = Path(bag_dir)
    available_scenes = []
    
    # Check which scenes need bag creation
    scenes_to_create = []
    for scene_token in scenes:
        if not check_bag_exists(bag_dir, scene_token, dataset_version):
            scenes_to_create.append(scene_token)
        else:
            print(f"✓ Bag already exists for scene {scene_token}, skipping creation")
            available_scenes.append(scene_token)
    
    # Only load NuScenes if we need to create bags
    if scenes_to_create:
        print(f"\nLoading NuScenes dataset for {len(scenes_to_create)} scene(s)...")
        from nuscenes.nuscenes import NuScenes
        from nuscenes.can_bus.can_bus_api import NuScenesCanBus
        
        nusc = NuScenes(version=dataset_version, dataroot=str(nuscenes_data_dir), verbose=True)
        nusc_can = NuScenesCanBus(dataroot=str(nuscenes_data_dir))
        
        # Create bags for scenes that need them
        for scene_token in scenes_to_create:
            print(f"\n🔧 Creating rosbag for scene {scene_token}...")
            try:
                create_rosbag_from_scene(
                    data_dir=nuscenes_data_dir,
                    output_dir=bag_output_dir,
                    scene_token=scene_token,
                    dataset_version=dataset_version,
                    nusc=nusc,  # Pass pre-loaded instance
                    nusc_can=nusc_can  # Pass pre-loaded instance
                )
                print(f"✓ Successfully created bag for scene {scene_token}")
                available_scenes.append(scene_token)
            except Exception as e:
                error_msg = f"Failed to create bag for scene {scene_token}: {str(e)}"
                print(f"✗ {error_msg}")
                if failure_log:
                    with open(failure_log, "a") as flog:
                        flog.write(f"{error_msg}\n")
                # Continue to next scene instead of stopping
                continue
    
    print(f"\n✓ {len(available_scenes)}/{len(scenes)} rosbags available")
    return available_scenes

