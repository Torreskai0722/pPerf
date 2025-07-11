#!/usr/bin/env python3
"""
Script to create JSON entries for rain-augmented nuScenes data.
This script finds the original scene and sample data, then creates modified entries
for the rain-augmented data with updated LIDAR and CAMERA file paths.
"""

import json
import os
import glob
from pathlib import Path

# Configuration
ORIGINAL_SCENE_TOKEN = "2f0e54af35964a3fb347359836bec035"
RAIN_INTENSITIES = [25, 50, 100]

# File paths
SCENE_JSON_PATH = "data/nuscenes/v1.0-trainval/scene.json"
MINI_SAMPLE_JSON_PATH = "data/nuscenes/v1.0-trainval/sample.json"
MINI_SAMPLE_DATA_JSON_PATH = "data/nuscenes/v1.0-trainval/sample_data.json"

# Output file paths (new files)
OUTPUT_SCENE_JSON_PATH = "data/nuscenes/v1.0-trainval/scene_rain.json"
OUTPUT_SAMPLE_JSON_PATH = "data/nuscenes/v1.0-trainval/sample_rain.json"
OUTPUT_SAMPLE_DATA_JSON_PATH = "data/nuscenes/v1.0-trainval/sample_data_rain.json"

def load_json_file(file_path):
    """Load JSON file and return the data."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def save_json_file(file_path, data):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def find_original_scene(scene_data):
    """Find the original scene entry."""
    for scene in scene_data:
        if scene.get("token") == ORIGINAL_SCENE_TOKEN:
            return scene
    return None

def find_samples_for_scene(sample_data, first_sample_token, last_sample_token):
    """Find all samples for a given scene."""
    samples = []
    in_scene = False
    
    for sample in sample_data:
        if sample.get("token") == first_sample_token:
            in_scene = True
        if in_scene:
            samples.append(sample)
        if sample.get("token") == last_sample_token:
            break
    
    return samples

def find_sample_data_for_samples(sample_data_list, all_sample_data):
    """Find all sample_data entries for given samples."""
    sample_tokens = [sample["token"] for sample in sample_data_list]
    related_sample_data = []
    
    for data in all_sample_data:
        if data.get("sample_token") in sample_tokens:
            related_sample_data.append(data)
    
    return related_sample_data

def create_rain_scene_entry(original_scene, rain_intensity):
    """Create a modified scene entry for rain-augmented data."""
    rain_scene = original_scene.copy()
    rain_scene["token"] = f"{original_scene['token']}_rainrate{rain_intensity}"
    rain_scene["name"] = f"{original_scene['name']}_rainrate{rain_intensity}"
    rain_scene["description"] = f"{original_scene['description']} (Rain intensity: {rain_intensity}mm/h)"
    
    # Update first and last sample tokens
    rain_scene["first_sample_token"] = f"{original_scene['first_sample_token']}_rainrate{rain_intensity}"
    rain_scene["last_sample_token"] = f"{original_scene['last_sample_token']}_rainrate{rain_intensity}"
    
    return rain_scene

def create_rain_sample_entry(original_sample, rain_intensity):
    """Create a modified sample entry for rain-augmented data."""
    rain_sample = original_sample.copy()
    rain_sample["token"] = f"{original_sample['token']}_rainrate{rain_intensity}"
    rain_sample["scene_token"] = f"{original_sample['scene_token']}_rainrate{rain_intensity}"
    
    # Update prev/next references
    if original_sample.get("prev"):
        rain_sample["prev"] = f"{original_sample['prev']}_rainrate{rain_intensity}"
    if original_sample.get("next"):
        rain_sample["next"] = f"{original_sample['next']}_rainrate{rain_intensity}"
    
    return rain_sample

def create_rain_sample_data_entry(original_data, rain_intensity):
    """Create a modified sample_data entry for rain-augmented data."""
    rain_data = original_data.copy()
    rain_data["token"] = f"{original_data['token']}_rainrate{rain_intensity}"
    rain_data["sample_token"] = f"{original_data['sample_token']}_rainrate{rain_intensity}"
    
    # Keep original ego_pose_token and calibrated_sensor_token unchanged
    # rain_data["ego_pose_token"] = original_data["ego_pose_token"]  # Already copied
    # rain_data["calibrated_sensor_token"] = original_data["calibrated_sensor_token"]  # Already copied
    
    # Update file path for LIDAR_TOP and CAM_FRONT data
    filename = original_data.get("filename", "")
    
    if "LIDAR_TOP" in filename:
        if filename.startswith("samples/"):
            # Replace the path prefix but keep the original filename
            rain_data["filename"] = filename.replace("samples/", f"rain/rainy_{rain_intensity}/samples/")
        elif filename.startswith("sweeps/"):
            # Replace the path prefix but keep the original filename
            rain_data["filename"] = filename.replace("sweeps/", f"rain/rainy_{rain_intensity}/sweeps/")
    
    elif "/CAM_FRONT/" in filename:
        if filename.startswith("samples/"):
            # Replace the path prefix but keep the original filename
            rain_data["filename"] = filename.replace("samples/", f"rain/rainy_{rain_intensity}/samples/")
        elif filename.startswith("sweeps/"):
            # Replace the path prefix but keep the original filename
            rain_data["filename"] = filename.replace("sweeps/", f"rain/rainy_{rain_intensity}/sweeps/")
    
    # Update prev/next references
    if original_data.get("prev"):
        rain_data["prev"] = f"{original_data['prev']}_rainrate{rain_intensity}"
    if original_data.get("next"):
        rain_data["next"] = f"{original_data['next']}_rainrate{rain_intensity}"
    
    return rain_data

def main():
    print("Loading original data...")
    
    # Load original data
    scene_data = load_json_file(SCENE_JSON_PATH)
    sample_data = load_json_file(MINI_SAMPLE_JSON_PATH)
    sample_data_list = load_json_file(MINI_SAMPLE_DATA_JSON_PATH)
    
    if not scene_data:
        print("Error: Could not load scene data")
        return
    
    # Find original scene
    original_scene = find_original_scene(scene_data)
    if not original_scene:
        print(f"Error: Could not find scene with token {ORIGINAL_SCENE_TOKEN}")
        return
    
    print(f"Found original scene: {original_scene['name']}")
    print(f"First sample token: {original_scene['first_sample_token']}")
    print(f"Last sample token: {original_scene['last_sample_token']}")
    
    # Find samples for this scene
    samples = find_samples_for_scene(sample_data, 
                                   original_scene['first_sample_token'], 
                                   original_scene['last_sample_token'])
    
    if not samples:
        print("Warning: Could not find samples for this scene in trainval dataset")
        print("This might be because the scene is not in the trainval dataset")
        print("Creating template entries based on file structure...")
        
        # Create template entries based on the file structure
        create_template_entries(original_scene)
        return
    
    print(f"Found {len(samples)} samples")
    
    # Find sample_data for these samples
    related_sample_data = find_sample_data_for_samples(samples, sample_data_list)
    print(f"Found {len(related_sample_data)} sample_data entries")
    
    # Create rain-augmented entries and append to original files
    for rain_intensity in RAIN_INTENSITIES:
        print(f"\nProcessing rain intensity {rain_intensity}...")
        
        # Create rain scene entry
        rain_scene = create_rain_scene_entry(original_scene, rain_intensity)
        
        # Create rain sample entries
        rain_samples = [create_rain_sample_entry(sample, rain_intensity) for sample in samples]
        
        # Create rain sample_data entries
        rain_sample_data = [create_rain_sample_data_entry(data, rain_intensity) for data in related_sample_data]
        
        # Append to original files (modify in place)
        scene_data.append(rain_scene)
        sample_data.extend(rain_samples)
        sample_data_list.extend(rain_sample_data)
        
        print(f"Appended rain-augmented data for intensity {rain_intensity}")
    
    # Save updated files
    print("\nSaving updated JSON files...")
    save_json_file(SCENE_JSON_PATH, scene_data)
    save_json_file(MINI_SAMPLE_JSON_PATH, sample_data)
    save_json_file(MINI_SAMPLE_DATA_JSON_PATH, sample_data_list)
    print("All files updated successfully!")

def create_template_entries(original_scene):
    """Create template entries when original samples are not found."""
    print("Creating template entries based on file structure...")
    
    # Load original data to append to
    scene_data = load_json_file(SCENE_JSON_PATH)
    sample_data = load_json_file(MINI_SAMPLE_JSON_PATH)
    sample_data_list = load_json_file(MINI_SAMPLE_DATA_JSON_PATH)
    
    for rain_intensity in RAIN_INTENSITIES:
        print(f"\nProcessing rain intensity {rain_intensity}...")
        
        # Create rain scene entry
        rain_scene = create_rain_scene_entry(original_scene, rain_intensity)
        
        # Get all LIDAR and CAMERA files for this rain intensity
        rain_dir = f"data/nuscenes/rain/rainy_{rain_intensity}"
        
        # LIDAR files
        lidar_sample_files = glob.glob(f"{rain_dir}/samples/LIDAR_TOP/*.pcd.bin")
        lidar_sweep_files = glob.glob(f"{rain_dir}/sweeps/LIDAR_TOP/*.pcd.bin")
        
        # CAMERA files - look for CAM_FRONT only
        cam_sample_files = []
        cam_sweep_files = []
        
        # Find CAM_FRONT directories only
        samples_dir = f"{rain_dir}/samples"
        sweeps_dir = f"{rain_dir}/sweeps"
        
        if os.path.exists(samples_dir):
            cam_front_dir = f"{samples_dir}/CAM_FRONT"
            if os.path.exists(cam_front_dir):
                cam_sample_files.extend(glob.glob(f"{cam_front_dir}/*.jpg"))
        
        if os.path.exists(sweeps_dir):
            cam_front_dir = f"{sweeps_dir}/CAM_FRONT"
            if os.path.exists(cam_front_dir):
                cam_sweep_files.extend(glob.glob(f"{cam_front_dir}/*.jpg"))
        
        print(f"Found {len(lidar_sample_files)} LIDAR sample files and {len(lidar_sweep_files)} LIDAR sweep files")
        print(f"Found {len(cam_sample_files)} CAM_FRONT sample files and {len(cam_sweep_files)} CAM_FRONT sweep files")
        
        # Sort files by timestamp
        lidar_sample_files.sort()
        lidar_sweep_files.sort()
        cam_sample_files.sort()
        cam_sweep_files.sort()
        
        # Create template sample entries with proper linking
        rain_samples = []
        for i, sample_file in enumerate(lidar_sample_files):
            filename = os.path.basename(sample_file)
            timestamp = filename.split("__")[-1].replace(".pcd.bin", "")
            
            sample_entry = {
                "token": f"sample_{timestamp}_rainrate{rain_intensity}",
                "timestamp": int(timestamp),
                "scene_token": rain_scene["token"],
                "next": f"sample_{lidar_sample_files[i+1].split('__')[-1].replace('.pcd.bin', '')}_rainrate{rain_intensity}" if i < len(lidar_sample_files) - 1 else "",
                "prev": f"sample_{lidar_sample_files[i-1].split('__')[-1].replace('.pcd.bin', '')}_rainrate{rain_intensity}" if i > 0 else ""
            }
            rain_samples.append(sample_entry)
        
        # Create template sample_data entries
        rain_sample_data = []
        
        # Add LIDAR sample data entries
        for i, sample_file in enumerate(lidar_sample_files):
            filename = os.path.basename(sample_file)
            timestamp = filename.split("__")[-1].replace(".pcd.bin", "")
            
            data_entry = {
                "token": f"sample_data_lidar_{timestamp}_rainrate{rain_intensity}",
                "sample_token": f"sample_{timestamp}_rainrate{rain_intensity}",
                "ego_pose_token": "ego_pose_original",  # Use original token
                "calibrated_sensor_token": "calibrated_sensor_LIDAR_TOP_original",  # Use original token
                "filename": f"rain/rainy_{rain_intensity}/samples/LIDAR_TOP/{filename}",
                "fileformat": "pcd.bin",
                "width": 0,
                "height": 0,
                "timestamp": int(timestamp),
                "is_key_frame": True,
                "next": f"sample_data_lidar_{lidar_sample_files[i+1].split('__')[-1].replace('.pcd.bin', '')}_rainrate{rain_intensity}" if i < len(lidar_sample_files) - 1 else "",
                "prev": f"sample_data_lidar_{lidar_sample_files[i-1].split('__')[-1].replace('.pcd.bin', '')}_rainrate{rain_intensity}" if i > 0 else ""
            }
            rain_sample_data.append(data_entry)
        
        # Add LIDAR sweep data entries
        for i, sweep_file in enumerate(lidar_sweep_files):
            filename = os.path.basename(sweep_file)
            timestamp = filename.split("__")[-1].replace(".pcd.bin", "")
            
            data_entry = {
                "token": f"sample_data_lidar_sweep_{timestamp}_rainrate{rain_intensity}",
                "sample_token": f"sample_{timestamp}_rainrate{rain_intensity}",
                "ego_pose_token": "ego_pose_original",  # Use original token
                "calibrated_sensor_token": "calibrated_sensor_LIDAR_TOP_original",  # Use original token
                "filename": f"rain/rainy_{rain_intensity}/samples/LIDAR_TOP/{filename}",
                "fileformat": "pcd.bin",
                "width": 0,
                "height": 0,
                "timestamp": int(timestamp),
                "is_key_frame": False,
                "next": f"sample_data_lidar_sweep_{lidar_sweep_files[i+1].split('__')[-1].replace('.pcd.bin', '')}_rainrate{rain_intensity}" if i < len(lidar_sweep_files) - 1 else "",
                "prev": f"sample_data_lidar_sweep_{lidar_sweep_files[i-1].split('__')[-1].replace('.pcd.bin', '')}_rainrate{rain_intensity}" if i > 0 else ""
            }
            rain_sample_data.append(data_entry)
        
        # Add CAM_FRONT sample data entries
        for i, sample_file in enumerate(cam_sample_files):
            filename = os.path.basename(sample_file)
            timestamp = filename.split("__")[-1].replace(".jpg", "")
            
            data_entry = {
                "token": f"sample_data_cam_{timestamp}_rainrate{rain_intensity}",
                "sample_token": f"sample_{timestamp}_rainrate{rain_intensity}",
                "ego_pose_token": "ego_pose_original",  # Use original token
                "calibrated_sensor_token": "calibrated_sensor_CAM_FRONT_original",  # Use original token
                "filename": f"rain/rainy_{rain_intensity}/samples/CAM_FRONT/{filename}",
                "fileformat": "jpg",
                "width": 1600,
                "height": 900,
                "timestamp": int(timestamp),
                "is_key_frame": True,
                "next": f"sample_data_cam_{cam_sample_files[i+1].split('__')[-1].replace('.jpg', '')}_rainrate{rain_intensity}" if i < len(cam_sample_files) - 1 else "",
                "prev": f"sample_data_cam_{cam_sample_files[i-1].split('__')[-1].replace('.jpg', '')}_rainrate{rain_intensity}" if i > 0 else ""
            }
            rain_sample_data.append(data_entry)
        
        # Add CAM_FRONT sweep data entries
        for i, sweep_file in enumerate(cam_sweep_files):
            filename = os.path.basename(sweep_file)
            timestamp = filename.split("__")[-1].replace(".jpg", "")
            
            data_entry = {
                "token": f"sample_data_cam_sweep_{timestamp}_rainrate{rain_intensity}",
                "sample_token": f"sample_{timestamp}_rainrate{rain_intensity}",
                "ego_pose_token": "ego_pose_original",  # Use original token
                "calibrated_sensor_token": "calibrated_sensor_CAM_FRONT_original",  # Use original token
                "filename": f"rain/rainy_{rain_intensity}/samples/CAM_FRONT/{filename}",
                "fileformat": "jpg",
                "width": 1600,
                "height": 900,
                "timestamp": int(timestamp),
                "is_key_frame": False,
                "next": f"sample_data_cam_sweep_{cam_sweep_files[i+1].split('__')[-1].replace('.jpg', '')}_rainrate{rain_intensity}" if i < len(cam_sweep_files) - 1 else "",
                "prev": f"sample_data_cam_sweep_{cam_sweep_files[i-1].split('__')[-1].replace('.jpg', '')}_rainrate{rain_intensity}" if i > 0 else ""
            }
            rain_sample_data.append(data_entry)
        
        # Append to original files (modify in place)
        scene_data.append(rain_scene)
        sample_data.extend(rain_samples)
        sample_data_list.extend(rain_sample_data)
        
        print(f"Appended template rain-augmented data for intensity {rain_intensity}")
    
    # Save updated files
    print("\nSaving updated JSON files...")
    save_json_file(SCENE_JSON_PATH, scene_data)
    save_json_file(MINI_SAMPLE_JSON_PATH, sample_data)
    save_json_file(MINI_SAMPLE_DATA_JSON_PATH, sample_data_list)
    print("All files updated successfully!")

if __name__ == "__main__":
    main() 