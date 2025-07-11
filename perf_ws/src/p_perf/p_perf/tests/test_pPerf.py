#!/usr/bin/env python3

import os
import sys
import cv2
import matplotlib
# Set matplotlib to use non-interactive backend to avoid Qt issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Set environment variables to avoid Qt/OpenCV display issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':0'

# Add the parent directory to the path to import p_perf modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p_perf.nuscenes_instance import get_nuscenes_instance
from p_perf.utils import load_sweep_sd


def visualize_first_image_frame(scene_token, camera_channel="CAM_FRONT", save_path=None):
    """
    Visualize the first image frame of a given scene.
    
    Args:
        scene_token (str): The scene token to visualize
        camera_channel (str): Camera channel to use (default: "CAM_FRONT")
        save_path (str, optional): Path to save the visualization image
    
    Returns:
        str: Path to the saved image or None if failed
    """
    try:
        # Get NuScenes instance
        nusc = get_nuscenes_instance()
        
        # Get the scene
        scene = nusc.get('scene', scene_token)
        print(f"Scene: {scene['name']} - {scene['description']}")
        
        # Get all sample data tokens for the camera channel
        sd_tokens = load_sweep_sd(nusc, scene, sensor_channel=camera_channel)
        
        if not sd_tokens:
            print(f"No sample data found for camera {camera_channel}")
            return None
        
        # Get the first frame
        first_sd_token = sd_tokens[0]
        first_sd = nusc.get('sample_data', first_sd_token)
        
        # Get the image path
        image_path = os.path.join(nusc.dataroot, first_sd['filename'])
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        
        # Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image from: {image_path}")
            return None
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(f"First Frame - {scene['name']}\n{scene['description']}\nCamera: {camera_channel}")
        plt.axis('off')
        
        # Save or show the image
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def visualize_first_20_frames(scene_token, camera_channel="CAM_FRONT", interval=2, save_dir=None):
    """
    Visualize the first 20 image frames of a given scene with specified interval.
    
    Args:
        scene_token (str): The scene token to visualize
        camera_channel (str): Camera channel to use (default: "CAM_FRONT")
        interval (int): Interval between frames in seconds (default: 2)
        save_dir (str, optional): Directory to save frames as images
    """
    try:
        # Get NuScenes instance
        nusc = get_nuscenes_instance()
        
        # Get the scene
        scene = nusc.get('scene', scene_token)
        print(f"Scene: {scene['name']} - {scene['description']}")
        
        # Get all sample data tokens for the camera channel
        sd_tokens = load_sweep_sd(nusc, scene, sensor_channel=camera_channel)
        
        if not sd_tokens:
            print(f"No sample data found for camera {camera_channel}")
            return
        
        # Limit to first 20 frames
        frames_to_show = min(20, len(sd_tokens))
        print(f"Processing first {frames_to_show} frames...")
        
        # Create save directory if specified
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            scene_save_dir = save_dir / f"{scene['name']}_{camera_channel}"
            scene_save_dir.mkdir(exist_ok=True)
        
        for i in range(frames_to_show):
            # Get the frame data
            sd_token = sd_tokens[i]
            sd = nusc.get('sample_data', sd_token)
            
            # Get the image path
            image_path = os.path.join(nusc.dataroot, sd['filename'])
            
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue
            
            # Load the image
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to load image from: {image_path}")
                continue
            
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.title(f"Frame {i+1}/{frames_to_show} - {scene['name']}\n{scene['description']}\nCamera: {camera_channel}\nTimestamp: {sd['timestamp']}")
            plt.axis('off')
            
            # Save frame if directory is specified
            if save_dir:
                frame_save_path = scene_save_dir / f"frame_{i+1:02d}.png"
                plt.savefig(frame_save_path, dpi=300, bbox_inches='tight')
                print(f"Saved frame {i+1}/{frames_to_show} to {frame_save_path}")
            else:
                # Try to display (may not work in headless environment)
                try:
                    plt.show()
                    time.sleep(interval)
                except:
                    print(f"Display not available, showing frame {i+1}/{frames_to_show} info only")
            
            plt.close()
            
            print(f"Processed frame {i+1}/{frames_to_show}")
        
        if save_dir:
            print(f"All frames saved to: {scene_save_dir}")
        print("Finished processing frames!")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    # Example usage
    scenes = [
    # '2f0e54af35964a3fb347359836bec035',  # ENVIRONMENT: Crowded city (scene 429)
    # '3a1850241080418b88dcee97c7d17ed7',  # ENVIRONMENT: Crowded city (scene 27)
    
    # 'bcb05cffb5814973a6cff4fbdca2b99b',  # ENVIRONMENT: Highspeed highway (scene 194)
    # '74d03e3b673f4a72a3f76e1b0c7840bd',  # ENVIRONMENT: Highspeed highway (scene 612)
    
    # '32185f91e68f4069ab3cdd2f4f1a4ff1',  # ENVIRONMENT: Suburban (scene 709)
    # '93032908bafd484ea10a39b0d5d3fff6',  # ENVIRONMENT: Suburban (scene 143)
    
    '7ca5e90766dc4b7bb6a9b3b0d95c99da',  # WEATHER: Clear (scene 201)
    'bed8426a524d45afab05b19cf02386b2',  # WEATHER: Clear (scene 97)
    
    '5fc7c5dfc56b4971b14005bc53f69908',  # WEATHER: Rainy (scene 389)
    '0ced08ea43754420a23b2fbec667a763',  # WEATHER: Rainy (scene 364)
    
    'b519ec833e23450a8bd3340b67f2516b',  # LIGHTING: Day (scene 314)
    '8edbc31083ab4fb187626e5b3c0411f7',  # LIGHTING: Day (scene 37)
    
    # 'f634de95cc7043b8b38ceaac67d472cf',  # LIGHTING: Night (scene 784)
    # 'b2119bd765a94b0cb705424800307bbf'   # LIGHTING: Night (scene 768)
    ]
    
    # Create output directory for saving frames
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Visualize first 20 frames and save them
    for scene_token in scenes:
        print(f"\n{'='*60}")
        print(f"Processing scene: {scene_token}")
        print(f"{'='*60}")
        visualize_first_20_frames(scene_token, interval=0, save_dir=output_dir)
        print(f"Completed scene: {scene_token}\n")
