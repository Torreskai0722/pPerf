from pylisa.lisa import Lisa
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import os
import json
from p_perf.utils import load_sweep_sd
from p_perf.nuscenes_instance import get_nuscenes_instance
import imgaug.augmenters as iaa
import cv2
from PIL import Image






def aug_pcd_rain_snow(weather, rain_drop, bin_file, output_file):
    lisa = Lisa(atm_model=weather)
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)[:, :4]
    aug_pc_rr = lisa.augment(point_cloud, rain_drop)[:, :4].astype(np.float32)
    
    # Add a column of zeros to make shape (N, 5)
    zeros_col = np.zeros((aug_pc_rr.shape[0], 1), dtype=np.float32)
    aug_pc_rr = np.hstack([aug_pc_rr, zeros_col])

    aug_pc_rr.tofile(output_file)
    print(f"WEATHER: {weather}      RR: {rain_drop}    OUTPUT: {'/'.join(output_file.split('/')[-3:])}")


def aug_image_rain(rain_rate, image_file, output_file):
    """
    Augment image with rain effects using imgaug
    """
    # Load image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not load image {image_file}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width = image_rgb.shape[:2]
    
    # Create deterministic seed based on filename and rain rate
    # Extract filename without path and extension
    filename = os.path.basename(image_file)
    filename_no_ext = os.path.splitext(filename)[0]
    
    # Create a hash-based seed for deterministic results
    import hashlib
    seed_string = f"{filename_no_ext}_{rain_rate}"
    seed = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
    
    # Set random seed for deterministic rain
    np.random.seed(seed)
    
    # Create rain augmentation based on rain rate
    if rain_rate == 25:
        rain_speed = 0.1
        drop_size = (0.05, 0.10)  # Smaller drops for light rain
    elif rain_rate == 50:
        rain_speed = 0.15
        drop_size = (0.08, 0.15)  # Medium drops for moderate rain
    else:  # rain_rate == 100
        rain_speed = 0.2
        drop_size = (0.12, 0.25)  # Larger drops for heavy rain
    
    # Scale drop size for larger images (900x1600)
    # For larger images, we need bigger drops to maintain visibility
    if width > 1000 or height > 1000:
        drop_size = (drop_size[0] * 1.5, drop_size[1] * 1.5)
    
    # Create rain augmenter with deterministic seed
    rain_aug = iaa.Rain(
        drop_size=drop_size,
        speed=(rain_speed, rain_speed * 1.5),
        angle=(-30, 30),
        blur_factor=0.1,
        seed=seed  # Set seed for deterministic behavior
    )
    
    # Apply rain augmentation
    augmented_image = rain_aug.augment_image(image_rgb)
    
    # Convert back to BGR and save
    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, augmented_image_bgr)
    
    print(f"IMAGE RAIN: RR: {rain_rate}    OUTPUT: {'/'.join(output_file.split('/')[-3:])}")


if __name__=='__main__':
    nusc = get_nuscenes_instance()
    scene_token = '2f0e54af35964a3fb347359836bec035'

    scene = nusc.get('scene', scene_token)  
    lidar_sd_token = load_sweep_sd(nusc, scene, sensor_channel='LIDAR_TOP')
    image_sd_token = load_sweep_sd(nusc, scene, sensor_channel='CAM_FRONT')
    lidar_sd_token_filenames = [nusc.get('sample_data', sd_token)['filename'] for sd_token in lidar_sd_token]
    image_sd_token_filenames = [nusc.get('sample_data', sd_token)['filename'] for sd_token in image_sd_token]

    data_dir = '/mnt/nas/Nuscenes'
    weather = 'rain'
    rain_rate = [25, 50, 100]

    output_dir = f'/mmdetection3d_ros2/data/nuscenes/weather/{weather}'

    count = 0
    # for filename in lidar_sd_token_filenames:
    #     for rr in rain_rate:
    #             lidar_output_sample_dir = f'{output_dir}/rainy_{rr}/samples/LIDAR_TOP'
    #             lidar_output_sweep_dir = f'{output_dir}/rainy_{rr}/sweeps/LIDAR_TOP'
    #             os.makedirs(lidar_output_sample_dir, exist_ok=True)
    #             os.makedirs(lidar_output_sweep_dir, exist_ok=True)

    #             file_name = f'{data_dir}/{filename}'
    #             print(file_name)
    #             output_file = f'{output_dir}/rainy_{rr}/{filename}'

    #             aug_pcd_rain_snow(weather, rr, file_name, output_file)
    
    for filename in image_sd_token_filenames:
        for rr in rain_rate:
            image_output_sample_dir = f'{output_dir}/rainy_{rr}/samples/CAM_FRONT'
            image_output_sweep_dir = f'{output_dir}/rainy_{rr}/sweeps/CAM_FRONT'
            os.makedirs(image_output_sample_dir, exist_ok=True)
            os.makedirs(image_output_sweep_dir, exist_ok=True)

            file_name = f'{data_dir}/{filename}'
            print(file_name)
            output_file = f'{output_dir}/rainy_{rr}/{filename}'

            aug_image_rain(rr, file_name, output_file)
    
    print("Weather augmentation completed!")
    



