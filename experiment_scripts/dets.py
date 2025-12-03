#!/usr/bin/env python3

import os
import ast
from itertools import product
import pandas as pd
from nuscenes.nuscenes import NuScenes
import sys
from pathlib import Path

# Add the p_perf package to the path
sys.path.append('/mmdetection3d_ros2/perf_ws/src/p_perf')

from p_perf.pPerf_manager import pPerfConfigManager
from p_perf.config.constant import (
    image_models, 
    lidar_models, 
    scenes,
    get_abbreviated_name,
    get_abbreviated_scene,
    get_full_model_name,
    get_full_scene_token
)
from p_perf.utils import (
    ensure_bags_exist,
    ExperimentCSVManager,
    create_failure_log,
    ExperimentRunner
)

# ============================================================================
# CONFIGURATION
# ============================================================================

bag_dir = "/mmdetection3d_ros2/data/"
dataset_version = "v1.0-trainval"
nuscenes_data_dir = Path("/mnt/nas/Nuscenes")

OVERWRITE = True   
CONTINUE = False  # Set to True to continue from existing mapping file
LOGGING_DELAY = False

# Parameter sweep setup
depths = [-1]
image_queues = [1]
lidar_queues = [1]
publishing_rate = [10]
output_name = "test"

# ============================================================================
# SETUP
# ============================================================================

output_base = f"/mmdetection3d_ros2/outputs/{output_name}"
os.makedirs(output_base, exist_ok=True)

# Create failure log
failure_log = create_failure_log(output_base)

# Create base configuration
base_config_file = os.path.join(output_base, "base_config.yaml")
manager = pPerfConfigManager(base_config_file)
manager.create_base_config(
    num_det_inferencers=2,  # LiDAR + Image detection
    num_seg_inferencers=0,  # Segmentation
    use_sim_time=True,
    logging_delay=LOGGING_DELAY
)

# ============================================================================
# EXPERIMENT MAPPING
# ============================================================================

# Create combinations for full stack testing
combinations = list(product(depths, image_models, lidar_models, scenes, image_queues, lidar_queues, publishing_rate))

# Define CSV columns
csv_columns = ["run_index", "scene", "depth", "image_model", "lidar_model", "image_queue", "lidar_queue", "publishing_rate", "status", "start_time"]

# Row formatter for CSV
def format_csv_row(index, combo):
    """Format combination tuple into CSV row with abbreviated names."""
    depth, img_model, lidar_model, scene, img_q, lidar_q, pub_rate = combo
    abbreviated_scene = get_abbreviated_scene(scene)
    abbreviated_img_model = get_abbreviated_name(img_model)
    abbreviated_lidar_model = (get_abbreviated_name(lidar_model[0]), lidar_model[1])
    return [index, abbreviated_scene, depth, abbreviated_img_model, abbreviated_lidar_model, img_q, lidar_q, pub_rate, "pending", ""]

# Create or load mapping file
mapping_file = os.path.join(output_base, "full_stack_mapping.csv")
csv_manager = ExperimentCSVManager(mapping_file, csv_columns)
df = csv_manager.create_mapping(combinations, format_csv_row, OVERWRITE, CONTINUE)

# ============================================================================
# ROSBAG PREPARATION
# ============================================================================

available_scenes = ensure_bags_exist(
    scenes=scenes,
    bag_dir=bag_dir,
    nuscenes_data_dir=nuscenes_data_dir,
    dataset_version=dataset_version,
    failure_log=failure_log
)

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

# Create nsys base command
nsys_base = ExperimentRunner.create_nsys_base_cmd()

# Create experiment runner
runner = ExperimentRunner(
    output_base=output_base,
    failure_log=failure_log,
    nsys_base=nsys_base,
    timeout=300,
    cleanup_memory=True  # Automatically clean memory after each run
)

def update_config_for_run(row, run_index):
    """Update configuration for a specific experiment run."""
    # Read abbreviated names from CSV and convert back to full names for config
    abbreviated_img_model = row["image_model"]
    abbreviated_lidar_model = row["lidar_model"]
    abbreviated_scene = row["scene"]
    img_q = row["image_queue"]
    lidar_q = row["lidar_queue"]
    pub_rate = row["publishing_rate"]
    
    # Convert abbreviated names to full names for configuration
    img_model = get_full_model_name(abbreviated_img_model)
    lidar_model_abbreviated_tuple = ast.literal_eval(abbreviated_lidar_model)
    lidar_model_name = get_full_model_name(lidar_model_abbreviated_tuple[0])
    lidar_mode = lidar_model_abbreviated_tuple[1]
    scene = get_full_scene_token(abbreviated_scene)
    
    # Update sensor replayer configuration
    manager.update_sensor_replayer(
        bag_dir=bag_dir,
        scene=scene,
        publishing_rate=pub_rate,
        index=run_index,
    )
    
    # Define inferencer configurations to update
    inferencer_configs = [
        {
            'section': 'det_inferencers',
            'inferencer_index': 0,
            'mode': "lidar",
            'model_name': lidar_model_name,
            'data_dir': output_base
        },
        {
            'section': 'det_inferencers', 
            'inferencer_index': 1,
            'mode': "image",
            'model_name': img_model,
            'data_dir': output_base
        }
    ]
    
    # Update all inferencers
    for config in inferencer_configs:
        manager.update_inferencer(
            config['section'], 
            config['inferencer_index'],
            mode=config['mode'],
            model_name=config['model_name'],
            index=run_index,
            data_dir=config['data_dir'],
            lidar_model_mode=lidar_mode,
            lidar_queue=lidar_q,
            image_queue=img_q
        )
    
    # Save the updated configuration
    manager.save_config(base_config_file)
    print(f"📋 Configuration for run {run_index}:")
    manager.list_inferencers()
    
    return base_config_file

def build_launch_cmd(config_file):
    """Build ROS2 launch command."""
    return ExperimentRunner.build_ros2_launch_cmd(
        package="p_perf",
        launch_file="full_stack.launch.py",
        config_file=config_file
    )

# Run experiments
# max_runs: -1 = run all, >0 = run that many experiments
df = runner.run_experiments(
    df=df,
    csv_path=mapping_file,
    config_updater=update_config_for_run,
    launch_cmd_builder=build_launch_cmd,
    max_runs=-1  # Change to positive number to limit runs (e.g., 1 for testing)
)

# ============================================================================
# POST-PROCESSING
# ============================================================================

# Load NuScenes for post-processing
nusc = NuScenes(version=dataset_version, dataroot=str(nuscenes_data_dir))

def parse_row_for_postprocessing(row):
    """Parse row to extract scene and other information for post-processing."""
    abbreviated_scene = row["scene"]
    scene = get_full_scene_token(abbreviated_scene)
    return {'scene': scene}

# Post-process experiments
# max_runs: -1 = process all, >0 = process that many experiments
df = runner.post_process_experiments(
    df=df,
    csv_path=mapping_file,
    nusc=nusc,
    row_parser=parse_row_for_postprocessing,
    publish_mode="bag",
    cleanup_json=False,  # Set to True to delete JSON files after processing
    max_runs=-1  # Change to positive number to limit post-processing (e.g., 1 for testing)
)

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "="*60)
print("ALL RUNS COMPLETED")
print("="*60)
print(f"Results saved to: {output_base}")
print(f"Mapping file: {mapping_file}")
print(f"Failure log: {failure_log}")
