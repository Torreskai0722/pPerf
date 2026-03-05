#!/usr/bin/env python3

import os
import ast
from itertools import product
from pathlib import Path
from nuscenes.nuscenes import NuScenes
import sys

# Add the p_perf package to the path
sys.path.append('/mmdetection3d_ros2/perf_ws/src/p_perf')

from p_perf.pPerf_manager import pPerfConfigManager
from p_perf.config.constant import (
    image_models,
    seg_models,
    get_abbreviated_name,
    get_abbreviated_scene,
    get_full_model_name,
    get_full_scene_token,
)
from p_perf.utils import (
    ensure_bags_exist,
    ExperimentCSVManager,
    create_failure_log,
    ExperimentRunner,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

bag_dir = "/mmdetection3d_ros2/data/bag"
# dataset_version = "v1.0-trainval"
dataset_version = "v1.0-mini"
# nuscenes_data_dir = Path("/mnt/nas/Nuscenes")
nuscenes_data_dir = Path("/mmdetection3d_ros2/data/nuscenes")
num_runs = -1

OVERWRITE = True
CONTINUE = False  # Set to True to continue from existing mapping file
LOGGING_DELAY = False
POST_PROCESS = True
KERNEL_PROCESS = True

# Parameter sweep setup
scenes = ["c5224b9b454b4ded9b5d2d2634bbda8a"]
depths = [-1]
image_queues = [1]
lidar_queues = [1]
publishing_rate = [10]
output_name = "det_1I1S-1"

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
    num_det_inferencers=1,  # Image detection
    num_seg_inferencers=1,  # Segmentation
    use_sim_time=True,
    logging_delay=LOGGING_DELAY,
)

# ============================================================================
# EXPERIMENT MAPPING
# ============================================================================

# Create combinations for det + seg testing
combinations = list(
    product(depths, image_models, seg_models, scenes, image_queues, lidar_queues, publishing_rate)
)

# Define CSV columns
csv_columns = [
    "run_index",
    "scene",
    "depth",
    "image_model",
    "seg_model",
    "image_queue",
    "lidar_queue",
    "publishing_rate",
    "status",
    "start_time",
]


def format_csv_row(index, combo):
    """Format combination tuple into CSV row with abbreviated names."""
    depth, img_model, seg_model, scene, img_q, lidar_q, pub_rate = combo
    abbreviated_scene = get_abbreviated_scene(scene)
    abbreviated_img_model = get_abbreviated_name(img_model)
    abbreviated_seg_model = (get_abbreviated_name(seg_model[0]), seg_model[1])
    return [
        index,
        abbreviated_scene,
        depth,
        abbreviated_img_model,
        abbreviated_seg_model,
        img_q,
        lidar_q,
        pub_rate,
        "pending",
        "",
    ]


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
    failure_log=failure_log,
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
    cleanup_memory=True,  # Automatically clean memory after each run
)


def update_config_for_run(row, run_index):
    """Update configuration for a specific experiment run."""
    # Read abbreviated names from CSV and convert back to full names for config
    abbreviated_img_model = row["image_model"]
    abbreviated_seg_model = row["seg_model"]
    abbreviated_scene = row["scene"]
    depth = row["depth"]
    img_q = row["image_queue"]
    lidar_q = row["lidar_queue"]
    pub_rate = row["publishing_rate"]

    # Convert abbreviated names to full names for configuration
    img_model = get_full_model_name(abbreviated_img_model)
    seg_model_abbreviated_tuple = ast.literal_eval(abbreviated_seg_model)
    seg_model_name = get_full_model_name(seg_model_abbreviated_tuple[0])
    seg_mode = seg_model_abbreviated_tuple[1]
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
            "section": "det_inferencers",
            "inferencer_index": 0,
            "mode": "image",
            "model_name": img_model,
            "data_dir": output_base,
        },
        {
            "section": "seg_inferencers",
            "inferencer_index": 0,
            "mode": seg_mode,
            "model_name": seg_model_name,
            "data_dir": output_base,
        },
    ]

    # Update all inferencers
    for config in inferencer_configs:
        manager.update_inferencer(
            config["section"],
            config["inferencer_index"],
            mode=config["mode"],
            model_name=config["model_name"],
            index=run_index,
            data_dir=config["data_dir"],
            lidar_queue=lidar_q,
            image_queue=img_q,
            depth=depth,
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
        config_file=config_file,
    )


# Run experiments
# max_runs: -1 = run all, >0 = run that many experiments
df = runner.run_experiments(
    df=df,
    csv_path=mapping_file,
    config_updater=update_config_for_run,
    launch_cmd_builder=build_launch_cmd,
    max_runs=num_runs,  # Change to positive number to limit runs (e.g., 1 for testing)
)

# ============================================================================
# POST-PROCESSING
# ============================================================================

if POST_PROCESS:
    # Load NuScenes for post-processing
    nusc = NuScenes(version=dataset_version, dataroot=str(nuscenes_data_dir))

    def parse_row_for_postprocessing(row):
        """Parse row to extract scene and other information for post-processing."""
        abbreviated_scene = row["scene"]
        scene = get_full_scene_token(abbreviated_scene)
        return {"scene": scene}

    # Post-process experiments
    # max_runs: -1 = process all, >0 = process that many experiments
    df = runner.post_process_experiments(
        df=df,
        csv_path=mapping_file,
        nusc=nusc,
        row_parser=parse_row_for_postprocessing,
        publish_mode="bag",
        cleanup_json=False,  # Set to True to delete JSON files after processing
        max_runs=num_runs,  # Change to positive number to limit post-processing (e.g., 1 for testing)
        process_kernels=KERNEL_PROCESS,
    )

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 60)
print("ALL RUNS COMPLETED")
print("=" * 60)
print(f"Results saved to: {output_base}")
print(f"Mapping file: {mapping_file}")
print(f"Failure log: {failure_log}")