#!/usr/bin/env python3

import os
import subprocess
import csv
from itertools import product
from subprocess import TimeoutExpired
import pandas as pd
import ast
import time
from nuscenes.nuscenes import NuScenes
import json
import sys

# Add the p_perf package to the path
sys.path.append('/mmdetection3d_ros2/perf_ws/src/p_perf')

from p_perf.pPerf_manager import pPerfConfigManager
from p_perf.post_process.timing_post import timing_processor
from p_perf.config.constant import image_models, lidar_models, seg_models
from p_perf.post_process.kernel_processor import KernelProcessor
from p_perf.post_process.layer_processor import LayerProcessor

# Base nsys command
nsys_base = [
    "nsys", "profile",
    "--trace=cuda,nvtx,cudnn",
    "--backtrace=none",
    "--force-overwrite", "true",
]

bag_dir = "/mmdetection3d_ros2/data/"
OVERWRITE = True
LOGGING_DELAY = False

# Parameter sweep setup
scenes = ['2f0e54af35964a3fb347359836bec035']
depths = [-1]
image_queues = [1]
lidar_queues = [1]
publishing_rate = [10]
output_name = "det_1I1S"

output_base = f"/mmdetection3d_ros2/outputs/{output_name}"
os.makedirs(output_base, exist_ok=True)

# Create base configuration
base_config_file = os.path.join(output_base, "base_config.yaml")
manager = pPerfConfigManager(base_config_file)
manager.create_base_config(
    num_det_inferencers=1,  # LiDAR + Image detection
    num_seg_inferencers=1,  # Segmentation
    use_sim_time=True,
    logging_delay=LOGGING_DELAY
)

# Create combinations for full stack testing
combinations = list(product(depths, seg_models, lidar_models, scenes, image_queues, lidar_queues, publishing_rate))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "full_stack_mapping.csv")
if OVERWRITE:   
    with open(mapping_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_index", "scene", "depth", "seg_model", "lidar_model", "image_queue", "lidar_queue", "publishing_rate", "status", "start_time"])
        for i, (depth, seg_model, lidar_model, scene, img_q, lidar_q, publishing_rate) in enumerate(combinations):
            writer.writerow([i, scene, depth, seg_model, lidar_model, img_q, lidar_q, publishing_rate, "pending", ""])

# Now run them and update status
df = pd.read_csv(mapping_file)

# Failure log file
failure_log = os.path.join(output_base, "failures.log")
with open(failure_log, "w") as flog:
    flog.write("Failed Runs Log\n")
    flog.write("================\n")


print("\n" + "="*60)
print("EXPERIMENT EXECUTION PHASE")
print("="*60)


for i, row in df.iterrows():
    # if df.at[i, "status"] == "success":
    #     continue

    depth = row["depth"]
    seg_model_tuple = ast.literal_eval(row["seg_model"])
    lidar_model_tuple = ast.literal_eval(row["lidar_model"])
    scene = row["scene"]
    img_q = row["image_queue"]
    lidar_q = row["lidar_queue"]
    publishing_rate = row["publishing_rate"]

    prefix = f"{output_base}/test_run_{i}"

    lidar_model_name, lidar_mode = lidar_model_tuple
    seg_model_name, seg_mode = seg_model_tuple
    df.at[i, "status"] = "pending"

    # Update configuration using pPerfConfigManager
    try:
        
        # Update sensor replayer configuration
        manager.update_sensor_replayer(
            bag_dir=bag_dir,
            scene=scene,
            publishing_rate=publishing_rate,
            index=i,
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
                'section': 'seg_inferencers',
                'inferencer_index': 0,
                'mode': seg_mode,
                'model_name': seg_model_name,
                'data_dir': output_base
            }
        ]
        
        # Update all inferencers in a loop
        for config in inferencer_configs:
            manager.update_inferencer(
                config['section'], 
                config['inferencer_index'],
                mode=config['mode'],
                model_name=config['model_name'],
                index=i,  # experiment run index
                data_dir=config['data_dir'],
                lidar_model_mode=lidar_mode,
                #QoS
                lidar_queue=lidar_q,
                image_queue=img_q
            )
        
        # Save the updated configuration
        config_file = base_config_file
        manager.save_config(config_file)
        print(f"✓ Updated configuration for run {i} and saved to: {config_file}")
        
        # Display current configuration for debugging
        print(f"\n📋 Configuration for run {i}:")
        manager.list_inferencers()
        
    except Exception as e:
        error_msg = f"Failed to update configuration for run {i}: {str(e)}"
        print(f"Error: {error_msg}")
        with open(failure_log, "a") as flog:
            flog.write(f"{error_msg}\n")
        df.at[i, "status"] = "config_error"
        df.at[i, "start_time"] = time.time()
        continue

    # Launch full stack using the generated YAML configuration
    ros2_cmd = [
        "ros2", "launch", "p_perf", "full_stack.launch.py",
        f"config_file:={config_file}"
    ]

    full_cmd = nsys_base + ["-o", prefix] + ros2_cmd

    print(f"\n>>> Running Full Stack ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")

    start_time = time.time()
    try:
        subprocess.run(full_cmd, check=True, timeout=300)  # Increased timeout for full stack
        df.at[i, "status"] = "success"
        df.at[i, "start_time"] = start_time
        print(f"Full stack run {i} completed successfully")
    except TimeoutExpired as e:
        error_msg = f"Full stack run {i} timed out after {e.timeout} seconds"
        print(f"Error: {error_msg}")
        with open(failure_log, "a") as flog:
            flog.write(f"{error_msg}\n")
        df.at[i, "status"] = "timeout"
        df.at[i, "start_time"] = start_time
        print(f"Marked run {i} as timeout and continuing to next experiment")
    except Exception as e:
        error_msg = f"Full stack run {i} failed with unexpected error: {str(e)}"
        print(f"Error: {error_msg}")
        with open(failure_log, "a") as flog:
            flog.write(f"{error_msg}\n")
        df.at[i, "status"] = "error"
        df.at[i, "start_time"] = start_time
        print(f"Marked run {i} as error and continuing to next experiment")
    finally:
        df.to_csv(mapping_file, index=False)
        print(f"Successfully saved status for run {i} to {mapping_file}")




nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/nas/Nuscenes')

for i, row in df.iterrows():
    prefix = f"{output_base}/test_run_{i}"
    depth = row["depth"]
    lidar_model_tuple = ast.literal_eval(row["lidar_model"])
    lidar_model = lidar_model_tuple[0]
    lidar_model_mode = lidar_model_tuple[1]
    seg_model = row["seg_model"]
    publishing_rate = row["publishing_rate"]
    
    if df.at[i, "status"] != "success":
        continue

    print(f"\n--- Processing Run {i}/{len(df)} ---")

    # EVALUATION PIPELINE OF INFERENCE TIME
    print(f"Processing run {i}: RAW JSON")
    raw_timing_json = f"{prefix}.json"
    nsys_report = f"{prefix}.nsys-rep"

    if not os.path.exists(raw_timing_json):
        if os.path.exists(nsys_report):
            print(f"Raw timing JSON file not found. Generating from {nsys_report}")
            try:
                subprocess.run([
                    "nsys", "export",
                    "--type", "json",
                    "--output", raw_timing_json,
                    nsys_report
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to export from {nsys_report}: {e}")
                continue
        else:
            print(f"Both {raw_timing_json} and {nsys_report} do not exist. Skipping.")
            continue

    # Process timing data
    try:
        timing_analyzer = timing_processor(nusc, raw_timing_json, output_base, i, scene=row["scene"], publish_mode="bag")
        timing_analyzer.parse_json()
        layer_records, kernel_records = timing_analyzer.generate_mapping()
        print(f"✓ Timing analysis completed for run {i}")
    except Exception as e:
        print(f"✗ Timing analysis failed for run {i}: {e}")
        continue

    # Delete the corresponding .json file after processing to save disk space
    if os.path.exists(raw_timing_json):
        os.remove(raw_timing_json)
    
    # Cleanup
    timing_analyzer.cleanup()

    print(f"✓ Run {i} post-processing completed")
    time.sleep(5)  # Brief pause between runs


print("\n" + "="*60)
print("ALL RUNS COMPLETED")
print("="*60)
print(f"Results saved to: {output_base}")
print(f"Mapping file: {mapping_file}")
print(f"Failure log: {failure_log}") 