import os
import subprocess
import csv
from itertools import product
from subprocess import TimeoutExpired
import pandas as pd
import json
import ast
import time

from p_perf.post_process.lidar_eval import lidar_evaluater
from p_perf.config.constant import nus_lidar_classes, kitti_lidar_classes
from p_perf.post_process.timing_post import timing_processor

from nuscenes import NuScenes
DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
nusc = NuScenes(
            version='v1.0-mini',
            dataroot=DATA_ROOT,
            verbose=False
        )


# Base nsys command
nsys_base = [
    "nsys", "profile",
    "--trace=cuda,nvtx,cudnn",
    "--backtrace=none",
    "--force-overwrite", "true",
    "--export=json",
]
bag_dir = "/mmdetection3d_ros2/data/"

# Output folder
output_base = "/mmdetection3d_ros2/outputs/lidar_base"
os.makedirs(output_base, exist_ok=True)

scenes = ['c5224b9b454b4ded9b5d2d2634bbda8a', 'd25718445d89453381c659b9c8734939', '2fc3753772e241f2ab2cd16a784cc680']

# Failure log file
failure_log = os.path.join(output_base, "failures.log")
with open(failure_log, "w") as flog:
    flog.write("Failed Runs Log\n")
    flog.write("================\n")

# Parameter sweep setup
depths = [0]

lidar_models = [
    ('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'nus', 0.5),    # VOXEL BASED
    ('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'nus', 0.5),        # VOXEL BASED
    ('hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'nus', 0.5)      # VOXEL BASED
    # ('point-rcnn_8xb2_kitti-3d-3class', 'kitti-3class', 0.5),            # POINT BASED
    # ('3dssd_4x4_kitti-3d-car', 'kitti-car', 0.1),                       # POINT BASED
    # ('pv_rcnn_8xb2-80e_kitti-3d-3class', 'kitti-3class', 0.1),             # POINT + VOXEL (Hybrid)
]

# Generate all combinations
combinations = list(product(depths, lidar_models, scenes))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "param_mapping.csv")
with open(mapping_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["run_index", "scene", "depth","lidar_model", "status", "start_time"])
    for i, (depth, lidar_model, scene) in enumerate(combinations):
        writer.writerow([i, scene, depth, lidar_model, "pending", "pending"])

# Now run them and update status
df = pd.read_csv(mapping_file)

for i, row in df.iterrows():

    depth = row["depth"]
    lidar_model_tuple = ast.literal_eval(row["lidar_model"])  # Safely convert string to tuple
    lidar_model = lidar_model_tuple[0]
    lidar_model_mode = lidar_model_tuple[1]
    lidar_thresh = lidar_model_tuple[2]
    scene = row["scene"]

    print(lidar_model)
    print(lidar_model_mode)

    prefix = f"{output_base}/test_run_{i}"

    ros2_cmd = [
        "ros2", "launch", "p_perf", "pPerf_base_lidar.launch.py",
        f"idx:={i}",
        f"bag_dir:={bag_dir}",
        f"scene:={scene}",
        f"data_dir:={output_base}",
        f"scene:={scene}",
        "sensor_expected_models:=1",
        f"lidar_depth:={depth}",
        f"lidar_model_name:={lidar_model}",
        f"lidar_model_mode:={lidar_model_mode}",
        f"lidar_model_thresh:={lidar_thresh}"
    ]


    full_cmd = nsys_base + ["-o", prefix] + ros2_cmd

    print(f"\n>>> Running ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")


    start_time = time.time()
    subprocess.run(full_cmd, check=True, timeout=180)

    # Save status to CSV after each run
    df.at[i, "status"] = "success"
    df.at[i, "start_time"] = start_time
    df.to_csv(mapping_file, index=False)


for i, row in df.iterrows():
    depth = row["depth"]
    lidar_model_tuple = ast.literal_eval(row["lidar_model"])  # Safely convert string to tuple
    lidar_model = lidar_model_tuple[0]
    lidar_model_mode = lidar_model_tuple[1]
    lidar_thresh = lidar_model_tuple[2]
    scene = row["scene"]

    # # RUNNING POST PROCESSING
    # # EVALUATION PIPELINE OF INFERENCE TIME
    raw_timing_json = f"{prefix}.json"
    timing_analyzer = timing_processor(raw_timing_json, output_base, i)
    timing_analyzer.parse_json()
    layer_records, kernel_records = timing_analyzer.generate_mapping()

    # Delete the corresponding .json file after processing
    json_path = f"{prefix}.json"
    if os.path.exists(json_path):
        os.remove(json_path)

    delay_csv = f"{output_base}/delays_{i}.csv"
    # EVALUATION PIPELINE OF LIDAR ACCURACY 
    lidar_pred_file = f"{output_base}/lidar_pred_{i}.json"
    if 'nus' in lidar_model_mode or 'car' in lidar_model_mode:
        lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i, nus_lidar_classes)
    else:
        lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i, kitti_lidar_classes)
    pred_boxes = lidar_evaluate.load_prediction_of_sample_tokens([], all=True)
    lidar_evaluate.evaluate(pred_boxes, 'still')
    lidar_evaluate.evaluate(pred_boxes, 'stream')


    # clean up the output directory
    if os.path.exists(f"{output_base}/delays_{i}.csv.lock"):
        os.remove(f"{output_base}/delays_{i}.csv.lock")

