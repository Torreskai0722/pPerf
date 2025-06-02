import os
import subprocess
import csv
from itertools import product
from subprocess import TimeoutExpired
import pandas as pd
import json
import ast

from p_perf.post_process.lidar_eval import lidar_evaluater
from p_perf.post_process.image_eval import image_evaluater
from p_perf.config.constant import nusc, nus_lidar_classes, kitti_lidar_classes
from p_perf.post_process.pseudo_gt import generate_pseudo_coco_gt, load_model
from p_perf.post_process.image_eval import generate_coco_gt, change_pred_imageid

# Base nsys command
nsys_base = [
    "nsys", "profile",
    "--trace=cuda,nvtx,cudnn",
    "--backtrace=none",
    "--force-overwrite", "true",
    # "--export=json",
]

# Output folder
output_base = "/mmdetection3d_ros2/outputs/lidar_base"
os.makedirs(output_base, exist_ok=True)

scenes = ['cc8c0bf57f984915a77078b10eb33198']

# Failure log file
failure_log = os.path.join(output_base, "failures.log")
with open(failure_log, "w") as flog:
    flog.write("Failed Runs Log\n")
    flog.write("================\n")

# Parameter sweep setup
lidar_sample_freqs = [10]
depths = [0]

lidar_models = [
    # ('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'nus', 0.5),    # VOXEL BASED
    # ('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'nus', 0.5),        # VOXEL BASED
    # ('point-rcnn_8xb2_kitti-3d-3class', 'kitti-3class', 0.5),            # POINT BASED
    # ('3dssd_4x4_kitti-3d-car', 'kitti-car', 0.1),                       # POINT BASED
    ('pv_rcnn_8xb2-80e_kitti-3d-3class', 'kitti-3class', 0.1),             # POINT + VOXEL (Hybrid)
]

# Generate all combinations
combinations = list(product(lidar_sample_freqs, depths, lidar_models, scenes))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "param_mapping.csv")
with open(mapping_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["run_index", "scene", "lidar_sample_freq", "depth","lidar_model", "status"])
    for i, (lidar_freq, depth, lidar_model, scene) in enumerate(combinations):
        writer.writerow([i, scene, lidar_freq, depth, lidar_model, "pending"])

# Now run them and update status
df = pd.read_csv(mapping_file)

for i, row in df.iterrows():
    lidar_freq = row["lidar_sample_freq"]
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
        f"data_dir:={output_base}",
        f"scene:={scene}",
        "sensor_expected_models:=1",
        f"lidar_sample_freq:={lidar_freq}",
        f"lidar_depth:={depth}",
        f"lidar_model_name:={lidar_model}",
        f"lidar_model_mode:={lidar_model_mode}",
        f"lidar_model_thresh:={lidar_thresh}"
    ]


    full_cmd = nsys_base + ["-o", prefix] + ros2_cmd

    print(f"\n>>> Running ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")

    try:
        subprocess.run(full_cmd, check=True, timeout=180)

    except subprocess.CalledProcessError as e:
        print(f"*** Run {i} failed with return code {e.returncode}")
        df.at[i, "status"] = f"failed ({e.returncode})"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} failed: lidar_freq={lidar_freq}, "
                       f"depth={depth}, lidar_model={lidar_model}, "
                       f"return_code={e.returncode}\n")
    except TimeoutExpired:
        print(f"*** Run {i} timed out after 180s")
        df.at[i, "status"] = "timeout"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} timeout: lidar_freq={lidar_freq}, "
                       f"depth={depth}, lidar_model={lidar_model}\n")
    except Exception as e:
        print(f"*** Run {i} failed with unexpected error: {e}")
        df.at[i, "status"] = "error"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} error: lidar_freq={lidar_freq}, "
                       f"depth={depth}, lidar_model={lidar_model}, "
                       f"error={str(e)}\n")

    # # RUNNING POST PROCESSING


    # # EVALUATION PIPELINE OF INFERENCE TIME
    # command = ["python3", "/mmdetection3d_ros2/perf_ws/src/p_perf/p_perf/post_process/timing_post.py", prefix]
    # print(f"\nRunning: {' '.join(command)}")
    # result = subprocess.run(command, capture_output=True, text=True)

    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)

    # # Delete the corresponding .json file after processing
    # json_path = f"{prefix}.json"
    # if os.path.exists(json_path):
    #     os.remove(json_path)

    delay_csv = f"{output_base}/delays_{i}.csv"
    # EVALUATION PIPELINE OF LIDAR ACCURACY 
    lidar_pred_file = f"{output_base}/lidar_pred_{i}.json"
    if 'nus' in lidar_model_mode or 'car' in lidar_model_mode:
        lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i, nus_lidar_classes)
    else:
        lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i, kitti_lidar_classes)
    pred_boxes = lidar_evaluate.load_prediction_of_sample_tokens([], all=True)
    lidar_evaluate.evaluate(pred_boxes)


    # clean up the output directory
    if os.path.exists(f"{output_base}/delays_{i}.csv.lock"):
        os.remove(f"{output_base}/delays_{i}.csv.lock")

    # Save status to CSV after each run
    df.at[i, "status"] = "success"
    df.to_csv(mapping_file, index=False)