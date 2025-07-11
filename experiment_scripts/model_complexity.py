import os
import subprocess
import csv
from itertools import product
from subprocess import TimeoutExpired
import pandas as pd
import json
import ast
import time
import torch

def clear_gpu():
    """Clear GPU memory and wait for it to be ready"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)  # Give some time for GPU to stabilize


# Define FLOPS metrics and their weights
flop_metrics = {
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,  # FMA counts as 2 FLOPS
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,  # ADD counts as 1 FLOPS
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,  # MUL counts as 1 FLOPS
}

# Base ncu command for FLOPS analysis
ncu_base = [
    "ncu",
    "--nvtx",
    "--nvtx-include", "regex:Frame[0-9]+/",
    "--target-processes", "all",
    "--import-source", "no",
    "--metrics", ",".join(flop_metrics.keys()),
    "--csv",
    "-f"
]

# Output folder
output_base = "/mmdetection3d_ros2/outputs/model_complexity"
os.makedirs(output_base, exist_ok=True)

# Define models to test
lidar_models = [
    ('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'nus'),    # VOXEL BASED
    ('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'nus'),        # VOXEL BASED
    ('hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'nus'),      # VOXEL BASED
    ('point-rcnn_8xb2_kitti-3d-3class', 'kitti'),            # POINT BASED
    ('3dssd_4x4_kitti-3d-car', 'kitti'),                       # POINT BASED
    ('pv_rcnn_8xb2-80e_kitti-3d-3class', 'kitti'),          # POINT + VOXEL (Hybrid)
]

image_models = [
    ('faster-rcnn_r50_fpn_1x_coco', 'coco'),      # TWO STAGE
    ('yolov3_d53_mstrain-608_273e_coco', 'coco'), # ONE STAGE ANCHOR
    ('yolox_x_8x8_300e_coco', 'coco'),            # ONE STAGE ANCHOR FREE
    ('centernet_r18-dcnv2_8xb16-crop512-140e_coco', 'coco'),   # ONE STAGE ANCHOR FREE
    ('detr_r50_8xb2-150e_coco', 'coco')           # TRANSFORMER
]

# Test scenes
scenes = ['2f0e54af35964a3fb347359836bec035']

# Failure log file
failure_log = os.path.join(output_base, "failures.log")
with open(failure_log, "w") as flog:
    flog.write("Failed Runs Log\n")
    flog.write("================\n")

# Generate all combinations
combinations = []
# Add lidar model combinations
for model in lidar_models:
    for scene in scenes:
        combinations.append(('lidar', model, scene))
# Add image model combinations
for model in image_models:
    for scene in scenes:
        combinations.append(('image', model, scene))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "param_mapping.csv")
# if not os.path.exists(mapping_file):
#     with open(mapping_file, mode='w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["run_index", "sensor_type", "scene", "model", "status", "start_time"])
#         for i, (sensor_type, model, scene) in enumerate(combinations):
#             writer.writerow([i, sensor_type, scene, model, "pending", "pending"])

# Now run them and update status
df = pd.read_csv(mapping_file)


for i, row in df.iterrows():
    if row["status"] == "success":
        continue
    sensor_type = row["sensor_type"]
    model_tuple = ast.literal_eval(row["model"])  # Safely convert string to tuple
    model_name = model_tuple[0]
    lidar_model_mode = model_tuple[1]
    scene = row["scene"]

    print(f"\nProcessing {sensor_type} model: {model_name}")
    print(f"Model mode: {lidar_model_mode}")

    prefix = f"{output_base}/test_run_{i}"

    # Clear GPU before each run
    clear_gpu()

    # Build the pPerf_complexity.py command
    if sensor_type == 'lidar':
        python_cmd = [
            "python3", "/mmdetection3d_ros2/perf_ws/src/p_perf/p_perf/pPerf_complexity.py",
            f"--model_name={model_name}",
            f"--sensor_type={sensor_type}",
            f"--lidar_model_mode={lidar_model_mode}",
            f"--scene_token={scene}"
        ]
    else:
        python_cmd = [
            "python3", "/mmdetection3d_ros2/perf_ws/src/p_perf/p_perf/pPerf_complexity.py",
            f"--model_name={model_name}",
            f"--sensor_type={sensor_type}",
            f"--scene_token={scene}"
        ]

    # Build the full NCU command
    ncu_cmd = ncu_base + python_cmd

    try:
        start_time = time.time()
        
        # Run NCU profiling for FLOPS analysis
        print(f"\n>>> Running {i}th NCU profiling for FLOPS analysis: {' '.join(ncu_cmd)}\n")
        with open(f"{prefix}_flops.csv", 'w') as f:
            subprocess.run(ncu_cmd, 
                         stdout=f, 
                         check=True, 
                         timeout=1800)  # 30 minutes timeout

    except subprocess.CalledProcessError as e:
        print(f"*** Run {i} failed with return code {e.returncode}")
        df.at[i, "status"] = f"failed ({e.returncode})"
        with open(failure_log, "a") as flog:
            flog.write(f"sensor_type={sensor_type}, model={model_name}, "
                      f"return_code={e.returncode}\n")
    except TimeoutExpired:
        print(f"*** Run {i} timed out after 1800s")
        df.at[i, "status"] = "timeout"
        with open(failure_log, "a") as flog:
            flog.write(f"sensor_type={sensor_type}, model={model_name}\n")
    except Exception as e:
        print(f"*** Run {i} failed with unexpected error: {e}")
        df.at[i, "status"] = "error"
        with open(failure_log, "a") as flog:
            flog.write(f"sensor_type={sensor_type}, model={model_name}, "
                      f"error={str(e)}\n")

    # Save status to CSV after each run
    df.at[i, "status"] = "success"
    df.at[i, "start_time"] = start_time
    df.to_csv(mapping_file, index=False)

print("\nAll runs completed!") 