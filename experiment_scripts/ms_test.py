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

from p_perf.post_process.lidar_eval import lidar_evaluater
from p_perf.post_process.image_eval import image_evaluater
from p_perf.config.constant import nus_lidar_classes, kitti_lidar_classes
from p_perf.post_process.pseudo_gt import generate_pseudo_coco_gt, load_model
from p_perf.post_process.image_eval import generate_coco_gt, change_pred_imageid
from p_perf.post_process.timing_post import timing_processor

DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
nusc = NuScenes(
    version='v1.0-mini',
    dataroot=DATA_ROOT,
    verbose=True
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
output_base = "/mmdetection3d_ros2/outputs/ms"
os.makedirs(output_base, exist_ok=True)

scenes = ['c5224b9b454b4ded9b5d2d2634bbda8a', 'd25718445d89453381c659b9c8734939', '2fc3753772e241f2ab2cd16a784cc680']

# Failure log file
failure_log = os.path.join(output_base, "failures.log")
with open(failure_log, "w") as flog:
    flog.write("Failed Runs Log\n")
    flog.write("================\n")

# Parameter sweep setup
depths = [0]
image_models = [    
    'faster-rcnn_r50_fpn_1x_coco',      # TWO STAGE
    'yolov3_d53_mstrain-608_273e_coco', # ONE STAGE ANCHOR
    'yolox_x_8x8_300e_coco',            # ONE STAGE ANCHOR FREE
    'centernet_r18-dcnv2_8xb16-crop512-140e_coco',   # ONE STAGE ANCHOR FREE
    'detr_r50_8xb2-150e_coco'           # TRANSFORMER
]

lidar_models = [
    ('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'nus', 0.5),    # VOXEL BASED
    ('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'nus', 0.5),        # VOXEL BASED
    ('hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'nus', 0.5),      # VOXEL BASED
    ('point-rcnn_8xb2_kitti-3d-3class', 'kitti-3class', 0.2),            # POINT BASED
    ('3dssd_4x4_kitti-3d-car', 'kitti-car', 0.2),                       # POINT BASED
    ('pv_rcnn_8xb2-80e_kitti-3d-3class', 'kitti-3class', 0.2),             # POINT + VOXEL (Hybrid)
]

# Create combinations for one lidar and one image model
model_combinations = []
for depth in depths:
    for img_model in image_models:
        for lidar_model in lidar_models:
            model_combinations.append((depth, img_model, lidar_model, None))

# Combine all combinations
all_combinations = model_combinations

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "ms_param_mapping.csv")
with open(mapping_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["run_index", "scene", "depth", "image_model", "lidar_model", "mm_model", "status", "start_time"])
    for i, (depth, img_model, lidar_model, mm_model) in enumerate(all_combinations):
        for scene in scenes:
            writer.writerow([i, scene, depth, img_model, lidar_model, mm_model, "pending", ""])

# Now run them and update status
df = pd.read_csv(mapping_file)

for i, row in df.iterrows():
    depth = row["depth"]
    img_model = row["image_model"]
    lidar_model = ast.literal_eval(row["lidar_model"]) if pd.notna(row["lidar_model"]) else None
    mm_model = ast.literal_eval(row["mm_model"]) if pd.notna(row["mm_model"]) else None
    scene = row["scene"]

    prefix = f"{output_base}/ms_test_run_{i}"

    # Base ROS 2 launch command
    ros2_cmd = [
        "ros2", "launch", "p_perf", "pPerf_ms.launch.py",
        f"idx:={i}",
        f"bag_dir:={bag_dir}",
        f"data_dir:={output_base}",
        f"scene:={scene}",
        f"image_depth:={depth}",
        f"lidar_depth:={depth}",
        f"image_model_name:={img_model}",
        f"lidar_model_name:={lidar_model[0]}",
        f"lidar_model_mode:={lidar_model[1]}",
        f"lidar_model_thresh:={lidar_model[2]}"
    ]

    full_cmd = nsys_base + ["-o", prefix] + ros2_cmd

    print(f"\n>>> Running ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")

    start_time = time.time()
    try:
        # Run with stdout and stderr piped to the logger
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output in real-time
        while True:
            output = proc.stdout.readline()
            if output == '' and proc.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Read any remaining stderr
        for error in proc.stderr:
            if error:
                print(f"ERROR: {error.strip()}")
        
        # Wait for process to complete
        return_code = proc.wait()
        if return_code != 0:
            print(f"Process exited with code {return_code}")
            df.at[i, "status"] = "failed"
            with open(failure_log, "a") as flog:
                flog.write(f"\nRun {i} failed with return code {return_code}\n")
                flog.write(f"Command: {' '.join(full_cmd)}\n")
        else:
            df.at[i, "status"] = "success"
            df.at[i, "start_time"] = start_time
            
    except Exception as e:
        print(f"Error running command: {str(e)}")
        df.at[i, "status"] = "failed"
        with open(failure_log, "a") as flog:
            flog.write(f"\nRun {i} failed: {str(e)}\n")
            flog.write(f"Command: {' '.join(full_cmd)}\n")
    
    df.to_csv(mapping_file, index=False)

# # Process results
# for i, row in df.iterrows():
#     if df.at[i, "status"] != "success":
#         continue

#     prefix = f"{output_base}/ms_test_run_{i}"
#     depth = row["depth"]
    
    # Process timing data
    raw_timing_json = f"{prefix}.json"
    timing_analyzer = timing_processor(raw_timing_json, output_base, i)
    timing_analyzer.parse_json()
    layer_records, kernel_records = timing_analyzer.generate_mapping()

    # Delete the corresponding .json file after processing
    if os.path.exists(raw_timing_json):
        os.remove(raw_timing_json)

    delay_csv = f"{output_base}/delays_{i}.csv"

    # Process lidar results if applicable
    if pd.notna(row["lidar_model"]):
        lidar_model_tuple = ast.literal_eval(row["lidar_model"])
        lidar_model = lidar_model_tuple[0]
        lidar_model_mode = lidar_model_tuple[1]
        lidar_thresh = lidar_model_tuple[2]

        lidar_pred_file = f"{output_base}/lidar_pred_{i}.json"
        if 'nus' in lidar_model_mode or 'car' in lidar_model_mode:
            lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i, nus_lidar_classes)
        else:
            lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i, kitti_lidar_classes)
        pred_boxes = lidar_evaluate.load_prediction_of_sample_tokens([], all=True)
        lidar_evaluate.evaluate(pred_boxes, 'still')
        lidar_evaluate.evaluate(pred_boxes, 'stream')

    # Process image results if applicable
    if pd.notna(row["image_model"]):
        image_pred_file = f"{output_base}/image_pred_{i}.json"
        image_still_gt = f"{output_base}/image_still_gt_{i}.json"
        image_stream_gt = f"{output_base}/image_stream_gt_{i}.json"

        with open(image_pred_file, 'r') as f:
            data = json.load(f)
        tokens = [d['image_id'] for d in data if 'image_id' in d]
        tokens = list(set(tokens))

        config_dir = '/mmdetection3d_ros2/DINO/dino_package/config'
        config_path = f'{config_dir}/DINO/DINO_4scale_swin.py'
        ckpt_path = f'{config_dir}/ckpts/checkpoint0029_4scale_swin.pth'
        id2name_path = '/mmdetection3d_ros2/DINO/dino_package/util/coco_id2name.json'

        with open(id2name_path) as f:
            id2name = {int(k): v for k, v in json.load(f).items()}
        model, postprocessors = load_model(config_path, ckpt_path)
        generate_pseudo_coco_gt(nusc, tokens, model, postprocessors, id2name, delay_csv, image_still_gt, streaming=False)
        generate_pseudo_coco_gt(nusc, tokens, model, postprocessors, id2name, delay_csv, image_stream_gt, streaming=True)

        change_pred_imageid(image_pred_file, image_still_gt)

        image_evaluate = image_evaluater(image_pred_file, image_still_gt, nusc, output_base, i)
        image_evaluate.mAP_evaluate(mode='still')
        image_evaluate.mAP_evaluate(mode='stream')

    # Clean up
    if os.path.exists(f"{output_base}/delays_{i}.csv.lock"):
        os.remove(f"{output_base}/delays_{i}.csv.lock") 