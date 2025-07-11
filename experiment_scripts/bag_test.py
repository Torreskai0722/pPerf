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
import torch

from p_perf.post_process.lidar_eval import lidar_evaluater
from p_perf.post_process.image_eval import image_evaluater
from p_perf.config.constant import nus_lidar_classes, kitti_lidar_classes
from p_perf.post_process.pseudo_gt import generate_pseudo_coco_gt, load_model
from p_perf.post_process.image_eval import generate_coco_gt, change_pred_imageid
from p_perf.post_process.timing_post import timing_processor
from p_perf.nuscenes_instance import get_nuscenes_instance
from p_perf.post_process.detection_parser import determine_parsing_strategy, parse_lidar_detections_for_run, parse_image_detections_for_run, parse_multimodal_detections_for_run

# Import cleanup utilities
from cleanup_utils import cleanup_between_runs, clear_gpu

# Base nsys command
nsys_base = [
    "nsys", "profile",
    "--trace=cuda,nvtx,cudnn",
    "--backtrace=none",
    "--force-overwrite", "true",
]
bag_dir = "/mmdetection3d_ros2/data/"
OVERWRITE = True

# Output folder
output_base = "/mmdetection3d_ros2/outputs/MACs_1"
os.makedirs(output_base, exist_ok=True)

# scenes = [
#     '2f0e54af35964a3fb347359836bec035',  # ENVIRONMENT: Crowded city (scene 429)
#     '3a1850241080418b88dcee97c7d17ed7',  # ENVIRONMENT: Crowded city (scene 27)
    
#     'bcb05cffb5814973a6cff4fbdca2b99b',  # ENVIRONMENT: Highspeed highway (scene 194)
#     '74d03e3b673f4a72a3f76e1b0c7840bd',  # ENVIRONMENT: Highspeed highway (scene 612)
    
#     '32185f91e68f4069ab3cdd2f4f1a4ff1',  # ENVIRONMENT: Suburban (scene 709)
#     '93032908bafd484ea10a39b0d5d3fff6',  # ENVIRONMENT: Suburban (scene 143)
    
#     # '7ca5e90766dc4b7bb6a9b3b0d95c99da',  # WEATHER: Clear (scene 201)
#     # 'bed8426a524d45afab05b19cf02386b2',  # WEATHER: Clear (scene 97)
    
#     # '5fc7c5dfc56b4971b14005bc53f69908',  # WEATHER: Rainy (scene 389)
#     # '0ced08ea43754420a23b2fbec667a763',  # WEATHER: Rainy (scene 364)
    
#     # 'b519ec833e23450a8bd3340b67f2516b',  # LIGHTING: Day (scene 314)
#     # '8edbc31083ab4fb187626e5b3c0411f7',  # LIGHTING: Day (scene 37)
    
#     # 'f634de95cc7043b8b38ceaac67d472cf',  # LIGHTING: Night (scene 784)
#     # 'b2119bd765a94b0cb705424800307bbf'   # LIGHTING: Night (scene 768)
# ]



scenes = ['2f0e54af35964a3fb347359836bec035']

# Failure log file
failure_log = os.path.join(output_base, "failures.log")
with open(failure_log, "w") as flog:
    flog.write("Failed Runs Log\n")
    flog.write("================\n")

# Parameter sweep setup
depths = [0]

# Image models for different detection approaches
image_models = [    
    'faster-rcnn_r50_fpn_1x_coco',      # TWO STAGE
    'yolov3_d53_mstrain-608_273e_coco', # ONE STAGE ANCHOR
    'yolox_x_8x8_300e_coco',            # ONE STAGE ANCHOR FREE
    'detr_r50_8xb2-150e_coco',          # TRANSFORMER                         # No image model
]

# Lidar models for different detection approaches
lidar_models = [
    ('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'nus', 0.5),    # VOXEL BASED
    ('centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'nus', 0.5),        # VOXEL BASED
    ('hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'nus', 0.5),      # VOXEL BASED
    ('point-rcnn_8xb2_kitti-3d-3class', 'kitti-3class', 0.2),            # POINT BASED
    ('3dssd_4x4_kitti-3d-car', 'kitti-car', 0.2),                       # POINT BASED
]


image_queues = [1]
lidar_queues = [1]
publishing_rate = [10, 13, 15]

combinations = list(product(depths, image_models, lidar_models, scenes, image_queues, lidar_queues, publishing_rate))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "param_mapping.csv")
if OVERWRITE:   
    with open(mapping_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_index", "scene", "depth", "image_model", "lidar_model", "image_queue", "lidar_queue", "publishing_rate", "status", "start_time"])
        for i, (depth, img_model, lidar_model, scene, img_q, lidar_q, publishing_rate) in enumerate(combinations):
            writer.writerow([i, scene, depth, img_model, lidar_model, img_q, lidar_q, publishing_rate, "pending", ""])

# Now run them and update status
df = pd.read_csv(mapping_file)

for i, row in df.iterrows():
    # if df.at[i, "status"] == "success":
    #     continue

    depth = row["depth"]
    img_model = row["image_model"]
    lidar_model_tuple = ast.literal_eval(row["lidar_model"])
    lidar_model = lidar_model_tuple[0]
    lidar_model_mode = lidar_model_tuple[1]
    lidar_thresh = lidar_model_tuple[2]
    scene = row["scene"]
    img_q = row["image_queue"]
    lidar_q = row["lidar_queue"]
    publishing_rate = row["publishing_rate"]

    prefix = f"{output_base}/test_run_{i}"

    # if "detr" not in img_model:
    #     continue

    df.at[i, "status"] = "pending"

    clear_gpu()

    # # Base ROS 2 launch command
    ros2_cmd = [
        "ros2", "launch", "p_perf", "pPerf_bag.launch.py",
        f"idx:={i}",
        f"bag_dir:={bag_dir}",
        f"data_dir:={output_base}",
        f"scene:={scene}",
        "sensor_expected_models:=2",
        f"image_depth:={depth}",
        f"image_model_name:={img_model}",
        f"image_queue:={img_q}",
        f"lidar_depth:={depth}",
        f"lidar_model_name:={lidar_model}",
        f"lidar_model_mode:={lidar_model_mode}",
        f"lidar_model_thresh:={lidar_thresh}",
        f"lidar_queue:={lidar_q}",
        f"publishing_rate:={publishing_rate}"
    ]


    full_cmd = nsys_base + ["-o", prefix] + ros2_cmd

    print(f"\n>>> Running ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")

    start_time = time.time()
    try:
        subprocess.run(full_cmd, check=True, timeout=180)
        df.at[i, "status"] = "success"
        df.at[i, "start_time"] = start_time
        print(f"Run {i} completed successfully")
    except subprocess.CalledProcessError as e:
        error_msg = f"Run {i} failed with error code {e.returncode}: {str(e)}"
        print(f"Error: {error_msg}")
        with open(failure_log, "a") as flog:
            flog.write(f"{error_msg}\n")
        df.at[i, "status"] = "failed"
        df.at[i, "start_time"] = start_time
        print(f"Marked run {i} as failed and continuing to next experiment")
    except TimeoutExpired as e:
        error_msg = f"Run {i} timed out after {e.timeout} seconds"
        print(f"Error: {error_msg}")
        with open(failure_log, "a") as flog:
            flog.write(f"{error_msg}\n")
        df.at[i, "status"] = "timeout"
        df.at[i, "start_time"] = start_time
        print(f"Marked run {i} as timeout and continuing to next experiment")
    except Exception as e:
        error_msg = f"Run {i} failed with unexpected error: {str(e)}"
        print(f"Error: {error_msg}")
        with open(failure_log, "a") as flog:
            flog.write(f"{error_msg}\n")
        df.at[i, "status"] = "error"
        df.at[i, "start_time"] = start_time
        print(f"Marked run {i} as error and continuing to next experiment")
    finally:
        try:
            df.to_csv(mapping_file, index=False)
            print(f"Successfully saved status for run {i} to {mapping_file}")
        except Exception as e:
            print(f"Failed to save status for run {i}: {str(e)}")
            # Try to save to a backup file
            backup_file = f"{mapping_file}.backup"
            try:
                df.to_csv(backup_file, index=False)
                print(f"Saved backup status to {backup_file}")
            except Exception as e2:
                print(f"Failed to save backup status: {str(e2)}")

        

nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/nas/Nuscenes')

# nusc = NuScenes(version='v1.0-trainval-rain', dataroot='/mmdetection3d_ros2/data/nuscenes')
for i, row in df.iterrows():

    prefix = f"{output_base}/test_run_{i}"
    depth = row["depth"]
    lidar_model_tuple = ast.literal_eval(row["lidar_model"])
    lidar_model = lidar_model_tuple[0]
    lidar_model_mode = lidar_model_tuple[1]
    lidar_thresh = lidar_model_tuple[2]
    image_model = row["image_model"]
    publishing_rate = row["publishing_rate"]
    if df.at[i, "status"] != "success":
        continue


    # if "detr" not in row["image_model"]:
    #     continue


    # EVALUATION PIPELINE OF INFERENCE TIME
    print(f"Processing run {i}/{len(df)}: RAW JSON")
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

    timing_analyzer = timing_processor(nusc, raw_timing_json, output_base, i, scene=row["scene"], publish_mode="bag")
    timing_analyzer.parse_json()
    layer_records, kernel_records = timing_analyzer.generate_mapping()

    # Delete the corresponding .json file after processing to save disk space
    if os.path.exists(raw_timing_json):
        os.remove(raw_timing_json)


    # # PARSE RAW DETECTIONS
    # try:
    #     print(f"Processing run {i}/{len(df)}: PARSING DETECTIONS")
        
    #     # # Get model names
    #     img_model = row["image_model"]
    #     lidar_model_tuple = ast.literal_eval(row["lidar_model"])
    #     lidar_model = lidar_model_tuple[0]
        
    #     # Determine parsing strategy
    #     parsing_tasks = determine_parsing_strategy(
    #         img_model, lidar_model, lidar_model_mode, lidar_thresh, 
    #         row["scene"], output_base, i
    #     )
        
    #     # Execute parsing tasks
    #     for task in parsing_tasks:
    #         print(f"Parsing {task['type']} detections for model: {task['model_name']}")
            
    #         if task['type'] == 'multimodal':
    #             parse_multimodal_detections_for_run(**task['params'])
    #         elif task['type'] == 'lidar':
    #             parse_lidar_detections_for_run(**task['params'])
    #         elif task['type'] == 'image':
    #             parse_image_detections_for_run(**task['params'])
    #         else:
    #             print(f"Unknown parsing task type: {task['type']}")
    # except Exception as e:
    #     print(f"Error parsing detections: {e}")
    #     continue

    # delay_csv = f"{output_base}/delays_{i}.csv"
    # # EVALUATION PIPELINE OF LIDAR ACCURACY 
    # lidar_pred_file = f"{output_base}/lidar_pred_{i}.json"
    # if 'nus' in lidar_model_mode or 'car' in lidar_model_mode:
    #     lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i, nus_lidar_classes)
    # else:
    #     lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i, kitti_lidar_classes)
    # pred_boxes = lidar_evaluate.load_prediction_of_sample_tokens([], all=True)
    # lidar_evaluate.evaluate(pred_boxes, 'still')
    # lidar_evaluate.evaluate(pred_boxes, 'stream')


    # # EVALUATION PIPELINE OF IMAGE ACCURACY
    # image_pred_file = f"{output_base}/image_pred_{i}.json"
    # image_still_gt = f"{output_base}/image_still_gt_{i}.json"
    # image_stream_gt = f"{output_base}/image_stream_gt_{i}.json"

    # with open(image_pred_file, 'r') as f:
    #     data = json.load(f)  # should be a list of dicts
    # tokens = [d['image_id'] for d in data if 'image_id' in d]
    # tokens = list(set(tokens))

    # config_dir = '/mmdetection3d_ros2/DINO/dino_package/config'
    # config_path = f'{config_dir}/DINO/DINO_4scale_swin.py'
    # ckpt_path = f'{config_dir}/ckpts/checkpoint0029_4scale_swin.pth'
    # id2name_path = '/mmdetection3d_ros2/DINO/dino_package/util/coco_id2name.json'

    # with open(id2name_path) as f:
    #     id2name = {int(k): v for k, v in json.load(f).items()}
    # model, postprocessors = load_model(config_path, ckpt_path)
    # generate_pseudo_coco_gt(nusc, tokens, model, postprocessors, id2name, delay_csv, image_still_gt, streaming=False)
    # generate_pseudo_coco_gt(nusc, tokens, model, postprocessors, id2name, delay_csv, image_stream_gt, streaming=True)

    # # modify the image_id in prediction as the prediction's image_id is still nuscene token
    # change_pred_imageid(image_pred_file, image_still_gt)

    # image_evaluate = image_evaluater(image_pred_file, image_still_gt, nusc, output_base, i)
    # image_evaluate.mAP_evaluate(mode='still')
    # image_evaluate.mAP_evaluate(mode='stream')

    # # clean up the output directory
    # if os.path.exists(f"{output_base}/delays_{i}.csv.lock"):
    #     os.remove(f"{output_base}/delays_{i}.csv.lock")

    
    # time.sleep(10)


