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

from p_perf.post_process.lidar_eval import lidar_evaluater
from p_perf.post_process.image_eval import image_evaluater
from p_perf.post_process.pseudo_gt import generate_pseudo_coco_gt, load_model
from p_perf.post_process.image_eval import generate_coco_gt, change_pred_imageid
from p_perf.post_process.timing_post import timing_processor
from p_perf.nuscenes_instance import get_nuscenes_instance
from nuscenes import NuScenes

def clear_gpu():
    """Clear GPU memory and wait for it to be ready"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)  # Give some time for GPU to stabilize


OVERWRITE = True

# Base nsys command
nsys_base = [
    "nsys", "profile",
    "--trace=cuda,nvtx,cudnn",
    "--backtrace=none",
    "--force-overwrite", "true",
    # "--export=json",
]

bag_dir = "/mmdetection3d_ros2/data/bag"

# Output folder
output_base = "/mmdetection3d_ros2/outputs/test"
os.makedirs(output_base, exist_ok=True)

scenes = ['2f0e54af35964a3fb347359836bec035']

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
    'detr_r50_8xb2-150e_coco'           # TRANSFORMER
]

# Generate all combinations
combinations = list(product(depths, image_models, scenes))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "param_mapping.csv")
if OVERWRITE:
    with open(mapping_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_index", "scene", "depth", "image_model", "status", "start_time"])
        for i, (depth, img_model, scene) in enumerate(combinations):
            writer.writerow([i, scene, depth, img_model, "pending"])


# Now run them and update status
df = pd.read_csv(mapping_file)


for i, row in df.iterrows():
    depth = row["depth"]
    img_model = row["image_model"]
    scene = row["scene"]

    prefix = f"{output_base}/image_base_run_{i}"

    ros2_cmd = [
        "ros2", "launch", "p_perf", "pPerf_base_image.launch.py",
        f"idx:={i}",
        f"bag_dir:={bag_dir}",
        f"data_dir:={output_base}",
        f"scene:={scene}",
        "sensor_expected_models:=1",
        f"image_depth:={depth}",
        f"image_model_name:={img_model}",
    ]


    full_cmd = nsys_base + ["-o", prefix] + ros2_cmd

    print(f"\n>>> Running ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")

    try:
        start_time = time.time()
        subprocess.run(full_cmd, check=True, timeout=350)

    except subprocess.CalledProcessError as e:
        print(f"*** Run {i} failed with return code {e.returncode}")
        df.at[i, "status"] = f"failed ({e.returncode})"
        with open(failure_log, "a") as flog:
            flog.write(f"depth={depth}, img_model={img_model}"
                       f"return_code={e.returncode}\n")
    except TimeoutExpired:
        print(f"*** Run {i} timed out after 180s")
        df.at[i, "status"] = "timeout"
        with open(failure_log, "a") as flog:
            flog.write(f"depth={depth}, img_model={img_model}\n")
    except Exception as e:
        print(f"*** Run {i} failed with unexpected error: {e}")
        df.at[i, "status"] = "error"
        with open(failure_log, "a") as flog:
            flog.write(f"depth={depth}, img_model={img_model}"
                       f"error={str(e)}\n")
            
    # Save status to CSV after each run
    df.at[i, "status"] = "success"
    df.at[i, "start_time"] = start_time
    df.to_csv(mapping_file, index=False)

nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/nas/Nuscenes', verbose=True)

for i, row in df.iterrows():
    depth = row["depth"]
    img_model = row["image_model"]
    scene = row["scene"]
    prefix = f"{output_base}/image_base_run_{i}"

    # RUNNING POST PROCESSING
    # EVALUATION PIPELINE OF INFERENCE TIME
    print(f"Processing run {i} RAW JSON")
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

    # delay_csv = f"{output_base}/delays_{i}.csv"
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
    # generate_pseudo_coco_gt(nusc, tokens, model, postprocessors, id2name, delay_csv, image_still_gt, streaming=False, model_name=img_model)
    # generate_pseudo_coco_gt(nusc, tokens, model, postprocessors, id2name, delay_csv, image_stream_gt, streaming=True, model_name=img_model)

    # # modify the image_id in prediction as the prediction's image_id is still nuscene token
    # change_pred_imageid(image_pred_file, image_still_gt)

    # image_evaluate = image_evaluater(image_pred_file, image_still_gt, nusc, output_base, i)
    # image_evaluate.mAP_evaluate(mode='still')
    # image_evaluate.mAP_evaluate(mode='stream')

    # # clean up the output directory
    # if os.path.exists(f"{output_base}/delays_{i}.csv.lock"):
    #     os.remove(f"{output_base}/delays_{i}.csv.lock")

