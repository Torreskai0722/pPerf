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
from p_perf.post_process.image_eval import image_evaluater
from p_perf.post_process.pseudo_gt import generate_pseudo_coco_gt, load_model
from p_perf.post_process.image_eval import generate_coco_gt, change_pred_imageid
from p_perf.post_process.timing_post import timing_processor

from nuscenes.nuscenes import NuScenes
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

# Output folder
output_base = "/mmdetection3d_ros2/outputs/image_base"
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

# Generate all combinations
combinations = list(product(depths, image_models, scenes))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "param_mapping.csv")
with open(mapping_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["run_index", "scene", "depth", "image_model", "status"])
    for i, (depth, img_model, scene) in enumerate(combinations):
        writer.writerow([i, scene, depth, img_model, "pending"])


# Now run them and update status
df = pd.read_csv(mapping_file)

for i, row in df.iterrows():
    depth = row["depth"]
    img_model = row["image_model"]
    scene = row["scene"]

    prefix = f"{output_base}/test_run_{i}"

    ros2_cmd = [
        "ros2", "launch", "p_perf", "pPerf_base_image.launch.py",
        f"idx:={i}",
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
        subprocess.run(full_cmd, check=True, timeout=180)

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

    # RUNNING POST PROCESSING
    # EVALUATION PIPELINE OF INFERENCE TIME
    raw_timing_json = f"{prefix}.json"
    timing_analyzer = timing_processor(raw_timing_json, output_base, i)
    timing_analyzer.parse_json()
    layer_records, kernel_records = timing_analyzer.generate_mapping()

    # Delete the corresponding .json file after processing to save disk space
    if os.path.exists(raw_timing_json):
        os.remove(raw_timing_json)

    delay_csv = f"{output_base}/delays_{i}.csv"
    image_pred_file = f"{output_base}/image_pred_{i}.json"
    image_still_gt = f"{output_base}/image_still_gt_{i}.json"
    image_stream_gt = f"{output_base}/image_stream_gt_{i}.json"

    with open(image_pred_file, 'r') as f:
        data = json.load(f)  # should be a list of dicts
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

    # modify the image_id in prediction as the prediction's image_id is still nuscene token
    change_pred_imageid(image_pred_file, image_still_gt)

    image_evaluate = image_evaluater(image_pred_file, image_still_gt, nusc, output_base, i)
    image_evaluate.mAP_evaluate(mode='still')
    image_evaluate.mAP_evaluate(mode='stream')

    # clean up the output directory
    if os.path.exists(f"{output_base}/delays_{i}.csv.lock"):
        os.remove(f"{output_base}/delays_{i}.csv.lock")


    # Save status to CSV after each run
    df.at[i, "status"] = "success"
    df.at[i, "start_time"] = start_time
    df.to_csv(mapping_file, index=False)
