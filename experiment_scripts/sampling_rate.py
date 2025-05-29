import os
import subprocess
import csv
from itertools import product
from subprocess import TimeoutExpired
import pandas as pd
import json

from p_perf.post_process.lidar_eval import lidar_evaluater
from p_perf.post_process.image_eval import image_evaluater
from p_perf.config.constant import nusc
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
output_base = "/mmdetection3d_ros2/outputs/test"
os.makedirs(output_base, exist_ok=True)

scenes = ['cc8c0bf57f984915a77078b10eb33198']

# Failure log file
failure_log = os.path.join(output_base, "failures.log")
with open(failure_log, "w") as flog:
    flog.write("Failed Runs Log\n")
    flog.write("================\n")

# Parameter sweep setup
image_sample_freqs = [10]
lidar_sample_freqs = [10]
depths = [0]
image_models = [    
    'faster-rcnn_r50_fpn_1x_coco',
    # 'yolox_x_8x8_300e_coco',
    # 'tood_x101-64x4d_fpn_ms-2x_coco',
    # 'detr_r50_8xb2-150e_coco'
]

lidar_models = [
    # 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',
    'hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d',
    # 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d',
]

# Generate all combinations
combinations = list(product(image_sample_freqs, lidar_sample_freqs, depths, image_models, lidar_models, scenes))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "param_mapping.csv")
with open(mapping_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["run_index", "scene", "image_sample_freq", "lidar_sample_freq", "depth", "image_model", "lidar_model", "status"])
    for i, (img_freq, lidar_freq, depth, img_model, lidar_model, scene) in enumerate(combinations):
        writer.writerow([i, scene, img_freq, lidar_freq, depth, img_model, lidar_model, "pending"])

# Now run them and update status
df = pd.read_csv(mapping_file)

for i, row in df.iterrows():

    img_freq = row["image_sample_freq"]
    lidar_freq = row["lidar_sample_freq"]
    depth = row["depth"]
    img_model = row["image_model"]
    lidar_model = row["lidar_model"]

    prefix = f"{output_base}/test_run_{i}"

    # Base ROS 2 launch command
    ros2_cmd = [
        "ros2", "launch", "p_perf", "pPerf_test.launch.py",
        f"idx:={i}",
        f"data_dir:={output_base}",
        f"scene:={scene}",
        "sensor_expected_models:=2",
        f"image_sample_freq:={img_freq}",
        f"image_depth:={depth}",
        f"image_model_name:={img_model}",
        f"lidar_sample_freq:={lidar_freq}",
        f"lidar_depth:={depth}",
        f"lidar_model_name:={lidar_model}"
    ]


    full_cmd = nsys_base + ["-o", prefix] + ros2_cmd

    print(f"\n>>> Running ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")

    try:
        subprocess.run(full_cmd, check=True, timeout=180)

    except subprocess.CalledProcessError as e:
        print(f"*** Run {i} failed with return code {e.returncode}")
        df.at[i, "status"] = f"failed ({e.returncode})"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} failed: img_freq={img_freq}, lidar_freq={lidar_freq}, "
                       f"depth={depth}, img_model={img_model}, lidar_model={lidar_model}, "
                       f"return_code={e.returncode}\n")
    except TimeoutExpired:
        print(f"*** Run {i} timed out after 180s")
        df.at[i, "status"] = "timeout"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} timeout: img_freq={img_freq}, lidar_freq={lidar_freq}, "
                       f"depth={depth}, img_model={img_model}, lidar_model={lidar_model}\n")
    except Exception as e:
        print(f"*** Run {i} failed with unexpected error: {e}")
        df.at[i, "status"] = "error"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} error: img_freq={img_freq}, lidar_freq={lidar_freq}, "
                       f"depth={depth}, img_model={img_model}, lidar_model={lidar_model}, "
                       f"error={str(e)}\n")

    # EVALUATION PIPELINE OF INFERENCE TIME
    # command = ["python3", "src/p_perf/p_perf/pPerf_post.py", prefix]
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
    lidar_evaluate = lidar_evaluater(lidar_pred_file, nusc, output_base, i)
    pred_boxes = lidar_evaluate.load_prediction_of_sample_tokens([], all=True)
    lidar_evaluate.evaluate(pred_boxes)

    # EVALUATION PIPELINE OF IMAGE ACCURACY
    image_pred_file = f"{output_base}/image_pred_{i}.json"
    image_gt_file = f"{output_base}/image_gt_{i}.json"

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
    generate_pseudo_coco_gt(nusc, tokens, model, postprocessors, id2name, delay_csv, image_gt_file)

    # generate_pseudo_coco_gt(nusc, tokens, None, None, None, delay_csv, image_gt_file)

    # modify the image_id in prediction as the prediction's image_id is still nuscene token
    change_pred_imageid(image_pred_file, image_gt_file)

    image_evaluate = image_evaluater(image_pred_file, image_gt_file, nusc, output_base, i)
    image_evaluate.mAP_evaluate()

    # clean up the output directory
    if os.path.exists(f"{output_base}/delays_{i}.csv.lock"):
        os.remove(f"{output_base}/delays_{i}.csv.lock")

    # Save status to CSV after each run
    df.at[i, "status"] = "success"
    df.to_csv(mapping_file, index=False)