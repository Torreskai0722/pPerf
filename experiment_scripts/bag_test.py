import os
import subprocess
import csv
from itertools import product
from subprocess import TimeoutExpired
import pandas as pd

# Base nsys command
nsys_base = [
    "nsys", "profile",
    "--trace=cuda,nvtx,cudnn",
    "--backtrace=none",
    "--force-overwrite", "true",
    "--export=json",
]

# Output folder
output_base = "/mmdetection3d_ros2/outputs/bag"
os.makedirs(output_base, exist_ok=True)

run_time = 30

# Failure log file
failure_log = os.path.join(output_base, "failures.log")
with open(failure_log, "w") as flog:
    flog.write("Failed Runs Log\n")
    flog.write("================\n")

# Parameter sweep setup
image_sample_freqs = [10]
lidar_sample_freqs = [14]
depths = [1]
image_models = [
    'faster-rcnn_r50_fpn_1x_coco',
    'detr_r50_8xb2-150e_coco',
    'yolov3_d53_320_273e_coco',
    'centernet_r18-dcnv2_8xb16-crop512-140e_coco'
]

lidar_models = [
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',
    'hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d',
    'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d',
]

# Generate all combinations
combinations = list(product(image_sample_freqs, lidar_sample_freqs, depths, image_models, lidar_models))

# Create mapping CSV with all combinations marked "pending"
mapping_file = os.path.join(output_base, "param_mapping.csv")
with open(mapping_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["run_index", "image_sample_freq", "lidar_sample_freq", "depth", "image_model", "lidar_model", "status"])
    for i, (img_freq, lidar_freq, depth, img_model, lidar_model) in enumerate(combinations):
        writer.writerow([i, img_freq, lidar_freq, depth, img_model, lidar_model, "pending"])

# Now run them and update status
df = pd.read_csv(mapping_file)

for i, row in df.iterrows():

    img_freq = row["image_sample_freq"]
    lidar_freq = row["lidar_sample_freq"]
    depth = row["depth"]
    img_model = row["image_model"]
    lidar_model = row["lidar_model"]

    output_file = f"{output_base}/v1_3_run_{i}"

    # Base ROS 2 launch command
    ros2_cmd = [
        "ros2", "launch", "p_perf", "pPerf_bag.launch.py",
        f"idx:={i}",
        f"data_dir:={output_base}",
        f"sensor_run_time:={run_time}",
        "sensor_expected_models:=2",
        f"image_sample_freq:={img_freq}",
        f"image_depth:={depth}",
        f"image_model_name:={img_model}",
        f"lidar_sample_freq:={lidar_freq}",
        f"lidar_depth:={depth}",
        f"lidar_model_name:={lidar_model}"
    ]


    full_cmd = nsys_base + ["-o", output_file] + ros2_cmd

    print(f"\n>>> Running ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")

    try:
        subprocess.run(full_cmd, check=True, timeout=180)
        df.at[i, "status"] = "success"
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

    try:
        command = ["python3", "src/p_perf/p_perf/pPerf_post.py", output_file]
        print(f"\nRunning: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Delete the corresponding .json file after processing
        json_path = f"{output_file}.json"
        if os.path.exists(json_path):
            os.remove(json_path)

    except subprocess.TimeoutExpired:
        print(f"*** Post-processing for run {i} timed out")
        df.at[i, "status"] += " + post-timeout"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} post-processing timeout for {output_file}\n")

    except subprocess.CalledProcessError as e:
        print(f"*** Post-processing failed with code {e.returncode}")
        df.at[i, "status"] += f" + post-failed ({e.returncode})"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} post-processing failed: return_code={e.returncode}\n")

    except Exception as e:
        print(f"*** Unexpected error during post-processing: {e}")
        df.at[i, "status"] += " + post-error"
        with open(failure_log, "a") as flog:
            flog.write(f"Run {i} post-processing error: {str(e)}\n")


    # Save status to CSV after each run
    df.to_csv(mapping_file, index=False)