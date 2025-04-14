# Modular and reusable version of the unified script with CV plots

import pandas as pd
import numpy as np
import os
import json
import csv
from typing import List, Optional
import matplotlib.pyplot as plt

# Constants
TOP_KERNEL = 5
INPUT_DIR = 'faster_rcnn_log'
OUTPUT_DIR = 'sampling_faster_rcnn'
BASE_DIR = '/mmdetection3d_ros2/pPerf_ws'
os.makedirs(f"{BASE_DIR}/{OUTPUT_DIR}", exist_ok=True)
plot_dir = f"{BASE_DIR}/{OUTPUT_DIR}/layer_cv_plots"
os.makedirs(plot_dir, exist_ok=True)
csv_path = f"{BASE_DIR}/{INPUT_DIR}/param_mapping.csv"
image_model_name = "faster-rcnn_r50_fpn_1x_coco"

image_models = [
    'faster-rcnn_r50_fpn_1x_coco',
    'detr_r50_8xb2-150e_coco',
    'yolov3_d53_320_273e_coco',
    'centernet_r18-dcnv2_8xb16-crop512-140e_coco'
]

def read_csv(csv_path: str) -> List[dict]:
    with open(csv_path, newline='') as csvfile:
        return list(csv.DictReader(csvfile))

def filter_indices(
    rows: List[dict],
    depth: Optional[int] = None,
    image_model: Optional[str] = None,
    lidar_model: Optional[str] = None,
    image_sample_freq: Optional[int] = None,
    lidar_sample_freq: Optional[int] = None,
) -> List[int]:
    filtered_indices = []
    for idx, row in enumerate(rows):
        if depth is not None and int(row.get("depth", -1)) != depth:
            continue
        if image_model is not None and row.get("image_model") != image_model:
            continue
        if lidar_model is not None and row.get("lidar_model") != lidar_model:
            continue
        if image_sample_freq is not None and int(row.get("image_sample_freq", -1)) != image_sample_freq:
            continue
        if lidar_sample_freq is not None and int(row.get("lidar_sample_freq", -1)) != lidar_sample_freq:
            continue
        filtered_indices.append(idx)
    return filtered_indices

def compute_layer_stats(selected_indices: List[int], rows: List[dict]) -> pd.DataFrame:
    stats = []
    for i in selected_indices:
        run_info = rows[i]
        run_index = int(run_info["run_index"])
        image_sample_freq = int(run_info["image_sample_freq"])
        lidar_sample_freq = int(run_info["lidar_sample_freq"])
        df = pd.read_csv(f'{BASE_DIR}/{INPUT_DIR}/v1_3_run_{i}_mapping.csv')

        layer_stats = df.groupby(["Model Name", "Layer Name"]).agg({
            "Layer Elapsed Time": ["mean", "std", "max", "min"],
            "Layer GPU Wait Time": ["mean", "std", "max", "min"],
            "Layer Memcpy Bytes": ["mean", "std", "max", "min"]
        })
        layer_stats.columns = ['_'.join(col).strip() for col in layer_stats.columns.values]
        layer_stats = layer_stats.reset_index()
        layer_stats["run_index"] = run_index
        layer_stats["image_sample_freq"] = image_sample_freq
        layer_stats["lidar_sample_freq"] = lidar_sample_freq
        stats.append(layer_stats)
    return pd.concat(stats, ignore_index=True)

def merge_lidar_model(final_df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    param_df = pd.read_csv(csv_path)
    param_df_mapping = param_df[["run_index", "lidar_model"]].drop_duplicates()
    final_df["run_index"] = final_df["run_index"].astype(int)
    param_df_mapping["run_index"] = param_df_mapping["run_index"].astype(int)
    return final_df.merge(param_df_mapping, on="run_index", how="left")

def plot_cv(
    df: pd.DataFrame, 
    value_std: str, 
    value_mean: str, 
    cv_label: str, 
    ylabel: str, 
    file_suffix: str, 
    layer: str
):
    grouped = df.groupby(["image_sample_freq", "lidar_sample_freq", "lidar_model"]).agg({
        value_std: "mean",
        value_mean: "mean"
    }).reset_index()

    grouped[cv_label] = grouped[value_std] / grouped[value_mean]
    grouped["combo"] = grouped["image_sample_freq"].astype(str) + "-" + grouped["lidar_sample_freq"].astype(str)

    pivot = grouped.pivot_table(index="combo", columns="lidar_model", values=cv_label)
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop(columns=["mean"])

    ax = pivot.plot(kind="bar", figsize=(14, 6))
    plt.title(f"{ylabel} CV for Layer: {layer}")
    plt.xlabel("ImageFreq-LiDARFreq")
    plt.ylabel(f"Coefficient of Variation ({ylabel})")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="LiDAR Model")
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{layer}_{file_suffix}.png")
    plt.close()

def plot_runwise_cv_by_combo(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    ylabel: str,
    file_suffix: str,
    layer: str
):
    df = df.copy()
    df["cv"] = df[std_col] / df[value_col]
    df["combo"] = (
        df["image_sample_freq"].astype(str) + "-" +
        df["lidar_sample_freq"].astype(str) + "-" +
        df["lidar_model"]
    )

    combos = df["combo"].unique()

    plt.figure(figsize=(16, 6))
    for combo in combos:
        sub_df = df[df["combo"] == combo]
        sub_df = sub_df.sort_values("run_index")
        plt.plot(
            sub_df["run_index"], sub_df["cv"], marker='o', label=combo
        )

    plt.title(f"Run-wise CV for {ylabel} - Layer: {layer}")
    plt.xlabel("Run Index")
    plt.ylabel(f"Coefficient of Variation ({ylabel})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="ImageFreq-LidarFreq-Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{layer}_{file_suffix}_runwise_line.png")
    plt.close()



# ===== MAIN SCRIPT =====
if __name__=='__main__':
    rows = read_csv(csv_path)
    selected_indices = filter_indices(rows, depth=0, image_model=image_model_name)
    print(f"Found {len(selected_indices)} runs with depth == 0.")

    final_df = compute_layer_stats(selected_indices, rows)
    final_df.to_csv(f"{BASE_DIR}/{OUTPUT_DIR}/layer_stats.csv", index=False)

    final_df = merge_lidar_model(final_df, csv_path)
    image_model_df = final_df[final_df["Model Name"] == image_model_name]
    unique_layers = image_model_df["Layer Name"].unique()

    for layer in unique_layers:
        layer_df = image_model_df[image_model_df["Layer Name"] == layer]
        plot_cv(layer_df, "Layer Elapsed Time_std", "Layer Elapsed Time_mean", "Elapsed CV", "Elapsed Time", "elapsed_time_cv", layer)
        plot_cv(layer_df, "Layer GPU Wait Time_std", "Layer GPU Wait Time_mean", "GPU Wait CV", "GPU Wait Time", "gpu_wait_cv", layer)
        if layer.lower() == "e2e":
            plot_cv(layer_df, "Layer Memcpy Bytes_std", "Layer Memcpy Bytes_mean", "Memcpy CV", "Memcpy Bytes", "memcpy_cv", layer)

        plot_runwise_cv_by_combo(
            layer_df, 
            value_col="Layer Elapsed Time_mean", 
            std_col="Layer Elapsed Time_std", 
            ylabel="Elapsed Time", 
            file_suffix="elapsed_time_cv", 
            layer=layer
        )

        plot_runwise_cv_by_combo(
            layer_df, 
            value_col="Layer GPU Wait Time_mean", 
            std_col="Layer GPU Wait Time_std", 
            ylabel="GPU Wait Time", 
            file_suffix="gpu_wait_cv", 
            layer=layer
        )

        if layer.lower() == "e2e":
            plot_runwise_cv_by_combo(
                layer_df, 
                value_col="Layer Memcpy Bytes_mean", 
                std_col="Layer Memcpy Bytes_std", 
                ylabel="Memcpy Bytes", 
                file_suffix="memcpy_cv", 
                layer=layer
            )
    print(f"✅ All CV plots saved to {plot_dir}")
