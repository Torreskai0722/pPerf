import os
import glob
import pandas as pd
import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ast


# Set up NuScenes
BASE = '/home/mg/pdnn/pPerf'
# BASE = '/mmdetection3d_ros2'
nusc = NuScenes(version='v1.0-mini', dataroot=f'{BASE}/data/nuscenes', verbose=False)

DELAY_DIR = f'{BASE}/outputs/nusc_scene0'  # Folder containing delay_{run_index}.csv

def count_points(token):
    try:
        file_path, _, _ = nusc.get_sample_data(token)
        pts = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
        return len(pts)
    except Exception as e:
        print(f"[WARN] Failed on token {token}: {e}")
        return None

def load_all_lidar_delays():
    all_data = []
    
    for csv_file in glob.glob(os.path.join(DELAY_DIR, "delays_*.csv")):
        df = pd.read_csv(csv_file)
        lidar_df = df[df['sensor_type'] == 'lidar'].copy()
        all_data.append(lidar_df)

    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

def plot_box_by_point_bins(df, n_bins=10):
    print("[INFO] Counting points...")
    df['num_points'] = df['input_token'].apply(count_points)
    df.dropna(subset=['num_points', 'decode_delay'], inplace=True)

    min_pts = int(df['num_points'].min())
    max_pts = int(df['num_points'].max())
    print(f"[INFO] Point count range: {min_pts} - {max_pts}")

    bins = np.linspace(min_pts, max_pts, n_bins + 1)
    df['point_bin'] = pd.cut(df['num_points'], bins=bins, include_lowest=True)
    
    grouped = df.groupby('point_bin')['decode_delay'].apply(list)

    plt.figure(figsize=(14, 6))
    plt.boxplot(grouped.dropna(), positions=np.arange(len(grouped)), patch_artist=True)

    bin_labels = [f"{int(b.left)}-{int(b.right)}" for b in grouped.index]
    plt.xticks(ticks=np.arange(len(grouped)), labels=bin_labels, rotation=45)
    plt.xlabel("LiDAR Point Count Range (binned)")
    plt.ylabel("Decode Delay (s)")
    plt.title("Decode Delay Distribution vs LiDAR Point Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("decode_delay_vs_points.png")


def plot_metric_by_sampling_frequency(data_dir, image_model_name, lidar_model_name, metric='process_time', sensor_filter='lidar', freq_type='image_sample_freq'):
    # Step 1: Read param_mapping.csv
    param_csv = f'{data_dir}/param_mapping.csv'
    param_df = pd.read_csv(param_csv)

    # Match exact model pair
    param_df = param_df[param_df['image_model'] == image_model_name]
    param_df = param_df[param_df['lidar_model'].str.contains(lidar_model_name)]

    # Check result
    if len(param_df) == 0:
        print("[WARN] No matching experiments found for given model pair.")
        return

    all_rows = []

    # Step 2: Loop over matching run_index files
    for _, row in param_df.iterrows():
        run_index = row['run_index']
        freq = row[freq_type]

        delay_path = os.path.join(data_dir, f'delays_{run_index}.csv')
        if not os.path.exists(delay_path):
            print(f"[WARN] Missing file: {delay_path}")
            continue

        delay_df = pd.read_csv(delay_path)
        delay_df = delay_df[delay_df['sensor_type'] == sensor_filter].copy()
        delay_df['sampling_freq'] = freq
        all_rows.append(delay_df[['sampling_freq', metric]])

    if not all_rows:
        print("[WARN] No valid delay data found.")
        return

    merged_df = pd.concat(all_rows, ignore_index=True)
    merged_df.dropna(subset=[metric, 'sampling_freq'], inplace=True)

    # Step 3: Group and Plot
    grouped = merged_df.groupby('sampling_freq')[metric].apply(list)

    save_fig_dir = f'{DELAY_DIR}/visualization/{metric}_VS_rate'
    os.makedirs(save_fig_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.boxplot(grouped, positions=np.arange(len(grouped)), patch_artist=True)
    plt.xticks(ticks=np.arange(len(grouped)), labels=grouped.index.astype(str), rotation=45)
    plt.xlabel(f"{freq_type} (Hz)")
    plt.ylabel(metric.replace('_', ' ').capitalize())
    plt.title(f"model pair:\n{image_model_name} + {lidar_model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/{sensor_filter}_{image_model_name}_{lidar_model_name}.png')



def plot_metric_by_fixed_freq(data_dir, fixed_model_name, model_type='image',
                                        image_freq_val=10, lidar_freq_val=10,
                                        metric='process_time', sensor_filter='lidar'):

    assert model_type in ['image', 'lidar'], "model_type must be 'image' or 'lidar'"

    param_csv = os.path.join(data_dir, 'param_mapping.csv')
    param_df = pd.read_csv(param_csv)

    # Parse lidar_model field if needed
    if 'lidar_model' in param_df.columns:
        param_df['lidar_model_parsed'] = param_df['lidar_model'].apply(
            lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith("(") else x
        )
    else:
        raise ValueError("Missing 'lidar_model' column in param_mapping.csv")

    # Replace original column for easier use
    if model_type == 'lidar':
        param_df['lidar_model'] = param_df['lidar_model_parsed']

    # Step 1: Filter by fixed model and freq pair
    col_name = 'image_model' if model_type == 'image' else 'lidar_model'
    opposite_col = 'lidar_model' if model_type == 'image' else 'image_model'

    filtered_df = param_df[
        (param_df[col_name] == fixed_model_name) &
        (param_df['image_sample_freq'] == image_freq_val) &
        (param_df['lidar_sample_freq'] == lidar_freq_val)
    ]

    if filtered_df.empty:
        print(f"[WARN] No runs found for {model_type} model: {fixed_model_name} at ({image_freq_val}Hz, {lidar_freq_val}Hz)")
        return

    box_data = {}

    for _, row in filtered_df.iterrows():
        run_index = row['run_index']
        other_model = row[opposite_col]
        if isinstance(other_model, str) and other_model.startswith("("):
            try:
                other_model = ast.literal_eval(other_model)[0]
            except:
                pass  # keep original string if parsing fails

        delay_csv = os.path.join(data_dir, f'delays_{run_index}.csv')
        if not os.path.exists(delay_csv):
            print(f"[WARN] Missing file: {delay_csv}")
            continue

        delay_df = pd.read_csv(delay_csv)
        delay_df = delay_df[delay_df['sensor_type'] == sensor_filter]

        if delay_df.empty or metric not in delay_df.columns:
            continue

        box_data.setdefault(other_model, []).extend(delay_df[metric].dropna().tolist())

    if not box_data:
        print("[WARN] No valid data found for plotting.")
        return

    # Step 2: Plotting
    save_dir = os.path.join(data_dir, 'visualization', f'{metric}_VS_model')
    os.makedirs(save_dir, exist_ok=True)

    sorted_keys = sorted(box_data.keys())
    plt.figure(figsize=(max(12, len(sorted_keys) * 1.5), 6))
    plt.boxplot([box_data[k] for k in sorted_keys], patch_artist=True)
    plt.xticks(ticks=np.arange(1, len(sorted_keys) + 1), labels=sorted_keys, rotation=45, ha='right')
    plt.xlabel(f"{opposite_col.replace('_', ' ').capitalize()} (varied)")
    plt.ylabel(metric.replace('_', ' ').capitalize())
    plt.title(f"{metric.replace('_', ' ').capitalize()} for {fixed_model_name}\nImage={image_freq_val}Hz, Lidar={lidar_freq_val}Hz")
    plt.grid(True)
    plt.tight_layout()

    safe_model_name = fixed_model_name.replace("/", "_").replace(" ", "_")
    save_path = os.path.join(save_dir, f'{safe_model_name}_{image_freq_val}Hz_{lidar_freq_val}Hz.png')
    plt.savefig(save_path)
    print(f"[INFO] Saved plot to {save_path}")


if __name__ == '__main__':
    # df_all = load_all_lidar_delays()
    # plot_box_by_point_bins(df_all)
    img_freqs = [10, 11, 12]
    lidar_freqs = [10]

    metrics = ['process_delay', 'comm_delay', 'process_time', 'decode_delay']

    image_models = [    
        'faster-rcnn_r50_fpn_1x_coco',      # TWO STAGE
        'yolov3_d53_mstrain-608_273e_coco', # ONE STAGE ANCHOR
        'yolox_x_8x8_300e_coco',            # ONE STAGE ANCHOR FREE
        'centernet_r18-dcnv2_8xb16-crop512-140e_coco',   # ONE STAGE ANCHOR FREE
        'detr_r50_8xb2-150e_coco'           # TRANSFORMER
    ]

    lidar_models = [
        'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',    # VOXEL BASED
        'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d',         # VOXEL BASED
        'point-rcnn_8xb2_kitti-3d-3class',            # POINT BASED
        '3dssd_4x4_kitti-3d-car',                       # POINT BASED
        'pv_rcnn_8xb2-80e_kitti-3d-3class',             # POINT + VOXEL (Hybrid)
    ]

    for image_model in image_models:
        for lidar_model in lidar_models:
            for metric in metrics:
                for model_type in ['image', 'lidar']:
                    plot_metric_by_sampling_frequency(DELAY_DIR, image_model, lidar_model, metric, 'image')


    for image_model in image_models:
        for img_freq in img_freqs:
            for metric in metrics:
                plot_metric_by_fixed_freq(DELAY_DIR, image_model, 'image', img_freq, 10, metric, 'image')


    for lidar_model in lidar_models:
        for img_freq in img_freqs:
            for metric in metrics:
                plot_metric_by_fixed_freq(DELAY_DIR, lidar_model, 'lidar', img_freq, 10, metric, 'lidar')