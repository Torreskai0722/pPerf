import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Model name mappings for shorter display names
model_name_mappings = {
    'faster-rcnn_r50_fpn_1x_coco': 'FasterRCNN',
    'yolov3_d53_mstrain-608_273e_coco': 'Yolo3',
    'yolox_x_8x8_300e_coco': 'YoloX',
    'detr_r50_8xb2-150e_coco': 'DETR',
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d': 'PointPillars',
    'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d': 'CenterPoint',
    'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d': 'SSN',
    'point-rcnn_8xb2_kitti-3d-3class': 'PointRCNN',
    '3dssd_4x4_kitti-3d-car': '3DSSD',
}

image_models = [
    'faster-rcnn_r50_fpn_1x_coco',
    'yolov3_d53_mstrain-608_273e_coco',
    'yolox_x_8x8_300e_coco',
    'detr_r50_8xb2-150e_coco'
]

lidar_models = [
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',
    'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d',
    'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d',
    'point-rcnn_8xb2_kitti-3d-3class',
    '3dssd_4x4_kitti-3d-car',
]

# Specify the model pair you want to analyze
# Example: model1 = 'faster-rcnn_r50_fpn_1x_coco', model2 = 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d'
model1 = 'faster-rcnn_r50_fpn_1x_coco'
model2 = 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d'

PRIORITY_PAIRS = [
    (0, 0), (0, -1), (-1, 0)
]

Y_AXIS_RANGE = 140

plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'figure.titlesize': 25
})

def get_model_type(model_name):
    if model_name in image_models:
        return 'image'
    elif model_name in lidar_models:
        return 'lidar'
    else:
        raise ValueError(f"Unknown model: {model_name}")

def remove_outliers(data: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """
    Remove outliers from data using the IQR method.
    """
    if len(data) < 4:
        return data
    q1 = np.percentile(data, 28)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def collect_inference_stats(param_mapping, main_model, counter_model, multi_tenant_dir):
    main_type = get_model_type(main_model)
    counter_type = get_model_type(counter_model)
    stats = {}
    for image_priority, lidar_priority in PRIORITY_PAIRS:
        if main_type == 'image' and counter_type == 'lidar':
            submask = (
                (param_mapping['image_model'] == main_model) &
                (param_mapping['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == counter_model)) &
                (param_mapping['image_priority'] == image_priority) &
                (param_mapping['lidar_priority'] == lidar_priority)
            )
        elif main_type == 'lidar' and counter_type == 'image':
            submask = (
                (param_mapping['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == main_model)) &
                (param_mapping['image_model'] == counter_model) &
                (param_mapping['image_priority'] == image_priority) &
                (param_mapping['lidar_priority'] == lidar_priority)
            )
        else:
            continue  # Only support image-lidar or lidar-image pairs
        rows = param_mapping[submask]
        all_e2e = []
        for _, row in rows.iterrows():
            run_index = row['run_index']
            timings_path = os.path.join(multi_tenant_dir, f"layer_timings_{run_index}.csv")
            if not os.path.exists(timings_path):
                continue
            timings = pd.read_csv(timings_path)
            e2e_times = timings[(timings['Model'] == main_model) & (timings['Layer'] == 'e2e')]['Elapsed Time'].values
            # Remove outliers before adding
            e2e_times = remove_outliers(np.array(e2e_times))
            all_e2e.extend(e2e_times)
        if len(all_e2e) > 0:
            stats[(image_priority, lidar_priority)] = {
                'mean': np.mean(all_e2e),
                'range': np.max(all_e2e) - np.min(all_e2e),
                'count': len(all_e2e)
            }
    return stats

def plot_mean_range_bar(stats, main_model, counter_model, output_dir):
    labels = []
    means = []
    ranges = []
    for (image_priority, lidar_priority), stat in stats.items():
        mean = stat.get('mean', 0)
        range_ = stat.get('range', 0)
        # Handle NaN or inf
        if not np.isfinite(mean):
            mean = 0
        if not np.isfinite(range_):
            range_ = 0
        labels.append(f"img_prio={image_priority},lidar_prio={lidar_priority}")
        means.append(mean)
        ranges.append(range_)
    if not means:
        print(f"No data for {main_model} vs {counter_model}")
        return
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(max(8, len(labels)*1.5), 6))
    bar1 = plt.bar(x - width/2, means, width, color='skyblue', label='Mean')
    bar2 = plt.bar(x + width/2, ranges, width, color='salmon', label='Range')
    # plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel("Inference Time (ms)")
    # plt.title(f"{model_name_mappings.get(main_model, main_model)} (with {model_name_mappings.get(counter_model, counter_model)})\nInference Time Mean and Range by Priority Pair")
    plt.ylim(0, Y_AXIS_RANGE)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{model_name_mappings.get(main_model, main_model)}_main_{model_name_mappings.get(counter_model, counter_model)}_counter_mean_range_bar.png"))
    plt.close()

def main():
    multi_tenant_dir = "outputs/ms"
    output_dir = "tools/priority_bar_plots"
    param_mapping = pd.read_csv(os.path.join(multi_tenant_dir, "param_mapping.csv"))
    # Plot for model1 as main, model2 as counter
    stats1 = collect_inference_stats(param_mapping, model1, model2, multi_tenant_dir)
    plot_mean_range_bar(stats1, model1, model2, output_dir)
    # Plot for model2 as main, model1 as counter
    stats2 = collect_inference_stats(param_mapping, model2, model1, multi_tenant_dir)
    plot_mean_range_bar(stats2, model2, model1, output_dir)

if __name__ == "__main__":
    main() 