import csv
from collections import defaultdict
from p_perf.nuscenes_instance import get_nuscenes_instance
from p_perf.utils import load_sweep_sd
import os

def analyze_flops(nusc, csv_file: str, sensor_type: str = 'lidar', scene_token: str = None) -> dict:
    """
    Analyze FLOPs from a CSV file and convert frame IDs to sample data tokens.
    
    Args:
        nusc: NuScenes instance
        csv_file (str): Path to the CSV file containing FLOPs metrics
        sensor_type (str): Type of sensor ('lidar' or 'image')
        scene_token (str): NuScenes scene token for frame-to-token conversion
        
    Returns:
        dict: Dictionary mapping sample data tokens to FLOPs
    """
    flop_metrics = {
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    }

    flops_per_frame = defaultdict(int)
    frame_to_token = {}

    # If scene_token is provided, create frame to token mapping
    if scene_token:
        scene = nusc.get('scene', scene_token)
        sensor_mode = 'LIDAR_TOP' if sensor_type == 'lidar' else 'CAM_FRONT'
        sd_tokens = load_sweep_sd(nusc, scene, sensor_channel=sensor_mode)
        
        # Create mapping of frame index to sample data token
        for frame_idx, sd_token in enumerate(sd_tokens):
            frame_to_token[f"Frame{frame_idx}"] = sd_token

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nvtx_range = row['thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg'].strip()
            metric_name = row['Metric Name'].strip()
            metric_val_str = row['Metric Value'].strip()

            # Skip rows without NVTX range or irrelevant metric
            if not nvtx_range or metric_name not in flop_metrics:
                continue

            # Extract frame ID from the NVTX range, e.g., Frame12
            if "Frame" in nvtx_range:
                try:
                    start = nvtx_range.index("Frame")
                    end = nvtx_range.index(":", start) if ":" in nvtx_range[start:] else len(nvtx_range)
                    frame_id = nvtx_range[start:end]
                except ValueError:
                    continue  # malformed string

                try:
                    val = float(metric_val_str)
                    flops = val * flop_metrics[metric_name]
                    flops_per_frame[frame_id] += flops
                except ValueError:
                    continue  # non-numeric value

    # Convert frame IDs to sample data tokens if mapping exists
    if frame_to_token:
        flops_per_token = {frame_to_token[frame_id]: flops 
                          for frame_id, flops in flops_per_frame.items() 
                          if frame_id in frame_to_token}
        
        # Save results to CSV
        output_dir = os.path.dirname(csv_file)
        output_file = os.path.join(output_dir, f"flops_analysis_{os.path.basename(csv_file)}")
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame ID', 'Sample Data Token', 'FLOPs'])
            for frame_id, flops in flops_per_frame.items():
                if frame_id in frame_to_token:
                    writer.writerow([frame_id, frame_to_token[frame_id], flops])
        
        return flops_per_token
    


if __name__ == "__main__":
    # Example usage
    nusc = get_nuscenes_instance()
    for i in range(11):
        csv_file = f"/mmdetection3d_ros2/outputs/model_complexity/test_run_{i}_flops.csv"
        flops = analyze_flops(nusc, csv_file, sensor_type='lidar', scene_token='2f0e54af35964a3fb347359836bec035')
        print(f"Run {i} FLOPs analysis saved to {os.path.join(os.path.dirname(csv_file), f'flops_analysis_test_run_{i}.csv')}")
