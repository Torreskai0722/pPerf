import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import glob
import ast

# Model name mappings for shorter display names
model_name_mappings = {
    # Image models
    'faster-rcnn_r50_fpn_1x_coco': 'FasterRCNN',
    'yolov3_d53_mstrain-608_273e_coco': 'Yolo3',
    'yolox_x_8x8_300e_coco': 'YoloX',
    'detr_r50_8xb2-150e_coco': 'DETR',
    
    # Lidar models
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d': 'PointPillars',
    'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d': 'CenterPoint',
    'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d': 'SSN',
    'point-rcnn_8xb2_kitti-3d-3class': 'PointRCNN',
    '3dssd_4x4_kitti-3d-car': '3DSSD',
    'pv_rcnn_8xb2-80e_kitti-3d-3class': 'PVRCNN'
}

image_models = [    
    'faster-rcnn_r50_fpn_1x_coco',      # TWO STAGE
    'yolov3_d53_mstrain-608_273e_coco', # ONE STAGE ANCHOR
    'yolox_x_8x8_300e_coco',            # ONE STAGE ANCHOR FREE
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

def set_plot_text_sizes():
    """
    Set larger text sizes for all plot elements.
    """
    plt.rcParams.update({
        'font.size': 28,           # Default font size
        'axes.titlesize': 28,      # Title font size
        'axes.labelsize': 28,      # Axis label font size
        'xtick.labelsize': 28,     # X-axis tick label font size
        'ytick.labelsize': 28,     # Y-axis tick label font size
        'legend.fontsize': 20,     # Legend font size
        'figure.titlesize': 28     # Figure title font size
    })

def read_param_mapping(output_dir: str) -> pd.DataFrame:
    """
    Read the param_mapping.csv file from the output directory.
    
    Args:
        output_dir: Path to the output directory containing param_mapping.csv
        
    Returns:
        DataFrame containing the parameter mapping information
    """
    param_mapping_path = os.path.join(output_dir, 'param_mapping.csv')
    if not os.path.exists(param_mapping_path):
        raise FileNotFoundError(f"param_mapping.csv not found in {output_dir}")
    return pd.read_csv(param_mapping_path)

def get_run_indices(param_mapping: pd.DataFrame, model_name: str, model_type: str, **filter_params) -> List[int]:
    """
    Find run indices based on model name, type, and additional parameters.
    
    Args:
        param_mapping: DataFrame from read_param_mapping
        model_name: Name of the model to find
        model_type: Type of model [image, lidar]
        **filter_params: Additional parameters to filter by (e.g., scene, depth, queue sizes)
        
    Returns:
        List of run indices matching the model and parameters
    """
    # Start with model type specific mask
    if model_type == "image":
        mask = param_mapping['image_model'] == model_name
    elif model_type == "lidar":
        # For lidar models, we need to handle the tuple string format
        mask = param_mapping['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Apply additional filters
    for param, value in filter_params.items():
        if param in param_mapping.columns:
            mask &= (param_mapping[param] == value)
    
    return param_mapping[mask]['run_index'].tolist()

def get_counter_models(param_mapping: pd.DataFrame, run_indices: List[int], model_type: str) -> List[str]:
    """
    Get the list of counter models for the given run indices.
    
    Args:
        param_mapping: DataFrame from read_param_mapping
        run_indices: List of run indices to check
        model_type: Type of the model [image, lidar]
        
    Returns:
        List of counter model names
    """
    counter_models = set()
    for idx in run_indices:
        if model_type == "image":
            counter_model = param_mapping.loc[param_mapping['run_index'] == idx, 'lidar_model'].iloc[0]
            counter_models.add(ast.literal_eval(counter_model)[0])
        else:  # lidar
            counter_model = param_mapping.loc[param_mapping['run_index'] == idx, 'image_model'].iloc[0]
            counter_models.add(counter_model)
    return sorted(list(counter_models))

def read_layer_timings(output_dir: str, run_index: int) -> pd.DataFrame:
    """
    Read layer timing information for a specific run.
    
    Args:
        output_dir: Path to the output directory
        run_index: Index of the run to read
        
    Returns:
        DataFrame containing layer timing information
    """
    timing_file = os.path.join(output_dir, f'layer_timings_{run_index}.csv')
    if not os.path.exists(timing_file):
        raise FileNotFoundError(f"layer_timings_{run_index}.csv not found in {output_dir}")
    return pd.read_csv(timing_file)

def read_delays(output_dir: str, run_index: int) -> pd.DataFrame:
    """
    Read delay information for a specific run.
    
    Args:
        output_dir: Path to the output directory
        run_index: Index of the run to read
        
    Returns:
        DataFrame containing delay information
    """
    delay_file = os.path.join(output_dir, f'delays_{run_index}.csv')
    if not os.path.exists(delay_file):
        raise FileNotFoundError(f"delays_{run_index}.csv not found in {output_dir}")
    return pd.read_csv(delay_file)

def remove_outliers(data: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """
    Remove outliers from data using the IQR method.
    
    Args:
        data: Input data array
        iqr_factor: Factor to multiply IQR by for outlier detection (default: 1.5)
        
    Returns:
        Data array with outliers removed
    """
    if len(data) < 4:  # Need at least 4 points for meaningful quartiles
        return data
        
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def load_model_complexities(model_complexity_dir: str) -> Dict[str, float]:
    """
    Load model complexities from the model complexity directory.
    
    Args:
        model_complexity_dir: Directory containing model complexity files
        
    Returns:
        Dictionary mapping model names to their complexity values
    """
    complexities = {}
    param_mapping = read_param_mapping(model_complexity_dir)
    model_complexity_param_mapping = param_mapping.groupby('model')['run_index'].apply(list).to_dict()
    
    for model_name, run_indices in model_complexity_param_mapping.items():
        try:
            model_complexity_info = f'flops_analysis_test_run_{run_indices[0]}_flops.csv'
            complexity_file = os.path.join(model_complexity_dir, model_complexity_info)
            if os.path.exists(complexity_file):
                model_complexity = pd.read_csv(complexity_file)['FLOPs'].mean()
                model_name = ast.literal_eval(model_name)[0]
                complexities[model_name] = model_complexity
        except Exception as e:
            print(f"Warning: Could not load complexity for {model_name}: {e}")
    
    return complexities

def normalize_complexity_for_plotting(complexity: float, max_range: float = 200.0) -> float:
    """
    Normalize complexity to a fixed range for plotting.
    
    Args:
        complexity: Raw complexity value (FLOPs)
        max_range: Maximum value for normalization (default: 200.0)
        
    Returns:
        Normalized complexity value between 0 and max_range
    """
    if complexity > 0:
        log_complexity = np.log10(complexity)
        # Use a more dynamic range to better show differences
        # Assuming log complexity values typically range from 6 to 12
        # Scale more aggressively to show differences
        normalized = (log_complexity - 6) / (12 - 6) * max_range
        # Apply some non-linear scaling to emphasize differences
        normalized = normalized ** 0.7  # Makes differences more visible
        return max(0, min(max_range, normalized))  # Clamp to 0-max_range
    return 0.0

def get_model_short_name(model_name: str) -> str:
    """
    Get the short display name for a model.
    
    Args:
        model_name: Full model name
        
    Returns:
        Short display name
    """
    return model_name_mappings.get(model_name, model_name)

def safe_layer_name(layer: str) -> str:
    """
    Convert layer name to a safe filename.
    
    Args:
        layer: Layer name
        
    Returns:
        Safe filename string
    """
    return layer.replace('/', '_').replace('\\', '_')

def create_output_directory(output_dir: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True) 