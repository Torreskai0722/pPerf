import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import ast

# Import common utilities
from utils import (
    model_name_mappings, image_models, lidar_models,
    read_param_mapping, get_run_indices, get_counter_models, 
    read_layer_timings, read_delays, remove_outliers,
    load_model_complexities, normalize_complexity_for_plotting,
    get_model_short_name, safe_layer_name, create_output_directory,
    set_plot_text_sizes
)

Y_AXIS_RANGE = 150

def create_counter_model_bar_plot(counter_data: Dict[str, np.ndarray], model_complexities: Dict[str, float], 
                                 output_dir: str, model_name_short: str, layer: str):
    """
    Create a bar plot showing mean, range, and model complexity for each counter model.
    
    Args:
        counter_data: Dictionary mapping counter model names to their elapsed time data
        model_complexities: Dictionary mapping model names to complexity values
        output_dir: Directory to save the plot
        model_name_short: Short name of the primary model
        layer: Layer name for the plot
    """
    if not counter_data:
        return
    
    # Prepare data for the bar plot
    counter_models = []
    mean_values = []
    range_values = []
    complexity_values = []
    
    for counter_model, elapsed_data in counter_data.items():
        if len(elapsed_data) > 0:
            counter_models.append(get_model_short_name(counter_model))
            mean_values.append(np.mean(elapsed_data))
            range_values.append(np.max(elapsed_data) - np.min(elapsed_data))
            
            # Get complexity value
            if counter_model in model_complexities:
                complexity = model_complexities[counter_model]
                # Normalize complexity to a reasonable range for plotting
                normalized_complexity = normalize_complexity_for_plotting(complexity, 200.0)
                complexity_values.append(normalized_complexity)
            else:
                complexity_values.append(0)
    
    if not counter_models:
        return
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(20, 9))
    
    x = np.arange(len(counter_models))
    width = 0.25
    
    # Plot bars
    bars1 = ax.bar(x - width, mean_values, width, label='Mean Time (ms)', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x, range_values, width, label='Range (ms)', color='lightcoral', alpha=0.8)
    bars3 = ax.bar(x + width, complexity_values, width, label='Model Complexity (normalized)', color='lightgreen', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add labels for non-zero values
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=20)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Customize the plot
    ax.set_xlabel('Counter Models')
    ax.set_ylabel('Values')
    ax.set_xticks(x)
    ax.set_xticklabels(counter_models)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    safe_layer = safe_layer_name(layer)
    plt.savefig(os.path.join(output_dir, f'{model_name_short}_{safe_layer}_counter_analysis.png'))
    plt.close()

def analyze_model_variations(baseline_timings: pd.DataFrame, multi_tenant_dir: str,
                           output_dir: str, model_name: str, model_type: str, 
                           model_complexities: Dict[str, float] = None, **filter_params):
    """
    Analyze variations in model performance in multi-tenant scenarios.
    Shows the model's baseline performance and its performance with each counter model.
    
    Args:
        baseline_timings: DataFrame containing the model's baseline layer timings
        multi_tenant_dir: Directory containing multi-tenant experiment results
        output_dir: Directory to save analysis results
        model_name: Name of the model being analyzed
        model_type: Type of the model being analyzed, [image, lidar]
        model_complexities: Dictionary mapping model names to complexity values
        **filter_params: Additional parameters to filter runs (e.g., scene, depth, queue sizes)
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    model_name_short = get_model_short_name(model_name)
    
    # Read param mapping from multi-tenant directory
    param_mapping = read_param_mapping(multi_tenant_dir)
    
    # Get run indices for this model with additional filters
    model_run_indices = get_run_indices(param_mapping, model_name, model_type, **filter_params)
    print(f"Found model run indices: {model_run_indices}")
    
    # Get list of counter models
    counter_models = get_counter_models(param_mapping, model_run_indices, model_type)
    print(f"Found counter models: {counter_models}")
    
    # Get unique layers
    layer_names = baseline_timings['Layer'].unique()
    
    # Create statistics file
    stats_file = os.path.join(output_dir, f'{model_name}_layer_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Layer Statistics for {model_name_short} ({model_type})\n")
        if filter_params:
            f.write("Filter Parameters:\n")
            for param, value in filter_params.items():
                f.write(f"  {param}: {value}\n")
        f.write("=" * 80 + "\n\n")
        
        # For each layer
        for layer in layer_names:
            if 'decode' in layer:
                continue
            plt.figure(figsize=(15, 8))
            
            # Get baseline data and remove outliers
            baseline_data = baseline_timings[baseline_timings['Layer'] == layer]['Elapsed Time'].values
            if len(baseline_data) == 0:
                continue
            
            # Remove outliers from baseline data
            baseline_data = remove_outliers(baseline_data)
            if len(baseline_data) == 0:
                continue
            
            # Get baseline GPU metrics data and remove outliers
            baseline_gpu_idle = baseline_timings[baseline_timings['Layer'] == layer]['GPU Wait Time'].values
            baseline_gpu_runtime = baseline_timings[baseline_timings['Layer'] == layer]['GPU Computation Time'].values
            baseline_gpu_turnaround = baseline_timings[baseline_timings['Layer'] == layer]['GPU Turnaround Time'].values
            
            baseline_gpu_idle = remove_outliers(baseline_gpu_idle)
            baseline_gpu_runtime = remove_outliers(baseline_gpu_runtime)
            baseline_gpu_turnaround = remove_outliers(baseline_gpu_turnaround)
            
            # Prepare data and labels for boxplot
            plot_data = [baseline_data]
            plot_labels = ['Baseline']
            
            # Prepare data for GPU metrics
            gpu_idle_data = []
            gpu_runtime_data = []
            gpu_turnaround_data = []
            timeline_data = {}  # For timeline graph
            
            # For each counter model
            counter_data = {}
            # Get run indices where this specific counter model is paired with our target model
            for run_idx in model_run_indices:
                try:
                    if model_type == "image":
                        counter_model = param_mapping.loc[param_mapping['run_index'] == run_idx, 'lidar_model'].iloc[0]
                        counter_model = ast.literal_eval(counter_model)[0]
                        if 'pv_rcnn_8xb2-80e_kitti-3d-3class' in counter_model:
                            continue
                    else:
                        counter_model = param_mapping.loc[param_mapping['run_index'] == run_idx, 'image_model'].iloc[0]

                    timing = read_layer_timings(multi_tenant_dir, run_idx)

                    # Get data for this model when running with this counter model
                    layer_elapsed = timing[
                        (timing['Layer'] == layer) & 
                        (timing['Model'] == model_name)
                    ]['Elapsed Time'].values

                    layer_GPU_idle = timing[
                        (timing['Layer'] == layer) & 
                        (timing['Model'] == model_name)
                    ]['GPU Wait Time'].values

                    layer_GPU_runtime = timing[
                        (timing['Layer'] == layer) & 
                        (timing['Model'] == model_name)
                    ]['GPU Computation Time'].values

                    layer_start_time = timing[
                        (timing['Layer'] == layer) & 
                        (timing['Model'] == model_name)
                    ]['Start Timestamp'].values

                    layer_GPU_turnaround = timing[
                        (timing['Layer'] == layer) & 
                        (timing['Model'] == model_name)
                    ]['GPU Turnaround Time'].values

                    if len(layer_elapsed) > 0:
                        counter_data[counter_model] = layer_elapsed, layer_GPU_idle, layer_GPU_runtime, layer_start_time, layer_GPU_turnaround
                        
                        # Store timeline data
                        if counter_model not in timeline_data:
                            timeline_data[counter_model] = []
                        for i, (elapsed, start_time) in enumerate(zip(layer_elapsed, layer_start_time)):
                            timeline_data[counter_model].append((start_time, elapsed))
                    else:
                        print(f"No data found for {counter_model} at layer {layer}, index {run_idx}")
                except FileNotFoundError:
                    continue
            
            if counter_data:
                # Remove outliers from counter model data and prepare for different plots
                for counter_model, (elapsed, gpu_idle, gpu_runtime, start_time, gpu_turnaround) in counter_data.items():
                    # Remove outliers from elapsed time
                    elapsed_clean = remove_outliers(np.array(elapsed))
                    if len(elapsed_clean) > 0:
                        plot_data.append(elapsed_clean)
                        plot_labels.append(get_model_short_name(counter_model))
                    
                    # Remove outliers from GPU metrics
                    gpu_idle_clean = remove_outliers(np.array(gpu_idle))
                    gpu_runtime_clean = remove_outliers(np.array(gpu_runtime))
                    gpu_turnaround_clean = remove_outliers(np.array(gpu_turnaround))

                    if len(gpu_idle_clean) > 0:
                        gpu_idle_data.append(gpu_idle_clean)
                    if len(gpu_runtime_clean) > 0:
                        gpu_runtime_data.append(gpu_runtime_clean)
                    if len(gpu_turnaround_clean) > 0:
                        gpu_turnaround_data.append(gpu_turnaround_clean)
            
            # Create main elapsed time boxplot (e2e)
            plt.figure(figsize=(15, 8))
            plt.boxplot(plot_data, labels=plot_labels, boxprops=dict(linewidth=3),
                        whiskerprops=dict(linewidth=3), capprops=dict(linewidth=3), medianprops=dict(linewidth=3))
            # plt.title(f'Model: {model_name_short}')
            plt.ylabel('e2e')
            plt.ylim(0, Y_AXIS_RANGE)  # Fixed y-axis range for e2e plots

            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Create counter model bar plot - extract just the elapsed time data (with outliers removed)
            elapsed_counter_data = {}
            for counter_model, (elapsed, _, _, _, _) in counter_data.items():
                # Remove outliers from elapsed time data to match box plot behavior
                elapsed_clean = remove_outliers(np.array(elapsed))
                if len(elapsed_clean) > 0:
                    elapsed_counter_data[counter_model] = elapsed_clean
            create_counter_model_bar_plot(elapsed_counter_data, model_complexities, output_dir, model_name_short, layer)
            
            plt.tight_layout()
            
            # Save elapsed time plot
            safe_layer = safe_layer_name(layer)
            plt.savefig(os.path.join(output_dir, f'{model_name_short}_{safe_layer}_elapsed_variation.png'))
            plt.close()
            
            # Create GPU idle time boxplot (no global range)
            if gpu_idle_data:
                plt.figure(figsize=(15, 8))
                gpu_idle_labels = ['Baseline'] + [get_model_short_name(counter_model) for counter_model in counter_data.keys()]
                plt.boxplot([baseline_gpu_idle] + gpu_idle_data, labels=gpu_idle_labels, boxprops=dict(linewidth=3),
                            whiskerprops=dict(linewidth=3), capprops=dict(linewidth=3), medianprops=dict(linewidth=3))
                # plt.title(f'Model: {model_name_short}')
                plt.ylabel('e2e')
    
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{model_name_short}_{safe_layer}_gpu_idle_variation.png'))
                plt.close()
            
            # Create GPU runtime boxplot (no global range)
            if gpu_runtime_data:
                plt.figure(figsize=(15, 8))
                gpu_runtime_labels = ['Baseline'] + [get_model_short_name(counter_model) for counter_model in counter_data.keys()]
                plt.boxplot([baseline_gpu_runtime] + gpu_runtime_data, labels=gpu_runtime_labels, boxprops=dict(linewidth=3),
                            whiskerprops=dict(linewidth=3), capprops=dict(linewidth=3), medianprops=dict(linewidth=3))
                # plt.title(f'Model: {model_name_short}')
                plt.ylabel('e2e')
    
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{model_name_short}_{safe_layer}_gpu_runtime_variation.png'))
                plt.close()

            if gpu_turnaround_data:
                plt.figure(figsize=(15, 8))
                gpu_turnaround_labels = ['Baseline'] + [get_model_short_name(counter_model) for counter_model in counter_data.keys()]
                plt.boxplot([baseline_gpu_turnaround] + gpu_turnaround_data, labels=gpu_turnaround_labels, boxprops=dict(linewidth=3),
                            whiskerprops=dict(linewidth=3), capprops=dict(linewidth=3), medianprops=dict(linewidth=3))
                # plt.title(f'Model: {model_name_short}')
                plt.ylabel('e2e')
                plt.ylim(0, Y_AXIS_RANGE)  # Fixed y-axis range for GPU turnaround plots
    
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{model_name_short}_{safe_layer}_gpu_turnaround_variation.png'))
                plt.close()

            # Create timeline graph (e2e)
            if timeline_data:
                plt.figure(figsize=(28, 8))
                colors = plt.cm.Set3(np.linspace(0, 1, len(timeline_data) + 1))
                
                # Plot baseline timeline (if available)
                baseline_start_times = baseline_timings[baseline_timings['Layer'] == layer]['Start Timestamp'].values
                baseline_start_times = [start_time - baseline_start_times.min() for start_time in baseline_start_times]
                baseline_elapsed = baseline_timings[baseline_timings['Layer'] == layer]['Elapsed Time'].values
                baseline_iqr = np.percentile(baseline_elapsed, 75) - np.percentile(baseline_elapsed, 28)
                if len(baseline_start_times) > 0:
                    plt.scatter(baseline_start_times, baseline_elapsed, 
                              label='Baseline', color=colors[0], alpha=1, s=28)
                    plt.plot(baseline_start_times, baseline_elapsed, 
                                color=colors[0], alpha=1)  # line connecting points
                
                # Plot counter model timelines
                for i, (counter_model, timeline_points) in enumerate(timeline_data.items(), 1):
                    if timeline_points:
                        start_times, elapsed_times = zip(*timeline_points)
                        start_times = np.array(start_times)  # Convert to numpy array
                        elapsed_times = np.array(elapsed_times)
                        start_times = [start_time - start_times.min() for start_time in start_times]
                        plt.scatter(start_times, elapsed_times, 
                                    label=get_model_short_name(counter_model), 
                                    color=colors[i], alpha=1, s=10)

                        plt.plot(start_times, elapsed_times, 
                                color=colors[i], alpha=1)  # line connecting points

                
                # plt.title(f'Model: {get_model_short_name(model_name)}')
                plt.xlabel('Start Timestamp')
                plt.ylabel('e2e')
                plt.ylim(0, Y_AXIS_RANGE)  # Fixed y-axis range for e2e plots
                
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{model_name}_{safe_layer}_timeline.png'))
                plt.close()
            
            # Write statistics
            f.write(f"\nLayer: {layer}\n")
            f.write("-" * 40 + "\n")
            
            # Baseline statistics
            baseline_mean = np.mean(baseline_data)
            baseline_std = np.std(baseline_data)
            f.write("Baseline Statistics (outliers removed):\n")
            f.write(f"Mean: {baseline_mean:.3f} ms\n")
            f.write(f"Std: {baseline_std:.3f} ms\n")
            f.write(f"Min: {np.min(baseline_data):.3f} ms\n")
            f.write(f"Max: {np.max(baseline_data):.3f} ms\n")
            f.write(f"Range: {np.max(baseline_data) - np.min(baseline_data):.3f} ms\n")
            f.write(f"Sample size: {len(baseline_data)}\n")
            
            # Add complexity information if available
            if model_complexities and model_name in model_complexities:
                complexity = model_complexities[model_name]
                f.write(f"Model Complexity: {complexity:.2e} FLOPs\n")
            
            # Counter model statistics and sensitivity metrics
            f.write("\nSensitivity Analysis:\n")
            f.write("-" * 40 + "\n")
            
            for i, counter_model in enumerate(counter_models):
                if i + 1 < len(plot_data):  # +1 because plot_data[0] is baseline
                    counter_data = plot_data[i + 1]
                    counter_mean = np.mean(counter_data)
                    counter_std = np.std(counter_data)
                    
                    # Calculate sensitivity metrics
                    mean_increase = ((counter_mean - baseline_mean) / baseline_mean) * 100
                    std_increase = ((counter_std - baseline_std) / baseline_std) * 100
                    max_increase = ((np.max(counter_data) - np.max(baseline_data)) / np.max(baseline_data)) * 100
                    min_increase = ((np.min(counter_data) - np.min(baseline_data)) / np.min(baseline_data)) * 100
                    
                    # Calculate stability metrics
                    cv_baseline = (baseline_std / baseline_mean) * 100  # Coefficient of variation
                    cv_counter = (counter_std / counter_mean) * 100
                    stability_change = cv_counter - cv_baseline
                    
                    f.write(f"\nWith {get_model_short_name(counter_model)}:\n")
                    f.write(f"Mean: {counter_mean:.3f} ms (Change: {mean_increase:+.1f}%)\n")
                    f.write(f"Std: {counter_std:.3f} ms (Change: {std_increase:+.1f}%)\n")
                    f.write(f"Min: {np.min(counter_data):.3f} ms (Change: {min_increase:+.1f}%)\n")
                    f.write(f"Max: {np.max(counter_data):.3f} ms (Change: {max_increase:+.1f}%)\n")
                    f.write(f"Sample size: {len(counter_data)}\n")
                    
                    # Add counter model complexity if available
                    if model_complexities and counter_model in model_complexities:
                        counter_complexity = model_complexities[counter_model]
                        f.write(f"Counter Model Complexity: {counter_complexity:.2e} FLOPs\n")
                    
                    # Write sensitivity metrics
                    f.write("\nSensitivity Metrics:\n")
                    f.write(f"Mean Time Increase: {mean_increase:+.1f}%\n")
                    f.write(f"Variability Increase: {std_increase:+.1f}%\n")
                    f.write(f"Stability Change: {stability_change:+.1f}% (negative is better)\n")
                    f.write(f"Max Impact: {max_increase:+.1f}%\n")
                    f.write(f"Min Impact: {min_increase:+.1f}%\n")
                    
                    # Add sensitivity rating
                    sensitivity = abs(mean_increase) + abs(std_increase) + abs(stability_change)
                    if sensitivity < 10:
                        rating = "Low"
                    elif sensitivity < 28:
                        rating = "Medium"
                    else:
                        rating = "High"
                    f.write(f"Sensitivity Rating: {rating} ({sensitivity:.1f})\n")
            
            f.write("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    # Set larger text sizes for all plots
    set_plot_text_sizes()
    
    # Example usage
    lidar_base_dir = "outputs/lidar_base"
    image_base_dir = "outputs/image_base"
    multi_tenant_dir = "outputs/ms"
    model_complexity_dir = "outputs/model_complexity"

    base_output_dir = "tools/results/reports_ms_10"

    # Load model complexities
    model_complexities = load_model_complexities(model_complexity_dir)
    print("Loaded model complexities:")
    for model_name, complexity in model_complexities.items():
        print(f"  {model_name}: {complexity:.2e}")

    for model_name in image_models:
        model_name_short = get_model_short_name(model_name)
        output_dir = f"{base_output_dir}/{model_name_short}"
        model_type = "image"
        baseline_param_mapping = read_param_mapping(image_base_dir)
        base_index = get_run_indices(baseline_param_mapping, model_name, model_type)
        print(f"\nModel: {model_name}, Base index: {base_index}")
        baseline_timings = pd.read_csv(os.path.join(image_base_dir, f"layer_timings_{base_index[0]}.csv"))

        print(f"Analyzing {model_name_short} ({model_type})", '-'*10)
        analyze_model_variations(baseline_timings, multi_tenant_dir, output_dir, model_name, model_type, 
                               model_complexities=model_complexities, scene="2f0e54af35964a3fb347359836bec035", lidar_priority=-1, image_priority=0) 


    for model_name in lidar_models:
        model_name = model_name[0]
        model_name_short = get_model_short_name(model_name)
        output_dir = f"{base_output_dir}/{model_name_short}"
        model_type = "lidar"
        baseline_param_mapping = read_param_mapping(lidar_base_dir)
        base_index = get_run_indices(baseline_param_mapping, model_name, model_type)
        print(f"Model: {model_name}, Base index: {base_index}")
        baseline_timings = pd.read_csv(os.path.join(lidar_base_dir, f"layer_timings_{base_index[0]}.csv"))

        print("\n", f"Analyzing {model_name_short} ({model_type})", "-"*10)
        analyze_model_variations(baseline_timings, multi_tenant_dir, output_dir, model_name, model_type, 
                               model_complexities=model_complexities, scene="2f0e54af35964a3fb347359836bec035", lidar_priority=-1, image_priority=0) 