#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import ast
import glob

# Import common utilities
from utils import (
    image_models, lidar_models,
    read_param_mapping, get_run_indices, get_counter_models, 
    read_layer_timings, remove_outliers, get_model_short_name,
    set_plot_text_sizes
)

# Scene information mapping
scene_info = {
    '2f0e54af35964a3fb347359836bec035': 'Clear',
    '3a1850241080418b88dcee97c7d17ed7': 'Crowded City',
    'bcb05cffb5814973a6cff4fbdca2b99b': 'Highway',
    '74d03e3b673f4a72a3f76e1b0c7840bd': 'Highway',
    '32185f91e68f4069ab3cdd2f4f1a4ff1': 'Suburban',
    '93032908bafd484ea10a39b0d5d3fff6': 'Suburban',
    '2f0e54af35964a3fb347359836bec035_rainrate25': 'Rainy 25',
    '2f0e54af35964a3fb347359836bec035_rainrate50': 'Rainy 50',
    '2f0e54af35964a3fb347359836bec035_rainrate100': 'Rainy 100'
}

FIELD = 'Elapsed Time'

def load_baseline_data(baseline_dir: str, baseline_scene: str, model_name: str, model_type: str) -> Dict[str, List[float]]:
    """
    Load baseline data for a specific model and scene.
    
    Args:
        baseline_dir: Directory containing baseline experiment results
        baseline_scene: Scene ID for baseline data
        model_name: Name of the model
        model_type: Type of model [image, lidar]
        
    Returns:
        Dictionary containing baseline timing data for different layers
    """
    try:
        param_mapping = read_param_mapping(baseline_dir)
        run_indices = get_run_indices(param_mapping, model_name, model_type, scene=baseline_scene)
        
        if not run_indices:
            print(f"No baseline data found for {model_name} in scene {baseline_scene}")
            return {}
        
        # Use the first run index
        run_index = run_indices[0]
        timing_data = read_layer_timings(baseline_dir, run_index)
        
        baseline_data = {}
        # Get e2e timing data
        e2e_times = timing_data[
            (timing_data['Layer'] == 'e2e') & 
            (timing_data['Model'] == model_name)
        ][FIELD].values
        
        if len(e2e_times) > 0:
            baseline_data['e2e'] = remove_outliers(e2e_times)
        
        return baseline_data
        
    except Exception as e:
        print(f"Error loading baseline data for {model_name}: {e}")
        return {}

def get_model_pair_name(image_model: str, lidar_model: str) -> str:
    """Generate a readable name for a model pair."""
    img_short = get_model_short_name(image_model)
    lidar_short = get_model_short_name(lidar_model)
    return f"{img_short} + {lidar_short}"

def get_scene_category(scene_id: str) -> str:
    """Get the category/type of a scene."""
    scene_name = scene_info.get(scene_id, scene_id)
    if 'Crowded City' in scene_name:
        return 'Crowded City'
    elif 'Highway' in scene_name:
        return 'Highway'
    elif 'Suburban' in scene_name:
        return 'Suburban'
    elif 'Clear' in scene_name:
        return 'Clear Weather'
    elif 'Rainy' in scene_name:
        return 'Rainy Weather'
    elif 'Day' in scene_name:
        return 'Day Lighting'
    elif 'Night' in scene_name:
        return 'Night Lighting'
    else:
        return 'Unknown'

def create_frame_count_report(multi_tenant_dir: str, output_dir: str, 
                            selected_scenes: List[str] = None,
                            baseline_dir: str = None, baseline_scene: str = None):
    """
    Create a comprehensive report of frame counts for each model in different scenes.
    
    Args:
        multi_tenant_dir: Directory containing multi-tenant experiment results
        output_dir: Directory to save the report
        selected_scenes: List of scene IDs to analyze (if None, analyze all available)
        baseline_dir: Directory containing baseline experiment results (optional)
        baseline_scene: Scene ID for baseline data (optional)
    """
    print("Creating frame count report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read parameter mapping
    param_mapping = read_param_mapping(multi_tenant_dir)
    
    # Filter successful runs only
    successful_runs = param_mapping[param_mapping['status'] == 'success'].copy()
    
    if successful_runs.empty:
        print("No successful runs found!")
        return
    
    # Filter by selected scenes if specified
    if selected_scenes:
        successful_runs = successful_runs[successful_runs['scene'].isin(selected_scenes)]
    
    # Get unique model pairs
    model_pairs = set()
    for _, row in successful_runs.iterrows():
        image_model = row['image_model']
        lidar_model = ast.literal_eval(row['lidar_model'])[0]
        if image_model != 'none' and lidar_model != 'none':
            model_pairs.add((image_model, lidar_model))
    
    # Initialize report data structures
    report_data = {
        'model_pairs': {},
        'individual_models': {},
        'summary': {
            'total_runs': len(successful_runs),
            'total_model_pairs': len(model_pairs),
            'scenes_analyzed': list(successful_runs['scene'].unique()),
            'missing_data': []
        }
    }
    
    # Analyze each model pair
    for image_model, lidar_model in model_pairs:
        pair_name = get_model_pair_name(image_model, lidar_model)
        print(f"  Analyzing frame counts for: {pair_name}")
        
        # Filter runs for this specific model pair
        pair_runs = successful_runs[
            (successful_runs['image_model'] == image_model) &
            (successful_runs['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == lidar_model))
        ]
        
        pair_data = {
            'image_model': image_model,
            'lidar_model': lidar_model,
            'scenes': {}
        }
        
        for _, run in pair_runs.iterrows():
            scene_id = run['scene']
            run_index = run['run_index']
            
            try:
                timing_data = read_layer_timings(multi_tenant_dir, run_index)
                
                # Get frame counts for each model
                image_e2e_times = timing_data[
                    (timing_data['Layer'] == 'e2e') & 
                    (timing_data['Model'] == image_model)
                ][FIELD].values
                
                lidar_e2e_times = timing_data[
                    (timing_data['Layer'] == 'e2e') & 
                    (timing_data['Model'] == lidar_model)
                ][FIELD].values
                
                if scene_id not in pair_data['scenes']:
                    pair_data['scenes'][scene_id] = {
                        'runs': [],
                        'total_image_frames': 0,
                        'total_lidar_frames': 0,
                        'avg_image_frames_per_run': 0,
                        'avg_lidar_frames_per_run': 0
                    }
                
                run_data = {
                    'run_index': run_index,
                    'image_frames': len(image_e2e_times),
                    'lidar_frames': len(lidar_e2e_times),
                    'image_frames_after_outlier_removal': len(remove_outliers(image_e2e_times)) if len(image_e2e_times) > 0 else 0,
                    'lidar_frames_after_outlier_removal': len(remove_outliers(lidar_e2e_times)) if len(lidar_e2e_times) > 0 else 0
                }
                
                pair_data['scenes'][scene_id]['runs'].append(run_data)
                pair_data['scenes'][scene_id]['total_image_frames'] += len(image_e2e_times)
                pair_data['scenes'][scene_id]['total_lidar_frames'] += len(lidar_e2e_times)
                
            except Exception as e:
                print(f"    Error processing run {run_index} for scene {scene_id}: {e}")
                report_data['summary']['missing_data'].append({
                    'model_pair': pair_name,
                    'scene': scene_id,
                    'run_index': run_index,
                    'error': str(e)
                })
                continue
        
        # Calculate averages for each scene
        for scene_id, scene_data in pair_data['scenes'].items():
            num_runs = len(scene_data['runs'])
            if num_runs > 0:
                scene_data['avg_image_frames_per_run'] = scene_data['total_image_frames'] / num_runs
                scene_data['avg_lidar_frames_per_run'] = scene_data['total_lidar_frames'] / num_runs
        
        report_data['model_pairs'][pair_name] = pair_data
    
    # Analyze individual models
    all_models = set()
    for image_model, lidar_model in model_pairs:
        all_models.add(image_model)
        all_models.add(lidar_model)
    
    for model_name in all_models:
        model_name_short = get_model_short_name(model_name)
        print(f"  Analyzing individual model: {model_name_short}")
        
        # Determine model type
        model_type = "image" if model_name in [pair[0] for pair in model_pairs] else "lidar"
        
        # Get run indices for this model
        model_run_indices = get_run_indices(param_mapping, model_name, model_type)
        
        # Get counter models
        counter_models = get_counter_models(param_mapping, model_run_indices, model_type)
        
        model_data = {
            'model_name': model_name,
            'model_type': model_type,
            'counter_models': {}
        }
        
        for counter_model in counter_models:
            counter_model_short = get_model_short_name(counter_model)
            
            # Filter runs for this specific counter model
            counter_runs = successful_runs.copy()
            if model_type == "image":
                counter_runs = counter_runs[
                    (counter_runs['image_model'] == model_name) &
                    (counter_runs['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == counter_model))
                ]
            else:  # lidar
                counter_runs = counter_runs[
                    (counter_runs['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == model_name)) &
                    (counter_runs['image_model'] == counter_model)
                ]
            
            counter_data = {
                'scenes': {}
            }
            
            for _, run in counter_runs.iterrows():
                scene_id = run['scene']
                run_index = run['run_index']
                
                try:
                    timing_data = read_layer_timings(multi_tenant_dir, run_index)
                    
                    # Get e2e timing data for both models
                    model_e2e_times = timing_data[
                        (timing_data['Layer'] == 'e2e') & 
                        (timing_data['Model'] == model_name)
                    ][FIELD].values

                    # Determine counter model name for this run
                    if model_type == "image":
                        counter_model_name = counter_model
                    else:
                        counter_model_name = counter_model

                    counter_e2e_times = timing_data[
                        (timing_data['Layer'] == 'e2e') & 
                        (timing_data['Model'] == counter_model_name)
                    ][FIELD].values

                    # Only include if both have data
                    if len(model_e2e_times) > 0 and len(counter_e2e_times) > 0:
                        if scene_id not in scene_data:
                            scene_data[scene_id] = []
                        scene_data[scene_id].extend(model_e2e_times)
                
                except Exception as e:
                    print(f"    Error processing run {run_index} for scene {scene_id}: {e}")
                    continue
            
            # Calculate averages for each scene
            for scene_id, scene_data in counter_data['scenes'].items():
                num_runs = len(scene_data['runs'])
                if num_runs > 0:
                    scene_data['avg_frames_per_run'] = scene_data['total_frames'] / num_runs
            
            model_data['counter_models'][counter_model_short] = counter_data
        
        report_data['individual_models'][model_name_short] = model_data
    
    # Analyze baseline data if available
    if baseline_dir and baseline_scene:
        print(f"  Analyzing baseline data for scene: {baseline_scene}")
        report_data['baseline'] = analyze_baseline_frame_counts(baseline_dir, baseline_scene, model_pairs)
    
    # Write the report
    write_frame_count_report(report_data, output_dir)
    
    print(f"Frame count report saved to {output_dir}")

def analyze_baseline_frame_counts(baseline_dir: str, baseline_scene: str, model_pairs: set) -> Dict:
    """Analyze frame counts in baseline data."""
    try:
        baseline_param_mapping = read_param_mapping(baseline_dir)
        baseline_runs = baseline_param_mapping[
            (baseline_param_mapping['scene'] == baseline_scene) &
            (baseline_param_mapping['status'] == 'success')
        ]
        
        baseline_data = {
            'scene': baseline_scene,
            'model_pairs': {},
            'individual_models': {}
        }
        
        # Analyze model pairs in baseline
        for image_model, lidar_model in model_pairs:
            pair_name = get_model_pair_name(image_model, lidar_model)
            
            pair_runs = baseline_runs[
                (baseline_runs['image_model'] == image_model) &
                (baseline_runs['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == lidar_model))
            ]
            
            if not pair_runs.empty:
                run = pair_runs.iloc[0]  # Use first successful run
                run_index = run['run_index']
                
                try:
                    timing_data = read_layer_timings(baseline_dir, run_index)
                    
                    image_e2e_times = timing_data[
                        (timing_data['Layer'] == 'e2e') & 
                        (timing_data['Model'] == image_model)
                    ][FIELD].values
                    
                    lidar_e2e_times = timing_data[
                        (timing_data['Layer'] == 'e2e') & 
                        (timing_data['Model'] == lidar_model)
                    ][FIELD].values
                    
                    baseline_data['model_pairs'][pair_name] = {
                        'run_index': run_index,
                        'image_frames': len(image_e2e_times),
                        'lidar_frames': len(lidar_e2e_times),
                        'image_frames_after_outlier_removal': len(remove_outliers(image_e2e_times)) if len(image_e2e_times) > 0 else 0,
                        'lidar_frames_after_outlier_removal': len(remove_outliers(lidar_e2e_times)) if len(lidar_e2e_times) > 0 else 0
                    }
                    
                except Exception as e:
                    print(f"    Error processing baseline run {run_index} for {pair_name}: {e}")
        
        return baseline_data
        
    except Exception as e:
        print(f"Error analyzing baseline frame counts: {e}")
        return {}

def write_frame_count_report(report_data: Dict, output_dir: str):
    """Write the frame count report to files."""
    
    # Write detailed report
    report_file = os.path.join(output_dir, 'frame_count_report.txt')
    with open(report_file, 'w') as f:
        f.write("Frame Count Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total runs analyzed: {report_data['summary']['total_runs']}\n")
        f.write(f"Total model pairs: {report_data['summary']['total_model_pairs']}\n")
        f.write(f"Scenes analyzed: {', '.join(report_data['summary']['scenes_analyzed'])}\n")
        f.write(f"Missing data entries: {len(report_data['summary']['missing_data'])}\n\n")
        
        # Model pairs analysis
        f.write("MODEL PAIRS ANALYSIS\n")
        f.write("-" * 25 + "\n\n")
        
        for pair_name, pair_data in report_data['model_pairs'].items():
            f.write(f"Model Pair: {pair_name}\n")
            f.write(f"  Image Model: {get_model_short_name(pair_data['image_model'])}\n")
            f.write(f"  Lidar Model: {get_model_short_name(pair_data['lidar_model'])}\n")
            f.write("  Scene Analysis:\n")
            
            for scene_id, scene_data in pair_data['scenes'].items():
                scene_name = scene_info.get(scene_id, scene_id)
                f.write(f"    {scene_name}:\n")
                f.write(f"      Runs: {len(scene_data['runs'])}\n")
                f.write(f"      Total Image Frames: {scene_data['total_image_frames']}\n")
                f.write(f"      Total Lidar Frames: {scene_data['total_lidar_frames']}\n")
                f.write(f"      Avg Image Frames/Run: {scene_data['avg_image_frames_per_run']:.1f}\n")
                f.write(f"      Avg Lidar Frames/Run: {scene_data['avg_lidar_frames_per_run']:.1f}\n")
                
                # Show individual run details
                for run_data in scene_data['runs']:
                    f.write(f"        Run {run_data['run_index']}: {run_data['image_frames']} image, {run_data['lidar_frames']} lidar frames\n")
            f.write("\n")
        
        # Individual models analysis
        f.write("INDIVIDUAL MODELS ANALYSIS\n")
        f.write("-" * 30 + "\n\n")
        
        for model_name, model_data in report_data['individual_models'].items():
            f.write(f"Model: {model_name} ({model_data['model_type']})\n")
            
            for counter_model, counter_data in model_data['counter_models'].items():
                f.write(f"  With Counter Model: {counter_model}\n")
                
                for scene_id, scene_data in counter_data['scenes'].items():
                    scene_name = scene_info.get(scene_id, scene_id)
                    f.write(f"    {scene_name}:\n")
                    f.write(f"      Runs: {len(scene_data['runs'])}\n")
                    f.write(f"      Total Frames: {scene_data['total_frames']}\n")
                    f.write(f"      Avg Frames/Run: {scene_data['avg_frames_per_run']:.1f}\n")
                    
                    # Show individual run details
                    for run_data in scene_data['runs']:
                        f.write(f"        Run {run_data['run_index']}: {run_data['frames']} frames\n")
                f.write("\n")
        
        # Baseline analysis
        if 'baseline' in report_data:
            f.write("BASELINE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            baseline_scene_name = scene_info.get(report_data['baseline']['scene'], report_data['baseline']['scene'])
            f.write(f"Baseline Scene: {baseline_scene_name}\n\n")
            
            f.write("Model Pairs in Baseline:\n")
            for pair_name, pair_data in report_data['baseline']['model_pairs'].items():
                f.write(f"  {pair_name}:\n")
                f.write(f"    Run Index: {pair_data['run_index']}\n")
                f.write(f"    Image Frames: {pair_data['image_frames']}\n")
                f.write(f"    Lidar Frames: {pair_data['lidar_frames']}\n")
                f.write(f"    Image Frames (after outlier removal): {pair_data['image_frames_after_outlier_removal']}\n")
                f.write(f"    Lidar Frames (after outlier removal): {pair_data['lidar_frames_after_outlier_removal']}\n")
            f.write("\n")
        
        # Missing data
        if report_data['summary']['missing_data']:
            f.write("MISSING DATA\n")
            f.write("-" * 15 + "\n")
            for missing in report_data['summary']['missing_data']:
                f.write(f"  {missing['model_pair']} - Scene {missing['scene']} - Run {missing['run_index']}: {missing['error']}\n")
    
    # Write CSV summary
    csv_file = os.path.join(output_dir, 'frame_count_summary.csv')
    with open(csv_file, 'w') as f:
        f.write("Model_Pair,Scene,Image_Frames,Lidar_Frames,Image_Frames_After_Outliers,Lidar_Frames_After_Outliers,Runs\n")
        
        for pair_name, pair_data in report_data['model_pairs'].items():
            for scene_id, scene_data in pair_data['scenes'].items():
                scene_name = scene_info.get(scene_id, scene_id)
                f.write(f"{pair_name},{scene_name},{scene_data['total_image_frames']},{scene_data['total_lidar_frames']},")
                f.write(f"{sum(run['image_frames_after_outlier_removal'] for run in scene_data['runs'])},{sum(run['lidar_frames_after_outlier_removal'] for run in scene_data['runs'])},{len(scene_data['runs'])}\n")
    
    print(f"Detailed report: {report_file}")
    print(f"CSV summary: {csv_file}")

def analyze_scene_influence_on_model_pairs(multi_tenant_dir: str, output_dir: str, 
                                         selected_scenes: List[str] = None,
                                         selected_model_pairs: List[Tuple[str, str]] = None,
                                         baseline_dir: str = None,
                                         baseline_scene: str = None,
                                         analyze_individual_models: bool = False,
                                         generate_frame_report: bool = True):
    """
    Analyze the influence of scenes on model pairs.
    
    Args:
        multi_tenant_dir: Directory containing multi-tenant experiment results
        output_dir: Directory to save analysis results
        selected_scenes: List of scene IDs to analyze (if None, analyze all available)
        selected_model_pairs: List of (image_model, lidar_model) tuples to analyze
        baseline_dir: Directory containing baseline experiment results (optional)
        baseline_scene: Scene ID for baseline data (optional)
        analyze_individual_models: Whether to also analyze individual models (default: False)
        generate_frame_report: Whether to generate frame count report (default: True)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate frame count report if requested
    if generate_frame_report:
        print("Generating frame count report...")
        create_frame_count_report(multi_tenant_dir, output_dir, selected_scenes, baseline_dir, baseline_scene)
        print()
    
    # Read parameter mapping
    param_mapping = read_param_mapping(multi_tenant_dir)
    
    # Filter successful runs only
    successful_runs = param_mapping[param_mapping['status'] == 'success'].copy()
    
    if successful_runs.empty:
        print("No successful runs found!")
        return
    
    # Filter by selected scenes if specified
    if selected_scenes:
        successful_runs = successful_runs[successful_runs['scene'].isin(selected_scenes)]
    
    # Get unique model pairs
    model_pairs = set()
    for _, row in successful_runs.iterrows():
        image_model = row['image_model']
        lidar_model = ast.literal_eval(row['lidar_model'])[0]
        if image_model != 'none' and lidar_model != 'none':
            model_pairs.add((image_model, lidar_model))
    
    # Filter by selected model pairs if specified
    if selected_model_pairs:
        model_pairs = model_pairs.intersection(set(selected_model_pairs))
    
    print(f"Found {len(model_pairs)} model pairs to analyze")
    print(f"Available scenes: {successful_runs['scene'].unique()}")
    
    # Create comprehensive analysis for each model pair
    for image_model, lidar_model in model_pairs:
        print(f"\nAnalyzing model pair: {get_model_pair_name(image_model, lidar_model)}")
        
        # Filter runs for this specific model pair
        pair_runs = successful_runs[
            (successful_runs['image_model'] == image_model) &
            (successful_runs['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == lidar_model))
        ]
        
        if pair_runs.empty:
            print(f"No runs found for {get_model_pair_name(image_model, lidar_model)}")
            continue
        
        # Create analysis for this model pair
        analyze_single_model_pair(pair_runs, multi_tenant_dir, output_dir, image_model, lidar_model,
                                baseline_dir, baseline_scene)
        
        # Analyze individual models if requested
        if analyze_individual_models:
            print(f"Analyzing individual models for pair: {get_model_pair_name(image_model, lidar_model)}")
            
            # Analyze image model
            analyze_individual_model(multi_tenant_dir, output_dir, image_model, "image", 
                                   selected_scenes, baseline_dir, baseline_scene)
            
            # Analyze lidar model
            analyze_individual_model(multi_tenant_dir, output_dir, lidar_model, "lidar", 
                                   selected_scenes, baseline_dir, baseline_scene)

def analyze_single_model_pair(pair_runs: pd.DataFrame, multi_tenant_dir: str, 
                            output_dir: str, image_model: str, lidar_model: str,
                            baseline_dir: str = None, baseline_scene: str = None):
    """Analyze a single model pair across different scenes."""
    
    pair_name = get_model_pair_name(image_model, lidar_model)
    safe_pair_name = pair_name.replace(' ', '_').replace('+', '_plus_')
    
    # Collect data for each scene
    scene_data = {}
    scene_categories = {}
    
    for _, run in pair_runs.iterrows():
        scene_id = run['scene']
        run_index = run['run_index']
        
        try:
            # Read timing data
            timing_data = read_layer_timings(multi_tenant_dir, run_index)
            
            # Get e2e timing data for each model (same approach as model_variation_analysis)
            image_e2e_times = timing_data[
                (timing_data['Layer'] == 'e2e') & 
                (timing_data['Model'] == image_model)
            ][FIELD].values
            
            lidar_e2e_times = timing_data[
                (timing_data['Layer'] == 'e2e') & 
                (timing_data['Model'] == lidar_model)
            ][FIELD].values
            
            # Only include data if both models have e2e times
            if len(image_e2e_times) > 0 and len(lidar_e2e_times) > 0:
                if scene_id not in scene_data:
                    scene_data[scene_id] = {'image': [], 'lidar': []}
                    scene_categories[scene_id] = get_scene_category(scene_id)
                
                # Remove outliers before adding to the data
                image_e2e_clean = remove_outliers(image_e2e_times)
                lidar_e2e_clean = remove_outliers(lidar_e2e_times)
                
                scene_data[scene_id]['image'].extend(image_e2e_clean)
                scene_data[scene_id]['lidar'].extend(lidar_e2e_clean)
            
        except Exception as e:
            print(f"Error processing run {run_index} for scene {scene_id}: {e}")
            continue
    
    if not scene_data:
        print(f"No valid data found for {pair_name}")
        return
    
    # Load baseline data if available
    baseline_data = {}
    if baseline_dir and baseline_scene:
        print(f"Loading baseline data for scene {baseline_scene}")
        baseline_data['image'] = load_baseline_data(baseline_dir, baseline_scene, image_model, "image")
        baseline_data['lidar'] = load_baseline_data(baseline_dir, baseline_scene, lidar_model, "lidar")
    
    # Create visualization
    create_scene_influence_plot(scene_data, scene_categories, pair_name, safe_pair_name, output_dir, 
                               image_model, lidar_model, baseline_data, baseline_scene)
    
    # Create statistical analysis
    create_statistical_analysis(scene_data, scene_categories, pair_name, safe_pair_name, output_dir, 
                               baseline_data, baseline_scene)

def analyze_individual_model(multi_tenant_dir: str, output_dir: str, model_name: str, model_type: str,
                           selected_scenes: List[str], baseline_dir: str = None, baseline_scene: str = None):
    """
    Analyze individual model performance across different scenes and counter models.
    
    Args:
        multi_tenant_dir: Directory containing multi-tenant experiment results
        output_dir: Directory to save analysis results
        model_name: Name of the model to analyze
        model_type: Type of model [image, lidar]
        selected_scenes: List of scene IDs to analyze
        baseline_dir: Directory containing baseline experiment results (optional)
        baseline_scene: Scene ID for baseline data (optional)
    """
    model_name_short = get_model_short_name(model_name)
    safe_model_name = model_name_short.replace(' ', '_')
    
    # Create output directory for this model
    model_output_dir = os.path.join(output_dir, f'individual_models/{safe_model_name}')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Read parameter mapping
    param_mapping = read_param_mapping(multi_tenant_dir)
    
    # Filter successful runs
    successful_runs = param_mapping[param_mapping['status'] == 'success'].copy()
    
    # Filter by selected scenes
    if selected_scenes:
        successful_runs = successful_runs[successful_runs['scene'].isin(selected_scenes)]
    
    # Get run indices for this model
    model_run_indices = get_run_indices(param_mapping, model_name, model_type)
    
    # Get counter models
    counter_models = get_counter_models(param_mapping, model_run_indices, model_type)
    
    print(f"Analyzing {model_name_short} ({model_type})")
    print(f"Counter models: {counter_models}")
    print(f"Scenes: {selected_scenes}")
    
    # Collect all data for combined plot
    all_counter_data = {}
    all_baseline_data = {}
    
    # Create analysis for each counter model
    for counter_model in counter_models:
        if "pv_rcnn" in counter_model:
            continue
        print(f"  Analyzing with counter model: {get_model_short_name(counter_model)}")
        
        # Load baseline data for this specific counter model if available
        counter_baseline_data = {}
        if baseline_dir and baseline_scene:
            try:
                # Read baseline parameter mapping
                baseline_param_mapping = read_param_mapping(baseline_dir)
                
                # Find runs where this model runs with this specific counter model in the baseline scene
                if model_type == "image":
                    baseline_runs = baseline_param_mapping[
                        (baseline_param_mapping['image_model'] == model_name) &
                        (baseline_param_mapping['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == counter_model)) &
                        (baseline_param_mapping['scene'] == baseline_scene) &
                        (baseline_param_mapping['status'] == 'success')
                    ]
                else:  # lidar
                    baseline_runs = baseline_param_mapping[
                        (baseline_param_mapping['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == model_name)) &
                        (baseline_param_mapping['image_model'] == counter_model) &
                        (baseline_param_mapping['scene'] == baseline_scene) &
                        (baseline_param_mapping['status'] == 'success')
                    ]
                
                if not baseline_runs.empty:
                    # Use the first successful run
                    baseline_run_index = baseline_runs.iloc[0]['run_index']
                    baseline_timing_data = read_layer_timings(baseline_dir, baseline_run_index)
                    
                    # Get e2e timing data for this model
                    baseline_e2e_times = baseline_timing_data[
                        (baseline_timing_data['Layer'] == 'e2e') & 
                        (baseline_timing_data['Model'] == model_name)
                    ][FIELD].values
                    
                    if len(baseline_e2e_times) > 0:
                        counter_baseline_data['e2e'] = remove_outliers(baseline_e2e_times)
                        print(f"    Loaded baseline data for {model_name_short} with {get_model_short_name(counter_model)}")
                    else:
                        print(f"    No baseline e2e data found for {model_name_short} with {get_model_short_name(counter_model)}")
                else:
                    print(f"    No baseline runs found for {model_name_short} with {get_model_short_name(counter_model)}")
                    
            except Exception as e:
                print(f"    Error loading baseline data for {model_name_short} with {get_model_short_name(counter_model)}: {e}")
        
        # Store baseline data for this counter model
        all_baseline_data[counter_model] = counter_baseline_data
        
        # Filter runs for this specific counter model
        counter_runs = successful_runs.copy()
        if model_type == "image":
            counter_runs = counter_runs[
                (counter_runs['image_model'] == model_name) &
                (counter_runs['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == counter_model))
            ]
        else:  # lidar
            counter_runs = counter_runs[
                (counter_runs['lidar_model'].apply(lambda x: ast.literal_eval(x)[0] == model_name)) &
                (counter_runs['image_model'] == counter_model)
            ]
        
        if counter_runs.empty:
            print(f"    No runs found for {model_name_short} with {get_model_short_name(counter_model)}")
            continue
        
        # Collect data for each scene
        scene_data = {}
        
        for _, run in counter_runs.iterrows():
            scene_id = run['scene']
            run_index = run['run_index']
            
            try:
                timing_data = read_layer_timings(multi_tenant_dir, run_index)
                
                # Get e2e timing data for both models
                model_e2e_times = timing_data[
                    (timing_data['Layer'] == 'e2e') & 
                    (timing_data['Model'] == model_name)
                ][FIELD].values

                # Determine counter model name for this run
                if model_type == "image":
                    counter_model_name = counter_model
                else:
                    counter_model_name = counter_model

                counter_e2e_times = timing_data[
                    (timing_data['Layer'] == 'e2e') & 
                    (timing_data['Model'] == counter_model_name)
                ][FIELD].values

                # Only include if both have data
                if len(model_e2e_times) > 0 and len(counter_e2e_times) > 0:
                    if scene_id not in scene_data:
                        scene_data[scene_id] = []
                    # Remove outliers before adding to the data
                    model_e2e_clean = remove_outliers(model_e2e_times)
                    scene_data[scene_id].extend(model_e2e_clean)
                
            except Exception as e:
                print(f"    Error processing run {run_index} for scene {scene_id}: {e}")
                continue
        
        if not scene_data:
            print(f"    No valid data found for {model_name_short} with {get_model_short_name(counter_model)}")
            continue
        
        # Store data for combined plot
        all_counter_data[counter_model] = scene_data
    
    # Create combined plot with all counter models
    if all_counter_data:
        create_combined_model_plot(all_counter_data, model_name, model_type, 
                                 model_output_dir, all_baseline_data, baseline_scene)

def create_combined_model_plot(all_counter_data: Dict, model_name: str, model_type: str, 
                             output_dir: str, all_baseline_data: Dict = None, baseline_scene: str = None):
    """
    Create a combined boxplot for a model showing all counter models together.
    Scenes are organized so that scenes belonging to the same counter model are grouped closer together.
    Each counter model includes its own baseline data.
    
    Args:
        all_counter_data: Dictionary mapping counter models to their scene data
        model_name: Name of the model being analyzed
        model_type: Type of model [image, lidar]
        output_dir: Directory to save the plot
        all_baseline_data: Dictionary mapping counter models to their baseline data (optional)
        baseline_scene: Baseline scene ID (optional)
    """
    model_name_short = get_model_short_name(model_name)
    
    # Prepare data for plotting with organized grouping
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    # Organize data by counter model, then by scene within each counter model
    counter_models = list(all_counter_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(counter_models)))
    
    for i, counter_model in enumerate(counter_models):
        counter_model_short = get_model_short_name(counter_model)
        scene_data = all_counter_data[counter_model]
        
        # Add baseline for this counter model if available
        if all_baseline_data and counter_model in all_baseline_data and 'e2e' in all_baseline_data[counter_model]:
            plot_data.append(all_baseline_data[counter_model]['e2e'])
            baseline_scene_name = scene_info.get(baseline_scene, baseline_scene)
            plot_labels.append(f"{baseline_scene_name}")
            plot_colors.append(colors[i])  # Same color as the counter model
        
        # Add all scenes for this counter model
        for scene_id, timing_data in scene_data.items():
            plot_data.append(timing_data)
            scene_name = scene_info.get(scene_id, scene_id)
            plot_labels.append(f"{scene_name}")
            plot_colors.append(colors[i])
    
    # Create the combined plot
    plt.figure(figsize=(len(plot_data) * 1.5, 8))
    box_plot = plt.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.6)
    
    # Color boxes by counter model
    for patch, color in zip(box_plot['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    

    
    plt.ylim(0, 150)
    
    # Add legend for counter models
    legend_elements = []
    for i, counter_model in enumerate(counter_models):
        counter_model_short = get_model_short_name(counter_model)
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=counter_model_short))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name_short}_combined_all_counter_models.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Created combined plot: {model_name_short}_combined_all_counter_models.png")



def create_scene_influence_plot(scene_data: Dict, scene_categories: Dict, 
                              pair_name: str, safe_pair_name: str, output_dir: str,
                              image_model: str, lidar_model: str, 
                              baseline_data: Dict = None, baseline_scene: str = None):
    """Create the main scene influence plot."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Prepare data for plotting
    scenes = list(scene_data.keys())
    scene_labels = [scene_info.get(scene, scene) for scene in scenes]
    categories = [scene_categories[scene] for scene in scenes]
    
    # Add baseline if available
    if baseline_data and baseline_scene:
        baseline_scene_name = scene_info.get(baseline_scene, baseline_scene)
        scenes.insert(0, baseline_scene)
        scene_labels.insert(0, f"{baseline_scene_name}")
        categories.insert(0, get_scene_category(baseline_scene))
    
    # Image model boxplot
    image_data = []
    if baseline_data and baseline_scene and 'image' in baseline_data and 'e2e' in baseline_data['image']:
        image_data.append(baseline_data['image']['e2e'])
    
    image_data.extend([scene_data[scene]['image'] for scene in scene_data.keys()])
    
    box1 = ax1.boxplot(image_data, labels=scene_labels, patch_artist=True, widths=0.6)
    
    # Color boxes by scene category
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(categories))))
    category_colors = {cat: colors[i] for i, cat in enumerate(set(categories))}
    
    for patch, category in zip(box1['boxes'], categories):
        patch.set_facecolor(category_colors[category])
        patch.set_alpha(0.7)
    
        ax1.set_title('Image Model')
    ax1.set_ylim(0, 125)
    ax1.grid(True, alpha=0.3)

    
    # Add category legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=cat) 
                      for cat, color in category_colors.items()]
    # ax1.legend(handles=legend_elements, loc='upper right')
    
    # Lidar model boxplot
    lidar_data = []
    if baseline_data and baseline_scene and 'lidar' in baseline_data and 'e2e' in baseline_data['lidar']:
        lidar_data.append(baseline_data['lidar']['e2e'])
    
    lidar_data.extend([scene_data[scene]['lidar'] for scene in scene_data.keys()])
    
    box2 = ax2.boxplot(lidar_data, labels=scene_labels, patch_artist=True, widths=0.6)
    
    for patch, category in zip(box2['boxes'], categories):
        patch.set_facecolor(category_colors[category])
        patch.set_alpha(0.7)
    
        ax2.set_title('Lidar Model')
    ax2.set_ylim(0, 150)
    ax2.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{safe_pair_name}_scene_influence.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_analysis(scene_data: Dict, scene_categories: Dict, 
                              pair_name: str, safe_pair_name: str, output_dir: str,
                              baseline_data: Dict = None, baseline_scene: str = None):
    """Create statistical analysis of scene influence."""
    
    stats_file = os.path.join(output_dir, f'{safe_pair_name}_scene_statistics.txt')
    
    with open(stats_file, 'w') as f:
        f.write(f"Scene Influence Analysis for {pair_name}\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall statistics
        all_image_times = []
        all_lidar_times = []
        for scene_data_values in scene_data.values():
            all_image_times.extend(scene_data_values['image'])
            all_lidar_times.extend(scene_data_values['lidar'])
        
        f.write("Overall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Image Model - Mean: {np.mean(all_image_times):.2f}ms, Std: {np.std(all_image_times):.2f}ms\n")
        f.write(f"Lidar Model - Mean: {np.mean(all_lidar_times):.2f}ms, Std: {np.std(all_lidar_times):.2f}ms\n")
        f.write(f"Total Inference Time - Mean: {np.mean(all_image_times) + np.mean(all_lidar_times):.2f}ms\n\n")
        
        # Baseline comparison if available
        if baseline_data and baseline_scene:
            f.write("Baseline Comparison:\n")
            f.write("-" * 20 + "\n")
            baseline_scene_name = scene_info.get(baseline_scene, baseline_scene)
            f.write(f"Baseline Scene: {baseline_scene_name}\n")
            
            if 'image' in baseline_data and 'e2e' in baseline_data['image']:
                baseline_img_mean = np.mean(baseline_data['image']['e2e'])
                f.write(f"Baseline Image Model - Mean: {baseline_img_mean:.2f}ms\n")
            
            if 'lidar' in baseline_data and 'e2e' in baseline_data['lidar']:
                baseline_lidar_mean = np.mean(baseline_data['lidar']['e2e'])
                f.write(f"Baseline Lidar Model - Mean: {baseline_lidar_mean:.2f}ms\n")
            
            f.write("\n")
        
        # Per-scene statistics
        f.write("Per-Scene Statistics:\n")
        f.write("-" * 20 + "\n")
        
        for scene_id, data in scene_data.items():
            scene_name = scene_info.get(scene_id, scene_id)
            category = scene_categories[scene_id]
            
            img_mean = np.mean(data['image'])
            img_std = np.std(data['image'])
            lidar_mean = np.mean(data['lidar'])
            lidar_std = np.std(data['lidar'])
            total_mean = img_mean + lidar_mean
            
            f.write(f"\nScene: {scene_name} ({category})\n")
            f.write(f"  Image Model: {img_mean:.2f} ± {img_std:.2f}ms\n")
            f.write(f"  Lidar Model: {lidar_mean:.2f} ± {lidar_std:.2f}ms\n")
            f.write(f"  Total: {total_mean:.2f}ms\n")
        
        # Scene category analysis
        f.write("\n\nScene Category Analysis:\n")
        f.write("-" * 25 + "\n")
        
        category_stats = {}
        for scene_id, data in scene_data.items():
            category = scene_categories[scene_id]
            if category not in category_stats:
                category_stats[category] = {'image': [], 'lidar': []}
            
            category_stats[category]['image'].extend(data['image'])
            category_stats[category]['lidar'].extend(data['lidar'])
        
        for category, stats in category_stats.items():
            img_mean = np.mean(stats['image'])
            img_std = np.std(stats['image'])
            lidar_mean = np.mean(stats['lidar'])
            lidar_std = np.std(stats['lidar'])
            
            f.write(f"\n{category}:\n")
            f.write(f"  Image Model: {img_mean:.2f} ± {img_std:.2f}ms\n")
            f.write(f"  Lidar Model: {lidar_mean:.2f} ± {lidar_std:.2f}ms\n")
            f.write(f"  Total: {img_mean + lidar_mean:.2f}ms\n")



if __name__ == "__main__":
    # Set plot parameters
    set_plot_text_sizes()
    
    # Configuration
    multi_tenant_dir = "outputs/rain_aug"
    output_dir = "tools/scene_influence_analysis_env"
    
    # Baseline configuration (optional)
    baseline_dir = "outputs/Image_Lidar_full" # Set to None if no baseline data
    baseline_scene = "2f0e54af35964a3fb347359836bec035"  # Set to None if no baseline data
    
    # Define scenes to analyze (you can modify this list)
    selected_scenes = [
        # '2f0e54af35964a3fb347359836bec035',  # Crowded City
        # 'bcb05cffb5814973a6cff4fbdca2b99b',  # Highway
        # '32185f91e68f4069ab3cdd2f4f1a4ff1',  # Suburban
        # '7ca5e90766dc4b7bb6a9b3b0d95c99da',  # Clear Weather
        # '5fc7c5dfc56b4971b14005bc53f69908'   # Rainy Weather
        '2f0e54af35964a3fb347359836bec035_rainrate25',
        '2f0e54af35964a3fb347359836bec035_rainrate50',
        '2f0e54af35964a3fb347359836bec035_rainrate100'
    ]
    
    # Define specific model pairs to analyze (optional - if None, analyze all)
    selected_model_pairs = None  # Set to None to analyze all pairs
    
    # Whether to analyze individual models
    analyze_individual_models = False  # Set to True to enable individual model analysis
    
    # Whether to generate frame count report
    generate_frame_report = True  # Set to True to generate frame count report
    
    print("Starting Scene Influence Analysis...")
    print(f"Analyzing scenes: {[scene_info.get(s, s) for s in selected_scenes]}")
    if baseline_dir and baseline_scene:
        print(f"Baseline directory: {baseline_dir}")
        print(f"Baseline scene: {scene_info.get(baseline_scene, baseline_scene)}")
    if analyze_individual_models:
        print("Individual model analysis: ENABLED")
    if generate_frame_report:
        print("Frame count report: ENABLED")
    
    # Run the analysis
    analyze_scene_influence_on_model_pairs(
        multi_tenant_dir=multi_tenant_dir,
        output_dir=output_dir,
        selected_scenes=selected_scenes,
        selected_model_pairs=selected_model_pairs,
        baseline_dir=baseline_dir,
        baseline_scene=baseline_scene,
        analyze_individual_models=analyze_individual_models,
        generate_frame_report=generate_frame_report
    )


    print(f"\nAnalysis complete! Results saved to {output_dir}") 