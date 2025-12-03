import sys
sys.path.append('/home/mg/pdnn/pPerf/perf_ws/src/p_perf')
from p_perf.post_process.kernel_processor import KernelProcessor
from p_perf.post_process.layer_processor import LayerProcessor
from p_perf.config.constant import (
    generate_image_lidar_combinations,
    generate_image_lidar_seg_combinations,
    scenes,
    get_abbreviated_name,
    get_abbreviated_scene,
    get_full_model_name,
    get_full_scene_token
)

import os
import pandas as pd
from pathlib import Path
import ast

# output_dirs = ["full_stack_1I1L1S-1", 'det_1I1L-1', 'det_1L1S-1']
output_dirs = ['full_stack_1I1L1S-MPS-LOW-10-2']
# for output_dir in output_dirs:
#     output_base = f"/home/mg/pdnn/pPerf/outputs/{output_dir}"

#     mapping_file = os.path.join(output_base, "full_stack_mapping.csv")
#     df = pd.read_csv(mapping_file)

#     kernel_processor = KernelProcessor(output_base)
    
#     print(f"\nProcessing {output_dir}...")
    
#     # Create plots from existing CSV files for each run
#     successful_plots = 0
#     failed_plots = 0
    
#     for i, row in df.iterrows():

#         if i!=5:
#             continue
        
#         run_index = row["run_index"]
        
#         print(f"Creating plots for run {run_index}...")
        
#         # Use existing CSV files to create plots
#         try:
#             kernel_processor.plot_multi_model_memcpy_vs_inference_subplots(
#                 run_index=run_index,
#                 csv_files=None,  # Auto-discover CSV files
#                 csv_dir=f"{output_base}/memcpy_analysis",
#                 output_dir=f"{output_base}/memcpy_analysis",
#                 save_plot=True,
#                 max_points_per_model=250
#             )
#             successful_plots += 1
#             print(f"  ✓ Successfully created plots for run {run_index}")
#         except Exception as e:
#             failed_plots += 1
#             print(f"  ✗ Failed to create plots for run {run_index}: {e}")
    
#     print(f"\nSummary for {output_dir}:")
#     print(f"  Successful plots: {successful_plots}")
#     print(f"  Failed plots: {failed_plots}")


# Code for full analysis (commented out)
for output_dir in output_dirs:
    output_base = f"/home/mg/pdnn/pPerf/outputs/{output_dir}"

    mapping_file = os.path.join(output_base, "full_stack_mapping.csv")
    df = pd.read_csv(mapping_file)

    layer_processor = LayerProcessor(output_base)
    kernel_processor = KernelProcessor(output_base)
    all_correlations = []

    for i, row in df.iterrows():
        if df.at[i, "status"] != "success":
            continue

        run_index = row["run_index"]

        
        layer_processor.modify_timings(
            run_index=run_index,
            add_inference=True,
            save_modified_timings=True
        )


    # Extract unique combinations from the CSV file (which now has abbreviated names)
    print(f"\nExtracting unique model combinations from CSV...")
    
    # Get unique image models (abbreviated)
    unique_image_models_abbrev = df['image_model'].dropna().unique().tolist()
    
    # Get unique lidar models (abbreviated tuples)
    unique_lidar_models_abbrev = []
    for lidar_str in df['lidar_model'].dropna().unique():
        try:
            lidar_tuple = ast.literal_eval(lidar_str)
            unique_lidar_models_abbrev.append(lidar_tuple)
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse lidar_model: {lidar_str}")
    
    # Get unique seg models (abbreviated tuples) - NEW
    unique_seg_models_abbrev = []
    if 'seg_model' in df.columns:
        for seg_str in df['seg_model'].dropna().unique():
            try:
                seg_tuple = ast.literal_eval(seg_str)
                unique_seg_models_abbrev.append(seg_tuple)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse seg_model: {seg_str}")
    
    print(f"Found {len(unique_image_models_abbrev)} unique image models")
    print(f"Found {len(unique_lidar_models_abbrev)} unique lidar models")
    print(f"Found {len(unique_seg_models_abbrev)} unique seg models")
    
    layer_name = 'e2e'
    metric = ['decode_head_h', 'decode_head_w']
    
    # Handle two cases: with or without seg models
    if unique_seg_models_abbrev:
        # Case 1: Full stack with segmentation (3 models)
        print(f"\nProcessing full stack combinations (image + lidar + seg)...")
        for abbreviated_img_model in unique_image_models_abbrev:
            for abbreviated_lidar_tuple in unique_lidar_models_abbrev:
                for abbreviated_seg_tuple in unique_seg_models_abbrev:
                    # Convert abbreviated names back to full names for processing
                    image_model = get_full_model_name(abbreviated_img_model)
                    lidar_model = get_full_model_name(abbreviated_lidar_tuple[0])
                    lidar_mode = abbreviated_lidar_tuple[1]
                    seg_model = get_full_model_name(abbreviated_seg_tuple[0])
                    seg_mode = abbreviated_seg_tuple[1]
                    
                    # Analyze all three models in this combination
                    for target_model in [image_model, lidar_model, seg_model]:
                        try:
                            # For filtering, use abbreviated names (what's in the CSV)
                            # For target_model (actual model file to load), use full name
                            layer_processor.plot_layer_boxplot(
                                layer_name=layer_name,
                                target_model=target_model,  # Full name for loading timing files
                                mapping_file='full_stack_mapping.csv',
                                metric=metric,
                                save_plot=True,
                                remove_outliers=False,
                                # Use abbreviated names for filtering (to match CSV)
                                image_model=abbreviated_img_model,
                                lidar_model=str(abbreviated_lidar_tuple),  # Keep as tuple string
                                seg_model=str(abbreviated_seg_tuple),  # Keep as tuple string
                            )
                            print(f"  ✓ Created plot for {target_model}")
                        except Exception as e:
                            print(f"  ✗ Failed to create plot for {target_model}: {e}")
    else:
        # Case 2: Detection only (2 models - backward compatible)
        print(f"\nProcessing detection-only combinations (image + lidar)...")
        for abbreviated_img_model in unique_image_models_abbrev:
            for abbreviated_lidar_tuple in unique_lidar_models_abbrev:
                # Convert abbreviated names back to full names for processing
                image_model = get_full_model_name(abbreviated_img_model)
                lidar_model = get_full_model_name(abbreviated_lidar_tuple[0])
                lidar_mode = abbreviated_lidar_tuple[1]
                
                # Analyze both image and lidar models in this combination
                for target_model in [image_model, lidar_model]:
                    try:
                        # For filtering, use abbreviated names (what's in the CSV)
                        # For target_model (actual model file to load), use full name
                        layer_processor.plot_layer_boxplot(
                            layer_name=layer_name,
                            target_model=target_model,  # Full name for loading timing files
                            mapping_file='full_stack_mapping.csv',
                            metric='scene',
                            save_plot=True,
                            remove_outliers=False,
                            # Use abbreviated names for filtering (to match CSV)
                            image_model=abbreviated_img_model,
                            lidar_model=str(abbreviated_lidar_tuple),  # Keep as tuple string
                        )
                        print(f"  ✓ Created plot for {target_model}")
                    except Exception as e:
                        print(f"  ✗ Failed to create plot for {target_model}: {e}")

    # model_names = ['faster-rcnn_r50_fpn_1x_coco', '3dssd_4x4_kitti-3d-car', 'cascade_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k']
    # layer_name = 'inference'
    # for model_name in model_names:
    #     layer_processor.plot_layer_boxplot(
    #         run_indices=[6, 7, 8],
    #         mapping_file=mapping_file,
    #         metric='run_index',
    #         layer_name=layer_name,
    #         model_name=model_name,
    #         save_plot=True,
    #     )

    #     _, correlations = kernel_processor.memcpy_analysis(
    #         run_index=run_index,
    #         output_dir=f"{output_base}/memcpy_analysis",
    #         create_plots=False,
    #         alignment_threshold_ms=2
    #     )

    #     kernel_processor.plot_multi_model_memcpy_vs_inference_subplots(
    #         run_index=run_index,
    #         csv_files=None,  # Auto-discover CSV files
    #         csv_dir=f"{output_base}/memcpy_analysis",
    #         output_dir=f"{output_base}/memcpy_analysis",
    #         save_plot=True,
    #         max_points_per_model=250
    #     )

    #     # Add run information to correlations and store in list
    #     for model_name, correlation in correlations.items():
    #         all_correlations.append({
    #             'run_index': run_index,
    #             'model_name': model_name,
    #             'correlation': correlation,
    #         })
        
    #     count += 1

    # Save all correlations to CSV
    # correlations_df = pd.DataFrame(all_correlations)
    # correlations_csv_path = f"{output_base}/memcpy_analysis/all_correlations.csv"
    # correlations_df.to_csv(correlations_csv_path, index=False)
    # print(f"Saved all correlations to: {correlations_csv_path}")
    # print(f"Total correlation records: {len(correlations_df)}")



# processor = KernelProcessor("outputs/det_1I1L-1")

# try:
#     processor.plot_target_model_multi_experiment(
#         target_model="faster-rcnn_r50_fpn_1x_coco",
#         model_mode="image_model",
#         mapping_file="full_stack_mapping.csv",
#         csv_dir="outputs/det_1I1L-1/memcpy_analysis",  # Directory with CSV files from memcpy_analysis
#         output_dir="outputs/det_1I1L-1/multi_experiment_plots",
#         save_plot=True,
#         max_points_per_experiment=100
#     )
#     print("Multi-run plot from CSV files created successfully!")
# except Exception as e:
#     print(f"Error creating multi-run plot from CSV: {e}")


# ==================== NEW RAIN RATE ANALYSIS WITH BASELINE ====================

# Rain rate analysis using model combinations and baseline comparison
# print("\n=== Rain Rate Analysis with Baseline Comparison ===")

# # Directories containing rain rate experiments and baseline
# output_base = "/home/mg/pdnn/pPerf/outputs"
# rain_output_dirs = [f'{output_base}/det_1I1L-2', 
#                 f'{output_base}/det_1L1S-2',
#                 f'{output_base}/seg-1_exps', 
#                 f'{output_base}/lidar_det-1_exps', 
#                 f'{output_base}/image_det-1_exps'
#                 ]


# for rain_output_dir in rain_output_dirs:
#     layer_processor = LayerProcessor(rain_output_dir)


#     mapping_file = os.path.join(rain_output_dir, "full_stack_mapping.csv")
#     df = pd.read_csv(mapping_file)

#     layer_processor = LayerProcessor(rain_output_dir)

#     for i, row in df.iterrows():
#         # if df.at[i, "status"] != "success":
#         #     continue
        
#         run_index = row["run_index"]

#         layer_processor.modify_timings(
#             run_index=run_index,
#             add_inference=True,
#             save_modified_timings=True
#         )

# for rain_output_dir in rain_output_dirs:
#     # Create layer processor instance for rain analysis
#     rain_layer_processor = LayerProcessor(rain_output_dir)

#     # Get models from constants and iterate through combinations
#     mapping_file = os.path.join(rain_output_dir, "full_stack_mapping.csv")
#     df = pd.read_csv(mapping_file)

#     # Extract unique models from the mapping CSV file
#     print(f"Extracting unique models from {mapping_file}...")
#     print(f"Available columns in CSV: {df.columns.tolist()}")
    
#     # Check and extract unique image models
#     if 'image_model' in df.columns:
#         image_models = df['image_model'].dropna().unique().tolist()
#         print(f"Found image models: {image_models}")
#     else:
#         image_models = [None]
#         print("No image_model column found - skipping image models")
    
#     # Check and extract unique lidar models (handle string representation of tuples)
#     if 'lidar_model' in df.columns:
#         lidar_model_strings = df['lidar_model'].dropna().unique()
#         lidar_models = []
#         for lidar_str in lidar_model_strings:
#             try:
#                 # Convert string representation of tuple back to tuple
#                 lidar_tuple = ast.literal_eval(lidar_str)
#                 lidar_models.append(lidar_tuple)
#             except (ValueError, SyntaxError):
#                 # If it's not a tuple string, treat as simple string
#                 lidar_models.append(lidar_str)
#         print(f"Found lidar models: {lidar_models}")
#     else:
#         lidar_models = [None]
#         print("No lidar_model column found - skipping lidar models")
    
#     # Check and extract unique seg models
#     if 'seg_model' in df.columns:
#         seg_model_strings = df['seg_model'].dropna().unique()
#         seg_models = []
#         for seg_str in seg_model_strings:
#             try:
#                 # Convert string representation of tuple back to tuple
#                 seg_tuple = ast.literal_eval(seg_str)
#                 seg_models.append(seg_tuple)
#             except (ValueError, SyntaxError):
#                 # If it's not a tuple string, treat as simple string
#                 seg_models.append(seg_str)
#         print(f"Found seg models: {seg_models}")
#     else:
#         seg_models = [None]  # Use None for cases where there's no seg model
#         print("No seg_model column found - skipping seg models")

#     # Check if we have any valid models to process
#     has_valid_models = any([
#         image_models != [None],
#         lidar_models != [None], 
#         seg_models != [None]
#     ])
    
#     if not has_valid_models:
#         print("No valid models found in any category - skipping this directory")
#         continue

#     success_count = 0
#     total_count = 0

#     # Go through every model combination from the CSV
#     for image_model in image_models:
#         for lidar_model in lidar_models:
#             for seg_model in seg_models:
                
#                 # Get the run indices for this combination
#                 from p_perf.post_process.utils import get_run_indices_by_models
#                 rain_mapping_path = os.path.join(rain_output_dir, "full_stack_mapping.csv")
                
#                 # print(f"Getting run indices for {image_model}, {lidar_model}, {seg_model}")
#                 run_indices = get_run_indices_by_models(
#                     rain_mapping_path,
#                     image_model=image_model,
#                     lidar_model=lidar_model,
#                     seg_model=seg_model,
#                 )
                
#                 if not run_indices:
#                     print(f"No successful runs found for combination: {image_model}, {lidar_model}, {seg_model}")
#                     continue
                    
#                 print(f"\nFound {len(run_indices)} runs for combination:")
#                 print(f"  Image: {image_model}")
#                 print(f"  LiDAR: {lidar_model}")
#                 print(f"  Seg: {seg_model}")
#                 print(f"  Run indices: {run_indices}")
                
#                 # Analyze each model type in this combination
#                 models_to_analyze = []
                
#                 # Only add image model if it exists and is not None
#                 if image_model is not None:
#                     models_to_analyze.append(('image', image_model))
                
#                 # Only add lidar model if it exists and is not None
#                 if lidar_model is not None:
#                     models_to_analyze.append(('lidar', lidar_model[0]))
                
#                 # Only add seg model if it exists and is not None
#                 if seg_model is not None:
#                     models_to_analyze.append(('seg', seg_model[0]))
                
#                 # Skip if no valid models to analyze
#                 if not models_to_analyze:
#                     print(f"  No valid models to analyze for this combination - skipping")
#                     continue
                
#                 for model_type, target_model in models_to_analyze:
#                     try:
#                         total_count += 1
#                         print(f"  Analyzing {model_type} model: {target_model}")
                        
#                         rain_layer_processor.plot_rain_rate_analysis(
#                             rain_output_dir=rain_output_dir,
#                             baseline_output_dir=None,  # No baseline comparison for now
#                             target_model=target_model,
#                             layer_name='inference',
#                             image_model=image_model,
#                             lidar_model=lidar_model,
#                             seg_model=seg_model,
#                             save_plot=True,
#                             remove_outliers=True,
#                             figsize=(12, 8)
#                         )
                        
#                         print(f"    ✓ Successfully analyzed {target_model}")
#                         success_count += 1
                        
#                     except Exception as e:
#                         print(f"    ✗ Failed to analyze {target_model}: {e}")

#     print(f"\n=== Rain Rate Analysis Complete ===")
#     print(f"Successfully generated {success_count} out of {total_count} analyses")


# ==================== EXAMPLE: Using Image-Lidar Combinations ====================
# 
# # Generate all image-lidar model combinations
# image_lidar_combinations = generate_image_lidar_combinations()
# 
# print(f"\nGenerated {len(image_lidar_combinations)} image-lidar combinations")
# 
# # Example: Iterate through combinations and perform analysis
# for image_model, lidar_model in image_lidar_combinations:
#     lidar_model_name = lidar_model[0]  # Extract model name from tuple
#     lidar_model_mode = lidar_model[1]  # Extract mode from tuple
#     
#     print(f"\nProcessing combination:")
#     print(f"  Image model: {image_model}")
#     print(f"  Lidar model: {lidar_model_name} (mode: {lidar_model_mode})")
#     
#     # Example analysis for each combination
#     # layer_processor.plot_layer_boxplot(
#     #     layer_name='inference',
#     #     target_model=image_model,  # or lidar_model_name
#     #     mapping_file='full_stack_mapping.csv',
#     #     metric='some_metric',
#     #     save_plot=True,
#     #     image_model=image_model,
#     #     lidar_model=lidar_model_name,
#     # )
# 
# ==================== EXAMPLE: Using Image-Lidar-Seg Combinations ====================
# 
# # Generate all image-lidar-seg model combinations
# full_combinations = generate_image_lidar_seg_combinations()
# 
# print(f"\nGenerated {len(full_combinations)} full model combinations")
# 
# # Example: Iterate through combinations
# for image_model, lidar_model, seg_model in full_combinations:
#     lidar_model_name = lidar_model[0]
#     lidar_model_mode = lidar_model[1]
#     seg_model_name = seg_model[0]
#     seg_model_mode = seg_model[1]
#     
#     print(f"\nProcessing combination:")
#     print(f"  Image model: {image_model}")
#     print(f"  Lidar model: {lidar_model_name} (mode: {lidar_model_mode})")
#     print(f"  Seg model: {seg_model_name} (mode: {seg_model_mode})")
#     
#     # Example analysis for each combination
#     # layer_processor.plot_layer_boxplot(
#     #     layer_name='inference',
#     #     target_model=image_model,
#     #     mapping_file='full_stack_mapping.csv',
#     #     metric='some_metric',
#     #     save_plot=True,
#     #     image_model=image_model,
#     #     lidar_model=lidar_model_name,
#     #     seg_model=seg_model_name,
#     # )

