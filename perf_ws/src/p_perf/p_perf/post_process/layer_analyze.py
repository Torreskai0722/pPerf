import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Optional, List

def filter_param_mapping(csv_path: str,
                         scenes: Optional[List[str]] = None,
                         image_models: Optional[List[str]] = None,
                         lidar_models: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # Extract actual lidar model name from the tuple string
    if 'lidar_model' in df.columns:
       df['lidar_model'] = df['lidar_model'].apply(lambda x: eval(x)[0])  # replace the original column

    if scenes:
        df = df[df['scene'].isin(scenes)]
    if image_models:
        df = df[df['image_model'].isin(image_models)]
    if lidar_models:
        df = df[df['lidar_model'].isin(lidar_models)]

    return df.reset_index(drop=True)


def summarize_variation(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(['Model', 'Layer'])['Elapsed Time'].agg(['mean', 'std', 'count'])
    stats['cv'] = stats['std'] / stats['mean']
    return stats.reset_index()


def identify_contributing_layers(df: pd.DataFrame, target_model: str) -> pd.DataFrame:
    if 'run_index' not in df.columns:
        raise ValueError("Missing 'run_index' column.")

    # Only use data from the target model
    model_df = df[df['Model'] == target_model]

    # Get E2E time per run for the target model
    e2e_times = model_df[model_df['Layer'] == 'e2e'].groupby('run_index')['Elapsed Time'].mean()

    # Get per-layer per-run elapsed times (excluding e2e itself)
    layer_times = model_df[model_df['Layer'] != 'e2e'].pivot_table(
        index='run_index', columns='Layer', values='Elapsed Time', aggfunc='mean'
    )

    # Add the e2e time for correlation/covariance computation
    layer_times['e2e'] = e2e_times

    # Compute covariance with e2e
    covs = layer_times.cov()['e2e'].drop('e2e')

    # Build result
    contrib_df = pd.DataFrame({
        'layer': covs.index,
        'cov_with_e2e': covs.values
    })
    contrib_df['abs_cov'] = contrib_df['cov_with_e2e'].abs()
    contrib_df.sort_values('abs_cov', ascending=False, inplace=True)

    return contrib_df.reset_index(drop=True)


def load_model_layers_across_runs(param_df: pd.DataFrame,
                                  target_model: str,
                                  data_dir: str) -> pd.DataFrame:
    all_rows = []
    for _, row in param_df.iterrows():
        run_index = row['run_index']
        timing_file = os.path.join(data_dir, f"layer_timings_{run_index}.csv")
        if not os.path.exists(timing_file):
            continue
        df = pd.read_csv(timing_file)
        df = df[df['Model'] == target_model]
        df['run_index'] = run_index
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def most_heavy_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify layers whose timing most correlates with E2E time, using normalized data.

    Args:
        df: A DataFrame for a single model, across multiple runs. Must contain 'run_index', 'Layer',
            'Elapsed Time', and 'Model' columns. Should already be filtered to one model.

    Returns:
        DataFrame of layers with correlation to e2e, sorted by absolute correlation.
    """
    if df.empty or 'run_index' not in df.columns:
        return pd.DataFrame(columns=['layer', 'corr_with_e2e', 'abs_corr'])

    # Get per-run E2E time
    e2e_times = df[df['Layer'] == 'e2e'].groupby('run_index')['Elapsed Time'].mean()

    # Get per-layer per-run elapsed times (excluding e2e)
    layer_times = df[df['Layer'] != 'e2e'].pivot_table(
        index='run_index', columns='Layer', values='Elapsed Time', aggfunc='mean'
    )


    # Add E2E column
    layer_times['e2e'] = e2e_times

    # Normalize each column (z-score)
    normalized = (layer_times - layer_times.mean()) / layer_times.std(ddof=0)


    # Compute correlation with E2E
    corrs = normalized.corr()['e2e'].drop('e2e')
    result = pd.DataFrame({
        'layer': corrs.index,
        'corr_with_e2e': corrs.values
    })
    result['abs_corr'] = result['corr_with_e2e'].abs()
    result.sort_values('abs_corr', ascending=False, inplace=True)
    return result.reset_index(drop=True)


def generate_pairwise_plots(param_df: pd.DataFrame,
                            fixed_type: str,
                            fixed_models: list,
                            scenes: list,
                            data_dir: str,
                            output_dir):
    """
    Generate E2E timing boxplots and timelines.

    Args:
        param_df: Filtered param_mapping DataFrame (must include run_index, lidar_model_name, image_model)
        fixed_type: "image" or "lidar"
        fixed_models: list of fixed model names (image or lidar)
        group_col: column to group on ("lidar_model_name" or "image_model")
        scenes: list of scene tokens to iterate over
        save_prefix: used in filenames
        data_dir: path where layer_timings_{idx}.csv and plots live
    """
    for fixed_model in fixed_models:
        for scene in scenes:
            if fixed_type == "image":
                filtered = param_df[(param_df["image_model"] == fixed_model) & (param_df["scene"] == scene)]
                group_col = "lidar_model"
            else:
                filtered = param_df[(param_df["lidar_model"] == fixed_model) & (param_df["scene"] == scene)]
                group_col = "image_model"

            records = []
            for _, row in filtered.iterrows():
                run_index = row["run_index"]
                varying_model = row[group_col]
                timing_file = os.path.join(data_dir, f"layer_timings_{run_index}.csv")
                if not os.path.exists(timing_file):
                    continue
                df = pd.read_csv(timing_file)
                e2e_df = df[(df["Layer"] == "e2e") & (df["Model"] == fixed_model)]
                for _, r in e2e_df.iterrows():
                    records.append({
                        "start_timestamp": r["Start Timestamp"],
                        "elapsed_time": r["Elapsed Time"],
                        group_col: varying_model
                    })

            if not records:
                continue

            e2e_df = pd.DataFrame(records)
            e2e_df["start_timestamp"] -= e2e_df["start_timestamp"].min()

            # Boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=e2e_df, x=group_col, y="elapsed_time")
            plt.title(f"Fixed {fixed_type.capitalize()}: {fixed_model}\nScene: {scene}")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"boxplot_{fixed_model}_{scene}.png"))
            plt.close()

            # Timeline
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=e2e_df, x="start_timestamp", y="elapsed_time", hue=group_col, marker='o')
            plt.title(f"Fixed {fixed_type.capitalize()}: {fixed_model}\nScene: {scene}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"timeline_{fixed_model}_{scene}.png"))
            plt.close()


def main():
    image_models = [    
        'faster-rcnn_r50_fpn_1x_coco',      # TWO STAGE
        'yolov3_d53_mstrain-608_273e_coco', # ONE STAGE ANCHOR
        'yolox_x_8x8_300e_coco',            # ONE STAGE ANCHOR FREE
        'centernet_r18-dcnv2_8xb16-crop512-140e_coco',   # ONE STAGE ANCHOR FREE
        'detr_r50_8xb2-150e_coco'           # TRANSFORMER
    ]

    lidar_models = [
        'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',
        'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d',
        'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d'
    ]

    scenes = ['c5224b9b454b4ded9b5d2d2634bbda8a', 'd25718445d89453381c659b9c8734939', '2fc3753772e241f2ab2cd16a784cc680']

    
    data_dir = '/home/mg/pdnn/pPerf/outputs/nusc_centerpoint'
    param_csv = f"{data_dir}/param_mapping.csv"

    # filtered_df = filter_param_mapping(
    #     csv_path=param_csv,
    #     scenes=scenes,
    #     image_models=image_models,
    #     lidar_models=lidar_models 
    # )

    # output_dir = f'{data_dir}/model_pair'
    # os.makedirs(output_dir, exist_ok=True)

    # generate_pairwise_plots(filtered_df,
    #                         fixed_type="image",
    #                         fixed_models=image_models,
    #                         scenes=scenes,
    #                         data_dir=data_dir,
    #                         output_dir=output_dir)
    
    # generate_pairwise_plots(filtered_df,
    #                     fixed_type="lidar",
    #                     fixed_models=lidar_models,
    #                     scenes=scenes,
    #                     data_dir=data_dir,
    #                     output_dir=output_dir)
    

    filtered_df = filter_param_mapping(
        csv_path=param_csv,
        # scenes=['c5224b9b454b4ded9b5d2d2634bbda8a'],
        lidar_models=['centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d']
    )

    for image_model in image_models:
        image_df = load_model_layers_across_runs(filtered_df, image_model, data_dir)
        print('-'*10, image_model, '-'*10)
        print(most_heavy_layers(image_df))


    lidar_model = 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d'
    lidar_df = load_model_layers_across_runs(filtered_df, lidar_model, data_dir)
    print('-'*10, lidar_model, '-'*10)
    print(most_heavy_layers(lidar_df))
    


if __name__ == "__main__":
    main()
