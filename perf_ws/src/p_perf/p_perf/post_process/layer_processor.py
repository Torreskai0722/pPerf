import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from p_perf.post_process.utils import get_run_indices_by_models, extract_rain_rate_from_scene, get_baseline_scene_name

class LayerProcessor:
    """
    A processor for analyzing layer timings from performance experiments.
    
    This class can:
    1. Add inference layers to layer timing data
    2. Generate per-model CSV files with layer statistics
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the LayerProcessor.
        
        Args:
            output_dir: Directory containing the experiment outputs
        """
        self.output_dir = Path(output_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the processor."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_mapping_csv(self, mapping_file: str) -> pd.DataFrame:
        """
        Load the full stack mapping CSV file.
        
        Args:
            mapping_file: Path to the mapping CSV file
            
        Returns:
            DataFrame containing the mapping information
        """
        mapping_path = self.output_dir / mapping_file
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
            
        self.logger.info(f"Loading mapping file: {mapping_path}")
        mapping_df = pd.read_csv(mapping_path)
        self.logger.info(f"Loaded {len(mapping_df)} experiment runs")
        return mapping_df
    
    def load_layer_timings(self, run_index: int) -> pd.DataFrame:
        """
        Load layer timings for a specific run index.
        
        Args:
            run_index: The run index to load
            
        Returns:
            DataFrame containing the layer timings with run_index column added
        """
        timings_file = f"layer_timings_{run_index}.csv"
        timings_path = self.output_dir / timings_file
        
        if not timings_path.exists():
            raise FileNotFoundError(f"Layer timings file not found: {timings_path}")
            
        self.logger.info(f"Loading layer timings: {timings_path}")
        timings_df = pd.read_csv(timings_path)
        
        # Add run_index column to identify which run this data belongs to
        timings_df['run_index'] = run_index
        
        self.logger.info(f"Loaded {len(timings_df)} layer records for run {run_index}")
        return timings_df
    
    def add_inference_layers(self, timings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add inference layers to the timings data.
        
        An inference layer is calculated as: e2e - data_preprocessing
        Start timestamp: end of data_preprocessing
        End timestamp: same as e2e end timestamp
        
        Args:
            timings_df: DataFrame containing layer timings
            
        Returns:
            DataFrame with inference layers added
        """
        self.logger.info("Adding inference layers...")
        
        # Create a copy to avoid modifying the original
        df = timings_df.copy()
        
        # Find all unique inputs and models
        inputs = df['Input'].unique()
        models = df['Model'].unique()
        
        inference_rows = []
        
        for input_id in inputs:
            for model in models:
                if model == 'pending':
                    continue
                    
                # Get e2e and data_preprocessing layers for this input/model combination
                e2e_data = df[(df['Input'] == input_id) & 
                             (df['Model'] == model) & 
                             (df['Layer'] == 'e2e')]
                
                # Try different possible data preprocessing layer names
                data_prep_layer_names = ['data_preprocessing']
                data_prep_data = None
                
                for layer_name in data_prep_layer_names:
                    data_prep_data = df[(df['Input'] == input_id) & 
                                       (df['Model'] == model) & 
                                       (df['Layer'] == layer_name)]
                    if len(data_prep_data) > 0:
                        break
                
                if len(e2e_data) > 0 and data_prep_data is not None and len(data_prep_data) > 0:
                    e2e_row = e2e_data.iloc[0]
                    data_prep_row = data_prep_data.iloc[0]
                    
                    # Calculate inference timing
                    inference_start = data_prep_row['End Timestamp']
                    inference_end = e2e_row['End Timestamp']
                    inference_elapsed = e2e_row['Elapsed Time'] - data_prep_row['Elapsed Time']
                    
                    # Create inference row
                    inference_row = {
                        'Input': input_id,
                        'Model': model,
                        'Layer': 'inference',
                        'Start Timestamp': inference_start,
                        'End Timestamp': inference_end,
                        'Elapsed Time': inference_elapsed,
                        'GPU Turnaround Time': 0.0,
                        'GPU Computation Time': 0.0,
                        'GPU Wait Time': 0.0,
                        'Internal Memcpy Size': 0,
                        'External Memcpy Size': 0,
                        'run_index': e2e_row['run_index']  # Preserve the run_index
                    }
                    
                    inference_rows.append(inference_row)
        
        # Add inference rows to the dataframe
        if inference_rows:
            inference_df = pd.DataFrame(inference_rows)
            df = pd.concat([df, inference_df], ignore_index=True)
            self.logger.info(f"Added {len(inference_rows)} inference layers")
        else:
            self.logger.warning("No inference layers could be calculated")
            
        return df
    
    def filter_relevant_layers(self, timings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out image_decode and lidar_decode layers.
        
        Args:
            timings_df: DataFrame containing layer timings
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = timings_df[~timings_df['Layer'].isin(['image_decode', 'lidar_decode'])]
        self.logger.info(f"Filtered out decode layers, remaining: {len(filtered_df)} records")
        return filtered_df

    def _remove_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove outliers using the IQR (Interquartile Range) method.
        
        Args:
            df: DataFrame containing the data
            column: Column name to check for outliers
            
        Returns:
            DataFrame with outliers removed
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        return filtered_df

    
    def modify_timings(self, 
                           run_index: int, 
                           add_inference: bool = True,
                           save_modified_timings: bool = True) -> pd.DataFrame:
        """
        Analyze a single run index.
        
        Args:
            run_index: The run index to analyze
            add_inference: Whether to add inference layers
            
        Returns:
            DataFrame with processed layer timings including run_index
        """
        self.logger.info(f"Analyzing single run: {run_index}")
        
        # Load timings (this now includes run_index column)
        timings_df = self.load_layer_timings(run_index)
        
        # Filter out decode layers
        timings_df = self.filter_relevant_layers(timings_df)
        
        # Add inference layers if requested
        if add_inference:
            timings_df = self.add_inference_layers(timings_df)
        
        if save_modified_timings:
            self.save_modified_timings(timings_df, run_index)
        
        return timings_df
    
    def save_modified_timings(self, timings: pd.DataFrame, run_index: int) -> None:
        """
        Save the combined layer timings with inference layers to a single CSV file.
        
        Args:
            combined_timings: Combined DataFrame with all layer timings including inference
            mapping_df: DataFrame containing the mapping information
        """
        self.logger.info("Saving combined timings with inference layers...")
        
        # Save combined data to a single file
        combined_file = self.output_dir / f"layer_timings_{run_index}.csv"
        timings.to_csv(combined_file, index=False)
        self.logger.info(f"Saved combined timings to: {combined_file}")
    
    def plot_layer_boxplot(self, 
                          run_indices: List[int], 
                          layer_name: str, 
                          model_name: str,
                          mapping_file: str,
                          metric: str,
                          save_plot: bool = True,
                          remove_outliers: bool = True,
                          figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Create a box plot of layer timings for specified run indices, layer, and model.
        
        Args:
            run_indices: List of run indices to include in the plot
            layer_name: Name of the layer to plot (e.g., 'inference', 'e2e', 'data_preprocessing')
            model_name: Name of the model to plot
            mapping_file: Path to the mapping CSV file
            metric: Column name from mapping CSV to include as metric data
            save_plot: Whether to save the plot to file
            remove_outliers: Whether to remove outliers using IQR method (default: True)
            figsize: Figure size as (width, height) tuple
        """
        self.logger.info(f"Creating box plot for layer '{layer_name}' and model '{model_name}' across {len(run_indices)} runs")
        
        # Load mapping CSV after the logger info
        mapping_df = self.load_mapping_csv(mapping_file)
        
        # Collect timing data for all specified run indices
        all_timings = []
        
        for run_idx in run_indices:
            try:
                # Load timings for this run
                timings_df = self.load_layer_timings(run_idx)
                
                # Filter for the specific layer and model
                filtered_data = timings_df[
                    (timings_df['Layer'] == layer_name) & 
                    (timings_df['Model'] == model_name)
                ]
                
                # Get metric value from mapping CSV for this run_index
                metric_value = None
                matching_rows = mapping_df[mapping_df['run_index'] == run_idx]
                if len(matching_rows) > 0:
                    if metric in matching_rows.columns:
                        metric_value = matching_rows.iloc[0][metric]
                    else:
                        self.logger.warning(f"Metric '{metric}' not found in mapping CSV columns")
                else:
                    self.logger.warning(f"Run index {run_idx} not found in mapping CSV")
                
                if len(filtered_data) > 0:
                    # Add the timing data with run index label and metric
                    for _, row in filtered_data.iterrows():
                        all_timings.append({
                            'run_index': run_idx,
                            'elapsed_time': row['Elapsed Time'],
                            'metric': metric_value
                        })
                else:
                    self.logger.warning(f"No data found for layer '{layer_name}' and model '{model_name}' in run {run_idx}")
                    
            except FileNotFoundError:
                self.logger.warning(f"Layer timings file not found for run {run_idx}")
                continue
        
        if not all_timings:
            self.logger.error(f"No timing data found for layer '{layer_name}' and model '{model_name}' across specified runs")
            return
        
        # Convert to DataFrame for plotting
        plot_df = pd.DataFrame(all_timings)
        
        # Remove outliers if requested
        if remove_outliers:
            original_count = len(plot_df)
            plot_df = self._remove_outliers(plot_df, 'elapsed_time')
            removed_count = original_count - len(plot_df)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} outliers from {original_count} data points")
        
        # Set seaborn style and context for larger text
        sns.set_style("white")  # Clean white background without grid
        sns.set_context("poster", font_scale=1.2)  # Use poster context for large text
        
        # Create the box plot
        plt.figure(figsize=figsize)
        
        # Use seaborn for better-looking box plots matching the reference style
        ax = sns.boxplot(data=plot_df, x='metric', y='elapsed_time', 
                        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                        medianprops=dict(color='orange', linewidth=3),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='white', 
                                      markeredgecolor='black', markersize=6, alpha=0.7))
        
        # Customize the plot using seaborn
        # ax.set_title(f'Layer Timing Distribution: {layer_name} ({model_name})', 
        #             fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label
        
        # Rotate x-axis labels if needed
        # plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot if requested
        if save_plot:
            plot_filename = f"boxplot_{layer_name}_{model_name}_runs_{'_'.join(map(str, run_indices))}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved box plot to: {plot_path}")
        

        plt.close()
        
        # Log summary statistics
        summary_stats = plot_df.groupby('run_index')['elapsed_time'].describe()
        self.logger.info(f"Summary statistics for {layer_name} ({model_name}):")
        self.logger.info(f"\n{summary_stats}")
        
        # Log metric information
        if 'metric' in plot_df.columns:
            metric_info = plot_df[['run_index', 'metric']].drop_duplicates()
            self.logger.info(f"Metric '{metric}' values by run_index:")
            for _, row in metric_info.iterrows():
                self.logger.info(f"  Run {row['run_index']}: {row['metric']}")

    def plot_rain_rate_analysis(self, 
                               rain_output_dir: str,
                               baseline_output_dir: Optional[str],
                               target_model: str,
                               layer_name: str, 
                               image_model: Optional[str] = None,
                               lidar_model: Optional[Union[str, Tuple[str, str]]] = None,
                               seg_model: Optional[Union[str, Tuple[str, str]]] = None,
                               rain_mapping_file: str = "full_stack_mapping.csv",
                               baseline_mapping_file: str = "full_stack_mapping.csv",
                               save_plot: bool = True,
                               remove_outliers: bool = True,
                               figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create a box plot analyzing rain rate influence on layer timings with optional baseline comparison.
        
        Args:
            rain_output_dir: Directory containing the rain rate experiment outputs
            baseline_output_dir: Directory containing the baseline (no rain) experiment outputs (optional)
            target_model: The specific model name to analyze (e.g., 'faster-rcnn_r50_fpn_1x_coco')
            layer_name: Name of the layer to plot (e.g., 'inference', 'e2e', 'data_preprocessing')
            image_model: Image model name to filter by (optional)
            lidar_model: LiDAR model name to filter by (optional)
            seg_model: Segmentation model name to filter by (optional)
            rain_mapping_file: Name of the mapping CSV file in rain output directory
            baseline_mapping_file: Name of the mapping CSV file in baseline output directory
            save_plot: Whether to save the plot to file
            remove_outliers: Whether to remove outliers using IQR method (default: True)
            figsize: Figure size as (width, height) tuple
        """
        self.logger.info(f"Creating rain rate analysis for model '{target_model}' and layer '{layer_name}'")
        
        # Setup paths
        rain_output_path = Path(rain_output_dir)
        rain_mapping_path = rain_output_path / rain_mapping_file
        
        if not rain_mapping_path.exists():
            raise FileNotFoundError(f"Rain mapping file not found: {rain_mapping_path}")
        
        # Collect timing data for all rain rates and baseline
        all_timings = []
        
        # Process rain rate data
        self.logger.info("Processing rain rate data...")
        rain_run_indices = get_run_indices_by_models(
            str(rain_mapping_path),
            image_model=image_model,
            lidar_model=lidar_model,
            seg_model=seg_model,
        )
        
        print(f"Rain run indices: {rain_run_indices}")
        rain_mapping_df = pd.read_csv(rain_mapping_path)
        
        for run_idx in rain_run_indices:
            try:
                # Get scene information to extract rain rate
                matching_rows = rain_mapping_df[rain_mapping_df['run_index'] == run_idx]
                if len(matching_rows) == 0:
                    continue
                    
                scene_name = matching_rows.iloc[0]['scene']
                rain_rate = extract_rain_rate_from_scene(scene_name)
                
                # Load timings for this run from rain directory
                timings_file = rain_output_path / f"layer_timings_{run_idx}.csv"
                if not timings_file.exists():
                    self.logger.warning(f"Rain timings file not found: {timings_file}")
                    continue
                    
                timings_df = pd.read_csv(timings_file)
                
                # Filter for the specific layer and target model
                filtered_data = timings_df[
                    (timings_df['Layer'] == layer_name) & 
                    (timings_df['Model'] == target_model)
                ]
                
                if len(filtered_data) > 0:
                    for _, timing_row in filtered_data.iterrows():
                        all_timings.append({
                            'rain_rate': rain_rate,
                            'elapsed_time': timing_row['Elapsed Time'],
                            'run_index': run_idx,
                            'source': 'rain' if rain_rate > 0 else 'unknown'
                        })
                else:
                    self.logger.warning(f"No rain data found for layer '{layer_name}' and model '{target_model}' in run {run_idx}")
                    
            except Exception as e:
                self.logger.warning(f"Error loading rain run {run_idx}: {e}")
                continue
        
        # Process baseline data if provided
        if baseline_output_dir:
            self.logger.info("Processing baseline data...")
            baseline_output_path = Path(baseline_output_dir)
            baseline_mapping_path = baseline_output_path / baseline_mapping_file
            
            if baseline_mapping_path.exists():
                baseline_run_indices = get_run_indices_by_models(
                    str(baseline_mapping_path),
                    image_model=image_model,
                    lidar_model=lidar_model,
                    seg_model=seg_model,
                    status_filter='success'
                )
                
                for run_idx in baseline_run_indices:
                    try:
                        # Load timings for this run from baseline directory
                        timings_file = baseline_output_path / f"layer_timings_{run_idx}.csv"
                        if not timings_file.exists():
                            self.logger.warning(f"Baseline timings file not found: {timings_file}")
                            continue
                            
                        timings_df = pd.read_csv(timings_file)
                        
                        # Filter for the specific layer and target model
                        filtered_data = timings_df[
                            (timings_df['Layer'] == layer_name) & 
                            (timings_df['Model'] == target_model)
                        ]
                        
                        if len(filtered_data) > 0:
                            for _, timing_row in filtered_data.iterrows():
                                all_timings.append({
                                    'rain_rate': 0,
                                    'elapsed_time': timing_row['Elapsed Time'],
                                    'run_index': run_idx,
                                    'source': 'baseline'
                                })
                        else:
                            self.logger.warning(f"No baseline data found for layer '{layer_name}' and model '{target_model}' in run {run_idx}")
                            
                    except Exception as e:
                        self.logger.warning(f"Error loading baseline run {run_idx}: {e}")
                        continue
            else:
                self.logger.warning(f"Baseline mapping file not found: {baseline_mapping_path}")
        
        if not all_timings:
            self.logger.error(f"No timing data found for model '{target_model}' and layer '{layer_name}'")
            return
        
        # Convert to DataFrame for plotting
        plot_df = pd.DataFrame(all_timings)
        
        # Remove outliers if requested
        if remove_outliers:
            original_count = len(plot_df)
            plot_df = self._remove_outliers(plot_df, 'elapsed_time')
            removed_count = original_count - len(plot_df)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} outliers from {original_count} data points")
        
        # Set seaborn style and context for larger text
        sns.set_style("white")  # Clean white background without grid
        sns.set_context("poster", font_scale=3)  # Use poster context for large text
        
        # Create the box plot
        plt.figure(figsize=figsize)
        
        # Sort by rain rate for proper ordering
        plot_df = plot_df.sort_values('rain_rate')
        
        # Use seaborn for better-looking box plots
        ax = sns.boxplot(data=plot_df, x='rain_rate', y='elapsed_time', 
                        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                        medianprops=dict(color='orange', linewidth=3),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='white', 
                                      markeredgecolor='black', markersize=6, alpha=0.7))
        
        # Customize the plot
        # ax.set_title(f'Rain Rate Impact on {layer_name.title()} Time\n{target_model}', 
        #             fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Set larger font sizes for tick labels
        ax.tick_params(axis='x', labelsize=40)  # X-axis tick labels
        ax.tick_params(axis='y', labelsize=40)  # Y-axis tick labels

        plt.tight_layout()
        
        # Save the plot if requested
        if save_plot:
            # Use rain_run_indices to create filename with run indices
            if rain_run_indices:
                run_indices_str = "_".join(map(str, sorted(rain_run_indices)))
                plot_filename = f"rain_rate_analysis_{layer_name}_{target_model.replace('/', '_')}_runs{run_indices_str}.png"
            else:
                plot_filename = f"rain_rate_analysis_{layer_name}_{target_model.replace('/', '_')}.png"
            
            plot_path = rain_output_path / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved rain rate analysis plot to: {plot_path}")
        
        plt.show()
        plt.close()
        
        # Log summary statistics by rain rate
        summary_stats = plot_df.groupby('rain_rate')['elapsed_time'].describe()
        self.logger.info(f"Summary statistics for {layer_name} ({target_model}) by rain rate:")
        self.logger.info(f"\n{summary_stats}")
        
        # Calculate percentage increase from baseline if baseline data exists
        if 0 in plot_df['rain_rate'].values:
            baseline_median = plot_df[plot_df['rain_rate'] == 0]['elapsed_time'].median()
            self.logger.info(f"\nPercentage increase from baseline (rain rate 0):")
            for rain_rate in sorted(plot_df['rain_rate'].unique()):
                if rain_rate != 0:
                    rain_median = plot_df[plot_df['rain_rate'] == rain_rate]['elapsed_time'].median()
                    pct_increase = ((rain_median - baseline_median) / baseline_median) * 100
                    self.logger.info(f"  Rain rate {rain_rate}: {pct_increase:.2f}% increase")