#!/usr/bin/env python3
"""
Performance Analysis Script for DNN Inference Experiments

This script analyzes delays and layer timings data from experiment outputs to understand:
- End-to-end inference time performance
- Communication delays
- Decode delays
- Layer-wise timing breakdowns
- Performance variations across model pairs and publishing rates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class PerformanceAnalyzer:
    def __init__(self, data_dir="outputs/MACs_1"):
        self.data_dir = Path(data_dir)
        self.param_mapping = None
        self.delays_data = {}
        self.layer_timings_data = {}
        self.analysis_results = {}
        
    def load_parameter_mapping(self):
        """Load the parameter mapping CSV file"""
        param_file = self.data_dir / "param_mapping.csv"
        if param_file.exists():
            self.param_mapping = pd.read_csv(param_file)
            print(f"Loaded parameter mapping with {len(self.param_mapping)} experiments")
            return True
        else:
            print(f"Parameter mapping file not found: {param_file}")
            return False
    
    def load_delays_data(self):
        """Load all delays CSV files"""
        delays_files = glob.glob(str(self.data_dir / "delays_*.csv"))
        print(f"Found {len(delays_files)} delays files")
        
        for file_path in delays_files:
            try:
                run_index = int(file_path.split('_')[-1].split('.')[0])
                df = pd.read_csv(file_path)
                self.delays_data[run_index] = df
                print(f"Loaded delays for run {run_index}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def load_layer_timings_data(self):
        """Load all layer timings CSV files"""
        layer_files = glob.glob(str(self.data_dir / "layer_timings_*.csv"))
        print(f"Found {len(layer_files)} layer timings files")
        
        for file_path in layer_files:
            try:
                run_index = int(file_path.split('_')[-1].split('.')[0])
                df = pd.read_csv(file_path)
                self.layer_timings_data[run_index] = df
                print(f"Loaded layer timings for run {run_index}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def analyze_delays_by_model_pair(self):
        """Analyze delays grouped by model pair and publishing rate"""
        if not self.param_mapping is not None:
            print("Parameter mapping not loaded")
            return
        
        results = []
        
        for run_index, delays_df in self.delays_data.items():
            if run_index not in self.param_mapping['run_index'].values:
                continue
                
            param_row = self.param_mapping[self.param_mapping['run_index'] == run_index].iloc[0]
            image_model = param_row['image_model']
            lidar_model = param_row['lidar_model']
            publishing_rate = param_row['publishing_rate']
            
            # Extract model name from lidar_model tuple string
            if isinstance(lidar_model, str) and lidar_model.startswith("('"):
                lidar_model_name = lidar_model.split("'")[1]
            else:
                lidar_model_name = str(lidar_model)
            
            # Group by model type (image/lidar)
            for model_type in ['image', 'lidar']:
                type_data = delays_df[delays_df['model_type'] == model_type]
                
                if len(type_data) == 0:
                    continue
                
                # Calculate statistics
                stats = {
                    'run_index': run_index,
                    'image_model': image_model,
                    'lidar_model': lidar_model_name,
                    'model_type': model_type,
                    'publishing_rate': publishing_rate,
                    'count': len(type_data),
                    'e2e_mean': type_data['e2e_delay'].mean(),
                    'e2e_std': type_data['e2e_delay'].std(),
                    'e2e_min': type_data['e2e_delay'].min(),
                    'e2e_max': type_data['e2e_delay'].max(),
                    'comm_mean': type_data['comm_delay'].mean(),
                    'comm_std': type_data['comm_delay'].std(),
                    'decode_mean': type_data['decode_delay'].mean(),
                    'decode_std': type_data['decode_delay'].std(),
                    'inference_mean': type_data['inference_delay'].mean(),
                    'inference_std': type_data['inference_delay'].std(),
                }
                results.append(stats)
        
        self.analysis_results['delays_by_model_pair'] = pd.DataFrame(results)
        print(f"Analyzed delays for {len(results)} model-type combinations")
    
    def analyze_layer_timings_by_model_pair(self):
        """Analyze layer timings grouped by model pair"""
        if not self.param_mapping is not None:
            print("Parameter mapping not loaded")
            return
        
        results = []
        
        for run_index, layer_df in self.layer_timings_data.items():
            if run_index not in self.param_mapping['run_index'].values:
                continue
                
            param_row = self.param_mapping[self.param_mapping['run_index'] == run_index].iloc[0]
            image_model = param_row['image_model']
            lidar_model = param_row['lidar_model']
            publishing_rate = param_row['publishing_rate']
            
            # Extract model name from lidar_model tuple string
            if isinstance(lidar_model, str) and lidar_model.startswith("('"):
                lidar_model_name = lidar_model.split("'")[1]
            else:
                lidar_model_name = str(lidar_model)
            
            # Filter for e2e entries to get overall model performance
            e2e_data = layer_df[layer_df['Layer'] == 'e2e']
            
            for _, row in e2e_data.iterrows():
                model_name = row['Model']
                
                # Determine if this is image or lidar model
                if model_name in [image_model]:
                    model_type = 'image'
                elif model_name in [lidar_model_name]:
                    model_type = 'lidar'
                else:
                    continue
                
                # Get all layers for this model
                model_layers = layer_df[layer_df['Model'] == model_name]
                
                stats = {
                    'run_index': run_index,
                    'image_model': image_model,
                    'lidar_model': lidar_model_name,
                    'model_type': model_type,
                    'model_name': model_name,
                    'publishing_rate': publishing_rate,
                    'e2e_elapsed': row['Elapsed Time'],
                    'e2e_gpu_turnaround': row['GPU Turnaround Time'],
                    'e2e_gpu_computation': row['GPU Computation Time'],
                    'e2e_gpu_wait': row['GPU Wait Time'],
                    'total_layers': len(model_layers),
                    'total_internal_memcpy': model_layers['Internal Memcpy Size'].sum(),
                    'total_external_memcpy': model_layers['External Memcpy Size'].sum(),
                }
                results.append(stats)
        
        self.analysis_results['layer_timings_by_model_pair'] = pd.DataFrame(results)
        print(f"Analyzed layer timings for {len(results)} model instances")
    
    def create_delays_visualizations(self):
        """Create visualizations for delays analysis"""
        if 'delays_by_model_pair' not in self.analysis_results:
            print("Delays analysis not performed")
            return
        
        df = self.analysis_results['delays_by_model_pair']
        
        # Create output directory
        output_dir = Path("analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 1. E2E Delay by Model Pair and Publishing Rate
        plt.figure(figsize=(15, 8))
        
        # Create model pair identifier
        df['model_pair'] = df['image_model'] + ' + ' + df['lidar_model']
        
        # Plot for each model type
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Image models
        image_data = df[df['model_type'] == 'image']
        for model_pair in image_data['model_pair'].unique():
            pair_data = image_data[image_data['model_pair'] == model_pair]
            ax1.plot(pair_data['publishing_rate'], pair_data['e2e_mean'], 
                    marker='o', label=model_pair, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Publishing Rate (Hz)')
        ax1.set_ylabel('Mean E2E Delay (seconds)')
        ax1.set_title('Image Model E2E Delays by Publishing Rate')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Lidar models
        lidar_data = df[df['model_type'] == 'lidar']
        for model_pair in lidar_data['model_pair'].unique():
            pair_data = lidar_data[lidar_data['model_pair'] == model_pair]
            ax2.plot(pair_data['publishing_rate'], pair_data['e2e_mean'], 
                    marker='s', label=model_pair, linewidth=2, markersize=8)
        
        ax2.set_xlabel('Publishing Rate (Hz)')
        ax2.set_ylabel('Mean E2E Delay (seconds)')
        ax2.set_title('Lidar Model E2E Delays by Publishing Rate')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'delays_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Box Plots for Each Model Pair
        self._create_model_pair_boxplots(df, output_dir)
        
        # 3. Delay Components Breakdown
        plt.figure(figsize=(15, 10))
        
        # Calculate average delays by model type
        avg_delays = df.groupby('model_type').agg({
            'e2e_mean': 'mean',
            'comm_mean': 'mean',
            'decode_mean': 'mean',
            'inference_mean': 'mean'
        }).reset_index()
        
        # Create stacked bar chart
        model_types = avg_delays['model_type']
        comm_delays = avg_delays['comm_mean']
        decode_delays = avg_delays['decode_mean']
        inference_delays = avg_delays['inference_mean']
        
        x = np.arange(len(model_types))
        width = 0.6
        
        plt.bar(x, comm_delays, width, label='Communication Delay', alpha=0.8)
        plt.bar(x, decode_delays, width, bottom=comm_delays, label='Decode Delay', alpha=0.8)
        plt.bar(x, inference_delays, width, bottom=comm_delays+decode_delays, label='Inference Delay', alpha=0.8)
        
        plt.xlabel('Model Type')
        plt.ylabel('Delay (seconds)')
        plt.title('Average Delay Components by Model Type')
        plt.xticks(x, model_types)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'delay_components.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Publishing Rate Impact
        plt.figure(figsize=(12, 8))
        
        # Group by publishing rate and model type
        rate_impact = df.groupby(['publishing_rate', 'model_type'])['e2e_mean'].mean().reset_index()
        
        for model_type in rate_impact['model_type'].unique():
            type_data = rate_impact[rate_impact['model_type'] == model_type]
            plt.plot(type_data['publishing_rate'], type_data['e2e_mean'], 
                    marker='o', label=f'{model_type} models', linewidth=2, markersize=8)
        
        plt.xlabel('Publishing Rate (Hz)')
        plt.ylabel('Average E2E Delay (seconds)')
        plt.title('Impact of Publishing Rate on E2E Delays')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'publishing_rate_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_pair_boxplots(self, df, output_dir):
        """Create box plots for each model pair showing delay distributions"""
        # Get unique model pairs
        df['model_pair'] = df['image_model'] + ' + ' + df['lidar_model']
        model_pairs = df['model_pair'].unique()
        
        # Create a comprehensive box plot for all model pairs
        plt.figure(figsize=(20, 12))
        
        # Create subplots for each model pair
        n_pairs = len(model_pairs)
        cols = 3
        rows = (n_pairs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Define colors for different publishing rates and model types
        rate_colors = {8: 'lightblue', 10: 'lightgreen', 12: 'lightcoral'}
        model_colors = {'image': 'skyblue', 'lidar': 'salmon'}
        
        for idx, model_pair in enumerate(model_pairs):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Get data for this model pair
            pair_data = df[df['model_pair'] == model_pair]
            
            # Create box plot data
            box_data = []
            labels = []
            colors = []
            
            for rate in sorted(pair_data['publishing_rate'].unique()):
                rate_data = pair_data[pair_data['publishing_rate'] == rate]
                
                # Get raw delay data for this model pair and rate
                for run_idx in rate_data['run_index']:
                    if run_idx in self.delays_data:
                        delays_df = self.delays_data[run_idx]
                        
                        # Get both image and lidar data for this run
                        for model_type in ['image', 'lidar']:
                            type_delays = delays_df[delays_df['model_type'] == model_type]['e2e_delay']
                            if len(type_delays) > 0:
                                box_data.append(type_delays.values)
                                labels.append(f'{rate}Hz-{model_type}')
                                colors.append(f'{rate}-{model_type}')
            
            if box_data:
                # Create box plot
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                
                # Color the boxes with different patterns for image vs lidar
                for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
                    rate = int(color.split('-')[0])
                    model_type = color.split('-')[1]
                    
                    if model_type == 'image':
                        patch.set_facecolor(rate_colors.get(rate, 'gray'))
                        patch.set_alpha(0.7)
                    else:  # lidar
                        patch.set_facecolor(rate_colors.get(rate, 'gray'))
                        patch.set_alpha(0.4)
                        # Add hatching for lidar to distinguish from image
                        patch.set_hatch('///')
                
                ax.set_title(f'{model_pair}', fontsize=10, wrap=True)
                ax.set_xlabel('Publishing Rate - Model Type')
                ax.set_ylabel('E2E Delay (seconds)')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                # Add legend
                legend_elements = []
                for rate in sorted(rate_colors.keys()):
                    # Image legend element
                    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=rate_colors[rate], alpha=0.7, 
                                                       label=f'{rate} Hz - Image'))
                    # Lidar legend element with hatching
                    lidar_rect = plt.Rectangle((0,0),1,1, facecolor=rate_colors[rate], alpha=0.4, 
                                             label=f'{rate} Hz - Lidar')
                    lidar_rect.set_hatch('///')
                    legend_elements.append(lidar_rect)
                
                ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_pair}', fontsize=10, wrap=True)
        
        # Hide empty subplots
        for idx in range(len(model_pairs), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_pair_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual box plot files for each model pair
        for model_pair in model_pairs:
            self._create_individual_model_pair_boxplot(model_pair, df, output_dir)
    
    def _create_individual_model_pair_boxplot(self, model_pair, df, output_dir):
        """Create individual box plot for a specific model pair"""
        # Get data for this model pair
        pair_data = df[df['model_pair'] == model_pair]
        
        # Create figure with subplots for different delay types
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        delay_types = [
            ('e2e_delay', 'E2E Delay', ax1),
            ('comm_delay', 'Communication Delay', ax2),
            ('decode_delay', 'Decode Delay', ax3),
            ('inference_delay', 'Inference Delay', ax4)
        ]
        
        # Define colors for different publishing rates and model types
        rate_colors = {8: 'lightblue', 10: 'lightgreen', 12: 'lightcoral'}
        model_colors = {'image': 'skyblue', 'lidar': 'salmon'}
        
        for delay_col, delay_name, ax in delay_types:
            # Create box plot data
            box_data = []
            labels = []
            colors = []
            
            for rate in sorted(pair_data['publishing_rate'].unique()):
                rate_data = pair_data[pair_data['publishing_rate'] == rate]
                
                # Get raw delay data for this model pair and rate
                for run_idx in rate_data['run_index']:
                    if run_idx in self.delays_data:
                        delays_df = self.delays_data[run_idx]
                        
                        # Get both image and lidar data for this run
                        for model_type in ['image', 'lidar']:
                            type_delays = delays_df[delays_df['model_type'] == model_type][delay_col]
                            if len(type_delays) > 0:
                                box_data.append(type_delays.values)
                                labels.append(f'{rate}Hz-{model_type}')
                                colors.append(f'{rate}-{model_type}')
            
            if box_data:
                # Create box plot
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                
                # Color the boxes with different patterns for image vs lidar
                for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
                    rate = int(color.split('-')[0])
                    model_type = color.split('-')[1]
                    
                    if model_type == 'image':
                        patch.set_facecolor(rate_colors.get(rate, 'gray'))
                        patch.set_alpha(0.7)
                    else:  # lidar
                        patch.set_facecolor(rate_colors.get(rate, 'gray'))
                        patch.set_alpha(0.4)
                        # Add hatching for lidar to distinguish from image
                        patch.set_hatch('///')
                
                ax.set_title(f'{delay_name} by Publishing Rate and Model Type')
                ax.set_xlabel('Publishing Rate - Model Type')
                ax.set_ylabel(f'{delay_name} (seconds)')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                # Add legend
                legend_elements = []
                for rate in sorted(rate_colors.keys()):
                    # Image legend element
                    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=rate_colors[rate], alpha=0.7, 
                                                       label=f'{rate} Hz - Image'))
                    # Lidar legend element with hatching
                    lidar_rect = plt.Rectangle((0,0),1,1, facecolor=rate_colors[rate], alpha=0.4, 
                                             label=f'{rate} Hz - Lidar')
                    lidar_rect.set_hatch('///')
                    legend_elements.append(lidar_rect)
                
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{delay_name} by Publishing Rate and Model Type')
        
        # Add overall title
        fig.suptitle(f'Delay Analysis for Model Pair: {model_pair}', fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Create safe filename
        safe_filename = f"boxplot_{model_pair.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')}.png"
        plt.savefig(output_dir / safe_filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_layer_timings_visualizations(self):
        """Create visualizations for layer timings analysis"""
        if 'layer_timings_by_model_pair' not in self.analysis_results:
            print("Layer timings analysis not performed")
            return
        
        df = self.analysis_results['layer_timings_by_model_pair']
        
        # Create output directory
        output_dir = Path("analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 1. GPU Utilization Analysis
        plt.figure(figsize=(15, 10))
        
        # Calculate GPU utilization percentage
        df['gpu_utilization'] = (df['e2e_gpu_computation'] / df['e2e_gpu_turnaround']) * 100
        
        # Create subplots for different metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # GPU Utilization by Model Type
        model_types = df['model_type'].unique()
        utilizations = [df[df['model_type'] == mt]['gpu_utilization'].mean() for mt in model_types]
        
        bars1 = ax1.bar(model_types, utilizations, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_title('Average GPU Utilization by Model Type')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, util in zip(bars1, utilizations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{util:.1f}%', ha='center', va='bottom')
        
        # E2E Time by Model Type
        e2e_times = [df[df['model_type'] == mt]['e2e_elapsed'].mean() for mt in model_types]
        
        bars2 = ax2.bar(model_types, e2e_times, alpha=0.7, color=['lightgreen', 'orange'])
        ax2.set_ylabel('E2E Time (seconds)')
        ax2.set_title('Average E2E Time by Model Type')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars2, e2e_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{time_val:.2f}s', ha='center', va='bottom')
        
        # GPU Wait Time vs Computation Time
        ax3.scatter(df[df['model_type'] == 'image']['e2e_gpu_computation'], 
                   df[df['model_type'] == 'image']['e2e_gpu_wait'], 
                   alpha=0.6, label='Image Models', s=50)
        ax3.scatter(df[df['model_type'] == 'lidar']['e2e_gpu_computation'], 
                   df[df['model_type'] == 'lidar']['e2e_gpu_wait'], 
                   alpha=0.6, label='Lidar Models', s=50)
        ax3.set_xlabel('GPU Computation Time (seconds)')
        ax3.set_ylabel('GPU Wait Time (seconds)')
        ax3.set_title('GPU Wait Time vs Computation Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Memory Copy Analysis
        total_memcpy = df['total_internal_memcpy'] + df['total_external_memcpy']
        ax4.scatter(df[df['model_type'] == 'image']['e2e_elapsed'], 
                   total_memcpy[df['model_type'] == 'image'], 
                   alpha=0.6, label='Image Models', s=50)
        ax4.scatter(df[df['model_type'] == 'lidar']['e2e_elapsed'], 
                   total_memcpy[df['model_type'] == 'lidar'], 
                   alpha=0.6, label='Lidar Models', s=50)
        ax4.set_xlabel('E2E Time (seconds)')
        ax4.set_ylabel('Total Memory Copy Size (bytes)')
        ax4.set_title('E2E Time vs Memory Copy Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'layer_timings_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Performance Comparison
        plt.figure(figsize=(14, 8))
        
        # Group by model name and calculate averages
        model_performance = df.groupby(['model_name', 'model_type']).agg({
            'e2e_elapsed': 'mean',
            'gpu_utilization': 'mean',
            'total_layers': 'mean'
        }).reset_index()
        
        # Create scatter plot
        colors = {'image': 'blue', 'lidar': 'red'}
        
        for model_type in model_performance['model_type'].unique():
            type_data = model_performance[model_performance['model_type'] == model_type]
            plt.scatter(type_data['e2e_elapsed'], type_data['gpu_utilization'], 
                       s=type_data['total_layers']*10, alpha=0.7, 
                       label=f'{model_type} models', c=colors[model_type])
            
            # Add model name labels
            for _, row in type_data.iterrows():
                plt.annotate(row['model_name'], 
                           (row['e2e_elapsed'], row['gpu_utilization']),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
        
        plt.xlabel('Average E2E Time (seconds)')
        plt.ylabel('Average GPU Utilization (%)')
        plt.title('Model Performance: E2E Time vs GPU Utilization\n(Bubble size = number of layers)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.analysis_results:
            print("No analysis results available")
            return
        
        output_dir = Path("analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / "performance_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall Statistics
            f.write("1. OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total experiments analyzed: {len(self.param_mapping)}\n")
            f.write(f"Delays files processed: {len(self.delays_data)}\n")
            f.write(f"Layer timings files processed: {len(self.layer_timings_data)}\n\n")
            
            # Publishing Rates
            f.write("2. PUBLISHING RATES\n")
            f.write("-" * 40 + "\n")
            rates = self.param_mapping['publishing_rate'].unique()
            f.write(f"Publishing rates tested: {sorted(rates)} Hz\n\n")
            
            # Model Pairs
            f.write("3. MODEL PAIRS\n")
            f.write("-" * 40 + "\n")
            image_models = self.param_mapping['image_model'].unique()
            f.write(f"Image models: {list(image_models)}\n")
            
            lidar_models = []
            for lidar_model in self.param_mapping['lidar_model'].unique():
                if isinstance(lidar_model, str) and lidar_model.startswith("('"):
                    lidar_models.append(lidar_model.split("'")[1])
                else:
                    lidar_models.append(str(lidar_model))
            f.write(f"Lidar models: {list(set(lidar_models))}\n\n")
            
            # Delays Analysis Summary
            if 'delays_by_model_pair' in self.analysis_results:
                df = self.analysis_results['delays_by_model_pair']
                f.write("4. DELAYS ANALYSIS SUMMARY\n")
                f.write("-" * 40 + "\n")
                
                # Overall statistics
                f.write(f"Average E2E delay (all models): {df['e2e_mean'].mean():.3f} ± {df['e2e_mean'].std():.3f} seconds\n")
                f.write(f"Average communication delay: {df['comm_mean'].mean():.3f} ± {df['comm_mean'].std():.3f} seconds\n")
                f.write(f"Average decode delay: {df['decode_mean'].mean():.3f} ± {df['decode_mean'].std():.3f} seconds\n")
                f.write(f"Average inference delay: {df['inference_mean'].mean():.3f} ± {df['inference_mean'].std():.3f} seconds\n\n")
                
                # Best and worst performers
                best_e2e = df.loc[df['e2e_mean'].idxmin()]
                worst_e2e = df.loc[df['e2e_mean'].idxmax()]
                
                f.write("Best E2E Performance:\n")
                f.write(f"  Model pair: {best_e2e['image_model']} + {best_e2e['lidar_model']}\n")
                f.write(f"  Model type: {best_e2e['model_type']}\n")
                f.write(f"  Publishing rate: {best_e2e['publishing_rate']} Hz\n")
                f.write(f"  E2E delay: {best_e2e['e2e_mean']:.3f} seconds\n\n")
                
                f.write("Worst E2E Performance:\n")
                f.write(f"  Model pair: {worst_e2e['image_model']} + {worst_e2e['lidar_model']}\n")
                f.write(f"  Model type: {worst_e2e['model_type']}\n")
                f.write(f"  Publishing rate: {worst_e2e['publishing_rate']} Hz\n")
                f.write(f"  E2E delay: {worst_e2e['e2e_mean']:.3f} seconds\n\n")
            
            # Layer Timings Summary
            if 'layer_timings_by_model_pair' in self.analysis_results:
                df = self.analysis_results['layer_timings_by_model_pair']
                f.write("5. LAYER TIMINGS SUMMARY\n")
                f.write("-" * 40 + "\n")
                
                # GPU utilization
                f.write(f"Average GPU utilization: {df['gpu_utilization'].mean():.1f} ± {df['gpu_utilization'].std():.1f}%\n")
                f.write(f"Average E2E elapsed time: {df['e2e_elapsed'].mean():.3f} ± {df['e2e_elapsed'].std():.3f} seconds\n")
                f.write(f"Average GPU turnaround time: {df['e2e_gpu_turnaround'].mean():.3f} ± {df['e2e_gpu_turnaround'].std():.3f} seconds\n\n")
                
                # Model-specific insights
                f.write("Model-specific insights:\n")
                for model_type in df['model_type'].unique():
                    type_data = df[df['model_type'] == model_type]
                    f.write(f"  {model_type.capitalize()} models:\n")
                    f.write(f"    Average E2E time: {type_data['e2e_elapsed'].mean():.3f} seconds\n")
                    f.write(f"    Average GPU utilization: {type_data['gpu_utilization'].mean():.1f}%\n")
                    f.write(f"    Average layers: {type_data['total_layers'].mean():.1f}\n\n")
            
            # Recommendations
            f.write("6. RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("Based on the analysis:\n")
            f.write("1. Consider the trade-off between publishing rate and E2E delay\n")
            f.write("2. Monitor GPU utilization to identify bottlenecks\n")
            f.write("3. Analyze communication delays for network optimization\n")
            f.write("4. Consider model complexity vs performance trade-offs\n")
            f.write("5. Investigate GPU wait times for potential optimization\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generated successfully!\n")
            f.write("Check the analysis_outputs directory for visualizations.\n")
            f.write("=" * 80 + "\n")
        
        print(f"Summary report saved to: {report_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Performance Analysis...")
        print("=" * 50)
        
        # Load data
        print("1. Loading parameter mapping...")
        if not self.load_parameter_mapping():
            return
        
        print("2. Loading delays data...")
        self.load_delays_data()
        
        print("3. Loading layer timings data...")
        self.load_layer_timings_data()
        
        # Perform analysis
        print("4. Analyzing delays by model pair...")
        self.analyze_delays_by_model_pair()
        
        print("5. Analyzing layer timings by model pair...")
        self.analyze_layer_timings_by_model_pair()
        
        # Create visualizations
        print("6. Creating delays visualizations...")
        self.create_delays_visualizations()
        
        print("7. Creating layer timings visualizations...")
        self.create_layer_timings_visualizations()
        
        # Generate report
        print("8. Generating summary report...")
        self.generate_summary_report()
        
        print("\nAnalysis complete! Check the 'analysis_outputs' directory for results.")
        print("=" * 50)

def main():
    """Main function to run the analysis"""
    analyzer = PerformanceAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
