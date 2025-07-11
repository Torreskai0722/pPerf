#!/usr/bin/env python3
"""
Performance Analysis Script for Multi-Model Inference Experiments

This script analyzes the performance data from the MACs experiments, including:
- E2E inference time analysis
- Communication delay analysis  
- Decode delay analysis
- Layer timing analysis
- Performance comparison across different model pairs and publishing rates
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

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceAnalyzer:
    def __init__(self, data_dir="outputs/MACs"):
        self.data_dir = Path(data_dir)
        self.param_mapping = None
        self.delays_data = {}
        self.layer_timings_data = {}
        self.results = {}
        
    def load_param_mapping(self):
        """Load the parameter mapping file"""
        param_file = self.data_dir / "param_mapping.csv"
        if param_file.exists():
            self.param_mapping = pd.read_csv(param_file)
            print(f"Loaded parameter mapping with {len(self.param_mapping)} experiments")
            return True
        else:
            print("Parameter mapping file not found!")
            return False
    
    def load_delays_data(self):
        """Load all delays CSV files"""
        delays_files = glob.glob(str(self.data_dir / "delays_*.csv"))
        print(f"Found {len(delays_files)} delays files")
        
        for file_path in delays_files:
            run_index = int(file_path.split('_')[-1].split('.')[0])
            try:
                df = pd.read_csv(file_path)
                self.delays_data[run_index] = df
                print(f"Loaded delays_{run_index}.csv with {len(df)} records")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def load_layer_timings_data(self):
        """Load all layer timings CSV files"""
        layer_files = glob.glob(str(self.data_dir / "layer_timings_*.csv"))
        print(f"Found {len(layer_files)} layer timings files")
        
        for file_path in layer_files:
            run_index = int(file_path.split('_')[-1].split('.')[0])
            try:
                df = pd.read_csv(file_path)
                self.layer_timings_data[run_index] = df
                print(f"Loaded layer_timings_{run_index}.csv with {len(df)} records")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def analyze_delays_by_model_pair(self):
        """Analyze delays for each model pair and publishing rate"""
        if not self.param_mapping is not None:
            print("Parameter mapping not loaded!")
            return
        
        results = []
        
        for _, row in self.param_mapping.iterrows():
            run_index = row['run_index']
            image_model = row['image_model']
            lidar_model = row['lidar_model']
            publishing_rate = row['publishing_rate']
            
            if run_index not in self.delays_data:
                continue
                
            delays_df = self.delays_data[run_index]
            
            # Separate image and lidar data
            image_delays = delays_df[delays_df['model_type'] == 'image']
            lidar_delays = delays_df[delays_df['model_type'] == 'lidar']
            
            # Analyze image model delays
            if len(image_delays) > 0:
                image_stats = {
                    'run_index': run_index,
                    'model_type': 'image',
                    'model_name': image_model,
                    'publishing_rate': publishing_rate,
                    'count': len(image_delays),
                    'e2e_mean': image_delays['e2e_delay'].mean(),
                    'e2e_std': image_delays['e2e_delay'].std(),
                    'e2e_min': image_delays['e2e_delay'].min(),
                    'e2e_max': image_delays['e2e_delay'].max(),
                    'comm_mean': image_delays['comm_delay'].mean(),
                    'comm_std': image_delays['comm_delay'].std(),
                    'decode_mean': image_delays['decode_delay'].mean(),
                    'decode_std': image_delays['decode_delay'].std(),
                    'inference_mean': image_delays['inference_delay'].mean(),
                    'inference_std': image_delays['inference_delay'].std(),
                }
                results.append(image_stats)
            
            # Analyze lidar model delays
            if len(lidar_delays) > 0:
                lidar_stats = {
                    'run_index': run_index,
                    'model_type': 'lidar',
                    'model_name': lidar_model,
                    'publishing_rate': publishing_rate,
                    'count': len(lidar_delays),
                    'e2e_mean': lidar_delays['e2e_delay'].mean(),
                    'e2e_std': lidar_delays['e2e_delay'].std(),
                    'e2e_min': lidar_delays['e2e_delay'].min(),
                    'e2e_max': lidar_delays['e2e_delay'].max(),
                    'comm_mean': lidar_delays['comm_delay'].mean(),
                    'comm_std': lidar_delays['comm_delay'].std(),
                    'decode_mean': lidar_delays['decode_delay'].mean(),
                    'decode_std': lidar_delays['decode_delay'].std(),
                    'inference_mean': lidar_delays['inference_delay'].mean(),
                    'inference_std': lidar_delays['inference_delay'].std(),
                }
                results.append(lidar_stats)
        
        self.delays_analysis = pd.DataFrame(results)
        return self.delays_analysis
    
    def analyze_layer_timings_by_model_pair(self):
        """Analyze layer timings for each model pair"""
        if not self.param_mapping is not None:
            print("Parameter mapping not loaded!")
            return
        
        results = []
        
        for _, row in self.param_mapping.iterrows():
            run_index = row['run_index']
            image_model = row['image_model']
            lidar_model = row['lidar_model']
            publishing_rate = row['publishing_rate']
            
            if run_index not in self.layer_timings_data:
                continue
                
            layer_df = self.layer_timings_data[run_index]
            
            # Analyze image model layers
            image_layers = layer_df[layer_df['Model'] == image_model]
            if len(image_layers) > 0:
                for layer_name in image_layers['Layer'].unique():
                    layer_data = image_layers[image_layers['Layer'] == layer_name]
                    if len(layer_data) > 0:
                        layer_stats = {
                            'run_index': run_index,
                            'model_type': 'image',
                            'model_name': image_model,
                            'layer_name': layer_name,
                            'publishing_rate': publishing_rate,
                            'count': len(layer_data),
                            'elapsed_mean': layer_data['Elapsed Time'].mean(),
                            'elapsed_std': layer_data['Elapsed Time'].std(),
                            'gpu_computation_mean': layer_data['GPU Computation Time'].mean(),
                            'gpu_computation_std': layer_data['GPU Computation Time'].std(),
                            'gpu_wait_mean': layer_data['GPU Wait Time'].mean(),
                            'gpu_wait_std': layer_data['GPU Wait Time'].std(),
                        }
                        results.append(layer_stats)
            
            # Analyze lidar model layers
            lidar_layers = layer_df[layer_df['Model'] == lidar_model]
            if len(lidar_layers) > 0:
                for layer_name in lidar_layers['Layer'].unique():
                    layer_data = lidar_layers[lidar_layers['Layer'] == layer_name]
                    if len(layer_data) > 0:
                        layer_stats = {
                            'run_index': run_index,
                            'model_type': 'lidar',
                            'model_name': lidar_model,
                            'layer_name': layer_name,
                            'publishing_rate': publishing_rate,
                            'count': len(layer_data),
                            'elapsed_mean': layer_data['Elapsed Time'].mean(),
                            'elapsed_std': layer_data['Elapsed Time'].std(),
                            'gpu_computation_mean': layer_data['GPU Computation Time'].mean(),
                            'gpu_computation_std': layer_data['GPU Computation Time'].std(),
                            'gpu_wait_mean': layer_data['GPU Wait Time'].mean(),
                            'gpu_wait_std': layer_data['GPU Wait Time'].std(),
                        }
                        results.append(layer_stats)
        
        self.layer_analysis = pd.DataFrame(results)
        return self.layer_analysis
    
    def create_delays_visualizations(self):
        """Create visualizations for delays analysis"""
        if not hasattr(self, 'delays_analysis'):
            print("Delays analysis not performed yet!")
            return
        
        # Create output directory
        output_dir = Path("analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 1. E2E Delay by Publishing Rate
        plt.figure(figsize=(15, 10))
        
        # Image models
        plt.subplot(2, 2, 1)
        image_data = self.delays_analysis[self.delays_analysis['model_type'] == 'image']
        for model in image_data['model_name'].unique():
            model_data = image_data[image_data['model_name'] == model]
            plt.errorbar(model_data['publishing_rate'], model_data['e2e_mean'], 
                        yerr=model_data['e2e_std'], marker='o', label=model, capsize=5)
        plt.xlabel('Publishing Rate (Hz)')
        plt.ylabel('E2E Delay (s)')
        plt.title('Image Model E2E Delay vs Publishing Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Lidar models
        plt.subplot(2, 2, 2)
        lidar_data = self.delays_analysis[self.delays_analysis['model_type'] == 'lidar']
        for model in lidar_data['model_name'].unique():
            model_data = lidar_data[lidar_data['model_name'] == model]
            plt.errorbar(model_data['publishing_rate'], model_data['e2e_mean'], 
                        yerr=model_data['e2e_std'], marker='s', label=model, capsize=5)
        plt.xlabel('Publishing Rate (Hz)')
        plt.ylabel('E2E Delay (s)')
        plt.title('Lidar Model E2E Delay vs Publishing Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Communication Delay Analysis
        plt.subplot(2, 2, 3)
        for model_type in ['image', 'lidar']:
            type_data = self.delays_analysis[self.delays_analysis['model_type'] == model_type]
            for model in type_data['model_name'].unique():
                model_data = type_data[type_data['model_name'] == model]
                marker = 'o' if model_type == 'image' else 's'
                plt.errorbar(model_data['publishing_rate'], model_data['comm_mean'], 
                            yerr=model_data['comm_std'], marker=marker, 
                            label=f"{model} ({model_type})", capsize=5)
        plt.xlabel('Publishing Rate (Hz)')
        plt.ylabel('Communication Delay (s)')
        plt.title('Communication Delay vs Publishing Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 3. Decode Delay Analysis
        plt.subplot(2, 2, 4)
        for model_type in ['image', 'lidar']:
            type_data = self.delays_analysis[self.delays_analysis['model_type'] == model_type]
            for model in type_data['model_name'].unique():
                model_data = type_data[type_data['model_name'] == model]
                marker = 'o' if model_type == 'image' else 's'
                plt.errorbar(model_data['publishing_rate'], model_data['decode_mean'], 
                            yerr=model_data['decode_std'], marker=marker, 
                            label=f"{model} ({model_type})", capsize=5)
        plt.xlabel('Publishing Rate (Hz)')
        plt.ylabel('Decode Delay (s)')
        plt.title('Decode Delay vs Publishing Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'delays_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Detailed breakdown by model pair
        self._create_model_pair_breakdown(output_dir)
    
    def _create_model_pair_breakdown(self, output_dir):
        """Create detailed breakdown by model pair"""
        # Group by model pairs
        model_pairs = self.param_mapping.groupby(['image_model', 'lidar_model']).size().reset_index()
        
        for _, pair in model_pairs.iterrows():
            image_model = pair['image_model']
            lidar_model = pair['lidar_model']
            
            # Get runs for this pair
            pair_runs = self.param_mapping[
                (self.param_mapping['image_model'] == image_model) & 
                (self.param_mapping['lidar_model'] == lidar_model)
            ]
            
            if len(pair_runs) == 0:
                continue
            
            plt.figure(figsize=(15, 10))
            
            # E2E delays for this pair
            plt.subplot(2, 2, 1)
            for run_idx in pair_runs['run_index']:
                if run_idx in self.delays_data:
                    delays_df = self.delays_data[run_idx]
                    publishing_rate = pair_runs[pair_runs['run_index'] == run_idx]['publishing_rate'].iloc[0]
                    
                    # Image model
                    image_data = delays_df[delays_df['model_type'] == 'image']
                    if len(image_data) > 0:
                        plt.scatter([publishing_rate] * len(image_data), image_data['e2e_delay'], 
                                  alpha=0.6, label=f'Image ({publishing_rate}Hz)', s=20)
                    
                    # Lidar model
                    lidar_data = delays_df[delays_df['model_type'] == 'lidar']
                    if len(lidar_data) > 0:
                        plt.scatter([publishing_rate] * len(lidar_data), lidar_data['e2e_delay'], 
                                  alpha=0.6, label=f'Lidar ({publishing_rate}Hz)', s=20, marker='s')
            
            plt.xlabel('Publishing Rate (Hz)')
            plt.ylabel('E2E Delay (s)')
            plt.title(f'E2E Delays: {image_model} + {lidar_model}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Communication delays
            plt.subplot(2, 2, 2)
            for run_idx in pair_runs['run_index']:
                if run_idx in self.delays_data:
                    delays_df = self.delays_data[run_idx]
                    publishing_rate = pair_runs[pair_runs['run_index'] == run_idx]['publishing_rate'].iloc[0]
                    
                    image_data = delays_df[delays_df['model_type'] == 'image']
                    if len(image_data) > 0:
                        plt.scatter([publishing_rate] * len(image_data), image_data['comm_delay'], 
                                  alpha=0.6, s=20)
                    
                    lidar_data = delays_df[delays_df['model_type'] == 'lidar']
                    if len(lidar_data) > 0:
                        plt.scatter([publishing_rate] * len(lidar_data), lidar_data['comm_delay'], 
                                  alpha=0.6, s=20, marker='s')
            
            plt.xlabel('Publishing Rate (Hz)')
            plt.ylabel('Communication Delay (s)')
            plt.title(f'Communication Delays: {image_model} + {lidar_model}')
            plt.grid(True, alpha=0.3)
            
            # Inference delays
            plt.subplot(2, 2, 3)
            for run_idx in pair_runs['run_index']:
                if run_idx in self.delays_data:
                    delays_df = self.delays_data[run_idx]
                    publishing_rate = pair_runs[pair_runs['run_index'] == run_idx]['publishing_rate'].iloc[0]
                    
                    image_data = delays_df[delays_df['model_type'] == 'image']
                    if len(image_data) > 0:
                        plt.scatter([publishing_rate] * len(image_data), image_data['inference_delay'], 
                                  alpha=0.6, s=20)
                    
                    lidar_data = delays_df[delays_df['model_type'] == 'lidar']
                    if len(lidar_data) > 0:
                        plt.scatter([publishing_rate] * len(lidar_data), lidar_data['inference_delay'], 
                                  alpha=0.6, s=20, marker='s')
            
            plt.xlabel('Publishing Rate (Hz)')
            plt.ylabel('Inference Delay (s)')
            plt.title(f'Inference Delays: {image_model} + {lidar_model}')
            plt.grid(True, alpha=0.3)
            
            # Decode delays
            plt.subplot(2, 2, 4)
            for run_idx in pair_runs['run_index']:
                if run_idx in self.delays_data:
                    delays_df = self.delays_data[run_idx]
                    publishing_rate = pair_runs[pair_runs['run_index'] == run_idx]['publishing_rate'].iloc[0]
                    
                    image_data = delays_df[delays_df['model_type'] == 'image']
                    if len(image_data) > 0:
                        plt.scatter([publishing_rate] * len(image_data), image_data['decode_delay'], 
                                  alpha=0.6, s=20)
                    
                    lidar_data = delays_df[delays_df['model_type'] == 'lidar']
                    if len(lidar_data) > 0:
                        plt.scatter([publishing_rate] * len(lidar_data), lidar_data['decode_delay'], 
                                  alpha=0.6, s=20, marker='s')
            
            plt.xlabel('Publishing Rate (Hz)')
            plt.ylabel('Decode Delay (s)')
            plt.title(f'Decode Delays: {image_model} + {lidar_model}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_filename = f"model_pair_{image_model.replace('/', '_')}_{lidar_model.replace('/', '_').replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')}.png"
            plt.savefig(output_dir / safe_filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_layer_timings_visualizations(self):
        """Create visualizations for layer timings analysis"""
        if not hasattr(self, 'layer_analysis'):
            print("Layer analysis not performed yet!")
            return
        
        output_dir = Path("analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Layer timing breakdown by model
        plt.figure(figsize=(20, 12))
        
        # Image models layer analysis
        plt.subplot(2, 2, 1)
        image_layers = self.layer_analysis[self.layer_analysis['model_type'] == 'image']
        if len(image_layers) > 0:
            layer_summary = image_layers.groupby(['model_name', 'layer_name'])['elapsed_mean'].mean().reset_index()
            pivot_data = layer_summary.pivot(index='layer_name', columns='model_name', values='elapsed_mean')
            pivot_data.plot(kind='bar', ax=plt.gca())
            plt.title('Image Model Layer Timings')
            plt.ylabel('Elapsed Time (s)')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Lidar models layer analysis
        plt.subplot(2, 2, 2)
        lidar_layers = self.layer_analysis[self.layer_analysis['model_type'] == 'lidar']
        if len(lidar_layers) > 0:
            layer_summary = lidar_layers.groupby(['model_name', 'layer_name'])['elapsed_mean'].mean().reset_index()
            pivot_data = layer_summary.pivot(index='layer_name', columns='model_name', values='elapsed_mean')
            pivot_data.plot(kind='bar', ax=plt.gca())
            plt.title('Lidar Model Layer Timings')
            plt.ylabel('Elapsed Time (s)')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # GPU computation vs wait time
        plt.subplot(2, 2, 3)
        for model_type in ['image', 'lidar']:
            type_data = self.layer_analysis[self.layer_analysis['model_type'] == model_type]
            if len(type_data) > 0:
                for model in type_data['model_name'].unique():
                    model_data = type_data[type_data['model_name'] == model]
                    marker = 'o' if model_type == 'image' else 's'
                    plt.scatter(model_data['gpu_computation_mean'], model_data['gpu_wait_mean'], 
                              marker=marker, label=f"{model} ({model_type})", alpha=0.7)
        plt.xlabel('GPU Computation Time (s)')
        plt.ylabel('GPU Wait Time (s)')
        plt.title('GPU Computation vs Wait Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Layer timing by publishing rate
        plt.subplot(2, 2, 4)
        for model_type in ['image', 'lidar']:
            type_data = self.layer_analysis[self.layer_analysis['model_type'] == model_type]
            if len(type_data) > 0:
                for model in type_data['model_name'].unique():
                    model_data = type_data[type_data['model_name'] == model]
                    marker = 'o' if model_type == 'image' else 's'
                    plt.scatter(model_data['publishing_rate'], model_data['elapsed_mean'], 
                              marker=marker, label=f"{model} ({model_type})", alpha=0.7)
        plt.xlabel('Publishing Rate (Hz)')
        plt.ylabel('Layer Elapsed Time (s)')
        plt.title('Layer Timing vs Publishing Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'layer_timings_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not hasattr(self, 'delays_analysis') or not hasattr(self, 'layer_analysis'):
            print("Analyses not performed yet!")
            return
        
        output_dir = Path("analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Create summary report
        with open(output_dir / 'performance_summary_report.txt', 'w') as f:
            f.write("PERFORMANCE ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("1. OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total experiments analyzed: {len(self.param_mapping)}\n")
            f.write(f"Total delay records: {sum(len(df) for df in self.delays_data.values())}\n")
            f.write(f"Total layer timing records: {sum(len(df) for df in self.layer_timings_data.values())}\n\n")
            
            # Publishing rates used
            f.write("2. PUBLISHING RATES ANALYZED\n")
            f.write("-" * 30 + "\n")
            rates = sorted(self.param_mapping['publishing_rate'].unique())
            f.write(f"Publishing rates: {rates} Hz\n\n")
            
            # Model pairs analyzed
            f.write("3. MODEL PAIRS ANALYZED\n")
            f.write("-" * 25 + "\n")
            model_pairs = self.param_mapping.groupby(['image_model', 'lidar_model']).size()
            for (img, lidar), count in model_pairs.items():
                f.write(f"Image: {img}\n")
                f.write(f"Lidar: {lidar}\n")
                f.write(f"Experiments: {count}\n")
                f.write("-" * 10 + "\n")
            f.write("\n")
            
            # Performance summary by model type
            f.write("4. PERFORMANCE SUMMARY BY MODEL TYPE\n")
            f.write("-" * 40 + "\n")
            
            for model_type in ['image', 'lidar']:
                f.write(f"\n{model_type.upper()} MODELS:\n")
                type_data = self.delays_analysis[self.delays_analysis['model_type'] == model_type]
                
                for model in type_data['model_name'].unique():
                    model_data = type_data[type_data['model_name'] == model]
                    f.write(f"\n  {model}:\n")
                    f.write(f"    E2E Delay: {model_data['e2e_mean'].mean():.4f} ± {model_data['e2e_std'].mean():.4f} s\n")
                    f.write(f"    Communication Delay: {model_data['comm_mean'].mean():.4f} ± {model_data['comm_std'].mean():.4f} s\n")
                    f.write(f"    Decode Delay: {model_data['decode_mean'].mean():.4f} ± {model_data['decode_std'].mean():.4f} s\n")
                    f.write(f"    Inference Delay: {model_data['inference_mean'].mean():.4f} ± {model_data['inference_std'].mean():.4f} s\n")
            
            # Publishing rate impact
            f.write("\n5. PUBLISHING RATE IMPACT ANALYSIS\n")
            f.write("-" * 35 + "\n")
            
            for rate in rates:
                f.write(f"\nPublishing Rate: {rate} Hz\n")
                rate_data = self.delays_analysis[self.delays_analysis['publishing_rate'] == rate]
                
                for model_type in ['image', 'lidar']:
                    type_data = rate_data[rate_data['model_type'] == model_type]
                    if len(type_data) > 0:
                        f.write(f"  {model_type.capitalize()} models:\n")
                        f.write(f"    Avg E2E Delay: {type_data['e2e_mean'].mean():.4f} s\n")
                        f.write(f"    Avg Communication Delay: {type_data['comm_mean'].mean():.4f} s\n")
                        f.write(f"    Avg Decode Delay: {type_data['decode_mean'].mean():.4f} s\n")
                        f.write(f"    Avg Inference Delay: {type_data['inference_mean'].mean():.4f} s\n")
            
            # Layer timing insights
            f.write("\n6. LAYER TIMING INSIGHTS\n")
            f.write("-" * 25 + "\n")
            
            for model_type in ['image', 'lidar']:
                f.write(f"\n{model_type.upper()} MODEL LAYERS:\n")
                type_data = self.layer_analysis[self.layer_analysis['model_type'] == model_type]
                
                if len(type_data) > 0:
                    layer_summary = type_data.groupby(['model_name', 'layer_name']).agg({
                        'elapsed_mean': 'mean',
                        'gpu_computation_mean': 'mean',
                        'gpu_wait_mean': 'mean'
                    }).reset_index()
                    
                    for model in layer_summary['model_name'].unique():
                        model_layers = layer_summary[layer_summary['model_name'] == model]
                        f.write(f"\n  {model}:\n")
                        for _, layer in model_layers.iterrows():
                            f.write(f"    {layer['layer_name']}: {layer['elapsed_mean']:.4f}s "
                                  f"(GPU comp: {layer['gpu_computation_mean']:.4f}s, "
                                  f"wait: {layer['gpu_wait_mean']:.4f}s)\n")
        
        print(f"Summary report saved to {output_dir / 'performance_summary_report.txt'}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Performance Analysis...")
        print("=" * 50)
        
        # Load data
        if not self.load_param_mapping():
            return
        
        self.load_delays_data()
        self.load_layer_timings_data()
        
        # Perform analyses
        print("\nAnalyzing delays...")
        self.analyze_delays_by_model_pair()
        
        print("\nAnalyzing layer timings...")
        self.analyze_layer_timings_by_model_pair()
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.create_delays_visualizations()
        self.create_layer_timings_visualizations()
        
        # Generate report
        print("\nGenerating summary report...")
        self.generate_summary_report()
        
        print("\nAnalysis complete! Check the 'analysis_outputs' directory for results.")

def main():
    """Main function to run the analysis"""
    analyzer = PerformanceAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 