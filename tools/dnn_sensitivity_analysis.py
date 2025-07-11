import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import scipy.stats
import ast

# Import common utilities
from utils import (
    model_name_mappings, image_models, lidar_models,
    read_param_mapping, get_run_indices, get_counter_models, 
    read_layer_timings, get_model_short_name, create_output_directory
)

class E2ECorrelationAnalyzer:
    """E2E correlation analysis for layer-to-end-to-end latency relationships."""
    
    def __init__(self, baseline_timings: pd.DataFrame, multi_tenant_dir: str, 
                 model_name: str, model_type: str, **filter_params):
        self.baseline_timings = baseline_timings
        self.multi_tenant_dir = multi_tenant_dir
        self.model_name = model_name
        self.model_type = model_type
        self.filter_params = filter_params
        
        # Load parameter mapping and get run data
        self.param_mapping = read_param_mapping(multi_tenant_dir)
        self.model_run_indices = get_run_indices(self.param_mapping, model_name, model_type, **filter_params)
        self.counter_models = get_counter_models(self.param_mapping, self.model_run_indices, model_type)
        
    def compute_layer_e2e_correlation(self, output_dir=None):
        """
        For each counter model, compute Pearson correlation between each layer's runtime and the e2e latency.
        """
        # Exclude 'e2e' from the list of layers to be analyzed
        all_layers = sorted([layer for layer in self.baseline_timings['Layer'].unique() if layer != 'e2e'])
        corr_matrix = pd.DataFrame(index=all_layers, columns=self.counter_models, dtype=float)

        for counter_model in self.counter_models:
            run_indices = [idx for idx in self.model_run_indices if (
                (self.model_type == 'image' and ast.literal_eval(self.param_mapping.loc[self.param_mapping['run_index'] == idx, 'lidar_model'].iloc[0])[0] == counter_model)
                or (self.model_type == 'lidar' and self.param_mapping.loc[self.param_mapping['run_index'] == idx, 'image_model'].iloc[0] == counter_model)
            )]
            if not run_indices:
                continue
            
            # Load timings for the first run of this counter model
            timing = read_layer_timings(self.multi_tenant_dir, run_indices[0])
            
            # Get e2e latency per input from layer_timings.csv
            e2e_latency = timing[timing['Layer'] == 'e2e'].set_index('Input')['GPU Turnaround Time']
            
            for layer in all_layers:
                layer_data = timing[timing['Layer'] == layer]
                if layer_data.empty:
                    corr = np.nan
                else:
                    merged = pd.merge(layer_data[['Input', 'Elapsed Time']], e2e_latency.rename('E2E Latency'), left_on='Input', right_index=True)
                    if len(merged) < 2:
                        corr = np.nan
                    else:
                        corr, _ = scipy.stats.pearsonr(merged['Elapsed Time'], merged['E2E Latency'])
                corr_matrix.loc[layer, counter_model] = corr

        # Compute summary statistics
        summary = pd.DataFrame({
            'Layer': corr_matrix.index,
            'MeanCorr': corr_matrix.mean(axis=1, skipna=True),
            'StdCorr': corr_matrix.std(axis=1, skipna=True),
            'MaxCorr': corr_matrix.max(axis=1, skipna=True),
            'MinCorr': corr_matrix.min(axis=1, skipna=True),
        })
        summary = summary.sort_values('MeanCorr', ascending=False).reset_index(drop=True)

        # Optionally save the full correlation matrix
        if output_dir is not None:
            create_output_directory(output_dir)
            corr_matrix.to_csv(os.path.join(output_dir, f'{self.model_name}_layer_e2e_corr_matrix.csv'))
            summary.to_csv(os.path.join(output_dir, f'{self.model_name}_layer_e2e_corr_summary.csv'), index=False)
        
        return summary, corr_matrix
    
    def generate_correlation_report(self, output_dir: str):
        """Generate e2e correlation analysis report."""
        create_output_directory(output_dir)
        
        # Compute correlations
        corr_summary, corr_matrix = self.compute_layer_e2e_correlation(output_dir)
        
        # Create detailed report
        with open(os.path.join(output_dir, f'{self.model_name}_e2e_correlation_report.txt'), 'w') as f:
            f.write(f"E2E Correlation Analysis for {get_model_short_name(self.model_name)}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model Information:\n")
            f.write(f"  Name: {self.model_name}\n")
            f.write(f"  Type: {self.model_type}\n")
            f.write(f"  Counter Models: {', '.join(self.counter_models)}\n\n")
            
            f.write(f"Correlation Summary (Top 10 by MeanCorr):\n")
            f.write(corr_summary.head(10).to_string(index=False))
            f.write("\n\n")
            
            f.write(f"Full Correlation Summary:\n")
            f.write(corr_summary.to_string(index=False))
            f.write("\n\n")
        
        # Create visualizations
        self._create_correlation_visualizations(output_dir, corr_summary, corr_matrix)
        
        print('Layer-TotalRuntime Correlation Summary:')
        print(corr_summary.head())
        
        return corr_summary, corr_matrix
    
    def _create_correlation_visualizations(self, output_dir: str, corr_summary: pd.DataFrame, corr_matrix: pd.DataFrame):
        """Create correlation visualizations."""
        # Top layers by mean correlation
        plt.figure(figsize=(12, 8))
        top_layers = corr_summary.head(15)
        plt.barh(top_layers['Layer'], top_layers['MeanCorr'])
        plt.xlabel('Mean Correlation with E2E Latency')
        plt.title(f'Top Layers by E2E Correlation - {get_model_short_name(self.model_name)}')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.model_name}_top_e2e_correlation.png'))
        plt.close()
        
        # Correlation heatmap
        if len(corr_matrix) > 0 and len(corr_matrix.columns) > 0:
            plt.figure(figsize=(12, max(8, len(corr_matrix) * 0.3)))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
            plt.title(f'Layer-E2E Correlation Heatmap - {get_model_short_name(self.model_name)}')
            plt.xlabel('Counter Models')
            plt.ylabel('Layers')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{self.model_name}_e2e_correlation_heatmap.png'))
            plt.close()

def analyze_e2e_correlation(baseline_timings: pd.DataFrame, multi_tenant_dir: str,
                          output_dir: str, model_name: str, model_type: str, **filter_params):
    """Main function to perform e2e correlation analysis."""
    print(f"Analyzing E2E correlation for {get_model_short_name(model_name)} ({model_type})")
    
    # Create analyzer
    analyzer = E2ECorrelationAnalyzer(baseline_timings, multi_tenant_dir, model_name, model_type, **filter_params)
    
    # Generate report
    corr_summary, corr_matrix = analyzer.generate_correlation_report(output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return analyzer

if __name__ == "__main__":
    # Example usage
    lidar_base_dir = "outputs/lidar_base"
    image_base_dir = "outputs/image_base"
    multi_tenant_dir = "outputs/ms"
    
    # Analyze image models
    for model_name in image_models:
        model_name_short = get_model_short_name(model_name)
        output_dir = f"tools/results/reports_ms_00/dnn_sensitivity/{model_name_short}"
        model_type = "image"
        
        baseline_param_mapping = read_param_mapping(image_base_dir)
        base_index = get_run_indices(baseline_param_mapping, model_name, model_type)
        baseline_timings = pd.read_csv(os.path.join(image_base_dir, f"layer_timings_{base_index[0]}.csv"))
        
        print(f"\nAnalyzing {model_name_short} ({model_type})")
        analyze_e2e_correlation(baseline_timings, multi_tenant_dir, output_dir, model_name, model_type, image_priority=0, lidar_priority=0)
    
    # Analyze lidar models
    for model_tuple in lidar_models:
        model_name = model_tuple[0]
        model_name_short = get_model_short_name(model_name)
        output_dir = f"tools/results/reports_ms_00/dnn_sensitivity/{model_name_short}"
        model_type = "lidar"
        
        baseline_param_mapping = read_param_mapping(lidar_base_dir)
        base_index = get_run_indices(baseline_param_mapping, model_name, model_type)
        baseline_timings = pd.read_csv(os.path.join(lidar_base_dir, f"layer_timings_{base_index[0]}.csv"))
        
        print(f"\nAnalyzing {model_name_short} ({model_type})")
        analyze_e2e_correlation(baseline_timings, multi_tenant_dir, output_dir, model_name, model_type, image_priority=0, lidar_priority=0) 