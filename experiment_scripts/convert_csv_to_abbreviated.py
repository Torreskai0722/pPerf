#!/usr/bin/env python3
"""
Script to convert existing full_stack_mapping.csv files from full model names 
to abbreviated names using the model_name_mappings and scenes_mapping.
"""

import sys
sys.path.append('/home/mg/pdnn/pPerf/perf_ws/src/p_perf')

import pandas as pd
import ast
import os
from p_perf.config.constant import (
    get_abbreviated_name,
    get_abbreviated_scene
)


def convert_csv_to_abbreviated(csv_path, backup=True):
    """
    Convert CSV file from full model names to abbreviated names.
    
    Args:
        csv_path: Path to the CSV file to convert
        backup: If True, creates a backup before modifying
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False
    
    # Create backup if requested
    if backup:
        backup_path = csv_path.replace('.csv', '_backup.csv')
        df_backup = pd.read_csv(csv_path)
        df_backup.to_csv(backup_path, index=False)
        print(f"✓ Created backup at: {backup_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    print(f"\nProcessing {len(df)} rows from {csv_path}")
    
    # Track changes
    changes_made = 0
    
    # Convert scene tokens to abbreviated names
    if 'scene' in df.columns:
        original_scenes = df['scene'].tolist()
        df['scene'] = df['scene'].apply(get_abbreviated_scene)
        scene_changes = sum(1 for orig, new in zip(original_scenes, df['scene']) if orig != new)
        if scene_changes > 0:
            print(f"  - Abbreviated {scene_changes} scene tokens")
            changes_made += scene_changes
    
    # Convert image model names to abbreviated names
    if 'image_model' in df.columns:
        # Filter out NaN values
        mask = df['image_model'].notna()
        original_models = df.loc[mask, 'image_model'].tolist()
        df.loc[mask, 'image_model'] = df.loc[mask, 'image_model'].apply(get_abbreviated_name)
        model_changes = sum(1 for orig, new in zip(original_models, df.loc[mask, 'image_model']) if orig != new)
        if model_changes > 0:
            print(f"  - Abbreviated {model_changes} image model names")
            changes_made += model_changes
    
    # Convert lidar model tuples to abbreviated names (keep tuple structure)
    if 'lidar_model' in df.columns:
        mask = df['lidar_model'].notna()
        original_lidars = []
        new_lidars = []
        
        for idx, lidar_str in df.loc[mask, 'lidar_model'].items():
            try:
                # Parse the tuple string
                lidar_tuple = ast.literal_eval(lidar_str)
                original_lidars.append(lidar_tuple)
                
                # Create abbreviated tuple (abbreviated_name, mode)
                abbreviated_tuple = (get_abbreviated_name(lidar_tuple[0]), lidar_tuple[1])
                new_lidars.append(abbreviated_tuple)
                
                # Update the dataframe
                df.at[idx, 'lidar_model'] = str(abbreviated_tuple)
            except (ValueError, SyntaxError) as e:
                print(f"  Warning: Could not parse lidar_model at row {idx}: {e}")
        
        lidar_changes = sum(1 for orig, new in zip(original_lidars, new_lidars) if orig[0] != new[0])
        if lidar_changes > 0:
            print(f"  - Abbreviated {lidar_changes} lidar model names")
            changes_made += lidar_changes
    
    # Convert seg model tuples to abbreviated names (if present)
    if 'seg_model' in df.columns:
        mask = df['seg_model'].notna()
        original_segs = []
        new_segs = []
        
        for idx, seg_str in df.loc[mask, 'seg_model'].items():
            try:
                # Parse the tuple string
                seg_tuple = ast.literal_eval(seg_str)
                original_segs.append(seg_tuple)
                
                # Create abbreviated tuple (abbreviated_name, mode)
                abbreviated_tuple = (get_abbreviated_name(seg_tuple[0]), seg_tuple[1])
                new_segs.append(abbreviated_tuple)
                
                # Update the dataframe
                df.at[idx, 'seg_model'] = str(abbreviated_tuple)
            except (ValueError, SyntaxError) as e:
                print(f"  Warning: Could not parse seg_model at row {idx}: {e}")
        
        seg_changes = sum(1 for orig, new in zip(original_segs, new_segs) if orig[0] != new[0])
        if seg_changes > 0:
            print(f"  - Abbreviated {seg_changes} seg model names")
            changes_made += seg_changes
    
    # Save the updated CSV
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Successfully updated {csv_path}")
    print(f"  Total changes: {changes_made}")
    
    return True


def main():
    """Main function to process CSV files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert full_stack_mapping.csv to use abbreviated model names'
    )
    parser.add_argument(
        'csv_path',
        help='Path to the CSV file to convert'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file'
    )
    
    args = parser.parse_args()
    
    # Convert the CSV
    success = convert_csv_to_abbreviated(
        args.csv_path,
        backup=not args.no_backup
    )
    
    if success:
        print("\n✓ Conversion completed successfully!")
    else:
        print("\n✗ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

