#!/usr/bin/env python3
"""
YAML Configuration Manager for pPerf
Dynamically read, modify, and write YAML configuration files
"""

import yaml
import os
import argparse
import copy
from typing import Dict, Any, List, Optional
from pathlib import Path


class pPerfConfigManager:
    """Manages YAML configuration files for pPerf launch system"""
    
    def __init__(self, config_file: str):
        """Initialize with a YAML configuration file"""
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                print(f"✓ Loaded configuration from {self.config_file}")
                return config or {}
        except FileNotFoundError:
            print(f"⚠ Configuration file {self.config_file} not found, creating new one")
            return {}
        except yaml.YAMLError as e:
            print(f"✗ Error parsing YAML file: {e}")
            return {}
    
    def save_config(self, output_file: Optional[str] = None) -> None:
        """Save configuration to YAML file"""
        output_path = Path(output_file) if output_file else self.config_file
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2, sort_keys=False)
            print(f"✓ Configuration saved to {output_path}")
        except Exception as e:
            print(f"✗ Error saving configuration: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return copy.deepcopy(self.config)
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set entire configuration"""
        self.config = copy.deepcopy(config)
    
    def update_sensor_replayer(self, **kwargs) -> None:
        """Update sensor replayer configuration"""
        if 'sensor_replayer' not in self.config:
            self.config['sensor_replayer'] = {}
        
        for key, value in kwargs.items():
            self.config['sensor_replayer'][key] = value
            print(f"✓ Updated sensor_replayer.{key} = {value}")
    
    def add_det_inferencer(self, **kwargs) -> None:
        """Add a detection inferencer"""
        if 'det_inferencers' not in self.config:
            self.config['det_inferencers'] = []
        
        # Auto-assign inferencer_index if not provided
        if 'inferencer_index' not in kwargs:
            kwargs['inferencer_index'] = len(self.config['det_inferencers'])
        
        # Auto-assign experiment run index if not provided
        if 'index' not in kwargs:
            kwargs['index'] = 0
        
        self.config['det_inferencers'].append(kwargs)
        mps_info = f" (MPS: {kwargs['cuda_mps_thread_percentage']}%)" if 'cuda_mps_thread_percentage' in kwargs else ""
        print(f"✓ Added det_inferencer_{kwargs['inferencer_index']} (exp_run_{kwargs['index']}) with mode: {kwargs.get('mode', 'unknown')}{mps_info}")
    
    def add_seg_inferencer(self, **kwargs) -> None:
        """Add a segmentation inferencer"""
        if 'seg_inferencers' not in self.config:
            self.config['seg_inferencers'] = []
        
        # Auto-assign inferencer_index if not provided
        if 'inferencer_index' not in kwargs:
            kwargs['inferencer_index'] = len(self.config['seg_inferencers'])
        
        # Auto-assign experiment run index if not provided
        if 'index' not in kwargs:
            kwargs['index'] = 0
        
        self.config['seg_inferencers'].append(kwargs)
        mps_info = f" (MPS: {kwargs['cuda_mps_thread_percentage']}%)" if 'cuda_mps_thread_percentage' in kwargs else ""
        print(f"✓ Added seg_inferencer_{kwargs['inferencer_index']} (exp_run_{kwargs['index']}) with mode: {kwargs.get('mode', 'unknown')}{mps_info}")
    
    def update_det_inferencer(self, inferencer_index: int, **kwargs) -> None:
        """Update a specific detection inferencer by inferencer_index"""
        if 'det_inferencers' not in self.config:
            print(f"✗ No det_inferencers configured")
            return
        
        # Find inferencer by inferencer_index
        inferencer = None
        for i, inf in enumerate(self.config['det_inferencers']):
            if inf.get('inferencer_index') == inferencer_index:
                inferencer = inf
                break
        
        if inferencer is None:
            print(f"✗ det_inferencer with inferencer_index {inferencer_index} not found")
            return
        
        for key, value in kwargs.items():
            inferencer[key] = value
            print(f"✓ Updated det_inferencer_{inferencer_index}.{key} = {value}")
    
    def update_seg_inferencer(self, inferencer_index: int, **kwargs) -> None:
        """Update a specific segmentation inferencer by inferencer_index"""
        if 'seg_inferencers' not in self.config:
            print(f"✗ No seg_inferencers configured")
            return
        
        # Find inferencer by inferencer_index
        inferencer = None
        for i, inf in enumerate(self.config['seg_inferencers']):
            if inf.get('inferencer_index') == inferencer_index:
                inferencer = inf
                break
        
        if inferencer is None:
            print(f"✗ seg_inferencer with inferencer_index {inferencer_index} not found")
            return
        
        for key, value in kwargs.items():
            inferencer[key] = value
            print(f"✓ Updated seg_inferencer_{inferencer_index}.{key} = {value}")
    

    
    def update_inferencer(self, section: str, inferencer_index: int, **kwargs) -> None:
        """General function to update any inferencer in any section by inferencer_index"""
        if section not in self.config:
            print(f"✗ Section '{section}' not found")
            return
        
        if not isinstance(self.config[section], list):
            print(f"✗ Section '{section}' is not a list of inferencers")
            return
        
        # Find inferencer by inferencer_index
        inferencer = None
        for i, inf in enumerate(self.config[section]):
            if inf.get('inferencer_index') == inferencer_index:
                inferencer = inf
                break
        
        if inferencer is None:
            print(f"✗ {section} with inferencer_index {inferencer_index} not found")
            return
        
        # Update the inferencer
        for key, value in kwargs.items():
            inferencer[key] = value
            print(f"✓ Updated {section}_{inferencer_index}.{key} = {value}")
    
    def get_inferencer(self, section: str, inferencer_index: int) -> Optional[Dict[str, Any]]:
        """Get an inferencer from a specific section by inferencer_index"""
        if section not in self.config:
            print(f"✗ Section '{section}' not found")
            return None
        
        if not isinstance(self.config[section], list):
            print(f"✗ Section '{section}' is not a list of inferencers")
            return None
        
        # Find inferencer by inferencer_index
        for inf in self.config[section]:
            if inf.get('inferencer_index') == inferencer_index:
                return inf
        
        print(f"✗ {section} with inferencer_index {inferencer_index} not found")
        return None

    
    def remove_det_inferencer(self, inferencer_index: int) -> None:
        """Remove a detection inferencer by inferencer_index"""
        if 'det_inferencers' not in self.config:
            print(f"✗ No det_inferencers configured")
            return
        
        # Find and remove inferencer by inferencer_index
        for i, inferencer in enumerate(self.config['det_inferencers']):
            if inferencer.get('inferencer_index') == inferencer_index:
                removed = self.config['det_inferencers'].pop(i)
                print(f"✓ Removed det_inferencer_{inferencer_index} (mode: {removed.get('mode', 'unknown')})")
                
                # Reindex remaining inferencers
                for j, remaining_inf in enumerate(self.config['det_inferencers']):
                    remaining_inf['inferencer_index'] = j
                return
        
        print(f"✗ det_inferencer with inferencer_index {inferencer_index} not found")
    
    def remove_seg_inferencer(self, inferencer_index: int) -> None:
        """Remove a segmentation inferencer by inferencer_index"""
        if 'seg_inferencers' not in self.config:
            print(f"✗ No seg_inferencers configured")
            return
        
        # Find and remove inferencer by inferencer_index
        for i, inferencer in enumerate(self.config['seg_inferencers']):
            if inferencer.get('inferencer_index') == inferencer_index:
                removed = self.config['seg_inferencers'].pop(i)
                print(f"✓ Removed seg_inferencer_{inferencer_index} (mode: {removed.get('mode', 'unknown')})")
                
                # Reindex remaining inferencers
                for j, remaining_inf in enumerate(self.config['seg_inferencers']):
                    remaining_inf['inferencer_index'] = j
                return
        
        print(f"✗ seg_inferencer with inferencer_index {inferencer_index} not found")
    
    def clear_section(self, section: str) -> None:
        """Clear a specific section"""
        if section in self.config:
            self.config[section] = [] if section.endswith('s') else {}
            print(f"✓ Cleared {section} section")
        else:
            print(f"✗ Section '{section}' not found")
    
    def list_inferencers(self) -> None:
        """List all configured inferencers"""
        print("\n📋 Current Configuration:")
        print("=" * 50)
        
        # Sensor Replayer
        if 'sensor_replayer' in self.config:
            print(f"🔴 Sensor Replayer:")
            for key, value in self.config['sensor_replayer'].items():
                print(f"   {key}: {value}")
        else:
            print("🔴 Sensor Replayer: Not configured")
        
        # Detection Inferencers
        if 'det_inferencers' in self.config and self.config['det_inferencers']:
            print(f"\n🟡 Detection Inferencers ({len(self.config['det_inferencers'])}):")
            for i, inferencer in enumerate(self.config['det_inferencers']):
                exp_run = inferencer.get('index', 0)
                inf_idx = inferencer.get('inferencer_index', i)
                mps_info = f" [MPS: {inferencer['cuda_mps_thread_percentage']}%]" if 'cuda_mps_thread_percentage' in inferencer else ""
                print(f"   [{inf_idx}] {inferencer.get('mode', 'unknown')} - {inferencer.get('model_name', 'unknown')} (exp_run_{exp_run}){mps_info}")
        else:
            print("\n🟡 Detection Inferencers: None configured")
        
        # Segmentation Inferencers
        if 'seg_inferencers' in self.config and self.config['seg_inferencers']:
            print(f"\n🟢 Segmentation Inferencers ({len(self.config['seg_inferencers'])}):")
            for i, inferencer in enumerate(self.config['seg_inferencers']):
                exp_run = inferencer.get('index', 0)
                inf_idx = inferencer.get('inferencer_index', i)
                mps_info = f" [MPS: {inferencer['cuda_mps_thread_percentage']}%]" if 'cuda_mps_thread_percentage' in inferencer else ""
                print(f"   [{inf_idx}] {inferencer.get('mode', 'unknown')} - {inferencer.get('model_name', 'unknown')} (exp_run_{exp_run}){mps_info}")
        else:
            print("\n🟢 Segmentation Inferencers: None configured")
        
        print("=" * 50)

    def create_base_config(self, num_det_inferencers: int, num_seg_inferencers: int, use_sim_time=True, logging_delay=True) -> None:
        """Create a base configuration with default values"""
        if self.config_file.exists():
            print(f"✗ Configuration file {self.config_file} already exists, skipping creation")
            return
        
        expected_models = num_det_inferencers + num_seg_inferencers
        self.update_sensor_replayer(
            use_sim_time=use_sim_time,
            expected_models=expected_models,
            bag_dir="/mmdetection3d_ros2/data/bag",
        )

        for i in range(num_det_inferencers):
            self.add_det_inferencer(
                use_sim_time=use_sim_time,
                inferencer_index=i,
                mode='lidar',
                index=i,
                input_type='bag',
                logging_delay=logging_delay
            )
            
        for i in range(num_seg_inferencers):
            self.add_seg_inferencer(
                use_sim_time=use_sim_time,
                inferencer_index=i,
                mode='sem_seg',
                index=i,
                input_type='bag',
                logging_delay=logging_delay
            )
        
        self.save_config()

