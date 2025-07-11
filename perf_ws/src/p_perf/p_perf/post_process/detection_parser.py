#!/usr/bin/env python3

import json
import os
import numpy as np
import mmengine
from typing import Dict, Any
from p_perf.post_process.lidar_eval import lidar_output_to_nusc_box, lidar_nusc_box_to_global
from p_perf.post_process.image_eval import image_output_to_coco
from p_perf.config.constant import nus_lidar_classes, kitti_lidar_classes
from p_perf.utils import get_closest_token_from_timestamp, build_channel_timestamp_token_map
from p_perf.nuscenes_instance import get_nuscenes_instance


class DetectionParser:
    """Parser for raw detection data saved during inference"""
    
    def __init__(self, nusc=None):
        self.nusc = nusc if nusc is not None else get_nuscenes_instance()
    
    def parse_raw_detections(self, raw_detections_file, output_file, mode, scene, 
                           input_type="publisher", lidar_model_mode="nus", lidar_model_thresh=0.2):
        if mode == 'lidar' or mode == 'multi-modal':
            self._parse_3d_detections(raw_detections_file, output_file, scene, input_type, 
                                    lidar_model_mode, lidar_model_thresh)
        elif mode == 'image':
            self._parse_2d_detections(raw_detections_file, output_file, scene, input_type)
    
    def parse_lidar_detections(self, raw_detections_file, output_file, scene, 
                             input_type="publisher", lidar_model_mode="nus", lidar_model_thresh=0.2):
        """Parse raw lidar detections specifically"""
        self._parse_3d_detections(raw_detections_file, output_file, scene, input_type, 
                                lidar_model_mode, lidar_model_thresh)
    
    def parse_image_detections(self, raw_detections_file, output_file, scene, input_type="publisher"):
        """Parse raw image detections specifically"""
        self._parse_2d_detections(raw_detections_file, output_file, scene, input_type)
    
    def _parse_3d_detections(self, raw_file, output_file, scene, input_type, lidar_model_mode, lidar_model_thresh):
        with open(raw_file, 'r') as f:
            raw_dets = json.load(f)
        
        print(f"Found {len(raw_dets)} raw detection entries")
        if raw_dets:
            # Debug: show structure of first detection
            first_token, first_pred = raw_dets[0]
            print(f"First detection token: {first_token}")
            print(f"First detection keys: {list(first_pred.keys())}")
            if 'bboxes_3d' in first_pred:
                print(f"bboxes_3d shape: {len(first_pred['bboxes_3d'])} boxes")
                if first_pred['bboxes_3d']:
                    print(f"First box dimensions: {len(first_pred['bboxes_3d'][0])}")
        
        nusc_annos = {}
        token_mapping = build_channel_timestamp_token_map(self.nusc, scene, "LIDAR_TOP")
        
        for token, pred_data in raw_dets:
            if input_type != "publisher":
                token = get_closest_token_from_timestamp(token, token_mapping)
            
            pred = self._reconstruct_prediction_object(pred_data)
            
            boxes = lidar_output_to_nusc_box(pred, token, lidar_model_thresh, lidar_model_mode)
            boxes = lidar_nusc_box_to_global(self.nusc, token, boxes)

            annos = []
            for box in boxes:
                if lidar_model_mode == 'nus' or 'car' in lidar_model_mode:
                    name = nus_lidar_classes[box.label]
                else:
                    name = kitti_lidar_classes[box.label]
                nusc_anno = dict(
                    sample_token=token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=float(box.score),
                    attribute_name='')
                annos.append(nusc_anno)
            nusc_annos[token] = annos
        
        nusc_submission = {
            'meta': {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False
            },
            'results': nusc_annos
        }
        
        mmengine.dump(nusc_submission, output_file)
        print(f"Parsed 3D detections written to {output_file}")
    
    def _parse_2d_detections(self, raw_file, output_file, scene, input_type):
        with open(raw_file, 'r') as f:
            raw_dets = json.load(f)
        
        coco_predictions = []
        token_mapping = build_channel_timestamp_token_map(self.nusc, scene, "CAM_FRONT")
        
        for token, pred_data in raw_dets:
            if input_type != "publisher":
                token = get_closest_token_from_timestamp(token, token_mapping)
            
            pred = self._reconstruct_prediction_object(pred_data)
            coco_pred = image_output_to_coco(pred, token)
            coco_predictions.extend(coco_pred)
        
        with open(output_file, 'w') as f:
            json.dump(coco_predictions, f, indent=2)
        
        print(f"Parsed 2D detections written to {output_file}")
    
    def _reconstruct_prediction_object(self, pred_data):
        class PredictionObject:
            def __init__(self, data):
                # For 3D detections
                if 'bboxes_3d' in data:
                    import torch
                    from mmdet3d.structures import LiDARInstance3DBoxes
                    bboxes_3d = torch.tensor(data['bboxes_3d'])
                    
                    # Handle different box dimensions (7 or 9)
                    if bboxes_3d.size(-1) == 9:
                        # If 9 dimensions, take only the first 7 (x, y, z, l, w, h, yaw)
                        # Assuming format: [x, y, z, l, w, h, yaw, vx, vy]
                        bboxes_3d = bboxes_3d[:, :7]
                    elif bboxes_3d.size(-1) != 7:
                        raise ValueError(f"Unexpected box dimension: {bboxes_3d.size(-1)}. Expected 7 or 9.")
                    
                    self.bboxes_3d = LiDARInstance3DBoxes(bboxes_3d)
                if 'scores_3d' in data:
                    import torch
                    self.scores_3d = torch.tensor(data['scores_3d'])
                if 'labels_3d' in data:
                    import torch
                    self.labels_3d = torch.tensor(data['labels_3d'])
                
                # For 2D detections
                if 'bboxes' in data:
                    import torch
                    self.bboxes = torch.tensor(data['bboxes'])
                if 'scores' in data:
                    import torch
                    self.scores = torch.tensor(data['scores'])
                if 'labels' in data:
                    import torch
                    self.labels = torch.tensor(data['labels'])
                
                # Handle nested structures (for cases where data might be wrapped)
                if 'pred_instances_3d' in data:
                    instances_3d = data['pred_instances_3d']
                    if 'bboxes_3d' in instances_3d:
                        import torch
                        from mmdet3d.structures import LiDARInstance3DBoxes
                        bboxes_3d = torch.tensor(instances_3d['bboxes_3d'])
                        
                        # Handle different box dimensions (7 or 9)
                        if bboxes_3d.size(-1) == 9:
                            # If 9 dimensions, take only the first 7 (x, y, z, l, w, h, yaw)
                            bboxes_3d = bboxes_3d[:, :7]
                        elif bboxes_3d.size(-1) != 7:
                            raise ValueError(f"Unexpected box dimension: {bboxes_3d.size(-1)}. Expected 7 or 9.")
                        
                        self.bboxes_3d = LiDARInstance3DBoxes(bboxes_3d)
                    if 'scores_3d' in instances_3d:
                        import torch
                        self.scores_3d = torch.tensor(instances_3d['scores_3d'])
                    if 'labels_3d' in instances_3d:
                        import torch
                        self.labels_3d = torch.tensor(instances_3d['labels_3d'])
                
                if 'pred_instances' in data:
                    instances = data['pred_instances']
                    if 'bboxes' in instances:
                        import torch
                        self.bboxes = torch.tensor(instances['bboxes'])
                    if 'scores' in instances:
                        import torch
                        self.scores = torch.tensor(instances['scores'])
                    if 'labels' in instances:
                        import torch
                        self.labels = torch.tensor(instances['labels'])
        
        return PredictionObject(pred_data)


def parse_multimodal_detections_for_run(run_index, output_base, scene, input_type="publisher", 
                                      lidar_model_mode="nus", lidar_model_thresh=0.2, model_name=None):
    """Parse multi-modal detections (both image and lidar) for a specific run"""
    parser = DetectionParser()
    
    # Construct raw file path based on model name
    if model_name:
        raw_file = os.path.join(output_base, f"raw_detections_{model_name}_{run_index}.json")
    else:
        raw_file = os.path.join(output_base, f"raw_detections_{run_index}.json")
    
    lidar_output_file = os.path.join(output_base, f"lidar_pred_{run_index}.json")
    image_output_file = os.path.join(output_base, f"image_pred_{run_index}.json")
    
    if os.path.exists(raw_file):
        # For multi-modal models, we need to check if the raw data contains both types of detections
        with open(raw_file, 'r') as f:
            raw_dets = json.load(f)
        
        # Check if we have both 2D and 3D detections in the same file
        has_3d = any('bboxes_3d' in pred_data or 'pred_instances_3d' in pred_data 
                    for _, pred_data in raw_dets)
        has_2d = any('bboxes' in pred_data or 'pred_instances' in pred_data 
                    for _, pred_data in raw_dets)
        
        if has_3d:
            print(f"Parsing 3D detections from multi-modal model...")
            parser.parse_lidar_detections(raw_file, lidar_output_file, scene, input_type, 
                                        lidar_model_mode, lidar_model_thresh)
        
        if has_2d:
            print(f"Parsing 2D detections from multi-modal model...")
            parser.parse_image_detections(raw_file, image_output_file, scene, input_type)
        
        if not has_3d and not has_2d:
            print(f"Warning: No valid detections found in multi-modal file: {raw_file}")
    else:
        print(f"Raw multi-modal detections file not found: {raw_file}")


def parse_detections_for_run(run_index, output_base, mode, scene, input_type="publisher", 
                           lidar_model_mode="nus", lidar_model_thresh=0.2, model_name=None):
    """
    Parse detections for a specific run
    
    Args:
        run_index: Index of the run
        output_base: Base output directory
        mode: 'lidar', 'image', or 'multi-modal'
        scene: Scene token
        input_type: 'publisher' or 'bag'
        lidar_model_mode: Mode for lidar models
        lidar_model_thresh: Threshold for lidar detections
        model_name: Name of the model (for finding raw detection files)
    """
    if mode == 'multi-modal':
        parse_multimodal_detections_for_run(run_index, output_base, scene, input_type, 
                                          lidar_model_mode, lidar_model_thresh, model_name)
    elif mode == 'lidar':
        parse_lidar_detections_for_run(run_index, output_base, scene, input_type, 
                                     lidar_model_mode, lidar_model_thresh, model_name)
    elif mode == 'image':
        parse_image_detections_for_run(run_index, output_base, scene, input_type, model_name)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def parse_lidar_detections_for_run(run_index, output_base, scene, input_type="publisher", 
                                 lidar_model_mode="nus", lidar_model_thresh=0.2, model_name=None):
    """Parse lidar detections for a specific run"""
    parser = DetectionParser()
    
    # Construct raw file path based on model name
    if model_name:
        raw_file = os.path.join(output_base, f"raw_detections_{model_name}_{run_index}.json")
    else:
        raw_file = os.path.join(output_base, f"raw_detections_{run_index}.json")
    
    output_file = os.path.join(output_base, f"lidar_pred_{run_index}.json")
    
    if os.path.exists(raw_file):
        parser.parse_lidar_detections(raw_file, output_file, scene, input_type, 
                                    lidar_model_mode, lidar_model_thresh)
    else:
        print(f"Raw lidar detections file not found: {raw_file}")


def parse_image_detections_for_run(run_index, output_base, scene, input_type="publisher", model_name=None):
    """Parse image detections for a specific run"""
    parser = DetectionParser()
    
    # Construct raw file path based on model name
    if model_name:
        raw_file = os.path.join(output_base, f"raw_detections_{model_name}_{run_index}.json")
    else:
        raw_file = os.path.join(output_base, f"raw_detections_{run_index}.json")
    
    output_file = os.path.join(output_base, f"image_pred_{run_index}.json")
    
    if os.path.exists(raw_file):
        parser.parse_image_detections(raw_file, output_file, scene, input_type)
    else:
        print(f"Raw image detections file not found: {raw_file}") 


def determine_parsing_strategy(img_model, lidar_model, lidar_model_mode, lidar_thresh, scene, output_base, run_index):
    """
    Determine the appropriate parsing strategy based on model combination
    
    Returns:
        list: List of parsing functions to call with their parameters
    """
    parsing_tasks = []
    
    # Check for multi-modal models
    if "BEVFusion" in lidar_model:
        # BEVFusion outputs both 2D and 3D detections in one file
        parsing_tasks.append({
            'type': 'multimodal',
            'model_name': lidar_model,
            'params': {
                'run_index': run_index,
                'output_base': output_base,
                'scene': scene,
                'input_type': "bag",
                'lidar_model_mode': lidar_model_mode,
                'lidar_model_thresh': lidar_thresh,
                'model_name': lidar_model
            }
        })
    else:
        # Handle separate image and lidar models
        # Parse lidar detections if we have a valid lidar model
        if lidar_model and lidar_model not in ['none', 'None', '']:
            parsing_tasks.append({
                'type': 'lidar',
                'model_name': lidar_model,
                'params': {
                    'run_index': run_index,
                    'output_base': output_base,
                    'scene': scene,
                    'input_type': "bag",
                    'lidar_model_mode': lidar_model_mode,
                    'lidar_model_thresh': lidar_thresh,
                    'model_name': lidar_model
                }
            })
        
        # Parse image detections if we have a valid image model
        if img_model and img_model not in ['none', 'None', '']:
            parsing_tasks.append({
                'type': 'image',
                'model_name': img_model,
                'params': {
                    'run_index': run_index,
                    'output_base': output_base,
                    'scene': scene,
                    'input_type': "bag",
                    'model_name': img_model
                }
            })
    
    return parsing_tasks