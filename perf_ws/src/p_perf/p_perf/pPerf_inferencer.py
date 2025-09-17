# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import Compose
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import torch
import cv2
import numpy as np # Added for bddLaneSegInferencer

from mmdet3d.apis import LidarDet3DInferencer, LidarSeg3DInferencer
from mmdet.apis import DetInferencer, init_detector
from mmseg.apis import MMSegInferencer, init_model
from mmdet3d.utils import ConfigType as ConfigType3D
from mmdet.utils import ConfigType as ConfigType2D

from p_perf.pPerf_pipeline import pPerfLiDARLoader, pPerfImageLoader

from typing import Union, Tuple, List, Dict

from mmseg.registry import MODELS

# Predefined point cloud range for pPerf3dDetLoader so that all models can be tested on the same point cloud range
POINT_CLOUD_RANGE = [-50, -50, -5, 50, 50, 3]


class pPerf3dSegInferencer(LidarSeg3DInferencer):

    def _init_pipeline(self, cfg: ConfigType3D) -> Compose:
        """Initialize the test pipeline with pPerf3dDetLoader.
        
        Args:
            cfg: Configuration object containing pipeline settings.
            
        Returns:
            Compose: A composed pipeline with pPerf3dDetLoader and Pack3DDetInputs.
        """
        # Check for pipeline in both possible locations
        if hasattr(cfg, 'test_pipeline'):
            pipeline_cfg = cfg.test_pipeline
        else:
            pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        
        # Find the Pack3DDetInputs transform
        pack_idx = self._get_transform_idx(pipeline_cfg, 'Pack3DDetInputs')
        if pack_idx == -1:
            raise ValueError('Pack3DDetInputs is not found in the test pipeline')
        
        # Get the Pack3DDetInputs configuration
        pack_cfg = pipeline_cfg[pack_idx]
        
        # Extract load_dim from LoadPointsFromFile
        load_points_idx = self._get_transform_idx(pipeline_cfg, 'LoadPointsFromFile')
        if load_points_idx == -1:
            raise ValueError('LoadPointsFromFile is not found in the test pipeline')
        load_points_cfg = pipeline_cfg[load_points_idx]
        load_dim = load_points_cfg.get('load_dim', 5)

        load_multi_sweeps_idx = self._get_transform_idx(pipeline_cfg, 'LoadPointsFromMultiSweeps')
        if load_multi_sweeps_idx != -1:
            load_multi_sweeps_cfg = pipeline_cfg[load_multi_sweeps_idx]
            use_dim = load_multi_sweeps_cfg.get('use_dim', None)
        else:
            use_dim = None
        
        # Extract point_cloud_range from PointsRangeFilter
        points_range_idx = self._get_transform_idx(pipeline_cfg, 'PointsRangeFilter')
        point_cloud_range = None
        if points_range_idx != -1:
            points_range_cfg = pipeline_cfg[points_range_idx]
            point_cloud_range = points_range_cfg.get('point_cloud_range', None)
        
        # Create new pipeline with only pPerf3dDetLoader and Pack3DDetInputs
        new_pipeline = [
            dict(type='pPerfLiDARLoader', 
                 coord_type='LIDAR',
                 load_dim=load_dim,
                 use_dim=use_dim,
                 point_cloud_range=POINT_CLOUD_RANGE),
            pack_cfg  # Keep the original Pack3DDetInputs configuration
        ]
        
        return Compose(new_pipeline)


class pPerf3dDetInferencer(LidarDet3DInferencer):

    def _init_pipeline(self, cfg: ConfigType3D) -> Compose:
        """Initialize the test pipeline with pPerf3dDetLoader.
        
        Args:
            cfg: Configuration object containing pipeline settings.
            
        Returns:
            Compose: A composed pipeline with pPerf3dDetLoader and Pack3DDetInputs.
        """
        # Check for pipeline in both possible locations
        if hasattr(cfg, 'test_pipeline'):
            pipeline_cfg = cfg.test_pipeline
        else:
            pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        
        # Find the Pack3DDetInputs transform
        pack_idx = self._get_transform_idx(pipeline_cfg, 'Pack3DDetInputs')
        if pack_idx == -1:
            raise ValueError('Pack3DDetInputs is not found in the test pipeline')
        
        # Get the Pack3DDetInputs configuration
        pack_cfg = pipeline_cfg[pack_idx]
        
        # Extract load_dim from LoadPointsFromFile
        load_points_idx = self._get_transform_idx(pipeline_cfg, 'LoadPointsFromFile')
        if load_points_idx == -1:
            raise ValueError('LoadPointsFromFile is not found in the test pipeline')
        load_points_cfg = pipeline_cfg[load_points_idx]
        load_dim = load_points_cfg.get('load_dim', 5)

        load_multi_sweeps_idx = self._get_transform_idx(pipeline_cfg, 'LoadPointsFromMultiSweeps')
        if load_multi_sweeps_idx != -1:
            load_multi_sweeps_cfg = pipeline_cfg[load_multi_sweeps_idx]
            use_dim = load_multi_sweeps_cfg.get('use_dim', None)
        else:
            use_dim = None
        
        # Extract point_cloud_range from PointsRangeFilter
        points_range_idx = self._get_transform_idx(pipeline_cfg, 'PointsRangeFilter')
        point_cloud_range = None
        if points_range_idx != -1:
            points_range_cfg = pipeline_cfg[points_range_idx]
            point_cloud_range = points_range_cfg.get('point_cloud_range', None)
        
        # Create new pipeline with only pPerf3dDetLoader and Pack3DDetInputs
        new_pipeline = [
            dict(type='pPerfLiDARLoader', 
                 coord_type='LIDAR',
                 load_dim=load_dim,
                 use_dim=use_dim,
                 point_cloud_range=POINT_CLOUD_RANGE),
            pack_cfg  # Keep the original Pack3DDetInputs configuration
        ]
        
        return Compose(new_pipeline)
    

class pPerf2dDetInferencer(DetInferencer):

    
    def _init_pipeline(self, cfg: ConfigType2D) -> Compose:
        """Initialize the test pipeline with pPerf2dDetLoader.
        
        Args:
            cfg: Configuration object containing pipeline settings.
            
        Returns:
            Compose: A composed pipeline with pPerf2dDetLoader.
        """
        # Check for pipeline in both possible locations
        if hasattr(cfg, 'test_pipeline'):
            pipeline_cfg = cfg.test_pipeline
        else:
            pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        
        # Extract scale and keep_ratio from Resize transform
        resize_idx = self._get_transform_idx(pipeline_cfg, 'Resize')
        scale = None
        keep_ratio = False
        if resize_idx != -1:
            resize_cfg = pipeline_cfg[resize_idx]
            scale = resize_cfg.get('scale', None)
            keep_ratio = resize_cfg.get('keep_ratio', False)
        
        # Extract meta_keys from PackDetInputs transform
        pack_idx = self._get_transform_idx(pipeline_cfg, 'PackDetInputs')
        meta_keys = None
        if pack_idx != -1:
            pack_cfg = pipeline_cfg[pack_idx]
            meta_keys = pack_cfg.get('meta_keys', meta_keys)

        # Create new pipeline with only pPerf2dDetLoader
        new_pipeline = [
            dict(type='pPerfImageLoader', 
                 scale=scale,
                 keep_ratio=keep_ratio,
                 meta_keys=meta_keys),
        ]
        return Compose(new_pipeline)


class bddDetInferencer(DetInferencer):

    def __init__(self, model_name: str):

        self.model_name = model_name
        self.model_cfg_base = '/mmdetection3d_ros2/data/models/det/configs/det/'
        self.model_url = "https://dl.cv.ethz.ch/bdd100k/det/models/"
        self.model_cfg = self.model_cfg_base + self.model_name + '.py'
        self.model_ckpt = self.model_url + self.model_name + '.pth'

        self.cfg = Config.fromfile(self.model_cfg)
        self._add_preprocessor(self.cfg)

        self.pipeline = self._init_pipeline(self.cfg)

        self.model = init_detector(self.cfg, self.model_ckpt, device='cuda:0', palette='random')

    def _init_pipeline(self, cfg: ConfigType2D) -> Compose:
        if hasattr(cfg, 'test_pipeline'):
            pipeline_cfg = cfg.test_pipeline
        else:
            pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        
        # Find MultiScaleFlipAug and extract img_scale
        scale = None
        keep_ratio = False
        
        for transform in pipeline_cfg:
            if transform['type'] == 'MultiScaleFlipAug':
                # Extract scale from MultiScaleFlipAug
                scale = transform.get('img_scale', None)
                
                # Look for Resize in the nested transforms
                if 'transforms' in transform:
                    for nested_transform in transform['transforms']:
                        if nested_transform['type'] == 'Resize':
                            keep_ratio = nested_transform.get('keep_ratio', False)
                break


        # Create new pipeline with only pPerf2dDetLoader
        new_pipeline = [
            dict(type='pPerfImageLoader', 
                    scale=scale,
                    keep_ratio=keep_ratio,
                    meta_keys=None),
        ]
        
        return Compose(new_pipeline)
    

    def _add_preprocessor(self, cfg: ConfigType2D):
        """Add data preprocessor to the model configuration.
        
        Args:
            cfg: Configuration object containing model settings.
        """
        # Check if img_norm_cfg exists in the configuration
        if hasattr(cfg, 'img_norm_cfg'):
            img_norm_cfg = cfg.img_norm_cfg
            mean = img_norm_cfg.get('mean', [123.675, 116.28, 103.53])
            std = img_norm_cfg.get('std', [58.395, 57.12, 57.375])
            bgr_to_rgb = img_norm_cfg.get('to_rgb', True)
        else:
            # Default values if img_norm_cfg is not found
            mean = [123.675, 116.28, 103.53]
            std = [58.395, 57.12, 57.375]
            bgr_to_rgb = True
        
        # Extract pad_size_divisor from the pipeline configuration
        pad_size_divisor = 1  # Default value
        
        # Check for pipeline in both possible locations (same logic as _init_pipeline)
        if hasattr(cfg, 'test_pipeline'):
            pipeline_cfg = cfg.test_pipeline
        else:
            pipeline_cfg = None
        
        if pipeline_cfg:
            for transform in pipeline_cfg:
                if transform['type'] == 'MultiScaleFlipAug' and 'transforms' in transform:
                    for nested_transform in transform['transforms']:
                        if nested_transform['type'] == 'Pad':
                            pad_size_divisor = nested_transform.get('size_divisor', 1)
                            break
                elif transform['type'] == 'Pad':
                    pad_size_divisor = transform.get('size_divisor', 1)
                    break
        
        # Create data_preprocessor configuration
        data_preprocessor = dict(
            type='DetDataPreprocessor',
            mean=mean,
            std=std,
            bgr_to_rgb=bgr_to_rgb,
            pad_size_divisor=pad_size_divisor,
        )
        
        # Add data_preprocessor to the model configuration
        if hasattr(cfg, 'model'):
            cfg.model.data_preprocessor = data_preprocessor
        else:
            # If model section doesn't exist, create it
            print(f"DEBUG: model section does not exist in the configuration")
        
        return cfg
    
    def __call__(self, img, return_datasamples=False) -> List[Dict]:
        results_dict = {'predictions': [], 'visualization': []}

        input = self.pipeline(img)
        input['inputs'] = [input['inputs']]
        input['data_samples'] = [input['data_samples']]
        with torch.no_grad():
            pred = self.model.test_step(input)
        
        pred = self.postprocess(preds=pred, return_datasamples=return_datasamples)
        results_dict['predictions'].extend(pred['predictions'])
        return results_dict
    
    def show_result(self, img, results_dict):
        """Visualize detection results by drawing bounding boxes and labels.
        
        Args:
            img: Input image (numpy array)
            results_dict: Detection results dictionary from the model
        """
        # Create a copy of the image for visualization
        vis_img = img.copy()
        
        # Extract predictions from results_dict
        predictions = results_dict.get('predictions', [])
        if not predictions:
            print("No predictions found in results_dict")
            return vis_img
        
        # Get the first prediction (assuming single image input)
        prediction = predictions[0]
        
        # Extract detection information
        if hasattr(prediction, 'pred_instances'):
            instances = prediction.pred_instances
            
            # Get bounding boxes, labels, and scores
            bboxes = instances.bboxes.cpu().numpy()
            labels = instances.labels.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            
            # Define colors for different classes (you can customize this)
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 128),  # Purple
                (128, 128, 0),  # Olive
            ]
            
            # Draw bounding boxes and labels (only for scores > 0.75)
            for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                # Only show detections with confidence score > 0.75
                if score <= 0.75:
                    continue
                    
                # Get color for this class
                color = colors[label % len(colors)]
                
                # Convert bbox coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # Create label text
                label_text = f"Class {label}: {score:.2f}"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    vis_img, 
                    (x1, y1 - text_height - baseline - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_img, 
                    label_text, 
                    (x1, y1 - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
        
        # Save the image
        output_path = f'detection_result_{self.model_name}.jpg'
        cv2.imwrite(output_path, vis_img)
        print(f"Visualization saved to: {output_path}")
        
        return vis_img


class bddSegInferencer(MMSegInferencer):
    '''
    BDD100K Lane Segmentation Inferencer
    Args:
        model_name: Name of the model to use
        mode: Mode of the model to use (e.g. 'ins_seg', 'pan_seg', 'sem_seg', 'drivable')
    '''

    def __init__(self, model_name: str, mode: str):
        self.model_name = model_name
        self.model_cfg_base = f'/mmdetection3d_ros2/data/models/{mode}/configs/{mode}/'
        self.model_url = f"https://dl.cv.ethz.ch/bdd100k/{mode}/models/"
        self.model_cfg = self.model_cfg_base + self.model_name + '.py'
        self.model_ckpt = self.model_url + self.model_name + '.pth'

        if mode == 'sem_seg' or mode == 'drivable':
            scope = 'mmseg'
            self.preprocess_type = 'SegDataPreProcessor'
        else:
            scope = 'mmdet'
            self.preprocess_type = 'DetDataPreprocessor'

        self.cfg = Config.fromfile(self.model_cfg)
        self._add_preprocessor(self.cfg)

        # Remove init_cfg from backbone to avoid loading weights from checkpoint
        if self.cfg.model.type == 'EncoderDecoder':
            if 'init_cfg' in self.cfg.model.backbone:
                self.cfg.model.backbone.init_cfg = None
        elif self.cfg.model.type == 'MultimodalEncoderDecoder':
            for k, v in self.cfg.model.items():
                if isinstance(v, dict) and 'init_cfg' in v:
                    self.cfg.model[k].init_cfg = None
        
        init_default_scope(self.cfg.get('default_scope', scope))

        self.pipeline = self._init_pipeline(self.cfg)

        self.model = MODELS.build(self.cfg.model)
        load_checkpoint(self.model, self.model_ckpt, map_location='cpu')
        self.model.cfg = self.cfg  # save the config in the model for convenience
        self.model.to(device='cuda:0')
        self.model.eval()
        
        self.num_pred_imgs = 0

    def _init_pipeline(self, cfg: ConfigType2D) -> Compose:
        if hasattr(cfg, 'test_pipeline'):
            pipeline_cfg = cfg.test_pipeline
        else:
            pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        
        # Find MultiScaleFlipAug and extract img_scale
        scale = None
        keep_ratio = False
        
        for transform in pipeline_cfg:
            if transform['type'] == 'MultiScaleFlipAug':
                # Extract scale from MultiScaleFlipAug
                scale = transform.get('img_scale', None)
                
                # Look for Resize in the nested transforms
                if 'transforms' in transform:
                    for nested_transform in transform['transforms']:
                        if nested_transform['type'] == 'Resize':
                            keep_ratio = nested_transform.get('keep_ratio', False)
                break

        # Create new pipeline with only pPerfImageLoader
        new_pipeline = [
            dict(type='pPerfImageLoader', 
                    scale=scale,
                    keep_ratio=keep_ratio,
                    meta_keys=None),
        ]
        
        return Compose(new_pipeline)

    def _add_preprocessor(self, cfg: ConfigType2D):
        """Add data preprocessor to the model configuration.
        
        Args:
            cfg: Configuration object containing model settings.
        """
        # Check if img_norm_cfg exists in the configuration
        if hasattr(cfg, 'img_norm_cfg'):
            img_norm_cfg = cfg.img_norm_cfg
            mean = img_norm_cfg.get('mean', [123.675, 116.28, 103.53])
            std = img_norm_cfg.get('std', [58.395, 57.12, 57.375])
            bgr_to_rgb = img_norm_cfg.get('to_rgb', True)
        else:
            # Default values if img_norm_cfg is not found
            mean = [123.675, 116.28, 103.53]
            std = [58.395, 57.12, 57.375]
            bgr_to_rgb = True
        
        # Extract size, size_divisor, and pad_val from the pipeline configuration
        size = None
        size_divisor = None
        pad_val = 0
        seg_pad_val = 255
        
        # Check for pipeline in both possible locations
        if hasattr(cfg, 'test_pipeline'):
            pipeline_cfg = cfg.test_pipeline
        else:
            pipeline_cfg = None
        
        if pipeline_cfg:
            for transform in pipeline_cfg:
                if transform['type'] == 'MultiScaleFlipAug' and 'transforms' in transform:
                    for nested_transform in transform['transforms']:
                        if nested_transform['type'] == 'Pad':
                            size_divisor = nested_transform.get('size_divisor', None)
                            pad_val = nested_transform.get('pad_val', None)
                            size = nested_transform.get('size', None)
                            seg_pad_val = nested_transform.get('seg_pad_val', 255)
                            break
                elif transform['type'] == 'Pad':
                    size_divisor = transform.get('size_divisor', None)
                    size = transform.get('size', None)
                    pad_val = transform.get('pad_val', 0)
                    seg_pad_val = transform.get('seg_pad_val', 255)
                    break
        
        if self.preprocess_type == 'SegDataPreProcessor':
            # Create data_preprocessor configuration
            data_preprocessor = dict(
                type=self.preprocess_type,
                mean=mean,
                std=std,
                bgr_to_rgb=bgr_to_rgb,
                size=size,
                size_divisor=size_divisor,
                pad_val=pad_val,
                seg_pad_val=seg_pad_val,
            )
        else:
            # Create data_preprocessor configuration
            data_preprocessor = dict(
                type=self.preprocess_type,
                mean=mean,
                std=std,
                bgr_to_rgb=bgr_to_rgb,
                pad_size_divisor=size_divisor,
            )
        
        # Add data_preprocessor to the model configuration
        if hasattr(cfg, 'model'):
            cfg.model.data_preprocessor = data_preprocessor
        else:
            # If model section doesn't exist, create it
            print(f"DEBUG: model section does not exist in the configuration")
        
        return cfg
    
    def __call__(self, img, return_datasamples=False) -> List[Dict]:
        results_dict = {'predictions': [], 'visualization': []}

        input_data = self.pipeline(img)
        input_data['inputs'] = [input_data['inputs']]
        input_data['data_samples'] = [input_data['data_samples']]
        
        with torch.no_grad():
            pred = self.model.test_step(input_data)[0]
        
        return pred
    
    def show_result(self, img, results_dict):
        """Visualize segmentation results by overlaying the segmentation mask.
        
        Args:
            img: Input image (numpy array)
            results_dict: Segmentation results dictionary from the model
        """
        # Create a copy of the image for visualization
        vis_img = img.copy()
        
        # Extract predictions from results_dict
        predictions = results_dict.get('predictions', [])
        if not predictions:
            print("No predictions found in results_dict")
            return vis_img
        
        # Get the first prediction (assuming single image input)
        prediction = predictions[0]
        
        # Extract segmentation information
        if hasattr(prediction, 'pred_sem_seg'):
            seg_map = prediction.pred_sem_seg.data.cpu().numpy()
            
            # Get the segmentation mask (assuming single channel)
            if len(seg_map.shape) == 3:
                seg_mask = seg_map[0]  # Take first channel if multiple
            else:
                seg_mask = seg_map
            
            # Create a colored mask overlay
            colored_mask = np.zeros_like(vis_img)
            
            # Define colors for different classes (you can customize this)
            colors = [
                (0, 0, 0),      # Black (background)
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 128),  # Purple
            ]
            
            # Apply colors to the mask
            for class_id in range(1, min(len(colors), int(seg_mask.max()) + 1)):
                mask = (seg_mask == class_id)
                colored_mask[mask] = colors[class_id]
            
            # Blend the original image with the colored mask
            alpha = 0.6
            vis_img = cv2.addWeighted(vis_img, 1 - alpha, colored_mask, alpha, 0)
        
        # Save the image
        output_path = f'lane_segmentation_result_{self.model_name}.jpg'
        cv2.imwrite(output_path, vis_img)
        print(f"Visualization saved to: {output_path}")
        
        return vis_img