# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import Compose

import torch
import cv2
import numpy as np  # For 2D segmentation visualization (show_result)

from mmdet3d.apis import LidarDet3DInferencer, LidarSeg3DInferencer
from mmdet.apis import DetInferencer, init_detector
from mmseg.apis import MMSegInferencer, init_model
from mmdet3d.utils import ConfigType as ConfigType3D
from mmdet.utils import ConfigType as ConfigType2D

from p_perf.pPerf_pipeline import pPerfLiDARLoader, pPerfImageLoader
from p_perf.general_utils import get_segmentation_palette

from typing import List, Dict

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

class pPerf2dSegInferencer(MMSegInferencer):
    """2D segmentation inferencer with custom pipeline (pPerfImageLoader). Uses MMSeg model zoo; no _add_preprocessor (config has data preprocessor)."""

    def __init__(self, model: str, mode: str = 'sem_seg', device: str = 'cuda:0', **kwargs):
        """Args: model: MMSeg model-zoo alias. mode: e.g. 'sem_seg', 'ins_seg', 'pan_seg', 'drivable' (stored for compatibility with node/profiler)."""
        super().__init__(model=model, device=device, **kwargs)
        self.model_name = model
        self.mode = mode

    def _init_pipeline(self, cfg: ConfigType2D) -> Compose:
        """Initialize the test pipeline with pPerfImageLoader."""
        if hasattr(cfg, 'test_pipeline'):
            pipeline_cfg = cfg.test_pipeline
        else:
            pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        scale = None
        keep_ratio = False
        for transform in pipeline_cfg:
            if transform.get('type') == 'MultiScaleFlipAug':
                scale = transform.get('img_scale', None)
                if 'transforms' in transform:
                    for nested_transform in transform['transforms']:
                        if nested_transform.get('type') == 'Resize':
                            keep_ratio = nested_transform.get('keep_ratio', False)
                            break
                break

        new_pipeline = [
            dict(type='pPerfImageLoader', scale=scale, keep_ratio=keep_ratio, meta_keys=None),
        ]
        return Compose(new_pipeline)

    def __call__(self, img, return_datasamples=False) -> List[Dict]:
        input_data = self.pipeline(img)
        input_data['inputs'] = [input_data['inputs']]
        input_data['data_samples'] = [input_data['data_samples']]
        with torch.no_grad():
            pred = self.model.test_step(input_data)[0]
        return pred

    def show_result(self, img, results_dict):
        """Visualize segmentation results by overlaying the segmentation mask."""
        vis_img = np.asarray(img).copy() if img is not None else np.zeros((512, 512, 3), dtype=np.uint8)
        # Accept DataSample (has .get or .pred_sem_seg) or dict
        if hasattr(results_dict, 'pred_sem_seg'):
            prediction = results_dict.pred_sem_seg
        else:
            prediction = results_dict.get('pred_sem_seg', []) if hasattr(results_dict, 'get') else []
        if not prediction:
            print("No prediction found in results_dict")
            cv2.imwrite(f'lane_segmentation_result_{self.model_name}.jpg', vis_img)
            return vis_img
        # prediction may be PixelData (.data) or a wrapper with .pred_sem_seg
        if hasattr(prediction, 'data'):
            seg_map = prediction.data.cpu().numpy() if hasattr(prediction.data, 'cpu') else np.asarray(prediction.data)
        elif hasattr(prediction, 'pred_sem_seg'):
            seg_map = prediction.pred_sem_seg.data.cpu().numpy()
        else:
            print("Prediction has no .data or .pred_sem_seg")
            cv2.imwrite(f'lane_segmentation_result_{self.model_name}.jpg', vis_img)
            return vis_img
        if len(seg_map.shape) == 3:
            seg_mask = seg_map[0]
        else:
            seg_mask = seg_map
        # Ensure vis_img is 3-channel and same spatial size as seg_mask
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        if vis_img.shape[:2] != seg_mask.shape[:2]:
            vis_img = cv2.resize(vis_img, (seg_mask.shape[1], seg_mask.shape[0]))
        colored_mask = np.zeros_like(vis_img)
        colors = get_segmentation_palette('bdd')  # 19 classes; extend if needed for COCO-Stuff etc.
        max_cls = int(seg_mask.max())
        for class_id in np.unique(seg_mask):
            class_id = int(class_id)
            if class_id < len(colors):
                colored_mask[seg_mask == class_id] = colors[class_id]
            else:
                # Fallback for datasets with many classes (e.g. COCO-Stuff)
                colored_mask[seg_mask == class_id] = (class_id % 256, (class_id * 7) % 256, (class_id * 13) % 256)
        alpha = 0.6
        vis_img = cv2.addWeighted(vis_img, 1 - alpha, colored_mask, alpha, 0)
        output_path = f'lane_segmentation_result_{self.model_name}.jpg'
        cv2.imwrite(output_path, vis_img)
        print(f"Visualization saved to: {output_path}")
        return vis_img


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

