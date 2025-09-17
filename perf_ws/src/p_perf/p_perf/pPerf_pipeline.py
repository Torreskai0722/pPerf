# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Union, Tuple

import numpy as np
import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import to_tensor

from mmdet.structures import DetDataSample

from mmengine.registry import TRANSFORMS

from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.structures.points import LiDARPoints

from mmengine.dataset import Compose
from mmdet3d.utils import ConfigType as ConfigType3D
from mmdet.utils import ConfigType as ConfigType2D

import torch


@TRANSFORMS.register_module()
class pPerfLiDARLoader(BaseTransform):
    """Combined loader for pPerf 3D detection with multi-sweep support.
    
    This class combines the functionality of LidarDet3DInferencerLoader and
    LoadPointsFromMultiSweeps, but removes the from_file functionality and
    only handles the case where lidar_sweeps is not in results with pad_empty_sweeps=False.
    
    Added keys:
      - points
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, 
                 coord_type='LIDAR',
                 load_dim: int = 5,
                 use_dim: Optional[List[int]] = None,
                 point_cloud_range: Optional[List[float]] = None,
                 **kwargs) -> None:
        super().__init__()
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)
        self.coord_type = coord_type
        
        # Dimension parameters
        self.load_dim = load_dim
        self.use_dim = use_dim if use_dim is not None else [0, 1, 2, 3]
        
        # Range filter parameters
        if point_cloud_range is not None:
            self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        else:
            self.pcd_range = None

    def transform(self, single_input: dict) -> dict:
        """Transform function to load points and handle multi-sweep processing.
        
        Args:
            single_input (dict): Single input containing points data.

        Returns:
            dict: The dict contains loaded points and meta information.
        """
        assert 'points' in single_input, "key 'points' must be in input dict"
        points = single_input['points']
        result = {}
        
        # Handle points loading (only from ndarray, no file loading)
        if not isinstance(single_input['points'], np.ndarray):
            raise ValueError('Unsupported input points type: '
                             f"{type(single_input['points'])}")

        # Load base points
        points = LiDARPoints(
            points, points_dim=points.shape[-1])

        points = points[:, self.use_dim]
        
        # Apply range filtering
        points_mask = points.in_range_3d(self.pcd_range)
        points = points[points_mask]
        
        # Handle semantic and instance masks if present
        pts_instance_mask = single_input.get('pts_instance_mask', None)
        pts_semantic_mask = single_input.get('pts_semantic_mask', None)
        
        if pts_instance_mask is not None:
            result['pts_instance_mask'] = pts_instance_mask[points_mask.numpy()]
        
        if pts_semantic_mask is not None:
            result['pts_semantic_mask'] = pts_semantic_mask[points_mask.numpy()]


        result['points'] = points
        result['box_type_3d'] = self.box_type_3d
        result['box_mode_3d'] = self.box_mode_3d
        
        return result

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = f'{self.__class__.__name__}(use_dim={self.use_dim}'
        if self.pcd_range is not None:
            repr_str += f', point_cloud_range={self.pcd_range.tolist()}'
        repr_str += ')'
        return repr_str
    


@TRANSFORMS.register_module()
class pPerfImageLoader(BaseTransform):
    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 keep_ratio: bool = False,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        super().__init__()
        self.scale = (scale, scale) if isinstance(scale, int) else scale
        self.keep_ratio = keep_ratio
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        # -------------------------
        # Step 1: Load and Resize
        # -------------------------
        assert 'img' in results, "key 'img' must be in input dict"
        img = results['img']
        ori_shape = img.shape[:2]

        if self.scale is not None:
            if self.keep_ratio:
                # For keep_ratio=True, use imrescale with the scale
                img = mmcv.imrescale(img, self.scale)
                new_h, new_w = img.shape[:2]
                h, w = ori_shape
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                # For keep_ratio=False, use imresize with the target size
                # Convert (width, height) to (height, width) for imresize
                if isinstance(self.scale, (tuple, list)) and len(self.scale) == 2:
                    target_size = (self.scale[1], self.scale[0])  # (height, width)
                else:
                    target_size = self.scale
                img, w_scale, h_scale = mmcv.imresize(img, target_size)
        else:
            # No scaling, just use original dimensions
            w_scale = 1.0
            h_scale = 1.0

        # -------------------------
        # Step 2: Convert to Tensor
        # -------------------------
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)

        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        # -------------------------
        # Step 3: Pack Metainfo
        # -------------------------
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = ori_shape
        results['scale_factor'] = (w_scale, h_scale)

        # -------------------------
        # Step 4: Create DetDataSample
        # -------------------------
        data_sample = DetDataSample()
        img_meta = {}
        if self.meta_keys is not None:
            for key in self.meta_keys:
                if key in results:
                    img_meta[key] = results[key]
        else:
            img_meta = results
        data_sample.set_metainfo(img_meta)

        # -------------------------
        # Step 5: Return like PackDetInputs
        # -------------------------
        result = {
            'inputs': img_tensor,
            'data_samples': data_sample
        }

        return result
