# Copyright (c) OpenMMLab. All rights reserved.
"""In-memory MMlab pipeline transformations used by the testbed."""

from typing import Any, Dict, Optional, Sequence


POINT_CLOUD_RANGE = (-50, -50, -5, 50, 50, 3)


class InMemoryImageLoader:
    """Turn an image array into the dictionary expected by MMlab pipelines."""

    def __call__(self, results: Any) -> Dict[str, Any]:
        image = results["img"] if isinstance(results, dict) else results
        shape = image.shape[:2]
        return {"img": image, "img_shape": shape, "ori_shape": shape,
                "img_path": None}


class InMemoryPointLoader:
    """Accept an NxD array without writing a temporary point-cloud file."""

    def __init__(self, coord_type: str = "LIDAR", load_dim: int = 5,
                 use_dim: Optional[Sequence[int]] = None,
                 point_cloud_range: Sequence[float] = POINT_CLOUD_RANGE):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = tuple(use_dim) if use_dim is not None else None
        self.point_cloud_range = tuple(point_cloud_range)

    def __call__(self, results: Any) -> Dict[str, Any]:
        import numpy as np  # pylint: disable=import-outside-toplevel
        values = results["points"] if isinstance(results, dict) else results
        points = np.asarray(values, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must be an NxD array with x/y/z")
        if self.use_dim is not None:
            points = points[:, self.use_dim]
        lower = np.asarray(self.point_cloud_range[:3])
        upper = np.asarray(self.point_cloud_range[3:])
        xyz = points[:, :3]
        points = points[((xyz >= lower) & (xyz <= upper)).all(axis=1)]
        output = dict(results) if isinstance(results, dict) else {}
        output["points"] = points
        return output
