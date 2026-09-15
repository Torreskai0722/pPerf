"""MMlab inferencer adapters and ROS message conversion helpers."""

from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class AdapterSpec:
    """A supported task/modality and its MMlab inferencer class."""

    module: str
    class_name: str


ADAPTERS: Dict[Tuple[str, str], AdapterSpec] = {
    ("detection", "image"): AdapterSpec("mmdet.apis", "DetInferencer"),
    ("detection", "lidar"): AdapterSpec(
        "mmdet3d.apis", "LidarDet3DInferencer"),
    ("semantic_segmentation", "image"): AdapterSpec(
        "mmseg.apis", "MMSegInferencer"),
    ("instance_segmentation", "image"): AdapterSpec(
        "mmdet.apis", "DetInferencer"),
    ("panoptic_segmentation", "image"): AdapterSpec(
        "mmdet.apis", "DetInferencer"),
    ("drivable_segmentation", "image"): AdapterSpec(
        "mmseg.apis", "MMSegInferencer"),
    ("lidar_segmentation", "lidar"): AdapterSpec(
        "mmdet3d.apis", "LidarSeg3DInferencer"),
}


def select_point_features(points: Any, count: Any = None) -> Any:
    """Select the leading PointCloud2 features expected by a LiDAR model."""
    if count is None:
        return points
    if getattr(points, "ndim", None) != 2 or points.shape[1] < count:
        raise ValueError(
            f"point input has shape {getattr(points, 'shape', None)}, "
            f"expected at least {count} features"
        )
    return points[:, :count]


def adapter_spec(task: str, modality: str) -> AdapterSpec:
    """Return the supported adapter spec for a task and modality."""
    try:
        return ADAPTERS[(task, modality)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported task/modality: {task}/{modality}"
        ) from exc


def override_resize_scale(inferencer: Any, scale: Any) -> Tuple[int, int]:
    """Override the single configured image resize transform."""
    selected_scale = tuple(scale)
    transforms = [
        transform for transform in inferencer.pipeline.transforms
        if hasattr(transform, "scale")
    ]
    if len(transforms) != 1:
        raise ValueError(
            "inference_resize_scale requires exactly one pipeline transform "
            f"with a scale attribute, found {len(transforms)}"
        )
    transforms[0].scale = selected_scale
    return selected_scale


def create_inferencer(task: str, modality: str, model_name: str,
                      resize_scale: Any = None, model_config: str = None,
                      checkpoint: str = None,
                      checkpoint_sha256: str = None) -> Any:
    """Construct the selected MMlab inferencer without visualization."""
    spec = adapter_spec(task, modality)
    module = importlib.import_module(spec.module)
    arguments = {"model": model_config or model_name}
    if checkpoint is not None:
        if checkpoint_sha256 is None:
            raise ValueError("an exact checkpoint requires checkpoint_sha256")
        observed = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if observed != checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 differs before trusted load")
        arguments["weights"] = checkpoint
    force_safe = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "").lower()
    if checkpoint is not None and force_safe in ("1", "y", "yes", "true"):
        raise ValueError(
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD conflicts with pinned load"
        )
    prior = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    if checkpoint is not None:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        inferencer = getattr(module, spec.class_name)(**arguments)
    finally:
        if checkpoint is not None:
            if prior is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = prior
    selected_scale = None
    if resize_scale is not None:
        selected_scale = override_resize_scale(inferencer, resize_scale)
    return InferencerAdapter(inferencer, modality, selected_scale)


class InferencerAdapter:
    """Expose only MMEngine preprocessing and the attached model."""

    def __init__(self, inferencer: Any, modality: str = "image",
                 resize_scale: Any = None):
        """Retain the inferencer and its effective preprocessing settings."""
        self.inferencer = inferencer
        self.model = inferencer.model
        self.modality = modality
        self.resize_scale = resize_scale

    def preprocess(self, value: Any) -> Any:
        """Produce one model batch without forward/visualize/postprocess."""
        if self.modality == "lidar" and not isinstance(value, dict):
            value = {"points": value}
        batches = self.inferencer.preprocess([value], batch_size=1)
        batch = next(iter(batches))
        # MMDetection collates ``(original_input, model_data)`` tuples into a
        # two-element list. Other inferencers return model data directly.
        if (isinstance(batch, (tuple, list)) and len(batch) == 2 and
                isinstance(batch[1], dict)):
            return batch[1]
        return batch


def decode_image(message: Any) -> Any:
    """Decode a ROS Image as a zero-copy shaped NumPy view when possible."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    encodings = {
        "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
        "mono8": (np.uint8, 1), "mono16": (np.uint16, 1),
    }
    try:
        dtype, channels = encodings[message.encoding.lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported image encoding: {message.encoding}"
        ) from exc
    array = np.frombuffer(message.data, dtype=dtype)
    expected = message.height * message.width * channels
    if array.size < expected:
        raise ValueError("image data is shorter than declared dimensions")
    shape = ((message.height, message.width) if channels == 1 else
             (message.height, message.width, channels))
    image = array[:expected].reshape(shape)
    if message.encoding.lower() in ("bgr8", "bgra8"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return image


def decode_compressed_image(message: Any) -> Any:
    """Decode a ROS CompressedImage into an MMlab-compatible BGR array."""
    import cv2  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("compressed image could not be decoded")
    return image


def decode_pointcloud2(message: Any) -> Any:
    """Decode PointCloud2 model fields into a dense floating-point array."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    type_map = {1: "i1", 2: "u1", 3: "i2", 4: "u2",
                5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    names, formats, offsets = [], [], []
    for field in message.fields:
        if field.name in ("x", "y", "z", "intensity", "ring"):
            names.append(field.name)
            formats.append(type_map[field.datatype])
            offsets.append(field.offset)
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("point cloud must contain x, y, and z fields")
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype({"names": names,
                      "formats": [endian + item for item in formats],
                      "offsets": offsets, "itemsize": message.point_step})
    records = np.frombuffer(message.data, dtype=dtype,
                            count=message.width * message.height)
    columns = [records[name].astype(np.float32, copy=False) for name in names]
    return np.column_stack(columns)
