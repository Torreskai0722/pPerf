"""Deterministic preprocessing and decoding for repeated source inputs."""

import hashlib
from pathlib import Path
import random

import numpy as np

from .adapters import decode_compressed_image, decode_pointcloud2, select_point_features


def decode_source(record, model):
    """Decode retained original CDR using the same path as the ROS callback."""
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, PointCloud2
    raw = Path(record["cdr_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["original_serialized_sha256"]:
        raise ValueError("fixed source CDR changed")
    if model["modality"] == "image":
        return decode_compressed_image(deserialize_message(raw, CompressedImage))
    return select_point_features(
        decode_pointcloud2(deserialize_message(raw, PointCloud2)),
        model.get("point_feature_count"))


def reset_preprocessing(seed):
    """Reset CPU transform RNGs; configured pipelines use Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def tensor_hash(value):
    """Hash model input tensors outside timed measurements, including layout."""
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, (torch.Tensor, np.ndarray)):
            array = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        else:
            raise TypeError(f"unsupported input tensor container: {type(item)}")

    visit(value)
    return digest.hexdigest()


def verify_preprocessing(adapter, value, seed):
    """Compare independently recomputed transforms and model input tensors."""
    import torch
    hashes = []
    for _ in range(2):
        reset_preprocessing(seed)
        with torch.no_grad():
            prepared = adapter.preprocess(value)
            inputs = adapter.model.data_preprocessor(prepared, training=False)["inputs"]
        hashes.append(tensor_hash(inputs))
    if hashes[0] != hashes[1]:
        raise ValueError("repeated preprocessing produces different input tensors")
    return {"seed": seed, "input_tensor_sha256": hashes[0], "checks": 2,
            "equal": True, "timed": False}
