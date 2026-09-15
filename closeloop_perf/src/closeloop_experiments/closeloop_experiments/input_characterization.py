"""Prepare offline input-characteristic evidence from campaign bags."""

import json
from pathlib import Path

import numpy as np

from .corrupt_rosbag import _header_timestamp_ns, _point_dtype


def characterize_inputs(study, destination):
    """Measure camera/LiDAR payloads and write analyzer-ready JSON."""
    import cv2
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, PointCloud2

    rows = []
    for condition_id, condition in study["data"]["conditions"].items():
        manifest = json.loads(Path(condition["dataset_manifest"]).read_text())
        expected = {
            (record["topic"], int(record["output_header_timestamp_ns"]))
            for record in manifest["sensor_records"]
        }
        observed = set()
        bags = sorted(Path(condition["bag_directory"]).rglob("*.mcap"))
        if len(bags) != 1:
            raise ValueError(
                f"{condition_id} requires exactly one MCAP, found {len(bags)}"
            )
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bags[0]), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        while reader.has_next():
            topic, raw, _timestamp = reader.read_next()
            common = {
                "condition_id": condition_id,
                "effect": condition["effect"],
                "corruption_type": condition["corruption_type"],
                "severity": condition["severity"],
                "rain_rate_mm_per_hour": condition.get(
                    "rain_rate_mm_per_hour"),
            }
            if topic == "/CAM_FRONT/image_rect_compressed":
                message = deserialize_message(raw, CompressedImage)
                timestamp = _header_timestamp_ns(message)
                encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
                gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError(
                        f"cannot decode {condition_id} image {timestamp}")
                values = gray.astype(np.float32)
                rows.append({
                    **common,
                    "modality": "image",
                    "input_timestamp_ns": timestamp,
                    "image_width": int(gray.shape[1]),
                    "image_height": int(gray.shape[0]),
                    "jpeg_bytes": int(encoded.size),
                    "luminance_mean": float(values.mean()),
                    "luminance_std": float(values.std()),
                    "horizontal_gradient_mean": float(
                        np.abs(np.diff(values, axis=1)).mean()),
                    "vertical_gradient_mean": float(
                        np.abs(np.diff(values, axis=0)).mean()),
                })
            elif topic == "/LIDAR_TOP":
                message = deserialize_message(raw, PointCloud2)
                timestamp = _header_timestamp_ns(message)
                points = np.frombuffer(
                    bytes(message.data), dtype=_point_dtype(message),
                    count=int(message.width) * int(message.height),
                )
                radius = np.hypot(points["x"], points["y"])
                row = {
                    **common,
                    "modality": "lidar",
                    "input_timestamp_ns": timestamp,
                    "point_count": int(len(points)),
                    "range_mean_m": float(radius.mean()),
                    "range_std_m": float(radius.std()),
                    "points_0_5m": int((radius < 5).sum()),
                    "points_5_20m": int(
                        ((radius >= 5) & (radius < 20)).sum()),
                    "points_20_40m": int(
                        ((radius >= 20) & (radius < 40)).sum()),
                    "points_40m_plus": int((radius >= 40).sum()),
                }
                if "intensity" in (points.dtype.names or ()):
                    row.update(
                        intensity_mean=float(points["intensity"].mean()),
                        intensity_std=float(points["intensity"].std()),
                    )
                rows.append(row)
            else:
                continue
            observed.add((topic, timestamp))
        if observed != expected:
            raise ValueError(
                f"{condition_id} characterized inputs differ from manifest")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination
