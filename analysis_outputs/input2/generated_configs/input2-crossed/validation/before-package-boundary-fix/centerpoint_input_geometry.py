"""Reconstruct CenterPoint input geometry offline, without model/GPU execution."""

import argparse
import copy
import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml

from .crossed_evidence import read_json, read_jsonl, sha256, write_csv, write_json


def scan_work(keys, max_points):
    """Count loop iterations in point_to_voxelidx for valid ordered voxels.

    Each point scans all predecessors, stopping at the max_points-th match.
    This describes logical work, not hardware cycles or memory transactions.
    """
    # The CUDA code records the first match before checking the stopping limit.
    stop_at = max(2, max_points)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    limits = np.full(len(counts), len(keys), dtype=np.int64)
    large = counts > stop_at
    limits[large] = order[starts[large] + stop_at - 1] + 1
    iterations = np.minimum(np.arange(len(keys)), limits[inverse]).sum()
    return int(iterations), len(counts)


def reconstruct(root):
    # These imports use the installed workload's CPU transforms and ROS reader.
    import rosbag2_py
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmdet3d.utils import register_all_modules
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py.point_cloud2 import dtype_from_fields

    torch.set_num_threads(3)
    register_all_modules(init_default_scope=True)
    root = Path(root)
    destination = root / "analysis/input2-crossed/input-data/centerpoint_diagnosis"
    timings = list(csv.DictReader((destination / "frames.csv").open()))
    wanted = {r["source_frame_id"] for r in timings}
    rows, evidence, settings = [], {}, {}
    for scene in dict.fromkeys(r["scene_id"] for r in timings):
        run_id = next(r["run_id"] for r in timings if r["scene_id"] == scene)
        run = root / "runs" / run_id
        config = yaml.safe_load((run / "config.yaml").read_text())
        model = config["models"][0]
        config_path = Path(model["model_config"])
        if sha256(config_path) != model["model_config_sha256"]:
            raise ValueError("Installed model config differs from collected evidence")
        cfg = Config.fromfile(config_path)
        pipeline_cfg = copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
        pipeline_cfg[0]["type"] = "LidarDet3DInferencerLoader"
        pipeline = Compose(pipeline_cfg)
        voxel = cfg.model.data_preprocessor.voxel_layer
        size = np.asarray(voxel.voxel_size, dtype=np.float32)
        lower = np.asarray(voxel.point_cloud_range[:3], dtype=np.float32)
        grid = np.rint((np.asarray(voxel.point_cloud_range[3:]) - lower) / size).astype(np.int64)
        settings[scene] = dict(pipeline=pipeline_cfg, voxel_layer=dict(voxel))
        manifest_path = Path(config["replay"]["controlled_bag_manifest"])
        manifest = read_json(manifest_path)
        index = {r["output_header_timestamp_ns"]: r for r in read_jsonl(manifest["frame_index"])
                 if r["topic"] == "/LIDAR_TOP"}
        bag = rosbag2_py.SequentialReader()
        bag.open(rosbag2_py.StorageOptions(uri=manifest["bag_path"], storage_id="mcap"),
                 rosbag2_py.ConverterOptions("", ""))
        while bag.has_next():
            topic, raw, _ = bag.read_next()
            if topic != "/LIDAR_TOP":
                continue
            msg = deserialize_message(raw, PointCloud2)
            identity = index[msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec]
            if identity["source_frame_id"] not in wanted:
                continue
            if hashlib.sha256(raw).hexdigest() != identity["output_serialized_sha256"]:
                raise ValueError("Controlled source frame checksum differs")
            dtype = dtype_from_fields(msg.fields, point_step=msg.point_step)
            if msg.is_bigendian:
                dtype = dtype.newbyteorder(">")
            records = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
            names = [f.name for f in msg.fields if f.name in ("x", "y", "z", "intensity", "ring")]
            points = np.column_stack([records[n].astype(np.float32, copy=False) for n in names])
            points = points[:, :model.get("point_feature_count", points.shape[1])]
            processed = pipeline({"points": points})["inputs"]["points"].numpy()
            coordinates = np.floor((processed[:, :3] - lower) / size).astype(np.int64)
            if not np.all((coordinates >= 0) & (coordinates < grid)):
                raise ValueError("Geometry reconstruction includes invalid voxel coordinates")
            keys = (coordinates[:, 0] * grid[1] + coordinates[:, 1]) * grid[2] + coordinates[:, 2]
            work, occupied = scan_work(keys, voxel.max_num_points)
            rows.append(dict(scene_id=scene, source_frame_id=identity["source_frame_id"],
                             raw_points=len(points), voxel_input_points=len(processed),
                             occupied_voxels=occupied, point_scan_iterations=work,
                             input_payload_sha256=identity["payload_sha256"]))
        for path in (config_path, manifest_path, Path(manifest["frame_index"])):
            evidence[str(path)] = sha256(path)
        print(scene, "reconstructed", sum(r["scene_id"] == scene for r in rows), flush=True)
    if {r["source_frame_id"] for r in rows} != wanted:
        raise ValueError("Not all timed source frames were reconstructed")
    for path in [Path("/opt/mmcv/mmcv/ops/csrc/common/cuda/voxelization_cuda_kernel.cuh"),
                 Path("/opt/mmcv/mmcv/ops/csrc/pytorch/cuda/voxelization_cuda.cu"),
                 Path("/opt/mmdetection3d/mmdet3d/datasets/transforms/loading.py")]:
        evidence[str(path)] = sha256(path)
    write_csv(destination / "input_geometry.csv", rows)
    write_json(destination / "input_geometry_provenance.json", {
        "method": "CPU reconstruction with installed inferencer transforms; no GPU inference",
        "settings": settings, "sources_sha256": evidence, "analyzer_sha256": sha256(__file__),
        "limitation": "Reconstructed geometry/logical work; not a recorded GPU voxel count or cycle counter",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root")
    reconstruct(parser.parse_args().artifact_root)
