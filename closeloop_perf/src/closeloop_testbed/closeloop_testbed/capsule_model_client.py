"""Disposable, ROS-free model process for one exact capsule frame."""

import argparse
import json
import os
from pathlib import Path

from closeloop_profiler.capsule_profiler import GpuForensicsProfiler
from .config import load_runtime_config
from closeloop_profiler.native_agent import NativeAgentBridge
from closeloop_testbed.resource_control import (
    apply_process_resources, configure_model_thread_pools,
)
from closeloop_profiler.source_frames import load_capture_input

from .adapters import create_inferencer
from .model_execution import select_model, write_status


def decode_dataset_input(payload, encoding):
    """Adapt verified NuScenes bytes without constructing a ROS message."""
    import cv2  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    if encoding == "nuscenes_compressed_image":
        image = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("compressed dataset image could not be decoded")
        return image
    if encoding == "nuscenes_pointcloud_xyzir_f32":
        if len(payload) % (5 * 4):
            raise ValueError("NuScenes point cloud has an invalid shape")
        values = np.frombuffer(payload, dtype="<f4")
        return values.reshape((-1, 5))
    raise ValueError(f"unsupported dataset payload encoding {encoding}")


def main(argv=None) -> int:
    """Warm one model, enter the native barrier, and forward exactly once."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--barrier-timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    config = load_runtime_config(args.config).data
    model = select_model(config, args.model_id)
    status_path = Path(args.status)
    status = {
        "schema": "kernel_capsule_client_status_v1",
        "model_id": args.model_id,
        "pid": os.getpid(),
        "state": "initializing",
        "model_forward_count": 0,
        "ros_subscription_count": 0,
        "rosbag_work_count": 0,
        "error": None,
    }
    write_status(status_path, status)
    profiler = None
    frame_returned = False
    try:
        apply_process_resources(model)
        configure_model_thread_pools(model)
        identity, payload, payload_encoding = load_capture_input(
            Path(os.environ["PPERF_KERNEL_CAPSULE_INPUT"])
        )
        converted = decode_dataset_input(payload, payload_encoding)
        inferencer = create_inferencer(
            model["task"],
            model["modality"],
            model["mmlab_model"],
            model.get("inference_resize_scale"),
            model.get("model_config"),
            model.get("checkpoint"),
            model.get("checkpoint_sha256"),
        )
        profiler = GpuForensicsProfiler(
            inferencer.model,
            config["run"]["id"],
            config["replay"]["scene_token"],
            args.model_id,
            model["architecture_profile"],
            model["module_annotation_depth"],
            options={"mode": "kernel_capsule", "checkpoint_required": True},
        )
        for index in range(model["warmup_count"]):
            profiler.run(
                inferencer,
                model["warmup_input"],
                f"warmup-{index}",
                warmup=True,
            )
        profiler.validate_warmup()
        profiler.release_cached_memory()
        bridge = NativeAgentBridge.from_environment()
        bridge.report_warmup_complete(args.model_id)
        status["state"] = "warmup_ready"
        write_status(status_path, status)
        specification = json.loads(Path(
            os.environ["PPERF_KERNEL_CAPSULE_SPEC"]
        ).read_text(encoding="utf-8"))
        input_id = str(specification["inference_inputs"][args.model_id])
        bridge.set_frame(
            args.model_id,
            input_id,
            identity.ros_header_timestamp_ns,
        )
        bridge.wait_capture_epoch(args.barrier_timeout_seconds)
        status["state"] = "capturing"
        write_status(status_path, status)
        try:
            profiler.run(inferencer, converted, input_id)
            frame_returned = True
            status["model_forward_count"] = 1
            bridge.report_frame_end(True)
            raise RuntimeError(
                "capsule_invalid: selected frame returned before shutdown"
            )
        except Exception:
            if not frame_returned:
                bridge.report_frame_end(False)
            raise
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        write_status(status_path, status)
        raise
    finally:
        if profiler is not None:
            profiler.close()
    return 0
