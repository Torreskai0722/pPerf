"""One-process-per-model MMlab inference ROS 2 node."""

import argparse
from collections import defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from closeloop_profiler.architecture_profiles import architecture_profile_metadata
from closeloop_profiler.components import create_model_profiler
from closeloop_profiler.source_frames import message_header_timestamp_ns
from .config import load_runtime_config
from closeloop_testbed.resource_control import (
    apply_process_resources,
    configure_model_thread_pools,
    MPS_ENVIRONMENT_VARIABLES,
)
from .adapters import (
    create_inferencer,
    decode_compressed_image,
    decode_image,
    decode_pointcloud2,
    select_point_features,
)
from .green_context import GreenContextError, activate_green_context
from .paired import (
    PairedTrialScheduler, ResidentSubgraphReplay, paired_participant_ids,
)
from .communication import BatchedJsonlWriter, StreamOrderTracker
from .capsule_helpers import select_cta_target
from .input_handling import (
    configured_input_queue_depth, model_input_record, model_input_segment,
)
from .model_execution import (
    select_model, update_profile_evidence, write_status,
)


def main(argv=None) -> int:
    """Start a configured model process and acknowledge drained replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-directory", required=True)
    args, ros_args = parser.parse_known_args(argv)
    config = load_runtime_config(args.config).data
    model_config = select_model(config, args.model_id)
    input_queue_depth = configured_input_queue_depth(model_config)
    resource_status = apply_process_resources(model_config)
    pool_status = configure_model_thread_pools(model_config)
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import CompressedImage, Image, PointCloud2
        from std_msgs.msg import String
    except ImportError as exc:
        raise RuntimeError(
            "model_node requires ROS 2 Python packages"
        ) from exc

    run_directory = Path(args.run_directory)
    status_path = run_directory / f"model_{args.model_id}.json"
    observed_mps = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    status = {
        "model_id": args.model_id,
        "pid": os.getpid(),
        "state": "starting",
        "inputs": 0,
        "error": None,
        "input_queue_depth": input_queue_depth,
        "configured_cuda_mps_active_thread_percentage": model_config.get(
            "mps_percentage"
        ),
        "process_observed_cuda_mps_active_thread_percentage": observed_mps,
        "process_observed_mps_environment": {
            name: os.environ.get(name) for name in MPS_ENVIRONMENT_VARIABLES
        },
    }
    status.update({
        name: model_config[name]
        for name in (
            "mmlab_model", "model_config", "checkpoint",
            "model_config_sha256", "checkpoint_sha256",
        )
        if name in model_config
    })
    status.update(resource_status)
    status.update(pool_status)
    status.update(
        architecture_profile_metadata(model_config["architecture_profile"])
    )
    green_configuration = model_config.get("green_context")
    if green_configuration is not None:
        status["green_context"] = {
            "requested_sm_count": green_configuration["sm_count"],
            "selected_partition_index": green_configuration[
                "partition_index"
            ],
            "active": False,
        }
    write_status(status_path, status)
    profiler = None
    input_log = None
    paired_log = None
    scheduler_thread = None
    node = None
    ros_initialized = False
    green_activation = None
    try:
        rclpy.init(args=ros_args)
        ros_initialized = True
        node = Node(model_config["node_name"])
        if green_configuration is not None:
            try:
                green_activation = activate_green_context(
                    green_configuration, config["gpu"]["index"]
                )
                status["green_context"] = green_activation.evidence
                write_status(status_path, status)
            except GreenContextError as exc:
                status["green_context"] = exc.evidence
                raise
        inferencer = create_inferencer(
            model_config["task"],
            model_config["modality"],
            model_config["mmlab_model"],
            model_config.get("inference_resize_scale"),
            model_config.get("model_config"),
            model_config.get("checkpoint"),
            model_config.get("checkpoint_sha256"),
        )
        status["checkpoint_load_policy"] = (
            "trusted_sha256_pinned_weights_only_false"
            if model_config.get("checkpoint") else "model_alias_default"
        )
        if inferencer.resize_scale is not None:
            status["inference_resize_scale"] = list(inferencer.resize_scale)
            write_status(status_path, status)
        profiler = create_model_profiler(config, inferencer, args.model_id)
        if config["replay"].get("controlled_bag_manifest"):
            bag_manifest = json.loads(Path(config["replay"]["controlled_bag_manifest"]).read_text())
            profiler.scene_token = bag_manifest["input_scenes"][model_config["modality"]]["scene_token"]
        cta_configuration = model_config.get("nvbit_cta_profile")
        cta_arm = None
        cta_disarm = None
        cta_input_begin = None
        cta_input_end = None
        cta_targets = {}
        if cta_configuration:
            tracker = ctypes.CDLL(None)
            cta_arm = tracker.pperf_nvbit_cta_arm
            cta_arm.argtypes = [ctypes.c_char_p]
            cta_arm.restype = ctypes.c_int
            cta_disarm = tracker.pperf_nvbit_cta_disarm
            cta_disarm.argtypes = []
            cta_disarm.restype = ctypes.c_int
            if cta_configuration.get("mode", "controlled") in (
                "passive", "mixed"
            ):
                cta_input_begin = tracker.pperf_nvbit_cta_input_begin
                cta_input_begin.argtypes = []
                cta_input_begin.restype = ctypes.c_int
                cta_input_end = tracker.pperf_nvbit_cta_input_end
                cta_input_end.argtypes = []
                cta_input_end.restype = ctypes.c_int
            cta_targets = cta_configuration["targets"]
        ready = node.create_publisher(String, "/closeloop/ready", 10)
        ack = node.create_publisher(String, "/closeloop/ack", 10)
        completion_seen = threading.Event()
        paired_stop = threading.Event()
        last_input = [time.monotonic()]
        acknowledged = [False]
        active_segment = {}
        input_order = StreamOrderTracker()
        last_callback_exit_ns = [None]
        paired_config = model_config.get("paired_trial")
        if paired_config is None:
            input_log = BatchedJsonlWriter(
                run_directory / f"model_{args.model_id}_inputs.jsonl"
            )
        else:
            fixed_input = Path(model_config["warmup_input"])
            fixed_input_sha256 = hashlib.sha256(
                fixed_input.read_bytes()
            ).hexdigest()
            status.update(
                {
                    "paired_trial": dict(paired_config),
                    "fixed_input_sha256": fixed_input_sha256,
                    "fixed_input_reused": None,
                }
            )
            paired_log = (
                run_directory / f"model_{args.model_id}_paired.jsonl"
            ).open("w", encoding="utf-8", buffering=1)

        def input_callback(message):
            callback_entry_ns = time.monotonic_ns()
            input_id = status["inputs"]
            segment = model_input_segment(active_segment, model_config["modality"])
            order = input_order.observe(
                segment.get("segment_id", "unassigned"),
                model_config["input_topic"],
                message_header_timestamp_ns(message),
            )
            record = model_input_record(
                args.model_id,
                input_id,
                model_config["input_topic"],
                message,
                callback_entry_ns,
                segment,
                order,
            )
            record["previous_callback_exit_monotonic_ns"] = (
                last_callback_exit_ns[0]
            )
            if segment.get("scene_token"):
                profiler.scene_token = segment["scene_token"]
            message_kind = model_config.get(
                "input_message_type",
                (
                    "compressed_image"
                    if model_config["modality"] == "image"
                    else "pointcloud2"
                ),
            )
            decoders = {
                "image": decode_image,
                "compressed_image": decode_compressed_image,
                "pointcloud2": decode_pointcloud2,
            }
            try:
                record["decode_start_monotonic_ns"] = time.monotonic_ns()
                converted = decoders[message_kind](message)
                record["decode_end_monotonic_ns"] = time.monotonic_ns()
                if message_kind == "pointcloud2":
                    record["input_point_count"] = int(converted.shape[0])
                    converted = select_point_features(
                        converted, model_config.get("point_feature_count")
                    )
                record["model_pipeline_start_monotonic_ns"] = (
                    time.monotonic_ns()
                )
                target = select_cta_target(
                    cta_targets,
                    segment.get("scene_index"),
                    order.get("message_order"),
                )
                if target:
                    result = cta_arm(target["label"].encode("utf-8"))
                    if result:
                        raise RuntimeError(
                            f"NVBit CTA arm failed with status {result}"
                        )
                    record["nvbit_cta_target_label"] = target["label"]
                elif cta_input_begin is not None:
                    result = cta_input_begin()
                    if result:
                        raise RuntimeError(
                            "NVBit CTA input begin failed with status "
                            f"{result}"
                        )
                try:
                    profiler.run(inferencer, converted, input_id)
                finally:
                    if target and cta_disarm is not None:
                        result = cta_disarm()
                        if result:
                            raise RuntimeError(
                                "NVBit CTA disarm failed with status "
                                f"{result}"
                            )
                    elif cta_input_end is not None:
                        result = cta_input_end()
                        if result:
                            raise RuntimeError(
                                "NVBit CTA input end failed with status "
                                f"{result}"
                            )
                record["model_pipeline_end_monotonic_ns"] = (
                    time.monotonic_ns()
                )
                record["inference_completion_monotonic_ns"] = getattr(
                    profiler, "last_completion_monotonic_ns", None
                )
                record["completed"] = True
                status["inputs"] += 1
                last_input[0] = time.monotonic()
            finally:
                callback_exit_ns = time.monotonic_ns()
                record["model_callback_exit_monotonic_ns"] = callback_exit_ns
                record.setdefault("completed", False)
                input_log.write(record)
                last_callback_exit_ns[0] = callback_exit_ns

        def completion_callback(_message):
            completion_seen.set()

        reliability = (
            ReliabilityPolicy.RELIABLE
            if model_config["qos"] == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        qos = QoSProfile(
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=input_queue_depth,
        )
        message_kind = model_config.get(
            "input_message_type",
            (
                "compressed_image"
                if model_config["modality"] == "image"
                else "pointcloud2"
            ),
        )
        message_types = {
            "image": Image,
            "compressed_image": CompressedImage,
            "pointcloud2": PointCloud2,
        }
        message_type = message_types[message_kind]
        if paired_config is None:
            segment_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )

            def segment_callback(message):
                active_segment.clear()
                active_segment.update(json.loads(message.data))

            node.create_subscription(
                String,
                "/closeloop/replay_segment",
                segment_callback,
                segment_qos,
            )
            node.create_subscription(
                message_type, model_config["input_topic"], input_callback, qos
            )
        else:
            scheduler = PairedTrialScheduler(
                paired_config["trials"],
                paired_config["period_seconds"],
                paired_config["deadline_offset_seconds"],
                paired_config["launch_tolerance_seconds"],
                timing_diagnostics_only=paired_config.get(
                    "timing_diagnostics_only", False),
            )
            epoch_seen = threading.Event()
            paired_error = []
            paired_finished = threading.Event()
            completion_condition = threading.Condition()
            completed_models = defaultdict(set)
            work_records = {}
            trial_complete = node.create_publisher(
                String, "/closeloop/trial_complete", 10
            )
            expected_models = set(paired_participant_ids(config))

            def trial_complete_callback(message):
                try:
                    record = json.loads(message.data)
                    trial = int(record["trial"])
                    model_id = str(record["model_id"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    return
                if model_id in expected_models:
                    with completion_condition:
                        completed_models[trial].add(model_id)
                        completion_condition.notify_all()

            node.create_subscription(
                String,
                "/closeloop/trial_complete",
                trial_complete_callback,
                10,
            )

            replay = None
            if (
                paired_config["role"] == "co_runner"
                and paired_config["k_replays"]
            ):
                replay = ResidentSubgraphReplay(
                    profiler.model, paired_config["replay_module"]
                )
                replay.start_capture()

            def paired_callback(trial, scheduled, actual):
                target = select_cta_target(cta_targets, trial=trial)
                if target:
                    result = cta_arm(target["label"].encode("utf-8"))
                    if result:
                        raise RuntimeError(
                            f"NVBit CTA arm failed with status {result}"
                        )
                elif cta_input_begin is not None:
                    result = cta_input_begin()
                    if result:
                        raise RuntimeError(
                            "NVBit CTA input begin failed with status "
                            f"{result}"
                        )
                try:
                    profiler.run(
                        inferencer, model_config["warmup_input"],
                        f"paired-{trial}"
                    )
                finally:
                    if target and cta_disarm is not None:
                        result = cta_disarm()
                        if result:
                            raise RuntimeError(
                                "NVBit CTA disarm failed with status "
                                f"{result}"
                            )
                    elif cta_input_end is not None:
                        result = cta_input_end()
                        if result:
                            raise RuntimeError(
                                "NVBit CTA input end failed with status "
                                f"{result}"
                            )
                service_ns = (
                    replay.replay(
                        paired_config["k_replays"], trial, args.model_id
                    )
                    if replay is not None
                    else 0
                )
                status["inputs"] += 1
                last_input[0] = time.monotonic()
                work_records[trial] = {
                    "trial": trial,
                    "scheduled_time": scheduled,
                    "actual_time": actual,
                    "work_completed_time": last_input[0],
                    "launch_error_seconds": actual - scheduled,
                    "subgraph_kernel_time_ns": service_ns,
                    "fixed_input_sha256": fixed_input_sha256,
                    "resident_input_sha256": status.get(
                        "resident_input_sha256"
                    ),
                    "nvbit_cta_target_label": (
                        target["label"] if target else None
                    ),
                }

            def paired_barrier(trial, boundary):
                message = String()
                message.data = json.dumps(
                    {
                        "trial": trial,
                        "model_id": args.model_id,
                        "completed_time": time.monotonic(),
                        "cuda_synchronized": True,
                    },
                    sort_keys=True,
                )
                trial_complete.publish(message)
                with completion_condition:
                    while (
                        completed_models[trial] != expected_models
                        and not paired_stop.is_set()
                        and time.monotonic() < boundary
                    ):
                        completion_condition.wait(
                            min(0.05, max(0.0, boundary - time.monotonic()))
                        )
                    quiescent = completed_models[trial] == expected_models
                record = work_records[trial]
                record["quiescent_time"] = time.monotonic()
                record["quiescence_verified"] = quiescent
                paired_log.write(json.dumps(record, sort_keys=True) + "\n")
                return quiescent

            def epoch_callback(message):
                nonlocal scheduler_thread
                if epoch_seen.is_set():
                    return
                epoch = json.loads(message.data)["monotonic_epoch_seconds"]
                epoch_seen.set()
                status["replay_monotonic_epoch_seconds"] = epoch

                def run_periodic():
                    try:
                        scheduler.run(
                            epoch, paired_callback, paired_barrier, paired_stop
                        )
                    except Exception as exc:  # worker evidence must survive
                        paired_error.append(str(exc))
                    finally:
                        paired_finished.set()

                scheduler_thread = threading.Thread(
                    target=run_periodic, daemon=True
                )
                scheduler_thread.start()

            epoch_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            node.create_subscription(
                String, "/closeloop/replay_epoch", epoch_callback, epoch_qos
            )
        node.create_subscription(
            String, "/closeloop/replay_complete", completion_callback, 10
        )
        warmup = model_config["warmup_input"]
        for index in range(model_config["warmup_count"]):
            if cta_input_begin is not None:
                result = cta_input_begin()
                if result:
                    raise RuntimeError(
                        f"NVBit CTA warmup begin failed with status {result}"
                    )
            try:
                profiler.run(
                    inferencer, warmup, f"warmup-{index}", warmup=True
                )
            finally:
                if cta_input_end is not None:
                    result = cta_input_end()
                    if result:
                        raise RuntimeError(
                            "NVBit CTA warmup end failed with status "
                            f"{result}"
                        )
        profiler.validate_warmup()
        if paired_config is not None and replay is not None:
            status["resident_input_sha256"] = replay.finish_capture()
            status["resident_input_reused"] = True
        profiler.release_cached_memory()
        update_profile_evidence(status, profiler)
        status["state"] = "ready"
        write_status(status_path, status)
        ready_message = String()
        ready_message.data = args.model_id
        # Publish repeatedly to tolerate discovery races.
        ready.publish(ready_message)

        def timer_callback():
            ready.publish(ready_message)
            quiet = model_config.get("quiet_period_seconds", 0.5)
            workload_done = (
                completion_seen.is_set()
                if paired_config is None
                else paired_finished.is_set()
            )
            if (
                workload_done
                and not acknowledged[0]
                and time.monotonic() - last_input[0] >= quiet
            ):
                if paired_config is not None:
                    status.update(scheduler.summary())
                    status["subgraph_kernel_time_ns"] = sum(
                        record["subgraph_kernel_time_ns"]
                        for record in work_records.values()
                    )
                    if paired_error:
                        status["state"] = "failed"
                        status["error"] = paired_error[0]
                        write_status(status_path, status)
                        ack_message = String()
                        ack_message.data = args.model_id
                        ack.publish(ack_message)
                        acknowledged[0] = True
                        return
                    status["fixed_input_reused"] = (
                        hashlib.sha256(fixed_input.read_bytes()).hexdigest()
                        == fixed_input_sha256
                    )
                    if not status["fixed_input_reused"]:
                        status["state"] = "failed"
                        status["error"] = "fixed paired input changed"
                        write_status(status_path, status)
                        ack_message = String()
                        ack_message.data = args.model_id
                        ack.publish(ack_message)
                        acknowledged[0] = True
                        return
                ack_message = String()
                ack_message.data = args.model_id
                ack.publish(ack_message)
                acknowledged[0] = True
                status["state"] = "acknowledged"
                update_profile_evidence(status, profiler)
                write_status(status_path, status)

        node.create_timer(0.2, timer_callback)
        rclpy.spin(node)
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        if profiler is not None:
            update_profile_evidence(status, profiler)
        write_status(status_path, status)
        raise
    finally:
        if input_log is not None:
            input_log.close()
        if paired_log is not None:
            paired_log.close()
        if "replay" in locals() and replay is not None:
            replay.close()
        if profiler is not None:
            profiler.close()
        if node is not None:
            node.destroy_node()
        if ros_initialized:
            rclpy.try_shutdown()
    return 0
