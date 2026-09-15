"""Transparent rosbag-to-model timestamp relay."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from .config import load_runtime_config
from closeloop_testbed.resource_control import apply_process_resources
from closeloop_profiler.source_frames import message_header_timestamp_ns

from .communication import (
    BatchedJsonlWriter,
    RELAY_ID,
    StreamOrderTracker,
    relay_routes,
)


def _message_kind(config, route):
    consumers = [
        model for model in config["models"]
        if model["input_topic"] == route["input_topic"]
    ]
    kinds = {
        model.get(
            "input_message_type",
            "compressed_image" if model["modality"] == "image"
            else "pointcloud2",
        )
        for model in consumers
    }
    if len(kinds) == 1:
        return kinds.pop()
    source = route["source_topic"].lower()
    if "image" in source or "cam" in source:
        return "compressed_image"
    if "lidar" in source or "point" in source:
        return "pointcloud2"
    raise ValueError(
        f"cannot determine message type for {route['source_topic']}"
    )


def main(argv=None):
    """Relay configured input topics unchanged and record publish timing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory", required=True)
    args, ros_args = parser.parse_known_args(argv)
    config = load_runtime_config(args.config).data
    resources = apply_process_resources(config["replay"])
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
        raise RuntimeError("relay_node requires ROS 2 Python packages") from exc

    run_directory = Path(args.run_directory)
    writer = BatchedJsonlWriter(
        run_directory / "communication_relay.jsonl"
    )
    tracker = StreamOrderTracker()
    counters = Counter()
    active_segment = {}
    routes = relay_routes(config)
    rclpy.init(args=ros_args)
    node = Node("closeloop_communication_relay")
    ready = node.create_publisher(String, "/closeloop/ready", 10)
    ack = node.create_publisher(String, "/closeloop/ack", 10)
    message_types = {
        "image": Image,
        "compressed_image": CompressedImage,
        "pointcloud2": PointCloud2,
    }

    segment_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    def segment_callback(message):
        active_segment.clear()
        active_segment.update(json.loads(message.data))

    node.create_subscription(
        String, "/closeloop/replay_segment", segment_callback, segment_qos
    )

    def completion_callback(_message):
        writer.flush()
        ack_message = String()
        ack_message.data = RELAY_ID
        ack.publish(ack_message)

    node.create_subscription(
        String, "/closeloop/replay_complete", completion_callback, 10
    )

    for route in routes:
        kind = _message_kind(config, route)
        consumers = [
            model for model in config["models"]
            if model["input_topic"] == route["input_topic"]
        ]
        reliable = consumers and all(
            model["qos"] == "reliable" for model in consumers
        )
        qos = QoSProfile(
            reliability=(
                ReliabilityPolicy.RELIABLE
                if reliable else ReliabilityPolicy.BEST_EFFORT
            ),
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        publisher = node.create_publisher(
            message_types[kind], route["input_topic"], qos
        )

        def relay_callback(message, selected=route, output=publisher):
            segment = dict(active_segment)
            timestamp_ns = message_header_timestamp_ns(message)
            order = tracker.observe(
                segment.get("segment_id", "unassigned"),
                selected["input_topic"],
                timestamp_ns,
            )
            pre_publish_ns = time.monotonic_ns()
            output.publish(message)
            writer.write({
                "schema": "communication_relay_v1",
                **segment,
                **selected,
                **order,
                "original_source_timestamp_ns": timestamp_ns,
                "relay_pre_publish_monotonic_ns": pre_publish_ns,
                "relay_post_publish_monotonic_ns": time.monotonic_ns(),
            })
            counters["messages"] += 1
            counters["duplicates"] += int(order["duplicate"])
            counters["out_of_order"] += int(order["out_of_order"])

        node.create_subscription(
            message_types[kind], route["raw_topic"], relay_callback, qos
        )

    ready_message = String()
    ready_message.data = RELAY_ID
    node.create_timer(0.2, lambda: ready.publish(ready_message))
    try:
        rclpy.spin(node)
    finally:
        writer.close()
        status = {
            "schema": "communication_relay_status_v1",
            "pid": os.getpid(),
            "routes": routes,
            "batch_size": writer.batch_size,
            "batches_written": writer.batches_written,
            **resources,
            **dict(counters),
        }
        (run_directory / "communication_relay_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
