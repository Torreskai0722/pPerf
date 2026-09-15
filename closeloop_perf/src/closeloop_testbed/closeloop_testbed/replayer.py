"""Ordered MCAP replay coordinator."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Dict, Iterable, List, Set

from .config import load_runtime_config
from closeloop_testbed.resource_control import apply_process_resources

from .communication import RELAY_ID, raw_topic
from .paired import paired_participant_ids


def scene_name(metadata_path: str, scene_token: str) -> str:
    """Resolve one scene token using the configured metadata file."""
    records = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    matches = [
        record["name"]
        for record in records
        if record.get("token") == scene_token
    ]
    if len(matches) != 1:
        raise ValueError(f"scene token {scene_token!r} was not found uniquely")
    return matches[0]


def discover_bags(directory: str, scene_token: str) -> List[Path]:
    """Return sorted MCAP files whose names contain the scene token."""
    root = Path(directory)
    matches = sorted(
        path
        for path in root.rglob("*.mcap")
        if scene_token in path.name or scene_token in str(path.parent)
    )
    if not matches:
        raise FileNotFoundError(
            f"no MCAP files for scene {scene_token!r} in {directory}"
        )
    return matches


def ordered_scene_tokens(replay_config):
    """Return configured scene tokens with legacy single-scene fallback."""
    return replay_config.get(
        "scene_tokens", [replay_config["scene_token"]]
    )


def playback_segments(replay_config):
    """Resolve every scene and MCAP into one ordered playback plan."""
    segments = []
    for pass_index in range(replay_config.get("repeat_count", 1)):
        for scene_index, token in enumerate(ordered_scene_tokens(replay_config)):
            name = scene_name(replay_config["metadata_path"], token)
            bags = discover_bags(replay_config["bag_directory"], name)
            for bag_index, bag in enumerate(bags):
                segments.append({
                    "segment_id": (
                        f"pass-{pass_index}-scene-{scene_index}-bag-{bag_index}"
                    ),
                    "pass_index": pass_index,
                    "scene_index": scene_index,
                    "scene_token": token,
                    "scene_name": name,
                    "bag_index": bag_index,
                    "bag_path": str(bag.resolve()),
                })
    return segments


def play_ordered_segments(segments, play_one):
    """Play segments synchronously, stopping after the first failure."""
    intervals = []
    for segment in segments:
        interval = play_one(segment)
        intervals.append(interval)
        if interval["completion_status"] != "completed":
            break
    return intervals


def relay_remappings(replay_config):
    """Remap configured rosbag topics to the relay's internal topics."""
    return {
        source: raw_topic(source) for source in replay_config["topics"]
    }


def replay_command(
    bags: Iterable[Path],
    rate: float,
    topics: Iterable[str],
    remappings: Dict[str, str],
) -> List[str]:
    """Build an exact one-pass rosbag replay command."""
    command = ["ros2", "bag", "play"] + [str(path) for path in bags]
    command += [
        "--rate",
        str(rate),
        "--storage",
        "mcap",
        "--start-paused",
        "--disable-keyboard-controls",
        "--topics",
    ]
    command += list(topics)
    if remappings:
        command += ["--remap"]
        for source, target in sorted(remappings.items()):
            command.append(f"{source}:={target}")
    return command


def resume_player(
    node, process: subprocess.Popen, timeout: float, resume_at: float = None
) -> float:
    """Resume a paused rosbag player after its ROS service is discoverable."""
    from rosbag2_interfaces.srv import Resume

    client = node.create_client(Resume, "/rosbag2_player/resume")
    deadline = time.monotonic() + timeout
    try:
        while not client.wait_for_service(timeout_sec=0.1):
            if process.poll() is not None:
                raise RuntimeError(
                    "rosbag player exited before its resume service was ready"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("rosbag player resume service timeout")

        if resume_at is not None:
            time.sleep(max(0.0, resume_at - time.monotonic()))

        future = client.call_async(Resume.Request())
        resumed_at = time.monotonic()
        while not future.done():
            if process.poll() is not None:
                raise RuntimeError(
                    "rosbag player exited before acknowledging resume"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("rosbag player resume request timeout")
            time.sleep(0.01)
        if future.exception() is not None:
            raise RuntimeError(
                f"rosbag player resume failed: {future.exception()}"
            )
        return resumed_at
    finally:
        node.destroy_client(client)


def stop_process(
    process: subprocess.Popen, first_signal: int = signal.SIGTERM
) -> None:
    """Stop a surviving child after replay coordination fails."""
    if process.poll() is None:
        process.send_signal(first_signal)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def wait_for_player(process: subprocess.Popen, timeout: float = None) -> tuple:
    """Wait for a player and report an expected partial-playback cutoff."""
    if timeout is None:
        return process.wait(), False
    try:
        return process.wait(timeout=max(0.0, timeout)), False
    except subprocess.TimeoutExpired:
        stop_process(process, signal.SIGINT)
        return process.returncode, True


class IdentityTracker:
    """Track exact model identities for readiness and completion."""

    def __init__(self, expected: Iterable[str]):
        """Initialize expected, ready, and acknowledged identities."""
        self.expected: Set[str] = set(expected)
        self.ready: Set[str] = set()
        self.acknowledged: Set[str] = set()

    def mark_ready(self, model_id: str) -> bool:
        """Mark an expected model ready; ignore unknown identities."""
        if model_id in self.expected:
            self.ready.add(model_id)
        return self.ready == self.expected

    def acknowledge(self, model_id: str) -> bool:
        """Record completion from an expected model."""
        if model_id in self.expected:
            self.acknowledged.add(model_id)
        return self.acknowledged == self.expected


def main(argv=None) -> int:
    """Run the ROS 2 replay coordination node."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory", required=True)
    args, ros_args = parser.parse_known_args(argv)
    config = load_runtime_config(args.config).data
    apply_process_resources(config["replay"])
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
    except ImportError as exc:
        raise RuntimeError(
            "replayer_node requires ROS 2 Python packages"
        ) from exc

    rclpy.init(args=ros_args)
    node = Node("closeloop_replayer")
    paired = all("paired_trial" in model for model in config["models"])
    relay_enabled = (
        not paired and "input" in config["recording"]["scopes"]
    )
    expected = paired_participant_ids(config) if paired else [
        model["id"] for model in config["models"]
    ]
    if relay_enabled:
        expected.append(RELAY_ID)
    tracker = IdentityTracker(expected)
    lock = threading.Lock()
    done = threading.Event()

    def ready_callback(message):
        with lock:
            tracker.mark_ready(message.data)

    def ack_callback(message):
        with lock:
            if tracker.acknowledge(message.data):
                done.set()

    node.create_subscription(String, "/closeloop/ready", ready_callback, 10)
    node.create_subscription(String, "/closeloop/ack", ack_callback, 10)
    completion = node.create_publisher(
        String, "/closeloop/replay_complete", 10
    )
    epoch_publisher = node.create_publisher(
        String,
        "/closeloop/replay_epoch",
        QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        ),
    )
    segment_publisher = node.create_publisher(
        String,
        "/closeloop/replay_segment",
        QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        ),
    )
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    replay_success = False
    playback_mode = config["replay"].get("playback_mode", "full")
    duration_seconds = config["replay"].get("duration_seconds")
    partial_cutoff_applied = False
    bags_started = 0
    bags_completed = 0
    replay_epoch = None
    replay_resumed_at = None
    playback_intervals = []
    try:
        deadline = (
            time.monotonic() + config["replay"]["readiness_timeout_seconds"]
        )
        while (
            time.monotonic() < deadline and tracker.ready != tracker.expected
        ):
            time.sleep(0.05)
        if tracker.ready != tracker.expected:
            error = "readiness timeout"
        elif paired:
            replay_epoch = time.monotonic() + 2.0
            epoch_message = String()
            epoch_message.data = json.dumps(
                {"monotonic_epoch_seconds": replay_epoch}
            )
            epoch_publisher.publish(epoch_message)
            replay_success = done.wait(
                config["replay"]["completion_timeout_seconds"]
            )
            error = None if replay_success else "paired completion failed"
        else:
            segments = playback_segments(config["replay"])
            replay_success = True
            partial_deadline = None

            def play_one(segment):
                nonlocal bags_started, bags_completed, partial_deadline
                nonlocal partial_cutoff_applied, replay_epoch
                nonlocal replay_resumed_at, replay_success
                interval = dict(segment)
                interval["process_started_monotonic_ns"] = time.monotonic_ns()
                interval["resume_monotonic_ns"] = None
                interval["end_monotonic_ns"] = None
                interval["returncode"] = None
                interval["partial_cutoff"] = False
                interval["completion_status"] = "started"
                segment_message = String()
                segment_message.data = json.dumps(segment, sort_keys=True)
                segment_publisher.publish(segment_message)
                command = replay_command(
                    [Path(segment["bag_path"])],
                    config["replay"]["rate"],
                    config["replay"]["topics"],
                    (
                        relay_remappings(config["replay"])
                        if relay_enabled
                        else config["replay"]["remappings"]
                    ),
                )
                process = subprocess.Popen(command)
                bags_started += 1
                try:
                    if replay_epoch is None:
                        replay_epoch = time.monotonic() + 2.0
                        epoch_message = String()
                        epoch_message.data = json.dumps(
                            {"monotonic_epoch_seconds": replay_epoch}
                        )
                        epoch_publisher.publish(epoch_message)
                    replay_resumed_at = resume_player(
                        node,
                        process,
                        config["replay"].get(
                            "player_startup_timeout_seconds", 10.0
                        ),
                        replay_epoch if bags_started == 1 else None,
                    )
                    interval["resume_monotonic_ns"] = int(
                        replay_resumed_at * 1_000_000_000
                    )
                    if playback_mode == "partial":
                        if partial_deadline is None:
                            partial_deadline = (
                                time.monotonic()
                                + duration_seconds / config["replay"]["rate"]
                            )
                        remaining = partial_deadline - time.monotonic()
                    else:
                        remaining = None
                    returncode, cutoff = wait_for_player(process, remaining)
                except Exception as exc:
                    stop_process(process)
                    interval["completion_status"] = "failed"
                    interval["error"] = str(exc)
                    interval["end_monotonic_ns"] = time.monotonic_ns()
                    raise
                interval["returncode"] = returncode
                interval["partial_cutoff"] = cutoff
                interval["end_monotonic_ns"] = time.monotonic_ns()
                if cutoff:
                    partial_cutoff_applied = True
                    interval["completion_status"] = "partial_cutoff"
                elif returncode != 0:
                    replay_success = False
                    interval["completion_status"] = "failed"
                else:
                    bags_completed += 1
                    interval["completion_status"] = "completed"
                return interval

            playback_intervals = play_ordered_segments(segments, play_one)
            replay_success = replay_success and (
                len(playback_intervals) == len(segments)
                and all(
                    interval["completion_status"] in (
                        "completed", "partial_cutoff"
                    )
                    for interval in playback_intervals
                )
            )
            message = String()
            message.data = json.dumps(
                {
                    "scene_tokens": ordered_scene_tokens(config["replay"]),
                    "playback_intervals": playback_intervals,
                },
                sort_keys=True,
            )
            completion.publish(message)
            done.wait(config["replay"]["completion_timeout_seconds"])
            error = (
                None
                if replay_success and done.is_set()
                else "completion failed"
            )
    except Exception as exc:  # result evidence must survive testbed failures
        error = str(exc)
    except KeyboardInterrupt:
        error = "interrupted"
    result = {
        "replay_success": replay_success,
        "all_acknowledged": tracker.acknowledged == tracker.expected,
        "ready_models": sorted(tracker.ready),
        "acknowledged_models": sorted(tracker.acknowledged),
        "playback_mode": playback_mode,
        "requested_duration_seconds": duration_seconds,
        "partial_cutoff_applied": partial_cutoff_applied,
        "bags_started": bags_started,
        "bags_completed": bags_completed,
        "replay_monotonic_epoch_seconds": replay_epoch,
        "replay_resumed_at_monotonic_seconds": replay_resumed_at,
        "scene_tokens": ordered_scene_tokens(config["replay"]),
        "repeat_count": config["replay"].get("repeat_count", 1),
        "playback_intervals": playback_intervals,
        "communication_profile_enabled": relay_enabled,
        "error": error,
    }
    output = Path(args.run_directory) / "testbed_result.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rclpy.try_shutdown()
    spinner.join(timeout=5)
    node.destroy_node()
    return 0 if error is None else 1
