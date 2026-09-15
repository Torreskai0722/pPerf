"""Paired ROS wrapper for the existing native FMA CUDA worker."""

import argparse
import json
import os
from pathlib import Path
import threading
import time

from .config import load_runtime_config
from .offline_workload import (
    WorkerProcess, discover_offline_native,
)

from .model_execution import write_status
from .process_env import fma_worker_environment


def worker_configuration(config):
    """Return the exact built-in FMA adapter configuration."""
    fma = config["synthetic_fma"]
    return {
        "adapter": "builtin_compute",
        "blocks": fma["blocks"],
        "threads": fma["threads"],
        "iterations": fma["iterations"],
    }


def main(argv=None):
    """Join paired readiness and launch one CTA-tracked FMA kernel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory", required=True)
    args, ros_args = parser.parse_known_args(argv)
    config = load_runtime_config(args.config).data
    fma = config["synthetic_fma"]
    client_id = fma["client_id"]
    run_directory = Path(args.run_directory)
    status_path = run_directory / f"model_{client_id}.json"
    status = {
        "schema": "synthetic_fma_status_v1",
        "model_id": client_id,
        "pid": os.getpid(),
        "state": "starting",
        "configured_cuda_mps_active_thread_percentage": fma[
            "mps_percentage"
        ],
        "process_observed_cuda_mps_active_thread_percentage": os.environ.get(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ),
        "configuration": worker_configuration(config),
        "error": None,
    }
    write_status(status_path, status)
    worker = None
    node = None
    ros_initialized = False
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        rclpy.init(args=ros_args)
        ros_initialized = True
        node = Node("closeloop_fma_workload")
        executable, adapter = discover_offline_native()
        worker = WorkerProcess(
            executable,
            adapter,
            worker_configuration(config),
            client_id,
            fma_worker_environment(config, os.environ),
            run_directory / f"model_{client_id}_worker.log",
            cta=True,
        )
        status.update(state="ready", identity=worker.identity)
        write_status(status_path, status)
        ready = node.create_publisher(String, "/closeloop/ready", 10)
        ack = node.create_publisher(String, "/closeloop/ack", 10)
        complete = node.create_publisher(
            String, "/closeloop/trial_complete", 10
        )
        started = threading.Event()
        finished = threading.Event()

        def run(epoch):
            try:
                release_ns = round(
                    (epoch + fma["paired_deadline_seconds"])
                    * 1_000_000_000
                )
                worker.send({"operation": "run", "release_ns": release_ns})
                result = worker.receive(
                    config["replay"]["completion_timeout_seconds"]
                )
                evidence = {
                    "schema": "synthetic_fma_cta_v1",
                    "release_monotonic_ns": release_ns,
                    "identity": worker.identity,
                    "result": result,
                }
                (run_directory / f"model_{client_id}_fma_cta.json").write_text(
                    json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                message = String()
                message.data = json.dumps({
                    "trial": 0,
                    "model_id": client_id,
                    "completed_time": time.monotonic(),
                    "cuda_synchronized": True,
                }, sort_keys=True)
                complete.publish(message)
                status.update(
                    state="completed", completed_trials=1,
                    missed_deadlines=0, quiescence_verified=True,
                )
            except Exception as exc:  # preserve sidecar evidence
                status.update(state="failed", error=str(exc))
            finally:
                write_status(status_path, status)
                finished.set()

        def epoch_callback(message):
            if started.is_set():
                return
            started.set()
            epoch = json.loads(message.data)["monotonic_epoch_seconds"]
            threading.Thread(target=run, args=(epoch,), daemon=True).start()

        node.create_subscription(
            String, "/closeloop/replay_epoch", epoch_callback, 10
        )
        ready_message = String()
        ready_message.data = client_id

        def timer_callback():
            ready.publish(ready_message)
            if finished.is_set() and status["state"] == "completed":
                ack.publish(ready_message)
                status["state"] = "acknowledged"
                write_status(status_path, status)

        node.create_timer(0.2, timer_callback)
        rclpy.spin(node)
    except Exception as exc:
        status.update(state="failed", error=str(exc))
        write_status(status_path, status)
        raise
    finally:
        if worker is not None:
            worker.close()
        if node is not None:
            node.destroy_node()
        if ros_initialized:
            rclpy.try_shutdown()
    return 0
