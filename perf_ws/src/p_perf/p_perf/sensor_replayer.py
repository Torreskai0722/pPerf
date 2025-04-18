#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pandas as pd
import subprocess
import time
import os
from glob import glob
from p_perf.pPerf import pPerf

class SensorReplayer(Node):
    def __init__(self):
        super().__init__('sensor_replayer')

        # PARAMETERS
        self.declare_parameter('expected_models', 2)
        self.declare_parameter('index', 0)
        self.declare_parameter('gpu_duration', 0.05)
        self.declare_parameter('run_time', 30)  # seconds
        self.declare_parameter('bag_dir', '')            # directory containing .mcap files
        self.declare_parameter('data_dir', '')    

        self.expected_models = self.get_parameter('expected_models').value
        self.data_dir = self.get_parameter('data_dir').value
        self.index = self.get_parameter('index').value
        self.gpu_duration = self.get_parameter('gpu_duration').value
        self.run_time = self.get_parameter('run_time').value
        self.bag_dir = self.get_parameter('bag_dir').value

        # Discover .mcap files
        self.bag_files = sorted(glob(os.path.join(self.bag_dir, "*.mcap")))
        if not self.bag_files:
            raise RuntimeError(f"No .mcap files found in: {self.bag_dir}")
        self.get_logger().info(f"Found {len(self.bag_files)} .mcap files.")

        self.models_ready_count = 0
        self.replay_started = False
        self.replay_start_time = None
        self.shutdown_requested = False

        self.profiler = pPerf('', 0, self.gpu_duration)

        # ROS 2 communication
        self.terminate_publisher = self.create_publisher(String, 'terminate_inferencers', 10)
        self.create_subscription(String, 'inferencer_ready', self.inferencer_ready_callback, 10)

    def inferencer_ready_callback(self, msg):
        try:
            self.models_ready_count += int(msg.data)
            self.get_logger().info(f"Inferencer ready: {self.models_ready_count}/{self.expected_models}")
            if self.models_ready_count == self.expected_models and not self.replay_started:
                self.profiler.start_gpu_monitoring()
                self.get_logger().info("All inferencers ready. Starting profiling and replay.")
                self.replay_start_time = time.time()
                self.create_timer(0.5, self.check_shutdown)
                self.start_rosbag_sequence()
        except ValueError:
            self.get_logger().warn("Invalid readiness message.")

    def start_rosbag_sequence(self):
        self.replay_started = True
        for bag_path in self.bag_files:
            if self.shutdown_requested:
                self.get_logger().warn("Replay stopped early due to time limit.")
                break

            cmd = [
                "ros2", "bag", "play", bag_path,
                "--remap", "/LIDAR_TOP:=/lidar_data", "CAM_FRONT/image_rect_compressed:=/image_data",
                "--topics", "/LIDAR_TOP", "/CAM_FRONT/image_rect_compressed"
            ]


            self.get_logger().info(f"Replaying: {bag_path}")
            proc = subprocess.Popen(cmd)
            while proc.poll() is None:
                if self.time_exceeded():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return
                time.sleep(0.1)  # Avoid busy wait

        self.shutdown()

    def check_shutdown(self):
        if self.time_exceeded():
            self.shutdown()

    def time_exceeded(self):
        if self.replay_start_time is None:
            return False
        elapsed = time.time() - self.replay_start_time
        return elapsed >= self.run_time

    def shutdown(self):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True

        self.get_logger().info("SensorReplayer shutting down...")

        terminate_msg = String()
        terminate_msg.data = "TERMINATE"
        self.terminate_publisher.publish(terminate_msg)

        self.profiler.stop_gpu_monitoring()
        pd.DataFrame(self.profiler.gpu_stats).to_csv(f"{self.data_dir}/gpu_{self.index}.csv", index=False)

        self.destroy_node()
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = SensorReplayer()
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Shutdown").info("Done.")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
