#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pandas as pd
import subprocess
import os
from glob import glob
from p_perf.pPerf import pPerf
import time

from nuscenes.nuscenes import NuScenes
DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
nusc = NuScenes(
    version='v1.0-mini',
    dataroot=DATA_ROOT,
    verbose=True
)

class SensorReplayer(Node):
    def __init__(self):
        super().__init__('sensor_replayer')

        # PARAMETERS
        self.declare_parameter('expected_models', 2)
        self.declare_parameter('index', 0)
        self.declare_parameter('gpu_duration', 0.05)
        self.declare_parameter('bag_dir', '')
        self.declare_parameter('data_dir', '')
        self.declare_parameter('scene', '')
        self.declare_parameter('is_ms_mode', False)    

        self.expected_models = self.get_parameter('expected_models').value
        self.data_dir = self.get_parameter('data_dir').value
        self.index = self.get_parameter('index').value
        self.gpu_duration = self.get_parameter('gpu_duration').value
        self.bag_dir = self.get_parameter('bag_dir').value
        self.is_ms_mode = self.get_parameter('is_ms_mode').value
        print(self.bag_dir)

        scene_token = self.get_parameter('scene').value
        self.scene = nusc.get('scene', scene_token)['name']

        print(self.scene)
        
        # Discover .mcap files
        all_bags = glob(os.path.join(self.bag_dir, '**', "*.mcap"), recursive=True)
        print(all_bags)
        self.bag_file = sorted([f for f in all_bags if self.scene in os.path.basename(f)])
        if not self.bag_file:
            raise RuntimeError(f"No .mcap files found for {self.scene}")

        self.models_ready_count = 0
        self.replay_started = False

        self.profiler = pPerf('', 0, self.gpu_duration)

        # ROS 2 communication
        self.terminate_publisher = self.create_publisher(String, 'terminate_inferencers', 10)

        self.inferencer_sub = self.create_subscription(String, 'inferencer_ready', self.inferencer_ready_callback, 10)

    def inferencer_ready_callback(self, msg):
        try:
            self.models_ready_count += int(msg.data)
            self.get_logger().info(f"Inferencer ready: {self.models_ready_count}/{self.expected_models}")
            if self.models_ready_count == self.expected_models and not self.replay_started:
                self.profiler.start_gpu_monitoring()
                self.get_logger().info("All inferencers ready. Starting profiling and replay.")
                self.start_rosbag_sequence()
        except ValueError:
            self.get_logger().warn("Invalid readiness message.")


    def start_rosbag_sequence(self):
        self.replay_started = True
        for bag_path in self.bag_file:
            cmd = [
                "ros2", "bag", "play", bag_path, "--clock", 
                "--remap", 
                "/LIDAR_TOP:=/lidar_data", 
                "/CAM_FRONT/image_rect_compressed:=/image_data/CAM_FRONT",
                # "/CAM_FRONT_LEFT/image_rect_compressed:=/image_data/CAM_FRONT_LEFT",
                # "/CAM_FRONT_RIGHT/image_rect_compressed:=/image_data/CAM_FRONT_RIGHT",
                # "/CAM_BACK/image_rect_compressed:=/image_data/CAM_BACK",
                # "/CAM_BACK_LEFT/image_rect_compressed:=/image_data/CAM_BACK_LEFT",
                # "/CAM_BACK_RIGHT/image_rect_compressed:=/image_data/CAM_BACK_RIGHT",
                "--topics", 
                "/LIDAR_TOP", 
                "/CAM_FRONT/image_rect_compressed",
                # "/CAM_FRONT_LEFT/image_rect_compressed",
                # "/CAM_FRONT_RIGHT/image_rect_compressed",
                # "/CAM_BACK/image_rect_compressed",
                # "/CAM_BACK_LEFT/image_rect_compressed",
                # "/CAM_BACK_RIGHT/image_rect_compressed"
            ]

            self.get_logger().info(f"Replaying: {bag_path}")
            proc = subprocess.Popen(cmd)
            proc.wait()
            
            # Ensure the process is terminated
            if proc.poll() is None:
                proc.terminate()
                proc.wait()

        self.shutdown()

    def shutdown(self):
        self.get_logger().info("SensorReplayer shutting down...")

        # Kill any remaining rosbag processes
        try:
            subprocess.run(["pkill", "-f", "ros2 bag play"], check=False)
        except Exception as e:
            self.get_logger().warn(f"Error killing rosbag processes: {e}")

        terminate_msg = String()
        terminate_msg.data = "TERMINATE"
        self.terminate_publisher.publish(terminate_msg)

        self.profiler.stop_gpu_monitoring()
        pd.DataFrame(self.profiler.gpu_stats).to_csv(f"{self.data_dir}/gpu_{self.index}.csv", index=False)

        # Clean up ROS 2 resources
        self.destroy_publisher(self.terminate_publisher)
        self.destroy_subscription(self.inferencer_sub)
        
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
