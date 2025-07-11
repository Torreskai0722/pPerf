#!/usr/bin/python3

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage
import numpy as np
import cv2
import os
import time
import torch

import pandas as pd
import json

from p_perf.pPerf import pPerf
from p_perf.utils import load_sweep_sd, get_paths_from_sd, convert_to_kitti

from nuscenes.nuscenes import NuScenes
DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
nusc = NuScenes(
    version='v1.0-mini',
    dataroot=DATA_ROOT,
    verbose=True
)


class SensorPublisherNode(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # RUNNING REQUIREMENTS
        self.declare_parameter('expected_models', 2)
        self.declare_parameter('index', 0)
        self.declare_parameter('data_dir', '')
        self.declare_parameter('lidar_model_mode', 'nus')

        self.expected_models = self.get_parameter('expected_models').value
        self.index = self.get_parameter('index').value
        self.data_dir = self.get_parameter('data_dir').value
        self.lidar_model_mode = self.get_parameter('lidar_model_mode').value  

        # BASIC EXPERIMENT PARAMETERS
        self.declare_parameter('publish_freq_lidar', 20)
        self.declare_parameter('publish_freq_image', 12)
        self.declare_parameter('scene', 'cc8c0bf57f984915a77078b10eb33198')
        self.declare_parameter('gpu_duration', 0.025)

        self.publish_freq_lidar = self.get_parameter('publish_freq_lidar').value
        self.publish_freq_image = self.get_parameter('publish_freq_image').value
        scene_token = self.get_parameter('scene').value
        self.gpu_duration = self.get_parameter('gpu_duration').value

        # COMMUNICATION EXPERIMENT PARAMETERS
        self.declare_parameter('lidar_queue', 1)
        self.declare_parameter('image_queue', 1)

        self.lidar_queue = self.get_parameter('lidar_queue').value
        self.image_queue = self.get_parameter('image_queue').value

        # Define a custom QoS profile
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.cam_keys = [
            'CAM_FRONT', 
            # 'CAM_FRONT_LEFT', 
            # 'CAM_FRONT_RIGHT',
            # 'CAM_BACK', 
            # 'CAM_BACK_LEFT', 
            # 'CAM_BACK_RIGHT'
        ]

        # PUBLISHER && SUBSCRIBER
        self.lidar_publisher = self.create_publisher(PointCloud2, 'lidar_data', lidar_qos)
        self.image_publishers = {
            cam: self.create_publisher(CompressedImage, f'image_data/{cam}', image_qos)
            for cam in self.cam_keys
        }
        self.terminate_publisher = self.create_publisher(String, 'terminate_inferencers', 10)
        self.create_subscription(String, 'inferencer_ready', self.inferencer_ready_callback, 10)
        self.image_timers = {}

        # DATA LOADING SECTION
        self.scene = nusc.get('scene', scene_token)
        self.lidar_tokens = load_sweep_sd(nusc, self.scene, 'LIDAR_TOP')
        self.lidar_files = get_paths_from_sd(nusc, self.lidar_tokens)

        self.image_tokens = {
            cam: load_sweep_sd(nusc, self.scene, cam)
            for cam in self.cam_keys
        }

        self.image_files = {
            cam: get_paths_from_sd(nusc, self.image_tokens[cam])
            for cam in self.cam_keys
        }


        self.len_lidar_msgs = len(self.lidar_files)
        self.len_image_msgs = len(self.image_files)

        self.lidar_index = 0
        self.image_index = 0
        self.start_time = None
        self.models_ready_count = 0

        # GPU PROFILING
        self.profiler = pPerf('', None, 0, GPU_monitoring=True)

        self.lidar_data = []
        self.image_data = {cam: [] for cam in self.cam_keys}
        self.image_indices = {cam: 0 for cam in self.cam_keys}

        self.pub_lidar_count = 0
        self.pub_image_count = 0
        self.preloading_done = False
        self.preload_all_data()

    
    def preload_all_data(self):
        for i in range(self.len_lidar_msgs):
            try:
                path = self.lidar_files[i]
                token = self.lidar_tokens[i]
                if 'kitti' in self.lidar_model_mode:
                    points = self.convert_to_kitti(nusc, token)
                elif self.lidar_model_mode == 'nus':
                    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
                else:
                    raise ValueError("Unsupported lidar_model_mode")

                self.lidar_data.append((token, points))

            except Exception as e:
                self.get_logger().error(f"Failed to load LIDAR {path}: {e}")

        for cam in self.cam_keys:
                for i in range(len(self.image_files[cam])):
                    try:
                        path = self.image_files[cam][i]
                        token = self.image_tokens[cam][i]
                        img = cv2.imread(path)
                        if img is None:
                            raise ValueError("cv2.imread returned None")

                        self.image_data[cam].append((token, img))

                    except Exception as e:
                        self.get_logger().error(f"Failed to load IMAGE {path} from {cam}: {e}")
        
        self.preloading_done = True

    def inferencer_ready_callback(self, msg):
        try:
            self.models_ready_count += int(msg.data)
            self.get_logger().info(f"Received inferencer readiness. Count: {self.models_ready_count}/{self.expected_models}")

            if self.models_ready_count == self.expected_models:
                while not self.preloading_done:
                    self.get_logger().info("waiting for preloading")
                self.start_publishing()

        except ValueError:
            self.get_logger().warn("Received invalid readiness message.")

    def start_publishing(self):
        self.get_logger().info("All expected models are ready. Starting data publishing.")
        self.lidar_timer = self.create_timer(1.0 / self.publish_freq_lidar, self.publish_lidar)
        for cam in self.cam_keys:
            timer = self.create_timer(1.0 / self.publish_freq_image, lambda c=cam: self.publish_image(c))
            self.image_timers[cam] = timer

        self.profiler.start_gpu_monitoring()
        self.profiler.start_cpu_ram_monitoring()


    def publish_lidar(self):
        if self.lidar_index >= self.len_lidar_msgs:
            self.shutdown()
            return

        self.pub_lidar_count += 1
        
        token, points = self.lidar_data[self.lidar_index]
        
        torch.cuda.nvtx.range_push(f"{token}.lidar_encode")
        msg = PointCloud2()
        msg.header.frame_id = token
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_dense = True
        msg.is_bigendian = False

        itemsize = np.dtype(np.float32).itemsize
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=1 * itemsize, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=2 * itemsize, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=3 * itemsize, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring', offset=4 * itemsize, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = itemsize * len(msg.fields)
        msg.row_step = msg.point_step * points.shape[0]
        msg.data = points.tobytes()
        torch.cuda.nvtx.range_pop()


        self.lidar_publisher.publish(msg)
        self.lidar_index += 1


    def publish_image(self, cam):
        idx = self.image_indices[cam]
        if idx >= len(self.image_data[cam]):
            self.get_logger().info(f"{cam} finished publishing.")
            self.destroy_timer(self.image_timers[cam])
            return

        token, img = self.image_data[cam][idx]
        torch.cuda.nvtx.range_push(f"{token}.{cam}.image_encode")

        msg = CompressedImage()
        msg.header.frame_id = token
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()

        torch.cuda.nvtx.range_pop()

        self.image_publishers[cam].publish(msg)
        self.image_indices[cam] += 1
        


    def shutdown(self):
        self.get_logger().info("Sensor Publisher shutting down............")

        print(f'pub lidar count: {self.pub_lidar_count}')
        print(f'pub image count: {self.pub_image_count}')
        terminate_msg = String()
        terminate_msg.data = "TERMINATE"
        self.terminate_publisher.publish(terminate_msg)

        self.profiler.stop_gpu_monitoring()
        self.profiler.stop_cpu_ram_monitoring()
        pd.DataFrame(self.profiler.gpu_stats).to_csv(f"{self.data_dir}/gpu_{self.index}.csv", index=False)
        pd.DataFrame(self.profiler.cpu_stats).to_csv(f"{self.data_dir}/cpu_{self.index}.csv", index=False)

        self.destroy_timer(self.lidar_timer)
        for timer in self.image_timers.values():
            self.destroy_timer(timer)
        self.destroy_node()

        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = SensorPublisherNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info('Done')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
