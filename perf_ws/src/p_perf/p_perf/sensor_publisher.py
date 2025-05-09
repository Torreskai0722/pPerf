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
from p_perf.pPerf import pPerf
from p_perf.utils import list_filenames
import pandas as pd
import json

LIDAR_DIR = '/mmdetection3d_ros2/data/nuscenes/sweeps/LIDAR_TOP'
IMAGE_DIR = '/mmdetection3d_ros2/data/nuscenes/sweeps/CAM_FRONT'

class SensorPublisherNode(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # RUNNING REQUIREMENTS
        self.declare_parameter('expected_models', 2)
        self.declare_parameter('index', 0)
        self.declare_parameter('data_dir', '')

        self.expected_models = self.get_parameter('expected_models').value
        self.index = self.get_parameter('index').value
        self.data_dir = self.get_parameter('data_dir').value
        json_path = '/mmdetection3d_ros2/data/nuscenes/v1.0-mini/sample_data.json'
        self.filename_to_token = self.load_filename_to_token_map_from_json(json_path)

        # BASIC EXPERIMENT PARAMETERS
        self.declare_parameter('publish_freq_lidar', 20)
        self.declare_parameter('publish_freq_image', 12)
        self.declare_parameter('run_time', 30)
        self.declare_parameter('gpu_duration', 0.05)

        self.publish_freq_lidar = self.get_parameter('publish_freq_lidar').value
        self.publish_freq_image = self.get_parameter('publish_freq_image').value
        self.run_time = self.get_parameter('run_time').value
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

        # PUBLISHER && SUBSCRIBER
        self.lidar_publisher = self.create_publisher(PointCloud2, 'lidar_data', lidar_qos)
        self.image_publisher = self.create_publisher(CompressedImage, 'image_data', image_qos)
        self.terminate_publisher = self.create_publisher(String, 'terminate_inferencers', 10)
        self.create_subscription(String, 'inferencer_ready', self.inferencer_ready_callback, 10)

        # DATA LOADING SECTION
        self.lidar_files = list_filenames(LIDAR_DIR, 'bin')
        self.image_files = list_filenames(IMAGE_DIR, 'jpg')

        self.max_lidar_msgs = min(len(self.lidar_files), int(self.run_time * self.publish_freq_lidar))
        self.max_image_msgs = min(len(self.image_files), int(self.run_time * self.publish_freq_image))

        self.lidar_index = 0
        self.image_index = 0
        self.start_time = None
        self.models_ready_count = 0

        self.profiler = pPerf('', 0, self.gpu_duration)

        self.lidar_data = []
        self.image_data = []

        self.pub_lidar_count = 0
        self.pub_image_count = 0
        self.preloading_done = False
        self.preload_all_data()


    def load_filename_to_token_map_from_json(self, json_path):
        with open(json_path, 'r') as f:
            sample_data = json.load(f)
        mapping = {}
        for item in sample_data:
            filename = os.path.basename(item['filename'])  # Extract only the filename, e.g., 'n015...bin'
            mapping[filename] = item['token']
        return mapping


    def preload_all_data(self):
        for i in range(self.max_lidar_msgs):
            try:
                path = self.lidar_files[i]
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
                filename = os.path.basename(path)
                token = self.filename_to_token.get(filename)

                dtype = np.float32
                itemsize = np.dtype(dtype).itemsize

                msg = PointCloud2()
                msg.header.frame_id = token
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.height = 1
                msg.width = points.shape[0]
                msg.is_dense = True
                msg.is_bigendian = False
                msg.fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=1*itemsize, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=2*itemsize, datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=3*itemsize, datatype=PointField.FLOAT32, count=1),
                    PointField(name='ring', offset=4*itemsize, datatype=PointField.FLOAT32, count=1)
                ]
                msg.point_step = itemsize * len(msg.fields)
                msg.row_step = msg.point_step * points.shape[0]
                msg.data = points.tobytes()

                self.lidar_data.append(msg)

            except Exception as e:
                self.get_logger().error(f"Failed to load LIDAR {path}: {e}")

        for i in range(self.max_image_msgs):
            try:
                path = self.image_files[i]
                img = cv2.imread(path)
                filename = os.path.basename(path)
                token = self.filename_to_token.get(filename)
                if img is None:
                    raise ValueError("cv2.imread returned None")

                msg = CompressedImage()
                msg.header.frame_id = token
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.format = "jpeg"  # or "png" depending on your preference
                msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
                self.image_data.append(msg)

            except Exception as e:
                self.get_logger().error(f"Failed to load IMAGE {path}: {e}")
        
        self.preloading_done = True


    def inferencer_ready_callback(self, msg):
        try:
            self.models_ready_count += int(msg.data)
            self.get_logger().info(f"Received inferencer readiness. Count: {self.models_ready_count}/{self.expected_models}")

            if self.models_ready_count == self.expected_models:
                while not self.preloading_done:
                    print("waiting for preloading")
                self.profiler.start_gpu_monitoring()
                self.start_publishing()

        except ValueError:
            self.get_logger().warn("Received invalid readiness message.")

    def start_publishing(self):
        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info("All expected models are ready. Starting data publishing.")

            self.lidar_timer = self.create_timer(1.0 / self.publish_freq_lidar, self.publish_lidar)
            self.image_timer = self.create_timer(1.0 / self.publish_freq_image, self.publish_image)

    def publish_lidar(self):
        if self.lidar_index >= self.max_lidar_msgs or time.time() - self.start_time > self.run_time:
            self.shutdown()
            return

        self.pub_lidar_count += 1

        msg, input_name = self.lidar_data[self.lidar_index]
        msg.header.frame_id = input_name
        msg.header.stamp = self.get_clock().now().to_msg()
        self.lidar_publisher.publish(msg)
        self.lidar_index += 1

    def publish_image(self):
        if self.image_index >= self.max_image_msgs or time.time() - self.start_time > self.run_time:
            self.shutdown()
            return
        
        self.pub_image_count += 1

        msg, input_name = self.image_data[self.image_index]
        msg.header.frame_id = input_name
        msg.header.stamp = self.get_clock().now().to_msg()
        self.image_publisher.publish(msg)
        self.image_index += 1


    def shutdown(self):
        self.get_logger().info("Sensor Publisher shutting down............")

        print(f'pub lidar count: {self.pub_lidar_count}')
        print(f'pub image count: {self.pub_image_count}')
        terminate_msg = String()
        terminate_msg.data = "TERMINATE"
        self.terminate_publisher.publish(terminate_msg)
        self.profiler.stop_gpu_monitoring()
        pd.DataFrame(self.profiler.gpu_stats).to_csv(f"{self.data_dir}/gpu_{self.index}.csv", index=False)

        self.destroy_timer(self.lidar_timer)
        self.destroy_timer(self.image_timer)
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
