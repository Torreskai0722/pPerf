#!/usr/bin/python3

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField, Image
import numpy as np
from cv_bridge import CvBridge
import cv2
import os
import time
from pPerf_v1_3.pPerf import pPerf
from pPerf_v1_3.utils import list_filenames
import pandas as pd
import random

LIDAR_DIR = '/mmdetection3d_ros2/dataset/nuscenes/v1.0-mini/sweeps/LIDAR_TOP'
IMAGE_DIR = '/mmdetection3d_ros2/dataset/nuscenes/v1.0-mini/sweeps/CAM_FRONT'

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

        # BASIC EXPERIMENT PARAMETERS
        self.declare_parameter('publish_freq_lidar', 20)
        self.declare_parameter('publish_freq_image', 12)
        self.declare_parameter('run_time', 30)
        self.declare_parameter('gpu_duration', 0.05)

        self.publish_freq_lidar = self.get_parameter('publish_freq_lidar').value
        self.publish_freq_image = self.get_parameter('publish_freq_image').value
        self.run_time = self.get_parameter('run_time').value
        self.gpu_duration = self.get_parameter('gpu_duration').value

        # HITRATE EXPERIMENT PARAMETER
        self.declare_parameter('critical_duration', 0.5)
        self.declare_parameter('critical_count', 3)

        self.critical_duration = self.get_parameter('critical_duration').value
        self.critical_count = self.get_parameter('critical_count').value

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
        self.image_publisher = self.create_publisher(Image, 'image_data', image_qos)
        self.terminate_publisher = self.create_publisher(String, 'terminate_inferencers', 10)
        self.create_subscription(String, 'inferencer_ready', self.inferencer_ready_callback, 10)

        # DATA LOADING SECTION
        self.lidar_files = list_filenames(LIDAR_DIR, 'bin')
        self.image_files = list_filenames(IMAGE_DIR, 'jpg')

        self.max_lidar_msgs = min(len(self.lidar_files), int(self.run_time * self.publish_freq_lidar))
        self.max_image_msgs = min(len(self.image_files), int(self.run_time * self.publish_freq_image))

        self.select_random_intervals(interval_duration=self.critical_duration, count=self.critical_count)

        self.lidar_index = 0
        self.image_index = 0
        self.start_time = None
        self.models_ready_count = 0

        self.bridge = CvBridge()
        self.profiler = pPerf('', 0, self.gpu_duration)

        self.lidar_data = []
        self.image_data = []

        self.pub_lidar_count = 0
        self.pub_image_count = 0
        self.preloading_done = False
        self.preload_all_data()



    def preload_all_data(self):
        for i in range(self.max_lidar_msgs):
            try:
                path = self.lidar_files[i]
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]
                input_name = os.path.basename(path).split('.')[0]

                dtype = np.float32
                itemsize = np.dtype(dtype).itemsize

                msg = PointCloud2()
                msg.header.frame_id = f"lidar_frame|{input_name}|unknown"
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.height = 1
                msg.width = points.shape[0]
                msg.is_dense = True
                msg.is_bigendian = False
                msg.point_step = itemsize * 4
                msg.row_step = msg.point_step * points.shape[0]
                msg.fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=1*itemsize, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=2*itemsize, datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=3*itemsize, datatype=PointField.FLOAT32, count=1)
                ]
                msg.data = points.tobytes()

                self.lidar_data.append((msg, input_name))

            except Exception as e:
                self.get_logger().error(f"Failed to load LIDAR {path}: {e}")

        for i in range(self.max_image_msgs):
            try:
                path = self.image_files[i]
                img = cv2.imread(path)
                if img is None:
                    raise ValueError("cv2.imread returned None")

                input_name = os.path.basename(path).split('.')[0]
                msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                msg.header.frame_id = f"camera_frame|{input_name}|unknown"
                msg.header.stamp = self.get_clock().now().to_msg()

                self.image_data.append((msg, input_name))

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
            df = pd.read_csv(f"{self.data_dir}/run_{self.index}.csv")
            self.critical_frames = set(df['frame'].astype(str).tolist())

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
        is_critical = input_name in self.critical_frames
        msg.header.frame_id = f"lidar_frame|{input_name}|{'critical' if is_critical else 'normal'}"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.lidar_publisher.publish(msg)
        self.lidar_index += 1

    def publish_image(self):
        if self.image_index >= self.max_image_msgs or time.time() - self.start_time > self.run_time:
            self.shutdown()
            return
        
        self.pub_image_count += 1

        msg, input_name = self.image_data[self.image_index]
        is_critical = input_name in self.critical_frames
        msg.header.frame_id = f"camera_frame|{input_name}|{'critical' if is_critical else 'normal'}"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.image_publisher.publish(msg)
        self.image_index += 1


    def select_random_intervals(self, interval_duration=0.5, count=3):
        max_time_lidar = self.max_lidar_msgs / self.publish_freq_lidar
        max_time_image = self.max_image_msgs / self.publish_freq_image
        max_time = min(max_time_lidar, max_time_image)

        lidar_frames_per_interval = int(interval_duration * self.publish_freq_lidar)
        image_frames_per_interval = int(interval_duration * self.publish_freq_image)

        max_start_time = max_time - interval_duration
        intervals = [random.uniform(0, max_start_time) for _ in range(count)]
        intervals.sort()

        records = []

        for t_start in intervals:
            t_end = t_start + interval_duration

            lidar_start_idx = int(t_start * self.publish_freq_lidar)
            image_start_idx = int(t_start * self.publish_freq_image)

            lidar_end_idx = min(lidar_start_idx + lidar_frames_per_interval, self.max_lidar_msgs)
            image_end_idx = min(image_start_idx + image_frames_per_interval, self.max_image_msgs)

            lidar_names = [os.path.basename(f).split('.')[0] for f in self.lidar_files[lidar_start_idx:lidar_end_idx]]
            image_names = [os.path.basename(f).split('.')[0] for f in self.image_files[image_start_idx:image_end_idx]]

            for lidar_name in lidar_names:
                records.append({
                    "start_time": round(t_start, 4),
                    "end_time": round(t_end, 4),
                    "sensor_type": "lidar",
                    "frame": lidar_name,
                    "paired_frames": image_names,
                    "hit status": False
                })

            for image_name in image_names:
                records.append({
                    "start_time": round(t_start, 4),
                    "end_time": round(t_end, 4),
                    "sensor_type": "image",
                    "frame": image_name,
                    "paired_frames": lidar_names,
                    "hit status": False
                })

        output_path = os.path.join(self.data_dir, f"run_{self.index}.csv")
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        self.get_logger().info(f"Saved {len(records)} entries to {output_path} using index-based interval selection.")

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
