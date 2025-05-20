#!/usr/bin/python3

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from cv_bridge import CvBridge
from mmdet3d.apis import LidarDet3DInferencer
from mmdet.apis import DetInferencer
from p_perf.pPerf import pPerf
import warnings
import pandas as pd
from filelock import FileLock
import os
import cv2

warnings.filterwarnings("ignore")

WARM_PCD = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_IMAGE = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # INPUT MODE
        self.declare_parameter('input_type', 'publisher')
        self.input_type = self.get_parameter('input_type').value   

        # MODEL RELATED PARAMETERS
        self.declare_parameter('mode', 'lidar')
        self.declare_parameter('model_name', '')
        self.declare_parameter('depth', 0)

        self.mode = self.get_parameter('mode').value.lower()
        self.model_name = self.get_parameter('model_name').value
        self.depth = self.get_parameter('depth').value   

        # BASIC PARAMETER FOR THE INTERENCER
        self.declare_parameter('index', 0)
        self.declare_parameter('data_dir', None)
        self.declare_parameter('sample_freq', 5)

        self.sample_freq = self.get_parameter('sample_freq').value
        self.index = self.get_parameter('index').value
        self.data_dir = self.get_parameter('data_dir').value

        # COMMUNICATION EXPERIMENT PARAMETERS
        self.declare_parameter('lidar_queue', 5)
        self.declare_parameter('image_queue', 5)

        self.lidar_queue = self.get_parameter('lidar_queue').value
        self.image_queue = self.get_parameter('image_queue').value

        # Define a custom QoS profile
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.lidar_queue
        )

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.image_queue
        )

        # INTIALIZATION OF LOCAL DATA
        self.latest_data = None
        self.latest_name = ''
        self.sub_lidar_count = 0
        self.sub_image_count = 0

        self.delay_csv = os.path.join(self.data_dir, f"delays_{self.index}.csv")
        self.delay_log = []

        self.ready_publisher = self.create_publisher(String, 'inferencer_ready', 10)

        # INIT OF MODELS BASED ON MODE
        if self.mode == 'lidar':
            self.subscriber = self.create_subscription(PointCloud2, 'lidar_data', self.lidar_callback, lidar_qos)
            self.inferencer = LidarDet3DInferencer(self.model_name)
        else:
            self.subscriber = self.create_subscription(CompressedImage, 'image_data', self.image_callback, image_qos)
            self.inferencer = DetInferencer(self.model_name)
        self.inferencer.show_progress = False

        # CALLBACK TO PROCESS LATEST DAATA
        self.timer = self.create_timer(1.0 / self.sample_freq, self.process_latest_data)

        # Subscribe to termination signal
        self.create_subscription(String, 'terminate_inferencers', self.terminate_callback, 10)

        # WARMUP
        self.profiler = pPerf(self.model_name, self.inferencer, self.depth)
        if self.mode == 'lidar':
            warm_data = dict(points=WARM_PCD)
        else:
            warm_data = WARM_IMAGE
        self.profiler.warm_up(warm_data)
        self.profiler.register_hooks(warm_data)

        # INFERENCER READY MSG FOR SENSOR PUBLISHER
        self.get_logger().info(f"{self.mode.capitalize()} model '{self.model_name}' is ready.")
        msg = String()
        msg.data = "1"
        self.ready_publisher.publish(msg)

    def lidar_callback(self, msg):
        frame_id = msg.header.frame_id
        if self.input_type == "publisher":
            input_name = frame_id
        else:
            input_name = frame_id.split('/')[-1]

        self.latest_name = input_name

        points = []
        for p in pc2.read_points(msg, field_names=['x', 'y', 'z', 'intensity', 'ring'], skip_nans=True):
            x = p[0]
            y = p[1]
            z = p[2]
            intensity = p[3]
            ring = p[4]
            points.append([x, y, z, intensity, ring])
        
        self.latest_data = dict(points=np.array(points, dtype=np.float32))
        self.sub_lidar_count += 1

        if self.input_type == 'publisher':
            self._log_delay(msg, input_name)

    def image_callback(self, msg):
        frame_id = msg.header.frame_id
        if self.input_type == "publisher":
            input_name = frame_id
        else:
            input_name = frame_id.split('/')[-1]

        self.latest_name = input_name

        try:
            # Decode JPEG or PNG encoded image bytes to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode compressed image")

            self.latest_data = img
            self.sub_image_count += 1
        except Exception as e:
            self.get_logger().error(f"Error decoding compressed image: {e}")
            self.latest_data = None
            return

        if self.input_type == 'publisher':
            self._log_delay(msg, input_name)


    def _log_delay(self, msg, input_name):
        recv_time = self.get_clock().now().nanoseconds / 1e9
        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.delay_log.append({
            'frame': input_name,
            'sensor_type': self.mode,
            'delay': recv_time - sent_time
        })

    def process_latest_data(self):
        """Processes the latest received data at a controlled rate."""
        if self.latest_data is None:
            return  # No new data received yet, skip processing
        
        self.profiler.run_inference(self.latest_data, self.latest_name)
        self.latest_data = None
            

    def terminate_callback(self, msg):
        if msg.data.strip() == "TERMINATE":
            self.get_logger().info(f"{self.mode} Inferencer shutting down.................")

            print(f'sub lidar count: {self.sub_lidar_count}')
            print(f'sub image count: {self.sub_image_count}')

            # Shared delay file path
            lock_path = self.delay_csv + ".lock"

            # Append mode with locking
            with FileLock(lock_path):
                delay_df = pd.DataFrame(self.delay_log)
                if os.path.exists(self.delay_csv):
                    delay_df.to_csv(self.delay_csv, mode='a', header=False, index=False)
                else:
                    delay_df.to_csv(self.delay_csv, mode='w', header=True, index=False)

            self.get_logger().info(f"Appended communication delays to {self.delay_csv}")

            self.destroy_timer(self.timer)
            self.destroy_node()
            raise SystemExit
    
def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except SystemExit:                 
        rclpy.logging.get_logger("Quitting").info('Done')
    rclpy.shutdown()

if __name__ == '__main__':
    main()