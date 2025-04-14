#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from cv_bridge import CvBridge
from mmdet3d.apis import LidarDet3DInferencer
from mmdet.apis import DetInferencer
from pPerf_v1_3.pPerf import pPerf
import warnings
import pandas as pd
from filelock import FileLock
import os

warnings.filterwarnings("ignore")

WARM_PCD = '/mmdetection3d_ros2/pPerf_ws/src/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915245547336.pcd.bin'
WARM_IMAGE = '/mmdetection3d_ros2/pPerf_ws/src/cabc30fc-fd79926f.jpg'

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

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
        self.declare_parameter('lidar_queue', 10)
        self.declare_parameter('image_queue', 10)

        self.lidar_queue = self.get_parameter('lidar_queue').value
        self.image_queue = self.get_parameter('image_queue').value

        # INTIALIZATION OF LOCAL DATA
        self.latest_data = None
        self.latest_name = ''
        self.latest_critical = False

        self.hit_csv = f"{self.data_dir}/run_{self.index}.csv"
        self.hit_df = pd.read_csv(self.hit_csv)
        self.delay_csv = os.path.join(self.data_dir, f"delays_{self.index}.csv")

        self.delay_log = []
        self.ready_publisher = self.create_publisher(String, 'inferencer_ready', 10)

        # INIT OF MODELS BASED ON MODE
        if self.mode == 'lidar':
            self.subscriber = self.create_subscription(PointCloud2, 'lidar_data', self.data_callback, self.lidar_queue)
            self.inferencer = LidarDet3DInferencer(self.model_name)
            self.inferencer.show_progress = False
        else:
            self.subscriber = self.create_subscription(Image, 'image_data', self.data_callback, self.image_queue)
            self.inferencer = DetInferencer(self.model_name)
            self.inferencer.show_progress = False

        # CALLBACK TO PROCESS LATEST DAATA
        self.timer = self.create_timer(1.0 / self.sample_freq, self.process_latest_data)

        # Subscribe to termination signal
        self.create_subscription(String, 'terminate_inferencers', self.terminate_callback, 10)

        # WARMUP
        self.profiler = pPerf(self.model_name, self.depth)
        self.profiler.warm_up(self.inferencer, WARM_PCD if self.mode == 'lidar' else WARM_IMAGE, mode=self.mode)
        self.profiler.register_hooks(self.inferencer.model)

        # INFERENCER READY MSG FOR SENSOR PUBLISHER
        self.get_logger().info(f"{self.mode.capitalize()} model '{self.model_name}' is ready.")
        msg = String()
        msg.data = "1"
        self.ready_publisher.publish(msg)

    def data_callback(self, msg):
        """Decode and store the latest sensor data."""
        frame_id = msg.header.frame_id
        parts = frame_id.split('|')
        input_name = parts[1]
        mode = parts[0]
 
        self.latest_name = input_name
        self.latest_critical = (parts[2] == "critical")

        self.latest_name = input_name

        if self.mode == 'lidar':
            # Ensure 5 values per point: x, y, z, intensity, ringfield
            points = []
            for p in pc2.read_points(msg, field_names=None, skip_nans=True):
                x = p[0]
                y = p[1]
                z = p[2]
                intensity = p[3] if len(p) > 3 else 0.0
                ringfield = p[4] if len(p) > 4 else 0.0
                points.append([x, y, z, intensity, ringfield])

            points = np.array(points, dtype=np.float32)
            self.latest_data = dict(points=points)

        else:
            bridge = CvBridge()
            try:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_data = img
            except Exception as e:
                self.get_logger().error(f"Error converting image: {e}")
                self.latest_data = None
        
        # Communication delay calculation
        recv_time = self.get_clock().now().nanoseconds / 1e9  # Current node time in seconds
        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9  # Time from header
        
        print(recv_time, flush=True)
        print(sent_time, flush=True)

        delay = recv_time - sent_time
        self.delay_log.append({
            'frame': input_name,
            'sensor_type': mode,
            'critical': self.latest_critical,
            'delay': delay
        })

    def process_latest_data(self):
        """Processes the latest received data at a controlled rate."""
        if self.latest_data is None:
            return  # No new data received yet, skip processing
        
        self.profiler.run_inference(self.inferencer, self.latest_data, self.latest_name)
        self.latest_data = None

        if self.latest_critical:
            lock_path = self.hit_csv + ".lock"
            with FileLock(lock_path):
                hit_df = pd.read_csv(self.hit_csv)
                mask = (hit_df['frame'] == self.latest_name) & (hit_df['sensor_type'] == self.mode)
                if mask.any():
                    hit_df.loc[mask, 'hit status'] = True
                    hit_df.to_csv(self.hit_csv, index=False)
            

    def terminate_callback(self, msg):
        if msg.data.strip() == "TERMINATE":
            self.get_logger().info(f"{self.mode} Inferencer shutting down.................")

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