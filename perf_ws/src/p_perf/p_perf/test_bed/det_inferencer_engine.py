#!/usr/bin/python3

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
import message_filters
import csv

import time

import numpy as np
import torch

import warnings
import pandas as pd
from filelock import FileLock
import os
import json
import cv2

from p_perf.pPerf import pPerf
from p_perf.pPerf_inferencer import pPerf3dDetInferencer, pPerf2dDetInferencer

from p_perf.general_utils import convert_to_kitti_ros

from LUT.engine_selector import EngineSelector

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")

WARM_PCD_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_PCD = np.fromfile(WARM_PCD_PATH, dtype=np.float32).reshape(-1, 5)
WARM_IMAGE_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'
WARM_IMAGE = cv2.imread(WARM_IMAGE_PATH)
if WARM_IMAGE is None:
    raise FileNotFoundError(f"Image not found: {WARM_IMAGE_PATH}")

class DetInferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # INPUT MODE
        self.declare_parameter('input_type', 'publisher')
        self.declare_parameter('lidar_model_mode', 'nus')

        self.input_type = self.get_parameter('input_type').value  
        self.lidar_model_mode = self.get_parameter('lidar_model_mode').value  

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

        self.index = self.get_parameter('index').value
        self.data_dir = self.get_parameter('data_dir').value

        # COMMUNICATION EXPERIMENT PARAMETERS
        self.declare_parameter('lidar_queue', 1)
        self.declare_parameter('image_queue', 1)

        self.lidar_queue = self.get_parameter('lidar_queue').value
        self.image_queue = self.get_parameter('image_queue').value

        # MESSAGE FILTER PARAMETERS
        self.declare_parameter('use_message_filters', True)
        self.declare_parameter('sync_queue_size', 3)
        self.declare_parameter('sync_slop', 0.05)

        self.use_message_filters = self.get_parameter('use_message_filters').value
        self.sync_queue_size = self.get_parameter('sync_queue_size').value
        self.sync_slop = self.get_parameter('sync_slop').value

        # Define a custom QoS profile
        self.lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.lidar_queue
        )

        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.image_queue
        )

        # INTIALIZATION OF LOCAL DATA
        self.sub_lidar_count = 0
        self.sub_image_count = 0

        # ENGINE SELECTION PARAMETERS
        self.declare_parameter('use_engine_selector', False)
        self.declare_parameter('engine_config_map', '{}')  # JSON string of engine map
        
        self.use_engine_selector = self.get_parameter('use_engine_selector').value
        self.engine_config_map_str = self.get_parameter('engine_config_map').value
        self.engine_selector = None
        
        # Create engine selector if enabled
        if self.use_engine_selector:
            self.engine_selector = self._create_engine_selector(self.engine_config_map_str)

        # DELAY LOGGING
        self.declare_parameter('logging_delay', True)
        self.logging_delay = self.get_parameter('logging_delay').value
        self.dets = []
        self.delay_log = []

        if self.logging_delay:
            self.delay_csv = os.path.join(self.data_dir, f"delays_{self.index}_det.csv")

            with open(self.delay_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_stamp', 'input_token', 'comm_delay', 'decode_delay', 'inference_delay', 'e2e_delay', 'model_name', 'model_type'])

        # INIT OF MODELS BASED ON MODE
        self.ready_publisher = self.create_publisher(String, 'inferencer_ready', 10)
        
        # PID announcement publisher for resource manager
        self.pid_publisher = self.create_publisher(String, 'pid_announcement', 10)
        
        # Flag to track if PID has been announced
        self.pid_announced = False
        
        # Output publishers
        self._init_output_publishers()

        # Initialize models and subscribers
        self._init_models()
        self._init_subscribers()
        self._init_warmup()
        self._publish_ready_message()

        # self.profiler.summary()

    def _init_models(self):
        """Initialize models - can be overridden by child classes"""
        if self.mode == 'lidar':
            self.inferencer = pPerf3dDetInferencer(self.model_name)
            self.inferencer.show_progress = False

        elif self.mode == 'image':
            self.inferencer = pPerf2dDetInferencer(self.model_name)
            self.inferencer.show_progress = False

    def _init_output_publishers(self):
        """Initialize output publishers - can be overridden by child classes."""
        self.lidar_output_topic = '/lidar_bbox'
        self.image_output_topic = '/image_bbox'
        self.lidar_output_pub = self.create_publisher(String, self.lidar_output_topic, 10)
        self.image_output_pub = self.create_publisher(String, self.image_output_topic, 10)

    def _publish_output(self, det, frame_stamp, is_lidar: bool):
        """Publish raw detection output with sensor-frame timestamp."""
        msg = String()
        msg.data = json.dumps({'stamp': frame_stamp, 'data': str(det)})
        if is_lidar:
            self.lidar_output_pub.publish(msg)
        else:
            self.image_output_pub.publish(msg)

    def _init_subscribers(self):
        """Initialize subscribers - can be overridden by child classes"""
        if self.use_message_filters:
            self._init_message_filters('lidar_data', 'image_data/CAM_FRONT')
        else:
            if self.mode == 'lidar':
                self.subscriber = self.create_subscription(PointCloud2, 'lidar_data', self.lidar_callback, self.lidar_qos)
            elif self.mode == 'image':
                self.subscriber = self.create_subscription(CompressedImage, 'image_data/CAM_FRONT', self.image_callback, self.image_qos)

        # Subscribe to termination signals
        self.create_subscription(String, 'terminate_inferencers', self._terminate_callback, 5)
        
        # Subscribe to targeted termination signal
        self.create_subscription(String, 'terminate_inferencer', self._terminate_targeted_callback, 10)

    def _init_message_filters(self, lidar_topic: str, image_topic: str):
        """Initialize message filters for time-aligned lidar + image inputs."""
        self.lidar_sub = message_filters.Subscriber(
            self,
            PointCloud2,
            lidar_topic,
            qos_profile=self.lidar_qos
        )
        self.image_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            image_topic,
            qos_profile=self.image_qos
        )
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.image_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False
        )
        self.time_sync.registerCallback(self.synced_callback)

    def synced_callback(self, lidar_msg, image_msg):
        """Handle time-aligned lidar + image messages."""
        if self.mode == 'lidar':
            self.lidar_callback(lidar_msg)
        elif self.mode == 'image':
            self.image_callback(image_msg)

    def _init_warmup(self):
        """Initialize warmup - can be overridden by child classes"""
        # WARMUP
        self.profiler = pPerf(
            self.model_name, 
            self.inferencer, 
            self.depth, 
            self.mode,
            engine_selector=self.engine_selector
        )
        warm_data = dict(points=WARM_PCD) if self.mode == 'lidar' else dict(img=WARM_IMAGE)

        self.profiler.warm_up(warm_data)
        self.profiler.register_hooks(warm_data)
        

    def _publish_ready_message(self):
        """Publish ready message - can be overridden by child classes"""
        # INFERENCER READY MSG FOR SENSOR PUBLISHER
        self.get_logger().info(f"{self.mode.capitalize()} model '{self.model_name}' is ready.")
        msg = String()
        msg.data = "1"
        self.ready_publisher.publish(msg)

    def _create_engine_selector(self, engine_map_str: str) -> EngineSelector:
        """
        Create EngineSelector from engine configuration parameters.
        
        Args:
            engine_map_str: JSON string of engine map, e.g., '{"1": 0, "2": 1, "3": 0}'
            
        Returns:
            EngineSelector instance or None if configuration is invalid
        """
        try:
            # Parse engine map from JSON string
            engine_map = json.loads(engine_map_str) if engine_map_str else {}
            
            self.get_logger().info(
                f"Created engine selector: {len(engine_map)} mappings"
            )
            
            if engine_map:
                self.get_logger().info(f"Engine map: {engine_map}")
            
            return EngineSelector(engine_map)
            
        except json.JSONDecodeError as e:
            self.get_logger().error(
                f"Invalid JSON in engine_config_map parameter: {e}. "
                f"Using default PyTorch execution."
            )
            return None
        except Exception as e:
            self.get_logger().error(
                f"Error creating engine selector: {e}. "
                f"Using default PyTorch execution."
            )
            return None

    def _announce_pid(self):
        """Announce the PID of the inferencer to the resource manager."""
        pid = os.getpid()
        announcement = {
            'pid': pid,
            'node_name': f'{self.mode}_inferencer_{self.index}',
            'model_name': self.model_name,
            'mode': self.mode
        }
        
        msg = String()
        msg.data = json.dumps(announcement)
        self.pid_publisher.publish(msg)
        self.get_logger().info(f"PID {pid} announced for {self.mode} model '{self.model_name}'")


    def lidar_callback(self, msg):
        recv_time_sim = self.get_clock().now().nanoseconds / 1e9
        recv_time = time.time()
        frame_id = msg.header.frame_id
        token = frame_id if self.input_type == "publisher" else msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sub_lidar_count += 1

        torch.cuda.nvtx.range_push(f"{token}.lidar_decode")
        input_data = self._decode_lidar(msg, self.lidar_model_mode)
        torch.cuda.nvtx.range_pop()

        decode_time = time.time()

        # Inference
        process_start = time.time()
        det = self.profiler.run_inference(input_data, token)
        process_end = time.time()
        
        self._publish_output(det, sent_time, is_lidar=True)

        # Process delay logging
        comm_delay = recv_time_sim - sent_time
        decode_delay = decode_time - recv_time
        inference_delay = process_end - process_start
        e2e_delay = inference_delay + decode_delay + abs(comm_delay)

        # Announce PID if not already announced
        if not self.pid_announced:
            self._announce_pid()
            self.pid_announced = True

    def image_callback(self, msg):
        recv_time_sim = self.get_clock().now().nanoseconds / 1e9
        recv_time = time.time()
        frame_id = msg.header.frame_id
        token = frame_id if self.input_type == "publisher" else msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sub_image_count += 1

        # Decode
        torch.cuda.nvtx.range_push(f"{token}.image_decode")
        input_data = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(input_data, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().error("Error decoding image during processing")
            return
        input_data = img
        torch.cuda.nvtx.range_pop()

        decode_time = time.time()

        # Inference
        process_start = time.time()
        det = self.profiler.run_inference(dict(img=input_data), token)
        process_end = time.time()
        
        self._publish_output(det, sent_time, is_lidar=False)

        # Process delay logging
        comm_delay = recv_time_sim - sent_time
        decode_delay = decode_time - recv_time
        inference_delay = process_end - process_start
        e2e_delay = inference_delay + decode_delay + abs(comm_delay)

        # Announce PID if not already announced
        if not self.pid_announced:
            self._announce_pid()
            self.pid_announced = True

    def _decode_lidar(self, msg, mode='nus'):
        # Each point consists of 5 float32 values: x, y, z, intensity, ring
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('ring', np.float32)
        ])

        # Read binary buffer
        points_array = np.frombuffer(msg.data, dtype=dtype)

        if mode == 'nus':
            points = np.stack(
                [points_array['x'], points_array['y'], points_array['z'],
                points_array['intensity'], points_array['ring']],
                axis=-1
            )
        elif 'kitti' in mode:
            # First convert to KITTI format
            kitti_points = convert_to_kitti_ros(points_array)
            # Then stack the points
            points = np.stack(
                [kitti_points['x'], kitti_points['y'], kitti_points['z'],
                kitti_points['intensity']],
                axis=-1
            )
        else:
            raise ValueError(f"Unsupported lidar model mode: {mode}")

        return dict(points=points.astype(np.float32))

    def _terminate_callback(self, msg):
        """Handle broadcast termination signal."""
        if msg.data.strip() == "TERMINATE":
            self.get_logger().info(f"{self.mode} Inferencer shutting down.................")
            self.destroy_node()
            raise SystemExit
    
    def _terminate_targeted_callback(self, msg):
        """Handle targeted termination signal with PID specification."""
        try:
            data = json.loads(msg.data)
            target_pid = data.get('pid')
            my_pid = os.getpid()
            
            if target_pid == my_pid:
                self.get_logger().info(f"{self.mode} Inferencer (PID {my_pid}) received termination request")
                self.destroy_node()
                raise SystemExit
                
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f"Invalid termination message: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in targeted termination: {e}")
    

def main(args=None):
    rclpy.init(args=args)
    node = DetInferenceNode()
    try:
        rclpy.spin(node)
    except SystemExit:                 
        rclpy.logging.get_logger("Quitting").info('Done')
    rclpy.shutdown()

if __name__ == '__main__':
    main()