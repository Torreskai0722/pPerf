#!/usr/bin/python3

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
import csv

import numpy as np
import torch
from mmdet3d.apis import LidarDet3DInferencer
from mmdet.apis import DetInferencer

import warnings
import pandas as pd
from filelock import FileLock
import os
import json
import cv2
import mmengine

from p_perf.pPerf import pPerf
from p_perf.post_process.lidar_eval import lidar_nusc_box_to_global, lidar_output_to_nusc_box
from p_perf.post_process.image_eval import image_output_to_coco, generate_coco_gt, change_pred_imageid
from p_perf.config.constant import lidar_classes, nusc


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
        self.declare_parameter('lidar_queue', 1)
        self.declare_parameter('image_queue', 1)

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
        self.latest_token = ''
        self.sub_lidar_count = 0
        self.sub_image_count = 0
        self.token_sent_time = {}
        self.token_recv_time = {}

        self.dets = []
        self.lidar_pred_json = os.path.join(self.data_dir, f"lidar_pred_{self.index}.json")
        self.image_pred_json = os.path.join(self.data_dir, f"image_pred_{self.index}.json")
        self.image_gt_json = os.path.join(self.data_dir, f"image_gt_{self.index}.json")
        self.delay_csv = os.path.join(self.data_dir, f"delays_{self.index}.csv")
        self.delay_log = []

        with open(self.delay_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input_token', 'sensor_type', 'comm_delay', 'decode_delay', 'process_delay', 'process_time'])

        # INIT OF MODELS BASED ON MODE
        self.ready_publisher = self.create_publisher(String, 'inferencer_ready', 10)

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
        self.create_subscription(String, 'terminate_inferencers', self._terminate_callback, 10)

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
        input_name = frame_id if self.input_type == "publisher" else msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        recv_time = self.get_clock().now().nanoseconds / 1e9

        self.latest_token = input_name
        self.latest_data = msg  # Store PointCloud2 message
        self.sub_lidar_count += 1

        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.token_sent_time[input_name] = sent_time
        self.token_recv_time[input_name] = recv_time
                

    def image_callback(self, msg):
        frame_id = msg.header.frame_id
        input_name = frame_id if self.input_type == "publisher" else msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        recv_time = self.get_clock().now().nanoseconds / 1e9

        self.latest_token = input_name
        self.latest_data = msg.data  # Store raw compressed data
        self.sub_image_count += 1

        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.token_sent_time[input_name] = sent_time
        self.token_recv_time[input_name] = recv_time


    def process_latest_data(self):
        if self.latest_data is None:
            return

        # DECODE BEFORE INFERENCE
        if self.mode == 'lidar':
            points = []
            for p in pc2.read_points(self.latest_data, field_names=['x', 'y', 'z', 'intensity', 'ring'], skip_nans=True):
                points.append([p[0], p[1], p[2], p[3], p[4]])
            input_data = dict(points=np.array(points, dtype=np.float32))
        else:
            np_arr = np.frombuffer(self.latest_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().error("Error decoding image during processing")
                self.latest_data = None
                self.latest_token = ''
                return
            input_data = img
        
        # COMM + DECODE DELAY LOGGING
        decode_time = self.get_clock().now().nanoseconds / 1e9
        self._log_delay(decode_time)

        # INFERENCE
        start_time = self.get_clock().now().nanoseconds / 1e9
        det = self.profiler.run_inference(input_data, self.latest_token)     
        end_time = self.get_clock().now().nanoseconds / 1e9

        # PROCESS DELAY LOGGING
        sent_time = self.token_sent_time.get(self.latest_token)
        if sent_time is not None:
            process_delay = end_time - sent_time
            self.delay_log.append({
                'input_token': self.latest_token,
                'sensor_type': self.mode,
                'process_delay': process_delay,
                'process_time': end_time - start_time
            })

        if self.mode == 'lidar':
            self.dets.append((self.latest_token, det['predictions'][0].pred_instances_3d))
        else:
            self.dets.append((self.latest_token, det['predictions'][0].pred_instances))

        # RESET
        self.latest_data = None
        self.latest_token = ''


    def _log_delay(self, decode_time):
        sent_time = self.token_sent_time.get(self.latest_token)
        recv_time = self.token_recv_time.get(self.latest_token)

        self.delay_log.append({
            'input_token': self.latest_token,
            'sensor_type': self.mode,
            'comm_delay': recv_time - sent_time,
            'decode_delay': decode_time - recv_time
        })


    def _save_lidar_detections(self):
        nusc_annos = {}
        for det in self.dets:
            token = det[0]
            boxes = lidar_output_to_nusc_box(det[1], token)
            boxes = lidar_nusc_box_to_global(nusc, token, boxes)

            annos = []
            for box in boxes:
                name = lidar_classes[box.label]
                nusc_anno = dict(
                    sample_token=token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name='')
                annos.append(nusc_anno)
            nusc_annos[token] = annos
        
        nusc_submission = {
            'meta': {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False
            },
            'results': nusc_annos
        }
        
        mmengine.dump(nusc_submission, self.lidar_pred_json)
        print(f"Results written to {self.lidar_pred_json}")


    def _save_image_detections(self):
        coco_predictions = []
        for det in self.dets:
            coco_pred = image_output_to_coco(det[1], det[0])
            coco_predictions.extend(coco_pred)
        
        with open(self.image_pred_json, 'w') as f:
            json.dump(coco_predictions, f, indent=2)


    def _terminate_callback(self, msg):
        if msg.data.strip() == "TERMINATE":
            self.get_logger().info(f"{self.mode} Inferencer shutting down.................")

            print(f'sub lidar count: {self.sub_lidar_count}')
            print(f'sub image count: {self.sub_image_count}')

            # save lidar detections
            if self.mode == 'lidar':
                self._save_lidar_detections()
            else:
                self._save_image_detections()

            # Shared delay file path
            lock_path = self.delay_csv + ".lock"

            # Append mode with locking
            with FileLock(lock_path):
                delay_df = pd.DataFrame(self.delay_log)
                delay_df = delay_df.groupby(['input_token', 'sensor_type'], as_index=False).agg({
                    'comm_delay': 'first',
                    'decode_delay': 'first',
                    'process_delay': 'first',
                    'process_time': 'first'
                })

                if os.path.exists(self.delay_csv):
                    delay_df.to_csv(self.delay_csv, mode='a', header=False, index=False)

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