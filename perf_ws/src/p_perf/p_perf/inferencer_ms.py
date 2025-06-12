#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer

import os
import csv
import json
import time
import threading
import numpy as np
import pandas as pd
import torch
import cv2
import mmengine
import warnings
from queue import Queue
import concurrent.futures

from mmdet3d.apis import LidarDet3DInferencer
from mmdet.apis import DetInferencer

from p_perf.pPerf import pPerf
from p_perf.BEVInferencer import BEVFormerInferencer
from p_perf.utils import get_closest_token_from_timestamp, build_channel_timestamp_token_map
from p_perf.config.constant import nus_lidar_classes, kitti_lidar_classes
from p_perf.post_process.lidar_eval import lidar_output_to_nusc_box, lidar_nusc_box_to_global
from p_perf.post_process.image_eval import image_output_to_coco
from filelock import FileLock
from nuscenes.nuscenes import NuScenes

warnings.filterwarnings("ignore")

WARM_PCD = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_IMAGE = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'

DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)


class ModelRunner:
    def __init__(self, name, inferencer, mode, stream, depth):
        self.name = name
        self.inferencer = inferencer
        self.mode = mode
        self.stream = stream
        self.profiler = pPerf(name, inferencer, depth, mode)
        self._warm_up()

    def _warm_up(self):
        with torch.cuda.stream(self.stream):
            if self.mode == 'lidar':
                warm_data = dict(points=WARM_PCD)
            elif self.mode == 'image':
                warm_data = WARM_IMAGE
            else:
                return  # multi-modal will be warmed up in main
            self.profiler.warm_up(warm_data)
            self.profiler.register_hooks(warm_data)

    def run(self, data, token):
        with torch.cuda.stream(self.stream):
            return self.profiler.run_inference(data, token)


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # PARAMETERS
        self.declare_parameter('lidar_models_str', '')
        self.declare_parameter('image_models_str', '')
        self.declare_parameter('multi_modal_model', None)
        self.declare_parameter('depth', 0)
        self.declare_parameter('scene', '')
        self.declare_parameter('data_dir', None)
        self.declare_parameter('index', 0)
        self.declare_parameter('lidar_queue', 1)
        self.declare_parameter('image_queue', 1)
        self.declare_parameter('input_type', 'publisher')
        self.declare_parameter('lidar_model_mode', 'nus')
        self.declare_parameter('lidar_model_thresh', 0.5)

        # Get parameters
        lidar_models_str = self.get_parameter('lidar_models_str').value
        image_models_str = self.get_parameter('image_models_str').value
        self.lidar_models = [m.strip() for m in lidar_models_str.split(',')] if lidar_models_str else []
        self.image_models = [m.strip() for m in image_models_str.split(',')] if image_models_str else []
        self.mm_model = self.get_parameter('multi_modal_model').value
        self.depth = self.get_parameter('depth').value
        self.scene_token = self.get_parameter('scene').value
        self.data_dir = self.get_parameter('data_dir').value
        self.index = self.get_parameter('index').value
        self.lidar_queue = self.get_parameter('lidar_queue').value
        self.image_queue = self.get_parameter('image_queue').value
        self.input_type = self.get_parameter('input_type').value
        self.lidar_model_mode = self.get_parameter('lidar_model_mode').value
        self.lidar_model_thresh = self.get_parameter('lidar_model_thresh').value

        # Create thread pool
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.lidar_models) + len(self.image_models) + (1 if self.mm_model == 'bevfusion' else 0)
        )

        # Track active futures for each model
        self.active_futures = {model: None for model in self.lidar_models + self.image_models}
        if self.mm_model == 'bevfusion':
            self.active_futures[self.mm_model] = None

        # Initialize data structures
        self.runners = {}
        self.lidar_dets = []
        self.image_dets = []
        self.delay_log = []
        self.token_delays = {}
        self.sub_lidar_count = 0
        self.sub_image_count = 0
        self.models_loaded = 0
        self.expected_models = len(self.lidar_models) + len(self.image_models) + (1 if self.mm_model == 'bevfusion' else 0)

        # Setup output files
        self.lidar_pred_json = os.path.join(self.data_dir, f"lidar_pred_{self.index}.json")
        self.image_pred_json = os.path.join(self.data_dir, f"image_pred_{self.index}.json")
        self.image_gt_json = os.path.join(self.data_dir, f"image_gt_{self.index}.json")
        self.delay_csv = os.path.join(self.data_dir, f"delays_{self.index}.csv")

        with open(self.delay_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_stamp', 'input_token', 'comm_delay', 'decode_delay', 'inference_delay', 'e2e_delay', 'model_name'])

        # Define QoS profiles
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

        # Initialize scene and camera sweeps
        self.scene = nusc.get('scene', self.scene_token)
        self.cam_sweeps = None

        # Initialize model runners
        self._init_model_runners()

        # Setup subscriptions
        self._setup_subscriptions(lidar_qos, image_qos)

        # Setup ready publisher
        self.ready_publisher = self.create_publisher(String, 'inferencer_ready', 10)
        msg = String()
        msg.data = "1"
        self.ready_publisher.publish(msg)

        # Setup terminate subscription
        self.terminate_sub = self.create_subscription(String, 'terminate_inferencers', self._terminate_callback, 10)

    def _init_model_runners(self):
        # Initialize lidar models
        for name in self.lidar_models:
            model = LidarDet3DInferencer(name)
            model.show_progress = False
            stream = torch.cuda.Stream()
            self.runners[name] = ModelRunner(name, model, 'lidar', stream, self.depth)

        # Initialize image models
        for name in self.image_models:
            model = DetInferencer(name)
            model.show_progress = False
            stream = torch.cuda.Stream()
            self.runners[name] = ModelRunner(name, model, 'image', stream, self.depth)

        # Initialize multi-modal model only if specified
        if self.mm_model and self.mm_model == 'bevfusion':
            config = '/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
            ckpt = '/mmdetection3d_ros2/perf_ws/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
            model = BEVFormerInferencer(config, ckpt)
            stream = torch.cuda.Stream()
            self.cam_sweeps = model.load_all_camera_sweeps(self.scene_token)
            self.runners[self.mm_model] = ModelRunner(self.mm_model, model, 'multi-modal', stream, self.depth)

            # Warm up multi-modal model
            sample_token = self.scene['first_sample_token']
            sample = nusc.get('sample', sample_token)
            warm_lidar_token = sample['data']['LIDAR_TOP']
            warm_cam_tokens = {
                cam: sample['data'][cam]
                for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            }
            warm_data = (warm_lidar_token, warm_cam_tokens, self.cam_sweeps)
            self.runners[self.mm_model].profiler.warm_up(warm_data)
            self.runners[self.mm_model].profiler.register_hooks(warm_data)

    def _setup_subscriptions(self, lidar_qos, image_qos):
        # Setup lidar subscription
        if self.lidar_models or self.mm_model == 'bevfusion':
            self.sub_lidar = self.create_subscription(
                PointCloud2, 'lidar_data', self.lidar_callback, lidar_qos)

        # Setup image subscription
        if self.image_models or self.mm_model == 'bevfusion':
            self.sub_image = self.create_subscription(
                CompressedImage, 'image_data/CAM_FRONT', self.image_callback, image_qos)

        # Setup multi-modal synchronization if needed
        if self.mm_model == 'bevfusion':
            self.cam_keys = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            
            self.image_subscribers = {
                cam: Subscriber(self, CompressedImage, f'image_data/{cam}', qos_profile=image_qos)
                for cam in self.cam_keys
            }
            
            self.lidar_subscriber = Subscriber(self, PointCloud2, 'lidar_data', qos_profile=lidar_qos)
            
            self.ts = ApproximateTimeSynchronizer(
                [self.lidar_subscriber] + list(self.image_subscribers.values()),
                queue_size=10,
                slop=0.1
            )
            self.ts.registerCallback(self.synced_callback)

    def lidar_callback(self, msg):
        recv_time = time.time()
        frame_id = msg.header.frame_id
        token = frame_id if self.input_type == "publisher" else msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sub_lidar_count += 1

        # Decode
        torch.cuda.nvtx.range_push(f"{token}.lidar_decode")
        input_data = self._decode_lidar(msg)
        torch.cuda.nvtx.range_pop()
        decode_time = time.time()

        # Run inference on all lidar models concurrently using thread pool
        for model_name in self.lidar_models:
            runner = self.runners[model_name]
            # Check if model is still processing previous frame
            if self.active_futures[model_name] is not None:
                if not self.active_futures[model_name].done():
                    self.get_logger().warn(f"Model {model_name} still processing previous frame, skipping current frame")
                    continue
                # Clean up completed future
                self.active_futures[model_name] = None
            
            # Submit new task
            future = self.thread_pool.submit(self._run_lidar_inference, runner, input_data, token)
            self.active_futures[model_name] = future

        # Store communication and decode delays for this token
        self.token_delays[token] = {
            'comm_delay': recv_time - sent_time,
            'decode_delay': decode_time - recv_time
        }

    def _run_lidar_inference(self, runner, input_data, token):
        process_start = time.time()
        det = runner.run(input_data, token)
        process_end = time.time()
        inference_delay = process_end - process_start
        
        # Get communication and decode delays for this token
        token_delays = self.token_delays.get(token, {'comm_delay': 0, 'decode_delay': 0})
        
        # Log delays for this specific model
        self._log_delay(
            time.time(),
            token,
            token_delays['comm_delay'],
            token_delays['decode_delay'],
            inference_delay + token_delays['decode_delay'],  # e2e includes decode
            inference_delay,
            runner.name
        )
        
        self.lidar_dets.append((token, det['predictions'][0].pred_instances_3d))

    def image_callback(self, msg):
        recv_time = time.time()
        frame_id = msg.header.frame_id
        token = frame_id if self.input_type == "publisher" else msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sub_image_count += 1

        # Decode
        torch.cuda.nvtx.range_push(f"{token}.image_decode")
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().error("Error decoding image during processing")
            return
        torch.cuda.nvtx.range_pop()
        decode_time = time.time()

        # Run inference on all image models concurrently using thread pool
        for model_name in self.image_models:
            runner = self.runners[model_name]
            # Check if model is still processing previous frame
            if self.active_futures[model_name] is not None:
                if not self.active_futures[model_name].done():
                    self.get_logger().warn(f"Model {model_name} still processing previous frame, skipping current frame")
                    continue
                # Clean up completed future
                self.active_futures[model_name] = None
            
            # Submit new task
            future = self.thread_pool.submit(self._run_image_inference, runner, img, token)
            self.active_futures[model_name] = future

        # Store communication and decode delays for this token
        self.token_delays[token] = {
            'comm_delay': recv_time - sent_time,
            'decode_delay': decode_time - recv_time
        }

    def _run_image_inference(self, runner, img, token):
        process_start = time.time()
        det = runner.run(img, token)
        process_end = time.time()
        inference_delay = process_end - process_start
        
        # Get communication and decode delays for this token
        token_delays = self.token_delays.get(token, {'comm_delay': 0, 'decode_delay': 0})
        
        # Log delays for this specific model
        self._log_delay(
            time.time(),
            token,
            token_delays['comm_delay'],
            token_delays['decode_delay'],
            inference_delay + token_delays['decode_delay'],  # e2e includes decode
            inference_delay,
            runner.name
        )
        
        self.image_dets.append((token, det['predictions'][0].pred_instances))

    def synced_callback(self, lidar_msg, *cam_msgs):
        recv_time_sim = self.get_clock().now().nanoseconds / 1e9
        lidar_token = lidar_msg.header.frame_id if self.input_type == "publisher" else lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec / 1e9
        sent_time = lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec / 1e9

        cam_tokens = {
            cam: msg.header.frame_id
            for cam, msg in zip(self.cam_keys, cam_msgs)
        }

        # Run inference on multi-modal model
        process_start = time.time()
        data = (lidar_token, cam_tokens, self.cam_sweeps)
        det = self.runners[self.mm_model].run(data, lidar_token)
        process_end = time.time()

        # Log delays
        comm_delay = recv_time_sim - sent_time
        inference_delay = process_end - process_start
        e2e_delay = inference_delay + comm_delay

        self._log_delay(sent_time, lidar_token, comm_delay, 0, e2e_delay, inference_delay, 'multi-modal')
        self.lidar_dets.append((lidar_token, det.pred_instances_3d))

    def _decode_lidar(self, msg):
        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('intensity', np.float32), ('ring', np.float32)
        ])
        points_array = np.frombuffer(msg.data, dtype=dtype)
        points = np.stack([
            points_array['x'], points_array['y'], points_array['z'],
            points_array['intensity'], points_array['ring']
        ], axis=-1)
        return dict(points=points.astype(np.float32))

    def _log_delay(self, sent_time, token, comm_delay, decode_delay, e2e_delay, inference_delay, model_name):
        self.delay_log.append({
            'time_stamp': sent_time,
            'input_token': token,
            'comm_delay': comm_delay,
            'decode_delay': decode_delay,
            'inference_delay': inference_delay,
            'e2e_delay': e2e_delay,
            'model_name': model_name
        })

    def _save_3d_detections(self):
        nusc_annos = {}
        token_mapping = build_channel_timestamp_token_map(nusc, self.scene, "LIDAR_TOP")
        for det in self.lidar_dets:
            token = det[0]
            if self.input_type != "publisher":
                token = get_closest_token_from_timestamp(token, token_mapping)
            boxes = lidar_output_to_nusc_box(det[1], token, self.lidar_model_thresh, self.lidar_model_mode)
            boxes = lidar_nusc_box_to_global(nusc, token, boxes)

            annos = []
            for box in boxes:
                name = nus_lidar_classes[box.label] if self.lidar_model_mode == 'nus' else kitti_lidar_classes[box.label]
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
        token_mapping = build_channel_timestamp_token_map(nusc, self.scene, "CAM_FRONT")
        for det in self.image_dets:
            token = det[0]
            if self.input_type != "publisher":
                token = get_closest_token_from_timestamp(token, token_mapping)
            coco_pred = image_output_to_coco(det[1], token)
            coco_predictions.extend(coco_pred)
        
        with open(self.image_pred_json, 'w') as f:
            json.dump(coco_predictions, f, indent=2)

    def _terminate_callback(self, msg):
        if msg.data.strip() == "TERMINATE":
            self.get_logger().info("Inferencer shutting down...")
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            self.get_logger().info(f'sub lidar count: {self.sub_lidar_count}')
            self.get_logger().info(f'sub image count: {self.sub_image_count}')

            # Save detections
            # if self.lidar_models or self.mm_model == 'bevfusion':
            #     self._save_3d_detections()
            # if self.image_models:
            #     self._save_image_detections()

            # Save delays
            lock_path = self.delay_csv + ".lock"
            with FileLock(lock_path):
                delay_df = pd.DataFrame(self.delay_log)
                if os.path.exists(self.delay_csv):
                    delay_df.to_csv(self.delay_csv, mode='a', header=False, index=False)

            self.get_logger().info(f"Appended communication delays to {self.delay_csv}")
            
            # Clean up ROS 2 resources
            self.destroy_subscription(self.sub_lidar)
            self.destroy_subscription(self.sub_image)
            self.destroy_publisher(self.ready_publisher)
            
            # Destroy the node
            self.destroy_node()
            
            # Shutdown ROS 2
            rclpy.shutdown()
            
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        # Create a multi-threaded executor
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info('Done')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
