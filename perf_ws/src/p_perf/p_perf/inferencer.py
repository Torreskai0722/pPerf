#!/usr/bin/python3

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
import csv
from message_filters import Subscriber, ApproximateTimeSynchronizer
import time

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
from turbojpeg import TurboJPEG

from p_perf.pPerf import pPerf
from p_perf.post_process.lidar_eval import lidar_nusc_box_to_global, lidar_output_to_nusc_box
from p_perf.post_process.image_eval import image_output_to_coco, generate_coco_gt, change_pred_imageid
from p_perf.config.constant import nus_lidar_classes, kitti_lidar_classes
from p_perf.BEVInferencer import BEVFormerInferencer
from p_perf.utils import get_closest_token_from_timestamp, build_channel_timestamp_token_map, convert_to_kitti_ros
from p_perf.nuscenes_instance import get_nuscenes_instance

warnings.filterwarnings("ignore")

WARM_PCD = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_IMAGE = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # Lazy load NuScenes instance
        self._nusc = None

        # Add memory management parameters
        self.max_dets_in_memory = 100  # Maximum number of detections to keep in memory
        self.save_interval = 50  # Save detections to disk every N frames
        self.max_buffer_size = 100  # Maximum number of items to keep in memory
        self.dets = []  # Will be managed as a fixed-size buffer
        self.delay_log = []  # Will be managed as a fixed-size buffer

        # Add precision control
        self.float_precision = 4  # Number of decimal places to keep

        # INPUT MODE
        self.declare_parameter('input_type', 'publisher')
        self.declare_parameter('lidar_model_mode', 'nus')
        self.declare_parameter('lidar_model_thresh', 0.2)
        self.declare_parameter('scene', '')

        self.input_type = self.get_parameter('input_type').value  
        self.lidar_model_mode = self.get_parameter('lidar_model_mode').value  
        self.lidar_model_thresh = self.get_parameter('lidar_model_thresh').value 
        self.scene = self.get_parameter('scene').value

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
        self.sub_lidar_count = 0
        self.sub_image_count = 0

        self.dets = []
        self.lidar_pred_json = os.path.join(self.data_dir, f"lidar_pred_{self.index}.json")
        self.image_pred_json = os.path.join(self.data_dir, f"image_pred_{self.index}.json")
        self.image_gt_json = os.path.join(self.data_dir, f"image_gt_{self.index}.json")
        self.delay_csv = os.path.join(self.data_dir, f"delays_{self.index}.csv")
        # Save raw detections with model name for better organization
        self.raw_detections_json = os.path.join(self.data_dir, f"raw_detections_{self.model_name}_{self.index}.json")
        self.delay_log = []

        self.jpeg = TurboJPEG()

        with open(self.delay_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_stamp', 'input_token', 'comm_delay', 'decode_delay', 'inference_delay', 'e2e_delay', 'model_name', 'model_type'])

        # INIT OF MODELS BASED ON MODE
        self.ready_publisher = self.create_publisher(String, 'inferencer_ready', 10)

        if self.mode == 'lidar':
            # Clear GPU cache before processing new frame
            torch.cuda.empty_cache()
            self.subscriber = self.create_subscription(PointCloud2, 'lidar_data', self.lidar_callback, lidar_qos)
            # Initialize model with memory optimizations
            torch.cuda.empty_cache()
            self.inferencer = LidarDet3DInferencer(self.model_name)
            self.inferencer.show_progress = False
            # Set model to eval mode and enable memory optimizations
            self.inferencer.model.eval()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        elif self.mode == 'image':
            # Clear GPU cache before processing new frame
            torch.cuda.empty_cache()
            self.subscriber = self.create_subscription(CompressedImage, 'image_data/CAM_FRONT', self.image_callback, image_qos)
            # Initialize model with memory optimizations
            torch.cuda.empty_cache()
            self.inferencer = DetInferencer(self.model_name)
            self.inferencer.show_progress = False
            # Set model to eval mode and enable memory optimizations
            self.inferencer.model.eval()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        elif self.mode == 'multi-modal':
            self.cam_keys = [
                    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                ]

            self.image_subscribers = {
                cam: Subscriber(self, CompressedImage, f'image_data/{cam}', qos_profile=image_qos)
                for cam in self.cam_keys
            }

            self.lidar_subscriber = Subscriber(self, PointCloud2, 'lidar_data', qos_profile=lidar_qos)

            self.ts = ApproximateTimeSynchronizer(
                [self.lidar_subscriber] + list(self.image_subscribers.values()),
                queue_size=10,
                slop=0.1  # 50ms tolerance
            )
            self.ts.registerCallback(self.synced_callback)

            # Preload cam_sweeps externally and assign to self.cam_sweeps
            config = '/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
            chkpt = '/mmdetection3d_ros2/perf_ws/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'

            self.inferencer = BEVFormerInferencer(config, chkpt)
            self.cam_sweeps = self.inferencer.load_all_camera_sweeps(self.scene)

            # self.scene = self.nusc.get('scene', self.scene)

            sample_token = self.scene['first_sample_token']
            # sample = self.nusc.get('sample', sample_token)

            # warm_lidar_token = sample['data']['LIDAR_TOP']
            # warm_cam_tokens = {
            #     cam: sample['data'][cam]
            #     for cam in [
            #         'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            #         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            #     ]
            # }


        # Subscribe to termination signal
        self.create_subscription(String, 'terminate_inferencers', self._terminate_callback, 10)

        # WARMUP
        self.profiler = pPerf(self.model_name, self.inferencer, self.depth, self.mode)
        if self.mode == 'lidar':
            warm_data = dict(points=WARM_PCD)
        elif self.mode == 'image':
            warm_data = WARM_IMAGE

        if self.mode == 'multi-modal':
            warm_data = (warm_lidar_token, warm_cam_tokens, self.cam_sweeps)
            self.profiler.warm_up(warm_data)
            self.profiler.register_hooks(warm_data)
        elif self.mode == 'lidar': 
            self.profiler.warm_up(warm_data)
            self.profiler.register_hooks(warm_data)
        elif self.mode == 'image':
            self.profiler.warm_up(warm_data)
            self.profiler.register_hooks(warm_data)

        # INFERENCER READY MSG FOR SENSOR PUBLISHER
        self.get_logger().info(f"{self.mode.capitalize()} model '{self.model_name}' is ready.")
        msg = String()
        msg.data = "1"
        self.ready_publisher.publish(msg)

    def _log_delay(self, sent_time, token, comm_delay, decode_delay, e2e_delay, inference_delay, model_name, model_type):
        """Store delay data directly"""
        self.delay_log.append({
            'time_stamp': sent_time,
            'input_token': token,
            'comm_delay': comm_delay,
            'decode_delay': decode_delay,
            'inference_delay': inference_delay,
            'e2e_delay': e2e_delay,
            'model_name': model_name,
            'model_type': model_type
        })

    def lidar_callback(self, msg):
        recv_time_sim = self.get_clock().now().nanoseconds / 1e9
        recv_time = time.time()
        frame_id = msg.header.frame_id
        token = frame_id if self.input_type == "publisher" else msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sub_lidar_count += 1

        # Decode
        torch.cuda.nvtx.range_push(f"{token}.lidar_decode")
        input_data = self._decode_lidar(msg, self.lidar_model_mode)
        torch.cuda.nvtx.range_pop()

        decode_time = time.time()

        # Inference
        process_start = time.time()
        det = self.profiler.run_inference(input_data, token)
        process_end = time.time()

        # Process delay logging
        comm_delay = recv_time_sim - sent_time
        decode_delay = decode_time - recv_time
        inference_delay = process_end - process_start
        e2e_delay = inference_delay + decode_delay + abs(comm_delay)

        self._log_delay(sent_time, token, comm_delay, decode_delay, e2e_delay, inference_delay, self.model_name, self.mode)
        # Store raw prediction data instead of processed instances
        self.dets.append((token, det['predictions'][0]))

    def image_callback(self, msg):
        recv_time_sim = self.get_clock().now().nanoseconds / 1e9
        recv_time = time.time()
        frame_id = msg.header.frame_id
        token = frame_id if self.input_type == "publisher" else msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        sent_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.sub_image_count += 1

        # Decode
        torch.cuda.nvtx.range_push(f"{token}.image_decode")
        # img = self.jpeg.decode(msg.data)
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
        det = self.profiler.run_inference(input_data, token)
        process_end = time.time()

        # Process delay logging
        comm_delay = recv_time_sim - sent_time
        decode_delay = decode_time - recv_time
        inference_delay = process_end - process_start
        e2e_delay = inference_delay + decode_delay + abs(comm_delay)

        self._log_delay(sent_time, token, comm_delay, decode_delay, e2e_delay, inference_delay, self.model_name, self.mode)
        # Store raw prediction data instead of processed instances
        self.dets.append((token, det['predictions'][0]))

    def synced_callback(self, lidar_msg, *cam_msgs):
        recv_time_sim = self.get_clock().now().nanoseconds / 1e9
        lidar_token = lidar_msg.header.frame_id if self.input_type == "publisher" else lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec / 1e9
        cam_tokens = {
            cam: msg.header.frame_id
            for cam, msg in zip(self.cam_keys, cam_msgs)
        }


        # Run inference
        process_start = time.time()
        data = (lidar_token, cam_tokens, self.cam_sweeps)
        det = self.profiler.run_inference(data, lidar_token)
        process_end = time.time()

        # LOGGING
        sent_time = lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec / 1e9
        comm_delay = recv_time_sim - sent_time
        inference_delay = process_end - process_start

        e2e_delay = inference_delay + comm_delay + abs(comm_delay)

        self._log_delay(sent_time, lidar_token, comm_delay, 0, e2e_delay, inference_delay, "BEVFusion", self.mode)
        # Store raw prediction data instead of processed instances
        self.dets.append((lidar_token, det))

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

    def _save_raw_detections(self):
        """Save raw detections in JSON format"""
        # Convert detections to serializable format
        serializable_dets = []
        for token, pred in self.dets:
            # Convert prediction object to dict
            pred_dict = {}
            
            # Handle different prediction object structures
            if hasattr(pred, 'pred_instances_3d'):
                # For 3D detections from lidar models
                instances_3d = pred.pred_instances_3d
                if hasattr(instances_3d, 'bboxes_3d'):
                    pred_dict['bboxes_3d'] = instances_3d.bboxes_3d.cpu().numpy().tolist()
                if hasattr(instances_3d, 'scores_3d'):
                    pred_dict['scores_3d'] = instances_3d.scores_3d.cpu().numpy().tolist()
                if hasattr(instances_3d, 'labels_3d'):
                    pred_dict['labels_3d'] = instances_3d.labels_3d.cpu().numpy().tolist()
            elif hasattr(pred, 'pred_instances'):
                # For 2D detections from image models
                instances = pred.pred_instances
                if hasattr(instances, 'bboxes'):
                    pred_dict['bboxes'] = instances.bboxes.cpu().numpy().tolist()
                if hasattr(instances, 'scores'):
                    pred_dict['scores'] = instances.scores.cpu().numpy().tolist()
                if hasattr(instances, 'labels'):
                    pred_dict['labels'] = instances.labels.cpu().numpy().tolist()
            elif hasattr(pred, 'bboxes_3d'):
                # Direct 3D attributes
                pred_dict['bboxes_3d'] = pred.bboxes_3d.cpu().numpy().tolist()
                pred_dict['scores_3d'] = pred.scores_3d.cpu().numpy().tolist()
                pred_dict['labels_3d'] = pred.labels_3d.cpu().numpy().tolist()
            elif hasattr(pred, 'bboxes'):
                # Direct 2D attributes
                pred_dict['bboxes'] = pred.bboxes.cpu().numpy().tolist()
                pred_dict['scores'] = pred.scores.cpu().numpy().tolist()
                pred_dict['labels'] = pred.labels.cpu().numpy().tolist()
            
            serializable_dets.append((token, pred_dict))
        
        with open(self.raw_detections_json, 'w') as f:
            json.dump(serializable_dets, f, indent=2)
        
        print(f"Raw detections written to {self.raw_detections_json}")

    # def _save_3d_detections(self):
    #     """Save 3D detections in original format"""
    #     nusc_annos = {}
    #     token_mapping = build_channel_timestamp_token_map(self.nusc, self.scene, "LIDAR_TOP")
        
    #     for token, pred in self.dets:
    #         if self.input_type != "publisher":
    #             token = get_closest_token_from_timestamp(token, token_mapping)
            
    #         boxes = lidar_output_to_nusc_box(
    #             pred,
    #             token,
    #             self.lidar_model_thresh,
    #             self.lidar_model_mode
    #         )
    #         boxes = lidar_nusc_box_to_global(self.nusc, token, boxes)

    #         annos = []
    #         for box in boxes:
    #             if self.lidar_model_mode == 'nus' or 'car' in self.lidar_model_mode:
    #                 name = nus_lidar_classes[box.label]
    #             else:
    #                 name = kitti_lidar_classes[box.label]
    #             nusc_anno = dict(
    #                 sample_token=token,
    #                 translation=box.center.tolist(),
    #                 size=box.wlh.tolist(),
    #                 rotation=box.orientation.elements.tolist(),
    #                 velocity=box.velocity[:2].tolist(),
    #                 detection_name=name,
    #                 detection_score=float(box.score),
    #                 attribute_name='')
    #             annos.append(nusc_anno)
    #         nusc_annos[token] = annos
        
    #     nusc_submission = {
    #         'meta': {
    #             'use_camera': False,
    #             'use_lidar': True,
    #             'use_radar': False,
    #             'use_map': False,
    #             'use_external': False
    #         },
    #         'results': nusc_annos
    #     }
        
    #     mmengine.dump(nusc_submission, self.lidar_pred_json)
    #     print(f"Results written to {self.lidar_pred_json}")

    # def _save_image_detections(self):
    #     """Save 2D detections in original format"""
    #     coco_predictions = []
    #     token_mapping = build_channel_timestamp_token_map(self.nusc, self.scene, "CAM_FRONT")
        
    #     for token, pred in self.dets:
    #         if self.input_type != "publisher":
    #             token = get_closest_token_from_timestamp(token, token_mapping)
            
    #         coco_pred = image_output_to_coco(pred, token)
    #         coco_predictions.extend(coco_pred)
        
    #     with open(self.image_pred_json, 'w') as f:
    #         json.dump(coco_predictions, f, indent=2)

    def _terminate_callback(self, msg):
        if msg.data.strip() == "TERMINATE":
            self.get_logger().info(f"{self.mode} Inferencer shutting down.................")

            print(f'sub lidar count: {self.sub_lidar_count}')
            print(f'sub image count: {self.sub_image_count}')

            # save lidar detections
            # if self.mode == 'lidar' or self.mode == 'multi-modal':
            #     self._save_3d_detections()
            # else:
            #     self._save_image_detections()
            
            # save raw detections
            self._save_raw_detections()
            # # Convert timestamps to tokens in delay log
            # if self.input_type != "publisher":
            #     lidar_token_mapping = build_channel_timestamp_token_map(self.nusc, self.scene, "LIDAR_TOP")
            #     image_token_mapping = build_channel_timestamp_token_map(self.nusc, self.scene, "CAM_FRONT")
                
            #     for delay in self.delay_log:
            #         if delay['model_type'] == 'lidar':
            #             delay['input_token'] = get_closest_token_from_timestamp(delay['input_token'], lidar_token_mapping)
            #         elif delay['model_type'] == 'image':
            #             delay['input_token'] = get_closest_token_from_timestamp(delay['input_token'], image_token_mapping)

            # Shared delay file path
            lock_path = self.delay_csv + ".lock"

            # Append mode with locking
            with FileLock(lock_path):
                delay_df = pd.DataFrame(self.delay_log)
                if os.path.exists(self.delay_csv):
                    delay_df.to_csv(self.delay_csv, mode='a', header=False, index=False)

            self.get_logger().info(f"Appended communication delays to {self.delay_csv}")

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