#!/usr/bin/python3

import torch
import os
import csv
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CompressedImage

from p_perf.test_bed.det_inferencer import DetInferenceNode
from p_perf.pPerf_inferencer import pPerf3dSegInferencer, bddSegInferencer


class SegmentationInferenceNode(DetInferenceNode):
    def __init__(self):
        # Set flag to track if spatial tiling has been configured
        self._spatial_tiling_configured = False
        
        # Call parent constructor which will call our overridden methods
        super().__init__()
        
        # Declare spatial tiling parameters
        self.declare_parameter('decode_head_h', 1)
        self.declare_parameter('decode_head_w', 1)
        self.decode_head_h = self.get_parameter('decode_head_h').value
        self.decode_head_w = self.get_parameter('decode_head_w').value
        
        # Configure spatial tiling NOW (after model is initialized)
        self._configure_spatial_tiling()
        
        # Override CSV filename for segmentation
        if self.logging_delay:
            self.delay_csv = os.path.join(self.data_dir, f"delays_{self.index}_seg.csv")

            with open(self.delay_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_stamp', 'input_token', 'comm_delay', 'decode_delay', 'inference_delay', 'e2e_delay', 'model_name', 'model_type'])
    
    def _configure_spatial_tiling(self):
        """Configure spatial tiling for segmentation models"""
        if self._spatial_tiling_configured:
            return
            
        if self.mode in ['sem_seg', 'ins_seg', 'pan_seg', 'drivable']:
            if hasattr(self.inferencer.model, 'decode_head') and \
               hasattr(self.inferencer.model.decode_head, 'bottleneck_spatial'):
                module = self.inferencer.model.decode_head.bottleneck_spatial
                module.splits_h = self.decode_head_h
                module.splits_w = self.decode_head_w
                self.get_logger().info(
                    f"Configured spatial tiling: {self.decode_head_h}x{self.decode_head_w} "
                    f"(halo={module.halo})"
                )
                self._spatial_tiling_configured = True
            else:
                self.get_logger().warning(
                    f"Model does not have decode_head.bottleneck_spatial, "
                    f"spatial tiling not configured"
                )

    def _init_output_publishers(self):
        """Initialize output publishers for segmentation."""
        self.lidar_output_topic = '/lidar_seg'
        self.image_output_topic = '/image_seg'
        self.lidar_output_pub = self.create_publisher(String, self.lidar_output_topic, 10)
        self.image_output_pub = self.create_publisher(String, self.image_output_topic, 10)

    def _init_subscribers(self):
        """Initialize subscribers - can be overridden by child classes"""
        if self.use_message_filters:
            self._init_message_filters('lidar_data', 'image_data/CAM_FRONT')
        else:
            if self.mode == '3d':
                self.subscriber = self.create_subscription(PointCloud2, 'lidar_data', self.lidar_callback, self.lidar_qos)
            elif self.mode in ['sem_seg', 'ins_seg', 'pan_seg', 'drivable']:
                self.subscriber = self.create_subscription(CompressedImage, 'image_data/CAM_FRONT', self.image_callback, self.image_qos)
            else:
                raise ValueError(f"Invalid mode '{self.mode}'. Supported modes: '3d', 'sem_seg', 'ins_seg', 'pan_seg', 'drivable'")

        # Subscribe to termination signal
        self.create_subscription(String, 'terminate_inferencers', self._terminate_callback, 5)

    def _init_models(self):
        """Override model initialization for segmentation"""
        if self.mode == '3d':
            # Initialize model with memory optimizations
            self.inferencer = pPerf3dSegInferencer(self.model_name)
            self.inferencer.show_progress = False

        elif self.mode in ['sem_seg', 'ins_seg', 'pan_seg', 'drivable']:
            # Initialize model with memory optimizations
            self.inferencer = bddSegInferencer(self.model_name, self.mode)
            self.inferencer.show_progress = False

        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Supported modes: '3d', 'sem_seg', 'ins_seg', 'pan_seg', 'drivable'")

    def _init_warmup(self):
        """Override warmup for segmentation"""
        from p_perf.pPerf import pPerf
        
        # Create new profiler with segmentation model (inherits engine_selector from parent)
        self.profiler = pPerf(
            self.model_name, 
            self.inferencer, 
            self.depth, 
            self.mode,
            engine_selector=self.engine_selector
        )
        
        # Select warm data based on mode - access from parent class
        from p_perf.test_bed.det_inferencer import WARM_PCD, WARM_IMAGE
        warm_data = dict(points=WARM_PCD) if self.mode == '3d' else dict(img=WARM_IMAGE)
        
        self.profiler.warm_up(warm_data)
        self.profiler.register_hooks(warm_data)

    def synced_callback(self, lidar_msg, image_msg):
        """Handle time-aligned lidar + image messages for segmentation modes."""
        if self.mode == '3d':
            self.lidar_callback(lidar_msg)
        elif self.mode in ['sem_seg', 'ins_seg', 'pan_seg', 'drivable']:
            self.image_callback(image_msg)
        else:
            self.get_logger().warning(
                f"Synced callback received for unsupported mode '{self.mode}'"
            )


    def _announce_pid(self):
        """Announce the PID of the segmentation inferencer to the resource manager."""
        import json
        import os
        
        pid = os.getpid()
        announcement = {
            'pid': pid,
            'node_name': f'{self.mode}_seg_inferencer_{self.index}',
            'model_name': self.model_name,
            'mode': self.mode
        }
        
        msg = String()
        msg.data = json.dumps(announcement)
        self.pid_publisher.publish(msg)
        self.get_logger().info(f"PID {pid} announced for {self.mode} segmentation model '{self.model_name}'")


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = SegmentationInferenceNode()
    try:
        rclpy.spin(node)
    except SystemExit:                 
        rclpy.logging.get_logger("Quitting").info('Done')
    rclpy.shutdown()

if __name__ == '__main__':
    main()
