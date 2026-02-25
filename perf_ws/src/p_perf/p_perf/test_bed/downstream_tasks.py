#!/usr/bin/python3

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
import message_filters


class _StampedString:
    """Minimal message wrapper with header for message_filters."""

    def __init__(self, stamp_msg: Header, data: str):
        self.header = stamp_msg
        self.data = data


class DownstreamTasksNode(Node):
    def __init__(self):
        super().__init__('downstream_tasks')

        # Sync configuration
        self.declare_parameter('bbox_sync_queue_size', 10)
        self.declare_parameter('bbox_sync_slop', 0.05)
        self.declare_parameter('seg_sync_queue_size', 10)
        self.declare_parameter('seg_sync_slop', 0.05)

        bbox_queue = self.get_parameter('bbox_sync_queue_size').value
        bbox_slop = self.get_parameter('bbox_sync_slop').value
        seg_queue = self.get_parameter('seg_sync_queue_size').value
        seg_slop = self.get_parameter('seg_sync_slop').value

        # Filters for bbox and seg topics (single 4-way sync)
        self._lidar_bbox_filter = message_filters.SimpleFilter()
        self._image_bbox_filter = message_filters.SimpleFilter()
        self._lidar_seg_filter = message_filters.SimpleFilter()
        self._image_seg_filter = message_filters.SimpleFilter()

        sync_queue = max(bbox_queue, seg_queue)
        sync_slop = max(bbox_slop, seg_slop)
        self._all_sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._lidar_bbox_filter,
                self._image_bbox_filter,
                self._lidar_seg_filter,
                self._image_seg_filter,
            ],
            queue_size=sync_queue,
            slop=sync_slop,
            allow_headerless=False
        )
        self._all_sync.registerCallback(self._all_sync_cb)

        # Subscriptions
        self.create_subscription(String, '/lidar_bbox', self._lidar_bbox_cb, 10)
        self.create_subscription(String, '/image_bbox', self._image_bbox_cb, 10)
        self.create_subscription(String, '/lidar_seg', self._lidar_seg_cb, 10)
        self.create_subscription(String, '/image_seg', self._image_seg_cb, 10)

    def _parse_stamp(self, msg: String) -> _StampedString:
        """Parse sensor-frame timestamp from message data."""
        stamp = None
        data = msg.data
        try:
            payload = json.loads(data)
            stamp = payload.get('stamp')
            data = payload.get('data', data)
        except json.JSONDecodeError:
            stamp = None

        if isinstance(stamp, (int, float)):
            sec = int(stamp)
            nanosec = int((stamp - sec) * 1e9)
            header = Header()
            header.stamp.sec = sec
            header.stamp.nanosec = nanosec
        else:
            # Fallback: use current time if stamp is missing/malformed
            header = Header()
            now = self.get_clock().now().to_msg()
            header.stamp = now
            self.get_logger().warning('Missing/invalid stamp in message, using current time.')

        return _StampedString(header, data)

    @staticmethod
    def _stamp_to_float(stamp_msg: Header) -> float:
        return float(stamp_msg.stamp.sec) + float(stamp_msg.stamp.nanosec) / 1e9

    def _lidar_bbox_cb(self, msg: String):
        self._lidar_bbox_filter.signalMessage(self._parse_stamp(msg))

    def _image_bbox_cb(self, msg: String):
        self._image_bbox_filter.signalMessage(self._parse_stamp(msg))

    def _lidar_seg_cb(self, msg: String):
        self._lidar_seg_filter.signalMessage(self._parse_stamp(msg))

    def _image_seg_cb(self, msg: String):
        self._image_seg_filter.signalMessage(self._parse_stamp(msg))

    def _all_sync_cb(self, lidar_bbox_msg, image_bbox_msg, lidar_seg_msg, image_seg_msg):
        min_stamp = min(
            self._stamp_to_float(lidar_bbox_msg.header),
            self._stamp_to_float(image_bbox_msg.header),
            self._stamp_to_float(lidar_seg_msg.header),
            self._stamp_to_float(image_seg_msg.header),
        )
        latency = time.time() - min_stamp
        self.get_logger().info(
            f'all fusion latency: {latency:.6f}s (min stamp {min_stamp:.6f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = DownstreamTasksNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger('Quitting').info('Done')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
