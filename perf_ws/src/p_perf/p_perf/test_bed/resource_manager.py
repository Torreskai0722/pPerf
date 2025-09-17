#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import psutil
import time
import os
import signal
import threading
from typing import Dict, List, Optional
import json
import pandas as pd

class ResourceManagerNode(Node):
    """
    Resource manager node that monitors external PIDs and profiles them using Linux perf.
    
    This node:
    1. Listens for PID announcements from inference nodes
    2. Starts perf profiling for each PID
    3. Monitors resource usage (CPU, RAM, GPU)
    4. Automatically stops profiling when PIDs terminate
    5. Saves profiling data and resource stats
    """
    
    def __init__(self):
        super().__init__('resource_manager')
        
        # Parameters
        self.declare_parameter('data_dir', '/tmp')
        self.declare_parameter('index', 0)
        self.declare_parameter('perf_output_dir', '/tmp/perf_data')
        self.declare_parameter('monitor_interval', 0.1)  # seconds
        
        self.data_dir = self.get_parameter('data_dir').value
        self.index = self.get_parameter('index').value
        self.perf_output_dir = self.get_parameter('perf_output_dir').value
        self.monitor_interval = self.get_parameter('monitor_interval').value
        
        # Create output directories
        os.makedirs(self.perf_output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # State tracking
        self.monitored_pids: Dict[int, Dict] = {}  # pid -> {process_info, perf_process, start_time}
        self.resource_stats: List[Dict] = []
        self.termination_requested = False
        
        # Subscribers
        self.pid_subscriber = self.create_subscription(
            String, 'pid_announcement', self.pid_announcement_callback, 10
        )
        
        # Publishers
        self.status_publisher = self.create_publisher(String, 'resource_manager_status', 10)
        
        # Termination subscriber
        self.terminate_subscriber = self.create_subscription(
            String, 'terminate_inferencers', self.terminate_callback, 5
        )

        
    def pid_announcement_callback(self, msg):
        """Handle PID announcements from inference nodes."""
        try:
            data = json.loads(msg.data)
            pid = data.get('pid')
            node_name = data.get('node_name', 'unknown')
            model_name = data.get('model_name', 'unknown')
            mode = data.get('mode', 'unknown')
            
            if pid and pid not in self.monitored_pids:
                self.get_logger().info(f"Received PID announcement: {node_name} (PID: {pid}, Model: {model_name}, Mode: {mode})")
                self._start_monitoring_pid(pid, node_name, model_name, mode)
                self.get_logger().info(f"Started monitoring PID {pid} ({node_name})")
            else:
                if pid in self.monitored_pids:
                    self.get_logger().warn(f"PID {pid} already being monitored")
                else:
                    self.get_logger().warn(f"Invalid PID announcement: {msg.data}")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse PID announcement: {e}")
        except Exception as e:
            self.get_logger().error(f"Error handling PID announcement: {e}")
    
    def _start_monitoring_pid(self, pid: int, node_name: str, model_name: str, mode: str):
        """Start perf profiling for a specific PID."""
        try:
            # Verify process exists
            if not psutil.pid_exists(pid):
                self.get_logger().warn(f"PID {pid} does not exist")
                return
            
            # Create perf output filename
            timestamp = int(time.time())
            perf_output = os.path.join(
                self.perf_output_dir, 
                f"perf_{node_name}_{model_name}_{mode}_{pid}_{timestamp}.data"
            )
            
            # Start perf record command
            perf_cmd = [
                'perf', 'record',
                '--pid', str(pid),
                '--output', perf_output,
                '--freq', '50',         # Sample frequency
                '--event', 'cpu-cycles,instructions,cache-misses,branch-misses',
                '--output-format', 'perf.data'
            ]
            
            # Start perf process
            perf_process = subprocess.Popen(
                perf_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Store minimal info
            self.monitored_pids[pid] = {
                'node_name': node_name,
                'model_name': model_name,
                'mode': mode,
                'perf_process': perf_process,
                'perf_output': perf_output,
                'start_time': time.time()
            }
            
            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'action': 'started_monitoring',
                'pid': pid,
                'node_name': node_name,
                'model_name': model_name,
                'mode': mode
            })
            self.status_publisher.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to start monitoring PID {pid}: {e}")
    
    
    def _stop_monitoring_pid(self, pid: int, reason: str):
        """Stop monitoring a specific PID."""
        if pid not in self.monitored_pids:
            return
        
        info = self.monitored_pids[pid]
        self.get_logger().info(f"Stopping monitoring PID {pid} ({reason})")
        
        try:
            # Stop perf process
            if info['perf_process'] and info['perf_process'].poll() is None:
                # Send SIGTERM to perf process group
                os.killpg(os.getpgid(info['perf_process'].pid), signal.SIGTERM)
                
                # Wait a bit for graceful shutdown
                try:
                    info['perf_process'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    os.killpg(os.getpgid(info['perf_process'].pid), signal.SIGKILL)
                    info['perf_process'].wait()
            
            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'action': 'stopped_monitoring',
                'pid': pid,
                'node_name': info['node_name'],
                'model_name': info['model_name'],
                'mode': info['mode'],
                'reason': reason,
                'duration': time.time() - info['start_time']
            })
            self.status_publisher.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error stopping monitoring PID {pid}: {e}")
    
    def terminate_callback(self, msg):
        """Handle termination request."""
        if msg.data.strip() == "TERMINATE":
            self.get_logger().info("Resource Manager shutting down...")
            
            # Stop monitoring all PIDs
            for pid in list(self.monitored_pids.keys()):
                self._stop_monitoring_pid(pid, "shutdown")
            
            # No resource stats to save
            self.destroy_node()
            raise SystemExit
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            'monitored_pids': len(self.monitored_pids),
            'active_pids': list(self.monitored_pids.keys()),
            'resource_stats_count': len(self.resource_stats),
            'termination_requested': self.termination_requested
        }


def main(args=None):
    rclpy.init(args=args)
    node = ResourceManagerNode()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Resource Manager").info('Shutdown complete')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main() 