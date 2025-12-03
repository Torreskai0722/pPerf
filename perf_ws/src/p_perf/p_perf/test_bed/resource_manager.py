#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import psutil
import time
from typing import Dict, Optional, List
import json
import csv
from datetime import datetime
from pathlib import Path

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be disabled.")

class ResourceMonitorNode(Node):
    """
    Resource monitor node for tracking inference process metrics.
    
    This node monitors per-process resource usage including:
    - CPU utilization
    - Memory usage (RAM)
    - Thread count
    - GPU utilization
    - GPU memory
    - GPU power consumption
    - GPU temperature
    
    On shutdown, saves all collected data to CSV files and calculates energy consumption.
    
    Topics:
    - Subscribes to: 'pid_announcement', 'terminate_inferencers'
    - Publishes to: 'resource_metrics'
    """
    
    def __init__(self):
        super().__init__('resource_monitor')
        
        # Parameters
        self.declare_parameter('index', 0)
        self.declare_parameter('data_dir', './')
        
        self.index = self.get_parameter('index').value
        self.data_dir = Path(self.get_parameter('data_dir').value)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.announced_pids: Dict[int, Dict] = {}  # pid -> {node_name, model_name, mode}
        
        # Resource monitoring - store all measurements for final CSV
        self.all_metrics: List[Dict] = []  # List of all metric snapshots with metadata
        self.process_metrics: Dict[int, Dict] = {}  # pid -> latest metrics (for publishing)
        self.monitoring_enabled = True
        self.monitoring_interval = 1.0  # seconds
        
        # Initialize GPU monitoring (single GPU)
        self.gpu_initialized = False
        self.gpu_handle = None
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_initialized = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Single GPU at index 0
                self.get_logger().info("GPU monitoring initialized for single GPU")
            except Exception as e:
                self.get_logger().warn(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_initialized = False
        else:
            self.get_logger().warn("GPU monitoring disabled (pynvml not available)")
        
        # Subscribers
        self.pid_subscriber = self.create_subscription(
            String, 'pid_announcement', self.pid_announcement_callback, 10
        )
        
        # Termination subscriber - listens to sensor_replayer's terminate command
        self.terminate_subscriber = self.create_subscription(
            String, 'terminate_inferencers', self.terminate_callback, 5
        )
        
        # Publishers
        self.metrics_publisher = self.create_publisher(String, 'resource_metrics', 10)
        
        # Start resource monitoring timer
        self.monitoring_timer = self.create_timer(
            self.monitoring_interval,
            self.monitor_resources_callback
        )
        
        self.get_logger().info(f"Resource Monitor started. Index: {self.index}, Data dir: {self.data_dir}")

        
    def pid_announcement_callback(self, msg):
        """Handle PID announcements from inference nodes."""
        try:
            data = json.loads(msg.data)
            pid = data.get('pid')
            node_name = data.get('node_name', 'unknown')
            model_name = data.get('model_name', 'unknown')
            mode = data.get('mode', 'unknown')
            
            if pid and pid not in self.announced_pids:
                self.get_logger().info(f"Monitoring PID {pid}: {node_name} (Model: {model_name}, Mode: {mode})")
                
                # Store PID info
                self.announced_pids[pid] = {
                    'node_name': node_name,
                    'model_name': model_name,
                    'mode': mode
                }
            elif pid in self.announced_pids:
                self.get_logger().debug(f"PID {pid} already being monitored")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse PID announcement: {e}")
        except Exception as e:
            self.get_logger().error(f"Error handling PID announcement: {e}")
    
    
    def monitor_resources_callback(self):
        """Periodically monitor resource usage for all tracked processes."""
        if not self.monitoring_enabled:
            return
        
        # Get all announced PIDs to monitor
        pids_to_monitor = list(self.announced_pids.keys())
        
        for pid in pids_to_monitor:
            try:
                if not psutil.pid_exists(pid):
                    # Process no longer exists, clean up
                    self.get_logger().info(f"Process {pid} no longer exists, removing from monitoring")
                    if pid in self.announced_pids:
                        del self.announced_pids[pid]
                    if pid in self.process_metrics:
                        del self.process_metrics[pid]
                    continue
                
                # Get process metrics
                metrics = self._get_process_metrics(pid)
                
                if metrics:
                    # Add timestamp and process info to metrics
                    timestamp = time.time()
                    metrics['timestamp'] = timestamp
                    
                    # Store latest metrics for publishing
                    self.process_metrics[pid] = metrics
                    
                    # Get process info
                    node_info = self.announced_pids[pid]
                    node_name = node_info.get('node_name', 'unknown')
                    model_name = node_info.get('model_name', 'unknown')
                    mode = node_info.get('mode', 'unknown')
                    
                    # Store complete record for CSV (flat structure)
                    record = {
                        'process_id': pid,
                        'timestamp': timestamp,
                        'node_name': node_name,
                        'model_name': model_name,
                        'mode': mode,
                        'cpu_percent': metrics.get('cpu_percent', 0),
                        'memory_mb': metrics.get('memory_mb', 0),
                        'memory_percent': metrics.get('memory_percent', 0),
                        'num_threads': metrics.get('num_threads', 0),
                        'gpu_utilization': metrics.get('gpu_utilization', 0),
                        'gpu_memory_mb': metrics.get('gpu_memory_mb', 0),
                        'gpu_device_id': metrics.get('gpu_device_id', -1),
                        'gpu_power_watts': metrics.get('gpu_power_watts', 0),
                        'gpu_power_limit_watts': metrics.get('gpu_power_limit_watts', 0),
                        'gpu_temperature_c': metrics.get('gpu_temperature_c', 0)
                    }
                    self.all_metrics.append(record)
                    
                    # Publish metrics
                    self._publish_process_metrics(pid, node_name, metrics)
                    
            except psutil.NoSuchProcess:
                # Process terminated between check and metric collection
                if pid in self.announced_pids:
                    del self.announced_pids[pid]
                if pid in self.process_metrics:
                    del self.process_metrics[pid]
            except Exception as e:
                self.get_logger().error(f"Error monitoring PID {pid}: {e}")
    
    def _get_process_metrics(self, pid: int) -> Optional[Dict]:
        """Get resource metrics for a specific process.
        
        Returns:
            Dictionary with metrics: {
                'cpu_percent': float,
                'memory_mb': float,
                'memory_percent': float,
                'num_threads': int,
                'gpu_utilization': float,  # Per-process GPU utilization (%)
                'gpu_memory_mb': float,    # Per-process GPU memory (MB)
                'gpu_device_id': int,      # GPU device this process is using
                'gpu_power_watts': float,  # GPU device power draw (Watts)
                'gpu_power_limit_watts': float,  # GPU device power limit (Watts)
                'gpu_temperature_c': float  # GPU device temperature (Celsius)
            }
        """
        try:
            process = psutil.Process(pid)
            
            # Get CPU and memory info
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            memory_percent = process.memory_percent()
            num_threads = process.num_threads()
            
            # Get GPU metrics for this process
            gpu_metrics = self._get_process_gpu_metrics(pid)
            
            metrics = {
                'cpu_percent': round(cpu_percent, 2),
                'memory_mb': round(memory_mb, 2),
                'memory_percent': round(memory_percent, 2),
                'num_threads': num_threads,
                'gpu_utilization': gpu_metrics.get('gpu_utilization', 0.0),
                'gpu_memory_mb': gpu_metrics.get('gpu_memory_mb', 0.0),
                'gpu_device_id': gpu_metrics.get('gpu_device_id', -1),
                'gpu_power_watts': gpu_metrics.get('gpu_power_watts', 0.0),
                'gpu_power_limit_watts': gpu_metrics.get('gpu_power_limit_watts', 0.0),
                'gpu_temperature_c': gpu_metrics.get('gpu_temperature_c', 0.0)
            }
            
            return metrics
            
        except psutil.NoSuchProcess:
            return None
        except Exception as e:
            self.get_logger().error(f"Error getting metrics for PID {pid}: {e}")
            return None
    
    def _get_process_gpu_metrics(self, pid: int) -> Dict:
        """Get GPU metrics for a specific process on single GPU.
        
        Returns:
            Dictionary with GPU metrics: {
                'gpu_utilization': float,  # GPU device compute utilization (%)
                'gpu_memory_mb': float,    # Per-process GPU memory (MB)
                'gpu_device_id': int,      # GPU device ID (always 0 for single GPU)
                'gpu_power_watts': float,  # GPU device power draw (Watts)
                'gpu_power_limit_watts': float,  # GPU device power limit (Watts)
                'gpu_temperature_c': float  # GPU device temperature (Celsius)
            }
        """
        if not self.gpu_initialized or self.gpu_handle is None:
            return {
                'gpu_utilization': 0.0,
                'gpu_memory_mb': 0.0,
                'gpu_device_id': -1,
                'gpu_power_watts': 0.0,
                'gpu_power_limit_watts': 0.0,
                'gpu_temperature_c': 0.0
            }
        
        try:
            # Get processes running on the GPU
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu_handle)
            except pynvml.NVMLError:
                # Fallback to graphics processes if compute processes fail
                try:
                    processes = pynvml.nvmlDeviceGetGraphicsRunningProcesses(self.gpu_handle)
                except pynvml.NVMLError:
                    return {
                        'gpu_utilization': 0.0,
                        'gpu_memory_mb': 0.0,
                        'gpu_device_id': 0,
                        'gpu_power_watts': 0.0,
                        'gpu_power_limit_watts': 0.0,
                        'gpu_temperature_c': 0.0
                    }
            
            # Find our process
            gpu_memory_mb = 0.0
            found_process = False
            for proc in processes:
                if proc.pid == pid:
                    # Get GPU memory used by this process
                    gpu_memory_mb = proc.usedGpuMemory / (1024 * 1024) if proc.usedGpuMemory else 0.0
                    found_process = True
                    break
            
            # Get device-level metrics (shared across all processes)
            # Get GPU utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_utilization = float(utilization.gpu)
            except pynvml.NVMLError:
                gpu_utilization = 0.0
            
            # Get GPU power consumption
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)  # milliwatts
                gpu_power_watts = power_mw / 1000.0
            except pynvml.NVMLError:
                gpu_power_watts = 0.0
            
            # Get GPU power limit
            try:
                power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle)
                gpu_power_limit_watts = power_limit_mw / 1000.0
            except pynvml.NVMLError:
                gpu_power_limit_watts = 0.0
            
            # Get GPU temperature
            try:
                gpu_temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                gpu_temperature = 0.0
            
            return {
                'gpu_utilization': round(gpu_utilization, 2),
                'gpu_memory_mb': round(gpu_memory_mb, 2),
                'gpu_device_id': 0,  # Single GPU, always device 0
                'gpu_power_watts': round(gpu_power_watts, 2),
                'gpu_power_limit_watts': round(gpu_power_limit_watts, 2),
                'gpu_temperature_c': round(gpu_temperature, 1)
            }
            
        except Exception as e:
            self.get_logger().error(f"Error getting GPU metrics for PID {pid}: {e}")
            return {
                'gpu_utilization': 0.0,
                'gpu_memory_mb': 0.0,
                'gpu_device_id': 0,
                'gpu_power_watts': 0.0,
                'gpu_power_limit_watts': 0.0,
                'gpu_temperature_c': 0.0
            }
    
    def _publish_process_metrics(self, pid: int, node_name: str, metrics: Dict):
        """Publish resource metrics for a process."""
        try:
            msg = String()
            msg.data = json.dumps({
                'pid': pid,
                'node_name': node_name,
                'timestamp': time.time(),
                'metrics': metrics
            })
            self.metrics_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing metrics for PID {pid}: {e}")
    
    def _save_data_to_files(self):
        """Save all collected metrics to a single CSV file."""
        if not self.all_metrics:
            self.get_logger().warn("No data collected to save")
            return
        
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Saving collected data to: {self.data_dir}")
        self.get_logger().info("=" * 80)
        
        # Create single CSV file with all data
        csv_file = self.data_dir / f'resource_{self.index}.csv'
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'process_id', 'timestamp', 'node_name', 'model_name', 'mode',
                'cpu_percent', 'memory_mb', 'memory_percent', 'num_threads',
                'gpu_utilization', 'gpu_memory_mb', 'gpu_device_id',
                'gpu_power_watts', 'gpu_power_limit_watts', 'gpu_temperature_c'
            ])
            
            # Write all records
            for record in self.all_metrics:
                writer.writerow([
                    record['process_id'],
                    record['timestamp'],
                    record['node_name'],
                    record['model_name'],
                    record['mode'],
                    record['cpu_percent'],
                    record['memory_mb'],
                    record['memory_percent'],
                    record['num_threads'],
                    record['gpu_utilization'],
                    record['gpu_memory_mb'],
                    record['gpu_device_id'],
                    record['gpu_power_watts'],
                    record['gpu_power_limit_watts'],
                    record['gpu_temperature_c']
                ])
        
        self.get_logger().info(f"  Saved {len(self.all_metrics)} records to: {csv_file}")
        
        # Calculate and log energy consumption
        self._calculate_energy_consumption()
    
    def _calculate_energy_consumption(self):
        """Calculate total energy consumption for each process."""
        self.get_logger().info("=" * 80)
        self.get_logger().info("Energy Consumption Summary")
        self.get_logger().info("=" * 80)
        
        # Group records by process_id
        process_records = {}
        for record in self.all_metrics:
            pid = record['process_id']
            if pid not in process_records:
                process_records[pid] = []
            process_records[pid].append(record)
        
        # Sort each process's records by timestamp
        for pid in process_records:
            process_records[pid].sort(key=lambda x: x['timestamp'])
        
        total_energy_wh = 0.0
        
        for pid, records in process_records.items():
            node_name = records[0]['node_name']
            model_name = records[0]['model_name']
            
            # Calculate energy using trapezoidal rule
            energy_wh = 0.0
            for i in range(1, len(records)):
                prev_record = records[i-1]
                curr_record = records[i]
                
                prev_power = prev_record['gpu_power_watts']
                curr_power = curr_record['gpu_power_watts']
                prev_time = prev_record['timestamp']
                curr_time = curr_record['timestamp']
                
                # Time delta in hours
                time_delta_hours = (curr_time - prev_time) / 3600.0
                
                # Average power over interval
                avg_power = (prev_power + curr_power) / 2.0
                
                # Energy = Power × Time
                energy_wh += avg_power * time_delta_hours
            
            energy_kwh = energy_wh * 0.001
            total_energy_wh += energy_wh
            
            self.get_logger().info(
                f"  {node_name:30s} (PID {pid:6d}): {energy_wh:8.3f} Wh = {energy_kwh:.6f} kWh ({len(records)} samples)"
            )
        
        total_energy_kwh = total_energy_wh * 0.001
        self.get_logger().info("-" * 80)
        self.get_logger().info(
            f"  {'TOTAL':30s}            : {total_energy_wh:8.3f} Wh = {total_energy_kwh:.6f} kWh"
        )
        self.get_logger().info("=" * 80)
    
    def terminate_callback(self, msg):
        """Handle monitor shutdown request and save all logged data."""
        if msg.data.strip() == "TERMINATE":
            self.get_logger().info("Resource Monitor received TERMINATE command...")
            
            # Stop monitoring
            self.monitoring_enabled = False
            if self.monitoring_timer:
                self.monitoring_timer.cancel()
            
            # Save all collected data to CSV files
            self._save_data_to_files()
            
            # Shutdown GPU monitoring
            if self.gpu_initialized:
                try:
                    pynvml.nvmlShutdown()
                    self.get_logger().info("GPU monitoring shutdown complete")
                except Exception as e:
                    self.get_logger().warn(f"Error shutting down GPU monitoring: {e}")
            
            self.destroy_node()
            raise SystemExit
    
    def get_status(self) -> Dict:
        """Get current monitoring status and metrics."""
        return {
            'monitored_pids': list(self.announced_pids.keys()),
            'monitored_processes': len(self.process_metrics),
            'process_metrics': self.process_metrics,
            'gpu_monitoring_enabled': self.gpu_initialized,
            'monitoring_enabled': self.monitoring_enabled,
            'monitoring_interval': self.monitoring_interval,
            'index': self.index,
            'data_dir': str(self.data_dir),
            'total_records': len(self.all_metrics)
        }


def main(args=None):
    rclpy.init(args=args)
    node = ResourceMonitorNode()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Resource Monitor").info('Shutdown complete')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main() 