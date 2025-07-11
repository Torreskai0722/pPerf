#!/usr/bin/env python3
"""
Scheduling Policy Setup Script for pPerf Experiments

This script provides comprehensive control over kernel scheduling policies and CPU scheduling
for performance-critical applications. It supports:

1. Kernel Scheduling Policies:
   - SCHED_FIFO (First In, First Out)
   - SCHED_RR (Round Robin)
   - SCHED_OTHER (Normal scheduling)
   - SCHED_BATCH (Batch processing)
   - SCHED_IDLE (Idle priority)
   - SCHED_DEADLINE (Deadline scheduling)

2. CPU Scheduling Features:
   - Thread affinity (CPU pinning)
   - Process priority (nice values)
   - I/O priority
   - Memory allocation policies

3. Real-time Configuration:
   - RT priority settings
   - CPU isolation
   - Interrupt handling

Author: System
Date: $(date)
"""

import os
import sys
import subprocess
import argparse
import psutil
import threading
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class SchedPolicy(Enum):
    """Kernel scheduling policies"""
    SCHED_OTHER = "other"
    SCHED_FIFO = "fifo"
    SCHED_RR = "rr"
    SCHED_BATCH = "batch"
    SCHED_IDLE = "idle"
    SCHED_DEADLINE = "deadline"

@dataclass
class SchedulingConfig:
    """Configuration for scheduling policies"""
    policy: SchedPolicy
    priority: int
    cpu_affinity: List[int]
    nice_value: int
    io_priority: int
    reset_on_fork: bool = False
    deadline_runtime: Optional[int] = None
    deadline_period: Optional[int] = None
    deadline_deadline: Optional[int] = None

class SchedulingManager:
    """Manages kernel and CPU scheduling policies"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.configs: Dict[str, SchedulingConfig] = {}
        self.applied_configs: Dict[int, SchedulingConfig] = {}
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load scheduling configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            for process_name, config_data in data.items():
                self.configs[process_name] = SchedulingConfig(
                    policy=SchedPolicy(config_data['policy']),
                    priority=config_data['priority'],
                    cpu_affinity=config_data['cpu_affinity'],
                    nice_value=config_data['nice_value'],
                    io_priority=config_data['io_priority'],
                    reset_on_fork=config_data.get('reset_on_fork', False),
                    deadline_runtime=config_data.get('deadline_runtime'),
                    deadline_period=config_data.get('deadline_period'),
                    deadline_deadline=config_data.get('deadline_deadline')
                )
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_config(self, config_file: str):
        """Save current configuration to JSON file"""
        data = {}
        for process_name, config in self.configs.items():
            data[process_name] = {
                'policy': config.policy.value,
                'priority': config.priority,
                'cpu_affinity': config.cpu_affinity,
                'nice_value': config.nice_value,
                'io_priority': config.io_priority,
                'reset_on_fork': config.reset_on_fork,
                'deadline_runtime': config.deadline_runtime,
                'deadline_period': config.deadline_period,
                'deadline_deadline': config.deadline_deadline
            }
        
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_system_info(self) -> Dict:
        """Get system information for scheduling decisions"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'load_avg': os.getloadavg(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent
        }
        return info
    
    def set_process_scheduling(self, pid: int, config: SchedulingConfig) -> bool:
        """Set scheduling policy for a specific process"""
        try:
            # Set kernel scheduling policy
            chrt_cmd = ['chrt']
            
            if config.reset_on_fork:
                chrt_cmd.append('-R')
            
            if config.policy == SchedPolicy.SCHED_DEADLINE:
                chrt_cmd.extend(['-d', str(config.priority)])
                if config.deadline_runtime:
                    chrt_cmd.extend(['-T', str(config.deadline_runtime)])
                if config.deadline_period:
                    chrt_cmd.extend(['-P', str(config.deadline_period)])
                if config.deadline_deadline:
                    chrt_cmd.extend(['-D', str(config.deadline_deadline)])
            else:
                chrt_cmd.extend(['-' + config.policy.value[0], str(config.priority)])
            
            chrt_cmd.extend(['--pid', str(pid)])
            
            result = subprocess.run(chrt_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to set scheduling policy: {result.stderr}")
                return False
            
            # Set CPU affinity
            if config.cpu_affinity:
                affinity_mask = sum(1 << cpu for cpu in config.cpu_affinity)
                taskset_cmd = ['taskset', '-p', str(affinity_mask), str(pid)]
                result = subprocess.run(taskset_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Failed to set CPU affinity: {result.stderr}")
                    return False
            
            # Set process priority (nice value)
            if config.nice_value != 0:
                renice_cmd = ['renice', str(config.nice_value), '-p', str(pid)]
                result = subprocess.run(renice_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Failed to set nice value: {result.stderr}")
                    return False
            
            # Set I/O priority
            if config.io_priority != 0:
                ionice_cmd = ['ionice', '-p', str(pid), '-n', str(config.io_priority)]
                result = subprocess.run(ionice_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Failed to set I/O priority: {result.stderr}")
                    return False
            
            self.applied_configs[pid] = config
            return True
            
        except Exception as e:
            print(f"Error setting process scheduling: {e}")
            return False
    
    def set_thread_scheduling(self, thread_id: int, config: SchedulingConfig) -> bool:
        """Set scheduling policy for a specific thread"""
        try:
            # For threads, we need to use pthread_setschedparam equivalent
            # This is more complex and requires C extensions or specific libraries
            # For now, we'll use the process-level approach
            return self.set_process_scheduling(thread_id, config)
        except Exception as e:
            print(f"Error setting thread scheduling: {e}")
            return False
    
    def apply_config_to_processes(self, process_patterns: List[str]) -> Dict[str, bool]:
        """Apply scheduling configs to processes matching patterns"""
        results = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_name = proc.info['name']
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                for pattern, config in self.configs.items():
                    if pattern in proc_name or pattern in cmdline:
                        success = self.set_process_scheduling(proc.info['pid'], config)
                        results[f"{proc_name}({proc.info['pid']})"] = success
                        break
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return results
    
    def setup_rt_environment(self):
        """Setup real-time environment for optimal performance"""
        try:
            # Check if running with sufficient privileges
            if os.geteuid() != 0:
                print("Warning: Some RT features require root privileges")
            
            # Set CPU governor to performance
            try:
                subprocess.run(['cpupower', 'frequency-set', '-g', 'performance'], 
                             capture_output=True, check=True)
                print("Set CPU governor to performance")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Could not set CPU governor (cpupower not available)")
            
            # Disable CPU frequency scaling
            try:
                for cpu in range(psutil.cpu_count()):
                    scaling_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                    if os.path.exists(scaling_file):
                        with open(scaling_file, 'w') as f:
                            f.write("performance")
            except PermissionError:
                print("Could not disable CPU frequency scaling (requires root)")
            
            # Set real-time limits
            try:
                # Increase real-time priority limit
                subprocess.run(['ulimit', '-r', '99'], check=True)
                print("Set real-time priority limit to 99")
            except subprocess.CalledProcessError:
                print("Could not set real-time priority limit")
            
        except Exception as e:
            print(f"Error setting up RT environment: {e}")
    
    def monitor_scheduling(self, duration: int = 60) -> Dict:
        """Monitor scheduling performance for a duration"""
        start_time = time.time()
        monitoring_data = {
            'start_time': start_time,
            'end_time': start_time + duration,
            'samples': []
        }
        
        def monitor_loop():
            while time.time() < start_time + duration:
                sample = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'load_avg': os.getloadavg(),
                    'processes': {}
                }
                
                # Monitor configured processes
                for pid, config in self.applied_configs.items():
                    try:
                        proc = psutil.Process(pid)
                        sample['processes'][pid] = {
                            'cpu_percent': proc.cpu_percent(),
                            'memory_percent': proc.memory_percent(),
                            'num_threads': proc.num_threads(),
                            'status': proc.status()
                        }
                    except psutil.NoSuchProcess:
                        continue
                
                monitoring_data['samples'].append(sample)
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        monitor_thread.join()
        
        return monitoring_data
    
    def generate_report(self, monitoring_data: Dict) -> str:
        """Generate a performance report from monitoring data"""
        if not monitoring_data['samples']:
            return "No monitoring data available"
        
        report = []
        report.append("=== Scheduling Performance Report ===")
        report.append(f"Duration: {monitoring_data['end_time'] - monitoring_data['start_time']:.2f} seconds")
        report.append(f"Samples: {len(monitoring_data['samples'])}")
        
        # Calculate averages
        cpu_percents = [s['cpu_percent'] for s in monitoring_data['samples']]
        memory_percents = [s['memory_percent'] for s in monitoring_data['samples']]
        
        report.append(f"Average CPU usage: {sum(cpu_percents)/len(cpu_percents):.2f}%")
        report.append(f"Average memory usage: {sum(memory_percents)/len(memory_percents):.2f}%")
        report.append(f"Peak CPU usage: {max(cpu_percents):.2f}%")
        report.append(f"Peak memory usage: {max(memory_percents):.2f}%")
        
        # Process-specific statistics
        if monitoring_data['samples'][0]['processes']:
            report.append("\nProcess Statistics:")
            for pid in monitoring_data['samples'][0]['processes'].keys():
                proc_cpu = [s['processes'][pid]['cpu_percent'] 
                           for s in monitoring_data['samples'] 
                           if pid in s['processes']]
                if proc_cpu:
                    report.append(f"  PID {pid}: Avg CPU {sum(proc_cpu)/len(proc_cpu):.2f}%")
        
        return '\n'.join(report)

def create_default_config() -> Dict[str, SchedulingConfig]:
    """Create default scheduling configuration for pPerf experiments"""
    configs = {}
    
    # High-priority configuration for inference threads
    configs['inference'] = SchedulingConfig(
        policy=SchedPolicy.SCHED_FIFO,
        priority=80,
        cpu_affinity=[0, 1, 2, 3],  # Dedicated cores for inference
        nice_value=-10,
        io_priority=0,  # High I/O priority
        reset_on_fork=True
    )
    
    # Medium priority for data processing
    configs['data_processing'] = SchedulingConfig(
        policy=SchedPolicy.SCHED_RR,
        priority=60,
        cpu_affinity=[4, 5, 6, 7],
        nice_value=-5,
        io_priority=1,
        reset_on_fork=False
    )
    
    # Lower priority for background tasks
    configs['background'] = SchedulingConfig(
        policy=SchedPolicy.SCHED_OTHER,
        priority=0,
        cpu_affinity=[8, 9, 10, 11],
        nice_value=10,
        io_priority=3,
        reset_on_fork=False
    )
    
    # Real-time configuration for critical paths
    configs['realtime'] = SchedulingConfig(
        policy=SchedPolicy.SCHED_FIFO,
        priority=90,
        cpu_affinity=[0, 1],  # Dedicated real-time cores
        nice_value=-20,
        io_priority=0,
        reset_on_fork=True
    )
    
    return configs

def main():
    parser = argparse.ArgumentParser(description='Setup kernel and CPU scheduling policies')
    parser.add_argument('--config', '-c', type=str, help='Configuration file (JSON)')
    parser.add_argument('--save-config', '-s', type=str, help='Save default config to file')
    parser.add_argument('--apply', '-a', action='store_true', help='Apply configurations')
    parser.add_argument('--monitor', '-m', type=int, default=0, help='Monitor for N seconds')
    parser.add_argument('--setup-rt', action='store_true', help='Setup real-time environment')
    parser.add_argument('--process-patterns', nargs='+', help='Process patterns to apply config to')
    parser.add_argument('--pid', type=int, help='Apply config to specific PID')
    parser.add_argument('--policy', type=str, choices=[p.value for p in SchedPolicy], 
                       help='Scheduling policy')
    parser.add_argument('--priority', type=int, help='Priority value')
    parser.add_argument('--cpu-affinity', type=str, help='CPU affinity (comma-separated)')
    parser.add_argument('--nice', type=int, help='Nice value')
    parser.add_argument('--io-priority', type=int, help='I/O priority')
    
    args = parser.parse_args()
    
    # Initialize scheduling manager
    manager = SchedulingManager(args.config)
    
    if args.save_config:
        # Save default configuration
        default_configs = create_default_config()
        manager.configs = default_configs
        manager.save_config(args.save_config)
        print(f"Default configuration saved to {args.save_config}")
        return
    
    if args.setup_rt:
        # Setup real-time environment
        manager.setup_rt_environment()
        print("Real-time environment setup completed")
    
    if args.pid and (args.policy or args.priority or args.cpu_affinity or args.nice or args.io_priority):
        # Apply specific configuration to PID
        config = SchedulingConfig(
            policy=SchedPolicy(args.policy) if args.policy else SchedPolicy.SCHED_OTHER,
            priority=args.priority if args.priority else 0,
            cpu_affinity=[int(x) for x in args.cpu_affinity.split(',')] if args.cpu_affinity else [],
            nice_value=args.nice if args.nice else 0,
            io_priority=args.io_priority if args.io_priority else 0
        )
        
        success = manager.set_process_scheduling(args.pid, config)
        print(f"Applied config to PID {args.pid}: {'Success' if success else 'Failed'}")
    
    if args.apply and args.process_patterns:
        # Apply configurations to matching processes
        results = manager.apply_config_to_processes(args.process_patterns)
        print("Applied configurations:")
        for process, success in results.items():
            print(f"  {process}: {'Success' if success else 'Failed'}")
    
    if args.monitor > 0:
        # Monitor performance
        print(f"Monitoring for {args.monitor} seconds...")
        monitoring_data = manager.monitor_scheduling(args.monitor)
        report = manager.generate_report(monitoring_data)
        print(report)
    
    # Show system information
    if not any([args.save_config, args.setup_rt, args.pid, args.apply, args.monitor]):
        print("=== System Information ===")
        info = manager.get_system_info()
        for key, value in info.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 