import threading
import time
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage
)
import torch
import functools
import inspect
from collections import deque
import types
from addict import Dict as AddictDict
from mmengine.config import ConfigDict


class pPerf:
    def __init__(self, model_name, target_depth, monitor_interval=0):
        self.hook_handles = []
        self.model_name = model_name
        self.target_depth = target_depth
        self.gpu_stats = []
        self.gpu_monitor_thread = None
        self.monitoring = False
        self.monitor_interval = monitor_interval  # seconds
        self.warming = False

    def _time_start_hook(self, module, input):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f'{module.hook_name}')

    def _time_end_hook(self, module, input, output):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    def is_hookable(self, module):
        """Returns True if the module is not a container and performs computation."""
        container_types = (torch.nn.ModuleList, torch.nn.ModuleDict)
        return not isinstance(module, container_types)

    def register_hooks(self, model):
        if self.target_depth < 0:
            return

        visited = set()
        queue = deque([(model, self.model_name, 0)])

        while queue:
            current_module, current_prefix, depth = queue.popleft()

            # Skip already visited modules
            if current_prefix in visited:
                continue
            visited.add(current_prefix)

            # Only register hooks if we're at the target depth
            if depth == self.target_depth:
                current_module.hook_name = current_prefix
                print(f"Registering hook for: {current_prefix}")
                pre_handle = current_module.register_forward_pre_hook(self._time_start_hook)
                post_handle = current_module.register_forward_hook(self._time_end_hook)
                self.hook_handles.extend([pre_handle, post_handle])
                continue  # Do not expand further

            # Expand children
            for child_name, child_module in current_module.named_children():
                child_prefix = f"{current_prefix}.{child_name}"

                if self.is_hookable(child_module):
                    # Regular module: enqueue for hook or further traversal
                    queue.append((child_module, child_prefix, depth + 1))
                else:
                    # Container module: recurse one level deeper
                    for grand_name, grand_child in child_module.named_children():
                        grand_prefix = f"{child_prefix}.{grand_name}"
                        queue.append((grand_child, grand_prefix, depth + 1))


    def unregister_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def warm_up(self, inferencer, warm_data, mode, num_warmups=10):
        self.warming = True
        for _ in range(num_warmups):
            if mode == 'lidar':
                inferencer(dict(points=warm_data))
            elif mode == 'image':
                inferencer(warm_data)
            else:
                print("Mode is invalid")
                return
        self.warming = False

    def run_inference(self, inferencer, data, input_name):
        
        torch.cuda.nvtx.range_push(f'{input_name}.{self.model_name}.e2e')
        result = inferencer(data)
        torch.cuda.nvtx.range_pop()
        return result

    def start_gpu_monitoring(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        def monitor():
            while self.monitoring:
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0

                self.gpu_stats.append({
                    'time': time.time(),
                    'gpu_util': util.gpu,
                    'mem_util': util.memory,
                    'mem_used_MB': mem.used / (1024 ** 2),
                    'power_W': power
                })
                time.sleep(self.monitor_interval)

        self.monitoring = True
        self.gpu_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.gpu_monitor_thread.start()

    def stop_gpu_monitoring(self):
        self.monitoring = False
        if self.gpu_monitor_thread is not None:
            self.gpu_monitor_thread.join()
        nvmlShutdown()
