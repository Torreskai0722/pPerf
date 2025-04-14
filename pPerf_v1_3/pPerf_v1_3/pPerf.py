
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
from collections import deque
import os

class pPerf:
    def __init__(self, model_name, target_depth, monitor_interval=0):
        """
        Performance Profiler for Deep Learning Models.

        :param model_name: Name of the model being profiled.
        :param target_depth: The depth at which hooks should be registered.
        """
        self.hook_handles = []
        self.timing = {}  # Dictionary to store inference times
        self.model_name = model_name
        self.target_depth = target_depth
        self.gpu_stats = []
        self.gpu_monitor_thread = None
        self.monitoring = False
        self.monitor_interval = monitor_interval  # seconds


    def _time_start_hook(self, module, input):
        """Hook to start the timer at the beginning of the forward pass."""
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f'{module.hook_name}')
        # print(f'{module.hook_name} start')

    def _time_end_hook(self, module, input, output):
        """Hook to end the timer at the end of the forward pass."""
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        # print(f'{module.hook_name} end')

    def wrap_roi_head_predict(self, roi_head, input_name):
        """Wraps roi_head.predict() with NVTX push/pop for profiling and restores after inference."""
        if hasattr(roi_head, "predict"):

            # If already wrapped, retrieve the original function
            if hasattr(roi_head, "_original_predict"):
                original_predict = roi_head._original_predict
            else:
                original_predict = roi_head.predict  # enter here if no been wrapped yet
                roi_head._original_predict = original_predict

            @functools.wraps(original_predict)
            def nvtx_wrapped_predict(*args, **kwargs):
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f'{input_name}.{self.model_name}.roi_head')
                output = original_predict(*args, **kwargs)  # Call original predict
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                return output

            # Store the original function inside roi_head
            roi_head.predict = nvtx_wrapped_predict


    def register_hooks(self, model):
        """
        Registers hooks at the specified depth using breadth-first traversal.

        :param model: The root model.
        :param input: The understand different
        """
        if self.target_depth < 0:
            return

        queue = deque([(model, f"{self.model_name}", 0)])  # (submodule, prefix, current_depth)

        while queue:
            submodule, prefix, depth = queue.popleft()

            if depth == self.target_depth:
                for child_name, child in submodule.named_children():
                    full_name = f"{prefix}.{child_name}"
                    child.hook_name = full_name  # Assign a unique identifier

                    pre_handle = child.register_forward_pre_hook(self._time_start_hook)
                    post_handle = child.register_forward_hook(self._time_end_hook)

                    self.hook_handles.extend([pre_handle, post_handle])

            elif depth < self.target_depth:
                for child_name, child in submodule.named_children():
                    full_name = f"{prefix}.{child_name}" if prefix else child_name
                    queue.append((child, full_name, depth + 1))

    def unregister_hooks(self):
        """Unregister all hooks by removing each handle."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()  # Clear the handles after unregistering


    def warm_up(self, inferencer, warm_data, mode, num_warmups=10,):
        """
        Perform a model warm-up phase to stabilize inference timing.
        :param inferencer: The inference engine.
        :param warm_data: The warm-up data file.
        :param num_warmups: Number of warm-up iterations.
        """
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
        """
        Runs inference while tracking execution time using NVTX profiling.

        :param inferencer: The inference engine.
        :param lidar_files: List of Lidar point cloud files.
        """

        if "pv_rcnn" in self.model_name and self.target_depth == 0:
            self.wrap_roi_head_predict(inferencer.model.roi_head, input_name)

        torch.cuda.nvtx.range_push(f'{input_name}.{self.model_name}.e2e')
        inferencer(data)
        torch.cuda.nvtx.range_pop()

    def start_gpu_monitoring(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0

        def monitor():
            while self.monitoring:
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W

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