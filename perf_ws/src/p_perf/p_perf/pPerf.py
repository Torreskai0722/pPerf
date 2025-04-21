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

    def is_forward_overridden(self, module):
        return 'forward' in module.__class__.__dict__

    def register_hooks(self, model):
        if self.target_depth < 0:
            return

        queue = deque([(model, f"{self.model_name}", 0)])

        while queue:
            submodule, prefix, depth = queue.popleft()

            if depth == self.target_depth:
                for child_name, child in submodule.named_children():
                    if not self.is_forward_overridden(child):
                        continue

                    full_name = f"{prefix}.{child_name}"
                    child.hook_name = full_name
                    pre_handle = child.register_forward_pre_hook(self._time_start_hook)
                    post_handle = child.register_forward_hook(self._time_end_hook)
                    self.hook_handles.extend([pre_handle, post_handle])

            elif depth < self.target_depth:
                for child_name, child in submodule.named_children():
                    full_name = f"{prefix}.{child_name}" if prefix else child_name
                    queue.append((child, full_name, depth + 1))

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

    # -------------- Auto Wrapping Missed Methods ----------------

    def find_non_forward_methods(self, module):
        """
        Identify non-forward, public methods defined in the module's class
        that may be missed by PyTorch forward hooks.
        """
        interesting = []
        module_cls = module.__class__

        for name, fn in inspect.getmembers(module, predicate=inspect.ismethod):
            if name.startswith('_') or name in ['forward', '__call__']:
                continue
            if name in module_cls.__dict__:
                interesting.append(name)

        return interesting


    def wrap_module_method_with_nvtx(self, module, method_name, nvtx_name):
        original_method = getattr(module, method_name)
        marker = f'_original_{method_name}'
        if hasattr(module, marker):
            return  # already wrapped

        setattr(module, marker, original_method)

        @functools.wraps(original_method)
        def wrapped_method(*args, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(nvtx_name)
            result = original_method(*args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return result

        setattr(module, method_name, wrapped_method)

    def auto_patch_missed_methods(self, model):
        for module_name, submodule in model.named_modules():
            method_names = self.find_non_forward_methods(submodule)
            for method in method_names:
                nvtx_tag = f"{self.model_name}.{module_name}.{method}"
                self.wrap_module_method_with_nvtx(submodule, method, nvtx_tag)