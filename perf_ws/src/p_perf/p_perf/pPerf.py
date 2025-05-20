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
import functools
import inspect
import time


class pPerf:
    def __init__(self, model_name, inferencer, depth, monitor_interval=0.1):
        self.model_name = model_name
        self.method_timings = {}  # {method_id: (start, end, tag)}
        self.filtered_methods = []  # methods selected after filtering
        self.module_method_map = {}  # for looking up method handles
        self.target_depth = depth

        # Model inferencing
        self.inferencer = inferencer

        # GPU monitoring
        self.gpu_stats = []
        self.gpu_monitor_thread = None
        self.monitoring = False
        self.monitor_interval = monitor_interval  # seconds
        self.warming = False

    def _trace_wrapper(self, fn, tag, method_id):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.time()
            self.method_timings[method_id] = (start, end, tag)
            return result
        return wrapped

    def trace_and_record_times(self, warmup_data):
        """
        Wraps all model methods, runs inference, and records time per method.
        Uses full module names like 'backbone.middle_encoder.conv1'.
        """
        print("[AutoProfiler] Wrapping all model methods to record timing...")

        # Build name -> module map (like model.backbone.middle_encoder)
        module_to_name = {}
        model = self.inferencer.model
        for name, module in model.named_modules():
            module_to_name[id(module)] = name  # id ensures match with method_id

        for module in model.modules():
            module_id = id(module)
            qualified_name = module_to_name.get(module_id, module.__class__.__name__)
            for name, method in inspect.getmembers(module, predicate=inspect.ismethod):
                method_id = (module_id, name)
                tag = f"{self.model_name}.{qualified_name}.{name}"
                marker_name = f"_original_{name}"

                if not hasattr(module, marker_name):
                    setattr(module, marker_name, method)
                    wrapped = self._trace_wrapper(method, tag, method_id)
                    setattr(module, name, wrapped)
                    self.module_method_map[method_id] = (module, name, tag)

        print("[AutoProfiler] Running input to measure execution time...")
        self.inferencer(warmup_data)
        print(f"[AutoProfiler] Recorded {len(self.method_timings)} method timings.")
        

    def filter_nested_ranges(self, model, tolerance_ms=0.5):
        """
        Filters overlapping methods (within tolerance_ms),
        and keeps only methods belonging to model's top-level children and their descendants.
        """
        print(f"[AutoProfiler] Filtering methods within {tolerance_ms} ms...")

        # Step 1: Get all valid modules = children + their descendants
        valid_modules = set()
        for _, child in model.named_children():
            for submodule in child.modules():
                valid_modules.add(id(submodule))

        # Step 2: Prepare timing ranges
        ranges = [
            (method_id, start, end, tag)
            for method_id, (start, end, tag) in self.method_timings.items()
        ]
        keep_set = set(method_id for method_id, *_ in ranges)

        # Step 3: Time-based filtering
        for i in range(len(ranges)):
            id_i, start_i, end_i, _ = ranges[i]
            for j in range(i + 1, len(ranges)):
                id_j, start_j, end_j, _ = ranges[j]

                start_diff = abs((start_i - start_j) * 1000)
                end_diff = abs((end_i - end_j) * 1000)

                if start_diff < tolerance_ms and end_diff < tolerance_ms:
                    if start_i <= start_j and end_i >= end_j:
                        keep_set.discard(id_j)
                    elif start_j <= start_i and end_j >= end_i:
                        keep_set.discard(id_i)

        # Step 4: Source-based filtering
        filtered = []
        for method_id in keep_set:
            module, _, _ = self.module_method_map[method_id]
            if id(module) in valid_modules:
                filtered.append(method_id)

        # Step 5: Compute nesting depth using only filtered methods
        timings = {
            method_id: (start, end, tag)
            for method_id, (start, end, tag) in self.method_timings.items()
            if method_id in filtered  # use filtered, not keep_set
        }

        # Sort filtered methods by start time
        sorted_timings = sorted(timings.items(), key=lambda x: x[1][0])

        # Compute depth using a nesting stack
        self.method_depths = {}  # Maps method_id -> depth
        active_stack = []

        for method_id, (start, end, _) in sorted_timings:
            while active_stack and active_stack[-1][1] <= start:
                active_stack.pop()

            depth = len(active_stack)
            self.method_depths[method_id] = depth
            active_stack.append((method_id, end))

        self.filtered_methods = filtered
        print(f"[AutoProfiler] Kept {len(filtered)} methods after time + module filtering with depth info.")

    def _nvtx_wrapper(self, fn, tag):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(tag)
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return result
        return wrapped

    def wrap_filtered_methods_with_nvtx(self):
        """
        Wraps only filtered methods at the target depth with NVTX markers.
        """
        count = 0
        for method_id in self.filtered_methods:
            if self.method_depths[method_id] != self.target_depth:
                continue
            module, name, tag = self.module_method_map[method_id]
            original = getattr(module, name)
            setattr(module, name, self._nvtx_wrapper(original, tag))
            count += 1


    def summary(self):
        print("\n[AutoProfiler] Final NVTX-annotated methods by depth:")
        max_depth = max(self.method_depths.values(), default=-1)
        for d in range(max_depth + 1):
            print(f"\n  --- Depth {d} ---")
            for method_id in self.filtered_methods:
                if self.method_depths[method_id] == d:
                    _, _, tag = self.module_method_map[method_id]
                    print(f"{tag}")

    # ACTUAL METHODS USED IN INFERENCER
    # During inferencing, the pipeline should be 
    # warm_up ==> register_hooks ==> run_inference
    
    def warm_up(self, warm_data, num_warmups=10):
        for _ in range(num_warmups):
            self.inferencer(warm_data)

    def register_hooks(self, warm_data):
        self.trace_and_record_times(warm_data)
        self.filter_nested_ranges(self.inferencer.model)
        self.wrap_filtered_methods_with_nvtx()
        print("MAX_DEPTH is: ", max(self.method_depths.values(), default=-1))
    
    def run_inference(self, data, input_name):
        torch.cuda.nvtx.range_push(f'{input_name}.{self.model_name}.e2e')
        result = self.inferencer(data)
        torch.cuda.nvtx.range_pop()
        return result
    
    # GPU Profiling
    def start_gpu_monitoring(self):
        # If self monitoring is false, skip both start and stop
        if self.monitoring == False:
            return
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
        if self.monitoring == False:
            return
        self.monitoring = False
        if self.gpu_monitor_thread is not None:
            self.gpu_monitor_thread.join()
        nvmlShutdown()