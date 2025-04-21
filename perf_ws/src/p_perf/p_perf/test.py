import numpy as np
from mmdet3d.apis import LidarDet3DInferencer, inference_multi_modality_detector, init_model
from mmdet.apis import DetInferencer
from utils import list_filenames
import mmcv
import matplotlib.pyplot as plt
from mmdet.visualization import DetLocalVisualizer
import os

lidar_paths = list_filenames('/mmdetection3d_ros2/data/v1.0-mini/sweeps/LIDAR_TOP', 'bin')

inferencer = LidarDet3DInferencer('pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d')


import sys
import torch
import functools
import inspect

class AutoProfiler:
    def __init__(self, model_name="model"):
        self.called_cuda_methods = set()
        self.model_name = model_name
        self.enabled = False

    def _trace_calls(self, frame, event, arg):
        if event != "call":
            return

        code = frame.f_code
        func_name = code.co_name
        filename = code.co_filename
        if "site-packages" in filename or "<frozen" in filename:
            return

        # Check for CUDA tensors in local variables
        for val in frame.f_locals.values():
            try:
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    module = frame.f_globals.get("__name__")
                    key = (module, func_name)
                    self.called_cuda_methods.add(key)
                    break
            except Exception:
                continue

    def profile_model(self, model, inferencer, input_data):
        """Runs the model once to trace CUDA-using functions."""
        print("[AutoProfiler] Starting warmup trace...")
        self.called_cuda_methods.clear()
        sys.setprofile(self._trace_calls)
        inferencer(input_data) 
        sys.setprofile(None)
        print(f"[AutoProfiler] Found {len(self.called_cuda_methods)} functions using CUDA.")

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

    def wrap_traced_methods(self, model):
        """Wraps methods that were traced as CUDA-using with NVTX profiling."""
        print("[AutoProfiler] Wrapping traced methods with NVTX...")
        count = 0
        for module in model.modules():
            module_name = module.__class__.__name__
            for name, method in inspect.getmembers(module, predicate=inspect.ismethod):
                key = (method.__module__, name)
                if key in self.called_cuda_methods:
                    tag = f"{self.model_name}.{module_name}.{name}"
                    marker_name = f"_original_{name}"
                    if not hasattr(module, marker_name):
                        setattr(module, marker_name, method)
                        setattr(module, name, self._nvtx_wrapper(method, tag))
                        count += 1
        print(f"[AutoProfiler] Wrapped {count} methods.")

    def print_traced_methods(self):
        if not self.called_cuda_methods:
            print("[AutoProfiler] No CUDA-using methods traced.")
            return

        print("\n[AutoProfiler] CUDA-using methods:")
        for module_name, method_name in sorted(self.called_cuda_methods):
            print(f"  - {module_name}.{method_name}")



model = inferencer.model
points = np.fromfile(lidar_paths[0], dtype=np.float32).reshape(-1, 5)
input_tensor = dict(points=np.array(points, dtype=np.float32))  # or your actual test input

profiler = AutoProfiler(model_name="centerpoint")

# Step 1: Trace methods that touch CUDA
profiler.profile_model(model, inferencer, input_tensor)

# Step2:
profiler.print_traced_methods()

# Step 3: Wrap only those methods with NVTX
profiler.wrap_traced_methods(model)

for path in lidar_paths[1:10]:
    input_name = os.path.basename(path).split('.')[0]
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    input_tensor = dict(points=np.array(points, dtype=np.float32)) 

    torch.cuda.nvtx.range_push(f'{input_name}.e2e')
    inferencer(input_tensor)
    torch.cuda.nvtx.range_pop()


