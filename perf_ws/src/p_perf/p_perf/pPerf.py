import threading
import time
import functools
import inspect
import os
from typing import Dict, List, Set, Tuple, Any, Optional

import torch
import psutil
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage
)


class pPerf:
    """
    Performance profiler for PyTorch models with GPU/CPU monitoring and NVTX annotation.
    
    This class provides comprehensive profiling capabilities including:
    - Method-level timing and NVTX annotation
    - Pipeline transform profiling
    - GPU utilization and memory monitoring
    - CPU and RAM monitoring
    - Nested method filtering and depth analysis
    """
    
    def __init__(self, 
                 model_name: str, 
                 inferencer: Any, 
                 depth: int, 
                 mode: str = 'lidar', 
                 monitor_interval: float = 0.01, 
                 GPU_monitoring: bool = True, 
                 CPU_monitoring: bool = True, 
                 ms_sync: bool = False):
        """
        Initialize the performance profiler.
        
        Args:
            model_name: Name of the model being profiled
            inferencer: The model inferencer object
            depth: Target depth for NVTX annotation
            mode: Profiling mode ('lidar', 'image')
            monitor_interval: Interval for resource monitoring (seconds)
            GPU_monitoring: Whether to monitor GPU resources
            CPU_monitoring: Whether to monitor CPU/RAM resources
            ms_sync: Whether to use millisecond-level stream synchronization
        """
        # Core configuration
        self.model_name = model_name
        self.inferencer = inferencer
        self.target_depth = depth
        self.mode = mode.lower()
        self.ms_sync = ms_sync
        self.monitor_interval = monitor_interval
        
        # Method profiling state
        self._init_method_profiling_state()
        
        # Pipeline profiling state
        self._init_pipeline_profiling_state()
        
        # Resource monitoring state
        self._init_resource_monitoring_state(GPU_monitoring, CPU_monitoring)
        
        # Profiling workflow state
        self.warming = False

    # ============================================================================
    # INITIALIZATION HELPERS
    # ============================================================================
    
    def _init_method_profiling_state(self):
        """Initialize method profiling related state variables."""
        self.method_timings: Dict[Tuple, Tuple[float, float, str]] = {}
        self.filtered_methods: List[Tuple] = []
        self.module_method_map: Dict[Tuple, Tuple[Any, str, str]] = {}
        self.method_called: Set[Tuple] = set()
        self.method_depths: Dict[Tuple, int] = {}

    def _init_pipeline_profiling_state(self):
        """Initialize pipeline profiling related state variables."""
        self.pipeline_timings: Dict[str, Tuple[float, float, str]] = {}
        self.pipeline_transform_map: Dict[str, Tuple[Any, int, str]] = {}
        self.pipeline_transform_called: Set[str] = set()

    def _init_resource_monitoring_state(self, GPU_monitoring: bool = False, CPU_monitoring: bool = False):
        """Initialize resource monitoring related state variables."""
        # GPU monitoring
        self.gpu_stats: List[Dict[str, Any]] = []
        self.gpu_monitor_thread: Optional[threading.Thread] = None
        self.GPU_monitoring = GPU_monitoring
        
        # CPU monitoring
        self.cpu_stats: List[Dict[str, Any]] = []
        self.cpu_monitor_thread: Optional[threading.Thread] = None
        self.CPU_monitoring = CPU_monitoring

    # ============================================================================
    # CORE PROFILING WORKFLOW
    # ============================================================================
    
    def warm_up(self, warm_data: Any, num_warmups: int = 10):
        """Warm up the model before profiling."""

        for _ in range(num_warmups):
            self.inferencer(warm_data)

    def register_hooks(self, warm_data: Any):
        """Register profiling hooks and prepare for inference."""

        # Wrap pipeline transforms
        self.trace_and_record_pipeline_times(warm_data)
        
        # Trace and record method timings
        self.trace_and_record_times(warm_data)
        
        # Clean up unused hooks
        self._purge_unused_hooks()
        
        # Filter and prepare methods for NVTX annotation
        self.filter_nested_ranges(self.inferencer.model)
        self._unwrap_all_traced_methods()
        self.wrap_filtered_methods_with_nvtx()
        
        max_depth = max(self.method_depths.values(), default=-1)


    def run_inference(self, data: Any, input_name: str) -> Any:
        """Run inference with NVTX annotation."""
        torch.cuda.nvtx.range_push(f'{input_name}.{self.model_name}.e2e')
        
        try:
            result = self.inferencer(data, return_datasamples=True)
            return result
        finally:
            torch.cuda.nvtx.range_pop()

    # ============================================================================
    # MODEL PROFILING
    # ============================================================================
    
    def trace_and_record_times(self, warmup_data: Any):
        """Wrap all model methods and pipeline transforms to record timing."""

        
        # Wrap model methods
        self._wrap_model_methods()
        
        # Run inference to collect timing data
        self.inferencer(warmup_data)

    def _wrap_model_methods(self):
        """Wrap all model methods with timing wrappers."""
        # Build name -> module map
        module_to_name = {}
        model = self.inferencer.model
        for name, module in model.named_modules():
            module_to_name[id(module)] = name

        # Wrap each method
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

    def _trace_wrapper(self, fn: callable, tag: str, method_id: Tuple) -> callable:
        """Create a timing wrapper for a method."""
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            self.method_called.add(method_id)
            
            # Synchronize CUDA if needed
            if self.ms_sync:
                current_stream = torch.cuda.current_stream()
                current_stream.synchronize()
            else:
                torch.cuda.synchronize()
            
            start = time.time()
            result = fn(*args, **kwargs)
            
            if self.ms_sync:
                current_stream = torch.cuda.current_stream()
                current_stream.synchronize()
            else:
                torch.cuda.synchronize()
            
            end = time.time()
            self.method_timings[method_id] = (start, end, tag)
            return result
        return wrapped

    def _unwrap_all_traced_methods(self):
        """Restore all wrapped methods to their original versions."""
        for method_id, (module, name, _) in self.module_method_map.items():
            original = getattr(module, f"_original_{name}", None)
            if original is not None:
                setattr(module, name, original)
        
        # Also unwrap pipeline transforms
        self._unwrap_all_pipeline_transforms()

    def _purge_unused_hooks(self):
        """Remove hooks for methods that were not called during tracing."""
        # Purge unused method hooks
        # unused = set(self.module_method_map.keys()) - self.method_called
        # for method_id in unused:
        #     module, name, _ = self.module_method_map[method_id]
        #     original = getattr(module, f"_original_{name}", None)
        #     if original:
        #         setattr(module, name, original)
        #     self.method_timings.pop(method_id, None)

        # Purge unused pipeline transform hooks
        unused_pipeline = set(self.pipeline_transform_map.keys()) - self.pipeline_transform_called
        for transform_id in unused_pipeline:
            pipeline_obj, idx, _ = self.pipeline_transform_map[transform_id]
            if hasattr(pipeline_obj, '_original_transforms') and idx in pipeline_obj._original_transforms:
                pipeline_obj.transforms[idx] = pipeline_obj._original_transforms[idx]
            self.pipeline_timings.pop(transform_id, None)

        # Update maps to only include used methods/transforms
        self.module_method_map = {
            mid: self.module_method_map[mid] for mid in self.method_called
        }
        self.pipeline_transform_map = {
            tid: self.pipeline_transform_map[tid] for tid in self.pipeline_transform_called
        }

    # ============================================================================
    # PIPELINE PROFILING
    # ============================================================================
    
    def trace_and_record_pipeline_times(self, warmup_data: Any):
        """Wrap all pipeline transforms to record timing."""


        if not hasattr(self.inferencer, 'pipeline'):
            return

        pipeline = self.inferencer.pipeline
        
        # Store original transforms and wrap them
        for i, transform in enumerate(pipeline.transforms):
            transform_id = f"pipeline_transform_{i}"
            tag = f"{self.model_name}.data_preprocessor.pipeline_{i}"
            
            # Store the original transform
            if not hasattr(pipeline, "_original_transforms"):
                setattr(pipeline, "_original_transforms", {})
            pipeline._original_transforms[i] = transform
            
            # Wrap the transform
            wrapped = self._trace_pipeline_wrapper(transform, tag, transform_id)
            pipeline.transforms[i] = wrapped
            self.pipeline_transform_map[transform_id] = (pipeline, i, tag)

    def _trace_pipeline_wrapper(self, fn: callable, tag: str, transform_id: str) -> callable:
        """Create a timing wrapper for a pipeline transform."""
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            self.pipeline_transform_called.add(transform_id)
            start = time.time()
            result = fn(*args, **kwargs)
            end = time.time()
            self.pipeline_timings[transform_id] = (start, end, tag)
            return result
        return wrapped

    def _unwrap_all_pipeline_transforms(self):
        """Restore all pipeline transforms to their original versions."""
        if not hasattr(self.inferencer, 'pipeline'):
            return
            
        pipeline = self.inferencer.pipeline
        if hasattr(pipeline, '_original_transforms'):
            for transform_id, (pipeline_obj, idx, _) in self.pipeline_transform_map.items():
                if idx in pipeline_obj._original_transforms:
                    pipeline_obj.transforms[idx] = pipeline_obj._original_transforms[idx]
                else:
                    print(f'[AutoProfiler] PIPELINE UNWRAP FAILED for {transform_id}')

    # ============================================================================
    # METHOD FILTERING AND DEPTH ANALYSIS
    # ============================================================================
    
    def filter_nested_ranges(self, model: torch.nn.Module, tolerance_ms: int = 1):
        """
        Filter overlapping methods and compute nesting depths.
        
        Args:
            model: The model to analyze
            tolerance_ms: Time tolerance for considering methods overlapping (milliseconds)
        """


        # Get valid modules (top-level children and their descendants)
        valid_modules = self._get_valid_modules(model)
        
        # Prepare timing ranges
        ranges = self._prepare_timing_ranges()
        
        # # Apply time-based filtering
        # keep_set = self._apply_time_based_filtering(ranges, tolerance_ms)
        
        # # Apply source-based filtering
        # filtered = self._apply_source_based_filtering(keep_set, valid_modules)

        filtered = [range[0] for range in ranges]
        
        # Compute nesting depths
        self._compute_nesting_depths(filtered)
        
        self.filtered_methods = filtered

    def _get_valid_modules(self, model: torch.nn.Module) -> Set[int]:
        """Get set of valid module IDs (top-level children and their descendants)."""
        valid_modules = set()
        for _, child in model.named_children():
            for submodule in child.modules():
                valid_modules.add(id(submodule))
        return valid_modules

    def _prepare_timing_ranges(self) -> List[Tuple]:
        """Prepare timing ranges from both model methods and pipeline transforms."""
        ranges = []
        
        # Add model method timings
        for method_id, (start, end, tag) in self.method_timings.items():
            ranges.append((method_id, start, end, tag))
        
        # Add pipeline transform timings
        for transform_id, (start, end, tag) in self.pipeline_timings.items():
            ranges.append((transform_id, start, end, tag))
        
        return ranges

    def _apply_time_based_filtering(self, ranges: List[Tuple], tolerance_ms: int) -> Set:
        """Apply time-based filtering to remove overlapping methods."""
        # Sort ranges by start time first
        sorted_ranges = sorted(ranges, key=lambda x: x[1])  # Sort by start time (index 1)
        keep_set = set(item[0] for item in sorted_ranges)
        tolerance_sec = tolerance_ms / 1000.0  # Convert to seconds

        for i in range(len(sorted_ranges)):
            id_i, start_i, end_i, _ = sorted_ranges[i]
            
            # Only check ranges that could potentially overlap
            for j in range(i + 1, len(sorted_ranges)):
                id_j, start_j, end_j, _ = sorted_ranges[j]
                
                # Early termination: if start_j is too far from start_i, 
                # all subsequent ranges will be even further
                if start_j - start_i > tolerance_sec:
                    break
                    
                start_diff = abs(start_i - start_j)
                end_diff = abs(end_i - end_j)

                if start_diff < tolerance_sec and end_diff < tolerance_sec:
                    if start_i <= start_j and end_i >= end_j:
                        keep_set.discard(id_j)
                    elif start_j <= start_i and end_j >= end_i:
                        keep_set.discard(id_i)

        return keep_set

    def _apply_source_based_filtering(self, keep_set: Set, valid_modules: Set[int]) -> List:
        """Apply source-based filtering to keep only valid modules."""
        filtered = []
        for item_id in keep_set:
            if item_id in self.module_method_map:
                # This is a model method
                module, _, _ = self.module_method_map[item_id]
                if id(module) in valid_modules:
                    filtered.append(item_id)
            else:
                # This is a pipeline transform, keep it
                filtered.append(item_id)
        return filtered

    def _compute_nesting_depths(self, filtered: List):
        """Compute nesting depths for filtered methods."""
        # Prepare timings for depth computation
        timings = {}
        
        for method_id in filtered:
            if method_id in self.method_timings:
                timings[method_id] = self.method_timings[method_id]
        
        for transform_id in filtered:
            if transform_id in self.pipeline_timings:
                timings[transform_id] = self.pipeline_timings[transform_id]

        # Sort by start time and compute depths
        sorted_timings = sorted(timings.items(), key=lambda x: x[1][0])
        
        self.method_depths = {}
        active_stack = []

        for method_id, (start, end, _) in sorted_timings:
            while active_stack and active_stack[-1][1] <= start:
                active_stack.pop()

            depth = len(active_stack)
            self.method_depths[method_id] = depth
            active_stack.append((method_id, end))

    # ============================================================================
    # NVTX ANNOTATION
    # ============================================================================
    
    def wrap_filtered_methods_with_nvtx(self):
        """Wrap filtered methods at target depth with NVTX markers.
        
        Note: Pipeline transforms are always wrapped regardless of depth.
        """
        count = 0
        for method_id in self.filtered_methods:
            if method_id in self.module_method_map:
                # This is a model method - apply depth check
                # if self.method_depths[method_id] != self.target_depth:
                #     continue
                module, name, tag = self.module_method_map[method_id]
                original = getattr(module, name)
                setattr(module, name, self._nvtx_wrapper(original, tag))
                count += 1
            else:
                # This is a pipeline transform - always wrap regardless of depth
                try:
                    pipeline_obj, idx, tag = self.pipeline_transform_map[method_id]
                    original = pipeline_obj.transforms[idx]
                    pipeline_obj.transforms[idx] = self._nvtx_wrapper(original, tag)
                    count += 1
                except:
                    print([key for key in self.pipeline_transform_map.keys()])
                    print(method_id, 'not found in pipeline_transform_map')
                    pass
        


    def _nvtx_wrapper(self, fn: callable, tag: str) -> callable:
        """Create an NVTX wrapper for a function."""
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if self.ms_sync:
                current_stream = torch.cuda.current_stream()
                current_stream.synchronize()
            else:
                torch.cuda.synchronize()
            
            torch.cuda.nvtx.range_push(tag)
            result = fn(*args, **kwargs)
            
            if self.ms_sync:
                current_stream = torch.cuda.current_stream()
                current_stream.synchronize()
            else:
                torch.cuda.synchronize()
            
            torch.cuda.nvtx.range_pop()
            return result
        return wrapped

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def summary(self):
        """Print a summary of the profiling results."""
        print("\n[AutoProfiler] Final NVTX-annotated methods by depth:")
        max_depth = max(self.method_depths.values(), default=-1)
        
        for d in range(max_depth + 1):
            print(f"\n  --- Depth {d} ---")
            for method_id in self.filtered_methods:
                if self.method_depths[method_id] == d:
                    if method_id in self.module_method_map:
                        _, _, tag = self.module_method_map[method_id]
                        timing = self.method_timings[method_id]
                    else:
                        _, _, tag = self.pipeline_transform_map[method_id]
                        timing = self.pipeline_timings[method_id]
                    print(f"    {tag}: {timing}")

    def get_profiling_stats(self) -> Dict[str, Any]:
        """Get comprehensive profiling statistics."""
        return {
            'model_name': self.model_name,
            'mode': self.mode,
            'target_depth': self.target_depth,
            'total_methods_traced': len(self.method_timings),
            'total_pipeline_transforms_traced': len(self.pipeline_timings),
            'filtered_methods': len(self.filtered_methods),
            'max_depth': max(self.method_depths.values(), default=-1),
            'gpu_stats_count': len(self.gpu_stats),
            'cpu_stats_count': len(self.cpu_stats),
        }