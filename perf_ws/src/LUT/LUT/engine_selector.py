from torch.utils._python_dispatch import TorchDispatchMode
from typing import Dict, Any
import torch

from torch.utils.cpp_extension import load
# Build/load the extension (first run compiles; later runs reuse the build cache)
load(
    name="my_cudnn_prof_ext_v2",
    sources=["/mmdetection3d_ros2/perf_ws/src/LUT/LUT/engine_cfg_perf.cpp"],
    with_cuda=True,
    extra_cflags=["-O0", "-g", "-fno-omit-frame-pointer"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
    is_python_module=False,   # <-- critical
)

_SPATIAL_OPS = {
    torch.ops.aten.convolution.default,
    torch.ops.aten._convolution.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv3d.default,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.max_pool3d.default,
    torch.ops.aten.avg_pool3d.default,
    # add more here if you discover other heavy ops
}


class EngineSelector(TorchDispatchMode):
    """
    Intercepts convolution operations and routes them to specific cuDNN engine configs.
    
    This class uses TorchDispatchMode to intercept aten convolution operations and
    replaces them with calls to the cuDNN backend API using pre-selected engine configs.
    
    Args:
        engine_config_map: Dictionary mapping op_id (str or int) to engine_index (int)
                          Example: {"1": 0, "2": 3, "5": 1} means:
                          - 1st conv uses engine config 0
                          - 2nd conv uses engine config 3
                          - 5th conv uses engine config 1
                          Operations not in the map will use the default PyTorch path.
    """

    def __init__(self, engine_config_map: Dict[Any, int]):
        super().__init__()
        # Normalize keys to strings for consistent lookup
        self.engine_config_map = {str(k): v for k, v in engine_config_map.items()}
        self.conv_count = 0
        # If map is empty, we can skip all dispatch overhead
        self.has_any_mapping = bool(self.engine_config_map)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # If no mappings at all, just pass through immediately
        if not self.has_any_mapping:
            return func(*args, **kwargs)

        # Only intercept convolution operations
        if func in _SPATIAL_OPS:
            self.conv_count += 1
            op_id = str(self.conv_count)
            
            # Check if we have an engine config for this op
            engine_index = self.engine_config_map.get(op_id)
            
            if engine_index is not None:
                # Route to our custom engine execution
                return self._execute_with_engine(args, kwargs, engine_index)
        
        # Default: execute normally
        return func(*args, **kwargs)

    def _execute_with_engine(self, args: tuple, kwargs: dict, engine_index: int):
        """
        Execute convolution using a specific engine config.
        
        Args from aten.convolution.default:
            args[0]: input tensor (NCHW)
            args[1]: weight tensor
            args[2]: bias tensor (can be None)
            args[3]: stride
            args[4]: padding
            args[5]: dilation
            args[6]: transposed (bool)
            args[7]: output_padding
            args[8]: groups
        """
        input_tensor = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else [1, 1]
        padding = args[4] if len(args) > 4 else [0, 0]
        dilation = args[5] if len(args) > 5 else [1, 1]
        groups = args[8] if len(args) > 8 else 1
        
        # Ensure inputs are contiguous and float32 (required by our C++ extension)
        input_tensor = input_tensor.contiguous().float()
        weight = weight.contiguous().float()
        if bias is not None:
            bias = bias.contiguous().float()
        
        # Normalize stride/padding/dilation to lists
        stride = self._as_list(stride)
        padding = self._as_list(padding)
        dilation = self._as_list(dilation)
        
        # Call our custom engine selection function
        output = torch.ops.LUT_perf.execute_conv2d_with_engine(
            input_tensor,
            weight,
            bias,
            stride,
            padding,
            dilation,
            int(groups),
            int(engine_index)
        )
        
        return output
    
    @staticmethod
    def _as_list(v):
        """Convert stride/padding/dilation to list of 2 ints."""
        if isinstance(v, (list, tuple)):
            return [int(v[0]), int(v[1])]
        return [int(v), int(v)]
    
    def reset_count(self):
        """Reset the convolution counter."""
        self.conv_count = 0