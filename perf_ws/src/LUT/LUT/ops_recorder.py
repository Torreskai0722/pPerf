from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any, Dict, List
import torch
import inspect
from LUT.utils import deepcopy_to_cpu

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

_MM_PREFIXES = ("mmdet", "mmdet3d", "mmseg")

class OpRecorder(TorchDispatchMode):
    """
    Intercepts PyTorch ops, but only RECORDS spatial ops (conv/pool/etc.).

    For recorded ops, stores:
      - 'func'  : aten op handle
      - 'args'  : deep-copied args (tensors on CPU)
      - 'kwargs': deep-copied kwargs (tensors on CPU)
      - 'shapes': list of tensor shapes (for quick inspection)
      - 'caller': optional mmdet/mmdet3d/mmseg caller info
      - 'op_id' : unique identifier - the global count of the convolution operator
    """

    def __init__(self, record_caller: bool = False, max_tensors_per_op: int = 4):
        super().__init__()
        self.record_caller = record_caller
        self.max_tensors_per_op = max_tensors_per_op
        self.events: List[Dict[str, Any]] = []
        # Counters for op_id generation
        self.global_conv_count = 0
        self.module_conv_counts: Dict[str, int] = {}  # module_name -> count

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in _SPATIAL_OPS:
            tensors = list(OpRecorder._gather_tensors((args, kwargs)))
            shapes = [tuple(t.shape) for t in tensors[: self.max_tensors_per_op]]

            caller = self._find_mm_caller() if self.record_caller else None

            # Generate op_id for convolution operations
            op_id = None
            if func.__name__.startswith("convolution"):
                self.global_conv_count += 1
                # op_id is just the global count
                op_id = str(self.global_conv_count)

            args_copy = deepcopy_to_cpu(args)
            kwargs_copy = deepcopy_to_cpu(kwargs)

            self.events.append({
                "op_name": func.__name__,   # e.g. "convolution.default"
                "func": func,               # aten op handle
                "shapes": shapes,
                "num_tensors": len(tensors),
                "args": args_copy,
                "kwargs": kwargs_copy,
                "caller": caller,
                "op_id": op_id,             # op_id for convolution ops, None for others
            })

        return func(*args, **kwargs)


    def _find_mm_caller(self):
        """
        Find the innermost Python frame whose module looks like
        mmdet/mmdet3d/mmseg. Returns {} if none is found.
        """
        stack = inspect.stack()
        caller_info: Dict[str, Any] = {}

        # [0] is __torch_dispatch__; start from [1]
        for frame_info in stack[1:]:
            frame = frame_info.frame
            module_name = frame.f_globals.get("__name__", "")
            filename = frame_info.filename or ""

            if module_name.startswith(_MM_PREFIXES):
                caller_info = {
                    "module": module_name,
                    "function": frame_info.function,
                    "file": filename,
                    "lineno": frame_info.lineno,
                }
                break

        del stack
        return caller_info or None
    
    @staticmethod
    def _gather_tensors(obj):
        """Recursively yield all tensors inside a nested structure."""
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                yield from OpRecorder._gather_tensors(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                yield from OpRecorder._gather_tensors(x)

    @staticmethod
    def summarize_arg(arg, indent=0):
        pad = "  " * indent
        if isinstance(arg, torch.Tensor):
            print(f"{pad}Tensor(shape={tuple(arg.shape)}, dtype={arg.dtype}, device={arg.device})")
        elif isinstance(arg, (list, tuple)):
            print(f"{pad}{type(arg).__name__}(len={len(arg)})")
            for i, x in enumerate(arg):
                print(f"{pad}  [{i}]:")
                OpRecorder.summarize_arg(x, indent + 2)
        elif isinstance(arg, dict):
            print(f"{pad}dict(len={len(arg)})")
            for k, v in arg.items():
                print(f"{pad}  {k!r}:")
                OpRecorder.summarize_arg(v, indent + 2)
        else:
            # Non-tensor leaf: just type + repr truncated
            r = repr(arg)
            if len(r) > 80:
                r = r[:77] + "..."
            print(f"{pad}{type(arg).__name__}: {r}")
