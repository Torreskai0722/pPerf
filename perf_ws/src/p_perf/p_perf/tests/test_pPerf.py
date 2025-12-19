from p_perf.pPerf_inferencer import pPerf2dDetInferencer, pPerf3dDetInferencer, bddDetInferencer, bddSegInferencer, pPerf3dSegInferencer
from p_perf.pPerf import pPerf
import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet3d.apis import LidarDet3DInferencer, LidarSeg3DInferencer
from mmengine.dataset import Compose

from p_perf.general_utils import convert_to_kitti_ros, visualize_segmentation_with_legend

WARM_IMAGE_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'
WARM_IMAGE = cv2.imread(WARM_IMAGE_PATH)
WARM_PCD_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_PCD = np.fromfile(WARM_PCD_PATH, dtype=np.float32).reshape(-1, 5)


import time
from typing import Any, Dict, List

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from op_sched.spatial_tile import SpatiallyTiledModule

from torch.utils.cpp_extension import load

# Build/load the extension (first run compiles; later runs reuse the build cache)
load(
    name="my_cudnn_prof_ext",
    sources=["/mmdetection3d_ros2/perf_ws/src/p_perf/p_perf/tests/my_cudnn_profile.cpp"],
    with_cuda=True,
    extra_cflags=["-O0", "-g", "-fno-omit-frame-pointer"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
    is_python_module=False,   # <-- critical
)

# torch.backends.cuda.matmul.allow_tf32 = True   # or False
# torch.backends.cudnn.allow_tf32 = True         # or False
# torch.backends.cudnn.benchmark = True          # affects algo selection in PyTorch paths
# torch.backends.cudnn.deterministic = False     # affects allowed algos

def _as_pair(v):
    # conv.stride/padding/dilation may be int or tuple
    if isinstance(v, (list, tuple)):
        return [int(v[0]), int(v[1])]
    return [int(v), int(v)]

def profile_conv_from_module(conv, x, max_engine_variants=25, iters=1):
    # Ensure inputs match your C++ checks: CUDA, contiguous, float32, NCHW
    x = x.contiguous().float()
    w = conv.weight.contiguous().float()
    b = conv.bias
    if b is None:
        b = torch.zeros((w.shape[0],), device=w.device, dtype=torch.float32)
    else:
        b = b.contiguous().float()

    stride   = _as_pair(conv.stride)
    padding  = _as_pair(conv.padding)
    dilation = _as_pair(conv.dilation)
    groups   = int(conv.groups)
    

    # Call your extension op
    y, engine_table = torch.ops.my_cudnn_prof.profile_conv2d(
        x, w, b, stride, padding, dilation, groups, int(max_engine_variants), int(iters)
    )
    return y, engine_table

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

# Whitelist of spatial ops you care about (extend as needed)
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

def _is_spatial_op(func) -> bool:
    """Return True only for spatial ops we want to record."""
    return func in _SPATIAL_OPS


def _gather_tensors(obj):
    """Recursively yield all tensors inside a nested structure."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _gather_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _gather_tensors(x)


import inspect

_MM_PREFIXES = ("mmdet", "mmdet3d", "mmseg")

def build_conv2d_from_aten_args(args: tuple, device: str = "cuda"):
    """
    Given the args tuple from aten.convolution.default, reconstruct an nn.Conv2d
    with identical hyperparameters and weights.
    """
    input_tensor = args[0]
    weight = args[1]
    bias = args[2]
    stride = args[3]
    padding = args[4]
    dilation = args[5]
    # args[6] = transposed (we assume False for normal Conv2d)
    # args[7] = output_padding (unused for regular conv)
    groups = args[8] if len(args) > 8 else 1

    in_channels = input_tensor.shape[1]
    out_channels = weight.shape[0]
    kH, kW = weight.shape[2], weight.shape[3]

    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kH, kW),
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        groups=groups,
        bias=(bias is not None),
    ).to(device)

    with torch.no_grad():
        conv.weight.copy_(weight.to(device))
        if bias is not None:
            conv.bias.copy_(bias.to(device))

    return conv


def replay_sep_aspp_conv_with_tiling(
    rec,
    splits_h: int = 1,
    splits_w: int = 1,
    device: str = "cuda",
    check_diff: bool = False,
):
    """
    Replay only the convolution from
    mmseg.models.decode_heads.sep_aspp_head.forward:100,
    but run it via SpatiallyTiledModule instead of the raw aten op.

    Args:
        rec:       OpRecorder instance with recorded events
        splits_h:  # of tiles along height
        splits_w:  # of tiles along width
        device:    "cuda", "cuda:0", or "cpu"
        check_diff: if True, also run the original conv and print max |diff|
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available but device='{device}' requested.")

    target_events = []
    for e in rec.events:
        caller = e.get("caller") or {}
        # print(caller.get("module"), caller.get("function"), caller.get("lineno"))
        if (
            caller.get("module") == "mmseg.models.decode_heads.sep_aspp_head"
            and caller.get("function") == "forward"
            and caller.get("lineno") == 206
            and e["op_name"].startswith("convolution")
        ):
            target_events.append(e)
    # print(f"Found {len(target_events)} target events")
    if not target_events:
        print("No matching sep_aspp_head.forward:100 convolution events found.")
        return

    # If there are multiple, just handle them all in order
    for idx, e in enumerate(target_events):
        # Move args/kwargs back to target device
        args = _move_to_device(e["args"], device)
        kwargs = _move_to_device(e["kwargs"], device)

        # Build Conv2d module from this aten.convolution.default
        conv = build_conv2d_from_aten_args(args, device=device)

        # # Wrap it in your spatial tiling wrapper
        # tiled_module = SpatiallyTiledModule(
        #     conv,
        #     splits_h=splits_h,
        #     splits_w=splits_w,
        # ).to(device)

        x = args[0]  # input feature: N x C x H x W

        y_prof, engine_table = profile_conv_from_module(
            conv, x, max_engine_variants=15
        )

        print(f"[Event {idx}] engine candidates: {engine_table.shape[0]}")
        print(engine_table)

        # # Nice NVTX label to see in Nsight
        # caller = e.get("caller") or {}
        # nvtx_name = (
        #     f"SEP_ASPP_TILED | {e['op_name']} | "
        #     f"{caller.get('module')}.{caller.get('function')}:{caller.get('lineno')}"
        # )

        # with torch.no_grad():
        #     # Optionally compare with original monolithic conv for sanity
        #     if check_diff:
        #         # Original op
        #         nvtx.range_push("SEP_ASPP_ORIGINAL_CONV")
        #         y_orig = e["func"](*args, **kwargs)
        #         nvtx.range_pop()

        #     # Tiled op
        #     nvtx.range_push(nvtx_name)
        #     y_tiled = tiled_module(x)
        #     nvtx.range_pop()

        #     if check_diff:
        #         # Make sure spatial partitioning is numerically correct
        #         diff = (y_tiled - y_orig).abs().max().item()
        #         print(f"[Event {idx}] max |tiled - original| = {diff:.6e}")

def _find_mm_caller():
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


def _deepcopy_to_cpu(obj):
    """Deep-copy args/kwargs, putting all tensors on CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_deepcopy_to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: _deepcopy_to_cpu(v) for k, v in obj.items()}
    else:
        return obj

def _move_to_device(obj, device: str):
    """Recursively move all tensors in a nested structure to the given device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


# -------------------------------------------------------------------
# OpRecorder
# -------------------------------------------------------------------

class OpRecorder(TorchDispatchMode):
    """
    Intercepts PyTorch ops, but only RECORDS spatial ops (conv/pool/etc.).

    For recorded ops, stores:
      - 'func'  : aten op handle
      - 'args'  : deep-copied args (tensors on CPU)
      - 'kwargs': deep-copied kwargs (tensors on CPU)
      - 'shapes': list of tensor shapes (for quick inspection)
      - 'caller': optional mmdet/mmdet3d/mmseg caller info
    """

    def __init__(self, record_caller: bool = False, max_tensors_per_op: int = 4):
        super().__init__()
        self.record_caller = record_caller
        self.max_tensors_per_op = max_tensors_per_op
        self.events: List[Dict[str, Any]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if _is_spatial_op(func):
            tensors = list(_gather_tensors((args, kwargs)))
            shapes = [tuple(t.shape) for t in tensors[: self.max_tensors_per_op]]

            caller = _find_mm_caller() if self.record_caller else None

            args_copy = _deepcopy_to_cpu(args)
            kwargs_copy = _deepcopy_to_cpu(kwargs)

            self.events.append({
                "op_name": func.__name__,   # e.g. "convolution.default"
                "func": func,               # aten op handle
                "shapes": shapes,
                "num_tensors": len(tensors),
                "args": args_copy,
                "kwargs": kwargs_copy,
                "caller": caller,
            })

        return func(*args, **kwargs)


def summarize_arg(arg, indent=0):
    pad = "  " * indent
    if isinstance(arg, torch.Tensor):
        print(f"{pad}Tensor(shape={tuple(arg.shape)}, dtype={arg.dtype}, device={arg.device})")
    elif isinstance(arg, (list, tuple)):
        print(f"{pad}{type(arg).__name__}(len={len(arg)})")
        for i, x in enumerate(arg):
            print(f"{pad}  [{i}]:")
            summarize_arg(x, indent + 2)
    elif isinstance(arg, dict):
        print(f"{pad}dict(len={len(arg)})")
        for k, v in arg.items():
            print(f"{pad}  {k!r}:")
            summarize_arg(v, indent + 2)
    else:
        # Non-tensor leaf: just type + repr truncated
        r = repr(arg)
        if len(r) > 80:
            r = r[:77] + "..."
        print(f"{pad}{type(arg).__name__}: {r}")




# model_name = 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d'
model_name = 'deeplabv3+_r50-d8_512x1024_40k_sem_seg_bdd100k'
# model_name = 'detr_r50_8xb2-150e_coco'
inferencer = bddSegInferencer(model_name, mode='sem_seg')

inferencer.model.eval()
rec = OpRecorder(record_caller=True)

with torch.no_grad(), rec:
    # inferencer(dict(points=WARM_PCD))
    inferencer(dict(img=WARM_IMAGE))
    print("done")

# Print the recorded events
print(f"Recorded {len(rec.events)} ops")
# for i, e in enumerate(rec.events):
#     print(f"=== Event {i} ===")
#     print(e["op_name"])
#     print("caller =", e["caller"])

#     print("args summary:")
#     for j, a in enumerate(e["args"]):
#         print(f"  arg[{j}]:")
#         summarize_arg(a, indent=2)

#     print("kwargs summary:")
#     for k, v in e["kwargs"].items():
#         print(f"  {k!r}:")
#         summarize_arg(v, indent=2)

#     print("--------------------------------")

splits_hs = [1]
splits_ws = [1]
for splits_h in splits_hs:
    for splits_w in splits_ws:
        replay_sep_aspp_conv_with_tiling(
            rec,
            splits_h=splits_h,
            splits_w=splits_w,
            device="cuda:0",
            check_diff=False,   # optional: sanity check correctness
        )