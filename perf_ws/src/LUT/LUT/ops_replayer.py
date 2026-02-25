from p_perf.pPerf_inferencer import pPerf2dDetInferencer, pPerf3dDetInferencer, bddDetInferencer, bddSegInferencer, pPerf3dSegInferencer
import cv2
import numpy as np
import torch
import torch.nn as nn

from LUT.utils import move_to_device
from LUT.ops_recorder import OpRecorder

from torch.utils.cpp_extension import load
# Build/load the extension (first run compiles; later runs reuse the build cache)
load(
    name="my_cudnn_prof_ext",
    sources=["/mmdetection3d_ros2/perf_ws/src/LUT/LUT/engine_cfg_perf.cpp"],
    with_cuda=True,
    extra_cflags=["-O0", "-g", "-fno-omit-frame-pointer"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
    is_python_module=False,   # <-- critical
)

# torch.backends.cuda.matmul.allow_tf32 = True   # or False
# torch.backends.cudnn.allow_tf32 = True         # or False
# torch.backends.cudnn.benchmark = True          # affects algo selection in PyTorch paths
# torch.backends.cudnn.deterministic = True     # affects allowed algos

WARM_IMAGE_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'
WARM_IMAGE = cv2.imread(WARM_IMAGE_PATH)
WARM_PCD_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_PCD = np.fromfile(WARM_PCD_PATH, dtype=np.float32).reshape(-1, 5)


class ops_replayer:
    """
    Replays convolution events by profiling their engine variants.
    
    Takes in conv_events and replays every convolution operation.
    """
    
    def __init__(self, conv_events, device: str = "cuda", max_engine_variants: int = 30, scale_info: str = ""):
        """
        Initialize the ops replayer.
        
        Args:
            conv_events: List of convolution event dicts to replay
            device: Target device for execution (default: "cuda")
            max_engine_variants: Maximum number of engine variants to profile (default: 30)
            scale_info: String describing the input scale (e.g., "1280x720") to distinguish different input sizes
        """
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available but device='{device}' requested.")
        
        # Normalize input to list
        if isinstance(conv_events, dict):
            conv_events = [conv_events]
        elif not isinstance(conv_events, list):
            raise TypeError(f"conv_events must be a dict or list of dicts, got {type(conv_events)}")
        
        self.conv_events = conv_events
        self.device = device
        self.max_engine_variants = max_engine_variants
        self.scale_info = scale_info
        self.results = []
    
    @staticmethod
    def _as_pair(v):
        """conv.stride/padding/dilation may be int or tuple"""
        if isinstance(v, (list, tuple)):
            return [int(v[0]), int(v[1])]
        return [int(v), int(v)]
    
    def _profile_conv_from_module(self, conv, x, max_engine_variants=25, iters=1, module_name="unknown"):
        """Profile convolution from a module."""
        # Ensure inputs match your C++ checks: CUDA, contiguous, float32, NCHW
        x = x.contiguous().float()
        w = conv.weight.contiguous().float()
        b = conv.bias
        if b is None:
            b = torch.zeros((w.shape[0],), device=w.device, dtype=torch.float32)
        else:
            b = b.contiguous().float()

        stride   = self._as_pair(conv.stride)
        padding  = self._as_pair(conv.padding)
        dilation = self._as_pair(conv.dilation)
        groups   = int(conv.groups)
        
        # Call your extension op
        y, engine_table = torch.ops.LUT_perf.profile_conv2d(
            x, w, b, stride, padding, dilation, groups, int(max_engine_variants), int(iters), module_name
        )
        return y, engine_table

    def _build_conv2d(self, args: tuple, device: str = "cuda"):
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
    
    def replay_conv(self, event):
        """
        Replay a single convolution event.
        
        Args:
            event: Single event dict to replay
            
        Returns:
            Tuple (event, output_tensor, engine_table) or None if event is invalid
        """
        # Validate event structure
        if not isinstance(event, dict):
            print(f"Skipping non-dict event: {type(event)}")
            return None
        
        if not event["op_name"].startswith("convolution"):
            print(f"Skipping non-convolution op: {event['op_name']}")
            return None
        
        # Move args/kwargs back to target device
        args = move_to_device(event["args"], self.device)
        kwargs = move_to_device(event["kwargs"], self.device)
        
        # Build Conv2d module from this aten.convolution.default
        conv = self._build_conv2d(args, device=self.device)
        
        x = args[0]  # input feature: N x C x H x W
        
        # Use op_id from the event (recorded during OpRecorder)
        op_id = event.get("op_id", "unknown")
        if op_id is None:
            # Fallback if op_id wasn't set (shouldn't happen for convolution ops)
            op_id = "unknown"
        
        # Include scale information and input shape in the module name
        input_shape = f"{x.shape[2]}x{x.shape[3]}"  # HxW
        if self.scale_info:
            module_name = f"{op_id}_scale{self.scale_info}_input{input_shape}"
        else:
            module_name = f"{op_id}_input{input_shape}"
        
        y_prof, engine_table = self._profile_conv_from_module(
            conv, x, max_engine_variants=self.max_engine_variants, module_name=module_name
        )
        
        # print(f"op_id: {op_id}, engine candidates: {engine_table.shape[0]}")
        # print(engine_table)
        # print(y_prof)
        
        return (event, y_prof, engine_table)
    
    def replay_all(self):
        """
        Replay all convolution events.
        
        Returns:
            List of tuples (event, output_tensor, engine_table) for each replayed event
        """
        if not self.conv_events:
            print("No events provided to replay.")
            return []
        
        self.results = []
        
        # Process each event
        for idx, event in enumerate(self.conv_events):
            try:
                result = self.replay_conv(event)
                if result is not None:
                    self.results.append(result)
            except Exception as e:
                print("================================================")
                print(f"Error replaying event {idx}: {e}")
                print(event["caller"])
                print("groups: ", event["args"][8], "\n")
                # break
        
        return self.results


if __name__ == "__main__":
    # Define different scales for testing (must be divisible by 32 for optimal performance)
    # First scale is always the default from config: (1280, 720)
    # Additional scales are appended after the default
    # All scales must be smaller than default (width <= 1280, height <= 720)
    portions = [0.5, 0.6, 0.7, 0.8]
    scales = []
    for portion in portions:
        scales.append((int(WARM_IMAGE.shape[1] * portion), int(WARM_IMAGE.shape[0] * portion)))
    
    # model_name = 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d'
    model_name = 'deeplabv3+_r50-d8_512x1024_40k_sem_seg_bdd100k'
    # model_name = 'detr_r50_8xb2-150e_coco'
    
    # Initialize inferencer with additional scales
    # Pipeline at index 0 will be the default scale from config
    # Pipelines at indices 1, 2, ... will be the additional scales provided
    inferencer = bddSegInferencer(model_name, mode='sem_seg')

    inferencer.model.eval()
    
    # Load the original image - no need to resize manually
    # Resizing will be handled by the pipelines
    image = WARM_IMAGE
    
    print(f"Original image shape: {image.shape}")
    print(f"Number of pipelines: {len(inferencer.pipelines)}")
    print(f"Pipeline scales: {inferencer.scales}")
    
    all_results = {}
    
    # Run inference using different pipeline indices (different scales)
    for pipeline_idx in range(len(inferencer.pipelines)):
        print(f"\n{'='*60}")
        print(f"Processing with pipeline {pipeline_idx} (scale: {inferencer.scales[pipeline_idx]})...")
        print(f"{'='*60}")
        
        rec = OpRecorder(record_caller=True)
        
        with torch.no_grad(), rec:
            # Call inferencer with (index, img) tuple
            # The pipeline at the specified index will handle the resizing
            inferencer((pipeline_idx, dict(img=image)))
            print(f"Inference done for pipeline {pipeline_idx}")

        # Print the recorded events
        print(f"Recorded {len(rec.events)} ops for pipeline {pipeline_idx}")

        # Filter convolution events
        conv_events = [e for e in rec.events if e["op_name"].startswith("convolution")]
        print(f"Found {len(conv_events)} convolution ops for pipeline {pipeline_idx}")

        # Get scale info for this pipeline
        scale = inferencer.scales[pipeline_idx]
        scale_info = f"{scale[0]}x{scale[1]}"
        
        # Replay all convolution events
        replayer = ops_replayer(
            conv_events,
            device="cuda:0",
            max_engine_variants=40,
            scale_info=scale_info,
        )

        # Option 1: Replay a specific conv layer multiple times
        for index in range(1):
            results = replayer.replay_conv(conv_events[63])
        
        # Option 2: Uncomment to replay all conv layers once
        # results = replayer.replay_all()
        
        all_results[f"pipeline_{pipeline_idx}"] = replayer.results
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for pipeline_name, results in all_results.items():
        print(f"{pipeline_name}: {len(results)} convolution operations profiled")