import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from p_perf.config.constant import image_models, lidar_models, seg_models
from p_perf.pPerf_inferencer import (
    pPerf2dDetInferencer,
    pPerf2dSegInferencer,
    pPerf3dDetInferencer,
)


PROFILE_RANGE_PREFIX = "profile_kernel_capture"
WARM_IMAGE_PATH = "/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg"
WARM_PCD_PATH = "/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin"
RUNNER_MODE_ENV = "P_PERF_KERNEL_PROFILER_MODE"
RUNNER_MODEL_ENV = "P_PERF_KERNEL_PROFILER_MODEL_NAME"
RUNNER_DEVICE_ENV = "P_PERF_KERNEL_PROFILER_DEVICE"
THEORETICAL_OCCUPANCY_CANDIDATES = [
    "launch__occupancy_per_sm",
    "launch__occupancy_limit_active_warps_pct",
    "launch__occupancy_limit_active_threads_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


# User configuration for direct script execution.
PROFILE_ALL_MODELS = True
MODEL_NAME = "faster-rcnn_r50_fpn_1x_coco"
OUTPUT_DIR = "kernel_profiles"
OUTPUT_CSV = None
DEVICE = "cuda:0"
KEEP_REPORT = False


def extract_nvtx_info(action: Any) -> str:
    """Return the joined push/pop NVTX range stack for a single action."""
    try:
        nvtx_state = action.nvtx_state()
        range_names: List[str] = []
        for domain_id in nvtx_state.domains():
            domain_info = nvtx_state.domain_by_id(domain_id)
            push_pop = domain_info.push_pop_ranges()
            if push_pop:
                range_names.extend(push_pop)
        return "/".join(range_names) if range_names else ""
    except Exception:
        return ""


def get_all_model_names() -> List[str]:
    """Return all currently enabled model names from constant.py."""
    lidar_names = [
        model_info[0] if isinstance(model_info, tuple) else model_info
        for model_info in lidar_models
    ]
    seg_names = [
        model_info[0] if isinstance(model_info, tuple) else model_info
        for model_info in seg_models
    ]
    return list(image_models) + lidar_names + seg_names


class KernelProfiler:
    """Profile the kernels launched by a single model under normal PyTorch execution."""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: str = "cuda:0",
        keep_report: bool = False,
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device
        self.keep_report = keep_report

    def profile(self, output_csv: Optional[str] = None) -> pd.DataFrame:
        """Run Nsight Compute and save the per-kernel profile CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._run_ncu_profile()
        profile_df = self.extract_kernel_metrics(report_path)

        if output_csv is None:
            output_path = self.output_dir / f"{self.model_name}.csv"
        else:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        profile_df.to_csv(output_path, index=False)

        if not self.keep_report and report_path.exists():
            report_path.unlink()

        return profile_df

    def _run_ncu_profile(self) -> Path:
        """Run Nsight Compute against the built-in profiling runner."""
        report_basename = f"{self.model_name}_kernel_profile"
        report_base = Path(tempfile.gettempdir()) / report_basename
        nvtx_filter = f"regex:.*{PROFILE_RANGE_PREFIX}.*/"
        runner_env = os.environ.copy()
        runner_env[RUNNER_MODE_ENV] = "runner"
        runner_env[RUNNER_MODEL_ENV] = self.model_name
        runner_env[RUNNER_DEVICE_ENV] = self.device

        cmd = [
            "ncu",
            "-o",
            str(report_base),
            "--force-overwrite",
            "--nvtx",
            "--nvtx-include",
            nvtx_filter,
            "--section",
            "LaunchStats",
            "--section",
            "Occupancy",
            sys.executable,
            str(Path(__file__).resolve()),
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=runner_env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Nsight Compute profiling failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        return report_base.with_suffix(".ncu-rep")

    def extract_kernel_metrics(self, report_path: Path) -> pd.DataFrame:
        """Extract per-kernel metrics from an Nsight Compute report."""
        ncu_report = self._load_ncu_report()
        report = ncu_report.load_report(str(report_path))
        extracted_rows: List[Dict[str, Any]] = []

        kernel_id = 1
        for range_idx in range(report.num_ranges()):
            current_range = report.range_by_idx(range_idx)
            for action_idx in range(current_range.num_actions()):
                action = current_range.action_by_idx(action_idx)
                nvtx_text = extract_nvtx_info(action)
                range_metadata = self._parse_nvtx_metadata(nvtx_text)
                kernel_name = action.name()

                theoretical_occupancy = self._normalize_occupancy_percent(
                    self._get_first_metric(action, THEORETICAL_OCCUPANCY_CANDIDATES)
                )

                extracted_rows.append(
                    {
                        "model_name": self.model_name,
                        "kernel_id": kernel_id,
                        "kernel_name": kernel_name,
                        "input_shape": range_metadata.get("input_shape", "unknown"),
                        "theoretical_occupancy": theoretical_occupancy,
                        "engine": self._infer_engine_name(kernel_name),
                        "engine_detail": kernel_name,
                        "block_size": self._get_first_metric(action, ["launch__block_size"]),
                        "grid_size": self._get_first_metric(action, ["launch__grid_size"]),
                        "duration_ns": self._get_first_metric(
                            action, ["gpu__time_duration.sum"]
                        ),
                        "registers_per_thread": self._get_first_metric(
                            action, ["launch__registers_per_thread"]
                        ),
                        "static_shared_memory_b": self._get_first_metric(
                            action, ["launch__shared_mem_per_block_static"]
                        ),
                        "dynamic_shared_memory_b": self._get_first_metric(
                            action, ["launch__shared_mem_per_block_dynamic"]
                        ),
                    }
                )
                kernel_id += 1

        if not extracted_rows:
            raise RuntimeError(
                f"No kernels were extracted from {report_path}. "
                "Verify that Nsight Compute captured the NVTX range."
            )

        return pd.DataFrame(extracted_rows)

    def _load_ncu_report(self) -> Any:
        """Load the Nsight Compute Python report API."""
        try:
            import ncu_report  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import ncu_report. Make sure the Nsight Compute Python "
                "extras are available in your current Python environment."
            ) from exc
        return ncu_report

    @staticmethod
    def _get_first_metric(action: Any, metric_names: List[str]) -> Optional[float]:
        for metric_name in metric_names:
            try:
                if metric_name in action:
                    return action[metric_name].value()
            except Exception:
                continue
        return None

    @staticmethod
    def _parse_nvtx_metadata(nvtx_text: str) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        for chunk in nvtx_text.split("|"):
            chunk = chunk.strip()
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            metadata[key.strip()] = value.strip()
        return metadata

    @staticmethod
    def _normalize_occupancy_percent(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        numeric_value = float(value)
        if numeric_value <= 1.0:
            return numeric_value * 100.0
        return numeric_value

    @staticmethod
    def _infer_engine_name(kernel_name: str) -> str:
        name = kernel_name.lower()
        if "scudnn" in name or "cudnn" in name:
            return "cudnn"
        if "cutlass" in name:
            return "cutlass"
        if "triton" in name:
            return "triton"
        if "cublas" in name:
            return "cublas"
        if "nccl" in name:
            return "nccl"
        if "memcpy" in name or "host2device" in name or "device2host" in name:
            return "memcpy"
        return "unknown"

    @staticmethod
    def _resolve_model_type(model_name: str) -> str:
        if model_name in image_models:
            return "image"

        for lidar_model_info in lidar_models:
            lidar_model_name = (
                lidar_model_info[0]
                if isinstance(lidar_model_info, tuple)
                else lidar_model_info
            )
            if model_name == lidar_model_name:
                return "lidar"

        for seg_model_info in seg_models:
            seg_model_name = (
                seg_model_info[0] if isinstance(seg_model_info, tuple) else seg_model_info
            )
            if model_name == seg_model_name:
                return "seg"

        raise ValueError(f"Could not infer model type for '{model_name}'")

    @staticmethod
    def _build_inferencer(model_name: str, device: str) -> Any:
        model_type = KernelProfiler._resolve_model_type(model_name)
        if model_type == "image":
            return pPerf2dDetInferencer(model_name, device=device)
        if model_type == "lidar":
            return pPerf3dDetInferencer(model_name, device=device)
        if model_type == "seg":
            return pPerf2dSegInferencer(
                model_name,
                mode=KernelProfiler._resolve_segmentation_mode(model_name),
                device=device,
            )
        raise ValueError(f"Unsupported model type for '{model_name}'")

    @staticmethod
    def _resolve_segmentation_mode(model_name: str) -> str:
        for seg_model_info in seg_models:
            if not isinstance(seg_model_info, tuple):
                continue
            seg_model_name, seg_mode = seg_model_info
            if model_name == seg_model_name:
                return seg_mode
        return "sem_seg"

    @staticmethod
    def _get_warm_input(model_name: str) -> Tuple[Dict[str, np.ndarray], str]:
        model_type = KernelProfiler._resolve_model_type(model_name)

        if model_type in {"image", "seg"}:
            if not Path(WARM_IMAGE_PATH).exists():
                raise FileNotFoundError(f"Warm image not found: {WARM_IMAGE_PATH}")
            image = np.asarray(Image.open(WARM_IMAGE_PATH).convert("RGB"))
            return {"img": image}, f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}"

        if not Path(WARM_PCD_PATH).exists():
            raise FileNotFoundError(f"Warm point cloud not found: {WARM_PCD_PATH}")
        points = np.fromfile(WARM_PCD_PATH, dtype=np.float32).reshape(-1, 5)
        return {"points": points}, f"{points.shape[0]}x{points.shape[1]}"

    @staticmethod
    def run_profiled_inference(model_name: str, device: str) -> None:
        """Run warmup and one profiled inference inside a well-known NVTX range."""
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available but '{device}' was requested.")

        inferencer = KernelProfiler._build_inferencer(model_name, device)
        if hasattr(inferencer, "show_progress"):
            inferencer.show_progress = False
        if hasattr(inferencer, "model"):
            inferencer.model.eval()

        warm_input, input_shape = KernelProfiler._get_warm_input(model_name)

        with torch.no_grad():
            inferencer(warm_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            nvtx_label = (
                f"{PROFILE_RANGE_PREFIX} | model_name={model_name} | input_shape={input_shape}"
            )
            torch.cuda.nvtx.range_push(nvtx_label)
            try:
                inferencer(warm_input)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()


def main() -> None:
    if os.environ.get(RUNNER_MODE_ENV) == "runner":
        runner_model_name = os.environ.get(RUNNER_MODEL_ENV)
        runner_device = os.environ.get(RUNNER_DEVICE_ENV, DEVICE)
        if not runner_model_name:
            raise ValueError(f"{RUNNER_MODEL_ENV} must be set in runner mode")
        KernelProfiler.run_profiled_inference(runner_model_name, runner_device)
        return

    model_names = get_all_model_names() if PROFILE_ALL_MODELS else [MODEL_NAME]

    for model_name in model_names:
        profiler = KernelProfiler(
            model_name=model_name,
            output_dir=OUTPUT_DIR,
            device=DEVICE,
            keep_report=KEEP_REPORT,
        )

        output_csv = OUTPUT_CSV
        if OUTPUT_CSV is not None and PROFILE_ALL_MODELS:
            output_path = Path(OUTPUT_CSV)
            output_csv = str(output_path.parent / f"{model_name}.csv")

        print(f"Profiling model: {model_name}")
        profiler.profile(output_csv=output_csv)


if __name__ == "__main__":
    main()
