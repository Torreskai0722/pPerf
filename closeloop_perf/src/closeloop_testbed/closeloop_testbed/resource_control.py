"""Apply reproducible CPU affinity and native thread-pool limits."""

import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Optional, Union


LIBRARY_ENVIRONMENT_VARIABLES = {
    "openmp": "OMP_NUM_THREADS",
    "mkl": "MKL_NUM_THREADS",
    "openblas": "OPENBLAS_NUM_THREADS",
    "numexpr": "NUMEXPR_NUM_THREADS",
}
MODEL_LIBRARY_KEYS = frozenset((
    "opencv",
    "pytorch_intraop",
    "pytorch_interop",
))
SUPPORTED_LIBRARY_KEYS = frozenset(
    tuple(LIBRARY_ENVIRONMENT_VARIABLES) + tuple(MODEL_LIBRARY_KEYS)
)
MPS_ENVIRONMENT_VARIABLES = (
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "CUDA_MPS_LOG_DIRECTORY",
    "CUDA_MPS_PIPE_DIRECTORY",
)


def sanitize_mps_environment(environment):
    """Return an environment copy without inherited MPS client settings."""
    sanitized = dict(environment)
    removed = {
        name: sanitized.pop(name)
        for name in MPS_ENVIRONMENT_VARIABLES if name in sanitized
    }
    return sanitized, removed


def _nvidia_smi(command, run_command=subprocess.run):
    result = run_command(
        command, check=True, capture_output=True, text=True)
    return {
        "command": command,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "returncode": result.returncode,
    }


def maximum_supported_clocks(gpu_index, run_command=subprocess.run):
    """Return the highest jointly advertised memory/graphics clock pair."""
    evidence = _nvidia_smi([
        "nvidia-smi", "-i", str(gpu_index),
        "--query-supported-clocks=memory,graphics",
        "--format=csv,noheader,nounits",
    ], run_command)
    pairs = []
    for line in evidence["stdout"].splitlines():
        values = [int(value) for value in re.findall(r"\d+", line)]
        if len(values) >= 2:
            pairs.append((values[0], values[1]))
    if not pairs:
        raise RuntimeError("GPU did not report supported clock pairs")
    memory, graphics = max(pairs, key=lambda pair: (pair[0], pair[1]))
    return {
        "memory_clock_mhz": memory,
        "graphics_clock_mhz": graphics,
        "query": evidence,
    }


def current_gpu_clocks(gpu_index, run_command=subprocess.run):
    """Return current numeric graphics and memory clocks with evidence."""
    evidence = _nvidia_smi([
        "nvidia-smi", "-i", str(gpu_index),
        "--query-gpu=clocks.current.graphics,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ], run_command)
    values = [int(value) for value in re.findall(r"\d+", evidence["stdout"])]
    if len(values) < 2:
        raise RuntimeError("GPU did not report current clocks")
    return {
        "graphics_clock_mhz": values[0],
        "memory_clock_mhz": values[1],
        "query": evidence,
    }


class GPUClockLock:
    """Lock one clock pair and reset both controls on every exit path."""

    def __init__(self, gpu_index, graphics_clock_mhz, memory_clock_mhz,
                 run_command=subprocess.run):
        """Configure one exact GPU clock pair."""
        self.gpu_index = int(gpu_index)
        self.graphics = int(graphics_clock_mhz)
        self.memory = int(memory_clock_mhz)
        self.run_command = run_command
        self.evidence = {
            "requested": {
                "graphics_clock_mhz": self.graphics,
                "memory_clock_mhz": self.memory,
            },
            "lock_commands": [],
            "reset_commands": [],
        }

    def _command(self, arguments):
        return _nvidia_smi(
            ["nvidia-smi", "-i", str(self.gpu_index)] + arguments,
            self.run_command)

    def __enter__(self):
        """Apply both clock locks and retain command evidence."""
        try:
            self.evidence["lock_commands"].append(self._command([
                "--lock-gpu-clocks=" + f"{self.graphics},{self.graphics}"
            ]))
            self.evidence["lock_commands"].append(self._command([
                "--lock-memory-clocks=" + f"{self.memory},{self.memory}"
            ]))
            self.evidence["observed_after_lock"] = current_gpu_clocks(
                self.gpu_index, self.run_command)
            return self
        except BaseException:
            self.reset()
            raise

    def reset(self):
        """Attempt both resets and raise only after both were attempted."""
        errors = []
        for option in ("--reset-gpu-clocks", "--reset-memory-clocks"):
            try:
                self.evidence["reset_commands"].append(
                    self._command([option]))
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def __exit__(self, _type, _value, _traceback):
        """Reset both clock locks."""
        self.reset()


def resolved_library_thread_counts(
        settings: Union[Dict[str, Any], Optional[int]]
) -> Dict[str, int]:
    """Resolve legacy and per-library thread controls without ambiguity."""
    if isinstance(settings, dict):
        fallback = settings.get("cpu_thread_count")
        overrides = settings.get("library_thread_counts", {})
    else:
        fallback = settings
        overrides = {}
    counts: Dict[str, int] = {}
    if fallback is not None:
        counts.update({
            "openmp": fallback,
            "mkl": fallback,
            "openblas": fallback,
            "numexpr": fallback,
            "opencv": fallback,
            "pytorch_intraop": fallback,
            "pytorch_interop": 1,
        })
    counts.update(overrides)
    return dict(sorted(counts.items()))


def thread_environment(
        settings: Union[Dict[str, Any], Optional[int]]
) -> Dict[str, str]:
    """Return native-library environment limits for one process."""
    counts = resolved_library_thread_counts(settings)
    environment = {
        variable: str(counts[library])
        for library, variable in LIBRARY_ENVIRONMENT_VARIABLES.items()
        if library in counts
    }
    if "openmp" in counts:
        environment["OMP_DYNAMIC"] = "FALSE"
    if "mkl" in counts:
        environment["MKL_DYNAMIC"] = "FALSE"
    return environment


def apply_process_resources(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply configured affinity and environment before workers are created."""
    affinity = settings.get("cpu_affinity")
    thread_count = settings.get("cpu_thread_count")
    counts = resolved_library_thread_counts(settings)
    os.environ.update(thread_environment(settings))
    if affinity is not None:
        os.sched_setaffinity(0, set(affinity))
    return {
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_affinity_source": (
            "explicit" if affinity is not None else "inherited"
        ),
        "cpu_thread_count": thread_count,
        "library_thread_counts": counts,
    }


def configure_model_thread_pools(
        settings: Union[Dict[str, Any], Optional[int]], cv2_module=None,
        torch_module=None) -> Dict[str, Any]:
    """Set OpenCV and PyTorch compute pools before model construction."""
    counts = resolved_library_thread_counts(settings)
    opencv_count = counts.get("opencv")
    torch_count = counts.get("pytorch_intraop")
    interop_count = counts.get("pytorch_interop")
    if opencv_count is not None and cv2_module is None:
        import cv2 as cv2_module  # pylint: disable=import-outside-toplevel
    if ((torch_count is not None or interop_count is not None) and
            torch_module is None):
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    if opencv_count is not None:
        cv2_module.setNumThreads(opencv_count)
    if torch_count is not None:
        torch_module.set_num_threads(torch_count)
    if interop_count is not None:
        torch_module.set_num_interop_threads(interop_count)
    observed = {
        library: count
        for library, count in counts.items()
        if library in LIBRARY_ENVIRONMENT_VARIABLES
    }
    if opencv_count is not None:
        observed["opencv"] = cv2_module.getNumThreads()
    if torch_count is not None:
        observed["pytorch_intraop"] = torch_module.get_num_threads()
    if interop_count is not None:
        observed["pytorch_interop"] = (
            torch_module.get_num_interop_threads()
        )
    return {
        "opencv_thread_count": observed.get("opencv"),
        "torch_thread_count": observed.get("pytorch_intraop"),
        "torch_interop_thread_count": observed.get("pytorch_interop"),
        "observed_library_thread_counts": dict(sorted(observed.items())),
    }


def detect_cpu_topology() -> Dict[str, Any]:
    """Return stable logical-CPU and physical-core topology metadata."""
    cpu_root = Path("/sys/devices/system/cpu")
    records = []
    for path in sorted(
            cpu_root.glob("cpu[0-9]*"),
            key=lambda item: int(item.name[3:])):
        cpu = int(path.name[3:])
        topology = path / "topology"
        try:
            core = int(
                (topology / "core_id").read_text(encoding="utf-8").strip()
            )
            package = int(
                (topology / "physical_package_id").read_text(
                    encoding="utf-8"
                ).strip()
            )
        except (OSError, ValueError):
            core = cpu
            package = 0
        records.append({"cpu": cpu, "core": core, "package": package})
    if not records:
        records = [
            {"cpu": cpu, "core": cpu, "package": 0}
            for cpu in range(os.cpu_count() or 1)
        ]
    return {
        "logical_cpu_count": len(records),
        "physical_core_count": len({
            (record["package"], record["core"]) for record in records
        }),
        "cpus": records,
    }
