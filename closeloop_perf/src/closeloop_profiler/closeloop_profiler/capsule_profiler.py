"""In-process marker companion for the native live kernel-capsule agent."""

import json
import os
from pathlib import Path
from typing import Any, Dict

from .profiler import ModelProfiler
from .native_agent import NativeAgentBridge


class GpuForensicsProfiler(ModelProfiler):
    """Retain model ranges while an injected native agent captures launches."""

    profiler_id = "gpu_forensics_v1"
    replay_backend_id = "kernel_capsule_v3"

    def __init__(
        self, *args: Any, options: Dict[str, Any] = None, **kwargs: Any
    ):
        """Attach model ranges and validate the native-agent setup."""
        super().__init__(*args, **kwargs)
        self.options = dict(options or {})
        self.capture_specification = os.environ.get(
            "PPERF_KERNEL_CAPSULE_SPEC"
        )
        if self.options.get("mode") != "kernel_capsule":
            raise ValueError(
                "gpu_forensics_v1 requires profiler mode kernel_capsule"
            )
        if self.options.get("checkpoint_required", True) is not True:
            raise ValueError("kernel_capsule_v3 requires checkpoints")
        if self.capture_specification:
            path = Path(self.capture_specification)
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema") != "kernel_capsule_capture_spec_v3":
                raise ValueError(
                    "invalid kernel capsule capture specification"
                )
        self.native_agent = NativeAgentBridge.from_environment(required=False)

    def _push(self, tag: str) -> None:
        """Mirror semantic ownership before emitting the matching NVTX push."""
        if self.native_agent is not None:
            try:
                value = json.loads(tag[len("closeloop:"):])
            except (json.JSONDecodeError, TypeError):
                value = {}
            owner = value.get("owner") or value.get("module")
            self.native_agent.push_owner(
                owner or value.get("event", "unknown")
            )
        super()._push(tag)

    def _pop(self) -> None:
        """Keep native and NVTX ownership stacks balanced."""
        super()._pop()
        if self.native_agent is not None:
            self.native_agent.pop_owner()
