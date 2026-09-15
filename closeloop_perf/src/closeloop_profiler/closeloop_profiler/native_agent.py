"""Small ctypes bridge to the injected native capsule agent."""

import ctypes
import os
from pathlib import Path
from typing import Optional


class NativeAgentError(RuntimeError):
    """Raised when the injected agent C ABI rejects a lifecycle marker."""


class NativeAgentBridge:
    """Call lifecycle markers exported by ``CUDA_INJECTION64_PATH``."""

    def __init__(self, library_path: Path):
        self.library_path = Path(library_path)
        self.library = ctypes.CDLL(str(self.library_path))
        self._warmup = self.library.pperf_agent_report_warmup_complete
        self._warmup.argtypes = [ctypes.c_char_p]
        self._warmup.restype = ctypes.c_int
        self._wait = self.library.pperf_agent_wait_capture_epoch
        self._wait.argtypes = [ctypes.c_uint64]
        self._wait.restype = ctypes.c_int
        self._frame = self.library.pperf_agent_set_frame
        self._frame.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
        ]
        self._frame.restype = ctypes.c_int
        self._push_owner = self.library.pperf_agent_push_owner
        self._push_owner.argtypes = [ctypes.c_char_p]
        self._push_owner.restype = ctypes.c_int
        self._pop_owner = self.library.pperf_agent_pop_owner
        self._pop_owner.argtypes = []
        self._pop_owner.restype = ctypes.c_int
        self._frame_end = self.library.pperf_agent_report_frame_end
        self._frame_end.argtypes = [ctypes.c_int]
        self._frame_end.restype = ctypes.c_int

    @classmethod
    def from_environment(
        cls, required: bool = True
    ) -> Optional["NativeAgentBridge"]:
        """Load the already-injected library named by the CUDA contract."""
        value = os.environ.get("CUDA_INJECTION64_PATH")
        if not value:
            if required:
                raise NativeAgentError("CUDA_INJECTION64_PATH is not set")
            return None
        return cls(Path(value))

    @staticmethod
    def _bytes(value: str) -> bytes:
        return str(value).encode("utf-8")

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise NativeAgentError(
                f"native capsule agent rejected {operation}: {result}"
            )

    def report_warmup_complete(self, model_id: str) -> None:
        """Move the native state from warming to ready."""
        self._check(self._warmup(self._bytes(model_id)), "warmup completion")

    def wait_capture_epoch(self, timeout_seconds: float) -> None:
        """Wait until both clients share the coordinator release epoch."""
        timeout_ms = max(1, int(float(timeout_seconds) * 1000))
        self._check(self._wait(timeout_ms), "capture epoch")

    def set_frame(
        self, model_id: str, input_id: str, header_timestamp_ns: Optional[int]
    ) -> None:
        """Mirror the exact model/input identity into native matching."""
        stamp = 0 if header_timestamp_ns is None else header_timestamp_ns
        self._check(
            self._frame(
                self._bytes(model_id), self._bytes(input_id), int(stamp)
            ),
            "frame identity",
        )

    def push_owner(self, owner: str) -> None:
        """Mirror one framework/NVTX owner on the application thread."""
        self._check(self._push_owner(self._bytes(owner)), "owner push")

    def pop_owner(self) -> None:
        """Pop one mirrored framework/NVTX owner."""
        self._check(self._pop_owner(), "owner pop")

    def report_frame_end(self, success: bool) -> None:
        """Report an unexpected normal frame return or an inference error."""
        self._check(self._frame_end(int(bool(success))), "frame end")
