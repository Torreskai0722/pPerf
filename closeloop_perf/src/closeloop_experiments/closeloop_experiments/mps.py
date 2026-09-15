"""Run-owned NVIDIA MPS lifecycle management."""

import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Dict, Optional


MAX_UNIX_SOCKET_PATH_BYTES = 107


class MPSError(RuntimeError):
    """Raised when safe MPS setup or restoration fails."""


def validate_mps_pipe_directory(pipe_directory: Path) -> None:
    """Reject paths too long for the MPS privileged UNIX socket."""
    socket_path = Path(pipe_directory) / "control_privileged"
    length = len(os.fsencode(socket_path))
    if length > MAX_UNIX_SOCKET_PATH_BYTES:
        raise MPSError(
            f"MPS socket path is {length} bytes; maximum is "
            f"{MAX_UNIX_SOCKET_PATH_BYTES}: {socket_path}"
        )


class MPSManager:
    """Own an MPS daemon and restore the GPU compute mode in all cases."""

    def __init__(self, gpu_index: int, run_directory: Path,
                 run_command: Callable = subprocess.run,
                 popen: Callable = subprocess.Popen):
        self.gpu_index = gpu_index
        self.run_directory = Path(run_directory)
        self.run_command = run_command
        self.popen = popen
        self.pipe_directory = self.run_directory / "p"
        self.log_directory = self.run_directory / "mps" / "log"
        self.original_mode: Optional[str] = None
        self.process = None
        self.evidence = {
            "gpu_index": gpu_index,
            "pipe_directory": str(self.pipe_directory),
            "log_directory": str(self.log_directory),
            "server_state": "not_started",
            "server_ids": [],
        }

    @property
    def environment(self) -> Dict[str, str]:
        """Return run-scoped MPS environment variables."""
        return {
            "CUDA_VISIBLE_DEVICES": str(self.gpu_index),
            "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_directory),
            "CUDA_MPS_LOG_DIRECTORY": str(self.log_directory),
        }

    def _query_mode(self) -> str:
        result = self.run_command(
            ["nvidia-smi", "-i", str(self.gpu_index),
             "--query-gpu=compute_mode", "--format=csv,noheader"],
            check=True, capture_output=True, text=True)
        return result.stdout.strip()

    def _set_mode(self, mode: str) -> None:
        option = {"Default": "0", "Exclusive_Process": "3",
                  "EXCLUSIVE_PROCESS": "3", "0": "0", "3": "3"}.get(mode, mode)
        self.run_command(["nvidia-smi", "-i", str(self.gpu_index),
                          "-c", str(option)], check=True,
                         capture_output=True, text=True)

    def _server_ids(self) -> list:
        result = self.run_command(
            ["nvidia-cuda-mps-control"], input="get_server_list\n",
            text=True, env={**os.environ, **self.environment}, check=True,
            capture_output=True,
        )
        return [
            line.strip() for line in result.stdout.splitlines()
            if line.strip().isdigit()
        ]

    def wait_for_server(self, timeout: float) -> list:
        """Wait until a launched CUDA client creates an MPS server."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            server_ids = self._server_ids()
            if server_ids:
                self.evidence.update(
                    server_state="active", server_ids=server_ids
                )
                return server_ids
            time.sleep(0.1)
        raise MPSError("MPS server identity was not observed after launch")

    def start(self) -> None:
        """Set exclusive-process mode and start a run-scoped daemon."""
        validate_mps_pipe_directory(self.pipe_directory)
        self.original_mode = self._query_mode()
        self.evidence["original_compute_mode"] = self.original_mode
        self.pipe_directory.mkdir(parents=True, exist_ok=False)
        self.log_directory.mkdir(parents=True, exist_ok=False)
        try:
            self._set_mode("Exclusive_Process")
            env = os.environ.copy()
            env.update(self.environment)
            self.process = self.popen(["nvidia-cuda-mps-control", "-d"],
                                      env=env, start_new_session=True)
            self.evidence["daemon_launcher_pid"] = getattr(
                self.process, "pid", None
            )
            time.sleep(0.1)
            if self.process.poll() not in (None, 0):
                raise MPSError("MPS daemon failed to start")
            self.evidence["server_state"] = "control_active"
        except Exception:
            self.stop()
            raise

    def model_environment(self, percentage: int) -> Dict[str, str]:
        """Return environment for one model process."""
        env = self.environment.copy()
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)
        return env

    def stop(self) -> None:
        """Quit only this daemon and restore the original compute mode."""
        errors = []
        if self.process is not None:
            try:
                self.run_command(
                    ["nvidia-cuda-mps-control"], input="quit\n", text=True,
                    env={**os.environ, **self.environment}, check=True,
                    capture_output=True)
                self.evidence["quit_succeeded"] = True
            except Exception as exc:  # restoration must still run
                self.evidence["quit_succeeded"] = False
                errors.append(exc)
        if self.original_mode is not None:
            try:
                self._set_mode(self.original_mode)
                self.evidence["compute_mode_restored"] = True
            except Exception as exc:
                self.evidence["compute_mode_restored"] = False
                errors.append(exc)
        self.evidence["server_state"] = "stopped"
        if errors:
            raise MPSError("; ".join(str(error) for error in errors))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback):
        self.stop()
