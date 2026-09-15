"""Persistent native CUDA workload process used by reusable testbeds."""

import hashlib
import json
from pathlib import Path
import select
import subprocess
import time
from typing import Any, Dict, Tuple


class OfflineWorkloadError(RuntimeError):
    """Raised when the native workload process fails."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workload_config(workload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in workload.items()
            if key not in ("adapter", "library")}


def discover_offline_native() -> Tuple[Path, Path]:
    """Resolve the testbed's installed worker and built-in adapter."""
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = Path(get_package_prefix("closeloop_testbed"))
    except (ImportError, LookupError) as exc:
        raise OfflineWorkloadError("closeloop_testbed is not installed") from exc
    worker = prefix / "lib/closeloop_testbed/pperf_offline_worker"
    adapter = prefix / "lib/libpperf_offline_compute_adapter.so"
    if not worker.is_file() or not adapter.is_file():
        raise OfflineWorkloadError("offline worker/adapter is not installed")
    return worker.resolve(), adapter.resolve()


class WorkerProcess:
    """One persistent native CUDA client with a JSON-line control pipe."""

    def __init__(self, executable: Path, adapter: Path,
                 workload: Dict[str, Any], client_id: str,
                 environment: Dict[str, str], log_path: Path,
                 cta: bool = False):
        self.client_id = client_id
        self.log = Path(log_path).open("wb")
        command = [
            str(executable), "--adapter", str(adapter), "--config-json",
            _canonical(_workload_config(workload)), "--client-id", client_id,
        ]
        if cta:
            command.append("--cta")
        self.process = subprocess.Popen(
            command, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True,
            bufsize=1, start_new_session=True)
        response = self._read(120.0)
        if response.get("status") != "ready":
            raise OfflineWorkloadError(f"{client_id} worker did not become ready")
        self.identity = dict(response["identity"])
        self.identity["adapter_library_sha256"] = _sha256_file(adapter)

    def _read(self, timeout: float) -> Dict[str, Any]:
        if self.process.stdout is None:
            raise OfflineWorkloadError("worker stdout is unavailable")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [self.process.stdout], [], [], deadline - time.monotonic())
            if not ready:
                break
            line = self.process.stdout.readline()
            if not line:
                raise OfflineWorkloadError(
                    f"{self.client_id} worker exited with {self.process.poll()}")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                self.log.write(line.encode("utf-8"))
                self.log.flush()
        raise TimeoutError(f"{self.client_id} worker response timeout")

    def send(self, value: Dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise OfflineWorkloadError("worker stdin is unavailable")
        self.process.stdin.write(_canonical(value) + "\n")
        self.process.stdin.flush()

    def receive(self, timeout: float = 120.0) -> Dict[str, Any]:
        response = self._read(timeout)
        if response.get("status") != "ok":
            raise OfflineWorkloadError(str(response.get("error", response)))
        return dict(response["result"])

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send({"operation": "shutdown"})
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()
        if self.process.returncode not in (0, -15):
            raise OfflineWorkloadError(
                f"{self.client_id} worker cleanup exit {self.process.returncode}")
