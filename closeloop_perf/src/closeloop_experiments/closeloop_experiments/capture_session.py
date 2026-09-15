"""Owned multi-client MPS capture session for the capsule CLI."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, Optional

from .agent_backend import UnixAgentReplayBackend
from .config import load_run_config
from .mps import MPSManager


class CaptureSessionError(RuntimeError):
    """Raised when an owned capture client or coordinator fails."""


def resolve_source_config(run_directory: Path) -> Path:
    """Return the byte-identical original config used by a source run."""
    run_directory = Path(run_directory)
    manifest_path = run_directory / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureSessionError(
            f"cannot read source run manifest: {exc}"
        ) from exc
    expected_hash = manifest.get("config_sha256")
    if not isinstance(expected_hash, str):
        raise CaptureSessionError("source manifest has no config_sha256")
    candidates = []
    recorded = manifest.get("config_source")
    if isinstance(recorded, str):
        candidates.append(Path(recorded))
    for argument in manifest.get("command", []):
        if (isinstance(argument, str) and
                argument.startswith("config_file:=")):
            candidates.append(Path(argument.split(":=", 1)[1]))
    candidates.append(run_directory / "config.yaml")
    for candidate in candidates:
        try:
            raw = candidate.expanduser().resolve().read_bytes()
        except OSError:
            continue
        if hashlib.sha256(raw).hexdigest() == expected_hash:
            return candidate.expanduser().resolve()
    raise CaptureSessionError(
        "no byte-identical source configuration is available; the original "
        "config path recorded by the run must remain readable"
    )


def discover_agent_library(explicit: Optional[Path] = None) -> Path:
    """Resolve the installed injection library without guessing silently."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    environment = os.environ.get("PPERF_KERNEL_CAPSULE_AGENT_LIBRARY")
    if environment:
        candidates.append(Path(environment))
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = Path(get_package_prefix("closeloop_profiler"))
        candidates.append(prefix / "lib/libpperf_kernel_capsule_agent.so")
    except (ImportError, LookupError):
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise CaptureSessionError(
        "libpperf_kernel_capsule_agent.so is not installed; pass "
        "--agent-library or set PPERF_KERNEL_CAPSULE_AGENT_LIBRARY"
    )


def discover_nvbit_cta_library(agent_library: Path) -> Optional[Path]:
    """Return the co-installed optional NVBit CTA preload tool."""
    candidates = []
    environment = os.environ.get("PPERF_NVBIT_CTA_TRACKER_LIBRARY")
    if environment:
        candidates.append(Path(environment))
    candidates.append(
        Path(agent_library).with_name("libpperf_nvbit_cta_tracker.so")
    )
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = Path(get_package_prefix("closeloop_profiler"))
        candidates.append(prefix / "lib/libpperf_nvbit_cta_tracker.so")
    except (ImportError, LookupError):
        pass
    return next(
        (
            candidate.resolve()
            for candidate in candidates if candidate.is_file()
        ),
        None,
    )


def discover_model_client() -> Path:
    """Resolve the installed client so preloads reach the target process."""
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = Path(get_package_prefix("closeloop_testbed"))
    except (ImportError, LookupError) as exc:
        raise CaptureSessionError(
            "closeloop_testbed is not installed"
        ) from exc
    executable = prefix / "lib/closeloop_testbed/capsule_model_client"
    if not executable.is_file():
        raise CaptureSessionError(
            "closeloop_testbed capsule_model_client is not installed"
        )
    return executable.resolve()


def nvbit_cupti_incompatibility() -> Optional[str]:
    """Explain legacy CUPTI's single-subscriber preload limit, if known."""
    candidates = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home) / "include/cuda.h")
    candidates.append(Path("/usr/local/cuda/include/cuda.h"))
    for header in candidates:
        try:
            lines = header.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("#define CUDA_VERSION "):
                version = int(line.rsplit(maxsplit=1)[-1])
                if version < 13030:
                    return (
                        f"CUDA {version} permits one CUPTI callback "
                        "subscriber; NVBit 1.8 and the capsule agent need "
                        "separate subscribers (CUDA 13.3+ is required)"
                    )
                return None
    return (
        "CUDA version is unknown, so NVBit/CUPTI callback coexistence "
        "cannot be established safely"
    )


class ManagedCaptureSessionBackend(UnixAgentReplayBackend):
    """Lazily own MPS, injected clients, and their coordinator."""

    def __init__(
        self,
        source_run_directory: Path,
        output_directory: Path,
        agent_library: Optional[Path] = None,
        timeout_seconds: float = 300.0,
        popen=subprocess.Popen,
    ):
        self.source_run_directory = Path(source_run_directory)
        self.output_directory = Path(output_directory)
        self.agent_library = Path(agent_library) if agent_library else None
        self.popen = popen
        self.session = None
        self.mps = None
        self.processes = []
        self.logs = []
        self.heartbeat_stop = threading.Event()
        self.heartbeat_failure = None
        self.heartbeat_thread = None
        super().__init__(Path("/not-started"), timeout_seconds, True)

    @staticmethod
    def _wait_for_path(path: Path, processes, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            failed = [
                process.returncode for process in processes
                if process.poll() is not None
            ]
            if failed:
                raise CaptureSessionError(
                    f"capture process exited before readiness: {failed}"
                )
            time.sleep(0.05)
        raise TimeoutError(f"capture readiness timeout: {path}")

    def _wait_for_client(self, status_path: Path, socket_path: Path) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            failed = [
                process.returncode for process in self.processes
                if process.poll() is not None
            ]
            if failed:
                raise CaptureSessionError(
                    f"capture client exited before readiness: {failed}"
                )
            if socket_path.exists() and status_path.is_file():
                try:
                    status = json.loads(
                        status_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    status = {}
                if status.get("state") == "warmup_ready":
                    return
                if status.get("state") == "failed":
                    raise CaptureSessionError(str(status.get("error")))
            time.sleep(0.05)
        raise TimeoutError(f"client warmup timeout: {status_path}")

    def _heartbeat(self) -> None:
        while not self.heartbeat_stop.wait(1.0):
            try:
                self._request("heartbeat")
            except Exception as exc:  # clients enforce lease expiry
                self.heartbeat_failure = exc
                return

    def _start(
        self, capture_specification: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.session is not None:
            return
        self.agent_library = discover_agent_library(self.agent_library)
        nvbit_cta_library = discover_nvbit_cta_library(self.agent_library)
        nvbit_unavailable_reason = (
            nvbit_cupti_incompatibility()
            if nvbit_cta_library is not None else None
        )
        if nvbit_unavailable_reason is not None:
            # ponytail: remove with a subscriber-scoped CUPTI v2 migration.
            nvbit_cta_library = None
        model_client = discover_model_client()
        config_path = resolve_source_config(self.source_run_directory)
        config = load_run_config(config_path).data
        selected_clients = None
        if capture_specification is not None:
            selected = [capture_specification["aggressor"]] + list(
                capture_specification["victim_kernels"]
            )
            selected_clients = {
                str(value["model_identity"]) for value in selected
            }
        models = [
            model for model in config["models"]
            if selected_clients is None or model["id"] in selected_clients
        ]
        if selected_clients is not None and {
            model["id"] for model in models
        } != selected_clients:
            raise CaptureSessionError(
                "capture specification names an unavailable model client"
            )
        if len(models) < 2 or not config["gpu"]["mps_enabled"]:
            raise CaptureSessionError(
                "automatic kernel capture requires at least two MPS models"
            )
        self.session = tempfile.TemporaryDirectory(prefix="pperf-capsule-")
        session_root = Path(self.session.name)
        self.mps = MPSManager(config["gpu"]["index"], session_root)
        self.mps.start()
        specification = self.output_directory / "capture_specification.json"
        capture_input = self.output_directory / "capture_input.json"
        for model in models:
            model_id = model["id"]
            socket_path = session_root / f"agent-{model_id}.sock"
            status_path = session_root / f"status-{model_id}.json"
            log = (
                self.output_directory / f"capture_client_{model_id}.log"
            ).open("wb")
            self.logs.append(log)
            environment = os.environ.copy()
            environment.update(self.mps.model_environment(
                model["mps_percentage"]
            ))
            environment.update({
                "CUDA_INJECTION64_PATH": str(self.agent_library),
                "PPERF_KERNEL_CAPSULE_SPEC": str(specification),
                "PPERF_KERNEL_CAPSULE_CLIENT_ID": model_id,
                "PPERF_KERNEL_CAPSULE_AGENT_SOCKET": str(socket_path),
                "PPERF_KERNEL_CAPSULE_INPUT": str(capture_input),
            })
            if nvbit_unavailable_reason is not None:
                environment["PPERF_NVBIT_CTA_UNAVAILABLE_REASON"] = (
                    nvbit_unavailable_reason
                )
            if nvbit_cta_library is not None:
                preload = environment.get("LD_PRELOAD", "").strip()
                environment["LD_PRELOAD"] = " ".join(
                    value for value in (str(nvbit_cta_library), preload)
                    if value
                )
                environment["PPERF_NVBIT_CTA_TRACKER_LIBRARY"] = str(
                    nvbit_cta_library
                )
            process = self.popen(
                [
                    str(model_client), "--config", str(config_path),
                    "--model-id", model_id, "--status", str(status_path),
                    "--barrier-timeout-seconds", str(self.timeout_seconds),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.processes.append(process)
            self._wait_for_client(status_path, socket_path)
        coordinator_socket = session_root / "coordinator.sock"
        coordinator_log = (
            self.output_directory / "capture_coordinator.log"
        ).open("wb")
        self.logs.append(coordinator_log)
        command = [
            "python3", "-m", "closeloop_experiments.capsule_coordinator",
            "--listen", str(coordinator_socket),
        ]
        for model in models:
            command.extend([
                "--agent",
                f"{model['id']}="
                f"{session_root / ('agent-' + model['id'] + '.sock')}",
            ])
        coordinator = self.popen(
            command,
            stdout=coordinator_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes.append(coordinator)
        self._wait_for_path(
            coordinator_socket, [coordinator], self.timeout_seconds
        )
        self.socket_path = coordinator_socket
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat, name="capsule-lease", daemon=True
        )
        self.heartbeat_thread.start()

    def _check_heartbeat(self) -> None:
        if self.heartbeat_failure is not None:
            raise CaptureSessionError(
                f"coordinator lease heartbeat failed: {self.heartbeat_failure}"
            )

    def capture_one_instrumented_frame(
        self, capture_specification: Dict[str, Any]
    ):
        """Start the owned session and capture the selected frame once."""
        self._start(capture_specification)
        self._check_heartbeat()
        return super().capture_one_instrumented_frame(capture_specification)

    def _request(self, operation: str, **payload: Any) -> Dict[str, Any]:
        if operation != "heartbeat":
            self._check_heartbeat()
        return super()._request(operation, **payload)

    def close(self) -> None:
        """Shutdown agents explicitly, then stop owned processes and MPS."""
        self.heartbeat_stop.set()
        if self.session is None:
            return
        coordinator_alive = (
            self.processes and self.processes[-1].poll() is None
        )
        if coordinator_alive:
            original_timeout = self.timeout_seconds
            self.timeout_seconds = min(original_timeout, 2.0)
            try:
                super().close()
            except Exception:
                pass
            finally:
                self.timeout_seconds = original_timeout
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=2)
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()
        if self.mps is not None:
            self.mps.stop()
        for log in self.logs:
            log.close()
        self.session.cleanup()
        self.session = None
