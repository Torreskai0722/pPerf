"""Immutable Level-1 experiment orchestration."""

from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import yaml

from closeloop_profiler.architecture_profiles import architecture_profile_metadata
from .config import RunConfig
from closeloop_profiler.components import (
    recording_config, scheduler_script_path,
)
from .mps import MPSManager
from closeloop_testbed.resource_control import (detect_cpu_topology,
                               sanitize_mps_environment)


class PreflightError(RuntimeError):
    """Raised when the host is unsafe or unsuitable for a run."""


class RunFailure(RuntimeError):
    """Raised when an owned experiment process fails."""


def _command_output(command: List[str], run_command=subprocess.run) -> str:
    result = run_command(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def detected_nsys_version(run_command=subprocess.run) -> str:
    """Return the installed Nsight Systems version string."""
    output = _command_output(["nsys", "--version"], run_command)
    match = re.search(r"(\d+(?:\.\d+)+)", output)
    if not match:
        raise PreflightError(f"cannot parse Nsight Systems version: {output}")
    return match.group(1)


def detected_ncu_version(run_command=subprocess.run) -> str:
    """Return the installed Nsight Compute version."""
    output = _command_output(["ncu", "--version"], run_command)
    match = re.search(r"Version\s+(\d+(?:\.\d+)+)", output)
    if not match:
        raise PreflightError(f"cannot parse Nsight Compute version: {output}")
    return match.group(1)


def detect_gpu_hardware(
    index: int, run_command=subprocess.run
) -> Dict[str, str]:
    """Return the immutable GPU identity recorded with every run."""
    fields = (
        "name,uuid,pci.bus_id,driver_version,memory.total,compute_cap"
    )
    output = _command_output([
        "nvidia-smi", "-i", str(index), "--query-gpu=" + fields,
        "--format=csv,noheader,nounits",
    ], run_command)
    values = [value.strip() for value in output.split(",")]
    names = fields.split(",")
    if len(values) != len(names):
        raise PreflightError(f"cannot parse GPU identity: {output}")
    return dict(zip(names, values))


def build_nsys_command(config: RunConfig, output_prefix: Path) -> List[str]:
    """Construct the Nsight command required by the selected profiler."""
    data = config.data
    recording = recording_config(data)
    nsys = recording.get("nsys")
    launch = [
        "ros2", "launch", data["ros"]["launch_package"],
        data["ros"]["launch_file"],
        "config_file:=" + str(config.run_directory / "resolved_config.yaml"),
        "run_directory:=" + str(config.run_directory),
    ]
    if nsys is None:
        return launch
    diagnostic = "preprocessing" in recording["scopes"]
    traces = list(nsys["trace"])
    if diagnostic:
        for trace in ("osrt", "openmp"):
            if trace not in traces:
                traces.append(trace)
    command = [
        "nsys", "profile", "--trace=" + ",".join(traces),
        "--sample=" + ("process-tree" if diagnostic else "none"),
        "--backtrace=" + ("dwarf" if diagnostic else "none"),
        "--cpuctxsw=" + (
            "process-tree" if (
                diagnostic or nsys.get("cpu_context_switch", False)
            )
            else "none"
        ),
        "--wait=all",
        "--force-overwrite=false", "--output", str(output_prefix),
    ]
    if diagnostic:
        command.extend([
            "--osrt-threshold=1000",
            "--osrt-backtrace-threshold=0",
        ])
    if nsys["gpu_context_switch"] and not data["gpu"]["mps_enabled"]:
        command.append("--gpuctxsw=true")
    if nsys.get("gpu_metrics_devices"):
        command.extend([
            "--gpu-metrics-devices=" + nsys["gpu_metrics_devices"],
            "--gpu-metrics-frequency="
            + str(nsys["gpu_metrics_frequency_hz"]),
            "--gpu-metrics-set=" + nsys["gpu_metrics_set"],
        ])
    command.extend(launch)
    return command


def scheduler_tracefs_available() -> bool:
    """Return whether scheduler tracepoints are mounted in this environment."""
    roots = (Path("/sys/kernel/tracing"),
             Path("/sys/kernel/debug/tracing"))
    return any(
        (root / "events" / "sched" / "sched_switch" / "format").is_file()
        for root in roots
    )


def build_profile_command(config: RunConfig,
                          output_prefix: Path) -> List[str]:
    """Wrap Nsight with the selected profiler's external collectors."""
    command = build_nsys_command(config, output_prefix)
    recording = recording_config(config.data)
    ncu = recording.get("ncu")
    if ncu:
        return [
            "ncu", "--mps", "client", "--target-processes", "all",
        ] + command
    if "preprocessing" not in recording["scopes"]:
        return command
    backend = recording.get("preprocessing", {}).get(
        "scheduler_backend", "bpftrace"
    )
    if backend == "none":
        return command
    return [
        "bpftrace", "-q",
        "-o", str(config.run_directory / "scheduler_events.csv"),
        "-c", shlex.join(command),
        str(scheduler_script_path()),
    ]


def build_ncu_control_command(config: RunConfig,
                              output_prefix: Path) -> List[str]:
    """Return the MPS control profiler for per-model client injection."""
    ncu = recording_config(config.data)["ncu"]
    command = [
            "ncu", "--mps", "control", "--mps-num-clients", "2",
            "--mps-timeout", "300", "--replay-mode", "kernel",
            "--clock-control", "none", "--apply-rules", "no",
            "--kernel-name", ncu["kernel_name"],
            "--launch-count", str(ncu["launch_count"]),
            "--force-overwrite", "--export", str(output_prefix),
    ]
    for section in ncu["sections"]:
        command.extend(["--section", section])
    return command


def initial_model_manifest(config: RunConfig) -> Dict[str, Dict[str, object]]:
    """Return configured architecture metadata before model processes run."""
    result = {}
    for model in config.models:
        result[model["id"]] = architecture_profile_metadata(
            model["architecture_profile"])
        result[model["id"]].update({
            name: model[name]
            for name in (
                "mmlab_model", "model_config", "checkpoint",
                "model_config_sha256", "checkpoint_sha256",
            )
            if name in model
        })
    return result


class ExperimentRunner:
    """Preflight, execute, terminate, export, and report one immutable run."""

    def __init__(self, config: RunConfig, run_command=subprocess.run,
                 popen=subprocess.Popen):
        self.config = config
        self.run_command = run_command
        self.popen = popen
        self.run_directory = config.run_directory
        self.process = None

    def _check_gpu_idle(self) -> None:
        try:
            output = _command_output([
                "nvidia-smi", "-i", str(self.config.data["gpu"]["index"]),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                self.run_command)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PreflightError(f"cannot inspect GPU: {exc}") from exc
        occupants = [line.strip() for line in output.splitlines()
                     if line.strip() and line.strip() != "[N/A]"]
        if occupants:
            raise PreflightError("configured GPU is not idle; PIDs: " +
                                 ", ".join(occupants))

    def _check_foreign_mps(self) -> None:
        result = self.run_command(["pgrep", "-x", "nvidia-cuda-mps-control"],
                                  check=False, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            raise PreflightError("a foreign MPS control daemon is active")

    def preflight(self) -> None:
        """Perform read-only environment and collision checks."""
        data = self.config.data
        if self.run_directory.exists():
            raise PreflightError(
                f"run directory already exists: {self.run_directory}")
        output_parent = self.run_directory.parent
        existing_parent = output_parent
        while (
                not existing_parent.exists() and
                existing_parent != existing_parent.parent):
            existing_parent = existing_parent.parent
        if not os.access(existing_parent, os.W_OK):
            raise PreflightError(
                f"output root is not writable: {output_parent}")
        ros_distro = os.environ.get("ROS_DISTRO")
        if ros_distro != data["ros"]["distribution"]:
            raise PreflightError(
                f"ROS_DISTRO is {ros_distro!r}, expected "
                f"{data['ros']['distribution']!r}")
        if shutil.which("ros2") is None:
            raise PreflightError("ros2 is not on PATH")
        try:
            package_prefix = Path(_command_output(
                ["ros2", "pkg", "prefix", data["ros"]["launch_package"]],
                self.run_command))
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PreflightError(
                "configured ROS launch package is unavailable") from exc
        launch_file = (package_prefix / "share" /
                       data["ros"]["launch_package"] / "launch" /
                       data["ros"]["launch_file"])
        if not launch_file.is_file():
            raise PreflightError(
                f"ROS launch file is unavailable: {launch_file}")
        recording = recording_config(data)
        nsys = recording.get("nsys")
        if nsys:
            installed = detected_nsys_version(self.run_command)
            if installed != nsys["version"]:
                raise PreflightError(
                    f"Nsight Systems {installed} installed, expected "
                    f"{nsys['version']}")
        if recording.get("ncu"):
            installed_ncu = detected_ncu_version(self.run_command)
            if installed_ncu != recording["ncu"]["version"]:
                raise PreflightError(
                    f"Nsight Compute {installed_ncu} installed, expected "
                    f"{recording['ncu']['version']}"
                )
        self._check_gpu_idle()
        self._check_foreign_mps()
        if "preprocessing" in recording["scopes"]:
            backend = recording.get("preprocessing", {}).get(
                "scheduler_backend", "bpftrace"
            )
            if backend == "bpftrace":
                if shutil.which("bpftrace") is None:
                    raise PreflightError(
                        "preprocess_contention_v1 requires bpftrace"
                    )
                if not scheduler_script_path().is_file():
                    raise PreflightError(
                        "packaged scheduler trace program is unavailable"
                    )
                if not scheduler_tracefs_available():
                    raise PreflightError(
                        "preprocess_contention_v1 requires mounted scheduler "
                        "tracepoints; expose tracefs and grant BPF/"
                        "performance monitoring permissions"
                    )

    def command(self) -> List[str]:
        """Return the fully resolved profile command."""
        return build_profile_command(
            self.config, self.run_directory / "profile"
        )

    def _atomic_json(self, path: Path, value: Dict[str, Any]) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix="." + path.name, dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(value, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            path.chmod(0o644)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _terminate_owned_group(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        process_group = os.getpgid(self.process.pid)
        for sig, grace in ((signal.SIGINT, 5), (signal.SIGTERM, 3)):
            os.killpg(process_group, sig)
            try:
                self.process.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                pass
        os.killpg(process_group, signal.SIGKILL)
        self.process.wait()

    def _export_sqlite(self) -> None:
        report = self.run_directory / "profile.nsys-rep"
        if not report.exists():
            raise RunFailure("Nsight Systems report was not produced")
        self.run_command([
            "nsys", "export", "--type", "sqlite", "--force-overwrite=false",
            "--output", str(self.run_directory / "profile.sqlite"),
            str(report)],
            check=True)

    def _merge_model_statuses(self, manifest: Dict[str, Any]) -> None:
        """Merge every readable final model status into the manifest."""
        for model in self.config.models:
            status_path = self.run_directory / f"model_{model['id']}.json"
            if not status_path.is_file():
                continue
            try:
                model_status = json.loads(
                    status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            manifest["models"][model["id"]] = model_status
        fma = self.config.data.get("synthetic_fma")
        if fma:
            status_path = self.run_directory / f"model_{fma['client_id']}.json"
            if status_path.is_file():
                try:
                    manifest["synthetic_fma"] = json.loads(
                        status_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    pass

    def run(self, dry_run: bool = False) -> Optional[Dict[str, Any]]:
        """Run an experiment; dry-run performs only read-only checks."""
        self.preflight()
        command = self.command()
        if dry_run:
            print(" ".join(command))
            return None
        self.run_directory.mkdir(parents=True, exist_ok=False)
        config_copy = self.run_directory / "config.yaml"
        config_copy.write_bytes(self.config.original_bytes)
        config_copy.chmod(0o444)
        resolved_copy = self.run_directory / "resolved_config.yaml"
        resolved_copy.write_text(
            yaml.safe_dump(self.config.data, sort_keys=False), encoding="utf-8"
        )
        resolved_copy.chmod(0o444)
        manifest = {
            "schema_version": 2, "run_id": self.config.data["run"]["id"],
            "config_sha256": self.config.sha256, "state": "running",
            "config_source": str(self.config.source.resolve()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "cpu_topology": detect_cpu_topology(),
            "gpu_hardware": detect_gpu_hardware(
                self.config.data["gpu"]["index"], self.run_command
            ),
            "recording": recording_config(self.config.data),
            "gpu": dict(self.config.data["gpu"]),
            "models": initial_model_manifest(self.config),
        }
        if "synthetic_fma" in self.config.data:
            manifest["synthetic_fma"] = dict(
                self.config.data["synthetic_fma"]
            )
        self._atomic_json(self.run_directory / "run_manifest.json", manifest)
        mps_context = (MPSManager(self.config.data["gpu"]["index"],
                                  self.run_directory,
                                  self.run_command, self.popen)
                       if self.config.data["gpu"]["mps_enabled"]
                       else nullcontext())
        status = "failed"
        error = None
        ncu_control = None
        ncu_log = None
        try:
            with mps_context:
                environment = os.environ.copy()
                removed_mps_environment = {}
                if not self.config.data["gpu"]["mps_enabled"]:
                    environment, removed_mps_environment = (
                        sanitize_mps_environment(environment))
                manifest["removed_mps_environment_variables"] = sorted(
                    removed_mps_environment
                )
                environment["ROS_DOMAIN_ID"] = str(
                    self.config.data["ros"]["domain_id"])
                environment["RMW_IMPLEMENTATION"] = (
                    self.config.data["ros"]["middleware"])
                if recording_config(self.config.data).get("ncu"):
                    ncu_log = (
                        self.run_directory / "ncu_control.log"
                    ).open("wb")
                    ncu_control = self.popen(
                        build_ncu_control_command(
                            self.config, self.run_directory / "profile"
                        ),
                        stdout=ncu_log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                with (self.run_directory / "runner.log").open("wb") as log:
                    self.process = self.popen(command, stdout=log,
                                              stderr=subprocess.STDOUT,
                                              env=environment,
                                              start_new_session=True)
                    if isinstance(mps_context, MPSManager):
                        mps_context.wait_for_server(
                            self.config.data["replay"][
                                "readiness_timeout_seconds"
                            ]
                        )
                    try:
                        return_code = self.process.wait(
                            timeout=self.config.data["run"][
                                "timeout_seconds"])
                    except subprocess.TimeoutExpired as exc:
                        self._terminate_owned_group()
                        raise RunFailure("experiment timed out") from exc
                    if return_code:
                        raise RunFailure(
                            "profiled testbed exited with status "
                            f"{return_code}")
                if ncu_control is not None:
                    try:
                        control_return_code = ncu_control.wait(timeout=300)
                    except subprocess.TimeoutExpired as exc:
                        os.killpg(os.getpgid(ncu_control.pid), signal.SIGTERM)
                        ncu_control.wait()
                        raise RunFailure(
                            "Nsight Compute MPS control timed out"
                        ) from exc
                    if control_return_code:
                        raise RunFailure(
                            "Nsight Compute MPS control exited with status "
                            f"{control_return_code}"
                        )
                if recording_config(self.config.data).get("nsys"):
                    self._export_sqlite()
                if (
                    recording_config(self.config.data).get("ncu")
                    and not (self.run_directory / "profile.ncu-rep").is_file()
                ):
                    raise RunFailure("Nsight Compute report was not produced")
                result_path = self.run_directory / "testbed_result.json"
                if not result_path.exists():
                    raise RunFailure("testbed result was not produced")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                        not result.get("replay_success") or
                        not result.get("all_acknowledged")):
                    raise RunFailure("testbed did not complete successfully")
                for model in self.config.models:
                    status_path = (self.run_directory /
                                   f"model_{model['id']}.json")
                    if not status_path.exists():
                        raise RunFailure(
                            f"status for model {model['id']} was not produced")
                    model_status = json.loads(
                        status_path.read_text(encoding="utf-8"))
                    manifest["models"][model["id"]] = model_status
                    if model_status.get("state") != "acknowledged":
                        raise RunFailure(
                            f"model {model['id']} did not acknowledge "
                            "completion")
                fma = self.config.data.get("synthetic_fma")
                if fma:
                    status_path = (
                        self.run_directory / f"model_{fma['client_id']}.json"
                    )
                    evidence_path = self.run_directory / (
                        f"model_{fma['client_id']}_fma_cta.json"
                    )
                    if (
                        not status_path.is_file()
                        or not evidence_path.is_file()
                    ):
                        raise RunFailure(
                            "synthetic FMA evidence is incomplete"
                        )
                    fma_status = json.loads(
                        status_path.read_text(encoding="utf-8")
                    )
                    manifest["synthetic_fma"] = fma_status
                    if (
                        fma_status.get("state") != "acknowledged"
                        or str(fma_status.get(
                            "process_observed_cuda_mps_active_thread_"
                            "percentage"
                        )) != str(fma["mps_percentage"])
                    ):
                        raise RunFailure(
                            "synthetic FMA did not acknowledge its MPS cap"
                        )
            status = "success"
        except KeyboardInterrupt:
            self._terminate_owned_group()
            error = "interrupted"
            raise
        except Exception as exc:
            self._terminate_owned_group()
            error = str(exc)
            raise
        finally:
            if ncu_control is not None and ncu_control.poll() is None:
                os.killpg(os.getpgid(ncu_control.pid), signal.SIGTERM)
                ncu_control.wait()
            if ncu_log is not None:
                ncu_log.close()
            if isinstance(mps_context, MPSManager):
                manifest["mps"] = dict(mps_context.evidence)
            self._merge_model_statuses(manifest)
            manifest.update({
                "state": status,
                "error": error,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            self._atomic_json(
                self.run_directory / "run_manifest.json", manifest)
        return manifest
