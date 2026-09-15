"""Offline two-client MPS kernel-capsule campaign."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .capsule_coordinator import AgentEndpoint, KernelCapsuleCoordinator
from .capture_session import discover_agent_library
from closeloop_profiler.gpu_forensics_store import GpuForensicsStore
from closeloop_profiler.kernel_capsule import capsule_v3_from_dict
from .mps import MPSManager


SCHEMA = "offline_kernel_capsule_v1"
VARIANTS = ("pair", "aggressor_only", "victim_only")
ROLES = ("aggressor", "victim")


class OfflineCapsuleError(RuntimeError):
    """Raised when offline campaign identity or evidence is incomplete."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workload_config(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: item for key, item in value.items()
        if key not in ("adapter", "library")
    }


def _validate_workload(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise OfflineCapsuleError(f"{name} must be a mapping")
    required = ("adapter", "blocks", "threads", "iterations")
    if any(key not in value for key in required):
        raise OfflineCapsuleError(
            f"{name} requires adapter, blocks, threads, and iterations"
        )
    if not isinstance(value["adapter"], str) or not value["adapter"]:
        raise OfflineCapsuleError(f"{name}.adapter must be non-empty")
    for key in ("blocks", "threads", "iterations"):
        if not isinstance(value[key], int) or value[key] <= 0:
            raise OfflineCapsuleError(f"{name}.{key} must be positive")
    if value["threads"] > 1024:
        raise OfflineCapsuleError(f"{name}.threads exceeds CUDA limit")
    return dict(value)


def load_offline_spec(path: Path) -> Dict[str, Any]:
    """Load and normalize one strict offline capsule YAML specification."""
    path = Path(path).resolve()
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OfflineCapsuleError(f"cannot read offline spec: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise OfflineCapsuleError(f"schema must be {SCHEMA}")
    gpu = value.get("gpu")
    clients = value.get("clients")
    cases = value.get("cases")
    if not isinstance(gpu, dict) or not isinstance(gpu.get("index"), int):
        raise OfflineCapsuleError("gpu.index must be an integer")
    if not isinstance(clients, dict) or set(clients) != set(ROLES):
        raise OfflineCapsuleError("clients must define aggressor and victim")
    for role in ROLES:
        percentage = clients[role].get("mps_active_thread_percentage")
        if not isinstance(percentage, int) or not 1 <= percentage <= 100:
            raise OfflineCapsuleError(
                f"clients.{role}.mps_active_thread_percentage invalid"
            )
    aggressor = _validate_workload(value.get("aggressor"), "aggressor")
    if "release_offset_ns" in aggressor:
        raise OfflineCapsuleError("aggressor release offset is fixed at zero")
    if not isinstance(cases, list) or not cases:
        raise OfflineCapsuleError("cases must be a non-empty list")
    normalized_cases = []
    identifiers = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise OfflineCapsuleError(f"cases[{index}] must be a mapping")
        case_id = case.get("id")
        if (
            not isinstance(case_id, str) or not case_id or
            case_id in identifiers
        ):
            raise OfflineCapsuleError("case IDs must be non-empty and unique")
        identifiers.add(case_id)
        offset = case.get("victim_release_offset_ns")
        if not isinstance(offset, int):
            raise OfflineCapsuleError(
                f"case {case_id} victim_release_offset_ns must be signed int"
            )
        track_cta = case.get("track_cta", False)
        if not isinstance(track_cta, bool):
            raise OfflineCapsuleError(f"case {case_id} track_cta must be bool")
        normalized_cases.append({
            "id": case_id,
            "victim": _validate_workload(
                case.get("victim"), f"case {case_id}.victim"
            ),
            "victim_release_offset_ns": offset,
            "track_cta": track_cta,
        })
    warmups = value.get("warmups", 5)
    repetitions = value.get("timing_repetitions", 10)
    if not isinstance(warmups, int) or warmups < 0:
        raise OfflineCapsuleError("warmups must be a non-negative integer")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise OfflineCapsuleError(
            "timing_repetitions must be a positive integer"
        )
    return {
        "schema": SCHEMA,
        "source": str(path),
        "source_sha256": _sha256_bytes(path.read_bytes()),
        "gpu": {"index": gpu["index"]},
        "clients": clients,
        "warmups": warmups,
        "timing_repetitions": repetitions,
        "aggressor": aggressor,
        "cases": normalized_cases,
    }


def _package_prefix() -> Path:
    try:
        from ament_index_python.packages import get_package_prefix
        return Path(get_package_prefix("closeloop_testbed"))
    except (ImportError, LookupError) as exc:
        raise OfflineCapsuleError(
            "closeloop_testbed is not installed"
        ) from exc


def discover_offline_native() -> Tuple[Path, Path]:
    """Resolve the installed worker and built-in adapter."""
    prefix = _package_prefix()
    worker = prefix / "lib/closeloop_testbed/pperf_offline_worker"
    adapter = prefix / "lib/libpperf_offline_compute_adapter.so"
    if not worker.is_file() or not adapter.is_file():
        raise OfflineCapsuleError(
            "offline native worker/adapter is not installed"
        )
    return worker.resolve(), adapter.resolve()


def _adapter_path(
    workload: Dict[str, Any], builtin: Path, spec_directory: Path
) -> Path:
    if workload["adapter"] == "builtin_compute":
        return builtin
    raw = workload.get("library", workload["adapter"])
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = spec_directory / path
    if not path.is_file():
        raise OfflineCapsuleError(f"adapter library is absent: {path}")
    return path.resolve()


class WorkerProcess:
    """One persistent native CUDA client with a JSON-line control pipe."""

    def __init__(
        self, executable: Path, adapter: Path, workload: Dict[str, Any],
        client_id: str, environment: Dict[str, str], log_path: Path,
        cta: bool = False,
    ):
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
            bufsize=1, start_new_session=True,
        )
        response = self._read(120.0)
        if response.get("status") != "ready":
            raise OfflineCapsuleError(
                f"{client_id} worker did not become ready"
            )
        self.identity = dict(response["identity"])
        self.identity["adapter_library_sha256"] = _sha256_file(adapter)

    def _read(self, timeout: float) -> Dict[str, Any]:
        if self.process.stdout is None:
            raise OfflineCapsuleError("worker stdout is unavailable")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [self.process.stdout], [], [], deadline - time.monotonic()
            )
            if not ready:
                break
            line = self.process.stdout.readline()
            if not line:
                raise OfflineCapsuleError(
                    f"{self.client_id} worker exited with "
                    f"{self.process.poll()}"
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # NVBit 1.8 writes its startup warning to stdout.
                self.log.write(line.encode("utf-8"))
                self.log.flush()
        raise TimeoutError(f"{self.client_id} worker response timeout")

    def send(self, value: Dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise OfflineCapsuleError("worker stdin is unavailable")
        self.process.stdin.write(_canonical(value) + "\n")
        self.process.stdin.flush()

    def receive(self, timeout: float = 120.0) -> Dict[str, Any]:
        response = self._read(timeout)
        if response.get("status") != "ok":
            raise OfflineCapsuleError(str(response.get("error", response)))
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
            raise OfflineCapsuleError(
                f"{self.client_id} worker cleanup exit "
                f"{self.process.returncode}"
            )


def _variant_order(repetition: int) -> Tuple[str, ...]:
    return VARIANTS if repetition % 2 == 0 else tuple(reversed(VARIANTS))


def _enabled(variant: str, role: str) -> bool:
    return variant == "pair" or variant == f"{role}_only"


def _combined_output(records: Iterable[Dict[str, Any]]) -> str:
    values = {
        record["client_id"]: record["output_sha256"] for record in records
    }
    return _sha256_bytes(_canonical(values).encode("utf-8"))


def _relationship(
    start: int, end: int, aggressor_start: Optional[int],
    aggressor_end: Optional[int],
) -> Tuple[int, str]:
    if aggressor_start is None or aggressor_end is None:
        return 0, "solo"
    overlap = max(0, min(end, aggressor_end) - max(start, aggressor_start))
    if overlap == 0:
        return 0, "no_overlap"
    if start < aggressor_start:
        return overlap, "beginning"
    if end > aggressor_end:
        return overlap, "tail"
    return overlap, "contained"


def _normalize_record(
    case: Dict[str, Any], mode: str, variant: str, repetition: int,
    records: List[Dict[str, Any]], identities: Dict[str, Dict[str, Any]],
    checkpoint_restores: int = 0,
) -> Dict[str, Any]:
    activities = []
    for record in records:
        role = record["client_id"]
        descriptors = identities[role]["launches"]
        for activity in record.get("launch_activities", []):
            ordinal = int(activity.get("ordinal", 0))
            launch_id = "aggressor" if role == "aggressor" else (
                f"victim-{ordinal}"
            )
            descriptor = descriptors[ordinal]
            activities.append({
                **activity,
                "launch_id": launch_id,
                "client_id": role,
                "context_handle": str(activity.get("context_handle", mode)),
                "stream_handle": str(activity.get("stream_handle", mode)),
                "submission_ordinal": ordinal,
                "driver_command_fingerprint": activity.get(
                    "driver_command_fingerprint",
                    _sha256_bytes(
                        f"{identities[role]['workload_fingerprint']}:{ordinal}"
                        .encode("utf-8")
                    ),
                ),
                "workload_fingerprint": identities[role][
                    "workload_fingerprint"
                ],
                "launch_descriptor": descriptor,
            })
    activities.sort(key=lambda item: item["driver_issue_start_ns"])
    aggressor = next(
        (item for item in activities if item["launch_id"] == "aggressor"),
        None,
    )
    for activity in activities:
        overlap, relation = _relationship(
            int(activity["gpu_start_ns"]), int(activity["gpu_end_ns"]),
            None if aggressor is None else int(aggressor["gpu_start_ns"]),
            None if aggressor is None else int(aggressor["gpu_end_ns"]),
        )
        activity["aggressor_overlap_duration_ns"] = overlap
        activity["overlap_relationship"] = relation
        ready = int(activity["driver_issue_end_ns"])
        activity["replay_ready_ns"] = ready
        activity["admission_wait_ns"] = (
            int(activity["gpu_start_ns"]) - ready
        )
        activity["gpu_duration_ns"] = (
            int(activity["gpu_end_ns"]) - int(activity["gpu_start_ns"])
        )
        activity["issue_to_gpu_start_wait_ns"] = (
            int(activity["gpu_start_ns"]) -
            int(activity["driver_issue_end_ns"])
        )
    starts = [int(item["gpu_start_ns"]) for item in activities]
    ends = [int(item["gpu_end_ns"]) for item in activities]
    victim = next(
        (item for item in activities if item["client_id"] == "victim"), None
    )
    return {
        "offline_case_id": case["id"],
        "execution_mode": mode,
        "phase": "timing",
        "variant": variant,
        "repetition": repetition,
        "checkpoint_restore_count": checkpoint_restores,
        "victim_delay_ns": None if victim is None else victim[
            "admission_wait_ns"
        ],
        "aggressor_duration_ns": None if aggressor is None else aggressor[
            "gpu_duration_ns"
        ],
        "victim_order": ["victim-0"],
        "launch_order": [item["launch_id"] for item in activities],
        "fingerprint_match": all(
            record.get("workload_fingerprint") ==
            identities[record["client_id"]]["workload_fingerprint"]
            for record in records
        ),
        "foreign_work_count": sum(
            int(record.get("foreign_work_count", 0)) for record in records
        ),
        "profiler_perturbation": "none",
        "output_sha256": _combined_output(records),
        "launch_activities": activities,
        "workload_start_ns": min(starts) if starts else None,
        "workload_end_ns": max(ends) if ends else None,
        "workload_duration_ns": max(ends) - min(starts) if starts else None,
        "actual_relative_gpu_start_ns": None if not aggressor or not victim
        else int(victim["gpu_start_ns"]) - int(aggressor["gpu_start_ns"]),
        "requested_victim_release_offset_ns": case[
            "victim_release_offset_ns"
        ],
    }


class DirectSession:
    """Two persistent timing workers under one already-owned MPS daemon."""

    def __init__(
        self, executable: Path, adapters: Dict[str, Path],
        workloads: Dict[str, Dict[str, Any]], mps: MPSManager,
        clients: Dict[str, Dict[str, Any]], log_directory: Path,
        tracker: Optional[Path] = None,
    ):
        self.workers = {}
        cta = tracker is not None
        for role in ROLES:
            environment = os.environ.copy()
            environment.update(mps.model_environment(
                clients[role]["mps_active_thread_percentage"]
            ))
            environment.pop("CUDA_INJECTION64_PATH", None)
            if tracker is not None:
                environment["NOBANNER"] = "1"
                preload = environment.get("LD_PRELOAD", "").strip()
                environment["LD_PRELOAD"] = " ".join(
                    item for item in (str(tracker), preload) if item
                )
            self.workers[role] = WorkerProcess(
                executable, adapters[role], workloads[role], role,
                environment, log_directory / f"{role}.log", cta,
            )
        self.identities = {
            role: worker.identity for role, worker in self.workers.items()
        }

    def run(self, variant: str, victim_offset_ns: int) -> List[Dict[str, Any]]:
        release = time.monotonic_ns() + 20_000_000
        for role, worker in self.workers.items():
            worker.send({
                "operation": "run",
                "release_ns": release + (
                    victim_offset_ns if role == "victim" else 0
                ),
                "enabled": _enabled(variant, role),
            })
        return [self.workers[role].receive() for role in ROLES]

    def close(self) -> None:
        errors = []
        for worker in self.workers.values():
            try:
                worker.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise OfflineCapsuleError(
                "; ".join(str(value) for value in errors)
            )


def _launch_spec(
    descriptor: Dict[str, Any], role: str, source_id: str,
    sequence_index: int = 0, occurrence: int = 0,
) -> Dict[str, Any]:
    return {
        "source_kernel_id": source_id,
        "model_identity": role,
        "capture_matcher": {
            "launch_api": descriptor["launch_api"],
            "symbol": descriptor["symbol"],
            "mangled_symbol": descriptor["symbol"],
            "grid": descriptor["grid"],
            "block": descriptor["block"],
            "dynamic_shared_memory": descriptor[
                "dynamic_shared_memory"
            ],
            "framework_owner": "",
        },
        "frame_local_launch_occurrence_ordinal": occurrence,
        "frame_local_launch_sequence_index": sequence_index,
        "sequence_anchors": [],
        "stream_priority": 0,
        "stream_id": 0,
        "launch_start_ns": 0,
        "launch_end_ns": 1,
        "start_ns": 0,
        "end_ns": 1,
        "source_gpu_interval": {"start_ns": 0, "end_ns": 1},
    }


def _capture_spec(identities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if len(identities["aggressor"]["launches"]) != 1:
        raise OfflineCapsuleError(
            "live v3 comparison requires one aggressor launch"
        )
    aggressor = _launch_spec(
        identities["aggressor"]["launches"][0], "aggressor", "a"
    )
    victims = []
    occurrences = {}
    for index, descriptor in enumerate(identities["victim"]["launches"]):
        key = _canonical({
            name: descriptor[name] for name in (
                "launch_api", "symbol", "grid", "block",
                "dynamic_shared_memory",
            )
        })
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        victims.append(_launch_spec(
            descriptor, "victim", f"v-{index}", index, occurrence
        ))
    return {
        "schema": "kernel_capsule_capture_spec_v3",
        "source_run_id": "offline_builtin_compute",
        "aggressor": aggressor,
        "victim_kernels": victims,
        "victim_head_source_kernel_id": "v-0",
        "inference_inputs": {"aggressor": "offline", "victim": "offline"},
        "source_frame": {"ros_header_timestamp_ns": 1},
        "overlap_set_complete": True,
        "overlap_evidence": {
            "complete": True,
            "aggressor_source_interval": {
                "start_ns": 0, "end_ns": 1,
            },
            "victim_source_intervals": [{
                "source_kernel_id": value["source_kernel_id"],
                "source_gpu_interval": {
                    "start_ns": 0, "end_ns": 1,
                },
                "overlap_duration_ns": 1,
                "stream_membership": "offline_single_stream",
            } for value in victims],
        },
        "submission_policy": "common_epoch_dependency_burst",
        "client_release_offsets_ns": {"aggressor": 0, "victim": 0},
    }


class OnlineSession:
    """Capture the same worker pair once, then drive live agent replay."""

    def __init__(
        self, executable: Path, adapters: Dict[str, Path],
        workloads: Dict[str, Dict[str, Any]], identities: Dict[str, Any],
        mps: MPSManager, clients: Dict[str, Dict[str, Any]],
        agent_library: Path, directory: Path,
    ):
        self.processes = []
        self.logs = []
        self.directory = directory
        specification = _capture_spec(identities)
        spec_path = directory / "capture_specification.json"
        spec_path.write_text(
            json.dumps(specification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        endpoints = []
        for role in ROLES:
            socket_path = directory / f"agent-{role}.sock"
            status_path = directory / f"status-{role}.json"
            environment = os.environ.copy()
            environment.update(mps.model_environment(
                clients[role]["mps_active_thread_percentage"]
            ))
            environment.update({
                "CUDA_INJECTION64_PATH": str(agent_library),
                "PPERF_KERNEL_CAPSULE_SPEC": str(spec_path),
                "PPERF_KERNEL_CAPSULE_CLIENT_ID": role,
                "PPERF_KERNEL_CAPSULE_AGENT_SOCKET": str(socket_path),
            })
            log = (directory / f"capture_{role}.log").open("wb")
            self.logs.append(log)
            process = subprocess.Popen([
                str(executable), "--adapter", str(adapters[role]),
                "--config-json", _canonical(_workload_config(workloads[role])),
                "--client-id", role, "--capture", "--status",
                str(status_path),
            ], env=environment, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            self.processes.append(process)
            self._wait_ready(socket_path, status_path)
            endpoints.append(AgentEndpoint(role, socket_path, 120.0))
        self.coordinator = KernelCapsuleCoordinator(endpoints, 120.0)
        self.capsule = self.coordinator.capture(specification)["capsule"]
        capsule_v3_from_dict(self.capsule).validate()
        self.identities = {}
        for role in ROLES:
            status = json.loads(
                (directory / f"status-{role}.json").read_text(encoding="utf-8")
            )
            self.identities[role] = status["identity"]
            self.identities[role]["adapter_library_sha256"] = _sha256_file(
                adapters[role]
            )

    def _wait_ready(self, socket_path: Path, status_path: Path) -> None:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            failures = [
                process.returncode for process in self.processes
                if process.poll() is not None
            ]
            if failures:
                raise OfflineCapsuleError(
                    f"online capture worker exited early: {failures}"
                )
            if socket_path.exists() and status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("state") == "warmup_ready":
                    return
            time.sleep(0.05)
        raise TimeoutError(f"online capture readiness timeout: {socket_path}")

    def run(self, variant: str, victim_offset_ns: int) -> Dict[str, Any]:
        restored = self.coordinator.restore(self.capsule["capsule_id"])
        record = self.coordinator.replay({
            "capsule_id": self.capsule["capsule_id"],
            "variant": variant,
            "instrumentation": "none",
            "client_release_offsets_ns": {
                "aggressor": 0, "victim": victim_offset_ns,
            },
        })
        if not record.get("fingerprint_match", False) or any(
            int(record.get(key, 0)) for key in (
                "foreign_work_count", "non_capsule_cuda_work_count"
            )
        ):
            raise OfflineCapsuleError("live replay integrity rejection")
        record["checkpoint_restore_count"] = restored[
            "restored_client_count"
        ]
        by_client = {role: {
            "client_id": role,
            "output_sha256": record["client_output_sha256"][role],
            "workload_fingerprint": self.identities[role][
                "workload_fingerprint"
            ],
            "launch_activities": [],
            "foreign_work_count": 0,
        } for role in ROLES}
        for activity in record["launch_activities"]:
            role = activity["client_id"]
            ordinal = 0 if role == "aggressor" else int(
                activity["launch_id"].split("-")[-1]
            )
            by_client[role]["launch_activities"].append({
                **activity, "ordinal": ordinal,
            })
        return {
            "clients": list(by_client.values()),
            "checkpoint_restore_count": record[
                "checkpoint_restore_count"
            ],
            "raw": record,
        }

    def close(self) -> None:
        errors = []
        try:
            self.coordinator.dispatch({"operation": "shutdown"})
        except Exception as exc:
            errors.append(exc)
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        for log in self.logs:
            log.close()
        if errors:
            raise OfflineCapsuleError(
                "; ".join(str(value) for value in errors)
            )


def _percentile(values: Iterable[int], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise OfflineCapsuleError("cannot summarize empty timing values")
    return float(ordered[max(0, math.ceil(percentile * len(ordered)) - 1)])


def _median(values: Iterable[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def build_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-kernel/workload distributions and context deltas."""
    rows = []
    keys = sorted({
        (record["offline_case_id"], record["execution_mode"],
         record["variant"]) for record in records
    })
    for case_id, mode, variant in keys:
        selected = [
            record for record in records
            if (record["offline_case_id"], record["execution_mode"],
                record["variant"]) == (case_id, mode, variant)
        ]
        workload = [record["workload_duration_ns"] for record in selected]
        rows.append({
            "case_id": case_id, "execution_mode": mode,
            "variant": variant, "scope": "workload",
            "metric": "duration_ns", "median": _median(workload),
            "p95": _percentile(workload, 0.95),
        })
        for launch_id in ("aggressor", "victim-0"):
            activities = [
                activity for record in selected
                for activity in record["launch_activities"]
                if activity["launch_id"] == launch_id
            ]
            if not activities:
                continue
            for metric in (
                "gpu_duration_ns", "admission_wait_ns",
                "aggressor_overlap_duration_ns",
            ):
                values = [int(activity[metric]) for activity in activities]
                rows.append({
                    "case_id": case_id, "execution_mode": mode,
                    "variant": variant, "scope": launch_id,
                    "metric": metric, "median": _median(values),
                    "p95": _percentile(values, 0.95),
                })
    lookup = {
        (row["case_id"], row["execution_mode"], row["variant"],
         row["scope"], row["metric"]): row for row in rows
    }
    pair_minus_solo = []
    for row in rows:
        if row["variant"] != "pair" or row["scope"] == "workload":
            continue
        solo = "aggressor_only" if row["scope"] == "aggressor" else (
            "victim_only"
        )
        other = lookup.get((
            row["case_id"], row["execution_mode"], solo, row["scope"],
            row["metric"],
        ))
        if other:
            pair_minus_solo.append({
                **{key: row[key] for key in (
                    "case_id", "execution_mode", "scope", "metric"
                )},
                "median_delta": row["median"] - other["median"],
                "p95_delta": row["p95"] - other["p95"],
            })
    for case_id, mode in sorted({
        (case_id, mode) for case_id, mode, _variant in keys
    }):
        pair = lookup.get((
            case_id, mode, "pair", "workload", "duration_ns"
        ))
        if pair is None:
            continue
        for solo in ("aggressor_only", "victim_only"):
            other = lookup.get((
                case_id, mode, solo, "workload", "duration_ns"
            ))
            if other:
                pair_minus_solo.append({
                    "case_id": case_id, "execution_mode": mode,
                    "scope": f"workload_vs_{solo}",
                    "metric": "duration_ns",
                    "median_delta": pair["median"] - other["median"],
                    "p95_delta": pair["p95"] - other["p95"],
                })
    mode_deltas = []
    for row in rows:
        if row["execution_mode"] != "direct":
            continue
        online = lookup.get((
            row["case_id"], "live_capsule", row["variant"], row["scope"],
            row["metric"],
        ))
        if online:
            for statistic in ("median", "p95"):
                delta = row[statistic] - online[statistic]
                mode_deltas.append({
                    **{key: row[key] for key in (
                        "case_id", "variant", "scope", "metric"
                    )},
                    "statistic": statistic,
                    "offline_minus_online": delta,
                    "percent": None if online[statistic] == 0 else
                    100.0 * delta / online[statistic],
                })
    return {
        "schema": "offline_vs_online_report_v1",
        "interpretation": (
            "lightweight-context/capsule-path effects; not DETR/DeepLab "
            "runtime-context reproduction"
        ),
        "summaries": rows,
        "pair_minus_solo": pair_minus_solo,
        "offline_minus_online": mode_deltas,
    }


class OfflineCampaign:
    """Own MPS, run direct/live variants, and write canonical evidence."""

    def __init__(
        self, spec_path: Path, output: Path, compare_online: bool = False,
        nvbit_cta: bool = False, agent_library: Optional[Path] = None,
        tracker_library: Optional[Path] = None,
    ):
        self.spec = load_offline_spec(spec_path)
        self.output = Path(output).resolve()
        self.compare_online = compare_online
        self.nvbit_cta = nvbit_cta
        self.worker, self.builtin_adapter = discover_offline_native()
        self.agent_library = discover_agent_library(agent_library) \
            if compare_online else None
        self.tracker_library = Path(tracker_library).resolve() \
            if tracker_library else self._default_tracker()
        self.records: List[Dict[str, Any]] = []
        self.cta_records: List[Dict[str, Any]] = []

    def _default_tracker(self) -> Optional[Path]:
        candidate = self.builtin_adapter.with_name(
            "libpperf_nvbit_cta_tracker.so"
        )
        return candidate if candidate.is_file() else None

    def _workloads(self, case: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {"aggressor": self.spec["aggressor"],
                "victim": case["victim"]}

    def _adapters(
        self, workloads: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Path]:
        directory = Path(self.spec["source"]).parent
        return {role: _adapter_path(
            workloads[role], self.builtin_adapter, directory
        ) for role in ROLES}

    @staticmethod
    def _identity_key(case: Dict[str, Any]) -> str:
        return _canonical(case["victim"])

    def _run_direct_group(
        self, cases: List[Dict[str, Any]], mps: MPSManager, root: Path,
    ) -> Dict[str, Dict[str, Any]]:
        workloads = self._workloads(cases[0])
        session = DirectSession(
            self.worker, self._adapters(workloads), workloads, mps,
            self.spec["clients"], root,
        )
        try:
            for case in cases:
                for repetition in range(self.spec["warmups"]):
                    for variant in _variant_order(repetition):
                        session.run(
                            variant, case["victim_release_offset_ns"]
                        )
                for repetition in range(self.spec["timing_repetitions"]):
                    for variant in _variant_order(repetition):
                        raw = session.run(
                            variant, case["victim_release_offset_ns"]
                        )
                        self.records.append(_normalize_record(
                            case, "direct", variant, repetition, raw,
                            session.identities,
                        ))
            return session.identities
        finally:
            session.close()

    def _run_online_group(
        self, cases: List[Dict[str, Any]], identities: Dict[str, Any],
        mps: MPSManager, root: Path,
    ) -> Dict[str, Any]:
        workloads = self._workloads(cases[0])
        session = OnlineSession(
            self.worker, self._adapters(workloads), workloads, identities,
            mps, self.spec["clients"], self.agent_library, root,
        )
        try:
            if {
                role: (
                    session.identities[role]["workload_fingerprint"],
                    session.identities[role]["adapter_library_sha256"],
                )
                for role in ROLES
            } != {
                role: (
                    identities[role]["workload_fingerprint"],
                    identities[role]["adapter_library_sha256"],
                )
                for role in ROLES
            }:
                raise OfflineCapsuleError(
                    "direct/live adapter fingerprints differ"
                )
            for case in cases:
                for repetition in range(self.spec["warmups"]):
                    for variant in _variant_order(repetition):
                        session.run(
                            variant, case["victim_release_offset_ns"]
                        )
                for repetition in range(self.spec["timing_repetitions"]):
                    for variant in _variant_order(repetition):
                        raw = session.run(
                            variant, case["victim_release_offset_ns"]
                        )
                        record = _normalize_record(
                            case, "live_capsule", variant, repetition,
                            raw["clients"], session.identities,
                            raw["checkpoint_restore_count"],
                        )
                        record["raw_online_record"] = raw["raw"]
                        self.records.append(record)
            return session.capsule
        finally:
            session.close()

    def _run_cta(
        self, case: Dict[str, Any], mps: MPSManager, root: Path,
    ) -> Dict[str, Any]:
        if self.tracker_library is None or not self.tracker_library.is_file():
            raise OfflineCapsuleError("NVBit CTA tracker is not installed")
        workloads = self._workloads(case)
        session = DirectSession(
            self.worker, self._adapters(workloads), workloads, mps,
            self.spec["clients"], root, self.tracker_library,
        )
        try:
            # ponytail: one second covers NVBit's one-time disassembly; split
            # prepare/launch RPCs only if collector startup becomes material.
            release = time.monotonic_ns() + 1_000_000_000
            for role, worker in session.workers.items():
                worker.send({
                    "operation": "run",
                    "release_ns": release + (
                        case["victim_release_offset_ns"]
                        if role == "victim" else 0
                    ),
                    "enabled": True,
                })
            records = [session.workers[role].receive() for role in ROLES]
            intervals = []
            statuses = []
            instances = []
            calibrations = []
            for record in records:
                role = record["client_id"]
                descriptors = session.identities[role]["launches"]
                launch_id = "aggressor" if role == "aggressor" else "victim-0"
                for value in record["cta_intervals"]:
                    intervals.append({**value, "launch_id": launch_id,
                                      "client_id": role})
                for value in record["cta_collection_status"]:
                    statuses.append({**value, "launch_id": launch_id,
                                     "client_id": role})
                calibrations.extend(record["clock_calibrations"])
                instances.append({
                    "launch_id": launch_id, "client_id": role,
                    "workload_fingerprint": session.identities[role][
                        "workload_fingerprint"
                    ],
                    "launch_descriptor": descriptors[0],
                })
            result = {
                "offline_case_id": case["id"],
                "execution_mode": "direct",
                "phase": "cta", "variant": "pair", "repetition": 0,
                "checkpoint_restore_count": 0,
                "victim_order": ["victim-0"],
                "fingerprint_match": True, "foreign_work_count": 0,
                "profiler_perturbation": "nvbit_cta",
                "timing_excluded_from_conclusions": True,
                "kernel_instances": instances,
                "cta_intervals": intervals,
                "cta_collection_status": statuses,
                "clock_calibrations": calibrations,
            }
            self.cta_records.append(result)
            return {"available": True, "record_count": len(intervals)}
        finally:
            session.close()

    def _acceptance(self) -> Dict[str, Any]:
        expected = len(self.spec["cases"]) * len(VARIANTS) * self.spec[
            "timing_repetitions"
        ]
        modes = ["direct"] + (["live_capsule"] if self.compare_online else [])
        counts = {mode: sum(
            record["execution_mode"] == mode for record in self.records
        ) for mode in modes}
        if any(value != expected for value in counts.values()):
            raise OfflineCapsuleError(
                f"timing iteration count mismatch: {counts}, "
                f"expected {expected}"
            )
        if any(not record["fingerprint_match"] for record in self.records):
            raise OfflineCapsuleError("adapter/launch integrity mismatch")
        if any(
            int(activity["gpu_end_ns"]) <= int(activity["gpu_start_ns"])
            for record in self.records
            for activity in record["launch_activities"]
        ):
            raise OfflineCapsuleError("invalid CUPTI timing interval")
        if self.compare_online:
            for case in self.spec["cases"]:
                for variant in VARIANTS:
                    hashes = {}
                    for mode in modes:
                        values = {
                            record["output_sha256"] for record in self.records
                            if record["offline_case_id"] == case["id"] and
                            record["variant"] == variant and
                            record["execution_mode"] == mode
                        }
                        if len(values) != 1:
                            raise OfflineCapsuleError(
                                "nondeterministic workload output"
                            )
                        hashes[mode] = next(iter(values))
                    if hashes["direct"] != hashes["live_capsule"]:
                        raise OfflineCapsuleError(
                            "direct/live output hash mismatch"
                        )
        cta = {"requested": self.nvbit_cta}
        if self.nvbit_cta:
            intervals = [
                value for record in self.cta_records
                for value in record["cta_intervals"]
            ]
            statuses = [
                value for record in self.cta_records
                for value in record["cta_collection_status"]
            ]
            keys = {
                (value["launch_id"], value["cta_id"])
                for value in intervals
            }
            co_residency = sum(
                1 for index, first in enumerate(intervals)
                for second in intervals[index + 1:]
                if first["client_id"] != second["client_id"] and
                first.get("sm_id") is not None and
                first.get("sm_id") == second.get("sm_id") and
                first.get("entry_ns") is not None and
                first.get("exit_ns") is not None and
                second.get("entry_ns") is not None and
                second.get("exit_ns") is not None and
                min(first["exit_ns"], second["exit_ns"]) >
                max(first["entry_ns"], second["entry_ns"])
            )
            cta.update({
                "available": bool(self.cta_records),
                "row_count": len(intervals),
                "unique_rows": len(keys) == len(intervals),
                "complete": bool(statuses) and all(
                    value["complete_count"] == value["expected_count"]
                    for value in statuses
                ),
                "same_sm_co_residency_intervals": co_residency,
            })
        cta_passed = (
            not self.nvbit_cta or
            cta.get("available", False) and
            cta.get("unique_rows", False) and
            cta.get("complete", False) and
            cta.get("same_sm_co_residency_intervals", 0) > 0
        )
        return {
            "status": "passed" if cta_passed else
            "timing_passed_optional_cta_failed",
            "timing_iterations": counts,
            "expected_per_mode": expected, "cta": cta,
        }

    def _write_database(self, normalized: Dict[str, Any]) -> None:
        with GpuForensicsStore(
            self.output / "gpu_forensics.sqlite"
        ) as store:
            store.set_metadata("offline_capsule", normalized)
            for case in self.spec["cases"]:
                store.record_offline_case(case["id"], case)
            for record in self.records + self.cta_records:
                record["iteration_id"] = store.record_iteration(
                    record["phase"], record["variant"],
                    record["repetition"], record,
                    record["offline_case_id"], record["execution_mode"],
                )

    def _write_csvs(self) -> None:
        timing_fields = (
            "offline_case_id", "execution_mode", "variant", "repetition",
            "requested_victim_release_offset_ns",
            "actual_relative_gpu_start_ns", "workload_start_ns",
            "workload_end_ns", "workload_duration_ns", "output_sha256",
        )
        with (self.output / "offline_timing.csv").open(
            "w", encoding="utf-8", newline=""
        ) as output:
            writer = csv.DictWriter(output, fieldnames=timing_fields)
            writer.writeheader()
            writer.writerows({key: record.get(key) for key in timing_fields}
                             for record in self.records)
        kernel_fields = (
            "offline_case_id", "execution_mode", "variant", "repetition",
            "launch_id", "client_id", "driver_issue_start_ns",
            "driver_issue_end_ns", "gpu_start_ns", "gpu_end_ns",
            "gpu_duration_ns", "replay_ready_ns", "admission_wait_ns",
            "issue_to_gpu_start_wait_ns", "aggressor_overlap_duration_ns",
            "overlap_relationship",
        )
        with (self.output / "offline_kernel_timing.csv").open(
            "w", encoding="utf-8", newline=""
        ) as output:
            writer = csv.DictWriter(output, fieldnames=kernel_fields)
            writer.writeheader()
            for record in self.records:
                for activity in record["launch_activities"]:
                    writer.writerow({
                        **{key: record.get(key) for key in kernel_fields},
                        **{key: activity.get(key) for key in kernel_fields},
                    })
        cta_fields = (
            "offline_case_id", "execution_mode", "launch_id", "client_id",
            "cta_id", "sm_id", "entry_ns", "exit_ns", "clock_error_ns",
            "observation_status", "entry_observed", "exit_observed",
        )
        with (self.output / "offline_cta.csv").open(
            "w", encoding="utf-8", newline=""
        ) as output:
            writer = csv.DictWriter(output, fieldnames=cta_fields)
            writer.writeheader()
            for record in self.cta_records:
                for interval in record["cta_intervals"]:
                    writer.writerow({
                        **{key: record.get(key) for key in cta_fields},
                        **{key: interval.get(key) for key in cta_fields},
                    })

    def run(self) -> Dict[str, Any]:
        """Execute the requested campaign without mutating existing output."""
        self.output.mkdir(parents=True, exist_ok=False)
        temporary = tempfile.TemporaryDirectory(prefix="pperf-offline-")
        runtime = Path(temporary.name)
        groups = {}
        for case in self.spec["cases"]:
            groups.setdefault(self._identity_key(case), []).append(case)
        captures = {}
        identities = {}
        cta_status = {"requested": self.nvbit_cta, "available": False}
        try:
            with MPSManager(self.spec["gpu"]["index"], runtime) as mps:
                for index, cases in enumerate(groups.values()):
                    direct_root = runtime / f"direct-{index}"
                    direct_root.mkdir()
                    direct_identity = self._run_direct_group(
                        cases, mps, direct_root
                    )
                    identities[cases[0]["id"]] = direct_identity
                    if self.compare_online:
                        online_root = runtime / f"online-{index}"
                        online_root.mkdir()
                        captures[cases[0]["id"]] = self._run_online_group(
                            cases, direct_identity, mps, online_root
                        )
                if self.nvbit_cta:
                    for case in self.spec["cases"]:
                        if not case["track_cta"]:
                            continue
                        cta_root = runtime / f"cta-{case['id']}"
                        cta_root.mkdir()
                        try:
                            cta_status = {
                                "requested": True,
                                **self._run_cta(case, mps, cta_root),
                            }
                        except Exception as exc:
                            cta_status = {
                                "requested": True, "available": False,
                                "error": str(exc),
                                "ordinary_timing_usable": True,
                            }
            normalized = {
                **self.spec,
                "worker": str(self.worker),
                "worker_sha256": _sha256_file(self.worker),
                "identities": identities,
                "live_capsules": captures,
                "nvbit_cta": cta_status,
            }
            acceptance = self._acceptance()
            self._write_database(normalized)
            self._write_csvs()
            report = build_report(self.records)
            report["acceptance"] = acceptance
            report["nvbit_cta"] = cta_status
            if self.compare_online and self.nvbit_cta:
                report["online_cta"] = {
                    "available": False,
                    "reason": (
                        "live agent and NVBit require separate legacy "
                        "CUPTI callback subscribers on CUDA 12.6"
                    ),
                }
            (self.output / "offline_capsule.json").write_text(
                json.dumps(normalized, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (self.output / "offline_vs_online_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            markdown = [
                "# Offline capsule comparison", "",
                report["interpretation"] + ".", "",
                f"Timing iterations: `{acceptance['timing_iterations']}`.",
                f"NVBit CTA: `{cta_status}`.", "",
                "See `offline_vs_online_report.json` for median/p95, "
                "pair-minus-solo, and direct-minus-live deltas.",
            ]
            (self.output / "offline_vs_online_report.md").write_text(
                "\n".join(markdown) + "\n", encoding="utf-8"
            )
            temporary.cleanup()
            return report
        except Exception as exc:
            if runtime.exists():
                shutil.copytree(
                    runtime, self.output / "runtime_failure",
                    dirs_exist_ok=True,
                )
            temporary.cleanup()
            (self.output / "offline_capsule_failure.json").write_text(
                json.dumps({
                    "schema": "offline_capsule_failure_v1",
                    "error_type": type(exc).__name__, "error": str(exc),
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            raise


def main(argv=None) -> int:
    """CLI for direct offline and optional live-capsule comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("output")
    parser.add_argument("--compare-online", action="store_true")
    parser.add_argument("--nvbit-cta", action="store_true")
    parser.add_argument("--agent-library")
    parser.add_argument("--tracker-library")
    args = parser.parse_args(argv)
    try:
        OfflineCampaign(
            Path(args.spec), Path(args.output), args.compare_online,
            args.nvbit_cta,
            Path(args.agent_library) if args.agent_library else None,
            Path(args.tracker_library) if args.tracker_library else None,
        ).run()
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
