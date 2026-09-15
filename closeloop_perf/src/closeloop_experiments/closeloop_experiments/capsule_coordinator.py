"""Host coordinator for dependency-burst live replay agents."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
import time
from typing import Any, Dict, Iterable, List
import uuid

from .rpc_transport import receive_json_line

PROTOCOL = "kernel_capsule_agent_v3"


class AgentEndpoint:
    """One request/response endpoint exposed by an injected MPS client."""

    def __init__(
        self, client_id: str, socket_path: Path, timeout_seconds: float = 120.0
    ):
        """Configure one stable client identity and local agent socket."""
        self.client_id = client_id
        self.socket_path = Path(socket_path)
        self.timeout_seconds = timeout_seconds

    def request(self, operation: str, **payload: Any) -> Dict[str, Any]:
        """Send one local RPC to the blocked live client agent."""
        message = {
            "protocol": PROTOCOL,
            "operation": operation,
            "client_id": self.client_id,
            **payload,
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_seconds)
            connection.connect(str(self.socket_path))
            connection.sendall(
                (json.dumps(message, sort_keys=True) + "\n").encode("utf-8")
            )
            response = receive_json_line(connection)
        if response.get("protocol") != PROTOCOL:
            raise RuntimeError("injected agent protocol mismatch")
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("error", "agent failure")))
        return dict(response.get("result", {}))


class KernelCapsuleCoordinator:
    """Coordinate exact capture and replay for participating clients."""

    def __init__(
        self, agents: Iterable[AgentEndpoint], lease_seconds: float = 5.0
    ):
        """Require at least two distinct live MPS client endpoints."""
        self.agents = {agent.client_id: agent for agent in agents}
        if len(self.agents) < 2:
            raise ValueError("kernel_capsule_v3 requires at least two clients")
        self.capsule = None
        self.launch_clients: Dict[str, str] = {}
        self.lease_id = str(uuid.uuid4())
        self.lease_seconds = float(lease_seconds)
        self.shutdown_requested = False

    def _lease(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "lease_expires_monotonic_ns": (
                time.monotonic_ns() + int(self.lease_seconds * 1e9)
            ),
        }

    def _parallel(
        self, operation: str, **payload: Any
    ) -> List[Dict[str, Any]]:
        payload = {**self._lease(), **payload}
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = [
                executor.submit(agent.request, operation, **payload)
                for agent in self.agents.values()
            ]
            return [future.result() for future in futures]

    @staticmethod
    def _merge_capabilities(
        records: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        records = list(records)
        names = set().union(
            *(set(record.get("collectors", {})) for record in records)
        )
        merged = {}
        for name in names:
            values = [
                record.get("collectors", {}).get(name, {})
                for record in records
            ]
            merged[name] = {
                "available": all(
                    bool(value.get("available", False)) for value in values
                ),
                "version": next(
                    (
                        value.get("version")
                        for value in values
                        if value.get("version")
                    ),
                    None,
                ),
                "reason": next(
                    (
                        value.get("reason")
                        for value in values
                        if not value.get("available", False)
                    ),
                    None,
                ),
            }
        return {"collectors": merged}

    def capabilities(self) -> Dict[str, Any]:
        """Report collectors available in both live client processes."""
        return self._merge_capabilities(self._parallel("capabilities"))

    def capture(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        """Arm both clients before the same single instrumented frame."""
        capture_release_ns = time.monotonic_ns() + 10_000_000
        fragments = self._parallel(
            "capture_one_instrumented_frame",
            capture_specification=specification,
            coordinator_capture_release_ns=capture_release_ns,
        )
        values = [fragment["capsule_fragment"] for fragment in fragments]
        capsule_ids = {value["capsule_id"] for value in values}
        if len(capsule_ids) != 1:
            raise RuntimeError("capsule_invalid: client capsule IDs differ")
        launches = []
        dependencies = []
        clients = []
        victim_ids = []
        aggressor_ids = []
        for value in values:
            launches.extend(value.get("launches", []))
            dependencies.extend(value.get("dependencies", []))
            clients.extend(value.get("clients", []))
            victim_ids.extend(value.get("victim_launch_ids", []))
            if value.get("aggressor_launch_id"):
                aggressor_ids.append(value["aggressor_launch_id"])
            for launch in value.get("launches", []):
                self.launch_clients[launch["launch_id"]] = launch["client_id"]
        if len(aggressor_ids) != 1 or not victim_ids:
            raise RuntimeError("capsule_invalid: capture roles are incomplete")
        first = values[0]
        builds = {json.dumps(
            value.get("agent_build", {}), sort_keys=True
        ) for value in values}
        if len(builds) != 1:
            raise RuntimeError("capsule_invalid: agent builds differ")
        victim_ids.sort(
            key=lambda value: int(value.rsplit("-", 1)[-1])
        )
        source_victims = specification["victim_kernels"]
        head_source_id = specification["victim_head_source_kernel_id"]
        head_index = next(
            index
            for index, value in enumerate(source_victims)
            if value["source_kernel_id"] == head_source_id
        )
        self.capsule = {
            "schema": "kernel_capsule_v3",
            "capsule_id": first["capsule_id"],
            "source_run_id": first["source_run_id"],
            "aggressor_launch_id": aggressor_ids[0],
            "victim_launch_ids": victim_ids,
            "launches": launches,
            "dependencies": dependencies,
            "clients": clients,
            "capture_complete": all(
                value.get("capture_complete", False) for value in values
            ),
            "applications_blocked": all(
                value.get("applications_blocked", False) for value in values
            ),
            "recursive_callbacks_disabled": all(
                value.get("recursive_callbacks_disabled", False)
                for value in values
            ),
            "no_later_model_work_submitted": all(
                value.get("no_later_model_work_submitted", False)
                for value in values
            ),
            "portable": False,
            "capabilities": {},
            "capture_completion_mode": "terminal_stream_synchronize",
            "terminal_streams": sorted(set().union(*(
                set(value.get("terminal_streams", [])) for value in values
            ))),
            "checkpoint_generation": min(
                int(value.get("checkpoint_generation", 0))
                for value in values
            ),
            "capture_matcher_evidence": {
                value["clients"][0]["client_id"]: value.get(
                    "capture_matcher_evidence", {}
                )
                for value in values
            },
            "agent_build": first.get("agent_build", {}),
            "victim_set": "gpu_execution_overlap_all_streams",
            "overlap_set_complete": all(
                value.get("overlap_set_complete", False)
                for value in values
            ),
            "overlap_evidence": first.get("overlap_evidence", {}),
            "quiescent_checkpoint": all(
                value.get("quiescent_checkpoint", False)
                for value in values
            ),
            "submission_gate_enforced": all(
                value.get("submission_gate_enforced", False)
                for value in values
            ),
            "event_operations": [
                operation
                for value in values
                for operation in value.get("event_operations", [])
            ],
            "submission_policy": "common_epoch_dependency_burst",
            "client_release_offsets_ns": specification[
                "client_release_offsets_ns"
            ],
            "victim_head_launch_id": f"victim-{head_index}",
        }
        return {"capsule": self.capsule}

    def restore(self, capsule_id: str) -> Dict[str, Any]:
        """Restore both client checkpoints concurrently."""
        records = self._parallel("restore_checkpoint", capsule_id=capsule_id)
        restored = sum(
            int(record.get("restored", False)) for record in records
        )
        return {"restored_client_count": restored}

    def replay(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Release all clients at one epoch for prompt captured-order issue."""
        if self.capsule is None:
            raise RuntimeError("no live capsule is captured")
        offsets = {
            str(client): int(offset)
            for client, offset in request.get(
                "client_release_offsets_ns",
                self.capsule["client_release_offsets_ns"],
            ).items()
        }
        if set(offsets) != set(self.agents):
            raise RuntimeError(
                "capsule_invalid: release offsets must name every client"
            )
        self._parallel(
            "prepare_replay",
            capsule_id=request["capsule_id"],
            variant=request["variant"],
            instrumentation=request["instrumentation"],
            submission_policy="common_epoch_dependency_burst",
            aggressor_priority=request.get("aggressor_priority"),
        )
        release_ns = time.monotonic_ns() + 5_000_000
        records = self._parallel(
            "execute_replay",
            capsule_id=request["capsule_id"],
            variant=request["variant"],
            instrumentation=request["instrumentation"],
            coordinator_release_ns=release_ns,
            submission_policy="common_epoch_dependency_burst",
            client_release_offsets_ns=offsets,
            aggressor_priority=request.get("aggressor_priority"),
        )
        merged: Dict[str, Any] = {
            "victim_order": self.capsule["victim_launch_ids"],
            "launch_order": [],
            "launch_fingerprints": {},
            "fingerprint_match": True,
            "foreign_work_count": 0,
            "non_capsule_cuda_work_count": 0,
            "frame_replay_count": 0,
            "model_forward_count": 0,
            "rosbag_work_count": 0,
            "cta_intervals": [],
            "cta_collection_status": [],
            "clock_calibrations": [],
            "cta_collection_valid": True,
            "cta_collection_errors": [],
            "metric_samples": [],
            "launch_activities": [],
            "source_gpu_offsets_used_for_submission": False,
            "priority_attribute_applied": False,
        }
        client_digests = {
            "checkpoint_state_sha256": {},
            "allocation_map_sha256": {},
            "output_sha256": {},
        }
        for record in records:
            client_id = str(record.get("client_id", "unknown"))
            for key in (
                "foreign_work_count",
                "non_capsule_cuda_work_count",
                "frame_replay_count",
                "model_forward_count",
                "rosbag_work_count",
            ):
                merged[key] += int(record.get(key, 0))
            merged["fingerprint_match"] &= bool(
                record.get("fingerprint_match", False)
            )
            merged["launch_order"].extend(record.get("launch_order", []))
            merged["launch_fingerprints"].update(
                record.get("launch_fingerprints", {})
            )
            merged["cta_intervals"].extend(record.get("cta_intervals", []))
            merged["cta_collection_status"].extend(
                record.get("cta_collection_status", [])
            )
            merged["clock_calibrations"].extend(
                record.get("clock_calibrations", [])
            )
            merged["cta_collection_valid"] &= bool(
                record.get("cta_collection_valid", True)
            )
            if record.get("cta_collection_error"):
                merged["cta_collection_errors"].append({
                    "client_id": client_id,
                    "error": record["cta_collection_error"],
                })
            merged["metric_samples"].extend(record.get("metric_samples", []))
            merged["launch_activities"].extend(
                record.get("launch_activities", [])
            )
            if record.get("priority_attribute_applied", False):
                merged["priority_attribute_applied"] = True
                merged["aggressor_priority"] = int(
                    record["aggressor_priority"]
                )
            for key in (
                "victim_delay_ns",
                "aggressor_duration_ns",
                "profiler_perturbation",
            ):
                if record.get(key) is not None:
                    merged[key] = record[key]
            for key, values in client_digests.items():
                if record.get(key) is not None:
                    values[client_id] = record[key]
        for key, values in client_digests.items():
            if key == "output_sha256":
                merged["client_output_sha256"] = dict(values)
            encoded = json.dumps(
                values, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            merged[key] = hashlib.sha256(encoded).hexdigest()
        issue_starts = {
            value["launch_id"]: int(value["driver_issue_start_ns"])
            for value in merged["launch_activities"]
        }
        merged["launch_order"].sort(key=lambda value: issue_starts[value])
        return merged

    def aggressor_priority_range(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query launch-priority support from the aggressor-owning client."""
        client_id = self.launch_clients["aggressor"]
        payload = {
            key: value for key, value in request.items()
            if key not in ("operation", "protocol")
        }
        payload = {**self._lease(), **payload}
        return self.agents[client_id].request(
            "aggressor_priority_range", **payload
        )

    def resource(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route one selected-resource request to the owning live client."""
        client_id = self.launch_clients[request["launch_id"]]
        payload = {
            key: value for key, value in request.items()
            if key not in ("operation", "protocol")
        }
        payload = {**self._lease(), **payload}
        return self.agents[client_id].request(
            "collect_resource_profile", **payload
        )

    def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one planner protocol request."""
        operation = request.get("operation")
        if operation == "capabilities":
            return self.capabilities()
        if operation == "capture_one_instrumented_frame":
            return self.capture(request["capture_specification"])
        if operation == "restore_checkpoints":
            return self.restore(request["capsule_id"])
        if operation == "replay":
            return self.replay(request)
        if operation == "aggressor_priority_range":
            return self.aggressor_priority_range(request)
        if operation == "collect_resource_profile":
            return self.resource(request)
        if operation == "heartbeat":
            records = self._parallel("lease_heartbeat")
            return {"acknowledged_client_count": len(records)}
        if operation == "shutdown":
            records = self._parallel("shutdown")
            self.shutdown_requested = True
            return {"shutdown_client_count": len(records)}
        raise ValueError(f"unknown coordinator operation: {operation}")


def serve(coordinator: KernelCapsuleCoordinator, socket_path: Path) -> None:
    """Serve planner requests until the process is terminated."""
    socket_path = Path(socket_path)
    if socket_path.exists():
        raise FileExistsError(socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        server.listen(4)
        try:
            while not coordinator.shutdown_requested:
                connection, _ = server.accept()
                with connection:
                    try:
                        request = receive_json_line(connection)
                        result = coordinator.dispatch(request)
                        response = {
                            "protocol": PROTOCOL,
                            "status": "ok",
                            "result": result,
                        }
                    except (
                        Exception
                    ) as exc:  # preserve protocol error evidence
                        response = {
                            "protocol": PROTOCOL,
                            "status": "error",
                            "error": str(exc),
                        }
                    connection.sendall(
                        (json.dumps(response, sort_keys=True) + "\n").encode(
                            "utf-8"
                        )
                    )
        finally:
            socket_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    """Start a host coordinator for two named injected-agent sockets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", required=True)
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        help="CLIENT_ID=/path/to/agent.sock; specify exactly twice",
    )
    parser.add_argument("--lease-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    agents = []
    for value in args.agent:
        client_id, path = value.split("=", 1)
        agents.append(AgentEndpoint(client_id, Path(path)))
    serve(
        KernelCapsuleCoordinator(agents, args.lease_seconds),
        Path(args.listen),
    )
    return 0
