"""Exact live-kernel capsule records and replay-integrity verification."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CAPSULE_SCHEMA = "kernel_capsule_v2"
CAPSULE_SCHEMA_V3 = "kernel_capsule_v3"
LEGACY_CAPSULE_SCHEMA = "kernel_capsule_v1"
ANALYSIS_CAPSULE_SCHEMAS = frozenset(
    (LEGACY_CAPSULE_SCHEMA, CAPSULE_SCHEMA, CAPSULE_SCHEMA_V3)
)
SUPPORTED_LAUNCH_APIS = frozenset(
    (
        "cuLaunchKernel",
        "cuLaunchKernelEx",
        "cuLaunchCooperativeKernel",
    )
)


class CapsuleInvalid(RuntimeError):
    """Raised when exact live-kernel replay invariants do not hold."""


def read_capsule_manifest(path: Path) -> Dict[str, Any]:
    """Read v1, v2, or v3 JSON for offline analysis only."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") not in ANALYSIS_CAPSULE_SCHEMAS:
        raise CapsuleInvalid("capsule_invalid: unknown capsule schema")
    return value


@dataclass(frozen=True)
class ParameterValue:
    """One parameter copied from the driver's exact parameter layout."""

    index: int
    offset: int
    size: int
    raw_bytes_hex: str
    pointer_allocation_id: Optional[str] = None
    pointer_offset: Optional[int] = None

    def validate(self) -> None:
        """Reject malformed or ambiguous captured parameter bytes."""
        try:
            raw = bytes.fromhex(self.raw_bytes_hex)
        except ValueError as exc:
            raise CapsuleInvalid(
                "capsule_invalid: malformed parameter bytes"
            ) from exc
        if len(raw) != self.size or self.offset < 0 or self.size < 0:
            raise CapsuleInvalid("capsule_invalid: invalid parameter layout")
        pointer_fields = (
            self.pointer_allocation_id is not None,
            self.pointer_offset is not None,
        )
        if any(pointer_fields) and not all(pointer_fields):
            raise CapsuleInvalid(
                "capsule_invalid: incomplete pointer allocation identity"
            )


@dataclass(frozen=True)
class LaunchFingerprint:
    """Fields that make one nested CUDA Driver command replay-equivalent."""

    launch_id: str
    client_id: str
    context_handle: str
    function_handle: str
    code_object_sha256: str
    symbol: str
    symbol_sha256: str
    launch_api: str
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    cluster: Optional[Tuple[int, int, int]]
    dynamic_shared_memory: int
    stream_handle: str
    stream_priority: int
    function_attributes: Dict[str, int]
    parameters: Tuple[ParameterValue, ...]
    # v2 scheduling inputs. These remain parser-only provenance for old
    # artifacts and are omitted from v3 manifests.
    relative_ready_offset_ns: int = 0
    relative_launch_offset_ns: int = 0
    submission_policy: Optional[str] = None
    submission_sequence_index: Optional[int] = None
    client_release_offset_ns: int = 0
    source_host_launch_interval: Dict[str, int] = field(default_factory=dict)
    source_gpu_interval: Dict[str, int] = field(default_factory=dict)
    source_runtime_api: Optional[str] = None
    captured_driver_api: Optional[str] = None
    replay_api: Optional[str] = None
    equivalence_level: str = "gpu_driver_command"
    runtime_dispatch_preserved: bool = False
    driver_command_fingerprint: str = ""
    launch_attributes: Tuple[Dict[str, Any], ...] = ()
    parameter_mode: str = "kernel_params"
    packed_parameter_bytes_hex: Optional[str] = None
    source_runtime_correlation_id: Optional[int] = None
    nested_driver_correlation_id: Optional[int] = None
    cupti_correlation_id: Optional[int] = None
    framework_owner: Optional[str] = None
    frame_local_launch_occurrence_ordinal: Optional[int] = None
    frame_local_launch_sequence_index: Optional[int] = None
    runtime_stub_handle: Optional[str] = None

    def validate(self) -> None:
        """Validate fields that can be checked without issuing a launch."""
        driver_api = self.captured_driver_api or self.launch_api
        if driver_api not in SUPPORTED_LAUNCH_APIS:
            raise CapsuleInvalid(
                f"capsule_invalid: unsupported Driver API {driver_api}"
            )
        if self.launch_api != driver_api:
            raise CapsuleInvalid(
                "capsule_invalid: launch_api is not the captured Driver API"
            )
        if self.replay_api != driver_api:
            raise CapsuleInvalid(
                "capsule_invalid: replay API differs from Driver capture"
            )
        if self.nested_driver_correlation_id is None:
            raise CapsuleInvalid(
                "capsule_invalid: nested Driver correlation missing"
            )
        if self.source_runtime_api is not None and (
            self.source_runtime_correlation_id is None
            or not self.runtime_stub_handle
        ):
            raise CapsuleInvalid(
                "capsule_invalid: Runtime-to-Driver correlation incomplete"
            )
        if (
            self.equivalence_level != "gpu_driver_command"
            or self.runtime_dispatch_preserved
        ):
            raise CapsuleInvalid(
                "capsule_invalid: replay is not Driver-command equivalent"
            )
        if not self.function_handle or not self.context_handle:
            raise CapsuleInvalid("capsule_invalid: missing live CUDA handle")
        if len(self.code_object_sha256) != 64:
            raise CapsuleInvalid("capsule_invalid: missing code-object hash")
        try:
            int(self.code_object_sha256, 16)
        except ValueError as exc:
            raise CapsuleInvalid(
                "capsule_invalid: malformed code-object hash"
            ) from exc
        expected_symbol = hashlib.sha256(
            self.symbol.encode("utf-8")
        ).hexdigest()
        if self.symbol_sha256 != expected_symbol:
            raise CapsuleInvalid("capsule_invalid: symbol hash mismatch")
        if any(value <= 0 for value in self.grid + self.block):
            raise CapsuleInvalid(
                "capsule_invalid: non-positive launch geometry"
            )
        if self.cluster is not None and any(
            value <= 0 for value in self.cluster
        ):
            raise CapsuleInvalid("capsule_invalid: invalid cluster geometry")
        for value in (
            self.frame_local_launch_occurrence_ordinal,
            self.frame_local_launch_sequence_index,
        ):
            if value is not None and value < 0:
                raise CapsuleInvalid(
                    "capsule_invalid: negative frame-local launch identity"
                )
        expected_offset = 0
        for parameter in self.parameters:
            parameter.validate()
            if parameter.offset < expected_offset:
                raise CapsuleInvalid(
                    "capsule_invalid: overlapping parameter layout"
                )
            expected_offset = parameter.offset + parameter.size
        if self.parameter_mode not in {"kernel_params", "packed_extra"}:
            raise CapsuleInvalid(
                "capsule_invalid: unsupported Driver parameter mode"
            )
        if self.parameter_mode == "packed_extra":
            if self.packed_parameter_bytes_hex is None:
                raise CapsuleInvalid(
                    "capsule_invalid: packed Driver parameters missing"
                )
            try:
                bytes.fromhex(self.packed_parameter_bytes_hex)
            except ValueError as exc:
                raise CapsuleInvalid(
                    "capsule_invalid: malformed packed Driver parameters"
                ) from exc
        if self.driver_command_fingerprint != self.command_digest:
            raise CapsuleInvalid(
                "capsule_invalid: Driver-command fingerprint mismatch"
            )

    def validate_v3_schedule(self) -> None:
        """Validate v3 schedule evidence outside command equivalence."""
        if self.submission_policy != "common_epoch_dependency_burst":
            raise CapsuleInvalid(
                "capsule_invalid: v3 submission policy is not a "
                "dependency burst"
            )
        if self.submission_sequence_index is None or (
            self.submission_sequence_index < 0
        ):
            raise CapsuleInvalid(
                "capsule_invalid: v3 submission sequence is missing"
            )
        for name, interval in (
            ("source host launch", self.source_host_launch_interval),
            ("source GPU", self.source_gpu_interval),
        ):
            if set(interval) != {"start_ns", "end_ns"} or int(
                interval["end_ns"]
            ) < int(interval["start_ns"]):
                raise CapsuleInvalid(
                    f"capsule_invalid: {name} provenance is incomplete"
                )

    @property
    def command_digest(self) -> str:
        """Hash only fields passed to the CUDA Driver launch command."""
        payload = {
            "context_handle": self.context_handle,
            "function_handle": self.function_handle,
            "code_object_sha256": self.code_object_sha256,
            "captured_driver_api": self.captured_driver_api,
            "replay_api": self.replay_api,
            "grid": self.grid,
            "block": self.block,
            "cluster": self.cluster,
            "dynamic_shared_memory": self.dynamic_shared_memory,
            "stream_handle": self.stream_handle,
            "stream_priority": self.stream_priority,
            "function_attributes": self.function_attributes,
            "launch_attributes": self.launch_attributes,
            "parameter_mode": self.parameter_mode,
            "parameters": [asdict(value) for value in self.parameters],
            "packed_parameter_bytes_hex": self.packed_parameter_bytes_hex,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def digest(self) -> str:
        """Return a stable fingerprint excluding runtime observations."""
        payload = asdict(self)
        payload.pop("cupti_correlation_id", None)
        payload.pop("source_runtime_correlation_id", None)
        payload.pop("nested_driver_correlation_id", None)
        payload.pop("framework_owner", None)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CapsuleDependency:
    """One replay ordering or event dependency."""

    source_launch_id: str
    target_launch_id: str
    kind: str
    event_handle: Optional[str] = None


@dataclass(frozen=True)
class MpsClientState:
    """MPS state captured before the client is paused."""

    client_id: str
    server_pid: int
    client_pid: int
    context_handle: str
    active_thread_percentage: int
    client_priority: int
    connection_count: int
    checkpoint_id: str
    checkpoint_saved: bool
    checkpoint_generation: int = 1


@dataclass
class KernelCapsule:
    """A non-portable capsule whose handles remain owned by live clients."""

    capsule_id: str
    source_run_id: str
    aggressor_launch_id: str
    victim_launch_ids: List[str]
    launches: List[LaunchFingerprint]
    dependencies: List[CapsuleDependency]
    clients: List[MpsClientState]
    capture_complete: bool
    applications_blocked: bool
    recursive_callbacks_disabled: bool
    no_later_model_work_submitted: bool
    portable: bool = False
    schema: str = CAPSULE_SCHEMA
    capabilities: Dict[str, Any] = field(default_factory=dict)
    capture_completion_mode: str = "terminal_stream_synchronize"
    terminal_streams: List[str] = field(default_factory=list)
    checkpoint_generation: int = 1
    capture_matcher_evidence: Dict[str, Any] = field(default_factory=dict)
    agent_build: Dict[str, Any] = field(default_factory=dict)
    victim_set: str = "gpu_execution_overlap_all_streams"
    overlap_set_complete: bool = True
    overlap_evidence: Dict[str, Any] = field(default_factory=dict)
    quiescent_checkpoint: bool = True
    submission_gate_enforced: bool = True
    event_operations: List[Dict[str, Any]] = field(default_factory=list)
    submission_policy: Optional[str] = None
    client_release_offsets_ns: Dict[str, int] = field(default_factory=dict)
    victim_head_launch_id: Optional[str] = None

    def launch_map(self) -> Dict[str, LaunchFingerprint]:
        """Return launches by stable capsule identity."""
        return {launch.launch_id: launch for launch in self.launches}

    def validate(self) -> None:
        """Validate capture completeness, dependency closure, and identity."""
        if self.schema == LEGACY_CAPSULE_SCHEMA:
            raise CapsuleInvalid(
                "capsule_invalid: v1 capsule is not a v2 replay input"
            )
        if (
            self.schema not in {CAPSULE_SCHEMA, CAPSULE_SCHEMA_V3}
            or self.portable
        ):
            raise CapsuleInvalid("capsule_invalid: capsule is not live")
        if not (
            self.capture_complete
            and self.applications_blocked
            and self.recursive_callbacks_disabled
            and self.no_later_model_work_submitted
        ):
            raise CapsuleInvalid(
                "capsule_invalid: capture boundary incomplete"
            )
        if self.capture_completion_mode != "terminal_stream_synchronize":
            raise CapsuleInvalid(
                "capsule_invalid: unsafe capture completion mode"
            )
        if (
            self.victim_set != "gpu_execution_overlap_all_streams"
            or not self.overlap_set_complete
            or not self.overlap_evidence
        ):
            raise CapsuleInvalid(
                "capsule_invalid: overlap-set evidence is incomplete"
            )
        intervals = self.overlap_evidence.get(
            "victim_source_intervals", []
        )
        if len(intervals) != len(self.victim_launch_ids) or any(
            int(record.get("overlap_duration_ns", 0)) <= 0
            or not record.get("source_gpu_interval")
            or not record.get("stream_membership")
            for record in intervals
        ):
            raise CapsuleInvalid(
                "capsule_invalid: victim source overlap is not positive"
            )
        if not self.quiescent_checkpoint or not self.submission_gate_enforced:
            raise CapsuleInvalid(
                "capsule_invalid: capture checkpoint/gate is incomplete"
            )
        launches = self.launch_map()
        if len(launches) != len(self.launches):
            raise CapsuleInvalid("capsule_invalid: duplicate launch ID")
        if self.aggressor_launch_id not in launches:
            raise CapsuleInvalid("capsule_invalid: aggressor launch absent")
        if not self.victim_launch_ids or any(
            launch_id not in launches for launch_id in self.victim_launch_ids
        ):
            raise CapsuleInvalid("capsule_invalid: victim closure incomplete")
        if self.schema == CAPSULE_SCHEMA_V3 and (
            self.victim_head_launch_id not in self.victim_launch_ids
        ):
            raise CapsuleInvalid(
                "capsule_invalid: v3 victim head launch mapping is missing"
            )
        for launch in self.launches:
            launch.validate()
            if self.schema == CAPSULE_SCHEMA_V3:
                launch.validate_v3_schedule()
        if self.schema == CAPSULE_SCHEMA_V3:
            if self.submission_policy != "common_epoch_dependency_burst":
                raise CapsuleInvalid(
                    "capsule_invalid: v3 submission policy is missing"
                )
            client_ids = {client.client_id for client in self.clients}
            if set(self.client_release_offsets_ns) != client_ids:
                raise CapsuleInvalid(
                    "capsule_invalid: v3 client release offsets are incomplete"
                )
            sequences: Dict[str, set] = {}
            for launch in self.launches:
                if launch.client_release_offset_ns != (
                    self.client_release_offsets_ns[launch.client_id]
                ):
                    raise CapsuleInvalid(
                        "capsule_invalid: v3 launch release offset changed"
                    )
                sequences.setdefault(launch.client_id, set()).add(
                    launch.submission_sequence_index
                )
            if any(
                values != set(range(len(values)))
                for values in sequences.values()
            ):
                raise CapsuleInvalid(
                    "capsule_invalid: v3 client submission sequence has gaps"
                )
        for client in self.clients:
            if not client.checkpoint_saved:
                raise CapsuleInvalid("capsule_invalid: checkpoint missing")
            if client.checkpoint_generation <= 0:
                raise CapsuleInvalid(
                    "capsule_invalid: checkpoint generation missing"
                )
        if self.checkpoint_generation <= 0:
            raise CapsuleInvalid(
                "capsule_invalid: checkpoint generation missing"
            )
        terminal_launches = [launches[self.aggressor_launch_id]] + [
            launches[launch_id] for launch_id in self.victim_launch_ids
        ]
        if not {
            launch.stream_handle for launch in terminal_launches
        }.issubset(set(self.terminal_streams)):
            raise CapsuleInvalid(
                "capsule_invalid: terminal captured streams are incomplete"
            )
        if not self.capture_matcher_evidence:
            raise CapsuleInvalid(
                "capsule_invalid: capture matcher evidence missing"
            )
        if not self.agent_build.get("version"):
            raise CapsuleInvalid("capsule_invalid: agent build missing")
        contexts = {client.context_handle for client in self.clients}
        if any(
            launch.context_handle not in contexts for launch in self.launches
        ):
            raise CapsuleInvalid("capsule_invalid: launch context not live")
        valid_ids = set(launches)
        for dependency in self.dependencies:
            if (
                dependency.source_launch_id not in valid_ids
                or dependency.target_launch_id not in valid_ids
            ):
                raise CapsuleInvalid(
                    "capsule_invalid: dependency references absent launch"
                )
            if dependency.kind == "event" and not dependency.event_handle:
                raise CapsuleInvalid(
                    "capsule_invalid: event dependency lacks original handle"
                )
            if dependency.kind not in {"same_stream", "event"}:
                raise CapsuleInvalid(
                    "capsule_invalid: unsupported replay dependency"
                )
            if self.schema == CAPSULE_SCHEMA_V3:
                source = launches[dependency.source_launch_id]
                target = launches[dependency.target_launch_id]
                if (
                    source.client_id == target.client_id
                    and source.submission_sequence_index
                    >= target.submission_sequence_index
                ):
                    raise CapsuleInvalid(
                        "capsule_invalid: dependency contradicts "
                        "submission order"
                    )
        for operation in self.event_operations:
            if (
                operation.get("source_launch_id") not in valid_ids
                or operation.get("target_launch_id") not in valid_ids
                or not operation.get("event_handle")
            ):
                raise CapsuleInvalid(
                    "capsule_invalid: unresolved external event operation"
                )

    def verify_observed_launches(
        self, observed: Iterable[LaunchFingerprint]
    ) -> None:
        """Require byte-identical launches on the original clients/contexts."""
        expected = self.launch_map()
        actual = {launch.launch_id: launch for launch in observed}
        if set(actual) != set(expected):
            raise CapsuleInvalid(
                "capsule_invalid: launch count/order set mismatch"
            )
        for launch_id, expected_launch in expected.items():
            actual_launch = actual[launch_id]
            expected_launch.validate()
            actual_launch.validate()
            if expected_launch.digest != actual_launch.digest:
                raise CapsuleInvalid(
                    f"capsule_invalid: launch fingerprint mismatch {launch_id}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe capsule manifest."""
        value = asdict(self)
        value["schema"] = self.schema
        for launch in value["launches"]:
            if self.schema == CAPSULE_SCHEMA_V3:
                launch.pop("relative_ready_offset_ns", None)
                launch.pop("relative_launch_offset_ns", None)
            launch["fingerprint_sha256"] = self.launch_map()[
                launch["launch_id"]
            ].digest
        return value

    def write(self, path: Path) -> None:
        """Atomically write the live capsule manifest."""
        self.validate()
        path = Path(path)
        temporary = path.with_name("." + path.name + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _tuple3(value: Optional[Iterable[int]]) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    result = tuple(int(item) for item in value)
    if len(result) != 3:
        raise CapsuleInvalid("capsule_invalid: launch dimension is not 3D")
    return result


def capsule_from_dict(
    value: Dict[str, Any], required_schema: str = CAPSULE_SCHEMA
) -> KernelCapsule:
    """Parse and validate a capsule manifest from a connector or agent."""
    if value.get("schema") != required_schema:
        version = required_schema.rsplit("_", 1)[-1]
        raise CapsuleInvalid(
            "capsule_invalid: "
            f"{value.get('schema', 'unknown')} capsule is not "
            f"a {version} replay input (required {required_schema})"
        )
    launches = []
    for record in value.get("launches", []):
        record = dict(record)
        record.pop("fingerprint_sha256", None)
        record["grid"] = _tuple3(record["grid"])
        record["block"] = _tuple3(record["block"])
        record["cluster"] = _tuple3(record.get("cluster"))
        record["parameters"] = tuple(
            ParameterValue(**parameter)
            for parameter in record.get("parameters", [])
        )
        record["launch_attributes"] = tuple(
            dict(attribute)
            for attribute in record.get("launch_attributes", [])
        )
        launches.append(LaunchFingerprint(**record))
    capsule = KernelCapsule(
        capsule_id=str(value["capsule_id"]),
        source_run_id=str(value["source_run_id"]),
        aggressor_launch_id=str(value["aggressor_launch_id"]),
        victim_launch_ids=list(value["victim_launch_ids"]),
        launches=launches,
        dependencies=[
            CapsuleDependency(**record)
            for record in value.get("dependencies", [])
        ],
        clients=[
            MpsClientState(**record) for record in value.get("clients", [])
        ],
        capture_complete=bool(value.get("capture_complete")),
        applications_blocked=bool(value.get("applications_blocked")),
        recursive_callbacks_disabled=bool(
            value.get("recursive_callbacks_disabled")
        ),
        no_later_model_work_submitted=bool(
            value.get("no_later_model_work_submitted")
        ),
        portable=bool(value.get("portable", False)),
        schema=str(value.get("schema", "")),
        capabilities=dict(value.get("capabilities", {})),
        capture_completion_mode=str(
            value.get("capture_completion_mode", "")
        ),
        terminal_streams=list(value.get("terminal_streams", [])),
        checkpoint_generation=int(value.get("checkpoint_generation", 0)),
        capture_matcher_evidence=dict(
            value.get("capture_matcher_evidence", {})
        ),
        agent_build=dict(value.get("agent_build", {})),
        victim_set=str(value.get("victim_set", "")),
        overlap_set_complete=bool(value.get("overlap_set_complete")),
        overlap_evidence=dict(value.get("overlap_evidence", {})),
        quiescent_checkpoint=bool(value.get("quiescent_checkpoint")),
        submission_gate_enforced=bool(
            value.get("submission_gate_enforced")
        ),
        event_operations=[
            dict(record) for record in value.get("event_operations", [])
        ],
        submission_policy=value.get("submission_policy"),
        client_release_offsets_ns={
            str(key): int(offset)
            for key, offset in value.get(
                "client_release_offsets_ns", {}
            ).items()
        },
        victim_head_launch_id=value.get("victim_head_launch_id"),
    )
    capsule.validate()
    return capsule


def capsule_v3_from_dict(value: Dict[str, Any]) -> KernelCapsule:
    """Parse v3 for replay, rejecting analysis-only v1/v2 artifacts."""
    return capsule_from_dict(value, required_schema=CAPSULE_SCHEMA_V3)
