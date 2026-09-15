"""Select one self-contained GPU-overlap MPS target from an Nsight trace."""

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class _InputIdentity:
    """Read-only source identity used by offline target selection."""

    value: Dict[str, Any]

    @property
    def ros_header_timestamp_ns(self) -> Optional[int]:
        return self.value.get("ros_header_timestamp_ns")

    @property
    def content_key(self) -> tuple:
        common = (
            self.value.get("topic"), self.value.get("message_type"),
            self.ros_header_timestamp_ns,
        )
        if self.value.get("schema") == "dataset_input_identity_v2":
            return common
        return common + (
            self.value.get("bag_timestamp_ns"),
            self.value.get("raw_cdr_sha256"),
            self.value.get("raw_cdr_size"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.value)


def _load_source_frame_index(path: Path) -> List[_InputIdentity]:
    """Read a profiler-owned source identity index without runtime imports."""
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if value.get("schema") != "source_frame_identity_v1":
                    raise ValueError("source frame schema mismatch")
                records.append(_InputIdentity(value))
            except (AttributeError, TypeError, ValueError,
                    json.JSONDecodeError) as exc:
                raise TraceSelectionError(
                    f"invalid source frame index line {line_number}"
                ) from exc
    ordinals = [record.value.get("ordinal") for record in records]
    if ordinals != list(range(len(records))):
        raise TraceSelectionError("source frame ordinals are not contiguous")
    return records


LAUNCH_APIS = frozenset(
    (
        "cudaLaunchKernel_v7000",
        "cudaLaunchKernelExC_v11060",
        "cuLaunchKernel",
        "cuLaunchKernelEx",
        "cuLaunchCooperativeKernel",
        "cudaLaunchCooperativeKernel",
    )
)
SYNCHRONIZATION_APIS = frozenset(
    (
        "cudaDeviceSynchronize",
        "cudaEventSynchronize_v3020",
        "cudaStreamSynchronize_v3020",
        "cuCtxSynchronize",
        "cuEventSynchronize",
        "cuStreamSynchronize",
    )
)
UNRESOLVED_DEPENDENCY_APIS = frozenset(
    (
        "cudaStreamWaitEvent",
        "cuStreamWaitEvent",
        "cudaGraphLaunch",
        "cuGraphLaunch",
        "cudaMemcpyAsync",
        "cuMemcpyAsync",
    )
)


def decode_global_id(global_id: int) -> Tuple[int, int]:
    """Decode an Nsight serialized global identifier into PID and TID."""
    return ((global_id >> 24) & 0xFFFFFF, global_id & 0xFFFFFF)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class KernelRecord:
    """One CUDA kernel plus host-launch and stream provenance."""

    kernel_id: int
    pid: int
    context_id: int
    stream_id: int
    stream_priority: int
    correlation_id: int
    name: str
    start_ns: int
    end_ns: int
    launch_start_ns: int
    launch_end_ns: int
    launch_api: str
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    registers_per_thread: int
    static_shared_memory: int
    dynamic_shared_memory: int
    graph_node_id: Optional[int]
    launch_type: Optional[int]
    mangled_name: Optional[str] = None

    @property
    def duration_ns(self) -> int:
        """Return measured GPU execution duration."""
        return self.end_ns - self.start_ns

    def compact_launch(self) -> Dict[str, Any]:
        """Return stable source fields used by a capture specification."""
        return {
            "source_kernel_id": self.kernel_id,
            "pid": self.pid,
            "context_id": self.context_id,
            "stream_id": self.stream_id,
            "stream_priority": self.stream_priority,
            "correlation_id": self.correlation_id,
            "name": self.name,
            "mangled_name": self.mangled_name,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "launch_start_ns": self.launch_start_ns,
            "launch_end_ns": self.launch_end_ns,
            "launch_api": self.launch_api,
            "grid": list(self.grid),
            "block": list(self.block),
            "registers_per_thread": self.registers_per_thread,
            "static_shared_memory": self.static_shared_memory,
            "dynamic_shared_memory": self.dynamic_shared_memory,
            "graph_node_id": self.graph_node_id,
            "launch_type": self.launch_type,
        }


@dataclass(frozen=True)
class TraceRecords:
    """Common CUDA and NVTX records reconstructed from one Nsight export."""

    kernels: List[KernelRecord]
    streams: Dict[Tuple[int, int, int], List[KernelRecord]]
    ready_times: Dict[int, int]
    unresolved_times: Dict[int, List[int]]
    ranges: List[Dict[str, Any]]


@dataclass(frozen=True)
class DelayEpisode:
    """One kernel and its reconstructed ready-to-start wait."""

    victim: KernelRecord
    ready_ns: int
    delay_ns: int
    inference: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class DependencyEdge:
    """One required ordering edge within the victim capsule."""

    source_kernel_id: int
    target_kernel_id: int
    kind: str


@dataclass(frozen=True)
class VictimOverlapSet:
    """Complete cross-stream victim set that executes with one aggressor."""

    kernels: Tuple[KernelRecord, ...]
    edges: Tuple[DependencyEdge, ...]
    head_kernel_id: int
    context_id: int

    @property
    def signature_payload(self) -> Dict[str, Any]:
        """Return a PID-independent ordered sequence and dependency graph."""
        stream_labels: Dict[Tuple[int, int], int] = {}
        sequence = []
        for kernel in self.kernels:
            key = (kernel.context_id, kernel.stream_id)
            if key not in stream_labels:
                stream_labels[key] = len(stream_labels)
            sequence.append(
                {
                    "name": kernel.name,
                    "stream": stream_labels[key],
                    "priority": kernel.stream_priority,
                    "grid": list(kernel.grid),
                    "block": list(kernel.block),
                    "dynamic_shared_memory": kernel.dynamic_shared_memory,
                }
            )
        positions = {
            kernel.kernel_id: index
            for index, kernel in enumerate(self.kernels)
        }
        edges = sorted(
            (
                positions[edge.source_kernel_id],
                positions[edge.target_kernel_id],
                edge.kind,
            )
            for edge in self.edges
        )
        return {"sequence": sequence, "edges": edges}

    @property
    def signature(self) -> str:
        """Return the dependency-graph signature hash."""
        return _digest(self.signature_payload)


class TraceSelectionError(RuntimeError):
    """Raised when the source trace cannot yield one safe capsule target."""


class TraceTargetSelector:
    """Rank source episodes by aggregate victim ready-to-start delay."""

    def __init__(
        self,
        sqlite_path: Path,
        run_directory: Optional[Path] = None,
        aggressor_min_duration_ms: float = 1.0,
    ):
        """Configure selection from one exported Nsight SQLite trace."""
        self.sqlite_path = Path(sqlite_path)
        self.run_directory = Path(run_directory or self.sqlite_path.parent)
        self.minimum_ns = int(aggressor_min_duration_ms * 1_000_000)
        self.models = self._model_identities()

    def _model_identities(self) -> Dict[int, str]:
        identities = {}
        for path in self.run_directory.glob("model_*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                identities[int(record["pid"])] = str(record["model_id"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return identities

    def _source_config_path(self) -> Path:
        """Return the original config whose hash is fixed by the run."""
        manifest = json.loads(
            (self.run_directory / "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected = manifest["config_sha256"]
        candidates = [manifest.get("config_source")]
        candidates.extend(
            argument.split(":=", 1)[1]
            for argument in manifest.get("command", [])
            if (
                isinstance(argument, str)
                and argument.startswith("config_file:=")
            )
        )
        candidates.append(str(self.run_directory / "config.yaml"))
        for value in candidates:
            if not value:
                continue
            candidate = Path(value).expanduser().resolve()
            try:
                raw = candidate.read_bytes()
            except OSError:
                continue
            if hashlib.sha256(raw).hexdigest() == expected:
                return candidate
        raise TraceSelectionError("source run config is unavailable")

    def _model_input_records(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Load input IDs written outside the source inference ranges."""
        records = {}
        for path in self.run_directory.glob("model_*_inputs.jsonl"):
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        key = (
                            str(record["model_id"]),
                            str(record["input_id"]),
                        )
                    except (KeyError, TypeError, json.JSONDecodeError) as exc:
                        raise TraceSelectionError(
                            f"invalid model input record {path}:"
                            f"{line_number}"
                        ) from exc
                    if key in records:
                        raise TraceSelectionError(
                            f"ambiguous model input identity {key}"
                        )
                    records[key] = record
        return records

    @staticmethod
    def _nvtx_ranges(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Load machine-readable inference and ownership ranges."""
        if "NVTX_EVENTS" not in TraceTargetSelector._tables(connection):
            return []
        ranges = []
        for start, end, global_tid, text in connection.execute(
            "SELECT start, end, globalTid, text FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND text LIKE 'closeloop:%'"
        ):
            if global_tid is None or text is None:
                continue
            try:
                tag = json.loads(str(text)[len("closeloop:"):])
            except (TypeError, json.JSONDecodeError):
                continue
            event = tag.get("event")
            if event not in ("inference", "method", "module"):
                continue
            pid, _ = decode_global_id(int(global_tid))
            ranges.append({
                "start_ns": int(start),
                "end_ns": int(end),
                "pid": pid,
                "tag": tag,
            })
        return ranges

    @staticmethod
    def _enclosing_range(
        kernel: KernelRecord,
        ranges: Iterable[Dict[str, Any]],
        event: str,
    ) -> Optional[Dict[str, Any]]:
        matches = [
            record for record in ranges
            if record["pid"] == kernel.pid
            and record["tag"].get("event") == event
            and record["start_ns"] <= kernel.launch_start_ns
            and kernel.launch_end_ns <= record["end_ns"]
        ]
        if not matches:
            return None
        matches.sort(key=lambda record: (
            record["end_ns"] - record["start_ns"], record["start_ns"]
        ))
        return matches[0]

    @staticmethod
    def _launch_matcher(kernel: KernelRecord) -> Dict[str, Any]:
        """Return source fields the live callback must match exactly."""
        matcher = {
            "launch_api": kernel.launch_api,
            "symbol": kernel.name,
            "grid": list(kernel.grid),
            "block": list(kernel.block),
            "dynamic_shared_memory": kernel.dynamic_shared_memory,
        }
        if kernel.mangled_name:
            matcher["mangled_symbol"] = kernel.mangled_name
        return matcher

    def _identity_aware_episode(
        self,
        aggressor: KernelRecord,
        closure: VictimOverlapSet,
        kernels: List[KernelRecord],
        ranges: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Map selected launches to one timestamp-identified dataset input."""
        index_path = self.run_directory / "source_frames.jsonl"
        input_records = self._model_input_records()
        if not input_records or not ranges:
            return None
        frames = (
            _load_source_frame_index(index_path) if index_path.is_file()
            else []
        )
        frame_by_stamp: DefaultDict[int, List[Any]] = defaultdict(list)
        for frame in frames:
            if frame.ros_header_timestamp_ns is not None:
                frame_by_stamp[frame.ros_header_timestamp_ns].append(frame)

        dataset_models = None
        metadata_path = None
        remappings = {}
        try:
            config = yaml.safe_load(
                self._source_config_path().read_text(encoding="utf-8")
            )
            if not isinstance(config, dict):
                raise ValueError("source run config is not a mapping")
            dataset_models = {
                str(model["id"]): model for model in config["models"]
            }
            metadata_path = config["replay"]["metadata_path"]
            remappings = config["replay"].get("remappings", {})
        except (KeyError, OSError, RuntimeError, ValueError):
            if not frames:
                return None

        selected = [aggressor] + list(closure.kernels)
        evidence = {}
        selected_frames = []
        inference_inputs: DefaultDict[str, set] = defaultdict(set)
        for kernel in selected:
            inference = self._enclosing_range(kernel, ranges, "inference")
            if inference is None:
                raise TraceSelectionError(
                    "selected launch has no enclosing inference NVTX range"
                )
            model_id = str(inference["tag"].get("model", ""))
            input_id = str(inference["tag"].get("input", ""))
            if not model_id or input_id.startswith("warmup-"):
                raise TraceSelectionError(
                    "selected launch has invalid inference identity"
                )
            input_record = input_records.get((model_id, input_id))
            if input_record is None:
                raise TraceSelectionError(
                    f"missing input identity for {model_id}/{input_id}"
                )
            stamp = input_record.get("ros_header_timestamp_ns")
            if dataset_models is not None:
                model = dataset_models.get(model_id)
                if model is None:
                    raise TraceSelectionError(
                        f"source config has no model {model_id}"
                    )
                destinations = [
                    source for source, destination in remappings.items()
                    if destination == model["input_topic"]
                ]
                if len(destinations) > 1:
                    raise TraceSelectionError(
                        f"source topic for model {model_id} is ambiguous"
                    )
                source_topic = (
                    destinations[0] if destinations
                    else model["input_topic"]
                )
                message_types = {
                    "compressed_image": (
                        "sensor_msgs/msg/CompressedImage"
                    ),
                    "pointcloud2": "sensor_msgs/msg/PointCloud2",
                }
                message_type = message_types.get(
                    model.get("input_message_type", "compressed_image")
                )
                if message_type is None:
                    raise TraceSelectionError(
                        f"model {model_id} has no dataset input encoding"
                    )
                selected_frames.append(_InputIdentity({
                    "schema": "dataset_input_identity_v2",
                    "ros_header_timestamp_ns": int(stamp),
                    "topic": source_topic,
                    "message_type": message_type,
                    "dataset_metadata_path": str(metadata_path),
                }))
            else:
                matches = frame_by_stamp.get(stamp, [])
                if len(matches) != 1:
                    raise TraceSelectionError(
                        f"source identity for ROS timestamp {stamp} is "
                        f"not unique"
                    )
                selected_frames.append(matches[0])
            inference_inputs[model_id].add(input_id)

            frame_kernels = sorted(
                (
                    item for item in kernels
                    if item.pid == kernel.pid
                    and inference["start_ns"] <= item.launch_start_ns
                    and item.launch_end_ns <= inference["end_ns"]
                ),
                key=lambda item: (
                    item.launch_start_ns, item.launch_end_ns, item.kernel_id
                ),
            )
            position = next(
                index for index, item in enumerate(frame_kernels)
                if item.kernel_id == kernel.kernel_id
            )
            matcher = self._launch_matcher(kernel)

            def framework_owner(item: KernelRecord) -> Optional[str]:
                owner_range = self._enclosing_range(item, ranges, "method")
                if owner_range is None:
                    owner_range = self._enclosing_range(
                        item, ranges, "module"
                    )
                if owner_range is None:
                    return None
                return (
                    owner_range["tag"].get("owner")
                    or owner_range["tag"].get("module")
                )

            owner = framework_owner(kernel)
            same_matcher = [
                item for item in frame_kernels
                if self._launch_matcher(item) == matcher
                and framework_owner(item) == owner
            ]
            occurrence = next(
                index for index, item in enumerate(same_matcher)
                if item.kernel_id == kernel.kernel_id
            )

            def anchor(
                item: Optional[KernelRecord],
            ) -> Optional[Dict[str, Any]]:
                if item is None:
                    return None
                return {
                    **self._launch_matcher(item),
                    "relative_sequence_offset": (
                        item.launch_start_ns - kernel.launch_start_ns
                    ),
                }

            evidence[kernel.kernel_id] = {
                "model_identity": model_id,
                "inference_input_id": input_id,
                "inference_ros_header_timestamp_ns": stamp,
                "framework_owner": owner,
                "frame_local_launch_sequence_index": position,
                "frame_local_launch_occurrence_ordinal": occurrence,
                "capture_matcher": {**matcher, "framework_owner": owner},
                "sequence_anchors": {
                    "previous": anchor(
                        frame_kernels[position - 1] if position else None
                    ),
                    "next": anchor(
                        frame_kernels[position + 1]
                        if position + 1 < len(frame_kernels) else None
                    ),
                },
            }

        content_keys = {frame.content_key for frame in selected_frames}
        if len(content_keys) != 1:
            raise TraceSelectionError(
                "selected cross-model launches do not share one source frame"
            )
        if any(len(values) != 1 for values in inference_inputs.values()):
            raise TraceSelectionError(
                "selected model launches span multiple inference inputs"
            )
        return {
            "source_frame": selected_frames[0].to_dict(),
            "inference_inputs": {
                model: next(iter(values))
                for model, values in sorted(inference_inputs.items())
            },
            "launch_evidence": evidence,
        }

    @staticmethod
    def _tables(connection: sqlite3.Connection) -> set:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    def _kernels(self, connection: sqlite3.Connection) -> List[KernelRecord]:
        tables = self._tables(connection)
        required = {
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "StringIds",
        }
        if not required.issubset(tables):
            raise TraceSelectionError(
                "source trace lacks CUDA kernel/runtime activity tables"
            )
        priorities: Dict[Tuple[int, int, int], int] = {}
        if "TARGET_INFO_CUDA_STREAM" in tables:
            for stream, process, context, priority in connection.execute(
                "SELECT streamId, processId, contextId, priority "
                "FROM TARGET_INFO_CUDA_STREAM"
            ):
                priorities[(int(process), int(context), int(stream))] = int(
                    priority
                )
        runtime: Dict[Tuple[int, int], Tuple[int, int, str]] = {}
        query = (
            "SELECT r.correlationId, r.globalTid, r.start, r.end, s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId "
            "WHERE r.correlationId IS NOT NULL"
        )
        for correlation, global_tid, start, end, name in connection.execute(
            query
        ):
            if global_tid is None:
                continue
            pid, _ = decode_global_id(int(global_tid))
            key = (pid, int(correlation))
            candidate = (int(start), int(end), str(name))
            previous = runtime.get(key)
            if previous is None or candidate[1] > previous[1]:
                runtime[key] = candidate
        kernel_query = (
            "SELECT k.rowid, k.globalPid, k.contextId, k.streamId, "
            "k.correlationId, n.value, m.value, k.start, k.end, k.gridX, "
            "k.gridY, "
            "k.gridZ, k.blockX, k.blockY, k.blockZ, "
            "k.registersPerThread, k.staticSharedMemory, "
            "k.dynamicSharedMemory, k.graphNodeId, k.launchType "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds n ON n.id=k.demangledName "
            "LEFT JOIN StringIds m ON m.id=k.mangledName "
            "ORDER BY k.start, k.end, k.rowid"
        )
        records = []
        for row in connection.execute(kernel_query):
            (
                rowid,
                global_pid,
                context,
                stream,
                correlation,
                name,
                mangled_name,
                start,
                end,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                registers,
                static_shared,
                dynamic_shared,
                graph_node,
                launch_type,
            ) = row
            pid, _ = decode_global_id(int(global_pid))
            launch = runtime.get((pid, int(correlation or -1)))
            if launch is None:
                launch = (int(start), int(start), "unknown")
            records.append(
                KernelRecord(
                    int(rowid),
                    pid,
                    int(context),
                    int(stream),
                    priorities.get((pid, int(context), int(stream)), 0),
                    int(correlation or -1),
                    str(name),
                    int(start),
                    int(end),
                    launch[0],
                    launch[1],
                    launch[2],
                    (int(grid_x), int(grid_y), int(grid_z)),
                    (int(block_x), int(block_y), int(block_z)),
                    int(registers),
                    int(static_shared),
                    int(dynamic_shared),
                    int(graph_node) if graph_node is not None else None,
                    int(launch_type) if launch_type is not None else None,
                    str(mangled_name) if mangled_name is not None else None,
                )
            )
        return records

    @staticmethod
    def _stream_sequences(
        kernels: Iterable[KernelRecord],
    ) -> Dict[Tuple[int, int, int], List[KernelRecord]]:
        streams: DefaultDict[Tuple[int, int, int], List[KernelRecord]] = (
            defaultdict(list)
        )
        for kernel in kernels:
            streams[(kernel.pid, kernel.context_id, kernel.stream_id)].append(
                kernel
            )
        return dict(streams)

    @staticmethod
    def _ready_times(
        streams: Dict[Tuple[int, int, int], List[KernelRecord]],
    ) -> Dict[int, int]:
        ready = {}
        for sequence in streams.values():
            predecessor_end = 0
            for kernel in sequence:
                ready[kernel.kernel_id] = max(
                    kernel.launch_end_ns, predecessor_end
                )
                predecessor_end = kernel.end_ns
        return ready

    @staticmethod
    def _boundary_times(
        connection: sqlite3.Connection,
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        sync: DefaultDict[int, List[int]] = defaultdict(list)
        unresolved: DefaultDict[int, List[int]] = defaultdict(list)
        query = (
            "SELECT r.globalTid, r.start, s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId"
        )
        for global_tid, start, name in connection.execute(query):
            plain = str(name).split("_v", 1)[0]
            if global_tid is not None and (
                str(name) in SYNCHRONIZATION_APIS
                or plain in SYNCHRONIZATION_APIS
            ):
                pid, _ = decode_global_id(int(global_tid))
                sync[pid].append(int(start))
            if global_tid is not None and plain in UNRESOLVED_DEPENDENCY_APIS:
                pid, _ = decode_global_id(int(global_tid))
                unresolved[pid].append(int(start))
        return (
            {pid: sorted(values) for pid, values in sync.items()},
            {pid: sorted(values) for pid, values in unresolved.items()},
        )

    @staticmethod
    def _strictly_overlaps(
        left_start: int, left_end: int, right_start: int, right_end: int
    ) -> bool:
        """Return whether two half-open intervals intersect positively."""
        return left_start < right_end and left_end > right_start

    @staticmethod
    def _aggressor_signature(kernel: KernelRecord) -> str:
        return _digest(
            {
                "name": kernel.name,
                "grid": kernel.grid,
                "block": kernel.block,
                "dynamic_shared_memory": kernel.dynamic_shared_memory,
                "launch_api": kernel.launch_api,
            }
        )

    def load_trace(self) -> TraceRecords:
        """Parse the shared records used by selection and trace analyses."""
        with sqlite3.connect(str(self.sqlite_path)) as connection:
            kernels = self._kernels(connection)
            streams = self._stream_sequences(kernels)
            ready_times = self._ready_times(streams)
            _, unresolved_times = self._boundary_times(connection)
            ranges = self._nvtx_ranges(connection)
        return TraceRecords(
            kernels, streams, ready_times, unresolved_times, ranges
        )

    def enumerate_delay_episodes(
        self, trace: Optional[TraceRecords] = None,
        delay_threshold_ns: int = 0, non_warmup_only: bool = False,
    ) -> List[DelayEpisode]:
        """Return every kernel whose ready-to-start delay is above a limit."""
        trace = trace or self.load_trace()
        episodes = []
        for victim in trace.kernels:
            inference = None
            if non_warmup_only:
                inference = self._enclosing_range(
                    victim, trace.ranges, "inference"
                )
                if (
                    inference is None
                    or str(inference["tag"].get("input", "")).startswith(
                        "warmup-"
                    )
                ):
                    continue
            ready_ns = trace.ready_times[victim.kernel_id]
            delay_ns = victim.start_ns - ready_ns
            if delay_ns > delay_threshold_ns:
                episodes.append(DelayEpisode(
                    victim, ready_ns, delay_ns, inference
                ))
        return episodes

    def _overlap_set(
        self,
        head: KernelRecord,
        aggressor: KernelRecord,
        kernels: List[KernelRecord],
        streams: Dict[Tuple[int, int, int], List[KernelRecord]],
        unresolved_times: Dict[int, List[int]],
        ranges: List[Dict[str, Any]],
    ) -> VictimOverlapSet:
        """
        Build the complete same-input/context positive-overlap set.

        The live agent checkpoints immediately before the first selected host
        launch.  Therefore every CUDA launch between that boundary and the
        final selected submission must itself be selected (or be a resolved
        internal event operation).  The SQLite selector cannot resolve event
        handles, so it conservatively rejects those episodes.
        """
        inference = self._enclosing_range(head, ranges, "inference")
        if inference is None:
            episode = [
                item for item in kernels if item.pid == head.pid
            ]
        else:
            episode = [
                item for item in kernels
                if item.pid == head.pid
                and inference["start_ns"] <= item.launch_start_ns
                and item.launch_end_ns <= inference["end_ns"]
            ]
        overlapping_contexts = {
            item.context_id for item in episode
            if self._strictly_overlaps(
                item.start_ns, item.end_ns,
                aggressor.start_ns, aggressor.end_ns,
            )
        }
        if overlapping_contexts != {head.context_id}:
            raise TraceSelectionError(
                "overlap episode spans multiple victim CUDA contexts"
            )
        episode = [
            item for item in episode if item.context_id == head.context_id
        ]

        selected = sorted(
            (
                item for item in episode
                if self._strictly_overlaps(
                    item.start_ns, item.end_ns,
                    aggressor.start_ns, aggressor.end_ns,
                )
            ),
            key=lambda item: (
                item.launch_start_ns, item.launch_end_ns, item.kernel_id
            ),
        )
        if not selected:
            raise TraceSelectionError("victim overlap set is empty")
        if any(item.graph_node_id is not None for item in selected):
            raise TraceSelectionError(
                "victim overlap set has unresolved graph dependencies"
            )
        if any(item.launch_api not in LAUNCH_APIS for item in selected):
            raise TraceSelectionError(
                "victim overlap set contains an unsupported launch"
            )

        selected_ids = {item.kernel_id for item in selected}
        checkpoint_boundary = min(item.launch_start_ns for item in selected)
        # Freeze begins in the last selected launch callback. Stream
        # synchronization completes the capture after the application has
        # already stopped submitting work.
        freeze_boundary = max(item.launch_end_ns for item in selected)
        post_checkpoint_foreign = [
            item for item in episode
            if item.kernel_id not in selected_ids
            and checkpoint_boundary <= item.launch_start_ns <= freeze_boundary
        ]
        if post_checkpoint_foreign:
            raise TraceSelectionError(
                "selected kernel depends on post-checkpoint non-overlap work"
            )
        if any(
            checkpoint_boundary <= value <= freeze_boundary
            for value in unresolved_times.get(head.pid, [])
        ):
            raise TraceSelectionError(
                "overlap episode has unresolved event or memcpy dependency"
            )

        edges = []
        for sequence in streams.values():
            selected_sequence = [
                item for item in sequence if item.kernel_id in selected_ids
            ]
            for source, target in zip(
                selected_sequence, selected_sequence[1:]
            ):
                edges.append(DependencyEdge(
                    source.kernel_id, target.kernel_id, "same_stream"
                ))
        edges.sort(key=lambda edge: (
            edge.source_kernel_id, edge.target_kernel_id, edge.kind
        ))
        return VictimOverlapSet(
            tuple(selected), tuple(edges), head.kernel_id, head.context_id
        )

    def select(self) -> Dict[str, Any]:
        """Return exactly one top aggregate-delay pair and one episode."""
        trace = self.load_trace()
        kernels = trace.kernels
        streams = trace.streams
        ready_times = trace.ready_times
        unresolved_times = trace.unresolved_times
        ranges = trace.ranges
        aggressors = [
            kernel
            for kernel in kernels
            if kernel.duration_ns >= self.minimum_ns
        ]
        groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        seen_occurrences = set()
        rejected = 0
        for episode in self.enumerate_delay_episodes(trace):
            victim = episode.victim
            ready = episode.ready_ns
            delay = episode.delay_ns
            for aggressor in aggressors:
                if aggressor.pid == victim.pid:
                    continue
                if not (
                    aggressor.start_ns < victim.start_ns
                    and aggressor.end_ns > ready
                ):
                    continue
                try:
                    closure = self._overlap_set(
                        victim,
                        aggressor,
                        kernels,
                        streams,
                        unresolved_times,
                        ranges,
                    )
                except TraceSelectionError:
                    rejected += 1
                    continue
                occurrence_key = (aggressor.kernel_id, closure.signature)
                if occurrence_key in seen_occurrences:
                    continue
                seen_occurrences.add(occurrence_key)
                delayed = []
                for item in closure.kernels:
                    item_ready = ready_times[item.kernel_id]
                    item_delay = item.start_ns - item_ready
                    if item_delay > 0 and (
                        aggressor.start_ns < item.start_ns
                        and aggressor.end_ns > item_ready
                    ):
                        delayed.append((item_delay, item_ready, item))
                if not delayed:
                    rejected += 1
                    continue
                delay, ready, victim = max(
                    delayed,
                    key=lambda item: (item[0], item[2].kernel_id),
                )
                aggressor_signature = self._aggressor_signature(aggressor)
                key = (aggressor_signature, closure.signature)
                group = groups.setdefault(
                    key,
                    {
                        "aggregate_excess_wait_ns": 0,
                        "occurrence_count": 0,
                        "representative": None,
                    },
                )
                group["aggregate_excess_wait_ns"] += delay
                group["occurrence_count"] += 1
                representative = group["representative"]
                if (
                    representative is None
                    or delay > representative["excess_wait_ns"]
                ):
                    group["representative"] = {
                        "excess_wait_ns": delay,
                        "ready_ns": ready,
                        "aggressor": aggressor,
                        "victim": victim,
                        "closure": closure,
                    }
        if not groups:
            if rejected:
                raise TraceSelectionError(
                    "no_self_contained_overlap_episode"
                )
            raise TraceSelectionError(
                "no cross-client >=1 ms aggressor overlaps victim ready delay"
            )
        selected_key, selected = max(
            groups.items(),
            key=lambda item: (
                item[1]["aggregate_excess_wait_ns"],
                item[1]["representative"]["excess_wait_ns"],
                item[0],
            ),
        )
        representative = selected["representative"]
        aggressor = representative["aggressor"]
        victim = representative["victim"]
        closure = representative["closure"]
        identity = self._identity_aware_episode(
            aggressor, closure, kernels, ranges
        )
        aggressor_record = aggressor.compact_launch()
        victim_records = [
            kernel.compact_launch() for kernel in closure.kernels
        ]
        for kernel, record in zip(closure.kernels, victim_records):
            overlap_start = max(kernel.start_ns, aggressor.start_ns)
            overlap_end = min(kernel.end_ns, aggressor.end_ns)
            record.update({
                "source_gpu_interval": {
                    "start_ns": kernel.start_ns,
                    "end_ns": kernel.end_ns,
                },
                "overlap_duration_ns": overlap_end - overlap_start,
                "stream_membership": {
                    "pid": kernel.pid,
                    "context_id": kernel.context_id,
                    "stream_id": kernel.stream_id,
                    "stream_priority": kernel.stream_priority,
                },
            })
        if identity is not None:
            aggressor_record.update(
                identity["launch_evidence"][aggressor.kernel_id]
            )
            for kernel, record in zip(closure.kernels, victim_records):
                record.update(identity["launch_evidence"][kernel.kernel_id])
        result = {
            "schema": "target_selection_v2",
            "schema_version": 2,
            "selection_policy": {
                "source": str(self.sqlite_path),
                "aggressor_min_duration_ms": self.minimum_ns / 1_000_000,
                "aggressor_mode": "individual",
                "selected_case_count": 1,
                "pair_ranking": "aggregate_excess_wait",
                "representative_episode": "maximum_excess_wait",
                "victim_set": "gpu_execution_overlap_all_streams",
                "overlap_interval_semantics": "half_open_positive",
                "capture_completion_mode": (
                    "terminal_stream_synchronize"
                ),
                "capture_completion_is_selection_boundary": False,
                "coverage_calculated": False,
            },
            "selected_pair": {
                "aggressor_signature": selected_key[0],
                "victim_overlap_signature": selected_key[1],
                "aggregate_excess_wait_ns": selected[
                    "aggregate_excess_wait_ns"
                ],
                "occurrence_count": selected["occurrence_count"],
            },
            "source_episode": {
                "excess_wait_ns": representative["excess_wait_ns"],
                "ready_ns": representative["ready_ns"],
                "aggressor_model": self.models.get(
                    aggressor.pid, str(aggressor.pid)
                ),
                "victim_model": self.models.get(victim.pid, str(victim.pid)),
                "aggressor": aggressor_record,
                "aggressor_source_interval": {
                    "start_ns": aggressor.start_ns,
                    "end_ns": aggressor.end_ns,
                },
                "victim_head_kernel_id": victim.kernel_id,
                "victim_kernels": victim_records,
                "victim_dependencies": [
                    asdict(edge) for edge in closure.edges
                ],
                "victim_set": "gpu_execution_overlap_all_streams",
                "overlap_set_complete": True,
                "victim_context_id": closure.context_id,
            },
            "diagnostics": {
                "eligible_aggressor_occurrences": len(aggressors),
                "ranked_pair_count_not_retained": len(groups),
                "rejected_non_self_contained_overlap_episodes": rejected,
                "sub_millisecond_kernels_are_diagnostic_only": True,
            },
        }
        if identity is not None:
            result["source_frame"] = identity["source_frame"]
            result["source_episode"]["inference_inputs"] = identity[
                "inference_inputs"
            ]
            result["selection_policy"]["source_frame_identity"] = (
                "unique_ros_timestamp_nuscenes"
            )
        return result

    def write(self, output_path: Path) -> Dict[str, Any]:
        """Select a target and atomically write ``target_selection.json``."""
        result = self.select()
        output_path = Path(output_path)
        temporary = output_path.with_name("." + output_path.name + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
        return result
