"""Fixed-phase planner for one exact live MPS kernel capsule."""

from abc import ABC, abstractmethod
import csv
import json
from pathlib import Path
import shutil
import statistics
from typing import Any, Dict, Iterable, List, Optional

from closeloop_profiler.gpu_forensics_store import GpuForensicsStore
from closeloop_profiler.kernel_capsule import (
    CAPSULE_SCHEMA_V3,
    KernelCapsule,
    capsule_from_dict,
    capsule_v3_from_dict,
)
from closeloop_analyzer.target_selection import TraceTargetSelector
from closeloop_profiler.source_frames import extract_dataset_input, input_identity_from_dict


class ReplayBackend(ABC):
    """Boundary implemented by the injected live-client replay agents."""

    @abstractmethod
    def capture_one_instrumented_frame(
        self, capture_specification: Dict[str, Any]
    ) -> KernelCapsule:
        """Capture once after model warmup and pause both live clients."""

    @abstractmethod
    def capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Return required and optional collector availability."""

    @abstractmethod
    def restore_checkpoints(self, capsule: KernelCapsule) -> int:
        """Restore every client checkpoint before one repetition."""

    @abstractmethod
    def replay(
        self, capsule: KernelCapsule, variant: str, instrumentation: str
    ) -> Dict[str, Any]:
        """Issue capsule launches with their captured relative ordering."""

    def aggressor_priority_range(
        self, capsule: KernelCapsule
    ) -> Dict[str, Any]:
        """Return supported per-launch priorities for the aggressor context."""
        raise NotImplementedError("launch-priority counterfactual unavailable")

    def replay_with_aggressor_priority(
        self, capsule: KernelCapsule, priority: int
    ) -> Dict[str, Any]:
        """Replay the pair with a declared aggressor-priority change."""
        raise NotImplementedError("launch-priority counterfactual unavailable")

    @abstractmethod
    def collect_resource_profile(
        self, capsule: KernelCapsule, launch_id: str
    ) -> Dict[str, Any]:
        """Collect NCU/CUDA launch facts, excluding its replay timing."""

    def close(self) -> None:
        """Release disposable live clients; manual backends may do nothing."""


class ReplayIntegrityError(RuntimeError):
    """Raised when replay evidence is incomplete or internally inconsistent."""


def capture_specification(target: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the one-frame capture request from exactly one target."""
    if target.get("schema") != "target_selection_v2":
        raise ValueError("v2 replay requires target_selection_v2")
    pair = target.get("selected_pair", {})
    episode = target.get("source_episode", {})
    source_frame = target.get("source_frame")
    if not pair or not episode or not source_frame:
        raise ValueError("target selection does not contain exactly one pair")
    return {
        "schema": "kernel_capsule_capture_spec_v2",
        "selected_case_count": 1,
        "after_existing_model_warmup": True,
        "additional_capsule_warmup": False,
        "capture_operation_not_replay": True,
        "source_pair": pair,
        "source_frame": source_frame,
        "inference_inputs": episode.get("inference_inputs", {}),
        "aggressor": episode["aggressor"],
        "ready_ns": episode["ready_ns"],
        "victim_head_kernel_id": episode["victim_head_kernel_id"],
        "victim_kernels": episode["victim_kernels"],
        "victim_dependencies": episode["victim_dependencies"],
        "victim_set": episode["victim_set"],
        "overlap_set_complete": episode["overlap_set_complete"],
        "overlap_evidence": {
            "aggressor_source_interval": episode[
                "aggressor_source_interval"
            ],
            "victim_source_intervals": [
                {
                    "source_kernel_id": launch["source_kernel_id"],
                    "source_gpu_interval": launch["source_gpu_interval"],
                    "overlap_duration_ns": launch["overlap_duration_ns"],
                    "stream_membership": launch["stream_membership"],
                }
                for launch in episode["victim_kernels"]
            ],
            "complete": episode["overlap_set_complete"],
        },
        "checkpoint": {
            "mode": "quiescent_before_first_selected_launch",
            "synchronize_previously_known_streams": True,
            "save_both_clients": True,
        },
        "submission_gate": {
            "after_checkpoint_allow_selected_launches_only": True,
            "allow_resolved_internal_event_operations": True,
            "reject_memcpy_or_external_dependencies": True,
        },
        "completion_action": {
            "mode": "terminal_stream_synchronize",
            "enqueue_completion_event": False,
            "synchronize_terminal_captured_streams": True,
            "block_application_thread": True,
            "keep_context_modules_allocations_mps_alive": True,
            "disable_recursive_callbacks": True,
        },
        "capture": [
            "function_handle",
            "code_object_hash",
            "symbol_hash",
            "launch_api_variant",
            "grid",
            "block",
            "cluster",
            "dynamic_shared_memory",
            "stream",
            "stream_priority",
            "function_attributes",
            "parameter_layout_and_bytes",
            "pointer_allocation_identity_and_offset",
            "event_dependencies",
            "mps_identity_and_configuration",
            "relative_offsets",
            "cupti_correlations",
            "runtime_to_nested_driver_correlation",
            "driver_command_fingerprint",
            "framework_ownership",
            "frame_local_occurrence_and_sequence_anchors",
        ],
    }


def capture_specification_v3(
    target: Dict[str, Any],
    client_release_offsets_ns: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Generate a dependency-burst capture request from a v2 target."""
    specification = capture_specification(target)
    episode = target["source_episode"]
    launches = [episode["aggressor"]] + list(episode["victim_kernels"])
    clients = sorted({str(launch["model_identity"]) for launch in launches})
    supplied = {
        str(client): int(offset)
        for client, offset in (client_release_offsets_ns or {}).items()
    }
    unknown = set(supplied) - set(clients)
    if unknown:
        raise ValueError(
            "client release offsets name non-participating clients: "
            + ", ".join(sorted(unknown))
        )
    offsets = {client: supplied.get(client, 0) for client in clients}
    specification.update(
        {
            "schema": "kernel_capsule_capture_spec_v3",
            "submission_policy": "common_epoch_dependency_burst",
            "client_release_offsets_ns": offsets,
            "source_timing_is_provenance_only": True,
            "source_gpu_start_offsets_control_submission": False,
            "victim_head_source_kernel_id": episode[
                "victim_head_kernel_id"
            ],
        }
    )
    specification["capture"] = [
        value
        for value in specification["capture"]
        if value != "relative_offsets"
    ] + [
        "submission_sequence_index",
        "client_release_offset_ns",
        "source_host_launch_interval",
        "source_gpu_interval",
    ]
    return specification


class KernelReplayPlannerV2:
    """Capture once and execute fixed checkpoint-restored kernel phases."""

    planner_id = "KernelReplayPlannerV2"

    def __init__(
        self,
        source_run_directory: Path,
        output_directory: Path,
        backend: ReplayBackend,
        profiler_options: Optional[Dict[str, Any]] = None,
        analyzer_options: Optional[Dict[str, Any]] = None,
    ):
        """Configure one source trace, output directory, and live backend."""
        self.source_run_directory = Path(source_run_directory)
        self.output_directory = Path(output_directory)
        self.backend = backend
        self.profiler_options = dict(profiler_options or {})
        self.analyzer_options = dict(analyzer_options or {})
        self.timing_repetitions = int(
            self.profiler_options.get("timing_repetitions", 3)
        )
        self.deep_repetitions = int(
            self.profiler_options.get("deep_repetitions", 3)
        )
        if self.timing_repetitions != 3 or self.deep_repetitions != 3:
            raise ValueError(
                "kernel_capsule_v2 requires exactly three repetitions"
            )
        self.records: List[Dict[str, Any]] = []

    def make_capture_specification(
        self, target: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the schema-specific native capture request."""
        return capture_specification(target)

    def parse_captured_capsule(self, value: Dict[str, Any]) -> KernelCapsule:
        """Parse a backend dictionary as this planner's capsule version."""
        return capsule_from_dict(value)

    def validate_captured_capsule(self, capsule: KernelCapsule) -> None:
        """Validate this planner's captured capsule before persistence."""
        capsule.validate()

    def select_target(self) -> Dict[str, Any]:
        """Analyze the original trace and persist only the selected pair."""
        selector = TraceTargetSelector(
            self.source_run_directory / "profile.sqlite",
            self.source_run_directory,
            float(self.analyzer_options.get("aggressor_min_duration_ms", 1.0)),
        )
        return selector.write(self.output_directory / "target_selection.json")

    def prepare_capture_input(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve and verify the selected timestamp's dataset payload."""
        identity = input_identity_from_dict(target["source_frame"])
        return extract_dataset_input(
            identity,
            self.output_directory / "capture_input.bin",
            self.output_directory / "capture_input.json",
        )

    @staticmethod
    def _required_collectors(options: Dict[str, Any]) -> set:
        return set(
            options.get(
                "required_collectors",
                (
                    "cupti_activity",
                    "cuda_launch_capture",
                    "cupti_checkpoint",
                ),
            )
        )

    def _record_capabilities(
        self, store: GpuForensicsStore
    ) -> Dict[str, bool]:
        capabilities = self.backend.capabilities()
        required = self._required_collectors(self.profiler_options)
        available = {}
        for collector in sorted(required | set(capabilities)):
            record = dict(capabilities.get(collector, {}))
            present = bool(record.get("available", False))
            store.record_capability(
                collector,
                collector in required,
                present,
                record.get("reason"),
                record.get("version"),
            )
            available[collector] = present
            if collector in required and not present:
                raise ReplayIntegrityError(
                    f"required collector unavailable: {collector}"
                )
        return available

    def _restore_and_replay(
        self,
        store: GpuForensicsStore,
        capsule: KernelCapsule,
        phase: str,
        variant: str,
        instrumentation: str,
        repetition: int,
    ) -> Dict[str, Any]:
        restores = self.backend.restore_checkpoints(capsule)
        if restores != len(capsule.clients):
            raise ReplayIntegrityError(
                "capsule_invalid: every client checkpoint must be restored"
            )
        record = dict(
            self.backend.replay(
                capsule, variant=variant, instrumentation=instrumentation
            )
        )
        record.update(
            {
                "phase": phase,
                "variant": variant,
                "instrumentation": instrumentation,
                "repetition": repetition,
                "checkpoint_restore_count": restores,
            }
        )
        if record.get("frame_replay_count", 0) != 0:
            raise ReplayIntegrityError("frame replay executed after capture")
        if record.get("model_forward_count", 0) != 0:
            raise ReplayIntegrityError("model forward executed after capture")
        if record.get("rosbag_work_count", 0) != 0:
            raise ReplayIntegrityError("rosbag work executed after capture")
        if record.get("non_capsule_cuda_work_count", 0) != 0:
            raise ReplayIntegrityError(
                "capsule_invalid: non-capsule CUDA work"
            )
        selected_ids = list(capsule.victim_launch_ids)
        if variant != "victim_only":
            selected_ids.append(capsule.aggressor_launch_id)
        launch_positions = {
            launch.launch_id: index
            for index, launch in enumerate(capsule.launches)
        }
        expected_order = sorted(
            selected_ids,
            key=lambda launch_id: (
                capsule.launch_map()[launch_id].relative_launch_offset_ns,
                launch_positions[launch_id],
            ),
        )
        if record.get("victim_order") != capsule.victim_launch_ids:
            record["fingerprint_match"] = False
        if record.get("launch_order") != expected_order:
            record["fingerprint_match"] = False
        expected_fingerprints = {
            launch_id: capsule.launch_map()[launch_id].digest
            for launch_id in expected_order
        }
        if record.get("launch_fingerprints") != expected_fingerprints:
            record["fingerprint_match"] = False
        record["iteration_id"] = store.record_iteration(
            phase, variant, repetition, record
        )
        self.records.append(record)
        return record

    @staticmethod
    def _median(values: Iterable[int]) -> float:
        values = list(values)
        if not values:
            raise ReplayIntegrityError(
                "replay integrity gate lacks timing repetitions"
            )
        return float(statistics.median(values))

    def _integrity_gate(
        self, _target: Dict[str, Any], capsule: KernelCapsule
    ) -> Dict[str, Any]:
        pairs = [
            record
            for record in self.records
            if record["phase"] == "timing" and record["variant"] == "pair"
        ]
        victims = [
            record
            for record in self.records
            if record["phase"] == "timing"
            and record["variant"] == "victim_only"
        ]
        if len(pairs) != 3 or len(victims) != 3:
            raise ReplayIntegrityError(
                "replay integrity gate requires three matched runs"
            )
        all_records = pairs + victims
        if any(
            int(record.get("foreign_work_count", 0)) for record in all_records
        ):
            raise ReplayIntegrityError("foreign GPU work observed")
        if any(
            not record.get("fingerprint_match", False)
            for record in all_records
        ):
            raise ReplayIntegrityError(
                "capsule_invalid: launch fingerprint mismatch"
            )
        if any(
            record.get("victim_order") != capsule.victim_launch_ids
            for record in all_records
        ):
            raise ReplayIntegrityError(
                "capsule_invalid: victim order/count mismatch"
            )
        checkpoint_states = {
            record.get("checkpoint_state_sha256") for record in all_records
        }
        allocation_states = {
            record.get("allocation_map_sha256") for record in all_records
        }
        if None in checkpoint_states or len(checkpoint_states) != 1:
            raise ReplayIntegrityError(
                "capsule_invalid: checkpoint restoration is nondeterministic"
            )
        if None in allocation_states or len(allocation_states) != 1:
            raise ReplayIntegrityError(
                "capsule_invalid: allocation identity changed"
            )
        for variant_records in (pairs, victims):
            outputs = {
                record.get("output_sha256") for record in variant_records
            }
            if None in outputs or len(outputs) != 1:
                raise ReplayIntegrityError(
                    "capsule_invalid: checkpoint output is nondeterministic"
                )
        pair_by_rep = {record["repetition"]: record for record in pairs}
        victim_by_rep = {record["repetition"]: record for record in victims}
        excess = [
            int(pair_by_rep[index]["victim_delay_ns"])
            - int(victim_by_rep[index]["victim_delay_ns"])
            for index in range(3)
        ]
        aggressor_durations = [
            int(record["aggressor_duration_ns"]) for record in pairs
        ]
        return {
            "status": "passed",
            "observed_pair_minus_victim_only_admission_ns": excess,
            "observed_aggressor_durations_ns": aggressor_durations,
            "timing_repetitions": 3,
            "deterministic_checkpoint_state": True,
            "deterministic_outputs_and_allocations": True,
        }

    def _write_timing_csv(self) -> None:
        fields = (
            "phase",
            "variant",
            "repetition",
            "victim_delay_ns",
            "aggressor_duration_ns",
            "checkpoint_restore_count",
            "fingerprint_match",
            "foreign_work_count",
        )
        path = self.output_directory / "capsule_timing.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                if record["phase"] == "timing":
                    writer.writerow(
                        {name: record.get(name) for name in fields}
                    )

    def _write_cta_csv(self) -> None:
        fields = (
            "phase",
            "repetition",
            "kernel_instance_id",
            "launch_id",
            "client_id",
            "sm_id",
            "cta_id",
            "entry_ns",
            "exit_ns",
            "duration_ns",
            "clock_error_ns",
            "observation_status",
            "entry_observed",
            "exit_observed",
            "expected_count",
            "entered_count",
            "exited_count",
            "complete_count",
            "missing_count",
            "coverage_fraction",
            "quality",
        )
        path = self.output_directory / "capsule_cta_placement.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                statuses = {
                    value["launch_id"]: value
                    for value in record.get("cta_collection_status", [])
                }
                for cta in record.get("cta_intervals", []):
                    status = statuses.get(cta["launch_id"], {})
                    entry = cta.get("entry_ns")
                    exit_ = cta.get("exit_ns")
                    writer.writerow(
                        {
                            "phase": record["phase"],
                            "repetition": record["repetition"],
                            "kernel_instance_id": (
                                f"{record.get('iteration_id')}:"
                                f"{cta['launch_id']}"
                            ),
                            "duration_ns": (
                                None if entry is None or exit_ is None
                                else int(exit_) - int(entry)
                            ),
                            **{
                                name: cta.get(name)
                                for name in (
                                    "launch_id", "client_id", "sm_id",
                                    "cta_id", "entry_ns", "exit_ns",
                                    "clock_error_ns", "observation_status",
                                    "entry_observed", "exit_observed",
                                )
                            },
                            **{
                                name: status.get(name)
                                for name in (
                                    "expected_count", "entered_count",
                                    "exited_count", "complete_count",
                                    "missing_count", "coverage_fraction",
                                    "quality",
                                )
                            },
                        }
                    )

    def _write_kernel_timing_csv(self) -> None:
        """v2 has no normalized per-launch replay activity artifact."""

    def _run_timing_campaign(
        self,
        store: GpuForensicsStore,
        target: Dict[str, Any],
        capsule: KernelCapsule,
    ) -> Dict[str, Any]:
        """Run the historical three-pair v2 timing campaign."""
        for repetition in range(3):
            variants = (
                ("pair", "victim_only")
                if repetition % 2 == 0
                else ("victim_only", "pair")
            )
            for variant in variants:
                self._restore_and_replay(
                    store,
                    capsule,
                    "timing",
                    variant,
                    "none",
                    repetition,
                )
        return self._integrity_gate(target, capsule)

    def _run_optional_collector(
        self,
        store: GpuForensicsStore,
        capsule: KernelCapsule,
        collector: str,
        phase: str,
        instrumentation: str,
    ) -> bool:
        """Run the historical three-repetition v2 optional campaign."""
        for repetition in range(3):
            self._restore_and_replay(
                store,
                capsule,
                phase,
                "pair",
                instrumentation,
                repetition,
            )
        return True

    def _run_priority_counterfactual(
        self, store: GpuForensicsStore, capsule: KernelCapsule
    ) -> Optional[Dict[str, Any]]:
        """Keep historical v2 campaigns free of v3 interventions."""
        return None

    def _collect_resources(
        self,
        store: GpuForensicsStore,
        capsule: KernelCapsule,
        available: Dict[str, bool],
    ) -> None:
        """Keep the historical NCU-gated v2 resource campaign."""
        if not available.get("ncu_mps", False):
            return
        unique = []
        seen = set()
        for launch_id in [capsule.aggressor_launch_id] + (
            capsule.victim_launch_ids
        ):
            launch = capsule.launch_map()[launch_id]
            key = (
                launch.code_object_sha256,
                launch.symbol,
                launch.grid,
                launch.block,
                launch.dynamic_shared_memory,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(launch_id)
        for launch_id in unique:
            restores = self.backend.restore_checkpoints(capsule)
            if restores != len(capsule.clients):
                raise ReplayIntegrityError(
                    "capsule_invalid: checkpoint missing before resource "
                    "collection"
                )
            record = self.backend.collect_resource_profile(
                capsule, launch_id
            )
            record["timing_excluded_from_conclusions"] = True
            store.record_resource(launch_id, record)

    def run(self) -> Dict[str, Any]:
        """Execute the immutable capture and replay campaign."""
        self.output_directory.mkdir(parents=True, exist_ok=False)
        try:
            target = self.select_target()
            input_manifest = self.prepare_capture_input(target)
            specification = self.make_capture_specification(target)
            specification["source_run_id"] = self.source_run_directory.name
            specification["capture_input"] = input_manifest
            (self.output_directory / "capture_specification.json").write_text(
                json.dumps(specification, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            capsule = self.backend.capture_one_instrumented_frame(
                specification
            )
            if isinstance(capsule, dict):
                capsule = self.parse_captured_capsule(capsule)
            self.validate_captured_capsule(capsule)
            capsule.write(self.output_directory / "kernel_capsule.json")
            database = self.output_directory / "capsule_replay.sqlite"
            gate = None
            analysis = None
            priority_counterfactual = None
            with GpuForensicsStore(database) as store:
                store.record_target(capsule.capsule_id, target)
                store.record_capsule(capsule)
                available = self._record_capabilities(store)
                gate = self._run_timing_campaign(store, target, capsule)
                if gate is not None:
                    store.set_metadata("replay_integrity_gate", gate)
                    if available.get("nvbit_cta", False):
                        available["nvbit_cta"] = (
                            self._run_optional_collector(
                                store,
                                capsule,
                                "nvbit_cta",
                                "cta",
                                "nvbit_cta",
                            )
                        )
                    if available.get("cupti_pm", False):
                        available["cupti_pm"] = self._run_optional_collector(
                            store,
                            capsule,
                            "cupti_pm",
                            "device_metric",
                            "cupti_pm",
                        )
                    self._collect_resources(
                        store, capsule, available
                    )
                    stage_evidence = {
                        "pending_grid_count": (
                            1 + len(capsule.victim_launch_ids)
                        ),
                        "hidden_mps_task_count": 0,
                        "mps_active_thread_percentage": {
                            client.client_id: client.active_thread_percentage
                            for client in capsule.clients
                        },
                        "mps_client_priority": {
                            client.client_id: client.client_priority
                            for client in capsule.clients
                        },
                        **dict(
                            capsule.capabilities.get("stage_evidence", {})
                        ),
                    }
                    store.set_metadata("stage_evidence", stage_evidence)
                    priority_counterfactual = (
                        self._run_priority_counterfactual(store, capsule)
                    )
                    if priority_counterfactual is not None:
                        store.set_metadata(
                            "priority_counterfactual",
                            priority_counterfactual,
                        )
            self._write_timing_csv()
            self._write_kernel_timing_csv()
            self._write_cta_csv()
            if gate is not None:
                from closeloop_analyzer.mps_analyzer import (
                    MpsContentionAnalyzer,
                )

                analysis = MpsContentionAnalyzer(
                    self.output_directory, self.analyzer_options
                ).analyze()
            shutil.copy2(
                database, self.output_directory / "gpu_forensics.sqlite"
            )
            return {
                "planner": self.planner_id,
                "target": target,
                "capsule": capsule.to_dict(),
                "replay_integrity_gate": gate,
                "priority_counterfactual": priority_counterfactual,
                "analysis": analysis,
            }
        finally:
            self.backend.close()


# Import compatibility for callers that only name the historical planner.
# The alias executes the v2 backend and never accepts v1 artifacts.
class KernelReplayPlannerV3(KernelReplayPlannerV2):
    """Replay all clients as a common-epoch dependency submission burst."""

    planner_id = "KernelReplayPlannerV3"
    submission_policy = "common_epoch_dependency_burst"

    def __init__(
        self,
        source_run_directory: Path,
        output_directory: Path,
        backend: ReplayBackend,
        profiler_options: Optional[Dict[str, Any]] = None,
        analyzer_options: Optional[Dict[str, Any]] = None,
    ):
        """Configure the single-sample v3 sequence-matched campaign."""
        self.source_run_directory = Path(source_run_directory)
        self.output_directory = Path(output_directory)
        self.backend = backend
        self.profiler_options = dict(profiler_options or {})
        self.analyzer_options = dict(analyzer_options or {})
        self.timing_repetitions = int(
            self.profiler_options.get("timing_repetitions", 1)
        )
        self.deep_repetitions = int(
            self.profiler_options.get("deep_repetitions", 1)
        )
        self.sequence_match_attempts = int(
            self.profiler_options.get("sequence_match_attempts", 3)
        )
        self.priority_counterfactual_enabled = bool(
            self.profiler_options.get("priority_counterfactual_enabled", False)
        )
        self.priority_counterfactual_trials = int(
            self.profiler_options.get("priority_counterfactual_trials", 3)
        )
        self.priority_min_effect_ns = int(
            self.profiler_options.get("priority_min_effect_ns", 1)
        )
        if self.timing_repetitions != 1 or self.deep_repetitions != 1:
            raise ValueError(
                "kernel_capsule_v3 requires exactly one accepted sample"
            )
        if self.sequence_match_attempts != 3:
            raise ValueError(
                "kernel_capsule_v3 requires sequence_match_attempts=3"
            )
        if self.priority_counterfactual_trials != 3:
            raise ValueError(
                "kernel_capsule_v3 requires priority_counterfactual_trials=3"
            )
        if self.priority_min_effect_ns < 0:
            raise ValueError("priority_min_effect_ns cannot be negative")
        self.records: List[Dict[str, Any]] = []
        self.accepted_pair_integrity: Dict[str, Any] = {}

    def make_capture_specification(
        self, target: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build v3 capture input with signed per-client release offsets."""
        return capture_specification_v3(
            target,
            self.profiler_options.get("client_release_offsets_ns", {}),
        )

    def parse_captured_capsule(self, value: Dict[str, Any]) -> KernelCapsule:
        """Reject v1/v2 artifacts as v3 replay inputs."""
        return capsule_v3_from_dict(value)

    def validate_captured_capsule(self, capsule: KernelCapsule) -> None:
        """Reject v1/v2 objects as well as serialized replay artifacts."""
        if capsule.schema != CAPSULE_SCHEMA_V3:
            raise ReplayIntegrityError(
                "capsule_invalid: v1/v2 capsule is not a v3 replay input"
            )
        capsule.validate()

    @staticmethod
    def _selected_launch_ids(
        capsule: KernelCapsule, variant: str
    ) -> List[str]:
        selected = list(capsule.victim_launch_ids)
        if variant == "pair":
            selected.append(capsule.aggressor_launch_id)
        return selected

    @staticmethod
    def _relationship(
        victim_start: int,
        victim_end: int,
        aggressor_start: int,
        aggressor_end: int,
    ) -> str:
        overlap = max(
            0,
            min(victim_end, aggressor_end)
            - max(victim_start, aggressor_start),
        )
        if overlap == 0:
            return "no_overlap"
        if victim_start < aggressor_start:
            return "beginning"
        if victim_end > aggressor_end:
            return "tail"
        return "contained"

    def _normalize_launch_activities(
        self,
        capsule: KernelCapsule,
        variant: str,
        record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate CUPTI correlations and derive readiness/admission waits."""
        selected_ids = self._selected_launch_ids(capsule, variant)
        launches = capsule.launch_map()
        supplied = {
            str(activity.get("launch_id")): dict(activity)
            for activity in record.get("launch_activities", [])
        }
        if set(supplied) != set(selected_ids):
            raise ReplayIntegrityError(
                "capsule_invalid: complete per-launch activity is required"
            )
        normalized: Dict[str, Dict[str, Any]] = {}
        required_times = (
            "driver_issue_start_ns",
            "driver_issue_end_ns",
            "gpu_start_ns",
            "gpu_end_ns",
        )
        for launch_id in selected_ids:
            launch = launches[launch_id]
            activity = supplied[launch_id]
            if any(activity.get(name) is None for name in required_times):
                raise ReplayIntegrityError(
                    "capsule_invalid: complete per-launch activity is required"
                )
            issue_start, issue_end, gpu_start, gpu_end = (
                int(activity[name]) for name in required_times
            )
            expected_evidence = {
                "client_id": launch.client_id,
                "context_handle": launch.context_handle,
                "stream_handle": launch.stream_handle,
                "submission_ordinal": launch.submission_sequence_index,
                "driver_command_fingerprint": (
                    launch.driver_command_fingerprint
                ),
            }
            if any(
                name in activity and activity[name] != expected
                for name, expected in expected_evidence.items()
            ):
                raise ReplayIntegrityError(
                    "capsule_invalid: replay launch activity identity changed"
                )
            if issue_end < issue_start or gpu_end <= gpu_start:
                raise ReplayIntegrityError(
                    "capsule_invalid: invalid replay launch interval"
                )
            ordinal = launch.submission_sequence_index
            normalized[launch_id] = {
                **activity,
                "launch_id": launch_id,
                "client_id": launch.client_id,
                "context_handle": launch.context_handle,
                "stream_handle": launch.stream_handle,
                "submission_ordinal": ordinal,
                "driver_command_fingerprint": (
                    launch.driver_command_fingerprint
                ),
                "driver_issue_start_ns": issue_start,
                "driver_issue_end_ns": issue_end,
                "gpu_start_ns": gpu_start,
                "gpu_end_ns": gpu_end,
                "gpu_duration_ns": gpu_end - gpu_start,
                "issue_to_gpu_start_wait_ns": gpu_start - issue_end,
            }

        for client_id in {launches[value].client_id for value in selected_ids}:
            client = sorted(
                (
                    normalized[value]
                    for value in selected_ids
                    if launches[value].client_id == client_id
                ),
                key=lambda value: value["submission_ordinal"],
            )
            if any(
                later["driver_issue_start_ns"]
                < earlier["driver_issue_end_ns"]
                for earlier, later in zip(client, client[1:])
            ):
                raise ReplayIntegrityError(
                    "capsule_invalid: per-client submission order changed"
                )

        predecessors: Dict[str, List[str]] = {
            launch_id: [] for launch_id in selected_ids
        }
        for edge in capsule.dependencies:
            if (
                edge.source_launch_id in predecessors
                and edge.target_launch_id in predecessors
            ):
                predecessors[edge.target_launch_id].append(
                    edge.source_launch_id
                )
        pending = set(selected_ids)
        while pending:
            progressed = False
            for launch_id in list(pending):
                dependencies = predecessors[launch_id]
                if any(value in pending for value in dependencies):
                    continue
                activity = normalized[launch_id]
                ready = (
                    max(
                        normalized[value]["gpu_end_ns"]
                        for value in dependencies
                    )
                    if dependencies
                    else activity["driver_issue_end_ns"]
                )
                if dependencies and activity["gpu_start_ns"] < ready:
                    raise ReplayIntegrityError(
                        "capsule_invalid: replay dependency order changed"
                    )
                activity["replay_ready_ns"] = ready
                activity["admission_wait_ns"] = (
                    activity["gpu_start_ns"] - ready
                )
                pending.remove(launch_id)
                progressed = True
            if not progressed:
                raise ReplayIntegrityError(
                    "capsule_invalid: replay dependency graph is cyclic"
                )

        aggressor = normalized.get(capsule.aggressor_launch_id)
        for launch_id, activity in normalized.items():
            if aggressor is None or launch_id == capsule.aggressor_launch_id:
                overlap = 0
                relationship = "no_overlap"
            else:
                overlap = max(
                    0,
                    min(activity["gpu_end_ns"], aggressor["gpu_end_ns"])
                    - max(
                        activity["gpu_start_ns"],
                        aggressor["gpu_start_ns"],
                    ),
                )
                relationship = self._relationship(
                    activity["gpu_start_ns"],
                    activity["gpu_end_ns"],
                    aggressor["gpu_start_ns"],
                    aggressor["gpu_end_ns"],
                )
            activity["aggressor_overlap_duration_ns"] = overlap
            activity["overlap_relationship"] = relationship
        return sorted(
            normalized.values(),
            key=lambda value: (
                value["client_id"], value["submission_ordinal"]
            ),
        )

    @staticmethod
    def _source_interval_order(
        capsule: KernelCapsule,
        launch_ids: List[str],
        interval_name: str,
    ) -> List[str]:
        """Return source launch IDs ordered only by interval start time."""
        launches = capsule.launch_map()
        return sorted(
            launch_ids,
            key=lambda launch_id: (
                int(getattr(launches[launch_id], interval_name)["start_ns"]),
                int(getattr(launches[launch_id], interval_name)["end_ns"]),
                launch_id,
            ),
        )

    @staticmethod
    def _replay_interval_order(
        activities: List[Dict[str, Any]],
        start_name: str,
        end_name: str,
    ) -> List[str]:
        """Return replay launch IDs ordered only by interval start time."""
        return [
            value["launch_id"]
            for value in sorted(
                activities,
                key=lambda value: (
                    int(value[start_name]),
                    int(value[end_name]),
                    value["launch_id"],
                ),
            )
        ]

    def _record_order_diagnostics(
        self,
        capsule: KernelCapsule,
        record: Dict[str, Any],
        selected_ids: List[str],
    ) -> None:
        """Compare source and replay orders without comparing time offsets."""
        activities = record["launch_activities"]
        source_host_order = self._source_interval_order(
            capsule, selected_ids, "source_host_launch_interval"
        )
        source_gpu_order = self._source_interval_order(
            capsule, selected_ids, "source_gpu_interval"
        )
        replay_driver_order = self._replay_interval_order(
            activities, "driver_issue_start_ns", "driver_issue_end_ns"
        )
        replay_gpu_order = self._replay_interval_order(
            activities, "gpu_start_ns", "gpu_end_ns"
        )
        source_gpu_starts = [
            int(
                capsule.launch_map()[launch_id].source_gpu_interval[
                    "start_ns"
                ]
            )
            for launch_id in selected_ids
        ]
        replay_gpu_starts = [
            int(activity["gpu_start_ns"]) for activity in activities
        ]
        gpu_start_order_ambiguous = (
            len(set(source_gpu_starts)) != len(source_gpu_starts)
            or len(set(replay_gpu_starts)) != len(replay_gpu_starts)
        )
        gpu_start_order_match = (
            not gpu_start_order_ambiguous
            and replay_gpu_order == source_gpu_order
        )
        record.update({
            "source_host_launch_order": source_host_order,
            "replay_driver_issue_order": replay_driver_order,
            "host_launch_order_match": (
                replay_driver_order == source_host_order
            ),
            "source_gpu_start_order": source_gpu_order,
            "replay_gpu_start_order": replay_gpu_order,
            "gpu_start_order_ambiguous": gpu_start_order_ambiguous,
            "gpu_start_order_match": gpu_start_order_match,
            "gpu_sequence_match_status": (
                "ambiguous"
                if gpu_start_order_ambiguous
                else "matched" if gpu_start_order_match else "mismatched"
            ),
        })

    def _restore_and_replay(
        self,
        store: GpuForensicsStore,
        capsule: KernelCapsule,
        phase: str,
        variant: str,
        instrumentation: str,
        repetition: int,
        persist: bool = True,
        aggressor_priority: Optional[int] = None,
        counterfactual_condition: Optional[str] = None,
    ) -> Dict[str, Any]:
        restores = self.backend.restore_checkpoints(capsule)
        if restores != len(capsule.clients):
            raise ReplayIntegrityError(
                "capsule_invalid: every client checkpoint must be restored"
            )
        if aggressor_priority is None:
            replay = self.backend.replay(
                capsule, variant=variant, instrumentation=instrumentation
            )
        else:
            replay = self.backend.replay_with_aggressor_priority(
                capsule, aggressor_priority
            )
        record = dict(replay)
        record.update(
            {
                "phase": phase,
                "variant": variant,
                "instrumentation": instrumentation,
                "repetition": repetition,
                "checkpoint_restore_count": restores,
                "submission_policy": self.submission_policy,
                "counterfactual_condition": counterfactual_condition,
                "declared_aggressor_priority_intervention": (
                    aggressor_priority
                ),
            }
        )
        if record.get("source_gpu_offsets_used_for_submission") is not False:
            raise ReplayIntegrityError(
                "capsule_invalid: source GPU offsets controlled submission"
            )
        for key, message in (
            ("frame_replay_count", "frame replay executed after capture"),
            ("model_forward_count", "model forward executed after capture"),
            ("rosbag_work_count", "rosbag work executed after capture"),
            (
                "non_capsule_cuda_work_count",
                "capsule_invalid: non-capsule CUDA work",
            ),
        ):
            if int(record.get(key, 0)):
                raise ReplayIntegrityError(message)
        activities = self._normalize_launch_activities(
            capsule, variant, record
        )
        record["launch_activities"] = activities
        expected_ids = self._selected_launch_ids(capsule, variant)
        self._record_order_diagnostics(capsule, record, expected_ids)
        expected_fingerprints = {
            launch_id: capsule.launch_map()[
                launch_id
            ].driver_command_fingerprint
            for launch_id in expected_ids
        }
        if record.get("launch_fingerprints") != expected_fingerprints:
            record["fingerprint_match"] = False
        launch_order = record.get("launch_order", [])
        if set(launch_order) != set(expected_ids) or len(launch_order) != len(
            expected_ids
        ):
            record["fingerprint_match"] = False
        else:
            positions = {
                launch_id: index
                for index, launch_id in enumerate(launch_order)
            }
            for client_id in {
                capsule.launch_map()[value].client_id for value in expected_ids
            }:
                expected_client_order = sorted(
                    (
                        value for value in expected_ids
                        if capsule.launch_map()[value].client_id == client_id
                    ),
                    key=lambda value: capsule.launch_map()[
                        value
                    ].submission_sequence_index,
                )
                if expected_client_order != sorted(
                    expected_client_order, key=lambda value: positions[value]
                ):
                    record["fingerprint_match"] = False
        if record.get("victim_order") != capsule.victim_launch_ids:
            record["fingerprint_match"] = False
        head = next(
            value
            for value in activities
            if value["launch_id"] == capsule.victim_head_launch_id
        )
        record["victim_head_launch_id"] = capsule.victim_head_launch_id
        record["victim_delay_ns"] = head["admission_wait_ns"]
        aggressor = next(
            (
                value
                for value in activities
                if value["launch_id"] == capsule.aggressor_launch_id
            ),
            None,
        )
        record["aggressor_duration_ns"] = (
            aggressor["gpu_duration_ns"] if aggressor else None
        )
        if persist:
            self._persist_replay_record(store, record)
        return record

    def _persist_replay_record(
        self, store: GpuForensicsStore, record: Dict[str, Any]
    ) -> None:
        """Persist only a replay record accepted by the v3 sequence gate."""
        record["iteration_id"] = store.record_iteration(
            record["phase"],
            record["variant"],
            int(record["repetition"]),
            record,
        )
        self.records.append(record)

    def _collect_resources(
        self,
        store: GpuForensicsStore,
        capsule: KernelCapsule,
        _available: Dict[str, bool],
    ) -> None:
        """Collect CUDA launch/resource facts for every captured launch."""
        for launch_id in [capsule.aggressor_launch_id] + (
            capsule.victim_launch_ids
        ):
            record = self.backend.collect_resource_profile(
                capsule, launch_id
            )
            record["timing_excluded_from_conclusions"] = True
            store.record_resource(launch_id, record)

    @staticmethod
    def _victim_schedule(
        record: Dict[str, Any], capsule: KernelCapsule
    ) -> List[tuple]:
        victim_ids = set(capsule.victim_launch_ids)
        return sorted(
            (
                value["launch_id"],
                value["client_id"],
                value["context_handle"],
                value["stream_handle"],
                value["submission_ordinal"],
                value["driver_command_fingerprint"],
            )
            for value in record["launch_activities"]
            if value["launch_id"] in victim_ids
        )

    @staticmethod
    def _require_sequence_match(record: Dict[str, Any], variant: str) -> None:
        """Require an unambiguous source-matched GPU-start sequence."""
        if not record.get("gpu_start_order_match", False):
            status = record.get("gpu_sequence_match_status", "mismatched")
            raise ReplayIntegrityError(
                "capsule_invalid: "
                f"{variant} GPU-start sequence is {status}"
            )

    def _run_timing_campaign(
        self,
        store: GpuForensicsStore,
        target: Dict[str, Any],
        capsule: KernelCapsule,
    ) -> Dict[str, Any]:
        """Keep the first matching pair, then one matching victim baseline."""
        accepted_pair = None
        for attempt in range(1, self.sequence_match_attempts + 1):
            record = self._restore_and_replay(
                store,
                capsule,
                "timing",
                "pair",
                "none",
                0,
                persist=False,
            )
            if record.get("gpu_start_order_match", False):
                record["sequence_match_attempt"] = attempt
                accepted_pair = record
                break
        if accepted_pair is None:
            raise ReplayIntegrityError(
                "capsule_invalid: no unambiguous pair GPU-start sequence "
                f"match in {self.sequence_match_attempts} attempts"
            )

        baseline = self._restore_and_replay(
            store,
            capsule,
            "timing",
            "victim_only",
            "none",
            0,
            persist=False,
        )
        self._require_sequence_match(baseline, "victim-only baseline")

        # Validate the complete selected timing pair before either record is
        # admitted to in-memory, CSV, or SQLite evidence.
        gate = self._integrity_gate_records(
            target, capsule, accepted_pair, baseline
        )
        self.accepted_pair_integrity = {
            name: accepted_pair.get(name)
            for name in (
                "checkpoint_state_sha256",
                "allocation_map_sha256",
                "output_sha256",
            )
        }
        self._persist_replay_record(store, accepted_pair)
        self._persist_replay_record(store, baseline)
        return gate

    def _run_optional_collector(
        self,
        store: GpuForensicsStore,
        capsule: KernelCapsule,
        collector: str,
        phase: str,
        instrumentation: str,
    ) -> bool:
        """Run optional CTA/metric evidence once and retain only a match."""
        record = self._restore_and_replay(
            store,
            capsule,
            phase,
            "pair",
            instrumentation,
            0,
            persist=False,
        )
        if not record.get("gpu_start_order_match", False):
            status = record.get("gpu_sequence_match_status", "mismatched")
            store.record_capability(
                collector,
                False,
                False,
                f"discarded: GPU-start sequence {status}",
            )
            return False
        if int(record.get("foreign_work_count", 0)):
            raise ReplayIntegrityError("foreign GPU work observed")
        if not record.get("fingerprint_match", False):
            raise ReplayIntegrityError(
                "capsule_invalid: launch fingerprint mismatch"
            )
        if collector == "nvbit_cta":
            reason = None
            if any(
                record.get(name) != value
                for name, value in self.accepted_pair_integrity.items()
            ):
                reason = "integrity hash mismatch"
            elif not record.get("cta_collection_valid", False):
                errors = record.get("cta_collection_errors", [])
                reason = "CTA collection unavailable"
                if errors:
                    reason += ": " + "; ".join(
                        str(value.get("error")) for value in errors
                    )
            else:
                try:
                    self._validate_cta_evidence(capsule, record)
                except (KeyError, TypeError, ValueError) as error:
                    reason = f"invalid CTA evidence: {error}"
            if reason is not None:
                store.record_capability(
                    collector, False, False, f"discarded: {reason}"
                )
                return False
        self._persist_replay_record(store, record)
        return True

    @staticmethod
    def _validate_cta_evidence(
        capsule: KernelCapsule, record: Dict[str, Any]
    ) -> None:
        """Validate complete slot mapping while allowing missing boundaries."""
        launches = capsule.launch_map()
        selected = [capsule.aggressor_launch_id] + capsule.victim_launch_ids
        expected = {
            launch_id: (
                launches[launch_id].grid[0]
                * launches[launch_id].grid[1]
                * launches[launch_id].grid[2]
            )
            for launch_id in selected
        }
        statuses = {
            str(value["launch_id"]): value
            for value in record["cta_collection_status"]
        }
        if set(statuses) != set(expected):
            raise ValueError("collection status launch mapping changed")
        rows = record["cta_intervals"]
        if len(rows) != sum(expected.values()):
            raise ValueError("CTA slot count differs from captured grid")
        seen = set()
        observed = {
            launch_id: {"entered": 0, "exited": 0, "complete": 0}
            for launch_id in selected
        }
        for row in rows:
            launch_id = str(row["launch_id"])
            cta_id = int(row["cta_id"])
            identity = (launch_id, cta_id)
            if launch_id not in expected or not 0 <= cta_id < expected[
                launch_id
            ] or identity in seen:
                raise ValueError("CTA identity is out of bounds or duplicated")
            seen.add(identity)
            entered = bool(row["entry_observed"])
            exited = bool(row["exit_observed"])
            entry = row.get("entry_ns")
            exit_ = row.get("exit_ns")
            if entered != (entry is not None) or exited != (exit_ is not None):
                raise ValueError("CTA boundary flags disagree with timestamps")
            if (entered or exited) and row.get("clock_error_ns") is None:
                raise ValueError("observed CTA has no clock error")
            if entry is not None and exit_ is not None and int(entry) > int(
                exit_
            ):
                raise ValueError("CTA entry follows exit")
            observed[launch_id]["entered"] += entered
            observed[launch_id]["exited"] += exited
            observed[launch_id]["complete"] += entered and exited
        for launch_id, total in expected.items():
            status = statuses[launch_id]
            counts = observed[launch_id]
            if (
                int(status["expected_count"]) != total
                or int(status["entered_count"]) != counts["entered"]
                or int(status["exited_count"]) != counts["exited"]
                or int(status["complete_count"]) != counts["complete"]
                or int(status["missing_count"]) != total - counts["complete"]
                or int(status["dropped_entry_count"])
                != total - counts["entered"]
                or int(status["dropped_exit_count"])
                != total - counts["exited"]
            ):
                raise ValueError("CTA coverage counts are inconsistent")

    @staticmethod
    def _priority_sample(
        record: Dict[str, Any], capsule: KernelCapsule
    ) -> Dict[str, Any]:
        """Reduce one counterfactual replay to victim-performance outcomes."""
        victims = [
            activity for activity in record["launch_activities"]
            if activity["launch_id"] in capsule.victim_launch_ids
        ]
        return {
            "condition": record["counterfactual_condition"],
            "trial": int(record["repetition"]),
            "aggressor_priority": record.get("aggressor_priority"),
            "priority_attribute_applied": bool(
                record.get("priority_attribute_applied", False)
            ),
            "victim_delay_ns": int(record["victim_delay_ns"]),
            "victim_completion_latency_ns": (
                max(int(value["gpu_end_ns"]) for value in victims)
                - min(int(value["replay_ready_ns"]) for value in victims)
            ),
            "victim_gpu_duration_ns": sum(
                int(value["gpu_duration_ns"]) for value in victims
            ),
            "aggressor_duration_ns": record.get("aggressor_duration_ns"),
            "gpu_start_sequence": record["replay_gpu_start_order"],
            "gpu_sequence_match_status": record[
                "gpu_sequence_match_status"
            ],
        }

    def _classify_priority_metric(
        self,
        samples: Dict[str, List[Dict[str, Any]]],
        metric: str,
    ) -> Dict[str, Any]:
        """Compare greatest/least priority against paired sham variability."""
        normalized = f"pair_minus_victim_only_{metric}"
        greatest = {
            value["trial"]: int(value[normalized])
            for value in samples["greatest_priority"]
        }
        least = {
            value["trial"]: int(value[normalized])
            for value in samples["least_priority"]
        }
        captured = {
            value["trial"]: int(value[normalized])
            for value in samples["captured"]
        }
        sham = {
            value["trial"]: int(value[normalized])
            for value in samples["sham"]
        }
        effects = [
            greatest[trial] - least[trial]
            for trial in range(self.priority_counterfactual_trials)
        ]
        sham_effects = [
            captured[trial] - sham[trial]
            for trial in range(self.priority_counterfactual_trials)
        ]
        noise_floor = max(
            self.priority_min_effect_ns,
            *(abs(value) for value in sham_effects),
        )
        median_effect = float(statistics.median(effects))
        direction = 1 if median_effect > 0 else -1 if median_effect < 0 else 0
        matching = sum(
            abs(value) > noise_floor
            and direction != 0
            and (1 if value > 0 else -1) == direction
            for value in effects
        )
        if matching >= 2:
            status = "supported"
        elif all(abs(value) <= noise_floor for value in effects):
            status = "ruled_out_for_workload"
        else:
            status = "inconclusive"
        return {
            "status": status,
            "metric": metric,
            "greatest_minus_least_effects_ns": effects,
            "captured_minus_sham_effects_ns": sham_effects,
            "median_effect_ns": median_effect,
            "noise_floor_ns": noise_floor,
            "matching_direction_trials": matching,
        }

    def _run_priority_counterfactual(
        self, store: GpuForensicsStore, capsule: KernelCapsule
    ) -> Optional[Dict[str, Any]]:
        """Test whether aggressor launch priority changes victim outcomes."""
        if not self.priority_counterfactual_enabled:
            return None
        try:
            priority_range = dict(
                self.backend.aggressor_priority_range(capsule)
            )
        except (NotImplementedError, RuntimeError) as error:
            result = {
                "schema": "aggressor_priority_counterfactual_v1",
                "status": "not_observable",
                "reason": str(error),
                "samples": {},
            }
            self._write_priority_counterfactual(result)
            return result
        if (
            not priority_range.get("supported", False)
            or int(priority_range["least_priority"])
            == int(priority_range["greatest_priority"])
        ):
            result = {
                "schema": "aggressor_priority_counterfactual_v1",
                "status": "not_observable",
                "reason": "aggressor context exposes no priority range",
                "priority_range": priority_range,
                "samples": {},
            }
            self._write_priority_counterfactual(result)
            return result

        conditions = (
            ("captured", "pair", None),
            (
                "sham",
                "pair",
                int(priority_range["captured_priority"]),
            ),
            (
                "greatest_priority",
                "pair",
                int(priority_range["greatest_priority"]),
            ),
            (
                "least_priority",
                "pair",
                int(priority_range["least_priority"]),
            ),
            ("victim_only", "victim_only", None),
        )
        records: Dict[str, List[Dict[str, Any]]] = {
            condition[0]: [] for condition in conditions
        }
        for trial in range(self.priority_counterfactual_trials):
            ordered = conditions if trial % 2 == 0 else tuple(
                reversed(conditions)
            )
            for condition, variant, priority in ordered:
                instrumentation = (
                    "launch_priority_counterfactual"
                    if priority is not None else "none"
                )
                record = self._restore_and_replay(
                    store,
                    capsule,
                    "priority_counterfactual",
                    variant,
                    instrumentation,
                    trial,
                    persist=False,
                    aggressor_priority=priority,
                    counterfactual_condition=condition,
                )
                if int(record.get("foreign_work_count", 0)):
                    raise ReplayIntegrityError("foreign GPU work observed")
                if not record.get("fingerprint_match", False):
                    raise ReplayIntegrityError(
                        "capsule_invalid: base launch fingerprint changed "
                        "during priority counterfactual"
                    )
                if priority is not None and (
                    not record.get("priority_attribute_applied", False)
                    or int(record.get("aggressor_priority")) != priority
                ):
                    raise ReplayIntegrityError(
                        "capsule_invalid: aggressor priority was not applied"
                    )
                self._persist_replay_record(store, record)
                records[condition].append(
                    self._priority_sample(record, capsule)
                )

        baseline = {
            value["trial"]: value for value in records["victim_only"]
        }
        metrics = (
            "victim_delay_ns",
            "victim_completion_latency_ns",
            "victim_gpu_duration_ns",
        )
        for condition, values in records.items():
            if condition == "victim_only":
                continue
            for value in values:
                for metric in metrics:
                    value[f"pair_minus_victim_only_{metric}"] = (
                        int(value[metric])
                        - int(baseline[value["trial"]][metric])
                    )
        findings = {
            metric: self._classify_priority_metric(records, metric)
            for metric in metrics
        }
        statuses = {value["status"] for value in findings.values()}
        if "supported" in statuses:
            status = "supported"
        elif statuses == {"ruled_out_for_workload"}:
            status = "ruled_out_for_workload"
        else:
            status = "inconclusive"
        result = {
            "schema": "aggressor_priority_counterfactual_v1",
            "status": status,
            "primary_outcome": "victim performance",
            "priority_range": priority_range,
            "trial_count": self.priority_counterfactual_trials,
            "minimum_effect_ns": self.priority_min_effect_ns,
            "gpu_start_sequence_is_outcome_not_acceptance_gate": True,
            "metric_findings": findings,
            "samples": records,
        }
        self._write_priority_counterfactual(result)
        return result

    def _write_priority_counterfactual(
        self, result: Dict[str, Any]
    ) -> None:
        """Write one inspectable priority-validation result atomically."""
        output = self.output_directory / "priority_counterfactual.json"
        temporary = output.with_name("." + output.name + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)

    def _integrity_gate_records(
        self,
        _target: Dict[str, Any],
        capsule: KernelCapsule,
        pair: Dict[str, Any],
        victim: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and summarize one accepted pair and one baseline."""
        all_records = (pair, victim)
        if any(
            int(value.get("foreign_work_count", 0))
            for value in all_records
        ):
            raise ReplayIntegrityError("foreign GPU work observed")
        if any(
            not value.get("fingerprint_match", False)
            for value in all_records
        ):
            raise ReplayIntegrityError(
                "capsule_invalid: launch fingerprint mismatch"
            )
        if self._victim_schedule(pair, capsule) != self._victim_schedule(
            victim, capsule
        ):
            raise ReplayIntegrityError(
                "capsule_invalid: matched victim command schedules changed"
            )
        checkpoint_states = {
            value.get("checkpoint_state_sha256") for value in all_records
        }
        allocation_states = {
            value.get("allocation_map_sha256") for value in all_records
        }
        if None in checkpoint_states or len(checkpoint_states) != 1:
            raise ReplayIntegrityError(
                "capsule_invalid: checkpoint restoration changed"
            )
        if None in allocation_states or len(allocation_states) != 1:
            raise ReplayIntegrityError(
                "capsule_invalid: allocation identity changed"
            )
        if any(value.get("output_sha256") is None for value in all_records):
            raise ReplayIntegrityError(
                "capsule_invalid: checkpoint output is missing"
            )
        self._require_sequence_match(pair, "pair")
        self._require_sequence_match(victim, "victim-only baseline")
        relationships = {
            value["overlap_relationship"]
            for value in pair["launch_activities"]
            if value["launch_id"] in capsule.victim_launch_ids
        }
        return {
            "status": "passed",
            "sequence_match_status": "matched",
            "accepted_source_gpu_sequence": pair[
                "source_gpu_start_order"
            ],
            "accepted_replay_gpu_sequence": pair[
                "replay_gpu_start_order"
            ],
            "accepted_victim_only_source_gpu_sequence": victim[
                "source_gpu_start_order"
            ],
            "accepted_victim_only_replay_gpu_sequence": victim[
                "replay_gpu_start_order"
            ],
            "pair_minus_victim_only_admission_delay_ns": (
                int(pair["victim_delay_ns"])
                - int(victim["victim_delay_ns"])
            ),
            "aggressor_duration_ns": int(pair["aggressor_duration_ns"]),
            "observed_replay_overlap_relationships": sorted(relationships),
            "accepted_pair_attempt": int(pair["sequence_match_attempt"]),
            "sequence_match_attempts": self.sequence_match_attempts,
            "timing_repetitions": 1,
        }

    def _integrity_gate(
        self, _target: Dict[str, Any], capsule: KernelCapsule
    ) -> Dict[str, Any]:
        pairs = [
            value for value in self.records
            if value["phase"] == "timing" and value["variant"] == "pair"
        ]
        victims = [
            value for value in self.records
            if value["phase"] == "timing"
            and value["variant"] == "victim_only"
        ]
        if len(pairs) != 1 or len(victims) != 1:
            raise ReplayIntegrityError(
                "replay integrity gate requires one matched pair and baseline"
            )
        return self._integrity_gate_records(
            _target, capsule, pairs[0], victims[0]
        )

    def _write_kernel_timing_csv(self) -> None:
        fields = (
            "phase", "variant", "repetition", "counterfactual_condition",
            "aggressor_priority", "launch_id", "client_id",
            "context_handle", "stream_handle", "submission_ordinal",
            "driver_command_fingerprint", "driver_issue_start_ns",
            "driver_issue_end_ns", "gpu_start_ns", "gpu_end_ns",
            "gpu_duration_ns", "issue_to_gpu_start_wait_ns",
            "replay_ready_ns", "admission_wait_ns",
            "aggressor_overlap_duration_ns", "overlap_relationship",
        )
        path = self.output_directory / "capsule_kernel_timing.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                for activity in record.get("launch_activities", []):
                    writer.writerow(
                        {
                            "phase": record["phase"],
                            "variant": record["variant"],
                            "repetition": record["repetition"],
                            "counterfactual_condition": record.get(
                                "counterfactual_condition"
                            ),
                            "aggressor_priority": record.get(
                                "aggressor_priority"
                            ),
                            **{
                                name: activity.get(name)
                                for name in fields[5:]
                            },
                        }
                    )

    def _write_timing_csv(self) -> None:
        """Write per-repetition aggregate victim-head admission evidence."""
        fields = (
            "phase",
            "variant",
            "repetition",
            "victim_head_launch_id",
            "victim_head_admission_wait_ns",
            "aggressor_duration_ns",
            "checkpoint_restore_count",
            "fingerprint_match",
            "foreign_work_count",
            "overlap_relationships",
            "source_host_launch_order",
            "replay_driver_issue_order",
            "host_launch_order_match",
            "source_gpu_start_order",
            "replay_gpu_start_order",
            "gpu_start_order_ambiguous",
            "gpu_start_order_match",
            "gpu_sequence_match_status",
        )
        path = self.output_directory / "capsule_timing.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                if record["phase"] != "timing":
                    continue
                relationships = sorted(
                    {
                        value["overlap_relationship"]
                        for value in record.get("launch_activities", [])
                        if value["launch_id"].startswith("victim-")
                    }
                )
                writer.writerow(
                    {
                        "phase": record["phase"],
                        "variant": record["variant"],
                        "repetition": record["repetition"],
                        "victim_head_launch_id": record[
                            "victim_head_launch_id"
                        ],
                        "victim_head_admission_wait_ns": record[
                            "victim_delay_ns"
                        ],
                        "aggressor_duration_ns": record.get(
                            "aggressor_duration_ns"
                        ),
                        "checkpoint_restore_count": record[
                            "checkpoint_restore_count"
                        ],
                        "fingerprint_match": record.get("fingerprint_match"),
                        "foreign_work_count": record.get("foreign_work_count"),
                        "overlap_relationships": ";".join(relationships),
                        "source_host_launch_order": ";".join(record[
                            "source_host_launch_order"
                        ]),
                        "replay_driver_issue_order": ";".join(record[
                            "replay_driver_issue_order"
                        ]),
                        "host_launch_order_match": record[
                            "host_launch_order_match"
                        ],
                        "source_gpu_start_order": ";".join(record[
                            "source_gpu_start_order"
                        ]),
                        "replay_gpu_start_order": ";".join(record[
                            "replay_gpu_start_order"
                        ]),
                        "gpu_start_order_ambiguous": record[
                            "gpu_start_order_ambiguous"
                        ],
                        "gpu_start_order_match": record[
                            "gpu_start_order_match"
                        ],
                        "gpu_sequence_match_status": record[
                            "gpu_sequence_match_status"
                        ],
                    }
                )


# Import compatibility for callers that only name the historical planner.
# The alias executes the v2 backend and never accepts v1 artifacts.
KernelReplayPlannerV1 = KernelReplayPlannerV2
