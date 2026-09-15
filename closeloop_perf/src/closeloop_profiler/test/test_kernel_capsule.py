"""Tests for exact live capsule identity and fixed kernel-only replay."""

import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest

from closeloop_profiler.kernel_capsule import (
    CapsuleDependency,
    CapsuleInvalid,
    KernelCapsule,
    LaunchFingerprint,
    MpsClientState,
    ParameterValue,
    capsule_from_dict,
    capsule_v3_from_dict,
    read_capsule_manifest,
)
from closeloop_experiments.capsule_coordinator import KernelCapsuleCoordinator
from closeloop_experiments.kernel_replay import (
    KernelReplayPlannerV2,
    KernelReplayPlannerV3,
    ReplayIntegrityError,
    ReplayBackend,
    capture_specification_v3,
)
from closeloop_testbed.mps_microbenchmark import (
    validate_trials,
    validation_matrix,
)


def launch(launch_id, client, symbol):
    """Return one valid exact launch fingerprint."""
    value = LaunchFingerprint(
        launch_id=launch_id,
        client_id=client,
        context_handle=f"ctx-{client}",
        function_handle=f"func-{launch_id}",
        code_object_sha256="a" * 64,
        symbol=symbol,
        symbol_sha256=hashlib.sha256(symbol.encode()).hexdigest(),
        launch_api="cuLaunchKernel",
        grid=(1, 1, 1),
        block=(32, 1, 1),
        cluster=None,
        dynamic_shared_memory=0,
        stream_handle=f"stream-{client}",
        stream_priority=0,
        function_attributes={"registers": 16},
        parameters=(ParameterValue(0, 0, 8, "0100000000000000"),),
        relative_ready_offset_ns=0,
        relative_launch_offset_ns=0,
        source_runtime_api="cudaLaunchKernel_v7000",
        captured_driver_api="cuLaunchKernel",
        replay_api="cuLaunchKernel",
        source_runtime_correlation_id=10,
        nested_driver_correlation_id=11,
        cupti_correlation_id=11,
        runtime_stub_handle=f"stub-{launch_id}",
    )
    return replace(value, driver_command_fingerprint=value.command_digest)


def capsule(stage_evidence=None):
    """Return a valid two-client capsule with one victim edge."""
    launches = [
        launch("aggressor", "a", "persistent"),
        launch("victim-0", "v", "victim_a"),
        launch("victim-1", "v", "victim_b"),
    ]
    return KernelCapsule(
        capsule_id="capsule-1",
        source_run_id="source",
        aggressor_launch_id="aggressor",
        victim_launch_ids=["victim-0", "victim-1"],
        launches=launches,
        dependencies=[
            CapsuleDependency("victim-0", "victim-1", "same_stream")
        ],
        clients=[
            MpsClientState("a", 1, 2, "ctx-a", 100, 0, 2, "cp-a", True),
            MpsClientState("v", 1, 3, "ctx-v", 100, 0, 2, "cp-v", True),
        ],
        capture_complete=True,
        applications_blocked=True,
        recursive_callbacks_disabled=True,
        no_later_model_work_submitted=True,
        capabilities={"stage_evidence": stage_evidence or {}},
        terminal_streams=["stream-a", "stream-v"],
        capture_matcher_evidence={"a": ["aggressor"], "v": ["victim"]},
        agent_build={"version": "test"},
        overlap_evidence={
            "aggressor_source_interval": {"start_ns": 0, "end_ns": 2},
            "victim_source_intervals": [
                {
                    "overlap_duration_ns": 1,
                    "source_gpu_interval": {"start_ns": 0, "end_ns": 1},
                    "stream_membership": {
                        "context_id": 1, "stream_id": index,
                    },
                }
                for index in range(2)
            ],
            "complete": True,
        },
    )


def target(source_delay=1_000_000, aggressor_duration=2_000_000):
    """Return the selected-target subset consumed by the planner."""
    return {
        "schema": "target_selection_v2",
        "schema_version": 2,
        "selected_pair": {
            "aggressor_signature": "a",
            "victim_overlap_signature": "v",
            "aggregate_excess_wait_ns": source_delay,
            "occurrence_count": 1,
        },
        "source_episode": {
            "excess_wait_ns": source_delay,
            "ready_ns": 0,
            "aggressor": {
                "start_ns": 10,
                "end_ns": 10 + aggressor_duration,
                "launch_start_ns": 1,
                "launch_end_ns": 2,
                "model_identity": "a",
            },
            "aggressor_source_interval": {
                "start_ns": 10,
                "end_ns": 10 + aggressor_duration,
            },
            "victim_head_kernel_id": 1,
            "victim_kernels": [{
                "source_kernel_id": 1,
                "launch_start_ns": 3,
                "launch_end_ns": 4,
                "model_identity": "v",
                "source_gpu_interval": {"start_ns": 11, "end_ns": 12},
                "overlap_duration_ns": 1,
                "stream_membership": {
                    "pid": 2, "context_id": 1, "stream_id": 1,
                    "stream_priority": 0,
                },
            }],
            "victim_dependencies": [],
            "victim_set": "gpu_execution_overlap_all_streams",
            "overlap_set_complete": True,
            "inference_inputs": {"a": "0", "v": "0"},
        },
        "source_frame": {
            "schema": "source_frame_identity_v1",
            "ordinal": 0,
            "storage_ordinal": 0,
            "topic_ordinal": 0,
            "bag_path": "/test/source.mcap",
            "bag_timestamp_ns": 1,
            "ros_header_timestamp_ns": 1,
            "topic": "/camera",
            "message_type": "sensor_msgs/msg/CompressedImage",
            "serialization_format": "cdr",
            "raw_cdr_sha256": "0" * 64,
            "raw_cdr_size": 0,
        },
    }


class FakeBackend(ReplayBackend):
    """Deterministic live-agent stand-in with no frame/model/rosbag work."""

    def __init__(self, reproduce=True):
        self.capsule = capsule(
            {
                "calibrated_task_slot_capacity": 8,
                "task_slot_admission_signature": False,
            }
        )
        self.reproduce = reproduce
        self.restore_calls = 0
        self.replay_calls = []
        self.resource_calls = []

    def capture_one_instrumented_frame(self, specification):
        assert specification["additional_capsule_warmup"] is False
        assert specification["capture_operation_not_replay"] is True
        assert specification["schema"] == "kernel_capsule_capture_spec_v2"
        assert specification["victim_set"] == (
            "gpu_execution_overlap_all_streams"
        )
        assert specification["completion_action"]["mode"] == (
            "terminal_stream_synchronize"
        )
        assert not specification["completion_action"][
            "enqueue_completion_event"
        ]
        return self.capsule

    def capabilities(self):
        return {
            name: {"available": True, "version": "test"}
            for name in (
                "cupti_activity",
                "cuda_launch_capture",
                "cupti_checkpoint",
                "cupti_pm",
                "nvbit_cta",
                "ncu_mps",
            )
        }

    def restore_checkpoints(self, captured):
        self.restore_calls += 1
        return len(captured.clients)

    def replay(self, captured, variant, instrumentation):
        self.replay_calls.append((variant, instrumentation))
        baseline = 100_000
        pair_delay = baseline + (1_000_000 if self.reproduce else 0)
        order = list(captured.victim_launch_ids)
        launch_order = list(order)
        if variant == "pair":
            launch_order.insert(0, captured.aggressor_launch_id)
        ctas = []
        if instrumentation == "nvbit_cta":
            ctas = [
                {
                    "launch_id": "victim-0",
                    "sm_id": 0,
                    "cta_id": "0",
                    "entry_ns": 10,
                    "exit_ns": 20,
                    "clock_error_ns": 1,
                }
            ]
        samples = []
        if instrumentation == "cupti_pm":
            samples = [
                {
                    "timestamp_ns": 10,
                    "metric": "sm_active",
                    "value": 0.5,
                    "unit": "ratio",
                    "precision_class": "sampled",
                }
            ]
        return {
            "victim_delay_ns": pair_delay if variant == "pair" else baseline,
            "aggressor_duration_ns": 2_000_000 if variant == "pair" else None,
            "victim_order": order,
            "launch_order": launch_order,
            "launch_fingerprints": {
                launch_id: captured.launch_map()[launch_id].digest
                for launch_id in launch_order
            },
            "fingerprint_match": True,
            "checkpoint_state_sha256": "c" * 64,
            "allocation_map_sha256": "a" * 64,
            "output_sha256": ("p" if variant == "pair" else "v") * 64,
            "foreign_work_count": 0,
            "non_capsule_cuda_work_count": 0,
            "frame_replay_count": 0,
            "model_forward_count": 0,
            "rosbag_work_count": 0,
            "profiler_perturbation": instrumentation,
            "cta_intervals": ctas,
            "metric_samples": samples,
        }

    def collect_resource_profile(self, captured, launch_id):
        self.resource_calls.append(launch_id)
        return {
            "registers_per_thread": 16,
            "threads_per_block": 32,
            "warps_per_block": 1,
            "occupancy_blocks_per_sm": 4,
        }


class FakeAgentEndpoint:
    """Return one fixed client-local collector capability response."""

    def __init__(self, client_id, nvbit):
        """Configure a fake client and its optional NVBit state."""
        self.client_id = client_id
        self.nvbit = nvbit

    def request(self, operation, **_payload):
        """Respond to the coordinator capability probe."""
        assert operation == "capabilities"
        return {
            "collectors": {
                "cupti_activity": {"available": True, "version": "12.6"},
                "nvbit_cta": {"available": self.nvbit},
            }
        }


class StubPlanner(KernelReplayPlannerV2):
    """Avoid source selection so planner tests exercise only live phases."""

    selected_target = target()

    def select_target(self):
        value = self.selected_target
        (self.output_directory / "target_selection.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
        return value

    def prepare_capture_input(self, _target):
        """Avoid requiring a real MCAP in fixed-phase planner tests."""
        payload = self.output_directory / "capture_input.bin"
        manifest = self.output_directory / "capture_input.json"
        payload.write_bytes(b"")
        value = {
            "schema": "kernel_capsule_input_v2",
            "payload_path": str(payload),
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        return value


def v3_capsule():
    """Return a dependency-burst form of the test capsule."""
    value = capsule()
    sequences = {"aggressor": 0, "victim-0": 0, "victim-1": 1}
    value.launches = [
        replace(
            item,
            submission_policy="common_epoch_dependency_burst",
            submission_sequence_index=sequences[item.launch_id],
            client_release_offset_ns=0,
            source_host_launch_interval={
                "start_ns": 10 + index,
                "end_ns": 11 + index,
            },
            # Distant source starts must never schedule Driver issue.
            source_gpu_interval={
                "start_ns": index * 10_000_000,
                "end_ns": index * 10_000_000 + 1,
            },
        )
        for index, item in enumerate(value.launches)
    ]
    value.schema = "kernel_capsule_v3"
    value.submission_policy = "common_epoch_dependency_burst"
    value.client_release_offsets_ns = {"a": 0, "v": 0}
    value.victim_head_launch_id = "victim-1"
    return value


class V3Backend(FakeBackend):
    """Return complete Driver issue and CUPTI activity evidence."""

    def __init__(self, reproduce=True):
        super().__init__(reproduce=reproduce)
        self.capsule = v3_capsule()

    def capture_one_instrumented_frame(self, specification):
        assert specification["schema"] == "kernel_capsule_capture_spec_v3"
        assert specification["submission_policy"] == (
            "common_epoch_dependency_burst"
        )
        assert specification["client_release_offsets_ns"] == {
            "a": 0, "v": 0,
        }
        assert specification[
            "source_gpu_start_offsets_control_submission"
        ] is False
        return self.capsule

    def capabilities(self):
        return {
            name: {"available": True, "version": "test"}
            for name in (
                "cupti_activity",
                "cuda_launch_capture",
                "cupti_checkpoint",
            )
        }

    def replay(self, captured, variant, instrumentation):
        self.replay_calls.append((variant, instrumentation))
        pair_wait = 1_000_000 if self.reproduce else 0
        activities = [
            {
                "launch_id": "victim-0",
                "driver_issue_start_ns": 100,
                "driver_issue_end_ns": 110,
                "gpu_start_ns": 120,
                "gpu_end_ns": 220,
            },
            {
                "launch_id": "victim-1",
                "driver_issue_start_ns": 111,
                "driver_issue_end_ns": 120,
                "gpu_start_ns": 220 + (pair_wait if variant == "pair" else 0),
                "gpu_end_ns": 320 + (pair_wait if variant == "pair" else 0),
            },
        ]
        launch_order = ["victim-0", "victim-1"]
        if variant == "pair":
            activities.append(
                {
                    "launch_id": "aggressor",
                    "driver_issue_start_ns": 100,
                    "driver_issue_end_ns": 110,
                    "gpu_start_ns": 115,
                    "gpu_end_ns": 2_000_115,
                }
            )
            launch_order.insert(0, "aggressor")
        return {
            "victim_order": list(captured.victim_launch_ids),
            "launch_order": launch_order,
            "launch_fingerprints": {
                launch_id: captured.launch_map()[
                    launch_id
                ].driver_command_fingerprint
                for launch_id in launch_order
            },
            "launch_activities": activities,
            "fingerprint_match": True,
            "checkpoint_state_sha256": "c" * 64,
            "allocation_map_sha256": "a" * 64,
            "output_sha256": ("p" if variant == "pair" else "v") * 64,
            "foreign_work_count": 0,
            "non_capsule_cuda_work_count": 0,
            "frame_replay_count": 0,
            "model_forward_count": 0,
            "rosbag_work_count": 0,
            "profiler_perturbation": instrumentation,
            "source_gpu_offsets_used_for_submission": False,
        }


class StubPlannerV3(KernelReplayPlannerV3):
    """Avoid source and MCAP access in v3 campaign tests."""

    selected_target = target()

    def select_target(self):
        value = self.selected_target
        (self.output_directory / "target_selection.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
        return value

    def prepare_capture_input(self, _target):
        payload = self.output_directory / "capture_input.bin"
        manifest = self.output_directory / "capture_input.json"
        payload.write_bytes(b"")
        value = {
            "schema": "kernel_capsule_input_v2",
            "payload_path": str(payload),
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        return value


def test_capsule_rejects_pointer_without_allocation_offset():
    """Direct pointers always carry allocation identity and byte offset."""
    broken = launch("x", "a", "kernel")
    object.__setattr__(
        broken,
        "parameters",
        (ParameterValue(0, 0, 8, "00" * 8, "allocation", None),),
    )
    with pytest.raises(CapsuleInvalid, match="pointer allocation"):
        broken.validate()


def test_v1_manifest_is_analysis_readable_but_not_replayable(tmp_path):
    """Legacy evidence remains inspectable and cannot enter v2 replay."""
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"schema": "kernel_capsule_v1"}))
    assert read_capsule_manifest(path)["schema"] == "kernel_capsule_v1"
    with pytest.raises(CapsuleInvalid, match="not a v2 replay input"):
        capsule_from_dict({"schema": "kernel_capsule_v1"})


def test_v2_manifest_is_analysis_readable_but_not_v3_replayable(tmp_path):
    """The flawed v2 schedule stays available only to offline analysis."""
    path = tmp_path / "v2.json"
    path.write_text(json.dumps({"schema": "kernel_capsule_v2"}))
    assert read_capsule_manifest(path)["schema"] == "kernel_capsule_v2"
    with pytest.raises(CapsuleInvalid, match="not a v3 replay input"):
        capsule_v3_from_dict({"schema": "kernel_capsule_v2"})


def test_runtime_api_cannot_be_a_v2_replay_command():
    """The outer Runtime call is provenance and never a replay target."""
    value = launch("x", "a", "kernel")
    value = replace(
        value,
        launch_api="cudaLaunchKernel_v7000",
        captured_driver_api="cudaLaunchKernel_v7000",
        replay_api="cudaLaunchKernel_v7000",
    )
    value = replace(value, driver_command_fingerprint=value.command_digest)
    with pytest.raises(CapsuleInvalid, match="unsupported Driver API"):
        value.validate()


def test_packed_parameter_bytes_participate_in_driver_fingerprint():
    """Packed Driver buffers are retained independently of Runtime args."""
    value = launch("x", "a", "kernel")
    value = replace(
        value,
        parameter_mode="packed_extra",
        packed_parameter_bytes_hex="01020304",
    )
    value = replace(value, driver_command_fingerprint=value.command_digest)
    value.validate()
    changed = replace(value, packed_parameter_bytes_hex="01020305")
    with pytest.raises(CapsuleInvalid, match="fingerprint mismatch"):
        changed.validate()


def test_runtime_driver_correlation_and_extended_attributes_are_exact():
    """Nested correlation and extended attributes are mandatory evidence."""
    value = launch("x", "a", "kernel")
    missing = replace(value, nested_driver_correlation_id=None)
    with pytest.raises(CapsuleInvalid, match="Driver correlation missing"):
        missing.validate()
    extended = replace(
        value,
        launch_api="cuLaunchKernelEx",
        captured_driver_api="cuLaunchKernelEx",
        replay_api="cuLaunchKernelEx",
        launch_attributes=({"id": 8, "value_hex": "00" * 16},),
    )
    extended = replace(
        extended, driver_command_fingerprint=extended.command_digest
    )
    extended.validate()
    changed = replace(
        extended,
        launch_attributes=({"id": 8, "value_hex": "01" + "00" * 15},),
    )
    with pytest.raises(CapsuleInvalid, match="fingerprint mismatch"):
        changed.validate()


def test_host_coordinator_requires_two_clients_and_gates_optional_backends():
    """Collector availability is the intersection of both live clients."""
    with pytest.raises(ValueError, match="at least two"):
        KernelCapsuleCoordinator([FakeAgentEndpoint("a", True)])
    coordinator = KernelCapsuleCoordinator([
        FakeAgentEndpoint("a", True), FakeAgentEndpoint("v", False),
    ])
    collectors = coordinator.capabilities()["collectors"]
    assert collectors["cupti_activity"]["available"] is True
    assert collectors["nvbit_cta"]["available"] is False
    three = KernelCapsuleCoordinator([
        FakeAgentEndpoint("a", True),
        FakeAgentEndpoint("v", True),
        FakeAgentEndpoint("observer", True),
    ])
    assert three.capabilities()["collectors"]["cupti_activity"][
        "available"
    ] is True


def test_custom_cuda_and_cudnn_launch_fingerprints_are_exact():
    """Driver function, launch variant, and parameter bytes stay exact."""
    captured = capsule()
    custom = replace(
        captured.launches[0], symbol="custom_cuda_kernel",
        symbol_sha256=hashlib.sha256(b"custom_cuda_kernel").hexdigest(),
    )
    cudnn = replace(
        captured.launches[1], symbol="cudnn_implicit_gemm",
        symbol_sha256=hashlib.sha256(b"cudnn_implicit_gemm").hexdigest(),
        launch_api="cuLaunchKernelEx",
        captured_driver_api="cuLaunchKernelEx",
        replay_api="cuLaunchKernelEx",
    )
    custom = replace(
        custom, driver_command_fingerprint=custom.command_digest
    )
    cudnn = replace(cudnn, driver_command_fingerprint=cudnn.command_digest)
    captured.launches[0] = custom
    captured.launches[1] = cudnn
    captured.verify_observed_launches(list(captured.launches))
    changed = replace(
        cudnn,
        parameters=(ParameterValue(0, 0, 8, "0200000000000000"),),
    )
    changed = replace(
        changed, driver_command_fingerprint=changed.command_digest
    )
    observed = [custom, changed, captured.launches[2]]
    with pytest.raises(CapsuleInvalid, match="fingerprint mismatch"):
        captured.verify_observed_launches(observed)


def test_fixed_phases_restore_each_time_and_never_rerun_frame(tmp_path):
    """Three fixed repetitions differ only by aggressor suppression."""
    backend = FakeBackend()
    planner = StubPlanner(tmp_path / "source", tmp_path / "out", backend)
    result = planner.run()
    assert result["replay_integrity_gate"]["status"] == "passed"
    assert backend.restore_calls == 15  # 6 timing + 3 CTA + 3 PM + 3 NCU
    assert backend.replay_calls[:6] == [
        ("pair", "none"),
        ("victim_only", "none"),
        ("victim_only", "none"),
        ("pair", "none"),
        ("pair", "none"),
        ("victim_only", "none"),
    ]
    assert set(backend.resource_calls) == {"aggressor", "victim-0", "victim-1"}
    assert (tmp_path / "out" / "gpu_forensics.sqlite").is_file()
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        timing = db.execute(
            "SELECT variant, victim_order_json, raw_json "
            "FROM replay_iteration "
            "WHERE phase='timing' ORDER BY iteration_id"
        ).fetchall()
    assert len(timing) == 6
    pair = json.loads(timing[0][2])["launch_order"]
    victim = json.loads(timing[1][2])["launch_order"]
    assert pair[1:] == victim
    assert pair[0] == "aggressor"


def test_contention_similarity_does_not_control_replay_success(tmp_path):
    """Zero admission excess remains an observation, not a rejection."""
    backend = FakeBackend(reproduce=False)
    planner = StubPlanner(tmp_path / "source", tmp_path / "out", backend)
    result = planner.run()
    gate = result["replay_integrity_gate"]
    assert gate["status"] == "passed"
    assert gate["observed_pair_minus_victim_only_admission_ns"] == [0, 0, 0]
    assert backend.restore_calls == 15
    assert set(backend.resource_calls) == {
        "aggressor", "victim-0", "victim-1",
    }


def test_v3_capture_spec_uses_zero_offset_dependency_burst():
    """Source GPU offsets are retained as provenance, never a release input."""
    specification = capture_specification_v3(target())
    assert specification["submission_policy"] == (
        "common_epoch_dependency_burst"
    )
    assert specification["client_release_offsets_ns"] == {"a": 0, "v": 0}
    assert specification[
        "source_gpu_start_offsets_control_submission"
    ] is False
    assert "relative_offsets" not in specification["capture"]


def test_v3_capture_spec_accepts_signed_client_release_offsets():
    """Offsets remain explicit controlled inputs while zero stays default."""
    specification = capture_specification_v3(
        target(), {"a": -25, "v": 40}
    )
    assert specification["client_release_offsets_ns"] == {"a": -25, "v": 40}


def test_v3_chain_issue_and_gpu_completion_readiness(tmp_path):
    """A queued same-stream tail is ready only after predecessor completion."""
    backend = V3Backend()
    planner = StubPlannerV3(tmp_path / "source", tmp_path / "out", backend)
    result = planner.run()
    assert result["replay_integrity_gate"]["status"] == "passed"
    assert backend.restore_calls == 2
    assert backend.replay_calls == [
        ("pair", "none"), ("victim_only", "none")
    ]
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        rows = db.execute(
            "SELECT launch_id, driver_issue_start_ns, replay_ready_ns, "
            "admission_wait_ns FROM replay_launch_activity "
            "WHERE iteration_id=1 ORDER BY submission_ordinal"
        ).fetchall()
        schema_version = json.loads(db.execute(
            "SELECT value_json FROM metadata WHERE key='schema_version'"
        ).fetchone()[0])
    victim_tail = next(value for value in rows if value[0] == "victim-1")
    assert victim_tail[1] == 111  # no 10 ms source-GPU offset sleep
    assert victim_tail[2] == 220  # victim-0 GPU completion
    assert victim_tail[3] == 1_000_000
    assert schema_version == 4
    assert (tmp_path / "out" / "capsule_kernel_timing.csv").is_file()
    assert backend.resource_calls == ["aggressor", "victim-0", "victim-1"]


def test_v3_victim_only_suppresses_only_aggressor(tmp_path):
    """Matched victim-only evidence retains both prompt victim issues."""
    backend = V3Backend()
    planner = StubPlannerV3(tmp_path / "source", tmp_path / "out", backend)
    planner.run()
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        launch_ids = {
            value[0]
            for value in db.execute(
                "SELECT a.launch_id FROM replay_launch_activity a "
                "JOIN replay_iteration i USING(iteration_id) "
                "WHERE i.variant='victim_only'"
            )
        }
    assert launch_ids == {"victim-0", "victim-1"}


def test_v3_multistream_event_readiness_uses_source_completion(tmp_path):
    """Captured event ordering remains authoritative across victim streams."""
    backend = V3Backend()
    captured = backend.capsule
    event_tail = replace(
        captured.launches[2], stream_handle="stream-v-second"
    )
    captured.launches[2] = replace(
        event_tail, driver_command_fingerprint=event_tail.command_digest
    )
    captured.dependencies = [
        CapsuleDependency("victim-0", "victim-1", "event", "event-1")
    ]
    captured.event_operations = [{
        "event_handle": "event-1",
        "source_launch_id": "victim-0",
        "target_launch_id": "victim-1",
    }]
    captured.terminal_streams.append("stream-v-second")
    planner = StubPlannerV3(tmp_path / "source", tmp_path / "out", backend)
    result = planner.run()
    assert result["replay_integrity_gate"]["status"] == "passed"


class MissingActivityBackend(V3Backend):
    """Drop one required CUPTI correlation."""

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        record["launch_activities"].pop()
        return record


class NoOverlapBackend(V3Backend):
    """Produce internally valid replay with no pair overlap or excess."""

    def __init__(self):
        super().__init__(reproduce=False)

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        if variant == "pair":
            aggressor = next(
                value for value in record["launch_activities"]
                if value["launch_id"] == "aggressor"
            )
            aggressor["gpu_start_ns"] = 115
            aggressor["gpu_end_ns"] = 119
        return record


def test_v3_no_overlap_or_admission_excess_is_still_successful(tmp_path):
    """Contention similarity is never a replay acceptance condition."""
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", NoOverlapBackend()
    )
    result = planner.run()
    gate = result["replay_integrity_gate"]
    assert gate["status"] == "passed"
    assert gate["pair_minus_victim_only_admission_delay_ns"] == 0
    assert gate["observed_replay_overlap_relationships"] == ["no_overlap"]


def test_v3_launch_order_diagnostic_ignores_exact_offsets(tmp_path):
    """Order matches depend on sequence alone, not timestamp differences."""
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", V3Backend()
    )
    result = planner.run()
    gate = result["replay_integrity_gate"]
    assert gate["accepted_source_gpu_sequence"] == [
        "aggressor", "victim-0", "victim-1"
    ]
    assert gate["accepted_replay_gpu_sequence"] == [
        "aggressor", "victim-0", "victim-1"
    ]
    assert gate["sequence_match_status"] == "matched"
    header = (tmp_path / "out" / "capsule_timing.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "source_gpu_start_order" in header
    assert "replay_gpu_start_order" in header
    assert "gpu_start_order_match" in header


class RetryPairBackend(V3Backend):
    """Mismatch the first pair GPU order and match the second."""

    def __init__(self):
        super().__init__()
        self.pair_calls = 0

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        if variant == "pair" and instrumentation == "none":
            self.pair_calls += 1
            if self.pair_calls == 1:
                aggressor = next(
                    value for value in record["launch_activities"]
                    if value["launch_id"] == "aggressor"
                )
                aggressor["gpu_start_ns"] = 125
        return record


def test_v3_retries_pair_and_persists_only_first_match(tmp_path):
    """A rejected first pair is absent from every retained record set."""
    backend = RetryPairBackend()
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", backend
    )
    result = planner.run()
    assert backend.replay_calls == [
        ("pair", "none"),
        ("pair", "none"),
        ("victim_only", "none"),
    ]
    assert result["replay_integrity_gate"]["accepted_pair_attempt"] == 2
    assert len(planner.records) == 2
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        rows = db.execute(
            "SELECT variant, repetition FROM replay_iteration "
            "WHERE phase='timing' ORDER BY iteration_id"
        ).fetchall()
    assert rows == [("pair", 0), ("victim_only", 0)]


class AmbiguousPairBackend(V3Backend):
    """Return equal GPU starts for the pair on every attempt."""

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        if variant == "pair" and instrumentation == "none":
            aggressor = next(
                value for value in record["launch_activities"]
                if value["launch_id"] == "aggressor"
            )
            aggressor["gpu_start_ns"] = 120
        return record


def test_v3_ambiguous_gpu_starts_retry_then_fail(tmp_path):
    """Equal starts never acquire an order from interval ends or IDs."""
    backend = AmbiguousPairBackend()
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", backend
    )
    with pytest.raises(ReplayIntegrityError, match="capsule_invalid"):
        planner.run()
    assert backend.replay_calls == [("pair", "none")] * 3
    assert planner.records == []


class HostOrderMismatchBackend(V3Backend):
    """Change cross-client Driver issue order while preserving GPU order."""

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        if variant == "pair":
            aggressor = next(
                value for value in record["launch_activities"]
                if value["launch_id"] == "aggressor"
            )
            aggressor["driver_issue_start_ns"] = 112
            aggressor["driver_issue_end_ns"] = 113
        return record


def test_v3_host_order_mismatch_with_gpu_match_is_accepted(tmp_path):
    """Host issue order remains diagnostic and cannot reject a GPU match."""
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", HostOrderMismatchBackend()
    )
    result = planner.run()
    assert result["replay_integrity_gate"]["sequence_match_status"] == (
        "matched"
    )
    pair = next(
        value for value in planner.records if value["variant"] == "pair"
    )
    assert pair["host_launch_order_match"] is False
    assert pair["gpu_start_order_match"] is True


class VictimBaselineMismatchBackend(V3Backend):
    """Preserve the pair but reverse an independent victim-only GPU order."""

    def __init__(self):
        super().__init__()
        self.capsule.dependencies = []

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        if variant == "victim_only":
            head = next(
                value for value in record["launch_activities"]
                if value["launch_id"] == "victim-0"
            )
            head["gpu_start_ns"] = 300
            head["gpu_end_ns"] = 400
        return record


def test_v3_victim_only_gpu_order_mismatch_fails_without_timing_rows(tmp_path):
    """The filtered source victim sequence is an integrity requirement."""
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", VictimBaselineMismatchBackend()
    )
    with pytest.raises(ReplayIntegrityError, match="victim-only baseline"):
        planner.run()
    assert planner.records == []
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM replay_iteration WHERE phase='timing'"
        ).fetchone()[0] == 0


class OptionalSequenceBackend(V3Backend):
    """Expose CTA/metric collectors and mismatch only CTA evidence."""

    def capabilities(self):
        capabilities = super().capabilities()
        capabilities.update({
            "nvbit_cta": {"available": True, "version": "test"},
            "cupti_pm": {"available": True, "version": "test"},
        })
        return capabilities

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        if instrumentation == "nvbit_cta":
            aggressor = next(
                value for value in record["launch_activities"]
                if value["launch_id"] == "aggressor"
            )
            aggressor["gpu_start_ns"] = 125
            record["cta_intervals"] = [{
                "launch_id": "victim-0",
                "sm_id": 0,
                "cta_id": "0",
                "entry_ns": 10,
                "exit_ns": 20,
            }]
        if instrumentation == "cupti_pm":
            record["metric_samples"] = [{
                "timestamp_ns": 10,
                "metric": "sm_active",
                "value": 0.5,
            }]
        return record


def test_v3_optional_collectors_run_once_and_mismatch_is_omitted(tmp_path):
    """Optional sequence drift reduces capability without retaining data."""
    backend = OptionalSequenceBackend()
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", backend
    )
    planner.run()
    assert backend.replay_calls.count(("pair", "nvbit_cta")) == 1
    assert backend.replay_calls.count(("pair", "cupti_pm")) == 1
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        phases = db.execute(
            "SELECT phase, repetition FROM replay_iteration "
            "ORDER BY iteration_id"
        ).fetchall()
        capability = db.execute(
            "SELECT available, reason FROM capability "
            "WHERE collector='nvbit_cta'"
        ).fetchone()
        metric_count = db.execute(
            "SELECT COUNT(*) FROM metric_sample"
        ).fetchone()[0]
    assert phases == [("timing", 0), ("timing", 0), ("device_metric", 0)]
    assert capability[0] == 0
    assert "sequence mismatched" in capability[1]
    assert metric_count == 1


class CompleteCtaBackend(V3Backend):
    """Return one complete CTA row for each captured one-CTA launch."""

    def capabilities(self):
        capabilities = super().capabilities()
        capabilities["nvbit_cta"] = {
            "available": True, "version": "test",
        }
        return capabilities

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        if instrumentation != "nvbit_cta":
            return record
        intervals = {
            "aggressor": (116, 210),
            "victim-0": (121, 180),
            "victim-1": (1_000_221, 1_000_280),
        }
        record["cta_intervals"] = [
            {
                "launch_id": launch_id,
                "client_id": captured.launch_map()[launch_id].client_id,
                "cta_id": 0,
                "sm_id": 3,
                "entry_ns": entry,
                "exit_ns": exit_,
                "clock_error_ns": 2,
                "entry_observed": True,
                "exit_observed": True,
                "observation_status": "complete",
            }
            for launch_id, (entry, exit_) in intervals.items()
        ]
        record["cta_collection_status"] = [
            {
                "launch_id": launch_id,
                "expected_count": 1,
                "entered_count": 1,
                "exited_count": 1,
                "complete_count": 1,
                "missing_count": 0,
                "dropped_entry_count": 0,
                "dropped_exit_count": 0,
                "coverage_fraction": 1.0,
                "quality": "complete",
            }
            for launch_id in intervals
        ]
        record["clock_calibrations"] = [
            {
                "client_id": client_id,
                "clock_a": "globaltimer",
                "clock_b": "cupti_activity",
                "offset_ns": 5,
                "error_ns": 2,
                "precision_class": "bounded_profiler_calibration",
            }
            for client_id in ("a", "v")
        ]
        record["cta_collection_valid"] = True
        return record


class PartialCtaBackend(CompleteCtaBackend):
    """Retain an entered CTA whose exit boundary was not observed."""

    def replay(self, captured, variant, instrumentation):
        record = super().replay(captured, variant, instrumentation)
        if instrumentation != "nvbit_cta":
            return record
        row = next(
            value for value in record["cta_intervals"]
            if value["launch_id"] == "victim-0"
        )
        row.update({
            "exit_ns": None,
            "exit_observed": False,
            "observation_status": "entry_only",
        })
        status = next(
            value for value in record["cta_collection_status"]
            if value["launch_id"] == "victim-0"
        )
        status.update({
            "exited_count": 0,
            "complete_count": 0,
            "missing_count": 1,
            "dropped_exit_count": 1,
            "coverage_fraction": 0.0,
            "quality": "partial",
        })
        return record


def test_schema_v3_cta_views_and_resource_enrichment(tmp_path):
    """Complete CTA evidence populates canonical timelines and overlaps."""
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", CompleteCtaBackend()
    )
    planner.run()
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM cta_interval"
        ).fetchone()[0] == 3
        assert db.execute(
            "SELECT COUNT(DISTINCT launch_id || ':' || cta_id) "
            "FROM cta_interval"
        ).fetchone()[0] == 3
        assert db.execute(
            "SELECT MIN(coverage_fraction), MAX(coverage_fraction) "
            "FROM cta_kernel_summary"
        ).fetchone() == (1.0, 1.0)
        fractions = db.execute(
            "SELECT MAX(entered_fraction) FROM cta_dispatch_timeline "
            "GROUP BY kernel_instance_id"
        ).fetchall()
        assert fractions == [(1.0,), (1.0,), (1.0,)]
        assert db.execute(
            "SELECT COUNT(*) FROM cta_residency WHERE duration_ns >= 0"
        ).fetchone()[0] == 3
        assert db.execute(
            "SELECT COUNT(*) FROM cta_sm_timeline"
        ).fetchone()[0] == 3
        assert db.execute(
            "SELECT COUNT(*) FROM kernel_co_residency "
            "WHERE overlap_duration_ns > 0"
        ).fetchone()[0] >= 1
        resources = db.execute(
            "SELECT COUNT(*), MIN(registers_per_thread), "
            "MIN(threads_per_block) FROM kernel_instance"
        ).fetchone()
        assert resources == (8, 16, 32)
        assert db.execute(
            "SELECT COUNT(*) FROM clock_calibration"
        ).fetchone()[0] == 2
        forbidden = {"warp", "thread", "lane", "instruction"}
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert not any(
            name in table for table in tables for name in forbidden
        )
    header = (tmp_path / "out" / "capsule_cta_placement.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "kernel_instance_id" in header
    assert "coverage_fraction" in header
    assert "duration_ns" in header
    assert "observation_status" in header


def test_partial_cta_boundaries_are_retained_with_coverage(tmp_path):
    """A missing exit remains queryable but is excluded from residency."""
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", PartialCtaBackend()
    )
    planner.run()
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        partial = db.execute(
            "SELECT entry_ns, exit_ns, observation_status "
            "FROM cta_interval WHERE launch_id='victim-0'"
        ).fetchone()
        assert partial == (121, None, "entry_only")
        status = db.execute(
            "SELECT expected_count, entered_count, exited_count, "
            "complete_count, missing_count, dropped_exit_count, "
            "coverage_fraction FROM cta_collection_status "
            "WHERE launch_id='victim-0'"
        ).fetchone()
        assert status == (1, 1, 0, 0, 1, 1, 0.0)
        assert db.execute(
            "SELECT COUNT(*) FROM cta_residency "
            "WHERE launch_id='victim-0'"
        ).fetchone()[0] == 0


class PriorityEffectBackend(V3Backend):
    """Model a launch-priority intervention that delays victim admission."""

    def __init__(self):
        super().__init__()
        aggressor = self.capsule.launch_map()["aggressor"]
        aggressor = replace(
            aggressor,
            launch_api="cuLaunchKernelEx",
            captured_driver_api="cuLaunchKernelEx",
            replay_api="cuLaunchKernelEx",
        )
        aggressor = replace(
            aggressor,
            driver_command_fingerprint=aggressor.command_digest,
        )
        self.capsule.launches = [
            aggressor if value.launch_id == "aggressor" else value
            for value in self.capsule.launches
        ]

    def aggressor_priority_range(self, _captured):
        return {
            "supported": True,
            "least_priority": 0,
            "greatest_priority": -1,
            "captured_priority": 0,
            "aggressor_launch_api": "cuLaunchKernelEx",
        }

    def replay_with_aggressor_priority(self, captured, priority):
        record = super().replay(
            captured, "pair", "launch_priority_counterfactual"
        )
        if priority == -1:
            tail = next(
                value for value in record["launch_activities"]
                if value["launch_id"] == "victim-1"
            )
            tail["gpu_start_ns"] += 200_000
            tail["gpu_end_ns"] += 200_000
        record["priority_attribute_applied"] = True
        record["aggressor_priority"] = priority
        return record


def test_v3_priority_counterfactual_detects_victim_change(tmp_path):
    """High/low launch priority is paired against sham and victim-only."""
    backend = PriorityEffectBackend()
    planner = StubPlannerV3(
        tmp_path / "source",
        tmp_path / "out",
        backend,
        profiler_options={
            "priority_counterfactual_enabled": True,
            "priority_counterfactual_trials": 3,
            "priority_min_effect_ns": 1,
        },
    )
    result = planner.run()
    counterfactual = result["priority_counterfactual"]
    assert counterfactual["status"] == "supported"
    assert counterfactual["trial_count"] == 3
    assert counterfactual[
        "gpu_start_sequence_is_outcome_not_acceptance_gate"
    ] is True
    assert counterfactual["metric_findings"]["victim_delay_ns"][
        "status"
    ] == "supported"
    assert len(counterfactual["samples"]["greatest_priority"]) == 3
    assert len(counterfactual["samples"]["least_priority"]) == 3
    assert all(
        value["priority_attribute_applied"]
        for condition in ("sham", "greatest_priority", "least_priority")
        for value in counterfactual["samples"][condition]
    )
    assert (tmp_path / "out" / "priority_counterfactual.json").is_file()
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        rows = db.execute(
            "SELECT COUNT(*) FROM replay_iteration "
            "WHERE phase='priority_counterfactual'"
        ).fetchone()[0]
    assert rows == 15


class NoPriorityRangeBackend(PriorityEffectBackend):
    """Expose no meaningful per-launch priority range."""

    def aggressor_priority_range(self, _captured):
        return {
            "supported": False,
            "least_priority": 0,
            "greatest_priority": 0,
            "captured_priority": 0,
            "aggressor_launch_api": "cuLaunchKernelEx",
        }


def test_v3_priority_counterfactual_reports_not_observable(tmp_path):
    """Unsupported priority does not fail or add intervention iterations."""
    planner = StubPlannerV3(
        tmp_path / "source",
        tmp_path / "out",
        NoPriorityRangeBackend(),
        profiler_options={"priority_counterfactual_enabled": True},
    )
    result = planner.run()
    assert result["priority_counterfactual"]["status"] == "not_observable"
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        rows = db.execute(
            "SELECT COUNT(*) FROM replay_iteration "
            "WHERE phase='priority_counterfactual'"
        ).fetchone()[0]
    assert rows == 0


class GpuStartOrderMismatchBackend(V3Backend):
    """Capture victim-first GPU order but replay aggressor-first order."""

    def __init__(self):
        super().__init__()
        source_gpu_intervals = {
            "victim-0": {"start_ns": 100, "end_ns": 120},
            "aggressor": {"start_ns": 110, "end_ns": 1_000},
            "victim-1": {"start_ns": 900, "end_ns": 1_100},
        }
        self.capsule.launches = [
            replace(
                item,
                source_gpu_interval=source_gpu_intervals[item.launch_id],
            )
            for item in self.capsule.launches
        ]


def test_v3_all_gpu_start_order_mismatches_are_rejected_atomically(tmp_path):
    """Three GPU-order mismatches leave no timing evidence behind."""
    planner = StubPlannerV3(
        tmp_path / "source",
        tmp_path / "out",
        GpuStartOrderMismatchBackend(),
    )
    with pytest.raises(ReplayIntegrityError, match="capsule_invalid"):
        planner.run()
    assert planner.records == []
    with sqlite3.connect(
        str(tmp_path / "out" / "capsule_replay.sqlite")
    ) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM replay_iteration WHERE phase='timing'"
        ).fetchone()[0] == 0


def test_v3_missing_activity_is_rejected(tmp_path):
    """Every selected Driver command requires one complete GPU interval."""
    planner = StubPlannerV3(
        tmp_path / "source", tmp_path / "out", MissingActivityBackend()
    )
    with pytest.raises(
        ReplayIntegrityError, match="complete per-launch activity"
    ):
        planner.run()


def test_synthetic_matrix_fixes_three_trials_and_two_of_three_policy():
    """Every controlled stage condition uses the same acceptance policy."""
    conditions = validation_matrix(8)
    assert {condition.stage for condition in conditions} == {1, 2, 3, 4}
    assert all(condition.trials == 3 for condition in conditions)
    condition = conditions[0]
    observed = [
        {"status": status, "direct_wdu_observation": False}
        for status in ("supported", "supported", "inconclusive")
    ]
    assert validate_trials(condition, observed)["validated"] is True
    observed[0]["direct_wdu_observation"] = True
    assert validate_trials(condition, observed)["validated"] is False
