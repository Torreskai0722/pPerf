"""Tests for replay, identity, decoding, and adapter routing."""

from types import SimpleNamespace
import hashlib
import os
import signal
import subprocess

import numpy as np
import pytest

from closeloop_testbed.adapters import (
    InferencerAdapter, adapter_spec, create_inferencer, decode_image,
    decode_pointcloud2, override_resize_scale, select_point_features,
)
from closeloop_testbed.capsule_model_client import decode_dataset_input
from closeloop_testbed.replayer import (IdentityTracker, discover_bags,
                                        playback_segments,
                                        play_ordered_segments,
                                        replay_command, wait_for_player)
from closeloop_testbed.process_env import (
    fma_environment, fma_worker_environment, model_environment,
)
from closeloop_testbed.capsule_helpers import select_cta_target
from closeloop_testbed.input_handling import configured_input_queue_depth
from closeloop_testbed.model_execution import update_profile_evidence


def test_bag_discovery_and_command(tmp_path):
    """Matching MCAP files sort and command remappings are stable."""
    second = tmp_path / "scene-x-2.mcap"
    first = tmp_path / "scene-x-1.mcap"
    second.touch()
    first.touch()
    bags = discover_bags(str(tmp_path), "scene-x")
    assert bags == [first, second]
    command = replay_command([first], 1.5, ["/a"], {"/a": "/b"})
    assert command[:4] == ["ros2", "bag", "play", str(first)]
    assert "--start-paused" in command
    assert "--disable-keyboard-controls" in command
    assert command[-2:] == ["--remap", "/a:=/b"]


def test_two_scenes_play_sequentially_without_overlap(tmp_path):
    """Every bag in scene A completes before scene B starts."""
    metadata = tmp_path / "scene.json"
    metadata.write_text(
        '[{"token":"a","name":"scene-a"},'
        '{"token":"b","name":"scene-b"}]',
        encoding="utf-8",
    )
    for name in ("scene-a-1.mcap", "scene-a-2.mcap", "scene-b-1.mcap"):
        (tmp_path / name).touch()
    segments = playback_segments({
        "scene_token": "a",
        "scene_tokens": ["a", "b"],
        "metadata_path": str(metadata),
        "bag_directory": str(tmp_path),
    })
    clock = iter(range(6))
    active = []

    def play_one(segment):
        assert not active
        active.append(segment["segment_id"])
        start = next(clock)
        active.pop()
        return {
            **segment,
            "resume_monotonic_ns": start,
            "end_monotonic_ns": next(clock),
            "completion_status": "completed",
        }

    intervals = play_ordered_segments(segments, play_one)
    assert [item["scene_token"] for item in intervals] == ["a", "a", "b"]
    assert all(
        previous["end_monotonic_ns"] <= current["resume_monotonic_ns"]
        for previous, current in zip(intervals, intervals[1:])
    )


def test_exact_identity_tracking():
    """Unknown and repeated model messages do not satisfy coordination."""
    tracker = IdentityTracker(["a", "b"])
    assert not tracker.mark_ready("unknown")
    assert not tracker.mark_ready("a")
    assert tracker.mark_ready("b")
    assert not tracker.acknowledge("a")
    assert tracker.acknowledge("b")


def test_player_wait_distinguishes_completion_and_partial_cutoff():
    """A bounded wait interrupts the player and reports an expected cutoff."""

    class Process:
        """Minimal subprocess stand-in with controllable timeout behavior."""

        def __init__(self, times_out):
            self.returncode = None
            self.times_out = times_out
            self.signals = []

        def poll(self):
            """Return the current process state."""
            return self.returncode

        def wait(self, timeout=None):
            """Complete normally or time out until signaled."""
            if self.times_out and not self.signals:
                raise subprocess.TimeoutExpired(["ros2", "bag"], timeout)
            self.returncode = 0 if not self.signals else -self.signals[-1]
            return self.returncode

        def send_signal(self, selected):
            """Record the graceful cutoff signal."""
            self.signals.append(selected)

        def terminate(self):
            """Record a fallback termination."""
            self.send_signal(signal.SIGTERM)

        def kill(self):
            """Record a final forced termination."""
            self.send_signal(signal.SIGKILL)

    completed = Process(False)
    assert wait_for_player(completed, 5.0) == (0, False)

    partial = Process(True)
    returncode, cutoff = wait_for_player(partial, 0.25)
    assert cutoff
    assert returncode == -signal.SIGINT
    assert partial.signals == [signal.SIGINT]


def test_model_input_queue_depth_is_configurable():
    """Input history keeps legacy depth unless model overrides it."""
    assert configured_input_queue_depth({}) == 10
    assert configured_input_queue_depth({"input_queue_depth": 1}) == 1


def test_decoders_and_adapter():
    """Image and point messages become in-memory arrays."""
    image = SimpleNamespace(encoding="rgb8", height=1, width=2,
                            data=bytes([1, 2, 3, 4, 5, 6]))
    assert decode_image(image).shape == (1, 2, 3)
    fields = [SimpleNamespace(name=name, datatype=7, offset=index * 4)
              for index, name in enumerate(
                  ("x", "y", "z", "intensity", "ring"))]
    values = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    cloud = SimpleNamespace(fields=fields, is_bigendian=False, point_step=20,
                            data=values.tobytes(), width=1, height=1)
    np.testing.assert_allclose(decode_pointcloud2(cloud), values)
    np.testing.assert_allclose(select_point_features(values, 4), values[:, :4])
    with pytest.raises(ValueError, match="expected at least 6 features"):
        select_point_features(values, 6)
    assert adapter_spec("detection", "image").class_name == "DetInferencer"
    raw = SimpleNamespace(model="model",
                          preprocess=lambda inputs, batch_size: iter(
                              [(inputs, {"inputs": "batch"})]))
    assert InferencerAdapter(raw).preprocess("input") == {"inputs": "batch"}

    captured = []

    def lidar_preprocess(inputs, batch_size):
        captured.extend(inputs)
        return iter([{"inputs": "points"}])

    lidar_raw = SimpleNamespace(model="model", preprocess=lidar_preprocess)
    assert InferencerAdapter(lidar_raw, "lidar").preprocess("warmup.bin") == {
        "inputs": "points"}
    assert captured == [{"points": "warmup.bin"}]

    paired_raw = SimpleNamespace(
        model="model",
        preprocess=lambda inputs, batch_size: iter(
            [[list(inputs), {"inputs": "image"}]]))
    assert InferencerAdapter(paired_raw).preprocess("warmup.jpg") == {
        "inputs": "image"}


def test_dataset_payload_decoders():
    """Capsule inputs decode directly without ROS serialization support."""
    import cv2

    image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    np.testing.assert_array_equal(
        decode_dataset_input(
            encoded.tobytes(), "nuscenes_compressed_image"
        ),
        image,
    )
    points = np.array([[1, 2, 3, 4, 5]], dtype="<f4")
    np.testing.assert_allclose(
        decode_dataset_input(
            points.tobytes(), "nuscenes_pointcloud_xyzir_f32"
        ),
        points,
    )
    with pytest.raises(ValueError, match="invalid shape"):
        decode_dataset_input(
            b"not-a-point", "nuscenes_pointcloud_xyzir_f32"
        )


def test_override_resize_scale_updates_single_transform():
    """An explicit scale changes the model inferencer resize transform."""
    resize = SimpleNamespace(scale=(2048, 1024))
    inferencer = SimpleNamespace(
        pipeline=SimpleNamespace(
            transforms=[SimpleNamespace(), resize, SimpleNamespace()]
        )
    )

    selected = override_resize_scale(inferencer, [1024, 512])

    assert selected == (1024, 512)
    assert resize.scale == (1024, 512)


def test_override_resize_scale_requires_one_resize_transform():
    """Ambiguous pipeline overrides fail instead of silently changing it."""
    inferencer = SimpleNamespace(
        pipeline=SimpleNamespace(
            transforms=[
                SimpleNamespace(scale=(1, 1)),
                SimpleNamespace(scale=(2, 2)),
            ]
        )
    )

    with pytest.raises(ValueError, match="exactly one"):
        override_resize_scale(inferencer, [1024, 512])


def test_exact_checkpoint_uses_trusted_torch26_load_only_while_loading(
        tmp_path, monkeypatch):
    """A hash-pinned checkpoint opts out of weights-only then restores env."""
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"trusted")

    class FakeInferencer:
        """Capture the checkpoint policy seen by MMEngine construction."""

        def __init__(self, **arguments):
            assert arguments["weights"] == str(checkpoint)
            assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
            self.model = object()
            self.pipeline = SimpleNamespace(transforms=[])

    monkeypatch.setattr(
        "closeloop_testbed.adapters.importlib.import_module",
        lambda _name: SimpleNamespace(DetInferencer=FakeInferencer),
    )
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    create_inferencer(
        "detection", "image", "alias", checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(b"trusted").hexdigest(),
    )
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


def test_mps_environment_is_model_scoped(tmp_path):
    """Only callers creating model nodes receive the run-owned MPS settings."""
    config = {"gpu": {"mps_enabled": True, "index": 2}}
    model = {"mps_percentage": 35}
    environment = model_environment(config, model, tmp_path)
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "35"
    assert environment["CUDA_MPS_PIPE_DIRECTORY"].endswith("/p")
    config["gpu"]["mps_enabled"] = False
    assert model_environment(config, model, tmp_path) == {}


def test_nvbit_cta_environment_is_model_scoped(tmp_path):
    config = {"gpu": {"mps_enabled": False}}
    model = {
        "id": "victim",
        "nvbit_cta_profile": {
            "tracker_library": "/tracker.so",
            "kernel_name": "kernel",
            "grid": [2, 3, 4],
            "block": [128, 1, 1],
            "barrier_participants": ["victim", "aggressor"],
            "targets": [
                {"label": "affected", "scene_index": 1,
                 "message_order": 34, "kernel_occurrence": 1,
                 "align_launch": True},
                {"label": "unaffected", "scene_index": 1,
                 "message_order": 45, "kernel_occurrence": 1},
            ],
        },
    }
    environment = model_environment(config, model, tmp_path)
    assert environment["LD_PRELOAD"] == "/tracker.so"
    assert environment["PPERF_NVBIT_CTA_TARGET_GRID"] == "2,3,4"
    assert environment["PPERF_NVBIT_CTA_TARGET_OCCURRENCES"] == "1,1"
    assert environment["PPERF_NVBIT_CTA_ALIGN_LABELS"] == "affected"
    assert environment["PPERF_NVBIT_CTA_OUTPUT_PREFIX"].endswith(
        "/model_victim"
    )


def test_passive_nvbit_cta_environment_has_no_barrier(tmp_path):
    config = {"gpu": {"mps_enabled": False}}
    model = {
        "id": "victim",
        "nvbit_cta_profile": {
            "mode": "passive",
            "tracker_library": "/tracker.so",
            "record_capacity": 8,
            "targets": [
                {"label": "natural", "scene_index": 1,
                 "message_order": 34, "sequence_start": 3,
                 "sequence_end": 7},
            ],
        },
    }
    environment = model_environment(config, model, tmp_path)
    assert environment["PPERF_NVBIT_CTA_MODE"] == "passive"
    assert environment["PPERF_NVBIT_CTA_SEQUENCE_STARTS"] == "3"
    assert environment["PPERF_NVBIT_CTA_SEQUENCE_ENDS"] == "7"
    assert environment["PPERF_NVBIT_CTA_RECORD_CAPACITY"] == "8"
    assert "PPERF_NVBIT_CTA_BARRIER_PARTICIPANTS" not in environment


def test_paired_trial_target_and_mixed_tail_environment(tmp_path):
    targets = [{
        "label": "trial_0", "trial": 0, "kernel_occurrence": 0,
        "sequence_start": 265, "sequence_end": 290,
        "align_launch": True,
    }]
    assert select_cta_target(targets, trial=0) == targets[0]
    assert select_cta_target(targets, trial=1) is None
    config = {"gpu": {"mps_enabled": False}}
    model = {
        "id": "deeplabv3plus",
        "nvbit_cta_profile": {
            "mode": "mixed", "tracker_library": "/tracker.so",
            "kernel_name": "k0265", "grid": [4, 57, 8],
            "block": [128, 1, 1],
            "barrier_participants": ["faster_rcnn", "deeplabv3plus"],
            "targets": targets,
        },
    }
    environment = model_environment(config, model, tmp_path)
    assert environment["PPERF_NVBIT_CTA_MODE"] == "mixed"
    assert environment["PPERF_NVBIT_CTA_SEQUENCE_STARTS"] == "265"
    assert environment["PPERF_NVBIT_CTA_SEQUENCE_ENDS"] == "290"
    assert environment["PPERF_NVBIT_CTA_TARGET_OCCURRENCES"] == "0"
    assert environment["PPERF_NVBIT_CTA_ALIGN_LABELS"] == "trial_0"


def test_fma_sidecar_environment_observes_its_mps_cap(tmp_path):
    config = {
        "gpu": {"index": 2},
        "synthetic_fma": {
            "mps_percentage": 70,
            "cta": {"tracker_library": "/tracker.so",
                    "record_capacity": 4096},
        },
    }
    environment = fma_environment(config, tmp_path)
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "70"
    assert "LD_PRELOAD" not in environment
    assert environment["CUDA_MPS_PIPE_DIRECTORY"].endswith("/p")
    worker = fma_worker_environment(config, environment)
    assert worker["LD_PRELOAD"] == "/tracker.so"
    assert worker["PPERF_NVBIT_CTA_RECORD_CAPACITY"] == "4096"


def test_model_environment_applies_library_counts_before_import(tmp_path):
    """Native library controls are present when the model process starts."""
    config = {"gpu": {"mps_enabled": False, "index": 0}}
    model = {
        "library_thread_counts": {
            "openmp": 3,
            "mkl": 2,
            "openblas": 1,
            "numexpr": 2,
        },
    }

    environment = model_environment(config, model, tmp_path)

    assert environment == {
        "OMP_NUM_THREADS": "3",
        "OMP_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "2",
        "MKL_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "2",
    }


def test_status_profile_evidence_is_sorted():
    """Final model status receives the profiler's stable profile evidence."""
    profiler = SimpleNamespace(
        architecture_profile="profile-v1",
        architecture_profile_sha256="digest",
        observed_method_bindings=["model.predict"],
        observed_module_bindings=["roi_head", "backbone"],
    )
    status = {"state": "ready"}
    update_profile_evidence(status, profiler)
    assert status == {
        "architecture_profile": "profile-v1",
        "architecture_profile_sha256": "digest",
        "observed_method_bindings": ["model.predict"],
        "observed_module_bindings": ["backbone", "roi_head"],
        "profiler_id": "model_level1_v1",
        "state": "ready",
    }
