"""Tests for immutable run configuration validation."""

import hashlib

import pytest
import yaml

from closeloop_experiments.config import ConfigError, load_run_config


def valid_data(tmp_path):
    """Return a valid minimal configuration with real fixture paths."""
    (tmp_path / "bags").mkdir()
    (tmp_path / "scene.json").write_text("[]", encoding="utf-8")
    (tmp_path / "warmup.bin").write_bytes(b"x")
    return {
        "schema_version": 2,
        "run": {
            "id": "run-1",
            "experiment": "test",
            "provenance": "git:x",
            "timeout_seconds": 10,
        },
        "ros": {
            "distribution": "humble",
            "middleware": "rmw_fastrtps_cpp",
            "domain_id": 8,
            "launch_package": "closeloop_testbed",
            "launch_file": "testbed.launch.py",
        },
        "replay": {
            "scene_token": "token",
            "metadata_path": "scene.json",
            "bag_directory": "bags",
            "rate": 1.0,
            "topics": ["/lidar"],
            "remappings": {},
            "readiness_timeout_seconds": 2,
            "completion_timeout_seconds": 2,
        },
        "gpu": {"index": 0, "mps_enabled": False},
        "models": [
            {
                "id": "model",
                "node_name": "model_node",
                "task": "detection",
                "modality": "lidar",
                "mmlab_model": "model-config",
                "input_topic": "/lidar",
                "architecture_profile": "mmdet3d_voxel_two_stage_v1",
                "qos": "best_effort",
                "warmup_input": "warmup.bin",
                "warmup_count": 1,
                "launch_offset_seconds": 0,
                "module_annotation_depth": 2,
            }
        ],
        "recording": {
            "level": "level1",
            "scopes": ["model"],
            "nsys": {
                "version": "2025.3.1",
                "trace": ["cuda", "nvtx", "cudnn"],
                "sample": "none",
                "backtrace": "none",
                "gpu_context_switch": True,
            },
        },
    }


def write_config(tmp_path, data):
    """Serialize a test configuration."""
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_valid_resolves_paths_and_hashes_original(tmp_path):
    """Relative paths resolve beside config and hash covers original bytes."""
    path = write_config(tmp_path, valid_data(tmp_path))
    config = load_run_config(str(path), artifact_root=str(tmp_path / "artifacts"))
    assert config.data["replay"]["bag_directory"] == str(tmp_path / "bags")
    assert config.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config.run_directory == tmp_path / "artifacts" / "runs" / "run-1"


def test_ordered_scenes_preserve_primary_scene_token(tmp_path):
    """The legacy token remains the first token in an ordered scene list."""
    data = valid_data(tmp_path)
    data["replay"]["scene_tokens"] = ["token", "second"]
    assert load_run_config(str(write_config(tmp_path, data)))
    data["replay"]["scene_tokens"].reverse()
    with pytest.raises(ConfigError, match="first ordered scene token"):
        load_run_config(str(write_config(tmp_path, data)))


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update({"extra": True}),
        lambda data: data["models"].append(dict(data["models"][0])),
    ],
)
def test_unknown_and_duplicate_rejected(tmp_path, change):
    """Unknown fields and duplicate stable IDs are rejected."""
    data = valid_data(tmp_path)
    change(data)
    with pytest.raises(ConfigError):
        load_run_config(str(write_config(tmp_path, data)))


def test_mps_cross_fields(tmp_path):
    """MPS models require percentages and disable context tracing."""
    data = valid_data(tmp_path)
    data["gpu"]["mps_enabled"] = True
    with pytest.raises(ConfigError):
        load_run_config(str(write_config(tmp_path, data)))
    data["models"][0]["mps_percentage"] = 50
    data["recording"]["nsys"]["gpu_context_switch"] = False
    assert load_run_config(str(write_config(tmp_path, data)))


def test_unknown_profile_and_zero_warmup_are_rejected(tmp_path):
    """Profiles are known and warmup executes at least once."""
    data = valid_data(tmp_path)
    data["models"][0]["architecture_profile"] = "future-profile"
    with pytest.raises(ConfigError, match="unknown architecture profile"):
        load_run_config(str(write_config(tmp_path, data)))
    data["models"][0]["architecture_profile"] = "mmdet3d_voxel_two_stage_v1"
    data["models"][0]["warmup_count"] = 0
    with pytest.raises(ConfigError, match="minimum of 1"):
        load_run_config(str(write_config(tmp_path, data)))


def test_model_input_queue_depth_must_be_positive(tmp_path):
    """Model input history depth accepts one and rejects zero."""
    data = valid_data(tmp_path)
    data["models"][0]["input_queue_depth"] = 1
    assert load_run_config(str(write_config(tmp_path, data)))
    data["models"][0]["input_queue_depth"] = 0
    with pytest.raises(ConfigError, match="minimum of 1"):
        load_run_config(str(write_config(tmp_path, data)))


def test_exact_model_and_input_provenance_are_verified(tmp_path):
    """Pinned artifacts and condition manifests fail closed on hash drift."""
    data = valid_data(tmp_path)
    model_config = tmp_path / "model.py"
    checkpoint = tmp_path / "model.pth"
    dataset_manifest = tmp_path / "dataset.json"
    model_config.write_text("model = {}\n", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    data["models"][0].update({
        "model_config": model_config.name,
        "model_config_sha256": hashlib.sha256(
            model_config.read_bytes()
        ).hexdigest(),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
    })
    data["input_variation"] = {
        "condition_id": "clean-scene",
        "effect": "surrounding_environment",
        "source_scene_name": "scene-0001",
        "scene_factors": {
            "road_type": "urban", "traffic": "low",
            "object_density": "low", "complexity": "low",
            "lighting": "day",
        },
        "corruption": {
            "type": "clean", "severity": 0, "seed": 0,
            "dataset_manifest": dataset_manifest.name,
            "dataset_manifest_sha256": hashlib.sha256(
                dataset_manifest.read_bytes()
            ).hexdigest(),
        },
        "replicate": 1,
        "tenancy": "single",
    }
    assert load_run_config(str(write_config(tmp_path, data)))
    data["models"][0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="checkpoint hash mismatch"):
        load_run_config(str(write_config(tmp_path, data)))


def test_green_context_schema_and_execution_constraints(tmp_path):
    """Green Contexts require positive, distinct, non-MPS partitions."""
    data = valid_data(tmp_path)
    data["models"][0]["green_context"] = {
        "sm_count": 14,
        "partition_index": 0,
    }
    assert load_run_config(str(write_config(tmp_path, data)))

    data["models"][0]["green_context"]["sm_count"] = 0
    with pytest.raises(ConfigError, match="minimum of 1"):
        load_run_config(str(write_config(tmp_path, data)))
    data["models"][0]["green_context"]["sm_count"] = 14

    data["gpu"]["mps_enabled"] = True
    data["models"][0]["mps_percentage"] = 50
    data["recording"]["nsys"]["gpu_context_switch"] = False
    with pytest.raises(ConfigError, match="requires MPS to be disabled"):
        load_run_config(str(write_config(tmp_path, data)))

    data["gpu"]["mps_enabled"] = False
    data["models"][0].pop("mps_percentage")
    data["recording"]["nsys"]["gpu_context_switch"] = True
    data["models"][0]["paired_trial"] = {
        "direction": "solo",
        "role": "target",
        "trials": 5,
        "period_seconds": 1.0,
        "target_anchor_seconds": 0,
        "deadline_offset_seconds": 0,
        "launch_tolerance_seconds": 0.001,
        "k_replays": 0,
    }
    with pytest.raises(ConfigError, match="does not support paired"):
        load_run_config(str(write_config(tmp_path, data)))


def test_paired_trial_and_fixed_clocks_are_validated(tmp_path):
    """Paired timing and GPU clock values are recorded together."""
    data = valid_data(tmp_path)
    data["models"][0]["paired_trial"] = {
        "direction": "solo",
        "role": "target",
        "trials": 5,
        "period_seconds": 1.0,
        "target_anchor_seconds": 0.004,
        "deadline_offset_seconds": 0.004,
        "launch_tolerance_seconds": 0.0005,
        "k_replays": 0,
    }
    data["gpu"].update(
        {
            "graphics_clock_mhz": 2000,
            "memory_clock_mhz": 9000,
        }
    )
    assert load_run_config(str(write_config(tmp_path, data)))
    del data["gpu"]["memory_clock_mhz"]
    with pytest.raises(ConfigError, match="must be set together"):
        load_run_config(str(write_config(tmp_path, data)))


def test_inference_resize_scale_is_image_only(tmp_path):
    """Explicit inference resize has two dimensions and requires an image."""
    data = valid_data(tmp_path)
    data["models"][0]["inference_resize_scale"] = [1024, 512]
    with pytest.raises(ConfigError, match="cannot resize non-image"):
        load_run_config(str(write_config(tmp_path, data)))

    data["models"][0].update(
        {
            "modality": "image",
            "input_message_type": "compressed_image",
        }
    )
    assert load_run_config(str(write_config(tmp_path, data)))

    data["models"][0]["inference_resize_scale"] = [512]
    with pytest.raises(ConfigError, match="too short"):
        load_run_config(str(write_config(tmp_path, data)))


def test_partial_replay_requires_duration_and_full_rejects_it(tmp_path):
    """Partial playback has a duration and full is the default."""
    data = valid_data(tmp_path)
    assert load_run_config(str(write_config(tmp_path, data)))

    data["replay"]["playback_mode"] = "partial"
    with pytest.raises(ConfigError, match="duration_seconds is required"):
        load_run_config(str(write_config(tmp_path, data)))

    data["replay"]["duration_seconds"] = 5.0
    assert load_run_config(str(write_config(tmp_path, data)))

    data["replay"]["playback_mode"] = "full"
    with pytest.raises(ConfigError, match="only valid for partial"):
        load_run_config(str(write_config(tmp_path, data)))


def test_cpu_assignments_are_independent_disjoint_and_sized(tmp_path):
    """CPU controls are independent and configured CPU sets cannot overlap."""
    data = valid_data(tmp_path)
    data["replay"].update(
        {
            "cpu_affinity": [4, 5],
            "cpu_thread_count": 2,
        }
    )
    data["models"][0].update(
        {
            "cpu_affinity": [0, 1, 2, 3],
            "cpu_thread_count": 2,
        }
    )
    assert load_run_config(str(write_config(tmp_path, data)))

    data["models"][0]["cpu_affinity"] = [0, 1, 2, 4]
    with pytest.raises(ConfigError, match="assigned to both"):
        load_run_config(str(write_config(tmp_path, data)))

    data["models"][0]["cpu_affinity"] = [0, 1]
    data["models"][0]["cpu_thread_count"] = 3
    with pytest.raises(ConfigError, match="exceeds"):
        load_run_config(str(write_config(tmp_path, data)))

    del data["models"][0]["cpu_thread_count"]
    assert load_run_config(str(write_config(tmp_path, data)))

    data["models"][0].pop("cpu_affinity")
    data["models"][0]["cpu_thread_count"] = 3
    assert load_run_config(str(write_config(tmp_path, data)))

    data["replay"].pop("cpu_thread_count")
    assert load_run_config(str(write_config(tmp_path, data)))


def test_per_library_thread_counts_validate_against_affinity(tmp_path):
    """Known library pools are independent and cannot exceed their CPU set."""
    data = valid_data(tmp_path)
    data["models"][0].update(
        {
            "cpu_affinity": [0, 1, 2, 3],
            "cpu_thread_count": 4,
            "library_thread_counts": {
                "pytorch_intraop": 3,
                "pytorch_interop": 1,
                "opencv": 2,
                "openmp": 3,
                "mkl": 2,
                "openblas": 1,
                "numexpr": 2,
            },
        }
    )
    assert load_run_config(str(write_config(tmp_path, data)))

    data["models"][0]["library_thread_counts"]["opencv"] = 5
    with pytest.raises(ConfigError, match="opencv exceeds"):
        load_run_config(str(write_config(tmp_path, data)))


def test_preprocess_components_and_validation_metadata(tmp_path):
    """Profiler/analyzer pairs and validation identity are explicit."""
    data = valid_data(tmp_path)
    data["run"]["phase"] = "diagnostic"
    data["recording"].update({
        "scopes": ["model", "preprocessing"],
        "preprocessing": {"scheduler_backend": "bpftrace"},
    })
    assert load_run_config(str(write_config(tmp_path, data)))
    data["run"]["phase"] = "validation"
    with pytest.raises(ConfigError, match="validation run requires"):
        load_run_config(str(write_config(tmp_path, data)))
    data["run"].update(
        {
            "parent_run_id": "diagnostic-1",
            "hypothesis_id": "preprocess-contention",
            "intervention_id": "threads-3",
        }
    )
    assert load_run_config(str(write_config(tmp_path, data)))


def test_gpu_forensics_capsule_components_are_fixed_and_mps_only(tmp_path):
    """The exact capsule registers fixed selection and sample counts."""
    data = valid_data(tmp_path)
    data["recording"]["capsule"] = {
                "checkpoint_required": True,
                "timing_repetitions": 1,
                "deep_repetitions": 1,
                "sequence_match_attempts": 3,
                "required_collectors": [
                    "cupti_activity",
                    "cuda_launch_capture",
                    "cupti_checkpoint",
                ],
                "optional_collectors": [
                    "cupti_pm",
                    "ncu_mps",
                    "nvbit_cta",
                    "nvdebug_readonly",
                ],
    }
    with pytest.raises(ConfigError, match="requires MPS"):
        load_run_config(str(write_config(tmp_path, data)))
    data["gpu"]["mps_enabled"] = True
    data["models"][0]["mps_percentage"] = 100
    data["recording"]["nsys"]["gpu_context_switch"] = False
    assert load_run_config(str(write_config(tmp_path, data)))
    data["recording"]["capsule"]["timing_repetitions"] = 2
    with pytest.raises(ConfigError):
        load_run_config(str(write_config(tmp_path, data)))
    data["recording"]["capsule"]["timing_repetitions"] = 1
    data["recording"]["capsule"]["sequence_match_attempts"] = 2
    with pytest.raises(ConfigError):
        load_run_config(str(write_config(tmp_path, data)))
