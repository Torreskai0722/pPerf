"""Tests for Nsight command, manifests, and preflight helpers."""

import json

import pytest

import closeloop_experiments.runner as runner_module
from closeloop_experiments.config import RunConfig
from closeloop_experiments.runner import (
    build_ncu_control_command, build_nsys_command, build_profile_command,
    detected_nsys_version, ExperimentRunner, initial_model_manifest,
)


@pytest.mark.parametrize(
    "recording,collector",
    [
        ({
            "level": "level1", "scopes": ["model", "input"],
            "nsys": {"trace": ["cuda", "nvtx", "cudnn"],
                     "gpu_context_switch": True},
        }, "nsys"),
        ({
            "level": "level2a", "scopes": ["model"],
            "nsys": {"trace": ["cuda", "nvtx", "cudnn"],
                     "gpu_context_switch": True,
                     "gpu_metrics_devices": "0",
                     "gpu_metrics_frequency_hz": 10000,
                     "gpu_metrics_set": "ad10x"},
        }, "nsys"),
        ({
            "level": "level2b", "scopes": ["model"],
            "ncu": {"version": "2025.2.1.0", "kernel_name": "kernel",
                    "launch_count": 1, "sections": ["LaunchStats"]},
        }, "ncu"),
        ({
            "level": "level1", "scopes": ["model", "preprocessing"],
            "nsys": {"trace": ["cuda", "nvtx", "cudnn"],
                     "gpu_context_switch": True},
            "preprocessing": {"scheduler_backend": "bpftrace"},
        }, "bpftrace"),
    ],
)
def test_each_level_and_scope_reaches_command_and_manifest(
        tmp_path, monkeypatch, recording, collector):
    """Applied recording selection is identical in command and manifest."""
    data = {
        "run": {"id": collector, "timeout_seconds": 1},
        "ros": {"launch_package": "closeloop_testbed",
                "launch_file": "testbed.launch.py", "domain_id": 1,
                "middleware": "rmw_fastrtps_cpp"},
        "replay": {"scene_token": "scene"},
        "gpu": {"index": 0, "mps_enabled": False},
        "models": [{"id": "model",
                    "architecture_profile": "mmdet_two_stage_2d_v1"}],
        "recording": recording,
    }
    config = RunConfig(
        data, tmp_path / "run.yaml", b"schema_version: 2\n", "digest",
        tmp_path / "artifacts",
    )
    command = build_profile_command(config, tmp_path / "profile")
    assert command[0] == collector
    runner = ExperimentRunner(
        config, popen=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("stop after manifest")
        )
    )
    runner.preflight = lambda: None
    monkeypatch.setattr(runner_module, "detect_cpu_topology", lambda: {})
    monkeypatch.setattr(
        runner_module, "detect_gpu_hardware", lambda *_args: {}
    )
    with pytest.raises(RuntimeError, match="stop after manifest"):
        runner.run()
    manifest = json.loads((
        config.run_directory / "run_manifest.json"
    ).read_text(encoding="utf-8"))
    assert manifest["recording"] == recording


def test_nsys_command_can_capture_process_tree_context_switches(tmp_path):
    """Command disables sampling while optionally tracing thread scheduling."""
    data = {
        "run": {"id": "run"},
        "ros": {"launch_package": "closeloop_testbed",
                "launch_file": "testbed.launch.py"},
        "gpu": {"mps_enabled": False},
        "recording": {
            "level": "level2a", "scopes": ["model"],
            "nsys": {"trace": ["cuda", "nvtx", "cudnn"],
                     "cpu_context_switch": True,
                     "gpu_context_switch": True,
                     "gpu_metrics_devices": "0",
                     "gpu_metrics_frequency_hz": 10000,
                     "gpu_metrics_set": "ad10x"},
        },
    }
    config = RunConfig(data, tmp_path / "run.yaml", b"x", "hash", tmp_path)
    command = build_nsys_command(config, tmp_path / "profile")
    assert "--trace=cuda,nvtx,cudnn" in command
    assert "--sample=none" in command
    assert "--backtrace=none" in command
    assert "--cpuctxsw=process-tree" in command
    assert "--wait=all" in command
    assert "--force-overwrite=false" in command
    assert "--gpuctxsw=true" in command
    assert "--gpu-metrics-devices=0" in command
    assert "--gpu-metrics-frequency=10000" in command
    assert "--gpu-metrics-set=ad10x" in command

    data["recording"]["nsys"]["cpu_context_switch"] = False
    command = build_nsys_command(config, tmp_path / "profile")
    assert "--cpuctxsw=none" in command

    data["recording"] = {"level": "level2b", "scopes": ["model"], "ncu": {
        "version": "2025.2.1.0",
        "kernel_name": "regex:kernel",
        "launch_count": 1,
        "sections": ["LaunchStats", "Occupancy"],
    }}
    command = build_profile_command(config, tmp_path / "profile")
    assert command[:5] == [
        "ncu", "--mps", "client", "--target-processes", "all"
    ]
    assert command.count("--section") == 0
    assert command[-2:] == [
        "config_file:=" + str(config.run_directory / "resolved_config.yaml"),
        "run_directory:=" + str(config.run_directory),
    ]
    control = build_ncu_control_command(config, tmp_path / "profile")
    assert control[:5] == [
        "ncu", "--mps", "control", "--mps-num-clients", "2"
    ]
    assert control.count("--section") == 2


def test_version_parser():
    """Nsight's descriptive version output is parsed exactly."""
    class Result:
        """Completed process stand-in."""

        stdout = "NVIDIA Nsight Systems version 2025.3.1.90"

    def run_command(*_args, **_kwargs):
        return Result()

    assert detected_nsys_version(run_command) == "2025.3.1.90"


def test_preprocess_profile_wraps_nsys_with_scheduler_collection(tmp_path):
    """The targeted component selects detailed Nsight and bpftrace evidence."""
    data = {
        "run": {"id": "run"},
        "ros": {"launch_package": "closeloop_testbed",
                "launch_file": "testbed.launch.py"},
        "gpu": {"mps_enabled": False},
        "recording": {
            "level": "level1", "scopes": ["model", "preprocessing"],
            "nsys": {"trace": ["cuda", "nvtx", "cudnn"],
                     "gpu_context_switch": False},
            "preprocessing": {"scheduler_backend": "bpftrace"},
        },
    }
    config = RunConfig(data, tmp_path / "run.yaml", b"x", "hash", tmp_path)
    nsys = build_nsys_command(config, tmp_path / "profile")
    assert "--trace=cuda,nvtx,cudnn,osrt,openmp" in nsys
    assert "--sample=process-tree" in nsys
    assert "--backtrace=dwarf" in nsys
    command = build_profile_command(config, tmp_path / "profile")
    assert command[:2] == ["bpftrace", "-q"]
    assert str(tmp_path / "runs" / "run" / "scheduler_events.csv") in command
    assert command[command.index("-c") + 1].startswith("nsys profile ")


def test_initial_manifest_contains_profile_metadata(tmp_path):
    """The initial manifest identifies profiles before final statuses."""
    data = {
        "run": {"id": "run"},
        "models": [{"id": "image",
                    "architecture_profile": "mmdet_two_stage_2d_v1"}],
    }
    config = RunConfig(data, tmp_path / "run.yaml", b"x", "hash", tmp_path)
    models = initial_model_manifest(config)
    assert models["image"]["architecture_profile"] == (
        "mmdet_two_stage_2d_v1"
    )
    assert len(models["image"]["architecture_profile_sha256"]) == 64
    assert models["image"]["required_method_bindings"] == ["model.predict"]
    assert models["image"]["required_module_bindings"] == [
        "backbone", "data_preprocessor", "roi_head", "rpn_head",
    ]
    assert models["image"]["observed_method_bindings"] == []
    assert models["image"]["observed_module_bindings"] == []


def test_final_status_replaces_seeded_manifest_metadata(tmp_path):
    """A failed model status is retained even when orchestration also fails."""
    data = {
        "run": {"id": "run"},
        "models": [{"id": "image",
                    "architecture_profile": "mmdet_two_stage_2d_v1"}],
    }
    config = RunConfig(data, tmp_path / "run.yaml", b"x", "hash", tmp_path)
    runner = ExperimentRunner(config)
    runner.run_directory.mkdir(parents=True)
    status = {"model_id": "image", "state": "failed",
              "architecture_profile": "mmdet_two_stage_2d_v1",
              "architecture_profile_sha256": "digest",
              "observed_method_bindings": ["model.predict"],
              "observed_module_bindings": ["backbone"]}
    (runner.run_directory / "model_image.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    manifest = {"models": initial_model_manifest(config)}
    runner._merge_model_statuses(manifest)
    assert manifest["models"]["image"] == status
