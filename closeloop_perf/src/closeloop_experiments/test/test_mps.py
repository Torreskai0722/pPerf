"""Tests for run-owned MPS lifecycle management."""

from types import SimpleNamespace

import pytest

from closeloop_experiments.mps import (
    MPSError, MPSManager, validate_mps_pipe_directory
)


def test_start_stop_and_restore(tmp_path):
    """A daemonizing control process is quit and compute mode restored."""
    commands = []

    def run_command(command, **kwargs):
        commands.append((command, kwargs))
        if "--query-gpu=compute_mode" in command:
            return SimpleNamespace(stdout="Default\n", returncode=0)
        if command == ["nvidia-cuda-mps-control"] and (
                kwargs.get("input") == "get_server_list\n"):
            return SimpleNamespace(stdout="123\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    process = SimpleNamespace(poll=lambda: 0)
    manager = MPSManager(1, tmp_path, run_command=run_command,
                         popen=lambda *_args, **_kwargs: process)
    manager.start()
    assert manager.evidence["server_state"] == "control_active"
    manager.wait_for_server(0.1)
    assert manager.evidence["server_ids"] == ["123"]
    assert manager.model_environment(30)[
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "30"
    manager.stop()
    assert any(command == ["nvidia-cuda-mps-control"]
               for command, _kwargs in commands)
    assert commands[-1][0][-2:] == ["-c", "0"]


def test_overlong_socket_path_is_rejected_before_gpu_changes(tmp_path):
    """MPS rejects an overlong privileged socket before querying the GPU."""
    pipe = tmp_path / ("x" * 120)
    with pytest.raises(MPSError, match="socket path"):
        validate_mps_pipe_directory(pipe)
