"""Tests for deterministic CPU and native thread-pool controls."""

from types import SimpleNamespace

import pytest

from closeloop_testbed import resource_control


def test_process_resources_apply_affinity_and_thread_environment(monkeypatch):
    """Affinity and native-library limits are applied before workers exist."""
    captured = {}
    environment = {}
    monkeypatch.setattr(resource_control.os, "environ", environment)
    monkeypatch.setattr(
        resource_control.os, "sched_setaffinity",
        lambda pid, cpus: captured.update({"pid": pid, "cpus": cpus})
    )
    monkeypatch.setattr(
        resource_control.os, "sched_getaffinity",
        lambda _pid: {2, 3, 4, 5}
    )

    status = resource_control.apply_process_resources({
        "cpu_affinity": [2, 3, 4, 5],
        "cpu_thread_count": 2,
    })

    assert captured == {"pid": 0, "cpus": {2, 3, 4, 5}}
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["MKL_NUM_THREADS"] == "2"
    assert environment["OPENBLAS_NUM_THREADS"] == "2"
    assert environment["NUMEXPR_NUM_THREADS"] == "2"
    assert environment["OMP_DYNAMIC"] == "FALSE"
    assert status == {
        "cpu_affinity": [2, 3, 4, 5],
        "cpu_affinity_source": "explicit",
        "cpu_thread_count": 2,
        "library_thread_counts": {
            "mkl": 2,
            "numexpr": 2,
            "openblas": 2,
            "opencv": 2,
            "openmp": 2,
            "pytorch_interop": 1,
            "pytorch_intraop": 2,
        },
    }


def test_process_resources_apply_affinity_without_thread_limit(monkeypatch):
    """Affinity can be applied without changing native-library limits."""
    captured = {}
    environment = {}
    monkeypatch.setattr(resource_control.os, "environ", environment)
    monkeypatch.setattr(
        resource_control.os, "sched_setaffinity",
        lambda pid, cpus: captured.update({"pid": pid, "cpus": cpus})
    )
    monkeypatch.setattr(
        resource_control.os, "sched_getaffinity",
        lambda _pid: {2, 3, 4, 5}
    )

    status = resource_control.apply_process_resources({
        "cpu_affinity": [2, 3, 4, 5],
    })

    assert captured == {"pid": 0, "cpus": {2, 3, 4, 5}}
    assert environment == {}
    assert status == {
        "cpu_affinity": [2, 3, 4, 5],
        "cpu_affinity_source": "explicit",
        "cpu_thread_count": None,
        "library_thread_counts": {},
    }


def test_process_resources_apply_thread_limit_without_affinity(monkeypatch):
    """Native-library limits can be applied without pinning the process."""
    environment = {}
    monkeypatch.setattr(resource_control.os, "environ", environment)
    monkeypatch.setattr(
        resource_control.os, "sched_setaffinity",
        lambda _pid, _cpus: pytest.fail("affinity should not be changed")
    )
    monkeypatch.setattr(
        resource_control.os, "sched_getaffinity",
        lambda _pid: {0, 1, 2, 3}
    )

    status = resource_control.apply_process_resources({
        "cpu_thread_count": 2,
    })

    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["MKL_NUM_THREADS"] == "2"
    assert environment["OPENBLAS_NUM_THREADS"] == "2"
    assert environment["NUMEXPR_NUM_THREADS"] == "2"
    assert status == {
        "cpu_affinity": [0, 1, 2, 3],
        "cpu_affinity_source": "inherited",
        "cpu_thread_count": 2,
        "library_thread_counts": {
            "mkl": 2,
            "numexpr": 2,
            "openblas": 2,
            "opencv": 2,
            "openmp": 2,
            "pytorch_interop": 1,
            "pytorch_intraop": 2,
        },
    }


def test_model_compute_pools_match_configured_physical_cores():
    """Configured OpenCV and PyTorch workers use the requested core count."""
    cv2_module = SimpleNamespace(
        setNumThreads=lambda count: setattr(cv2_module, "count", count),
        getNumThreads=lambda: cv2_module.count,
    )
    torch_module = SimpleNamespace(
        set_num_threads=lambda count: setattr(
            torch_module, "thread_count", count),
        get_num_threads=lambda: torch_module.thread_count,
        set_num_interop_threads=lambda count: setattr(
            torch_module, "interop_count", count),
        get_num_interop_threads=lambda: torch_module.interop_count,
    )

    status = resource_control.configure_model_thread_pools(
        3, cv2_module=cv2_module, torch_module=torch_module
    )

    assert status == {
        "opencv_thread_count": 3,
        "torch_thread_count": 3,
        "torch_interop_thread_count": 1,
        "observed_library_thread_counts": {
            "mkl": 3,
            "numexpr": 3,
            "openblas": 3,
            "opencv": 3,
            "openmp": 3,
            "pytorch_interop": 1,
            "pytorch_intraop": 3,
        },
    }


def test_per_library_counts_override_legacy_fallback(monkeypatch):
    """Every supported library can override the process fallback."""
    environment = {}
    monkeypatch.setattr(resource_control.os, "environ", environment)
    monkeypatch.setattr(
        resource_control.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}
    )
    settings = {
        "cpu_thread_count": 4,
        "library_thread_counts": {
            "openmp": 3,
            "mkl": 2,
            "openblas": 1,
            "numexpr": 2,
            "opencv": 2,
            "pytorch_intraop": 3,
            "pytorch_interop": 2,
        },
    }

    status = resource_control.apply_process_resources(settings)

    assert environment["OMP_NUM_THREADS"] == "3"
    assert environment["MKL_NUM_THREADS"] == "2"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["NUMEXPR_NUM_THREADS"] == "2"
    assert status["library_thread_counts"]["opencv"] == 2
    assert status["library_thread_counts"]["pytorch_intraop"] == 3
    assert status["library_thread_counts"]["pytorch_interop"] == 2


def test_model_compute_pools_are_independently_configurable():
    """OpenCV and both PyTorch pools honor distinct requested counts."""
    cv2_module = SimpleNamespace(
        setNumThreads=lambda count: setattr(cv2_module, "count", count),
        getNumThreads=lambda: cv2_module.count,
    )
    torch_module = SimpleNamespace(
        set_num_threads=lambda count: setattr(
            torch_module, "thread_count", count),
        get_num_threads=lambda: torch_module.thread_count,
        set_num_interop_threads=lambda count: setattr(
            torch_module, "interop_count", count),
        get_num_interop_threads=lambda: torch_module.interop_count,
    )

    status = resource_control.configure_model_thread_pools(
        {"library_thread_counts": {
            "opencv": 2,
            "pytorch_intraop": 3,
            "pytorch_interop": 4,
        }},
        cv2_module=cv2_module,
        torch_module=torch_module,
    )

    assert status["observed_library_thread_counts"] == {
        "opencv": 2,
        "pytorch_interop": 4,
        "pytorch_intraop": 3,
    }


def test_gpu_clock_lock_resolves_pair_and_resets_on_all_exit_paths():
    """Clock selection is numeric and both reset commands survive failure."""
    commands = []

    def run_command(command, **_kwargs):
        commands.append(command)
        query = next(
            (item for item in command if item.startswith("--query")), "")
        if query.startswith("--query-supported"):
            stdout = "8000, 1900\n9000, 1800\n9000, 2000\n"
        elif query.startswith("--query-gpu"):
            stdout = "2000, 9000\n"
        else:
            stdout = ""
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    selected = resource_control.maximum_supported_clocks(0, run_command)
    assert selected["graphics_clock_mhz"] == 2000
    assert selected["memory_clock_mhz"] == 9000
    with pytest.raises(KeyboardInterrupt):
        with resource_control.GPUClockLock(
                0, 2000, 9000, run_command):
            raise KeyboardInterrupt
    assert any("--reset-gpu-clocks" in command for command in commands)
    assert any("--reset-memory-clocks" in command for command in commands)


def test_gpu_clock_lock_resets_after_setup_failure():
    """A partial clock setup still attempts both independent resets."""
    commands = []

    def run_command(command, **_kwargs):
        commands.append(command)
        if any(item.startswith("--lock-memory") for item in command):
            raise RuntimeError("memory lock failed")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    with pytest.raises(RuntimeError, match="memory lock failed"):
        with resource_control.GPUClockLock(0, 2000, 9000, run_command):
            pass
    assert any("--reset-gpu-clocks" in command for command in commands)
    assert any("--reset-memory-clocks" in command for command in commands)


def test_non_mps_environment_sanitation_is_copy_only():
    """Inherited MPS variables are removed without mutating the caller."""
    environment = {
        "PATH": "/bin",
        "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50",
    }
    sanitized, removed = resource_control.sanitize_mps_environment(
        environment)
    assert sanitized == {"PATH": "/bin"}
    assert removed == {
        "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50",
    }
    assert "CUDA_MPS_PIPE_DIRECTORY" in environment
