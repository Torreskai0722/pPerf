"""Small helper check for Phase 4 CTA artifact parsing."""

import gzip

from closeloop_analyzer.mps_two_model_cta_analysis import (
    CROSS_CLIENT_SMID_COMPARABLE,
    _co_residency_feasibility,
    _target_launch_key,
    _tracer_kernel_signature,
    _read_csv,
    _write_csv,
)


def test_write_csv_uses_union_of_fields(tmp_path):
    path = tmp_path / "rows.csv"
    _write_csv(path, [{"a": 1}, {"b": 2}])
    assert path.read_text().splitlines()[0] == "a,b"


def test_co_residency_rejects_shared_memory_overcommit():
    launch = {
        "block": [128, 1, 1], "registers_per_thread": 16,
        "static_shared_memory": 0, "dynamic_shared_memory": 65536,
    }
    limits = {
        "warp_size": 32, "max_threads_per_sm": 1536,
        "registers_per_sm": 65536, "shared_memory_per_sm": 102400,
        "max_blocks_per_sm": 24,
    }
    feasible, reason, *_ = _co_residency_feasibility(
        launch, launch, limits
    )
    assert not feasible
    assert reason == "shared_memory"
    assert not CROSS_CLIENT_SMID_COMPARABLE


def test_target_launch_key_allows_missing_target():
    launches = {
        ("model", 0): {
            "target_label": "captured", "launch_sequence_index": 7,
        },
    }
    assert _target_launch_key(launches, "model", "captured", 7) == (
        "model", 0
    )
    assert _target_launch_key(launches, "model", "dropped", 7) is None


def test_tracer_kernel_signature_includes_driver_api():
    launch = {
        "kernel_name": "kernel", "driver_launch_api": "cuLaunchKernel",
        "launch_type": "kernel", "grid": [1, 2, 3], "block": [4, 5, 6],
        "registers_per_thread": 7, "static_shared_memory": 8,
        "dynamic_shared_memory": 9,
    }
    signature = _tracer_kernel_signature(launch)
    launch["driver_launch_api"] = "cuLaunchKernelEx"
    assert signature != _tracer_kernel_signature(launch)


def test_read_csv_falls_back_to_gzip(tmp_path):
    path = tmp_path / "records.csv"
    with gzip.open(f"{path}.gz", "wt", encoding="utf-8") as output:
        output.write("value\n1\n")
    assert _read_csv(path) == [{"value": "1"}]
