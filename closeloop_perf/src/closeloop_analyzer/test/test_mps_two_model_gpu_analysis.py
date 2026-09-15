"""Small checks for the Phase 4 interval and rank helpers."""

from closeloop_analyzer.mps_two_model_gpu_analysis import (
    _correlation, _duration_metrics, _intersection_duration, _merged,
    _target_decisions,
)


def test_interval_and_rank_helpers():
    assert _merged([(5, 9), (1, 4), (3, 7)]) == [(1, 9)]
    assert _intersection_duration([(0, 5), (7, 10)], [(2, 8)]) == 4
    assert _correlation([1, 2, 3], [10, 20, 30]) == 1


def test_kernel_duration_metrics_use_p99_minus_observed_minimum():
    metrics = _duration_metrics([1.0, 2.0, 3.0])
    assert metrics["p0_ms"] == 1.0
    assert metrics["p50_ms"] == 2.0
    assert metrics["primary_range_ms"] == 1.98
    assert metrics["normalized_range"] == 0.99


def test_target_selection_uses_range_and_keeps_required_comparisons():
    rows = []
    for model, index, duration_range in (
        ("faster_rcnn", "K0255", 2.0),
        ("faster_rcnn", "K1091", 4.0),
        ("deeplabv3plus", "K0228", 3.0),
        ("deeplabv3plus", "K0265", 1.0),
    ):
        rows.append({
            "model_id": model, "kernel_index": index,
            "kernel_name": index, "run_count": 3,
            "inference_coverage_fraction": 1.0, "sample_count": 858,
            "sequence_index_stable": True, "fast_observation_count": 80,
            "slow_observation_count": 80,
            "primary_range_ms": duration_range,
        })
    decisions, selected = _target_decisions(rows, 3)
    assert {(row["model_id"], row["kernel_index"]) for row in selected} == {
        ("faster_rcnn", "K0255"), ("faster_rcnn", "K1091"),
        ("deeplabv3plus", "K0228"),
        ("deeplabv3plus", "K0265"),
    }
    assert all(row["cta_outcomes_examined"] is False for row in decisions)
