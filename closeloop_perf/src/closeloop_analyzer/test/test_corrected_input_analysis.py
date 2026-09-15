"""Tests for direct descriptive input-variation measurements."""

import json

import pytest

import closeloop_analyzer.input_data.corrected_input_analysis as analysis
from closeloop_analyzer.input_data.corrected_input_analysis import (
    _collect,
    _plot_heatmap,
    comparison_rows,
    context_comparison_rows,
    controlled_response_rows,
    distribution_metrics,
    ecdf_rows,
    input_change_rows,
    mechanism_comparison_rows,
    mechanism_rows,
    summarize,
    variation_metrics,
    wasserstein_1,
)


def test_variation_is_direct_weather_minus_clear_within_one_pair():
    """Absolute change and factor use no single-tenant denominator."""
    amplified = variation_metrics(15.0, 10.0)
    damped = variation_metrics(5.0, 10.0)
    assert amplified == {
        "variation_change_ms": 5.0,
        "variation_factor": 1.5,
        "percentage_change": 50.0,
        "observed_label": "observed_amplification",
    }
    assert damped["variation_change_ms"] == -5.0
    assert damped["variation_factor"] == 0.5
    assert damped["observed_label"] == "observed_damping"


def test_outputs_are_descriptive_distribution_records_only():
    """Summary, comparison, and ECDF records contain no inferential fields."""
    common = {
        "run_id": "run",
        "pair_id": "3dssd+detr",
        "lidar_model": "3dssd",
        "camera_model": "detr",
        "model_id": "3dssd",
        "mps_mode": "off",
    }
    frames = [{
        **common, "condition_id": condition,
        "inference_e2e_ms": value
    } for condition, values in (
        ("weather-clean", (10.0, 11.0, 12.0)),
        ("weather-rain-s1", (10.0, 14.0, 18.0)),
    ) for value in values]
    summary = summarize(frames)
    comparisons = comparison_rows(summary,
                                  {"weather-rain-s1": "weather-clean"})
    ecdfs = ecdf_rows(frames)
    clean = next(row for row in summary
                 if row["condition_id"] == "weather-clean")
    expected = {
        **{
            key: clean[key]
            for key in (
                "pair_id",
                "lidar_model",
                "camera_model",
                "target_model",
                "co_runner",
                "condition_id",
                "mps_mode",
                "analysis_scope",
            )
        },
        "frame_count": 3,
        "minimum_ms": 10.0,
        "maximum_ms": 12.0,
        "median_ms": 11.0,
        "iqr_ms": 1.0,
        "p5_ms": pytest.approx(10.1),
        "p95_ms": pytest.approx(11.9),
        "width_ms": pytest.approx(1.98),
    }
    assert {key: clean[key] for key in expected} == expected
    encoded = json.dumps([summary, comparisons, ecdfs]).lower()
    for forbidden in (
            "p_value",
            "bootstrap",
            "holm",
            "directional_supported",
            "inconclusive",
            "statistical_significance",
    ):
        assert forbidden not in encoded


def test_distribution_has_requested_width_and_ecdf_endpoints():
    """Minimum-to-P99 is primary and each empirical CDF terminates at one."""
    metrics = distribution_metrics([1, 2, 3, 4, 5])
    assert metrics["p0_p99_range_ms"] == pytest.approx(
        metrics["p99_ms"] - metrics["minimum_ms"])
    assert metrics["width_ms"] == metrics["p0_p99_range_ms"]
    frames = [{
        "pair_id": "p",
        "model_id": "m",
        "condition_id": "c",
        "mps_mode": "on",
        "inference_e2e_ms": value,
    } for value in (2, 1, 3)]
    rows = ecdf_rows(frames)
    assert [row["latency_ms"] for row in rows] == [1.0, 2.0, 3.0]
    assert rows[-1]["ecdf"] == 1.0


def test_distribution_keeps_shared_analyzer_compatibility():
    """Goal 1 helpers remain usable by the existing report analyzers."""
    metrics = distribution_metrics([1, 2, 3, 4, 5])
    assert metrics["sample_count"] == metrics["frame_count"] == 5
    assert metrics["primary_range_ms"] == pytest.approx(
        metrics["p99_ms"] - metrics["p0_ms"]
    )
    assert wasserstein_1([0, 1], [1, 2]) == pytest.approx(1.0)


def test_input_and_mechanism_changes_use_matching_clear_payloads():
    """Synthetic payload and GPU mechanisms keep direct clear baselines."""
    inputs = [{
        "condition_id": condition,
        "effect": "weather",
        "corruption_type": "clean" if condition == "clear" else "rain",
        "severity": 0 if condition == "clear" else 3,
        "modality": "lidar",
        "input_timestamp_ns": timestamp,
        "point_count": value,
    } for condition, values in (
        ("clear", (100, 120)), ("rain", (80, 90))
    ) for timestamp, value in enumerate(values)]
    changes = input_change_rows(inputs, {"rain": "clear"})
    points = next(row for row in changes if row["metric"] == "point_count")
    assert points["median_paired_change"] == -25
    assert points["clear_baseline_id"] == "clear"

    mechanisms = [{
        "pair_id": "3dssd+detr",
        "model_id": "3dssd",
        "condition_id": condition,
        "mps_mode": "off",
        "inference_e2e_ms": value,
    } for condition, values in (
        ("clear", (10, 11, 12)), ("rain", (10, 14, 18))
    ) for value in values]
    rows = mechanism_comparison_rows(mechanisms, {"rain": "clear"})
    latency = next(row for row in rows
                   if row["metric"] == "inference_e2e_ms")
    assert latency["observed_label"] == "observed_amplification"
    assert latency["clear_baseline_id"] == "clear"


def test_controlled_and_context_rows_keep_intrinsic_evidence_separate():
    """Synthetic input, isolated response, and pair response stay distinct."""
    input_changes = [{
        "condition_id": "rain", "clear_baseline_id": "clear",
        "modality": "lidar", "metric": "point_count",
        "adverse_median": 80, "clear_median": 100,
        "median_paired_change": -20, "median_ratio": 0.8,
    }]
    intrinsic = [{
        "pair_id": "", "target_model": "3dssd",
        "condition_id": "rain", "clear_baseline_id": "clear",
        "mps_mode": "off", "metric": "kernel_count",
        "adverse_median": 20, "clear_median": 25, "median_change": -5,
        "adverse_width": 2, "clear_width": 4, "variation_factor": 0.5,
        "observed_label": "observed_damping",
    }]
    study = {"data": {"models": {"3dssd": {
        "modality": "lidar", "architecture_profile": "point"
    }}}}
    controlled = controlled_response_rows(
        input_changes, intrinsic, [], study
    )
    assert controlled[0]["input_median_paired_change"] == -20
    assert controlled[0]["response_median_change"] == -5

    pair = [{
        **intrinsic[0], "pair_id": "3dssd+detr", "mps_mode": "on",
        "median_change": 2, "variation_factor": 1.2,
        "observed_label": "observed_amplification",
    }]
    context = context_comparison_rows(pair, intrinsic)[0]
    assert context["co_runner"] == "detr"
    assert context["isolated_observed_label"] == "observed_damping"
    assert context["pair_observed_label"] == "observed_amplification"


def test_mechanism_rows_retain_proposal_counts():
    frames = [{
        "run_id": "r", "model_id": "faster-rcnn", "frame_id": "0",
        "input_timestamp_ns": 1, "pair_id": "centerpoint+faster-rcnn",
        "condition_id": "clear", "mps_mode": "off",
        "inference_e2e_ms": 10, "gpu_kernel_active_ms": 8,
        "kernel_count": 100, "kernel_span_ms": 9, "waiting_ms": 1,
    }]
    structures = [{
        "run_id": "r", "model_id": "faster-rcnn", "input_id": "0",
        "structure_kind": "proposals", "structure_count": 37,
    }]
    assert mechanism_rows(frames, structures)[0]["proposal_count"] == 37


def test_diagnostic_warnings_keep_extractable_frames(tmp_path, monkeypatch):
    """Run diagnostics describe rows and never decide their inclusion."""
    run_id = "0-clear-3dssd-detr"
    run_directory = tmp_path / run_id
    run_directory.mkdir()
    (run_directory / "model_3dssd_inputs.jsonl").write_text(
        json.dumps({
            "input_id": "0", "ros_header_timestamp_ns": 123,
            "input_point_count": 456,
        }) + "\n", encoding="utf-8")
    evidence = {
        "errors": ["clock warning", "launch warning"],
        "frames": [{
            "model_id": "3dssd", "frame_id": "0",
            "inference_e2e_ms": 10,
            "memcpy_adjacent_waiting_ns": 20,
        }],
        "modules": [],
        "structures": [],
    }
    monkeypatch.setattr(
        analysis, "_validate_run",
        lambda *_args, **_kwargs: (evidence, {
            "3dssd": {"0": {
                "ros_header_timestamp_ns": 123,
                "input_point_count": 456,
            }}
        }))
    study = {"paths": {"output_root": tmp_path}}
    planned = {run_id: {
        "condition_id": "clear", "lidar_model": "3dssd",
        "camera_model": "detr", "models": ["3dssd", "detr"],
        "mps_enabled": False, "tenancy": "two_model",
    }}

    validation, rows, _modules, _structures = _collect(
        study, planned, {run_id: tmp_path / "config.yaml"})

    assert not validation[0]["valid"]
    assert validation[0]["extractable"]
    assert len(rows) == 1
    assert not rows[0]["run_valid"]
    assert rows[0]["run_diagnostics"] == "clock warning; launch warning"


def test_filtered_analysis_skips_empty_mps_heatmap(tmp_path):
    """A non-MPS subset does not require the unused MPS-on matrix."""
    _plot_heatmap([], {"data": {"conditions": {}}}, tmp_path, "on")
    assert not list(tmp_path.rglob("*"))
