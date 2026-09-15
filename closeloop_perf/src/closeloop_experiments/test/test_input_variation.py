"""Small checks for the input-variation matrix and robust metrics."""

from closeloop_experiments.input_variation import (
    _load_validation_policy,
    absolute_spread_difference_confidence_interval,
    amplification_confidence_interval, expected_runs,
    architecture_pathway_analysis, case_study_analysis, latency_metrics,
    ratio_of_ratios_confidence_interval, summarize_input_geometry,
    summarize_primary_findings,
)


def test_matrix_has_all_pairs_controls_and_fixed_trio():
    """One condition expands to 8 controls, 12 pairs, and one fixed trio."""
    models = {
        "lidar-a": {"group": "lidar_detection"},
        "lidar-b": {"group": "lidar_detection"},
        **{f"camera-{index}": {"group": "image_detection"}
           for index in range(6)},
    }
    study = {
        "data": {
            "models": models, "pair_condition_ids": ["clean"],
            "triple_condition_ids": ["clean"], "replicates": 1,
            "preselected_trio": ["lidar-a", "camera-0", "camera-1"],
        },
        "lidar": ["lidar-a", "lidar-b"],
        "camera": [f"camera-{index}" for index in range(6)],
    }
    runs = expected_runs(study)
    assert len(runs) == 21
    assert sum(value["tenancy"] == "single" for value in runs.values()) == 8
    assert sum(value["tenancy"] == "two_model"
               for value in runs.values()) == 12
    assert sum(value["tenancy"] == "three_model"
               for value in runs.values()) == 1


def test_robust_metrics_and_amplification_ci_are_reproducible():
    """Spread metrics use every value and deterministic moving blocks."""
    control = [10, 11, 12, 13, 14, 15, 16, 17]
    pair = [10, 12, 14, 16, 18, 20, 22, 24]
    metrics = latency_metrics(control)
    assert metrics["sample_count"] == len(control)
    assert metrics["iqr_ms"] > 0
    assert metrics["p95_p5_ms"] > metrics["iqr_ms"]
    first = amplification_confidence_interval(pair, control, 50, 7)
    assert first == amplification_confidence_interval(pair, control, 50, 7)
    assert first[0] > 0
    modifier = ratio_of_ratios_confidence_interval(
        pair, control, control, control, 50, 8
    )
    assert modifier == ratio_of_ratios_confidence_interval(
        pair, control, control, control, 50, 8
    )
    difference = absolute_spread_difference_confidence_interval(
        pair, control, 50, 9, confidence=0.99
    )
    assert difference == absolute_spread_difference_confidence_interval(
        pair, control, 50, 9, confidence=0.99
    )
    assert difference[0] < difference[1]


def test_validation_amendment_is_explicit_and_study_bound(tmp_path):
    """A post-registration policy must preserve and bind the original plan."""
    study = {
        "sha256": "abc", "data": {
            "minimum_samples": {"image": 50, "lidar": 80},
        },
    }
    (tmp_path / "validation_amendment.yaml").write_text("""
schema_version: 1
amended_at: now
study_sha256: abc
reason: ten-second runs
original_minimum_samples: {image: 50, lidar: 80}
minimum_samples: {image: 10, lidar: 10}
timestamp_comparability: source_manifest_window
minimum_window_fraction: 0.95
""")
    policy = _load_validation_policy(study, tmp_path)
    assert policy["minimum_samples"] == {"image": 10, "lidar": 10}
    assert policy["minimum_window_fraction"] == 0.95


def test_primary_summary_classifies_confidence_intervals():
    """The concise table distinguishes amplified, damped, and null effects."""
    study = {"data": {"models": {
        "camera": {"group": "image_detection"},
        "lidar": {"group": "lidar_detection"},
    }}}
    summaries = [{
        "tenancy": "two_model", "co_models": "camera;lidar",
        "model_id": model, "amplification_factor": value,
    } for model, value in (("camera", 3.0), ("lidar", 4.0))]
    contrasts = [{
        "contrast": "multi_tenant_weather_modifier",
        "tenancy": "two_model", "co_models": "camera;lidar",
        "model_id": model, "estimate": 1.5,
        "ci95": interval,
    } for model, interval in (
        ("camera", [1.1, 2.0]), ("lidar", [0.8, 1.2])
    )]
    rows = summarize_primary_findings(study, summaries, contrasts)
    assert rows[0]["weather_amplified"] == 1
    assert rows[1]["weather_inconclusive"] == 1


def test_input_geometry_connects_points_voxels_and_latency():
    """Input geometry is summarized and joined to architecture evidence."""
    study = {
        "sha256": "abc", "lidar": ["lidar"],
        "data": {
            "bootstrap_repetitions": 20, "random_seed": 1,
            "conditions": {
                "weather-clean": {
                    "effect": "weather", "corruption_type": "clean",
                    "severity": 0,
                },
                "weather-rain-s1": {
                    "effect": "weather", "corruption_type": "rain",
                    "severity": 1,
                },
            },
        },
    }
    geometry = []
    for condition, scale in (("weather-clean", 10),
                             ("weather-rain-s1", 9)):
        for index in range(5):
            geometry.append({
                "condition_id": condition,
                "effect": "weather",
                "corruption_type": (
                    "clean" if condition.endswith("clean") else "rain"
                ),
                "severity": 0 if condition.endswith("clean") else 1,
                "ros_header_timestamp_ns": index,
                "total_points": scale * (index + 1),
                "points_0_5m": scale * (index + 1),
                "points_5_10m": 1, "points_10_20m": 1,
                "points_20_40m": 1, "points_40m_plus": 1,
            })
    summary = summarize_input_geometry(study, geometry)
    rain = next(row for row in summary
                if row["condition_id"] == "weather-rain-s1")
    assert rain["total_points_median_ratio_to_clean"] == 0.9

    frames, structures, modules = [], [], []
    for index in range(5):
        frames.append({
            "run_id": "run", "model_id": "lidar", "tenancy": "single",
            "condition_id": "weather-clean", "frame_id": str(index),
            "ros_header_timestamp_ns": index,
            "inference_e2e_ms": index + 1,
        })
        structures.append({
            "run_id": "run", "model_id": "lidar",
            "input_id": str(index), "structure_kind": "voxels",
            "structure_count": index + 1,
        })
        modules.append({
            "run_id": "run", "model_id": "lidar",
            "input_id": str(index), "module": "middle_encoder",
            "duration_ms": index + 1,
        })
    _runs, pathways = architecture_pathway_analysis(
        study, geometry, frames, modules, structures
    )
    voxel_latency = next(
        row for row in pathways if row["pathway"] == "voxels_to_latency"
    )
    assert voxel_latency["median_spearman"] == 1.0


def test_case_study_uses_absolute_width_and_labels_inference_scope():
    """The fixed trio separates strict and descriptive case-study results."""
    study = {
        "data": {
            "preselected_trio": ["lidar", "camera-a", "camera-b"],
            "triple_condition_ids": ["weather-clean", "weather-rain-s3"],
            "random_seed": 1,
        },
    }
    summaries, frames = [], []
    for condition, shared_values in (
            ("weather-clean", [10, 11, 12, 13, 14, 15, 16, 17]),
            ("weather-rain-s3", [10, 13, 16, 19, 22, 25, 28, 31])):
        single_values = [10, 11, 12, 13, 14, 15, 16, 17]
        for tenancy, co_models, values in (
                ("single", "lidar", single_values),
                ("three_model", "lidar;camera-a;camera-b", shared_values)):
            run_id = condition + "-" + tenancy
            summaries.append({
                **latency_metrics(values), "condition_id": condition,
                "tenancy": tenancy, "co_models": co_models,
                "model_id": "lidar", "run_id": run_id,
            })
            frames.extend({
                "run_id": run_id, "model_id": "lidar",
                "inference_e2e_ms": value,
                "memcpy_adjacent_waiting_ns": value * 1e6,
                "gpu_kernel_active_ms": value / 10,
                "kernel_span_ms": value,
            } for value in values)
    rows = case_study_analysis(study, summaries, frames)
    assert rows[1]["shared_change_from_clean_ms"] > 0
    assert rows[1]["single_change_from_clean_ms"] == 0
    assert rows[1]["strict_finding_scope"] == "within_run_temporal_ci99"
    assert rows[1]["weather_change_scope"] == (
        "descriptive_no_confidence_interval"
    )
