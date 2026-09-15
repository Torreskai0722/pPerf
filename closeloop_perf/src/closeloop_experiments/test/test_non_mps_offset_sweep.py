"""Checks for the generalized 18-pair signed-offset sweep."""

import csv
from itertools import islice
from pathlib import Path

import pytest

from closeloop_experiments.non_mps_offset_sweep import (
    _bag_stat_plots,
    bag_correlation_supplements,
    copy_alignment_contributions,
    correlation_rows,
    envelope_overlap,
    load_study,
    offset_order,
    per_offset_rows,
    run_config,
)


STUDY = (
    Path(__file__).parents[3]
    / "studies/non_mps_offset_sweep/study.yaml"
)


def test_inventory_has_18_unique_pairs_and_every_model_with_3dssd():
    study = load_study(STUDY, check_paths=False)

    assert len(study["pairs"]) == len(set(study["pairs"])) == 18
    assert {pair[1] for pair in study["pairs"] if pair[0] == "3dssd"} == {
        "faster-rcnn", "yolov3", "detr", "mask-rcnn",
        "deeplabv3plus", "vit-upernet",
    }


def test_offset_order_and_image_deadline_are_exact():
    study = load_study(STUDY, check_paths=False)
    assert list(islice(offset_order(), 9)) == [
        0, -8, 8, -16, 16, -24, 24, -32, 32,
    ]
    for offset_ms in (-16, 0, 16):
        config = run_config(
            study, ("3dssd", "faster-rcnn"), offset_ms)
        image, lidar = config["models"]
        assert image["launch_offset_seconds"] == 0
        assert lidar["launch_offset_seconds"] == 0
        assert image["paired_trial"]["trials"] == 3
        assert lidar["paired_trial"]["trials"] == 3
        assert image["paired_trial"]["deadline_offset_seconds"] == 2.0
        assert lidar["paired_trial"]["deadline_offset_seconds"] == pytest.approx(
            2.0 + offset_ms / 1000)


def test_both_direction_alignment_and_envelope_overlap():
    left = [
        {"copy_class": "H2D", "start": 10_000_000, "end": 10_100_000},
        {"copy_class": "D2D", "start": 15_000_000, "end": 15_100_000},
        {"copy_class": "D2H", "start": 20_000_000, "end": 20_100_000},
    ]
    right = [
        {"copy_class": "H2D", "start": 12_000_000, "end": 12_100_000},
        {"copy_class": "D2H", "start": 22_000_000, "end": 22_100_000},
    ]
    left_active = [(10_000_000, 12_000_000), (20_000_000, 21_000_000)]
    right_active = [(10_000_000, 12_000_000), (19_000_000, 20_000_000)]

    left_scores = copy_alignment_contributions(left, right_active)
    right_scores = copy_alignment_contributions(right, left_active)

    assert [row["copy_class"] for row in left_scores] == ["H2D", "D2H"]
    assert [row["alignment_contribution"] for row in left_scores] == [0, 0.5]
    assert [row["alignment_contribution"] for row in right_scores] == [1, 0.5]
    assert envelope_overlap((10, 30), (20, 40)) == 10
    assert envelope_overlap((10, 20), (20, 30)) == 0


def test_warning_trials_remain_in_offset_rows():
    rows = [{
        "pair_id": "3dssd+faster-rcnn",
        "configured_signed_offset_ms": -8,
        "target_model": "3dssd",
        "target_direction": "lidar",
        "diagnostic_warnings": "launch tolerance exceeded",
        "gpu_envelope_overlap_ns": 0,
        "actual_signed_offset_ms": -7.5,
        "nvtx_inference_e2e_ms": 10,
        "gpu_envelope_duration_ns": 9,
        "alignment_score_a": 0.5,
        "max_copy_alignment": 1,
        "h2d_count": 1,
        "d2h_count": 1,
        "memcpy_adjacent_waiting_ns": 2,
    }]

    summary = per_offset_rows(rows)

    assert summary[0]["trial_count"] == 1
    assert summary[0]["warning_trial_count"] == 1
    assert summary[0]["zero_overlap_trial_count"] == 1

    rows[0]["gpu_envelope_overlap_ns"] = None
    assert correlation_rows(rows)[0]["lidar_sample_count"] == 1


def test_bag_correlations_respect_conditions_scenes_and_timestamps(tmp_path):
    root = tmp_path / "bag-analysis"
    root.mkdir()
    path = root / "frame_metrics.csv"
    fields = [
        "alignment", "condition_id", "inference_e2e_ms",
        "input_timestamp_ns", "model_id", "pair_id",
        "ros_header_timestamp_ns",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        for model in ("pointpillars", "faster-rcnn"):
            for condition, timestamps in (
                    ("c1", (1, 2, 3)), ("c2", (1, 2, 3)),
                    ("c3", (4, 5, 6))):
                for timestamp in timestamps:
                    writer.writerow({
                        "alignment": timestamp / 10,
                        "condition_id": condition,
                        "inference_e2e_ms": timestamp,
                        "input_timestamp_ns": timestamp,
                        "model_id": model,
                        "pair_id": "pointpillars+faster-rcnn",
                        "ros_header_timestamp_ns": timestamp,
                    })
    study = {
        "paths": {"bag_analysis_root": root},
        "bag_conditions": {
            "c1": {"scene_name": "scene-1"},
            "c2": {"scene_name": "scene-1"},
            "c3": {"scene_name": "scene-2"},
        },
    }

    result = bag_correlation_supplements(study, repetitions=10)

    assert len(result["by_condition"]) == 6
    assert len(result["by_scene"]) == 4
    assert len(result["timestamp_averaged_rows"]) == 12
    assert len(result["timestamp_averaged_correlations"]) == 2
    assert len(result["clustered_intervals"]) == 2
    assert all(row["spearman"] == pytest.approx(1)
               for row in result["timestamp_averaged_correlations"])

    trials = [
        {
            "alignment_score_a": value,
            "pair_id": "pointpillars+faster-rcnn",
            "target_direction": direction,
        }
        for direction in ("image", "lidar") for value in (0.1, 0.2, 0.3)
    ]
    plot_output = tmp_path / "analysis"
    _bag_stat_plots(study, trials, result, plot_output)
    assert {
        path.name for path in (plot_output / "plots").glob("*.png")
    } == {
        "18-pair-source-aware-correlation-intervals.png",
        "pointpillars-faster-rcnn-a-distributions.png",
        "pointpillars-faster-rcnn-source-aware-correlations.png",
    }
