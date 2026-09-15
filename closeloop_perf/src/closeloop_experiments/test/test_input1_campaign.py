"""Focused checks for the two-pass input1 physical-rain campaign."""

import os
from pathlib import Path
import tempfile

import pytest

from closeloop_profiler.architecture_profiles import get_architecture_profile
from closeloop_experiments.config import load_run_config
from closeloop_analyzer.input_data.corrected_input_analysis import (
    predictability_table_rows,
    variation_metrics,
)
from closeloop_experiments.corrected_input_variation import (
    expected_runs,
    load_study,
    materialize,
    model_pairs,
)
from closeloop_experiments.input_variation import _validate_source_passes
from closeloop_experiments.mps import MAX_UNIX_SOCKET_PATH_BYTES
from closeloop_testbed.replayer import playback_segments


ROOT = Path(__file__).parents[3]
STUDY = ROOT / "studies/input1/study.yaml"
V3_STUDY = ROOT / "studies/input_data_variation_matrix_v3/study.yaml"
V4_STUDY = ROOT / "studies/input-data-variation-rainrate-v4/study.yaml"


def test_input1_inventory_generates_exactly_210_two_pass_runs(tmp_path):
    study = load_study(STUDY, check_paths=False)
    planned = expected_runs(study)
    configs = materialize(study, tmp_path)

    assert len(study["lidar"]) == 3
    assert len(study["camera"]) == 7
    assert len(model_pairs(study)) == 21
    assert len(planned) == len(configs) == 210
    assert study["data"]["replay_passes"] == 2
    assert set(study["data"]["conditions"]) == {
        "clear", "r7p5", "r15", "r25", "r50",
    }
    for path in configs.values():
        config = load_run_config(path, check_paths=False).data
        assert config["replay"]["repeat_count"] == 2
        enabled = config["gpu"]["mps_enabled"]
        assert all((model.get("mps_percentage") == 100) == enabled
                   for model in config["models"])


def test_old_campaigns_default_to_one_pass_without_rewriting_configs(tmp_path):
    for path in (V3_STUDY, V4_STUDY):
        study = load_study(path, check_paths=False)
        assert study["data"].get("replay_passes", 1) == 1
        config = load_run_config(next(iter(materialize(study, tmp_path).values())),
                                 check_paths=False).data
        assert config["replay"].get("repeat_count", 1) == 1


def test_replayer_orders_two_uniquely_identified_passes(tmp_path):
    metadata = tmp_path / "scene.json"
    metadata.write_text('[{"token":"a","name":"scene-a"}]',
                        encoding="utf-8")
    (tmp_path / "scene-a.mcap").touch()
    segments = playback_segments({
        "scene_token": "a",
        "metadata_path": str(metadata),
        "bag_directory": str(tmp_path),
        "repeat_count": 2,
    })
    assert [row["pass_index"] for row in segments] == [0, 1]
    assert len({row["segment_id"] for row in segments}) == 2


def test_each_pass_must_meet_sample_and_source_window_coverage():
    source = set(range(20))
    records = [
        {"pass_index": pass_index, "ros_header_timestamp_ns": timestamp}
        for pass_index, timestamps in ((0, range(20)), (1, range(10)))
        for timestamp in timestamps
    ]
    errors = _validate_source_passes(
        "dino", records, 2, 10, source, 0.95,
    )
    assert not any("pass 0" in error for error in errors)
    assert any("pass 1 covers" in error for error in errors)


def test_pinned_new_models_and_dino_profile_are_resolved():
    models = load_study(STUDY, check_paths=False)["data"]["models"]
    assert models["pointpillars"]["model_config_sha256"] == (
        "1ffe085179a48b1bf47e15c12674fe0d58d518cb117ce7a2b7fa10dbbdbd4db1"
    )
    assert models["pointpillars"]["checkpoint_sha256"] == (
        "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
    )
    assert models["dino"]["checkpoint_sha256"] == (
        "55b2bba264ac6ea4d006ce7597fd265c778495f0c4988530f967eee13e2f2d59"
    )
    profile = get_architecture_profile(models["dino"]["architecture_profile"])
    assert profile.name == "mmdet_detr_2d_v1"
    assert {binding.module_path for binding in profile.module_bindings
            if binding.required} == {
        "data_preprocessor", "backbone", "neck", "positional_encoding",
        "encoder", "decoder", "bbox_head",
    }


def test_minimum_to_p99_change_and_complete_42_row_mode_tables():
    conditions = ("clear", "r7p5", "r15", "r25", "r50")
    baseline = {condition: "clear" for condition in conditions[1:]}
    pairs = [(f"lidar-{left}+camera-{right}", f"lidar-{left}",
              f"camera-{right}") for left in range(3) for right in range(7)]
    summary = [{
        "pair_id": pair,
        "target_model": target,
        "condition_id": condition,
        "mps_mode": mode,
        "p0_p99_range_ms": 10.0 if condition == "clear" else 12.0,
    } for mode in ("off", "on") for pair, lidar, camera in pairs
       for target in (lidar, camera) for condition in conditions]
    for mode in ("off", "on"):
        rows = predictability_table_rows(summary, conditions, baseline, mode)
        assert len(rows) == 42
        assert all(row["r50_percentage_change"] == pytest.approx(20.0)
                   for row in rows)
    assert variation_metrics(8.0, 10.0)["percentage_change"] == pytest.approx(
        -20.0
    )


def test_longest_generated_mps_socket_path_fits_linux_limit():
    study = load_study(STUDY, check_paths=False)
    with tempfile.TemporaryDirectory(prefix="p", dir="/tmp") as directory:
        configs = materialize(study, directory)
        paths = [
            load_run_config(
                path, check_paths=False, artifact_root=directory
            ).run_directory / "mps" / "pipe" / "control_privileged"
            for run_id, path in configs.items() if run_id.startswith("1-")
        ]
    assert max(len(os.fsencode(path)) for path in paths) <= (
        MAX_UNIX_SOCKET_PATH_BYTES
    )
