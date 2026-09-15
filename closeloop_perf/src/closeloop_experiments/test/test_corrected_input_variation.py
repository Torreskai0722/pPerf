"""Tests for the descriptive cross-modal campaign matrix."""

import gzip
import json
from pathlib import Path

import yaml

from closeloop_experiments.corrected_input_variation import (
    MPS_ORDER, _can_reuse_completed_config, archive_profiles,
    expected_isolated_runs, expected_runs, load_study, materialize,
    materialize_isolated, model_pairs,
)
from closeloop_experiments.config import load_run_config


STUDY = (
    Path(__file__).parents[3]
    / "studies/input_data_variation_matrix_v3/study.yaml"
)
V4_STUDY = (
    Path(__file__).parents[3]
    / "studies/input-data-variation-rainrate-v4/study.yaml"
)


def test_campaign_matrix_has_only_the_requested_360_pairs():
    """Eight models derive twelve pairs across 15 conditions and two modes."""
    study = load_study(STUDY, check_paths=False)
    planned = expected_runs(study)
    models = study["data"]["models"]
    pairs = model_pairs(study)

    assert len(models) == 8
    assert len(study["lidar"]) == 2
    assert len(study["camera"]) == 6
    assert len(pairs) == 12
    assert len(set(pairs)) == 12
    assert study["data"]["replicates"] == 1
    assert tuple(study["data"]["mps_modes"]) == MPS_ORDER
    assert len(study["data"]["conditions"]) == 15
    assert len(planned) == 360
    assert {item["mps_enabled"] for item in planned.values()} == {False, True}
    assert all(item["replicate"] == 1 for item in planned.values())
    assert all(item["tenancy"] == "two_model" for item in planned.values())
    assert all(len(item["models"]) == 2 for item in planned.values())
    assert all(
        models[item["models"][0]]["modality"]
        != models[item["models"][1]]["modality"]
        for item in planned.values()
    )
    forbidden = ("rain_clear", "clear_rain", "snow_clear", "clear_snow")
    assert not any(
        token in item["condition_id"]
        for item in planned.values()
        for token in forbidden
    )


def test_full_rainrate_campaign_has_60_non_mps_full_replays(tmp_path):
    """Five full-scene conditions cross twelve pairs without MPS fields."""
    study = load_study(V4_STUDY, check_paths=False)
    planned = expected_runs(study)
    configs = materialize(study, tmp_path)

    assert len(study["data"]["conditions"]) == 5
    assert study["data"]["mps_modes"] == [False]
    assert len(planned) == len(configs) == 60
    assert {item["mps_enabled"] for item in planned.values()} == {False}
    assert set(study["data"]["baseline_map"].values()) == {"clear-full"}
    for path in configs.values():
        config = load_run_config(path, check_paths=False).data
        assert config["replay"]["playback_mode"] == "full"
        assert "duration_seconds" not in config["replay"]
        assert config["gpu"]["mps_enabled"] is False
        assert all("mps_percentage" not in model
                   for model in config["models"])
        corruption = config["input_variation"]["corruption"]
        assert corruption["full_duration"] is True
        if corruption["type"] == "rain":
            assert corruption["severity"] is None
            assert corruption["rain_rate_mm_per_hour"] in (7.5, 15, 25, 50)


def test_weather_and_interaction_baselines_are_explicit():
    """Every adverse input maps to the intended same-scene clear input."""
    baseline = load_study(STUDY, check_paths=False)["data"]["baseline_map"]
    assert all(
        baseline[f"weather-{weather}-s{severity}"] == "weather-clean"
        for weather in ("rain", "snow")
        for severity in (1, 3, 5)
    )
    assert baseline["interaction-low-rain-s3"] == "env-low-urban"
    assert baseline["interaction-low-snow-s3"] == "env-low-urban"
    assert baseline["interaction-crowded-rain-s3"] == (
        "env-crowded-intersection"
    )
    assert baseline["interaction-crowded-snow-s3"] == (
        "env-crowded-intersection"
    )


def test_generated_configs_use_one_condition_for_both_models(tmp_path):
    """Every generated pair replays one bag and one pinned manifest."""
    study = load_study(STUDY, check_paths=False)
    configs = materialize(study, tmp_path)
    forbidden = ("rain_clear", "clear_rain", "snow_clear", "clear_snow")
    assert len(configs) == 360
    for run_id, path in configs.items():
        config = load_run_config(path, check_paths=False).data
        condition_id = config["input_variation"]["condition_id"]
        manifest = Path(
            config["input_variation"]["corruption"]["dataset_manifest"]
        )
        assert not any(token in condition_id for token in forbidden)
        assert Path(config["replay"]["bag_directory"]) == manifest.parent
        assert len(config["models"]) == 2
        assert all(name in run_id for name in (
            condition_id, config["models"][0]["id"],
            config["models"][1]["id"],
        ))
        enabled = config["gpu"]["mps_enabled"]
        assert run_id.startswith("1-" if enabled else "0-")
        if enabled:
            assert all(
                model.get("mps_percentage")
                == study["data"]["mps_percentage"]
                for model in config["models"]
            )
        else:
            assert all(
                "mps_percentage" not in model for model in config["models"]
            )


def test_isolated_evidence_only_adds_the_missing_3dssd_runs(tmp_path):
    """The prior validated study covers the other six requested models."""
    study = load_study(STUDY, check_paths=False)
    planned = expected_isolated_runs(study)
    configs = materialize_isolated(study, tmp_path)
    assert len(planned) == len(configs) == 15
    assert all(item["models"] == ["3dssd"] for item in planned.values())
    assert all(item["tenancy"] == "single" for item in planned.values())
    assert all(not item["mps_enabled"] for item in planned.values())
    for path in configs.values():
        config = load_run_config(path, check_paths=False).data
        assert [model["id"] for model in config["models"]] == ["3dssd"]
        assert config["input_variation"]["tenancy"] == "single"


def test_3dssd_non_mps_prerequisite_is_exactly_105_runs():
    """The requested subset has six pairs times 15 plus 15 isolated runs."""
    study = load_study(STUDY, check_paths=False)
    pairs = {
        pair for pair in model_pairs(study) if pair[0] == "3dssd"
    }
    pair_runs = [
        item for item in expected_runs(study).values()
        if not item["mps_enabled"]
        and (item["lidar_model"], item["camera_model"]) in pairs
    ]
    isolated = expected_isolated_runs(study)

    assert pairs == {("3dssd", camera) for camera in study["camera"]}
    assert len(pair_runs) == 90
    assert len(isolated) == 15


def test_expanded_bag_summary_has_18_unique_pairs():
    """Six new 3DSSD pairs extend the 12 Section 4.1 pairs to 18."""
    study = load_study(STUDY, check_paths=False)
    prior = {
        (lidar, camera)
        for lidar in ("pointpillars", "centerpoint")
        for camera in study["camera"]
    }
    added = {
        pair for pair in model_pairs(study) if pair[0] == "3dssd"
    }

    assert len(prior | added) == 18


def test_completed_profiles_are_losslessly_archived(tmp_path):
    """Storage reduction retains both raw Nsight artifacts byte-for-byte."""
    payloads = {
        "profile.nsys-rep": b"nsight-report" * 100,
        "profile.sqlite": b"sqlite-trace" * 100,
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    archive_profiles(tmp_path)
    archive_profiles(tmp_path)
    for name, payload in payloads.items():
        assert not (tmp_path / name).exists()
        with gzip.open(tmp_path / f"{name}.gz", "rb") as archived:
            assert archived.read() == payload
    assert (tmp_path / "profile_archive.json").is_file()


def test_completed_config_reuse_allows_only_provenance_and_domain_drift(
        tmp_path):
    """Reuse requires success, exact executed bytes, and harmless drift."""
    generated = tmp_path / "generated" / "run.yaml"
    run = tmp_path / "output" / "run"
    generated.parent.mkdir()
    run.mkdir(parents=True)
    prior = {"run": {"provenance": "old"}, "ros": {"domain_id": 70},
             "value": 1}
    expanded = {"run": {"provenance": "new"}, "ros": {"domain_id": 99},
                "value": 1}
    prior_bytes = yaml.safe_dump(prior, sort_keys=False).encode()
    expanded_bytes = yaml.safe_dump(expanded, sort_keys=False).encode()
    generated.write_bytes(prior_bytes)
    (run / "config.yaml").write_bytes(prior_bytes)
    (run / "run_manifest.json").write_text(
        json.dumps({"state": "success"}), encoding="utf-8")
    study = {"paths": {"output_root": tmp_path / "output"}}

    assert _can_reuse_completed_config(study, generated, expanded_bytes)
    expanded["value"] = 2
    assert not _can_reuse_completed_config(
        study, generated, yaml.safe_dump(expanded).encode())
    (run / "config.yaml").write_bytes(b"different")
    assert not _can_reuse_completed_config(study, generated, expanded_bytes)
