"""Unit checks for the fixed K1091 CTA-admission campaign."""

import json
from pathlib import Path

import pytest
import yaml

from closeloop_experiments.config import schema_v2_config
from closeloop_experiments.k1091_cta_mps_sweep import (
    BINS, K1091, MPS_VALUES, PROFILES, SweepError, admission_bins,
    campaign_conditions, materialize_campaign, successful_attempt,
)


TEMPLATE = Path(__file__).resolve().parents[3] / (
    "studies/mps_two_model_contention_cause/diagnostic_configs/"
    "mps2cause-cta-passive-r24.yaml"
)


def test_exact_ordered_cartesian_product_and_total():
    conditions = campaign_conditions()
    pairs = {
        (item.target_mps, item.corunner_mps)
        for item in conditions if item.profile == "real"
    }
    assert pairs == set((left, right) for left in MPS_VALUES
                        for right in MPS_VALUES)
    assert len(conditions) == 68
    assert {item.profile for item in conditions} == {*PROFILES, "target_only"}


def test_profile_generation_is_immutable_and_pinned(tmp_path):
    (tmp_path / "calibration_manifest.json").write_text(json.dumps({
        "paired_deadline_seconds_by_target_mps": {
            str(value): 2.05 + value / 100000 for value in MPS_VALUES
        },
    }), encoding="utf-8")
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(schema_v2_config(yaml.safe_load(TEMPLATE.read_text()))),
        encoding="utf-8",
    )
    manifest = materialize_campaign(template, tmp_path, 12345)
    assert len(manifest["conditions"]) == 68
    assert materialize_campaign(template, tmp_path, 12345) == manifest
    real = next(item for item in manifest["conditions"]
                if item["profile"] == "real")
    config = __import__("yaml").safe_load(
        Path(real["config_path"]).read_text(encoding="utf-8")
    )
    faster, deep = config["models"]
    assert faster["nvbit_cta_profile"]["targets"][0][
        "sequence_start"
    ] == K1091["sequence"]
    assert deep["nvbit_cta_profile"]["targets"][0][
        "sequence_end"
    ] == 290
    assert all(model["nvbit_cta_profile"]["mode"] == "mixed"
               for model in config["models"])
    synthetic = next(item for item in manifest["conditions"]
                     if item["profile"] == "synthetic_low"
                     and item["target_mps"] == 50)
    synthetic_config = __import__("yaml").safe_load(
        Path(synthetic["config_path"]).read_text(encoding="utf-8")
    )
    assert synthetic_config["synthetic_fma"][
        "paired_deadline_seconds"
    ] == pytest.approx(2.0505)
    path = Path(real["config_path"])
    path.chmod(0o644)
    path.write_text(path.read_text() + "# changed\n", encoding="utf-8")
    with pytest.raises(SweepError, match="immutable file differs"):
        materialize_campaign(template, tmp_path, 12345)


def test_half_open_300_bins_conserve_every_k1091_entry():
    entries = list(range(K1091["cta_count"]))
    rows = admission_bins(entries, 0, K1091["cta_count"] + 1)
    assert len(rows) == BINS
    assert sum(row["admitted_ctas"] for row in rows) == 2048
    assert rows[-1]["cumulative_admitted_ctas"] == 2048
    boundary = admission_bins([0, 10, 19], 0, 20, bins=2)
    assert [row["admitted_ctas"] for row in boundary] == [1, 2]
    with pytest.raises(ValueError, match="outside"):
        admission_bins([20], 0, 20, bins=2)


def test_resume_skips_valid_success_and_numbers_retries(tmp_path):
    record = {"condition_id": "condition", "run_prefix": "k1-00"}
    output = tmp_path
    first = output / "runs/k1-00-a1"
    first.mkdir(parents=True)
    (first / "run_manifest.json").write_text(
        json.dumps({"state": "failed"}), encoding="utf-8"
    )
    assert successful_attempt(record, output) == (None, 2)
    second = output / "runs/k1-00-a2"
    second.mkdir()
    (second / "run_manifest.json").write_text(
        json.dumps({"state": "success"}), encoding="utf-8"
    )
    (second / "campaign_validation.json").write_text(
        json.dumps({"valid": True}), encoding="utf-8"
    )
    assert successful_attempt(record, output) == (second, None)
