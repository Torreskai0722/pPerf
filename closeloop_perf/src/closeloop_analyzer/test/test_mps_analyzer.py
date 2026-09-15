"""Unit tests for conservative MPS stage skill outcomes."""

import json
import sqlite3

import pytest

from closeloop_analyzer.mps_analyzer import (
    AnalyzeMpsResourceResidencyV1,
    AnalyzeMpsTaskSlotAdmissionV1,
    AnalyzeMpsTpcEligibilityV1,
    AnalyzeMpsWduSelectionV1,
    MpsContentionAnalyzer,
)


def test_stage1_support_ruleout_and_missing_evidence():
    """Admission needs the knee and never claims direct WDU data."""
    skill = AnalyzeMpsTaskSlotAdmissionV1()
    supported = skill.analyze(
        {
            "pending_grid_count": 8,
            "calibrated_task_slot_capacity": 8,
            "hidden_mps_task_count": 0,
            "task_slot_admission_signature": True,
        }
    ).to_dict()
    assert supported["status"] == "supported"
    assert supported["precision_class"] == "calibrated_indirect"
    ruled = skill.analyze(
        {
            "pending_grid_count": 2,
            "calibrated_task_slot_capacity": 8,
            "hidden_mps_task_count": 0,
        }
    ).to_dict()
    assert ruled["status"] == "ruled_out"
    missing = skill.analyze({}).to_dict()
    assert missing["status"] == "inconclusive"
    assert missing["missing_evidence"]


def test_stage2_requires_ctas_admission_residency_and_bias():
    """Selection support is gated by stage 1 and block capacity."""
    skill = AnalyzeMpsWduSelectionV1()
    unavailable = skill.analyze({}).to_dict()
    assert unavailable["status"] == "not_observable"
    evidence = {
        "nvbit_cta_available": True,
        "aggressor_dispatch_continues_victim_absent": True,
        "victim_block_residency_capacity": True,
        "wdu_selection_signature": True,
    }
    supported = skill.analyze(evidence, {1: {"status": "ruled_out"}}).to_dict()
    assert supported["status"] == "supported"
    opposing = dict(evidence)
    opposing["aggressor_dispatch_continues_victim_absent"] = False
    assert (
        skill.analyze(opposing, {1: {"status": "supported"}}).status
        == "ruled_out"
    )
    partial = dict(evidence, cta_coverage_complete=False)
    assert skill.analyze(
        partial, {1: {"status": "supported"}}
    ).status == "inconclusive"


def test_stage3_missing_credit_state_is_inconclusive():
    """CTA absence alone cannot establish an MPS mask or credit restriction."""
    skill = AnalyzeMpsTpcEligibilityV1()
    base = {
        "nvbit_cta_available": True,
        "mps_active_thread_percentage": {"v": 50},
        "mps_client_priority": {"v": 0},
        "gpu_topology": {"sm_count": 56},
        "candidate_tpc_ids": [0, 1],
        "tpc_validation_signature": True,
        "victim_absent_from_candidate_tpcs": True,
    }
    assert skill.analyze(base).status == "inconclusive"
    base["mps_tpc_mask_or_credit_state"] = {"eligible": []}
    base["eligible_tpc_count"] = 0
    assert skill.analyze(base).status == "supported"
    base["eligible_tpc_count"] = 2
    base["victim_absent_from_candidate_tpcs"] = False
    assert skill.analyze(base).status == "ruled_out"


@pytest.mark.parametrize(
    "capacity,release,status",
    [
        ("insufficient_all", True, "supported"),
        ("verified_capacity_throughout", False, "ruled_out"),
        (None, None, "inconclusive"),
    ],
)
def test_stage4_resource_boundary_outcomes(capacity, release, status):
    """Residency follows verified per-SM capacity and release timing."""
    evidence = {
        "nvbit_cta_available": True,
        "resource_profile_available": True,
        "eligible_sm_capacity_during_delay": capacity,
        "victim_entry_follows_required_release": release,
        "occupancy_calculation_verified": True,
    }
    assert AnalyzeMpsResourceResidencyV1().analyze(evidence).status == status


def test_microbenchmark_needs_two_of_three_without_false_direct_claim():
    """Synthetic validation accepts 2/3 and forbids direct WDU claims."""
    trials = [
        {
            "stage": 1,
            "intended_status": "supported",
            "observed_status": value,
            "direct_wdu_claim": False,
        }
        for value in ("supported", "supported", "inconclusive")
    ]
    finding = (
        AnalyzeMpsTaskSlotAdmissionV1()
        .analyze(
            {
                "pending_grid_count": 8,
                "calibrated_task_slot_capacity": 8,
                "hidden_mps_task_count": 0,
                "task_slot_admission_signature": True,
                "microbenchmark_trials": trials,
            }
        )
        .to_dict()
    )
    assert finding["microbenchmark_validation"]["validated"] is True
    trials[0]["direct_wdu_claim"] = True
    finding = (
        AnalyzeMpsTaskSlotAdmissionV1()
        .analyze(
            {
                "pending_grid_count": 8,
                "calibrated_task_slot_capacity": 8,
                "hidden_mps_task_count": 0,
                "task_slot_admission_signature": True,
                "microbenchmark_trials": trials,
            }
        )
        .to_dict()
    )
    assert finding["microbenchmark_validation"]["validated"] is False


def test_legacy_schema_v2_artifact_remains_readable(tmp_path):
    """Coverage-aware loading does not require schema-v3 CTA tables."""
    database = tmp_path / "capsule_replay.sqlite"
    with sqlite3.connect(str(database)) as connection:
        connection.executescript(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value_json TEXT);"
            "CREATE TABLE capability(collector TEXT PRIMARY KEY, "
            "required INTEGER, available INTEGER, reason TEXT, version TEXT);"
            "CREATE TABLE microbenchmark_validation(stage INTEGER, "
            "condition_id TEXT, trial INTEGER, intended_status TEXT, "
            "observed_status TEXT, direct_wdu_claim INTEGER, "
            "evidence_json TEXT);"
            "CREATE TABLE kernel_resource_profile(launch_id TEXT PRIMARY KEY);"
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('schema_version', ?)",
            (json.dumps(2),),
        )
        connection.execute(
            "INSERT INTO capability VALUES "
            "('nvbit_cta', 0, 1, NULL, 'legacy')"
        )
    evidence = MpsContentionAnalyzer(tmp_path).load_evidence()
    assert evidence["nvbit_cta_available"] is True
    assert evidence["cta_coverage_complete"] is None
