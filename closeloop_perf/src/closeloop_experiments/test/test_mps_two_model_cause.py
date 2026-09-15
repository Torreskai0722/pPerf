"""Checks for the fixed Phase 4 GPU-only intervention matrix."""

from closeloop_experiments.mps_two_model_cause import (
    CONDITION_ORDER, MODEL_ORDER, _passive_domain_id, _run_config,
)


def test_passive_domain_wraps_after_ros_maximum():
    assert _passive_domain_id(1) == 225
    assert _passive_domain_id(8) == 232
    assert _passive_domain_id(9) == 33


def test_phase4_changes_only_ordered_mps_percentages():
    """The intervention, reversal, and negative control keep two models."""
    study = {
        "sha256": "digest",
        "data": {
            "output_root": "/tmp/out",
            "scene_tokens": ["scene-a", "scene-b"],
            "metadata_path": "/tmp/scene.json",
            "bag_directory": "/tmp/bags",
            "models": [{"id": model} for model in MODEL_ORDER],
            "conditions": {
                "discovery": [80, 80], "intervention": [80, 40],
                "reversal": [80, 80], "negative_control": [40, 80],
            },
            "replay_cpu_affinity": [12],
            "replay_cpu_thread_count": 1,
            "gpu": {"index": 0, "mps_enabled": True},
            "nsys_version": "1",
        },
    }
    configs = {
        condition: _run_config(study, condition, 1, ordinal)
        for ordinal, condition in enumerate(CONDITION_ORDER, 1)
    }
    assert all(len(config["models"]) == 2 for config in configs.values())
    assert all(
        config["replay"]["scene_tokens"] == ["scene-a", "scene-b"]
        for config in configs.values()
    )
    assert [
        model["mps_percentage"]
        for model in configs["intervention"]["models"]
    ] == [80, 40]
    assert [
        model["mps_percentage"]
        for model in configs["negative_control"]["models"]
    ] == [40, 80]
    assert configs["discovery"]["models"] == configs["reversal"]["models"]
