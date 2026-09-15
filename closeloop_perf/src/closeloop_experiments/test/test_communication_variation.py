"""Checks for the fixed Phase 2 communication matrix."""

from closeloop_experiments.communication_variation import (
    _run_config, condition_from_run_id
)


def test_phase2_keeps_two_scene_order_and_changes_only_cpu_threads():
    """The intervention preserves workload identity and depth one."""
    study = {
        "sha256": "digest",
        "data": {
            "output_root": "/tmp/out",
            "scene_tokens": ["scene-a", "scene-b"],
            "metadata_path": "/tmp/scene.json",
            "bag_directory": "/tmp/bags",
            "model": {
                "id": "model", "input_queue_depth": 1,
                "cpu_affinity": [0, 1],
            },
            "conditions": {
                "baseline": {
                    "model_cpu_thread_count": 3,
                    "replay_cpu_affinity": [2],
                    "replay_cpu_thread_count": 1,
                },
                "cpu1_intervention": {
                    "model_cpu_thread_count": 1,
                    "replay_cpu_affinity": [2],
                    "replay_cpu_thread_count": 1,
                },
            },
            "gpu": {"index": 0, "mps_enabled": False},
            "nsys_version": "1",
        },
    }
    baseline = _run_config(study, "baseline", 1, 0)
    intervention = _run_config(study, "cpu1_intervention", 1, 1)
    assert baseline["replay"]["scene_tokens"] == ["scene-a", "scene-b"]
    assert baseline["models"][0]["input_queue_depth"] == 1
    baseline["models"][0].pop("cpu_thread_count")
    intervention["models"][0].pop("cpu_thread_count")
    baseline["run"] = intervention["run"]
    baseline["ros"]["domain_id"] = intervention["ros"]["domain_id"]
    assert baseline == intervention


def test_reversal_condition_name_is_not_mistaken_for_replicate_suffix():
    """The `r` starting reversal is part of the condition name."""
    assert condition_from_run_id("comm-reversal-r3") == "reversal"
