"""Checks for the fixed Phase 3 three-model MPS matrix."""

from closeloop_experiments.mps_three_model import (
    MODEL_ORDER, MPS_CONFIGURATIONS, _run_config
)


def test_phase3_matrix_has_exact_order_and_no_default_client():
    """All ten cells preserve model, scene, and explicit percentage order."""
    study = {
        "sha256": "digest",
        "data": {
            "output_root": "/tmp/out",
            "scene_tokens": ["scene-a", "scene-b"],
            "metadata_path": "/tmp/scene.json",
            "bag_directory": "/tmp/bags",
            "models": [{"id": model} for model in MODEL_ORDER],
            "replay_cpu_affinity": [12],
            "replay_cpu_thread_count": 1,
            "gpu": {"index": 0, "mps_enabled": True},
            "nsys_version": "1",
        },
    }
    configs = [
        _run_config(study, number, triple)
        for number, triple in enumerate(MPS_CONFIGURATIONS, 1)
    ]
    assert len(configs) == 10
    assert len({config["run"]["id"] for config in configs}) == 10
    for config, triple in zip(configs, MPS_CONFIGURATIONS):
        assert config["replay"]["scene_tokens"] == ["scene-a", "scene-b"]
        assert tuple(model["id"] for model in config["models"]) == MODEL_ORDER
        assert tuple(
            model["mps_percentage"] for model in config["models"]
        ) == triple
        assert all(40 <= value <= 90 for value in triple)
