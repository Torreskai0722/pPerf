"""Explicit controlled bag selection and modality-specific NVTX identities."""

import hashlib
import json

import pytest

from closeloop_testbed.input_handling import model_input_segment
from closeloop_testbed.replayer import playback_segments


def test_controlled_replay_uses_explicit_bag_and_both_own_scene_identities(tmp_path):
    bag = tmp_path / "actually-constructed.mcap"
    bag.write_bytes(b"checked bag")
    scenes = {"lidar": {"scene_id": "A", "scene_token": "lidar-a"},
              "image": {"scene_id": "B", "scene_token": "camera-b"}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "controlled_bag_v1", "validation": {"valid": True},
                               "input_scenes": scenes, "bag_path": str(bag), "window_ns": 19_250_761_000,
                               "output_hashes": {str(bag): hashlib.sha256(bag.read_bytes()).hexdigest()}}))
    config = {"controlled_bag_manifest": str(path), "controlled_bag_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "rate": 1.0, "repeat_count": 1, "playback_mode": "full"}
    segment, = playback_segments(config)
    assert segment["bag_path"] == str(bag)
    assert model_input_segment(segment, "lidar")["scene_token"] == "lidar-a"
    assert model_input_segment(segment, "image")["scene_token"] == "camera-b"
    assert "scene_token" not in segment
    legacy = {"scene_token": "legacy", "scene_name": "original"}
    assert model_input_segment(legacy, "image") == legacy
    with pytest.raises(ValueError, match="one-pass"):
        playback_segments({**config, "repeat_count": 3})
    bag.write_bytes(b"changed bag")
    with pytest.raises(ValueError, match="bag hash mismatch"):
        playback_segments(config)
