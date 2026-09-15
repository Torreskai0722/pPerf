"""Tests for trace identity and timestamp-based dataset extraction."""

import hashlib
import json

import pytest

from closeloop_profiler.source_frames import (
    DatasetInputIdentity, SourceFrameError, SourceFrameIdentity,
    extract_dataset_input,
    load_capture_input, load_source_frame_index,
)


def test_legacy_v1_index_remains_readable(tmp_path):
    """Old raw-CDR identities remain readable without opening their MCAP."""
    index = tmp_path / "source_frames.jsonl"
    identity = SourceFrameIdentity(
        0, 2, 1, "/old/source.mcap", 30_000, 30_000,
        "/CAM_FRONT/image_rect_compressed",
        "sensor_msgs/msg/CompressedImage", "cdr", "0" * 64, 100,
    )
    index.write_text(json.dumps(identity.to_dict()) + "\n")
    assert load_source_frame_index(index) == [identity]


def test_dataset_resolves_by_timestamp_without_mcap(tmp_path):
    """Selection identity resolves one hashed dataset payload by timestamp."""

    dataset = tmp_path / "nuscenes"
    version = dataset / "v1.0-mini"
    image = dataset / "sweeps" / "CAM_FRONT" / "frame.jpg"
    version.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg-data")
    (version / "sample_data.json").write_text(json.dumps([{
        "token": "sample-data-1",
        "timestamp": 30,
        "filename": "sweeps/CAM_FRONT/frame.jpg",
    }]))
    identity = DatasetInputIdentity(
        ros_header_timestamp_ns=30_000,
        topic="/CAM_FRONT/image_rect_compressed",
        message_type="sensor_msgs/msg/CompressedImage",
        dataset_metadata_path=str(version / "scene.json"),
    )
    manifest = extract_dataset_input(
        identity, tmp_path / "input.bin", tmp_path / "input.json",
        data_root=dataset,
    )
    loaded_identity, payload, encoding = load_capture_input(
        tmp_path / "input.json"
    )
    assert loaded_identity == identity
    assert payload == b"jpeg-data"
    assert encoding == "nuscenes_compressed_image"
    assert manifest["dataset_source"]["sample_data_token"] == "sample-data-1"
    assert manifest["source_frame"]["schema"] == "dataset_input_identity_v2"


def test_capture_loader_rejects_changed_dataset_payload(tmp_path):
    """No client can load dataset bytes changed after extraction."""
    payload = tmp_path / "input.bin"
    payload.write_bytes(b"changed")
    manifest = {
        "schema": "kernel_capsule_input_v2",
        "source_frame": SourceFrameIdentity(
            0, 0, 0, "/bag", 1, 1, "/CAM_FRONT/image",
            "sensor_msgs/msg/CompressedImage", "cdr", "0" * 64, 1,
        ).to_dict(),
        "payload_path": str(payload),
        "payload_encoding": "nuscenes_compressed_image",
        "dataset_payload_sha256": hashlib.sha256(b"original").hexdigest(),
        "dataset_payload_size": len(b"original"),
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(SourceFrameError, match="dataset input payload"):
        load_capture_input(path)
