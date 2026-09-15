"""Tests for relay-to-model timing joins."""

import csv
import json

from closeloop_analyzer.communication import analyze_communication


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_communication_join_records_latency_and_unmatched(tmp_path):
    """Timestamp occurrence joins survive unequal relay and callback counts."""
    relay = {
        "segment_id": "scene-0-bag-0",
        "scene_index": 0,
        "scene_token": "scene-a",
        "input_topic": "/camera/front",
        "source_topic": "/CAM_FRONT/image_rect_compressed",
        "raw_topic": "/closeloop/raw/CAM_FRONT/image_rect_compressed",
        "original_source_timestamp_ns": 10,
        "timestamp_occurrence": 0,
        "relay_pre_publish_monotonic_ns": 100,
        "relay_post_publish_monotonic_ns": 110,
        "duplicate": False,
        "out_of_order": False,
    }
    _write_jsonl(tmp_path / "communication_relay.jsonl", [relay, {
        **relay, "original_source_timestamp_ns": 20,
        "relay_pre_publish_monotonic_ns": 200,
        "relay_post_publish_monotonic_ns": 210,
    }])
    _write_jsonl(tmp_path / "model_image_inputs.jsonl", [{
        "segment_id": "scene-0-bag-0",
        "input_topic": "/camera/front",
        "ros_header_timestamp_ns": 10,
        "timestamp_occurrence": 0,
        "model_callback_entry_monotonic_ns": 150,
        "model_callback_exit_monotonic_ns": 180,
        "previous_callback_exit_monotonic_ns": 120,
    }])
    (tmp_path / "testbed_result.json").write_text(
        '{"playback_intervals": []}', encoding="utf-8"
    )
    (tmp_path / "communication_relay_status.json").write_text(
        '{"messages": 2}', encoding="utf-8"
    )
    summary = analyze_communication(
        {"models": [{"id": "image", "input_topic": "/camera/front"}]},
        tmp_path,
    )
    assert summary["matched"] == 1
    assert summary["dropped_or_overwritten"] == 1
    rows = list(csv.DictReader(
        (tmp_path / "communication_latency.csv").open(encoding="utf-8")
    ))
    assert rows[0]["communication_latency_ns"] == "50"
    assert rows[0]["previous_callback_busy_residual_ns"] == "20"
    assert rows[0]["post_busy_communication_latency_ns"] == "30"
