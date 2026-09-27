"""Strict fixed-input occurrence identity validation for controlled replay."""

import hashlib
import json
from pathlib import Path


def validate_fixed_manifest(manifest):
    """Allow repeated sources only with a complete, unique nominal schedule."""
    if (manifest["schema"] != "fixed_input_bag_v1"
            or manifest["window_ns"] != 40_000_000_000
            or manifest["rates_hz"] != {"lidar": 20, "image": 12}
            or manifest["origin_ns"] != 1_000_000_000_000
            or set(manifest["sources"]) != {"lidar", "image"}):
        raise ValueError("invalid fixed-input protocol")
    raw = Path(manifest["frame_index"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["frame_index_sha256"]:
        raise ValueError("fixed occurrence index hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    counters = {m: 0 for m in manifest["sources"]}
    previous = None
    for row in rows:
        modality = row["modality"]
        source = manifest["sources"][modality]
        ordinal = counters[modality]
        timestamp = manifest["origin_ns"] + ordinal * 10**9 // manifest["rates_hz"][modality]
        if (row["occurrence_id"] != f"{modality}:{ordinal}"
                or row["output_bag_timestamp_ns"] != timestamp
                or row["output_header_timestamp_ns"] != timestamp
                or any(row.get(k) != source.get(k) for k in (
                    "source_frame_id", "source_scene", "source_bag", "source_bag_sha256",
                    "original_bag_timestamp_ns", "original_header_timestamp_ns", "payload_sha256",
                    "original_serialized_sha256", "non_timestamp_sha256", "topic", "frame_id"))):
            raise ValueError("fixed occurrence/source identity or timestamp differs")
        order = (timestamp, 0 if modality == "lidar" else 1)
        if previous is not None and order <= previous:
            raise ValueError("fixed occurrence order is not strictly increasing")
        previous = order
        counters[modality] += 1
    if counters != {"lidar": 800, "image": 480}:
        raise ValueError("fixed occurrence counts differ")
    if manifest["counts"] != {s["topic"]: counters[m] for m, s in manifest["sources"].items()}:
        raise ValueError("fixed topic counts differ")
    for modality, source in manifest["sources"].items():
        if manifest["input_scenes"][modality] != {"scene_id": source["source_scene"], "scene_token": source["scene_token"]}:
            raise ValueError("fixed actual scene identity differs")
    return rows
