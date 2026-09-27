"""Immutable repeated-payload bags on a common 20/12 Hz schedule."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

from closeloop_testbed.fixed_manifest import validate_fixed_manifest
from .controlled_bags import ORIGIN_NS, fingerprint, reader, sha256, write_json


def occurrences(sources):
    """Preserve source identity while assigning each occurrence a unique stamp."""
    raw = {m: Path(s["cdr_path"]).read_bytes() for m, s in sources.items()}
    for modality, source in sources.items():
        if hashlib.sha256(raw[modality]).hexdigest() != source["original_serialized_sha256"]:
            raise ValueError("source CDR checksum differs")
    schedule = sorted((i * 10**9 // rate, order, modality, i)
                      for order, (modality, rate) in enumerate((("lidar", 20), ("image", 12)))
                      for i in range(40 * rate))
    for delta, _, modality, ordinal in schedule:
        timestamp = ORIGIN_NS + delta
        output = bytearray(raw[modality])
        struct.pack_into(("<" if output[1] == 1 else ">") + "iI", output, 4,
                         *divmod(timestamp, 10**9))
        row = {**sources[modality], "modality": modality,
               "occurrence_id": f"{modality}:{ordinal}",
               "output_bag_timestamp_ns": timestamp, "output_header_timestamp_ns": timestamp,
               "output_serialized_sha256": hashlib.sha256(output).hexdigest()}
        yield timestamp, row, bytes(output)


def validate_bag(path, verify_sources=True):
    """Read back every occurrence against the exact original CDR transform."""
    manifest = json.loads(Path(path).read_text())
    rows = validate_fixed_manifest(manifest)
    if verify_sources:
        for source in manifest["sources"].values():
            if sha256(source["source_bag"]) != source["source_bag_sha256"]:
                raise ValueError("original source bag changed")
    for file, expected in manifest["output_hashes"].items():
        if sha256(file) != expected:
            raise ValueError("fixed output bag changed")
    bag = reader(manifest["bag_path"])
    from .controlled_bags import metadata
    if {t.name: metadata(t) for t in bag.get_all_topics_and_types()} != {
            s["topic"]: s["topic_metadata"] for s in manifest["sources"].values()}:
        raise ValueError("fixed topic metadata changed")
    for row, (timestamp, expected, raw) in zip(rows, occurrences(manifest["sources"])):
        if row != expected or not bag.has_next() or bag.read_next() != (row["topic"], raw, timestamp):
            raise ValueError("fixed bag does not match original payload/schedule")
    if bag.has_next():
        raise ValueError("extra fixed bag occurrences")
    return {"valid": True, "occurrences": len(rows),
            "checks": ["source_hashes", "payload", "non_timestamp_fields", "topic_metadata",
                       "original_identity", "occurrence_identity", "timestamps", "schedule", "counts"]}


def construct_bag(sources, bag_root):
    """Write one content-addressed immutable bag and original-source index."""
    import rosbag2_py
    identity = {"schema": "fixed_input_bag_v1", "sources": sources,
                "window_ns": 40_000_000_000, "origin_ns": ORIGIN_NS,
                "rates_hz": {"lidar": 20, "image": 12}, "start_phase_ns": 0,
                "schedule_interval": "[0,40s)", "merge_ties": ["lidar", "image"],
                "timestamp_transform": "occurrence bag/header at nominal schedule; original timestamps retained separately"}
    directory = Path(bag_root).resolve() / fingerprint(identity)[:20]
    path = directory / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if any(prior.get(k) != v for k, v in identity.items()):
            raise ValueError("immutable fixed bag changed")
        validate_bag(path)
        return path
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    for source in sources.values():
        writer.create_topic(rosbag2_py.TopicMetadata(**source["topic_metadata"]))
    index = directory / "source_frames.jsonl"
    counts = Counter()
    with index.open("x") as stream:
        for timestamp, row, raw in occurrences(sources):
            writer.write(row["topic"], raw, timestamp)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            counts[row["topic"]] += 1
    del writer
    manifest = {**identity, "bag_path": str(output), "frame_index": str(index),
                "frame_index_sha256": sha256(index), "counts": dict(counts),
                "input_scenes": {m: {"scene_id": s["source_scene"], "scene_token": s["scene_token"]} for m, s in sources.items()},
                "output_hashes": {str(p): sha256(p) for p in sorted(output.iterdir())}}
    write_json(path, manifest)
    manifest["validation"] = validate_bag(path)
    write_json(path, manifest)
    for file in directory.rglob("*"):
        if file.is_file():
            file.chmod(0o444)
    return path
