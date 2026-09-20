"""Construct and read back immutable, common-window MCAP inputs."""

from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import struct


TOPICS = {"lidar": "/LIDAR_TOP", "image": "/CAM_FRONT/image_rect_compressed"}
TYPES = {"lidar": "sensor_msgs/msg/PointCloud2",
         "image": "sensor_msgs/msg/CompressedImage"}
WINDOW_NS = 19_250_761_000
ORIGIN_NS = 1_000_000_000_000


def sha256(path):
    """Hash large bags without retaining their bytes in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically replace a generated JSON record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fingerprint(value):
    """Hash a canonical construction or experiment identity."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reader(path):
    """Open the installed rosbag2 MCAP implementation."""
    import rosbag2_py
    result = rosbag2_py.SequentialReader()
    result.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    return result


def metadata(topic):
    """Preserve the topic metadata supported by the installed ROS version."""
    return {key: getattr(topic, key) for key in (
        "name", "type", "serialization_format", "offered_qos_profiles")}


def inspect_source(scene, condition):
    """Verify the complete source's timestamps, types, and content hashes."""
    files = sorted(Path(condition["bag_directory"]).glob("*.mcap"))
    if not files:
        raise ValueError(f"no source MCAPs for {scene}")
    start = end = previous = None
    topics, counts, hashes = {}, Counter(), {}
    for path in files:
        hashes[str(path)] = sha256(path)
        source = reader(path)
        for topic in source.get_all_topics_and_types():
            info = metadata(topic)
            if topic.name in topics and topics[topic.name] != info:
                raise ValueError(f"topic metadata changes in {scene}")
            topics[topic.name] = info
        while source.has_next():
            topic, _raw, timestamp = source.read_next()
            if previous is not None and timestamp < previous:
                raise ValueError(f"nonmonotonic source bag: {scene}")
            start = timestamp if start is None else start
            end = previous = timestamp
            counts[topic] += 1
    for modality, topic in TOPICS.items():
        if (topics.get(topic, {}).get("type") != TYPES[modality]
                or topics[topic]["serialization_format"] != "cdr"
                or not counts[topic]):
            raise ValueError(f"missing or incompatible {modality} in {scene}")
    return {"scene_id": scene, "scene_token": condition["scene_token"],
            "files": hashes, "first_timestamp_ns": start,
            "last_timestamp_ns": end, "duration_ns": end - start,
            "topics": topics, "counts": dict(counts)}


def shifted_stream(source, modality, window_ns=WINDOW_NS):
    """Yield source records with equal bag/header shifts and exact identity."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    topic = TOPICS[modality]
    message_type = get_message(TYPES[modality])
    offset = ORIGIN_NS - source["first_timestamp_ns"]
    ordinal, previous_header = 0, None
    for path, bag_hash in source["files"].items():
        bag = reader(path)
        while bag.has_next():
            name, raw, timestamp = bag.read_next()
            if name != topic:
                continue
            topic_ordinal = ordinal
            ordinal += 1
            if timestamp - source["first_timestamp_ns"] > window_ns:
                continue
            message = deserialize_message(raw, message_type)
            stamp = message.header.stamp
            header_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if previous_header is not None and header_ns < previous_header:
                raise ValueError(f"nonmonotonic header: {path}/{topic}")
            previous_header = header_ns
            shifted = header_ns + offset
            if shifted <= 0:
                raise ValueError("shared origin produces a nonpositive header")
            # Both supported messages start with Header. Keep original CDR
            # padding too: Fast-CDR reserialization can leave padding undefined.
            if raw[:2] not in (b"\x00\x00", b"\x00\x01"):
                raise ValueError("unsupported CDR encapsulation")
            endian = "<" if raw[1] == 1 else ">"
            if struct.unpack_from(endian + "iI", raw, 4) != (stamp.sec, stamp.nanosec):
                raise ValueError("CDR header layout differs from decoded header")
            output = bytearray(raw)
            struct.pack_into(endian + "iI", output, 4, *divmod(shifted, 1_000_000_000))
            output = bytes(output)
            non_timestamp = hashlib.sha256(raw[:4] + raw[12:]).hexdigest()
            identity = {
                "source_frame_id": f"{source['scene_id']}:{modality}:{topic_ordinal}",
                "source_scene": source["scene_id"], "source_bag": path,
                "source_bag_sha256": bag_hash, "topic": topic,
                "topic_ordinal": topic_ordinal, "frame_id": message.header.frame_id,
                "original_bag_timestamp_ns": timestamp,
                "original_header_timestamp_ns": header_ns,
                "output_bag_timestamp_ns": timestamp + offset,
                "output_header_timestamp_ns": shifted,
                "timestamp_shift_ns": offset,
                "payload_sha256": hashlib.sha256(bytes(message.data)).hexdigest(),
                "original_serialized_sha256": hashlib.sha256(raw).hexdigest(),
                "output_serialized_sha256": hashlib.sha256(output).hexdigest(),
                "non_timestamp_sha256": non_timestamp,
            }
            yield timestamp + offset, identity, output


def merged_streams(sources, window_ns):
    """Merge by bag time; modality order breaks ties, preserving stream order."""
    streams = [shifted_stream(sources[modality], modality, window_ns)
               for modality in TOPICS]
    return heapq.merge(*streams, key=lambda row: row[0])


def validate_bag(manifest_path, verify_sources=True):
    """Read every output record and compare it to its transformed source."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    manifest = json.loads(Path(manifest_path).read_text())
    if verify_sources:
        for source in manifest["sources"].values():
            for path, expected in source["files"].items():
                if sha256(path) != expected:
                    raise ValueError(f"source bag changed: {path}")
    if sha256(manifest["frame_index"]) != manifest["frame_index_sha256"]:
        raise ValueError("controlled frame index hash mismatch")
    for path, expected in manifest["output_hashes"].items():
        if sha256(path) != expected:
            raise ValueError(f"controlled bag hash mismatch: {path}")
    bag = reader(manifest["bag_path"])
    expected_metadata = {TOPICS[m]: s["topics"][TOPICS[m]]
                         for m, s in manifest["sources"].items()}
    if {t.name: metadata(t) for t in bag.get_all_topics_and_types()} != expected_metadata:
        raise ValueError("controlled topic metadata differs")
    types = {TOPICS[m]: get_message(TYPES[m]) for m in TOPICS}
    counts, streams = Counter(), {topic: hashlib.sha256() for topic in types}
    expected = iter(merged_streams(manifest["sources"], manifest["window_ns"]))
    previous = None
    with Path(manifest["frame_index"]).open() as index:
        for line in index:
            row = json.loads(line)
            if not bag.has_next():
                raise ValueError("controlled bag ends before index")
            topic, raw, timestamp = bag.read_next()
            reference = next(expected, None)
            if reference is None or reference != (timestamp, row, raw):
                raise ValueError("controlled record differs from source transformation")
            if previous is not None and timestamp < previous:
                raise ValueError("controlled bag is nonmonotonic")
            previous = timestamp
            message = deserialize_message(raw, types[topic])
            stamp = message.header.stamp
            if stamp.sec * 1_000_000_000 + stamp.nanosec != row["output_header_timestamp_ns"]:
                raise ValueError("controlled header timestamp differs")
            if hashlib.sha256(raw[:4] + raw[12:]).hexdigest() != row["non_timestamp_sha256"]:
                raise ValueError("controlled non-timestamp fields differ")
            counts[topic] += 1
            streams[topic].update(json.dumps(row, sort_keys=True).encode())
    if bag.has_next() or next(expected, None) is not None:
        raise ValueError("controlled bag/index/source counts differ")
    if dict(counts) != manifest["counts"]:
        raise ValueError("controlled bag counts differ")
    return {"valid": True, "counts": dict(counts),
            "stream_sha256": {t: h.hexdigest() for t, h in streams.items()},
            "checks": ["payload", "non_timestamp_fields", "frame_id", "topic_metadata",
                       "bag_and_header_shift", "intervals", "monotonicity", "counts"]}


def construct_bag(sources, bag_root, window_ns=WINDOW_NS):
    """Deduplicate by source identity and settings, independently of runs."""
    import rosbag2_py
    if any(source["duration_ns"] < window_ns for source in sources.values()):
        raise ValueError("source is shorter than the frozen common window")
    identity = {"schema": "controlled_bag_v1", "sources": sources,
                "window_ns": window_ns, "origin_ns": ORIGIN_NS,
                "window": "inclusive, relative to first bag timestamp",
                "merge_ties": list(TOPICS), "topics": TOPICS,
                "serialization_transform": "patch_first_header_CDR_timestamp_only_v1",
                "timestamp_transform": "same additive offset for bag and header"}
    directory = Path(bag_root).resolve() / fingerprint(identity)[:20]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError(f"immutable controlled bag identity differs: {directory}")
        validation = validate_bag(manifest_path)
        if validation != manifest["validation"]:
            raise ValueError("controlled validation differs from retained evidence")
        return manifest_path
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    for modality, source in sources.items():
        writer.create_topic(rosbag2_py.TopicMetadata(**source["topics"][TOPICS[modality]]))
    index_path = directory / "source_frames.jsonl"
    counts = Counter()
    with index_path.open("x") as index:
        for timestamp, row, raw in merged_streams(sources, window_ns):
            writer.write(row["topic"], raw, timestamp)
            index.write(json.dumps(row, sort_keys=True) + "\n")
            counts[row["topic"]] += 1
    del writer
    manifest = {**identity, "bag_path": str(output), "frame_index": str(index_path),
                "frame_index_sha256": sha256(index_path), "counts": dict(counts),
                "input_scenes": {m: {"scene_id": s["scene_id"], "scene_token": s["scene_token"]}
                                 for m, s in sources.items()},
                "output_hashes": {str(p): sha256(p) for p in sorted(output.iterdir())}}
    write_json(manifest_path, manifest)
    manifest["validation"] = validate_bag(manifest_path)
    write_json(manifest_path, manifest)
    for path in directory.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return manifest_path
