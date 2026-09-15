"""Exact source-message identity for identity-aware capsule traces."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SOURCE_FRAME_SCHEMA = "source_frame_identity_v1"
DATASET_INPUT_SCHEMA = "dataset_input_identity_v2"
CAPTURE_INPUT_SCHEMA = "kernel_capsule_input_v2"


class SourceFrameError(RuntimeError):
    """Raised when a selected MCAP message cannot be proven exact."""


@dataclass(frozen=True)
class SourceFrameIdentity:
    """Identity and physical location of one serialized MCAP message."""

    ordinal: int
    storage_ordinal: int
    topic_ordinal: int
    bag_path: str
    bag_timestamp_ns: int
    ros_header_timestamp_ns: Optional[int]
    topic: str
    message_type: str
    serialization_format: str
    raw_cdr_sha256: str
    raw_cdr_size: int
    schema: str = SOURCE_FRAME_SCHEMA

    def validate(self) -> None:
        """Reject incomplete or non-CDR source identities."""
        if self.schema != SOURCE_FRAME_SCHEMA:
            raise SourceFrameError("source frame schema mismatch")
        if min(
            self.ordinal, self.storage_ordinal, self.topic_ordinal,
            self.bag_timestamp_ns, self.raw_cdr_size,
        ) < 0:
            raise SourceFrameError("source frame contains a negative field")
        if self.serialization_format != "cdr":
            raise SourceFrameError("source frame is not raw CDR")
        if len(self.raw_cdr_sha256) != 64:
            raise SourceFrameError("source frame SHA-256 is malformed")
        try:
            int(self.raw_cdr_sha256, 16)
        except ValueError as exc:
            raise SourceFrameError(
                "source frame SHA-256 is malformed"
            ) from exc
        if not self.topic or not self.message_type or not self.bag_path:
            raise SourceFrameError("source frame identity is incomplete")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SourceFrameIdentity":
        """Parse and validate one JSON index record."""
        fields = {
            name: value[name]
            for name in cls.__dataclass_fields__
            if name in value
        }
        result = cls(**fields)
        result.validate()
        return result

    @property
    def content_key(self) -> tuple:
        """Return the cross-model identity that must agree exactly."""
        return (
            self.topic,
            self.message_type,
            self.bag_timestamp_ns,
            self.ros_header_timestamp_ns,
            self.raw_cdr_sha256,
            self.raw_cdr_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe identity record."""
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class DatasetInputIdentity:
    """Dataset identity derivable from existing trace and run metadata."""

    ros_header_timestamp_ns: int
    topic: str
    message_type: str
    dataset_metadata_path: str
    schema: str = DATASET_INPUT_SCHEMA

    def validate(self) -> None:
        """Reject incomplete timestamp-based dataset identities."""
        if self.schema != DATASET_INPUT_SCHEMA:
            raise SourceFrameError("dataset input schema mismatch")
        if self.ros_header_timestamp_ns < 0:
            raise SourceFrameError("dataset input timestamp is negative")
        if not self.topic or not self.message_type:
            raise SourceFrameError("dataset input identity is incomplete")
        if not self.dataset_metadata_path:
            raise SourceFrameError("dataset metadata path is missing")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DatasetInputIdentity":
        """Parse and validate a timestamp-based identity."""
        result = cls(**{
            name: value[name]
            for name in cls.__dataclass_fields__
            if name in value
        })
        result.validate()
        return result

    @property
    def content_key(self) -> tuple:
        """Return the cross-model identity that must agree exactly."""
        return (
            self.topic,
            self.message_type,
            self.ros_header_timestamp_ns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe identity record."""
        self.validate()
        return asdict(self)


def input_identity_from_dict(value: Dict[str, Any]) -> Any:
    """Read a current dataset identity or a legacy v1 source identity."""
    if value.get("schema") == DATASET_INPUT_SCHEMA:
        return DatasetInputIdentity.from_dict(value)
    return SourceFrameIdentity.from_dict(value)


def message_header_timestamp_ns(message: Any) -> Optional[int]:
    """Return a message's header timestamp without serializing it again."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def load_source_frame_index(path: Path) -> List[SourceFrameIdentity]:
    """Load an identity index and require contiguous global ordinals."""
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(SourceFrameIdentity.from_dict(json.loads(line)))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise SourceFrameError(
                    f"invalid source frame index line {line_number}"
                ) from exc
    if [record.ordinal for record in records] != list(range(len(records))):
        raise SourceFrameError("source frame ordinals are not contiguous")
    return records


def _nuscenes_channel(topic: str) -> str:
    """Return the NuScenes sensor channel encoded by a ROS topic."""
    parts = [part for part in topic.split("/") if part]
    if not parts:
        raise SourceFrameError("source topic has no NuScenes channel")
    return parts[0]


def _nuscenes_data_root(identity: Any) -> Path:
    """Find the dataset beside the bag directory recorded by the trace."""
    if isinstance(identity, DatasetInputIdentity):
        metadata = Path(identity.dataset_metadata_path).resolve()
        return metadata.parent.parent
    for ancestor in Path(identity.bag_path).resolve().parents:
        candidate = ancestor / "nuscenes"
        if candidate.is_dir():
            return candidate
    raise SourceFrameError(
        "NuScenes data directory is not adjacent to the recorded bag"
    )


def extract_dataset_input(
    identity: Any,
    payload_path: Path,
    manifest_path: Path,
    *,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve one selected input by timestamp from NuScenes metadata."""
    identity.validate()
    if identity.ros_header_timestamp_ns is None:
        raise SourceFrameError("selected input has no unique ROS timestamp")
    if identity.ros_header_timestamp_ns % 1000:
        raise SourceFrameError(
            "selected timestamp is not a NuScenes timestamp"
        )
    root = Path(data_root or _nuscenes_data_root(identity)).resolve()
    timestamp_us = identity.ros_header_timestamp_ns // 1000
    channel = _nuscenes_channel(identity.topic)
    matches: Dict[Path, Dict[str, Any]] = {}
    for metadata_path in root.glob("*/sample_data.json"):
        try:
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceFrameError(
                f"invalid NuScenes sample-data metadata: {metadata_path}"
            ) from exc
        for record in records:
            filename = Path(str(record.get("filename", "")))
            if (
                int(record.get("timestamp", -1)) == timestamp_us
                and channel in filename.parts
            ):
                source_path = (
                    metadata_path.parent.parent / filename
                ).resolve()
                try:
                    source_path.relative_to(root)
                except ValueError as exc:
                    raise SourceFrameError(
                        "NuScenes metadata points outside the data directory"
                    ) from exc
                matches[source_path] = record
    if len(matches) != 1:
        raise SourceFrameError(
            "selected timestamp does not resolve to one NuScenes input"
        )
    source_path, record = next(iter(matches.items()))
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise SourceFrameError(
            f"selected NuScenes payload is unreadable: {source_path}"
        ) from exc
    encodings = {
        "sensor_msgs/msg/CompressedImage": "nuscenes_compressed_image",
        "sensor_msgs/msg/PointCloud2": "nuscenes_pointcloud_xyzir_f32",
    }
    try:
        payload_encoding = encodings[identity.message_type]
    except KeyError as exc:
        raise SourceFrameError(
            f"unsupported NuScenes input type {identity.message_type}"
        ) from exc
    observed_hash = hashlib.sha256(payload).hexdigest()
    payload_path = Path(payload_path)
    manifest_path = Path(manifest_path)
    payload_temporary = payload_path.with_name(
        "." + payload_path.name + ".tmp"
    )
    payload_temporary.write_bytes(payload)
    payload_temporary.replace(payload_path)
    payload_path.chmod(0o444)
    manifest = {
        "schema": CAPTURE_INPUT_SCHEMA,
        "source_frame": identity.to_dict(),
        "payload_path": str(payload_path.resolve()),
        "payload_encoding": payload_encoding,
        "dataset_payload_sha256": observed_hash,
        "dataset_payload_size": len(payload),
        "dataset_source": {
            "channel": channel,
            "filename": str(record["filename"]),
            "sample_data_token": str(record.get("token", "")),
            "timestamp_us": timestamp_us,
        },
    }
    manifest_temporary = manifest_path.with_name(
        "." + manifest_path.name + ".tmp"
    )
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_path)
    manifest_path.chmod(0o444)
    return manifest


def load_capture_input(path: Path) -> tuple:
    """Load and re-verify a dataset payload before model creation."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != CAPTURE_INPUT_SCHEMA:
        raise SourceFrameError("capture input schema mismatch")
    identity = input_identity_from_dict(value["source_frame"])
    payload = Path(value["payload_path"]).read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if (
        observed != value.get("dataset_payload_sha256")
        or len(payload) != int(value.get("dataset_payload_size", -1))
    ):
        raise SourceFrameError("dataset input payload SHA-256 mismatch")
    return identity, payload, str(value.get("payload_encoding", ""))
