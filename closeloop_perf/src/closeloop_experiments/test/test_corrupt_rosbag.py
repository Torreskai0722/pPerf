"""Focused checks for full-duration Astra/LISA rain corruption."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from closeloop_experiments.corrupt_rosbag import (
    CAMERA_TOPIC,
    LIDAR_TOPIC,
    CorruptionError,
    _corrupt_lidar_rain,
    _load_nuscenes_metadata,
    _point_records,
    _pylisa_initializer,
    _pylisa_point,
    complete_depth_ip_basic,
    derived_seed,
    lidar_to_camera,
    pair_camera_lidar_timestamps,
    paired_timestamps,
    project_depth,
    rain_rate_slug,
    resolve_astra_cache,
    validate_manifest,
)


def test_header_pairing_enforces_fifty_milliseconds():
    """The hard limit is inclusive and an unmatched camera fails."""
    assert pair_camera_lidar_timestamps([100], [50_000_100]) == [
        (100, 50_000_100, 50_000_000)
    ]
    assert paired_timestamps([100], [50_000_100]) == [(100, 50_000_100)]
    with pytest.raises(CorruptionError, match="limit is 50000000 ns"):
        pair_camera_lidar_timestamps([100], [50_000_101])


def test_motion_compensated_projection_uses_nearest_depth():
    """Identity metadata and the z-buffer preserve the nearest point."""
    identity = {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}
    calibrated = {"lidar": identity, "camera": identity}
    poses = {"lidar_pose": identity, "camera_pose": identity}
    lidar = {
        "calibrated_sensor_token": "lidar", "ego_pose_token": "lidar_pose"
    }
    camera = {
        "calibrated_sensor_token": "camera", "ego_pose_token": "camera_pose"
    }
    points = np.array([
        [0, 0, 5], [0, 0, 2], [1, 0, 1], [0, 0, -1],
    ], dtype=float)
    transformed = lidar_to_camera(
        points, lidar, camera, calibrated, poses
    )
    np.testing.assert_array_equal(transformed, points)
    depth = project_depth(
        transformed, np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
        width=3, height=3,
    )
    assert depth[1, 1] == 2
    assert depth[1, 2] == 1
    assert np.count_nonzero(depth) == 2


def test_ip_basic_completion_is_full_finite_and_deterministic():
    """The specified completion stages leave no holes."""
    sparse = np.zeros((40, 60), dtype=np.float32)
    sparse[35, 30] = 10
    sparse[20, 10] = 30
    first = complete_depth_ip_basic(sparse)
    second = complete_depth_ip_basic(sparse)
    assert np.isfinite(first).all()
    assert np.all(first > 0)
    np.testing.assert_array_equal(first, second)


def test_fractional_rate_has_canonical_path_without_rounding():
    """A fractional CLI float remains the physical 7.5 mm/h rate."""
    rate = float("7.5")
    assert rate == 7.5
    assert rain_rate_slug(rate) == "7p5"
    assert rain_rate_slug(15.0) == "15"
    with pytest.raises(CorruptionError):
        rain_rate_slug(float("nan"))


def test_astra_cache_defaults_to_an_isolated_scene_path():
    """Each scene avoids particle-duration conflicts by default."""
    expected = Path("/mmdetection3d_ros2/data/astra_particles/scene-0001")
    assert resolve_astra_cache("scene-0001") == expected
    assert resolve_astra_cache("scene-0001", "/tmp/cache") == Path(
        "/tmp/cache"
    )
    with pytest.raises(CorruptionError, match="invalid scene name"):
        resolve_astra_cache("../scene-0001")


def test_metadata_rejects_a_scene_name_token_mismatch(tmp_path):
    """A valid token cannot silently label output as another scene."""
    files = (
        "scene.json", "sample.json", "sample_data.json", "sensor.json",
        "calibrated_sensor.json", "ego_pose.json",
    )
    for name in files:
        values = (
            [{"token": "token", "name": "scene-0001"}]
            if name == "scene.json" else []
        )
        (tmp_path / name).write_text(json.dumps(values))
    with pytest.raises(CorruptionError, match="scene name/token mismatch"):
        _load_nuscenes_metadata(tmp_path, "token", "scene-0002")


def _pointcloud():
    fields = [
        SimpleNamespace(name=name, offset=index * 4, datatype=7, count=1)
        for index, name in enumerate(("x", "y", "z", "intensity", "ring"))
    ]
    values = np.array([
        (1, 2, 3, 255, 9),
        (4, 5, 6, 127.5, 10),
        (7, 8, 9, 0, 11),
    ], dtype=[(name, "<f4") for name in ("x", "y", "z", "intensity", "ring")])
    return SimpleNamespace(
        fields=fields, is_bigendian=False, point_step=20, row_step=60,
        width=3, height=1, is_dense=True, data=values.tobytes(),
    )


class _FakePool:
    def __init__(self):
        self.tasks = None

    def map(self, _function, tasks, chunksize):
        assert chunksize == 256
        self.tasks = tasks
        output = []
        for index, (point, _rate, seed) in enumerate(tasks):
            label = 0 if index == 1 else 2
            output.append([
                point[0] + (seed % 7) / 1000,
                point[1], point[2], point[3] / 2, label,
            ])
        return output


def test_lisa_normalization_preserves_fields_and_is_deterministic():
    """The wrapper normalizes intensity, drops label zero, and retains ring."""
    first_pool = _FakePool()
    first, stats = _corrupt_lidar_rain(
        _pointcloud(), first_pool, 7.5, 4200, 123
    )
    second_pool = _FakePool()
    second, second_stats = _corrupt_lidar_rain(
        _pointcloud(), second_pool, 7.5, 4200, 123
    )
    assert [task[0][3] for task in first_pool.tasks] == [1.0, 0.5, 0.0]
    assert first.data == second.data
    assert stats == second_stats
    records = _point_records(first)
    np.testing.assert_array_equal(records["ring"], [9, 11])
    np.testing.assert_allclose(records["intensity"], [127.5, 0])
    assert first.width == 2
    assert stats["lost_point_count"] == 1
    assert stats["output_points_finite"] is True


def test_official_lisa_point_seed_is_repeatable():
    """The repository's strongest-return implementation honors a point seed."""
    lisa_root = Path(__file__).resolve().parents[4] / "LISA"
    _pylisa_initializer(lisa_root, 7.5)
    task = (np.array([10.0, 0.0, 0.0, 0.5]), 7.5, 12345)
    np.testing.assert_array_equal(_pylisa_point(task), _pylisa_point(task))


def _record(topic, timestamp):
    record = {
        "topic": topic,
        "bag_timestamp_ns": timestamp,
        "output_bag_timestamp_ns": timestamp,
        "header_timestamp_ns": timestamp,
        "output_header_timestamp_ns": timestamp,
        "source_payload_sha256": "source",
        "output_payload_sha256": "output",
        "metadata_sample_data_token": f"metadata-{topic}-{timestamp}",
    }
    if topic == CAMERA_TOPIC:
        record["dense_depth_coverage"] = 1.0
        record["paired_lidar_header_timestamp_ns"] = timestamp
        record["lidar_pairing_delta_ns"] = 0
    else:
        record["output_points_finite"] = True
    return record


def test_full_duration_manifest_validation():
    """Full rain validation derives counts and duration from any scene."""
    camera_count, lidar_count = 3, 5
    duration_ns = 12_345_678_901
    window_start = 1_000_000_000
    records = [
        _record(CAMERA_TOPIC, index) for index in range(camera_count)
    ] + [
        _record(LIDAR_TOPIC, index) for index in range(lidar_count)
    ]
    manifest = {
        "schema_version": 1,
        "duration_ns": duration_ns,
        "duration_seconds": duration_ns / 1_000_000_000,
        "source_message_count": 17,
        "window_start_bag_timestamp_ns": window_start,
        "window_end_bag_timestamp_ns": window_start + duration_ns,
        "corruption": {
            "type": "rain", "severity": None, "full_duration": True,
            "rain_rate_mm_per_hour": 7.5, "seed": 4200,
        },
        "sensor_records": records,
        "preserved_stream_source_sha256": "same",
        "preserved_stream_output_sha256": "same",
        "pairing": {
            "tolerance_ns": 50_000_000,
            "camera_messages": camera_count,
            "lidar_messages": lidar_count,
            "paired_camera_messages": camera_count,
            "depth_pair_count": camera_count,
            "unique_depth_source_lidar_messages": camera_count,
            "maximum_delta_ns": 0,
        },
    }
    assert validate_manifest(manifest) == []
    broken = deepcopy(manifest)
    broken["duration_ns"] -= 1
    assert "full-duration differs from bag timestamp window" in \
        validate_manifest(broken)


def test_full_duration_clean_manifest_validation():
    """A full clean manifest proves byte-identical counts and streams."""
    records = [_record(CAMERA_TOPIC, 100), _record(LIDAR_TOPIC, 100)]
    for record in records:
        record["output_payload_sha256"] = record["source_payload_sha256"]
    manifest = {
        "schema_version": 1,
        "duration_ns": 19_249_668_000,
        "duration_seconds": 19.249668,
        "source_message_count": 10,
        "output_message_count": 10,
        "window_start_bag_timestamp_ns": 1_000,
        "window_end_bag_timestamp_ns": 19_249_669_000,
        "source_bag_sha256": "bag",
        "output_bag_files": [{"path": "scene.mcap", "sha256": "bag"}],
        "corruption": {
            "type": "clean", "severity": 0, "full_duration": True,
        },
        "sensor_records": records,
        "source_stream_sha256": "stream",
        "output_stream_sha256": "stream",
        "preserved_stream_source_sha256": "same",
        "preserved_stream_output_sha256": "same",
        "pairing": {
            "tolerance_ns": 50_000_000,
            "camera_messages": 1,
            "lidar_messages": 1,
            "paired_camera_messages": 1,
            "depth_pair_count": 1,
            "unique_depth_source_lidar_messages": 1,
            "maximum_delta_ns": 0,
        },
    }
    assert validate_manifest(manifest) == []
    manifest["output_stream_sha256"] = "changed"
    assert "full clean stream changed" in validate_manifest(manifest)


def test_full_duration_fixed_manifest_validation():
    """Fixed manifests require repeated asset hashes and point layout."""
    camera = _record(CAMERA_TOPIC, 100)
    lidar = _record(LIDAR_TOPIC, 100)
    camera.update(fixed_content_sha256="a" * 64,
                  output_content_sha256="a" * 64)
    lidar.update({
        "fixed_content_sha256": "b" * 64,
        "output_content_sha256": "b" * 64,
        "output_point_count": 34_752,
        "output_height": 1,
        "output_width": 34_752,
        "output_point_step": 20,
        "output_row_step": 695_040,
    })
    manifest = {
        "schema_version": 1,
        "duration_ns": 1_000,
        "duration_seconds": 0.000001,
        "source_message_count": 2,
        "output_message_count": 2,
        "window_start_bag_timestamp_ns": 10,
        "window_end_bag_timestamp_ns": 1_010,
        "corruption": {
            "type": "fixed", "severity": 0, "full_duration": True,
        },
        "fixed_assets": {
            "camera": {"sha256": "a" * 64},
            "lidar": {
                "sha256": "b" * 64,
                "point_count": 34_752,
                "feature_count": 5,
                "dtype": "float32",
                "finite": True,
            },
        },
        "sensor_records": [camera, lidar],
        "preserved_stream_source_sha256": "same",
        "preserved_stream_output_sha256": "same",
        "pairing": {
            "tolerance_ns": 50_000_000,
            "camera_messages": 1,
            "lidar_messages": 1,
            "paired_camera_messages": 1,
            "depth_pair_count": 1,
            "unique_depth_source_lidar_messages": 1,
            "maximum_delta_ns": 0,
        },
    }
    assert validate_manifest(manifest) == []
    lidar["output_width"] -= 1
    assert "a fixed PointCloud2 layout is invalid" in validate_manifest(
        manifest)


def test_legacy_manifest_still_validates():
    """Existing ten-second study manifests retain their validation behavior."""
    records = [
        {
            "topic": CAMERA_TOPIC, "bag_timestamp_ns": 1_000_000_000,
            "output_bag_timestamp_ns": 1_000_000_000,
            "header_timestamp_ns": 10, "output_header_timestamp_ns": 10,
        },
        {
            "topic": LIDAR_TOPIC, "bag_timestamp_ns": 1_020_000_000,
            "output_bag_timestamp_ns": 1_020_000_000,
            "header_timestamp_ns": 20, "output_header_timestamp_ns": 20,
        },
    ]
    manifest = {
        "schema_version": 1, "duration_seconds": 10.0,
        "corruption": {"type": "rain", "severity": 3, "seed": 19},
        "sensor_records": records,
        "preserved_stream_source_sha256": "same",
        "preserved_stream_output_sha256": "same",
        "pairing": {"paired_lidar_messages": 1},
    }
    assert validate_manifest(manifest) == []
    manifest["sensor_records"][1]["output_header_timestamp_ns"] += 1
    assert "a sensor header timestamp changed" in validate_manifest(manifest)
    assert derived_seed(7, CAMERA_TOPIC, 1) == derived_seed(
        7, CAMERA_TOPIC, 1
    )
