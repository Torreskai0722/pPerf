"""Focused checks for the seven-condition input2 campaign."""

import hashlib
import os
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

import closeloop_analyzer.input_data.corrected_input_analysis as analysis
from closeloop_experiments.config import load_run_config
from closeloop_experiments.corrected_input_variation import (
    expected_runs, load_study, materialize, model_pairs,
)
from closeloop_experiments.corrupt_rosbag import CAMERA_TOPIC, LIDAR_TOPIC
from closeloop_experiments.mps import MAX_UNIX_SOCKET_PATH_BYTES, MPSManager


ROOT = Path(__file__).parents[3]
STUDY = ROOT / "studies/input2/study.yaml"
SCENES = (
    "scene-0434", "scene-0245", "scene-0398", "scene-0184",
    "scene-0738", "scene-0770", "scene-0252-fixed",
)
LIDARS = ("3dssd", "centerpoint", "pointpillars")
CAMERAS = (
    "detr", "yolov3", "mask-rcnn", "dino", "vit-upernet", "faster-rcnn",
)


def test_input2_is_exactly_the_requested_252_run_matrix(tmp_path):
    study = load_study(STUDY, check_paths=False)
    planned = expected_runs(study)
    configs = materialize(study, tmp_path)

    assert tuple(study["data"]["conditions"]) == SCENES
    assert set(model_pairs(study)) == {
        (lidar, camera) for lidar in LIDARS for camera in CAMERAS
    }
    assert study["data"]["mps_modes"] == [False, True]
    assert study["data"]["replicates"] == study["data"]["replay_passes"] == 1
    assert study["data"]["conditions"]["scene-0252-fixed"][
        "replay_passes"] == 2
    assert study["data"]["baseline_map"] == {}
    assert "scene-0539" not in STUDY.read_text(encoding="utf-8")
    assert len(planned) == len(configs) == 252
    assert set(planned) == {
        f"{mode}-{scene}-{lidar}-{camera}"
        for mode in (0, 1) for scene in SCENES
        for lidar in LIDARS for camera in CAMERAS
    }
    for path in configs.values():
        config = load_run_config(path, check_paths=False).data
        condition = study["data"]["conditions"][
            config["input_variation"]["condition_id"]
        ]
        expected_passes = 2 if config["input_variation"][
            "condition_id"] == "scene-0252-fixed" else 1
        assert config["replay"]["repeat_count"] == expected_passes
        assert config["replay"]["playback_mode"] == "full"
        assert "duration_seconds" not in config["replay"]
        assert condition["scene_factors"]["lighting"] in ("daylight", "fixed")
        assert config["input_variation"]["corruption"]["full_duration"] is True


def test_scene_0252_fixed_decoded_contents_and_timestamps_match():
    """Every fixed sensor payload matches its asset without timing drift."""
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, PointCloud2

    repo = ROOT.parent
    source = (repo / "data/bag/NuScenes-v1.0-trainval-scene-0252/"
              "NuScenes-v1.0-trainval-scene-0252_0.mcap")
    fixed = repo / "data/input_variation/scene-0252-fixed"
    if not source.is_file() or not fixed.is_dir():
        pytest.skip("generated scene-0252-fixed artifact is unavailable")
    output = next(fixed.glob("*.mcap"))
    image_hash = hashlib.sha256(
        (repo / "data/warmup.jpg").read_bytes()).hexdigest()
    lidar_hash = hashlib.sha256(
        (repo / "data/warmup.bin").read_bytes()).hexdigest()

    def reader(path):
        value = rosbag2_py.SequentialReader()
        value.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        return value

    left, right = reader(source), reader(output)
    counts = {CAMERA_TOPIC: 0, LIDAR_TOPIC: 0}
    selected = {
        CAMERA_TOPIC: (1534867436112404000, image_hash),
        LIDAR_TOPIC: (1534867436097280000, lidar_hash),
    }
    selected_found = set()
    while left.has_next():
        assert right.has_next()
        source_topic, source_raw, source_timestamp = left.read_next()
        output_topic, output_raw, output_timestamp = right.read_next()
        assert (output_topic, output_timestamp) == (
            source_topic, source_timestamp)
        if source_topic not in counts:
            assert output_raw == source_raw
            continue
        message_type = (CompressedImage if source_topic == CAMERA_TOPIC
                        else PointCloud2)
        source_message = deserialize_message(source_raw, message_type)
        output_message = deserialize_message(output_raw, message_type)
        assert output_message.header == source_message.header
        expected = image_hash if source_topic == CAMERA_TOPIC else lidar_hash
        assert hashlib.sha256(bytes(output_message.data)).hexdigest() == expected
        header_ns = (source_message.header.stamp.sec * 1_000_000_000
                     + source_message.header.stamp.nanosec)
        if header_ns == selected[source_topic][0]:
            assert hashlib.sha256(bytes(source_message.data)).hexdigest() == (
                selected[source_topic][1]
            )
            selected_found.add(source_topic)
        if source_topic == LIDAR_TOPIC:
            assert (output_message.height, output_message.width,
                    output_message.point_step, output_message.row_step) == (
                        1, 34_752, 20, 695_040)
        counts[source_topic] += 1
    assert not right.has_next()
    assert selected_found == {CAMERA_TOPIC, LIDAR_TOPIC}
    assert counts == {CAMERA_TOPIC: 229, LIDAR_TOPIC: 384}


def test_input2_longest_mps_socket_path_fits_linux_limit():
    study = load_study(STUDY, check_paths=False)
    with tempfile.TemporaryDirectory(prefix="p", dir="/tmp") as directory:
        paths = [
            MPSManager(
                0, Path(directory) / "runs" / run_id
            ).pipe_directory / "control_privileged"
            for run_id in expected_runs(study) if run_id.startswith("1-")
        ]
    assert max(len(os.fsencode(path)) for path in paths) <= (
        MAX_UNIX_SOCKET_PATH_BYTES
    )


def test_input2_plot_files_labels_and_p1_p99_filter(tmp_path, monkeypatch):
    study = load_study(STUDY, check_paths=False)
    frames = []
    for mode in ("off", "on"):
        for lidar in LIDARS:
            for camera in CAMERAS:
                pair = f"{lidar}+{camera}"
                for scene in SCENES:
                    for model in (lidar, camera):
                        frames.extend({
                            "pair_id": pair,
                            "condition_id": scene,
                            "mps_mode": mode,
                            "model_id": model,
                            "lidar_model": lidar,
                            "camera_model": camera,
                            "inference_e2e_ms": value,
                            "run_valid": True,
                        } for value in range(101))

    observed = []
    scenes = []
    ecdf_labels = []
    heatmap_labels = []
    original = analysis.sns.violinplot
    original_step = plt.Axes.step
    original_set_xticks = plt.Axes.set_xticks

    def capture(**kwargs):
        observed.extend(kwargs["data"]["Inference time (ms)"])
        scenes.extend(kwargs["data"]["Scene"])
        return original(**kwargs)

    monkeypatch.setattr(analysis.sns, "violinplot", capture)

    def capture_step(axis, *args, **kwargs):
        ecdf_labels.append(kwargs.get("label"))
        return original_step(axis, *args, **kwargs)

    def capture_ticks(axis, ticks, labels=None, *args, **kwargs):
        if labels is not None:
            heatmap_labels.extend(labels)
        return original_set_xticks(axis, ticks, labels, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "step", capture_step)
    monkeypatch.setattr(plt.Axes, "set_xticks", capture_ticks)
    references = [{
        **row, "condition_id": "scene-0252"
    } for row in frames if row["condition_id"] == "scene-0434"]
    ecdfs = analysis.ecdf_rows(frames)
    summary = analysis.summarize(frames)
    for mode in ("off", "on"):
        analysis._plot_ecdfs(ecdfs, tmp_path, mode)
        analysis._plot_heatmap(summary, study, tmp_path, mode)
        analysis._plot_violins(frames, study, tmp_path, mode, references)

    violins = sorted(tmp_path.glob("violin_plots/*/*.png"))
    ecdf_plots = sorted(tmp_path.glob("plots/*/ecdf-*.png"))
    heatmaps = sorted(tmp_path.glob("plots/*/width-heatmap.png"))
    assert (len(ecdf_plots), len(violins), len(heatmaps)) == (36, 36, 2)
    assert {path.parent.name for path in violins} == {"mps", "non_mps"}
    assert set(scenes) == {"scene-0252", *SCENES}
    assert "scene-0252-fixed" in ecdf_labels
    assert "scene-0252-fixed" in heatmap_labels
    assert min(observed) == 1 and max(observed) == 99
    assert all(np.asarray(plt.imread(path)).size
               for path in (*ecdf_plots, *violins, *heatmaps))
