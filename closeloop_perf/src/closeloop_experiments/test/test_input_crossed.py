"""Controlled matrix, MCAP transformations, immutable selection, and resume."""

import json
from pathlib import Path

import pytest
import yaml

from closeloop_experiments import input_crossed as study_module
from closeloop_experiments.config import load_run_config
from closeloop_experiments.controlled_bags import (
    TOPICS, TYPES, WINDOW_NS, construct_bag, inspect_source, validate_bag,
    write_json,
)
from closeloop_experiments.mps import validate_mps_pipe_directory


STUDY = Path(__file__).parents[3] / "studies/input2_crossed/study.yaml"


def test_exact_matrix_resources_repetitions_order_and_paths(tmp_path):
    study = study_module.load_study(STUDY, check_paths=False)
    selection = {"selections": {study_module.block_id(p, mode): {"A": "scene-0770", "B": "scene-0398"}
                                for p in study_module.PAIRS for mode in (False, True)}}
    isolated = study_module.planned_runs(study, "isolated")
    screening = study_module.planned_runs(study, "screening")
    confirmation = study_module.planned_runs(study, "confirmation", selection)
    assert (len(isolated), len(screening), len(confirmation)) == (56, 40, 120)
    assert len({e["run_id"] for e in isolated + screening + confirmation}) == 216
    bag = tmp_path / "manifest.json"
    bag.write_text("{}")
    for entry in isolated + screening + confirmation:
        config = study_module.run_config(study, entry, bag, 0)
        assert config["replay"]["repeat_count"] == entry["replay_passes"] == 1
        assert config["input_variation"]["replicate"] == entry["repetition"]
        assert config["replay"]["cpu_affinity"] == list(range(12, 16))
        assert config["replay"]["cpu_thread_count"] == 4
        for model in config["models"]:
            assert model["cpu_affinity"] == list(range(6) if model["modality"] == "image" else range(6, 12))
            assert model["cpu_thread_count"] == 3
            assert model["warmup_count"] == 5
            assert model["module_annotation_depth"] == 0
            assert model["qos"] == "best_effort" and model["input_queue_depth"] == 1
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        load_run_config(config_path, check_paths=False)
        validate_mps_pipe_directory(Path("/mmdetection3d_ros2/analysis_outputs/input2/runs") / entry["run_id"] / "p")
    for repetition, order in enumerate(study_module.CELL_ORDERS, 1):
        rows = [e for e in confirmation if e["repetition"] == repetition]
        assert tuple(e["cell"] for e in rows[:4]) == order
    assert confirmation[0]["pair"] != confirmation[40]["pair"]


def make_source(root, scene, first, payload):
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CompressedImage, PointCloud2
    directory = root / scene
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(directory), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    for modality in TOPICS:
        writer.create_topic(rosbag2_py.TopicMetadata(name=TOPICS[modality], type=TYPES[modality], serialization_format="cdr", offered_qos_profiles=""))
    for delta in (0, 1_000_000_000, WINDOW_NS):
        for modality, cls in (("lidar", PointCloud2), ("image", CompressedImage)):
            message = cls()
            message.header.frame_id = scene + "-original-frame"
            header = first + delta - (35_000_000 if modality == "image" else 0)
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(header, 1_000_000_000)
            message.data = payload
            writer.write(TOPICS[modality], serialize_message(message), first + delta)
    del writer
    return inspect_source(scene, {"scene_token": scene + "-token", "bag_directory": str(directory)})


def test_actual_mcap_crossing_preserves_offsets_payloads_and_unchanged_streams(tmp_path):
    a = make_source(tmp_path, "a", 10_000_000_000, b"payload-a")
    b = make_source(tmp_path, "b", 80_000_000_000, b"payload-b")
    bags = {}
    for lidar, image in ((a, a), (a, b), (b, a), (b, b)):
        sources = {"lidar": lidar, "image": image}
        path = construct_bag(sources, tmp_path / "controlled")
        assert construct_bag(sources, tmp_path / "controlled") == path
        assert validate_bag(path)["valid"]
        bags[lidar["scene_id"] + image["scene_id"]] = path
        manifest = json.loads(path.read_text())
        frames = [json.loads(line) for line in Path(manifest["frame_index"]).read_text().splitlines()]
        assert [f["topic"] for f in frames[:2]] == list(TOPICS.values())
        for frame in frames:
            assert frame["output_bag_timestamp_ns"] - frame["output_header_timestamp_ns"] == (
                35_000_000 if frame["topic"] == TOPICS["image"] else 0)
            assert frame["frame_id"] == frame["source_scene"] + "-original-frame"
    study_module.assert_unchanged_streams(bags)
    manifest = json.loads(next(iter(bags.values())).read_text())
    index = Path(manifest["frame_index"])
    index.chmod(0o644)
    index.write_text(index.read_text() + "{}\n")
    with pytest.raises(ValueError, match="index hash mismatch"):
        validate_bag(next(iter(bags.values())))


def test_resume_revalidates_success_and_never_overwrites_existing_run(tmp_path, monkeypatch):
    study = study_module.load_study(STUDY, check_paths=False)
    root, ledger = study_module.paths(tmp_path)
    root.mkdir(parents=True)
    config_path = root / "run.yaml"
    config_path.write_text("test")
    artifact = tmp_path / "runs/existing"
    artifact.mkdir(parents=True)
    sentinel = artifact / "retained"
    sentinel.write_text("unchanged")
    from closeloop_experiments.controlled_bags import sha256
    entry = {"run_id": "existing", "phase": "screening", "config_path": str(config_path),
             "config_sha256": sha256(config_path), "artifact_directory": str(artifact), "status": "validated"}
    write_json(ledger, {"study_sha256": study["sha256"], "hardware": {"gpu_hardware": {}, "cpu_topology": {}},
                        "executions": [entry], "execution_order": []})
    monkeypatch.setattr(study_module, "detect_gpu_hardware", lambda _: {})
    monkeypatch.setattr(study_module, "detect_cpu_topology", lambda: {})
    monkeypatch.setattr(study_module, "load_run_config", lambda *a, **kw: type("Config", (), {"data": {}})())
    monkeypatch.setattr(study_module, "inspect_execution", lambda *a: {"accepted": False, "reasons": ["archive hash differs"]})
    monkeypatch.setattr(study_module, "ExperimentRunner", lambda _: pytest.fail("existing evidence must not be overwritten"))
    result = study_module.run(study, tmp_path, "screening")
    assert result["counts"]["blocked"] == 1
    assert sentinel.read_text() == "unchanged"
    assert "archive hash differs" in json.loads(ledger.read_text())["executions"][0]["blocker"]


def test_confirmation_reuses_twenty_distinct_screening_executions(tmp_path, monkeypatch):
    study = study_module.load_study(STUDY, check_paths=False)
    monkeypatch.setattr(study_module, "inspect_source", lambda scene, condition: {"scene_id": scene, "duration_ns": WINDOW_NS})
    monkeypatch.setattr(study_module, "detect_gpu_hardware", lambda _: {})
    monkeypatch.setattr(study_module, "detect_cpu_topology", lambda: {})

    def bag(sources, root):
        name = "+".join(sources[m]["scene_id"] for m in TOPICS)
        path = root / name / "manifest.json"
        write_json(path, {"input_scenes": {m: {"scene_id": s["scene_id"]} for m, s in sources.items()},
                          "validation": {"stream_sha256": {TOPICS[m]: s["scene_id"] for m, s in sources.items()}}})
        return path

    monkeypatch.setattr(study_module, "construct_bag", bag)
    manifest = study_module.prepare(study, tmp_path, tmp_path / "bags", "screening")
    for entry in manifest["executions"]:
        if entry["phase"] == "screening":
            Path(entry["artifact_directory"]).mkdir(parents=True)
            entry["status"] = "validated"
    write_json(study_module.paths(tmp_path)[1], manifest)
    selection = {"selections": {study_module.block_id(p, mode): {"A": "scene-0770", "B": "scene-0398"}
                                for p in study_module.PAIRS for mode in (False, True)}}
    monkeypatch.setattr(study_module, "selection_record", lambda *a: (selection, {"path": "selection", "sha256": "frozen"}))
    monkeypatch.setattr(study_module, "inspect_execution", lambda *a: {"accepted": True, "reasons": []})
    manifest = study_module.prepare(study, tmp_path, tmp_path / "bags", "confirmation", "selection")
    confirmation = [e for e in manifest["executions"] if e["phase"] == "confirmation"]
    reused = [e for e in confirmation if e["status"] == "validated"]
    assert len(reused) == 20 and len(confirmation) - len(reused) == 100
    assert len({e["artifact_directory"] for e in reused}) == 20
    assert {e["cell"] for e in reused} == {"AA", "BB"}
    assert all(e["selection_evidence"] for e in reused)
    resumed = study_module.prepare(study, tmp_path, tmp_path / "bags", "confirmation", "selection")
    assert resumed == manifest
