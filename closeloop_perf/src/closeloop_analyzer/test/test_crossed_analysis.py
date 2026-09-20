"""Statistics, deterministic selection, coverage, contrasts, and rejection."""

import copy
import json

import numpy as np
import pytest

from closeloop_analyzer.input_data.crossed_analysis import (
    choose_scenes, contrasts, freeze_selection, mode_comparisons,
)
from closeloop_analyzer.input_data.crossed_evidence import coverage_rows, inspect_execution, metrics


def test_untrimmed_linear_R_and_fragile_tail_are_explicit():
    values = [1, 2, 3, 100]
    result = metrics(values, 3, 20)
    p50, p99 = np.percentile(values, [50, 99], method="linear")
    assert result["R"] == (p99 - p50) / p50
    assert result["P99_minus_P50_ms"] == p99 - p50
    assert result["throughput_hz"] == .2
    assert result["unique_source_count"] == 3
    assert "fragile" in result["sample_warning"]
    from closeloop_analyzer.input_data.corrected_input_analysis import distribution_metrics
    existing = distribution_metrics(values)
    assert existing["R"] == result["R"]
    assert existing["normalized_range"] == (p99 - min(values)) / p50


def test_selection_ties_and_freeze(tmp_path):
    pair, scenes = ["lidar", "camera"], ["0770", "0398", "0184", "0245"]
    rows = {m: {s: {"R": 1.0} for s in scenes} for m in pair}
    result = choose_scenes(pair, scenes, rows)
    assert (result["selected_model"], result["A"], result["B"]) == ("lidar", "0770", "0398")
    rows["camera"]["0245"]["R"] = 2
    assert choose_scenes(pair, scenes, rows)["selected_model"] == "camera"
    path = tmp_path / "selection.json"
    freeze_selection(path, result)
    freeze_selection(path, result)
    changed = copy.deepcopy(result)
    changed["B"] = "0184"
    with pytest.raises(ValueError, match="frozen"):
        freeze_selection(path, changed)


def test_coverage_uses_relay_publication_not_scheduled_bag_inputs():
    frames = [{"topic": "/LIDAR_TOP", "output_header_timestamp_ns": t, "source_frame_id": str(t)} for t in (1, 2, 3)]
    relay = [{"source_topic": "/LIDAR_TOP", "original_source_timestamp_ns": 1, "relay_post_publish_monotonic_ns": 20}]
    inputs = [{"ros_header_timestamp_ns": 2, "completed": False}, {"ros_header_timestamp_ns": 99, "completed": True}]
    rows, unmatched, _ = coverage_rows(frames, relay, inputs, {"id": "lidar", "modality": "lidar"})
    assert sum(r["relay_published"] for r in rows) == 1
    assert sum(r["upstream_missing"] for r in rows) == 2
    assert rows[0]["dropped_or_overwritten"] == 1
    assert rows[1]["failed"] == 1
    assert len(unmatched) == 1


def test_own_and_corunner_contrasts_and_interaction():
    cells = dict(AA=1, AB=3, BA=4, BB=8)
    lidar, camera = contrasts(cells, "lidar"), contrasts(cells, "image")
    assert lidar[:2] == [("co_runner", "AB-AA", 2), ("co_runner", "BB-BA", 4)]
    assert camera[:2] == [("co_runner", "BA-AA", 3), ("co_runner", "BB-AB", 5)]
    assert lidar[-1] == ("interaction", "BB-BA-AB+AA", 2)


def test_success_without_evidence_cannot_be_reused(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({"state": "success"}))
    result = inspect_execution(tmp_path, {}, {})
    assert not result["accepted"]
    assert "missing/invalid recorded evidence" in result["reasons"][0]


def test_excluded_attempts_remain_in_archive_index(tmp_path, monkeypatch):
    from closeloop_analyzer.input_data import crossed_analysis
    monkeypatch.setattr(crossed_analysis, "archive_index", lambda path: {"profile.sqlite": {"sha256": "verified"}})
    manifest = {"executions": [{"run_id": "replacement", "phase": "screening", "status": "blocked",
        "attempts": [{"artifact_directory": str(tmp_path / "original"), "blocker": "missing completion trace"}]}]}
    _, _, blockers, archives = crossed_analysis.collect(manifest, tmp_path, ("screening",))
    assert len(blockers) == 1
    assert archives == [{"run_id": "original", "role": "excluded_attempt", "accepted": False,
                         "rejection_reasons": ["missing completion trace"],
                         "profile.sqlite": {"sha256": "verified"}}]


def test_modes_require_identical_actual_scene_combinations():
    row = dict(phase="confirmation", pair=["lidar", "camera"], model_id="lidar", lidar_scene="A", camera_scene="B",
               mps_enabled=False, R=1, P50_ms=2, P99_ms=4, throughput_hz=5, unique_source_count=20)
    assert mode_comparisons([row, {**row, "mps_enabled": True, "camera_scene": "C"}]) == []
    matched = mode_comparisons([row, {**row, "mps_enabled": True, "R": 2}])
    assert matched[0]["on_minus_off_mean"] == 1


def test_archive_hash_and_sqlite_integrity_are_checked_without_mutation(tmp_path):
    import gzip
    import sqlite3
    from closeloop_analyzer.input_data.crossed_evidence import archive_index, sha256
    path = tmp_path / "profile.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE NVTX_EVENTS (start INTEGER)")
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER)")
    archives = {}
    for name, data in (("profile.nsys-rep", b"retained Nsight bytes"), ("profile.sqlite", path.read_bytes())):
        archived = tmp_path / (name + ".gz")
        archived.write_bytes(gzip.compress(data))
        archives[name] = {"path": str(archived), "sha256": sha256(archived), "bytes": archived.stat().st_size}
    path.unlink()
    (tmp_path / "profile_archive.json").write_text(json.dumps(archives))
    before = {p.name: sha256(p) for p in tmp_path.iterdir()}
    assert archive_index(tmp_path)["profile.sqlite"]["restored_sha256"]
    assert {p.name: sha256(p) for p in tmp_path.iterdir()} == before
    archive = tmp_path / "profile.nsys-rep.gz"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive size/hash mismatch"):
        archive_index(tmp_path)


def test_selection_keeps_authored_scene_order_after_sorted_json_roundtrip(tmp_path):
    import yaml
    from closeloop_analyzer.input_data.crossed_analysis import select
    scenes = ["scene-0770", "scene-0398", "scene-0184", "scene-0245"]
    pair = ["lidar", "camera"]
    data = {"conditions": dict.fromkeys(scenes, {}), "pairs": [pair], "mps_modes": [False]}
    source = tmp_path / "study.yaml"
    source.write_text(yaml.safe_dump(data, sort_keys=False))
    manifest = json.loads(json.dumps({"study": data, "study_path": str(source),
                                      "study_sha256": "frozen", "window_ns": 1,
                                      "executions": []}, sort_keys=True))
    assert list(manifest["study"]["conditions"]) != scenes
    summaries = []
    for scene in scenes:
        run = tmp_path / scene
        run.mkdir()
        (run / "run_manifest.json").write_text("{}")
        manifest["executions"].append({"run_id": scene, "artifact_directory": str(run),
                                       "evidence": {"file_hashes": {}}})
        for model in pair:
            summaries.append({"phase": "screening", "pair": pair, "mps_enabled": False,
                              "model_id": model, "slot_id": scene, "scene_id": scene,
                              "lidar_scene": scene, "camera_scene": scene, "R": 1.0,
                              "sample_warning": "", "unique_source_count": 200})
    selected = select(manifest, summaries, tmp_path)["selections"]["lidar+camera:mps-off"]
    assert (selected["A"], selected["B"], selected["selected_model"]) == (scenes[0], scenes[1], "lidar")


def test_baseline_violins_group_modes_and_limit_only_display_to_p99(tmp_path, monkeypatch):
    import csv
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from closeloop_analyzer.input_data.crossed_analysis import isolated_violin_plots
    scenes = ["scene-0770", "scene-0398", "scene-0184", "scene-0245"]
    manifest = {"scene_order": scenes, "study": {"models": {"lidar": {}}, "mps_modes": [False, True]}}
    summaries, frames, expected, observed, positions = [], {}, [], [], []
    for mode in (False, True):
        for i, scene in enumerate(scenes):
            slot = f"{int(mode)}-{scene}"
            values = [1 + i + 10 * mode, 2 + i + 10 * mode, 1000 + i + 10 * mode]
            expected.append(values[:-1])
            summaries.append(dict(phase="isolated", slot_id=slot, execution_id=slot, model_id="lidar",
                                  mps_enabled=mode, scene_id=scene, completed_count=3, unique_source_count=3))
            frames[slot] = [dict(model_id="lidar", scene_id=scene, latency_ms=value,
                                 source_frame_id=f"{scene}:{n}") for n, value in enumerate(values)]
            frames[slot].append(dict(model_id="other", latency_ms=99999))
    summaries.append(dict(phase="confirmation", model_id="lidar"))
    original_frames, original_summaries = copy.deepcopy(frames), copy.deepcopy(summaries)
    violinplot, savefig = Axes.violinplot, Figure.savefig

    def capture_violin(axis, dataset, *args, **kwargs):
        observed.append(list(dataset[0]))
        positions.extend(kwargs["positions"])
        return violinplot(axis, dataset, *args, **kwargs)

    def check_figure(figure, *args, **kwargs):
        assert len(figure.axes) == 1
        assert "P99" in figure.axes[0].get_title()
        assert [t.get_text() for t in figure.axes[0].get_legend().get_texts()] == ["MPS off", "MPS on"]
        for axis in figure.axes:
            assert axis.get_xlabel() == "Scene" and axis.get_ylabel() == "Inference time (ms)"
            assert [t.get_text() for t in axis.get_xticklabels()] == [s[6:] for s in scenes]
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Axes, "violinplot", capture_violin)
    monkeypatch.setattr(Figure, "savefig", check_figure)
    result = isolated_violin_plots(manifest, summaries, frames, tmp_path)
    assert observed == expected  # Keeps the lower tail and omits only observations above each P99.
    assert frames == original_frames and summaries == original_summaries
    assert np.allclose(positions, [i + offset for offset in (-.18, .18) for i in range(4)])
    counts = list(csv.DictReader((tmp_path / "isolated_violin_display.csv").open()))
    assert len(counts) == 8 and all(r["displayed_count"] == "2" and r["omitted_above_P99"] == "1" for r in counts)
    for record in counts:
        values = [r["latency_ms"] for r in original_frames[record["execution_id"]] if r["model_id"] == "lidar"]
        assert float(record["P99_ms"]) == np.percentile(values, 99, method="linear")
    assert (tmp_path / result["plots"][0]).is_file()
    assert (tmp_path / "plots/isolated/isolated_baselines.pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="one execution"):
        isolated_violin_plots(manifest, summaries + [summaries[0]], frames, tmp_path)
