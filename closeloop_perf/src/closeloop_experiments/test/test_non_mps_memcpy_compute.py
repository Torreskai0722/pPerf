"""Tests for the bidirectional memcpy/compute-demand campaign."""

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import yaml

from closeloop_experiments.non_mps_memcpy_compute import (
    ALIGNMENT_METRIC,
    NonMpsMemcpyComputeDemandV2,
    StudyError,
    _json_bytes,
    _trace_run,
    alignment_score,
    calibrated_trial_period_ns,
    candidate_id,
    canonical_context_timeline,
    canonical_copy_timeline,
    load_candidate_manifest,
    load_study,
    select_offsets,
    validate_non_mps_host,
)


def _baseline(tmp_path):
    (tmp_path / "bags").mkdir()
    (tmp_path / "scene.json").write_text("[]", encoding="utf-8")
    (tmp_path / "image.bin").write_bytes(b"image")
    (tmp_path / "lidar.bin").write_bytes(b"lidar")
    models = [
        {
            "id": "image",
            "node_name": "image",
            "task": "detection",
            "modality": "image",
            "mmlab_model": "faster-rcnn",
            "architecture_profile": "mmdet_two_stage_2d_v1",
            "input_topic": "/image",
            "input_message_type": "image",
            "qos": "best_effort",
            "warmup_input": "image.bin",
            "warmup_count": 1,
            "launch_offset_seconds": 0,
            "module_annotation_depth": 0,
        },
        {
            "id": "lidar",
            "node_name": "lidar",
            "task": "detection",
            "modality": "lidar",
            "mmlab_model": "centerpoint",
            "architecture_profile": "mmdet3d_voxel_two_stage_v1",
            "input_topic": "/lidar",
            "input_message_type": "pointcloud2",
            "qos": "best_effort",
            "warmup_input": "lidar.bin",
            "warmup_count": 1,
            "launch_offset_seconds": 0,
            "module_annotation_depth": 0,
        },
    ]
    return {
        "schema_version": 2,
        "run": {
            "id": "baseline",
            "experiment": "test",
            "provenance": "fixture",
            "timeout_seconds": 10,
        },
        "ros": {
            "distribution": "humble",
            "middleware": "rmw_fastrtps_cpp",
            "domain_id": 8,
            "launch_package": "closeloop_testbed",
            "launch_file": "testbed.launch.py",
        },
        "replay": {
            "scene_token": "scene",
            "metadata_path": "scene.json",
            "bag_directory": "bags",
            "rate": 0.25,
            "playback_mode": "partial",
            "duration_seconds": 10,
            "topics": ["/image"],
            "remappings": {},
            "readiness_timeout_seconds": 2,
            "completion_timeout_seconds": 2,
        },
        "gpu": {"index": 0, "mps_enabled": False},
        "models": models,
        "recording": {
            "level": "level1", "scopes": ["model"],
            "nsys": {
                "version": "2025.3.1",
                "trace": ["cuda", "nvtx", "cudnn"],
                "sample": "none",
                "backtrace": "none",
                "gpu_context_switch": True,
            },
        },
    }


def _study(tmp_path):
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(yaml.safe_dump(_baseline(tmp_path)), encoding="utf-8")
    study = tmp_path / "study.yaml"
    study.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "skill": "NonMpsMemcpyComputeDemandV2",
                "baseline_config": "baseline.yaml",
                "directions": [
                    {
                        "id": "image-target",
                        "target_model_id": "image",
                        "co_runner_model_id": "lidar",
                        "replay_module": "pts_backbone",
                    },
                    {
                        "id": "lidar-target",
                        "target_model_id": "lidar",
                        "co_runner_model_id": "image",
                        "replay_module": "backbone",
                    },
                ],
                "fixed_input_paths": {
                    "image": "image.bin",
                    "lidar": "lidar.bin",
                },
                "k_values": [0, 1, 2, 4, 8],
                "trials_per_run": 5,
                "launch_tolerance_ms": 0.5,
                "matched_overlap_tolerance": 0.05,
                "alignment_difference_min": 0.05,
                "alignment_window_ms": 2.0,
                "non_overlap_latency_tolerance": 0.1,
                "period_guard_fraction": 0.25,
                "output_root": "campaign",
                "random_seed": 17,
            }
        ),
        encoding="utf-8",
    )
    return load_study(study)


def _calibration(study):
    copies = [
        {
            "copy_class": "H2D",
            "bytes": 100,
            "ordinal": 0,
            "start_ns": 10_000_000,
            "end_ns": 11_000_000,
        },
        {
            "copy_class": "D2H",
            "bytes": 50,
            "ordinal": 0,
            "start_ns": 19_000_000,
            "end_ns": 20_000_000,
        },
    ]
    models = {}
    contexts = [
        {"ordinal": 0, "start_ns": 0, "end_ns": 12_000_000},
        {"ordinal": 1, "start_ns": 18_000_000, "end_ns": 30_000_000},
    ]
    for model_id in study["inputs"]:
        models[model_id] = {
            str(k): {
                "median_execution_envelope_ns": 30_000_000,
                "worst_execution_envelope_ns": 35_000_000,
                "median_kernel_time_ns": 10_000_000,
                "median_kernel_latency_ns": 16_000_000,
                "median_subgraph_kernel_time_ns": k * 4_000_000,
                "canonical_context_timeline": contexts,
                **({"canonical_copy_timeline": copies} if k == 0 else {}),
            }
            for k in (0, 1, 2, 4, 8)
        }
    directions = {}
    for direction_id, direction in study["directions"].items():
        offsets = []
        labels = [
            "nonoverlap_left",
            "nonoverlap_right",
            "overlap_25_left",
            "overlap_25_right",
            "overlap_50_left",
            "overlap_50_right",
            "overlap_75_left",
            "overlap_75_right",
            "max_alignment",
            "max_overlap_low_alignment",
        ]
        deltas = [-40, 31, -20, 20, -15, 15, -10, 10, 0, 1]
        for index, (label, delta) in enumerate(zip(labels, deltas)):
            offsets.append(
                {
                    "offset_index": index,
                    "label": label,
                    "delta_ns": delta * 1_000_000,
                    "predicted_execution_overlap": 0.5,
                    "predicted_alignment": 1.0 if index == 8 else 0.0,
                }
            )
        directions[direction_id] = {
            **direction,
            "offsets": offsets,
            "matched_pair": {
                "high_alignment_offset_index": 8,
                "low_alignment_offset_index": 9,
                "execution_overlap_difference": 0.0,
                "alignment_difference": 1.0,
            },
        }
    manifest = {
        "schema_version": 1,
        "skill": "NonMpsMemcpyComputeDemandV2",
        "alignment_metric": ALIGNMENT_METRIC,
        "alignment_window_ns": 2_000_000,
        "study_sha256": study["sha256"],
        "baseline_config_sha256": study["baseline"].sha256,
        "resolved_clocks": {
            "graphics_clock_mhz": 2100,
            "memory_clock_mhz": 9000,
        },
        "fixed_input_sha256": {
            model_id: __import__("hashlib")
            .sha256(path.read_bytes())
            .hexdigest()
            for model_id, path in study["inputs"].items()
        },
        "trial_period_ns": 100_000_000,
        "models": models,
        "directions": directions,
    }
    output = study["output_root"] / "calibration_manifest.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(_json_bytes(manifest))
    return manifest


def test_100_cell_stable_bidirectional_grid_and_configs(tmp_path):
    study = _study(tmp_path)
    calibration = _calibration(study)
    campaign = NonMpsMemcpyComputeDemandV2(study)
    cells = campaign.candidates(calibration)
    assert len(cells) == len({item["candidate_id"] for item in cells}) == 100
    assert {item["direction_id"] for item in cells} == {
        "image-target",
        "lidar-target",
    }
    assert {item["k_replays"] for item in cells} == {0, 1, 2, 4, 8}
    assert candidate_id("image-target", 3, 8) == "nmcd-image-target-o03-k08"
    manifest = campaign.generate(calibration)
    negative = next(
        item
        for item in manifest["candidates"]
        if item["direction_id"] == "image-target"
        and item["configured_delta_ns"] < 0
    )
    config = yaml.safe_load(Path(negative["config_path"]).read_text())
    target = next(
        model
        for model in config["models"]
        if model["paired_trial"]["role"] == "target"
    )
    co = next(
        model
        for model in config["models"]
        if model["paired_trial"]["role"] == "co_runner"
    )
    assert target["paired_trial"]["target_anchor_seconds"] > 0
    assert co["paired_trial"]["deadline_offset_seconds"] == 0


def test_alignment_offset_selection_and_copy_signatures():
    target = [
        {
            "copy_class": "H2D",
            "bytes": 1,
            "ordinal": 0,
            "start_ns": 40,
            "end_ns": 50,
        },
        {
            "copy_class": "D2H",
            "bytes": 1,
            "ordinal": 0,
            "start_ns": 90,
            "end_ns": 100,
        },
    ]
    target_with_d2d = [
        target[0],
        {"copy_class": "D2D", "start_ns": 65, "end_ns": 75},
        target[1],
    ]
    co_context = [
        {"start_ns": 5, "end_ns": 15},
        {"start_ns": 55, "end_ns": 65},
    ]
    assert alignment_score(target_with_d2d, co_context, 25, 10) == 1
    assert alignment_score(target_with_d2d, co_context, -40, 10) == 0
    offsets, matched = select_offsets(
        target, co_context, 100, 100, 100, 10
    )
    assert len({item["delta_ns"] for item in offsets}) == 10
    assert offsets[0]["predicted_execution_overlap"] == 0
    assert offsets[1]["predicted_execution_overlap"] == 0
    assert matched["execution_overlap_difference"] <= 0.05
    assert matched["alignment_difference"] >= 0.05
    trials = [target for _ in range(5)]
    assert canonical_copy_timeline(trials) == target
    changed = [list(trial) for trial in trials]
    changed[-1] = [{**target[0], "bytes": 2}, target[1]]
    with pytest.raises(StudyError, match="signature changed"):
        canonical_copy_timeline(changed)
    contexts = [co_context for _ in range(5)]
    contexts[-1] = co_context + [{"start_ns": 75, "end_ns": 80}]
    assert canonical_context_timeline(contexts) == [
        {"ordinal": 0, "start_ns": 5, "end_ns": 15},
        {"ordinal": 1, "start_ns": 55, "end_ns": 65},
    ]


def test_trial_period_uses_signed_offset_paired_k8_envelope():
    """Signed anchors, both envelopes, guard, and rounding set one period."""
    models = {
        "image": {
            "0": {"worst_execution_envelope_ns": 50_000_000},
            "8": {"worst_execution_envelope_ns": 100_000_000},
        },
        "lidar": {
            "0": {"worst_execution_envelope_ns": 120_000_000},
            "8": {"worst_execution_envelope_ns": 125_000_000},
        },
    }
    directions = [
        {
            "target_model_id": "lidar",
            "co_runner_model_id": "image",
            "offsets": [{"delta_ns": -150_000_000}],
        },
        {
            "target_model_id": "image",
            "co_runner_model_id": "lidar",
            "offsets": [{"delta_ns": 140_000_000}],
        },
    ]
    assert calibrated_trial_period_ns(models, directions, 0.25) == 400_000_000


def test_restart_rejects_changed_generated_config(tmp_path):
    study = _study(tmp_path)
    campaign = NonMpsMemcpyComputeDemandV2(study)
    campaign.generate(_calibration(study))
    manifest = load_candidate_manifest(study)
    path = Path(manifest["candidates"][0]["config_path"])
    path.chmod(0o644)
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(StudyError, match="generated config changed"):
        load_candidate_manifest(study)


def test_strict_non_mps_host_rejection(tmp_path):
    study = _study(tmp_path)

    def daemon(arguments, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="123\n", stderr="")

    with pytest.raises(StudyError, match="daemon"):
        validate_non_mps_host(study, daemon)

    def exclusive(arguments, **_kwargs):
        output = "" if arguments[0] == "pgrep" else "Exclusive_Process\n"
        return SimpleNamespace(
            returncode=1 if arguments[0] == "pgrep" else 0,
            stdout=output,
            stderr="",
        )

    with pytest.raises(StudyError, match="compute mode"):
        validate_non_mps_host(study, exclusive)


def _synthetic_trace(path, target_pid, co_pid):
    target_global, co_global = target_pid << 24, co_pid << 24
    host_owner = (target_pid + 1000) << 24
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            "CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);"
            "CREATE TABLE NVTX_EVENTS("
            "start INTEGER,end INTEGER,globalTid INTEGER,text TEXT);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
            "start INTEGER,end INTEGER,globalPid INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY("
            "start INTEGER,end INTEGER,globalPid INTEGER,"
            "correlationId INTEGER,copyKind INTEGER,bytes INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME("
            "start INTEGER,end INTEGER,globalTid INTEGER,"
            "correlationId INTEGER,nameId INTEGER);"
            "CREATE TABLE ENUM_GPU_CTX_SWITCH("
            "id INTEGER PRIMARY KEY,name TEXT);"
            "CREATE TABLE GPU_CONTEXT_SWITCH_EVENTS("
            "tag INTEGER,contextId INTEGER,globalPid INTEGER,"
            "timestamp INTEGER,seqNo INTEGER);"
        )
        db.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [(1, "cudaMemcpyAsync_v3020"), (2, "cudaStreamSynchronize_v3020")],
        )
        db.executemany(
            "INSERT INTO ENUM_GPU_CTX_SWITCH VALUES (?,?)",
            [
                (1, "RESTORE_START"),
                (2, "RESTORE_END"),
                (3, "SAVE_START"),
                (4, "SAVE_END"),
            ],
        )
        sequence = 0
        for trial in range(5):
            base = trial * 100_000_000
            tags = [
                (
                    10,
                    30,
                    target_global,
                    {
                        "event": "inference",
                        "model": "target",
                        "input": f"paired-{trial}",
                    },
                ),
                (
                    12,
                    18,
                    co_global,
                    {
                        "event": "inference",
                        "model": "co",
                        "input": f"paired-{trial}",
                    },
                ),
                (
                    20,
                    24,
                    co_global,
                    {
                        "event": "compute_service_replay",
                        "model": "co",
                        "input": f"paired-{trial}",
                    },
                ),
            ]
            for start, end, owner, tag in tags:
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
                    (
                        base + start * 1_000_000,
                        base + end * 1_000_000,
                        owner,
                        "closeloop:" + json.dumps(tag),
                    ),
                )
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)",
                [
                    (base + 12_000_000, base + 14_000_000, target_global),
                    (base + 25_000_000, base + 28_000_000, target_global),
                    (base + 13_000_000, base + 17_000_000, co_global),
                    (base + 20_000_000, base + 23_000_000, co_global),
                ],
            )
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?)",
                [
                    (
                        base + 11_000_000,
                        base + 12_000_000,
                        target_global,
                        10 + trial,
                        1,
                        100,
                    ),
                    (
                        base + 28_000_000,
                        base + 29_000_000,
                        target_global,
                        20 + trial,
                        2,
                        50,
                    ),
                    (
                        base + 20_000_000,
                        base + 21_000_000,
                        target_global,
                        40 + trial,
                        8,
                        999,
                    ),
                    (
                        base + 12_100_000,
                        base + 12_200_000,
                        co_global,
                        30 + trial,
                        1,
                        100,
                    ),
                ],
            )
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                [
                    (
                        base + 10_500_000,
                        base + 10_600_000,
                        target_global,
                        10 + trial,
                        1,
                    ),
                    (
                        base + 10_600_000,
                        base + 10_800_000,
                        target_global,
                        99,
                        2,
                    ),
                    (
                        base + 27_500_000,
                        base + 27_600_000,
                        target_global,
                        20 + trial,
                        1,
                    ),
                    (
                        base + 27_600_000,
                        base + 27_900_000,
                        target_global,
                        98,
                        2,
                    ),
                ],
            )
            events = [
                (10, 1),
                (101, 2),
                (149, 3),
                (15, 4),
                (25, 1),
                (251, 2),
                (299, 3),
                (30, 4),
            ]
            for marker, tag in events:
                timestamp = (
                    base + marker * 1_000_000
                    if marker < 100
                    else base + marker * 100_000
                )
                sequence += 1
                db.execute(
                    "INSERT INTO GPU_CONTEXT_SWITCH_EVENTS VALUES (?,?,?,?,?)",
                    (tag, 1, host_owner, timestamp, sequence),
                )
            for timestamp_ms, tag in (
                (10, 1),
                (10.1, 2),
                (14.9, 3),
                (15, 4),
                (20, 1),
                (20.1, 2),
                (23.9, 3),
                (24, 4),
                (26, 1),
                (26.1, 2),
                (29.9, 3),
                (30, 4),
            ):
                sequence += 1
                db.execute(
                    "INSERT INTO GPU_CONTEXT_SWITCH_EVENTS VALUES (?,?,?,?,?)",
                    (
                        tag,
                        2,
                        co_global,
                        base + round(timestamp_ms * 1_000_000),
                        sequence,
                    ),
                )


def test_synthetic_trace_classification_alignment_and_descheduling(tmp_path):
    target_pid, co_pid = 101, 202
    _synthetic_trace(tmp_path / "profile.sqlite", target_pid, co_pid)
    for model_id, pid in (("target", target_pid), ("co", co_pid)):
        (tmp_path / f"model_{model_id}.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "pid": pid,
                    "state": "acknowledged",
                    "process_observed_mps_environment": {},
                    "fixed_input_sha256": model_id,
                    "resident_input_sha256": "resident",
                }
            ),
            encoding="utf-8",
        )
        records = [
            {
                "trial": trial,
                "actual_time": trial + (0.002 if model_id == "co" else 0),
                "launch_error_seconds": 0,
                "quiescence_verified": True,
            }
            for trial in range(5)
        ]
        (tmp_path / f"model_{model_id}_paired.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in records),
            encoding="utf-8",
        )
    candidate = {
        "candidate_id": "cell",
        "direction_id": "d",
        "target_model_id": "target",
        "co_runner_model_id": "co",
        "offset_index": 0,
        "offset_label": "max_alignment",
        "configured_delta_ns": 2_000_000,
        "k_replays": 1,
    }
    timeline = [
        {
            "copy_class": "H2D",
            "bytes": 100,
            "ordinal": 0,
            "start_ns": 1_000_000,
            "end_ns": 2_000_000,
        },
        {
            "copy_class": "D2H",
            "bytes": 50,
            "ordinal": 0,
            "start_ns": 18_000_000,
            "end_ns": 19_000_000,
        },
    ]
    calibration = {
        "alignment_metric": ALIGNMENT_METRIC,
        "alignment_window_ns": 2_000_000,
        "models": {
            "target": {"0": {"canonical_copy_timeline": timeline}},
            "co": {
                "1": {
                    "canonical_context_timeline": [
                        {"start_ns": 0, "end_ns": 3_000_000},
                        {"start_ns": 8_000_000, "end_ns": 12_000_000},
                    ]
                }
            },
        }
    }
    rows, evidence = _trace_run(tmp_path, candidate, {}, calibration)
    assert evidence["observable"] and not evidence["errors"]
    assert len(rows) == 5
    assert rows[0]["target_kernel_latency_ns"] == 16_000_000
    assert rows[0]["memcpy_adjacent_waiting_ns"] == 500_000
    assert rows[0]["inference_overlap_ns"] == 6_000_000
    assert rows[0]["target_inactive_ns"] == 10_000_000
    assert rows[0]["compute_related_descheduling_ns"] == 4_000_000
    assert rows[0]["subgraph_kernel_time_ns"] == 3_000_000
    assert rows[0]["h2d_signature"] == "[100]"
    assert rows[0]["d2h_signature"] == "[50]"
    assert rows[0]["alignment_operation_count"] == 2
    assert rows[0]["actual_alignment"] == 0.75
