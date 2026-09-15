"""Synthetic full-bag Green Context trace analysis tests."""

import json
import sqlite3

import pytest

from closeloop_experiments.non_mps_memcpy_compute import StudyError
from closeloop_experiments.full_bag_green import (
    _comparison_rows,
    _comparison_summary,
    _execution_mode,
    _write_correlation_comparison_scatter,
    extract_frame_rows,
    summarize_rows,
)


def _global_id(pid, tid=0):
    return (pid << 24) | tid


def _create_trace(path, include_segmentation=False):
    models = {"image-det": (101, 7), "lidar-det": (202, 8)}
    ranges = {
        "image-det": [(10, 20), (30, 50), (60, 90)],
        "lidar-det": [(12, 27), (35, 60), (65, 100)],
    }
    copies = {
        "image-det": [(14, 19), (34, 49), (64, 89)],
        "lidar-det": [(16, 25), (39, 55), (69, 95)],
    }
    active = {
        "image-det": [(11, 19), (31, 49), (61, 89), (93, 95)],
        "lidar-det": [(13, 26), (32, 34), (36, 59), (62, 64), (66, 99)],
    }
    if include_segmentation:
        models["segmentation"] = (303, 9)
        ranges["segmentation"] = [(10, 22), (30, 52), (60, 92)]
        copies["segmentation"] = [(14, 20), (34, 50), (64, 90)]
        active["segmentation"] = [(12, 21), (32, 51), (62, 91)]
    with sqlite3.connect(path) as database:
        database.executescript(
            """
            CREATE TABLE NVTX_EVENTS(
                start INTEGER, end INTEGER, globalTid INTEGER, text TEXT
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, globalPid INTEGER,
                contextId INTEGER, greenContextId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                start INTEGER, end INTEGER, globalPid INTEGER,
                correlationId INTEGER, copyKind INTEGER, bytes INTEGER
            );
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER, name TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER, nameId INTEGER
            );
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE GPU_CONTEXT_SWITCH_EVENTS(
                tag INTEGER, contextId INTEGER, globalPid INTEGER,
                timestamp INTEGER, seqNo INTEGER
            );
            CREATE TABLE ENUM_GPU_CTX_SWITCH(id INTEGER, name TEXT);
            INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES
                (1, 'HOST_TO_DEVICE'), (2, 'DEVICE_TO_HOST'),
                (3, 'DEVICE_TO_DEVICE');
            INSERT INTO StringIds VALUES
                (1, 'cudaMemcpyAsync_v11000'),
                (2, 'cudaStreamSynchronize_v3020'),
                (3, 'cudaEventRecord_v3020'),
                (4, 'cudaEventSynchronize_v3020');
            INSERT INTO ENUM_GPU_CTX_SWITCH VALUES
                (1, 'RESTORE_START'), (2, 'SAVE_END');
            """
        )
        sequence = 0
        for model_id, (pid, green_id) in models.items():
            global_pid = _global_id(pid)
            global_tid = _global_id(pid, 1)
            for frame, ((start_ms, end_ms), copy_starts) in enumerate(
                zip(ranges[model_id], copies[model_id])
            ):
                start, end = start_ms * 1_000_000, end_ms * 1_000_000
                tag = "closeloop:" + json.dumps({
                    "event": "inference",
                    "model": model_id,
                    "input": str(frame),
                })
                database.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
                    (start, end, global_tid, tag),
                )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
                    "(?,?,?,?,?)",
                    (start + 1_000_000, end - 1_000_000, global_pid,
                     green_id + 100, green_id),
                )
                first, second = (value * 1_000_000 for value in copy_starts)
                correlation = pid * 100 + frame
                for copy_start, kind, size, copy_correlation in (
                    (first, 1, 100, correlation),
                    (first + 500_000, 3, 200, correlation + 1000),
                    (second, 2, 50, correlation + 2000),
                ):
                    database.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES "
                        "(?,?,?,?,?,?)",
                        (copy_start, copy_start + 100_000, global_pid,
                         copy_correlation, kind, size),
                    )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
                    "(?,?,?,?,?)",
                    (first - 500_000, first - 400_000, global_tid,
                     correlation, 1),
                )
                wait = (frame + 1) * 100_000
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
                    "(?,?,?,?,?)",
                    (first - 300_000, first - 300_000 + wait, global_tid,
                     None, 2),
                )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
                    "(?,?,?,?,?)",
                    (end + 100_000, end + 110_000, global_tid, None, 3),
                )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
                    "(?,?,?,?,?)",
                    (end + 200_000, end + 400_000, global_tid, None, 4),
                )
            for left_ms, right_ms in active[model_id]:
                sequence += 1
                database.execute(
                    "INSERT INTO GPU_CONTEXT_SWITCH_EVENTS VALUES "
                    "(?,?,?,?,?)",
                    (1, green_id + 100, global_pid, left_ms * 1_000_000,
                     sequence),
                )
                sequence += 1
                database.execute(
                    "INSERT INTO GPU_CONTEXT_SWITCH_EVENTS VALUES "
                    "(?,?,?,?,?)",
                    (2, green_id + 100, global_pid, right_ms * 1_000_000,
                     sequence),
                )
    return models


def _write_model_evidence(path, models):
    for model_id, (pid, _green_id) in models.items():
        (path / f"model_{model_id}.json").write_text(
            json.dumps({
                "model_id": model_id,
                "pid": pid,
                "state": "acknowledged",
            }),
            encoding="utf-8",
        )
        (path / f"model_{model_id}_inputs.jsonl").write_text(
            "".join(
                json.dumps({
                    "input_id": str(frame),
                    "ros_header_timestamp_ns": frame * 1_000_000,
                }) + "\n"
                for frame in range(3)
            ),
            encoding="utf-8",
        )


def test_per_frame_alignment_latency_waiting_and_green_context(tmp_path):
    """Each frame uses only H2D/D2H and the other attributed context."""
    models = _create_trace(tmp_path / "profile.sqlite")
    _write_model_evidence(tmp_path, models)

    rows, evidence = extract_frame_rows(tmp_path)
    assert len(rows) == 6
    assert evidence["all_model_kernels_have_nonzero_green_context_id"]
    assert evidence["model_kernel_green_context_ids"] == {
        "image-det": [7],
        "lidar-det": [8],
    }
    for model_id in models:
        group = [row for row in rows if row["model_id"] == model_id]
        expected = {
            "image-det": [0.75, 1.0, 1.0],
            "lidar-det": [0.5, 0.5, 1.0],
        }
        assert [row["alignment"] for row in group] == expected[model_id]
        assert [row["alignment_copy_count"] for row in group] == [2, 2, 2]
        assert [row["h2d_copy_count"] for row in group] == [1, 1, 1]
        assert [row["d2h_copy_count"] for row in group] == [1, 1, 1]
        assert [row["memcpy_adjacent_waiting_ns"] for row in group] == [
            100_000,
            200_000,
            300_000,
        ]
        assert all(row["gpu_kernel_active_ns"] > 0 for row in group)
        assert all(
            row["completion_boundary"]
            == "post_nvtx_cuda_event_synchronize"
            for row in group
        )
        assert all(row["completion_synchronize_ns"] == 200_000 for row in group)
        assert all(
            row["inference_e2e_ns"] == row["host_nvtx_latency_ns"] + 400_000
            for row in group
        )

    summary = summarize_rows(rows)
    for values in summary.values():
        assert values["frame_count"] == 3
        assert values["p99_frame_count"] == 2
    image_values = summary["image-det"]
    assert image_values["spearman_alignment_inference_e2e"] is None
    assert image_values["inference_e2e_slope_positive"]

    green_rows = []
    for row in rows:
        changed = dict(row)
        changed["host_nvtx_latency_ns"] *= 2
        changed["host_nvtx_latency_ms"] *= 2
        changed["inference_e2e_ns"] *= 2
        changed["inference_e2e_ms"] *= 2
        changed["gpu_kernel_active_ns"] *= 2
        changed["gpu_kernel_active_ms"] *= 2
        changed["kernel_span_ns"] *= 2
        changed["memcpy_adjacent_waiting_ns"] *= 2
        green_rows.append(changed)
    matched = _comparison_rows(rows, green_rows)
    comparison = _comparison_summary(
        rows,
        green_rows,
        matched,
        {"models": summarize_rows(rows)},
        {"models": summarize_rows(green_rows)},
    )
    for values in comparison.values():
        assert values["matched_frame_count"] == 3
        assert values["frame_count_change_percent"] == 0
        assert values["median_p99_inference_e2e_change_percent"] == 100
        assert values["median_paired_inference_e2e_change_percent"] == 100


def test_kernel_intervals_cover_traces_without_context_switch_events(tmp_path):
    """MPS traces derive kernel active time without context-switch tables."""
    models = _create_trace(tmp_path / "profile.sqlite")
    _write_model_evidence(tmp_path, models)
    with sqlite3.connect(tmp_path / "profile.sqlite") as database:
        database.execute("DROP TABLE GPU_CONTEXT_SWITCH_EVENTS")
        database.execute("DROP TABLE ENUM_GPU_CTX_SWITCH")

    rows, evidence = extract_frame_rows(tmp_path)

    assert len(rows) == 6
    assert evidence["context_owner_global_ids"] == {
        "image-det": None,
        "lidar-det": None,
    }
    assert set(evidence["active_interval_sources"].values()) == {
        "kernel_intervals"
    }
    assert all(row["gpu_kernel_active_ns"] > 0 for row in rows)


def test_incomplete_restore_markers_do_not_reject_full_bag_trace(tmp_path):
    """Dropped save markers retain only the unambiguous active intervals."""
    models = _create_trace(tmp_path / "profile.sqlite")
    _write_model_evidence(tmp_path, models)
    pid, green_id = models["image-det"]
    with sqlite3.connect(tmp_path / "profile.sqlite") as database:
        database.executemany(
            "INSERT INTO GPU_CONTEXT_SWITCH_EVENTS VALUES (?,?,?,?,?)",
            [
                (1, green_id + 100, _global_id(pid), 94_000_000, 999),
                (2, green_id + 100, _global_id(pid), 105_000_000, 1000),
                (1, green_id + 100, _global_id(pid), 110_000_000, 1001),
            ],
        )

    rows, _evidence = extract_frame_rows(tmp_path)

    assert len(rows) == 6


def test_three_model_alignment_uses_other_context_union(tmp_path):
    """Each target uses the de-duplicated union of both co-runners."""
    models = _create_trace(
        tmp_path / "profile.sqlite", include_segmentation=True
    )
    _write_model_evidence(tmp_path, models)

    rows, evidence = extract_frame_rows(tmp_path)

    assert len(rows) == 9
    assert evidence["model_kernel_green_context_ids"] == {
        "image-det": [7],
        "lidar-det": [8],
        "segmentation": [9],
    }
    expected = {
        "image-det": [1.0, 1.0, 1.0],
        "lidar-det": [0.5, 0.5, 1.0],
        "segmentation": [1.0, 1.0, 1.0],
    }
    for model_id, values in expected.items():
        group = [row for row in rows if row["model_id"] == model_id]
        assert [row["alignment"] for row in group] == values
        assert all(
            row["other_model_ids"] == ";".join(
                item for item in sorted(models) if item != model_id
            )
            for row in group
        )


def test_summary_uses_p99_post_preprocess_e2e_as_primary_inference_time():
    """The p99 filter and primary correlation use completion-bounded time."""
    rows = [
        {
            "model_id": "image-det",
            "alignment": alignment,
            "inference_e2e_ns": inference_e2e,
            "gpu_kernel_active_ns": gpu_active,
            "kernel_span_ns": 10 - gpu_active,
            "host_nvtx_latency_ns": host_latency,
            "memcpy_adjacent_waiting_ns": gpu_active,
        }
        for alignment, inference_e2e, gpu_active, host_latency in (
            (0.0, 3, 1, 4),
            (0.3, 2, 2, 3),
            (0.6, 1, 3, 2),
            (1.0, 100, 100, 1),
        )
    ]
    values = summarize_rows(rows)["image-det"]
    assert values["p99_frame_count"] == 3
    assert values["spearman_alignment_inference_e2e"] == pytest.approx(-1.0)
    assert values["spearman_alignment_gpu_kernel_active"] == pytest.approx(1.0)
    assert values["spearman_alignment_gpu_kernel_span"] == pytest.approx(-1.0)
    assert values["spearman_alignment_host_nvtx_latency"] == pytest.approx(-1.0)


def test_three_condition_scatter_uses_p99_inference_e2e_time(tmp_path):
    """The combined plot includes all models and configurations."""
    run_info = []
    for index, configuration in enumerate(
        ("whole_gpu", "green_28sm", "green_14sm")
    ):
        rows = [
            {
                "model_id": model_id,
                "alignment": 0.1 + index / 10,
                "inference_e2e_ns": (10 + index) * 1_000_000,
                "gpu_kernel_active_ns": (10 + index) * 1_000_000,
            }
            for model_id in ("image-det", "lidar-det")
        ]
        run_info.append({
            "configuration": configuration,
            "config": {
                "models": [
                    {
                        "id": "image-det",
                        "mmlab_model": "faster-rcnn_r50_fpn_1x_coco",
                    },
                    {
                        "id": "lidar-det",
                        "mmlab_model": "pointpillars_hv_secfpn_nus-3d",
                    },
                ],
            },
            "rows": rows,
            "summary": {
                "models": {
                    model_id: {
                        "spearman_alignment_inference_e2e": 0.5,
                        "inference_e2e_p99_ns": 12_000_000,
                        "p99_frame_count": 1,
                        "frame_count": 1,
                    }
                    for model_id in ("image-det", "lidar-det")
                }
            },
        })
    output = tmp_path / "scatter.svg"
    _write_correlation_comparison_scatter(output, run_info)
    svg = output.read_text(encoding="utf-8")
    assert svg.count("<circle") == 6
    assert "post-preprocess inference" in svg
    assert "Green Context 28 SM" in svg
    assert "PointPillars" in svg
    assert "CenterPoint" not in svg
    assert (
        '<text x="490" y="305" text-anchor="end" font-size="10">'
        '11.0</text>'
    ) in svg


def test_execution_mode_requires_consistent_model_contexts():
    """Comparison modes accept all-or-none Green Context assignment."""
    assert _execution_mode({"models": [{}, {}]}) == "whole_gpu"
    assert _execution_mode({
        "models": [
            {"green_context": {"sm_count": 28}},
            {"green_context": {"sm_count": 28}},
        ]
    }) == "green_context"
    with pytest.raises(StudyError, match="all models"):
        _execution_mode({"models": [{"green_context": {}}, {}]})
