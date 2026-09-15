"""Synthetic coverage for observational MPS leftover enumeration."""

import csv
import json
from pathlib import Path
import sqlite3

from closeloop_analyzer.mps_leftover import (
    MpsLeftoverAnalyzer, compare_kernel_inventories,
    formal_leftover_metrics,
)


def _make_trace(path: Path) -> None:
    with sqlite3.connect(str(path)) as db:
        db.executescript("""
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
          start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER,
          correlationId INTEGER, nameId INTEGER, returnValue INTEGER,
          callchainId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
          start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
          greenContextId INTEGER, streamId INTEGER, correlationId INTEGER,
          globalPid INTEGER, demangledName INTEGER, shortName INTEGER,
          mangledName INTEGER, launchType INTEGER, cacheConfig INTEGER,
          registersPerThread INTEGER, gridX INTEGER, gridY INTEGER,
          gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER,
          staticSharedMemory INTEGER, dynamicSharedMemory INTEGER,
          localMemoryPerThread INTEGER, localMemoryTotal INTEGER,
          gridId INTEGER, sharedMemoryExecuted INTEGER, graphNodeId INTEGER,
          sharedMemoryLimitConfig INTEGER);
        CREATE TABLE NVTX_EVENTS(
          start INTEGER, end INTEGER, globalTid INTEGER, text TEXT);
        """)
        db.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [
                (1, "cudaLaunchKernel_v7000"),
                (2, "faster_aggressor"),
                (3, "deeplab_exact"),
                (4, "deeplab_delayed"),
                (5, "deeplab_graph_delayed"),
                (6, "deeplab_aggressor"),
                (7, "faster_delayed"),
                (8, "preprocess_or_warmup"),
            ],
        )
        global_ids = {10: 10 << 24, 20: 20 << 24}
        records = [
            # pid, correlation, name, stream, launch end, start, end, graph
            (10, 1, 2, 1, 999_900, 1_000_000, 2_200_000, None),
            (10, 2, 2, 2, 1_399_900, 1_400_000, 2_200_000, None),
            (10, 3, 2, 3, 2_299_900, 2_300_000, 3_600_000, None),
            (10, 4, 7, 4, 5_100_000, 6_200_001, 6_200_101, None),
            (20, 5, 3, 7, 10, 1_000_010, 1_000_110, None),
            (20, 6, 4, 8, 1_100_000, 2_100_001, 2_100_101, None),
            (20, 7, 5, 9, 2_400_000, 3_500_001, 3_500_101, 9),
            (20, 8, 6, 10, 5_199_900, 5_200_000, 6_300_000, None),
            (10, 9, 2, 11, 3_899_900, 3_900_000, 4_000_000, None),
            (10, 10, 8, 12, 7_899_900, 7_900_000, 8_000_000, None),
            (20, 11, 8, 13, 8_899_900, 8_900_000, 9_000_000, None),
        ]
        for (
            pid, correlation, name, stream, launch_end, start, end, graph
        ) in records:
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
                "(?,?,?,?,?,?,?,?)",
                (launch_end - 10, launch_end, 0, global_ids[pid],
                 correlation, 1, 0, None),
            )
            values = [
                start, end, 0, 1, 0, stream, correlation, global_ids[pid],
                name, name, None, 1, 0, 32 if correlation == 9 else 16,
                1, 1, 1, 32, 1, 1, 0, 0, 0,
                0, correlation, 0, graph, 0,
            ]
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ("
                + ",".join("?" for _ in values) + ")",
                values,
            )
        for pid, model in ((10, "faster_rcnn"), (20, "deeplabv3plus")):
            tag = "closeloop:" + json.dumps({
                "event": "inference", "input": "0", "model": model,
                "run": "synthetic", "scene": "scene", "schema_version": 1,
            })
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
                (0, 6_000_000 if pid == 10 else 7_000_000,
                 global_ids[pid], tag),
            )
        warmup_tag = "closeloop:" + json.dumps({
            "event": "inference", "input": "warmup-0",
            "model": "faster_rcnn", "run": "synthetic",
            "scene": "scene", "schema_version": 1,
        })
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
            (7_500_000, 8_500_000, global_ids[10], warmup_tag),
        )


def test_all_delay_aggressor_occurrences_are_retained_and_grouped(tmp_path):
    """Keep co-aggressors, directions, duplicates, and rejection evidence."""
    _make_trace(tmp_path / "profile.sqlite")
    for model, pid in (("faster_rcnn", 10), ("deeplabv3plus", 20)):
        (tmp_path / f"model_{model}.json").write_text(
            json.dumps({"model_id": model, "pid": pid, "inputs": 1}),
            encoding="utf-8",
        )

    result = MpsLeftoverAnalyzer(tmp_path, 1.0).analyze()

    assert result["schema"] == "mps_leftover_analysis_v3"
    assert result["occurrence_count"] == 4
    assert {item["direction"] for item in result["directions"]} == {
        "faster_rcnn->deeplabv3plus",
        "deeplabv3plus->faster_rcnn",
    }
    deeplab = result["models"]["deeplabv3plus"]
    assert deeplab["eligible_kernel_count"] == 4
    assert deeplab["delayed_kernel_count"] == 2
    assert deeplab["per_scene"]["unknown"] == {
        "eligible_kernel_count": 4, "delayed_kernel_count": 2,
    }
    faster = result["models"]["faster_rcnn"]
    gpu_complete = faster["gpu_complete_inference_duration"]
    assert gpu_complete["count"] == 1
    assert gpu_complete["max_ns"] == 6_200_101
    assert gpu_complete["p99_ns"] == 6_200_101
    assert gpu_complete["max_ns"] > 6_000_000
    assert faster["gpu_complete_inference_durations_ns"] == [6_200_101]

    inventory = result["kernel_inventory"]
    assert sum(row["kernel_count"] for row in inventory
               if row["model"] == "faster_rcnn") == 5
    assert sum(row["kernel_count"] for row in inventory
               if row["model"] == "deeplabv3plus") == 4
    faster_aggressors = [
        row for row in inventory
        if row["model"] == "faster_rcnn"
        and row["symbol"] == "faster_aggressor"
    ]
    assert sorted(row["kernel_count"] for row in faster_aggressors) == [1, 3]
    assert len({row["inventory_signature"]
                for row in faster_aggressors}) == 2
    assert (tmp_path / "mps_aggressor_kernels.csv").is_file()
    assert {item["evidence_grade"] for item in result["directions"]} == {
        "observed"
    }

    with (tmp_path / "mps_leftover_occurrences.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        occurrences = list(csv.DictReader(source))
    assert len(occurrences) == 4
    delayed = [
        row for row in occurrences
        if row["victim_kernel_name"] == "deeplab_delayed"
    ]
    assert len(delayed) == 2
    assert {row["co_aggressor_count"] for row in delayed} == {"2"}
    assert any(int(row["aggressor_duration_ns"]) < 1_000_000
               for row in delayed)
    assert any("below capsule" in row["capsule_rejection_reason"]
               for row in delayed)
    assert any("graph" in row["capsule_rejection_reason"]
               for row in occurrences)
    assert all(float(row["normalized_victim_start_position"]) >= 0.8
               for row in occurrences)
    assert all(float(row["normalized_victim_start_position"]) < 1.0
               for row in occurrences)
    assert all(int(row["terminal_overlap_ns"]) > 0
               for row in occurrences)
    inventory_signatures = {
        row["inventory_signature"] for row in inventory
    }
    assert all(
        row["aggressor_inventory_signature"] in inventory_signatures
        for row in occurrences
    )

    with (tmp_path / "mps_leftover_groups.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        groups = list(csv.DictReader(source))
    duplicate = next(
        row for row in groups
        if row["victim_kernel_name"] == "deeplab_delayed"
    )
    assert duplicate["occurrence_count"] == "2"
    assert (tmp_path / "mps_leftover_analysis.json").is_file()


def test_not_observed_grade_and_exact_threshold_exclusion(tmp_path):
    """Grade empty directions and keep the 1 ms comparison strict."""
    _make_trace(tmp_path / "profile.sqlite")
    for model, pid in (("faster_rcnn", 10), ("deeplabv3plus", 20)):
        (tmp_path / f"model_{model}.json").write_text(
            json.dumps({"model_id": model, "pid": pid, "inputs": 1}),
            encoding="utf-8",
        )
    one_ms = MpsLeftoverAnalyzer(tmp_path, 1.0).analyze()
    assert one_ms["models"]["deeplabv3plus"]["delayed_kernel_count"] == 2
    no_candidates = MpsLeftoverAnalyzer(tmp_path, 10.0).analyze()
    assert {row["evidence_grade"] for row in no_candidates["directions"]} == {
        "not_observed_in_trace"
    }


def test_inventory_comparison_reports_additions_and_removals():
    """Compare the resource signatures against one same-cap baseline."""
    metric = {
        "symbol": "kernel", "kernel_count_per_inference": 2,
        "duration_p50_ns": 10, "duration_p95_ns": 20,
        "summed_kernel_time_per_inference_ns": 30,
        "model_kernel_time_share": 0.5,
    }
    current = [{"inventory_signature": "kept", **metric}, {
        "inventory_signature": "added", **metric,
    }]
    baseline = [{"inventory_signature": "kept", **metric}, {
        "inventory_signature": "removed", **metric,
    }]
    changes = compare_kernel_inventories(current, baseline)
    assert {row["change"] for row in changes} == {
        "added", "present", "removed"
    }


def test_formal_leftover_requires_final_20_percent_and_terminal_overlap():
    """Apply inclusive 0.80, exclusive 1.0, and positive overlap exactly."""
    assert formal_leftover_metrics(80, 0, 100) == (0.8, 20)
    assert formal_leftover_metrics(99, 0, 100) == (0.99, 1)
    assert formal_leftover_metrics(79, 0, 100) is None
    assert formal_leftover_metrics(100, 0, 100) is None
    assert formal_leftover_metrics(101, 0, 100) is None
