"""Tests for preprocessing and scheduler-state correlation."""

import csv
import json
import sqlite3

from closeloop_analyzer.preprocessing_analyzer import (
    PreprocessAnalyzer, SCHEDULER_FIELDS
)


def encode_tag(run_id, scene_token, model_id, input_id, event_type):
    """Build the collector/analyzer NVTX interchange format."""
    value = {
        "event": event_type,
        "input": str(input_id),
        "model": model_id,
        "run": run_id,
        "scene": scene_token,
        "schema_version": 1,
    }
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    )
    return "closeloop:" + payload


def global_id(pid, tid):
    """Encode the Nsight global-ID layout used by the analyzer."""
    return (pid << 24) | tid


def test_analyzer_writes_compact_absolute_tail_planner_result(tmp_path):
    """Planner output uses exclusive p95/p99 and omits CV-driven bulk."""
    database = tmp_path / "profile.sqlite"
    connection = sqlite3.connect(str(database))
    connection.execute(
        "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, "
        "text TEXT, globalTid INTEGER)"
    )
    connection.execute(
        "CREATE TABLE SCHED_EVENTS (start INTEGER, cpu INTEGER, "
        "isSchedIn INTEGER, globalTid INTEGER)"
    )
    pid = 100
    calling_tid = 100
    worker_tid = 101
    stage_specs = [
        (1_000_000, 2_000_000, 1_100_000, 1_200_000),
        (3_000_000, 5_000_000, 3_100_000, 3_600_000),
        (6_000_000, 10_000_000, 6_100_000, 8_000_000),
    ]
    scheduler_rows = []
    for index, (start, end, wakeup, scheduled) in enumerate(stage_specs):
        tag = encode_tag(
            "run", "scene", "image", str(index), "preprocess"
        )
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            (start, end, tag, global_id(pid, calling_tid)),
        )
        scheduler_rows.extend([
            (scheduled, 2, 1, global_id(pid, worker_tid)),
            (end - 5, 2, 0, global_id(pid, worker_tid)),
        ])
    connection.executemany(
        "INSERT INTO SCHED_EVENTS VALUES (?, ?, ?, ?)", scheduler_rows
    )
    connection.commit()
    connection.close()

    fields = [
        "timestamp_ns", "event", "cpu", "prev_tid", "prev_state",
        "next_tid", "target_tid", "parent_tid", "child_tid",
        "orig_cpu", "dest_cpu", "comm",
    ]
    with (tmp_path / "scheduler_events.csv").open(
            "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for start, end, wakeup, scheduled in stage_specs:
            writer.writerow({
                "timestamp_ns": wakeup, "event": "wakeup", "cpu": 0,
                "prev_tid": 0, "prev_state": 0, "next_tid": 0,
                "target_tid": worker_tid, "parent_tid": 0,
                "child_tid": 0, "orig_cpu": 0, "dest_cpu": 2,
                "comm": "model_node",
            })
            writer.writerow({
                "timestamp_ns": scheduled, "event": "switch", "cpu": 2,
                "prev_tid": 0, "prev_state": 0,
                "next_tid": worker_tid, "target_tid": 0,
                "parent_tid": 0, "child_tid": 0, "orig_cpu": 0,
                "dest_cpu": 0, "comm": "idle",
            })
            writer.writerow({
                "timestamp_ns": end - 5, "event": "switch", "cpu": 2,
                "prev_tid": worker_tid, "prev_state": 1,
                "next_tid": 0, "target_tid": 0, "parent_tid": 0,
                "child_tid": 0, "orig_cpu": 0, "dest_cpu": 0,
                "comm": "model_node",
            })

    result = PreprocessAnalyzer(
        tmp_path,
        {"materiality_threshold_ms": 1.0},
    ).analyze()
    assert result["scheduler_source"] == "bpftrace"
    assert result["schema_version"] == 2
    assert result["objective"]["materiality_threshold_ms"] == 1.0
    assert result["material_actions"][0]["action"] == "preprocess"
    assert result["material_actions"][0]["exclusive_p95_ms"] == 3.8
    assert len(result["mechanism_coverage"]) == 4
    assert "verdict" not in result
    assert "stage_summaries" not in result
    assert (tmp_path / "preprocess_thread_states.csv").is_file()
    assert (tmp_path / "preprocess_stage_summary.csv").is_file()
    saved = json.loads(
        (tmp_path / "preprocess_analysis.json").read_text(encoding="utf-8")
    )
    assert saved["skill"] == "preprocess_contention"
    assert len(json.dumps(saved)) < 20_000


def test_scheduler_events_accept_headerless_bpftrace_output(tmp_path):
    """The bpftrace collector does not need an unsupported BEGIN probe."""
    scheduler_path = tmp_path / "scheduler_events.csv"
    scheduler_path.write_text(
        "100,switch,2,10,0,11,0,0,0,0,0,worker\n",
        encoding="utf-8",
    )

    analyzer = PreprocessAnalyzer(tmp_path)
    events = analyzer._raw_scheduler()

    assert tuple(SCHEDULER_FIELDS) == (
        "timestamp_ns", "event", "cpu", "prev_tid", "prev_state",
        "next_tid", "target_tid", "parent_tid", "child_tid",
        "orig_cpu", "dest_cpu", "comm",
    )
    assert len(events) == 1
    assert events[0].timestamp_ns == 100
    assert events[0].next_tid == 11
    assert events[0].comm == "worker"


def test_pool_identity_keeps_pytorch_owner_and_libgomp_backend():
    """PyTorch and libgomp describe one pool instead of two pools."""
    identity = PreprocessAnalyzer._pool_identity({
        "creator_owner": "libgomp.so.1",
        "creator_stack": (
            "GOMP_parallel@libgomp.so.1;"
            "at::TensorIteratorBase::for_each@libtorch_cpu.so"
        ),
    })

    assert identity == ("pytorch", "intra_op", "libgomp")
