"""Shared per-input timing extraction from immutable Nsight evidence."""

from bisect import bisect_left, bisect_right
import json
from pathlib import Path
import sqlite3
import sys

from .._common import (
    StudyError, active_context_intervals, context_owner,
    intersection_duration, memcopies, merge_intervals, nvtx_ranges,
    runtime_events, status_map, sync_wait, tables,
)


ALIGNMENT_WINDOW_NS = 2_000_000


def _pid(global_id):
    return ((int(global_id) >> 24) & 0xFFFFFF
            if global_id is not None else -1)


def _kernels(connection):
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in tables(connection):
        return []
    columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})")
    }
    green = "greenContextId" if "greenContextId" in columns else None
    selected = ["start", "end", "globalPid", "contextId"]
    if green:
        selected.append(green)
    result = []
    for values in connection.execute(
            f"SELECT {','.join(selected)} FROM {table}"):
        record = dict(zip(selected, values))
        if record["globalPid"] is None:
            continue
        result.append({
            "start": int(record["start"]),
            "end": int(record["end"]),
            "pid": _pid(record["globalPid"]),
            "context_id": int(record["contextId"]),
            "green_context_id": (
                int(record[green])
                if green and record[green] is not None else None
            ),
        })
    return result


def _input_records(run_directory, model_id):
    path = Path(run_directory) / f"model_{model_id}_inputs.jsonl"
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"invalid model input evidence: {path}") from exc
    result = {str(record["input_id"]): record for record in records}
    if len(result) != len(records):
        raise StudyError(f"duplicate input IDs in {path}")
    return result


def _frame_key(value):
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def _time_index(items):
    ordered = sorted(items, key=lambda item: item["start"])
    return ordered, [item["start"] for item in ordered]


def _in_time_range(index, interval):
    items, starts = index
    left = bisect_left(starts, interval["start"])
    right = bisect_left(starts, interval["end"])
    return [
        item for item in items[left:right] if item["end"] <= interval["end"]
    ]


def _alignment(copies, intervals, window_ns):
    ends = [end for _start, end in intervals]
    overlap = 0
    for copy in copies:
        start, end = copy["start"] - window_ns, copy["start"]
        index = bisect_right(ends, start)
        while index < len(intervals) and intervals[index][0] < end:
            left, right = intervals[index]
            overlap += max(0, min(end, right) - max(start, left))
            index += 1
    return overlap / (len(copies) * window_ns)


def _completion_boundary(inference, next_inference_start, runtime_indexes):
    candidates = []
    for events, starts in runtime_indexes.values():
        left = bisect_left(starts, inference["end"])
        right = bisect_left(starts, next_inference_start)
        for index in range(max(1, left), right):
            event = events[index]
            previous = events[index - 1]
            if (
                event["pid"] == inference["pid"]
                and event["name"]
                in ("cudaEventSynchronize", "cuEventSynchronize")
                and previous["name"]
                in ("cudaEventRecord", "cuEventRecord")
                and previous["start"] >= inference["end"]
            ):
                candidates.append(event)
                break
    if candidates:
        selected = min(candidates, key=lambda event: event["start"])
        return (
            selected["end"], "post_nvtx_cuda_event_synchronize",
            selected["end"] - selected["start"],
        )
    return inference["end"], "nvtx_includes_completion_synchronize", 0


def extract_frame_rows(
        run_directory, alignment_window_ns=ALIGNMENT_WINDOW_NS):
    """Extract per-input metrics from a single- or multi-model trace."""
    run_directory = Path(run_directory)
    statuses = status_map(run_directory)
    if not statuses:
        raise StudyError("frame extraction requires a model status")
    inputs = {
        model_id: _input_records(run_directory, model_id)
        for model_id in statuses
    }
    with sqlite3.connect(str(run_directory / "profile.sqlite")) as connection:
        ranges = nvtx_ranges(connection)
        kernels = _kernels(connection)
        copies = memcopies(connection)
        runtimes = runtime_events(connection)
        active = {}
        owners = {}
        interval_sources = {}
        for model_id, status in statuses.items():
            owner, errors = context_owner(
                connection, int(status["pid"]), kernels)
            if errors == ["GPU context-switch events absent"]:
                model_kernels = [
                    item for item in kernels
                    if item["pid"] == int(status["pid"])
                ]
                if not model_kernels:
                    raise StudyError(f"{model_id} has no attributed kernels")
                active[model_id] = merge_intervals(
                    (item["start"], item["end"]) for item in model_kernels
                )
                owners[model_id] = None
                interval_sources[model_id] = "kernel_intervals"
                continue
            if errors or owner is None:
                raise StudyError(
                    f"{model_id} context attribution failed: "
                    + "; ".join(errors)
                )
            intervals, _transitions, errors, _timestamps = (
                active_context_intervals(owner[1])
            )
            errors = [
                error for error in errors
                if not error.startswith((
                    "duplicate RESTORE_START", "unpaired RESTORE_START",
                    "unpaired SAVE_END",
                ))
            ]
            if errors:
                raise StudyError(
                    f"{model_id} context intervals failed: "
                    + "; ".join(errors)
                )
            active[model_id] = intervals
            owners[model_id] = int(owner[0])
            interval_sources[model_id] = "gpu_context_switch"

    rows = []
    model_ids = sorted(statuses)
    kernel_indexes = {
        int(status["pid"]): _time_index([
            item for item in kernels if item["pid"] == int(status["pid"])
        ]) for status in statuses.values()
    }
    copy_indexes = {
        int(status["pid"]): _time_index([
            item for item in copies if item["pid"] == int(status["pid"])
        ]) for status in statuses.values()
    }
    runtime_indexes = {
        global_tid: _time_index(events)
        for global_tid, events in runtimes.items()
    }
    for model_id in model_ids:
        other_ids = [item for item in model_ids if item != model_id]
        other_active = merge_intervals([
            interval for item in other_ids for interval in active[item]
        ])
        pid = int(statuses[model_id]["pid"])
        inference_ranges = [
            item for item in ranges
            if item["pid"] == pid
            and item["tag"].get("event") == "inference"
            and item["tag"].get("model") == model_id
            and str(item["tag"].get("input")) in inputs[model_id]
        ]
        inference_ranges.sort(key=lambda item: item["start"])
        next_start = {
            str(inference["tag"]["input"]): (
                inference_ranges[index + 1]["start"]
                if index + 1 < len(inference_ranges) else sys.maxsize
            )
            for index, inference in enumerate(inference_ranges)
        }
        by_input = {}
        for inference in inference_ranges:
            input_id = str(inference["tag"]["input"])
            if input_id in by_input:
                raise StudyError(
                    f"multiple inference ranges for {model_id}/{input_id}")
            by_input[input_id] = inference
        if set(by_input) != set(inputs[model_id]):
            raise StudyError(
                f"{model_id} inference/input evidence differs: "
                f"ranges={len(by_input)} inputs={len(inputs[model_id])}"
            )
        for input_id in sorted(by_input, key=_frame_key):
            inference = by_input[input_id]
            completion_end, boundary, completion_sync = _completion_boundary(
                inference, next_start[input_id], runtime_indexes)
            execution = {
                "start": inference["start"], "end": completion_end,
                "pid": pid,
            }
            target_kernels = _in_time_range(kernel_indexes[pid], execution)
            input_copies = _in_time_range(copy_indexes[pid], execution)
            if not target_kernels:
                raise StudyError(f"{model_id}/{input_id} has no kernels")
            if not input_copies:
                raise StudyError(f"{model_id}/{input_id} has no copies")
            green_ids = sorted({
                item["green_context_id"] for item in target_kernels
                if item["green_context_id"] not in (None, 0)
            })
            first_kernel = min(item["start"] for item in target_kernels)
            last_kernel = max(item["end"] for item in target_kernels)
            kernel_active = intersection_duration(
                [(item["start"], item["end"]) for item in target_kernels],
                active[model_id],
            )
            record = inputs[model_id][input_id]
            rows.append({
                "model_id": model_id,
                "other_model_ids": ";".join(other_ids),
                "frame_id": input_id,
                "ros_header_timestamp_ns": record.get(
                    "ros_header_timestamp_ns"),
                "host_nvtx_start_ns": inference["start"],
                "host_nvtx_end_ns": inference["end"],
                "host_nvtx_latency_ns": inference["end"] - inference["start"],
                "host_nvtx_latency_ms": (
                    inference["end"] - inference["start"]) / 1_000_000,
                "inference_e2e_end_ns": completion_end,
                "inference_e2e_ns": completion_end - inference["start"],
                "inference_e2e_ms": (
                    completion_end - inference["start"]) / 1_000_000,
                "completion_boundary": boundary,
                "completion_synchronize_ns": completion_sync,
                "kernel_span_ns": last_kernel - first_kernel,
                "kernel_span_ms": (last_kernel - first_kernel) / 1_000_000,
                "gpu_kernel_active_ns": kernel_active,
                "gpu_kernel_active_ms": kernel_active / 1_000_000,
                "memcpy_adjacent_waiting_ns": sync_wait(
                    input_copies,
                    {
                        global_tid: _in_time_range(index, inference)
                        for global_tid, index in runtime_indexes.items()
                        if index[0] and index[0][0]["pid"] == pid
                    },
                    inference,
                ),
                "alignment": _alignment(
                    input_copies, other_active, int(alignment_window_ns)),
                "alignment_window_ns": int(alignment_window_ns),
                "alignment_copy_count": len(input_copies),
                "h2d_copy_count": sum(
                    item["copy_class"] == "H2D" for item in input_copies),
                "d2h_copy_count": sum(
                    item["copy_class"] == "D2H" for item in input_copies),
                "kernel_count": len(target_kernels),
                "green_context_ids": ";".join(map(str, green_ids)),
            })
    evidence = {
        "completion_boundary_counts": {
            boundary: sum(row["completion_boundary"] == boundary
                          for row in rows)
            for boundary in sorted({row["completion_boundary"]
                                    for row in rows})
        },
        "context_owner_global_ids": owners,
        "active_interval_sources": interval_sources,
        "model_kernel_green_context_ids": {
            model_id: sorted({
                item["green_context_id"] for item in kernels
                if item["pid"] == int(statuses[model_id]["pid"])
                and item["green_context_id"] not in (None, 0)
            }) for model_id in model_ids
        },
        "all_model_kernels_have_nonzero_green_context_id": all(
            item["green_context_id"] not in (None, 0)
            for item in kernels
            if item["pid"] in {
                int(status["pid"]) for status in statuses.values()
            }
        ),
        "all_model_kernels_have_zero_green_context_id": all(
            item["green_context_id"] in (None, 0)
            for item in kernels
            if item["pid"] in {
                int(status["pid"]) for status in statuses.values()
            }
        ),
    }
    return rows, evidence
