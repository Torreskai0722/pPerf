"""Run and analyze one natural-timing full-bag Green Context experiment."""

import argparse
from bisect import bisect_left, bisect_right
import csv
import html
import json
from pathlib import Path
import sqlite3
import statistics
import sys

import yaml

from .config import ConfigError, load_run_config
from .non_mps_memcpy_compute import (
    StudyError,
    _active_context_intervals,
    _context_owner,
    _intersection_duration,
    _memcopies,
    _merge_intervals,
    _nvtx_ranges,
    _runtime_events,
    _spearman,
    _status_map,
    _sync_wait,
    _tables,
    _write_csv,
    _write_json,
)
from closeloop_analyzer.mps_leftover import _percentile
from closeloop_testbed.resource_control import (
    GPUClockLock,
    MPS_ENVIRONMENT_VARIABLES,
    current_gpu_clocks,
    maximum_supported_clocks,
)
from .runner import ExperimentRunner, PreflightError, RunFailure
from closeloop_analyzer.target_selection import decode_global_id


ALIGNMENT_WINDOW_NS = 2_000_000
ALIGNMENT_METRIC = "h2d_d2h_preceding_other_context_union_window_v2"
PRIMARY_INFERENCE_METRIC = (
    "post_cpu_preprocess_to_gpu_completion_wall_duration"
)
FRAME_SELECTION = "inference_e2e_time_at_or_below_per_model_configuration_p99"


def _pid(global_id):
    return decode_global_id(int(global_id))[0] if global_id is not None else -1


def _green_kernels(connection):
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in _tables(connection):
        return []
    columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})")
    }
    green = "greenContextId" if "greenContextId" in columns else None
    selected = ["start", "end", "globalPid", "contextId"]
    if green:
        selected.append(green)
    rows = []
    for values in connection.execute(
        f"SELECT {','.join(selected)} FROM {table}"
    ):
        record = dict(zip(selected, values))
        if record["globalPid"] is None:
            continue
        rows.append({
            "start": int(record["start"]),
            "end": int(record["end"]),
            "pid": _pid(record["globalPid"]),
            "context_id": int(record["contextId"]),
            "green_context_id": (
                int(record[green])
                if green and record[green] is not None else None
            ),
        })
    return rows


def _input_records(run_directory, model_id):
    path = Path(run_directory) / f"model_{model_id}_inputs.jsonl"
    try:
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
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
    return [item for item in items[left:right] if item["end"] <= interval["end"]]


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
    """Return the post-test-step CUDA completion boundary for one frame."""
    candidates = []
    for events, starts in runtime_indexes.values():
        left = bisect_left(starts, inference["end"])
        right = bisect_left(starts, next_inference_start)
        for index in range(max(1, left), right):
            event = events[index]
            previous = events[index - 1]
            if (
                event["pid"] == inference["pid"]
                and event["name"] in (
                    "cudaEventSynchronize", "cuEventSynchronize"
                )
                and previous["name"] in (
                    "cudaEventRecord", "cuEventRecord"
                )
                and previous["start"] >= inference["end"]
            ):
                candidates.append(event)
                break
    if candidates:
        selected = min(candidates, key=lambda event: event["start"])
        return (
            selected["end"],
            "post_nvtx_cuda_event_synchronize",
            selected["end"] - selected["start"],
        )
    return inference["end"], "nvtx_includes_completion_synchronize", 0


def extract_frame_rows(run_directory, alignment_window_ns=ALIGNMENT_WINDOW_NS):
    """Extract per-frame metrics from a single- or multi-model Nsight trace."""
    run_directory = Path(run_directory)
    statuses = _status_map(run_directory)
    if not statuses:
        raise StudyError("frame extraction requires at least one model status")
    inputs = {
        model_id: _input_records(run_directory, model_id)
        for model_id in statuses
    }
    sqlite_path = run_directory / "profile.sqlite"
    with sqlite3.connect(str(sqlite_path)) as connection:
        ranges = _nvtx_ranges(connection)
        kernels = _green_kernels(connection)
        memcopies = _memcopies(connection)
        runtimes = _runtime_events(connection)
        active = {}
        context_owners = {}
        active_interval_sources = {}
        for model_id, status in statuses.items():
            owner, owner_errors = _context_owner(
                connection, int(status["pid"]), kernels
            )
            if owner_errors == ["GPU context-switch events absent"]:
                model_kernels = [
                    item for item in kernels
                    if item["pid"] == int(status["pid"])
                ]
                if not model_kernels:
                    raise StudyError(f"{model_id} has no attributed kernels")
                active[model_id] = _merge_intervals(
                    (item["start"], item["end"])
                    for item in model_kernels
                )
                context_owners[model_id] = None
                active_interval_sources[model_id] = "kernel_intervals"
                continue
            if owner_errors or owner is None:
                raise StudyError(
                    f"{model_id} context attribution failed: "
                    + "; ".join(owner_errors)
                )
            intervals, _transitions, errors, _timestamps = (
                _active_context_intervals(owner[1])
            )
            errors = [
                error for error in errors
                if not error.startswith((
                    "duplicate RESTORE_START",
                    "unpaired RESTORE_START",
                    "unpaired SAVE_END",
                ))
            ]
            if errors:
                raise StudyError(
                    f"{model_id} context intervals failed: "
                    + "; ".join(errors)
                )
            active[model_id] = intervals
            context_owners[model_id] = int(owner[0])
            active_interval_sources[model_id] = "gpu_context_switch"

    rows = []
    model_ids = sorted(statuses)
    kernel_indexes = {
        int(status["pid"]): _time_index([
            item for item in kernels if item["pid"] == int(status["pid"])
        ])
        for status in statuses.values()
    }
    copy_indexes = {
        int(status["pid"]): _time_index([
            item for item in memcopies if item["pid"] == int(status["pid"])
        ])
        for status in statuses.values()
    }
    runtime_indexes = {
        global_tid: _time_index(events)
        for global_tid, events in runtimes.items()
    }
    for model_id in model_ids:
        other_ids = [item for item in model_ids if item != model_id]
        other_active = _merge_intervals([
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
        next_inference_start = {
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
                    f"multiple inference ranges for {model_id} input {input_id}"
                )
            by_input[input_id] = inference
        if set(by_input) != set(inputs[model_id]):
            raise StudyError(
                f"{model_id} inference/input evidence differs: "
                f"ranges={len(by_input)} inputs={len(inputs[model_id])}"
            )
        for input_id in sorted(by_input, key=_frame_key):
            inference = by_input[input_id]
            completion_end, completion_boundary, completion_sync = (
                _completion_boundary(
                    inference, next_inference_start[input_id], runtime_indexes
                )
            )
            execution = {
                "start": inference["start"],
                "end": completion_end,
                "pid": pid,
            }
            target_kernels = _in_time_range(kernel_indexes[pid], execution)
            copies = _in_time_range(copy_indexes[pid], execution)
            if not target_kernels:
                raise StudyError(f"{model_id} input {input_id} has no kernels")
            if not copies:
                raise StudyError(
                    f"{model_id} input {input_id} has no H2D/D2H copies"
                )
            green_ids = sorted({
                item["green_context_id"] for item in target_kernels
                if item["green_context_id"] not in (None, 0)
            })
            score = _alignment(copies, other_active, int(alignment_window_ns))
            first_kernel = min(item["start"] for item in target_kernels)
            last_kernel = max(item["end"] for item in target_kernels)
            kernel_active = _intersection_duration(
                [(item["start"], item["end"]) for item in target_kernels],
                active[model_id],
            )
            record = inputs[model_id][input_id]
            rows.append({
                "model_id": model_id,
                "other_model_ids": ";".join(other_ids),
                "frame_id": input_id,
                "ros_header_timestamp_ns": record.get(
                    "ros_header_timestamp_ns"
                ),
                "host_nvtx_start_ns": inference["start"],
                "host_nvtx_end_ns": inference["end"],
                "host_nvtx_latency_ns": inference["end"] - inference["start"],
                "host_nvtx_latency_ms": (
                    inference["end"] - inference["start"]
                ) / 1_000_000,
                "inference_e2e_end_ns": completion_end,
                "inference_e2e_ns": completion_end - inference["start"],
                "inference_e2e_ms": (
                    completion_end - inference["start"]
                ) / 1_000_000,
                "completion_boundary": completion_boundary,
                "completion_synchronize_ns": completion_sync,
                "kernel_span_ns": last_kernel - first_kernel,
                "kernel_span_ms": (last_kernel - first_kernel) / 1_000_000,
                "gpu_kernel_active_ns": kernel_active,
                "gpu_kernel_active_ms": kernel_active / 1_000_000,
                "memcpy_adjacent_waiting_ns": _sync_wait(
                    copies,
                    {
                        global_tid: _in_time_range(index, inference)
                        for global_tid, index in runtime_indexes.items()
                        if index[0] and index[0][0]["pid"] == pid
                    },
                    inference,
                ),
                "alignment": score,
                "alignment_window_ns": int(alignment_window_ns),
                "alignment_copy_count": len(copies),
                "h2d_copy_count": sum(
                    item["copy_class"] == "H2D" for item in copies
                ),
                "d2h_copy_count": sum(
                    item["copy_class"] == "D2H" for item in copies
                ),
                "kernel_count": len(target_kernels),
                "green_context_ids": ";".join(map(str, green_ids)),
            })
    evidence = {
        "completion_boundary_counts": {
            boundary: sum(
                row["completion_boundary"] == boundary for row in rows
            )
            for boundary in sorted({
                row["completion_boundary"] for row in rows
            })
        },
        "context_owner_global_ids": context_owners,
        "active_interval_sources": active_interval_sources,
        "model_kernel_green_context_ids": {
            model_id: sorted({
                item["green_context_id"]
                for item in kernels
                if item["pid"] == int(statuses[model_id]["pid"])
                and item["green_context_id"] not in (None, 0)
            })
            for model_id in model_ids
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


def _linear_slope(left, right):
    if len(left) < 2 or len(left) != len(right):
        return None
    mean_left = statistics.fmean(left)
    denominator = sum((value - mean_left) ** 2 for value in left)
    if not denominator:
        return None
    mean_right = statistics.fmean(right)
    return sum(
        (x - mean_left) * (y - mean_right)
        for x, y in zip(left, right)
    ) / denominator


def _p99_rows(rows):
    p99 = _percentile(
        [row["inference_e2e_ns"] for row in rows], 0.99
    )
    return p99, [row for row in rows if row["inference_e2e_ns"] <= p99]


def summarize_rows(rows):
    """Return post-preprocessing inference statistics by model."""
    result = {}
    for model_id in sorted({row["model_id"] for row in rows}):
        group = [row for row in rows if row["model_id"] == model_id]
        observed_e2e = [row["inference_e2e_ns"] for row in group]
        observed_active = [row["gpu_kernel_active_ns"] for row in group]
        p99, included = _p99_rows(group)
        alignments = [row["alignment"] for row in included]
        inference_e2e = [row["inference_e2e_ns"] for row in included]
        kernel_active = [row["gpu_kernel_active_ns"] for row in included]
        kernel_spans = [row["kernel_span_ns"] for row in included]
        host_latencies = [row["host_nvtx_latency_ns"] for row in included]
        waiting = [row["memcpy_adjacent_waiting_ns"] for row in included]
        correlation = _spearman(alignments, inference_e2e)
        slope = _linear_slope(alignments, inference_e2e)
        active_correlation = _spearman(alignments, kernel_active)
        active_slope = _linear_slope(alignments, kernel_active)
        span_correlation = _spearman(alignments, kernel_spans)
        span_slope = _linear_slope(alignments, kernel_spans)
        host_correlation = _spearman(alignments, host_latencies)
        host_slope = _linear_slope(alignments, host_latencies)
        wait_correlation = _spearman(alignments, waiting)
        result[model_id] = {
            "frame_count": len(group),
            "p99_frame_count": len(included),
            "inference_e2e_p99_ns": p99,
            "alignment_range": [min(alignments), max(alignments)],
            "median_alignment": statistics.median(alignments),
            "inference_e2e_ns_range": [
                min(inference_e2e), max(inference_e2e)
            ],
            "observed_inference_e2e_ns_range": [
                min(observed_e2e), max(observed_e2e)
            ],
            "median_inference_e2e_ns": statistics.median(inference_e2e),
            "gpu_kernel_active_ns_range": [
                min(kernel_active), max(kernel_active)
            ],
            "observed_gpu_kernel_active_ns_range": [
                min(observed_active), max(observed_active)
            ],
            "median_gpu_kernel_active_ns": statistics.median(kernel_active),
            "gpu_kernel_span_ns_range": [min(kernel_spans), max(kernel_spans)],
            "median_gpu_kernel_span_ns": statistics.median(kernel_spans),
            "host_nvtx_latency_ns_range": [
                min(host_latencies), max(host_latencies)
            ],
            "median_host_nvtx_latency_ns": statistics.median(host_latencies),
            "median_memcpy_adjacent_waiting_ns": statistics.median(waiting),
            "spearman_alignment_inference_e2e": correlation,
            "linear_slope_inference_e2e_ns_per_alignment": slope,
            "spearman_alignment_gpu_kernel_active": active_correlation,
            "linear_slope_gpu_kernel_active_ns_per_alignment": active_slope,
            "spearman_alignment_gpu_kernel_span": span_correlation,
            "linear_slope_gpu_kernel_span_ns_per_alignment": span_slope,
            "spearman_alignment_host_nvtx_latency": host_correlation,
            "linear_slope_host_nvtx_latency_ns_per_alignment": host_slope,
            "spearman_alignment_waiting": wait_correlation,
            "alignment_inference_e2e_correlation_positive": (
                correlation > 0 if correlation is not None else None
            ),
            "inference_e2e_slope_positive": (
                slope > 0 if slope is not None else None
            ),
        }
    return result


def _write_scatter(path, rows):
    groups = [
        (
            model_id,
            _p99_rows([
                row for row in rows if row["model_id"] == model_id
            ])[1],
        )
        for model_id in sorted({row["model_id"] for row in rows})
    ]
    width, panel_height, margin = 760, 300, 55
    height = panel_height * len(groups)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for panel, (model_id, group) in enumerate(groups):
        top = panel * panel_height
        plot_left, plot_right = margin, width - 25
        plot_top, plot_bottom = top + 35, top + panel_height - margin
        xs = [row["alignment"] for row in group]
        ys = [row["inference_e2e_ms"] for row in group]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = x_max - x_min or 1
        y_span = y_max - y_min or 1
        parts.extend([
            f'<text x="{margin}" y="{top + 20}" font-size="15">'
            f'{html.escape(model_id)}</text>',
            f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
            f'y2="{plot_bottom}" stroke="black"/>',
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
            f'y2="{plot_bottom}" stroke="black"/>',
            f'<text x="{(plot_left + plot_right) / 2}" y="{top + panel_height - 10}" '
            'text-anchor="middle" font-size="12">alignment A</text>',
            f'<text x="5" y="{(plot_top + plot_bottom) / 2}" font-size="12">'
            'Post-preprocess inference ms (≤p99)</text>',
        ])
        for x, y in zip(xs, ys):
            px = plot_left + (x - x_min) / x_span * (plot_right - plot_left)
            py = plot_bottom - (y - y_min) / y_span * (plot_bottom - plot_top)
            parts.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" '
                'fill="#2563eb" fill-opacity="0.7"/>'
            )
        parts.extend([
            f'<text x="{plot_left}" y="{plot_bottom + 16}" font-size="10">'
            f'{x_min:.3f}</text>',
            f'<text x="{plot_right}" y="{plot_bottom + 16}" '
            f'text-anchor="end" font-size="10">{x_max:.3f}</text>',
            f'<text x="{plot_left - 5}" y="{plot_bottom}" '
            f'text-anchor="end" font-size="10">{y_min:.3f}</text>',
            f'<text x="{plot_left - 5}" y="{plot_top + 8}" '
            f'text-anchor="end" font-size="10">{y_max:.3f}</text>',
        ])
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def _yes_no(value):
    return "unavailable" if value is None else "yes" if value else "no"


def _write_report(path, summary):
    lines = [
        "# Full-bag "
        + ("Green Context" if summary["execution_mode"] == "green_context"
           else "whole-GPU")
        + " alignment",
        "",
        "Alignment uses GPU timestamps: the 2 ms window immediately preceding "
        "each target H2D/D2H copy is intersected with the union of the other "
        "processes' GPU context-active intervals. D2D copies are excluded. "
        "Primary inference time begins after external CPU preprocessing and "
        "ends at the model's CUDA completion synchronization. It includes the "
        "model data_preprocessor, memcopies, synchronization waiting, "
        "inter-kernel gaps, and context-inactive periods. Frames above each "
        "model's inference-time p99 are excluded from the plot and "
        "correlations. Kernel-active union and host NVTX duration are retained "
        "as mechanism metrics.",
        "",
        "| Model | Frames ≤p99/total | Inference p99 (ms) | A range | "
        "Inference range (ms) | Spearman A–inference "
        "| Slope (ms/A) | Spearman A-waiting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, values in summary["models"].items():
        alignment = values["alignment_range"]
        latency = values["inference_e2e_ns_range"]
        correlation = values["spearman_alignment_inference_e2e"]
        slope = values["linear_slope_inference_e2e_ns_per_alignment"]
        wait = values["spearman_alignment_waiting"]
        lines.append(
            f"| {model_id} | {values['p99_frame_count']}/"
            f"{values['frame_count']} | "
            f"{values['inference_e2e_p99_ns'] / 1e6:.3f} | "
            f"{alignment[0]:.4f}–{alignment[1]:.4f} | "
            f"{latency[0] / 1e6:.3f}–{latency[1] / 1e6:.3f} | "
            f"{'unavailable' if correlation is None else f'{correlation:.4f}'} | "
            f"{'unavailable' if slope is None else f'{slope / 1e6:.4f}'} | "
            f"{'unavailable' if wait is None else f'{wait:.4f}'} |"
        )
    lines.extend(["", "## Direction", ""])
    for model_id, values in summary["models"].items():
        lines.append(
            f"- {model_id}: alignment–inference-time correlation positive: "
            f"{_yes_no(values['alignment_inference_e2e_correlation_positive'])}; "
            "inference-time slope positive: "
            f"{_yes_no(values['inference_e2e_slope_positive'])}."
        )
    lines.extend([
        "",
        "## Mechanism sensitivity",
        "",
        "| Model | Spearman A–kernel-active union | "
        "Spearman A–host NVTX duration |",
        "|---|---:|---:|",
    ])
    for model_id, values in summary["models"].items():
        active = values["spearman_alignment_gpu_kernel_active"]
        host = values["spearman_alignment_host_nvtx_latency"]
        lines.append(
            f"| {model_id} | "
            f"{'unavailable' if active is None else f'{active:.4f}'} | "
            f"{'unavailable' if host is None else f'{host:.4f}'} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_evidence_errors(run_directory, config, trace_evidence):
    errors = []
    statuses = _status_map(run_directory)
    mode = _execution_mode(config)
    for model in config["models"]:
        model_id = model["id"]
        status = statuses.get(model_id, {})
        if status.get("state") != "acknowledged":
            errors.append(f"{model_id} did not acknowledge completion")
        observed = status.get("process_observed_mps_environment", {})
        if any(observed.get(name) is not None for name in MPS_ENVIRONMENT_VARIABLES):
            errors.append(f"{model_id} observed MPS configuration")
        if mode == "green_context":
            green = status.get("green_context", {})
            requested = model["green_context"]
            if not green.get("active") or not green.get("current_context_active"):
                errors.append(f"{model_id} Green Context was not active/current")
            if green.get("actual_sm_count") != requested["sm_count"]:
                errors.append(f"{model_id} Green Context SM count differs")
        elif "green_context" in status:
            errors.append(f"{model_id} unexpectedly reported a Green Context")
    try:
        testbed = json.loads(
            (Path(run_directory) / "testbed_result.json").read_text()
        )
        if not testbed.get("replay_success") or not testbed.get("all_acknowledged"):
            errors.append("testbed completion evidence failed")
        if (
            testbed.get("playback_mode") != "full"
            or testbed.get("partial_cutoff_applied")
            or testbed.get("bags_started") != 1
            or testbed.get("bags_completed") != 1
        ):
            errors.append("the complete bag was not played exactly once")
    except (OSError, json.JSONDecodeError):
        errors.append("testbed result is missing")
    try:
        clock = json.loads(
            (Path(run_directory) / "green_context_clock_control.json").read_text()
        )
        if not clock.get("clocks_restored"):
            errors.append("GPU clock controls were not restored")
    except (OSError, json.JSONDecodeError):
        errors.append("GPU clock-control evidence is missing")
    if mode == "green_context":
        if not trace_evidence["all_model_kernels_have_nonzero_green_context_id"]:
            errors.append("model kernels lack nonzero Green Context IDs")
        if any(
            not ids
            for ids in trace_evidence["model_kernel_green_context_ids"].values()
        ):
            errors.append("a model has no observed Green Context ID")
    elif not trace_evidence["all_model_kernels_have_zero_green_context_id"]:
        errors.append("whole-GPU model kernels unexpectedly use Green Context IDs")
    return errors


def _execution_mode(config):
    green_count = sum("green_context" in model for model in config["models"])
    if green_count == len(config["models"]):
        return "green_context"
    if green_count == 0:
        return "whole_gpu"
    raise StudyError("all models must use Green Contexts or none may use them")


def analyze_full_bag(run_directory):
    """Write CSV, JSON, SVG, and Markdown results for a completed run."""
    run_directory = Path(run_directory).expanduser().resolve()
    config = yaml.safe_load(
        (run_directory / "config.yaml").read_text(encoding="utf-8")
    )
    rows, trace_evidence = extract_frame_rows(run_directory)
    model_summary = summarize_rows(rows)
    errors = _run_evidence_errors(run_directory, config, trace_evidence)
    summary = {
        "schema_version": 4,
        "execution_mode": _execution_mode(config),
        "green_context_sm_counts": {
            model["id"]: model["green_context"]["sm_count"]
            for model in config["models"] if "green_context" in model
        },
        "alignment_metric": ALIGNMENT_METRIC,
        "alignment_window_ns": ALIGNMENT_WINDOW_NS,
        "alignment_timestamp_domain": "gpu",
        "primary_inference_metric": PRIMARY_INFERENCE_METRIC,
        "frame_selection": FRAME_SELECTION,
        "kernel_active_mechanism_metric": (
            "target_kernel_gpu_context_active_union_duration"
        ),
        "host_sensitivity_metric": "host_nvtx_inference_range_duration",
        "valid": not errors,
        "errors": errors,
        "models": model_summary,
        "trace_evidence": trace_evidence,
        "artifacts": {
            "frames_csv": "green_context_frames.csv",
            "scatter_plot": "green_context_alignment_scatter.svg",
            "report": "green_context_alignment_report.md",
        },
    }
    _write_csv(run_directory / "green_context_frames.csv", rows)
    _write_scatter(run_directory / "green_context_alignment_scatter.svg", rows)
    _write_json(
        run_directory / "green_context_alignment_summary.json", summary
    )
    _write_report(
        run_directory / "green_context_alignment_report.md", summary
    )
    if errors:
        raise StudyError("; ".join(errors))
    return summary


def _validate_experiment(config):
    data = config.data
    if data["gpu"]["mps_enabled"]:
        raise StudyError("Green Context full-bag run requires MPS disabled")
    if len(data["models"]) < 2:
        raise StudyError("full-bag alignment requires at least two models")
    _execution_mode(data)
    if data["replay"].get("playback_mode", "full") != "full":
        raise StudyError("full-bag alignment requires complete playback")
    if not data["recording"]["nsys"]["gpu_context_switch"]:
        raise StudyError("GPU context-switch tracing is required")


def run_full_bag(config, dry_run=False):
    """Lock maximum clocks, run once, reset clocks, then analyze."""
    _validate_experiment(config)
    runner = ExperimentRunner(config)
    if dry_run:
        return runner.run(dry_run=True)
    runner.preflight()
    clocks = maximum_supported_clocks(config.data["gpu"]["index"])
    selected = {
        name: clocks[name]
        for name in ("graphics_clock_mhz", "memory_clock_mhz")
    }
    control = GPUClockLock(
        config.data["gpu"]["index"],
        selected["graphics_clock_mhz"],
        selected["memory_clock_mhz"],
    )
    record = {
        "resolved_clocks": selected,
        "maximum_clock_query": clocks["query"],
        "observed_before_lock": current_gpu_clocks(
            config.data["gpu"]["index"]
        ),
        "error": None,
    }
    try:
        with control:
            runner.run()
    except BaseException as exc:
        record["error"] = str(exc)
        raise
    finally:
        record["clock_control"] = control.evidence
        resets = control.evidence.get("reset_commands", [])
        record["clocks_restored"] = (
            len(resets) == 2
            and all(item.get("returncode") == 0 for item in resets)
        )
        try:
            record["observed_after_reset"] = current_gpu_clocks(
                config.data["gpu"]["index"]
            )
        except Exception as exc:
            record["observed_after_reset_error"] = str(exc)
        if config.run_directory.exists():
            _write_json(
                config.run_directory / "green_context_clock_control.json",
                record,
            )
    return analyze_full_bag(config.run_directory)


def _read_frame_rows(run_directory):
    path = Path(run_directory) / "green_context_frames.csv"
    integer_fields = {
        "ros_header_timestamp_ns",
        "host_nvtx_latency_ns",
        "inference_e2e_end_ns",
        "inference_e2e_ns",
        "completion_synchronize_ns",
        "gpu_kernel_active_ns",
        "kernel_span_ns",
        "memcpy_adjacent_waiting_ns",
    }
    float_fields = {"alignment"}
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
    return rows


def _workload_signature(config):
    return {
        "replay": {
            name: config["replay"].get(name)
            for name in ("scene_token", "rate", "playback_mode", "topics",
                         "remappings")
        },
        "models": [
            {
                name: model.get(name)
                for name in (
                    "id", "task", "modality", "mmlab_model",
                    "architecture_profile", "input_topic",
                    "input_message_type", "input_queue_depth", "qos",
                    "warmup_count", "launch_offset_seconds",
                )
            }
            for model in config["models"]
        ],
    }


def _comparison_rows(whole_rows, green_rows):
    whole = {
        (row["model_id"], row["ros_header_timestamp_ns"]): row
        for row in whole_rows
    }
    green = {
        (row["model_id"], row["ros_header_timestamp_ns"]): row
        for row in green_rows
    }
    rows = []
    for model_id, timestamp in sorted(set(whole).intersection(green)):
        baseline, partitioned = (
            whole[(model_id, timestamp)], green[(model_id, timestamp)]
        )
        inference_delta = (
            partitioned["inference_e2e_ns"]
            - baseline["inference_e2e_ns"]
        )
        rows.append({
            "model_id": model_id,
            "ros_header_timestamp_ns": timestamp,
            "whole_gpu_frame_id": baseline["frame_id"],
            "green_context_frame_id": partitioned["frame_id"],
            "whole_gpu_inference_e2e_ns": baseline["inference_e2e_ns"],
            "green_context_inference_e2e_ns": partitioned[
                "inference_e2e_ns"
            ],
            "inference_e2e_delta_ns": inference_delta,
            "inference_e2e_change_percent": (
                inference_delta / baseline["inference_e2e_ns"] * 100
            ),
            "whole_gpu_kernel_span_ns": baseline["kernel_span_ns"],
            "green_context_kernel_span_ns": partitioned["kernel_span_ns"],
            "whole_gpu_host_nvtx_latency_ns": baseline[
                "host_nvtx_latency_ns"
            ],
            "green_context_host_nvtx_latency_ns": partitioned[
                "host_nvtx_latency_ns"
            ],
            "host_nvtx_latency_delta_ns": (
                partitioned["host_nvtx_latency_ns"]
                - baseline["host_nvtx_latency_ns"]
            ),
            "whole_gpu_waiting_ns": baseline[
                "memcpy_adjacent_waiting_ns"
            ],
            "green_context_waiting_ns": partitioned[
                "memcpy_adjacent_waiting_ns"
            ],
            "waiting_delta_ns": (
                partitioned["memcpy_adjacent_waiting_ns"]
                - baseline["memcpy_adjacent_waiting_ns"]
            ),
            "whole_gpu_alignment": baseline["alignment"],
            "green_context_alignment": partitioned["alignment"],
        })
    return rows


def _difference(left, right):
    return None if left is None or right is None else right - left


def _comparison_summary(whole_rows, green_rows, matched, whole, green):
    result = {}
    model_ids = sorted({row["model_id"] for row in whole_rows + green_rows})
    for model_id in model_ids:
        whole_group = [row for row in whole_rows if row["model_id"] == model_id]
        green_group = [row for row in green_rows if row["model_id"] == model_id]
        paired = [row for row in matched if row["model_id"] == model_id]
        if not paired:
            raise StudyError(f"no source frames match for {model_id}")
        whole_stats, green_stats = (
            whole["models"][model_id], green["models"][model_id]
        )
        result[model_id] = {
            "whole_gpu_frame_count": len(whole_group),
            "green_context_frame_count": len(green_group),
            "frame_count_difference": len(green_group) - len(whole_group),
            "frame_count_change_percent": (
                (len(green_group) - len(whole_group))
                / len(whole_group) * 100
            ),
            "matched_frame_count": len(paired),
            "whole_gpu_median_p99_inference_e2e_ns": whole_stats[
                "median_inference_e2e_ns"
            ],
            "green_context_median_p99_inference_e2e_ns": green_stats[
                "median_inference_e2e_ns"
            ],
            "median_p99_inference_e2e_change_percent": (
                (green_stats["median_inference_e2e_ns"]
                 - whole_stats["median_inference_e2e_ns"])
                / whole_stats["median_inference_e2e_ns"] * 100
            ),
            "whole_gpu_median_matched_inference_e2e_ns": statistics.median(
                row["whole_gpu_inference_e2e_ns"] for row in paired
            ),
            "green_context_median_matched_inference_e2e_ns": statistics.median(
                row["green_context_inference_e2e_ns"] for row in paired
            ),
            "median_paired_inference_e2e_delta_ns": statistics.median(
                row["inference_e2e_delta_ns"] for row in paired
            ),
            "median_paired_inference_e2e_change_percent": statistics.median(
                row["inference_e2e_change_percent"] for row in paired
            ),
            "median_paired_host_nvtx_latency_delta_ns": statistics.median(
                row["host_nvtx_latency_delta_ns"] for row in paired
            ),
            "median_paired_waiting_delta_ns": statistics.median(
                row["waiting_delta_ns"] for row in paired
            ),
            "whole_gpu_spearman_alignment_inference_e2e": whole_stats[
                "spearman_alignment_inference_e2e"
            ],
            "green_context_spearman_alignment_inference_e2e": green_stats[
                "spearman_alignment_inference_e2e"
            ],
            "spearman_alignment_inference_e2e_difference": _difference(
                whole_stats["spearman_alignment_inference_e2e"],
                green_stats["spearman_alignment_inference_e2e"],
            ),
            "whole_gpu_linear_slope_inference_e2e_ns_per_alignment": whole_stats[
                "linear_slope_inference_e2e_ns_per_alignment"
            ],
            "green_context_linear_slope_inference_e2e_ns_per_alignment": green_stats[
                "linear_slope_inference_e2e_ns_per_alignment"
            ],
            "linear_slope_inference_e2e_ns_per_alignment_difference": _difference(
                whole_stats["linear_slope_inference_e2e_ns_per_alignment"],
                green_stats["linear_slope_inference_e2e_ns_per_alignment"],
            ),
            "whole_gpu_spearman_alignment_waiting": whole_stats[
                "spearman_alignment_waiting"
            ],
            "green_context_spearman_alignment_waiting": green_stats[
                "spearman_alignment_waiting"
            ],
        }
    return result


def _write_comparison_report(path, summary):
    sm_counts = sorted(set(summary["green_context_sm_counts"].values()))
    allocation = "/".join(map(str, sm_counts)) + " SM"
    lines = [
        f"# Whole GPU vs {allocation} Green Contexts",
        "",
        "Matched rows use the same model and ROS source timestamp in both "
        "one-pass full-bag runs.",
        "",
        "Primary inference time starts after external CPU preprocessing and "
        "ends at CUDA completion. It includes memcopies, synchronization "
        "waiting, inter-kernel gaps, context-inactive periods, and the model "
        "data_preprocessor. Per-run statistics exclude frames above inference "
        "time p99.",
        "",
        "| Model | Frames whole→green | Matched | Median inference time "
        "whole→green ≤p99 (ms) | P99-set median Δ | Matched median inference "
        "whole→green (ms) | Median paired Δ (ms) | Median paired Δ (%) | "
        "Host NVTX Δ (ms) | Waiting Δ (ms) | Spearman A–inference "
        "whole→green |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, values in summary["models"].items():
        lines.append(
            f"| {model_id} | {values['whole_gpu_frame_count']}→"
            f"{values['green_context_frame_count']} | "
            f"{values['matched_frame_count']} | "
            f"{values['whole_gpu_median_p99_inference_e2e_ns'] / 1e6:.3f}→"
            f"{values['green_context_median_p99_inference_e2e_ns'] / 1e6:.3f} | "
            f"{values['median_p99_inference_e2e_change_percent']:+.2f}% | "
            f"{values['whole_gpu_median_matched_inference_e2e_ns'] / 1e6:.3f}→"
            f"{values['green_context_median_matched_inference_e2e_ns'] / 1e6:.3f} | "
            f"{values['median_paired_inference_e2e_delta_ns'] / 1e6:+.3f} | "
            f"{values['median_paired_inference_e2e_change_percent']:+.2f}% | "
            f"{values['median_paired_host_nvtx_latency_delta_ns'] / 1e6:+.3f} | "
            f"{values['median_paired_waiting_delta_ns'] / 1e6:+.3f} | "
            f"{values['whole_gpu_spearman_alignment_inference_e2e']:.4f}→"
            f"{values['green_context_spearman_alignment_inference_e2e']:.4f} |"
        )
    lines.extend([
        "",
        "| Model | A–inference Spearman whole→green | Slope whole→green "
        "(ms/A) | A-waiting Spearman whole→green |",
        "|---|---:|---:|---:|",
    ])
    for model_id, values in summary["models"].items():
        whole_slope = values[
            "whole_gpu_linear_slope_inference_e2e_ns_per_alignment"
        ]
        green_slope = values[
            "green_context_linear_slope_inference_e2e_ns_per_alignment"
        ]
        lines.append(
            f"| {model_id} | "
            f"{values['whole_gpu_spearman_alignment_inference_e2e']:.4f}→"
            f"{values['green_context_spearman_alignment_inference_e2e']:.4f} | "
            f"{whole_slope / 1e6:.3f}→{green_slope / 1e6:.3f} | "
            f"{values['whole_gpu_spearman_alignment_waiting']:.4f}→"
            f"{values['green_context_spearman_alignment_waiting']:.4f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_full_bag_runs(whole_gpu_run, green_context_run, output_directory):
    """Compare identical source frames from whole-GPU and Green runs."""
    whole_gpu_run = Path(whole_gpu_run).expanduser().resolve()
    green_context_run = Path(green_context_run).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    whole_config = yaml.safe_load(
        (whole_gpu_run / "config.yaml").read_text(encoding="utf-8")
    )
    green_config = yaml.safe_load(
        (green_context_run / "config.yaml").read_text(encoding="utf-8")
    )
    if _execution_mode(whole_config) != "whole_gpu":
        raise StudyError("first comparison run must use the whole GPU")
    if _execution_mode(green_config) != "green_context":
        raise StudyError("second comparison run must use Green Contexts")
    if _workload_signature(whole_config) != _workload_signature(green_config):
        raise StudyError("comparison run workloads differ")
    whole_summary = json.loads(
        (whole_gpu_run / "green_context_alignment_summary.json").read_text()
    )
    green_summary = json.loads(
        (green_context_run / "green_context_alignment_summary.json").read_text()
    )
    if not whole_summary.get("valid") or not green_summary.get("valid"):
        raise StudyError("both comparison runs must be valid")
    whole_clock = json.loads(
        (whole_gpu_run / "green_context_clock_control.json").read_text()
    )
    green_clock = json.loads(
        (green_context_run / "green_context_clock_control.json").read_text()
    )
    if whole_clock["resolved_clocks"] != green_clock["resolved_clocks"]:
        raise StudyError("comparison run clock pairs differ")
    whole_rows, green_rows = (
        _read_frame_rows(whole_gpu_run), _read_frame_rows(green_context_run)
    )
    matched = _comparison_rows(whole_rows, green_rows)
    summary = {
        "schema_version": 3,
        "primary_inference_metric": PRIMARY_INFERENCE_METRIC,
        "frame_selection": FRAME_SELECTION,
        "whole_gpu_run": str(whole_gpu_run),
        "green_context_run": str(green_context_run),
        "resolved_clocks": whole_clock["resolved_clocks"],
        "green_context_sm_counts": green_summary["green_context_sm_counts"],
        "models": _comparison_summary(
            whole_rows, green_rows, matched, whole_summary, green_summary
        ),
        "artifacts": {
            "matched_frames_csv": "matched_frame_comparison.csv",
            "report": "comparison_report.md",
        },
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_csv(output_directory / "matched_frame_comparison.csv", matched)
    _write_json(output_directory / "comparison_summary.json", summary)
    _write_comparison_report(output_directory / "comparison_report.md", summary)
    return summary


def _correlation_comparison_report(path, summary):
    lines = [
        f"# {len(summary['runs'])}-configuration alignment correlation",
        "",
        "A uses target GPU H2D/D2H copy-start timestamps and the union of the "
        "other processes' GPU context-switch intervals. Inference time starts "
        "after external CPU preprocessing and ends at CUDA completion, "
        "including the model data_preprocessor, memcopies, synchronization "
        "waiting, inter-kernel gaps, and context-inactive periods. Each "
        "model/configuration includes only frames at or below its inference "
        "time p99. Kernel-active union is retained as a mechanism metric.",
        "",
        "| Model | Configuration | SM/model | Frames ≤p99/total | Inference "
        "p99 (ms) | A range | Inference range (ms) | Spearman A–inference | "
        "Δρ from whole | Slope (ms/A) | Δ slope from whole | Kernel-active ρ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        sm_count = row["sm_count"] if row["sm_count"] is not None else "all"
        lines.append(
            f"| {row['model_id']} | {row['configuration']} | {sm_count} | "
            f"{row['p99_frame_count']}/{row['processed_frame_count']} | "
            f"{row['inference_e2e_p99_ms']:.3f} | "
            f"{row['alignment_min']:.4f}–{row['alignment_max']:.4f} | "
            f"{row['inference_e2e_min_ms']:.3f}–"
            f"{row['inference_e2e_max_ms']:.3f} | "
            f"{row['spearman_alignment_inference_e2e']:.4f} | "
            f"{row['spearman_change_from_whole']:+.4f} | "
            f"{row['slope_inference_e2e_ms_per_alignment']:.3f} | "
            f"{row['slope_change_from_whole_ms_per_alignment']:+.3f} | "
            f"{row['spearman_alignment_gpu_kernel_active']:.4f} |"
        )
    lines.extend(["", "## Direction", ""])
    for row in summary["rows"]:
        lines.append(
            f"- {row['model_id']} / {row['configuration']}: correlation "
            f"positive: {_yes_no(row['correlation_positive'])}; slope "
            f"positive: {_yes_no(row['slope_positive'])}."
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_correlation_comparison_scatter(path, run_info):
    configurations = [item["configuration"] for item in run_info]
    model_ids = sorted(run_info[0]["summary"]["models"])
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    panel_width, panel_height = 360, 280
    left, top = 90, 70
    width = left + len(run_info) * panel_width + 30
    height = top + len(model_ids) * panel_height + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20">'
        'Alignment A vs post-preprocess inference time (≤p99)</text>',
    ]
    for row_index, model_id in enumerate(model_ids):
        model_rows = [
            row for item in run_info
            for row in _p99_rows([
                candidate for candidate in item["rows"]
                if candidate["model_id"] == model_id
            ])[1]
        ]
        x_max = max(row["alignment"] for row in model_rows)
        x_span = x_max or 1
        for column, item in enumerate(run_info):
            group = _p99_rows([
                row for row in item["rows"] if row["model_id"] == model_id
            ])[1]
            y_min = min(row["inference_e2e_ns"] / 1e6 for row in group)
            y_max = max(row["inference_e2e_ns"] / 1e6 for row in group)
            y_span = y_max - y_min or 1
            x0 = left + column * panel_width
            y0 = top + row_index * panel_height
            plot_left, plot_right = x0 + 45, x0 + panel_width - 20
            plot_top, plot_bottom = y0 + 35, y0 + panel_height - 45
            rho = item["summary"]["models"][model_id][
                "spearman_alignment_inference_e2e"
            ]
            p99 = item["summary"]["models"][model_id][
                "inference_e2e_p99_ns"
            ]
            title = {
                "whole_gpu": "Whole GPU",
                "green_28sm": "Green Context 28 SM",
                "green_14sm": "Green Context 14 SM",
                "green_18sm": "Green Context 18 SM",
            }.get(configurations[column], configurations[column])
            counts = item["summary"]["models"][model_id]
            parts.extend([
                f'<text x="{(plot_left + plot_right) / 2}" y="{y0 + 18}" '
                f'text-anchor="middle" font-size="14">{html.escape(title)}; '
                f'ρ={rho:.3f}; p99={p99 / 1e6:.1f} ms; '
                f'n={counts["p99_frame_count"]}/{counts["frame_count"]}</text>',
                f'<line x1="{plot_left}" y1="{plot_bottom}" '
                f'x2="{plot_right}" y2="{plot_bottom}" stroke="black"/>',
                f'<line x1="{plot_left}" y1="{plot_top}" '
                f'x2="{plot_left}" y2="{plot_bottom}" stroke="black"/>',
                f'<text x="{plot_left}" y="{plot_bottom + 16}" '
                'font-size="10">0</text>',
                f'<text x="{plot_right}" y="{plot_bottom + 16}" '
                f'text-anchor="end" font-size="10">{x_max:.3f}</text>',
                f'<text x="{plot_left - 5}" y="{plot_bottom}" '
                f'text-anchor="end" font-size="10">{y_min:.1f}</text>',
                f'<text x="{plot_left - 5}" y="{plot_top + 8}" '
                f'text-anchor="end" font-size="10">{y_max:.1f}</text>',
                f'<text x="{(plot_left + plot_right) / 2}" '
                f'y="{plot_bottom + 34}" text-anchor="middle" font-size="11">'
                'Alignment A</text>',
            ])
            for row in group:
                inference_e2e_ms = row["inference_e2e_ns"] / 1e6
                x = plot_left + row["alignment"] / x_span * (
                    plot_right - plot_left
                )
                y = plot_bottom - (inference_e2e_ms - y_min) / y_span * (
                    plot_bottom - plot_top
                )
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" '
                    f'fill="{colors[column]}" fill-opacity="0.55"/>'
                )
        row_middle = top + row_index * panel_height + panel_height / 2
        model_alias = next(
            (
                model["mmlab_model"]
                for model in run_info[0].get("config", {}).get("models", [])
                if model["id"] == model_id
            ),
            model_id,
        ).lower()
        model_title = next(
            (
                title for token, title in (
                    ("faster-rcnn", "Faster R-CNN"),
                    ("pointpillars", "PointPillars"),
                    ("centerpoint", "CenterPoint"),
                    ("deeplabv3plus", "DeepLabV3+"),
                )
                if token in model_alias
            ),
            model_id,
        )
        parts.append(
            f'<text x="18" y="{row_middle}" text-anchor="middle" '
            f'font-size="15" transform="rotate(-90 18 {row_middle})">'
            f'{html.escape(model_title)} '
            '— post-preprocess inference (ms)</text>'
        )
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def compare_alignment_correlations(run_directories, output_directory):
    """Compare GPU-side A/inference correlations across two or three runs."""
    if len(run_directories) not in (2, 3):
        raise StudyError("correlation comparison requires two or three runs")
    run_info = []
    for run_directory in run_directories:
        path = Path(run_directory).expanduser().resolve()
        config = yaml.safe_load(
            (path / "config.yaml").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (path / "green_context_alignment_summary.json").read_text()
        )
        clock = json.loads(
            (path / "green_context_clock_control.json").read_text()
        )
        if not summary.get("valid"):
            raise StudyError(f"comparison run is invalid: {path}")
        mode = _execution_mode(config)
        counts = sorted({
            model["green_context"]["sm_count"]
            for model in config["models"] if "green_context" in model
        })
        if mode == "green_context" and len(counts) != 1:
            raise StudyError(f"models use different SM counts: {path}")
        sm_count = counts[0] if counts else None
        run_info.append({
            "path": path,
            "config": config,
            "summary": summary,
            "clocks": clock["resolved_clocks"],
            "mode": mode,
            "sm_count": sm_count,
            "configuration": (
                "whole_gpu" if sm_count is None else f"green_{sm_count}sm"
            ),
            "rows": _read_frame_rows(path),
        })
    if len({item["configuration"] for item in run_info}) != len(run_info):
        raise StudyError("comparison configurations must be distinct")
    signature = _workload_signature(run_info[0]["config"])
    if any(_workload_signature(item["config"]) != signature
           for item in run_info[1:]):
        raise StudyError("comparison run workloads differ")
    if any(item["clocks"] != run_info[0]["clocks"] for item in run_info[1:]):
        raise StudyError("comparison run clock pairs differ")
    whole = next(
        (item for item in run_info if item["mode"] == "whole_gpu"), None
    )
    if whole is None:
        raise StudyError("correlation comparison requires one whole-GPU run")
    run_info.sort(key=lambda item: (
        item["mode"] != "whole_gpu", -(item["sm_count"] or 0)
    ))
    rows = []
    for model_id in sorted(whole["summary"]["models"]):
        baseline = whole["summary"]["models"][model_id]
        baseline_rho = baseline["spearman_alignment_inference_e2e"]
        baseline_slope = baseline[
            "linear_slope_inference_e2e_ns_per_alignment"
        ]
        for item in run_info:
            values = item["summary"]["models"][model_id]
            inference_e2e = values["inference_e2e_ns_range"]
            rho = values["spearman_alignment_inference_e2e"]
            slope = values[
                "linear_slope_inference_e2e_ns_per_alignment"
            ]
            alignment = values["alignment_range"]
            rows.append({
                "model_id": model_id,
                "configuration": item["configuration"],
                "sm_count": item["sm_count"],
                "processed_frame_count": values["frame_count"],
                "p99_frame_count": values["p99_frame_count"],
                "inference_e2e_p99_ms": (
                    values["inference_e2e_p99_ns"] / 1e6
                ),
                "alignment_min": alignment[0],
                "alignment_max": alignment[1],
                "inference_e2e_min_ms": inference_e2e[0] / 1e6,
                "inference_e2e_max_ms": inference_e2e[1] / 1e6,
                "spearman_alignment_inference_e2e": rho,
                "spearman_change_from_whole": rho - baseline_rho,
                "slope_inference_e2e_ms_per_alignment": slope / 1e6,
                "slope_change_from_whole_ms_per_alignment": (
                    slope - baseline_slope
                ) / 1e6,
                "spearman_alignment_host_nvtx_latency": values[
                    "spearman_alignment_host_nvtx_latency"
                ],
                "spearman_alignment_gpu_kernel_active": values[
                    "spearman_alignment_gpu_kernel_active"
                ],
                "spearman_alignment_waiting": values[
                    "spearman_alignment_waiting"
                ],
                "correlation_positive": rho > 0,
                "slope_positive": slope > 0,
            })
    output_directory = Path(output_directory).expanduser().resolve()
    summary = {
        "schema_version": 3,
        "alignment_timestamp_domain": "gpu",
        "primary_inference_metric": PRIMARY_INFERENCE_METRIC,
        "frame_selection": FRAME_SELECTION,
        "resolved_clocks": run_info[0]["clocks"],
        "runs": {
            item["configuration"]: str(item["path"]) for item in run_info
        },
        "rows": rows,
        "artifacts": {
            "csv": "correlation_comparison.csv",
            "report": "correlation_comparison.md",
            "scatter_plot": "correlation_scatter.svg",
        },
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_csv(output_directory / "correlation_comparison.csv", rows)
    _write_json(output_directory / "correlation_comparison.json", summary)
    _write_correlation_comparison_scatter(
        output_directory / "correlation_scatter.svg", run_info
    )
    _correlation_comparison_report(
        output_directory / "correlation_comparison.md", summary
    )
    return summary


def _parser():
    parser = argparse.ArgumentParser(
        description="Run, analyze, or compare full-bag GPU experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("config")
    run.add_argument("--dry-run", action="store_true")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("run_directory")
    compare = commands.add_parser("compare")
    compare.add_argument("whole_gpu_run")
    compare.add_argument("green_context_run")
    compare.add_argument("output_directory")
    correlations = commands.add_parser("compare-correlations")
    correlations.add_argument("run_directories", nargs="+")
    correlations.add_argument("--output", required=True)
    return parser


def main(argv=None):
    """Run, analyze, or compare full-bag GPU experiments."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_full_bag(load_run_config(args.config), args.dry_run)
        elif args.command == "analyze":
            result = analyze_full_bag(args.run_directory)
        elif args.command == "compare":
            result = compare_full_bag_runs(
                args.whole_gpu_run,
                args.green_context_run,
                args.output_directory,
            )
        else:
            result = compare_alignment_correlations(
                args.run_directories, args.output
            )
        if result is not None:
            print(json.dumps(result, sort_keys=True))
    except (ConfigError, StudyError, PreflightError, RunFailure, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
