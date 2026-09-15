"""GPU-only discovery analysis for the fixed Phase 4 model pair."""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np

from .input_data.corrected_input_analysis import _write_csv
from .mps_leftover import MpsLeftoverAnalyzer
from .mps_two_model_cta_analysis import (
    CROSS_CLIENT_SMID_COMPARABLE,
    _co_residency_feasibility,
)
from ._common import memcopies
from .target_selection import TraceTargetSelector


FEATURES = (
    "gpu_span_ms",
    "kernel_busy_ms",
    "kernel_sum_ms",
    "ready_delay_sum_ms",
    "ready_delay_max_ms",
    "cross_client_overlap_ms",
    "foreign_busy_ms",
    "gpu_execution_gap_ms",
    "h2d_ms",
    "d2h_ms",
)
KERNEL_QUANTILES = (0, 1, 5, 25, 50, 75, 95, 99)
KERNEL_SET_SIZES = (1, 2, 3, 5, 10, 20, 50)
MANDATORY_TARGETS = {
    ("faster_rcnn", "K0255"),
    ("deeplabv3plus", "K0265"),
}
DEVICE_LIMITS = {
    "warp_size": 32,
    "max_threads_per_sm": 1536,
    "registers_per_sm": 65536,
    "shared_memory_per_sm": 102400,
    "max_blocks_per_sm": 24,
}


def _merged(intervals, start=None, end=None):
    """Return clipped, merged half-open intervals."""
    result = []
    for left, right in sorted(intervals):
        left = max(left, start) if start is not None else left
        right = min(right, end) if end is not None else right
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], right))
        else:
            result.append((left, right))
    return result


def _duration(intervals):
    return sum(end - start for start, end in intervals)


def _intersection_duration(left, right):
    left, right = _merged(left), _merged(right)
    total = first = second = 0
    while first < len(left) and second < len(right):
        total += max(
            0,
            min(left[first][1], right[second][1])
            - max(left[first][0], right[second][0]),
        )
        if left[first][1] <= right[second][1]:
            first += 1
        else:
            second += 1
    return total


def _rank(values):
    """Return deterministic average ranks for a Spearman correlation."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    offset = 0
    while offset < len(values):
        limit = offset + 1
        while (
            limit < len(values)
            and values[order[limit]] == values[order[offset]]
        ):
            limit += 1
        ranks[order[offset:limit]] = (offset + limit - 1) / 2
        offset = limit
    return ranks


def _correlation(left, right):
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _overlapping(records, starts, start, end):
    return [
        item for item in records[:bisect_left(starts, end)]
        if item.end_ns > start
    ]


def _interval_index(records):
    """Index intervals for complete backwards overlap queries."""
    ordered = sorted(records, key=lambda item: (
        item.start_ns, item.end_ns, item.kernel_id,
    ))
    prefix_max = []
    maximum = 0
    for item in ordered:
        maximum = max(maximum, item.end_ns)
        prefix_max.append(maximum)
    return ordered, [item.start_ns for item in ordered], prefix_max


def _indexed_overlaps(index, start, end):
    records, starts, prefix_max = index
    position = bisect_left(starts, end) - 1
    result = []
    while position >= 0 and prefix_max[position] > start:
        item = records[position]
        if item.end_ns > start:
            result.append(item)
        position -= 1
    return sorted(result, key=lambda item: (
        item.start_ns, item.end_ns, item.pid, item.kernel_id,
    ))


def _kernel_signature(kernel):
    return MpsLeftoverAnalyzer._inventory_signature(kernel)


def _kernel_record(kernel, model, occurrence):
    return {
        "model_id": model,
        "pid": kernel.pid,
        "context_id": kernel.context_id,
        "stream_id": kernel.stream_id,
        "kernel_id": kernel.kernel_id,
        "kernel_name": kernel.name,
        "kernel_signature": _kernel_signature(kernel),
        "kernel_name_geometry_occurrence": occurrence,
        "start_ns": kernel.start_ns,
        "end_ns": kernel.end_ns,
        "duration_ns": kernel.duration_ns,
        "grid": list(kernel.grid),
        "block": list(kernel.block),
        "registers_per_thread": kernel.registers_per_thread,
        "static_shared_memory": kernel.static_shared_memory,
        "dynamic_shared_memory": kernel.dynamic_shared_memory,
        "expected_cta_count": int(np.prod(kernel.grid)),
    }


def _duration_metrics(values):
    """Return the required descriptive kernel-duration measurements."""
    values = np.asarray(values, dtype=float)
    percentiles = np.percentile(values, KERNEL_QUANTILES)
    result = {
        f"p{quantile}_ms": float(value)
        for quantile, value in zip(KERNEL_QUANTILES, percentiles)
    }
    result["primary_range_ms"] = result["p99_ms"] - result["p0_ms"]
    result["normalized_range"] = (
        result["primary_range_ms"] / result["p50_ms"]
        if result["p50_ms"] else None
    )
    return result


def _kernel_inventory(loaded):
    """Index logical per-inference launches and rank duration variation."""
    grouped = {}
    counts = defaultdict(list)
    inference_totals = defaultdict(int)
    run_ids = defaultdict(set)
    signatures = defaultdict(set)

    for run_id, (_selector, _trace, features, _ranges,
                 inference_kernels) in loaded.items():
        foreign_indices = {
            pid: _interval_index([
                kernel for kernel in _trace.kernels if kernel.pid == pid
            ])
            for pid in {kernel.pid for kernel in _trace.kernels}
        }
        for feature in features:
            model = feature["model_id"]
            key = (
                feature["pid"], feature["range_start_ns"],
                feature["range_end_ns"],
            )
            kernels = sorted(
                inference_kernels[key],
                key=lambda item: (
                    item.launch_start_ns, item.launch_end_ns,
                    item.kernel_id,
                ),
            )
            counts[model].append(len(kernels))
            inference_totals[model] += 1
            run_ids[model].add(run_id)
            occurrences = defaultdict(int)
            for sequence_index, kernel in enumerate(kernels, 1):
                signature = _kernel_signature(kernel)
                occurrence = occurrences[signature]
                occurrences[signature] += 1
                signatures[model].add(signature)
                group_key = (model, signature, occurrence)
                group = grouped.setdefault(group_key, {
                    "model_id": model,
                    "kernel_signature": signature,
                    "occurrence_within_signature": occurrence,
                    "kernel_name": kernel.name,
                    "launch_api": kernel.launch_api,
                    "grid": "x".join(map(str, kernel.grid)),
                    "block": "x".join(map(str, kernel.block)),
                    "registers_per_thread": kernel.registers_per_thread,
                    "static_shared_memory_bytes":
                    kernel.static_shared_memory,
                    "dynamic_shared_memory_bytes":
                    kernel.dynamic_shared_memory,
                    "durations": [],
                    "inference_latencies": [],
                    "sequence_indices": [],
                    "ready_delays": [],
                    "cross_client_overlaps": [],
                    "context_ids": set(),
                    "stream_ids": set(),
                    "run_ids": set(),
                })
                group["durations"].append(kernel.duration_ns / 1e6)
                group["inference_latencies"].append(
                    feature["inference_latency_ms"]
                )
                group["sequence_indices"].append(sequence_index)
                group["ready_delays"].append(max(
                    0, kernel.start_ns -
                    _trace.ready_times[kernel.kernel_id]
                ) / 1e6)
                foreign = [
                    item
                    for pid, index in foreign_indices.items()
                    if pid != kernel.pid
                    for item in _indexed_overlaps(
                        index, kernel.start_ns, kernel.end_ns
                    )
                ]
                group["cross_client_overlaps"].append(
                    _intersection_duration(
                        [(kernel.start_ns, kernel.end_ns)],
                        [(item.start_ns, item.end_ns) for item in foreign],
                    ) / 1e6
                )
                group["context_ids"].add(kernel.context_id)
                group["stream_ids"].add(kernel.stream_id)
                group["run_ids"].add(run_id)

    rows = []
    for model in sorted(counts):
        model_groups = [
            group for (group_model, _signature, _occurrence), group
            in grouped.items() if group_model == model
        ]
        model_groups.sort(key=lambda group: (
            float(np.median(group["sequence_indices"])),
            group["kernel_signature"],
            group["occurrence_within_signature"],
        ))
        for index, group in enumerate(model_groups, 1):
            durations = group.pop("durations")
            inference_latencies = group.pop("inference_latencies")
            sequence_indices = group.pop("sequence_indices")
            ready_delays = group.pop("ready_delays")
            overlaps = group.pop("cross_client_overlaps")
            context_ids = group.pop("context_ids")
            stream_ids = group.pop("stream_ids")
            observed_runs = group.pop("run_ids")
            duration_metrics = _duration_metrics(durations)
            p10, p90 = np.percentile(durations, (10, 90))
            rows.append({
                **group,
                "kernel_index": f"K{index:04d}",
                "sample_count": len(durations),
                "run_count": len(observed_runs),
                "inference_coverage_fraction": (
                    len(durations) / inference_totals[model]
                ),
                "sequence_index_min": min(sequence_indices),
                "sequence_index_p50": float(np.median(sequence_indices)),
                "sequence_index_max": max(sequence_indices),
                "sequence_index_stable": (
                    min(sequence_indices) == max(sequence_indices)
                ),
                "context_ids": ",".join(map(str, sorted(context_ids))),
                "stream_ids": ",".join(map(str, sorted(stream_ids))),
                "context_count": len(context_ids),
                "stream_count": len(stream_ids),
                "ready_delay_p50_ms": float(np.percentile(
                    ready_delays, 50
                )),
                "ready_delay_p99_ms": float(np.percentile(
                    ready_delays, 99
                )),
                "cross_client_overlap_p50_ms": float(np.percentile(
                    overlaps, 50
                )),
                "cross_client_overlap_p99_ms": float(np.percentile(
                    overlaps, 99
                )),
                "cross_client_overlap_positive_fraction": float(np.mean(
                    np.asarray(overlaps) > 0
                )),
                "fast_observation_count": sum(
                    value <= p10 for value in durations
                ),
                "slow_observation_count": sum(
                    value >= p90 for value in durations
                ),
                "inference_latency_spearman_rho": _correlation(
                    durations, inference_latencies
                ),
                **duration_metrics,
            })

    for model in counts:
        ranked = sorted(
            (row for row in rows if row["model_id"] == model),
            key=lambda row: row["primary_range_ms"], reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["variation_rank"] = rank

    summaries = []
    for model, values in sorted(counts.items()):
        model_rows = [row for row in rows if row["model_id"] == model]
        summaries.append({
            "model_id": model,
            "run_count": len(run_ids[model]),
            "inference_count": inference_totals[model],
            "kernel_launch_count_min": min(values),
            "kernel_launch_count_p50": float(np.median(values)),
            "kernel_launch_count_p99": float(np.percentile(values, 99)),
            "kernel_launch_count_max": max(values),
            "distinct_kernel_name_count": len({
                row["kernel_name"] for row in model_rows
            }),
            "distinct_kernel_signature_count": len(signatures[model]),
            "logical_kernel_index_count": len(model_rows),
            "always_present_logical_kernel_count": sum(
                row["inference_coverage_fraction"] == 1
                for row in model_rows
            ),
        })
    return summaries, rows


def _kernel_set_correlations(loaded, kernel_rows):
    """Measure whether cumulative high-range kernels track inference time."""
    result = []
    for model in sorted({row["model_id"] for row in kernel_rows}):
        eligible = sorted(
            (
                row for row in kernel_rows
                if row["model_id"] == model
                and row["inference_coverage_fraction"] == 1
            ),
            key=lambda row: row["primary_range_ms"], reverse=True,
        )
        selections = {
            f"top_{size}_by_p99_p0": eligible[:size]
            for size in KERNEL_SET_SIZES if size <= len(eligible)
        }
        selections["all_always_present"] = eligible
        sums = {name: [] for name in selections}
        all_kernel_sums = []
        inference_latencies = []

        for _run_id, (_selector, _trace, features, _ranges,
                      inference_kernels) in loaded.items():
            for feature in features:
                if feature["model_id"] != model:
                    continue
                key = (
                    feature["pid"], feature["range_start_ns"],
                    feature["range_end_ns"],
                )
                occurrences = defaultdict(int)
                durations = {}
                for kernel in sorted(
                    inference_kernels[key],
                    key=lambda item: (
                        item.launch_start_ns, item.launch_end_ns,
                        item.kernel_id,
                    ),
                ):
                    signature = _kernel_signature(kernel)
                    occurrence = occurrences[signature]
                    occurrences[signature] += 1
                    durations[(signature, occurrence)] = (
                        kernel.duration_ns / 1e6
                    )
                for name, selected in selections.items():
                    sums[name].append(sum(
                        durations[(
                            row["kernel_signature"],
                            row["occurrence_within_signature"],
                        )]
                        for row in selected
                    ))
                all_kernel_sums.append(feature["kernel_sum_ms"])
                inference_latencies.append(feature["inference_latency_ms"])

        inference_metrics = _duration_metrics(inference_latencies)
        for name, selected in selections.items():
            metrics = _duration_metrics(sums[name])
            result.append({
                "model_id": model,
                "selection": name,
                "kernel_count": len(selected),
                "sample_count": len(inference_latencies),
                "kernel_indices": ",".join(
                    row["kernel_index"] for row in selected
                ),
                "sum_duration_spearman_rho": _correlation(
                    sums[name], inference_latencies
                ),
                "inference_primary_range_ms":
                inference_metrics["primary_range_ms"],
                "sum_to_inference_range_fraction": (
                    metrics["primary_range_ms"]
                    / inference_metrics["primary_range_ms"]
                ),
                **metrics,
            })
        metrics = _duration_metrics(all_kernel_sums)
        result.append({
            "model_id": model,
            "selection": "all_launched",
            "kernel_count": "variable",
            "sample_count": len(inference_latencies),
            "kernel_indices": "all",
            "sum_duration_spearman_rho": _correlation(
                all_kernel_sums, inference_latencies
            ),
            "inference_primary_range_ms":
            inference_metrics["primary_range_ms"],
            "sum_to_inference_range_fraction": (
                metrics["primary_range_ms"]
                / inference_metrics["primary_range_ms"]
            ),
            **metrics,
        })
    return result


def _run_features(run_directory):
    selector = TraceTargetSelector(
        run_directory / "profile.sqlite", run_directory
    )
    trace = selector.load_trace()
    inference_index = MpsLeftoverAnalyzer._inference_index(trace.ranges)
    input_records = selector._model_input_records()
    by_pid = defaultdict(list)
    for kernel in trace.kernels:
        by_pid[kernel.pid].append(kernel)
    starts = {
        pid: [kernel.start_ns for kernel in records]
        for pid, records in by_pid.items()
    }
    with sqlite3.connect(str(run_directory / "profile.sqlite")) as database:
        copies = memcopies(database)
    copies_by_pid = defaultdict(list)
    for copy in copies:
        copies_by_pid[copy["pid"]].append(copy)

    inference_kernels = defaultdict(list)
    inferences = {}
    for kernel in trace.kernels:
        inference = MpsLeftoverAnalyzer._indexed_inference(
            kernel, inference_index
        )
        if inference is None:
            continue
        key = (kernel.pid, inference["start_ns"], inference["end_ns"])
        inferences[key] = inference
        inference_kernels[key].append(kernel)

    rows = []
    for key, own in inference_kernels.items():
        pid, range_start, range_end = key
        inference = inferences[key]
        model = str(
            inference["tag"].get("model") or selector.models.get(pid, pid)
        )
        input_id = str(inference["tag"].get("input", ""))
        if input_id.startswith("warmup-"):
            continue
        source = input_records.get((model, input_id), {})
        gpu_end = max(range_end, max(kernel.end_ns for kernel in own))
        gpu_start = min(kernel.start_ns for kernel in own)
        own_intervals = [(kernel.start_ns, kernel.end_ns) for kernel in own]
        own_busy = _merged(own_intervals, gpu_start, gpu_end)
        foreign = []
        for foreign_pid, records in by_pid.items():
            if foreign_pid != pid:
                foreign.extend(_overlapping(
                    records, starts[foreign_pid], gpu_start, gpu_end
                ))
        foreign_intervals = [
            (kernel.start_ns, kernel.end_ns) for kernel in foreign
        ]
        relevant_copies = [
            copy for copy in copies_by_pid[pid]
            if copy["start"] < gpu_end and copy["end"] > range_start
        ]
        ready_delays = [
            max(0, kernel.start_ns - trace.ready_times[kernel.kernel_id])
            for kernel in own
        ]
        run_suffix = run_directory.name[len("mps2cause-"):]
        condition, separator, replicate = run_suffix.rpartition("-r")
        if not separator or not replicate.isdigit():
            condition, replicate = run_suffix, 0
        row = {
            "run_id": run_directory.name,
            "condition": condition,
            "replicate": int(replicate),
            "model_id": model,
            "pid": pid,
            "context_ids": ",".join(
                str(value) for value in sorted({k.context_id for k in own})
            ),
            "stream_ids": ",".join(
                str(value) for value in sorted({k.stream_id for k in own})
            ),
            "input_id": input_id,
            "scene_index": source.get("scene_index"),
            "scene_name": source.get("scene_name"),
            "scene_token": source.get("scene_token"),
            "range_start_ns": range_start,
            "range_end_ns": range_end,
            "gpu_end_ns": gpu_end,
            "inference_latency_ms": (gpu_end - range_start) / 1e6,
            "gpu_span_ms": (gpu_end - gpu_start) / 1e6,
            "kernel_busy_ms": _duration(own_busy) / 1e6,
            "kernel_sum_ms": sum(k.duration_ns for k in own) / 1e6,
            "ready_delay_sum_ms": sum(ready_delays) / 1e6,
            "ready_delay_max_ms": max(ready_delays, default=0) / 1e6,
            "cross_client_overlap_ms": _intersection_duration(
                own_intervals, foreign_intervals
            ) / 1e6,
            "foreign_busy_ms": _duration(_merged(
                foreign_intervals, gpu_start, gpu_end
            )) / 1e6,
            "gpu_execution_gap_ms": (
                gpu_end - gpu_start - _duration(own_busy)
            ) / 1e6,
            "kernel_count": len(own),
            "foreign_kernel_count": len(foreign),
            "h2d_ms": sum(
                copy["end"] - copy["start"] for copy in relevant_copies
                if copy["copy_class"] == "H2D"
            ) / 1e6,
            "d2h_ms": sum(
                copy["end"] - copy["start"] for copy in relevant_copies
                if copy["copy_class"] == "D2H"
            ) / 1e6,
            "h2d_bytes": sum(
                copy["bytes"] for copy in relevant_copies
                if copy["copy_class"] == "H2D"
            ),
            "d2h_bytes": sum(
                copy["bytes"] for copy in relevant_copies
                if copy["copy_class"] == "D2H"
            ),
        }
        rows.append(row)
    return selector, trace, rows, inferences, inference_kernels


def _episode(selector, trace, rows, inferences, inference_kernels, row):
    key = (row["pid"], row["range_start_ns"], row["range_end_ns"])
    own = sorted(inference_kernels[key], key=lambda kernel: kernel.start_ns)
    other = [kernel for kernel in trace.kernels if kernel.pid != row["pid"]]
    occurrence = defaultdict(int)
    occurrences = {}
    for kernel in trace.kernels:
        model = selector.models.get(kernel.pid, str(kernel.pid))
        signature = (model, kernel.name, kernel.grid, kernel.block)
        occurrences[kernel.kernel_id] = occurrence[signature]
        occurrence[signature] += 1
    return {
        "run_id": row["run_id"],
        "model_id": row["model_id"],
        "input_id": row["input_id"],
        "scene_index": row["scene_index"],
        "scene_name": row["scene_name"],
        "scene_token": row["scene_token"],
        "inference_latency_ms": row["inference_latency_ms"],
        "range_start_ns": row["range_start_ns"],
        "range_end_ns": row["range_end_ns"],
        "gpu_end_ns": row["gpu_end_ns"],
        "kernels": [
            _kernel_record(
                kernel, row["model_id"], occurrences[kernel.kernel_id]
            )
            for kernel in own
        ],
        "foreign_kernels": [
            _kernel_record(
                kernel, selector.models.get(kernel.pid, str(kernel.pid)),
                occurrences[kernel.kernel_id],
            )
            for kernel in other
            if kernel.start_ns < row["gpu_end_ns"]
            and kernel.end_ns > own[0].start_ns
        ],
    }


def _target_decisions(kernel_rows, run_count):
    """Lock targets from discovery metrics before any CTA observation."""
    eligible = []
    for row in kernel_rows:
        reasons = []
        if row["run_count"] != run_count:
            reasons.append("not observed in every discovery run")
        if row["inference_coverage_fraction"] != 1:
            reasons.append("not present in every inference")
        if row["sample_count"] < 30:
            reasons.append("fewer than 30 observations")
        if not row["sequence_index_stable"]:
            reasons.append("frame-local sequence index varies")
        if min(
            row["fast_observation_count"], row["slow_observation_count"]
        ) < 3:
            reasons.append("fewer than three fast or slow observations")
        row["selection_eligible"] = not reasons
        row["eligibility_reason"] = (
            "eligible" if not reasons else "; ".join(reasons)
        )
        if not reasons:
            eligible.append(row)
    primary = {
        model: max(
            (row for row in eligible if row["model_id"] == model),
            key=lambda row: (
                row["primary_range_ms"], row["sample_count"],
                row["kernel_index"],
            ),
        )["kernel_index"]
        for model in sorted({row["model_id"] for row in kernel_rows})
    }
    selected = {
        (model, index) for model, index in primary.items()
    } | MANDATORY_TARGETS
    decisions = []
    selected_rows = []
    for row in kernel_rows:
        key = (row["model_id"], row["kernel_index"])
        reasons = []
        if primary.get(row["model_id"]) == row["kernel_index"]:
            reasons.append("primary highest eligible p99-p0")
        if key in MANDATORY_TARGETS:
            reasons.append("required comparison target")
        is_selected = key in selected
        if is_selected:
            selected_rows.append(row)
        decisions.append({
            **{key: value for key, value in row.items()
               if key not in ("kernel_name",)},
            "kernel_name": row["kernel_name"],
            "selected": is_selected,
            "decision": (
                "; ".join(reasons) if reasons else
                row["eligibility_reason"] if not row["selection_eligible"]
                else "eligible but lower p99-p0 rank"
            ),
            "selection_metric": "kernel duration p99-p0",
            "cta_outcomes_examined": False,
        })
    return decisions, sorted(selected_rows, key=lambda row: (
        row["model_id"], row["kernel_index"],
    ))


def _feature_identity_map(feature, kernels, model, index_by_key):
    occurrences = defaultdict(int)
    result = {}
    for sequence, kernel in enumerate(sorted(
        kernels,
        key=lambda item: (
            item.launch_start_ns, item.launch_end_ns, item.kernel_id,
        ),
    ), 1):
        signature = _kernel_signature(kernel)
        occurrence = occurrences[signature]
        occurrences[signature] += 1
        result[kernel.kernel_id] = {
            "model_id": model,
            "kernel_signature": signature,
            "occurrence_within_signature": occurrence,
            "kernel_index": index_by_key.get(
                (model, signature, occurrence), "unindexed"
            ),
            "launch_sequence_index": sequence,
            "input_id": feature["input_id"],
            "scene_name": feature["scene_name"],
            "scene_token": feature["scene_token"],
        }
    return result


def _matched_target_artifacts(loaded, kernel_rows, selected_rows):
    """Build same-source fast/slow episodes and passive capture windows."""
    index_by_key = {
        (
            row["model_id"], row["kernel_signature"],
            row["occurrence_within_signature"],
        ): row["kernel_index"]
        for row in kernel_rows
    }
    observations = defaultdict(list)
    identity_cache = {}
    feature_ranges = {}
    input_records = {}
    for run_id, (selector, trace, features, _ranges,
                 inference_kernels) in loaded.items():
        input_records[run_id] = selector._model_input_records()
        by_pid = defaultdict(list)
        for feature in features:
            by_pid[feature["pid"]].append(feature)
        feature_ranges[run_id] = {
            pid: (
                [item["range_start_ns"] for item in sorted(
                    values, key=lambda item: item["range_start_ns"]
                )],
                sorted(values, key=lambda item: item["range_start_ns"]),
            )
            for pid, values in by_pid.items()
        }
        for feature in features:
            key = (
                feature["pid"], feature["range_start_ns"],
                feature["range_end_ns"],
            )
            cache_key = (run_id, key)
            identities = _feature_identity_map(
                feature, inference_kernels[key], feature["model_id"],
                index_by_key,
            )
            identity_cache[cache_key] = identities
            selected = {
                row["kernel_index"]: row
                for row in selected_rows
                if row["model_id"] == feature["model_id"]
            }
            for kernel in inference_kernels[key]:
                identity = identities[kernel.kernel_id]
                row = selected.get(identity["kernel_index"])
                if row is None:
                    continue
                source = input_records[run_id].get((
                    feature["model_id"], str(feature["input_id"]),
                ), {})
                source_key = (
                    source.get("scene_token", feature["scene_token"]),
                    source.get("ros_header_timestamp_ns"),
                )
                observations[(
                    feature["model_id"], identity["kernel_index"],
                    source_key,
                )].append({
                    "run_id": run_id,
                    "selector": selector,
                    "trace": trace,
                    "feature": feature,
                    "inference_key": key,
                    "kernel": kernel,
                    "identity": identity,
                    "source": source,
                })

    matched_catalog = []
    timeline = []
    selection_records = []
    capture_targets = []
    client_windows = defaultdict(list)

    def enclosing_feature(run_id, kernel):
        starts, values = feature_ranges[run_id].get(kernel.pid, ((), ()))
        position = bisect_right(starts, kernel.launch_start_ns) - 1
        if position < 0:
            return None
        value = values[position]
        return value if kernel.launch_end_ns <= value["range_end_ns"] else None

    for selected in selected_rows:
        model = selected["model_id"]
        kernel_index = selected["kernel_index"]
        groups = [
            (source_key, values)
            for (group_model, group_index, source_key), values
            in observations.items()
            if group_model == model and group_index == kernel_index
            and len(values) == len(loaded)
        ]
        if not groups:
            raise ValueError(
                f"no all-run same-source group for {model} {kernel_index}"
            )
        source_key, matched = max(groups, key=lambda item: (
            max(value["kernel"].duration_ns for value in item[1]) -
            min(value["kernel"].duration_ns for value in item[1]),
            str(item[0]),
        ))
        fast = min(matched, key=lambda value: (
            value["kernel"].duration_ns, value["run_id"],
        ))
        slow = max(matched, key=lambda value: (
            value["kernel"].duration_ns, value["run_id"],
        ))
        label = f"{model}_{kernel_index.lower()}"
        selection_records.append({
            "target_label": label,
            "model_id": model,
            "kernel_index": kernel_index,
            "kernel_signature": selected["kernel_signature"],
            "occurrence_within_signature":
            selected["occurrence_within_signature"],
            "launch_sequence_index": selected["sequence_index_min"],
            "source_scene_token": source_key[0],
            "source_ros_header_timestamp_ns": source_key[1],
            "source_input_id": fast["feature"]["input_id"],
            "source_message_order": fast["source"].get("message_order"),
            "fast_run_id": fast["run_id"],
            "fast_duration_ms": fast["kernel"].duration_ns / 1e6,
            "slow_run_id": slow["run_id"],
            "slow_duration_ms": slow["kernel"].duration_ns / 1e6,
            "matched_range_ms": (
                slow["kernel"].duration_ns - fast["kernel"].duration_ns
            ) / 1e6,
            "same_source_frame": True,
            "same_launch_shape": True,
        })

        foreign_sequences = defaultdict(list)
        target_sequence = fast["identity"]["launch_sequence_index"]
        for observation in matched:
            target = observation["kernel"]
            trace = observation["trace"]
            for foreign in trace.kernels:
                if (
                    foreign.pid != target.pid
                    and foreign.start_ns < target.end_ns
                    and foreign.end_ns > target.start_ns
                ):
                    foreign_feature = enclosing_feature(
                        observation["run_id"], foreign
                    )
                    if foreign_feature is None:
                        continue
                    foreign_key = (
                        foreign.pid, foreign_feature["range_start_ns"],
                        foreign_feature["range_end_ns"],
                    )
                    identities = identity_cache[(
                        observation["run_id"], foreign_key,
                    )]
                    foreign_identity = identities[foreign.kernel_id]
                    foreign_sequences[
                        foreign_identity["model_id"]
                    ].append(foreign_identity["launch_sequence_index"])

        for episode_kind, observation in (("fast", fast), ("slow", slow)):
            target = observation["kernel"]
            trace = observation["trace"]
            ready_ns = trace.ready_times[target.kernel_id]
            stream = trace.streams[
                (target.pid, target.context_id, target.stream_id)
            ]
            position = next(
                index for index, item in enumerate(stream)
                if item.kernel_id == target.kernel_id
            )
            predecessor = stream[position - 1] if position else None
            foreign = [
                item for item in trace.kernels
                if item.pid != target.pid
                and item.start_ns < target.end_ns
                and item.end_ns > target.start_ns
            ]
            overlap_ns = _intersection_duration(
                [(target.start_ns, target.end_ns)],
                [(item.start_ns, item.end_ns) for item in foreign],
            )
            episode_id = f"{label}_{episode_kind}"
            matched_catalog.append({
                "episode_id": episode_id,
                "episode_kind": episode_kind,
                "target_label": label,
                "run_id": observation["run_id"],
                "model_id": model,
                "kernel_index": kernel_index,
                "kernel_signature": selected["kernel_signature"],
                "occurrence_within_signature":
                selected["occurrence_within_signature"],
                "launch_sequence_index":
                observation["identity"]["launch_sequence_index"],
                "input_id": observation["feature"]["input_id"],
                "scene_name": observation["feature"]["scene_name"],
                "scene_token": observation["feature"]["scene_token"],
                "ros_header_timestamp_ns":
                observation["source"].get("ros_header_timestamp_ns"),
                "launch_ready_ns": ready_ns,
                "gpu_start_ns": target.start_ns,
                "gpu_end_ns": target.end_ns,
                "duration_ns": target.duration_ns,
                "ready_to_start_delay_ns": target.start_ns - ready_ns,
                "predecessor_kernel_id": (
                    predecessor.kernel_id if predecessor else None
                ),
                "predecessor_kernel_name": (
                    predecessor.name if predecessor else None
                ),
                "predecessor_gpu_end_ns": (
                    predecessor.end_ns if predecessor else None
                ),
                "cta_count": int(np.prod(target.grid)),
                "foreign_kernel_count": len(foreign),
                "cross_client_overlap_ns": overlap_ns,
                "context_id": target.context_id,
                "stream_id": target.stream_id,
                "grid": "x".join(map(str, target.grid)),
                "block": "x".join(map(str, target.block)),
                "registers_per_thread": target.registers_per_thread,
                "static_shared_memory_bytes": target.static_shared_memory,
                "dynamic_shared_memory_bytes": target.dynamic_shared_memory,
            })
            timeline.append({
                "episode_id": episode_id,
                "role": "target",
                "model_id": model,
                "pid": target.pid,
                "context_id": target.context_id,
                "stream_id": target.stream_id,
                "kernel_id": target.kernel_id,
                "kernel_index": kernel_index,
                "kernel_signature": selected["kernel_signature"],
                "occurrence_within_signature":
                selected["occurrence_within_signature"],
                "launch_sequence_index":
                observation["identity"]["launch_sequence_index"],
                "kernel_name": target.name,
                "gpu_start_ns": target.start_ns,
                "gpu_end_ns": target.end_ns,
                "duration_ns": target.duration_ns,
                "overlap_with_target_ns": target.duration_ns,
                "relative_phase": "target",
                "grid": "x".join(map(str, target.grid)),
                "block": "x".join(map(str, target.block)),
                "registers_per_thread": target.registers_per_thread,
                "static_shared_memory_bytes": target.static_shared_memory,
                "dynamic_shared_memory_bytes": target.dynamic_shared_memory,
                "cta_count": int(np.prod(target.grid)),
            })
            for item in foreign:
                foreign_feature = enclosing_feature(
                    observation["run_id"], item
                )
                if foreign_feature is None:
                    foreign_identity = {
                        "kernel_index": "outside_inference",
                        "kernel_signature": _kernel_signature(item),
                        "occurrence_within_signature": None,
                        "launch_sequence_index": None,
                        "model_id": observation["selector"].models.get(
                            item.pid, str(item.pid)
                        ),
                    }
                else:
                    foreign_key = (
                        item.pid, foreign_feature["range_start_ns"],
                        foreign_feature["range_end_ns"],
                    )
                    foreign_identity = identity_cache[(
                        observation["run_id"], foreign_key,
                    )][item.kernel_id]
                overlap = min(target.end_ns, item.end_ns) - max(
                    target.start_ns, item.start_ns
                )
                feasible, reason, threads, registers, shared = (
                    _co_residency_feasibility({
                        "block": target.block,
                        "registers_per_thread":
                            target.registers_per_thread,
                        "static_shared_memory":
                            target.static_shared_memory,
                        "dynamic_shared_memory":
                            target.dynamic_shared_memory,
                    }, {
                        "block": item.block,
                        "registers_per_thread": item.registers_per_thread,
                        "static_shared_memory": item.static_shared_memory,
                        "dynamic_shared_memory": item.dynamic_shared_memory,
                    }, DEVICE_LIMITS)
                )
                if item.start_ns <= target.start_ns:
                    phase = (
                        "covers_target" if item.end_ns >= target.end_ns
                        else "already_running"
                    )
                else:
                    phase = "starts_during_target"
                timeline.append({
                    "episode_id": episode_id,
                    "role": "co_runner",
                    "model_id": foreign_identity["model_id"],
                    "pid": item.pid,
                    "context_id": item.context_id,
                    "stream_id": item.stream_id,
                    "kernel_id": item.kernel_id,
                    "kernel_index": foreign_identity["kernel_index"],
                    "kernel_signature":
                    foreign_identity["kernel_signature"],
                    "occurrence_within_signature":
                    foreign_identity["occurrence_within_signature"],
                    "launch_sequence_index":
                    foreign_identity["launch_sequence_index"],
                    "kernel_name": item.name,
                    "gpu_start_ns": item.start_ns,
                    "gpu_end_ns": item.end_ns,
                    "duration_ns": item.duration_ns,
                    "overlap_with_target_ns": overlap,
                    "relative_phase": phase,
                    "grid": "x".join(map(str, item.grid)),
                    "block": "x".join(map(str, item.block)),
                    "registers_per_thread": item.registers_per_thread,
                    "static_shared_memory_bytes": item.static_shared_memory,
                    "dynamic_shared_memory_bytes":
                    item.dynamic_shared_memory,
                    "cta_count": int(np.prod(item.grid)),
                    "resource_feasible_candidate": feasible,
                    "infeasible_reason": reason,
                    "combined_threads": threads,
                    "combined_allocated_registers": registers,
                    "combined_allocated_shared_memory": shared,
                    "cross_client_smid_comparable":
                        CROSS_CLIENT_SMID_COMPARABLE,
                    "physical_co_residency_supported": False,
                })

        source = fast["source"]
        capture_targets.append({
            "label": label,
            "target_model": model,
            "kernel_index": kernel_index,
            "kernel_signature": selected["kernel_signature"],
            "occurrence_within_signature":
            selected["occurrence_within_signature"],
            "scene_index": source.get("scene_index"),
            "scene_name": source.get("scene_name"),
            "scene_token": source.get("scene_token"),
            "message_order": source.get("message_order"),
            "ros_header_timestamp_ns": source.get(
                "ros_header_timestamp_ns"
            ),
            "input_id": fast["feature"]["input_id"],
        })
        for client_model in sorted({
            row["model_id"] for row in kernel_rows
        }):
            if client_model == model:
                start = max(1, target_sequence - 2)
                end = target_sequence + 2
            else:
                values = foreign_sequences[client_model]
                if not values:
                    raise ValueError(
                        f"no co-runner window for {label} {client_model}"
                    )
                start = max(1, min(values) - 2)
                end = max(values) + 2
            client_windows[client_model].append({
                "label": label,
                "scene_index": source.get("scene_index"),
                "message_order": source.get("message_order"),
                "sequence_start": start,
                "sequence_end": end,
            })

    capacity = {}
    for model, windows in client_windows.items():
        total = 0
        for window in windows:
            label = window["label"]
            target = next(
                item for item in capture_targets if item["label"] == label
            )
            maximum = 0
            for run_id, (_selector, _trace, features, _ranges,
                         inference_kernels) in loaded.items():
                feature = next(
                    item for item in features
                    if item["model_id"] == model
                    and item["input_id"] == target["input_id"]
                )
                key = (
                    feature["pid"], feature["range_start_ns"],
                    feature["range_end_ns"],
                )
                kernels = sorted(
                    inference_kernels[key],
                    key=lambda item: (
                        item.launch_start_ns, item.launch_end_ns,
                        item.kernel_id,
                    ),
                )
                end = min(window["sequence_end"], len(kernels))
                maximum = max(maximum, sum(
                    int(np.prod(item.grid))
                    for item in kernels[window["sequence_start"] - 1:end]
                ))
            window["maximum_expected_cta_records"] = maximum
            total += maximum
        if total > 1048576:
            raise ValueError(
                f"passive CTA plan exceeds fixed capacity for {model}: "
                f"{total}"
            )
        capacity[model] = total
    plan = {
        "schema": "mps_two_model_passive_capture_plan_v1",
        "selection_locked_before_cta": True,
        "mode": "passive",
        "launch_alignment": False,
        "launch_manipulation": False,
        "targets": capture_targets,
        "clients": {
            model: {
                "record_capacity": 1048576,
                "maximum_expected_cta_records": capacity[model],
                "windows": windows,
            }
            for model, windows in sorted(client_windows.items())
        },
        "completeness_gate": (
            "single-stream target interval must be bracketed by the "
            "captured co-runner window, or the captured window must reach "
            "the co-runner inference boundary; all launches and CTA "
            "boundaries must be complete with zero drops"
        ),
    }
    return selection_records, matched_catalog, timeline, plan


def analyze_discovery(run_root, output_root):
    """Analyze all low-overhead discovery repetitions and select episodes."""
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    loaded = {}
    for run_directory in sorted(run_root.glob("mps2cause-discovery-r*")):
        loaded[run_directory.name] = _run_features(run_directory)
        all_rows.extend(loaded[run_directory.name][2])
    if not all_rows:
        raise ValueError("no completed Phase 4 discovery runs")
    _write_csv(output_root / "discovery_inference_gpu_features.csv", all_rows)
    inventory, kernel_rows = _kernel_inventory(loaded)
    _write_csv(output_root / "kernel_inventory_summary.csv", inventory)
    _write_csv(output_root / "kernel_index_variation.csv", kernel_rows)
    decisions, selected_rows = _target_decisions(kernel_rows, len(loaded))
    _write_csv(
        output_root / "deterministic_target_selection.csv", decisions
    )
    (selection_records, matched_catalog, matched_timeline,
     passive_plan) = _matched_target_artifacts(
        loaded, kernel_rows, selected_rows
    )
    _write_csv(
        output_root / "selected_target_catalog.csv", selection_records
    )
    _write_csv(
        output_root / "matched_episode_catalog.csv", matched_catalog
    )
    _write_csv(
        output_root / "matched_kernel_timeline.csv", matched_timeline
    )
    (output_root / "passive_capture_plan.json").write_text(
        json.dumps(passive_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    kernel_sets = _kernel_set_correlations(loaded, kernel_rows)
    _write_csv(output_root / "kernel_set_correlation.csv", kernel_sets)

    victim_rows = [row for row in all_rows if row["model_id"] == "faster_rcnn"]
    outcome = np.asarray(
        [row["inference_latency_ms"] for row in victim_rows], dtype=float
    )
    correlations = [
        {
            "feature": feature,
            "spearman_rho": _correlation(
                outcome,
                np.asarray([row[feature] for row in victim_rows], dtype=float),
            ),
        }
        for feature in FEATURES
    ]
    correlations.sort(key=lambda row: abs(row["spearman_rho"]), reverse=True)
    _write_csv(
        output_root / "discovery_gpu_feature_correlations.csv", correlations
    )

    affected_row = max(
        victim_rows, key=lambda row: row["inference_latency_ms"]
    )
    same_scope = [
        row for row in victim_rows
        if row["run_id"] == affected_row["run_id"]
        and row["scene_index"] == affected_row["scene_index"]
    ]
    median = float(np.median([
        row["inference_latency_ms"] for row in same_scope
    ]))
    unaffected_row = min(
        same_scope, key=lambda row: abs(row["inference_latency_ms"] - median)
    )
    episodes = {}
    for name, row in (
        ("affected", affected_row), ("unaffected", unaffected_row)
    ):
        selector, trace, rows, inferences, kernels = loaded[row["run_id"]]
        episodes[name] = _episode(
            selector, trace, rows, inferences, kernels, row
        )

    affected_by_signature = defaultdict(list)
    unaffected_by_signature = defaultdict(list)
    for kernel in episodes["affected"]["kernels"]:
        affected_by_signature[kernel["kernel_signature"]].append(kernel)
    for kernel in episodes["unaffected"]["kernels"]:
        unaffected_by_signature[kernel["kernel_signature"]].append(kernel)
    candidates = []
    for signature in set(affected_by_signature) & set(unaffected_by_signature):
        pairs = zip(
            affected_by_signature[signature],
            unaffected_by_signature[signature],
        )
        for ordinal, (affected, unaffected) in enumerate(pairs):
            overlap = sum(
                max(
                    0,
                    min(affected["end_ns"], foreign["end_ns"])
                    - max(affected["start_ns"], foreign["start_ns"]),
                )
                for foreign in episodes["affected"]["foreign_kernels"]
            )
            candidates.append((
                affected["duration_ns"] - unaffected["duration_ns"] + overlap,
                overlap,
                ordinal,
                affected,
                unaffected,
            ))
    if not candidates:
        raise ValueError("affected and unaffected episodes share no kernel")
    _, _, ordinal, affected_kernel, unaffected_kernel = max(candidates)
    aggressor = max(
        episodes["affected"]["foreign_kernels"],
        key=lambda kernel: max(
            0,
            min(affected_kernel["end_ns"], kernel["end_ns"])
            - max(affected_kernel["start_ns"], kernel["start_ns"]),
        ),
    )
    overlap = max(
        0,
        min(affected_kernel["end_ns"], aggressor["end_ns"])
        - max(affected_kernel["start_ns"], aggressor["start_ns"]),
    )
    unaffected_aggressors = [
        kernel for kernel in episodes["unaffected"]["foreign_kernels"]
        if kernel["kernel_signature"] == aggressor["kernel_signature"]
    ] or episodes["unaffected"]["foreign_kernels"]
    unaffected_aggressor = max(
        unaffected_aggressors,
        key=lambda kernel: max(
            0,
            min(unaffected_kernel["end_ns"], kernel["end_ns"])
            - max(unaffected_kernel["start_ns"], kernel["start_ns"]),
        ),
    )
    selection = {
        "selection_rule": (
            "maximum Faster R-CNN GPU-complete inference; same-run, "
            "same-scene median-nearest control; shared kernel instance "
            "maximizing duration "
            "delta plus cross-client overlap"
        ),
        "affected": {
            key: value for key, value in episodes["affected"].items()
            if key not in ("kernels", "foreign_kernels")
        },
        "unaffected": {
            key: value for key, value in episodes["unaffected"].items()
            if key not in ("kernels", "foreign_kernels")
        },
        "victim_kernel_ordinal_within_signature": ordinal,
        "affected_victim_kernel": affected_kernel,
        "unaffected_victim_kernel": unaffected_kernel,
        "affected_aggressor_kernel": aggressor,
        "unaffected_aggressor_kernel": unaffected_aggressor,
        "affected_victim_aggressor_overlap_ns": overlap,
    }
    (output_root / "target_episodes.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "mps_two_model_gpu_discovery_v1",
        "run_count": len(loaded),
        "inference_count": len(all_rows),
        "victim_inference_count": len(victim_rows),
        "kernel_inventory": inventory,
        "kernel_set_correlations": kernel_sets,
        "deterministic_target_selection": selection_records,
        "passive_capture_plan": passive_plan,
        "ranked_gpu_feature_correlations": correlations,
        "target_episodes": selection,
    }
    (output_root / "discovery_gpu_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def analyze_device_metrics(run_directory, output_root, target_path):
    """Join the selected input identities to sampled GPU device metrics."""
    targets = json.loads(target_path.read_text(encoding="utf-8"))
    _, _, rows, _, _ = _run_features(run_directory)
    selected = {
        (target["model_id"], str(target["input_id"])): label
        for label, target in (
            ("affected", targets["affected"]),
            ("unaffected", targets["unaffected"]),
        )
    }
    metric_rows = []
    with sqlite3.connect(str(run_directory / "profile.sqlite")) as database:
        for row in rows:
            label = selected.get((row["model_id"], row["input_id"]))
            if label is None:
                continue
            query = database.execute(
                "SELECT t.metricName,m.value FROM GPU_METRICS m "
                "JOIN TARGET_INFO_GPU_METRICS t "
                "ON t.typeId=m.typeId AND t.metricId=m.metricId "
                "WHERE m.timestamp>=? AND m.timestamp<?",
                (row["range_start_ns"], row["gpu_end_ns"]),
            )
            values = defaultdict(list)
            for metric, value in query:
                values[str(metric)].append(float(value))
            for metric in sorted(values):
                samples = np.asarray(values[metric], dtype=float)
                metric_rows.append({
                    "episode": label,
                    "model_id": row["model_id"],
                    "input_id": row["input_id"],
                    "metric": metric,
                    "sample_count": len(samples),
                    "mean": float(np.mean(samples)),
                    "p50": float(np.percentile(samples, 50)),
                    "p95": float(np.percentile(samples, 95)),
                    "maximum": float(np.max(samples)),
                })
    if len({row["episode"] for row in metric_rows}) != 2:
        raise ValueError("device metrics do not cover both target inputs")
    _write_csv(output_root / "target_device_metrics.csv", metric_rows)
    return metric_rows


def analyze_postpassive(run_root, output_root, catalog_path):
    """Compare postpassive low-overhead runs at one locked logical index."""
    output_root.mkdir(parents=True, exist_ok=True)
    catalog = list(csv.DictReader(catalog_path.open(encoding="utf-8")))
    target = next(row for row in catalog if row["kernel_index"] == "K1091")
    sequence = int(target["launch_sequence_index"])
    signature = target["kernel_signature"]
    source_message = int(target["source_message_order"])
    inference_rows = []
    kernel_rows = []
    co_rows = []
    for run_directory in sorted(
        run_root.glob("mps2cause-postpassive-*-r1")
    ):
        selector, trace, features, _, inference_kernels = _run_features(
            run_directory
        )
        condition = run_directory.name[
            len("mps2cause-postpassive-"):-len("-r1")
        ]
        feature_by_key = {
            (row["pid"], row["range_start_ns"], row["range_end_ns"]): row
            for row in features
        }
        locations = {}
        for key, kernels in inference_kernels.items():
            feature = feature_by_key.get(key)
            if feature is None:
                continue
            occurrences = defaultdict(int)
            for index, kernel in enumerate(sorted(
                kernels,
                key=lambda item: (
                    item.launch_start_ns, item.launch_end_ns, item.kernel_id,
                ),
            ), 1):
                observed_signature = _kernel_signature(kernel)
                occurrence = occurrences[observed_signature]
                occurrences[observed_signature] += 1
                locations[kernel.kernel_id] = {
                    "model_id": feature["model_id"],
                    "input_id": feature["input_id"],
                    "launch_sequence_index": index,
                    "occurrence_within_signature": occurrence,
                }
        inputs = selector._model_input_records()
        inference_rows.extend({"postpassive_condition": condition, **row}
                              for row in features)
        for feature in features:
            if feature["model_id"] != "faster_rcnn":
                continue
            key = (
                feature["pid"], feature["range_start_ns"],
                feature["range_end_ns"],
            )
            kernels = sorted(
                inference_kernels[key],
                key=lambda item: (
                    item.launch_start_ns, item.launch_end_ns, item.kernel_id,
                ),
            )
            if len(kernels) < sequence:
                continue
            kernel = kernels[sequence - 1]
            if _kernel_signature(kernel) != signature:
                continue
            source = inputs.get(("faster_rcnn", feature["input_id"]), {})
            foreign = [
                candidate for candidate in trace.kernels
                if candidate.pid != kernel.pid
                and candidate.start_ns < kernel.end_ns
                and candidate.end_ns > kernel.start_ns
            ]
            overlap = _intersection_duration(
                [(kernel.start_ns, kernel.end_ns)],
                [(item.start_ns, item.end_ns) for item in foreign],
            )
            target_resource = _kernel_record(kernel, "faster_rcnn", 0)
            row = {
                "run_id": run_directory.name,
                "condition": condition,
                "model_id": "faster_rcnn",
                "input_id": feature["input_id"],
                "scene_index": source.get("scene_index"),
                "message_order": source.get("message_order"),
                "ros_header_timestamp_ns": source.get(
                    "ros_header_timestamp_ns"
                ),
                "kernel_index": "K1091",
                "kernel_signature": signature,
                "launch_sequence_index": sequence,
                "occurrence_within_signature": locations[
                    kernel.kernel_id
                ]["occurrence_within_signature"],
                "context_id": kernel.context_id,
                "stream_id": kernel.stream_id,
                "grid": "x".join(map(str, kernel.grid)),
                "block": "x".join(map(str, kernel.block)),
                "kernel_start_ns": kernel.start_ns,
                "kernel_end_ns": kernel.end_ns,
                "kernel_duration_ns": kernel.duration_ns,
                "ready_to_start_delay_ns": max(
                    0, kernel.start_ns - trace.ready_times[kernel.kernel_id]
                ),
                "cross_client_overlap_ns": overlap,
                "foreign_kernel_count": len(foreign),
                "inference_latency_ms": feature["inference_latency_ms"],
                "locked_source_frame": (
                    source.get("message_order") == source_message
                ),
            }
            kernel_rows.append(row)
            for candidate in foreign:
                candidate_model = selector.models.get(
                    candidate.pid, str(candidate.pid)
                )
                candidate_resource = _kernel_record(
                    candidate, candidate_model, 0
                )
                feasible, reason, threads, registers, shared = (
                    _co_residency_feasibility(
                        target_resource, candidate_resource, DEVICE_LIMITS
                    )
                )
                co_rows.append({
                    "run_id": run_directory.name,
                    "condition": condition,
                    "target_input_id": feature["input_id"],
                    "co_model_id": candidate_model,
                    "co_input_id": locations.get(
                        candidate.kernel_id, {}
                    ).get("input_id", ""),
                    "co_launch_sequence_index": locations.get(
                        candidate.kernel_id, {}
                    ).get("launch_sequence_index", ""),
                    "co_occurrence_within_signature": locations.get(
                        candidate.kernel_id, {}
                    ).get("occurrence_within_signature", ""),
                    "co_kernel_name": candidate.name,
                    "co_kernel_signature": _kernel_signature(candidate),
                    "co_context_id": candidate.context_id,
                    "co_stream_id": candidate.stream_id,
                    "co_start_ns": candidate.start_ns,
                    "co_end_ns": candidate.end_ns,
                    "overlap_ns": max(
                        0, min(kernel.end_ns, candidate.end_ns)
                        - max(kernel.start_ns, candidate.start_ns)
                    ),
                    "relative_phase_ns": candidate.start_ns - kernel.start_ns,
                    "resource_feasible_candidate": feasible,
                    "infeasible_reason": reason,
                    "combined_threads": threads,
                    "combined_allocated_registers": registers,
                    "combined_allocated_shared_memory": shared,
                    "cross_client_smid_comparable": False,
                    "physical_co_residency_supported": False,
                })
    summaries = []
    by_source = defaultdict(list)
    for row in kernel_rows:
        by_source[(
            row["scene_index"], row["message_order"],
            row["ros_header_timestamp_ns"],
        )].append(row)
    common_sources = {
        key for key, rows in by_source.items()
        if {row["condition"] for row in rows} == {
            "intervention", "reversal", "negative_control"
        }
    }
    common_rows = [
        row for key in sorted(common_sources) for row in by_source[key]
    ]
    for condition in ("intervention", "reversal", "negative_control"):
        inferences = [
            row for row in inference_rows
            if row["postpassive_condition"] == condition
            and row["model_id"] == "faster_rcnn"
        ]
        kernels = [row for row in kernel_rows if row["condition"] == condition]
        target_frame = next((
            row for row in kernels if row["locked_source_frame"]
        ), None)
        inference_metrics = _duration_metrics([
            row["inference_latency_ms"] for row in inferences
        ])
        kernel_metrics = (
            _duration_metrics([
                row["kernel_duration_ns"] / 1e6 for row in kernels
            ])
            if kernels else
            {**{f"p{quantile}_ms": "" for quantile in KERNEL_QUANTILES},
             "primary_range_ms": "", "normalized_range": ""}
        )
        summaries.append({
            "condition": condition,
            "inference_sample_count": len(inferences),
            "k1091_sample_count": len(kernels),
            **{f"inference_{key}": value
               for key, value in inference_metrics.items()},
            **{f"k1091_{key}": value for key, value in kernel_metrics.items()},
            "target_frame_kernel_duration_ms":
                target_frame["kernel_duration_ns"] / 1e6
                if target_frame else "",
            "target_frame_cross_client_overlap_ms":
                target_frame["cross_client_overlap_ns"] / 1e6
                if target_frame else "",
            "target_frame_inference_latency_ms":
                target_frame["inference_latency_ms"] if target_frame else "",
            "target_frame_exact_index_available": bool(target_frame),
            "common_source_frame_count": len(common_sources),
        })
    _write_csv(output_root / "postpassive_inference_features.csv",
               inference_rows)
    _write_csv(output_root / "postpassive_k1091_instances.csv", kernel_rows)
    _write_csv(output_root / "postpassive_k1091_corunners.csv", co_rows)
    _write_csv(
        output_root / "postpassive_k1091_common_frames.csv", common_rows)
    _write_csv(output_root / "postpassive_condition_summary.csv", summaries)
    result = {
        "schema": "mps_two_model_postpassive_validation_v1",
        "target": target,
        "run_count": 3,
        "common_source_frame_count": len(common_sources),
        "summaries": summaries,
        "claim_boundary": (
            "GPU-side descriptive validation; no cross-client physical SM "
            "or named global-resource claim"
        ),
    }
    (output_root / "postpassive_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--postpassive-target", type=Path)
    args = parser.parse_args(argv)
    if args.postpassive_target:
        result = analyze_postpassive(
            args.run_root, args.output_root, args.postpassive_target
        )
        print(json.dumps(result["summaries"], indent=2))
        return 0
    result = analyze_discovery(args.run_root, args.output_root)
    device_run = args.run_root / "mps2cause-device-metrics"
    if device_run.is_dir():
        analyze_device_metrics(
            device_run,
            args.output_root,
            args.output_root / "target_episodes.json",
        )
    print(json.dumps(result["ranked_gpu_feature_correlations"], indent=2))
    return 0
