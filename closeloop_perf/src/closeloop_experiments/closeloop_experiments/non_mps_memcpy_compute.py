"""Bidirectional non-MPS memcpy-alignment and compute-demand campaign."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys

import yaml

from .config import ConfigError, load_run_config, schema_v2_config
from closeloop_testbed.resource_control import (
    GPUClockLock,
    MPS_ENVIRONMENT_VARIABLES,
    current_gpu_clocks,
    maximum_supported_clocks,
)
from .runner import ExperimentRunner, PreflightError, RunFailure
from closeloop_analyzer.target_selection import decode_global_id

SKILL = "NonMpsMemcpyComputeDemandV2"
ALIGNMENT_METRIC = "h2d_d2h_preceding_context_window_v1"
K_VALUES = (0, 1, 2, 4, 8)
TRIALS = 5
OFFSET_COUNT = 10
CALIBRATION_PERIOD_SECONDS = 10.0
REQUIRED_STUDY_FIELDS = {
    "schema_version",
    "skill",
    "baseline_config",
    "directions",
    "fixed_input_paths",
    "k_values",
    "trials_per_run",
    "launch_tolerance_ms",
    "matched_overlap_tolerance",
    "alignment_difference_min",
    "alignment_window_ms",
    "non_overlap_latency_tolerance",
    "period_guard_fraction",
    "output_root",
    "random_seed",
}


class StudyError(RuntimeError):
    """Raised when study configuration or campaign evidence is invalid."""


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_immutable(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise StudyError(f"immutable file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o444)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _resolve(source, value):
    path = Path(value).expanduser()
    return (
        (source.parent / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )


def load_study(path, check_paths=True):
    """Load and strictly validate one bidirectional V2 study."""
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise StudyError(f"cannot load study: {exc}") from exc
    if not isinstance(data, dict):
        raise StudyError("study root must be an object")
    missing = REQUIRED_STUDY_FIELDS.difference(data)
    unknown = set(data).difference(REQUIRED_STUDY_FIELDS)
    if missing or unknown:
        raise StudyError(
            f"study fields missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    expected = {
        "schema_version": 1,
        "skill": SKILL,
        "k_values": list(K_VALUES),
        "trials_per_run": TRIALS,
    }
    for name, value in expected.items():
        if data[name] != value:
            raise StudyError(f"{name} must be {value!r} for {SKILL}")
    if not isinstance(data["random_seed"], int):
        raise StudyError("random_seed must be an integer")
    for name in (
        "matched_overlap_tolerance",
        "alignment_difference_min",
        "non_overlap_latency_tolerance",
        "period_guard_fraction",
    ):
        if not 0 < float(data[name]) < 1:
            raise StudyError(f"{name} must be between zero and one")
    if not 0 < float(data["launch_tolerance_ms"]) <= 0.5:
        raise StudyError("launch_tolerance_ms must be in (0, 0.5]")
    if not 0 < float(data["alignment_window_ms"]):
        raise StudyError("alignment_window_ms must be positive")
    baseline_path = _resolve(source, data["baseline_config"])
    try:
        baseline = load_run_config(str(baseline_path), check_paths=check_paths)
    except ConfigError as exc:
        raise StudyError(f"invalid baseline: {exc}") from exc
    if baseline.data["gpu"]["mps_enabled"]:
        raise StudyError("non-MPS study cannot enable MPS")
    if not baseline.data["recording"]["nsys"]["gpu_context_switch"]:
        raise StudyError("GPU context-switch tracing is required")
    if any("mps_percentage" in model for model in baseline.models):
        raise StudyError("non-MPS study cannot set MPS percentages")
    model_ids = {model["id"] for model in baseline.models}
    if len(model_ids) != 2:
        raise StudyError("baseline must contain exactly two models")
    directions = data["directions"]
    if not isinstance(directions, list) or len(directions) != 2:
        raise StudyError("directions must contain the two role directions")
    required_direction = {
        "id",
        "target_model_id",
        "co_runner_model_id",
        "replay_module",
    }
    pairs = set()
    for direction in directions:
        if (
            not isinstance(direction, dict)
            or set(direction) != required_direction
        ):
            raise StudyError(
                "each direction requires id, role IDs, and replay_module"
            )
        pair = (direction["target_model_id"], direction["co_runner_model_id"])
        if set(pair) != model_ids or pair[0] == pair[1]:
            raise StudyError("each direction must assign both baseline models")
        pairs.add(pair)
    if len(pairs) != 2 or {pair[::-1] for pair in pairs} != pairs:
        raise StudyError("directions must reverse target and co-runner roles")
    if len({direction["id"] for direction in directions}) != 2:
        raise StudyError("direction IDs must be unique")
    if set(data["fixed_input_paths"]) != model_ids:
        raise StudyError("fixed_input_paths must cover both model IDs")
    inputs = {
        model_id: _resolve(source, value)
        for model_id, value in data["fixed_input_paths"].items()
    }
    if check_paths:
        missing_inputs = [
            str(path) for path in inputs.values() if not path.is_file()
        ]
        if missing_inputs:
            raise StudyError(
                "fixed inputs missing: " + ", ".join(missing_inputs)
            )
    output_root = _resolve(source, data["output_root"])
    return {
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": data,
        "baseline": baseline,
        "inputs": inputs,
        "directions": {item["id"]: item for item in directions},
        "output_root": output_root,
    }


def candidate_id(direction_id, offset_index, k_replays):
    """Return stable identity independent of randomized execution order."""
    return (
        f"nmcd-{direction_id}-o{int(offset_index):02d}-k{int(k_replays):02d}"
    )


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(
        (int(start), int(end)) for start, end in intervals if end > start
    ):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _union_duration(intervals, start=None, end=None):
    if start is not None and end is not None:
        intervals = [
            (max(start, left), min(end, right))
            for left, right in intervals
            if left < end and right > start
        ]
    return sum(right - left for left, right in _merge_intervals(intervals))


def _intersection_duration(left, right):
    left, right = _merge_intervals(left), _merge_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(
            0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0])
        )
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def alignment_score(
    target_copies, co_context_timeline, delta_ns, window_ns
):
    """Average co-runner context occupancy before target H2D/D2H copies."""
    window_ns = int(window_ns)
    if window_ns <= 0:
        raise ValueError("alignment window must be positive")
    starts = [
        int(item["start_ns"])
        for item in target_copies
        if item.get("copy_class") in ("H2D", "D2H")
    ]
    if not starts:
        return None
    shifted = _merge_intervals(
        (
            int(item["start_ns"]) + int(delta_ns),
            int(item["end_ns"]) + int(delta_ns),
        )
        for item in co_context_timeline
    )
    overlap = sum(
        _union_duration(shifted, start - window_ns, start)
        for start in starts
    )
    return overlap / (len(starts) * window_ns)


def execution_overlap(target_duration_ns, co_duration_ns, delta_ns):
    """Return target-normalized predicted inference-range overlap."""
    overlap = max(
        0,
        min(target_duration_ns, delta_ns + co_duration_ns) - max(0, delta_ns),
    )
    return overlap / target_duration_ns if target_duration_ns else 0.0


def select_offsets(
    target_timeline,
    co_context_timeline,
    target_duration_ns,
    co_duration_ns,
    co_k8_duration_ns,
    alignment_window_ns,
    overlap_tolerance=0.05,
    alignment_difference_min=0.05,
):
    """Select ten unique signed offsets and preregister a matched pair."""
    margin = max(
        1, round(max(target_duration_ns, co_k8_duration_ns) * 0.25)
    )
    lower = -int(co_k8_duration_ns) - margin
    upper = int(target_duration_ns) + margin
    steps = 20001
    grid = sorted(
        {
            round(lower + index * (upper - lower) / (steps - 1))
            for index in range(steps)
        }
    )
    metrics = {
        delta: (
            execution_overlap(target_duration_ns, co_duration_ns, delta),
            alignment_score(
                target_timeline,
                co_context_timeline,
                delta,
                alignment_window_ns,
            )
            or 0.0,
        )
        for delta in grid
    }
    peak = max(value[0] for value in metrics.values())
    peak_center = statistics.median(
        [delta for delta, value in metrics.items() if value[0] == peak]
    )
    selected = [("nonoverlap_left", lower), ("nonoverlap_right", upper)]
    used = {lower, upper}

    def choose(label, candidates, key):
        for delta in sorted(candidates, key=key):
            if delta not in used:
                selected.append((label, delta))
                used.add(delta)
                return
        raise StudyError("offset calibration could not produce unique cells")

    for fraction in (0.25, 0.50, 0.75):
        desired = fraction * peak
        choose(
            f"overlap_{int(fraction * 100)}_left",
            [delta for delta in grid if delta < peak_center],
            lambda delta, wanted=desired: abs(metrics[delta][0] - wanted),
        )
        choose(
            f"overlap_{int(fraction * 100)}_right",
            [delta for delta in grid if delta > peak_center],
            lambda delta, wanted=desired: abs(metrics[delta][0] - wanted),
        )
    choose(
        "max_alignment", grid, lambda delta: (-metrics[delta][1], abs(delta))
    )
    peak_deltas = [
        delta for delta in grid if abs(metrics[delta][0] - peak) <= 1e-12
    ]
    choose(
        "max_overlap_low_alignment",
        peak_deltas,
        lambda delta: (metrics[delta][1], abs(delta)),
    )
    result = [
        {
            "offset_index": index,
            "label": label,
            "delta_ns": delta,
            "predicted_execution_overlap": metrics[delta][0],
            "predicted_alignment": metrics[delta][1],
        }
        for index, (label, delta) in enumerate(selected)
    ]
    pairs = []
    for left_index, left in enumerate(result):
        for right in result[left_index + 1:]:
            overlap_difference = abs(
                left["predicted_execution_overlap"]
                - right["predicted_execution_overlap"]
            )
            alignment_difference = abs(
                left["predicted_alignment"] - right["predicted_alignment"]
            )
            if (
                overlap_difference <= overlap_tolerance
                and alignment_difference >= alignment_difference_min
            ):
                pairs.append(
                    (alignment_difference, -overlap_difference, left, right)
                )
    if not pairs:
        raise StudyError("no matched-overlap pair meets alignment separation")
    _alignment, _overlap, first, second = max(
        pairs, key=lambda item: (item[0], item[1])
    )
    high, low = sorted(
        (first, second),
        key=lambda item: item["predicted_alignment"],
        reverse=True,
    )
    matched = {
        "high_alignment_offset_index": high["offset_index"],
        "low_alignment_offset_index": low["offset_index"],
        "execution_overlap_difference": abs(
            high["predicted_execution_overlap"]
            - low["predicted_execution_overlap"]
        ),
        "alignment_difference": (
            high["predicted_alignment"] - low["predicted_alignment"]
        ),
    }
    return result, matched


def canonical_copy_timeline(trials):
    """Median five matching copy signatures by class, bytes, and ordinal."""
    if len(trials) != TRIALS:
        raise StudyError(f"copy calibration requires {TRIALS} trials")
    signatures = [
        [
            (item["copy_class"], item["bytes"], item["ordinal"])
            for item in trial
        ]
        for trial in trials
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise StudyError("H2D/D2H count or byte signature changed")
    return [
        {
            "copy_class": key[0],
            "bytes": key[1],
            "ordinal": key[2],
            "start_ns": int(
                statistics.median(trial[index]["start_ns"] for trial in trials)
            ),
            "end_ns": int(
                statistics.median(trial[index]["end_ns"] for trial in trials)
            ),
        }
        for index, key in enumerate(signatures[0])
    ]


def canonical_context_timeline(trials):
    """Return the observed context timeline most typical of five trials."""
    if len(trials) != TRIALS:
        raise StudyError(f"context calibration requires {TRIALS} trials")
    timelines = [
        _merge_intervals(
            (item["start_ns"], item["end_ns"]) for item in trial
        )
        for trial in trials
    ]
    if any(not timeline for timeline in timelines):
        raise StudyError("active context timeline is empty")
    selected = min(
        range(TRIALS),
        key=lambda index: sum(
            _union_duration(timelines[index])
            + _union_duration(other)
            - 2 * _intersection_duration(timelines[index], other)
            for other in timelines
        ),
    )
    return [
        {
            "ordinal": index,
            "start_ns": start,
            "end_ns": end,
        }
        for index, (start, end) in enumerate(timelines[selected])
    ]


def calibrated_trial_period_ns(models, directions, guard_fraction):
    """Round the guarded worst signed-offset paired K=8 envelope."""
    envelopes = []
    for item in directions:
        target = models[item["target_model_id"]]["0"][
            "worst_execution_envelope_ns"
        ]
        co_runner = models[item["co_runner_model_id"]]["8"][
            "worst_execution_envelope_ns"
        ]
        for offset in item["offsets"]:
            delta = offset["delta_ns"]
            anchor = max(0, -delta)
            envelopes.append(
                max(anchor + target, anchor + delta + co_runner)
            )
    worst = max(envelopes)
    guarded = worst * (1 + guard_fraction)
    return max(
        100_000_000,
        math.ceil(guarded / 100_000_000) * 100_000_000,
    )


def validate_non_mps_host(study, run_command=subprocess.run, environment=None):
    """Reject active MPS or non-default GPU compute mode."""
    environment = os.environ if environment is None else environment
    daemon = run_command(
        ["pgrep", "-x", "nvidia-cuda-mps-control"],
        check=False,
        capture_output=True,
        text=True,
    )
    if daemon.returncode == 0 and daemon.stdout.strip():
        raise StudyError("an MPS control daemon is active")
    gpu_index = study["baseline"].data["gpu"]["index"]
    mode = run_command(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=compute_mode",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if mode.stdout.strip().lower() not in ("default", "0"):
        raise StudyError("GPU compute mode is not Default")
    return {
        "mps_daemon_active": False,
        "compute_mode": mode.stdout.strip(),
        "inherited_mps_environment_variables": sorted(
            name for name in MPS_ENVIRONMENT_VARIABLES if name in environment
        ),
        "gpu_context_switch_tracing": True,
    }


def _paired_model(
    model,
    direction,
    role,
    trials,
    period,
    anchor,
    deadline,
    tolerance,
    k_replays,
    replay_module=None,
):
    result = copy.deepcopy(model)
    result["launch_offset_seconds"] = 0
    result["paired_trial"] = {
        "direction": direction,
        "role": role,
        "trials": trials,
        "period_seconds": period,
        "target_anchor_seconds": anchor,
        "deadline_offset_seconds": deadline,
        "launch_tolerance_seconds": tolerance,
        "k_replays": k_replays,
    }
    if replay_module is not None:
        result["paired_trial"]["replay_module"] = replay_module
    return result


def _run_metadata(
    data, identifier, experiment, provenance, output_root, parent="calibration"
):
    run = data["run"]
    for name in ("parent_run_id", "hypothesis_id", "intervention_id"):
        run.pop(name, None)
    run.update(
        {
            "id": identifier,
            "experiment": experiment,
            "provenance": provenance,
            "output_root": str(output_root),
            "phase": "validation",
            "parent_run_id": parent,
            "hypothesis_id": "memcpy_alignment_compute_service_demand",
            "intervention_id": identifier,
        }
    )


def _model_replay_modules(study):
    return {
        direction["co_runner_model_id"]: direction["replay_module"]
        for direction in study["directions"].values()
    }


def _calibration_config(study, model_id, k_replays, clocks):
    data = copy.deepcopy(study["baseline"].data)
    identifier = f"nmcd-cal-{model_id}-k{k_replays:02d}"
    _run_metadata(
        data,
        identifier,
        SKILL + " calibration",
        "study-sha256:" + study["sha256"],
        study["output_root"] / "calibration" / "runs",
    )
    data["gpu"].update(clocks)
    model = next(model for model in data["models"] if model["id"] == model_id)
    model["warmup_input"] = str(study["inputs"][model_id])
    replay_module = _model_replay_modules(study)[model_id]
    data["models"] = [
        _paired_model(
            model,
            f"calibration-{model_id}",
            "co_runner" if k_replays else "target",
            TRIALS,
            CALIBRATION_PERIOD_SECONDS,
            0,
            0,
            study["data"]["launch_tolerance_ms"] / 1000.0,
            k_replays,
            replay_module if k_replays else None,
        )
    ]
    return data


def _write_configs(study, records, directory):
    result = []
    for record, data in records:
        content = yaml.safe_dump(
            schema_v2_config(data), sort_keys=False).encode()
        path = directory / f"{record['candidate_id']}.yaml"
        _write_immutable(path, content)
        load_run_config(str(path), check_paths=False)
        result.append(
            {
                **record,
                "config_path": str(path),
                "config_sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return result


def _load_jsonl(path):
    try:
        with Path(path).open(encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"invalid JSONL evidence: {path}") from exc


def _tables(connection):
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(connection, table):
    return {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})")
    }


def _pid(global_id):
    return decode_global_id(int(global_id))[0] if global_id is not None else -1


def _nvtx_ranges(connection):
    if "NVTX_EVENTS" not in _tables(connection):
        return []
    result = []
    for start, end, global_tid, text in connection.execute(
        "SELECT start,end,globalTid,text FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND text LIKE 'closeloop:%'"
    ):
        try:
            tag = json.loads(str(text)[len("closeloop:"):])
        except json.JSONDecodeError:
            continue
        result.append(
            {
                "start": int(start),
                "end": int(end),
                "pid": _pid(global_tid),
                "tag": tag,
            }
        )
    return result


def _kernels(connection):
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in _tables(connection):
        return []
    return [
        {"start": int(start), "end": int(end), "pid": _pid(global_pid)}
        for start, end, global_pid in connection.execute(
            "SELECT start,end,globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
        if global_pid is not None
    ]


def _copy_kind(value, enum_names):
    name = str(enum_names.get(value, value)).upper().replace("_", "")
    if value == 1 or any(token in name for token in ("HTOD", "HOSTTODEVICE")):
        return "H2D"
    if value == 2 or any(token in name for token in ("DTOH", "DEVICETOHOST")):
        return "D2H"
    return None


def _memcopies(connection):
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if table not in _tables(connection):
        return []
    columns = _columns(connection, table)
    kind_column = next(
        (
            name
            for name in ("copyKind", "kind", "memcpyKind")
            if name in columns
        ),
        None,
    )
    bytes_column = next(
        (name for name in ("bytes", "size", "byteCount") if name in columns),
        None,
    )
    correlation = "correlationId" if "correlationId" in columns else None
    selected = ["start", "end", "globalPid"]
    selected += [correlation] if correlation else []
    selected += [kind_column] if kind_column else []
    selected += [bytes_column] if bytes_column else []
    enum_names = {}
    for enum_table in ("ENUM_CUDA_MEMCPY_OPER", "ENUM_CUDA_MEMCPY_KIND"):
        if enum_table in _tables(connection):
            enum_columns = _columns(connection, enum_table)
            name_column = "name" if "name" in enum_columns else "label"
            enum_names.update(
                dict(
                    connection.execute(
                        f"SELECT id,{name_column} FROM {enum_table}"
                    )
                )
            )
    result = []
    for row in connection.execute(f"SELECT {','.join(selected)} FROM {table}"):
        values = dict(zip(selected, row))
        kind = _copy_kind(values.get(kind_column), enum_names)
        if kind and values["globalPid"] is not None:
            result.append(
                {
                    "start": int(values["start"]),
                    "end": int(values["end"]),
                    "pid": _pid(values["globalPid"]),
                    "copy_class": kind,
                    "bytes": int(values.get(bytes_column) or 0),
                    "correlation": (
                        int(values[correlation])
                        if correlation and values[correlation] is not None
                        else None
                    ),
                }
            )
    return result


def _runtime_events(connection):
    required = {"CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"}
    if not required.issubset(_tables(connection)):
        return {}
    events = defaultdict(list)
    for start, end, global_tid, correlation, name in connection.execute(
        "SELECT r.start,r.end,r.globalTid,r.correlationId,s.value "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId ORDER BY r.start"
    ):
        events[int(global_tid)].append(
            {
                "start": int(start),
                "end": int(end),
                "pid": _pid(global_tid),
                "correlation": (
                    int(correlation) if correlation is not None else None
                ),
                "name": str(name).split("_v", 1)[0],
            }
        )
    return events


def _find_inference(ranges, pid, model_id, trial):
    matches = [
        item
        for item in ranges
        if item["pid"] == pid
        and item["tag"].get("event") == "inference"
        and item["tag"].get("model") == model_id
        and item["tag"].get("input") == f"paired-{trial}"
    ]
    if len(matches) != 1:
        raise StudyError(
            f"expected one inference range for {model_id} trial {trial}"
        )
    return matches[0]


def _in_range(items, interval, pid=None):
    return [
        item
        for item in items
        if (pid is None or item["pid"] == pid)
        and item["start"] >= interval["start"]
        and item["end"] <= interval["end"]
    ]


def _copy_trial(items, inference, pid):
    copies = _in_range(items, inference, pid)
    ordinals = defaultdict(int)
    result = []
    for item in sorted(
        copies, key=lambda value: (value["start"], value["end"])
    ):
        copy_class = item["copy_class"]
        ordinal = ordinals[copy_class]
        ordinals[copy_class] += 1
        result.append(
            {
                "copy_class": copy_class,
                "bytes": item["bytes"],
                "ordinal": ordinal,
                "start_ns": item["start"] - inference["start"],
                "end_ns": item["end"] - inference["start"],
            }
        )
    return result


def _calibration_trace(run_directory, model_id, k_replays):
    status = json.loads(
        (Path(run_directory) / f"model_{model_id}.json").read_text()
    )
    if status.get("state") != "acknowledged":
        raise StudyError(f"calibration {model_id} K={k_replays} failed")
    pid = int(status["pid"])
    with sqlite3.connect(str(Path(run_directory) / "profile.sqlite")) as db:
        ranges = _nvtx_ranges(db)
        kernels = _kernels(db)
        memcopies = _memcopies(db)
        owner, owner_errors = _context_owner(db, pid, kernels)
    if owner is None or owner_errors:
        raise StudyError("; ".join(owner_errors))
    active, _transitions, context_errors, _timestamps = (
        _active_context_intervals(owner[1])
    )
    if context_errors:
        raise StudyError("; ".join(context_errors))
    trials = []
    for trial in range(TRIALS):
        inference = _find_inference(ranges, pid, model_id, trial)
        replay = [
            item
            for item in ranges
            if item["pid"] == pid
            and item["tag"].get("event") == "compute_service_replay"
            and item["tag"].get("input") == f"paired-{trial}"
        ]
        replay_copies = [
            item
            for item in memcopies
            if item["pid"] == pid
            and any(
                item["start"] < region["end"] and item["end"] > region["start"]
                for region in replay
            )
        ]
        if replay_copies:
            raise StudyError("H2D/D2H operation observed inside replay range")
        inference_kernels = _in_range(kernels, inference, pid)
        if not inference_kernels:
            raise StudyError("calibration inference has no kernels")
        replay_kernels = [
            item
            for item in kernels
            if item["pid"] == pid
            and any(
                item["start"] < region["end"] and item["end"] > region["start"]
                for region in replay
            )
        ]
        envelope_end = max(
            [inference["end"]] + [item["end"] for item in replay]
        )
        trials.append(
            {
                "copies": _copy_trial(memcopies, inference, pid),
                "active_context": [
                    {
                        "start_ns": max(left, inference["start"])
                        - inference["start"],
                        "end_ns": min(right, envelope_end)
                        - inference["start"],
                    }
                    for left, right in active
                    if left < envelope_end and right > inference["start"]
                ],
                "execution_envelope_ns": envelope_end - inference["start"],
                "kernel_latency_ns": (
                    max(item["end"] for item in inference_kernels)
                    - min(item["start"] for item in inference_kernels)
                ),
                "kernel_time_ns": _union_duration(
                    [
                        (item["start"], item["end"])
                        for item in inference_kernels
                    ]
                ),
                "subgraph_kernel_time_ns": _union_duration(
                    [(item["start"], item["end"]) for item in replay_kernels]
                ),
            }
        )
    return trials


def _completed_run(config, record, extra_artifacts=()):
    if not config.run_directory.exists():
        return False
    required = [
        config.run_directory / name
        for name in (
            "config.yaml",
            "run_manifest.json",
            "testbed_result.json",
            "profile.nsys-rep",
            "profile.sqlite",
            "runner.log",
        )
    ]
    required.extend(
        config.run_directory / f"model_{model['id']}.json"
        for model in config.models
    )
    required.extend(
        config.run_directory / f"model_{model['id']}_paired.jsonl"
        for model in config.models
    )
    required.extend(extra_artifacts)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise StudyError(
            f"incomplete existing run {record['candidate_id']}: "
            + ", ".join(path.name for path in missing)
        )
    manifest = json.loads(
        (config.run_directory / "run_manifest.json").read_text()
    )
    if (
        manifest.get("state") != "success"
        or manifest.get("config_sha256") != record["config_sha256"]
        or _sha256(config.run_directory / "config.yaml")
        != record["config_sha256"]
    ):
        raise StudyError(
            "existing run does not match immutable config: "
            + record["candidate_id"]
        )
    evidence_path = config.run_directory / "campaign_evidence.json"
    if evidence_path in required:
        evidence = json.loads(evidence_path.read_text())
        observed = {
            path.name: _sha256(path)
            for path in required
            if path != evidence_path
        }
        if evidence.get("artifact_sha256") != observed:
            raise StudyError(
                f"existing artifacts changed: {record['candidate_id']}"
            )
    return True


def calibrate_study(
    study, clocks=None, dry_run=False, run_command=subprocess.run
):
    """Run fixed-input solo/subgraph calibration and select offsets."""
    clocks = clocks or maximum_supported_clocks(
        study["baseline"].data["gpu"]["index"], run_command
    )
    resolved_clocks = {
        name: clocks[name]
        for name in ("graphics_clock_mhz", "memory_clock_mhz")
    }
    records = []
    for model_id in sorted(study["inputs"]):
        for k_replays in K_VALUES:
            identifier = f"nmcd-cal-{model_id}-k{k_replays:02d}"
            records.append(
                (
                    {
                        "candidate_id": identifier,
                        "model_id": model_id,
                        "k_replays": k_replays,
                    },
                    _calibration_config(
                        study, model_id, k_replays, resolved_clocks
                    ),
                )
            )
    candidates = _write_configs(
        study,
        records,
        study["output_root"] / "calibration" / "generated_configs",
    )
    candidate_manifest = {
        "schema_version": 1,
        "skill": SKILL,
        "alignment_metric": ALIGNMENT_METRIC,
        "alignment_window_ns": round(
            study["data"]["alignment_window_ms"] * 1_000_000
        ),
        "study_sha256": study["sha256"],
        "baseline_config_sha256": study["baseline"].sha256,
        "resolved_clocks": resolved_clocks,
        "candidates": candidates,
    }
    _write_immutable(
        study["output_root"] / "calibration_candidate_manifest.json",
        _json_bytes(candidate_manifest),
    )
    host = validate_non_mps_host(study, run_command)
    control = GPUClockLock(
        study["baseline"].data["gpu"]["index"],
        resolved_clocks["graphics_clock_mhz"],
        resolved_clocks["memory_clock_mhz"],
        run_command,
    )
    error = None
    completed = resumed = 0
    try:
        if dry_run:
            for candidate in candidates:
                config = load_run_config(
                    candidate["config_path"],
                    artifact_root=str(study["output_root"]),
                )
                if not config.run_directory.exists():
                    ExperimentRunner(config, run_command=run_command).run(True)
            return {"dry_run": True, "candidate_count": len(candidates)}
        with control:
            for candidate in candidates:
                config = load_run_config(
                    candidate["config_path"],
                    artifact_root=str(study["output_root"]),
                )
                evidence_path = (
                    config.run_directory / "calibration_evidence.json"
                )
                if _completed_run(config, candidate, (evidence_path,)):
                    resumed += 1
                    continue
                ExperimentRunner(config, run_command=run_command).run()
                _write_json(
                    evidence_path,
                    {
                        "candidate_id": candidate["candidate_id"],
                        "config_sha256": candidate["config_sha256"],
                        "artifact_sha256": {
                            path.name: _sha256(path)
                            for path in _run_artifacts(config)
                        },
                    },
                )
                completed += 1
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        if not dry_run:
            _write_json(
                study["output_root"] / "calibration_clock_control.json",
                {
                    "host_preflight": host,
                    "resolved_clocks": resolved_clocks,
                    "clock_control": control.evidence,
                    "completed_candidates": completed,
                    "resumed_candidates": resumed,
                    "error": error,
                },
            )
    models = {}
    alignment_window_ns = round(
        study["data"]["alignment_window_ms"] * 1_000_000
    )
    for model_id in sorted(study["inputs"]):
        by_k = {}
        canonical = None
        for k_replays in K_VALUES:
            candidate = next(
                item
                for item in candidates
                if item["model_id"] == model_id
                and item["k_replays"] == k_replays
            )
            config = load_run_config(
                candidate["config_path"], check_paths=False,
                artifact_root=str(study["output_root"]),
            )
            trials = _calibration_trace(
                config.run_directory, model_id, k_replays
            )
            timeline = canonical_copy_timeline(
                [trial["copies"] for trial in trials]
            )
            context_timeline = canonical_context_timeline(
                [trial["active_context"] for trial in trials]
            )
            signature = [
                (item["copy_class"], item["bytes"], item["ordinal"])
                for item in timeline
            ]
            if canonical is None:
                canonical = signature
            elif signature != canonical:
                raise StudyError(
                    f"{model_id} H2D/D2H signature changed at K={k_replays}"
                )
            by_k[str(k_replays)] = {
                "median_execution_envelope_ns": int(
                    statistics.median(
                        trial["execution_envelope_ns"] for trial in trials
                    )
                ),
                "worst_execution_envelope_ns": max(
                    trial["execution_envelope_ns"] for trial in trials
                ),
                "median_kernel_time_ns": int(
                    statistics.median(
                        trial["kernel_time_ns"] for trial in trials
                    )
                ),
                "median_kernel_latency_ns": int(
                    statistics.median(
                        trial["kernel_latency_ns"] for trial in trials
                    )
                ),
                "median_subgraph_kernel_time_ns": int(
                    statistics.median(
                        trial["subgraph_kernel_time_ns"] for trial in trials
                    )
                ),
                "canonical_context_timeline": context_timeline,
            }
            if k_replays == 0:
                by_k["0"]["canonical_copy_timeline"] = timeline
        models[model_id] = by_k
    directions = {}
    for direction_id, direction in study["directions"].items():
        target = models[direction["target_model_id"]]
        co_runner = models[direction["co_runner_model_id"]]
        offsets, matched = select_offsets(
            target["0"]["canonical_copy_timeline"],
            co_runner["0"]["canonical_context_timeline"],
            target["0"]["median_execution_envelope_ns"],
            co_runner["0"]["median_execution_envelope_ns"],
            co_runner["8"]["worst_execution_envelope_ns"],
            alignment_window_ns,
            overlap_tolerance=study["data"]["matched_overlap_tolerance"],
            alignment_difference_min=study["data"][
                "alignment_difference_min"
            ],
        )
        directions[direction_id] = {
            **direction,
            "offsets": offsets,
            "matched_pair": matched,
        }
    trial_period_ns = calibrated_trial_period_ns(
        models,
        directions.values(),
        study["data"]["period_guard_fraction"],
    )
    manifest = {
        "schema_version": 1,
        "skill": SKILL,
        "alignment_metric": ALIGNMENT_METRIC,
        "alignment_window_ns": alignment_window_ns,
        "study_sha256": study["sha256"],
        "baseline_config_sha256": study["baseline"].sha256,
        "resolved_clocks": resolved_clocks,
        "fixed_input_sha256": {
            model_id: _sha256(path)
            for model_id, path in study["inputs"].items()
        },
        "trial_period_ns": trial_period_ns,
        "models": models,
        "directions": directions,
    }
    _write_immutable(
        study["output_root"] / "calibration_manifest.json",
        _json_bytes(manifest),
    )
    return manifest


def load_calibration_manifest(study):
    """Load calibration only when immutable study inputs still match."""
    path = study["output_root"] / "calibration_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError("calibration manifest is missing or invalid") from exc
    if (
        manifest.get("skill") != SKILL
        or manifest.get("alignment_metric") != ALIGNMENT_METRIC
        or manifest.get("alignment_window_ns")
        != round(study["data"]["alignment_window_ms"] * 1_000_000)
        or manifest.get("study_sha256") != study["sha256"]
        or manifest.get("baseline_config_sha256") != study["baseline"].sha256
    ):
        raise StudyError("calibration manifest does not match this study")
    observed = {
        model_id: _sha256(path) for model_id, path in study["inputs"].items()
    }
    if manifest.get("fixed_input_sha256") != observed:
        raise StudyError("fixed calibration inputs changed")
    return manifest


class NonMpsMemcpyComputeDemandV2:
    """Generate the randomized bidirectional 100-cell campaign."""

    def __init__(self, study):
        self.study = study

    def candidates(self, calibration=None):
        """Return stable cells in one reproducibly randomized order."""
        calibration = calibration or load_calibration_manifest(self.study)
        cells = []
        for direction_id, direction in calibration["directions"].items():
            for offset in direction["offsets"]:
                for k_replays in K_VALUES:
                    cells.append(
                        {
                            "candidate_id": candidate_id(
                                direction_id, offset["offset_index"], k_replays
                            ),
                            "direction_id": direction_id,
                            "target_model_id": direction["target_model_id"],
                            "co_runner_model_id": direction[
                                "co_runner_model_id"
                            ],
                            "offset_index": offset["offset_index"],
                            "offset_label": offset["label"],
                            "configured_delta_ns": offset["delta_ns"],
                            "predicted_execution_overlap": offset[
                                "predicted_execution_overlap"
                            ],
                            "predicted_alignment": offset[
                                "predicted_alignment"
                            ],
                            "k_replays": k_replays,
                        }
                    )
        random.Random(self.study["data"]["random_seed"]).shuffle(cells)
        for position, cell in enumerate(cells):
            cell["randomized_position"] = position
        return cells

    def _config(self, cell, calibration, fixed_inputs):
        data = copy.deepcopy(self.study["baseline"].data)
        _run_metadata(
            data,
            cell["candidate_id"],
            SKILL,
            "study-sha256:" + self.study["sha256"],
            self.study["output_root"] / "runs",
            f"nmcd-cal-{cell['target_model_id']}-k00",
        )
        data["gpu"].update(calibration["resolved_clocks"])
        delta = cell["configured_delta_ns"] / 1_000_000_000
        anchor = max(0.0, -delta)
        period = calibration["trial_period_ns"] / 1_000_000_000
        tolerance = self.study["data"]["launch_tolerance_ms"] / 1000.0
        direction = calibration["directions"][cell["direction_id"]]
        models = []
        for model in data["models"]:
            model["warmup_input"] = str(fixed_inputs[model["id"]])
            if model["id"] == cell["target_model_id"]:
                models.append(
                    _paired_model(
                        model,
                        cell["direction_id"],
                        "target",
                        TRIALS,
                        period,
                        anchor,
                        anchor,
                        tolerance,
                        0,
                    )
                )
            else:
                models.append(
                    _paired_model(
                        model,
                        cell["direction_id"],
                        "co_runner",
                        TRIALS,
                        period,
                        anchor,
                        anchor + delta,
                        tolerance,
                        cell["k_replays"],
                        direction["replay_module"],
                    )
                )
        data["models"] = models
        return data

    def generate(self, calibration=None):
        """Write immutable fixed inputs, configs, and candidate manifest."""
        calibration = calibration or load_calibration_manifest(self.study)
        fixed_inputs = {}
        for model_id, source in self.study["inputs"].items():
            destination = (
                self.study["output_root"]
                / "fixed_inputs"
                / (model_id + source.suffix)
            )
            _write_immutable(destination, source.read_bytes())
            fixed_inputs[model_id] = destination
        records = [
            (cell, self._config(cell, calibration, fixed_inputs))
            for cell in self.candidates(calibration)
        ]
        candidates = _write_configs(
            self.study,
            records,
            self.study["output_root"] / "generated_configs",
        )
        manifest = {
            "schema_version": 1,
            "skill": SKILL,
            "alignment_metric": ALIGNMENT_METRIC,
            "alignment_window_ns": calibration["alignment_window_ns"],
            "study_path": str(self.study["source"]),
            "study_sha256": self.study["sha256"],
            "baseline_config_sha256": self.study["baseline"].sha256,
            "calibration_manifest_sha256": _sha256(
                self.study["output_root"] / "calibration_manifest.json"
            ),
            "resolved_clocks": calibration["resolved_clocks"],
            "trial_period_ns": calibration["trial_period_ns"],
            "fixed_input_sha256": calibration["fixed_input_sha256"],
            "fixed_input_paths": {
                model_id: str(path) for model_id, path in fixed_inputs.items()
            },
            "random_seed": self.study["data"]["random_seed"],
            "directions": calibration["directions"],
            "candidates": candidates,
        }
        if (
            len(candidates) != 100
            or len({item["candidate_id"] for item in candidates}) != 100
        ):
            raise StudyError("campaign grid must contain 100 unique cells")
        _write_immutable(
            self.study["output_root"] / "candidate_manifest.json",
            _json_bytes(manifest),
        )
        return manifest


def load_candidate_manifest(study):
    """Verify study, calibration, configs, and fixed inputs for restart."""
    path = study["output_root"] / "candidate_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError("candidate manifest is missing or invalid") from exc
    calibration_path = study["output_root"] / "calibration_manifest.json"
    if (
        manifest.get("skill") != SKILL
        or manifest.get("alignment_metric") != ALIGNMENT_METRIC
        or manifest.get("alignment_window_ns")
        != round(study["data"]["alignment_window_ms"] * 1_000_000)
        or manifest.get("study_sha256") != study["sha256"]
        or manifest.get("baseline_config_sha256") != study["baseline"].sha256
        or manifest.get("calibration_manifest_sha256")
        != _sha256(calibration_path)
        or len(manifest.get("candidates", [])) != 100
    ):
        raise StudyError("candidate manifest does not match immutable inputs")
    for candidate in manifest["candidates"]:
        if _sha256(candidate["config_path"]) != candidate["config_sha256"]:
            raise StudyError(
                f"generated config changed: {candidate['candidate_id']}"
            )
    if manifest["fixed_input_sha256"] != {
        model_id: _sha256(path) for model_id, path in study["inputs"].items()
    }:
        raise StudyError("campaign fixed inputs changed")
    generated_inputs = manifest.get("fixed_input_paths", {})
    if set(generated_inputs) != set(manifest["fixed_input_sha256"]) or any(
        _sha256(generated_inputs[model_id]) != expected
        for model_id, expected in manifest["fixed_input_sha256"].items()
    ):
        raise StudyError("generated immutable fixed inputs changed")
    return manifest


def _run_artifacts(config):
    paths = [
        config.run_directory / name
        for name in (
            "config.yaml",
            "run_manifest.json",
            "testbed_result.json",
            "profile.nsys-rep",
            "profile.sqlite",
            "runner.log",
        )
    ]
    for model in config.models:
        paths.extend(
            (
                config.run_directory / f"model_{model['id']}.json",
                config.run_directory / f"model_{model['id']}_paired.jsonl",
            )
        )
    return paths


def _verify_run_evidence(config, manifest, candidate, observed_clocks):
    errors = []
    statuses = {}
    tolerance = candidate.get("launch_tolerance_seconds")
    if tolerance is None:
        tolerance = config.models[0]["paired_trial"][
            "launch_tolerance_seconds"
        ]
    for model in config.models:
        status = json.loads(
            (config.run_directory / f"model_{model['id']}.json").read_text()
        )
        statuses[model["id"]] = status
        if status.get("state") != "acknowledged":
            errors.append(f"model {model['id']} did not acknowledge")
        if any(
            status.get("process_observed_mps_environment", {}).get(name)
            is not None
            for name in MPS_ENVIRONMENT_VARIABLES
        ):
            errors.append(f"model {model['id']} observed MPS configuration")
        if status.get("completed_trials") != TRIALS:
            errors.append(f"model {model['id']} did not complete five trials")
        if status.get("missed_deadlines") != 0:
            errors.append(f"model {model['id']} missed a deadline")
        if status.get("quiescence_verified") is not True:
            errors.append(f"model {model['id']} lacks quiescence evidence")
        if status.get("max_abs_launch_error_seconds", math.inf) > tolerance:
            errors.append(f"model {model['id']} exceeded launch tolerance")
        expected_hash = manifest["fixed_input_sha256"][model["id"]]
        if status.get("fixed_input_sha256") != expected_hash:
            errors.append(f"model {model['id']} fixed input hash differs")
        if status.get("fixed_input_reused") is not True:
            errors.append(f"model {model['id']} fixed input was not reused")
        trial_records = _load_jsonl(
            config.run_directory / f"model_{model['id']}_paired.jsonl"
        )
        if len(trial_records) != TRIALS or not all(
            item.get("quiescence_verified") for item in trial_records
        ):
            errors.append(f"model {model['id']} paired log is incomplete")
    co_status = statuses[candidate["co_runner_model_id"]]
    if candidate["k_replays"]:
        if not co_status.get("resident_input_sha256"):
            errors.append("co-runner resident input hash is absent")
        if co_status.get("resident_input_reused") is not True:
            errors.append("co-runner resident input was not reused")
        if co_status.get("subgraph_kernel_time_ns", 0) <= 0:
            errors.append("co-runner added kernel-service time is absent")
    evidence = {
        "candidate_id": candidate["candidate_id"],
        "config_sha256": candidate["config_sha256"],
        "idle_clock_sample": observed_clocks,
        "non_mps_verified": not errors,
        "errors": errors,
        "artifact_sha256": {
            path.name: _sha256(path) for path in _run_artifacts(config)
        },
    }
    _write_json(config.run_directory / "campaign_evidence.json", evidence)
    if errors:
        raise StudyError("; ".join(errors))


def _completed_candidate(config, candidate):
    return _completed_run(
        config, candidate, (config.run_directory / "campaign_evidence.json",)
    )


def _gate_ids(manifest):
    selected = set()
    for direction_id, direction in manifest["directions"].items():
        offsets = {
            item["label"]: item["offset_index"]
            for item in direction["offsets"]
        }
        nonoverlap = offsets["nonoverlap_left"]
        maximum = offsets["max_alignment"]
        for offset_index in (nonoverlap, maximum):
            for k_replays in (0, 8):
                selected.add(
                    candidate_id(direction_id, offset_index, k_replays)
                )
    return selected


def run_campaign(
    study, selected=None, gate=False, dry_run=False, run_command=subprocess.run
):
    """Run immutable cells under one strict maximum-clock lock."""
    manifest = load_candidate_manifest(study)
    host = validate_non_mps_host(study, run_command)
    requested = set(selected or ())
    if gate:
        requested.update(_gate_ids(manifest))
    known = {candidate["candidate_id"] for candidate in manifest["candidates"]}
    if requested.difference(known):
        raise StudyError("unknown candidate selection")
    candidates = [
        candidate
        for candidate in manifest["candidates"]
        if not requested or candidate["candidate_id"] in requested
    ]
    if dry_run:
        for candidate in candidates:
            config = load_run_config(
                candidate["config_path"],
                artifact_root=str(study["output_root"]),
            )
            if not config.run_directory.exists():
                ExperimentRunner(config, run_command=run_command).run(True)
        return {"dry_run": True, "candidate_count": len(candidates)}
    clocks = manifest["resolved_clocks"]
    control = GPUClockLock(
        study["baseline"].data["gpu"]["index"],
        clocks["graphics_clock_mhz"],
        clocks["memory_clock_mhz"],
        run_command,
    )
    error = None
    completed = resumed = 0
    try:
        with control:
            for candidate in candidates:
                config = load_run_config(
                    candidate["config_path"],
                    artifact_root=str(study["output_root"]),
                )
                if _completed_candidate(config, candidate):
                    resumed += 1
                    continue
                observed = current_gpu_clocks(
                    config.data["gpu"]["index"], run_command
                )
                ExperimentRunner(config, run_command=run_command).run()
                _verify_run_evidence(config, manifest, candidate, observed)
                completed += 1
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        _write_json(
            study["output_root"] / "campaign_clock_control.json",
            {
                "host_preflight": host,
                "resolved_clocks": clocks,
                "clock_control": control.evidence,
                "completed_candidates": completed,
                "resumed_candidates": resumed,
                "error": error,
            },
        )
    return {"completed": completed, "resumed": resumed}


def _status_map(run_directory):
    result = {}
    for path in Path(run_directory).glob("model_*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            result[status["model_id"]] = status
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return result


def _context_owner(connection, process_pid, kernels):
    required = {"GPU_CONTEXT_SWITCH_EVENTS", "ENUM_GPU_CTX_SWITCH"}
    if not required.issubset(_tables(connection)):
        return None, ["GPU context-switch events absent"]
    rows = connection.execute(
        "SELECT e.timestamp,e.contextId,e.globalPid,t.name "
        "FROM GPU_CONTEXT_SWITCH_EVENTS e "
        "JOIN ENUM_GPU_CTX_SWITCH t ON t.id=e.tag "
        "ORDER BY e.timestamp,e.seqNo"
    ).fetchall()
    by_owner = defaultdict(list)
    for timestamp, context_id, global_pid, tag in rows:
        if global_pid is not None:
            by_owner[int(global_pid)].append(
                (int(timestamp), int(context_id), str(tag))
            )
    direct = [owner for owner in by_owner if _pid(owner) == process_pid]
    if len(direct) == 1:
        return (direct[0], by_owner[direct[0]]), []
    starts = [item["start"] for item in kernels if item["pid"] == process_pid]
    scores = []
    for owner, events in by_owner.items():
        active = _active_context_intervals(events)[0]
        active_starts = [left for left, _right in active]

        def is_active(start):
            index = bisect_right(active_starts, start) - 1
            return index >= 0 and start < active[index][1]

        hits = sum(is_active(start) for start in starts)
        scores.append((hits, owner))
    scores.sort(reverse=True)
    if not scores or scores[0][0] == 0:
        return None, ["no Docker PID-namespace context attribution"]
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        return None, ["ambiguous Docker PID-namespace context attribution"]
    return (scores[0][1], by_owner[scores[0][1]]), []


def _active_context_intervals(events):
    active_starts = {}
    restore_transitions = {}
    save_transitions = {}
    active = []
    transitions = []
    errors = []
    switch_timestamps = []
    for timestamp, context_id, tag in events:
        if tag == "RESTORE_START":
            if context_id in active_starts:
                errors.append(f"duplicate RESTORE_START for {context_id}")
            active_starts[context_id] = timestamp
            restore_transitions[context_id] = timestamp
        elif tag == "RESTORE_END":
            start = restore_transitions.pop(context_id, None)
            if start is not None and timestamp >= start:
                transitions.append((start, timestamp))
        elif tag == "SAVE_START":
            save_transitions[context_id] = timestamp
            switch_timestamps.append(timestamp)
        elif tag == "SAVE_END":
            start = active_starts.pop(context_id, None)
            if start is None or timestamp <= start:
                errors.append(f"unpaired SAVE_END for {context_id}")
            else:
                active.append((start, timestamp))
            transition = save_transitions.pop(context_id, None)
            if transition is not None and timestamp >= transition:
                transitions.append((transition, timestamp))
    errors.extend(
        f"unpaired RESTORE_START for {key}" for key in sorted(active_starts)
    )
    return (
        _merge_intervals(active),
        _merge_intervals(transitions),
        errors,
        switch_timestamps,
    )


def _complement(intervals, start, end):
    result = []
    cursor = start
    for left, right in _merge_intervals(
        [
            (max(start, left), min(end, right))
            for left, right in intervals
            if left < end and right > start
        ]
    ):
        if left > cursor:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return result


def _sync_wait(memcopies, runtimes, inference):
    correlations = {
        item["correlation"]
        for item in memcopies
        if item["correlation"] is not None
    }
    seen = set()
    waiting = 0
    for events in runtimes.values():
        for index, event in enumerate(events[:-1]):
            if (
                event["pid"] != inference["pid"]
                or event["correlation"] not in correlations
                or "Memcpy" not in event["name"]
                or not inference["start"] <= event["start"] < inference["end"]
            ):
                continue
            following = events[index + 1]
            if (
                following["name"]
                in ("cudaStreamSynchronize", "cuStreamSynchronize")
                and following["end"] <= inference["end"]
            ):
                key = (following["start"], following["end"])
                if key not in seen:
                    waiting += following["end"] - following["start"]
                    seen.add(key)
    return waiting


def _trace_run(run_directory, candidate, direction, calibration):
    """Extract all five preregistered trial metrics from one Nsight trace."""
    statuses = _status_map(run_directory)
    target_id = candidate["target_model_id"]
    co_id = candidate["co_runner_model_id"]
    errors = []
    if target_id not in statuses or co_id not in statuses:
        return [], {"errors": ["model status is missing"], "observable": False}
    for model_id in (target_id, co_id):
        status = statuses[model_id]
        if status.get("state") != "acknowledged":
            errors.append(f"{model_id} did not acknowledge")
        if any(
            status.get("process_observed_mps_environment", {}).get(name)
            is not None
            for name in MPS_ENVIRONMENT_VARIABLES
        ):
            errors.append(f"{model_id} observed MPS configuration")
    target_pid, co_pid = int(statuses[target_id]["pid"]), int(
        statuses[co_id]["pid"]
    )
    target_log = {
        int(item["trial"]): item
        for item in _load_jsonl(
            Path(run_directory) / f"model_{target_id}_paired.jsonl"
        )
    }
    co_log = {
        int(item["trial"]): item
        for item in _load_jsonl(
            Path(run_directory) / f"model_{co_id}_paired.jsonl"
        )
    }
    with sqlite3.connect(str(Path(run_directory) / "profile.sqlite")) as db:
        ranges = _nvtx_ranges(db)
        kernels = _kernels(db)
        memcopies = _memcopies(db)
        runtimes = _runtime_events(db)
        target_owner, target_owner_errors = _context_owner(
            db, target_pid, kernels
        )
        co_owner, co_owner_errors = _context_owner(db, co_pid, kernels)
    errors.extend(f"target {error}" for error in target_owner_errors)
    errors.extend(f"co-runner {error}" for error in co_owner_errors)
    target_active = transitions = switch_timestamps = []
    co_active = []
    observable = target_owner is not None and co_owner is not None
    if target_owner is not None:
        target_active, transitions, context_errors, switch_timestamps = (
            _active_context_intervals(target_owner[1])
        )
        errors.extend(f"target {error}" for error in context_errors)
    if co_owner is not None:
        co_active, _co_transitions, context_errors, _co_timestamps = (
            _active_context_intervals(co_owner[1])
        )
        errors.extend(f"co-runner {error}" for error in context_errors)
    target_timeline = calibration["models"][target_id]["0"][
        "canonical_copy_timeline"
    ]
    co_context_timeline = calibration["models"][co_id][
        str(candidate["k_replays"])
    ]["canonical_context_timeline"]
    alignment_window_ns = calibration["alignment_window_ns"]
    rows = []
    for trial in range(TRIALS):
        try:
            target_range = _find_inference(
                ranges, target_pid, target_id, trial
            )
            co_range = _find_inference(ranges, co_pid, co_id, trial)
            target_record, co_record = target_log[trial], co_log[trial]
        except (KeyError, StudyError) as exc:
            errors.append(str(exc))
            continue
        target_kernels = _in_range(kernels, target_range, target_pid)
        target_copies = _in_range(memcopies, target_range, target_pid)
        replay_ranges = [
            item
            for item in ranges
            if item["pid"] == co_pid
            and item["tag"].get("event") == "compute_service_replay"
            and item["tag"].get("input") == f"paired-{trial}"
        ]
        replay_intervals = [
            (item["start"], item["end"]) for item in replay_ranges
        ]
        replay_copies = [
            item
            for item in memcopies
            if item["pid"] == co_pid
            and any(
                item["start"] < right and item["end"] > left
                for left, right in replay_intervals
            )
        ]
        if not target_kernels:
            errors.append(f"target trial {trial} has no kernels")
            latency = None
        else:
            latency = max(item["end"] for item in target_kernels) - min(
                item["start"] for item in target_kernels
            )
        actual_delta_ns = round(
            (co_record["actual_time"] - target_record["actual_time"])
            * 1_000_000_000
        )
        inactive = (
            _complement(
                target_active, target_range["start"], target_range["end"]
            )
            if observable
            else []
        )
        target_interval = [(target_range["start"], target_range["end"])]
        co_interval = [(co_range["start"], co_range["end"])]
        signature = _copy_trial(memcopies, target_range, target_pid)
        signatures = {
            copy_class: [
                item["bytes"]
                for item in signature
                if item["copy_class"] == copy_class
            ]
            for copy_class in ("H2D", "D2H")
        }
        rows.append(
            {
                **{
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "direction_id",
                        "target_model_id",
                        "co_runner_model_id",
                        "offset_index",
                        "offset_label",
                        "configured_delta_ns",
                        "k_replays",
                    )
                },
                "trial": trial,
                "valid": not replay_copies and latency is not None,
                "target_kernel_latency_ns": latency,
                "configured_alignment": alignment_score(
                    target_timeline,
                    co_context_timeline,
                    candidate["configured_delta_ns"],
                    alignment_window_ns,
                ),
                "actual_delta_alignment": alignment_score(
                    target_timeline,
                    co_context_timeline,
                    actual_delta_ns,
                    alignment_window_ns,
                ),
                "actual_alignment": (
                    alignment_score(
                        [
                            {
                                "copy_class": item["copy_class"],
                                "start_ns": item["start"],
                            }
                            for item in target_copies
                        ],
                        [
                            {"start_ns": left, "end_ns": right}
                            for left, right in co_active
                        ],
                        0,
                        alignment_window_ns,
                    )
                    if observable
                    else None
                ),
                "alignment_metric": ALIGNMENT_METRIC,
                "alignment_window_ns": alignment_window_ns,
                "alignment_operation_count": len(target_copies),
                "memcpy_adjacent_waiting_ns": _sync_wait(
                    target_copies, runtimes, target_range
                ),
                "h2d_signature": json.dumps(signatures["H2D"]),
                "d2h_signature": json.dumps(signatures["D2H"]),
                "inference_overlap_ns": _intersection_duration(
                    target_interval, co_interval
                ),
                "normalized_inference_overlap": _intersection_duration(
                    target_interval, co_interval
                )
                / (target_range["end"] - target_range["start"]),
                "target_context_switch_count": sum(
                    target_range["start"] <= timestamp <= target_range["end"]
                    for timestamp in switch_timestamps
                ),
                "target_inactive_ns": _union_duration(inactive),
                "target_switch_transition_ns": _union_duration(
                    transitions, target_range["start"], target_range["end"]
                ),
                "compute_related_descheduling_ns": _intersection_duration(
                    inactive, replay_intervals
                ),
                "actual_delta_ns": actual_delta_ns,
                "target_launch_error_ns": round(
                    target_record["launch_error_seconds"] * 1_000_000_000
                ),
                "co_runner_launch_error_ns": round(
                    co_record["launch_error_seconds"] * 1_000_000_000
                ),
                "subgraph_kernel_time_ns": _intersection_duration(
                    [
                        (item["start"], item["end"])
                        for item in kernels
                        if item["pid"] == co_pid
                    ],
                    replay_intervals,
                ),
                "replay_memcpy_count": len(replay_copies),
                "target_fixed_input_sha256": statuses[target_id].get(
                    "fixed_input_sha256"
                ),
                "co_runner_fixed_input_sha256": statuses[co_id].get(
                    "fixed_input_sha256"
                ),
                "resident_input_sha256": statuses[co_id].get(
                    "resident_input_sha256"
                ),
                "target_quiescence_verified": target_record.get(
                    "quiescence_verified"
                ),
                "co_runner_quiescence_verified": co_record.get(
                    "quiescence_verified"
                ),
                "context_observable": observable,
            }
        )
    expected_target_signature = {
        copy_class: [
            item["bytes"]
            for item in target_timeline
            if item["copy_class"] == copy_class
        ]
        for copy_class in ("H2D", "D2H")
    }
    for row in rows:
        if (
            json.loads(row["h2d_signature"])
            != expected_target_signature["H2D"]
            or json.loads(row["d2h_signature"])
            != expected_target_signature["D2H"]
        ):
            row["valid"] = False
            errors.append(
                f"target copy signature changed in trial {row['trial']}"
            )
        if row["replay_memcpy_count"]:
            errors.append(
                f"replay range contains memcpy in trial {row['trial']}"
            )
    return rows, {
        "errors": sorted(set(errors)),
        "observable": observable,
        "trial_count": len(rows),
    }


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _rank(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            result[position] = rank
        index = end
    return result


def _spearman(left, right):
    if len(left) < 3 or len(left) != len(right):
        return None
    left, right = _rank(left), _rank(right)
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _write_csv(path, rows):
    path = Path(path)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        writer.writerows(rows)


def _clock_evidence_errors(study, manifest):
    errors = []
    try:
        record = json.loads(
            (study["output_root"] / "campaign_clock_control.json").read_text()
        )
        control = record["clock_control"]
        if record.get("resolved_clocks") != manifest["resolved_clocks"]:
            errors.append("campaign clock resolution differs")
        if control.get("requested") != manifest["resolved_clocks"]:
            errors.append("requested campaign clocks differ")
        for action in ("lock_commands", "reset_commands"):
            commands = control.get(action, [])
            if len(commands) != 2 or any(
                command.get("returncode") != 0 for command in commands
            ):
                errors.append(f"incomplete successful {action}")
        if record.get("error") is not None:
            errors.append("campaign execution recorded an error")
    except (OSError, KeyError, json.JSONDecodeError):
        errors.append("campaign clock-control evidence is missing")
    return errors


def analyze_campaign(study):
    """Write trial/run summaries and descriptive mechanism status."""
    manifest = load_candidate_manifest(study)
    calibration = load_calibration_manifest(study)
    campaign_errors = _clock_evidence_errors(study, manifest)
    trials = []
    runs = []
    for candidate in manifest["candidates"]:
        config = load_run_config(
            candidate["config_path"], check_paths=False,
            artifact_root=str(study["output_root"]),
        )
        if not _completed_candidate(config, candidate):
            raise StudyError(
                f"candidate is not complete: {candidate['candidate_id']}"
            )
        rows, trace = _trace_run(
            config.run_directory,
            candidate,
            manifest["directions"][candidate["direction_id"]],
            calibration,
        )
        evidence = json.loads(
            (config.run_directory / "campaign_evidence.json").read_text()
        )
        errors = campaign_errors + trace["errors"] + evidence.get("errors", [])
        valid = (
            not errors
            and trace["trial_count"] == TRIALS
            and all(row["valid"] for row in rows)
        )
        for row in rows:
            row["run_valid"] = valid
            row["validity_errors"] = "; ".join(errors)
        trials.extend(rows)
        runs.append(
            {
                **{
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "direction_id",
                        "target_model_id",
                        "co_runner_model_id",
                        "offset_index",
                        "offset_label",
                        "configured_delta_ns",
                        "predicted_execution_overlap",
                        "predicted_alignment",
                        "k_replays",
                    )
                },
                "valid": valid,
                "observable": trace["observable"],
                "validity_errors": "; ".join(errors),
                "trial_count": len(rows),
                "median_actual_delta_ns": _median(
                    [row["actual_delta_ns"] for row in rows]
                ),
                "median_configured_alignment": _median(
                    [row["configured_alignment"] for row in rows]
                ),
                "median_actual_delta_alignment": _median(
                    [row["actual_delta_alignment"] for row in rows]
                ),
                "median_actual_alignment": _median(
                    [row["actual_alignment"] for row in rows]
                ),
                "median_target_kernel_latency_ns": _median(
                    [row["target_kernel_latency_ns"] for row in rows]
                ),
                "median_memcpy_adjacent_waiting_ns": _median(
                    [row["memcpy_adjacent_waiting_ns"] for row in rows]
                ),
                "median_normalized_inference_overlap": _median(
                    [row["normalized_inference_overlap"] for row in rows]
                ),
                "median_target_context_switch_count": _median(
                    [row["target_context_switch_count"] for row in rows]
                ),
                "median_target_inactive_ns": _median(
                    [row["target_inactive_ns"] for row in rows]
                ),
                "median_target_switch_transition_ns": _median(
                    [row["target_switch_transition_ns"] for row in rows]
                ),
                "median_compute_related_descheduling_ns": _median(
                    [row["compute_related_descheduling_ns"] for row in rows]
                ),
                "median_subgraph_kernel_time_ns": _median(
                    [row["subgraph_kernel_time_ns"] for row in rows]
                ),
                "max_abs_launch_error_ns": max(
                    [abs(row["target_launch_error_ns"]) for row in rows]
                    + [abs(row["co_runner_launch_error_ns"]) for row in rows],
                    default=None,
                ),
                "replay_memcpy_count": sum(
                    row["replay_memcpy_count"] for row in rows
                ),
            }
        )
    summaries = []
    for run in runs:
        target_kernel = calibration["models"][run["target_model_id"]]["0"][
            "median_kernel_time_ns"
        ]
        demand_ratio = (
            (run["median_subgraph_kernel_time_ns"] or 0) / target_kernel
            if target_kernel
            else None
        )
        demand = (
            None
            if demand_ratio is None
            else (
                "low"
                if demand_ratio < 0.5
                else "medium" if demand_ratio <= 1.5 else "high"
            )
        )
        summaries.append(
            {
                **run,
                "compute_demand_ratio": demand_ratio,
                "compute_demand_regime": demand,
            }
        )
    directions = {}
    for direction_id, direction in manifest["directions"].items():
        group = [
            row for row in summaries if row["direction_id"] == direction_id
        ]
        matched = direction["matched_pair"]
        matched_pair_observable = direction.get(
            "matched_pair_meets_v2_separation", True
        )
        high_index = matched["high_alignment_offset_index"]
        low_index = matched["low_alignment_offset_index"]
        k0 = sorted(
            [row for row in group if row["k_replays"] == 0],
            key=lambda row: row["offset_index"],
        )
        high0 = next(row for row in k0 if row["offset_index"] == high_index)
        low0 = next(row for row in k0 if row["offset_index"] == low_index)
        alignments = [row["median_actual_alignment"] for row in k0]
        waits = [row["median_memcpy_adjacent_waiting_ns"] for row in k0]
        latencies = [row["median_target_kernel_latency_ns"] for row in k0]
        alignment_wait = (
            _spearman(alignments, waits)
            if None not in alignments + waits
            else None
        )
        alignment_latency = (
            _spearman(alignments, latencies)
            if None not in alignments + latencies
            else None
        )
        experiment1 = (
            matched_pair_observable
            and high0["median_memcpy_adjacent_waiting_ns"]
            > low0["median_memcpy_adjacent_waiting_ns"]
            and high0["median_target_kernel_latency_ns"]
            > low0["median_target_kernel_latency_ns"]
            and alignment_wait is not None
            and alignment_wait > 0
            and alignment_latency is not None
            and alignment_latency > 0
        )
        deltas = []
        for k_replays in K_VALUES:
            high = next(
                row
                for row in group
                if row["k_replays"] == k_replays
                and row["offset_index"] == high_index
            )
            low = next(
                row
                for row in group
                if row["k_replays"] == k_replays
                and row["offset_index"] == low_index
            )
            deltas.append(
                {
                    "k_replays": k_replays,
                    "delta_l_alignment_ns": (
                        high["median_target_kernel_latency_ns"]
                        - low["median_target_kernel_latency_ns"]
                    ),
                }
            )
        trend = _spearman(
            [item["k_replays"] for item in deltas],
            [item["delta_l_alignment_ns"] for item in deltas],
        )
        by_k_demand = {}
        for k_replays in K_VALUES:
            values = [
                row["compute_demand_ratio"]
                for row in group
                if row["k_replays"] == k_replays
                and row["compute_demand_ratio"] is not None
            ]
            by_k_demand[k_replays] = _median(values)
        highest_k = max(by_k_demand, key=lambda key: by_k_demand[key] or 0)
        highest = [row for row in group if row["k_replays"] == highest_k]
        descheduling_dominates = _median(
            [row["median_compute_related_descheduling_ns"] for row in highest]
        ) > _median(
            [row["median_memcpy_adjacent_waiting_ns"] for row in highest]
        )
        experiment2 = (
            matched_pair_observable
            and deltas[-1]["delta_l_alignment_ns"]
            < deltas[0]["delta_l_alignment_ns"]
            and trend is not None
            and trend < 0
            and descheduling_dominates
        )
        solo_latency = calibration["models"][direction["target_model_id"]][
            "0"
        ]["median_kernel_latency_ns"]
        nonoverlap = [
            row
            for row in group
            if row["offset_label"].startswith("nonoverlap_")
        ]
        nonoverlap_deviations = [
            abs(row["median_target_kernel_latency_ns"] - solo_latency)
            / solo_latency
            for row in nonoverlap
        ]
        nonoverlap_ok = all(
            value <= study["data"]["non_overlap_latency_tolerance"]
            for value in nonoverlap_deviations
        )
        represented = {row["compute_demand_regime"] for row in group}
        missing_regimes = {"low", "medium", "high"}.difference(represented)
        if not all(row["valid"] for row in group) or not nonoverlap_ok:
            status = "invalid"
        elif (
            not all(row["observable"] for row in group)
            or not matched_pair_observable
        ):
            status = "unobservable"
        elif not {"low", "medium", "high"}.issubset(represented):
            status = "regime_unrepresented"
        elif experiment1 and experiment2:
            status = "consistent"
        else:
            status = "not_observed"
        directions[direction_id] = {
            "status": status,
            "target_model_id": direction["target_model_id"],
            "co_runner_model_id": direction["co_runner_model_id"],
            "experiment1_consistent": experiment1,
            "experiment2_consistent": experiment2,
            "matched_pair_meets_v2_separation": matched_pair_observable,
            "alignment_waiting_spearman": alignment_wait,
            "alignment_latency_spearman": alignment_latency,
            "delta_l_alignment": deltas,
            "delta_l_k_spearman": trend,
            "highest_represented_k": highest_k,
            "compute_descheduling_exceeds_waiting": descheduling_dominates,
            "nonoverlap_latency_within_tolerance": nonoverlap_ok,
            "nonoverlap_max_latency_deviation": max(
                nonoverlap_deviations, default=None
            ),
            "represented_demand_regimes": sorted(
                item for item in represented if item is not None
            ),
            "missing_demand_regimes": sorted(missing_regimes),
            "demand_regime_status": (
                "regime_unrepresented" if missing_regimes else "represented"
            ),
        }
    statuses = [item["status"] for item in directions.values()]
    if "invalid" in statuses:
        overall = "invalid"
    elif "unobservable" in statuses:
        overall = "unobservable"
    elif "regime_unrepresented" in statuses:
        overall = "regime_unrepresented"
    elif statuses == ["consistent", "consistent"]:
        overall = "consistent_both"
    elif "consistent" in statuses:
        overall = "direction_dependent"
    else:
        overall = "not_observed"
    result = {
        "schema_version": 1,
        "skill": SKILL,
        "alignment_metric": ALIGNMENT_METRIC,
        "alignment_window_ns": calibration["alignment_window_ns"],
        "alignment_copy_classes": ["H2D", "D2H"],
        "analysis_provenance": manifest.get("analysis_provenance"),
        "recorded_source_root": manifest.get("recorded_source_root"),
        "status": overall,
        "all_runs_valid": all(row["valid"] for row in summaries),
        "invalid_runs": [
            {
                "candidate_id": row["candidate_id"],
                "errors": row["validity_errors"].split("; "),
            }
            for row in runs
            if not row["valid"]
        ],
        "directions": directions,
        "inference_policy": (
            "descriptive within-run repeats; no permutation p-values or "
            "clustered confidence claims"
        ),
    }
    output = study["output_root"]
    _write_csv(output / "per_trial.csv", trials)
    _write_csv(output / "per_run.csv", runs)
    _write_csv(output / "offset_k_summary.csv", summaries)
    _write_json(output / "validation_result.json", result)
    lines = [
        f"# {SKILL}",
        "",
        f"Overall status: `{overall}`.",
        "",
        *(
            [f"Analysis provenance: {manifest['analysis_provenance']}.", ""]
            if manifest.get("analysis_provenance")
            else []
        ),
        "Alignment is the mean fraction of the fixed arbitration window "
        "immediately preceding each target H2D/D2H copy start during which "
        "the co-runner GPU context is active. The window is "
        f"{calibration['alignment_window_ns'] / 1_000_000:g} ms; D2D copies "
        "are excluded.",
        "",
        "Five pairs in each cell are within-run repeats; no permutation "
        "p-values or clustered confidence claims are reported.",
        "",
        "| Direction | Experiment 1 | Experiment 2 | Non-overlap max "
        "deviation | Demand bands | Status |",
        "|---|---|---|---|---|---|",
    ]
    for direction_id, item in directions.items():
        lines.append(
            f"| {direction_id} | {item['experiment1_consistent']} | "
            f"{item['experiment2_consistent']} | "
            f"{item['nonoverlap_max_latency_deviation']:.1%} | "
            f"{', '.join(item['represented_demand_regimes'])} | "
            f"{item['status']} |"
        )
    lines.extend(
        [
            "",
            *(
                f"- {direction_id} demand coverage: "
                f"`{item['demand_regime_status']}`"
                + (
                    " (missing "
                    + ", ".join(item["missing_demand_regimes"])
                    + ")."
                    if item["missing_demand_regimes"]
                    else "."
                )
                for direction_id, item in directions.items()
            ),
        ]
    )
    if any(
        not item["matched_pair_meets_v2_separation"]
        for item in directions.values()
    ):
        lines.extend(
            [
                "",
                "The recorded high/low pairs do not meet the V2 alignment "
                "separation threshold, so their alignment contrast is "
                "unobservable. Independent validity checks still determine "
                "the reported direction and overall status.",
            ]
        )
    if result["invalid_runs"]:
        lines.extend(["", "Invalid run evidence:"])
        lines.extend(
            f"- `{item['candidate_id']}`: {'; '.join(item['errors'])}"
            for item in result["invalid_runs"]
        )
    lines.extend(["", "Experiment 2 is a compute-service-demand sweep."])
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def _explicit_clocks(args):
    values = (args.graphics_clock_mhz, args.memory_clock_mhz)
    if (values[0] is None) != (values[1] is None):
        raise StudyError("both explicit clocks are required together")
    return (
        {"graphics_clock_mhz": values[0], "memory_clock_mhz": values[1]}
        if values[0] is not None
        else None
    )


def _parser():
    parser = argparse.ArgumentParser(description=SKILL)
    parser.add_argument("--artifact-root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("study")
    validate.add_argument("--config-only", action="store_true")
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("study")
    calibrate.add_argument("--dry-run", action="store_true")
    calibrate.add_argument("--graphics-clock-mhz", type=int)
    calibrate.add_argument("--memory-clock-mhz", type=int)
    generate = commands.add_parser("generate")
    generate.add_argument("study")
    run = commands.add_parser("run")
    run.add_argument("study")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--gate", action="store_true")
    run.add_argument("--candidate", action="append")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("study")
    return parser


def main(argv=None):
    """Validate, calibrate, generate, run, or analyze one study."""
    args = _parser().parse_args(argv)
    try:
        study = load_study(args.study)
        study["output_root"] = Path(args.artifact_root).resolve()
        if args.command == "validate":
            evidence = (
                {
                    "config_valid": True,
                    "alignment_metric": ALIGNMENT_METRIC,
                    "alignment_window_ms": study["data"][
                        "alignment_window_ms"
                    ],
                }
                if args.config_only
                else {
                    "config_valid": True,
                    "alignment_metric": ALIGNMENT_METRIC,
                    "alignment_window_ms": study["data"][
                        "alignment_window_ms"
                    ],
                    "host": validate_non_mps_host(study),
                }
            )
            _write_json(
                study["output_root"] / "study_validation.json", evidence
            )
            print(json.dumps(evidence, sort_keys=True))
        elif args.command == "calibrate":
            result = calibrate_study(
                study, _explicit_clocks(args), args.dry_run
            )
            print(
                json.dumps(
                    {
                        "calibration": str(
                            study["output_root"] / "calibration_manifest.json"
                        ),
                        "result": result,
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "generate":
            manifest = NonMpsMemcpyComputeDemandV2(study).generate()
            print(
                json.dumps(
                    {
                        "candidates": len(manifest["candidates"]),
                        "manifest": str(
                            study["output_root"] / "candidate_manifest.json"
                        ),
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "run":
            print(
                json.dumps(
                    run_campaign(
                        study, args.candidate, args.gate, args.dry_run
                    ),
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(analyze_campaign(study), sort_keys=True))
    except (
        ConfigError,
        StudyError,
        PreflightError,
        RunFailure,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
