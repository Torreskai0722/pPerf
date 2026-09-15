"""Proven shared readers and statistics for offline analyzers."""

import hashlib
import json
import math
from collections import defaultdict
from bisect import bisect_right
from pathlib import Path
import statistics

import yaml


class StudyError(RuntimeError):
    """Raised when authored study or recorded evidence is inconsistent."""


def sha256(path):
    """Return the SHA-256 of one immutable source file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(path):
    """Read authored YAML without importing experiment runtime code."""
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise StudyError(f"cannot load study: {exc}") from exc
    if not isinstance(data, dict):
        raise StudyError("study root must be an object")
    return {
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": data,
    }


def tables(connection):
    """Return the SQLite table inventory."""
    return {
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(connection, table):
    return {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})")
    }


def _pid(global_id):
    if global_id is None:
        return -1
    value = int(global_id)
    return (value >> 24) & 0xFFFFFF


def nvtx_ranges(connection):
    """Decode completed closeloop NVTX ranges from an Nsight export."""
    if "NVTX_EVENTS" not in tables(connection):
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
        result.append({
            "start": int(start), "end": int(end),
            "pid": _pid(global_tid), "tag": tag,
        })
    return result


def _copy_kind(value, enum_names):
    name = str(enum_names.get(value, value)).upper().replace("_", "")
    if value == 1 or any(
            token in name for token in ("HTOD", "HOSTTODEVICE")):
        return "H2D"
    if value == 2 or any(
            token in name for token in ("DTOH", "DEVICETOHOST")):
        return "D2H"
    return None


def memcopies(connection):
    """Read normalized H2D and D2H records from an Nsight export."""
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if table not in tables(connection):
        return []
    columns = _columns(connection, table)
    kind_column = next((
        name for name in ("copyKind", "kind", "memcpyKind")
        if name in columns
    ), None)
    bytes_column = next((
        name for name in ("bytes", "size", "byteCount")
        if name in columns
    ), None)
    correlation = "correlationId" if "correlationId" in columns else None
    selected = ["start", "end", "globalPid"]
    selected += [correlation] if correlation else []
    selected += [kind_column] if kind_column else []
    selected += [bytes_column] if bytes_column else []
    enum_names = {}
    for enum_table in ("ENUM_CUDA_MEMCPY_OPER", "ENUM_CUDA_MEMCPY_KIND"):
        if enum_table not in tables(connection):
            continue
        enum_columns = _columns(connection, enum_table)
        name_column = "name" if "name" in enum_columns else "label"
        enum_names.update(dict(connection.execute(
            f"SELECT id,{name_column} FROM {enum_table}"
        )))
    result = []
    for row in connection.execute(f"SELECT {','.join(selected)} FROM {table}"):
        values = dict(zip(selected, row))
        kind = _copy_kind(values.get(kind_column), enum_names)
        if kind and values["globalPid"] is not None:
            result.append({
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
            })
    return result


def runtime_events(connection):
    """Read per-thread CUDA runtime calls from an Nsight export."""
    required = {"CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"}
    if not required.issubset(tables(connection)):
        return {}
    events = defaultdict(list)
    for start, end, global_tid, correlation, name in connection.execute(
        "SELECT r.start,r.end,r.globalTid,r.correlationId,s.value "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId ORDER BY r.start"
    ):
        events[int(global_tid)].append({
            "start": int(start), "end": int(end), "pid": _pid(global_tid),
            "correlation": (
                int(correlation) if correlation is not None else None
            ),
            "name": str(name).split("_v", 1)[0],
        })
    return events


def merge_intervals(intervals):
    """Merge overlapping half-open intervals."""
    merged = []
    for start, end in sorted(
            (int(start), int(end)) for start, end in intervals if end > start):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def intersection_duration(left, right):
    """Return duration covered by both interval sets."""
    left, right = merge_intervals(left), merge_intervals(right)
    first = second = total = 0
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


def status_map(run_directory):
    """Read every valid model status record in a run."""
    result = {}
    for path in Path(run_directory).glob("model_*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            result[status["model_id"]] = status
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return result


def active_context_intervals(events):
    """Reconstruct active CUDA-context and transition intervals."""
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
        merge_intervals(active), merge_intervals(transitions), errors,
        switch_timestamps,
    )


def context_owner(connection, process_pid, kernels):
    """Resolve one process to its Nsight GPU-context-switch owner."""
    required = {"GPU_CONTEXT_SWITCH_EVENTS", "ENUM_GPU_CTX_SWITCH"}
    if not required.issubset(tables(connection)):
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
                (int(timestamp), int(context_id), str(tag)))
    direct = [owner for owner in by_owner if _pid(owner) == process_pid]
    if len(direct) == 1:
        return (direct[0], by_owner[direct[0]]), []
    starts = [item["start"] for item in kernels if item["pid"] == process_pid]
    scores = []
    for owner, events in by_owner.items():
        active = active_context_intervals(events)[0]
        active_starts = [left for left, _right in active]

        def is_active(start):
            index = bisect_right(active_starts, start) - 1
            return index >= 0 and start < active[index][1]

        scores.append((sum(is_active(start) for start in starts), owner))
    scores.sort(reverse=True)
    if not scores or scores[0][0] == 0:
        return None, ["no Docker PID-namespace context attribution"]
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        return None, ["ambiguous Docker PID-namespace context attribution"]
    return (scores[0][1], by_owner[scores[0][1]]), []


def sync_wait(memcopy_records, runtimes, inference):
    """Sum stream synchronization immediately following measured copies."""
    correlations = {
        item["correlation"] for item in memcopy_records
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


def spearman(left, right):
    """Return Spearman correlation for two equal series."""
    if len(left) < 3 or len(left) != len(right):
        return None
    left, right = _rank(left), _rank(right)
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum(
        (x_value - left_mean) * (y_value - right_mean)
        for x_value, y_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None
