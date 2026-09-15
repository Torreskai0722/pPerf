"""Analyze preprocessing ranges with per-thread scheduler-state evidence."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

from ._manifest import EvidenceError, load_manifest, require
from ._output import result_directory, write_result


TAG_PREFIX = "closeloop:"
PREPROCESS_EVENTS = frozenset((
    "preprocess",
    "preprocess_step",
    "data_preprocessor_step",
))
SCHEDULER_FIELDS = (
    "timestamp_ns", "event", "cpu", "prev_tid", "prev_state",
    "next_tid", "target_tid", "parent_tid", "child_tid",
    "orig_cpu", "dest_cpu", "comm",
)


def decode_global_id(global_id: int) -> Tuple[int, int]:
    """Decode an Nsight serialized global ID into PID and TID."""
    return ((global_id >> 24) & 0xFFFFFF, global_id & 0xFFFFFF)


@dataclass(frozen=True)
class StageRange:
    """One measured preprocessing range."""

    start: int
    end: int
    pid: int
    tid: int
    model: str
    input_id: str
    stage: str

    @property
    def duration_ns(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SchedulerEvent:
    """One raw kernel scheduler event."""

    timestamp_ns: int
    event: str
    cpu: int
    prev_tid: int = 0
    prev_state: int = 0
    next_tid: int = 0
    target_tid: int = 0
    parent_tid: int = 0
    child_tid: int = 0
    orig_cpu: int = 0
    dest_cpu: int = 0
    comm: str = ""


@dataclass(frozen=True)
class StateInterval:
    """One reconstructed scheduling-state interval."""

    start: int
    end: int
    state: str
    cpu: Optional[int]


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - rank) +
        ordered[upper] * (rank - lower)
    )


def _correlation(left: List[float], right: List[float]) -> Optional[float]:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x_value - left_mean) * (y_value - right_mean)
        for x_value, y_value in zip(left, right)
    )
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum * right_sum)
    return numerator / denominator if denominator else None


def _action_name(stage: str) -> str:
    """Return a compact action label while preserving the full stage."""
    value = stage.rsplit(":", 1)[-1]
    return value.rsplit(".", 1)[-1]


def _confidence(
        trace_quality: float, ownership_quality: float,
        mechanism_signature: float, tail_association: float,
        validation_result: float = 0.0,
        opposing_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Return an inspectable evidence-strength score, not a probability."""
    components = {
        "trace_quality": trace_quality,
        "ownership_quality": ownership_quality,
        "mechanism_signature": mechanism_signature,
        "tail_association": tail_association,
        "validation_result": validation_result,
    }
    weights = {
        "trace_quality": 0.15,
        "ownership_quality": 0.15,
        "mechanism_signature": 0.20,
        "tail_association": 0.25,
        "validation_result": 0.25,
    }
    score = sum(
        components[name] * weights[name] for name in components
    ) - opposing_penalty
    cap = 0.95 if validation_result else 0.75
    score = max(0.0, min(cap, score))
    if score >= 0.85:
        label = "validated"
    elif score >= 0.70:
        label = "high"
    elif score >= 0.40:
        label = "medium"
    else:
        label = "low"
    return {
        "score": round(score, 3),
        "label": label,
        "cap": cap,
        "components": components,
        "opposing_penalty": opposing_penalty,
    }


class PreprocessAnalyzer:
    """Correlate preprocessing latency with exact scheduler state intervals."""

    analyzer_id = "preprocess_contention_analyzer_v1"

    def __init__(self, run_directory: Path,
                 options: Optional[Dict[str, Any]] = None,
                 output_directory: Optional[Path] = None):
        self.run_directory = Path(run_directory)
        self.output_directory = Path(output_directory or run_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.options = dict(options or {})
        self.sqlite_path = self.run_directory / "profile.sqlite"
        self.scheduler_path = self.run_directory / "scheduler_events.csv"

    @staticmethod
    def _tag(text: Optional[str]) -> Optional[Dict[str, Any]]:
        if not text or not text.startswith(TAG_PREFIX):
            return None
        try:
            value = json.loads(text[len(TAG_PREFIX):])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def _ranges(self, connection: sqlite3.Connection) -> List[StageRange]:
        records = []
        rows = connection.execute(
            "SELECT start, end, text, globalTid FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND text LIKE 'closeloop:%'"
        )
        exclude_warmup = self.options.get("exclude_warmup", True)
        for start, end, text, global_tid in rows:
            tag = self._tag(text)
            if tag is None or global_tid is None:
                continue
            event = tag.get("event")
            module = tag.get("module")
            if event not in PREPROCESS_EVENTS:
                is_data_preprocessor = (
                    event == "module" and module == "data_preprocessor"
                )
                if not is_data_preprocessor:
                    continue
            input_id = str(tag.get("input", ""))
            if exclude_warmup and input_id.startswith("warmup-"):
                continue
            pid, tid = decode_global_id(int(global_tid))
            if event == "module":
                stage = "data_preprocessor"
            elif event == "preprocess":
                stage = "preprocess"
            else:
                stage = str(module or event)
            records.append(StageRange(
                int(start), int(end), pid, tid,
                str(tag.get("model", "")), input_id, stage,
            ))
        return sorted(records, key=lambda item: (item.start, item.end))

    @staticmethod
    def _nsys_scheduler(
            connection: sqlite3.Connection
    ) -> List[Tuple[int, int, int, int, int]]:
        return [
            (int(start), int(cpu), int(is_in), *decode_global_id(int(gid)))
            for start, cpu, is_in, gid in connection.execute(
                "SELECT start, cpu, isSchedIn, globalTid "
                "FROM SCHED_EVENTS WHERE globalTid IS NOT NULL "
                "ORDER BY start"
            )
        ]

    @staticmethod
    def _thread_names(
            connection: sqlite3.Connection) -> Dict[Tuple[int, int], str]:
        tables = PreprocessAnalyzer._tables(connection)
        if not {"ThreadNames", "StringIds"}.issubset(tables):
            return {}
        names = {}
        query = (
            "SELECT t.globalTid, s.value FROM ThreadNames t "
            "JOIN StringIds s ON s.id = t.nameId"
        )
        for global_tid, name in connection.execute(query):
            names[decode_global_id(int(global_tid))] = str(name)
        return names

    @staticmethod
    def _tables(connection: sqlite3.Connection) -> set:
        return {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    @staticmethod
    def _stack_text(frames: List[Tuple[str, str]]) -> str:
        return ";".join(
            f"{symbol}@{module}" for symbol, module in frames
        )

    @staticmethod
    def _stack_owner(frames: List[Tuple[str, str]]) -> str:
        signatures = (
            "libgomp", "libomp", "libtorch", "libopenblas", "libmkl",
            "opencv", "libtbb", "libcuda", "libcudart",
        )
        for _symbol, module in frames:
            lowered = module.lower()
            if any(signature in lowered for signature in signatures):
                return Path(module).name
        ignored = ("libc.so", "libtoolsinjection", "[unknown]")
        for _symbol, module in frames[1:]:
            lowered = module.lower()
            if not any(item in lowered for item in ignored):
                return Path(module).name
        return ""

    @staticmethod
    def _osrt_calls(
            connection: sqlite3.Connection
    ) -> List[Dict[str, Any]]:
        tables = PreprocessAnalyzer._tables(connection)
        required = {"OSRT_API", "OSRT_CALLCHAINS", "StringIds"}
        if not required.issubset(tables):
            return []
        grouped: DefaultDict[
            Tuple[int, int, int, str, int],
            List[Tuple[int, str, str]],
        ] = defaultdict(list)
        query = (
            "SELECT o.start, o.end, o.globalTid, name.value, "
            "o.callchainId, c.stackDepth, symbol.value, module.value "
            "FROM OSRT_API o "
            "JOIN StringIds name ON name.id = o.nameId "
            "LEFT JOIN OSRT_CALLCHAINS c ON c.id = o.callchainId "
            "LEFT JOIN StringIds symbol ON symbol.id = c.symbol "
            "LEFT JOIN StringIds module ON module.id = c.module "
            "WHERE o.globalTid IS NOT NULL AND ("
            "name.value = 'pthread_create' OR "
            "name.value LIKE '%wait%' OR name.value LIKE '%join%' OR "
            "name.value LIKE '%sleep%' OR "
            "name.value LIKE '%mutex_lock%' OR "
            "name.value LIKE '%rwlock_%' OR "
            "name.value LIKE '%poll%' OR name.value LIKE '%select%' OR "
            "name.value LIKE '%receive%' OR name.value LIKE '%recv%' OR "
            "name.value = 'read') "
            "ORDER BY o.start, c.stackDepth"
        )
        for row in connection.execute(query):
            start, end, global_tid, name, chain_id = row[:5]
            depth, symbol, module = row[5:]
            if depth is not None:
                grouped[(
                    int(start), int(end), int(global_tid), str(name),
                    int(chain_id),
                )].append((int(depth), str(symbol), str(module)))
            else:
                grouped.setdefault((
                    int(start), int(end), int(global_tid), str(name),
                    int(chain_id),
                ), [])
        calls = []
        for key, entries in grouped.items():
            start, end, global_tid, name, _chain_id = key
            pid, tid = decode_global_id(int(global_tid))
            frames = [
                (symbol, module)
                for _depth, symbol, module in sorted(entries)
            ]
            calls.append({
                "start": int(start),
                "end": int(end),
                "pid": pid,
                "tid": tid,
                "api": str(name),
                "owner": PreprocessAnalyzer._stack_owner(frames),
                "stack": PreprocessAnalyzer._stack_text(frames),
            })
        return calls

    @staticmethod
    def _range_index(
            ranges: List[StageRange]
    ) -> Dict[int, Tuple[List[int], List[Tuple[int, int, StageRange]]]]:
        """Build non-overlapping segments labelled by innermost range."""
        by_pid: DefaultDict[
            int, List[Tuple[int, int, StageRange]]
        ] = defaultdict(list)
        for record in ranges:
            by_pid[record.pid].append((record.start, 1, record))
            by_pid[record.pid].append((record.end, 0, record))
        result = {}
        for pid, boundaries in by_pid.items():
            active: Dict[StageRange, None] = {}
            segments = []
            previous = None
            index = 0
            boundaries.sort(key=lambda item: (item[0], item[1]))
            while index < len(boundaries):
                timestamp = boundaries[index][0]
                if previous is not None and previous < timestamp and active:
                    record = min(
                        active, key=lambda item: item.duration_ns
                    )
                    segments.append((previous, timestamp, record))
                while (index < len(boundaries) and
                       boundaries[index][0] == timestamp):
                    _time, kind, record = boundaries[index]
                    if kind:
                        active[record] = None
                    else:
                        active.pop(record, None)
                    index += 1
                previous = timestamp
            result[pid] = (
                [segment[0] for segment in segments], segments
            )
        return result

    @staticmethod
    def _indexed_range(
            index: Dict[
                int, Tuple[List[int], List[Tuple[int, int, StageRange]]]
            ],
            pid: int, timestamp: int
    ) -> Optional[StageRange]:
        starts, segments = index.get(pid, ([], []))
        position = bisect_right(starts, timestamp) - 1
        if position < 0:
            return None
        start, end, record = segments[position]
        return record if start <= timestamp < end else None

    @staticmethod
    def _work_samples(
            connection: sqlite3.Connection,
            ranges: List[StageRange]
    ) -> List[Dict[str, Any]]:
        tables = PreprocessAnalyzer._tables(connection)
        required = {
            "COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS", "StringIds",
        }
        if not required.issubset(tables):
            return []
        range_index = PreprocessAnalyzer._range_index(ranges)
        counts: DefaultDict[
            Tuple[str, str, int, int, str, str], int
        ] = defaultdict(int)
        query = (
            "SELECT e.start, e.globalTid, symbol.value, module.value "
            "FROM COMPOSITE_EVENTS e "
            "JOIN SAMPLING_CALLCHAINS c "
            "ON c.id = e.id AND c.stackDepth = 0 "
            "JOIN StringIds symbol ON symbol.id = c.symbol "
            "JOIN StringIds module ON module.id = c.module "
            "WHERE e.globalTid IS NOT NULL ORDER BY e.start"
        )
        for timestamp, global_tid, symbol, module in connection.execute(
                query):
            pid, tid = decode_global_id(int(global_tid))
            record = PreprocessAnalyzer._indexed_range(
                range_index, pid, int(timestamp)
            )
            if record is None:
                continue
            key = (
                record.model, record.stage, pid, tid,
                str(symbol), str(module),
            )
            counts[key] += 1
        return [
            {
                "model": key[0],
                "stage": key[1],
                "pid": key[2],
                "tid": key[3],
                "symbol": key[4],
                "module": key[5],
                "samples": count,
            }
            for key, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    @staticmethod
    def _wait_rows(
            calls: List[Dict[str, Any]],
            ranges: List[StageRange]
    ) -> List[Dict[str, Any]]:
        wait_fragments = (
            "wait", "join", "sleep", "mutex_lock", "rwlock_",
            "poll", "select", "receive", "recv", "read",
        )
        range_index = PreprocessAnalyzer._range_index(ranges)
        rows = []
        for call in calls:
            if not any(
                    fragment in call["api"] for fragment in wait_fragments):
                continue
            record = PreprocessAnalyzer._indexed_range(
                range_index, call["pid"], call["start"]
            )
            if record is None:
                record = PreprocessAnalyzer._indexed_range(
                    range_index, call["pid"], call["end"] - 1
                )
            if record is None:
                continue
            overlap = max(
                0,
                min(record.end, call["end"])
                - max(record.start, call["start"]),
            )
            rows.append({
                "model": record.model,
                "input": record.input_id,
                "stage": record.stage,
                "pid": call["pid"],
                "tid": call["tid"],
                "api": call["api"],
                "overlap_ns": overlap,
                "owner": call["owner"],
                "stack": call["stack"],
            })
        return rows

    def _raw_scheduler(self) -> List[SchedulerEvent]:
        if not self.scheduler_path.is_file():
            return []
        records = []
        with self.scheduler_path.open(
                "r", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, fieldnames=SCHEDULER_FIELDS):
                try:
                    records.append(SchedulerEvent(
                        timestamp_ns=int(row["timestamp_ns"]),
                        event=row["event"],
                        cpu=int(row["cpu"]),
                        prev_tid=int(row["prev_tid"]),
                        prev_state=int(row["prev_state"]),
                        next_tid=int(row["next_tid"]),
                        target_tid=int(row["target_tid"]),
                        parent_tid=int(row["parent_tid"]),
                        child_tid=int(row["child_tid"]),
                        orig_cpu=int(row["orig_cpu"]),
                        dest_cpu=int(row["dest_cpu"]),
                        comm=row.get("comm", ""),
                    ))
                except (KeyError, TypeError, ValueError):
                    continue
        return records

    @staticmethod
    def _bpf_observations(
            events: Iterable[SchedulerEvent]
    ) -> DefaultDict[Tuple[int, int, int], List[int]]:
        observations: DefaultDict[Tuple[int, int, int], List[int]] = (
            defaultdict(list)
        )
        for event in events:
            if event.event != "switch":
                continue
            observations[(event.prev_tid, event.cpu, 0)].append(
                event.timestamp_ns
            )
            observations[(event.next_tid, event.cpu, 1)].append(
                event.timestamp_ns
            )
        return observations

    @staticmethod
    def _clock_offset(
            raw_events: List[SchedulerEvent],
            nsys_events: List[Tuple[int, int, int, int, int]],
            target_pids: set) -> int:
        """Estimate ``bpf_timestamp - nsys_timestamp`` from shared switches."""
        if not raw_events:
            return 0
        bpf = PreprocessAnalyzer._bpf_observations(raw_events)
        nsys = [
            (timestamp, cpu, is_in, tid)
            for timestamp, cpu, is_in, pid, tid in nsys_events
            if pid in target_pids
        ]
        candidates = []
        for timestamp, cpu, is_in, tid in nsys[:200]:
            values = bpf.get((tid, cpu, is_in), ())
            for bpf_timestamp in values[:200]:
                candidates.append(bpf_timestamp - timestamp)
            if len(candidates) >= 2000:
                break
        if not candidates:
            raise RuntimeError(
                "cannot align bpftrace and Nsight scheduler clocks"
            )
        sample = nsys[:500]
        best = None
        for candidate in candidates:
            residuals = []
            for timestamp, cpu, is_in, tid in sample:
                values = bpf.get((tid, cpu, is_in), ())
                if not values:
                    continue
                expected = timestamp + candidate
                index = bisect_left(values, expected)
                nearby = values[max(0, index - 1):index + 1]
                if nearby:
                    residuals.append(
                        min(abs(value - expected) for value in nearby)
                    )
            matches = sum(value <= 100_000 for value in residuals)
            median = statistics.median(residuals) if residuals else math.inf
            score = (matches, -median)
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None or best[0][0] < 2:
            raise RuntimeError(
                "insufficient shared scheduler events for clock alignment"
            )
        candidate = best[1]
        deltas = []
        for timestamp, cpu, is_in, tid in sample:
            values = bpf.get((tid, cpu, is_in), ())
            if not values:
                continue
            expected = timestamp + candidate
            index = bisect_left(values, expected)
            nearby = values[max(0, index - 1):index + 1]
            if not nearby:
                continue
            nearest = min(nearby, key=lambda value: abs(value - expected))
            if abs(nearest - expected) <= 100_000:
                deltas.append(nearest - timestamp)
        return int(statistics.median(deltas)) if deltas else candidate

    @staticmethod
    def _transition_events(
            events: Iterable[SchedulerEvent], offset: int,
            target_tids: set
    ) -> DefaultDict[int, List[Tuple[int, str, Optional[int]]]]:
        transitions: DefaultDict[
            int, List[Tuple[int, str, Optional[int]]]
        ] = defaultdict(list)
        for event in events:
            timestamp = event.timestamp_ns - offset
            if event.event == "switch":
                if event.prev_tid in target_tids:
                    runnable = (
                        event.prev_state == 0 or
                        bool(event.prev_state & 0x100)
                    )
                    transitions[event.prev_tid].append(
                        (timestamp,
                         "runnable_wait" if runnable else "blocked",
                         None)
                    )
                if event.next_tid in target_tids:
                    transitions[event.next_tid].append(
                        (timestamp, "running", event.cpu)
                    )
            elif (event.event in ("wakeup", "wakeup_new") and
                  event.target_tid in target_tids):
                transitions[event.target_tid].append(
                    (timestamp, "runnable_wait", None)
                )
            elif event.event == "exit" and event.target_tid in target_tids:
                transitions[event.target_tid].append(
                    (timestamp, "exited", None)
                )
        return transitions

    @staticmethod
    def _state_intervals(
            transitions: Dict[int, List[Tuple[int, str, Optional[int]]]],
            target_tids: set, start: int, end: int
    ) -> Dict[int, List[StateInterval]]:
        result = {}
        priorities = {
            "blocked": 0, "runnable_wait": 1,
            "running": 2, "exited": 3,
        }
        for tid in target_tids:
            items = sorted(
                transitions.get(tid, []),
                key=lambda item: (item[0], priorities[item[1]]),
            )
            state = "unknown"
            cpu = None
            state_start = start
            intervals = []
            for timestamp, next_state, next_cpu in items:
                if timestamp <= start:
                    state = next_state
                    cpu = next_cpu
                    continue
                if timestamp >= end:
                    break
                if timestamp > state_start:
                    intervals.append(StateInterval(
                        state_start, timestamp, state, cpu
                    ))
                state = next_state
                cpu = next_cpu
                state_start = timestamp
            if state_start < end:
                intervals.append(StateInterval(
                    state_start, end, state, cpu
                ))
            result[tid] = intervals
        return result

    @staticmethod
    def _nsys_running_intervals(
            events: List[Tuple[int, int, int, int, int]],
            target_tids: set, start: int, end: int
    ) -> Dict[int, List[StateInterval]]:
        transitions = defaultdict(list)
        for timestamp, cpu, is_in, _pid, tid in events:
            if tid not in target_tids:
                continue
            transitions[tid].append((
                timestamp, "running" if is_in else "unknown",
                cpu if is_in else None,
            ))
        return PreprocessAnalyzer._state_intervals(
            transitions, target_tids, start, end
        )

    @staticmethod
    def _thread_activity(
            events: Iterable[SchedulerEvent], offset: int,
            target_tids: set
    ) -> Dict[int, Dict[str, List[int]]]:
        activity: DefaultDict[
            int, DefaultDict[str, List[int]]
        ] = defaultdict(lambda: defaultdict(list))
        for event in events:
            timestamp = event.timestamp_ns - offset
            if event.event == "switch":
                if event.prev_tid in target_tids:
                    activity[event.prev_tid]["switches"].append(timestamp)
                if event.next_tid in target_tids:
                    activity[event.next_tid]["switches"].append(timestamp)
            elif (event.event in ("wakeup", "wakeup_new") and
                  event.target_tid in target_tids):
                activity[event.target_tid]["wakeups"].append(timestamp)
            elif (event.event == "migrate" and
                  event.target_tid in target_tids):
                activity[event.target_tid]["migrations"].append(timestamp)
        return {
            tid: {
                kind: sorted(timestamps)
                for kind, timestamps in kinds.items()
            }
            for tid, kinds in activity.items()
        }

    @staticmethod
    def _activity_count(timestamps: List[int], start: int, end: int) -> int:
        return bisect_left(timestamps, end) - bisect_left(timestamps, start)

    @staticmethod
    def _overlap(
            intervals: List[StateInterval], start: int,
            end: int) -> Dict[str, int]:
        totals = {
            "running": 0,
            "runnable_wait": 0,
            "blocked": 0,
            "unknown": 0,
            "exited": 0,
        }
        if not intervals:
            totals["unknown"] = end - start
            return totals
        starts = [interval.start for interval in intervals]
        index = max(0, bisect_right(starts, start) - 1)
        while index < len(intervals):
            interval = intervals[index]
            if interval.start >= end:
                break
            overlap = max(
                0, min(end, interval.end) - max(start, interval.start)
            )
            totals[interval.state] = (
                totals.get(interval.state, 0) + overlap
            )
            index += 1
        accounted = sum(totals.values())
        if accounted < end - start:
            totals["unknown"] += end - start - accounted
        return totals

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]],
                   fields: List[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _lifecycle_rows(
            self, raw: List[SchedulerEvent], offset: int,
            tid_to_pid: Dict[int, int],
            names: Dict[Tuple[int, int], str],
            nsys_events: List[Tuple[int, int, int, int, int]],
            osrt_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        first = {}
        last = {}
        cpus: DefaultDict[int, set] = defaultdict(set)
        for timestamp, cpu, _is_in, _pid, tid in nsys_events:
            first[tid] = min(timestamp, first.get(tid, timestamp))
            last[tid] = max(timestamp, last.get(tid, timestamp))
            cpus[tid].add(cpu)
        parents = {}
        exits = {}
        for event in raw:
            timestamp = event.timestamp_ns - offset
            if event.event == "fork":
                parents[event.child_tid] = (event.parent_tid, timestamp)
            elif event.event == "exit":
                exits[event.target_tid] = timestamp
        rows = []
        creates: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        for call in osrt_calls:
            if call["api"] == "pthread_create":
                creates[call["tid"]].append(call)
        for tid, pid in sorted(tid_to_pid.items(), key=lambda item: item[0]):
            parent_tid, created = parents.get(tid, ("", ""))
            creation = None
            if parent_tid != "":
                candidates = [
                    call for call in creates.get(parent_tid, ())
                    if call["start"] - 1_000_000 <= created
                    <= call["end"] + 1_000_000
                ]
                if candidates:
                    creation = min(
                        candidates,
                        key=lambda call: min(
                            abs(created - call["start"]),
                            abs(created - call["end"]),
                        ),
                    )
            rows.append({
                "pid": pid,
                "tid": tid,
                "name": names.get((pid, tid), ""),
                "parent_tid": parent_tid,
                "created_ns": created,
                "creation_api": (
                    creation["api"] if creation is not None else ""
                ),
                "creator_owner": (
                    creation["owner"] if creation is not None else ""
                ),
                "creator_stack": (
                    creation["stack"] if creation is not None else ""
                ),
                "first_scheduled_ns": first.get(tid, ""),
                "last_scheduled_ns": last.get(tid, ""),
                "exited_ns": exits.get(tid, ""),
                "cpus": " ".join(str(cpu) for cpu in sorted(cpus[tid])),
            })
        return rows

    @staticmethod
    def _merged_duration(intervals: List[Tuple[int, int]]) -> int:
        """Return the duration of the union of half-open intervals."""
        if not intervals:
            return 0
        ordered = sorted(intervals)
        start, end = ordered[0]
        total = 0
        for next_start, next_end in ordered[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        return total + end - start

    @classmethod
    def _exclusive_durations(
            cls, ranges: List[StageRange]) -> Dict[StageRange, int]:
        """Subtract nested work so parent stages are not duplicated."""
        grouped: DefaultDict[
            Tuple[int, str, str], List[StageRange]
        ] = defaultdict(list)
        for record in ranges:
            grouped[(record.pid, record.model, record.input_id)].append(
                record
            )
        result = {}
        for records in grouped.values():
            for parent in records:
                children = [
                    (child.start, child.end)
                    for child in records
                    if child != parent and
                    child.start >= parent.start and
                    child.end <= parent.end and
                    (child.start > parent.start or child.end < parent.end)
                ]
                result[parent] = max(
                    0,
                    parent.duration_ns - cls._merged_duration(children),
                )
        return result

    def _manifest(self) -> Dict[str, Any]:
        """Load run metadata when available."""
        path = self.run_directory / "run_manifest.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _cpu_topology(manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Return captured topology, with a local fallback for old runs."""
        captured = manifest.get("cpu_topology")
        if isinstance(captured, dict) and captured.get("cpus"):
            return captured
        cpu_root = Path("/sys/devices/system/cpu")
        records = []
        paths = sorted(
            cpu_root.glob("cpu[0-9]*"),
            key=lambda item: int(item.name[3:]),
        )
        for path in paths:
            cpu = int(path.name[3:])
            try:
                core = int(
                    (path / "topology" / "core_id").read_text(
                        encoding="utf-8"
                    ).strip()
                )
                package = int(
                    (path / "topology" / "physical_package_id").read_text(
                        encoding="utf-8"
                    ).strip()
                )
            except (OSError, ValueError):
                core = cpu
                package = 0
            records.append({"cpu": cpu, "core": core, "package": package})
        return {
            "logical_cpu_count": len(records),
            "physical_core_count": len({
                (item["package"], item["core"]) for item in records
            }),
            "cpus": records,
            "source": "analysis_host_fallback",
        }

    @staticmethod
    def _physical_cores(
            topology: Dict[str, Any], cpus: Iterable[int]) -> set:
        """Return physical-core identities represented by logical CPU IDs."""
        allowed = set(cpus)
        return {
            (item.get("package", 0), item.get("core", item["cpu"]))
            for item in topology.get("cpus", ())
            if item["cpu"] in allowed
        }

    @staticmethod
    def _pool_identity(
            lifecycle: Dict[str, Any]
    ) -> Optional[Tuple[str, str, str]]:
        """Map a creator stack to logical owner, pool kind, and backend."""
        owner = str(lifecycle.get("creator_owner", "")).lower()
        stack = str(lifecycle.get("creator_stack", "")).lower()
        evidence = owner + ";" + stack
        if "libgomp" in evidence or "libomp" in evidence:
            backend = "libgomp" if "libgomp" in evidence else "libomp"
            if any(fragment in evidence for fragment in (
                    "libtorch", "tensoriterator", "at::", "torch::")):
                return "pytorch", "intra_op", backend
            return "openmp", "native", backend
        if "opencv" in evidence or "cv2" in evidence:
            return "opencv", "parallel_for", "opencv"
        if "libtorch" in evidence:
            return "pytorch", "native", "libtorch"
        return None

    @classmethod
    def _pool_inventory(
            cls, lifecycle_rows: List[Dict[str, Any]]
    ) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, str]]:
        """Return unique compute-pool records and worker-to-pool ownership."""
        grouped: DefaultDict[
            Tuple[int, str, str, str], set
        ] = defaultdict(set)
        tid_to_pool = {}
        for row in lifecycle_rows:
            identity = cls._pool_identity(row)
            if identity is None:
                continue
            pid = int(row["pid"])
            owner, kind, backend = identity
            pool_id = f"{pid}:{owner}:{kind}:{backend}"
            tid = int(row["tid"])
            grouped[(pid, owner, kind, backend)].add(tid)
            tid_to_pool[tid] = pool_id
        inventories: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        for (pid, owner, kind, backend), tids in sorted(grouped.items()):
            inventories[pid].append({
                "pool_id": f"{pid}:{owner}:{kind}:{backend}",
                "framework_owner": owner,
                "pool_kind": kind,
                "execution_backend": backend,
                "observed_worker_threads": len(tids),
                "team_size_estimate": len(tids) + 1,
            })
        return dict(inventories), tid_to_pool

    @staticmethod
    def _action_libraries(
            work_samples: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Attribute sampled action work without double-counting backends."""
        grouped: DefaultDict[
            Tuple[str, str], DefaultDict[str, int]
        ] = defaultdict(lambda: defaultdict(int))
        operations: DefaultDict[
            Tuple[str, str, str], DefaultDict[str, int]
        ] = defaultdict(lambda: defaultdict(int))
        for row in work_samples:
            module = str(row["module"]).lower()
            symbol = str(row["symbol"])
            if "libgomp" in module or "libomp" in module:
                library = "pytorch"
                backend = "libgomp" if "libgomp" in module else "libomp"
            elif "libtorch" in module:
                library = "pytorch"
                backend = "libtorch"
            elif "opencv" in module or "cv2" in module:
                library = "opencv"
                backend = "opencv"
            else:
                library = "other"
                backend = Path(str(row["module"])).name
            key = (str(row["model"]), str(row["stage"]))
            count = int(row["samples"])
            grouped[key][f"{library}|{backend}"] += count
            if symbol and not symbol.startswith("0x"):
                operations[(key[0], key[1], library)][symbol] += count
        result = {}
        for key, counts in grouped.items():
            total = sum(counts.values())
            entries = []
            for encoded, count in sorted(
                    counts.items(), key=lambda item: -item[1]):
                library, backend = encoded.split("|", 1)
                operation_counts = operations.get(
                    (key[0], key[1], library), {}
                )
                top_operation = (
                    max(operation_counts, key=operation_counts.get)
                    if operation_counts else None
                )
                if top_operation and len(top_operation) > 80:
                    top_operation = top_operation[:77] + "..."
                entries.append({
                    "framework_owner": library,
                    "execution_backend": backend,
                    "sample_share": round(count / total, 3) if total else 0,
                    "top_operation": top_operation,
                })
            result[key] = entries[:2]
        return result

    @staticmethod
    def _pool_stage_statistics(
            thread_rows: List[Dict[str, Any]], tid_to_pool: Dict[int, str]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Summarize identified pool overlap for every action invocation."""
        invocations: DefaultDict[
            Tuple[str, str, int], DefaultDict[
                str, DefaultDict[str, float]
            ]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for row in thread_rows:
            if row["relation"] != "own":
                continue
            pool_id = tid_to_pool.get(int(row["tid"]))
            if pool_id is None:
                continue
            wall = int(row["wall_ns"])
            if not wall:
                continue
            key = (
                str(row["model"]), str(row["stage"]),
                int(row["stage_start_ns"]),
            )
            invocations[key][pool_id]["running"] += (
                int(row["running_ns"]) / wall
            )
            invocations[key][pool_id]["runnable_wait"] += (
                int(row["runnable_wait_ns"]) / wall
            )
            invocations[key][pool_id]["blocked"] += (
                int(row["blocked_ns"]) / wall
            )
        by_stage: DefaultDict[
            Tuple[str, str], List[Dict[str, Dict[str, float]]]
        ] = defaultdict(list)
        for (model, stage, _start), pools in invocations.items():
            by_stage[(model, stage)].append(dict(pools))
        result = {}
        for key, items in by_stage.items():
            pool_ids = sorted({
                pool_id for item in items for pool_id in item
            })
            pools = []
            for pool_id in pool_ids:
                running = [
                    item.get(pool_id, {}).get("running", 0.0)
                    for item in items
                ]
                runnable = [
                    item.get(pool_id, {}).get("runnable_wait", 0.0)
                    for item in items
                ]
                blocked = [
                    item.get(pool_id, {}).get("blocked", 0.0)
                    for item in items
                ]
                pools.append({
                    "pool_id": pool_id,
                    "running_p95": _percentile(running, 0.95),
                    "runnable_wait_p95": _percentile(runnable, 0.95),
                    "blocked_p95": _percentile(blocked, 0.95),
                })
            overlaps = sum(
                sum(
                    values.get("running", 0.0) +
                    values.get("runnable_wait", 0.0) >= 0.05
                    for values in item.values()
                ) >= 2
                for item in items
            )
            result[key] = {
                "pools": pools,
                "multi_pool_overlap_fraction": (
                    overlaps / len(items) if items else 0.0
                ),
            }
        return result

    @classmethod
    def _process_inventory(
            cls, manifest: Dict[str, Any], topology: Dict[str, Any],
            model_by_pid: Dict[int, str],
            lifecycle_rows: List[Dict[str, Any]],
            pool_inventory: Dict[int, List[Dict[str, Any]]],
            pool_statistics: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """Build compact process capacity, control, and pool metadata."""
        metadata_by_model = manifest.get("models", {})
        observed_cpus: DefaultDict[int, set] = defaultdict(set)
        tracked_threads: DefaultDict[int, int] = defaultdict(int)
        for row in lifecycle_rows:
            pid = int(row["pid"])
            tracked_threads[pid] += 1
            for value in str(row.get("cpus", "")).split():
                try:
                    observed_cpus[pid].add(int(value))
                except ValueError:
                    continue
        processes = []
        by_pid = {}
        for pid, model in sorted(
                model_by_pid.items(), key=lambda item: item[1]):
            metadata = metadata_by_model.get(model, {})
            cpus = metadata.get("cpu_affinity")
            if cpus is None:
                cpus = sorted(observed_cpus[pid])
            cpus = sorted(set(cpus or ()))
            physical = cls._physical_cores(topology, cpus)
            pools = []
            preprocess_pool_stats = {
                item["pool_id"]: item
                for item in pool_statistics.get(
                    (model, "preprocess"), {}
                ).get("pools", ())
            }
            for pool in pool_inventory.get(pid, ()):
                internal_id = pool["pool_id"]
                stable_id = (
                    f"{model}:{pool['framework_owner']}:"
                    f"{pool['pool_kind']}:{pool['execution_backend']}"
                )
                record = {
                    **pool,
                    "pool_id": stable_id,
                }
                metrics = preprocess_pool_stats.get(internal_id)
                if metrics is not None:
                    record["preprocess_p95"] = {
                        "running": round(metrics["running_p95"], 3),
                        "runnable_wait": round(
                            metrics["runnable_wait_p95"], 3
                        ),
                        "blocked": round(metrics["blocked_p95"], 3),
                    }
                pools.append(record)
            configured = metadata.get("library_thread_counts", {})
            observed = metadata.get(
                "observed_library_thread_counts", {}
            )
            process = {
                "model": model,
                "pid": pid,
                "effective_cpus": cpus,
                "affinity_source": metadata.get(
                    "cpu_affinity_source",
                    "inherited_or_full" if len(cpus) == topology.get(
                        "logical_cpu_count"
                    ) else "observed",
                ),
                "logical_cpu_capacity": len(cpus),
                "physical_core_capacity": len(physical),
                "cpu_thread_count": metadata.get("cpu_thread_count"),
                "library_thread_counts": configured,
                "observed_library_thread_counts": observed,
                "tracked_threads": tracked_threads[pid],
                "observed_compute_pools": pools,
            }
            processes.append(process)
            by_pid[pid] = process
        return processes, by_pid

    @staticmethod
    def _control_key(libraries: List[Dict[str, Any]]) -> str:
        """Return the most specific supported thread-control key."""
        if libraries:
            owner = libraries[0].get("framework_owner")
            if owner == "pytorch":
                return "pytorch_intraop"
            if owner == "opencv":
                return "opencv"
            if owner == "openmp":
                return "openmp"
        return "cpu_thread_count"

    @staticmethod
    def _validation_control(
            process: Dict[str, Any], control_key: str
    ) -> Dict[str, Any]:
        """Describe one executable matched-run thread-count control."""
        physical = max(1, process["physical_core_capacity"])
        logical = max(1, process["logical_cpu_capacity"])
        values = sorted({physical, logical})
        if control_key == "cpu_thread_count":
            path = f"models.{process['model']}.cpu_thread_count"
        else:
            path = (
                f"models.{process['model']}.library_thread_counts."
                f"{control_key}"
            )
        return {
            "type": "thread_count_override",
            "path": path,
            "values": values,
            "keep_other_controls_unchanged": True,
            "success": "same action p95 or p99 improves by at least 1 ms",
        }

    def analyze(self) -> Dict[str, Any]:
        """Write detailed tables and the planner-facing analysis result."""
        if not self.sqlite_path.is_file():
            raise RuntimeError(f"missing Nsight export: {self.sqlite_path}")
        with sqlite3.connect(str(self.sqlite_path)) as connection:
            ranges = self._ranges(connection)
            nsys_events = self._nsys_scheduler(connection)
            names = self._thread_names(connection)
            osrt_calls = self._osrt_calls(connection)
            work_samples = self._work_samples(connection, ranges)
        if not ranges:
            raise RuntimeError("no non-warmup preprocessing ranges found")

        target_pids = {record.pid for record in ranges}
        tid_to_pid = {
            tid: pid for _time, _cpu, _in, pid, tid in nsys_events
            if pid in target_pids
        }
        target_tids = set(tid_to_pid)
        analysis_start = min(record.start for record in ranges)
        analysis_end = max(record.end for record in ranges)
        raw = self._raw_scheduler()
        scheduler_source = "nsys_running_only"
        clock_offset = 0
        if raw:
            clock_offset = self._clock_offset(
                raw, nsys_events, target_pids
            )
            transitions = self._transition_events(
                raw, clock_offset, target_tids
            )
            intervals = self._state_intervals(
                transitions, target_tids, analysis_start, analysis_end
            )
            scheduler_source = "bpftrace"
        else:
            intervals = self._nsys_running_intervals(
                nsys_events, target_tids, analysis_start, analysis_end
            )
        activity = self._thread_activity(
            raw, clock_offset, target_tids
        )

        model_by_pid = {}
        for record in ranges:
            model_by_pid[record.pid] = record.model
        exclusive_durations = self._exclusive_durations(ranges)

        stage_rows = []
        thread_rows = []
        row_by_range = {}
        for record in ranges:
            totals = defaultdict(int)
            peer_totals = defaultdict(int)
            event_totals = defaultdict(int)
            peer_event_totals = defaultdict(int)
            calling_thread_states = defaultdict(int)
            for tid in sorted(target_tids):
                thread_pid = tid_to_pid[tid]
                own_process = thread_pid == record.pid
                states = self._overlap(
                    intervals.get(tid, []), record.start, record.end
                )
                for state, duration in states.items():
                    destination = totals if own_process else peer_totals
                    destination[state] += duration
                    if tid == record.tid:
                        calling_thread_states[state] += duration
                thread_activity = {}
                for kind in ("switches", "wakeups", "migrations"):
                    count = self._activity_count(
                        activity.get(tid, {}).get(kind, []),
                        record.start, record.end,
                    )
                    thread_activity[kind] = count
                    destination = (
                        event_totals if own_process
                        else peer_event_totals
                    )
                    destination[kind] += count
                if any(states[state] for state in (
                        "running", "runnable_wait", "blocked")):
                    thread_rows.append({
                        "model": record.model,
                        "input": record.input_id,
                        "stage": record.stage,
                        "stage_pid": record.pid,
                        "relation": "own" if own_process else "peer",
                        "thread_model": model_by_pid.get(thread_pid, ""),
                        "pid": thread_pid,
                        "tid": tid,
                        "thread_name": names.get((thread_pid, tid), ""),
                        "stage_start_ns": record.start,
                        "wall_ns": record.duration_ns,
                        "running_ns": states["running"],
                        "runnable_wait_ns": states["runnable_wait"],
                        "blocked_ns": states["blocked"],
                        "unknown_ns": states["unknown"],
                        **thread_activity,
                    })
            wall = record.duration_ns
            stage_row = {
                "model": record.model,
                "input": record.input_id,
                "stage": record.stage,
                "pid": record.pid,
                "calling_tid": record.tid,
                "start_ns": record.start,
                "end_ns": record.end,
                "wall_ns": wall,
                "exclusive_ns": exclusive_durations[record],
                "calling_thread_running_ns": (
                    calling_thread_states["running"]
                ),
                "calling_thread_runnable_wait_ns": (
                    calling_thread_states["runnable_wait"]
                ),
                "calling_thread_blocked_ns": (
                    calling_thread_states["blocked"]
                ),
                "running_thread_equivalents": (
                    totals["running"] / wall if wall else 0
                ),
                "runnable_wait_thread_equivalents": (
                    totals["runnable_wait"] / wall if wall else 0
                ),
                "blocked_thread_equivalents": (
                    totals["blocked"] / wall if wall else 0
                ),
                "unknown_thread_equivalents": (
                    totals["unknown"] / wall if wall else 0
                ),
                "peer_running_thread_equivalents": (
                    peer_totals["running"] / wall if wall else 0
                ),
                "peer_runnable_wait_thread_equivalents": (
                    peer_totals["runnable_wait"] / wall if wall else 0
                ),
                "peer_blocked_thread_equivalents": (
                    peer_totals["blocked"] / wall if wall else 0
                ),
                "switches": event_totals["switches"],
                "wakeups": event_totals["wakeups"],
                "migrations": event_totals["migrations"],
                "peer_switches": peer_event_totals["switches"],
                "peer_wakeups": peer_event_totals["wakeups"],
                "peer_migrations": peer_event_totals["migrations"],
            }
            stage_rows.append(stage_row)
            row_by_range[record] = stage_row

        grouped: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in stage_rows:
            grouped[(row["model"], row["stage"])].append(row)
        summaries = []
        materiality = float(
            self.options.get("materiality_threshold_ms", 1.0)
        )
        for (model, stage), rows in sorted(grouped.items()):
            wall_durations = [row["wall_ns"] / 1e6 for row in rows]
            exclusive = [row["exclusive_ns"] / 1e6 for row in rows]
            scheduled = [
                row["running_thread_equivalents"] +
                row["runnable_wait_thread_equivalents"]
                for row in rows
            ]
            runnable_wait = [
                row["runnable_wait_thread_equivalents"] for row in rows
            ]
            peer_scheduled = [
                row["peer_running_thread_equivalents"] +
                row["peer_runnable_wait_thread_equivalents"]
                for row in rows
            ]
            summary = {
                "model": model,
                "stage": stage,
                "action": _action_name(stage),
                "samples": len(rows),
                "exclusive_p95_ms": _percentile(exclusive, 0.95),
                "exclusive_p99_ms": _percentile(exclusive, 0.99),
                "wall_p95_ms": _percentile(wall_durations, 0.95),
                "wall_p99_ms": _percentile(wall_durations, 0.99),
                "scheduled_p95": _percentile(scheduled, 0.95),
                "scheduled_p99": _percentile(scheduled, 0.99),
                "runnable_wait_p95": _percentile(
                    runnable_wait, 0.95
                ),
                "runnable_wait_p99": _percentile(
                    runnable_wait, 0.99
                ),
                "peer_scheduled_p95": _percentile(
                    peer_scheduled, 0.95
                ),
                "peer_scheduled_p99": _percentile(
                    peer_scheduled, 0.99
                ),
                "p99_reliable": len(rows) >= 100,
            }
            summary["material"] = (
                summary["exclusive_p95_ms"] >= materiality or
                summary["exclusive_p99_ms"] >= materiality
            )
            summaries.append(summary)

        lifecycle_rows = self._lifecycle_rows(
            raw, clock_offset, tid_to_pid, names, nsys_events,
            osrt_calls,
        )
        wait_rows = self._wait_rows(osrt_calls, ranges)
        manifest = self._manifest()
        topology = self._cpu_topology(manifest)
        pools_by_pid, tid_to_pool = self._pool_inventory(lifecycle_rows)
        pool_statistics = self._pool_stage_statistics(
            thread_rows, tid_to_pool
        )
        processes, process_by_pid = self._process_inventory(
            manifest, topology, model_by_pid, lifecycle_rows,
            pools_by_pid, pool_statistics,
        )
        process_by_model = {
            process["model"]: process for process in processes
        }
        libraries_by_action = self._action_libraries(work_samples)
        material_summaries = [
            summary for summary in summaries if summary["material"]
        ]
        material_by_model: DefaultDict[
            str, List[Dict[str, Any]]
        ] = defaultdict(list)
        for summary in material_summaries:
            material_by_model[summary["model"]].append(summary)
        for values in material_by_model.values():
            values.sort(
                key=lambda item: max(
                    item["exclusive_p95_ms"],
                    item["exclusive_p99_ms"],
                ),
                reverse=True,
            )

        trace_quality = 1.0 if scheduler_source == "bpftrace" else 0.4
        findings = []
        intra_models = set()

        # Mechanism 1: one identified compute pool is larger than its process
        # capacity. Created or blocked threads outside that pool do not count.
        for process in processes:
            capacity = process["logical_cpu_capacity"]
            if not capacity:
                continue
            for pool in process["observed_compute_pools"]:
                team_size = pool["team_size_estimate"]
                if team_size <= capacity:
                    continue
                actions = material_by_model.get(process["model"], ())
                if not actions:
                    continue
                owner = pool["framework_owner"]
                related = [
                    action for action in actions
                    if any(
                        library["framework_owner"] == owner
                        for library in libraries_by_action.get(
                            (action["model"], action["stage"]), ()
                        )
                    )
                ]
                action = (related or list(actions))[0]
                libraries = libraries_by_action.get(
                    (action["model"], action["stage"]), []
                )
                control = self._validation_control(
                    process, self._control_key(libraries)
                )
                findings.append({
                    "id": (
                        f"single-pool:{process['model']}:"
                        f"{owner}:{action['action']}"
                    ),
                    "mechanism": "single_pool_oversubscription",
                    "scope": "affinity_local",
                    "status": "candidate",
                    "severity_ms": round(max(
                        action["exclusive_p95_ms"],
                        action["exclusive_p99_ms"],
                    ), 3),
                    "participants": [{
                        "model": process["model"],
                        "action": action["action"],
                        "stage": action["stage"],
                        "pool": pool["pool_id"],
                        "team_size_estimate": team_size,
                        "logical_cpu_capacity": capacity,
                    }],
                    "supporting_evidence": [
                        (
                            f"pool team estimate {team_size} exceeds "
                            f"{capacity} allowed logical CPUs"
                        ),
                        (
                            f"exclusive p95/p99 is "
                            f"{action['exclusive_p95_ms']:.3f}/"
                            f"{action['exclusive_p99_ms']:.3f} ms"
                        ),
                    ],
                    "opposing_evidence": (
                        ["p95 runnable waiting is below 0.25 threads"]
                        if action["runnable_wait_p95"] < 0.25 else []
                    ),
                    "missing_evidence": [
                        "matched run with only this pool count changed"
                    ],
                    "applicable_tests": [control],
                    "confidence": _confidence(
                        trace_quality, 1.0, 1.0,
                        0.5 if action["runnable_wait_p95"] >= 0.25 else 0.0,
                        opposing_penalty=(
                            0.05 if action["runnable_wait_p95"] < 0.25
                            else 0.0
                        ),
                    ),
                })

        # Mechanism 2a: independently created pools inside one process are
        # active during a material action and exceed local capacity together.
        for process in processes:
            capacity = process["logical_cpu_capacity"]
            candidates = []
            for action in material_by_model.get(process["model"], ()):
                stats = pool_statistics.get(
                    (action["model"], action["stage"]), {}
                )
                active_pools = [
                    pool for pool in stats.get("pools", ())
                    if pool["running_p95"] +
                    pool["runnable_wait_p95"] >= 0.05
                ]
                if (
                    capacity and len(active_pools) >= 2 and
                    action["scheduled_p95"] > capacity and
                    stats.get("multi_pool_overlap_fraction", 0.0) > 0
                ):
                    candidates.append((action, stats, active_pools))
            if not candidates:
                continue
            action, stats, active_pools = max(
                candidates,
                key=lambda item: max(
                    item[0]["exclusive_p95_ms"],
                    item[0]["exclusive_p99_ms"],
                ),
            )
            intra_models.add(process["model"])
            participants = []
            for pool in active_pools:
                encoded = pool["pool_id"].split(":")
                participants.append({
                    "model": process["model"],
                    "framework_owner": encoded[1],
                    "pool_kind": encoded[2],
                    "execution_backend": encoded[3],
                    "scheduled_p95": round(
                        pool["running_p95"] +
                        pool["runnable_wait_p95"], 3
                    ),
                })
            libraries = libraries_by_action.get(
                (action["model"], action["stage"]), []
            )
            findings.append({
                "id": (
                    f"multi-pool:intra:{process['model']}:"
                    f"{action['action']}"
                ),
                "mechanism": "multi_pool_cpu_contention",
                "scope": "intra_process",
                "status": "candidate",
                "severity_ms": round(max(
                    action["exclusive_p95_ms"],
                    action["exclusive_p99_ms"],
                ), 3),
                "action": {
                    "model": process["model"],
                    "action": action["action"],
                    "stage": action["stage"],
                    "exclusive_p95_ms": round(
                        action["exclusive_p95_ms"], 3
                    ),
                    "exclusive_p99_ms": round(
                        action["exclusive_p99_ms"], 3
                    ),
                },
                "participants": participants,
                "supporting_evidence": [
                    (
                        f"scheduled p95 {action['scheduled_p95']:.2f} "
                        f"exceeds local capacity {capacity}"
                    ),
                    (
                        f"multiple identified pools overlap in "
                        f"{stats['multi_pool_overlap_fraction']:.1%} "
                        "of action samples"
                    ),
                ],
                "opposing_evidence": [],
                "missing_evidence": [
                    "matched run changing one library pool at a time"
                ],
                "applicable_tests": [
                    self._validation_control(
                        process, self._control_key(libraries)
                    )
                ],
                "confidence": _confidence(
                    trace_quality, 1.0, 1.0, 0.5
                ),
            })

        # Mechanism 2b: represent cross-process contention as an action pair.
        # Exposure means the peer action was already active at target start,
        # avoiding the worst duration-dependent overlap-selection bias.
        material_keys = {
            (summary["model"], summary["stage"])
            for summary in material_summaries
        }
        range_index = self._range_index(ranges)
        directed: DefaultDict[
            Tuple[Tuple[str, str, int], Tuple[str, str, int]],
            List[StageRange],
        ] = defaultdict(list)
        for target in ranges:
            if (target.model, target.stage) not in material_keys:
                continue
            target_process = process_by_pid.get(target.pid)
            if target_process is None:
                continue
            target_cpus = set(target_process["effective_cpus"])
            for peer_pid, peer_process in process_by_pid.items():
                if peer_pid == target.pid:
                    continue
                if not target_cpus.intersection(
                        peer_process["effective_cpus"]):
                    continue
                peer = self._indexed_range(
                    range_index, peer_pid, target.start
                )
                if peer is None or (
                        peer.model, peer.stage) not in material_keys:
                    continue
                key = (
                    (target.model, target.stage, target.pid),
                    (peer.model, peer.stage, peer.pid),
                )
                directed[key].append(target)

        minimum_overlap = int(
            self.options.get("minimum_overlap_samples", 5)
        )
        pair_impacts: DefaultDict[
            Tuple[Tuple[str, str, int], Tuple[str, str, int]],
            List[Dict[str, Any]],
        ] = defaultdict(list)
        for (target_key, peer_key), exposed in directed.items():
            if len(exposed) < minimum_overlap:
                continue
            target_model, target_stage, target_pid = target_key
            peer_model, peer_stage, peer_pid = peer_key
            all_target = [
                record for record in ranges
                if record.model == target_model and
                record.stage == target_stage
            ]
            baseline = []
            for record in all_target:
                active = self._indexed_range(
                    range_index, peer_pid, record.start
                )
                if active is None or active.stage != peer_stage:
                    baseline.append(record)
            if not baseline:
                continue
            exposed_ms = [
                exclusive_durations[record] / 1e6 for record in exposed
            ]
            baseline_ms = [
                exclusive_durations[record] / 1e6 for record in baseline
            ]
            active_p95 = _percentile(exposed_ms, 0.95)
            active_p99 = _percentile(exposed_ms, 0.99)
            baseline_p95 = _percentile(baseline_ms, 0.95)
            baseline_p99 = _percentile(baseline_ms, 0.99)
            rows = [row_by_range[record] for record in exposed]
            combined = [
                row["running_thread_equivalents"] +
                row["runnable_wait_thread_equivalents"] +
                row["peer_running_thread_equivalents"] +
                row["peer_runnable_wait_thread_equivalents"]
                for row in rows
            ]
            canonical = tuple(sorted((target_key, peer_key)))
            pair_impacts[canonical].append({
                "model": target_model,
                "action": _action_name(target_stage),
                "stage": target_stage,
                "peer_model": peer_model,
                "peer_action": _action_name(peer_stage),
                "exposed_samples": len(exposed),
                "exclusive_p95_active_ms": active_p95,
                "exclusive_p99_active_ms": active_p99,
                "exclusive_p95_without_peer_action_ms": baseline_p95,
                "exclusive_p99_without_peer_action_ms": baseline_p99,
                "conditional_p95_delta_ms": active_p95 - baseline_p95,
                "conditional_p99_delta_ms": active_p99 - baseline_p99,
                "conditional_p99_reliable": len(exposed) >= 100,
                "combined_scheduled_p95": _percentile(combined, 0.95),
                "libraries": libraries_by_action.get(
                    (target_model, target_stage), []
                ),
                "target_pid": target_pid,
                "peer_pid": peer_pid,
            })

        for pair, impacts in pair_impacts.items():
            material_impacts = [
                impact for impact in impacts
                if impact["conditional_p95_delta_ms"] >= materiality or
                impact["conditional_p99_delta_ms"] >= materiality
            ]
            if not material_impacts:
                continue
            left_process = process_by_pid[pair[0][2]]
            right_process = process_by_pid[pair[1][2]]
            shared_cpus = sorted(
                set(left_process["effective_cpus"]).intersection(
                    right_process["effective_cpus"]
                )
            )
            combined_p95 = max(
                impact["combined_scheduled_p95"] for impact in impacts
            )
            if combined_p95 <= len(shared_cpus):
                continue
            participant_models = {pair[0][0], pair[1][0]}
            tests = []
            for model, stage, _pid in pair:
                process = process_by_model[model]
                libraries = libraries_by_action.get((model, stage), [])
                control = self._validation_control(
                    process, self._control_key(libraries)
                )
                if control not in tests:
                    tests.append(control)
            opposing = [
                (
                    f"{impact['model']} {impact['action']} has less than "
                    f"{materiality:g} ms conditional tail increase"
                )
                for impact in impacts
                if impact not in material_impacts
            ]
            small_tail_groups = any(
                not impact["conditional_p99_reliable"]
                for impact in impacts
            )
            severity = max(
                max(
                    impact["conditional_p95_delta_ms"],
                    impact["conditional_p99_delta_ms"],
                )
                for impact in material_impacts
            )
            findings.append({
                "id": (
                    "multi-pool:inter:" +
                    ":".join(sorted(
                        f"{model}.{_action_name(stage)}"
                        for model, stage, _pid in pair
                    ))
                ),
                "mechanism": "multi_pool_cpu_contention",
                "scope": (
                    "mixed" if participant_models.intersection(intra_models)
                    else "inter_process"
                ),
                "status": "candidate",
                "severity_ms": round(severity, 3),
                "shared_logical_cpus": len(shared_cpus),
                "combined_scheduled_p95": round(combined_p95, 3),
                "participants": [
                    {
                        key: (
                            round(value, 3)
                            if isinstance(value, float) else value
                        )
                        for key, value in impact.items()
                        if key not in (
                            "target_pid", "peer_pid",
                            "combined_scheduled_p95", "stage", "libraries",
                        )
                    }
                    for impact in impacts
                ],
                "supporting_evidence": [
                    (
                        f"combined scheduled p95 {combined_p95:.2f} "
                        f"exceeds shared capacity {len(shared_cpus)}"
                    ),
                    "at least one directional p95/p99 increase exceeds "
                    f"{materiality:g} ms",
                ],
                "opposing_evidence": opposing,
                "missing_evidence": [
                    "matched run changing one participant pool at a time",
                    *(
                        ["conditional p99 groups have fewer than 100 samples"]
                        if small_tail_groups else []
                    ),
                ],
                "applicable_tests": tests,
                "confidence": _confidence(
                    trace_quality,
                    1.0 if all(
                        impact["libraries"] for impact in impacts
                    ) else 0.5,
                    1.0,
                    1.0,
                    opposing_penalty=(
                        0.05 if small_tail_groups else 0.0
                    ),
                ),
            })

        # Mechanism 3: this is an observational SMT/topology candidate. A
        # physical-core-only rerun is required before claiming causality.
        cpu_records = topology.get("cpus", ())
        for process in processes:
            logical = process["logical_cpu_capacity"]
            physical = process["physical_core_capacity"]
            if not physical or physical >= logical:
                continue
            candidates = [
                action for action in material_by_model.get(
                    process["model"], ()
                )
                if action["scheduled_p95"] > physical and
                action["runnable_wait_p95"] >= 0.25
            ]
            if not candidates:
                continue
            action = candidates[0]
            representatives = []
            seen = set()
            for item in cpu_records:
                if item["cpu"] not in process["effective_cpus"]:
                    continue
                core = (item.get("package", 0), item.get("core"))
                if core not in seen:
                    seen.add(core)
                    representatives.append(item["cpu"])
            findings.append({
                "id": (
                    f"cpu-topology:{process['model']}:{action['action']}"
                ),
                "mechanism": "cpu_topology_contention",
                "scope": "process",
                "status": "candidate",
                "severity_ms": round(max(
                    action["exclusive_p95_ms"],
                    action["exclusive_p99_ms"],
                ), 3),
                "participants": [{
                    "model": process["model"],
                    "action": action["action"],
                    "logical_cpu_capacity": logical,
                    "physical_core_capacity": physical,
                    "scheduled_p95": round(
                        action["scheduled_p95"], 3
                    ),
                }],
                "supporting_evidence": [
                    (
                        f"{logical} logical CPUs represent only "
                        f"{physical} physical cores"
                    ),
                    (
                        f"scheduled p95 {action['scheduled_p95']:.2f} "
                        "exceeds physical-core capacity"
                    ),
                ],
                "opposing_evidence": [
                    "single run cannot separate SMT from pool contention"
                ],
                "missing_evidence": [
                    "physical-core-only affinity comparison"
                ],
                "applicable_tests": [{
                    "type": "physical_core_only_affinity",
                    "path": f"models.{process['model']}.cpu_affinity",
                    "value": representatives,
                    "keep_other_controls_unchanged": True,
                    "success": (
                        "same action p95 or p99 improves by at least 1 ms"
                    ),
                }],
                "confidence": _confidence(
                    trace_quality, 1.0, 0.75, 0.0,
                    opposing_penalty=0.05,
                ),
            })

        findings.sort(
            key=lambda item: (
                -item["confidence"]["score"],
                -item["severity_ms"],
                item["id"],
            )
        )
        all_findings = findings
        finding_limit = int(self.options.get("max_findings", 4))
        findings = []
        selected_signatures = set()
        for finding in all_findings:
            signature = (
                finding["mechanism"],
                finding["scope"],
            )
            if signature in selected_signatures:
                continue
            findings.append(finding)
            selected_signatures.add(signature)
            if len(findings) >= finding_limit:
                break
        findings.sort(
            key=lambda item: (
                -item["confidence"]["score"],
                -item["severity_ms"],
                item["id"],
            )
        )
        self._write_csv(
            self.output_directory / "preprocess_stage_metrics.csv",
            stage_rows,
            [
                "model", "input", "stage", "pid", "calling_tid",
                "start_ns", "end_ns", "wall_ns", "exclusive_ns",
                "calling_thread_running_ns",
                "calling_thread_runnable_wait_ns",
                "calling_thread_blocked_ns",
                "running_thread_equivalents",
                "runnable_wait_thread_equivalents",
                "blocked_thread_equivalents",
                "unknown_thread_equivalents",
                "peer_running_thread_equivalents",
                "peer_runnable_wait_thread_equivalents",
                "peer_blocked_thread_equivalents",
                "switches", "wakeups", "migrations",
                "peer_switches", "peer_wakeups", "peer_migrations",
            ],
        )
        self._write_csv(
            self.output_directory / "preprocess_thread_states.csv",
            thread_rows,
            [
                "model", "input", "stage", "stage_pid", "relation",
                "thread_model", "pid", "tid", "thread_name",
                "stage_start_ns",
                "wall_ns", "running_ns", "runnable_wait_ns",
                "blocked_ns", "unknown_ns",
                "switches", "wakeups", "migrations",
            ],
        )
        self._write_csv(
            self.output_directory / "thread_lifecycle.csv",
            lifecycle_rows,
            [
                "pid", "tid", "name", "parent_tid", "created_ns",
                "creation_api", "creator_owner", "creator_stack",
                "first_scheduled_ns", "last_scheduled_ns", "exited_ns",
                "cpus",
            ],
        )
        self._write_csv(
            self.output_directory / "preprocess_waits.csv",
            wait_rows,
            [
                "model", "input", "stage", "pid", "tid", "api",
                "overlap_ns", "owner", "stack",
            ],
        )
        self._write_csv(
            self.output_directory / "preprocess_work_samples.csv",
            work_samples,
            [
                "model", "stage", "pid", "tid", "symbol", "module",
                "samples",
            ],
        )
        summary_fields = [
            "model", "stage", "action", "samples",
            "exclusive_p95_ms", "exclusive_p99_ms",
            "wall_p95_ms", "wall_p99_ms",
            "scheduled_p95", "scheduled_p99",
            "runnable_wait_p95", "runnable_wait_p99",
            "peer_scheduled_p95", "peer_scheduled_p99",
            "p99_reliable", "material",
        ]
        self._write_csv(
            self.output_directory / "preprocess_stage_summary.csv",
            summaries, summary_fields,
        )

        mechanisms = (
            "single_pool_oversubscription",
            "multi_pool_cpu_contention",
            "cpu_topology_contention",
            "external_process_preemption",
        )
        mechanism_coverage = []
        for mechanism in mechanisms:
            matching = [
                finding for finding in all_findings
                if finding["mechanism"] == mechanism
            ]
            if matching:
                selected_matching = [
                    finding for finding in findings
                    if finding["mechanism"] == mechanism
                ]
                coverage = {
                    "mechanism": mechanism,
                    "status": "candidate",
                    "candidate_count": len(matching),
                    "finding_ids": [
                        finding["id"] for finding in selected_matching
                    ],
                }
                omitted_count = len(matching) - len(selected_matching)
                if omitted_count:
                    coverage["omitted_count"] = omitted_count
                if mechanism == "multi_pool_cpu_contention":
                    coverage["scopes"] = sorted({
                        finding["scope"] for finding in matching
                    })
            elif mechanism == "external_process_preemption":
                coverage = {
                    "mechanism": mechanism,
                    "status": "insufficient_evidence",
                    "missing_evidence": (
                        "system-wide scheduler attribution outside the "
                        "profiled process tree"
                    ),
                    "applicable_test": {
                        "type": "isolated_cpu_comparison",
                        "success": (
                            "same action p95 or p99 improves by at least 1 ms"
                        ),
                    },
                }
            else:
                coverage = {
                    "mechanism": mechanism,
                    "status": "no_supporting_evidence",
                }
            mechanism_coverage.append(coverage)

        max_actions = int(
            self.options.get("max_material_actions", 8)
        )
        selected_actions = sorted(
            material_summaries,
            key=lambda item: max(
                item["exclusive_p95_ms"],
                item["exclusive_p99_ms"],
            ),
            reverse=True,
        )[:max_actions]
        planner_actions = []
        for action in selected_actions:
            planner_actions.append({
                "model": action["model"],
                "stage": action["stage"],
                "action": action["action"],
                "samples": action["samples"],
                "exclusive_p95_ms": round(
                    action["exclusive_p95_ms"], 3
                ),
                "exclusive_p99_ms": round(
                    action["exclusive_p99_ms"], 3
                ),
                "scheduled_p95": round(action["scheduled_p95"], 3),
                "scheduled_p99": round(action["scheduled_p99"], 3),
                "runnable_wait_p95": round(
                    action["runnable_wait_p95"], 3
                ),
                "peer_scheduled_p95": round(
                    action["peer_scheduled_p95"], 3
                ),
                "p99_reliable": action["p99_reliable"],
                "libraries": libraries_by_action.get(
                    (action["model"], action["stage"]), []
                ),
            })

        control_records = []
        for process in processes:
            prefix = f"models.{process['model']}"
            control_records.append({
                "model": process["model"],
                "fallback": {
                    "path": f"{prefix}.cpu_thread_count",
                    "current": process["cpu_thread_count"],
                },
                "per_library_prefix": (
                    f"{prefix}.library_thread_counts"
                ),
                "supported_library_keys": [
                    "pytorch_intraop", "pytorch_interop", "opencv",
                    "openmp", "mkl", "openblas", "numexpr",
                ],
                "current": process["library_thread_counts"],
            })

        primary_mechanisms = sorted({
            finding["mechanism"] for finding in findings
        })
        result = {
            "schema_version": 2,
            "skill": "preprocess_contention",
            "profiler": "preprocess_contention_v1",
            "analyzer": self.analyzer_id,
            "run_id": manifest.get(
                "run_id", self.run_directory.name
            ),
            "objective": {
                "metrics": [
                    "exclusive_p95_ms", "exclusive_p99_ms"
                ],
                "materiality_threshold_ms": materiality,
                "scheduled_demand_definition": (
                    "running + runnable_wait; blocked is excluded"
                ),
            },
            "scheduler_source": scheduler_source,
            "scheduler_clock_offset_ns": clock_offset,
            "local_assessment": {
                "status": (
                    "contention_candidates" if findings
                    else "no_supported_cpu_contention"
                ),
                "primary_mechanisms": primary_mechanisms,
                "planner_must_validate": bool(findings),
            },
            "system": {
                "logical_cpus": topology.get("logical_cpu_count", 0),
                "physical_cores": topology.get("physical_core_count", 0),
                "smt_present": (
                    topology.get("logical_cpu_count", 0) >
                    topology.get("physical_core_count", 0)
                ),
            },
            "processes": processes,
            "material_actions": planner_actions,
            "findings": findings,
            "mechanism_coverage": mechanism_coverage,
            "controls": control_records,
            "evidence_quality": {
                "trace_quality": (
                    "complete_target_state_reconstruction"
                    if scheduler_source == "bpftrace"
                    else "running_state_only"
                ),
                "thread_scope": "all profiled model-process threads",
                "external_process_attribution": "incomplete",
                "conditional_p99_minimum_samples": 100,
                "notes": [
                    (
                        "framework owner and execution backend are separate; "
                        "PyTorch/libgomp threads are counted once"
                    ),
                    (
                        "action overlap is observational and cannot replace "
                        "a matched control run"
                    ),
                ],
            },
            "other_analyzers": {
                "gpu": "separate_analyzer_required",
            },
            "artifacts": {
                "stage_metrics": "preprocess_stage_metrics.csv",
                "stage_summary": "preprocess_stage_summary.csv",
                "thread_states": "preprocess_thread_states.csv",
                "thread_lifecycle": "thread_lifecycle.csv",
                "wait_calls": "preprocess_waits.csv",
                "work_samples": "preprocess_work_samples.csv",
            },
        }
        output_path = self.output_directory / "preprocess_analysis.json"
        output_path.write_text(
            json.dumps(
                result, separators=(",", ":"), sort_keys=True
            ) + "\n",
            encoding="utf-8",
        )
        return result


class PreprocessingAnalyzer:
    """Analyze preprocessing ranges, waits, pools, and scheduler evidence."""

    def analyze(self, source, output_root, options=None) -> dict:
        """Analyze a preprocessing-scoped run and return its summary."""
        options = dict(options or {})
        manifest = load_manifest(source, output_root)
        require(manifest, scope="preprocessing")
        run = Path(manifest["run_directory"])
        destination = result_directory(
            output_root, "preprocessing", manifest)
        scheduler = run / "scheduler_events.csv"
        if options.get("scheduler_conclusions") and not scheduler.is_file():
            raise EvidenceError(
                "scheduler conclusions require scheduler_events.csv evidence"
            )
        result = {
            "analysis_type": "preprocessing",
            "run_id": manifest.get("run_id"),
            "recording": manifest["recording"],
            "scheduler_evidence": scheduler.is_file(),
            "evidence": sorted(
                path.name for path in run.iterdir() if path.is_file()
            ),
            "options": options,
        }
        if (run / "profile.sqlite").is_file():
            result["details"] = PreprocessAnalyzer(
                run, options, destination).analyze()
        return write_result(output_root, "preprocessing", manifest, result)
