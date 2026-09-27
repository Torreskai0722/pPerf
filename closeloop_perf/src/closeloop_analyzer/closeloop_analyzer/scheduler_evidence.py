"""Namespace normalization and strict scheduler recording quality gates."""

from bisect import bisect_left
from dataclasses import replace
import numpy as np

from .preprocessing_analyzer import PreprocessAnalyzer


def namespace_mapping(events):
    """Return namespace-to-host identities, rejecting ambiguous PID reuse."""
    mapping = {}
    rows = []
    for e in events:
        if e.event != 'mapping':
            continue
        key = (e.target_tid, e.parent_tid)
        value = (e.prev_tid, e.next_tid)
        inode = e.child_tid & 0xffffffff
        if min(*key, *value, inode) <= 0:
            raise ValueError('invalid namespace identity')
        if key in mapping and mapping[key] != value:
            raise ValueError('ambiguous namespace identity reuse')
        mapping[key] = value
        rows.append({'timestamp_ns': e.timestamp_ns, 'host_pid': value[0],
                     'host_tid': value[1], 'namespace_pid': key[0],
                     'namespace_tid': key[1], 'namespace_inode': inode,
                     'name': e.comm})
    if not mapping:
        raise ValueError('no host/container thread mapping')
    return mapping, rows


def normalize_nsys(ranges, events, mapping):
    """Normalize Nsight namespace IDs to host IDs before any scheduler join."""
    host = set(mapping.values())
    def identity(pid, tid):
        key = (pid, tid)
        if key in mapping and key in host and mapping[key] != key:
            raise ValueError('ambiguous Nsight ID domain')
        if key in mapping:
            return mapping[key]
        if key in host:
            return key
        raise ValueError(f'unmapped Nsight identity {key}')
    converted = []
    for r in ranges:
        pid, tid = identity(r.pid, r.tid)
        converted.append(replace(r, pid=pid, tid=tid))
    normalized = [(t, cpu, inside, *identity(pid, tid))
                  for t, cpu, inside, pid, tid in events
                  if (pid, tid) in mapping or (pid, tid) in host]
    return converted, normalized, identity


def alignment_quality(raw, events, pids, strict=True):
    """Match unique switch edges per model and reject excessive residuals."""
    offset = PreprocessAnalyzer._clock_offset(raw, events, pids)
    observations = PreprocessAnalyzer._bpf_observations(raw)
    by_pid = {}
    for pid in sorted(pids):
        candidates = [e for e in events if e[3] == pid]
        # Equally spaced checks cover initialization, replay, and drain.
        indices = np.linspace(0, len(candidates) - 1, min(2000, len(candidates)), dtype=int)
        residuals, used, matched_switches = [], set(), set()
        for index in indices:
            t, cpu, inside, _, tid = candidates[index]
            values = observations.get((tid, cpu, inside), [])
            if not values:
                continue
            at = bisect_left(values, t + offset)
            nearest = min(values[max(0, at - 1):at + 1], key=lambda x: abs(x - t - offset))
            edge = (nearest, cpu, tid, inside)
            if edge in used:
                continue
            used.add(edge)
            residuals.append(abs(nearest - t - offset))
            if residuals[-1] <= 50_000:
                matched_switches.add((nearest, cpu))
        matched = sum(r <= 50_000 for r in residuals)
        p95 = float(np.percentile(residuals, 95)) if residuals else None
        by_pid[str(pid)] = {'matched_switch_edges': matched,
                            'matched_switches': len(matched_switches),
                            'sampled_switch_edges': len(residuals), 'residual_p95_ns': p95,
                            'valid': len(matched_switches) >= 50 and p95 is not None and p95 <= 50_000}
        if strict and not by_pid[str(pid)]['valid']:
            raise ValueError(f'clock alignment rejected for PID {pid}: {by_pid[str(pid)]}')
    return offset, by_pid


def trace_diagnostics(connection, log):
    """Retain native diagnostics and reject reported dropped/lost events."""
    tables = PreprocessAnalyzer._tables(connection)
    rows = []
    if 'DIAGNOSTIC_EVENT' in tables:
        cursor = connection.execute('SELECT * FROM DIAGNOSTIC_EVENT')
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor]
    import re
    messages = log.splitlines() + [str(row.get('text', '')) for row in rows]
    losses = [message for message in messages if re.search(
        r'(?:lost\s+[1-9]\d*|dropped\s+[1-9]\d*|[1-9]\d*\s+(?:events?\s+)?(?:lost|dropped)|Some events.*were lost)',
        message, re.I)]
    return {'diagnostic_events': rows, 'reported_losses': losses, 'valid': not losses}



def sanity_report(path):
    """Check a real CPU-only recorder trace before any measured model run."""
    from pathlib import Path
    import json
    import sqlite3
    path = Path(path)
    analyzer = PreprocessAnalyzer(path)
    raw = analyzer._raw_scheduler()
    mapping, rows = namespace_mapping(raw)
    with sqlite3.connect(path / 'profile.sqlite') as connection:
        ranges, events, _ = normalize_nsys(analyzer._ranges(connection), analyzer._nsys_scheduler(connection), mapping)
        diagnostics = trace_diagnostics(connection, (path / 'runner.log').read_text())
    if not diagnostics['valid']:
        raise ValueError('CPU sanity trace loss')
    if not ranges:
        raise ValueError('no CPU sanity NVTX ranges')
    offset, alignment = alignment_quality(raw, events, {r.pid for r in ranges})
    tids = {r.tid for r in ranges}
    intervals = analyzer._state_intervals(analyzer._transition_events(raw, offset, tids), tids,
                                          min(r.start for r in ranges), max(r.end for r in ranges))
    coverage = sum(r.duration_ns - analyzer._overlap(intervals[r.tid], r.start, r.end)['unknown']
                   for r in ranges) / sum(r.duration_ns for r in ranges)
    if coverage < .99:
        raise ValueError('CPU sanity coverage below 99%')
    result = {'valid': True, 'alignment': alignment, 'offset_ns': offset,
              'known_coverage': coverage, 'mapped_threads': len(mapping), 'diagnostics': diagnostics}
    (path / 'quality.json').write_text(json.dumps(result, indent=2) + '\n')
    (path / 'thread_mapping.json').write_text(json.dumps(rows, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import sys
    print(sanity_report(sys.argv[1]))
