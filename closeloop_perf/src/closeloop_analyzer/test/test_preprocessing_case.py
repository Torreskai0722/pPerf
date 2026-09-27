"""Case-study joins, one-pass filtering, and scheduler quality gates."""

import json
import sqlite3
from dataclasses import replace

import pytest

from closeloop_analyzer.preprocessing_analyzer import PreprocessAnalyzer, SchedulerEvent, StageRange
from closeloop_analyzer.preprocessing_case import frame_join, retain_once, metrics
from closeloop_analyzer.scheduler_evidence import namespace_mapping, normalize_nsys, alignment_quality


def test_namespace_mapping_preserves_external_identity_and_rejects_reuse():
    event = SchedulerEvent(0, 'mapping', 0, prev_tid=1000, next_tid=1001,
                           target_tid=10, parent_tid=11, child_tid=4026532000)
    mapping, _ = namespace_mapping([event])
    ranges, events, identity = normalize_nsys([StageRange(0, 10, 10, 11, 'vit', '0', 'preprocess')],
                                             [(0, 0, 1, 10, 11)], mapping)
    assert (ranges[0].pid, ranges[0].tid) == (1000, 1001)
    assert events == [(0, 0, 1, 1000, 1001)]
    with pytest.raises(ValueError, match='unmapped'):
        identity(10, 12)
    with pytest.raises(ValueError, match='reuse'):
        namespace_mapping([event, replace(event, next_tid=1002)])


def test_alignment_rejects_insufficient_and_shifted_edges(monkeypatch):
    monkeypatch.setattr(PreprocessAnalyzer, '_clock_offset', lambda *_: 1000000)
    raw = [SchedulerEvent(i * 1000000 + 1000000, 'switch', 0, next_tid=11) for i in range(100)]
    events = [(i * 1000000, 0, 1, 10, 11) for i in range(100)]
    assert alignment_quality(raw, events, {10})[1]['10']['matched_switch_edges'] == 100
    with pytest.raises(ValueError, match='alignment rejected'):
        alignment_quality(raw, events[:49], {10})
    with pytest.raises(ValueError, match='alignment rejected'):
        alignment_quality(raw, [(t + 100000, cpu, state, pid, tid) for t, cpu, state, pid, tid in events], {10})


def test_frame_join_and_filter_are_by_invocation_and_actual_scene(tmp_path):
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE NVTX_EVENTS (start,end,text,globalTid)')
    rows = []
    for i in range(200):
        for event, start, end in [('preprocess', 0, 1000000), ('inference', 1000000, (i + 2) * 1000000)]:
            tag = 'closeloop:' + json.dumps({'event': event, 'input': str(i), 'model': 'vit'})
            db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)', (start, end, tag, (10 << 24) | 11))
        rows.append({'model_id': 'vit', 'input_id': str(i), 'completed': True,
                     'source_frame_id': 'repeated', 'occurrence_id': f'image:{i}',
                     'source_scene': 'scene-a' if i < 100 else 'scene-b',
                     'model_callback_entry_monotonic_ns': 0, 'inference_completion_monotonic_ns': (i + 3) * 1000000})
    (tmp_path / 'model_vit_inputs.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    frames = frame_join(db, tmp_path)
    selected, audits = retain_once(frames)
    assert len(frames) == 200 and len(selected) == 196
    assert len(audits) == 2
    assert audits['vit/scene-a']['retained_unique_source_count'] == 1
    assert selected[0]['input_id'] == '1' and selected[0]['occurrence_id'] == 'image:1'
    assert selected[0]['frame_without_preprocessing_ms'] == 3
    assert frames[0]['input_id'] == '0'  # full export stays intact


def test_scheduler_waits_exclude_blocked_cuda_waits():
    raw = [SchedulerEvent(0, 'switch', 0, next_tid=10),
           SchedulerEvent(10, 'switch', 0, prev_tid=10, prev_state=1),
           SchedulerEvent(30, 'wakeup', 1, target_tid=10),
           SchedulerEvent(50, 'switch', 1, next_tid=10)]
    transitions = PreprocessAnalyzer._transition_events(raw, 0, {10})
    intervals = PreprocessAnalyzer._state_intervals(transitions, {10}, 0, 100)
    assert PreprocessAnalyzer._overlap(intervals[10], 0, 100) == {
        'running': 60, 'blocked': 20, 'runnable_wait': 20, 'unknown': 0, 'exited': 0}


def test_trace_loss_is_preserved_and_rejects_attribution():
    from closeloop_analyzer.scheduler_evidence import trace_diagnostics
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE DIAGNOSTIC_EVENT (text)')
    db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES (?)', ('Some events (1714) were lost (including Perf:1714).',))
    result = trace_diagnostics(db, '')
    assert not result['valid'] and len(result['reported_losses']) == 1


def test_scheduler_reader_filters_without_changing_raw_export(tmp_path):
    raw = ('30,switch,0,10,0,11,0,0,0,0,0,worker\n'
           '10,switch,0,99,0,98,0,0,0,0,0,external\n'
           '20,wakeup,1,0,0,0,10,0,0,0,0,worker\n')
    file = tmp_path / 'scheduler_events.csv'
    file.write_text(raw)
    events = PreprocessAnalyzer(tmp_path)._raw_scheduler(target_tids={10})
    assert [e.timestamp_ns for e in events] == [20, 30]
    assert file.read_text() == raw


def test_small_sample_preserves_occurrences_but_withholds_tails():
    rows = [{key: i for key in ('preprocessing_ms', 'frame_ms', 'frame_without_preprocessing_ms',
                               'calling_running_ms', 'calling_runnable_wait_ms', 'calling_blocked_ms',
                               'pool_running_ms', 'pool_runnable_wait_ms', 'pool_blocked_ms')}
            | {'occurrence_id': f'image:{i}', 'input_id': str(i)} for i in range(100)]
    low = metrics(rows[:99])
    assert low['count'] == 99 and len(low['retained_occurrences']) == 99
    assert low['preprocessing_p50_ms'] == 49 and not low['tail_reporting_allowed']
    assert low['preprocessing_p99_minus_p50_ms'] is None
    assert low['pool_runnable_wait_ms_p99'] is None
    assert metrics(rows)['tail_reporting_allowed']
