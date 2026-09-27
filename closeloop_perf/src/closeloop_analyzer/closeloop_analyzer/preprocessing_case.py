"""Offline fixed-input CPU contention report; never launches workloads."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
from scipy.stats import spearmanr

from .preprocessing_analyzer import PreprocessAnalyzer, decode_global_id
from .sample_filter import filter_frames
from .scheduler_evidence import namespace_mapping, normalize_nsys, alignment_quality, trace_diagnostics

MODELS = ('vit-upernet', 'centerpoint')
CONDITIONS = ('default', 'cpu', 'threads', 'both')
LABELS = ('Default', 'CPU\nassignment', 'Thread\nlimits', 'Both')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if rows:
        with Path(path).open('w') as out:
            writer = csv.DictWriter(out, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def frame_join(connection, run):
    """Join completed inference, preprocessing and callback evidence by invocation."""
    nvtx = defaultdict(dict)
    for start, end, text, gid in connection.execute(
            "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND text LIKE 'closeloop:%'"):
        tag = PreprocessAnalyzer._tag(text)
        if tag.get('event') not in ('inference', 'preprocess') or str(tag.get('input', '')).startswith('warmup-'):
            continue
        key = (tag['model'], str(tag['input']))
        event = tag['event']
        if event in nvtx[key]:
            raise ValueError(f'duplicate invocation range: {key}/{event}')
        nvtx[key][event] = (int(start), int(end), *decode_global_id(gid))
    frames = []
    for path in sorted(Path(run).glob('model_*_inputs.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row.get('completed'):
                continue
            key = (row['model_id'], str(row['input_id']))
            stages = nvtx[key]
            if set(stages) != {'preprocess', 'inference'} or not row.get('inference_completion_monotonic_ns'):
                raise ValueError(f'CUDA-completed frame join missing: {key}')
            pre, inf = stages['preprocess'], stages['inference']
            frame_ms = (row['inference_completion_monotonic_ns'] - row['model_callback_entry_monotonic_ns']) / 1e6
            pre_ms = (pre[1] - pre[0]) / 1e6
            if frame_ms < pre_ms:
                raise ValueError('frame latency shorter than preprocessing')
            frames.append({'model': key[0], 'input_id': key[1], 'source_frame_id': row['source_frame_id'],
                           'occurrence_id': row['occurrence_id'], 'scene': row['source_scene'],
                           'latency_ms': (inf[1] - inf[0]) / 1e6, 'preprocessing_ms': pre_ms,
                           'frame_ms': frame_ms, 'frame_without_preprocessing_ms': frame_ms - pre_ms,
                           'pre_start_ns': pre[0], 'pre_end_ns': pre[1], 'pid': pre[2], 'tid': pre[3]})
    if len({(r['model'], r['input_id']) for r in frames}) != len(frames):
        raise ValueError('duplicate completed invocation')
    return frames


def retain_once(frames):
    groups = defaultdict(list)
    for row in frames:
        groups[row['model'], row['scene']].append(row)
    retained, audits = [], {}
    for (model, scene), rows in sorted(groups.items()):
        selected, audit = filter_frames(rows)
        audit.update(original_unique_source_count=len({r['source_frame_id'] for r in rows}),
                     retained_invocations=[r['input_id'] for r in selected],
                     retained_occurrences=[r['occurrence_id'] for r in selected],
                     excluded_occurrences=[r['occurrence_id'] for r in rows if r not in selected])
        audits[model + '/' + scene] = audit
        retained.extend(selected)
    return retained, audits


def association(rows, left, right):
    a, b = [r[left] for r in rows], [r[right] for r in rows]
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(spearmanr(a, b).statistic)


def tail_value(value, count):
    """The frozen protocol requires 100 retained observations for tail reporting."""
    return float(value) if count >= 100 else None


def format_value(value):
    return "NA" if value is None else f"{value:.3f}"


def metrics(rows):
    p1, p50, p99 = np.percentile([r['preprocessing_ms'] for r in rows], [1, 50, 99])
    result = {'count': len(rows), 'tail_reporting_allowed': len(rows) >= 100, 'preprocessing_p50_ms': float(p50),
              'preprocessing_p99_minus_p50_ms': tail_value(p99 - p50, len(rows)),
              'preprocessing_p99_minus_p1_ms': tail_value(p99 - p1, len(rows)),
              'retained_occurrences': [r['occurrence_id'] for r in rows],
              'retained_invocations': [r['input_id'] for r in rows]}
    for key in ('calling_running_ms', 'calling_runnable_wait_ms', 'calling_blocked_ms',
                'pool_running_ms', 'pool_runnable_wait_ms', 'pool_blocked_ms'):
        result[key + '_p50'] = float(np.percentile([r[key] for r in rows], 50))
        result[key + '_p99'] = tail_value(np.percentile([r[key] for r in rows], 99), len(rows))
    for right in ('frame_ms', 'frame_without_preprocessing_ms', 'calling_runnable_wait_ms', 'pool_runnable_wait_ms'):
        result['spearman_preprocessing_' + right] = association(rows, 'preprocessing_ms', right)
    return result


def analyze_run(entry, output):
    """Preserve full exports, then diagnose only once-retained invocations."""
    started = time.monotonic()
    run = Path(entry['artifact_directory'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    analyzer = PreprocessAnalyzer(run, output_directory=output)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if manifest['state'] != 'success':
        raise ValueError('execution did not complete successfully')
    if sha256(entry['config_path']) != entry['config_sha256']:
        raise ValueError('configuration hash mismatch')
    for name, expected in entry['artifact_hashes'].items():
        if sha256(run / name) != expected:
            raise ValueError('immutable artifact changed: ' + name)
    with sqlite3.connect(run / 'profile.sqlite') as connection:
        ranges = analyzer._ranges(connection)
        events = analyzer._nsys_scheduler(connection)
        frames = frame_join(connection, run)
        names = analyzer._thread_names(connection)
        calls = analyzer._osrt_calls(connection)
        cpu_timings = []
        for (text,) in connection.execute("SELECT text FROM NVTX_EVENTS WHERE text LIKE 'closeloop:%'"):
            tag = analyzer._tag(text)
            if tag.get('event') == 'cpu_timing' and not str(tag.get('input', '')).startswith('warmup-'):
                cpu_timings.append({'model': tag['model'], 'input_id': str(tag['input']),
                                    'stage': tag.get('module', tag['timed_event']),
                                    'timed_event': tag['timed_event'],
                                    'wall_time_ns': tag['wall_time_ns'],
                                    'thread_time_ns': tag['thread_time_ns']})
        diagnostics = trace_diagnostics(connection, (run / 'runner.log').read_text())
    rejection_reasons = [] if diagnostics['valid'] else ['Nsight/BPF reported trace loss; causal attribution rejected']
    write_csv(output / 'full_frames.csv', frames)
    write_csv(output / 'full_cpu_timings.csv', cpu_timings)
    retained, audits = retain_once(frames)
    write_json(output / 'sample_filters.json', audits)
    mapping, mapping_rows = namespace_mapping(analyzer._raw_scheduler(event_types={'mapping'}))
    write_json(output / 'thread_mapping.json', mapping_rows)
    ranges, events, identity = normalize_nsys(ranges, events, mapping)
    names = {identity(*key): name for key, name in names.items() if key in mapping or key in mapping.values()}
    calls = [call for call in calls if (call['pid'], call['tid']) in mapping or (call['pid'], call['tid']) in mapping.values()]
    for call in calls:
        call['pid'], call['tid'] = identity(call['pid'], call['tid'])
    for frame in retained:
        frame['pid'], frame['tid'] = identity(frame['pid'], frame['tid'])
    pids = {r['pid'] for r in retained}
    model_tids = {tid for pid, tid in mapping.values() if pid in pids}
    raw = analyzer._raw_scheduler(target_tids=model_tids)
    offset, alignment = alignment_quality(raw, events, pids, strict=False)
    if not all(item['valid'] for item in alignment.values()):
        rejection_reasons.append('scheduler clock alignment failed; attribution is exploratory only')
    tid_to_pid = {tid: pid for pid, tid in mapping.values() if pid in pids}
    lifecycle = analyzer._lifecycle_rows(raw, offset, tid_to_pid, names, events, calls)
    pools, pool_by_tid = analyzer._pool_inventory(lifecycle)
    write_csv(output / 'thread_lifecycle.csv', lifecycle)
    write_json(output / 'worker_pools.json', pools)
    write_csv(output / 'full_stages.csv', [asdict(r) for r in ranges])
    start, end = min(r.start for r in ranges), max(r.end for r in ranges)
    tids = set(tid_to_pid)
    transitions = analyzer._transition_events(raw, offset, tids)
    intervals = analyzer._state_intervals(transitions, tids, start, end)
    state_rows, stage_rows, known = [], [], defaultdict(lambda: [0, 0])
    range_index = defaultdict(list)
    for r in ranges:
        range_index[r.model, r.input_id].append(r)
    for frame in retained:
        pid, calling = frame['pid'], frame['tid']
        workers = {tid for tid in pool_by_tid if tid_to_pid[tid] == pid and tid != calling}
        for scope, targets in (('calling', {calling}), ('pool', workers)):
            sums = defaultdict(int)
            for tid in sorted(targets):
                states = analyzer._overlap(intervals[tid], frame['pre_start_ns'], frame['pre_end_ns'])
                for state, duration in states.items():
                    sums[state] += duration
                known[frame['model']][0] += sum(states.values()) - states['unknown']
                known[frame['model']][1] += sum(states.values())
                state_rows.append({'model': frame['model'], 'input_id': frame['input_id'],
                                   'occurrence_id': frame['occurrence_id'], 'scope': scope, 'pid': pid,
                                   'tid': tid, 'pool': pool_by_tid.get(tid, ''), **states})
            for state in ('running', 'runnable_wait', 'blocked', 'unknown'):
                frame[scope + '_' + state + '_ms'] = sums[state] / 1e6
        frame['identified_workers'] = len(workers)
        for r in range_index[frame['model'], frame['input_id']]:
            stage_rows.append({**asdict(r), 'occurrence_id': frame['occurrence_id'],
                               'wall_ms': r.duration_ns / 1e6,
                               **analyzer._overlap(intervals[r.tid], r.start, r.end)})
    write_csv(output / 'retained_frames.csv', retained)
    write_csv(output / 'retained_thread_states.csv', state_rows)
    write_csv(output / 'retained_stages.csv', stage_rows)
    retained_keys = {(r['model'], r['input_id']) for r in retained}
    write_csv(output / 'retained_cpu_timings.csv', [r for r in cpu_timings if (r['model'], r['input_id']) in retained_keys])
    stage_groups = defaultdict(list)
    for row in stage_rows:
        stage_groups[row['model'], row['stage']].append(row)
    transform_summary = []
    for (model, stage), rows in sorted(stage_groups.items()):
        p1, p50, p99 = np.percentile([r['wall_ms'] for r in rows], [1, 50, 99])
        transform_summary.append({'model': model, 'stage': stage,
                                  'scope': 'internal_data_preprocessor' if stage.startswith('data_preprocessor') else 'cpu_preprocessing',
                                  'count': len(rows), 'p50_ms': float(p50),
                                  'p99_minus_p50_ms': tail_value(p99 - p50, len(rows)),
                                  'p99_minus_p1_ms': tail_value(p99 - p1, len(rows)),
                                  'retained_invocations': json.dumps([r['input_id'] for r in rows]),
                                  'retained_occurrences': json.dumps([r['occurrence_id'] for r in rows])})
    write_csv(output / 'transform_summary.csv', transform_summary)
    coverage = {model: values[0] / values[1] for model, values in known.items()}
    if any(v < .99 for v in coverage.values()):
        rejection_reasons.append('known scheduler-state coverage below 99%: ' + str(coverage))
    summary = {}
    for model in MODELS:
        rows = [r for r in retained if r['model'] == model]
        if len(rows) < 100:
            rejection_reasons.append('fewer than 100 retained observations; tail statistics withheld: ' + model)
        summary[model] = metrics(rows)
        status = manifest['models'][model]
        settings = status['effective_cpu_settings']
        if settings['gpu_clocks'] != '3105, 10501':
            rejection_reasons.append(model + ' GPU clock readback differs: ' + settings['gpu_clocks'])
        if not status['fixed_preprocessing']['equal'] or status.get('inference_resize_scale', [512, 512]) != [512, 512]:
            raise ValueError('fixed preprocessing contract failed')
    result = {'run_id': entry['run_id'], 'block': entry['block'], 'condition': entry['condition'],
              'analysis_sha256': {p.name: sha256(p) for p in
                                  (Path(__file__), Path(__file__).with_name('sample_filter.py'),
                                   Path(__file__).with_name('scheduler_evidence.py'),
                                   Path(__file__).with_name('preprocessing_analyzer.py'))},
              'quality': {'valid': not rejection_reasons, 'rejection_reasons': rejection_reasons, 'alignment': alignment, 'offset_ns': offset,
                          'known_state_coverage': coverage, 'trace_diagnostics': diagnostics},
              'models': summary, 'sample_filters': audits,
              'effective_settings': {m: manifest['models'][m]['effective_cpu_settings'] for m in MODELS},
              'actual_completions': {m: manifest['models'][m]['inputs'] for m in MODELS},
              'replay_and_drain': json.loads((run / 'testbed_result.json').read_text()),
              'recording_cost': {'elapsed_seconds': entry['elapsed_seconds'], 'artifact_bytes': entry['artifact_bytes'],
                                 'analysis_seconds': time.monotonic() - started,
                                 'overhead_relative_to_uninstrumented': 'not estimated; all executions identically recorded'}}
    write_json(output / 'result.json', result)
    roles = {}
    for role, namespace_pid in re.findall(
            r'\[(relay_node|replayer_node)-\d+\]: process started with pid \[(\d+)\]',
            (run / 'runner.log').read_text()):
        roles.update({r['host_pid']: role for r in mapping_rows if r['namespace_pid'] == int(namespace_pid)})
    return result, retained, {'intervals': intervals, 'run': str(run), 'offset': offset, 'roles': roles,
                              'mapping': mapping_rows, 'pool_by_tid': pool_by_tid}


def competitors(frame, trace, model_pids):
    """Who occupied eligible CPUs during this retained invocation's waits?"""
    start, end = frame['pre_start_ns'], frame['pre_end_ns']
    mapped = {r['host_tid']: r for r in trace['mapping']}
    def category(tid, info, name):
        if tid == 0:
            return 'idle'
        if name.startswith(('[NSys', '*')) or name in ('bpftrace', 'nsys'):
            return 'recording'
        pid = info.get('host_pid')
        return trace['roles'].get(pid, model_pids.get(pid, 'experiment_control' if info else 'external'))
    cpu_state, rows = {}, []
    raw = PreprocessAnalyzer(Path(trace['run']))._raw_scheduler(
        event_types={'switch'}, start_ns=start + trace['offset'] - 100_000_000,
        end_ns=end + trace['offset'] + 1_000_000)
    for e in raw:
        if e.event != 'switch' or e.cpu not in range(16):
            continue
        at = e.timestamp_ns - trace['offset']
        if at > end:
            continue
        previous = cpu_state.get(e.cpu)
        if previous and at > start:
            since, tid = previous
            begin, finish = max(start, since), min(end, at)
            if finish > begin:
                info = mapped.get(tid, {})
                pid = info.get('host_pid')
                name = info.get('name', e.comm if e.prev_tid == tid else '')
                rows.append({'model': frame['model'], 'input_id': frame['input_id'], 'occurrence_id': frame['occurrence_id'],
                             'cpu': e.cpu, 'tid': tid, 'pid': pid, 'category': category(tid, info, name),
                             'name': name,
                             'start_ns': begin, 'end_ns': finish,
                             'continuous': e.prev_tid == tid})
        cpu_state[e.cpu] = (at, e.next_tid)
    for cpu, (since, tid) in cpu_state.items():
        begin = max(start, since)
        if begin < end:
            info = mapped.get(tid, {})
            pid = info.get('host_pid')
            name = info.get('name', '')
            rows.append({'model': frame['model'], 'input_id': frame['input_id'], 'occurrence_id': frame['occurrence_id'],
                         'cpu': cpu, 'tid': tid, 'pid': pid, 'category': category(tid, info, name), 'name': name,
                         'start_ns': begin, 'end_ns': end, 'continuous': True})
    return rows


def make_figure(results, samples, traces, output):
    """Pool only retained rows, then select the predeclared matched P95 occurrence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    by_key = {(r['block'], r['condition']): r['run_id'] for r in results}
    baseline, both = by_key[1, 'default'], by_key[1, 'both']
    base_rows = [r for r in samples[baseline] if r['model'] == 'vit-upernet']
    peer = {r['occurrence_id']: r for r in samples[both] if r['model'] == 'vit-upernet'}
    p95 = float(np.percentile([r['preprocessing_ms'] for r in base_rows], 95))
    candidates = [r for r in base_rows if r['occurrence_id'] in peer]
    selected = min(candidates, key=lambda r: (abs(r['preprocessing_ms'] - p95), r['occurrence_id']))
    paired = peer[selected['occurrence_id']]
    fig = plt.figure(figsize=(7.0, 4.5), facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=(1.35, 1), hspace=.65, wspace=.3)
    for index, model in enumerate(MODELS):
        ax = fig.add_subplot(grid[0, index])
        data = [[r['preprocessing_ms'] for result in results if result['condition'] == condition
                 for r in samples[result['run_id']] if r['model'] == model] for condition in CONDITIONS]
        violins = ax.violinplot(data, showextrema=False, widths=.8)
        for body in violins['bodies']:
            body.set_facecolor('#4c78a8'); body.set_edgecolor('#333333'); body.set_alpha(.65)
        for x, values in enumerate(data, 1):
            p25, p50, p75 = np.percentile(values, [25, 50, 75])
            ax.vlines(x, p25, p75, color='#333333', lw=1.6)
            ax.scatter(x, p50, s=15, color='#333333', zorder=3)
        ax.set_xticks(range(1, 5), LABELS, fontsize=8)
        ax.set_ylabel('CPU preprocessing (ms)', fontsize=9)
        ax.set_title('ViT–UPerNet' if index == 0 else 'CenterPoint', fontsize=10)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    ax = fig.add_subplot(grid[1, :])
    colors = {'running': '#4c78a8', 'runnable_wait': '#d95f02', 'blocked': '#c7c7c7', 'unknown': '#cc0000'}
    timeline = []
    ylabels, ys = [], []
    for row_index, (run_id, frame, label) in enumerate(((baseline, selected, 'Default'), (both, paired, 'Both'))):
        trace = traces[run_id]
        workers = [tid for tid in trace['pool_by_tid'] if any(m['host_tid'] == tid and m['host_pid'] == frame['pid'] for m in trace['mapping'])]
        # Representative pool worker with the largest runnable wait, tied by TID.
        worker = max(workers, key=lambda tid: (PreprocessAnalyzer._overlap(trace['intervals'][tid], frame['pre_start_ns'], frame['pre_end_ns'])['runnable_wait'], -tid)) if workers else None
        for j, (tid, role) in enumerate(((frame['tid'], 'caller'), (worker, 'pool worker'))):
            y = 3 - row_index * 2 - j
            ys.append(y); ylabels.append(label + ' ' + role)
            if tid is None:
                continue
            for segment in trace['intervals'][tid]:
                start, end = max(segment.start, frame['pre_start_ns']), min(segment.end, frame['pre_end_ns'])
                if end <= start:
                    continue
                state = segment.state
                ax.broken_barh([((start - frame['pre_start_ns']) / 1e6, (end - start) / 1e6)], (y - .32, .64), facecolors=colors.get(state, 'white'))
                timeline.append({'condition': label, 'role': role, 'tid': tid, 'state': state,
                                 'start_relative_ms': (start - frame['pre_start_ns']) / 1e6,
                                 'end_relative_ms': (end - frame['pre_start_ns']) / 1e6})
    ax.set_yticks(ys, ylabels, fontsize=8)
    ax.set_xlabel('Time since preprocessing entry (ms)', fontsize=9)
    ax.set_title('Matched ViT occurrence ' + selected['occurrence_id'], fontsize=10)
    if not all(r['quality']['valid'] for r in results):
        ax.text(.99, .98, 'Exploratory: validity checks failed', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color='#8c2d04')
    ax.tick_params(labelsize=8); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=colors[state], label=label) for state, label in
                       (('running', 'Running'), ('runnable_wait', 'Runnable waiting'), ('blocked', 'Blocked'))],
              ncol=3, loc='upper center', bbox_to_anchor=(.5, -.35), frameon=False, fontsize=8)
    fig.subplots_adjust(left=.16, right=.985, top=.95, bottom=.18)
    fig.savefig(output / 'preprocessing_case.pdf', bbox_inches='tight')
    fig.savefig(output / 'preprocessing_case.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    write_json(output / 'timeline_selection.json', {'policy': 'ViT block 1 baseline nearest retained P95 among mutually retained occurrences',
               'retained_baseline_p95_ms': p95, 'default': selected, 'both': paired, 'timeline': timeline})
    for run_id, frame in ((baseline, selected), (both, paired)):
        model_pids = {r['pid']: r['model'] for r in samples[run_id]}
        write_csv(output / (run_id + '_competitors.csv'), competitors(frame, traces[run_id], model_pids))


def report(root, baseline_only=False):
    """Keep execution findings separate, gate the affirmative conclusion."""
    root = Path(root).resolve()
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    results, samples, traces, failures = [], {}, {}, []
    for entry in manifest['executions']:
        if entry['status'] == 'planned':
            continue
        print('Analyzing ' + entry['run_id'], flush=True)
        try:
            result, frames, trace = analyze_run(entry, output / 'runs' / entry['run_id'])
            results.append(result)
            samples[entry['run_id']] = frames
            traces[entry['run_id']] = trace
        except Exception as exc:
            failures.append({'run_id': entry['run_id'], 'error': str(exc)})
        if baseline_only:
            break
    write_json(output / 'recording_quality.json', {
        'analyzed_runs': [r['run_id'] for r in results],
        'valid_runs': [r['run_id'] for r in results if r['quality']['valid']],
        'rejections': {r['run_id']: r['quality']['rejection_reasons'] for r in results if not r['quality']['valid']},
        'failures': failures})
    if baseline_only:
        if failures or not results:
            raise ValueError('baseline recording rejected: ' + str(failures))
        result = results[0]
        rows = samples[result['run_id']]
        vit = result['models']['vit-upernet']
        diagnostic = {'run_id': result['run_id'], 'status': 'candidate; interventions required',
                      'diagnosed_model': 'vit-upernet', 'endpoint': 'pool_runnable_wait_ms_p99',
                      'mechanism': 'native-pool runnable scheduling delay during CPU preprocessing',
                      'baseline_metrics': vit,
                      'identified_worker_count': max(r['identified_workers'] for r in rows if r['model'] == 'vit-upernet'),
                      'quality': result['quality'], 'selection_precedes_interventions': True}
        write_json(output / 'baseline_diagnosis.json', diagnostic)
        return diagnostic
    comparisons = []
    for block in (1, 2, 3):
        for condition in ('cpu', 'threads', 'both'):
            for model in MODELS:
                a = next((r for r in results if (r['block'], r['condition']) == (block, 'default')), None)
                b = next((r for r in results if (r['block'], r['condition']) == (block, condition)), None)
                if a is None or b is None:
                    continue
                left = {r['occurrence_id']: r for r in samples[a['run_id']] if r['model'] == model}
                right = {r['occurrence_id']: r for r in samples[b['run_id']] if r['model'] == model}
                common = sorted(left.keys() & right.keys())
                matched_spreads = {}
                for label, data in (('default', left), (condition, right)):
                    p50, p99 = np.percentile([data[k]['preprocessing_ms'] for k in common], [50, 99])
                    matched_spreads[label] = {
                        'preprocessing_p99_minus_p50_ms': tail_value(p99 - p50, len(common)),
                        'pool_runnable_wait_ms_p99': tail_value(np.percentile([data[k]['pool_runnable_wait_ms'] for k in common], 99), len(common))}
                deltas = {key: float(np.median([right[k][key] - left[k][key] for k in common]))
                          for key in ('preprocessing_ms', 'frame_ms', 'frame_without_preprocessing_ms', 'pool_runnable_wait_ms')}
                comparisons.append({'block': block, 'condition': condition, 'model': model,
                                    'matched_count': len(common), 'tail_reporting_allowed': len(common) >= 100,
                                    'retained_occurrences': common,
                                    'default_invocations': [left[k]['input_id'] for k in common],
                                    'intervention_invocations': [right[k]['input_id'] for k in common],
                                    'matched_metrics': matched_spreads,
                                    'median_paired_differences_ms': deltas,
                                    'tail_spread_difference_ms': (b['models'][model]['preprocessing_p99_minus_p50_ms'] - a['models'][model]['preprocessing_p99_minus_p50_ms']) if min(len(left), len(right)) >= 100 else None,
                                    'pool_wait_p99_difference_ms': (b['models'][model]['pool_runnable_wait_ms_p99'] - a['models'][model]['pool_runnable_wait_ms_p99']) if min(len(left), len(right)) >= 100 else None})
    primary = [r for r in comparisons if r['condition'] == 'both' and r['model'] == 'vit-upernet']
    supported = (len(results) == 12 and not failures and all(r['quality']['valid'] for r in results) and len(primary) == 3 and
                 all(r['tail_spread_difference_ms'] < 0 and r['pool_wait_p99_difference_ms'] < 0 for r in primary))
    write_json(output / 'comparisons.json', comparisons)
    write_json(output / 'results.json', {'supported': supported, 'executions': results, 'failures': failures})
    write_csv(output / 'execution_metrics.csv', [
        {'run_id': r['run_id'], 'block': r['block'], 'condition': r['condition'], 'model': model,
         'valid_for_attribution': r['quality']['valid'],
         **{key: value for key, value in m.items() if not key.startswith('retained_')},
         'retained_invocations': json.dumps(m['retained_invocations']),
         'retained_occurrences': json.dumps(m['retained_occurrences'])}
        for r in results for model, m in r['models'].items()])
    if len(results) == 12 and not failures:
        make_figure(results, samples, traces, output)
    lines = ['# CenterPoint + ViT preprocessing contention case study', '',
             'Status: ' + ('primary confirmation passed' if supported else 'inconclusive; affirmative manuscript conclusion withheld'), '',
             'Fixed synchronized scene-0770:lidar:198 and scene-0770:image:120, chosen from the middle of the cached sequence without latency inspection. '
             'The 40 s bag repeats these sources at 20/12 Hz with distinct occurrence IDs. Five warmup inferences per model; ViT resize 512×512; '
             'launch offsets 0/1 s; replay waits for readiness. MPS disabled; requested GPU clock locks 3105/10501 MHz (readbacks below). '
             'Replay/relay placement is held at CPUs 12–15. Baseline means default model CPU settings.', '',
             'CPU preprocessing is the existing preprocess range and its constituent transforms. Frame latency spans callback entry to CUDA-confirmed completion, '
             'including decoding. Delivery delay and internal data_preprocessor ranges are separate. GPU voxelization and CUDA synchronization are not CPU runnable waiting.', '',
             'P1–P99 inclusive linear cutoffs are computed once per execution/model/actual scene from original completed non-warmup inference latencies. '
             'All component statistics and correlations use those retained invocation identities. Repeated source payloads are separate occurrences. '
             'Full frame/stage exports and retained IDs are preserved alongside every per-run result. The figure pools the three separately filtered executions without re-filtering. '
             'Figure summaries show median and interquartile range. Tails are withheld for any execution/model or matched subset with fewer than 100 retained observations; pooling does not repair that gate. '
             'The lower panel uses block 1 ViT, baseline occurrence nearest its retained preprocessing P95 among occurrences also retained in Both.', '',
             '| Block | Condition | Model | Retained | P50 (ms) | P99−P50 (ms) | P99−P1 (ms) | Pool wait P99 (thread-ms) | ρ(pre,frame) | ρ(pre,frame−pre) | ρ(pre,pool wait) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for result in results:
        for model, m in result['models'].items():
            values = [m[k] for k in ('preprocessing_p50_ms', 'preprocessing_p99_minus_p50_ms', 'preprocessing_p99_minus_p1_ms', 'pool_runnable_wait_ms_p99',
                      'spearman_preprocessing_frame_ms', 'spearman_preprocessing_frame_without_preprocessing_ms', 'spearman_preprocessing_pool_runnable_wait_ms')]
            formatted = ['NA' if v is None else f'{v:.3f}' for v in values]
            lines.append(f"| {result['block']} | {result['condition']} | {model} | {m['count']} | " + ' | '.join(formatted) + ' |')
    lines += ['', 'These are configuration-policy comparisons: affinity can itself change native defaults, so CPU placement and library parallelism are not independent factorial effects. '
              'Spearman associations are within execution and descriptive; preprocessing is part of frame latency, and the frame-minus-preprocessing diagnostic is reported explicitly. '
              'Repeated frames are not statistically independent observations. Results apply only to this recorded workload and hardware.', '',
              'Recording cost includes wall duration, artifact bytes, snapshot time, and offline analysis time in each result. '
              'Relative timing perturbation was not estimated because the protocol has no uninstrumented execution; instrumentation is identical across policies.', '',
              'CPU-sharing interference context: [Elmougy et al., Diagnosing the Interference on CPU-GPU Synchronization Caused by CPU Sharing in Multi-Tenant GPU Clouds (2021)](https://doi.org/10.1109/IPCCC51483.2021.9679439). '
              'Library-pool context: [PyTorch, Optimizing LibTorch-based inference engine memory usage and thread-pooling](https://pytorch.org/blog/optimizing-libtorch/). '
              'Their cloud/LibTorch settings differ from this ROS/OpenMMLab fixed-input GPU workload.', '',
              'Recording failures: ' + json.dumps(failures)]
    lines += ['', '## Recording quality and effective settings', '',
              '| Run | Causal attribution allowed | Minimum matched switches/model | Maximum alignment P95 (µs) | Minimum known state (%) |',
              '|---|---|---:|---:|---:|']
    for result in results:
        q = result['quality']
        lines.append(f"| {result['run_id']} | {q['valid']} | "
                     f"{min(a['matched_switches'] for a in q['alignment'].values())} | "
                     f"{max(a['residual_p95_ns'] for a in q['alignment'].values()) / 1000:.3f} | "
                     f"{100 * min(q['known_state_coverage'].values()):.3f} |")
    for result in results:
        if result['quality']['rejection_reasons']:
            lines += ['', result['run_id'] + ': ' + '; '.join(result['quality']['rejection_reasons']) + '.']
    lines += ['', '| Run | Model | Completed / retained | OpenCV / Torch intra / inter | Observed graphics / memory (MHz) |',
              '|---|---|---:|---:|---:|']
    for result in results:
        for model, settings in result['effective_settings'].items():
            lines.append(f"| {result['run_id']} | {model} | {result['actual_completions'][model]} / "
                         f"{result['models'][model]['count']} | {settings['opencv_threads']} / "
                         f"{settings['pytorch_intraop_threads']} / {settings['pytorch_interop_threads']} | {settings['gpu_clocks']} |")
    lines += ['', 'Native capacities and every per-thread CPU mask are retained in each result. '
              'Nsight helper threads are identified separately and retain the inherited CPU mask. '
              'Reported pool times below sum identified worker-thread intervals; they are not elapsed wall time. '
              'Every derived row references the retained invocation and occurrence lists in its execution result.', '',
              '| Run | Model | Caller running / runnable waiting / blocked P99 (ms) | Pool running / runnable waiting / blocked P99 (thread-ms) |',
              '|---|---|---:|---:|']
    for result in results:
        for model, m in result['models'].items():
            times = [' / '.join(format_value(m[scope + '_' + state + '_ms_p99']) for state in ('running', 'runnable_wait', 'blocked'))
                     for scope in ('calling', 'pool')]
            lines.append(f"| {result['run_id']} | {model} | {times[0]} | {times[1]} |")
    lines += ['', '## Matched occurrences', '',
              'Differences are intervention minus Default within block, after intersecting the two retained occurrence sets; no re-filtering. '
              'Pooled plots do not replace these comparisons. `comparisons.json` also contains P99−P50 and pool-wait P99 on each matched subset.', '',
              '| Block | Policy | Model | Matched | Median Δ preprocessing / frame / frame−preprocessing (ms) | Full retained Δ spread / pool-wait P99 (ms) |',
              '|---|---|---|---:|---:|---:|']
    for comparison in comparisons:
        d = comparison['median_paired_differences_ms']
        deltas = ' / '.join(f'{d[key]:.3f}' for key in ('preprocessing_ms', 'frame_ms', 'frame_without_preprocessing_ms'))
        lines.append(f"| {comparison['block']} | {comparison['condition']} | {comparison['model']} | "
                     f"{comparison['matched_count']} | {deltas} | {format_value(comparison['tail_spread_difference_ms'])} / "
                     f"{format_value(comparison['pool_wait_p99_difference_ms'])} |")
    lines += ['', '## Recording cost', '',
              '| Run | Lock/run/export wall time (s) | Raw artifact size (MB) | Offline analysis (s) |',
              '|---|---:|---:|---:|']
    for result in results:
        cost = result['recording_cost']
        lines.append(f"| {result['run_id']} | {cost['elapsed_seconds']:.2f} | {cost['artifact_bytes'] / 1e6:.2f} | {cost['analysis_seconds']:.2f} |")
    lines += ['', 'These costs include initialization and export and are not estimates of latency overhead. '
              'The collector and its helper threads may themselves compete for CPU time. '
              'Raw BPF switches, Nsight diagnostics, clock-lock/reset responses, completion records and drain boundaries remain immutable under `runs/`. '
              'The preliminary baseline diagnosis was saved before the first intervention and is retained independently of this final report. '
              'Reproduction commands and the frozen order are in `closeloop_perf/studies/preprocessing_case/README.md`.']
    if supported:
        lines += ['', 'Default CPU execution settings can introduce contention among preprocessing threads, amplifying preprocessing latency variation and contributing to frame-latency variation. Our profiler localizes this variability to CPU scheduling delays and supports targeted configuration changes.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return {'supported': supported, 'analyzed_runs': len(results),
            'valid_runs': sum(r['quality']['valid'] for r in results), 'failures': failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(report(args.artifact_root, args.baseline_only)))


if __name__ == '__main__':
    main()
