"""Prepare and execute the frozen CenterPoint/ViT CPU-policy case study."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import yaml

from . import corrected_input_variation as campaign
from .config import load_run_config, schema_v2_config
from .controlled_bags import sha256, write_json
from .fixed_input_bags import construct_bag, validate_bag
from .runner import ExperimentRunner, detect_gpu_hardware
from closeloop_testbed.resource_control import GPUClockLock, detect_cpu_topology

ORDER = (('default', 'cpu', 'threads', 'both'),
         ('threads', 'both', 'default', 'cpu'),
         ('both', 'threads', 'cpu', 'default'))
MODELS = ('vit-upernet', 'centerpoint')


def condition_config(base, condition, block):
    """Apply authored policies; omission preserves inherited model defaults."""
    result = copy.deepcopy(base)
    result['run']['id'] = f'preprocessing-b{block}-{condition}'
    result['run']['experiment'] = 'fixed-input preprocessing CPU policy comparison'
    result['gpu']['mps_enabled'] = False
    result['recording']['scopes'] = ['model', 'input', 'preprocessing']
    result['recording']['preprocessing'] = {'scheduler_backend': 'bpftrace'}
    result['replay'].update(cpu_affinity=[12, 13, 14, 15], cpu_thread_count=4)
    for model in result['models']:
        for key in ('cpu_affinity', 'cpu_thread_count', 'library_thread_counts', 'mps_percentage'):
            model.pop(key, None)
        model['launch_offset_seconds'] = 0 if model['id'] == 'vit-upernet' else 1
        model['warmup_count'] = 5
        if condition in ('cpu', 'both'):
            model['cpu_affinity'] = list(range(0, 6) if model['id'] == 'vit-upernet' else range(6, 12))
        if condition in ('threads', 'both'):
            model['cpu_thread_count'] = 3
    return schema_v2_config(result)


def prepare(inventory, candidates, root):
    """Verify pinned inputs and freeze twelve configurations before measuring."""
    root = Path(root).resolve()
    ledger = root / 'experiment_manifest.json'
    if ledger.exists():
        raise ValueError('study already prepared; refusing to overwrite')
    study = campaign.load_study(inventory, check_paths=False)
    selected = [c for c in json.loads(Path(candidates).read_text())['candidates']
                if c['lidar']['source_frame_id'] == 'scene-0770:lidar:198'
                and c['image']['source_frame_id'] == 'scene-0770:image:120']
    if len(selected) != 1:
        raise ValueError('frozen synchronized pair must occur exactly once')
    source = selected[0]
    sources = {m: source[m] for m in ('lidar', 'image')}
    for name in MODELS:
        model = study['data']['models'][name]
        for p, expected in ((Path(model['model_config']), model['model_config_sha256']),
                            (study['paths']['checkpoint_root'] / model['checkpoint'], model['checkpoint_sha256'])):
            if sha256(p) != expected:
                raise ValueError(f'pinned model artifact differs: {p}')
    topology = detect_cpu_topology()
    for cpus in (set(range(6)), set(range(6, 12))):
        cores = {(r['package'], r['core']) for r in topology['cpus'] if r['cpu'] in cpus}
        assert len(cores) == 3 and {r['cpu'] for r in topology['cpus'] if (r['package'], r['core']) in cores} == cpus
    import os
    if sorted(os.sched_getaffinity(0)) != list(range(16)):
        raise ValueError('baseline must inherit CPUs 0–15')
    bag = construct_bag(sources, root / 'bags')
    study['paths']['output_root'] = root / 'runs'
    base = campaign._run_config(study, 'replaced', {'condition_id': 'scene-0770',
                                'models': list(MODELS), 'mps_enabled': False}, 0)
    base['replay'].update(controlled_bag_manifest=str(bag), controlled_bag_manifest_sha256=sha256(bag),
                          bag_directory=str(bag.parent / 'bag'), repeat_count=1)
    base['input_variation']['corruption'].update(dataset_manifest=str(bag), dataset_manifest_sha256=sha256(bag))
    config_root = root / 'generated_configs'
    config_root.mkdir(parents=True, exist_ok=True)
    executions = []
    for block, order in enumerate(ORDER, 1):
        for condition in order:
            config = condition_config(base, condition, block)
            config['input_variation']['replicate'] = block
            path = config_root / (config['run']['id'] + '.yaml')
            with path.open('x') as out:
                yaml.safe_dump(config, out, sort_keys=False)
            path.chmod(0o444)
            resolved = load_run_config(path, artifact_root=str(root))
            executions.append({'run_id': config['run']['id'], 'condition': condition, 'block': block,
                               'config_path': str(path), 'config_sha256': sha256(path),
                               'artifact_directory': str(resolved.run_directory), 'status': 'planned'})
    result = {'schema': 'preprocessing_case_v1', 'selection_policy': 'fixed middle cached pair; no latency inspection',
              'source_candidate': source, 'candidates_sha256': sha256(candidates),
              'inventory_sha256': sha256(inventory), 'bag_manifest': str(bag), 'bag_manifest_sha256': sha256(bag),
              'executions': executions, 'execution_order': [], 'cpu_topology': topology,
              'gpu': detect_gpu_hardware(0), 'automatic_retries': False}
    write_json(ledger, result)
    return result


def run(root, limit):
    """Consume each slot once, stopping after failures or before unreviewed interventions."""
    root = Path(root).resolve()
    ledger = root / 'experiment_manifest.json'
    manifest = json.loads(ledger.read_text())
    if sha256(manifest['bag_manifest']) != manifest['bag_manifest_sha256']:
        raise ValueError('frozen bag manifest changed')
    validate_bag(manifest['bag_manifest'])
    count = 0
    for entry in manifest['executions']:
        if entry['status'] != 'planned':
            continue
        if count >= limit:
            break
        if manifest['execution_order'] and not (root / 'analysis/baseline_diagnosis.json').is_file():
            raise ValueError('offline first-baseline diagnosis required before interventions')
        if sha256(entry['config_path']) != entry['config_sha256']:
            raise ValueError('frozen run configuration changed')
        config = load_run_config(entry['config_path'], artifact_root=str(root))
        clock = GPUClockLock(0, 3105, 10501)
        entry.update(status='running', started_at=datetime.now(timezone.utc).isoformat())
        manifest['execution_order'].append(entry['run_id'])
        write_json(ledger, manifest)
        start = time.monotonic()
        try:
            with clock:
                result = ExperimentRunner(config).run()
            entry['status'] = result['state']
        except BaseException as exc:
            entry.update(status='failed', error=str(exc))
            raise
        finally:
            entry.update(clock_evidence=clock.evidence, elapsed_seconds=time.monotonic() - start,
                         finished_at=datetime.now(timezone.utc).isoformat())
            directory = Path(entry['artifact_directory'])
            entry['artifact_hashes'] = {p.name: sha256(p) for p in sorted(directory.glob('*')) if p.is_file()}
            entry['artifact_bytes'] = sum(p.stat().st_size for p in directory.glob('*') if p.is_file())
            write_json(ledger, manifest)
        count += 1
    return manifest


def main(argv=None):
    """Small preparation/run entrypoint; analysis remains offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--inventory')
    parser.add_argument('--candidates')
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if not args.inventory or not args.candidates:
            parser.error('prepare requires --inventory and --candidates')
        prepare(args.inventory, args.candidates, args.artifact_root)
    else:
        run(args.artifact_root, args.limit)


if __name__ == '__main__':
    main()
