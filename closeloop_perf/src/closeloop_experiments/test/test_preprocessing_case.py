"""Four policy conditions and frozen block order."""
from collections import Counter
from closeloop_experiments.preprocessing_case import condition_config, ORDER


def test_four_conditions_keep_replay_fixed_and_omit_unauthored_limits(monkeypatch):
    monkeypatch.setattr('closeloop_experiments.preprocessing_case.schema_v2_config', lambda x: x)
    base = {'run': {}, 'gpu': {}, 'recording': {}, 'replay': {},
            'models': [{'id': name, 'cpu_thread_count': 99, 'cpu_affinity': [9]}
                       for name in ('vit-upernet', 'centerpoint')]}
    assert Counter(c for block in ORDER for c in block) == dict.fromkeys(('default', 'cpu', 'threads', 'both'), 3)
    for condition in ORDER[0]:
        config = condition_config(base, condition, 1)
        assert config['replay']['cpu_affinity'] == [12, 13, 14, 15]
        for index, model in enumerate(config['models']):
            assert model.get('cpu_thread_count') == (3 if condition in ('threads', 'both') else None)
            assert model.get('cpu_affinity') == (list(range(index * 6, (index + 1) * 6)) if condition in ('cpu', 'both') else None)
            assert model['launch_offset_seconds'] == index
            assert model['warmup_count'] == 5
