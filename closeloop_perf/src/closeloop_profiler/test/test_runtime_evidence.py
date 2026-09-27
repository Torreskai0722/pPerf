"""Default readback must observe without setting a pool."""
from types import SimpleNamespace
from closeloop_profiler.runtime_evidence import runtime_snapshot


def test_default_snapshot_reads_all_pools_without_setters(monkeypatch):
    monkeypatch.setattr('closeloop_profiler.runtime_evidence.subprocess.run',
                        lambda *a, **k: SimpleNamespace(stdout='3105, 10501'))
    result = runtime_snapshot({}, SimpleNamespace(getNumThreads=lambda: 16),
                              SimpleNamespace(get_num_threads=lambda: 10, get_num_interop_threads=lambda: 16))
    assert result['configured_limits'] == {}
    assert result['opencv_threads'] == 16
    assert result['pytorch_intraop_threads'] == 10
    assert result['pytorch_interop_threads'] == 16
    assert result['threads'] and all(t['cpu_affinity'] for t in result['threads'])
    assert 'native_pool_capacities' in result
