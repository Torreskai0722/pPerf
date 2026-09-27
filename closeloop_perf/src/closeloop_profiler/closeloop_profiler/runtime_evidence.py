"""Read effective native pools and thread masks without changing settings."""

import os
from pathlib import Path
import time
import subprocess


THREAD_ENV_PREFIXES = ('OMP_', 'MKL_', 'OPENBLAS_', 'GOTO_', 'BLIS_',
                       'NUMEXPR_', 'VECLIB_', 'KMP_', 'TBB_', 'OPENCV_',
                       'TORCH_NUM_', 'ATEN_THREADING')


def runtime_snapshot(configured, cv2_module=None, torch_module=None):
    """Capture capacities, not runnable activity; the scheduler records that."""
    if cv2_module is None:
        import cv2 as cv2_module
    if torch_module is None:
        import torch as torch_module
    from threadpoolctl import threadpool_info
    start = time.monotonic_ns()
    threads = []
    for path in sorted(Path('/proc/self/task').iterdir()):
        try:
            status = dict(line.split(':', 1) for line in
                          (path / 'status').read_text().splitlines())
            threads.append({'tid': int(path.name), 'name': status['Name'].strip(),
                            'namespace_tids': list(map(int, status['NSpid'].split())),
                            'cpu_affinity': sorted(os.sched_getaffinity(int(path.name)))})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {'phase': 'after_initialization_and_warmup',
            'pid': os.getpid(), 'pid_namespace_inode': os.stat('/proc/self/ns/pid').st_ino,
            'configured_limits': dict(configured),
            'opencv_threads': cv2_module.getNumThreads(),
            'pytorch_intraop_threads': torch_module.get_num_threads(),
            'pytorch_interop_threads': torch_module.get_num_interop_threads(),
            'native_pool_capacities': threadpool_info(), 'threads': threads,
            'gpu_clocks': subprocess.run(
                ['nvidia-smi', '--query-gpu=clocks.current.graphics,clocks.current.memory',
                 '--format=csv,noheader,nounits'], check=True, capture_output=True,
                text=True).stdout.strip(),
            'environment': {k: v for k, v in os.environ.items()
                            if k.startswith(THREAD_ENV_PREFIXES)},
            'monotonic_ns': start, 'snapshot_cost_ns': time.monotonic_ns() - start}
