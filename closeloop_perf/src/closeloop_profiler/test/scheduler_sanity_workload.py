"""CPU-only real-recorder check: native threads, wakeups, forks, and NVTX."""

import ctypes
import json
import multiprocessing
import threading
import time


def worker():
    nvtx = ctypes.CDLL('libnvToolsExt.so.1')
    for i in range(300):
        tag = {'event': 'preprocess', 'model': str(multiprocessing.current_process().name),
               'input': str(i), 'scene': 'sanity'}
        nvtx.nvtxRangePushA(('closeloop:' + json.dumps(tag)).encode())
        sum(range(15000))
        time.sleep(0.002)
        nvtx.nvtxRangePop()


def process():
    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    processes = [multiprocessing.Process(target=process, name=f'model-{i}') for i in range(2)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    assert all(p.exitcode == 0 for p in processes)
