"""Tests for strict paired scheduling and resident replay."""

import threading

import pytest

from closeloop_testbed.paired import (
    PairedTrialError,
    PairedTrialScheduler,
    ResidentSubgraphReplay,
    signed_deadline_offsets,
)


class FakeClock:
    """Monotonic clock advanced explicitly by waits and work."""

    def __init__(self, tick=0.0001, oversleep=0):
        self.now = 0.0
        self.tick = tick
        self.oversleep = oversleep

    def monotonic(self):
        self.now += self.tick
        return self.now

    def wait(self, seconds):
        self.now += seconds + self.oversleep
        return False


def test_signed_delta_five_trials_barriers_and_shutdown():
    """Signed delta becomes non-negative absolute deadlines for five pairs."""
    assert signed_deadline_offsets(-0.003) == (0.003, 0.0)
    assert signed_deadline_offsets(0.003) == (0.0, 0.003)
    clock = FakeClock()
    stop = threading.Event()
    seen = []
    scheduler = PairedTrialScheduler(
        5, 0.1, 0.003, 0.0005, clock.monotonic, clock.wait
    )

    def callback(trial, scheduled, _actual):
        seen.append((trial, scheduled))
        clock.now += 0.01

    launches = scheduler.run(
        1.0, callback, lambda _trial, _boundary: True, stop
    )
    assert [round(item[1], 3) for item in seen] == [
        1.003,
        1.103,
        1.203,
        1.303,
        1.403,
    ]
    assert len(launches) == 5
    assert scheduler.summary()["quiescence_verified"]


def test_lateness_and_absent_quiescence_invalidate_instead_of_skip():
    """Late launches and failed process barriers invalidate the whole run."""
    clock = FakeClock()
    clock.now = 1.001
    scheduler = PairedTrialScheduler(
        5, 0.1, 0, 0.0005, clock.monotonic, clock.wait
    )
    with pytest.raises(PairedTrialError, match="launch error"):
        scheduler.run(
            1.0, lambda *_args: None, lambda *_args: True, threading.Event()
        )

    clock.now = 0
    scheduler = PairedTrialScheduler(
        5, 0.1, 0, 0.0005, clock.monotonic, clock.wait
    )
    with pytest.raises(PairedTrialError, match="quiescence"):
        scheduler.run(
            0, lambda *_args: None, lambda *_args: False, threading.Event()
        )


def test_scheduler_wakes_early_before_strict_launch_deadline():
    """A normal sub-millisecond sleep overshoot stays inside tolerance."""
    clock = FakeClock(tick=0.00005, oversleep=0.0006)
    scheduler = PairedTrialScheduler(
        1, 0.1, 0, 0.0005, clock.monotonic, clock.wait
    )
    launches = scheduler.run(
        0.01, lambda *_args: None, lambda *_args: True, threading.Event()
    )
    assert abs(launches[0].launch_error_seconds) <= 0.0005


def test_timing_diagnostics_do_not_reject_trials():
    """Timing warnings remain descriptive in the controlled offset sweep."""
    clock = FakeClock()
    clock.now = 1.001
    scheduler = PairedTrialScheduler(
        1, 0.1, 0, 0.0005, clock.monotonic, clock.wait,
        timing_diagnostics_only=True,
    )
    launches = scheduler.run(
        1.0, lambda *_args: None, lambda *_args: False, threading.Event()
    )
    assert len(launches) == 1
    assert not scheduler.summary()["quiescence_verified"]


def test_pre_stopped_scheduler_does_no_work():
    """Shutdown before the epoch completes without a launch."""
    stop = threading.Event()
    stop.set()
    scheduler = PairedTrialScheduler(5, 1, 0)
    assert (
        scheduler.run(1, lambda *_args: None, lambda *_args: True, stop) == []
    )


def test_resident_tensor_is_detached_once_and_reused_for_every_replay():
    """Warm-up capture retains one GPU tensor object across K replays."""

    class Tensor:
        is_cuda = True
        shape = (2, 3)
        dtype = "float32"
        device = "cuda:0"

        def detach(self):
            return self

        def data_ptr(self):
            return id(self)

    class Handle:
        def __init__(self, hooks, hook):
            self.hooks, self.hook = hooks, hook

        def remove(self):
            self.hooks.remove(self.hook)

    class Module:
        def __init__(self):
            self.hooks, self.seen = [], []

        def register_forward_pre_hook(self, hook):
            self.hooks.append(hook)
            return Handle(self.hooks, hook)

        def __call__(self, *inputs):
            for hook in list(self.hooks):
                hook(self, inputs)
            self.seen.append(inputs)

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self, _stream):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, _other):
            return 1.25

    class Nvtx:
        range_push = staticmethod(lambda _tag: None)
        range_pop = staticmethod(lambda: None)

    stream = object()
    cuda = type(
        "Cuda",
        (),
        {
            "nvtx": Nvtx(),
            "Event": Event,
            "default_stream": staticmethod(lambda: stream),
            "current_stream": staticmethod(lambda: stream),
        },
    )()
    torch = type(
        "Torch",
        (),
        {
            "cuda": cuda,
            "is_tensor": staticmethod(lambda value: isinstance(value, Tensor)),
        },
    )()
    module = Module()
    model = type(
        "Model", (), {"named_modules": lambda self: [("backbone", module)]}
    )()
    tensor = Tensor()
    replay = ResidentSubgraphReplay(model, "backbone", torch)
    replay.start_capture()
    module(tensor)
    fingerprint = replay.finish_capture()
    assert len(fingerprint) == 64
    assert replay.replay(2, 0, "image") == 1_250_000
    assert module.seen[-2][0] is tensor and module.seen[-1][0] is tensor
