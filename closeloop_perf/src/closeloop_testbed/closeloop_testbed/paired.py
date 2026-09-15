"""Absolute-deadline scheduling and resident subgraph replay."""

from dataclasses import asdict, dataclass
import hashlib
import json
import time


class PairedTrialError(RuntimeError):
    """Raised when a paired trial cannot meet its registered schedule."""


def paired_participant_ids(config):
    """Return every model and optional FMA client in the paired barrier."""
    result = [model["id"] for model in config["models"]]
    if "synthetic_fma" in config:
        result.append(config["synthetic_fma"]["client_id"])
    return result


def signed_deadline_offsets(delta_seconds):
    """Represent signed delta with non-negative offsets from one epoch."""
    delta = float(delta_seconds)
    target = max(0.0, -delta)
    return target, target + delta


@dataclass(frozen=True)
class PairedLaunch:
    """Timing evidence for one completed fixed-input trial."""

    trial: int
    scheduled_time: float
    actual_time: float
    completed_time: float
    quiescent_time: float
    quiescence_verified: bool = True
    deadline_missed: bool = False

    @property
    def launch_error_seconds(self):
        """Return signed launch error relative to the absolute deadline."""
        return self.actual_time - self.scheduled_time

    def to_dict(self):
        """Return a JSON-safe launch record."""
        return {
            **asdict(self),
            "launch_error_seconds": self.launch_error_seconds,
        }


class PairedTrialScheduler:
    """Execute every trial deadline or invalidate the process run."""

    def __init__(
        self,
        trials,
        period_seconds,
        deadline_offset_seconds,
        launch_tolerance_seconds=0.0005,
        monotonic=time.monotonic,
        wait=None,
        timing_diagnostics_only=False,
    ):
        self.trials = int(trials)
        self.period = float(period_seconds)
        self.deadline_offset = float(deadline_offset_seconds)
        self.launch_tolerance = float(launch_tolerance_seconds)
        self.timing_diagnostics_only = bool(timing_diagnostics_only)
        self.monotonic = monotonic
        self.wait = wait
        self.launches = []
        if self.trials < 1 or self.period <= 0:
            raise ValueError("trials and period must be positive")
        if self.deadline_offset < 0 or self.deadline_offset >= self.period:
            raise ValueError("deadline offset must be within the trial period")
        if self.launch_tolerance < 0:
            raise ValueError("launch tolerance cannot be negative")

    def run(self, epoch, callback, barrier, stop_event):
        """Run fixed trials and require both processes quiescent each time."""
        wait = self.wait or stop_event.wait
        epoch = float(epoch)
        for trial in range(self.trials):
            if stop_event.is_set():
                break
            deadline = epoch + self.deadline_offset + trial * self.period
            remaining = deadline - self.monotonic()
            if remaining > 0:
                if wait(max(0.0, remaining - 0.001)):
                    break
                while self.monotonic() < deadline:
                    if stop_event.is_set():
                        break
                if stop_event.is_set():
                    break
            actual = self.monotonic()
            error = actual - deadline
            if (abs(error) > self.launch_tolerance
                    and not self.timing_diagnostics_only):
                raise PairedTrialError(
                    f"trial {trial} launch error {error:.9f}s exceeds "
                    f"{self.launch_tolerance:.9f}s"
                )
            callback(trial, deadline, actual)
            completed = self.monotonic()
            next_boundary = epoch + (trial + 1) * self.period
            quiescence_verified = barrier(trial, next_boundary)
            if (not quiescence_verified
                    and not self.timing_diagnostics_only):
                raise PairedTrialError(
                    f"trial {trial} did not reach paired quiescence"
                )
            quiescent_time = self.monotonic()
            deadline_missed = quiescent_time >= next_boundary
            if deadline_missed and not self.timing_diagnostics_only:
                raise PairedTrialError(
                    f"trial {trial} missed the next absolute deadline"
                )
            self.launches.append(
                PairedLaunch(
                    trial, deadline, actual, completed, quiescent_time,
                    bool(quiescence_verified), deadline_missed,
                )
            )
        if not stop_event.is_set() and len(self.launches) != self.trials:
            raise PairedTrialError("paired trial sequence ended early")
        return list(self.launches)

    def summary(self):
        """Return strict schedule and quiescence evidence."""
        errors = [launch.launch_error_seconds for launch in self.launches]
        return {
            "completed_trials": len(self.launches),
            "missed_deadlines": (
                max(0, self.trials - len(self.launches))
                + sum(launch.deadline_missed for launch in self.launches)
            ),
            "max_abs_launch_error_seconds": max(map(abs, errors), default=0),
            "quiescence_verified": (
                len(self.launches) == self.trials
                and all(launch.quiescence_verified for launch in self.launches)
            ),
        }


def _detach_resident(value, torch_module):
    if torch_module.is_tensor(value):
        detached = value.detach()
        if not detached.is_cuda:
            raise PairedTrialError(
                "captured subgraph tensor is not GPU-resident"
            )
        return detached
    if isinstance(value, tuple):
        return tuple(_detach_resident(item, torch_module) for item in value)
    if isinstance(value, list):
        return [_detach_resident(item, torch_module) for item in value]
    if isinstance(value, dict):
        return {
            key: _detach_resident(item, torch_module)
            for key, item in value.items()
        }
    return value


def _resident_description(value, torch_module):
    if torch_module.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "data_ptr": int(value.data_ptr()),
        }
    if isinstance(value, tuple):
        return [
            "tuple",
            *(_resident_description(item, torch_module) for item in value),
        ]
    if isinstance(value, list):
        return [
            "list",
            *(_resident_description(item, torch_module) for item in value),
        ]
    if isinstance(value, dict):
        return {
            str(key): _resident_description(item, torch_module)
            for key, item in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    return {"type": type(value).__name__, "value": repr(value)}


class ResidentSubgraphReplay:
    """Capture a warm-up input and replay its module on the default stream."""

    def __init__(self, model, module_path, torch_module=None):
        if torch_module is None:
            # pylint: disable-next=import-outside-toplevel
            import torch as torch_module
        self.torch = torch_module
        try:
            self.module = dict(model.named_modules())[module_path]
        except KeyError as exc:
            raise PairedTrialError(
                f"subgraph module {module_path!r} was not found"
            ) from exc
        self.module_path = module_path
        self.inputs = None
        self._identity = None
        self._handle = None

    def start_capture(self):
        """Capture and detach the selected module's next warm-up input."""
        if self._handle is not None:
            return

        def capture(_module, inputs):
            if self.inputs is None:
                self.inputs = _detach_resident(inputs, self.torch)
                self._identity = _resident_description(self.inputs, self.torch)

        self._handle = self.module.register_forward_pre_hook(capture)

    def finish_capture(self):
        """Stop capture and require a stable GPU-resident input."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self.inputs is None:
            raise PairedTrialError(
                f"warm-up did not invoke {self.module_path!r}"
            )
        encoded = json.dumps(self._identity, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def replay(self, count, trial, model_id):
        """Replay the resident input and return measured CUDA time in ns."""
        count = int(count)
        if count <= 0:
            return 0
        if self.inputs is None:
            raise PairedTrialError("subgraph input has not been captured")
        cuda = self.torch.cuda
        stream = cuda.default_stream()
        current = getattr(cuda, "current_stream", None)
        if current is not None and current() != stream:
            raise PairedTrialError(
                "subgraph replay requires the default stream"
            )
        start = cuda.Event(enable_timing=True)
        end = cuda.Event(enable_timing=True)
        start.record(stream)
        for index in range(count):
            tag = "closeloop:" + json.dumps(
                {
                    "event": "compute_service_replay",
                    "input": f"paired-{trial}",
                    "model": model_id,
                    "module": self.module_path,
                    "replay_index": index,
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            cuda.nvtx.range_push(tag)
            try:
                self.module(*self.inputs)
            finally:
                cuda.nvtx.range_pop()
        end.record(stream)
        end.synchronize()
        if _resident_description(self.inputs, self.torch) != self._identity:
            raise PairedTrialError("resident subgraph input identity changed")
        return int(round(start.elapsed_time(end) * 1_000_000))

    def close(self):
        """Remove any pending capture hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
