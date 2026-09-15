"""Fixed K1091 CTA-admission sweep across ordered MPS caps."""

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import yaml

from .config import load_run_config, schema_v2_config
from .mps import MPSManager
from .offline_capsule import WorkerProcess, discover_offline_native
from closeloop_testbed.resource_control import GPUClockLock
from .runner import ExperimentRunner


MPS_VALUES = (50, 60, 70, 80)
SYNTHETIC_BLOCKS = {
    "synthetic_low": 28,
    "synthetic_medium": 56,
    "synthetic_high": 112,
}
PROFILES = (*SYNTHETIC_BLOCKS, "real")
BINS = 300
K1091 = {
    "model_id": "faster_rcnn",
    "sequence": 506,
    "kernel_name": "ampere_sgemm_128x64_tn",
    "grid": [8, 16, 16],
    "block": [128, 1, 1],
    "cta_count": 2048,
}
K0265 = {
    "model_id": "deeplabv3plus",
    "sequence": 265,
    "kernel_name": (
        "sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_"
        "nhwc_tilesize128x128x16_stage3_warpsize2x2x1_g1_tensor16x8x8_"
        "execute_kernel__5x_cudnn"
    ),
    "grid": [4, 57, 8],
    "block": [128, 1, 1],
    "cta_count": 1824,
}


class SweepError(RuntimeError):
    """Raised when immutable campaign or capture evidence is invalid."""


@dataclass(frozen=True)
class Condition:
    """One logical baseline or ordered MPS/profile cell."""

    condition_id: str
    profile: str
    target_mps: int
    corunner_mps: int = 0

    @property
    def baseline(self):
        """Return whether this is a reusable target-only baseline."""
        return self.profile == "target_only"


def campaign_conditions():
    """Return four baselines followed by the exact 64-cell ordered grid."""
    result = [
        Condition(f"k1091-baseline-t{target}", "target_only", target)
        for target in MPS_VALUES
    ]
    result.extend(
        Condition(
            f"k1091-{profile}-t{target}-c{corunner}",
            profile,
            target,
            corunner,
        )
        for target in MPS_VALUES
        for corunner in MPS_VALUES
        for profile in PROFILES
    )
    if len(result) != 68 or len(set(result)) != 68:
        raise SweepError("K1091 campaign must contain 68 unique conditions")
    return tuple(result)


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_immutable(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise SweepError(f"immutable file differs: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o444)
    return path


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _paired(role):
    return {
        "direction": "k1091_cta_admission",
        "role": role,
        "trials": 1,
        "period_seconds": 30.0,
        "target_anchor_seconds": 2.0,
        "deadline_offset_seconds": 2.0,
        "launch_tolerance_seconds": 0.005,
        "k_replays": 0,
    }


def _cta_profile(identity, tracker, participants, tail_end, align):
    return {
        "mode": "mixed",
        "tracker_library": str(tracker),
        "kernel_name": identity["kernel_name"],
        "grid": identity["grid"],
        "block": identity["block"],
        "barrier_participants": participants,
        "record_capacity": 300000,
        "targets": [{
            "label": "trial_0",
            "trial": 0,
            "kernel_occurrence": 0,
            "sequence_start": identity["sequence"],
            "sequence_end": tail_end,
            "align_launch": align,
        }],
    }


def build_condition_config(
    template, condition, iterations, output_root, tracker_library
):
    """Build one deterministic fixed-input run configuration."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    data = copy.deepcopy(template)
    models = {model["id"]: model for model in data["models"]}
    try:
        faster = models[K1091["model_id"]]
        deep = models[K0265["model_id"]]
    except KeyError as exc:
        raise SweepError("template requires Faster R-CNN and DeepLabV3+") \
            from exc
    run_prefix = f"k1-{campaign_conditions().index(condition):02d}"
    data["run"].update({
        "id": run_prefix + "-a1",
        "experiment": "K1091 CTA admission MPS sweep",
        "provenance": "fixed K1091/K0265 identity campaign",
        "output_root": str(Path(output_root) / "runs"),
        "timeout_seconds": 900,
    })
    data["ros"]["domain_id"] = (
        20 + campaign_conditions().index(condition) % 200
    )
    data["replay"]["communication_profile"] = False
    data["recording"] = {
        "level": "level1", "scopes": ["model"], "capsule": {},
    }
    faster["mps_percentage"] = condition.target_mps
    faster["launch_offset_seconds"] = 0
    faster["paired_trial"] = _paired("target")
    real = condition.profile == "real"
    participants = (
        [K1091["model_id"], K0265["model_id"]]
        if real else [K1091["model_id"]]
    )
    faster["nvbit_cta_profile"] = _cta_profile(
        K1091, tracker_library, participants, K1091["sequence"], real
    )
    if real:
        deep["mps_percentage"] = condition.corunner_mps
        deep["launch_offset_seconds"] = 0
        deep["paired_trial"] = _paired("co_runner")
        deep["nvbit_cta_profile"] = _cta_profile(
            K0265, tracker_library, participants, 290, True
        )
        data["models"] = [faster, deep]
        data.pop("synthetic_fma", None)
    else:
        data["models"] = [faster]
        data.pop("synthetic_fma", None)
        if not condition.baseline:
            deadline = 2.0
            calibration_path = Path(output_root) / "calibration_manifest.json"
            if calibration_path.is_file():
                calibration = json.loads(
                    calibration_path.read_text(encoding="utf-8")
                )
                deadline = calibration.get(
                    "paired_deadline_seconds_by_target_mps", {}
                ).get(str(condition.target_mps), deadline)
            data["synthetic_fma"] = {
                "client_id": "synthetic_fma",
                "blocks": SYNTHETIC_BLOCKS[condition.profile],
                "threads": 256,
                "iterations": iterations,
                "mps_percentage": condition.corunner_mps,
                "paired_deadline_seconds": deadline,
                "cta": {
                    "tracker_library": str(tracker_library),
                    "record_capacity": 4096,
                },
            }
    return schema_v2_config(data)


def _yaml_bytes(value):
    return yaml.safe_dump(schema_v2_config(value), sort_keys=False).encode()


def materialize_campaign(
    template_path, output_root, iterations, tracker_library=None,
    selected=None, manifest_name="campaign_manifest.json",
):
    """Write immutable configs and a 68-condition campaign manifest."""
    template_config = load_run_config(str(template_path), check_paths=False)
    template = template_config.data
    if tracker_library is None:
        tracker_library = template["models"][0]["nvbit_cta_profile"][
            "tracker_library"
        ]
    output_root = Path(output_root)
    selected = set(selected or [item.condition_id
                                for item in campaign_conditions()])
    records = []
    for condition in campaign_conditions():
        if condition.condition_id not in selected:
            continue
        data = build_condition_config(
            template, condition, iterations, output_root, tracker_library
        )
        path = output_root / "generated_configs" / (
            condition.condition_id + "-attempt1.yaml"
        )
        content = _yaml_bytes(data)
        _write_immutable(path, content)
        load_run_config(str(path), check_paths=False)
        records.append({
            **asdict(condition),
            "run_prefix": data["run"]["id"].rsplit("-a", 1)[0],
            "config_path": str(path.resolve()),
            "config_sha256": hashlib.sha256(content).hexdigest(),
        })
    manifest = {
        "schema": "k1091_cta_mps_sweep_v1",
        "template_path": str(Path(template_path).resolve()),
        "template_sha256": template_config.sha256,
        "output_root": str(output_root.resolve()),
        "iterations": iterations,
        "tracker_library": str(Path(tracker_library)),
        "conditions": records,
    }
    if len(selected) == 68 and len(records) != 68:
        raise SweepError("campaign manifest does not contain 68 conditions")
    _write_immutable(output_root / manifest_name,
                     _json_bytes(manifest))
    return manifest


def materialize_baselines(template_path, output_root, tracker_library=None):
    """Write only the four prerequisite target-only baseline configs."""
    selected = [
        condition.condition_id for condition in campaign_conditions()
        if condition.baseline
    ]
    return materialize_campaign(
        template_path, output_root, 1, tracker_library, selected,
        "baseline_manifest.json",
    )


def attempt_config(record, attempt):
    """Return a numbered immutable retry config without changing evidence."""
    if attempt < 1 or attempt > 3:
        raise ValueError("attempt must be in 1..3")
    source = Path(record["config_path"])
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    run_id = record["run_prefix"] + f"-a{attempt}"
    data["run"]["id"] = run_id
    path = source.with_name(run_id + ".yaml")
    _write_immutable(path, _yaml_bytes(data))
    return path


def successful_attempt(record, output_root):
    """Return the first valid immutable attempt, or the next free number."""
    runs = Path(output_root) / "runs"
    for attempt in range(1, 4):
        directory = runs / (record["run_prefix"] + f"-a{attempt}")
        validation = directory / "campaign_validation.json"
        manifest = directory / "run_manifest.json"
        if validation.is_file() and manifest.is_file():
            try:
                valid = json.loads(validation.read_text())["valid"]
                state = json.loads(manifest.read_text())["state"]
            except (KeyError, OSError, json.JSONDecodeError):
                valid = False
                state = "invalid"
            if valid and state == "success":
                return directory, None
        if not directory.exists():
            return None, attempt
    return None, None


def admission_bins(entries, start_ns, end_ns, bins=BINS):
    """Reduce exact CTA entries into half-open bins without interpolation."""
    start_ns, end_ns = int(start_ns), int(end_ns)
    if bins < 1 or end_ns <= start_ns:
        raise ValueError("bin horizon must be positive")
    counts = [0] * bins
    span = end_ns - start_ns
    for entry in entries:
        entry = int(entry)
        if entry < start_ns or entry >= end_ns:
            raise ValueError("CTA entry falls outside the half-open horizon")
        counts[min(bins - 1, (entry - start_ns) * bins // span)] += 1
    cumulative = 0
    rows = []
    for index, count in enumerate(counts):
        cumulative += count
        rows.append({
            "sample_index": index,
            "relative_bin_start_ns": span * index / bins,
            "relative_bin_end_ns": span * (index + 1) / bins,
            "admitted_ctas": count,
            "cumulative_admitted_ctas": cumulative,
        })
    return rows


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _complete_intervals(run_directory, model_id, sequence):
    status = json.loads((run_directory / f"model_{model_id}_cta_status.json")
                        .read_text(encoding="utf-8"))
    launch = next((item for item in status["launches"]
                   if int(item["launch_sequence_index"]) == sequence), None)
    if launch is None:
        raise SweepError(f"{model_id} sequence {sequence} was not captured")
    rows = [
        row for row in _read_csv(
            run_directory / f"model_{model_id}_cta_raw.csv"
        )
        if int(row["launch_slot"]) == int(launch["launch_slot"])
    ]
    if not rows or any(
        row["observation_status"] != "complete" for row in rows
    ):
        raise SweepError(f"{model_id} sequence {sequence} CTA data incomplete")
    intervals = [{
        "entry": int(row["entry_globaltimer_ns"]),
        "exit": int(row["exit_globaltimer_ns"]),
        "duration": int(row["duration_ns"]),
        "sm_id": int(row["sm_id"]),
    } for row in rows]
    return status, launch, intervals


def _target_capture(run_directory):
    status, launch, intervals = _complete_intervals(
        run_directory, K1091["model_id"], K1091["sequence"]
    )
    if (
        launch["kernel_name"] != K1091["kernel_name"]
        or launch["grid"] != K1091["grid"]
        or launch["block"] != K1091["block"]
        or len(intervals) != K1091["cta_count"]
    ):
        raise SweepError("captured K1091 identity or CTA count differs")
    collection = status["collection"]
    if any(int(collection.get(name, 0)) for name in (
        "dropped_capacity_record_count", "incomplete_entry_count",
        "incomplete_exit_count", "dropped_launch_count",
    )):
        raise SweepError("K1091 tracer reports dropped/incomplete records")
    entries = sorted(item["entry"] for item in intervals)
    exits = [item["exit"] for item in intervals]
    durations = sorted(item["duration"] for item in intervals)
    ready = int(launch["launch_ready_globaltimer_ns"])
    if entries[0] < ready:
        raise SweepError("K1091 CTA entry precedes launch-ready")
    sm_count = len({item["sm_id"] for item in intervals})
    wave_starts = entries[::max(1, sm_count)]
    gaps = [right - left for left, right in zip(
        wave_starts, wave_starts[1:]
    )]
    return {
        "status": status,
        "launch": launch,
        "intervals": intervals,
        "entries": entries,
        "first_entry": entries[0],
        "last_entry": entries[-1],
        "last_exit": max(exits),
        "ready": ready,
        "clock_error_ns": int(launch["launch_ready_error_ns"]),
        "sm_count": sm_count,
        "wave_count": len(wave_starts),
        "gap_p50_ns": _percentile(gaps, 50) if gaps else 0,
        "gap_max_ns": max(gaps, default=0),
        "service_p50_ns": _percentile(durations, 50),
        "service_p95_ns": _percentile(durations, 95),
        "service_p99_ns": _percentile(durations, 99),
    }


def _percentile(values, percentile):
    values = sorted(values)
    if not values:
        return 0
    position = (len(values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (
        position - lower
    )


def _overlap(first, second):
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def _union_length(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _synthetic_capture(run_directory, condition, target):
    path = run_directory / "model_synthetic_fma_fma_cta.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    result = evidence["result"]
    intervals = result["cta_intervals"]
    expected = SYNTHETIC_BLOCKS[condition.profile]
    if len(intervals) != expected or any(
        item["observation_status"] != "complete" for item in intervals
    ):
        raise SweepError("synthetic FMA CTA collection is incomplete")
    buffer_status = result["cta_buffer_status"]
    if any(int(buffer_status.get(name, 0)) for name in (
        "dropped_capacity_record_count", "incomplete_entry_count",
        "incomplete_exit_count",
    )):
        raise SweepError("synthetic FMA tracer reports dropped records")
    starts = [int(item["entry_globaltimer_ns"]) for item in intervals]
    ends = [int(item["exit_globaltimer_ns"]) for item in intervals]
    uncertainty = (
        max(int(item["clock_error_ns"]) for item in intervals)
        + target["clock_error_ns"]
    )
    contained = (
        min(starts) - uncertainty <= target["first_entry"]
        and max(ends) + uncertainty >= target["last_exit"]
    )
    if not contained:
        raise SweepError("FMA envelope does not contain complete K1091")
    descriptor = evidence["identity"]["launches"][0]
    return {
        "kind": "synthetic_fma",
        "offered_ctas": expected,
        "observed_ctas": len(intervals),
        "resources": descriptor,
        "start": min(starts),
        "end": max(ends),
        "overlap_ns": _overlap(
            (target["first_entry"], target["last_exit"]),
            (min(starts), max(ends)),
        ),
        "overlap_inventory": [],
        "contained": True,
    }


def _real_capture(run_directory, target):
    status = json.loads((run_directory / "model_deeplabv3plus_cta_status.json")
                        .read_text(encoding="utf-8"))
    target_status = status["targets"][0]
    faster_status = target["status"]["targets"][0]
    if any(int(status["collection"].get(name, 0)) for name in (
        "dropped_capacity_record_count", "incomplete_entry_count",
        "incomplete_exit_count", "dropped_launch_count",
    )):
        raise SweepError("DeepLab tail tracer reports dropped records")
    if not (
        target_status.get("launch_alignment_complete")
        and faster_status.get("launch_alignment_complete")
    ):
        raise SweepError("K1091/K0265 launch alignment did not complete")
    inventory = []
    tail_spans = []
    offered = observed = 0
    raw_by_slot = {}
    for row in _read_csv(
        run_directory / "model_deeplabv3plus_cta_raw.csv"
    ):
        raw_by_slot.setdefault(int(row["launch_slot"]), []).append(row)
    for launch in status["launches"]:
        sequence = int(launch["launch_sequence_index"])
        if sequence < 265 or sequence > 290:
            continue
        rows = raw_by_slot.get(int(launch["launch_slot"]), [])
        if not rows or any(
            row["observation_status"] != "complete" for row in rows
        ) or len(rows) != int(launch["expected_count"]):
            raise SweepError(f"DeepLab sequence {sequence} is incomplete")
        intervals = [{
            "entry": int(row["entry_globaltimer_ns"]),
            "exit": int(row["exit_globaltimer_ns"]),
        } for row in rows]
        offered += int(launch["expected_count"])
        observed += len(intervals)
        span = (min(item["entry"] for item in intervals),
                max(item["exit"] for item in intervals))
        tail_spans.append(span)
        overlap = _overlap(
            (target["first_entry"], target["last_exit"]), span
        )
        if overlap:
            inventory.append({
                "sequence": sequence,
                "kernel_name": launch["kernel_name"],
                "grid": launch["grid"],
                "block": launch["block"],
                "cta_count": len(intervals),
                "start_ns": span[0],
                "end_ns": span[1],
                "overlap_ns": overlap,
            })
    head = next((item for item in inventory if item["sequence"] == 265), None)
    if head is None or head["overlap_ns"] <= 0:
        raise SweepError("K0265 has no positive overlap with K1091")
    if (
        head["kernel_name"] != K0265["kernel_name"]
        or head["grid"] != K0265["grid"]
        or head["block"] != K0265["block"]
        or head["cta_count"] != K0265["cta_count"]
    ):
        raise SweepError("captured K0265 identity or CTA count differs")
    return {
        "kind": "deeplabv3plus",
        "offered_ctas": offered,
        "observed_ctas": observed,
        "resources": {
            "head": K0265,
            "tail_sequence": [265, 290],
        },
        "start": min(item[0] for item in tail_spans),
        "end": max(item[1] for item in tail_spans),
        "overlap_ns": _union_length([
            (max(target["first_entry"], item["start_ns"]),
             min(target["last_exit"], item["end_ns"]))
            for item in inventory
        ]),
        "overlap_inventory": inventory,
        "contained": (
            head["start_ns"] <= target["first_entry"]
            and head["end_ns"] >= target["last_exit"]
        ),
    }


def validate_attempt(run_directory, condition):
    """Apply strict identity, completeness, MPS, and overlap gates."""
    run_directory = Path(run_directory)
    errors = []
    target = corunner = None
    try:
        target = _target_capture(run_directory)
        if condition.profile in SYNTHETIC_BLOCKS:
            corunner = _synthetic_capture(run_directory, condition, target)
        elif condition.profile == "real":
            corunner = _real_capture(run_directory, target)
        for model_id, percentage in (
            (K1091["model_id"], condition.target_mps),
            *(([(K0265["model_id"], condition.corunner_mps)])
              if condition.profile == "real" else []),
        ):
            status = json.loads((run_directory / f"model_{model_id}.json")
                                .read_text(encoding="utf-8"))
            if str(status.get(
                "process_observed_cuda_mps_active_thread_percentage"
            )) != str(percentage):
                errors.append(f"{model_id} MPS cap differs")
    except (OSError, KeyError, ValueError, StopIteration, SweepError) as exc:
        errors.append(str(exc))
    evidence = {
        "schema": "k1091_campaign_validation_v1",
        "condition": asdict(condition),
        "valid": not errors,
        "errors": errors,
        "target": None if target is None else {
            key: target[key] for key in (
                "ready", "first_entry", "last_entry", "last_exit",
                "clock_error_ns", "sm_count", "wave_count",
            )
        },
        "corunner": corunner,
    }
    _write_json(run_directory / "campaign_validation.json", evidence)
    return evidence


def _condition_from_record(record):
    return Condition(**{
        key: record[key] for key in (
            "condition_id", "profile", "target_mps", "corunner_mps"
        )
    })


def run_campaign(manifest_path, selected=None, dry_run=False):
    """Run conditions sequentially, retaining up to three numbered attempts."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    selected = set(selected or [item["condition_id"]
                                for item in manifest["conditions"]])
    results = {}
    for record in manifest["conditions"]:
        condition = _condition_from_record(record)
        if condition.condition_id not in selected:
            continue
        for _ in range(3):
            success, attempt = successful_attempt(
                record, manifest["output_root"]
            )
            if success:
                results[condition.condition_id] = str(success)
                break
            if attempt is None:
                raise SweepError(
                    f"{condition.condition_id} exhausted three attempts"
                )
            config = load_run_config(
                str(attempt_config(record, attempt)),
                artifact_root=manifest["output_root"],
            )
            if dry_run:
                ExperimentRunner(config).run(dry_run=True)
                results[condition.condition_id] = "dry-run"
                break
            gpu = config.data["gpu"]
            control = GPUClockLock(
                gpu["index"], gpu["graphics_clock_mhz"],
                gpu["memory_clock_mhz"],
            )
            try:
                with control:
                    ExperimentRunner(config).run()
                evidence = validate_attempt(config.run_directory, condition)
            except KeyboardInterrupt:
                raise
            except Exception:
                if not config.run_directory.exists() or attempt == 3:
                    raise
                continue
            finally:
                if config.run_directory.is_dir():
                    _write_json(
                        config.run_directory / "clock_control.json", {
                            "control": control.evidence,
                            "clocks_restored": len(
                                control.evidence.get("reset_commands", [])
                            ) == 2,
                        }
                    )
            if evidence["valid"]:
                results[condition.condition_id] = str(config.run_directory)
                break
        else:
            raise SweepError(
                f"{condition.condition_id} exhausted three attempts"
            )
    return results


def calibrate_fma_iterations(
    output_root, gpu, longest_baseline_ns, tracker_library,
    run_command=subprocess.run, paired_deadlines=None,
):
    """Calibrate one 28-block/80%-MPS envelope in at most three attempts."""
    threshold = max(10_000_000, 2 * int(longest_baseline_ns))
    iterations = 1_000_000
    attempts = []
    worker_path, adapter = discover_offline_native()
    for attempt in range(1, 4):
        apps = run_command([
            "nvidia-smi", "-i", str(gpu["index"]),
            "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True)
        occupants = [line.strip() for line in apps.stdout.splitlines()
                     if line.strip() and line.strip() != "[N/A]"]
        foreign_mps = run_command(
            ["pgrep", "-x", "nvidia-cuda-mps-control"], check=False,
            capture_output=True, text=True,
        )
        if occupants or foreign_mps.returncode == 0:
            raise SweepError("calibration requires an idle GPU and no MPS")
        runtime = Path(output_root) / "calibration" / f"attempt{attempt}"
        runtime.mkdir(parents=True, exist_ok=False)
        control = GPUClockLock(
            gpu["index"], gpu["graphics_clock_mhz"],
            gpu["memory_clock_mhz"], run_command,
        )
        with control, MPSManager(
            gpu["index"], runtime, run_command=run_command
        ) as mps:
            worker = WorkerProcess(
                worker_path, adapter,
                {"adapter": "builtin_compute", "blocks": 28,
                 "threads": 256, "iterations": iterations},
                "synthetic_fma", {**os.environ,
                                  **mps.model_environment(80)},
                runtime / "worker.log",
            )
            try:
                worker.send({"operation": "run",
                             "release_ns": time.monotonic_ns() + 20_000_000})
                result = worker.receive()
            finally:
                worker.close()
        duration = int(result["launch_activities"][0]["gpu_duration_ns"])
        attempts.append({
            "attempt": attempt, "iterations": iterations,
            "duration_ns": duration,
        })
        if duration >= threshold:
            manifest = {
                "schema": "k1091_fma_calibration_v1",
                "threshold_ns": threshold,
                "longest_baseline_ns": int(longest_baseline_ns),
                "iterations": iterations,
                "attempts": attempts,
                "paired_deadline_seconds_by_target_mps": (
                    paired_deadlines or {}
                ),
            }
            _write_immutable(
                Path(output_root) / "calibration_manifest.json",
                _json_bytes(manifest),
            )
            return manifest
        iterations = max(
            iterations + 1,
            math.ceil(iterations * threshold * 1.1 / max(duration, 1)),
        )
    raise SweepError("FMA calibration failed after three attempts")


def calibrate_from_baselines(manifest_path, run_command=subprocess.run):
    """Load four valid baselines and lock the shared FMA iteration count."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if len(manifest["conditions"]) != 4 or any(
        item["profile"] != "target_only" for item in manifest["conditions"]
    ):
        raise SweepError("calibration requires the four-baseline manifest")
    envelopes = []
    deadlines = {}
    for record in manifest["conditions"]:
        run_directory = _successful_run(record, manifest["output_root"])
        target = _target_capture(run_directory)
        envelopes.append(target["last_exit"] - target["first_entry"])
        paired = json.loads(
            (run_directory / "model_faster_rcnn_paired.jsonl")
            .read_text(encoding="utf-8").splitlines()[0]
        )
        model = load_run_config(record["config_path"]).data["models"][0]
        ready_delay = (
            int(target["launch"]["host_launch_ready_ns"])
            - round(float(paired["scheduled_time"]) * 1_000_000_000)
        ) / 1_000_000_000
        deadlines[str(record["target_mps"])] = (
            model["paired_trial"]["deadline_offset_seconds"]
            + max(0.0, ready_delay - 0.005)
        )
    config = load_run_config(manifest["conditions"][0]["config_path"])
    return calibrate_fma_iterations(
        manifest["output_root"], config.data["gpu"], max(envelopes),
        manifest["tracker_library"], run_command, deadlines,
    )


def _summary_row(condition, run_directory, target, corunner):
    envelope = target["last_exit"] - target["first_entry"]
    dispatch = target["last_entry"] - target["first_entry"]
    overlap = 0 if corunner is None else corunner["overlap_ns"]
    return {
        **asdict(condition),
        "run_directory": str(run_directory),
        "target_identity": "K1091/sequence506",
        "corunner_identity": "" if corunner is None else corunner["kind"],
        "overlap_ns": overlap,
        "overlap_fraction": overlap / envelope if envelope else 0,
        "k1091_envelope_ns": envelope,
        "ready_to_first_cta_ns": target["first_entry"] - target["ready"],
        "dispatch_span_ns": dispatch,
        "admission_ctas_per_ms": K1091["cta_count"] * 1e6 /
        max(1, dispatch),
        "dispatch_wave_count": target["wave_count"],
        "interwave_gap_p50_ns": target["gap_p50_ns"],
        "interwave_gap_max_ns": target["gap_max_ns"],
        "client_local_sm_footprint": target["sm_count"],
        "cta_service_p50_ns": target["service_p50_ns"],
        "cta_service_p95_ns": target["service_p95_ns"],
        "cta_service_p99_ns": target["service_p99_ns"],
    }


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _successful_run(record, output_root):
    directory, _ = successful_attempt(record, output_root)
    if directory is None:
        raise SweepError(f"{record['condition_id']} has no valid attempt")
    return directory


def analyze_campaign(manifest_path, output_directory):
    """Write strict summaries, exact 300-bin series, figures, and report."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["conditions"]) != 68:
        raise SweepError("analysis requires all 68 conditions")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    captures = {}
    summaries = []
    corunners = []
    for record in manifest["conditions"]:
        condition = _condition_from_record(record)
        run_directory = _successful_run(record, manifest["output_root"])
        target = _target_capture(run_directory)
        corunner = None
        if condition.profile in SYNTHETIC_BLOCKS:
            corunner = _synthetic_capture(run_directory, condition, target)
        elif condition.profile == "real":
            corunner = _real_capture(run_directory, target)
        captures[condition.condition_id] = (condition, target, corunner,
                                            run_directory)
        summaries.append(_summary_row(
            condition, run_directory, target, corunner
        ))
        if corunner:
            envelope = target["last_exit"] - target["first_entry"]
            corunners.append({
                **asdict(condition),
                "run_directory": str(run_directory),
                "kind": corunner["kind"],
                "offered_ctas": corunner["offered_ctas"],
                "observed_ctas": corunner["observed_ctas"],
                "resources": json.dumps(corunner["resources"], sort_keys=True),
                "duration_ns": corunner["end"] - corunner["start"],
                "overlap_ns": corunner["overlap_ns"],
                "overlap_fraction": corunner["overlap_ns"] / envelope,
                "full_k1091_containment": corunner["contained"],
                "overlapping_kernel_inventory": json.dumps(
                    corunner["overlap_inventory"], sort_keys=True
                ),
            })
    _write_csv(output / "condition_summary.csv", summaries)
    _write_csv(output / "corunner_summary.csv", corunners)
    admission = []
    figure_series = {}
    for target_mps in MPS_VALUES:
        baseline = captures[f"k1091-baseline-t{target_mps}"]
        for corunner_mps in MPS_VALUES:
            series = [
                captures[f"k1091-{profile}-t{target_mps}-c{corunner_mps}"]
                for profile in PROFILES
            ] + [baseline]
            horizon = max(
                target["last_exit"] - target["ready"]
                for _, target, _, _ in series
            )
            key = (target_mps, corunner_mps)
            figure_series[key] = []
            for condition, target, _, run_directory in series:
                profile = condition.profile
                rows = admission_bins(
                    target["entries"], target["ready"],
                    target["ready"] + horizon,
                )
                if len(rows) != BINS or rows[-1][
                    "cumulative_admitted_ctas"
                ] != K1091["cta_count"]:
                    raise SweepError("300-bin K1091 conservation failed")
                decorated = [{
                    "target_mps": target_mps,
                    "corunner_mps": corunner_mps,
                    "profile": profile,
                    "source_run": str(run_directory),
                    **row,
                } for row in rows]
                admission.extend(decorated)
                figure_series[key].append((profile, decorated))
    _write_csv(output / "k1091_admission_300.csv", admission)
    _write_figures(output, figure_series)
    report = _report(manifest_path, manifest, summaries, corunners)
    (output / "report.md").write_text(report, encoding="utf-8")
    report_clean = (
        "TODO" not in report
        and "not guaranteed instantaneous GPU shares" in report
        and "no physical cross-client placement is inferred" in report
    )
    acceptance = {
        "schema": "k1091_cta_mps_sweep_acceptance_v1",
        "passed": (
            len(summaries) == 68
            and len(admission) == 16 * 5 * BINS
            and len(list(output.glob("k1091_admission_t*_c*.png"))) == 16
            and all(row["cta_service_p99_ns"] > 0 for row in summaries)
            and report_clean
        ),
        "condition_count": len(summaries),
        "admission_row_count": len(admission),
        "figure_count": len(list(output.glob(
            "k1091_admission_t*_c*.png"
        ))),
        "report_clean": report_clean,
    }
    _write_json(output / "acceptance.json", acceptance)
    if not acceptance["passed"]:
        raise SweepError("campaign acceptance failed")
    return acceptance


def _write_figures(output, figure_series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "synthetic_low": "FMA 28",
        "synthetic_medium": "FMA 56",
        "synthetic_high": "FMA 112",
        "real": "K0265-led real",
        "target_only": "target only",
    }
    for (target_mps, corunner_mps), series in figure_series.items():
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for profile, rows in series:
            x = [(
                row["relative_bin_start_ns"]
                + row["relative_bin_end_ns"]
            ) / 2e6 for row in rows]
            axes[0].plot(x, [row["admitted_ctas"] for row in rows],
                         label=labels[profile])
            axes[1].plot(
                x, [row["cumulative_admitted_ctas"] for row in rows],
                label=labels[profile],
            )
        axes[0].set_ylabel("K1091 CTA entries/bin")
        axes[1].set_ylabel("Cumulative K1091 CTAs")
        axes[1].set_xlabel("Time from K1091 launch-ready (ms)")
        axes[0].legend(ncol=2, fontsize=8)
        figure.suptitle(
            f"K1091 admission | target {target_mps}% / co-runner "
            f"{corunner_mps}% MPS caps"
        )
        figure.tight_layout()
        figure.savefig(
            output / f"k1091_admission_t{target_mps}_c{corunner_mps}.png",
            dpi=160,
        )
        plt.close(figure)


def _report(manifest_path, manifest, summaries, corunners):
    run_paths = sorted({row["run_directory"] for row in summaries})
    hashes = [
        f"- `{path}`: `{_sha256(Path(path) / 'config.yaml')}`"
        for path in run_paths
    ]
    real_inventory = [
        row for row in corunners if row["profile"] == "real"
    ]
    calibration_path = Path(manifest["output_root"]) / (
        "calibration_manifest.json"
    )
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.is_file() else {}
    )
    overlap_lines = []
    for row in real_inventory:
        sequences = [
            item["sequence"] for item in json.loads(
                row["overlapping_kernel_inventory"]
            )
        ]
        overlap_lines.append(
            f"- target/co-runner `{row['target_mps']}/{row['corunner_mps']}`: "
            f"DeepLab sequences `{sequences}`"
        )
    failed = []
    for directory in sorted(
            (Path(manifest["output_root"]) / "runs").glob("*")):
        validation = directory / "campaign_validation.json"
        run_manifest = directory / "run_manifest.json"
        state = None
        if run_manifest.is_file():
            state = json.loads(run_manifest.read_text()).get("state")
        valid = None
        if validation.is_file():
            valid = json.loads(validation.read_text()).get("valid")
        if state != "success" or valid is False:
            failed.append(f"- `{directory}`: state `{state}`, valid `{valid}`")
    attempts = calibration.get("attempts", [])
    calibration_text = (
        f"The 28-block/80% isolated calibration threshold was "
        f"`{calibration.get('threshold_ns')}` ns; attempts were "
        f"`{attempts}`. Baseline-derived FMA paired deadlines by target cap "
        f"were `{calibration.get('paired_deadline_seconds_by_target_mps')}`."
        if calibration else
        "Calibration evidence was supplied as the locked iteration count in "
        "the immutable campaign manifest."
    )
    return "\n".join([
        "# K1091 CTA admission MPS sweep", "",
        "## Configuration", "",
        "The campaign contains four reusable Faster R-CNN target-only "
        "baselines and 64 ordered MPS/profile pairs. Synthetic profiles use "
        "28, 56, and 112 identical 256-thread FMA CTAs. The real profile "
        "barrier-aligns Faster R-CNN sequence 506 (K1091) with DeepLabV3+ "
        "sequence 265 (K0265), then observes DeepLab sequences 265–290.", "",
        f"The locked FMA iteration count is `{manifest['iterations']}`. "
        "Each condition has one valid execution; failed numbered attempts "
        "remain beside successful evidence. MPS percentages are client "
        "active-thread caps, not guaranteed instantaneous GPU shares.", "",
        calibration_text, "",
        "K1091 is `ampere_sgemm_128x64_tn`, grid `8×16×16`, block "
        "`128×1×1`, and 2,048 CTAs. K0265 uses its pinned stored signature, "
        "grid `4×57×8`, block `128×1×1`, and 1,824 CTAs.", "",
        "## Reduction", "",
        "CTA admissions are exact entry events reduced into half-open bins; "
        "there is no tracker polling or interpolation. Every plotted series "
        "has 300 bins and conserves all 2,048 K1091 entries. Later DeepLab "
        "kernels are retained only when their measured intervals overlap "
        "K1091. SM identifiers are interpreted only in each MPS client’s "
        "local namespace; no physical cross-client placement is inferred.", "",
        f"All `{len(summaries)}` required conditions passed capture gates. "
        f"The real-stream table contains `{len(real_inventory)}` ordered "
        "MPS cells with measured overlap inventories.", "",
        "## Actual real-stream overlap sets", "", *overlap_lines, "",
        "## Failed attempts and exclusions", "",
        *(failed or ["No failed or excluded attempts were retained."]), "",
        "## Commands", "",
        "```bash",
        "ros2 run closeloop_experiments campaign k1091-cta-mps baselines "
        "TEMPLATE --artifact-root ARTIFACT_ROOT",
        "ros2 run closeloop_experiments campaign k1091-cta-mps run "
        "ARTIFACT_ROOT/baseline_manifest.json --artifact-root ARTIFACT_ROOT",
        "ros2 run closeloop_experiments campaign k1091-cta-mps calibrate "
        "ARTIFACT_ROOT/baseline_manifest.json --artifact-root ARTIFACT_ROOT",
        "ros2 run closeloop_experiments campaign k1091-cta-mps generate "
        "TEMPLATE --artifact-root ARTIFACT_ROOT",
        "ros2 run closeloop_analyzer analyze mps "
        "ARTIFACT_ROOT/runs/RUN_ID --output-root ARTIFACT_ROOT",
        "```", "",
        "## Raw configurations and SHA-256", "", *hashes, "",
        "Raw CTA CSV/status paths are rooted in the run directories listed "
        "in `condition_summary.csv` and `corunner_summary.csv`; those tables "
        "are the authoritative path inventory.", "",
        "## Limitations", "",
        "Compute level denotes offered CTA count with identical per-CTA FMA "
        "instructions, not a fraction of peak FLOP/s. K0265 alignment is a "
        "controlled barrier intervention rather than the natural scheduler "
        "phase. One execution per condition describes these captures and "
        "does not estimate run-to-run uncertainty.", "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    baselines = commands.add_parser("baselines")
    baselines.add_argument("template", type=Path)
    baselines.add_argument("--tracker-library")
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("baseline_manifest", type=Path)
    generate = commands.add_parser("generate")
    generate.add_argument("template", type=Path)
    generate.add_argument("--iterations", type=int)
    generate.add_argument("--tracker-library")
    run = commands.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--condition", action="append")
    run.add_argument("--dry-run", action="store_true")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    output_root = Path(args.artifact_root).resolve()
    if args.command == "baselines":
        result = materialize_baselines(
            args.template, output_root, args.tracker_library
        )
    elif args.command == "calibrate":
        result = calibrate_from_baselines(args.baseline_manifest)
    elif args.command == "generate":
        iterations = args.iterations
        if iterations is None:
            calibration = json.loads((
                output_root / "calibration_manifest.json"
            ).read_text(encoding="utf-8"))
            iterations = int(calibration["iterations"])
        result = materialize_campaign(
            args.template, output_root, iterations,
            args.tracker_library,
        )
    elif args.command == "run":
        result = run_campaign(args.manifest, args.condition, args.dry_run)
    else:
        result = analyze_campaign(
            args.manifest, output_root / "analysis" / "k1091-cta-mps"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
