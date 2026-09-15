"""Fixed-input signed-offset sweep for all 18 LiDAR-image pairs."""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics

import matplotlib
import numpy as np
import yaml

from .config import ConfigError, load_run_config, schema_v2_config
from .non_mps_memcpy_compute import (
    ALIGNMENT_METRIC,
    MPS_ENVIRONMENT_VARIABLES,
    StudyError,
    _active_context_intervals,
    _context_owner,
    _find_inference,
    _in_range,
    _intersection_duration,
    _kernels,
    _load_jsonl,
    _memcopies,
    _nvtx_ranges,
    _runtime_events,
    _sha256,
    _spearman,
    _status_map,
    _sync_wait,
    _union_duration,
    _write_csv,
    _write_immutable,
    _write_json,
)
from closeloop_testbed.resource_control import GPUClockLock
from .runner import ExperimentRunner

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LIDAR_ORDER = ("pointpillars", "centerpoint", "3dssd")
IMAGE_ORDER = (
    "faster-rcnn",
    "yolov3",
    "detr",
    "mask-rcnn",
    "deeplabv3plus",
    "vit-upernet",
)
BAG_BOOTSTRAP_REPETITIONS = 1000
REQUIRED_FIELDS = {
    "schema_version",
    "study_id",
    "section4_1_study",
    "3dssd_study",
    "base_config",
    "output_root",
    "bag_analysis_root",
    "trials_per_offset",
    "offset_step_ms",
    "image_deadline_seconds",
    "alignment_window_ms",
    "launch_tolerance_seconds",
    "graphics_clock_mhz",
    "memory_clock_mhz",
}


def _resolve(source, value):
    path = Path(value).expanduser()
    return (source.parent / path).resolve() if not path.is_absolute() else (
        path.resolve())


def _load_yaml(path):
    try:
        value = yaml.safe_load(Path(path).read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise StudyError(f"cannot load offset sweep input: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyError(f"YAML root is not an object: {path}")
    return value


def _model(source, checkpoint_root):
    model = copy.deepcopy(source)
    model.pop("group", None)
    model["checkpoint"] = str(Path(checkpoint_root) / model["checkpoint"])
    return model


def load_study(path, check_paths=True):
    """Load the pinned 18-pair sweep contract."""
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    data = _load_yaml(source)
    missing = REQUIRED_FIELDS - set(data)
    unknown = set(data) - REQUIRED_FIELDS
    if missing or unknown:
        raise StudyError(
            f"study fields missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    expected = {
        "schema_version": 1,
        "study_id": "non_mps_offset_sweep",
        "trials_per_offset": 3,
        "offset_step_ms": 8,
        "image_deadline_seconds": 2.0,
        "alignment_window_ms": 2.0,
    }
    for name, value in expected.items():
        if data[name] != value:
            raise StudyError(f"{name} must be {value!r}")

    section_path = _resolve(source, data["section4_1_study"])
    ssd_path = _resolve(source, data["3dssd_study"])
    section = _load_yaml(section_path)
    ssd = _load_yaml(ssd_path)
    aliases = {"yolo": "yolov3", "vit": "vit-upernet"}
    section_names = (
        "pointpillars", "centerpoint", "faster-rcnn", "yolo", "detr",
        "mask-rcnn", "deeplabv3plus", "vit",
    )
    models = {
        aliases.get(name, name): _model(
            section["models"][name], section["checkpoint_root"])
        for name in section_names
    }
    models["3dssd"] = _model(
        ssd["models"]["3dssd"], ssd["checkpoint_root"])
    if set(models) != set(LIDAR_ORDER + IMAGE_ORDER):
        raise StudyError("model sources do not produce the 18-pair inventory")

    paths = {
        name: _resolve(source, data[name])
        for name in ("base_config", "output_root", "bag_analysis_root")
    }
    paths.update({
        "warmup_image": _resolve(ssd_path, ssd["warmup_image"]),
        "warmup_lidar": _resolve(ssd_path, ssd["warmup_lidar"]),
        "metadata_path": _resolve(ssd_path, ssd["metadata_path"]),
    })
    condition = ssd["conditions"]["weather-clean"]
    if check_paths:
        for name in ("base_config", "warmup_image", "warmup_lidar",
                     "metadata_path"):
            if not paths[name].is_file():
                raise StudyError(f"missing {name}: {paths[name]}")
        if not Path(condition["bag_directory"]).is_dir():
            raise StudyError("fixed-input replay placeholder bag is missing")
        for name, model in models.items():
            artifacts = (
                (Path(model["model_config"]), model["model_config_sha256"]),
                (Path(model["checkpoint"]), model["checkpoint_sha256"]),
            )
            for artifact, expected_hash in artifacts:
                if not artifact.is_file() or _sha256(artifact) != expected_hash:
                    raise StudyError(f"{name} pinned artifact differs")
    study = {
        "source": source,
        "sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "data": data,
        "paths": paths,
        "models": models,
        "condition": condition,
        "bag_conditions": copy.deepcopy(section["conditions"]),
        "pairs": tuple((lidar, image) for lidar in LIDAR_ORDER
                       for image in IMAGE_ORDER),
    }
    if len(study["pairs"]) != 18 or len(set(study["pairs"])) != 18:
        raise StudyError("sweep must contain 18 unique pairs")
    return study


def offset_order(step_ms=8):
    """Yield 0, -step, +step, -2*step, +2*step, forever."""
    yield 0
    magnitude = int(step_ms)
    while True:
        yield -magnitude
        yield magnitude
        magnitude += int(step_ms)


def envelope_overlap(left, right):
    """Return the intersection of two measured GPU envelopes in ns."""
    if not left or not right:
        return None
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def copy_alignment_contributions(copies, co_active, window_ns=2_000_000):
    """Return per-H2D/D2H preceding-window occupancy; D2D is excluded."""
    result = []
    ordinals = {"H2D": 0, "D2H": 0}
    for item in sorted(copies, key=lambda row: (row["start"], row["end"])):
        copy_class = item.get("copy_class")
        if copy_class not in ordinals:
            continue
        start = int(item["start"])
        contribution = _union_duration(
            co_active, start - window_ns, start) / window_ns
        result.append({
            **item,
            "copy_ordinal": ordinals[copy_class],
            "alignment_contribution": contribution,
        })
        ordinals[copy_class] += 1
    return result


def _run_id(lidar, image, offset_ms):
    sign = "z" if offset_ms == 0 else "n" if offset_ms < 0 else "p"
    return f"nmos-{lidar}-{image}-{sign}{abs(int(offset_ms)):04d}"


def _configured_model(study, name, index, offset_ms, period):
    source = study["models"][name]
    model = copy.deepcopy(source)
    modality = model["modality"]
    model.update({
        "id": name,
        "node_name": "model_" + name.replace("-", "_"),
        "input_topic": "/camera/front" if modality == "image" else "/lidar/top",
        "input_message_type": (
            "compressed_image" if modality == "image" else "pointcloud2"),
        "input_queue_depth": 1,
        "qos": "best_effort",
        "warmup_input": str(study["paths"][
            "warmup_image" if modality == "image" else "warmup_lidar"]),
        "warmup_count": 5,
        "launch_offset_seconds": 0,
        "module_annotation_depth": 0,
        "quiet_period_seconds": 0.5,
        "cpu_affinity": list(range(index * 6, (index + 1) * 6)),
        "cpu_thread_count": 3,
    })
    image_deadline = study["data"]["image_deadline_seconds"]
    deadline = image_deadline if modality == "image" else (
        image_deadline + offset_ms / 1000)
    if deadline < 0:
        raise StudyError("LiDAR deadline precedes the paired replay epoch")
    model["paired_trial"] = {
        "direction": "lidar-deadline-minus-image-deadline",
        "role": "target" if modality == "image" else "co_runner",
        "trials": 3,
        "period_seconds": period,
        "target_anchor_seconds": image_deadline,
        "deadline_offset_seconds": deadline,
        "launch_tolerance_seconds": study["data"][
            "launch_tolerance_seconds"],
        "timing_diagnostics_only": True,
        "k_replays": 0,
    }
    return model


def run_config(study, pair, offset_ms, ordinal=0):
    """Build one three-trial config with an invariant 2 s image deadline."""
    lidar, image = pair
    data = copy.deepcopy(_load_yaml(study["paths"]["base_config"]))
    identifier = _run_id(lidar, image, offset_ms)
    data["run"].update({
        "id": identifier,
        "experiment": "18-pair non-MPS signed offset sweep",
        "provenance": "study-sha256:" + study["sha256"],
        "output_root": str(study["paths"]["output_root"] / "runs"),
        "phase": "validation",
        "parent_run_id": "non_mps_offset_sweep",
        "hypothesis_id": "memcpy_alignment",
        "intervention_id": identifier,
    })
    data["ros"]["domain_id"] = 70 + ordinal % 100
    condition = study["condition"]
    data["replay"].update({
        "scene_token": condition["scene_token"],
        "metadata_path": str(study["paths"]["metadata_path"]),
        "bag_directory": condition["bag_directory"],
        "cpu_affinity": [12, 13, 14, 15],
        "cpu_thread_count": 4,
    })
    data["gpu"].update({
        "mps_enabled": False,
        "graphics_clock_mhz": study["data"]["graphics_clock_mhz"],
        "memory_clock_mhz": study["data"]["memory_clock_mhz"],
    })
    period = max(4.0, 3.0 + offset_ms / 1000)
    data["models"] = [
        _configured_model(study, image, 0, offset_ms, period),
        _configured_model(study, lidar, 1, offset_ms, period),
    ]
    data["recording"]["nsys"]["gpu_context_switch"] = True
    return schema_v2_config(data)


def materialize_config(study, pair, offset_ms, ordinal=0):
    """Write and validate one immutable offset config."""
    data = run_config(study, pair, offset_ms, ordinal)
    path = (study["paths"]["output_root"] / "generated_configs" /
            f"{pair[0]}+{pair[1]}" / f"{data['run']['id']}.yaml")
    _write_immutable(
        path, yaml.safe_dump(schema_v2_config(data), sort_keys=False).encode())
    try:
        load_run_config(str(path))
    except ConfigError as exc:
        raise StudyError(f"invalid generated offset config: {exc}") from exc
    return path


def _model_diagnostics(status, record, tolerance):
    warnings = []
    if status.get("state") != "acknowledged":
        warnings.append(f"state={status.get('state')}")
    observed = status.get("process_observed_mps_environment", {})
    if any(observed.get(name) is not None for name in MPS_ENVIRONMENT_VARIABLES):
        warnings.append("MPS environment observed")
    if record is None:
        warnings.append("paired record missing")
    elif abs(record.get("launch_error_seconds", math.inf)) > tolerance:
        warnings.append("launch tolerance exceeded")
    return warnings


def extract_offset_run(study, config, pair, offset_ms):
    """Extract every completed target direction without diagnostic gates."""
    run_directory = config.run_directory
    statuses = _status_map(run_directory)
    logs = {
        model_id: {
            int(row["trial"]): row for row in _load_jsonl(
                run_directory / f"model_{model_id}_paired.jsonl")
        }
        for model_id in statuses
        if (run_directory / f"model_{model_id}_paired.jsonl").is_file()
    }
    with sqlite3.connect(str(run_directory / "profile.sqlite")) as db:
        ranges = _nvtx_ranges(db)
        kernels = _kernels(db)
        copies = _memcopies(db)
        runtimes = _runtime_events(db)
        active = {}
        context_errors = {}
        for model_id, status in statuses.items():
            owner, errors = _context_owner(db, int(status["pid"]), kernels)
            context_errors[model_id] = list(errors)
            if owner is None:
                active[model_id] = []
            else:
                intervals, _transitions, errors, _timestamps = (
                    _active_context_intervals(owner[1]))
                active[model_id] = intervals
                context_errors[model_id].extend(errors)

    model_ids = (pair[1], pair[0])
    pids = {name: int(statuses[name]["pid"]) for name in model_ids}
    tolerance = study["data"]["launch_tolerance_seconds"]
    window_ns = round(study["data"]["alignment_window_ms"] * 1_000_000)
    trials, per_copy = [], []
    for trial in range(3):
        records = {name: logs.get(name, {}).get(trial) for name in model_ids}
        inference, envelopes = {}, {}
        for name in model_ids:
            try:
                inference[name] = _find_inference(
                    ranges, pids[name], name, trial)
                gpu_events = (
                    _in_range(kernels, inference[name], pids[name])
                    + _in_range(copies, inference[name], pids[name]))
                envelopes[name] = (
                    min(item["start"] for item in gpu_events),
                    max(item["end"] for item in gpu_events),
                ) if gpu_events else None
            except StudyError:
                inference[name], envelopes[name] = None, None
        overlap = envelope_overlap(envelopes[model_ids[0]],
                                   envelopes[model_ids[1]])
        actual_offset_ms = None
        if all(records.values()):
            actual_offset_ms = (
                records[pair[0]]["actual_time"]
                - records[pair[1]]["actual_time"]
            ) * 1000
        for target in model_ids:
            record = records[target]
            if record is None:
                continue
            co_runner = next(name for name in model_ids if name != target)
            interval = inference[target]
            target_copies = (
                _in_range(copies, interval, pids[target]) if interval else [])
            contributions = copy_alignment_contributions(
                target_copies, active.get(co_runner, []), window_ns)
            warnings = (
                _model_diagnostics(statuses[target], record, tolerance)
                + [f"context: {item}" for item in context_errors[target]]
                + [f"co-context: {item}"
                   for item in context_errors[co_runner]]
            )
            if interval is None:
                warnings.append("inference NVTX range missing")
            envelope = envelopes[target]
            row = {
                "pair_id": f"{pair[0]}+{pair[1]}",
                "lidar_model": pair[0],
                "image_model": pair[1],
                "run_id": config.data["run"]["id"],
                "trial": trial,
                "target_model": target,
                "target_direction": (
                    "image" if target == pair[1] else "lidar"),
                "co_runner_model": co_runner,
                "configured_signed_offset_ms": offset_ms,
                "actual_signed_offset_ms": actual_offset_ms,
                "configured_image_deadline_seconds": 2.0,
                "configured_lidar_deadline_seconds": 2.0 + offset_ms / 1000,
                "nvtx_inference_e2e_ns": (
                    interval["end"] - interval["start"] if interval else None),
                "nvtx_inference_e2e_ms": (
                    (interval["end"] - interval["start"]) / 1_000_000
                    if interval else None),
                "gpu_envelope_start_ns": envelope[0] if envelope else None,
                "gpu_envelope_end_ns": envelope[1] if envelope else None,
                "gpu_envelope_duration_ns": (
                    envelope[1] - envelope[0] if envelope else None),
                "gpu_envelope_overlap_ns": overlap,
                "alignment_score_a": (
                    statistics.fmean(item["alignment_contribution"]
                                     for item in contributions)
                    if contributions else None),
                "max_copy_alignment": (
                    max(item["alignment_contribution"]
                        for item in contributions)
                    if contributions else None),
                "alignment_metric": ALIGNMENT_METRIC,
                "alignment_window_ns": window_ns,
                "h2d_count": sum(item["copy_class"] == "H2D"
                                 for item in contributions),
                "d2h_count": sum(item["copy_class"] == "D2H"
                                 for item in contributions),
                "memcpy_adjacent_waiting_ns": (
                    _sync_wait(target_copies, runtimes, interval)
                    if interval else None),
                "target_launch_error_ms": (
                    record.get("launch_error_seconds", 0) * 1000),
                "co_runner_launch_error_ms": (
                    records[co_runner].get("launch_error_seconds", 0) * 1000
                    if records[co_runner] else None),
                "target_quiescence_verified": record.get(
                    "quiescence_verified"),
                "co_runner_quiescence_verified": (
                    records[co_runner].get("quiescence_verified")
                    if records[co_runner] else None),
                "context_observable": not context_errors[target]
                and not context_errors[co_runner],
                "trace_status": "complete" if interval else "partial",
                "diagnostic_warnings": "; ".join(sorted(set(warnings))),
            }
            trials.append(row)
            for item in contributions:
                per_copy.append({
                    **{key: row[key] for key in (
                        "pair_id", "lidar_model", "image_model", "run_id",
                        "trial", "target_model", "target_direction",
                        "co_runner_model", "configured_signed_offset_ms",
                        "actual_signed_offset_ms", "alignment_window_ns",
                        "diagnostic_warnings")},
                    "copy_class": item["copy_class"],
                    "copy_ordinal": item["copy_ordinal"],
                    "copy_start_ns": item["start"],
                    "copy_end_ns": item["end"],
                    "copy_duration_ns": item["end"] - item["start"],
                    "copy_bytes": item.get("bytes", 0),
                    "alignment_contribution": item["alignment_contribution"],
                    "memcpy_adjacent_waiting_ns": _sync_wait(
                        [item], runtimes, interval),
                })
    return trials, per_copy


def per_offset_rows(trials):
    """Summarize retained trials by pair, signed offset, and target."""
    groups = {}
    for row in trials:
        key = (row["pair_id"], row["configured_signed_offset_ms"],
               row["target_model"], row["target_direction"])
        groups.setdefault(key, []).append(row)

    def median(rows, field):
        values = [row[field] for row in rows if row.get(field) is not None]
        return statistics.median(values) if values else None

    return [{
        "pair_id": key[0],
        "configured_signed_offset_ms": key[1],
        "target_model": key[2],
        "target_direction": key[3],
        "trial_count": len(rows),
        "warning_trial_count": sum(bool(row["diagnostic_warnings"])
                                   for row in rows),
        "zero_overlap_trial_count": sum(
            row["gpu_envelope_overlap_ns"] == 0 for row in rows),
        "median_actual_signed_offset_ms": median(
            rows, "actual_signed_offset_ms"),
        "median_nvtx_inference_e2e_ms": median(
            rows, "nvtx_inference_e2e_ms"),
        "median_gpu_envelope_duration_ns": median(
            rows, "gpu_envelope_duration_ns"),
        "median_gpu_envelope_overlap_ns": median(
            rows, "gpu_envelope_overlap_ns"),
        "median_alignment_score_a": median(rows, "alignment_score_a"),
        "median_max_copy_alignment": median(rows, "max_copy_alignment"),
        "median_h2d_count": median(rows, "h2d_count"),
        "median_d2h_count": median(rows, "d2h_count"),
        "median_memcpy_adjacent_waiting_ns": median(
            rows, "memcpy_adjacent_waiting_ns"),
    } for key, rows in sorted(groups.items())]


def _campaign_manifest(study):
    path = study["paths"]["output_root"] / "sweep_manifest.json"
    if not path.is_file():
        return {"schema_version": 1, "study_sha256": study["sha256"],
                "cells": [], "pairs": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("study_sha256") != study["sha256"]:
        raise StudyError("sweep manifest belongs to a different study")
    return value


def _cell_rows(study, cell):
    config = load_run_config(
        cell["config_path"], check_paths=False,
        artifact_root=str(study["paths"]["output_root"]),
    )
    return extract_offset_run(
        study, config, tuple(cell["pair"]), cell["offset_ms"])


def run_campaign(study, artifact_root, dry_run=False):
    """Run offsets in exact order until each sign independently separates."""
    manifest = _campaign_manifest(study)
    manifest_path = study["paths"]["output_root"] / "sweep_manifest.json"
    cells = {(tuple(cell["pair"]), cell["offset_ms"]): cell
             for cell in manifest["cells"]}
    clocks = {
        "graphics_clock_mhz": study["data"]["graphics_clock_mhz"],
        "memory_clock_mhz": study["data"]["memory_clock_mhz"],
    }
    control = GPUClockLock(0, clocks["graphics_clock_mhz"],
                           clocks["memory_clock_mhz"])
    ordinal = len(cells)
    context = control if not dry_run else __import__("contextlib").nullcontext()
    with context:
        for pair in study["pairs"]:
            prior = [cell for (cell_pair, _offset), cell in cells.items()
                     if cell_pair == pair
                     and cell.get("all_three_zero_overlap")]
            negative_active = not any(
                cell["offset_ms"] <= 0 for cell in prior)
            positive_active = not any(
                cell["offset_ms"] >= 0 for cell in prior)
            for offset_ms in offset_order(study["data"]["offset_step_ms"]):
                if not negative_active and not positive_active:
                    break
                if offset_ms < 0 and not negative_active:
                    continue
                if offset_ms > 0 and not positive_active:
                    continue
                key = (pair, offset_ms)
                cell = cells.get(key)
                if cell is None:
                    path = materialize_config(study, pair, offset_ms, ordinal)
                    cell = {
                        "execution_index": ordinal,
                        "pair": list(pair),
                        "offset_ms": offset_ms,
                        "run_id": _run_id(*pair, offset_ms),
                        "config_path": str(path),
                        "state": "planned",
                    }
                    manifest["cells"].append(cell)
                    cells[key] = cell
                    ordinal += 1
                    _write_json(manifest_path, manifest)
                config = load_run_config(
                    cell["config_path"], artifact_root=str(artifact_root)
                )
                run_manifest = config.run_directory / "run_manifest.json"
                state = None
                if run_manifest.is_file():
                    state = json.loads(run_manifest.read_text()).get("state")
                if state != "success":
                    if state is not None:
                        cell["state"] = state
                        cell["all_three_zero_overlap"] = False
                        _write_json(manifest_path, manifest)
                        print(json.dumps({
                            "pair": pair, "offset_ms": offset_ms,
                            "state": state, "retried": False,
                        }))
                        continue
                    ExperimentRunner(config).run(dry_run=dry_run)
                    if dry_run:
                        return {"dry_run": True, "first_cell": cell}
                separated = cell.get("all_three_zero_overlap")
                reused = separated is not None
                if separated is None:
                    rows, _copies = _cell_rows(study, cell)
                    cell["state"] = "success"
                    overlap_by_trial = {
                        row["trial"]: row["gpu_envelope_overlap_ns"]
                        for row in rows
                        if row["target_direction"] == "image"
                        and row["gpu_envelope_overlap_ns"] is not None
                    }
                    separated = (
                        len(overlap_by_trial) == 3
                        and all(value == 0
                                for value in overlap_by_trial.values())
                    )
                    cell["all_three_zero_overlap"] = separated
                    _write_json(manifest_path, manifest)
                if not reused:
                    print(json.dumps({
                        "pair": pair, "offset_ms": offset_ms,
                        "all_three_zero_overlap": separated,
                    }))
                if separated:
                    if offset_ms == 0:
                        negative_active = positive_active = False
                    elif offset_ms < 0:
                        negative_active = False
                    else:
                        positive_active = False
    manifest["pairs"] = [list(pair) for pair in study["pairs"]]
    manifest["complete"] = True
    manifest["clock_control"] = control.evidence
    _write_json(manifest_path, manifest)
    return manifest


def _rho(rows, predicate=lambda _row: True, field="alignment_score_a"):
    selected = [row for row in rows if predicate(row)
                and row.get(field) is not None
                and row.get("nvtx_inference_e2e_ms") is not None]
    return _spearman(
        [row[field] for row in selected],
        [row["nvtx_inference_e2e_ms"] for row in selected],
    )


def correlation_rows(trials):
    """Return one pooled/overlap/signed correlation row per pair."""
    result = []
    for pair_id in sorted({row["pair_id"] for row in trials}):
        row = {"pair_id": pair_id}
        for direction in ("image", "lidar"):
            values = [item for item in trials
                      if item["pair_id"] == pair_id
                      and item["target_direction"] == direction]
            row.update({
                f"{direction}_pooled_spearman": _rho(values),
                f"{direction}_overlap_only_spearman": _rho(
                    values, lambda item: (
                        item.get("gpu_envelope_overlap_ns") is not None
                        and item["gpu_envelope_overlap_ns"] > 0)),
                f"{direction}_negative_offset_spearman": _rho(
                    values, lambda item: item[
                        "configured_signed_offset_ms"] < 0),
                f"{direction}_positive_offset_spearman": _rho(
                    values, lambda item: item[
                        "configured_signed_offset_ms"] > 0),
                f"{direction}_max_copy_pooled_spearman": _rho(
                    values, field="max_copy_alignment"),
                f"{direction}_sample_count": len(values),
            })
        result.append(row)
    return result


def _bag_comparison(study, trials, correlations):
    path = study["paths"]["bag_analysis_root"] / "frame_metrics.csv"
    bag = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row.get("pair_id") and row.get("alignment") not in (None, ""):
                bag.append(row)
    sweep = {row["pair_id"]: row for row in correlations}
    result = []
    for pair_id in sorted(sweep):
        for direction in ("image", "lidar"):
            target = pair_id.split("+", 1)[1 if direction == "image" else 0]
            bag_rows = [row for row in bag if row["pair_id"] == pair_id
                        and row["model_id"] == target]
            bag_rho = _spearman(
                [float(row["alignment"]) for row in bag_rows],
                [float(row["inference_e2e_ms"]) for row in bag_rows],
            )
            corr = sweep[pair_id]
            negative = corr[f"{direction}_negative_offset_spearman"]
            positive = corr[f"{direction}_positive_offset_spearman"]
            pooled = corr[f"{direction}_pooled_spearman"]
            maximum = corr[f"{direction}_max_copy_pooled_spearman"]
            direction_gap = (
                abs(negative - positive)
                if negative is not None and positive is not None else None)
            averaging_gain = (
                abs(maximum) - abs(pooled)
                if maximum is not None and pooled is not None else None)
            # ponytail: descriptive two-signal comparison; add a formal model
            # only if this heuristic becomes a claimed causal classifier.
            larger = (
                "direction_dependence" if direction_gap is not None
                and averaging_gain is not None and direction_gap > averaging_gain
                else "memcpy_averaging" if direction_gap is not None
                and averaging_gain is not None else "not_observable"
            )
            result.append({
                "pair_id": pair_id,
                "target_model": target,
                "target_direction": direction,
                "bag_sample_count": len(bag_rows),
                "bag_spearman": bag_rho,
                "bag_median_copy_count": statistics.median(
                    float(row["alignment_copy_count"]) for row in bag_rows),
                "sweep_pooled_spearman": pooled,
                "sweep_overlap_only_spearman": corr[
                    f"{direction}_overlap_only_spearman"],
                "sweep_negative_offset_spearman": negative,
                "sweep_positive_offset_spearman": positive,
                "sweep_max_copy_spearman": maximum,
                "direction_dependence_gap": direction_gap,
                "memcpy_averaging_gain": averaging_gain,
                "larger_observed_contrast": larger,
            })
    return result


def _percentile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return (values[lower] * (upper - position)
            + values[upper] * (position - lower))


def _cluster_bootstrap_spearman(rows, cluster_field, repetitions, seed):
    """Return a percentile interval after resampling whole source clusters."""
    from scipy.stats import spearmanr

    clusters = {}
    for row in rows:
        clusters.setdefault(row[cluster_field], []).append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sampled = [row for key in rng.choices(keys, k=len(keys))
                   for row in clusters[key]]
        value = float(spearmanr(
            [row["alignment"] for row in sampled],
            [row["inference_e2e_ms"] for row in sampled],
        ).statistic)
        if not math.isnan(value):
            values.append(value)
    return {
        "cluster_count": len(keys),
        "valid_repetitions": len(values),
        "ci95_low": _percentile(values, 0.025),
        "ci95_high": _percentile(values, 0.975),
    }


def bag_correlation_supplements(study, repetitions=BAG_BOOTSTRAP_REPETITIONS):
    """Correlate bag rows by condition, source scene, and source timestamp."""
    path = study["paths"]["bag_analysis_root"] / "frame_metrics.csv"
    scene_by_condition = {
        condition_id: condition["scene_name"]
        for condition_id, condition in study["bag_conditions"].items()
    }
    rows = []
    with path.open(newline="", encoding="utf-8") as source:
        for raw in csv.DictReader(source):
            if (not raw.get("pair_id") or raw.get("alignment") in (None, "")
                    or raw.get("inference_e2e_ms") in (None, "")):
                continue
            timestamp = raw.get("input_timestamp_ns") or raw.get(
                "ros_header_timestamp_ns")
            if not timestamp:
                continue
            rows.append({
                **raw,
                "alignment": float(raw["alignment"]),
                "inference_e2e_ms": float(raw["inference_e2e_ms"]),
                "source_scene": scene_by_condition[raw["condition_id"]],
                "source_timestamp_ns": str(round(float(timestamp))),
            })

    targets = {}
    for row in rows:
        lidar, image = row["pair_id"].split("+", 1)
        direction = "image" if row["model_id"] == image else "lidar"
        targets.setdefault(
            (row["pair_id"], row["model_id"], direction), []).append(row)

    by_condition, by_scene = [], []
    averaged_rows, averaged_correlations, intervals = [], [], []
    for (pair_id, target, direction), target_rows in sorted(targets.items()):
        base = {
            "pair_id": pair_id,
            "target_model": target,
            "target_direction": direction,
        }
        for field, destination in (
                ("condition_id", by_condition),
                ("source_scene", by_scene)):
            groups = {}
            for row in target_rows:
                groups.setdefault(row[field], []).append(row)
            for name, values in sorted(groups.items()):
                destination.append({
                    **base,
                    field: name,
                    "condition_count": len({row["condition_id"]
                                            for row in values}),
                    "sample_count": len(values),
                    "spearman": _spearman(
                        [row["alignment"] for row in values],
                        [row["inference_e2e_ms"] for row in values]),
                })

        timestamp_groups = {}
        for row in target_rows:
            key = (row["source_scene"], row["source_timestamp_ns"])
            timestamp_groups.setdefault(key, []).append(row)
        target_averages = []
        for (scene, timestamp), values in sorted(timestamp_groups.items()):
            item = {
                **base,
                "source_scene": scene,
                "source_timestamp_ns": timestamp,
                "condition_count": len({row["condition_id"]
                                        for row in values}),
                "execution_row_count": len(values),
                "mean_alignment": statistics.fmean(
                    row["alignment"] for row in values),
                "mean_inference_e2e_ms": statistics.fmean(
                    row["inference_e2e_ms"] for row in values),
            }
            averaged_rows.append(item)
            target_averages.append(item)
        averaged_correlations.append({
            **base,
            "source_timestamp_count": len(target_averages),
            "spearman": _spearman(
                [row["mean_alignment"] for row in target_averages],
                [row["mean_inference_e2e_ms"] for row in target_averages]),
        })

        seed_text = f"{pair_id}|{target}|{direction}"
        seed = int(__import__("hashlib").sha256(
            seed_text.encode()).hexdigest()[:16], 16)
        scene_ci = _cluster_bootstrap_spearman(
            target_rows, "source_scene", repetitions, seed)
        timestamp_ci = _cluster_bootstrap_spearman(
            target_rows, "source_timestamp_ns", repetitions, seed + 1)
        intervals.append({
            **base,
            "sample_count": len(target_rows),
            "pooled_spearman": _spearman(
                [row["alignment"] for row in target_rows],
                [row["inference_e2e_ms"] for row in target_rows]),
            "bootstrap_repetitions": repetitions,
            **{f"scene_cluster_{name}": value
               for name, value in scene_ci.items()},
            **{f"timestamp_cluster_{name}": value
               for name, value in timestamp_ci.items()},
        })
    return {
        "by_condition": by_condition,
        "by_scene": by_scene,
        "timestamp_averaged_rows": averaged_rows,
        "timestamp_averaged_correlations": averaged_correlations,
        "clustered_intervals": intervals,
    }


def _plots(trials, correlations, output):
    plot_root = output / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for pair_id in sorted({row["pair_id"] for row in trials}):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, direction in zip(axes, ("image", "lidar")):
            rows = [row for row in trials if row["pair_id"] == pair_id
                    and row["target_direction"] == direction
                    and row["alignment_score_a"] is not None
                    and row["nvtx_inference_e2e_ms"] is not None]
            axis.scatter(
                [row["alignment_score_a"] for row in rows],
                [row["nvtx_inference_e2e_ms"] for row in rows],
                c=[row["configured_signed_offset_ms"] for row in rows],
                cmap="coolwarm", s=24, alpha=0.8,
            )
            axis.set(title=f"{direction} target", xlabel="alignment A",
                     ylabel="NVTX inference E2E (ms)")
            axis.grid(alpha=0.25)
        figure.suptitle(pair_id)
        figure.tight_layout()
        figure.savefig(plot_root / f"inference-vs-a-{pair_id}.png", dpi=160)
        plt.close(figure)

    columns = [
        f"{direction}_{scope}_spearman"
        for direction in ("image", "lidar")
        for scope in ("pooled", "overlap_only", "negative_offset",
                      "positive_offset")
    ]
    values = np.asarray([[
        row[column] if row[column] is not None else np.nan
        for column in columns] for row in correlations])
    figure, axis = plt.subplots(figsize=(13, 9))
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(columns)), columns, rotation=55, ha="right")
    axis.set_yticks(range(len(correlations)),
                    [row["pair_id"] for row in correlations])
    axis.set_title("Inference-time versus A Spearman correlations")
    figure.colorbar(image, ax=axis, label="Spearman rho")
    figure.tight_layout()
    figure.savefig(plot_root / "18-pair-bidirectional-correlation-heatmap.png",
                   dpi=160, bbox_inches="tight")
    plt.close(figure)


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values)


def _bag_stat_plots(study, trials, supplements, output):
    """Plot source-aware bag correlations and bag-versus-sweep A support."""
    plot_root = output / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    pair_id = "pointpillars+faster-rcnn"
    bag_path = study["paths"]["bag_analysis_root"] / "frame_metrics.csv"
    with bag_path.open(newline="", encoding="utf-8") as source:
        bag = [row for row in csv.DictReader(source)
               if row.get("pair_id") == pair_id
               and row.get("alignment") not in (None, "")]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, direction, target in zip(
            axes, ("image", "lidar"), ("faster-rcnn", "pointpillars")):
        bag_values = [float(row["alignment"]) for row in bag
                      if row["model_id"] == target]
        sweep_values = [row["alignment_score_a"] for row in trials
                        if row["pair_id"] == pair_id
                        and row["target_direction"] == direction
                        and row["alignment_score_a"] is not None]
        for label, values, color in (
                ("bag", bag_values, "tab:blue"),
                ("sweep", sweep_values, "tab:orange")):
            x, y = _ecdf(values)
            axis.step(
                x, y, where="post", color=color, linewidth=2,
                label=(f"{label}: n={len(values)}, "
                       f"median={statistics.median(values):.3f}"),
            )
        axis.set(title=f"{target} target", xlabel="alignment score A",
                 xlim=(-0.02, 0.65), ylim=(0, 1.01))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("empirical cumulative probability")
    figure.suptitle("PointPillars–Faster R-CNN: bag versus sweep A")
    figure.tight_layout()
    figure.savefig(
        plot_root / "pointpillars-faster-rcnn-a-distributions.png",
        dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True)
    scenes = sorted({row["source_scene"]
                     for row in supplements["by_scene"]
                     if row["pair_id"] == pair_id})
    colors = dict(zip(scenes, plt.get_cmap("tab10").colors))
    for axis, direction, target in zip(
            axes, ("image", "lidar"), ("faster-rcnn", "pointpillars")):
        conditions = [row for row in supplements["by_condition"]
                      if row["pair_id"] == pair_id
                      and row["target_direction"] == direction]
        scene_rows = {row["source_scene"]: row
                      for row in supplements["by_scene"]
                      if row["pair_id"] == pair_id
                      and row["target_direction"] == direction}
        condition_scene = {
            name: value["scene_name"]
            for name, value in study["bag_conditions"].items()
        }
        for y, scene in enumerate(scenes):
            values = [row for row in conditions
                      if condition_scene[row["condition_id"]] == scene]
            jitter = np.linspace(-0.22, 0.22, len(values))
            axis.scatter(
                [row["spearman"] for row in values], y + jitter,
                color=colors[scene], s=30, alpha=0.75,
                label="condition" if y == 0 else None,
            )
            axis.scatter(
                [scene_rows[scene]["spearman"]], [y], marker="D", s=65,
                color="black", label="scene pooled" if y == 0 else None,
            )
        averaged = next(
            row for row in supplements["timestamp_averaged_correlations"]
            if row["pair_id"] == pair_id
            and row["target_direction"] == direction)
        interval = next(
            row for row in supplements["clustered_intervals"]
            if row["pair_id"] == pair_id
            and row["target_direction"] == direction)
        pooled = interval["pooled_spearman"]
        axis.axvline(pooled, color="black", linestyle="--", linewidth=1.3,
                     label=f"raw pooled {pooled:.3f}")
        axis.axvline(
            averaged["spearman"], color="tab:purple", linestyle=":",
            linewidth=1.8,
            label=f"timestamp averaged {averaged['spearman']:.3f}")
        for y, prefix, color, label in (
                (-1, "scene_cluster", "tab:blue", "scene-cluster 95% CI"),
                (-2, "timestamp_cluster", "tab:orange",
                 "timestamp-cluster 95% CI")):
            low, high = (interval[f"{prefix}_ci95_low"],
                         interval[f"{prefix}_ci95_high"])
            axis.errorbar(
                pooled, y, xerr=[[pooled - low], [high - pooled]],
                fmt="s", color=color, capsize=4, label=label)
        axis.set(
            title=f"{target} target", xlabel="Spearman correlation",
            xlim=(-1, 1),
            yticks=[-2, -1, *range(len(scenes))],
            yticklabels=["timestamp-cluster CI", "scene-cluster CI", *scenes],
        )
        axis.axvline(0, color="grey", linewidth=0.8)
        axis.grid(axis="x", alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle(
        "PointPillars–Faster R-CNN: condition and source-aware correlations")
    figure.tight_layout()
    figure.savefig(
        plot_root / "pointpillars-faster-rcnn-source-aware-correlations.png",
        dpi=180, bbox_inches="tight")
    plt.close(figure)

    available_pairs = {row["pair_id"]
                       for row in supplements["clustered_intervals"]}
    pairs = [f"{lidar}+{image}" for lidar in LIDAR_ORDER
             for image in IMAGE_ORDER
             if f"{lidar}+{image}" in available_pairs]
    averaged_index = {
        (row["pair_id"], row["target_direction"]): row
        for row in supplements["timestamp_averaged_correlations"]
    }
    interval_index = {
        (row["pair_id"], row["target_direction"]): row
        for row in supplements["clustered_intervals"]
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 10), sharex=True,
                                sharey=True)
    for axis, direction in zip(axes, ("image", "lidar")):
        for y, name in enumerate(reversed(pairs)):
            interval = interval_index[(name, direction)]
            pooled = interval["pooled_spearman"]
            for shift, prefix, color, label in (
                    (-0.12, "scene_cluster", "tab:blue", "scene cluster"),
                    (0.12, "timestamp_cluster", "tab:orange",
                     "timestamp cluster")):
                low, high = (interval[f"{prefix}_ci95_low"],
                             interval[f"{prefix}_ci95_high"])
                axis.errorbar(
                    pooled, y + shift,
                    xerr=[[pooled - low], [high - pooled]], fmt="o",
                    markersize=3.5, capsize=2, color=color,
                    label=label if y == 0 else None)
            axis.scatter(
                averaged_index[(name, direction)]["spearman"], y,
                marker="x", color="tab:purple", s=28,
                label="timestamp averaged" if y == 0 else None)
        axis.set(title=f"{direction} target", xlabel="Spearman correlation",
                 xlim=(-1, 1))
        axis.axvline(0, color="grey", linewidth=0.8)
        axis.grid(axis="x", alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    axes[0].set_yticks(range(len(pairs)), list(reversed(pairs)), fontsize=8)
    figure.suptitle("Bag A versus inference time: clustered uncertainty")
    figure.tight_layout()
    figure.savefig(
        plot_root / "18-pair-source-aware-correlation-intervals.png",
        dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_campaign(study):
    """Write all retained sweep rows, correlations, plots, and bag comparison."""
    manifest = _campaign_manifest(study)
    trials, copies = [], []
    for cell in sorted(manifest["cells"], key=lambda item: item[
            "execution_index"]):
        cell_trials, cell_copies = _cell_rows(study, cell)
        trials.extend(cell_trials)
        copies.extend(cell_copies)
    offsets = per_offset_rows(trials)
    correlations = correlation_rows(trials)
    comparison = _bag_comparison(study, trials, correlations)
    bag_supplements = bag_correlation_supplements(study)
    output = study["paths"]["output_root"] / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_trial.csv", trials)
    _write_csv(output / "per_offset.csv", offsets)
    _write_csv(output / "per_copy_alignment.csv", copies)
    _write_csv(output / "correlations_18_pair.csv", correlations)
    _write_csv(output / "bag_vs_offset_correlations.csv", comparison)
    _write_csv(output / "bag_correlations_by_condition.csv",
               bag_supplements["by_condition"])
    _write_csv(output / "bag_correlations_by_source_scene.csv",
               bag_supplements["by_scene"])
    _write_csv(output / "bag_timestamp_averaged_rows.csv",
               bag_supplements["timestamp_averaged_rows"])
    _write_csv(output / "bag_correlations_timestamp_averaged.csv",
               bag_supplements["timestamp_averaged_correlations"])
    _write_csv(output / "bag_correlations_clustered_ci.csv",
               bag_supplements["clustered_intervals"])
    _write_csv(output / "faster_rcnn_explanation.csv", [
        row for row in comparison if row["target_model"] == "faster-rcnn"])
    _plots(trials, correlations, output)
    _bag_stat_plots(study, trials, bag_supplements, output)
    result = {
        "pair_count": len({row["pair_id"] for row in trials}),
        "trial_row_count": len(trials),
        "offset_row_count": len(offsets),
        "copy_row_count": len(copies),
        "correlation_pair_count": len(correlations),
        "bag_condition_correlation_row_count": len(
            bag_supplements["by_condition"]),
        "bag_scene_correlation_row_count": len(
            bag_supplements["by_scene"]),
        "bag_timestamp_averaged_row_count": len(
            bag_supplements["timestamp_averaged_rows"]),
        "bag_clustered_interval_row_count": len(
            bag_supplements["clustered_intervals"]),
        "source_aware_plot_count": 3,
        "warning_trial_row_count": sum(
            bool(row["diagnostic_warnings"]) for row in trials),
        "all_trial_rows_retained": True,
    }
    _write_json(output / "analysis_manifest.json", result)
    return result


def main(argv=None):
    """Plan, validate, dry-run, execute, or analyze the offset sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "validate", "run",
                                            "analyze"))
    parser.add_argument("study")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    study = load_study(args.study)
    study["paths"]["output_root"] = Path(args.artifact_root).resolve()
    if args.command == "plan":
        result = {"pairs": study["pairs"], "pair_count": len(study["pairs"])}
    elif args.command == "validate":
        result = {
            "pair_count": len(study["pairs"]),
            "first_offsets_ms": [
                value for _, value in zip(range(9), offset_order())],
        }
    elif args.command == "run":
        result = run_campaign(study, args.artifact_root, args.dry_run)
    else:
        result = analyze_campaign(study)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
