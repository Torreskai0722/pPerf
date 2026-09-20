"""Descriptive analysis for the cross-modal input-variation campaign."""
# flake8: noqa: E501

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
import gzip
import json
import os
from pathlib import Path
import shutil

import matplotlib
import numpy as np
import yaml

from .._common import (
    StudyError, load_study as _read_study, sha256 as _sha256,
)
from ._input_validation import validate_input_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

SCOPE = "single-run exploratory descriptive"
QUANTILES = (0, 1, 5, 25, 50, 75, 95, 99)


def _resolve(source, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def load_study(path):
    """Load the read-only campaign contract needed by this analyzer."""
    study = _read_study(path)
    data = study["data"]
    required = {
        "models", "conditions", "baseline_map", "mps_modes", "output_root",
    }
    missing = required - set(data)
    if missing:
        raise StudyError(f"study fields missing={sorted(missing)}")
    models = data["models"]
    lidar = tuple(sorted(
        name for name, model in models.items()
        if model.get("modality") == "lidar"
    ))
    camera = tuple(sorted(
        name for name, model in models.items()
        if model.get("modality") == "image"
    ))
    if not lidar or not camera or len(lidar) + len(camera) != len(models):
        raise StudyError("every model must be image or LiDAR")
    paths = {
        name: _resolve(study["source"], data[name])
        for name in (
            "base_config", "output_root", "generated_config_root",
            "checkpoint_root", "metadata_path", "warmup_image",
            "warmup_lidar",
        ) if name in data
    }
    return {**study, "paths": paths, "lidar": lidar, "camera": camera}


def _model_pairs(study):
    return tuple(
        (lidar, camera)
        for lidar in study["lidar"] for camera in study["camera"]
    )


def expected_runs(study):
    """Return the authored two-model matrix without materializing configs."""
    return {
        f"{int(mps)}-{condition}-{lidar}-{camera}": {
            "condition_id": condition,
            "lidar_model": lidar,
            "camera_model": camera,
            "models": [camera, lidar],
            "mps_enabled": mps,
            "replicate": 1,
            "tenancy": "two_model",
        }
        for mps in study["data"]["mps_modes"]
        for condition in study["data"]["conditions"]
        for lidar, camera in _model_pairs(study)
    }


def expected_isolated_runs(study):
    """Return the authored isolated 3DSSD analysis cells."""
    return {
        f"isolated-{condition}-3dssd": {
            "condition_id": condition,
            "models": ["3dssd"],
            "mps_enabled": False,
            "replicate": 1,
            "tenancy": "single",
        }
        for condition in study["data"]["conditions"]
    }


def _selected_runs(study, pairs=(), mps_modes=()):
    pairs, mps_modes = set(pairs), set(mps_modes)
    return {
        run_id: planned
        for run_id, planned in expected_runs(study).items()
        if (
            not pairs
            or f"{planned['lidar_model']}+{planned['camera_model']}" in pairs
        ) and (
            not mps_modes
            or ("on" if planned["mps_enabled"] else "off") in mps_modes
        )
    }


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _input_records(run_directory, model_id):
    return {
        str(record["input_id"]): record
        for record in (json.loads(line) for line in (
            Path(run_directory) / f"model_{model_id}_inputs.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip())
    }


@contextmanager
def _readable_profile(run_directory):
    """Temporarily restore an archived SQLite trace for existing analyzers."""
    run_directory = Path(run_directory)
    sqlite_path = run_directory / "profile.sqlite"
    archived = run_directory / "profile.sqlite.gz"
    created = False
    if not sqlite_path.is_file() and archived.is_file():
        temporary = run_directory / "profile.sqlite.analysis.tmp"
        with gzip.open(archived, "rb") as source, temporary.open("wb") as dst:
            shutil.copyfileobj(source, dst)
        os.replace(temporary, sqlite_path)
        created = True
    try:
        yield
    finally:
        if created and sqlite_path.is_file():
            sqlite_path.unlink()


def characterize_inputs(study):
    """Read payload characteristics prepared by the experiment package."""
    path = Path(study["paths"].get(
        "input_characteristics",
        study["paths"]["output_root"] / "input_characteristics.json",
    ))
    if not path.is_file():
        raise StudyError(
            "input characteristics evidence is missing; prepare the dataset "
            "before running offline analysis"
        )
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise StudyError(f"input characteristics evidence is invalid: {path}")
    return rows


def input_summary_rows(rows):
    """Summarize each measured input characteristic descriptively."""
    metadata = {
        "condition_id", "effect", "corruption_type", "severity",
        "rain_rate_mm_per_hour", "modality", "input_timestamp_ns",
    }
    result = []
    groups = {}
    for row in rows:
        groups.setdefault((row["condition_id"], row["modality"]), []).append(
            row)
    for (condition_id, modality), values in sorted(groups.items()):
        fields = sorted(set.intersection(*(
            set(value) - metadata for value in values)))
        for field in fields:
            metrics = distribution_metrics([value[field] for value in values])
            result.append({
                "condition_id": condition_id,
                "modality": modality,
                "metric": field,
                **metrics,
            })
    return result


def input_change_rows(rows, baseline_map):
    """Compare paired adverse payload metrics to their clear source inputs."""
    metadata = {
        "condition_id", "effect", "corruption_type", "severity",
        "rain_rate_mm_per_hour", "modality", "input_timestamp_ns",
    }
    indexed = {
        (row["condition_id"], row["modality"], row["input_timestamp_ns"]):
        row for row in rows
    }
    result = []
    for condition_id, clear_id in baseline_map.items():
        for modality in ("image", "lidar"):
            adverse = [
                row for row in rows
                if row["condition_id"] == condition_id
                and row["modality"] == modality
            ]
            paired = [
                (row, indexed.get((clear_id, modality,
                                   row["input_timestamp_ns"])))
                for row in adverse
            ]
            paired = [(left, right) for left, right in paired
                      if right is not None]
            if not paired:
                continue
            fields = sorted(set.intersection(*(
                (set(left) & set(right)) - metadata
                for left, right in paired)))
            for field in fields:
                adverse_values = [float(left[field]) for left, _ in paired]
                clear_values = [float(right[field]) for _, right in paired]
                adverse_median = float(np.median(adverse_values))
                clear_median = float(np.median(clear_values))
                changes = [left - right for left, right in zip(
                    adverse_values, clear_values)]
                result.append({
                    "condition_id": condition_id,
                    "clear_baseline_id": clear_id,
                    "modality": modality,
                    "metric": field,
                    "sample_count": len(paired),
                    "adverse_median": adverse_median,
                    "clear_median": clear_median,
                    "median_paired_change": float(np.median(changes)),
                    "median_ratio": (
                        adverse_median / clear_median
                        if clear_median else None),
                })
    return result


def distribution_metrics(values):
    """Return the campaign's absolute latency-distribution measurements."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("latency distribution is empty")
    percentiles = dict(zip(
        QUANTILES, np.percentile(values, QUANTILES, method="linear").astype(float)
    ))
    p0, p5, p25, median, p75, p95, p99 = (
        percentiles[key] for key in (0, 5, 25, 50, 75, 95, 99)
    )
    return {
        "frame_count": int(len(values)),
        "sample_count": int(len(values)),
        "minimum_ms": p0,
        "maximum_ms": float(np.max(values)),
        "median_ms": median,
        "iqr_ms": p75 - p25,
        **{f"p{key}_ms": value for key, value in percentiles.items()},
        "p0_p99_range_ms": p99 - p0,
        "primary_range_ms": p99 - p0,
        # Compatibility alias for older report consumers; this is P0-P99.
        "width_ms": p99 - p0,
        "normalized_range": (p99 - p0) / median if median else None,
        "R": (p99 - median) / median if median else None,
        "p50_p99_range_ms": p99 - median,
    }


def wasserstein_1(left, right):
    """Return the one-dimensional empirical Wasserstein distance."""
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(left, right))


def variation_metrics(adverse_width, clear_width):
    """Compare weather and clear widths directly within one model pair."""
    change = float(adverse_width) - float(clear_width)
    if clear_width == 0:
        return {
            "variation_change_ms": change,
            "variation_factor": None,
            "percentage_change": None,
            "observed_label": "not_defined_zero_clear_width",
        }
    factor = float(adverse_width) / float(clear_width)
    if adverse_width > clear_width:
        label = "observed_amplification"
    elif adverse_width < clear_width:
        label = "observed_damping"
    else:
        label = "observed_unchanged"
    return {
        "variation_change_ms": change,
        "variation_factor": factor,
        "percentage_change": (factor - 1.0) * 100.0,
        "observed_label": label,
    }


def _validate_run(run_directory, expected_config, planned, study):
    policy = {
        "minimum_samples": {"image": 10, "lidar": 10},
        "timestamp_comparability": "source_manifest_window",
        "minimum_window_fraction": 0.95,
    }
    with _readable_profile(run_directory):
        evidence = validate_input_run(
            run_directory, planned, study, expected_config, policy
        )
    inputs = {
        model_id: _input_records(run_directory, model_id)
        for model_id in planned["models"]
        if (Path(run_directory) / f"model_{model_id}_inputs.jsonl").is_file()
    }
    return evidence, inputs


def _validate_run_job(arguments):
    """Validate one independent run in a worker process."""
    return _validate_run(*arguments)


def _collect(study, planned_runs, configs):
    """Collect frame, module, and structure rows from planned runs."""
    output_root = study["paths"]["output_root"]
    validation, frames, modules, structures = [], [], [], []
    jobs = {
        run_id: (output_root / run_id, configs[run_id], planned, study)
        for run_id, planned in planned_runs.items()
        if (output_root / run_id).is_dir()
    }
    if len(jobs) < 2:
        results = {
            run_id: _validate_run_job(arguments)
            for run_id, arguments in jobs.items()
        }
    else:
        with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as executor:
            results = dict(zip(jobs, executor.map(
                _validate_run_job, jobs.values())))
    for run_id, planned in planned_runs.items():
        run_directory = output_root / run_id
        if not run_directory.is_dir():
            validation.append({
                "run_id": run_id,
                "model_id": (planned["models"][0]
                             if len(planned["models"]) == 1 else ""),
                "valid": False,
                "extractable": False,
                "errors": "run directory missing",
                "source": "goal1_campaign",
                "raw_artifact_path": str(run_directory),
                "analysis_scope": SCOPE,
            })
            continue
        evidence, inputs = results[run_id]
        errors = list(evidence["errors"])
        run_frames = evidence.get("frames", [])
        run_modules = evidence.get("modules", [])
        run_structures = evidence.get("structures", [])
        validation.append({
            "run_id": run_id,
            "model_id": (planned["models"][0]
                         if len(planned["models"]) == 1 else ""),
            "valid": not errors,
            "extractable": bool(run_frames),
            "errors": "; ".join(errors),
            "source": "goal1_campaign",
            "raw_artifact_path": str(run_directory),
            "analysis_scope": SCOPE,
        })
        lidar = planned.get("lidar_model", "")
        camera = planned.get("camera_model", "")
        common = {
            "run_id": run_id,
            "condition_id": planned["condition_id"],
            "lidar_model": lidar,
            "camera_model": camera,
            "pair_id": f"{lidar}+{camera}" if lidar and camera else "",
            "mps_mode": "on" if planned["mps_enabled"] else "off",
            "replicate": 1,
            "tenancy": planned["tenancy"],
            "analysis_scope": SCOPE,
            "run_valid": not errors,
            "run_diagnostics": "; ".join(errors),
        }
        for row in run_frames:
            record = inputs.get(row["model_id"], {}).get(
                str(row["frame_id"]), {})
            row.update(common)
            row["input_timestamp_ns"] = record.get(
                "ros_header_timestamp_ns")
            row["input_point_count"] = record.get("input_point_count")
            row["pass_index"] = record.get("pass_index", 0)
            row["segment_id"] = record.get("segment_id", "")
            row["waiting_ms"] = row["memcpy_adjacent_waiting_ns"] / 1_000_000
            frames.append(row)
        for row in run_modules + run_structures:
            record = inputs.get(row["model_id"], {}).get(
                str(row["input_id"]), {})
            row.update(common)
            row["input_timestamp_ns"] = record.get(
                "ros_header_timestamp_ns")
            row["input_point_count"] = record.get("input_point_count")
            row["pass_index"] = record.get("pass_index", 0)
            row["segment_id"] = record.get("segment_id", "")
        modules.extend(run_modules)
        structures.extend(run_structures)
    return validation, frames, modules, structures


def collect(study, pairs=(), mps_modes=()):
    """Collect the selected two-model campaign runs."""
    planned = _selected_runs(study, pairs, mps_modes)
    return _collect(study, planned, dict.fromkeys(planned))


def collect_reference_pairs(study):
    """Load the 12 validated non-MPS Section 4.1 pair results."""
    source = study["source"].parents[1] / "section4_1" / "study.yaml"
    reference = yaml.safe_load(source.read_bytes())
    output_root = (source.parent / reference["output_root"]).resolve()
    analysis_root = output_root / "analysis"
    validation_summary = _load_json(
        analysis_root / "validation_summary.json")
    if (
        not validation_summary.get("complete")
        or validation_summary.get("study_sha256") != _sha256(source)
        or validation_summary.get("valid_runs")
        != validation_summary.get("planned_runs")
    ):
        raise StudyError("Section 4.1 pair reference is not complete")

    aliases = {"vit": "vit-upernet", "yolo": "yolov3"}
    lidar = {"pointpillars", "centerpoint"}
    camera = set(study["camera"])
    result = []
    run_pairs = {}
    for name in ("frame_metrics.csv", "module_metrics.csv",
                 "structure_metrics.csv"):
        rows = []
        with (analysis_root / name).open(newline="", encoding="utf-8") as src:
            for raw in csv.DictReader(src):
                if raw.get("tenancy") != "two_model":
                    continue
                row = {key: _reference_value(value)
                       for key, value in raw.items()}
                row["model_id"] = aliases.get(
                    str(row["model_id"]), str(row["model_id"]))
                if name == "frame_metrics.csv":
                    other = aliases.get(
                        str(row["other_model_ids"]),
                        str(row["other_model_ids"]),
                    )
                    models = {str(row["model_id"]), other}
                    lidar_model = next(iter(models & lidar), None)
                    camera_model = next(iter(models & camera), None)
                    if lidar_model is None or camera_model is None:
                        continue
                    run_pairs[str(row["run_id"])] = (
                        lidar_model, camera_model)
                pair = run_pairs.get(str(row["run_id"]))
                if pair is None:
                    continue
                row.update({
                    "lidar_model": pair[0],
                    "camera_model": pair[1],
                    "pair_id": f"{pair[0]}+{pair[1]}",
                    "mps_mode": "off",
                    "replicate": 1,
                    "analysis_scope": SCOPE,
                    "run_valid": True,
                    "run_diagnostics": "",
                    "source": "reused_section4_1_pair",
                })
                if name == "frame_metrics.csv":
                    row["input_timestamp_ns"] = row.get(
                        "ros_header_timestamp_ns")
                    row["waiting_ms"] = (
                        float(row["memcpy_adjacent_waiting_ns"]) / 1_000_000)
                rows.append(row)
        result.append(rows)
    validation = [{
        "run_id": run_id,
        "model_id": "",
        "valid": True,
        "extractable": True,
        "errors": "",
        "source": "reused_section4_1_pair",
        "raw_artifact_path": str(output_root / run_id),
        "analysis_scope": SCOPE,
    } for run_id in sorted(run_pairs)]
    return (validation, *result)


def _reference_value(value):
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def collect_reference_isolated(study):
    """Reuse the validated Section 4.1 isolated runs for seven exact models."""
    source = study["source"].parents[1] / "section4_1" / "study.yaml"
    reference = yaml.safe_load(source.read_bytes())
    reference_sha = _sha256(source)
    output_root = (source.parent / reference["output_root"]).resolve()
    analysis_root = output_root / "analysis"
    validation_summary = _load_json(
        analysis_root / "validation_summary.json"
    )
    if (
        not validation_summary.get("complete")
        or validation_summary.get("study_sha256") != reference_sha
        or validation_summary.get("valid_runs")
        != validation_summary.get("planned_runs")
    ):
        raise StudyError("Section 4.1 isolated reference is not complete")

    current_to_reference = {"vit-upernet": "vit", "yolov3": "yolo"}
    reference_to_current = {
        value: key for key, value in current_to_reference.items()
    }
    fields = (
        "mmlab_model", "architecture_profile", "model_config_sha256",
        "checkpoint_sha256",
    )
    expected = {}
    for current_model, model in study["data"]["models"].items():
        if current_model == "3dssd":
            continue
        reference_model = current_to_reference.get(current_model,
                                                   current_model)
        old_model = reference["models"].get(reference_model)
        if old_model is None or any(
            old_model.get(field) != model.get(field) for field in fields
        ):
            raise StudyError(
                f"Section 4.1 model differs from {current_model}"
            )
        for condition_id, condition in study["data"]["conditions"].items():
            old_condition = reference["conditions"].get(condition_id)
            if old_condition is None:
                raise StudyError(
                    f"Section 4.1 condition is missing: {condition_id}"
                )
            old_bag = (source.parent /
                       old_condition["bag_directory"]).resolve()
            old_manifest = (source.parent /
                            old_condition["dataset_manifest"]).resolve()
            if (
                old_bag != Path(condition["bag_directory"]).resolve()
                or _sha256(old_manifest)
                != condition["dataset_manifest_sha256"]
            ):
                raise StudyError(
                    f"Section 4.1 input differs for {condition_id}"
                )
            run_id = (
                f"iv-{condition_id}-single-{reference_model}-r1"
            )
            expected[run_id] = {
                "condition_id": condition_id,
                "current_model": current_model,
                "reference_model": reference_model,
            }

    validation = []
    input_records = {}
    for run_id, planned in expected.items():
        run_directory = output_root / run_id
        config = yaml.safe_load(
            (run_directory / "config.yaml").read_bytes()
        )
        manifest = _load_json(run_directory / "run_manifest.json")
        clocks = _load_json(
            run_directory / "input_variation_clock_control.json"
        )
        model = config.get("models", [{}])[0]
        corruption = config.get("input_variation", {}).get(
            "corruption", {}
        )
        errors = []
        if manifest.get("state") != "success":
            errors.append(f"run state is {manifest.get('state')}")
        if not clocks.get("clocks_restored"):
            errors.append("GPU clocks were not restored")
        if config.get("gpu", {}).get("mps_enabled"):
            errors.append("isolated reference used MPS")
        if (
            len(config.get("models", [])) != 1
            or model.get("id") != planned["reference_model"]
        ):
            errors.append("isolated reference model differs")
        if any(
            model.get(field)
            != study["data"]["models"][planned["current_model"]].get(field)
            for field in fields
        ):
            errors.append("isolated reference model hash differs")
        if (
            config.get("input_variation", {}).get("condition_id")
            != planned["condition_id"]
            or corruption.get("dataset_manifest_sha256")
            != study["data"]["conditions"][planned["condition_id"]][
                "dataset_manifest_sha256"
            ]
        ):
            errors.append("isolated reference input differs")
        validation.append({
            "run_id": run_id,
            "model_id": planned["current_model"],
            "valid": not errors,
            "extractable": not errors,
            "errors": "; ".join(errors),
            "source": "reused_section4_1",
            "raw_artifact_path": str(run_directory),
            "analysis_scope": SCOPE,
        })
        if errors:
            continue
        input_records[run_id] = _input_records(
            run_directory, planned["reference_model"]
        )
    if any(not row["valid"] for row in validation):
        raise StudyError("Section 4.1 isolated reference validation failed")

    result = []
    for name in ("frame_metrics.csv", "module_metrics.csv",
                 "structure_metrics.csv"):
        rows = []
        with (analysis_root / name).open(newline="", encoding="utf-8") as src:
            for row in csv.DictReader(src):
                planned = expected.get(row["run_id"])
                if planned is None:
                    continue
                input_id = str(row.get("frame_id", row.get("input_id")))
                row = {key: _reference_value(value)
                       for key, value in row.items()}
                row["model_id"] = reference_to_current.get(
                    str(row["model_id"]), str(row["model_id"])
                )
                record = input_records[row["run_id"]][input_id]
                row.update({
                    "condition_id": planned["condition_id"],
                    "lidar_model": "",
                    "camera_model": "",
                    "pair_id": "",
                    "mps_mode": "off",
                    "replicate": 1,
                    "tenancy": "single",
                    "analysis_scope": SCOPE,
                    "input_timestamp_ns": record["ros_header_timestamp_ns"],
                    "input_point_count": record.get("input_point_count"),
                })
                if name == "frame_metrics.csv":
                    row["waiting_ms"] = (
                        float(row["memcpy_adjacent_waiting_ns"]) / 1_000_000
                    )
                rows.append(row)
        result.append(rows)
    return (validation, *result)


def collect_isolated(study):
    """Collect all eight isolated models, reusing validated prior runs."""
    reference = collect_reference_isolated(study)
    planned = expected_isolated_runs(study)
    current = _collect(study, planned, dict.fromkeys(planned))
    return tuple(left + right for left, right in zip(reference, current))


def collect_3dssd_isolated(study):
    """Collect only the 15 newly required isolated 3DSSD runs."""
    planned = expected_isolated_runs(study)
    return _collect(study, planned, dict.fromkeys(planned))


def summarize(frames):
    """Summarize latency for each pair, target, condition, and MPS mode."""
    groups = {}
    for row in frames:
        key = (
            row["pair_id"],
            row["lidar_model"],
            row["camera_model"],
            row["model_id"],
            row["condition_id"],
            row["mps_mode"],
        )
        groups.setdefault(key, []).append(row["inference_e2e_ms"])
    return [{
        "pair_id": key[0],
        "lidar_model": key[1],
        "camera_model": key[2],
        "target_model": key[3],
        "co_runner": key[2] if key[3] == key[1] else key[1],
        "condition_id": key[4],
        "mps_mode": key[5],
        "analysis_scope": SCOPE,
        **distribution_metrics(values),
    } for key, values in sorted(groups.items())]


def comparison_rows(summary, baseline_map):
    """Build direct adverse-versus-clear comparisons within each pair."""
    indexed = {
        (
            row["pair_id"], row["target_model"], row["condition_id"],
            row["mps_mode"]
        ): row
        for row in summary
    }
    result = []
    for row in summary:
        condition_id = row["condition_id"]
        if condition_id not in baseline_map:
            continue
        baseline_id = baseline_map[condition_id]
        baseline = indexed.get((row["pair_id"], row["target_model"],
                                baseline_id, row["mps_mode"]))
        if baseline is None:
            continue
        result.append({
            "pair_id":
            row["pair_id"],
            "lidar_model":
            row["lidar_model"],
            "camera_model":
            row["camera_model"],
            "target_model":
            row["target_model"],
            "co_runner":
            row["co_runner"],
            "condition_id":
            condition_id,
            "clear_baseline_id":
            baseline_id,
            "mps_mode":
            row["mps_mode"],
            "adverse_p0_p99_range_ms":
            row["p0_p99_range_ms"],
            "clear_p0_p99_range_ms":
            baseline["p0_p99_range_ms"],
            "adverse_width_ms":
            row["p0_p99_range_ms"],
            "clear_width_ms":
            baseline["p0_p99_range_ms"],
            "analysis_scope":
            SCOPE,
            **variation_metrics(row["p0_p99_range_ms"],
                                baseline["p0_p99_range_ms"]),
        })
    return result


def predictability_table_rows(summary, conditions, baseline_map, mode):
    """Pivot one complete MPS mode to one row per pair and evaluated model."""
    selected = [row for row in summary if row["mps_mode"] == mode]
    grouped = {}
    for row in selected:
        grouped.setdefault((row["pair_id"], row["target_model"]), {})[
            row["condition_id"]] = row
    rows = []
    for (pair_id, target_model), values in sorted(grouped.items()):
        if set(values) != set(conditions):
            continue
        row = {
            "pair_id": pair_id,
            "evaluated_model": target_model,
            "mps_mode": mode,
        }
        for condition_id in conditions:
            interval = values[condition_id]["p0_p99_range_ms"]
            row[f"{condition_id}_p0_p99_range_ms"] = interval
            clear_id = baseline_map.get(condition_id)
            if clear_id is not None:
                row[f"{condition_id}_percentage_change"] = variation_metrics(
                    interval, values[clear_id]["p0_p99_range_ms"]
                )["percentage_change"]
        rows.append(row)
    return rows


def ecdf_rows(frames):
    """Return every empirical latency CDF as a table."""
    groups = {}
    for row in frames:
        key = (row["pair_id"], row["model_id"], row["condition_id"],
               row["mps_mode"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(groups.items()):
        values.sort(key=lambda row: float(row["inference_e2e_ms"]))
        for rank, row in enumerate(values, 1):
            result.append({
                "pair_id": key[0],
                "target_model": key[1],
                "condition_id": key[2],
                "mps_mode": key[3],
                "latency_ms": row["inference_e2e_ms"],
                "pass_index": row.get("pass_index", 0),
                "segment_id": row.get("segment_id", ""),
                "ecdf": rank / len(values),
                "analysis_scope": SCOPE,
            })
    return result


def mechanism_rows(frames, structures):
    """Join frame latency and dynamic structure counts on model and input."""
    counts = {}
    for row in structures:
        counts.setdefault(
            (row["run_id"], row["model_id"], str(row["input_id"])),
            {})[row["structure_kind"]] = row["structure_count"]
    result = []
    for row in frames:
        dynamic = counts.get(
            (row["run_id"], row["model_id"], str(row["frame_id"])), {})
        result.append({
            "run_id": row["run_id"],
            "model_id": row["model_id"],
            "input_timestamp_ns": row["input_timestamp_ns"],
            "input_id": row["frame_id"],
            "pass_index": row.get("pass_index", 0),
            "segment_id": row.get("segment_id", ""),
            "pair_id": row["pair_id"],
            "condition_id": row["condition_id"],
            "mps_mode": row["mps_mode"],
            "inference_e2e_ms": row["inference_e2e_ms"],
            "gpu_kernel_active_ms": row["gpu_kernel_active_ms"],
            "kernel_count": row["kernel_count"],
            "kernel_span_ms": row["kernel_span_ms"],
            "waiting_ms": row["waiting_ms"],
            "input_point_count": row.get("input_point_count"),
            "occupied_voxel_count": dynamic.get("voxels"),
            "sampled_point_count": dynamic.get("sampled_points"),
            "proposal_count": dynamic.get("proposals"),
            "analysis_scope": SCOPE,
        })
    return result


def mechanism_comparison_rows(rows, baseline_map):
    """Compare adverse and clear mechanism-distribution widths directly."""
    fields = (
        "inference_e2e_ms", "gpu_kernel_active_ms", "kernel_count",
        "kernel_span_ms", "waiting_ms", "input_point_count",
        "occupied_voxel_count", "sampled_point_count", "proposal_count",
    )
    grouped = {}
    for row in rows:
        key = (row["pair_id"], row["model_id"], row["condition_id"],
               row["mps_mode"])
        grouped.setdefault(key, []).append(row)
    result = []
    for (pair_id, model_id, condition_id, mode), values in sorted(
            grouped.items()):
        clear_id = baseline_map.get(condition_id)
        clear = grouped.get((pair_id, model_id, clear_id, mode))
        if clear is None:
            continue
        for field in fields:
            adverse_values = [row.get(field) for row in values
                              if row.get(field) is not None]
            clear_values = [row.get(field) for row in clear
                            if row.get(field) is not None]
            if not adverse_values or not clear_values:
                continue
            adverse = distribution_metrics(adverse_values)
            baseline = distribution_metrics(clear_values)
            result.append({
                "pair_id": pair_id,
                "target_model": model_id,
                "condition_id": condition_id,
                "clear_baseline_id": clear_id,
                "mps_mode": mode,
                "metric": field,
                "adverse_median": adverse["median_ms"],
                "clear_median": baseline["median_ms"],
                "median_change": (
                    adverse["median_ms"] - baseline["median_ms"]),
                "adverse_p0_p99_range": adverse["p0_p99_range_ms"],
                "clear_p0_p99_range": baseline["p0_p99_range_ms"],
                "adverse_width": adverse["p0_p99_range_ms"],
                "clear_width": baseline["p0_p99_range_ms"],
                **variation_metrics(adverse["p0_p99_range_ms"],
                                    baseline["p0_p99_range_ms"]),
            })
    return result


def module_comparison_rows(rows, baseline_map):
    """Compare module-duration distributions within the same pair and mode."""
    grouped = {}
    for row in rows:
        key = (row["pair_id"], row["model_id"], row["module"],
               row["condition_id"], row["mps_mode"])
        grouped.setdefault(key, []).append(row["duration_ms"])
    result = []
    for (pair_id, model_id, module, condition_id, mode), values in sorted(
            grouped.items()):
        clear_id = baseline_map.get(condition_id)
        clear = grouped.get((pair_id, model_id, module, clear_id, mode))
        if clear is None:
            continue
        adverse = distribution_metrics(values)
        baseline = distribution_metrics(clear)
        result.append({
            "pair_id": pair_id,
            "target_model": model_id,
            "module": module,
            "condition_id": condition_id,
            "clear_baseline_id": clear_id,
            "mps_mode": mode,
            "adverse_median_ms": adverse["median_ms"],
            "clear_median_ms": baseline["median_ms"],
            "median_change_ms": (
                adverse["median_ms"] - baseline["median_ms"]),
            "adverse_p0_p99_range_ms": adverse["p0_p99_range_ms"],
            "clear_p0_p99_range_ms": baseline["p0_p99_range_ms"],
            "adverse_width_ms": adverse["p0_p99_range_ms"],
            "clear_width_ms": baseline["p0_p99_range_ms"],
            **variation_metrics(adverse["p0_p99_range_ms"],
                                baseline["p0_p99_range_ms"]),
        })
    return result


def mps_comparison_rows(rows):
    """Place each off/on adverse-versus-clear observation side by side."""
    indexed = {
        (row["pair_id"], row["target_model"], row["condition_id"],
         row["mps_mode"]): row
        for row in rows
    }
    result = []
    for key in sorted({key[:3] for key in indexed}):
        off = indexed.get((*key, "off"))
        on = indexed.get((*key, "on"))
        if off is None or on is None:
            continue
        result.append({
            "pair_id": key[0],
            "target_model": key[1],
            "condition_id": key[2],
            "mps_off_label": off["observed_label"],
            "mps_on_label": on["observed_label"],
            "mps_off_variation_factor": off["variation_factor"],
            "mps_on_variation_factor": on["variation_factor"],
            "factor_change_on_minus_off": (
                on["variation_factor"] - off["variation_factor"]
                if off["variation_factor"] is not None
                and on["variation_factor"] is not None else None),
        })
    return result


def controlled_response_rows(input_changes, mechanisms, modules, study):
    """Join controlled input changes to isolated computational responses."""
    model_modalities = {
        model_id: model["modality"]
        for model_id, model in study["data"]["models"].items()
    }
    responses = [{
        **row,
        "response_metric": row["metric"],
        "response_module": "",
    } for row in mechanisms]
    responses.extend({
        **row,
        "response_metric": "module_duration_ms",
        "response_module": row["module"],
        "adverse_median": row["adverse_median_ms"],
        "clear_median": row["clear_median_ms"],
        "median_change": row["median_change_ms"],
        "adverse_width": row["adverse_width_ms"],
        "clear_width": row["clear_width_ms"],
    } for row in modules)
    inputs = {}
    for row in input_changes:
        inputs.setdefault((row["condition_id"], row["modality"]), []).append(
            row
        )
    result = []
    for response in responses:
        model_id = response["target_model"]
        for input_row in inputs.get(
            (response["condition_id"], model_modalities[model_id]), []
        ):
            result.append({
                "target_model": model_id,
                "architecture_profile": study["data"]["models"][model_id][
                    "architecture_profile"
                ],
                "condition_id": response["condition_id"],
                "clear_baseline_id": response["clear_baseline_id"],
                "input_modality": input_row["modality"],
                "input_metric": input_row["metric"],
                "input_adverse_median": input_row["adverse_median"],
                "input_clear_median": input_row["clear_median"],
                "input_median_paired_change": input_row[
                    "median_paired_change"
                ],
                "input_median_ratio": input_row["median_ratio"],
                "response_metric": response["response_metric"],
                "response_module": response["response_module"],
                "response_adverse_median": response["adverse_median"],
                "response_clear_median": response["clear_median"],
                "response_median_change": response["median_change"],
                "response_adverse_width": response["adverse_width"],
                "response_clear_width": response["clear_width"],
                "response_variation_factor": response["variation_factor"],
                "response_observed_label": response["observed_label"],
                "analysis_scope": SCOPE,
            })
    return result


def context_comparison_rows(pair_rows, isolated_rows):
    """Place intrinsic isolated and co-run pair responses side by side."""
    isolated = {
        (row["target_model"], row["condition_id"], row["metric"]): row
        for row in isolated_rows if row["mps_mode"] == "off"
    }
    result = []
    for row in pair_rows:
        intrinsic = isolated.get(
            (row["target_model"], row["condition_id"], row["metric"])
        )
        if intrinsic is None:
            continue
        models = row["pair_id"].split("+")
        co_runner = next(
            (model for model in models if model != row["target_model"]), ""
        )
        result.append({
            "pair_id": row["pair_id"],
            "target_model": row["target_model"],
            "co_runner": co_runner,
            "condition_id": row["condition_id"],
            "clear_baseline_id": row["clear_baseline_id"],
            "mps_mode": row["mps_mode"],
            "metric": row["metric"],
            "isolated_median_change": intrinsic["median_change"],
            "isolated_variation_factor": intrinsic["variation_factor"],
            "isolated_observed_label": intrinsic["observed_label"],
            "pair_median_change": row["median_change"],
            "pair_variation_factor": row["variation_factor"],
            "pair_observed_label": row["observed_label"],
            "analysis_scope": SCOPE,
        })
    return result


def _plot_ecdfs(rows, output_root, mode):
    selected = [row for row in rows if row["mps_mode"] == mode]
    pairs = sorted({row["pair_id"] for row in selected})
    for pair_id in pairs:
        pair_rows = [row for row in selected if row["pair_id"] == pair_id]
        targets = sorted({row["target_model"] for row in pair_rows})
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
        for axis, target in zip(axes[0], targets):
            target_rows = [
                row for row in pair_rows if row["target_model"] == target
            ]
            for condition_id in sorted(
                {row["condition_id"]
                 for row in target_rows}):
                values = [
                    row for row in target_rows
                    if row["condition_id"] == condition_id
                ]
                axis.step(
                    [row["latency_ms"] for row in values],
                    [row["ecdf"] for row in values],
                    where="post",
                    label=condition_id,
                )
            axis.set(title=target,
                     xlabel="inference latency (ms)",
                     ylabel="ECDF")
            axis.grid(alpha=0.25)
        axes[0][-1].legend(fontsize=6,
                           bbox_to_anchor=(1.04, 1),
                           loc="upper left")
        figure.suptitle(f"{pair_id} | MPS {mode} | {SCOPE}")
        figure.tight_layout()
        path = output_root / "plots" / f"mps_{mode}" / f"ecdf-{pair_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)


def _plot_heatmap(summary, study, output_root, mode):
    rows = [row for row in summary if row["mps_mode"] == mode]
    if not rows:
        return
    conditions = list(study["data"]["conditions"])
    targets = sorted({(row["target_model"], row["co_runner"]) for row in rows})
    indexed = {
        (row["target_model"], row["co_runner"], row["condition_id"]):
        row["p0_p99_range_ms"]
        for row in rows
    }
    values = np.asarray([[
        indexed.get((target, co_runner, condition), np.nan)
        for condition in conditions
    ] for target, co_runner in targets])
    figure, axis = plt.subplots(figsize=(18, 9))
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(conditions)),
                    conditions,
                    rotation=60,
                    ha="right")
    axis.set_yticks(
        range(len(targets)),
        [f"{target} | {co_runner}" for target, co_runner in targets])
    axis.set_title(f"minimum-P99 latency range (ms) | MPS {mode} | {SCOPE}")
    figure.colorbar(image, ax=axis, label="P99 - minimum (ms)")
    figure.tight_layout()
    path = output_root / "plots" / f"mps_{mode}" / "width-heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _load_violin_reference(specification):
    """Load one relabeled condition from a prior frame-metrics CSV."""
    try:
        label, selection = specification.split("=", 1)
        condition_id, source = selection.split(":", 1)
    except ValueError as exc:
        raise StudyError(
            "violin reference must be LABEL=CONDITION:FRAME_METRICS_CSV"
        ) from exc
    with Path(source).open(newline="", encoding="utf-8") as stream:
        return [{
            **row,
            "condition_id": label,
            "inference_e2e_ms": float(row["inference_e2e_ms"]),
            "run_valid": row["run_valid"] == "True",
        } for row in csv.DictReader(stream)
                if row["condition_id"] == condition_id]


def _plot_violins(frames, study, output_root, mode, reference_frames=()):
    """Plot paired P1-P99 latency distributions for every model pair."""
    selected = [row for row in frames
                if row["mps_mode"] == mode and row["run_valid"]]
    pairs = {row["pair_id"] for row in selected}
    references = [row for row in reference_frames
                  if row["mps_mode"] == mode and row["run_valid"]
                  and row["pair_id"] in pairs]
    selected = references + selected
    conditions = list(dict.fromkeys(
        [row["condition_id"] for row in references]
        + list(study["data"]["conditions"])
    ))
    directory = "mps" if mode == "on" else "non_mps"
    colors = {"lidar": "#1f77b4", "image": "#ff7f0e"}
    for pair_id in sorted(pairs):
        pair_rows = [row for row in selected if row["pair_id"] == pair_id]
        records = {"Scene": [], "Inference time (ms)": [], "Modality": []}
        for condition_id in conditions:
            for modality, model_field in (
                    ("lidar", "lidar_model"), ("image", "camera_model")):
                values = np.asarray([
                    row["inference_e2e_ms"] for row in pair_rows
                    if row["condition_id"] == condition_id
                    and row["model_id"] == row[model_field]
                ], dtype=float)
                if not len(values):
                    raise StudyError(
                        f"missing {mode} {pair_id} {condition_id} {modality} frames"
                    )
                p1, p99 = np.percentile(values, (1, 99))
                trimmed = values[(values >= p1) & (values <= p99)]
                records["Scene"].extend([condition_id] * len(trimmed))
                records["Inference time (ms)"].extend(trimmed.tolist())
                records["Modality"].extend([modality] * len(trimmed))
        figure, axis = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=records,
            x="Scene",
            y="Inference time (ms)",
            hue="Modality",
            order=conditions,
            hue_order=["lidar", "image"],
            palette=colors,
            cut=0,
            inner="quart",
            density_norm="width",
            common_norm=False,
            linewidth=1.1,
            legend=False,
            ax=axis,
        )
        axis.set_title(
            f"{pair_id} | {'MPS' if mode == 'on' else 'Non-MPS'} | P1-P99"
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_root / "violin_plots" / directory / f"{pair_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)


def _fmt(value):
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.3f}"


def _report(study, output_root, manifest, input_summary, input_changes,
            isolated_comparisons, isolated_mechanisms, isolated_modules,
            comparisons, mechanism_comparisons, contexts, mps_comparisons,
            validation, isolated_validation):
    """Render the complete descriptive, evidence-bounded Goal 1 report."""
    conditions = study["data"]["conditions"]
    isolated_included = manifest["isolated_planned_run_count"] > 0
    modes = ["on" if value else "off"
             for value in study["data"]["mps_modes"]]
    isolated_completion = (
        f"Isolated evidence: **{manifest['isolated_valid_run_count']}/"
        f"{manifest['isolated_planned_run_count']} valid**, comprising "
        f"{manifest['isolated_reused_section4_1_run_count']} reused runs and "
        f"{manifest['isolated_goal1_valid_run_count']} newly executed 3DSSD "
        "runs."
        if isolated_included else
        "Isolated controls were outside this campaign and are not analyzed."
    )
    lines = [
        f"# {study['data']['study_id']} descriptive analysis", "",
        "## Scope and completion", "",
        f"Scope: **{SCOPE}**. One execution was used per condition. Frames "
        "within an execution are repeated observations, not independent "
        "experimental repetitions. No accuracy or confidence claim is made.",
        "",
        f"Pair campaign: **{manifest['valid_run_count']}/"
        f"{manifest['planned_run_count']} valid**; missing or invalid: "
        f"**{manifest['invalid_or_missing_run_count']}**. "
        f"{isolated_completion}", "",
        f"The pair matrix crosses {len(study['lidar'])} LiDAR models with "
        f"{len(study['camera'])} camera models, "
        f"{len(conditions)} same-condition inputs, and MPS modes "
        f"{', '.join(modes)}: {manifest['planned_run_count']} cells. Both "
        "tenants consume the same clear or adverse bag; labels below compare "
        "each adverse cell only with its configured clear baseline.", "",
        "## Actual input changes", "",
        "All values below come from decoded payloads in the immutable MCAP "
        "bags. Physical-rain conditions are deterministic corruptions of the "
        "listed full-duration clear source and retain source timestamps.", "",
        "### Clear-condition input context", "",
        "| condition | image luminance mean | image horizontal gradient | "
        "LiDAR points | LiDAR mean range (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    summaries = {
        (row["condition_id"], row["modality"], row["metric"]): row
        for row in input_summary
    }
    for condition_id, condition in conditions.items():
        if condition["corruption_type"] != "clean":
            continue
        lines.append(
            f"| {condition_id} | "
            f"{_fmt(summaries[(condition_id, 'image', 'luminance_mean')]['median_ms'])} | "
            f"{_fmt(summaries[(condition_id, 'image', 'horizontal_gradient_mean')]['median_ms'])} | "
            f"{_fmt(summaries[(condition_id, 'lidar', 'point_count')]['median_ms'])} | "
            f"{_fmt(summaries[(condition_id, 'lidar', 'range_mean_m')]['median_ms'])} |"
        )
    lines += [
        "", "### Paired adverse-minus-clear payload changes", "",
        "`median paired change` is computed at identical source timestamps. "
        "Ratios use adverse median divided by matching clear median.", "",
        "| condition | modality | metric | paired n | clear median | adverse "
        "median | median paired change | ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    selected_inputs = {
        "image": {"jpeg_bytes", "luminance_mean", "luminance_std",
                  "horizontal_gradient_mean", "vertical_gradient_mean"},
        "lidar": {"point_count", "range_mean_m", "points_0_5m",
                  "points_5_20m", "points_20_40m", "points_40m_plus",
                  "intensity_mean"},
    }
    for row in input_changes:
        if row["metric"] not in selected_inputs[row["modality"]]:
            continue
        lines.append(
            f"| {row['condition_id']} | {row['modality']} | "
            f"{row['metric']} | {row['sample_count']} | "
            f"{_fmt(row['clear_median'])} | {_fmt(row['adverse_median'])} | "
            f"{_fmt(row['median_paired_change'])} | "
            f"{_fmt(row['median_ratio'])} |"
        )

    lines += [
        "", "## Isolated-model computational response", "",
        ("Isolated MPS-disabled results are used only to describe intrinsic "
        "input sensitivity. `observed amplification` and `observed damping` "
        "mean that the adverse minimum-to-P99 latency range is respectively "
        "larger or smaller than the matching clear range; they do not imply a "
        "statistical or causal finding."
         if isolated_included else
         "Not produced: isolated controls were outside the v4 campaign."), "",
        "| model | adverse condition | median latency change (ms) | clear "
        "minimum-P99 range (ms) | adverse minimum-P99 range (ms) | factor | label |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    isolated_latency = {
        (row["target_model"], row["condition_id"]): row
        for row in isolated_mechanisms
        if row["metric"] == "inference_e2e_ms"
    }
    for row in isolated_comparisons:
        latency = isolated_latency[(row["target_model"],
                                     row["condition_id"])]
        lines.append(
            f"| {row['target_model']} | {row['condition_id']} | "
            f"{_fmt(latency['median_change'])} | "
            f"{_fmt(row['clear_width_ms'])} | {_fmt(row['adverse_width_ms'])} | "
            f"{_fmt(row['variation_factor'])} | {row['observed_label']} |"
        )
    lines += [
        "", "The latency table above stores width comparisons; exact median "
        "changes and compute-path measurements are in "
        "`isolated_mechanism_comparisons.csv`.", "",
        "### Isolated compute-path summary across adverse conditions", "",
        "| model | metric | observations | median adverse-clear median "
        "change | amplified | damped | unchanged |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    core_metrics = {
        "inference_e2e_ms", "gpu_kernel_active_ms", "kernel_count",
        "kernel_span_ms", "waiting_ms", "input_point_count",
        "occupied_voxel_count", "sampled_point_count", "proposal_count",
    }
    grouped = defaultdict(list)
    for row in isolated_mechanisms:
        if row["metric"] in core_metrics:
            grouped[(row["target_model"], row["metric"])].append(row)
    for (model_id, metric), rows in sorted(grouped.items()):
        labels = Counter(row["observed_label"] for row in rows)
        lines.append(
            f"| {model_id} | {metric} | {len(rows)} | "
            f"{_fmt(np.median([row['median_change'] for row in rows]))} | "
            f"{labels['observed_amplification']} | "
            f"{labels['observed_damping']} | "
            f"{labels['observed_unchanged']} |"
        )
    lines += [
        "", "### Module execution summary", "",
        "| model | module | adverse comparisons | median duration change "
        "(ms) | amplified widths | damped widths | unchanged widths |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in isolated_modules:
        grouped[(row["target_model"], row["module"])].append(row)
    for (model_id, module), rows in sorted(grouped.items()):
        labels = Counter(row["observed_label"] for row in rows)
        lines.append(
            f"| {model_id} | {module} | {len(rows)} | "
            f"{_fmt(np.median([row['median_change_ms'] for row in rows]))} | "
            f"{labels['observed_amplification']} | "
            f"{labels['observed_damping']} | "
            f"{labels['observed_unchanged']} |"
        )

    input_index = {
        (row["condition_id"], row["modality"], row["metric"]): row
        for row in input_changes
    }
    mechanism_index = {
        (row["target_model"], row["condition_id"], row["metric"]): row
        for row in isolated_mechanisms
    }
    dynamic_metric = {
        "3dssd": "sampled_point_count",
        "centerpoint": "occupied_voxel_count",
        "pointpillars": "occupied_voxel_count",
        "faster-rcnn": "proposal_count",
    }
    lines += [
        "", "## Controlled synthetic-input validation", "",
        "This table holds scene, model, weights, MPS-off isolation, and source "
        "timestamp alignment fixed while the deterministic weather payload "
        "changes. It validates whether computation changed alongside the "
        "measured input; it does not establish that a single measured input "
        "feature caused the computation change.", "",
        "| model | condition | input metric | input ratio | latency median "
        "change (ms) | latency width factor | GPU-active median change (ms) | "
        "kernel-count median change | dynamic metric change |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for latency in isolated_mechanisms:
        if latency["metric"] != "inference_e2e_ms":
            continue
        model_id = latency["target_model"]
        modality = study["data"]["models"][model_id]["modality"]
        input_metric = "point_count" if modality == "lidar" else "luminance_mean"
        input_row = input_index[(latency["condition_id"], modality,
                                 input_metric)]
        gpu = mechanism_index[(model_id, latency["condition_id"],
                               "gpu_kernel_active_ms")]
        kernels = mechanism_index[(model_id, latency["condition_id"],
                                   "kernel_count")]
        dynamic = dynamic_metric.get(model_id)
        dynamic_row = mechanism_index.get((model_id,
                                            latency["condition_id"], dynamic))
        dynamic_text = (
            f"{dynamic}: {_fmt(dynamic_row['median_change'])}"
            if dynamic_row else "fixed/no captured dynamic count"
        )
        lines.append(
            f"| {model_id} | {latency['condition_id']} | {input_metric} | "
            f"{_fmt(input_row['median_ratio'])} | "
            f"{_fmt(latency['median_change'])} | "
            f"{_fmt(latency['variation_factor'])} | "
            f"{_fmt(gpu['median_change'])} | "
            f"{_fmt(kernels['median_change'])} | {dynamic_text} |"
        )

    lines += [
        "", "## Same-condition pair amplification and damping", "",
        "Every label in this section compares an adverse pair only with its "
        "matching clear pair under the same MPS mode. Isolated widths are not "
        "used as denominators.", "",
        "| target | MPS | adverse pair observations | amplified | damped | "
        "unchanged | median width factor | minimum factor | maximum factor |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[(row["target_model"], row["mps_mode"])].append(row)
    for (model_id, mode), rows in sorted(grouped.items()):
        labels = Counter(row["observed_label"] for row in rows)
        factors = [row["variation_factor"] for row in rows
                   if row["variation_factor"] is not None]
        lines.append(
            f"| {model_id} | {mode} | {len(rows)} | "
            f"{labels['observed_amplification']} | "
            f"{labels['observed_damping']} | "
            f"{labels['observed_unchanged']} | {_fmt(np.median(factors))} | "
            f"{_fmt(min(factors))} | {_fmt(max(factors))} |"
        )
    lines += [
        "", "### Co-runner-specific latency-width response", "",
        "| pair | target | MPS | conditions | amplified | damped | unchanged "
        "| median factor |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[(row["pair_id"], row["target_model"],
                 row["mps_mode"])].append(row)
    for (pair_id, model_id, mode), rows in sorted(grouped.items()):
        labels = Counter(row["observed_label"] for row in rows)
        factors = [row["variation_factor"] for row in rows
                   if row["variation_factor"] is not None]
        lines.append(
            f"| {pair_id} | {model_id} | {mode} | {len(rows)} | "
            f"{labels['observed_amplification']} | "
            f"{labels['observed_damping']} | "
            f"{labels['observed_unchanged']} | {_fmt(np.median(factors))} |"
        )

    lines += [
        "", "## Compute, kernel, waiting, co-runner, and MPS mechanisms", "",
        "`gpu_kernel_active_ms` measures summed kernel service, `kernel_span_ms` "
        "measures first-to-last kernel extent, `waiting_ms` is the captured "
        "memcpy-adjacent non-kernel interval, and `kernel_count` records launch "
        "structure. Their joint movement is descriptive evidence: active-time "
        "or count changes are consistent with input-driven work changes; span "
        "and waiting changes without matching active-time changes are "
        "consistent with scheduling/interference. These measurements do not "
        "identify a specific GPU resource.", "",
        "| target | MPS | metric | observations | median adverse-clear median "
        "change | amplified widths | damped widths | unchanged widths |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in mechanism_comparisons:
        if row["metric"] in core_metrics:
            grouped[(row["target_model"], row["mps_mode"],
                     row["metric"])].append(row)
    for (model_id, mode, metric), rows in sorted(grouped.items()):
        labels = Counter(row["observed_label"] for row in rows)
        lines.append(
            f"| {model_id} | {mode} | {metric} | {len(rows)} | "
            f"{_fmt(np.median([row['median_change'] for row in rows]))} | "
            f"{labels['observed_amplification']} | "
            f"{labels['observed_damping']} | "
            f"{labels['observed_unchanged']} |"
        )
    lines += [
        "", "### Intrinsic-versus-pair label agreement", "",
        "| pair | target | MPS | latency comparisons | same isolated/pair "
        "label | different label |",
        "|---|---|---|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in contexts:
        if row["metric"] == "inference_e2e_ms":
            grouped[(row["pair_id"], row["target_model"],
                     row["mps_mode"])].append(row)
    for (pair_id, model_id, mode), rows in sorted(grouped.items()):
        same = sum(row["isolated_observed_label"]
                   == row["pair_observed_label"] for row in rows)
        lines.append(
            f"| {pair_id} | {model_id} | {mode} | {len(rows)} | "
            f"{same} | {len(rows) - same} |"
        )
    lines += [
        "", "### MPS label transitions", "",
        "MPS rows place the already valid within-mode off/on comparisons side "
        "by side; they do not compare raw adverse latency across modes.", "",
        "| target | off label → on label | observations |",
        "|---|---|---:|",
    ]
    transitions = Counter(
        (row["target_model"], row["mps_off_label"], row["mps_on_label"])
        for row in mps_comparisons
    )
    for (model_id, off, on), count in sorted(transitions.items()):
        lines.append(f"| {model_id} | {off} → {on} | {count} |")

    lines += [
        "", "## Architecture-family interpretation boundary", "",
        "- 3DSSD is point-based: raw point count/range changes can reach "
        "preprocessing and early point operations; `sampled_point_count` shows "
        "whether its captured final sampling cardinality changed.",
        "- CenterPoint is voxel-based: `occupied_voxel_count` separates raw "
        "point loss/addition from the realized sparse voxel workload.",
        "- PointPillars is voxel/pillar-based: `occupied_voxel_count` and its "
        "voxel encoder, middle encoder, backbone, neck, and head durations "
        "describe the realized pillar workload.",
        "- DeepLabV3+ and ViT-UPerNet are dense segmentation paths with fixed "
        "configured resize shapes; input sensitivity is assessed through "
        "preprocessing, module durations, kernel activity/count/span, and "
        "waiting rather than a proposal count.",
        "- DETR uses a transformer detector path with fixed configured input "
        "processing and query structure; the same fixed/dynamic measurements "
        "bound any observed response.",
        "- DINO uses the same measured transformer stages: backbone, neck, "
        "positional encoding, encoder, decoder, and bbox head. Warm-up must "
        "observe every required binding before a run is accepted.",
        "- Faster R-CNN is proposal-based: `proposal_count`, RPN, and ROI module "
        "durations expose its captured dynamic post-backbone path.",
        "- YOLOv3 is single-stage: no proposal-stage count is expected, so "
        "preprocessing and dense backbone/head GPU measurements are used.",
        "", "These are architecture-aware interpretations of measured "
        "execution structure, not accuracy claims or proof that weather alone "
        "caused any co-run latency change.", "",
        "## Completeness, exclusions, failures, and limitations", "",
    ]
    invalid = [row for row in validation if not row["valid"]]
    invalid_isolated = [row for row in isolated_validation if not row["valid"]]
    failed_attempts = []
    failure_roots = (
        study["paths"]["output_root"] / "failed_attempts",
        study["paths"]["output_root"].with_name(
            study["paths"]["output_root"].name + "_failed_attempts"
        ),
    )
    for failure_root in failure_roots:
        for manifest_path in sorted(failure_root.glob("*/run_manifest.json")):
            attempt = _load_json(manifest_path)
            result_path = manifest_path.parent / "testbed_result.json"
            result = _load_json(result_path) if result_path.is_file() else {}
            failed_attempts.append((manifest_path.parent, attempt, result))
    if invalid or invalid_isolated:
        lines.append("Invalid or missing runs:")
        lines.append("")
        for row in invalid + invalid_isolated:
            lines.append(f"- `{row['run_id']}`: {row['errors']}")
    else:
        lines.append("No planned run was excluded or invalid.")
    if failed_attempts:
        lines += ["", "Excluded execution attempts:", ""]
        for path, attempt, result in failed_attempts:
            lines.append(
                f"- `{path}`: {result.get('error', attempt.get('error'))}; "
                f"bags started {result.get('bags_started', 'n/a')}, "
                f"bags completed {result.get('bags_completed', 'n/a')}, "
                f"models acknowledged "
                f"{len(result.get('acknowledged_models', []))}. The "
                "zero-sample attempt was excluded and the required cell's "
                "next attempt succeeded."
            )
    lines += [
        "", "Limitations:", "",
        "- One execution per cell supports descriptive comparisons only; "
        "within-run frames do not provide run-to-run uncertainty.",
        "- Rain and snow are deterministic synthetic corruptions, not a claim "
        "about every natural weather process.",
        "- The study measures computation and latency, not prediction accuracy.",
        ("- Pair changes combine both tenants' same-condition input responses "
         "with GPU co-run effects; isolated evidence helps separate them but "
         "does not prove a unique causal mechanism."
         if isolated_included else
         "- Pair changes combine both tenants' input responses and co-run "
         "effects; without isolated controls they cannot be separated."),
        "- Nsight instrumentation perturbs timing; all matrix cells use the "
        "same profiling path, and conclusions remain descriptive.",
        "- No specific shared GPU resource is named without direct evidence.",
        "", "## Artifacts and regeneration", "",
        f"- Study contract: `{study['source']}` (SHA-256 "
        f"`{study['sha256']}`).",
        f"- Pair and new 3DSSD raw runs: `{study['paths']['output_root']}/<run_id>`.",
        f"- Preserved failed attempts: `{failure_root}/<run-id>-attempt-N`.",
        "- Isolated raw runs are listed in `isolated_validation.csv` when "
        "included.",
        f"- Analysis tables, plots, manifest, and this report: `{output_root}`.",
        "- Each Goal 1 raw directory retains `config.yaml`, run/model/status "
        "evidence, input JSONL, clock evidence, `profile.nsys-rep.gz`, "
        "`profile.sqlite.gz`, and `profile_archive.json`. Compression is "
        "checksum-verified and lossless.", "",
        "Run from `/mmdetection3d_ros2` inside `pPerf-host`:", "",
        "```bash",
        "source /opt/ros/humble/setup.bash",
        "source closeloop_perf/install/setup.bash",
        "ros2 run closeloop_analyzer analyze input-data RUN_DIRECTORY "
        f"--output-root {output_root}",
        "```", "",
        "Restore one archived trace without replacing the archive:", "",
        "```bash",
        "gzip -dk RUN/profile.sqlite.gz",
        "gzip -dk RUN/profile.nsys-rep.gz",
        "```", "",
        "All per-frame, per-module, per-structure, controlled-response, pair "
        "context, ECDF, and comparison rows are retained in the CSV artifacts; "
        "the tables above summarize rather than silently exclude them.", "",
    ]
    (output_root / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def analyze(study, output_root=None, pairs=(), mps_modes=(),
            include_section4_1_pairs=False, include_isolated=True,
            violin_references=()):
    """Write all descriptive tables, ECDFs, heatmaps, and mechanism joins."""
    output_root = Path(output_root
                       or study["paths"]["output_root"] / "analysis")
    output_root.mkdir(parents=True, exist_ok=True)
    validation, frames, modules, structures = collect(
        study, pairs, mps_modes)
    reference_frames = sum(
        (_load_violin_reference(item) for item in violin_references), []
    )
    if include_section4_1_pairs:
        reference = collect_reference_pairs(study)
        validation += reference[0]
        frames += reference[1]
        modules += reference[2]
        structures += reference[3]
    isolated = (
        collect_isolated(study) if include_isolated else ([], [], [], [])
    )
    (isolated_validation, isolated_frames, isolated_modules,
     isolated_structures) = isolated
    inputs = characterize_inputs(study)
    input_summary = input_summary_rows(inputs)
    input_changes = input_change_rows(
        inputs, study["data"]["baseline_map"]
    )
    summary = summarize(frames)
    comparisons = comparison_rows(summary, study["data"]["baseline_map"])
    predictability_tables = {
        mode: predictability_table_rows(
            summary, study["data"]["conditions"],
            study["data"]["baseline_map"], mode,
        )
        for mode in ("on" if value else "off"
                     for value in study["data"]["mps_modes"])
    }
    ecdfs = ecdf_rows(frames)
    mechanisms = mechanism_rows(frames, structures)
    mechanism_comparisons = mechanism_comparison_rows(
        mechanisms, study["data"]["baseline_map"])
    module_comparisons = module_comparison_rows(
        modules, study["data"]["baseline_map"])
    isolated_summary = summarize(isolated_frames)
    isolated_mechanisms = mechanism_rows(
        isolated_frames, isolated_structures)
    isolated_comparisons = comparison_rows(
        isolated_summary, study["data"]["baseline_map"]
    )
    isolated_mechanism_comparisons = mechanism_comparison_rows(
        isolated_mechanisms, study["data"]["baseline_map"]
    )
    isolated_module_comparisons = module_comparison_rows(
        isolated_modules, study["data"]["baseline_map"]
    )
    controlled = controlled_response_rows(
        input_changes, isolated_mechanism_comparisons,
        isolated_module_comparisons, study,
    )
    contexts = context_comparison_rows(
        mechanism_comparisons, isolated_mechanism_comparisons
    )
    mps_comparisons = mps_comparison_rows(comparisons)
    isolated_3dssd_validation = [
        row for row in isolated_validation if row["model_id"] == "3dssd"
    ]
    isolated_3dssd_frames = [
        row for row in isolated_frames if row["model_id"] == "3dssd"
    ]
    isolated_3dssd_modules = [
        row for row in isolated_modules if row["model_id"] == "3dssd"
    ]
    isolated_3dssd_structures = [
        row for row in isolated_structures if row["model_id"] == "3dssd"
    ]
    _write_csv(output_root / "validation.csv", validation)
    _write_csv(output_root / "isolated_validation.csv", isolated_validation)
    _write_csv(output_root / "isolated_3dssd_validation.csv",
               isolated_3dssd_validation)
    _write_csv(output_root / "input_metrics.csv", inputs)
    _write_csv(output_root / "input_summary.csv", input_summary)
    _write_csv(
        output_root / "input_changes.csv",
        input_changes,
    )
    _write_csv(output_root / "frame_metrics.csv", frames)
    _write_csv(output_root / "module_metrics.csv", modules)
    _write_csv(output_root / "structure_metrics.csv", structures)
    _write_csv(output_root / "mechanism_metrics.csv", mechanisms)
    _write_csv(output_root / "isolated_frame_metrics.csv", isolated_frames)
    _write_csv(output_root / "isolated_module_metrics.csv", isolated_modules)
    _write_csv(output_root / "isolated_structure_metrics.csv",
               isolated_structures)
    _write_csv(output_root / "isolated_latency_summary.csv",
               isolated_summary)
    _write_csv(output_root / "isolated_latency_comparisons.csv",
               isolated_comparisons)
    _write_csv(output_root / "isolated_mechanism_metrics.csv",
               isolated_mechanisms)
    _write_csv(output_root / "isolated_mechanism_comparisons.csv",
               isolated_mechanism_comparisons)
    _write_csv(output_root / "isolated_module_comparisons.csv",
               isolated_module_comparisons)
    _write_csv(output_root / "controlled_input_response.csv", controlled)
    _write_csv(output_root / "pair_context_response.csv", contexts)
    _write_csv(output_root / "isolated_3dssd_frame_metrics.csv",
               isolated_3dssd_frames)
    _write_csv(output_root / "isolated_3dssd_module_metrics.csv",
               isolated_3dssd_modules)
    _write_csv(output_root / "isolated_3dssd_structure_metrics.csv",
               isolated_3dssd_structures)
    _write_csv(output_root / "isolated_3dssd_summary.csv",
               summarize(isolated_3dssd_frames))
    _write_csv(output_root / "isolated_3dssd_mechanism_metrics.csv",
               mechanism_rows(isolated_3dssd_frames,
                              isolated_3dssd_structures))
    for mode in ("on" if value else "off"
                 for value in study["data"]["mps_modes"]):
        _write_csv(
            output_root / f"predictability_min_to_p99_mps_{mode}.csv",
            predictability_tables[mode],
        )
        _write_csv(
            output_root / f"latency_summary_mps_{mode}.csv",
            [row for row in summary if row["mps_mode"] == mode],
        )
        _write_csv(
            output_root / f"weather_comparisons_mps_{mode}.csv",
            [row for row in comparisons if row["mps_mode"] == mode],
        )
        _write_csv(
            output_root / f"mechanism_comparisons_mps_{mode}.csv",
            [row for row in mechanism_comparisons
             if row["mps_mode"] == mode],
        )
        _write_csv(
            output_root / f"module_comparisons_mps_{mode}.csv",
            [row for row in module_comparisons
             if row["mps_mode"] == mode],
        )
        _write_csv(
            output_root / f"ecdf_mps_{mode}.csv",
            [row for row in ecdfs if row["mps_mode"] == mode],
        )
        _plot_ecdfs(ecdfs, output_root, mode)
        _plot_heatmap(summary, study, output_root, mode)
        _plot_violins(frames, study, output_root, mode, reference_frames)
    _write_csv(output_root / "mps_comparisons.csv", mps_comparisons)
    manifest = {
        "analysis_scope": SCOPE,
        "study_id": study["data"]["study_id"],
        "study_sha256": study["sha256"],
        "violin_references": list(violin_references),
        "isolated_controls_included": include_isolated,
        "planned_run_count": len(validation),
        "valid_run_count": sum(row["valid"] for row in validation),
        "extractable_run_count": sum(
            row.get("extractable", row["valid"]) for row in validation),
        "invalid_or_missing_run_count":
        sum(not row["valid"] for row in validation),
        "latency_summary_row_count": len(summary),
        "comparison_row_count": len(comparisons),
        "predictability_table_row_counts": {
            mode: len(rows) for mode, rows in predictability_tables.items()
        },
        "isolated_planned_run_count": len(isolated_validation),
        "isolated_valid_run_count": sum(
            row["valid"] for row in isolated_validation),
        "isolated_extractable_run_count": sum(
            row.get("extractable", row["valid"])
            for row in isolated_validation),
        "isolated_reused_section4_1_run_count": sum(
            row["source"] == "reused_section4_1"
            for row in isolated_validation),
        "isolated_goal1_planned_run_count": len(isolated_3dssd_validation),
        "isolated_goal1_valid_run_count": sum(
            row["valid"] for row in isolated_3dssd_validation),
        "isolated_3dssd_planned_run_count": len(
            isolated_3dssd_validation),
        "isolated_3dssd_valid_run_count": sum(
            row["valid"] for row in isolated_3dssd_validation),
        "input_record_count": len(inputs),
        "mechanism_comparison_row_count": len(mechanism_comparisons),
        "module_comparison_row_count": len(module_comparisons),
        "controlled_input_response_row_count": len(controlled),
        "pair_context_response_row_count": len(contexts),
    }
    manifest["complete"] = (
        manifest["extractable_run_count"] == manifest["planned_run_count"]
        and manifest["isolated_extractable_run_count"]
        == manifest["isolated_planned_run_count"]
    )
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _report(
        study, output_root, manifest, input_summary, input_changes,
        isolated_comparisons, isolated_mechanism_comparisons,
        isolated_module_comparisons, comparisons, mechanism_comparisons,
        contexts, mps_comparisons, validation, isolated_validation,
    )
    return manifest


def main(argv=None):
    """Run descriptive analysis for one versioned campaign."""
    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("--output-root")
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--mps-mode", action="append",
                        choices=("off", "on"), default=[])
    parser.add_argument("--include-section4-1-pairs", action="store_true")
    parser.add_argument("--pair-only", action="store_true")
    parser.add_argument(
        "--violin-reference", action="append", default=[],
        metavar="LABEL=CONDITION:FRAME_METRICS_CSV",
    )
    args = parser.parse_args(argv)
    result = analyze(load_study(args.study), args.output_root,
                     args.pair, args.mps_mode,
                     args.include_section4_1_pairs,
                     not args.pair_only,
                     args.violin_reference)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
