"""Plan, run, validate, and analyze reproducible input-variation studies."""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics

import yaml

from .config import ConfigError, load_run_config, schema_v2_config
from .corrupt_rosbag import (
    LIDAR_TOPIC, _header_timestamp_ns, _point_dtype, transform_bag,
    validate_manifest,
)
from .full_bag_green import extract_frame_rows
from closeloop_analyzer.mps_leftover import _percentile
from .non_mps_memcpy_compute import StudyError, _nvtx_ranges, _spearman
from .runner import ExperimentRunner, PreflightError, RunFailure
from closeloop_testbed.resource_control import GPUClockLock
from closeloop_analyzer.target_selection import decode_global_id


REQUIRED_FIELDS = {
    "schema_version", "study_id", "preregistered_at", "trio_preregistered_at",
    "base_config", "output_root", "generated_config_root", "checkpoint_root",
    "corruptions_root", "models", "conditions", "pair_condition_ids",
    "triple_condition_ids", "preselected_trio", "replicates",
    "minimum_samples", "bootstrap_repetitions", "confidence_level",
    "outlier_policy", "random_seed", "warmup_image", "warmup_lidar",
    "metadata_path", "software_provenance", "software_provenance_sha256",
}

CASE_STUDY_BOOTSTRAP_REPETITIONS = 20_000


def _resolve(source, value):
    path = Path(value).expanduser()
    return (source.parent / path).resolve() if not path.is_absolute() else path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_validation_policy(study, output_root):
    """Load an auditable post-registration validation amendment, if present."""
    policy = {
        "minimum_samples": study["data"]["minimum_samples"],
        "timestamp_comparability": "control_subset",
        "minimum_window_fraction": None,
        "amendment": None,
    }
    candidates = [Path(output_root) / "validation_amendment.yaml"]
    if study.get("source"):
        candidates.append(
            Path(study["source"]).parent / "validation_amendment.yaml"
        )
    path = next((candidate for candidate in candidates
                 if candidate.is_file()), candidates[0])
    if not path.is_file():
        return policy
    amendment = yaml.safe_load(path.read_bytes())
    required = {
        "schema_version", "amended_at", "study_sha256", "reason",
        "original_minimum_samples", "minimum_samples",
        "timestamp_comparability", "minimum_window_fraction",
    }
    if not isinstance(amendment, dict) or set(amendment) != required:
        raise StudyError("validation amendment fields differ from schema")
    if amendment["schema_version"] != 1:
        raise StudyError("unsupported validation amendment schema")
    if amendment["study_sha256"] != study["sha256"]:
        raise StudyError("validation amendment study SHA-256 differs")
    if amendment["original_minimum_samples"] != study["data"][
            "minimum_samples"]:
        raise StudyError(
            "validation amendment does not preserve original minima"
        )
    minimum = amendment["minimum_samples"]
    if (set(minimum) != {"image", "lidar"}
            or any(not isinstance(value, int) or value < 2
                   for value in minimum.values())):
        raise StudyError("validation amendment minimum samples are invalid")
    if amendment["timestamp_comparability"] != "source_manifest_window":
        raise StudyError("unsupported amended timestamp comparability policy")
    fraction = amendment["minimum_window_fraction"]
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise StudyError("validation amendment window fraction is invalid")
    policy.update(
        minimum_samples=minimum,
        timestamp_comparability=amendment["timestamp_comparability"],
        minimum_window_fraction=float(fraction), amendment=amendment,
    )
    return policy


def load_study(path, check_paths=True):
    """Load the reusable study contract and enforce its planned matrix."""
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise StudyError(f"cannot load input-variation study: {exc}") from exc
    if not isinstance(data, dict):
        raise StudyError("study root must be an object")
    missing, unknown = REQUIRED_FIELDS - set(data), set(data) - REQUIRED_FIELDS
    if missing or unknown:
        raise StudyError(
            f"study fields missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if data["schema_version"] != 1 or data["replicates"] < 1:
        raise StudyError("schema_version must be 1 and replicates positive")
    models = data["models"]
    if not isinstance(models, dict) or len(models) != 8:
        raise StudyError("models must contain the eight preplanned variants")
    lidar = {name for name, model in models.items()
             if model.get("group") == "lidar_detection"}
    camera = set(models) - lidar
    groups = {model.get("group") for model in models.values()}
    if len(lidar) != 2 or len(camera) != 6 or groups != {
        "lidar_detection", "image_segmentation", "image_detection"
    }:
        raise StudyError("model groups must be 2 LiDAR and 6 camera variants")
    trio = data["preselected_trio"]
    if len(trio) != 3 or len(set(trio)) != 3 or not set(trio) <= set(models):
        raise StudyError("preselected_trio must name three distinct models")
    if {models[name]["group"] for name in trio} != {
        "lidar_detection", "image_segmentation", "image_detection"
    }:
        raise StudyError(
            "trio must contain LiDAR, segmentation, and detection"
        )
    conditions = data["conditions"]
    if not isinstance(conditions, dict) or not conditions:
        raise StudyError("conditions must be a non-empty mapping")
    selected = set(data["pair_condition_ids"])
    if not selected or not selected <= set(conditions):
        raise StudyError("pair_condition_ids must reference known conditions")
    effects = {conditions[name]["effect"] for name in selected}
    if effects != {"surrounding_environment", "weather", "interaction"}:
        raise StudyError(
            "pair conditions must separate all three effect classes"
        )
    if not set(data["triple_condition_ids"]) <= selected:
        raise StudyError("triple conditions must be pair conditions")
    if float(data["confidence_level"]) != 0.95:
        raise StudyError("this study requires 95% confidence intervals")
    paths = {
        name: _resolve(source, data[name]) for name in (
            "base_config", "output_root", "generated_config_root",
            "checkpoint_root", "corruptions_root", "software_provenance",
        )
    }
    if check_paths:
        for name in (
            "base_config", "checkpoint_root", "corruptions_root",
            "software_provenance",
        ):
            if not paths[name].exists():
                raise StudyError(f"{name} does not exist: {paths[name]}")
        if _sha256(paths["software_provenance"]) != data[
                "software_provenance_sha256"]:
            raise StudyError("software provenance SHA-256 differs")
    return {
        "source": source, "sha256": hashlib.sha256(raw).hexdigest(),
        "data": data, "paths": paths, "lidar": sorted(lidar),
        "camera": sorted(camera),
    }


def expected_runs(study):
    """Return the exact single/two/three-model matrix keyed by run ID."""
    data = study["data"]
    result = {}
    for condition_id in data["pair_condition_ids"]:
        for replicate in range(1, data["replicates"] + 1):
            for model in sorted(data["models"]):
                run_id = f"iv-{condition_id}-single-{model}-r{replicate}"
                result[run_id] = {
                    "condition_id": condition_id, "models": [model],
                    "tenancy": "single", "replicate": replicate,
                }
            for lidar in study["lidar"]:
                for camera in study["camera"]:
                    run_id = (
                        f"iv-{condition_id}-pair-{lidar}-{camera}-r{replicate}"
                    )
                    result[run_id] = {
                        "condition_id": condition_id,
                        "models": [camera, lidar], "tenancy": "two_model",
                        "replicate": replicate,
                    }
    for condition_id in data["triple_condition_ids"]:
        for replicate in range(1, data["replicates"] + 1):
            names = list(data["preselected_trio"])
            run_id = f"iv-{condition_id}-triple-{'-'.join(names)}-r{replicate}"
            result[run_id] = {
                "condition_id": condition_id, "models": names,
                "tenancy": "three_model", "replicate": replicate,
            }
    return result


def _condition_paths(study, condition):
    return {
        name: _resolve(study["source"], condition[name]) for name in (
            "source_bag", "bag_directory", "dataset_manifest"
        )
    }


def prepare_conditions(study, selected=None, workers=None):
    """Create missing immutable ten-second input MCAPs and manifests."""
    results = {}
    for condition_id, condition in study["data"]["conditions"].items():
        if selected and condition_id not in selected:
            continue
        paths = _condition_paths(study, condition)
        if paths["dataset_manifest"].is_file():
            manifest = json.loads(paths["dataset_manifest"].read_text())
            errors = validate_manifest(manifest)
            corruption = manifest.get("corruption", {})
            if (
                errors
                or manifest.get("scene_token") != condition["scene_token"]
                or corruption.get("type") != condition["corruption_type"]
                or corruption.get("severity") != condition["severity"]
                or corruption.get("seed") != condition["seed"]
            ):
                raise StudyError(
                    f"existing condition {condition_id} is incompatible: "
                    f"{errors}"
                )
            results[condition_id] = "existing"
            continue
        manifest = transform_bag(
            paths["source_bag"], paths["bag_directory"],
            condition["scene_name"], condition["scene_token"],
            condition["corruption_type"], condition["severity"],
            condition["seed"], study["paths"]["corruptions_root"], workers,
        )
        if manifest != paths["dataset_manifest"]:
            raise StudyError(f"condition manifest path differs: {manifest}")
        results[condition_id] = "generated"
    if not selected:
        validate_conditions(study)
    return results


def validate_conditions(study):
    """Validate every condition and cross-condition source comparability."""
    manifests = {}
    source_groups = {}
    for condition_id, condition in study["data"]["conditions"].items():
        path = _condition_paths(study, condition)["dataset_manifest"]
        try:
            manifest = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise StudyError(
                f"condition {condition_id} manifest is unreadable: {exc}"
            ) from exc
        errors = validate_manifest(manifest)
        corruption = manifest.get("corruption", {})
        expected = (
            condition["scene_name"], condition["scene_token"],
            condition["corruption_type"], condition["severity"],
            condition["seed"],
        )
        observed = (
            manifest.get("scene_name"), manifest.get("scene_token"),
            corruption.get("type"), corruption.get("severity"),
            corruption.get("seed"),
        )
        changed = [
            record["source_payload_sha256"]
            != record["output_payload_sha256"]
            for record in manifest.get("sensor_records", [])
        ]
        if expected != observed:
            errors.append("scene/corruption/severity/seed differs from plan")
        if corruption.get("type") == "clean" and any(changed):
            errors.append("clean target payload changed")
        if corruption.get("type") != "clean" and not all(changed):
            errors.append("a weather target payload did not change")
        if errors:
            raise StudyError(
                f"condition {condition_id} is invalid: {'; '.join(errors)}"
            )
        manifests[condition_id] = manifest
        source_groups.setdefault(manifest["source_bag_sha256"], []).append(
            (condition_id, manifest)
        )
    for group in source_groups.values():
        identities = {
            (
                manifest["source_stream_sha256"],
                manifest["window_start_bag_timestamp_ns"],
                manifest["window_end_bag_timestamp_ns"],
                tuple(record["source_payload_sha256"]
                      for record in manifest["sensor_records"]),
            )
            for _condition_id, manifest in group
        }
        if len(identities) != 1:
            names = ", ".join(condition_id for condition_id, _ in group)
            raise StudyError(
                f"conditions do not share an identical source window: {names}"
            )
    return manifests


def _model_config(study, name, index, count):
    source = study["data"]["models"][name]
    chunks = {1: [(0, 8)], 2: [(0, 6), (6, 12)],
              3: [(0, 4), (4, 8), (8, 12)]}[count]
    start, stop = chunks[index]
    model = {
        key: source[key] for key in (
            "task", "modality", "mmlab_model", "architecture_profile",
            "model_config", "model_config_sha256", "checkpoint",
            "checkpoint_sha256",
        )
    }
    model["checkpoint"] = str(
        study["paths"]["checkpoint_root"] / model["checkpoint"]
    )
    model.update({
        "id": name, "node_name": name.replace("-", "_"),
        "input_topic": "/camera/front" if source["modality"] == "image"
        else "/lidar/top",
        "input_message_type": "compressed_image"
        if source["modality"] == "image" else "pointcloud2",
        "input_queue_depth": 1, "qos": "best_effort",
        "warmup_input": str(_resolve(
            study["source"], study["data"][
                "warmup_image" if source["modality"] == "image"
                else "warmup_lidar"
            ]
        )),
        "warmup_count": 5, "launch_offset_seconds": index,
        "module_annotation_depth": 0, "quiet_period_seconds": 0.5,
        "cpu_affinity": list(range(start, stop)),
        "cpu_thread_count": min(3, stop - start),
    })
    if "inference_resize_scale" in source:
        model["inference_resize_scale"] = source["inference_resize_scale"]
    return model


def _validate_shared_paths(study):
    """Hash each shared model artifact once before expanding the matrix."""
    for name, model in study["data"]["models"].items():
        config = Path(model["model_config"])
        checkpoint = study["paths"]["checkpoint_root"] / model["checkpoint"]
        for kind, path, expected in (
            ("config", config, model["model_config_sha256"]),
            ("checkpoint", checkpoint, model["checkpoint_sha256"]),
        ):
            if not path.is_file():
                raise StudyError(f"{name} {kind} is missing: {path}")
            if _sha256(path) != expected:
                raise StudyError(f"{name} {kind} SHA-256 differs")
    for name in ("metadata_path", "warmup_image", "warmup_lidar"):
        path = _resolve(study["source"], study["data"][name])
        if not path.is_file():
            raise StudyError(f"{name} is missing: {path}")


def materialize_configs(study, output_root=None, config_root=None):
    """Write immutable validated run configs from the study matrix."""
    validate_conditions(study)
    _validate_shared_paths(study)
    try:
        base = yaml.safe_load(study["paths"]["base_config"].read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise StudyError(f"cannot load base config: {exc}") from exc
    output_root = Path(output_root or study["paths"]["output_root"]).resolve()
    config_root = Path(
        config_root or study["paths"]["generated_config_root"]
    ).resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for run_id, planned in expected_runs(study).items():
        condition = study["data"]["conditions"][planned["condition_id"]]
        condition_paths = _condition_paths(study, condition)
        manifest = condition_paths["dataset_manifest"]
        if not manifest.is_file():
            raise StudyError(f"condition manifest missing: {manifest}")
        config = copy.deepcopy(base)
        config["run"].update({
            "id": run_id,
            "experiment": "pPerf Section 4.1 input-data variation",
            "provenance": (
                f"study={study['sha256']} preregistered="
                f"{study['data']['preregistered_at']} software="
                f"{study['data']['software_provenance_sha256']}"
            ),
            "output_root": str(output_root),
        })
        config["replay"].update({
            "scene_token": condition["scene_token"],
            "metadata_path": str(_resolve(
                study["source"], study["data"]["metadata_path"]
            )),
            "bag_directory": str(condition_paths["bag_directory"]),
            "playback_mode": "full",
        })
        names = planned["models"]
        config["models"] = [
            _model_config(study, name, index, len(names))
            for index, name in enumerate(names)
        ]
        config["input_variation"] = {
            "condition_id": planned["condition_id"],
            "effect": condition["effect"],
            "source_scene_name": condition["scene_name"],
            "scene_factors": condition["scene_factors"],
            "corruption": {
                "type": condition["corruption_type"],
                "severity": condition["severity"],
                "seed": condition["seed"],
                "dataset_manifest": str(manifest),
                "dataset_manifest_sha256": _sha256(manifest),
            },
            "replicate": planned["replicate"],
            "tenancy": planned["tenancy"],
        }
        content = yaml.safe_dump(
            schema_v2_config(config), sort_keys=False).encode()
        path = config_root / f"{run_id}.yaml"
        if path.exists() and path.read_bytes() != content:
            raise StudyError(f"immutable generated config differs: {path}")
        if not path.exists():
            path.write_bytes(content)
            path.chmod(0o444)
        try:
            load_run_config(str(path), check_paths=False)
        except ConfigError as exc:
            raise StudyError(
                f"generated config {run_id} invalid: {exc}"
            ) from exc
        paths[run_id] = path
    return paths


def run_campaign(study, artifact_root, selected=None, dry_run=False):
    """Run every missing planned config, preserving failures for validation."""
    artifact_root = Path(artifact_root).resolve()
    configs = materialize_configs(
        study, output_root=artifact_root,
        config_root=artifact_root / "generated_configs" /
        study["data"]["study_id"],
    )
    results = {}
    for run_id, path in configs.items():
        if selected and run_id not in selected:
            continue
        config = load_run_config(str(path), artifact_root=str(artifact_root))
        manifest_path = config.run_directory / "run_manifest.json"
        if manifest_path.is_file():
            state = json.loads(manifest_path.read_text()).get("state")
            results[run_id] = (
                "existing-success" if state == "success" else state
            )
            continue
        try:
            if dry_run:
                ExperimentRunner(config).run(dry_run=True)
            else:
                gpu = config.data["gpu"]
                control = GPUClockLock(
                    gpu["index"], gpu["graphics_clock_mhz"],
                    gpu["memory_clock_mhz"],
                )
                try:
                    with control:
                        ExperimentRunner(config).run()
                finally:
                    if config.run_directory.is_dir():
                        _write_json(
                            config.run_directory /
                            "input_variation_clock_control.json",
                            {
                                "requested_graphics_clock_mhz": gpu[
                                    "graphics_clock_mhz"
                                ],
                                "requested_memory_clock_mhz": gpu[
                                    "memory_clock_mhz"
                                ],
                                "control": control.evidence,
                                "clocks_restored": len(control.evidence.get(
                                    "reset_commands", []
                                )) == 2 and all(
                                    item.get("returncode") == 0
                                    for item in control.evidence.get(
                                        "reset_commands", []
                                    )
                                ),
                            },
                        )
            results[run_id] = "dry-run" if dry_run else "success"
        except (ConfigError, PreflightError, RunFailure, OSError) as exc:
            results[run_id] = f"failed: {exc}"
    return results


def latency_metrics(values):
    """Return the preplanned robust latency/spread statistics."""
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("latency metrics require samples")
    median = statistics.median(values)
    p5, p25 = _percentile(values, 0.05), _percentile(values, 0.25)
    p75, p95 = _percentile(values, 0.75), _percentile(values, 0.95)
    spread = p95 - p5
    return {
        "sample_count": len(values), "median_ms": median,
        "p25_ms": p25, "p75_ms": p75, "iqr_ms": p75 - p25,
        "p5_ms": p5, "p95_ms": p95, "p95_p5_ms": spread,
        "normalized_latency_spread": spread / median if median else None,
    }


def _moving_block_sample(values, rng):
    count = len(values)
    block = max(1, round(math.sqrt(count)))
    result = []
    while len(result) < count:
        start = rng.randrange(count)
        result.extend(
            values[(start + offset) % count] for offset in range(block)
        )
    return result[:count]


def metric_confidence_intervals(values, repetitions, seed):
    """Return temporal moving-block bootstrap 95% intervals."""
    values, rng = list(map(float, values)), random.Random(seed)
    names = ("median_ms", "iqr_ms", "p95_p5_ms",
             "normalized_latency_spread")
    samples = {name: [] for name in names}
    for _ in range(repetitions):
        metrics = latency_metrics(_moving_block_sample(values, rng))
        for name in names:
            samples[name].append(metrics[name])
    return {
        name + "_ci95": [
            _percentile(samples[name], 0.025),
            _percentile(samples[name], 0.975),
        ] for name in names
    }


def amplification_confidence_interval(pair, control, repetitions, seed):
    """Bootstrap the co-execution/control normalized-spread ratio."""
    rng, values = random.Random(seed), []
    for _ in range(repetitions):
        numerator = latency_metrics(_moving_block_sample(pair, rng))[
            "normalized_latency_spread"
        ]
        denominator = latency_metrics(_moving_block_sample(control, rng))[
            "normalized_latency_spread"
        ]
        if denominator:
            values.append(numerator / denominator)
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def ratio_of_ratios_confidence_interval(numerator_a, denominator_a,
                                        numerator_b, denominator_b,
                                        repetitions, seed):
    """Bootstrap a normalized-spread ratio divided by a second ratio."""
    rng, values = random.Random(seed), []
    series = (numerator_a, denominator_a, numerator_b, denominator_b)
    for _ in range(repetitions):
        spreads = [
            latency_metrics(_moving_block_sample(value, rng))[
                "normalized_latency_spread"
            ] for value in series
        ]
        if all(spreads):
            values.append((spreads[0] / spreads[1]) /
                          (spreads[2] / spreads[3]))
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def absolute_spread_difference_confidence_interval(
        numerator, denominator, repetitions, seed, confidence=0.95):
    """Bootstrap a difference between absolute P95--P5 latency widths."""
    rng, values = random.Random(seed), []
    for _ in range(repetitions):
        numerator_width = latency_metrics(
            _moving_block_sample(numerator, rng)
        )["p95_p5_ms"]
        denominator_width = latency_metrics(
            _moving_block_sample(denominator, rng)
        )["p95_p5_ms"]
        values.append(numerator_width - denominator_width)
    tail = (1.0 - confidence) / 2.0
    return [_percentile(values, tail), _percentile(values, 1.0 - tail)]


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _module_and_structure_rows(run_directory, statuses, input_ids):
    sqlite_path = Path(run_directory) / "profile.sqlite"
    modules, structures = [], []
    with sqlite3.connect(str(sqlite_path)) as connection:
        for record in _nvtx_ranges(connection):
            tag = record["tag"]
            model = tag.get("model")
            input_id = str(tag.get("input"))
            if (tag.get("event") == "module" and model in statuses
                    and input_id in input_ids[model]):
                modules.append({
                    "model_id": model, "input_id": input_id,
                    "module": tag.get("module"),
                    "duration_ms": (record["end"] - record["start"]) / 1e6,
                })
        if "NVTX_EVENTS" in {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }:
            for start, global_tid, text in connection.execute(
                "SELECT start,globalTid,text FROM NVTX_EVENTS "
                "WHERE text LIKE 'closeloop:%'"
            ):
                try:
                    tag = json.loads(str(text)[len("closeloop:"):])
                except json.JSONDecodeError:
                    continue
                model = tag.get("model")
                input_id = str(tag.get("input"))
                if (tag.get("event") == "structure" and model in statuses
                        and input_id in input_ids[model]):
                    structures.append({
                        "model_id": model, "input_id": input_id,
                        "module": tag.get("module"),
                        "structure_kind": tag.get("structure_kind"),
                        "structure_count": tag.get("structure_count"),
                        "timestamp_ns": int(start),
                        "pid": decode_global_id(int(global_tid))[0],
                    })
    return modules, structures


def validate_run(run_directory, planned, study, expected_config=None,
                 validation_policy=None):
    """Validate one completed run and extract its full evidence."""
    run_directory = Path(run_directory)
    errors = []
    required = (
        "config.yaml", "run_manifest.json", "testbed_result.json",
        "profile.sqlite", "input_variation_clock_control.json",
    )
    for name in required:
        if not (run_directory / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return {"valid": False, "errors": errors}
    config = yaml.safe_load((run_directory / "config.yaml").read_bytes())
    manifest = _load_json(run_directory / "run_manifest.json")
    testbed = _load_json(run_directory / "testbed_result.json")
    clock = _load_json(run_directory / "input_variation_clock_control.json")
    if expected_config is not None and (
            (run_directory / "config.yaml").read_bytes()
            != Path(expected_config).read_bytes()):
        errors.append("executed config differs from immutable planned config")
    variation = config.get("input_variation", {})
    if manifest.get("state") != "success":
        errors.append(f"run state is {manifest.get('state')}")
    if not manifest.get("gpu_hardware"):
        errors.append("GPU hardware identity is missing")
    if not clock.get("clocks_restored"):
        errors.append("GPU clock controls were not restored")
    if (
        not testbed.get("replay_success")
        or not testbed.get("all_acknowledged")
    ):
        errors.append("replay or model acknowledgement failed")
    repeat_count = config.get("replay", {}).get("repeat_count", 1)
    intervals = testbed.get("playback_intervals", [])
    if (
        testbed.get("playback_mode") != "full"
        or testbed.get("repeat_count", 1) != repeat_count
        or testbed.get("bags_started") != len(intervals)
        or testbed.get("bags_completed") != len(intervals)
        or not intervals
        or any(item.get("completion_status") != "completed"
               for item in intervals)
    ):
        errors.append("full replay passes did not all complete exactly once")
    if variation.get("condition_id") != planned["condition_id"]:
        errors.append("condition differs from plan")
    if variation.get("tenancy") != planned["tenancy"]:
        errors.append("tenancy differs from plan")
    configured_models = [model["id"] for model in config["models"]]
    if configured_models != planned["models"]:
        errors.append("model identities/order differ from plan")
    condition_manifest = {}
    try:
        condition_manifest = _load_json(
            variation.get("corruption", {}).get("dataset_manifest", "missing")
        )
        errors.extend(validate_manifest(condition_manifest))
        if _sha256(variation["corruption"]["dataset_manifest"]) != variation[
                "corruption"]["dataset_manifest_sha256"]:
            errors.append("condition manifest hash differs")
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        errors.append(f"condition provenance is unreadable: {exc}")
    statuses, input_ids = {}, {}
    for model in config["models"]:
        model_id = model["id"]
        status_path = run_directory / f"model_{model_id}.json"
        inputs_path = run_directory / f"model_{model_id}_inputs.jsonl"
        if not status_path.is_file() or not inputs_path.is_file():
            errors.append(f"{model_id} status or input evidence missing")
            continue
        status = _load_json(status_path)
        records = [
            json.loads(line) for line in inputs_path.read_text().splitlines()
            if line.strip()
        ]
        statuses[model_id] = status
        input_ids[model_id] = {str(record["input_id"]) for record in records}
        policy = validation_policy or {
            "minimum_samples": study["data"]["minimum_samples"],
            "timestamp_comparability": "control_subset",
            "minimum_window_fraction": None,
        }
        minimum = policy["minimum_samples"][model["modality"]]
        if status.get("state") != "acknowledged":
            errors.append(f"{model_id} did not acknowledge")
        if (
            status.get("inputs") != len(records)
            or len(input_ids[model_id]) != len(records)
        ):
            errors.append(f"{model_id} sample counts or IDs differ")
        source_timestamps = None
        if policy["timestamp_comparability"] == "source_manifest_window":
            topic = ("/CAM_FRONT/image_rect_compressed"
                     if model["modality"] == "image" else "/LIDAR_TOP")
            source_timestamps = {
                int(record["output_header_timestamp_ns"])
                for record in condition_manifest.get("sensor_records", [])
                if record.get("topic") == topic
            }
        errors.extend(_validate_source_passes(
            model_id, records, repeat_count, minimum, source_timestamps,
            policy["minimum_window_fraction"],
        ))
        if any(value.startswith("warmup-") for value in input_ids[model_id]):
            errors.append(f"{model_id} warmup leaked into measured inputs")
        for name in ("mmlab_model", "model_config", "model_config_sha256",
                     "checkpoint", "checkpoint_sha256"):
            if status.get(name) != model.get(name):
                errors.append(f"{model_id} observed {name} differs")
    rows, modules, structures = [], [], []
    try:
        rows, trace = extract_frame_rows(run_directory)
        if len(rows) != sum(len(values) for values in input_ids.values()):
            errors.append("trace/sample row count differs")
        if any(str(row["frame_id"]).startswith("warmup-") for row in rows):
            errors.append("warmup range entered frame metrics")
        if any(row["completion_boundary"] not in (
                "post_nvtx_cuda_event_synchronize",
                "nvtx_includes_completion_synchronize") for row in rows):
            errors.append("timing completion boundary is not synchronized")
        if set(statuses) == set(configured_models):
            modules, structures = _module_and_structure_rows(
                run_directory, statuses, input_ids
            )
        warmups = {model["id"]: 0 for model in config["models"]}
        with sqlite3.connect(str(run_directory / "profile.sqlite")) as db:
            for record in _nvtx_ranges(db):
                tag = record["tag"]
                if (tag.get("event") == "inference"
                        and str(tag.get("input", "")).startswith("warmup-")
                        and tag.get("model") in warmups):
                    warmups[tag["model"]] += 1
        for model in config["models"]:
            if warmups[model["id"]] != model["warmup_count"]:
                errors.append(
                    f"{model['id']} warmup range count differs: "
                    f"{warmups[model['id']]} != {model['warmup_count']}"
                )
    except (StudyError, OSError, sqlite3.DatabaseError) as exc:
        errors.append(f"trace extraction failed: {exc}")
        trace = {}
    return {
        "valid": not errors, "errors": errors, "frames": rows,
        "modules": modules, "structures": structures,
        "trace_evidence": trace, "config": config,
    }


def _validate_source_passes(model_id, records, repeat_count, minimum,
                            source_timestamps=None, minimum_window_fraction=None):
    """Validate sample count and optional source coverage for every replay pass."""
    errors = []
    grouped = {}
    for record in records:
        grouped.setdefault(int(record.get("pass_index", 0)), []).append(record)
    expected = set(range(repeat_count))
    if set(grouped) != expected:
        errors.append(
            f"{model_id} pass indexes differ: "
            f"{sorted(grouped)} != {sorted(expected)}"
        )
    for pass_index in sorted(expected):
        selected = grouped.get(pass_index, [])
        if len(selected) < minimum:
            errors.append(
                f"{model_id} pass {pass_index} has {len(selected)} samples, "
                f"needs {minimum}"
            )
        if source_timestamps is None:
            continue
        measured = {
            int(record["ros_header_timestamp_ns"]) for record in selected
        }
        if not measured <= source_timestamps:
            errors.append(
                f"{model_id} pass {pass_index} has timestamps outside "
                "source manifest"
            )
        elif len(measured) > 1 and len(source_timestamps) > 1:
            coverage = ((max(measured) - min(measured)) /
                        (max(source_timestamps) - min(source_timestamps)))
            if coverage < minimum_window_fraction:
                errors.append(
                    f"{model_id} pass {pass_index} covers {coverage:.3f} of "
                    f"source window, needs {minimum_window_fraction:.3f}"
                )
    return errors


def discover_and_validate(study, output_root=None):
    """Discover planned outputs and report invalid or missing runs."""
    root = Path(output_root or study["paths"]["output_root"]).resolve()
    plan = expected_runs(study)
    configs = materialize_configs(study, output_root=root)
    policy = _load_validation_policy(study, root)
    results = {}
    for run_id, planned in plan.items():
        directory = root / run_id
        results[run_id] = (
            validate_run(
                directory, planned, study, configs[run_id], policy
            )
            if directory.is_dir()
            else {"valid": False, "errors": ["missing run directory"]}
        )
    for run_id, evidence in results.items():
        if not evidence["valid"]:
            continue
        planned = plan[run_id]
        if planned["tenancy"] == "single":
            continue
        for model_id in planned["models"]:
            control_id = (
                f"iv-{planned['condition_id']}-single-{model_id}-"
                f"r{planned['replicate']}"
            )
            control = results.get(control_id, {})
            if not control.get("valid"):
                evidence["errors"].append(
                    f"corresponding control {control_id} is not valid"
                )
                evidence["valid"] = False
                continue
            if policy["timestamp_comparability"] != "control_subset":
                continue
            measured = {
                row["ros_header_timestamp_ns"] for row in evidence["frames"]
                if row["model_id"] == model_id
            }
            reference = {
                row["ros_header_timestamp_ns"] for row in control["frames"]
                if row["model_id"] == model_id
            }
            if not measured or not measured <= reference:
                evidence["errors"].append(
                    f"{model_id} timestamps are not comparable to its control"
                )
                evidence["valid"] = False
    hardware = {}
    for run_id, evidence in results.items():
        if evidence["valid"]:
            identity = json.dumps(
                _load_json(root / run_id / "run_manifest.json").get(
                    "gpu_hardware"
                ), sort_keys=True
            )
            hardware.setdefault(identity, []).append(run_id)
    if len(hardware) > 1:
        for runs in hardware.values():
            for run_id in runs:
                results[run_id]["valid"] = False
                results[run_id]["errors"].append(
                    "GPU hardware differs across study runs"
                )
    unexpected = sorted(
        path.name for path in root.iterdir()
        if (path.is_dir() and path.name.startswith("iv-")
            and path.name not in plan)
    ) if root.is_dir() else []
    return results, unexpected


def extract_input_geometry(study):
    """Read radial LiDAR density from each immutable condition bag."""
    import numpy as np
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    rows = []
    bands = (
        ("points_0_5m", 0.0, 5.0),
        ("points_5_10m", 5.0, 10.0),
        ("points_10_20m", 10.0, 20.0),
        ("points_20_40m", 20.0, 40.0),
        ("points_40m_plus", 40.0, math.inf),
    )
    for condition_id, condition in study["data"]["conditions"].items():
        paths = _condition_paths(study, condition)
        manifest = _load_json(paths["dataset_manifest"])
        expected = {
            int(record["output_header_timestamp_ns"]): int(
                record["output_point_count"]
            )
            for record in manifest["sensor_records"]
            if record["topic"] == LIDAR_TOPIC
        }
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(
                uri=str(paths["bag_directory"]), storage_id="mcap"
            ),
            rosbag2_py.ConverterOptions("", ""),
        )
        observed = set()
        while reader.has_next():
            topic, raw, _timestamp_ns = reader.read_next()
            if topic != LIDAR_TOPIC:
                continue
            message = deserialize_message(raw, PointCloud2)
            timestamp_ns = _header_timestamp_ns(message)
            records = np.frombuffer(
                bytes(message.data), dtype=_point_dtype(message),
                count=int(message.width) * int(message.height),
            )
            radius = np.hypot(records["x"], records["y"])
            row = {
                "condition_id": condition_id,
                "effect": condition["effect"],
                "corruption_type": condition["corruption_type"],
                "severity": condition["severity"],
                "ros_header_timestamp_ns": timestamp_ns,
                "total_points": len(records),
            }
            for name, lower, upper in bands:
                row[name] = int(((radius >= lower) & (radius < upper)).sum())
            if expected.get(timestamp_ns) != len(records):
                raise StudyError(
                    f"{condition_id} LiDAR geometry differs from manifest"
                )
            observed.add(timestamp_ns)
            rows.append(row)
        if observed != set(expected):
            raise StudyError(
                f"{condition_id} LiDAR geometry timestamps are incomplete"
            )
    return rows


def _bootstrap_median_interval(values, repetitions, seed):
    """Bootstrap a median across repeated run-level diagnostics."""
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    rng = random.Random(seed)
    medians = [
        statistics.median(
            values[rng.randrange(len(values))] for _ in values
        )
        for _ in range(repetitions)
    ]
    return [_percentile(medians, 0.025), _percentile(medians, 0.975)]


def _paired_median_ratio_interval(numerator, denominator, repetitions, seed):
    """Moving-block bootstrap a paired median ratio."""
    if len(numerator) != len(denominator) or not numerator:
        return None
    rng, values, count = random.Random(seed), [], len(numerator)
    block = max(1, round(math.sqrt(count)))
    for _ in range(repetitions):
        indexes = []
        while len(indexes) < count:
            start = rng.randrange(count)
            indexes.extend(
                (start + offset) % count for offset in range(block)
            )
        indexes = indexes[:count]
        base = statistics.median(denominator[index] for index in indexes)
        if base:
            values.append(
                statistics.median(numerator[index] for index in indexes)
                / base
            )
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def summarize_input_geometry(study, rows):
    """Summarize point density and fixed-scene weather changes."""
    fields = (
        "total_points", "points_0_5m", "points_5_10m",
        "points_10_20m", "points_20_40m", "points_40m_plus",
    )
    by_condition = {
        condition_id: sorted(
            [row for row in rows if row["condition_id"] == condition_id],
            key=lambda row: row["ros_header_timestamp_ns"],
        )
        for condition_id in study["data"]["conditions"]
    }
    clean = by_condition.get("weather-clean", [])
    output = []
    for condition_id, condition_rows in by_condition.items():
        condition = study["data"]["conditions"][condition_id]
        record = {
            "condition_id": condition_id,
            "effect": condition["effect"],
            "corruption_type": condition["corruption_type"],
            "severity": condition["severity"],
            "sample_count": len(condition_rows),
            "status": "exploratory_diagnostic_post_hoc",
        }
        for field in fields:
            values = [row[field] for row in condition_rows]
            metrics = latency_metrics(values)
            record[field + "_median"] = metrics["median_ms"]
            record[field + "_normalized_spread"] = metrics[
                "normalized_latency_spread"
            ]
            if (condition["effect"] == "weather"
                    and condition_id != "weather-clean"):
                clean_values = [row[field] for row in clean]
                record[field + "_median_ratio_to_clean"] = (
                    metrics["median_ms"]
                    / latency_metrics(clean_values)["median_ms"]
                )
                record[field + "_median_ratio_ci95"] = (
                    _paired_median_ratio_interval(
                        values, clean_values,
                        study["data"]["bootstrap_repetitions"],
                        derived_analysis_seed(
                            study, "input-geometry", condition_id, field
                        ),
                    )
                )
        output.append(record)
    return output


def architecture_pathway_analysis(study, geometry_rows, frame_rows,
                                  module_rows, structure_rows):
    """Connect raw point density to voxels, modules, and latency."""
    geometry = {
        (row["condition_id"], int(row["ros_header_timestamp_ns"])): row
        for row in geometry_rows
    }
    voxels = {
        (row["run_id"], row["model_id"], str(row["input_id"])):
        float(row["structure_count"])
        for row in structure_rows if row["structure_kind"] == "voxels"
    }
    modules = {}
    for row in module_rows:
        key = (row["run_id"], row["model_id"], str(row["input_id"]))
        modules.setdefault(key, {})[row["module"]] = float(
            row["duration_ms"]
        )
    grouped = {}
    for frame in frame_rows:
        if frame["model_id"] not in study["lidar"]:
            continue
        key = (frame["run_id"], frame["model_id"], str(frame["frame_id"]))
        input_row = geometry.get(
            (frame["condition_id"], int(frame["ros_header_timestamp_ns"]))
        )
        voxel_count = voxels.get(key)
        if input_row is None or voxel_count is None:
            continue
        group = (
            frame["run_id"], frame["model_id"], frame["tenancy"],
            frame["condition_id"],
        )
        item = {
            "raw_points": float(input_row["total_points"]),
            "near_points": float(input_row["points_0_5m"]),
            "voxels": voxel_count,
            "latency": float(frame["inference_e2e_ms"]),
        }
        for module, duration in modules.get(key, {}).items():
            item["module:" + module] = duration
        grouped.setdefault(group, []).append(item)
    run_rows = []
    base_paths = (
        ("raw_points_to_voxels", "raw_points", "voxels"),
        ("near_points_to_voxels", "near_points", "voxels"),
        ("raw_points_to_latency", "raw_points", "latency"),
        ("voxels_to_latency", "voxels", "latency"),
    )
    for group, values in grouped.items():
        paths = list(base_paths)
        module_fields = sorted({
            key for value in values for key in value
            if key.startswith("module:")
        })
        paths.extend(
            ("voxels_to_" + field, "voxels", field)
            for field in module_fields
        )
        for pathway, source, target in paths:
            pairs = [
                (value[source], value[target]) for value in values
                if source in value and target in value
            ]
            rho = (
                _spearman([pair[0] for pair in pairs],
                          [pair[1] for pair in pairs])
                if len(pairs) >= 5 else None
            )
            if rho is not None:
                run_rows.append({
                    "run_id": group[0], "model_id": group[1],
                    "tenancy": group[2], "condition_id": group[3],
                    "pathway": pathway, "sample_count": len(pairs),
                    "spearman": rho,
                    "status": "exploratory_diagnostic_post_hoc",
                })
    summary = []
    repetitions = study["data"]["bootstrap_repetitions"]
    for key in sorted({
        (row["model_id"], row["tenancy"], row["pathway"])
        for row in run_rows
    }):
        values = [
            row["spearman"] for row in run_rows
            if (row["model_id"], row["tenancy"], row["pathway"]) == key
        ]
        summary.append({
            "model_id": key[0], "tenancy": key[1], "pathway": key[2],
            "run_count": len(values),
            "median_spearman": statistics.median(values),
            "median_spearman_ci95": _bootstrap_median_interval(
                values, repetitions,
                derived_analysis_seed(study, "architecture-pathway", *key),
            ),
            "positive_runs": sum(value > 0 for value in values),
            "status": "exploratory_diagnostic_post_hoc",
        })
    return run_rows, summary


def cross_tenant_pathway_analysis(study, frame_rows, structure_rows):
    """Relate a tenant's latency/waiting to overlapping co-runner work."""
    frames = {}
    for row in frame_rows:
        if row["tenancy"] != "two_model":
            continue
        frames.setdefault(row["run_id"], {}).setdefault(
            row["model_id"], []
        ).append(row)
    voxels = {
        (row["run_id"], row["model_id"], str(row["input_id"])):
        float(row["structure_count"])
        for row in structure_rows if row["structure_kind"] == "voxels"
    }
    run_rows = []
    fields = (
        ("target_wait_to_corunner_active",
         "memcpy_adjacent_waiting_ns", "gpu_kernel_active_ms"),
        ("target_latency_to_corunner_span",
         "inference_e2e_ms", "kernel_span_ms"),
        ("target_latency_to_corunner_latency",
         "inference_e2e_ms", "inference_e2e_ms"),
    )
    for run_id, models in frames.items():
        if len(models) != 2:
            continue
        names = list(models)
        for target_model in names:
            corunner_model = next(
                model for model in names if model != target_model
            )
            pairs = []
            for target in models[target_model]:
                best, best_overlap = None, 0
                for corunner in models[corunner_model]:
                    overlap = min(
                        target["host_nvtx_end_ns"],
                        corunner["host_nvtx_end_ns"],
                    ) - max(
                        target["host_nvtx_start_ns"],
                        corunner["host_nvtx_start_ns"],
                    )
                    if overlap > best_overlap:
                        best, best_overlap = corunner, overlap
                if best is not None:
                    pairs.append((target, best))
            target_kind = (
                "lidar" if target_model in study["lidar"] else "camera"
            )
            corunner_kind = (
                "lidar" if corunner_model in study["lidar"] else "camera"
            )
            for pathway, target_field, corunner_field in fields:
                rho = (
                    _spearman(
                        [pair[0][target_field] for pair in pairs],
                        [pair[1][corunner_field] for pair in pairs],
                    ) if len(pairs) >= 5 else None
                )
                if rho is not None:
                    run_rows.append({
                        "run_id": run_id, "target_model": target_model,
                        "corunner_model": corunner_model,
                        "target_kind": target_kind,
                        "corunner_kind": corunner_kind,
                        "pathway": pathway, "sample_count": len(pairs),
                        "overlap_fraction": (
                            len(pairs) / len(models[target_model])
                        ),
                        "spearman": rho,
                        "status": "exploratory_diagnostic_post_hoc",
                    })
            if corunner_kind == "lidar":
                voxel_pairs = [
                    (target, voxels.get((
                        run_id, corunner_model, str(corunner["frame_id"])
                    )))
                    for target, corunner in pairs
                ]
                voxel_pairs = [pair for pair in voxel_pairs
                               if pair[1] is not None]
                if len(voxel_pairs) >= 5:
                    rho = _spearman(
                        [pair[0]["memcpy_adjacent_waiting_ns"]
                         for pair in voxel_pairs],
                        [pair[1] for pair in voxel_pairs],
                    )
                    if rho is not None:
                        run_rows.append({
                            "run_id": run_id,
                            "target_model": target_model,
                            "corunner_model": corunner_model,
                            "target_kind": target_kind,
                            "corunner_kind": corunner_kind,
                            "pathway": "target_wait_to_corunner_voxels",
                            "sample_count": len(voxel_pairs),
                            "overlap_fraction": (
                                len(pairs) / len(models[target_model])
                            ),
                            "spearman": rho,
                            "status": "exploratory_diagnostic_post_hoc",
                        })
    summary = []
    repetitions = study["data"]["bootstrap_repetitions"]
    for key in sorted({
        (row["target_kind"], row["corunner_kind"], row["pathway"])
        for row in run_rows
    }):
        values = [
            row["spearman"] for row in run_rows
            if (row["target_kind"], row["corunner_kind"], row["pathway"])
            == key
        ]
        summary.append({
            "target_kind": key[0], "corunner_kind": key[1],
            "pathway": key[2], "run_count": len(values),
            "median_spearman": statistics.median(values),
            "median_spearman_ci95": _bootstrap_median_interval(
                values, repetitions,
                derived_analysis_seed(study, "cross-tenant-pathway", *key),
            ),
            "positive_runs": sum(value > 0 for value in values),
            "status": "exploratory_diagnostic_post_hoc",
        })
    return run_rows, summary


def analyze(study, output_root=None, analysis_root=None):
    """Validate and write metrics, mechanisms, tables, and plots."""
    validations, unexpected = discover_and_validate(study, output_root)
    output_root = Path(output_root or study["paths"]["output_root"]).resolve()
    validation_policy = _load_validation_policy(study, output_root)
    analysis_root = Path(
        analysis_root or study["paths"]["output_root"] / "analysis"
    ).resolve()
    valid = {run_id: value for run_id, value in validations.items()
             if value["valid"]}
    frame_rows, module_rows, structure_rows, summaries = [], [], [], []
    repetitions = study["data"]["bootstrap_repetitions"]
    for run_id, evidence in valid.items():
        planned = expected_runs(study)[run_id]
        condition = study["data"]["conditions"][planned["condition_id"]]
        for row in evidence["frames"]:
            enriched = dict(row, run_id=run_id,
                            condition_id=planned["condition_id"],
                            effect=condition["effect"],
                            corruption_type=condition["corruption_type"],
                            severity=condition["severity"],
                            tenancy=planned["tenancy"])
            frame_rows.append(enriched)
        for collection, target in ((evidence["modules"], module_rows),
                                   (evidence["structures"], structure_rows)):
            target.extend(dict(row, run_id=run_id,
                               condition_id=planned["condition_id"],
                               tenancy=planned["tenancy"])
                          for row in collection)
        for model_id in planned["models"]:
            values = [row["inference_e2e_ms"] for row in evidence["frames"]
                      if row["model_id"] == model_id]
            metrics = latency_metrics(values)
            metrics.update(metric_confidence_intervals(
                values, repetitions,
                derived_analysis_seed(study, run_id, model_id),
            ))
            summaries.append(dict(
                metrics, run_id=run_id, model_id=model_id,
                co_models=";".join(planned["models"]),
                condition_id=planned["condition_id"],
                effect=condition["effect"],
                corruption_type=condition["corruption_type"],
                severity=condition["severity"], tenancy=planned["tenancy"],
            ))
    by_key = {(row["condition_id"], row["model_id"], row["tenancy"]): row
              for row in summaries}
    for row in summaries:
        if row["tenancy"] == "single":
            continue
        control = by_key.get((row["condition_id"], row["model_id"], "single"))
        if not control or not control["normalized_latency_spread"]:
            continue
        row["amplification_factor"] = (
            row["normalized_latency_spread"] /
            control["normalized_latency_spread"]
        )
        pair = [item["inference_e2e_ms"] for item in frame_rows
                if item["run_id"] == row["run_id"]
                and item["model_id"] == row["model_id"]]
        single = [item["inference_e2e_ms"] for item in frame_rows
                  if item["run_id"] == control["run_id"]
                  and item["model_id"] == row["model_id"]]
        row["amplification_factor_ci95"] = amplification_confidence_interval(
            pair, single, repetitions,
            derived_analysis_seed(study, row["run_id"], "amplification"),
        )
    contrasts = effect_contrasts(study, summaries, frame_rows)
    mechanisms, exploratory = exploratory_analysis(
        study, summaries, frame_rows, module_rows, structure_rows
    )
    input_geometry = extract_input_geometry(study)
    input_geometry_summary = summarize_input_geometry(
        study, input_geometry
    )
    pathway_runs, pathway_summary = architecture_pathway_analysis(
        study, input_geometry, frame_rows, module_rows, structure_rows
    )
    cross_tenant_runs, cross_tenant_summary = (
        cross_tenant_pathway_analysis(study, frame_rows, structure_rows)
    )
    module_summaries = summarize_modules(module_rows)
    case_study = case_study_analysis(study, summaries, frame_rows)
    primary_summary = summarize_primary_findings(
        study, summaries, contrasts
    )
    validation_summary = {
        "schema_version": 1, "study_sha256": study["sha256"],
        "planned_runs": len(validations), "valid_runs": len(valid),
        "invalid_runs": {
            run_id: value["errors"] for run_id, value in validations.items()
            if not value["valid"]
        },
        "unexpected_runs": unexpected,
        "complete": len(valid) == len(validations) and not unexpected,
        "outlier_policy": study["data"]["outlier_policy"],
        "validation_policy": validation_policy,
    }
    _write_json(analysis_root / "validation_summary.json", validation_summary)
    _write_json(analysis_root / "latency_summary.json", summaries)
    _write_csv(analysis_root / "latency_summary.csv", summaries)
    _write_csv(analysis_root / "frame_metrics.csv", frame_rows)
    _write_csv(analysis_root / "module_metrics.csv", module_rows)
    _write_csv(analysis_root / "structure_metrics.csv", structure_rows)
    _write_json(analysis_root / "effect_contrasts.json", contrasts)
    _write_csv(analysis_root / "effect_contrasts.csv", contrasts)
    _write_json(analysis_root / "mechanism_summary.json", mechanisms)
    _write_csv(analysis_root / "mechanism_summary.csv", mechanisms)
    _write_json(analysis_root / "module_summary.json", module_summaries)
    _write_csv(analysis_root / "module_summary.csv", module_summaries)
    _write_json(analysis_root / "case_study_summary.json", case_study)
    _write_csv(analysis_root / "case_study_summary.csv", case_study)
    _write_json(analysis_root / "exploratory_summary.json", exploratory)
    _write_csv(analysis_root / "input_geometry_metrics.csv", input_geometry)
    _write_json(
        analysis_root / "input_geometry_summary.json",
        input_geometry_summary,
    )
    _write_csv(
        analysis_root / "input_geometry_summary.csv",
        input_geometry_summary,
    )
    _write_csv(
        analysis_root / "architecture_pathway_runs.csv", pathway_runs
    )
    _write_json(
        analysis_root / "architecture_pathway_summary.json",
        pathway_summary,
    )
    _write_csv(
        analysis_root / "architecture_pathway_summary.csv",
        pathway_summary,
    )
    _write_csv(
        analysis_root / "cross_tenant_pathway_runs.csv", cross_tenant_runs
    )
    _write_json(
        analysis_root / "cross_tenant_pathway_summary.json",
        cross_tenant_summary,
    )
    _write_csv(
        analysis_root / "cross_tenant_pathway_summary.csv",
        cross_tenant_summary,
    )
    _write_json(analysis_root / "primary_summary.json", primary_summary)
    _write_csv(analysis_root / "primary_summary.csv", primary_summary)
    _write_report(analysis_root / "validation_report.md", validation_summary)
    _write_table(analysis_root / "section4_1_table.tex", summaries)
    _write_primary_table(
        analysis_root / "section4_1_primary_table.tex", primary_summary
    )
    _write_plot(analysis_root / "section4_1_spread.svg", summaries)
    _write_modifier_plot(
        analysis_root / "section4_1_modifier_classification.svg",
        primary_summary,
    )
    _write_modifier_plot_png(
        analysis_root / "section4_1_modifier_classification.png",
        primary_summary,
    )
    return validation_summary


def _frame_values(rows, run_id, model_id, field):
    return [row[field] for row in rows if row["run_id"] == run_id
            and row["model_id"] == model_id and row.get(field) is not None]


def effect_contrasts(study, summaries, frame_rows):
    """Compare environment, weather, and their interaction independently."""
    index = {
        (row["condition_id"], row["tenancy"], row["co_models"],
         row["model_id"]): row for row in summaries
    }
    output = []
    environment_ids = [
        name for name in study["data"]["pair_condition_ids"]
        if study["data"]["conditions"][name]["effect"]
        == "surrounding_environment"
    ]
    keys = {(row["tenancy"], row["co_models"], row["model_id"])
            for row in summaries}
    for tenancy, co_models, model_id in sorted(keys):
        environment = [
            index.get((name, tenancy, co_models, model_id))
            for name in environment_ids
        ]
        environment = [row for row in environment if row]
        if len(environment) == len(environment_ids):
            values = [row["normalized_latency_spread"] for row in environment]
            if all(value is not None for value in values):
                output.append({
                    "contrast": "surrounding_environment_range",
                    "tenancy": tenancy, "co_models": co_models,
                    "model_id": model_id,
                    "condition_id": "all-clean-scenes",
                    "estimate": max(values) - min(values),
                    "minimum": min(values), "maximum": max(values),
                })
        clean = index.get(("weather-clean", tenancy, co_models, model_id))
        if clean:
            for condition_id in (
                "weather-rain-s1", "weather-rain-s3", "weather-rain-s5",
                "weather-snow-s1", "weather-snow-s3", "weather-snow-s5",
            ):
                weather = index.get(
                    (condition_id, tenancy, co_models, model_id)
                )
                if (not weather or not clean["normalized_latency_spread"]
                        or weather["normalized_latency_spread"] is None):
                    continue
                ratio = (weather["normalized_latency_spread"] /
                         clean["normalized_latency_spread"])
                pair_values = _frame_values(
                    frame_rows, weather["run_id"], model_id,
                    "inference_e2e_ms"
                )
                clean_values = _frame_values(
                    frame_rows, clean["run_id"], model_id,
                    "inference_e2e_ms"
                )
                output.append({
                    "contrast": "weather_to_fixed_scene_clean",
                    "tenancy": tenancy, "co_models": co_models,
                    "model_id": model_id, "condition_id": condition_id,
                    "estimate": ratio,
                    "ci95": amplification_confidence_interval(
                        pair_values, clean_values,
                        study["data"]["bootstrap_repetitions"],
                        derived_analysis_seed(
                            study, condition_id, tenancy, co_models, model_id
                        ),
                    ),
                })
        if tenancy != "single":
            single_clean = index.get(
                ("weather-clean", "single", model_id, model_id)
            )
            tenant_clean = index.get(
                ("weather-clean", tenancy, co_models, model_id)
            )
            for condition_id in (
                "weather-rain-s1", "weather-rain-s3", "weather-rain-s5",
                "weather-snow-s1", "weather-snow-s3", "weather-snow-s5",
            ):
                tenant_weather = index.get(
                    (condition_id, tenancy, co_models, model_id)
                )
                single_weather = index.get(
                    (condition_id, "single", model_id, model_id)
                )
                rows = (tenant_weather, tenant_clean,
                        single_weather, single_clean)
                if not all(rows) or any(
                        not row["normalized_latency_spread"] for row in rows):
                    continue
                tenant_ratio = (
                    tenant_weather["normalized_latency_spread"] /
                    tenant_clean["normalized_latency_spread"]
                )
                single_ratio = (
                    single_weather["normalized_latency_spread"] /
                    single_clean["normalized_latency_spread"]
                )
                values = [
                    _frame_values(frame_rows, row["run_id"], model_id,
                                  "inference_e2e_ms") for row in rows
                ]
                output.append({
                    "contrast": "multi_tenant_weather_modifier",
                    "tenancy": tenancy, "co_models": co_models,
                    "model_id": model_id, "condition_id": condition_id,
                    "estimate": tenant_ratio / single_ratio,
                    "multi_tenant_weather_ratio": tenant_ratio,
                    "single_weather_ratio": single_ratio,
                    "ci95": ratio_of_ratios_confidence_interval(
                        *values, study["data"]["bootstrap_repetitions"],
                        derived_analysis_seed(
                            study, "multi-tenant-weather", condition_id,
                            tenancy, co_models, model_id,
                        ),
                    ),
                })
        interaction_map = {
            "interaction-low-rain-s3": (
                "env-low-urban", "weather-rain-s3"
            ),
            "interaction-low-snow-s3": (
                "env-low-urban", "weather-snow-s3"
            ),
            "interaction-crowded-rain-s3": (
                "env-crowded-intersection", "weather-rain-s3"
            ),
            "interaction-crowded-snow-s3": (
                "env-crowded-intersection", "weather-snow-s3"
            ),
        }
        for condition_id, (scene_clean_id, fixed_weather_id) in (
                interaction_map.items()):
            joint = index.get((condition_id, tenancy, co_models, model_id))
            scene_clean = index.get(
                (scene_clean_id, tenancy, co_models, model_id)
            )
            fixed_weather = index.get(
                (fixed_weather_id, tenancy, co_models, model_id)
            )
            fixed_clean = index.get(
                ("weather-clean", tenancy, co_models, model_id)
            )
            if not all((joint, scene_clean, fixed_weather, fixed_clean)):
                continue
            spreads = [
                row["normalized_latency_spread"]
                for row in (joint, scene_clean, fixed_weather, fixed_clean)
            ]
            if any(value is None for value in spreads) or not (
                    scene_clean["normalized_latency_spread"]
                    and fixed_clean["normalized_latency_spread"]
                    and fixed_weather["normalized_latency_spread"]):
                continue
            scene_ratio = (joint["normalized_latency_spread"] /
                           scene_clean["normalized_latency_spread"])
            fixed_ratio = (fixed_weather["normalized_latency_spread"] /
                           fixed_clean["normalized_latency_spread"])
            output.append({
                "contrast": "environment_weather_interaction_modifier",
                "tenancy": tenancy, "co_models": co_models,
                "model_id": model_id, "condition_id": condition_id,
                "estimate": scene_ratio / fixed_ratio,
                "scene_weather_ratio": scene_ratio,
                "fixed_scene_weather_ratio": fixed_ratio,
                "ci95": ratio_of_ratios_confidence_interval(
                    *[
                        _frame_values(
                            frame_rows, row["run_id"], model_id,
                            "inference_e2e_ms",
                        ) for row in (
                            joint, scene_clean, fixed_weather, fixed_clean
                        )
                    ],
                    study["data"]["bootstrap_repetitions"],
                    derived_analysis_seed(
                        study, "environment-weather-interaction",
                        condition_id, tenancy, co_models, model_id,
                    ),
                ),
            })
    return output


def exploratory_analysis(study, summaries, frame_rows, module_rows,
                         structure_rows):
    """Search mechanism and severity relationships without promoting claims."""
    mechanisms = []
    mechanism_fields = (
        "gpu_kernel_active_ms", "kernel_span_ms", "kernel_count",
        "memcpy_adjacent_waiting_ns", "alignment_copy_count",
    )
    for summary in summaries:
        record = {
            key: summary[key] for key in (
                "run_id", "condition_id", "tenancy", "co_models", "model_id"
            )
        }
        record["latency_normalized_spread"] = summary[
            "normalized_latency_spread"
        ]
        for field in mechanism_fields:
            values = _frame_values(
                frame_rows, summary["run_id"], summary["model_id"], field
            )
            record[field + "_normalized_spread"] = (
                latency_metrics(values)["normalized_latency_spread"]
                if values else None
            )
        for kind in ("voxels", "proposals"):
            values = [row["structure_count"] for row in structure_rows
                      if row["run_id"] == summary["run_id"]
                      and row["model_id"] == summary["model_id"]
                      and row["structure_kind"] == kind]
            record[kind + "_normalized_spread"] = (
                latency_metrics(values)["normalized_latency_spread"]
                if values else None
            )
        mechanisms.append(record)
    correlations = []
    fields = [
        key for key in mechanisms[0]
        if key.endswith("_normalized_spread")
        and key != "latency_normalized_spread"
    ] if mechanisms else []
    for field in fields:
        pairs = [(row["latency_normalized_spread"], row[field])
                 for row in mechanisms if row.get(field) is not None]
        by_model = {}
        for model_id in sorted({row["model_id"] for row in mechanisms}):
            values = [
                (row["latency_normalized_spread"], row[field])
                for row in mechanisms
                if row["model_id"] == model_id and row.get(field) is not None
            ]
            by_model[model_id] = (
                _spearman([value[0] for value in values],
                          [value[1] for value in values])
                if len(values) >= 3 else None
            )
        interval = _spearman_confidence_interval(
            pairs, study["data"]["bootstrap_repetitions"],
            derived_analysis_seed(study, "exploratory-correlation", field),
        ) if len(pairs) >= 3 else None
        correlations.append({
            "relationship": f"latency_vs_{field}", "sample_count": len(pairs),
            "spearman": _spearman(
                [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            ) if len(pairs) >= 3 else None,
            "spearman_ci95": interval,
            "per_model_spearman": by_model,
            "status": "exploratory_not_preplanned",
        })
    severity_trends = []
    for tenancy, co_models, model_id in sorted({
        (row["tenancy"], row["co_models"], row["model_id"])
        for row in summaries
    }):
        clean = [row for row in summaries
                 if row["condition_id"] == "weather-clean"
                 and (row["tenancy"], row["co_models"], row["model_id"])
                 == (tenancy, co_models, model_id)]
        if not clean:
            continue
        for kind in ("rain", "snow"):
            rows = clean + [row for row in summaries
                            if row["effect"] == "weather"
                            and row["corruption_type"] == kind
                            and (row["tenancy"], row["co_models"],
                                 row["model_id"])
                            == (tenancy, co_models, model_id)]
            if len(rows) != 4:
                continue
            severity_trends.append({
                "tenancy": tenancy, "co_models": co_models,
                "model_id": model_id, "corruption_type": kind,
                "spearman_severity_normalized_spread": _spearman(
                    [row["severity"] for row in rows],
                    [row["normalized_latency_spread"] for row in rows],
                ),
                "status": "exploratory_candidate_requires_targeted_rerun",
            })
    return mechanisms, {
        "classification": "exploratory_not_preplanned",
        "correlations": correlations, "severity_trends": severity_trends,
        "module_rows_available": len(module_rows),
        "structure_rows_available": len(structure_rows),
        "claim_policy": (
            "Do not promote a candidate without a confidence interval, "
            "cross-run consistency, mechanism evidence, and targeted rerun."
        ),
    }


def _spearman_confidence_interval(pairs, repetitions, seed):
    """Bootstrap a cross-run Spearman association."""
    rng, values = random.Random(seed), []
    for _ in range(repetitions):
        sample = [pairs[rng.randrange(len(pairs))]
                  for _ in range(len(pairs))]
        value = _spearman(
            [pair[0] for pair in sample], [pair[1] for pair in sample]
        )
        if value is not None:
            values.append(value)
    return (
        [_percentile(values, 0.025), _percentile(values, 0.975)]
        if values else None
    )


def summarize_modules(module_rows):
    """Aggregate repeated module ranges without discarding samples."""
    groups = {}
    for row in module_rows:
        key = (
            row["run_id"], row["condition_id"], row["tenancy"],
            row["model_id"], row["module"],
        )
        groups.setdefault(key, []).append(row["duration_ms"])
    result = []
    for key, values in sorted(groups.items()):
        metrics = latency_metrics(values)
        result.append({
            "run_id": key[0], "condition_id": key[1],
            "tenancy": key[2], "model_id": key[3], "module": key[4],
            "sample_count": metrics["sample_count"],
            "median_ms": metrics["median_ms"],
            "iqr_ms": metrics["iqr_ms"],
            "p95_p5_ms": metrics["p95_p5_ms"],
            "normalized_latency_spread": metrics[
                "normalized_latency_spread"
            ],
        })
    return result


def case_study_analysis(study, summaries, frame_rows):
    """Summarize the preregistered trio using absolute latency width."""
    models = study["data"]["preselected_trio"]
    target_model = models[0]
    co_models = ";".join(models)
    index = {
        (row["condition_id"], row["tenancy"], row["co_models"],
         row["model_id"]): row for row in summaries
    }
    output = []
    for condition_id in study["data"]["triple_condition_ids"]:
        shared = index.get(
            (condition_id, "three_model", co_models, target_model)
        )
        single = index.get(
            (condition_id, "single", target_model, target_model)
        )
        if not shared or not single:
            continue
        shared_frames = [
            row for row in frame_rows
            if row["run_id"] == shared["run_id"]
            and row["model_id"] == target_model
        ]
        shared_values = [row["inference_e2e_ms"] for row in shared_frames]
        single_values = _frame_values(
            frame_rows, single["run_id"], target_model, "inference_e2e_ms"
        )
        waiting = [
            row["memcpy_adjacent_waiting_ns"] / 1e6
            for row in shared_frames
            if row.get("memcpy_adjacent_waiting_ns") is not None
        ]
        active = [
            row["gpu_kernel_active_ms"] for row in shared_frames
            if row.get("gpu_kernel_active_ms") is not None
        ]
        span = [
            row["kernel_span_ms"] for row in shared_frames
            if row.get("kernel_span_ms") is not None
        ]
        wait_pairs = [
            (row["inference_e2e_ms"],
             row["memcpy_adjacent_waiting_ns"] / 1e6)
            for row in shared_frames
            if row.get("memcpy_adjacent_waiting_ns") is not None
        ]
        record = {
            "condition_id": condition_id,
            "target_model": target_model,
            "co_models": co_models,
            "single_sample_count": len(single_values),
            "shared_sample_count": len(shared_values),
            "single_p95_p5_ms": single["p95_p5_ms"],
            "shared_p95_p5_ms": shared["p95_p5_ms"],
            "shared_minus_single_p95_p5_ms": (
                shared["p95_p5_ms"] - single["p95_p5_ms"]
            ),
            "shared_minus_single_ci99_ms": (
                absolute_spread_difference_confidence_interval(
                    shared_values, single_values,
                    CASE_STUDY_BOOTSTRAP_REPETITIONS,
                    derived_analysis_seed(
                        study, "case-study", condition_id, target_model
                    ),
                    confidence=0.99,
                )
            ),
            "shared_waiting_p95_p5_ms": (
                latency_metrics(waiting)["p95_p5_ms"] if waiting else None
            ),
            "shared_active_p95_p5_ms": (
                latency_metrics(active)["p95_p5_ms"] if active else None
            ),
            "shared_kernel_span_p95_p5_ms": (
                latency_metrics(span)["p95_p5_ms"] if span else None
            ),
            "latency_waiting_spearman": (
                _spearman(
                    [pair[0] for pair in wait_pairs],
                    [pair[1] for pair in wait_pairs],
                ) if len(wait_pairs) >= 3 else None
            ),
            "strict_finding_scope": "within_run_temporal_ci99",
            "weather_change_scope": "descriptive_no_confidence_interval",
        }
        for corunner_model in models[1:]:
            for field in ("kernel_span_ms", "gpu_kernel_active_ms"):
                values = _frame_values(
                    frame_rows, shared["run_id"], corunner_model, field
                )
                label = field.removesuffix("_ms")
                record[f"{corunner_model}_{label}_p95_p5_ms"] = (
                    latency_metrics(values)["p95_p5_ms"]
                    if values else None
                )
        output.append(record)
    clean = next(
        (row for row in output if row["condition_id"] == "weather-clean"),
        None,
    )
    if clean:
        for row in output:
            row["shared_change_from_clean_ms"] = (
                row["shared_p95_p5_ms"] - clean["shared_p95_p5_ms"]
            )
            row["single_change_from_clean_ms"] = (
                row["single_p95_p5_ms"] - clean["single_p95_p5_ms"]
            )
            row["waiting_change_from_clean_ms"] = (
                row["shared_waiting_p95_p5_ms"]
                - clean["shared_waiting_p95_p5_ms"]
            )
    return output


def summarize_primary_findings(study, summaries, contrasts):
    """Return concise pair/model rows for the paper table and figure."""
    modifiers = [row for row in contrasts
                 if row["contrast"] == "multi_tenant_weather_modifier"
                 and row["tenancy"] == "two_model"]
    output = []
    pairs = sorted({row["co_models"] for row in modifiers})
    for co_models in pairs:
        for model_id in co_models.split(";"):
            amplifications = [
                row["amplification_factor"] for row in summaries
                if row["tenancy"] == "two_model"
                and row["co_models"] == co_models
                and row["model_id"] == model_id
            ]
            rows = [row for row in modifiers
                    if row["co_models"] == co_models
                    and row["model_id"] == model_id]
            classifications = [
                "amplified" if row["ci95"][0] > 1 else
                "damped" if row["ci95"][1] < 1 else "inconclusive"
                for row in rows
            ]
            output.append({
                "co_models": co_models, "model_id": model_id,
                "model_group": study["data"]["models"][model_id]["group"],
                "median_contention_amplification": statistics.median(
                    amplifications
                ),
                "minimum_contention_amplification": min(amplifications),
                "maximum_contention_amplification": max(amplifications),
                "median_weather_modifier": statistics.median(
                    row["estimate"] for row in rows
                ),
                "weather_amplified": classifications.count("amplified"),
                "weather_damped": classifications.count("damped"),
                "weather_inconclusive": classifications.count(
                    "inconclusive"
                ),
            })
    return output


def derived_analysis_seed(study, *values):
    text = ":".join([str(study["data"]["random_seed"]), *map(str, values)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _write_report(path, summary):
    lines = [
        "# Input-variation validation", "",
        f"Planned runs: {summary['planned_runs']}",
        f"Valid runs: {summary['valid_runs']}",
        f"Complete: {summary['complete']}", "",
        "Outlier policy: " + summary["outlier_policy"], "",
        "## Invalid or missing runs", "",
    ]
    if summary["invalid_runs"]:
        for run_id, errors in summary["invalid_runs"].items():
            lines.append(f"- `{run_id}`: {'; '.join(errors)}")
    else:
        lines.append("None.")
    if summary["unexpected_runs"]:
        lines.extend(["", "## Unexpected runs", ""])
        lines.extend(f"- `{run_id}`" for run_id in summary["unexpected_runs"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table(path, rows):
    lines = [
        r"\begin{tabular}{llllrrrr}",
        r"Condition & Tenancy & Model & N & Median & IQR & P95--P5 & Amp. \\",
        r"\hline",
    ]
    for row in rows:
        amp = row.get("amplification_factor")
        lines.append(
            f"{row['condition_id']} & {row['tenancy']} & {row['model_id']} & "
            f"{row['sample_count']} & {row['median_ms']:.2f} & "
            f"{row['iqr_ms']:.2f} & {row['p95_p5_ms']:.2f} & "
            f"{'--' if amp is None else f'{amp:.2f}'} \\\\"
        )
    lines.append(r"\end{tabular}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_primary_table(path, rows):
    lines = [
        r"\begin{tabular}{lllrrr}",
        r"Co-runners & Measured model & Median AF & Amp. & Damp. & Null \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['co_models'].replace(';', ' + ')} & {row['model_id']} & "
            f"{row['median_contention_amplification']:.2f} & "
            f"{row['weather_amplified']} & {row['weather_damped']} & "
            f"{row['weather_inconclusive']} \\\\"
        )
    lines.append(r"\end{tabular}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_modifier_plot(path, rows):
    grouped = _group_modifier_rows(rows)
    width, margin, row_height = 900, 180, 34
    height = 105 + row_height * len(grouped)
    colors = {
        "weather_amplified": "#d95f02",
        "weather_damped": "#1b9e77",
        "weather_inconclusive": "#bdbdbd",
    }
    bar_width = width - margin - 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="10" y="20" font-size="14">Weather modifier outcomes '
        '(95% CI vs. 1; aggregated across co-runners)</text>',
        '<rect x="20" y="31" width="12" height="12" fill="#d95f02"/>',
        '<text x="38" y="41" font-size="10">amplified</text>',
        '<rect x="105" y="31" width="12" height="12" fill="#1b9e77"/>',
        '<text x="123" y="41" font-size="10">damped</text>',
        '<rect x="180" y="31" width="12" height="12" fill="#bdbdbd"/>',
        '<text x="198" y="41" font-size="10">inconclusive</text>',
    ]
    for index, row in enumerate(grouped):
        y, x = 60 + index * row_height, margin
        total = sum(row[field] for field in colors)
        parts.append(
            f'<text x="{margin - 8}" y="{y + 13}" text-anchor="end" '
            f'font-size="10">{row["model_id"]} (n={total})</text>'
        )
        for field in colors:
            count = row[field]
            segment = count / total * bar_width
            if count:
                parts.append(
                    f'<rect x="{x:.2f}" y="{y}" width="{segment:.2f}" '
                    f'height="16" fill="{colors[field]}"/>'
                )
            x += segment
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_modifier_plot_png(path, rows):
    from PIL import Image, ImageDraw

    grouped = _group_modifier_rows(rows)
    width, margin, row_height = 1400, 260, 60
    image = Image.new("RGB", (width, 150 + row_height * len(grouped)), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        "weather_amplified": "#d95f02",
        "weather_damped": "#1b9e77",
        "weather_inconclusive": "#bdbdbd",
    }
    bar_width = width - margin - 60
    draw.text(
        (20, 15),
        "Weather modifier outcomes (95% CI vs. 1; "
        "aggregated across co-runners)",
        fill="black",
    )
    legend_x = 20
    for field, label in (
        ("weather_amplified", "amplified"),
        ("weather_damped", "damped"),
        ("weather_inconclusive", "inconclusive"),
    ):
        draw.rectangle((legend_x, 45, legend_x + 22, 67), fill=colors[field])
        draw.text((legend_x + 30, 48), label, fill="black")
        legend_x += 150
    for index, row in enumerate(grouped):
        y, x = 100 + index * row_height, margin
        total = sum(row[field] for field in colors)
        draw.text((10, y + 7), f"{row['model_id']} (n={total})", fill="black")
        for field in colors:
            count = row[field]
            segment = count / total * bar_width
            if count:
                draw.rectangle(
                    (x, y, x + segment, y + 34), fill=colors[field]
                )
            x += segment
    image.save(path)


def _group_modifier_rows(rows):
    grouped = {}
    for row in rows:
        target = grouped.setdefault(row["model_id"], {
            "model_id": row["model_id"], "weather_amplified": 0,
            "weather_damped": 0, "weather_inconclusive": 0,
        })
        for field in (
            "weather_amplified", "weather_damped", "weather_inconclusive"
        ):
            target[field] += row[field]
    return list(grouped.values())


def _write_plot(path, rows):
    values = [
        row for row in rows
        if row.get("amplification_factor") is not None
    ]
    width, height, margin = 1000, max(300, 18 * len(values) + 80), 180
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    scale = (width - margin - 40) / max(
        2.0, max((row["amplification_factor"] for row in values), default=1)
    )
    parts.append(
        f'<line x1="{margin + scale}" y1="20" x2="{margin + scale}" '
        f'y2="{height - 30}" stroke="#666" stroke-dasharray="4 3"/>'
    )
    for index, row in enumerate(values):
        y = 35 + index * 18
        x = margin + row["amplification_factor"] * scale
        label = f"{row['condition_id']} / {row['model_id']}"
        parts.extend([
            f'<text x="{margin - 5}" y="{y + 4}" text-anchor="end" '
            f'font-size="9">{label}</text>',
            f'<circle cx="{x:.2f}" cy="{y}" r="3" fill="#2563eb"/>',
        ])
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(argv=None):
    """Run one phase of the repository-native input-variation workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare", "generate", "run", "validate", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("study")
        if name == "prepare":
            command.add_argument("--condition", action="append")
            command.add_argument("--workers", type=int)
        if name == "run":
            command.add_argument("--run-id", action="append")
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        study = load_study(args.study, check_paths=args.command != "plan")
        artifact_root = Path(args.artifact_root).resolve()
        study["paths"]["output_root"] = artifact_root
        study["paths"]["generated_config_root"] = (
            artifact_root / "generated_configs" / study["data"]["study_id"]
        )
        if args.command == "plan":
            result = {"study_sha256": study["sha256"],
                      "planned_runs": expected_runs(study)}
        elif args.command == "prepare":
            result = prepare_conditions(study, args.condition, args.workers)
        elif args.command == "generate":
            result = {name: str(path) for name, path in
                      materialize_configs(
                          study, output_root=artifact_root,
                          config_root=study["paths"][
                              "generated_config_root"],
                      ).items()}
        elif args.command == "run":
            result = run_campaign(
                study, artifact_root, args.run_id, args.dry_run)
        elif args.command == "validate":
            values, unexpected = discover_and_validate(study)
            result = {
                "valid": sum(value["valid"] for value in values.values()),
                "planned": len(values), "unexpected": unexpected,
                "errors": {
                    name: value["errors"] for name, value in values.items()
                    if not value["valid"]
                },
            }
        else:
            result = analyze(study)
    except (StudyError, ConfigError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
