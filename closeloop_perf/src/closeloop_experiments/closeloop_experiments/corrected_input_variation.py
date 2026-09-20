"""Plan and execute the descriptive cross-modal input campaign."""

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

from .config import ConfigError, load_run_config, schema_v2_config
from .input_variation import _write_json
from .non_mps_memcpy_compute import StudyError
from closeloop_testbed.resource_control import GPUClockLock
from .runner import ExperimentRunner

MPS_ORDER = (False, True)
REQUIRED_FIELDS = {
    "schema_version",
    "study_id",
    "base_config",
    "output_root",
    "generated_config_root",
    "checkpoint_root",
    "metadata_path",
    "warmup_image",
    "warmup_lidar",
    "replicates",
    "mps_modes",
    "mps_percentage",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "nsys_version",
    "replay_cpu_affinity",
    "replay_cpu_thread_count",
    "models",
    "conditions",
    "baseline_map",
}
OPTIONAL_FIELDS = {"replay_passes", "pairs", "common_window_ns", "confirmation_repetitions"}
MODEL_FIELDS = {
    "group",
    "task",
    "modality",
    "mmlab_model",
    "architecture_profile",
    "model_config",
    "model_config_sha256",
    "checkpoint",
    "checkpoint_sha256",
}


def _resolve(source, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (source.parent /
                                                      path).resolve()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _archive_profile(path):
    """Losslessly gzip one large profile after verifying its contents."""
    path = Path(path)
    archived = Path(str(path) + ".gz")
    if not path.is_file():
        if not archived.is_file():
            raise StudyError(f"profile artifact is missing: {path}")
        return {
            "path": str(archived),
            "bytes": archived.stat().st_size,
            "sha256": _sha256(archived),
        }
    original_sha256 = _sha256(path)
    if not archived.is_file():
        temporary = Path(str(archived) + ".tmp")
        with path.open("rb") as source, temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename=path.name, mode="wb", fileobj=raw, mtime=0
            ) as destination:
                shutil.copyfileobj(source, destination)
        os.replace(temporary, archived)
    digest = hashlib.sha256()
    with gzip.open(archived, "rb") as restored:
        for block in iter(lambda: restored.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != original_sha256:
        raise StudyError(f"compressed profile verification failed: {path}")
    path.unlink()
    return {
        "path": str(archived),
        "bytes": archived.stat().st_size,
        "sha256": _sha256(archived),
        "restored_sha256": original_sha256,
    }


def archive_profiles(run_directory):
    """Archive the two regenerable Nsight files in one completed run."""
    run_directory = Path(run_directory)
    evidence = {
        name: _archive_profile(run_directory / name)
        for name in ("profile.nsys-rep", "profile.sqlite")
    }
    _write_json(run_directory / "profile_archive.json", evidence)
    return evidence


def load_study(path, check_paths=True):
    """Load and validate a cross-modal campaign contract."""
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise StudyError(
            f"cannot load descriptive input study: {exc}") from exc
    if not isinstance(data, dict):
        raise StudyError("study root must be an object")
    missing = REQUIRED_FIELDS - set(data)
    unknown = set(data) - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if missing or unknown:
        raise StudyError(
            f"study fields missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if data["schema_version"] != 1 or data["replicates"] != 1:
        raise StudyError("schema_version must be 1 and replicates must be 1")
    replay_passes = data.get("replay_passes", 1)
    if type(replay_passes) is not int or replay_passes < 1:
        raise StudyError("replay_passes must be a positive integer")
    mps_modes = data["mps_modes"]
    if (not isinstance(mps_modes, list) or not mps_modes
            or any(type(mode) is not bool for mode in mps_modes)
            or len(set(mps_modes)) != len(mps_modes)):
        raise StudyError("mps_modes must contain a unique configured subset")

    models = data["models"]
    if not isinstance(models, dict) or not models:
        raise StudyError("model inventory must not be empty")
    for name, model in models.items():
        absent = MODEL_FIELDS - set(model)
        if absent:
            raise StudyError(
                f"model {name} is missing fields {sorted(absent)}")
    lidar = tuple(
        sorted(name for name, model in models.items()
               if model["modality"] == "lidar"))
    camera = tuple(
        sorted(name for name, model in models.items()
               if model["modality"] == "image"))
    if not lidar or not camera or len(lidar) + len(camera) != len(models):
        raise StudyError(
            "every model must be image or LiDAR, with at least one of each")

    conditions = data["conditions"]
    if not isinstance(conditions, dict) or not conditions:
        raise StudyError("campaign must contain at least one condition")
    for condition_id, condition in conditions.items():
        condition_passes = condition.get("replay_passes", replay_passes)
        if type(condition_passes) is not int or condition_passes < 1:
            raise StudyError(
                f"condition {condition_id} replay_passes must be a "
                "positive integer")
    baseline_map = data["baseline_map"]
    adverse = {
        name
        for name, condition in conditions.items()
        if condition["corruption_type"] not in ("clean", "fixed")
    }
    if set(baseline_map) != adverse:
        raise StudyError(
            "every adverse condition must have one clear baseline")
    for condition_id, baseline_id in baseline_map.items():
        if baseline_id not in conditions:
            raise StudyError(
                f"unknown baseline {baseline_id} for {condition_id}")
        if conditions[baseline_id]["corruption_type"] != "clean":
            raise StudyError(f"baseline {baseline_id} is not clear")

    paths = {
        name: _resolve(source, data[name])
        for name in (
            "base_config",
            "output_root",
            "generated_config_root",
            "checkpoint_root",
            "metadata_path",
            "warmup_image",
            "warmup_lidar",
        )
    }
    if check_paths:
        for name in ("base_config", "metadata_path", "warmup_image",
                     "warmup_lidar"):
            if not paths[name].is_file():
                raise StudyError(f"{name} is missing: {paths[name]}")
        for name, model in models.items():
            checkpoint = paths["checkpoint_root"] / model["checkpoint"]
            for kind, artifact, expected in (
                ("config", Path(model["model_config"]),
                 model["model_config_sha256"]),
                ("checkpoint", checkpoint, model["checkpoint_sha256"]),
            ):
                if not artifact.is_file() or _sha256(artifact) != expected:
                    raise StudyError(
                        f"{name} {kind} is missing or has a different SHA-256")
        for condition_id, condition in conditions.items():
            manifest = Path(condition["dataset_manifest"])
            if not Path(condition["bag_directory"]).is_dir():
                raise StudyError(f"condition bag is missing: {condition_id}")
            if not manifest.is_file() or _sha256(
                    manifest) != condition["dataset_manifest_sha256"]:
                raise StudyError(
                    f"condition manifest is missing or differs: {condition_id}"
                )
    study = {
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": data,
        "paths": paths,
        "lidar": lidar,
        "camera": camera,
    }
    return study


def model_pairs(study):
    """Derive every LiDAR-camera pair from the model inventory."""
    return tuple((lidar, camera) for lidar in study["lidar"]
                 for camera in study["camera"])


def expected_runs(study):
    """Return the configured descriptive matrix keyed by unique run ID."""
    result = {}
    for mps_enabled in study["data"]["mps_modes"]:
        state = "1" if mps_enabled else "0"
        for condition_id in study["data"]["conditions"]:
            for lidar, camera in model_pairs(study):
                run_id = f"{state}-{condition_id}-{lidar}-{camera}"
                if run_id in result:
                    raise StudyError(f"duplicate planned run ID: {run_id}")
                result[run_id] = {
                    "condition_id": condition_id,
                    "lidar_model": lidar,
                    "camera_model": camera,
                    "models": [camera, lidar],
                    "mps_enabled": mps_enabled,
                    "replicate": 1,
                    "tenancy": "two_model",
                }
    return result


def expected_isolated_runs(study):
    """Return the 15 missing isolated 3DSSD mechanism runs."""
    return {
        f"isolated-{condition_id}-3dssd": {
            "condition_id": condition_id,
            "models": ["3dssd"],
            "mps_enabled": False,
            "replicate": 1,
            "tenancy": "single",
        }
        for condition_id in study["data"]["conditions"]
    }


def _model_config(study, name, index, mps_enabled):
    source = study["data"]["models"][name]
    model = {
        key: copy.deepcopy(value)
        for key, value in source.items() if key != "group"
    }
    model.update({
        "id":
        name,
        "node_name":
        "model_" + name.replace("-", "_"),
        "checkpoint":
        str(study["paths"]["checkpoint_root"] / source["checkpoint"]),
        "input_topic":
        "/camera/front" if source["modality"] == "image" else "/lidar/top",
        "input_message_type":
        "compressed_image" if source["modality"] == "image" else "pointcloud2",
        "input_queue_depth":
        1,
        "qos":
        "best_effort",
        "warmup_input":
        str(study["paths"]["warmup_image" if source["modality"] ==
                           "image" else "warmup_lidar"]),
        "warmup_count":
        5,
        "launch_offset_seconds":
        index,
        "module_annotation_depth":
        0,
        "quiet_period_seconds":
        0.5,
        "cpu_affinity":
        list(range(index * 6, (index + 1) * 6)),
        "cpu_thread_count":
        3,
    })
    if mps_enabled:
        model["mps_percentage"] = study["data"]["mps_percentage"]
    return model


def _run_config(study, run_id, planned, ordinal):
    data = study["data"]
    condition = data["conditions"][planned["condition_id"]]
    config = copy.deepcopy(
        yaml.safe_load(study["paths"]["base_config"].read_bytes()))
    config["run"].update({
        "id": run_id,
        "experiment": "descriptive cross-modal input-data variation",
        "provenance": f"study={study['sha256']}",
        "output_root": str(study["paths"]["output_root"]),
    })
    config["ros"]["domain_id"] = 70 + ordinal % 100
    replay = {
        "scene_token":
        condition["scene_token"],
        "metadata_path":
        str(study["paths"]["metadata_path"]),
        "bag_directory":
        condition["bag_directory"],
        "cpu_affinity":
        data["replay_cpu_affinity"],
        "cpu_thread_count":
        data["replay_cpu_thread_count"],
        "playback_mode":
        "full",
    }
    replay["repeat_count"] = condition.get(
        "replay_passes", data.get("replay_passes", 1))
    config["replay"].update(replay)
    config["replay"].pop("duration_seconds", None)
    config["gpu"].update({
        "mps_enabled": planned["mps_enabled"],
        "graphics_clock_mhz": data["graphics_clock_mhz"],
        "memory_clock_mhz": data["memory_clock_mhz"],
    })
    config["models"] = [
        _model_config(study, name, index, planned["mps_enabled"])
        for index, name in enumerate(planned["models"])
    ]
    config["recording"]["nsys"].update({
        "version": data["nsys_version"],
        "gpu_context_switch": not planned["mps_enabled"],
    })
    corruption = {
        "type": condition["corruption_type"],
        "severity": condition["severity"],
        "seed": condition["seed"],
        "dataset_manifest": condition["dataset_manifest"],
        "dataset_manifest_sha256": condition["dataset_manifest_sha256"],
    }
    for field in ("full_duration", "rain_rate_mm_per_hour"):
        if field in condition:
            corruption[field] = condition[field]
    config["input_variation"] = {
        "condition_id": planned["condition_id"],
        "effect": condition["effect"],
        "source_scene_name": condition["scene_name"],
        "scene_factors": condition["scene_factors"],
        "corruption": corruption,
        "replicate": 1,
        "tenancy": "two_model",
    }
    return config


def _can_reuse_completed_config(study, path, content):
    """Allow only successful, byte-identical runs with harmless drift."""
    run_directory = study["paths"]["output_root"] / path.stem
    manifest = run_directory / "run_manifest.json"
    executed = run_directory / "config.yaml"
    if not manifest.is_file() or not executed.is_file():
        return False
    try:
        if json.loads(manifest.read_text(encoding="utf-8")).get(
                "state") != "success":
            return False
        prior_bytes = path.read_bytes()
        if executed.read_bytes() != prior_bytes:
            return False
        prior = yaml.safe_load(prior_bytes)
        expanded = yaml.safe_load(content)
        for config in (prior, expanded):
            config["run"].pop("provenance")
            config["ros"].pop("domain_id")
        return prior == expanded
    except (OSError, TypeError, json.JSONDecodeError, KeyError,
            yaml.YAMLError):
        return False


def materialize(study, artifact_root):
    """Write the immutable configured campaign matrix."""
    root = (Path(artifact_root).resolve() / "generated_configs" /
            study["data"]["study_id"])
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for ordinal, (run_id, planned) in enumerate(expected_runs(study).items()):
        config = _run_config(study, run_id, planned, ordinal)
        path = root / f"{run_id}.yaml"
        content = yaml.safe_dump(
            schema_v2_config(config), sort_keys=False).encode()
        if (path.exists() and path.read_bytes() != content
                and not _can_reuse_completed_config(study, path, content)):
            raise StudyError(f"immutable generated config differs: {path}")
        if not path.exists():
            path.write_bytes(content)
            path.chmod(0o444)
        try:
            load_run_config(str(path), check_paths=False)
        except ConfigError as exc:
            raise StudyError(
                f"generated config {run_id} is invalid: {exc}") from exc
        result[run_id] = path
    return result


def materialize_isolated(study, artifact_root):
    """Write configs for isolated 3DSSD evidence missing from Section 4.1."""
    root = (Path(artifact_root).resolve() / "generated_configs" /
            study["data"]["study_id"] / "isolated")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for ordinal, (run_id, planned) in enumerate(
            expected_isolated_runs(study).items()):
        config = _run_config(study, run_id, planned, ordinal)
        config["run"]["experiment"] = "isolated 3DSSD input mechanism"
        config["models"] = [_model_config(study, "3dssd", 0, False)]
        config["input_variation"]["tenancy"] = "single"
        path = root / f"{run_id}.yaml"
        content = yaml.safe_dump(
            schema_v2_config(config), sort_keys=False).encode()
        if path.exists() and path.read_bytes() != content:
            raise StudyError(f"immutable generated config differs: {path}")
        if not path.exists():
            path.write_bytes(content)
            path.chmod(0o444)
        try:
            load_run_config(str(path), check_paths=False)
        except ConfigError as exc:
            raise StudyError(
                f"generated config {run_id} is invalid: {exc}") from exc
        result[run_id] = path
    return result


def filtered_configs(study,
                     configs,
                     run_ids=(),
                     conditions=(),
                     pairs=(),
                     mps_modes=()):
    """Filter planned configs by run ID, condition, pair, and MPS mode."""
    planned = expected_runs(study)
    run_ids, conditions, pairs, mps_modes = map(
        set, (run_ids, conditions, pairs, mps_modes))
    unknown = run_ids - set(planned)
    if unknown:
        raise StudyError(f"unknown run IDs: {sorted(unknown)}")
    result = {}
    for run_id, path in configs.items():
        item = planned[run_id]
        pair = f"{item['lidar_model']}+{item['camera_model']}"
        mode = "on" if item["mps_enabled"] else "off"
        if run_ids and run_id not in run_ids:
            continue
        if conditions and item["condition_id"] not in conditions:
            continue
        if pairs and pair not in pairs:
            continue
        if mps_modes and mode not in mps_modes:
            continue
        result[run_id] = path
    return result


def validate_configs(configs):
    """Validate every selected config after study-level artifact checks."""
    return {
        run_id: load_run_config(str(path), check_paths=False).sha256
        for run_id, path in configs.items()
    }


def run_campaign(study, configs, artifact_root, dry_run=False):
    """Run missing configurations sequentially with fixed GPU clocks."""
    results = {}
    for run_id, path in configs.items():
        config = load_run_config(str(path), artifact_root=str(artifact_root))
        manifest_path = config.run_directory / "run_manifest.json"
        if manifest_path.is_file():
            state = json.loads(
                manifest_path.read_text(encoding="utf-8")).get("state")
            if state == "success":
                executed = config.run_directory / "config.yaml"
                if (not executed.is_file()
                        or executed.read_bytes() != path.read_bytes()):
                    raise StudyError(
                        f"executed config differs from generated config: "
                        f"{run_id}")
                archive_profiles(config.run_directory)
            results[run_id] = f"existing-{state}"
            continue
        if dry_run:
            ExperimentRunner(config).run(dry_run=True)
            results[run_id] = "dry-run"
            continue
        gpu = config.data["gpu"]
        control = GPUClockLock(gpu["index"], gpu["graphics_clock_mhz"],
                               gpu["memory_clock_mhz"])
        try:
            try:
                with control:
                    ExperimentRunner(config).run()
            finally:
                if config.run_directory.is_dir():
                    _write_json(
                        config.run_directory /
                        "input_variation_clock_control.json",
                        {
                            "requested_graphics_clock_mhz":
                            gpu["graphics_clock_mhz"],
                            "requested_memory_clock_mhz":
                            gpu["memory_clock_mhz"],
                            "control":
                            control.evidence,
                            "clocks_restored":
                            len(control.evidence.get(
                                "reset_commands", [])) == 2
                            and all(
                                item.get("returncode") == 0
                                for item in control.evidence.get(
                                    "reset_commands", [])),
                        },
                    )
        except Exception:
            if config.run_directory.is_dir():
                failed_root = study["paths"]["output_root"] / "failed_attempts"
                failed_root.mkdir(parents=True, exist_ok=True)
                attempt = 1
                destination = failed_root / f"{run_id}-attempt-{attempt}"
                while destination.exists():
                    attempt += 1
                    destination = failed_root / f"{run_id}-attempt-{attempt}"
                shutil.move(str(config.run_directory), destination)
            raise
        archive_profiles(config.run_directory)
        results[run_id] = "success"
        print(json.dumps({"run_id": run_id, "state": "success"}))
    return results


def main(argv=None):
    """Plan, generate, validate, dry-run, or execute the campaign."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    # The existing public campaign owns both descriptive and controlled studies.
    for value in arguments:
        if value.endswith((".yaml", ".yml")) and Path(value).is_file():
            if yaml.safe_load(Path(value).read_text()).get("study_id") == "input2-crossed":
                from .input_crossed import main as crossed_main
                return crossed_main(arguments)
            break
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=("plan", "generate", "validate", "run",
                                 "generate-isolated", "validate-isolated",
                                 "run-isolated"))
    parser.add_argument("study")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--mps-mode",
                        action="append",
                        choices=("off", "on"),
                        default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args(argv)
    study = load_study(args.study)
    study["paths"]["output_root"] = (
        Path(args.artifact_root).resolve() / "runs"
    )
    if args.command == "plan":
        result = expected_runs(study)
    else:
        isolated = args.command.endswith("-isolated")
        if isolated:
            configs = {
                run_id: path
                for run_id, path in materialize_isolated(
                    study, args.artifact_root).items()
                if (not args.run_id or run_id in args.run_id)
                and (not args.condition or expected_isolated_runs(study)[
                    run_id]["condition_id"] in args.condition)
            }
        else:
            configs = filtered_configs(
                study,
                materialize(study, args.artifact_root),
                args.run_id,
                args.condition,
                args.pair,
                args.mps_mode,
            )
        if args.command in ("generate", "generate-isolated"):
            result = {key: str(value) for key, value in configs.items()}
        elif args.command in ("validate", "validate-isolated"):
            result = validate_configs(configs)
        else:
            result = run_campaign(
                study, configs, args.artifact_root, args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
