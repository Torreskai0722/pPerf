"""Select and run the fixed two-model GPU contention experiment."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from .config import load_run_config, schema_v2_config
from .input_variation import _write_json
from closeloop_testbed.resource_control import GPUClockLock
from .runner import ExperimentRunner


MODEL_ORDER = ("faster_rcnn", "deeplabv3plus")
CONDITION_ORDER = (
    "discovery", "intervention", "reversal", "negative_control",
)


def _passive_domain_id(replicate):
    value = 224 + replicate
    return value if value <= 232 else value - 200


def load_study(path):
    """Load and verify the pinned pair, artifacts, and intervention matrix."""
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    data = yaml.safe_load(raw)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported Phase 4 study schema")
    if tuple(model["id"] for model in data["models"]) != MODEL_ORDER:
        raise ValueError("Phase 4 model order differs")
    if tuple(data["conditions"]) != CONDITION_ORDER:
        raise ValueError("Phase 4 condition order differs")
    if data["conditions"]["discovery"] != [80, 80]:
        raise ValueError("Phase 4 selected pair differs")
    if data["conditions"]["reversal"] != [80, 80]:
        raise ValueError("Phase 4 reversal differs")
    for model in data["models"]:
        for field in ("model_config", "checkpoint"):
            observed = hashlib.sha256(
                Path(model[field]).read_bytes()
            ).hexdigest()
            if observed != model[f"{field}_sha256"]:
                raise ValueError(f"{model['id']} {field} hash differs")
    return schema_v2_config({
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": data,
    })


def _run_config(study, condition, replicate, ordinal):
    data = study["data"]
    percentages = data["conditions"][condition]
    models = deepcopy(data["models"])
    for model, percentage in zip(models, percentages):
        model["mps_percentage"] = percentage
    return schema_v2_config({
        "schema_version": 1,
        "run": {
            "id": f"mps2cause-{condition}-r{replicate}",
            "experiment": "two-model GPU contention cause localization",
            "provenance": f"study={study['sha256']}",
            "timeout_seconds": 1800,
            "output_root": data["output_root"],
        },
        "ros": {
            "distribution": "humble",
            "middleware": "rmw_fastrtps_cpp",
            "domain_id": 180 + ordinal,
            "launch_package": "closeloop_testbed",
            "launch_file": "testbed.launch.py",
        },
        "replay": {
            "scene_token": data["scene_tokens"][0],
            "scene_tokens": data["scene_tokens"],
            "metadata_path": data["metadata_path"],
            "bag_directory": data["bag_directory"],
            "rate": 1.0,
            "playback_mode": "full",
            "topics": ["/CAM_FRONT/image_rect_compressed"],
            "remappings": {
                "/CAM_FRONT/image_rect_compressed": "/camera/front"
            },
            "readiness_timeout_seconds": 360,
            "player_startup_timeout_seconds": 15,
            "completion_timeout_seconds": 180,
            "communication_profile": True,
            "cpu_affinity": data["replay_cpu_affinity"],
            "cpu_thread_count": data["replay_cpu_thread_count"],
        },
        "gpu": deepcopy(data["gpu"]),
        "models": models,
        "nsys": {
            "version": data["nsys_version"],
            "trace": ["cuda", "nvtx", "cudnn"],
            "sample": "none",
            "backtrace": "none",
            "cpu_context_switch": False,
            "gpu_context_switch": False,
        },
        "instrumentation": {
            "profiler": {"id": "model_level1_v1"},
            "analyzer": {"id": "none_v1"},
        },
    })


def materialize(study, artifact_root):
    """Write the fixed four-condition, three-repetition run matrix."""
    root = (Path(artifact_root).resolve() / "generated_configs" /
            "mps-two-model")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    ordinal = 0
    for condition in CONDITION_ORDER:
        for replicate in range(1, study["data"]["replicates"] + 1):
            ordinal += 1
            config = _run_config(study, condition, replicate, ordinal)
            path = root / f"{config['run']['id']}.yaml"
            content = yaml.safe_dump(
                schema_v2_config(config), sort_keys=False).encode()
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f"immutable config differs: {path}")
            if not path.exists():
                path.write_bytes(content)
                path.chmod(0o444)
            load_run_config(str(path))
            result[config["run"]["id"]] = path
    return result


def run_campaign(study, selected, artifact_root, dry_run=False):
    """Run selected conditions sequentially with GPU clock evidence."""
    results = {}
    for run_id, path in materialize(study, artifact_root).items():
        condition = run_id[len("mps2cause-"):].rsplit("-r", 1)[0]
        if selected and condition not in selected:
            continue
        config = load_run_config(str(path), artifact_root=str(artifact_root))
        manifest_path = config.run_directory / "run_manifest.json"
        if manifest_path.is_file():
            state = json.loads(manifest_path.read_text())["state"]
            if state != "success":
                raise ValueError(f"existing {run_id} state is {state}")
            results[run_id] = "existing-success"
            continue
        if dry_run:
            ExperimentRunner(config).run(dry_run=True)
            results[run_id] = "dry-run"
            continue
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
                    config.run_directory / "phase4_clock_control.json",
                    {
                        "control": control.evidence,
                        "clocks_restored": len(
                            control.evidence.get("reset_commands", [])
                        ) == 2 and all(
                            item.get("returncode") == 0
                            for item in control.evidence.get(
                                "reset_commands", []
                            )
                        ),
                    },
                )
        results[run_id] = "success"
        print(json.dumps({"run_id": run_id, "state": "success"}))
    return results


def run_config_with_clocks(path, artifact_root, dry_run=False):
    """Run one diagnostic config with the campaign's clock controls."""
    config = load_run_config(
        str(Path(path).expanduser().resolve()),
        artifact_root=str(artifact_root),
    )
    if dry_run:
        ExperimentRunner(config).run(dry_run=True)
        return {config.data["run"]["id"]: "dry-run"}
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
                config.run_directory / "phase4_clock_control.json",
                {"control": control.evidence},
            )
    return {config.data["run"]["id"]: "success"}


def materialize_ncu(study, target_path, artifact_root):
    """Write two immutable one-kernel Nsight Compute diagnostics."""
    targets = json.loads(Path(target_path).read_text(encoding="utf-8"))
    selected = {
        "faster_rcnn": targets["affected_victim_kernel"]["kernel_name"],
        "deeplabv3plus": targets["affected_aggressor_kernel"][
            "kernel_name"
        ],
    }
    root = (Path(artifact_root).resolve() / "generated_configs" /
            "mps-two-model" / "diagnostic")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    sections = [
        "LaunchStats", "Occupancy", "SchedulerStats", "WarpStateStats",
        "SpeedOfLight", "MemoryWorkloadAnalysis",
        "MemoryWorkloadAnalysis_Tables",
    ]
    for ordinal, (model, kernel) in enumerate(selected.items(), 215):
        config = _run_config(study, "discovery", 1, ordinal)
        run_id = f"mps2cause-ncu-{model}-v5"
        config["run"].update({
            "id": run_id,
            "experiment": "two-model targeted Nsight Compute diagnostic",
            "provenance": "target_episodes.json; invasive counter pass",
        })
        config["ros"]["domain_id"] = ordinal
        config["replay"]["readiness_timeout_seconds"] = 900
        config["recording"] = {
            "level": "level2b",
            "scopes": ["model"],
            "ncu": {
                "version": "2025.2.1.0",
                "kernel_name": kernel,
                "launch_count": 1,
                "sections": sections,
            },
        }
        path = root / f"{run_id}.yaml"
        content = yaml.safe_dump(
            schema_v2_config(config), sort_keys=False).encode()
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"immutable config differs: {path}")
        if not path.exists():
            path.write_bytes(content)
            path.chmod(0o444)
        load_run_config(str(path))
        result[run_id] = str(path)
    return result


def materialize_passive_cta(
        study, plan_path, replicate, artifact_root):
    """Write one immutable natural-schedule CTA window configuration."""
    plan_path = Path(plan_path).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("mode") != "passive"
        or not plan.get("selection_locked_before_cta")
        or plan.get("launch_alignment")
        or plan.get("launch_manipulation")
    ):
        raise ValueError("passive CTA plan violates the observation gate")
    condition = plan.get("condition", "discovery")
    if condition not in CONDITION_ORDER:
        raise ValueError("passive CTA plan condition is invalid")
    config = _run_config(study, condition, 1, 224 + replicate)
    run_id = f"mps2cause-cta-passive-r{replicate}"
    config["run"].update({
        "id": run_id,
        "experiment": "two-model passive all-client CTA windows",
        "provenance": (
            f"passive_capture_plan_sha256="
            f"{hashlib.sha256(plan_path.read_bytes()).hexdigest()}"
        ),
    })
    config["ros"]["domain_id"] = _passive_domain_id(replicate)
    config["recording"] = {
        "level": "level1", "scopes": ["model"], "capsule": {},
    }
    tracker = (
        "/mmdetection3d_ros2/closeloop_perf/install/"
        "closeloop_profiler/lib/libpperf_nvbit_cta_tracker.so"
    )
    for model in config["models"]:
        client = plan["clients"][model["id"]]
        model["nvbit_cta_profile"] = {
            "mode": "passive",
            "tracker_library": tracker,
            "record_capacity": client["record_capacity"],
            "targets": [
                {
                    "label": window["label"],
                    "scene_index": window["scene_index"],
                    "message_order": window["message_order"],
                    "sequence_start": window["sequence_start"],
                    "sequence_end": window["sequence_end"],
                }
                for window in client["windows"]
            ],
        }
    root = (Path(artifact_root).resolve() / "generated_configs" /
            "mps-two-model" / "diagnostic")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.yaml"
    content = yaml.safe_dump(
        schema_v2_config(config), sort_keys=False).encode()
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"immutable config differs: {path}")
    if not path.exists():
        path.write_bytes(content)
        path.chmod(0o444)
    load_run_config(str(path))
    return {run_id: str(path)}


def materialize_postpassive(study, artifact_root):
    """Write one low-overhead run for intervention, reversal, and control."""
    root = (Path(artifact_root).resolve() / "generated_configs" /
            "mps-two-model" / "diagnostic")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for replicate, condition in enumerate(
        ("intervention", "reversal", "negative_control"), 29
    ):
        config = _run_config(study, condition, 1, 224 + replicate)
        run_id = f"mps2cause-postpassive-{condition}-r1"
        config["run"].update({
            "id": run_id,
            "experiment": "postpassive low-overhead GPU validation",
            "provenance": "postpassive_intervention_plan_v1",
        })
        config["ros"]["domain_id"] = _passive_domain_id(replicate)
        path = root / f"{run_id}.yaml"
        content = yaml.safe_dump(
            schema_v2_config(config), sort_keys=False).encode()
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"immutable config differs: {path}")
        if not path.exists():
            path.write_bytes(content)
            path.chmod(0o444)
        load_run_config(str(path))
        result[run_id] = str(path)
    return result


def main(argv=None):
    """Generate, validate, dry-run, or execute Phase 4."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "generate", "validate", "run", "run-config",
            "ncu-generate", "cta-passive-generate",
            "postpassive-generate",
        ),
    )
    parser.add_argument("study")
    parser.add_argument("--condition", action="append",
                        choices=CONDITION_ORDER)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-episodes")
    parser.add_argument("--passive-plan")
    parser.add_argument("--replicate", type=int, choices=range(1, 31),
                        default=1)
    args = parser.parse_args(argv)
    if args.command == "run-config":
        result = run_config_with_clocks(
            args.study, args.artifact_root, args.dry_run)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    study = load_study(args.study)
    if args.command == "postpassive-generate":
        result = materialize_postpassive(study, args.artifact_root)
    elif args.command == "cta-passive-generate":
        if not args.passive_plan:
            parser.error("cta-passive-generate requires --passive-plan")
        result = materialize_passive_cta(
            study, args.passive_plan, args.replicate, args.artifact_root
        )
    elif args.command == "ncu-generate":
        if not args.target_episodes:
            parser.error("ncu-generate requires --target-episodes")
        result = materialize_ncu(
            study, args.target_episodes, args.artifact_root)
    elif args.command == "generate":
        result = {key: str(value) for key, value in
                  materialize(study, args.artifact_root).items()}
    elif args.command == "validate":
        result = {
            key: load_run_config(str(value)).sha256
            for key, value in materialize(study, args.artifact_root).items()
        }
    else:
        result = run_campaign(
            study, set(args.condition or ()), args.artifact_root, args.dry_run
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
