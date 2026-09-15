"""Generate and run the fixed ten-cell three-model MPS experiment."""

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


MPS_CONFIGURATIONS = (
    (40, 40, 40), (60, 60, 60), (80, 80, 80), (90, 90, 90),
    (40, 60, 80), (40, 80, 60), (60, 40, 80), (60, 80, 40),
    (80, 40, 60), (80, 60, 40),
)
MODEL_ORDER = ("faster_rcnn", "deeplabv3plus", "detr")


def load_study(path):
    """Load and verify the exact ten configurations and model artifacts."""
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    data = yaml.safe_load(raw)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported three-model study schema")
    triples = tuple(
        tuple(values) for values in data["ordered_mps_configurations"]
    )
    if triples != MPS_CONFIGURATIONS or len(set(triples)) != 10:
        raise ValueError("ordered MPS configurations differ")
    if tuple(model["id"] for model in data["models"]) != MODEL_ORDER:
        raise ValueError("three-model order differs")
    if any(not 40 <= value <= 90 for triple in triples for value in triple):
        raise ValueError("MPS percentage is outside 40-90")
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


def _run_config(study, number, percentages):
    data = study["data"]
    models = deepcopy(data["models"])
    for model, percentage in zip(models, percentages):
        model["mps_percentage"] = percentage
    triple_id = "-".join(str(value) for value in percentages)
    return schema_v2_config({
        "schema_version": 1,
        "run": {
            "id": f"mps3-c{number:02d}-{triple_id}",
            "experiment": "three-model explicit non-default MPS leftover",
            "provenance": f"study={study['sha256']}",
            "timeout_seconds": 1800,
            "output_root": data["output_root"],
        },
        "ros": {
            "distribution": "humble",
            "middleware": "rmw_fastrtps_cpp",
            "domain_id": 160 + number,
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
    """Write exactly ten immutable ordered MPS configurations."""
    root = (Path(artifact_root).resolve() / "generated_configs" /
            "mps-three-model")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for number, percentages in enumerate(MPS_CONFIGURATIONS, 1):
        config = _run_config(study, number, percentages)
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
    if len(result) != 10:
        raise ValueError("three-model matrix must contain exactly ten runs")
    return result


def run_campaign(study, selected, artifact_root, dry_run=False):
    """Run the ten unique configurations sequentially with clock evidence."""
    results = {}
    for run_id, path in materialize(study, artifact_root).items():
        number = int(run_id.split("-", 2)[1][1:])
        if selected and number not in selected:
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
                    config.run_directory / "phase3_clock_control.json",
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


def main(argv=None):
    """Generate, validate, dry-run, or execute the fixed ten cells."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "validate", "run"))
    parser.add_argument("study")
    parser.add_argument("--configuration", type=int, action="append")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.configuration and any(
            not 1 <= number <= 10 for number in args.configuration):
        parser.error("--configuration must be between 1 and 10")
    study = load_study(args.study)
    if args.command == "generate":
        result = {key: str(value) for key, value in
                  materialize(study, args.artifact_root).items()}
    elif args.command == "validate":
        result = {
            key: load_run_config(str(value)).sha256
            for key, value in materialize(study, args.artifact_root).items()
        }
    else:
        result = run_campaign(
            study, set(args.configuration or ()), args.artifact_root,
            args.dry_run
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
