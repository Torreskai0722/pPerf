"""Generate and run the fixed ROS 2 communication-variation study."""

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


CONDITIONS = ("baseline", "cpu1_intervention", "reversal")


def condition_from_run_id(run_id):
    """Remove only the final replicate suffix from a run ID."""
    return run_id.rsplit("-r", 1)[0].removeprefix("comm-")


def load_study(path):
    """Load and validate the fixed communication study contract."""
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    data = yaml.safe_load(raw)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported communication study schema")
    if tuple(data["conditions"]) != CONDITIONS:
        raise ValueError("communication conditions differ from fixed order")
    if len(data["scene_tokens"]) != 2:
        raise ValueError("communication study requires exactly two scenes")
    for field in ("model_config", "checkpoint"):
        observed = hashlib.sha256(
            Path(data["model"][field]).read_bytes()
        ).hexdigest()
        if observed != data["model"][f"{field}_sha256"]:
            raise ValueError(f"model {field} hash differs")
    return schema_v2_config({
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": data,
    })


def _run_config(study, condition_id, replicate, ordinal):
    data = study["data"]
    condition = data["conditions"][condition_id]
    model = deepcopy(data["model"])
    model["cpu_thread_count"] = condition["model_cpu_thread_count"]
    return schema_v2_config({
        "schema_version": 1,
        "run": {
            "id": f"comm-{condition_id}-r{replicate}",
            "experiment": "ROS 2 relay-to-model communication variation",
            "provenance": f"study={study['sha256']}",
            "timeout_seconds": 1200,
            "output_root": data["output_root"],
        },
        "ros": {
            "distribution": "humble",
            "middleware": "rmw_fastrtps_cpp",
            "domain_id": 130 + ordinal,
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
            "readiness_timeout_seconds": 240,
            "player_startup_timeout_seconds": 15,
            "completion_timeout_seconds": 120,
            "communication_profile": True,
            "cpu_affinity": condition["replay_cpu_affinity"],
            "cpu_thread_count": condition["replay_cpu_thread_count"],
        },
        "gpu": deepcopy(data["gpu"]),
        "models": [model],
        "nsys": {
            "version": data["nsys_version"],
            "trace": ["cuda", "nvtx", "cudnn"],
            "sample": "none",
            "backtrace": "none",
            "cpu_context_switch": False,
            "gpu_context_switch": True,
        },
        "instrumentation": {
            "profiler": {"id": "model_level1_v1"},
            "analyzer": {"id": "none_v1"},
        },
    })


def materialize(study, artifact_root):
    """Write the nine immutable discovery/intervention configurations."""
    root = (Path(artifact_root).resolve() / "generated_configs" /
            "communication")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    ordinal = 0
    for condition_id in CONDITIONS:
        for replicate in range(1, study["data"]["replicates"] + 1):
            config = _run_config(study, condition_id, replicate, ordinal)
            ordinal += 1
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
    """Run selected conditions sequentially with clock-reset evidence."""
    results = {}
    for run_id, path in materialize(study, artifact_root).items():
        condition = condition_from_run_id(run_id)
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
                    config.run_directory / "phase2_clock_control.json",
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
    """Generate, validate, dry-run, or execute communication cells."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "validate", "run"))
    parser.add_argument("study")
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
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
            study, set(args.condition or ()), args.artifact_root, args.dry_run
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
