"""Versioned run configuration loading and semantic validation."""

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import yaml

from closeloop_profiler.architecture_profiles import (
    ArchitectureProfileError,
    get_architecture_profile,
)
from closeloop_profiler.components import recording_config
from .mps import MPSError, validate_mps_pipe_directory


class ConfigError(ValueError):
    """Raised when a run configuration is invalid."""


PATH_FIELDS = (
    ("replay", "metadata_path"),
    ("replay", "bag_directory"),
)


@dataclass(frozen=True)
class RunConfig:
    """Validated configuration plus immutable source metadata."""

    data: Dict[str, Any]
    source: Path
    original_bytes: bytes
    sha256: str
    artifact_root: Optional[Path] = None

    @property
    def run_directory(self) -> Path:
        """Return the configured immutable run output directory."""
        if self.artifact_root is None:
            raise ConfigError("an explicit artifact root is required")
        return self.artifact_root / "runs" / self.data["run"]["id"]

    @property
    def models(self) -> List[Dict[str, Any]]:
        """Return configured model records."""
        return self.data["models"]

    def with_artifact_root(self, root: str) -> "RunConfig":
        """Return this immutable config bound to an explicit artifact root."""
        configured = RunConfig(
            self.data, self.source, self.original_bytes, self.sha256,
            Path(root).expanduser().resolve(),
        )
        if self.data["gpu"]["mps_enabled"]:
            try:
                validate_mps_pipe_directory(configured.run_directory / "p")
            except MPSError as exc:
                raise ConfigError(str(exc)) from exc
        return configured


def schema_path() -> Path:
    """Return the bundled schema path."""
    return Path(__file__).with_name("schema") / "run_config.schema.json"


def schema_v2_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a generated run config in the sole writable schema version."""
    if not isinstance(data, dict) or not {"run", "models", "gpu"} <= data.keys():
        return data
    result = copy.deepcopy(data)
    instrumentation = result.pop("instrumentation", {})
    nsys = result.pop("nsys", None)
    ncu = result.pop("ncu", None)
    recording = result.get("recording")
    if recording is None:
        profiler = instrumentation.get("profiler", {})
        profiler_id = profiler.get("id", "model_level1_v1")
        scopes = ["model"]
        if result.get("input_variation") or result.get("replay", {}).get(
                "communication_profile"):
            scopes.append("input")
        if profiler_id == "preprocess_contention_v1":
            scopes.append("preprocessing")
        if ncu:
            recording = {"level": "level2b", "scopes": scopes, "ncu": ncu}
        else:
            nsys = dict(nsys or {})
            nsys.pop("enabled", None)
            recording = {
                "level": (
                    "level2a" if nsys.get("gpu_metrics_devices") else "level1"
                ),
                "scopes": scopes,
                "nsys": nsys,
            }
        options = dict(profiler.get("options", {}))
        if profiler_id == "preprocess_contention_v1":
            recording["preprocessing"] = {
                key: value for key, value in options.items()
                if key in {
                    "scheduler_backend", "off_cpu_stacks", "thread_lifecycle",
                    "data_preprocessor_methods",
                }
            }
        elif profiler_id == "gpu_forensics_v1":
            options.pop("mode", None)
            recording["capsule"] = options
        result["recording"] = recording
    result["schema_version"] = 2
    result["run"].pop("output_root", None)
    return result


def _semantic_validate(data: Dict[str, Any]) -> None:
    recording = recording_config(data)
    nsys = recording.get("nsys")
    ncu = recording.get("ncu")
    required_traces = {"cuda", "nvtx", "cudnn"}
    if nsys and set(nsys["trace"]) != required_traces:
        raise ConfigError("Level-1 trace must contain cuda, nvtx, and cudnn")
    playback_mode = data["replay"].get("playback_mode", "full")
    controlled = data["replay"].get("controlled_bag_manifest")
    if bool(controlled) != bool(data["replay"].get("controlled_bag_manifest_sha256")):
        raise ConfigError("controlled replay requires both manifest path and SHA-256")
    if controlled and (data["replay"].get("repeat_count", 1) != 1
                       or data["replay"]["rate"] != 1.0
                       or playback_mode != "full"
                       or data["replay"].get("scene_tokens")):
        raise ConfigError("controlled replay requires one full pass at rate 1 without scene discovery")
    scene_tokens = data["replay"].get("scene_tokens")
    if scene_tokens and scene_tokens[0] != data["replay"]["scene_token"]:
        raise ConfigError(
            "replay.scene_token must equal the first ordered scene token"
        )
    has_duration = "duration_seconds" in data["replay"]
    if playback_mode == "partial" and not has_duration:
        raise ConfigError(
            "replay.duration_seconds is required for partial playback"
        )
    if playback_mode == "full" and has_duration:
        raise ConfigError(
            "replay.duration_seconds is only valid for partial playback"
        )
    ids = [model["id"] for model in data["models"]]
    if len(ids) != len(set(ids)):
        raise ConfigError("model IDs must be unique")
    nodes = [model["node_name"] for model in data["models"]]
    if len(nodes) != len(set(nodes)):
        raise ConfigError("model node names must be unique")
    synthetic_fma = data.get("synthetic_fma")
    if synthetic_fma and synthetic_fma["client_id"] in ids:
        raise ConfigError("synthetic_fma client_id must differ from model IDs")
    mps = data["gpu"]["mps_enabled"]
    cpu_owners = {}
    resource_settings = [("replay", data["replay"])]
    resource_settings.extend(
        (f"model {model['id']}", model) for model in data["models"]
    )
    for owner, settings in resource_settings:
        affinity = settings.get("cpu_affinity")
        thread_count = settings.get("cpu_thread_count")
        configured_counts = {
            "cpu_thread_count": thread_count,
            **settings.get("library_thread_counts", {}),
        }
        if affinity is not None:
            for name, count in configured_counts.items():
                if count is not None and count > len(affinity):
                    raise ConfigError(
                        f"{owner} {name} exceeds its logical CPU affinity"
                    )
        if affinity is None:
            continue
        for cpu_id in affinity:
            previous = cpu_owners.get(cpu_id)
            if previous is not None:
                raise ConfigError(
                    f"CPU {cpu_id} is assigned to both {previous} and {owner}"
                )
            cpu_owners[cpu_id] = owner
    for model in data["models"]:
        pinned = (
            "model_config" in model,
            "model_config_sha256" in model,
            "checkpoint" in model,
            "checkpoint_sha256" in model,
        )
        if any(pinned) and not all(pinned):
            raise ConfigError(
                f"model {model['id']} must set model_config and checkpoint "
                "paths with both SHA-256 values"
            )
        try:
            get_architecture_profile(model["architecture_profile"])
        except ArchitectureProfileError as exc:
            raise ConfigError(str(exc)) from exc
        message_type = model.get("input_message_type")
        if model["modality"] == "lidar" and message_type not in (
            None,
            "pointcloud2",
        ):
            raise ConfigError(
                f"model {model['id']} has incompatible LiDAR message type"
            )
        if model["modality"] == "image" and message_type == "pointcloud2":
            raise ConfigError(
                f"model {model['id']} has incompatible image message type"
            )
        if (
            "point_feature_count" in model
            and model["modality"] != "lidar"
        ):
            raise ConfigError(
                f"model {model['id']} cannot select point features from "
                "non-LiDAR input"
            )
        if model["modality"] != "image" and "inference_resize_scale" in model:
            raise ConfigError(
                f"model {model['id']} cannot resize non-image inference input"
            )
        has_percentage = "mps_percentage" in model
        if mps and not has_percentage:
            raise ConfigError(
                f"model {model['id']} requires mps_percentage when "
                "MPS is enabled"
            )
        if not mps and has_percentage:
            raise ConfigError(
                f"model {model['id']} sets mps_percentage while "
                "MPS is disabled"
            )
        cta = model.get("nvbit_cta_profile")
        if cta:
            mode = cta.get("mode", "controlled")
            passive = mode == "passive"
            sequenced = mode in ("passive", "mixed")
            required_target_fields = (
                ("sequence_start", "sequence_end")
                if passive else
                ("kernel_occurrence", "sequence_start", "sequence_end")
                if sequenced else ("kernel_occurrence",)
            )
            if any(
                field not in target
                for target in cta["targets"]
                for field in required_target_fields
            ):
                raise ConfigError(
                    f"model {model['id']} NVBit CTA targets do not match "
                    f"{cta.get('mode', 'controlled')} mode"
                )
            if sequenced and any(
                target["sequence_start"] > target["sequence_end"]
                for target in cta["targets"]
            ):
                raise ConfigError(
                    f"model {model['id']} sequenced NVBit CTA window is "
                    "invalid"
                )
            if passive and any(
                target.get("align_launch", False)
                for target in cta["targets"]
            ):
                raise ConfigError(
                    f"model {model['id']} passive NVBit CTA window requests "
                    "launch alignment"
                )
    variation = data.get("input_variation")
    if variation:
        corruption = variation["corruption"]
        kind = corruption["type"]
        severity = corruption["severity"]
        rate = corruption.get("rain_rate_mm_per_hour")
        physical_rain = kind == "rain" and rate is not None
        if ((kind in ("clean", "fixed")
             and (severity != 0 or rate is not None))
                or (physical_rain and severity is not None)
                or (kind not in ("clean", "fixed", "rain") and severity not in
                    range(1, 6))
                or (kind == "rain" and not physical_rain and severity not in
                    range(1, 6))):
            raise ConfigError(
                "clean/fixed requires severity 0, physical rain requires a rate, "
                "and legacy weather requires severity 1-5"
            )
        if variation["tenancy"] != {
            1: "single", 2: "two_model", 3: "three_model"
        }.get(len(data["models"])):
            raise ConfigError(
                "input_variation.tenancy does not match model count"
            )
    green_models = [
        model for model in data["models"] if "green_context" in model
    ]
    if green_models:
        if mps:
            raise ConfigError("green_context requires MPS to be disabled")
        if any("paired_trial" in model for model in data["models"]):
            raise ConfigError(
                "green_context does not support paired execution"
            )
        partitions = [
            (model["green_context"]["sm_count"],
             model["green_context"]["partition_index"])
            for model in green_models
        ]
        if len(partitions) != len(set(partitions)):
            raise ConfigError("green_context partitions must be unique")
    paired = [model for model in data["models"] if "paired_trial" in model]
    if paired:
        if len(paired) != len(data["models"]):
            raise ConfigError(
                "paired_trial must be configured for every model"
            )
        shared = (
            "direction",
            "trials",
            "period_seconds",
            "target_anchor_seconds",
            "launch_tolerance_seconds",
        )
        for name in shared:
            if len({model["paired_trial"][name] for model in paired}) != 1:
                raise ConfigError(f"paired_trial {name} must match")
        if len({model["paired_trial"].get(
                "timing_diagnostics_only", False) for model in paired}) != 1:
            raise ConfigError(
                "paired_trial timing_diagnostics_only must match")
        roles = [model["paired_trial"]["role"] for model in paired]
        if len(paired) == 2 and sorted(roles) != ["co_runner", "target"]:
            raise ConfigError("two-model paired_trial requires both roles")
        period = paired[0]["paired_trial"]["period_seconds"]
        for model in paired:
            trial = model["paired_trial"]
            if trial["deadline_offset_seconds"] >= period:
                raise ConfigError(
                    "paired_trial deadline offset must be within period"
                )
            if (
                trial["role"] == "target"
                and trial["deadline_offset_seconds"]
                != trial["target_anchor_seconds"]
            ):
                raise ConfigError(
                    "paired target deadline must equal target anchor"
                )
            if trial["role"] == "target" and trial["k_replays"]:
                raise ConfigError("paired target cannot replay a subgraph")
            if trial["k_replays"] and not trial.get("replay_module"):
                raise ConfigError("subgraph replay requires replay_module")
    if synthetic_fma:
        if not mps:
            raise ConfigError("synthetic_fma requires MPS")
        if not paired or len(paired) != len(data["models"]):
            raise ConfigError("synthetic_fma requires paired model execution")
        period = paired[0]["paired_trial"]["period_seconds"]
        if synthetic_fma["paired_deadline_seconds"] >= period:
            raise ConfigError(
                "synthetic_fma paired deadline must be within period"
            )
    clocks = data["gpu"]
    if ("graphics_clock_mhz" in clocks) != ("memory_clock_mhz" in clocks):
        raise ConfigError(
            "gpu graphics_clock_mhz and memory_clock_mhz must be set together"
        )
    level = recording["level"]
    if level in ("level1", "level2a") and not nsys and "capsule" not in recording:
        raise ConfigError(f"recording {level} requires recording.nsys")
    if level in ("level1", "level2a") and ncu:
        raise ConfigError(f"recording {level} cannot use Nsight Compute")
    if level == "level2b" and not ncu:
        raise ConfigError("recording level2b requires recording.ncu")
    if level == "level2b" and nsys:
        raise ConfigError(
            "recording level2b forbids simultaneous Nsight Systems collection"
        )
    gpu_metric_fields = {
        "gpu_metrics_devices", "gpu_metrics_frequency_hz", "gpu_metrics_set"
    }
    configured_metrics = gpu_metric_fields.intersection(nsys or {})
    if level == "level2a" and configured_metrics != gpu_metric_fields:
        raise ConfigError(
            "recording level2a requires all Nsight Systems GPU metric fields"
        )
    if level == "level1" and configured_metrics:
        raise ConfigError("recording level1 cannot collect GPU metrics")
    scopes = set(recording["scopes"])
    if "preprocessing" in scopes and "preprocessing" not in recording:
        raise ConfigError(
            "preprocessing scope requires recording.preprocessing"
        )
    if "preprocessing" not in scopes and "preprocessing" in recording:
        raise ConfigError(
            "recording.preprocessing requires preprocessing scope"
        )
    if mps and nsys and nsys["gpu_context_switch"]:
        raise ConfigError("GPU context-switch tracing is unsupported with MPS")
    if "capsule" in recording:
        options = recording["capsule"]
        if not mps:
            raise ConfigError("kernel capsule recording requires MPS")
        if options.get("checkpoint_required", True) is not True:
            raise ConfigError("kernel capsule recording requires checkpoints")
        for name in ("timing_repetitions", "deep_repetitions"):
            if int(options.get(name, 1)) != 1:
                raise ConfigError(f"kernel capsule recording requires {name}=1")
        if int(options.get("sequence_match_attempts", 3)) != 3:
            raise ConfigError(
                "gpu_forensics_v1 requires sequence_match_attempts=3"
            )
        if int(options.get("priority_counterfactual_trials", 3)) != 3:
            raise ConfigError(
                "gpu_forensics_v1 requires priority_counterfactual_trials=3"
            )
        if int(options.get("priority_min_effect_ns", 1)) < 0:
            raise ConfigError("priority_min_effect_ns cannot be negative")
    phase = data["run"].get("phase", "measurement")
    if phase == "validation":
        missing = [
            name
            for name in ("parent_run_id", "hypothesis_id", "intervention_id")
            if not data["run"].get(name)
        ]
        if missing:
            raise ConfigError("validation run requires " + ", ".join(missing))


def load_run_config(path: str, check_paths: bool = True,
                    artifact_root: Optional[str] = None) -> RunConfig:
    """Load, validate, and resolve a YAML run configuration."""
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read configuration: {exc}") from exc
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError("configuration root must be an object")
    try:
        schema = json.loads(schema_path().read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(parsed)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        prefix = f"{location}: " if location else ""
        raise ConfigError(prefix + exc.message) from exc
    _semantic_validate(parsed)
    resolved = json.loads(json.dumps(parsed))
    base = source.parent
    for section, field in PATH_FIELDS:
        value = Path(resolved[section][field]).expanduser()
        resolved[section][field] = str(
            (base / value).resolve()
            if not value.is_absolute()
            else value.resolve()
        )
    controlled = resolved["replay"].get("controlled_bag_manifest")
    if controlled:
        path = Path(controlled).expanduser()
        path = path.resolve() if path.is_absolute() else (base / path).resolve()
        resolved["replay"]["controlled_bag_manifest"] = str(path)
        if check_paths:
            from closeloop_testbed.replayer import controlled_manifest
            try:
                controlled_manifest(resolved["replay"])
            except (OSError, ValueError, KeyError) as exc:
                raise ConfigError(f"invalid controlled bag: {exc}") from exc
    for model in resolved["models"]:
        for field in ("warmup_input", "model_config", "checkpoint"):
            if field not in model:
                continue
            value = Path(model[field]).expanduser()
            model[field] = str(
                (base / value).resolve()
                if not value.is_absolute()
                else value.resolve()
            )
    if "input_variation" in resolved:
        corruption = resolved["input_variation"]["corruption"]
        value = Path(corruption["dataset_manifest"]).expanduser()
        corruption["dataset_manifest"] = str(
            (base / value).resolve()
            if not value.is_absolute()
            else value.resolve()
        )
    if check_paths:
        metadata = Path(resolved["replay"]["metadata_path"])
        bags = Path(resolved["replay"]["bag_directory"])
        if not metadata.is_file():
            raise ConfigError("replay.metadata_path is not a file")
        if not bags.is_dir():
            raise ConfigError("replay.bag_directory is not a directory")
        for model in resolved["models"]:
            if not Path(model["warmup_input"]).is_file():
                raise ConfigError(
                    f"warmup_input for model {model['id']} is not a file"
                )
            if "checkpoint" in model:
                model_config = Path(model["model_config"])
                if not model_config.is_file():
                    raise ConfigError(
                        f"model_config for model {model['id']} is not a file"
                    )
                observed = hashlib.sha256(
                    model_config.read_bytes()
                ).hexdigest()
                if observed != model["model_config_sha256"]:
                    raise ConfigError(
                        f"model config hash mismatch for model {model['id']}"
                    )
                checkpoint = Path(model["checkpoint"])
                if not checkpoint.is_file():
                    raise ConfigError(
                        f"checkpoint for model {model['id']} is not a file"
                    )
                observed = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if observed != model["checkpoint_sha256"]:
                    raise ConfigError(
                        f"checkpoint hash mismatch for model {model['id']}"
                    )
        if "input_variation" in resolved:
            corruption = resolved["input_variation"]["corruption"]
            manifest = Path(corruption["dataset_manifest"])
            if not manifest.is_file():
                raise ConfigError(
                    "input_variation corruption dataset_manifest is not a file"
                )
            observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
            if observed != corruption["dataset_manifest_sha256"]:
                raise ConfigError(
                    "input_variation corruption dataset manifest hash mismatch"
                )
    config = RunConfig(
        resolved, source, raw, hashlib.sha256(raw).hexdigest()
    )
    return config.with_artifact_root(artifact_root) if artifact_root else config


def validate_config(path: str, check_paths: bool = True) -> RunConfig:
    """Validate a run configuration without binding an artifact root."""
    return load_run_config(path, check_paths=check_paths)
