"""Validate and extract one immutable input-variation run."""

import hashlib
import json
from pathlib import Path
import sqlite3

import yaml

from .._common import StudyError, nvtx_ranges
from ._frame_analysis import extract_frame_rows


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dataset_manifest_errors(value):
    """Validate the provenance fields required for analysis joins."""
    errors = []
    if value.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")
    corruption = value.get("corruption", {})
    kind = corruption.get("type")
    severity = corruption.get("severity")
    if kind in ("clean", "fixed") and severity != 0:
        errors.append(f"{kind} data requires severity 0")
    elif kind in ("rain", "snow") and not (
            severity in range(1, 6)
            or (
                kind == "rain"
                and corruption.get("full_duration") is True
                and isinstance(corruption.get("rain_rate_mm_per_hour"),
                               (int, float))
                and corruption["rain_rate_mm_per_hour"] > 0
            )):
        errors.append(f"{kind} data has invalid severity")
    elif kind not in ("clean", "fixed", "rain", "snow"):
        errors.append("unsupported corruption type")
    records = value.get("sensor_records", [])
    if {record.get("topic") for record in records} != {
            "/CAM_FRONT/image_rect_compressed", "/LIDAR_TOP"}:
        errors.append("sensor records must cover camera and LiDAR")
    if any(
        record.get("bag_timestamp_ns")
        != record.get("output_bag_timestamp_ns")
        for record in records
    ):
        errors.append("a bag timestamp changed")
    if any(
        record.get("header_timestamp_ns")
        != record.get("output_header_timestamp_ns")
        for record in records
    ):
        errors.append("a sensor header timestamp changed")
    if value.get("preserved_stream_source_sha256") != value.get(
            "preserved_stream_output_sha256"):
        errors.append("an unselected message changed")
    return errors


def _module_and_structure_rows(run_directory, statuses, input_ids):
    modules, structures = [], []
    with sqlite3.connect(str(Path(run_directory) / "profile.sqlite")) as db:
        for record in nvtx_ranges(db):
            tag = record["tag"]
            model = tag.get("model")
            input_id = str(tag.get("input"))
            if (
                tag.get("event") == "module"
                and model in statuses
                and input_id in input_ids[model]
            ):
                modules.append({
                    "model_id": model, "input_id": input_id,
                    "module": tag.get("module"),
                    "duration_ms": (record["end"] - record["start"]) / 1e6,
                })
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "NVTX_EVENTS" in tables:
            for start, global_tid, text in db.execute(
                "SELECT start,globalTid,text FROM NVTX_EVENTS "
                "WHERE text LIKE 'closeloop:%'"
            ):
                try:
                    tag = json.loads(str(text)[len("closeloop:"):])
                except json.JSONDecodeError:
                    continue
                model = tag.get("model")
                input_id = str(tag.get("input"))
                if (
                    tag.get("event") == "structure"
                    and model in statuses
                    and input_id in input_ids[model]
                ):
                    structures.append({
                        "model_id": model, "input_id": input_id,
                        "module": tag.get("module"),
                        "structure_kind": tag.get("structure_kind"),
                        "structure_count": tag.get("structure_count"),
                        "timestamp_ns": int(start),
                        "pid": (int(global_tid) >> 24) & 0xFFFFFF,
                    })
    return modules, structures


def _validate_source_passes(
    model_id, records, repeat_count, minimum, source_timestamps=None,
    minimum_window_fraction=None,
):
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
            coverage = (
                (max(measured) - min(measured))
                / (max(source_timestamps) - min(source_timestamps))
            )
            if coverage < minimum_window_fraction:
                errors.append(
                    f"{model_id} pass {pass_index} covers {coverage:.3f} of "
                    f"source window, needs {minimum_window_fraction:.3f}"
                )
    return errors


def validate_input_run(
    run_directory, planned, study, expected_config=None,
    validation_policy=None,
):
    """Validate one completed run and extract its analysis evidence."""
    run_directory = Path(run_directory)
    errors = []
    required = (
        "config.yaml", "run_manifest.json", "testbed_result.json",
        "profile.sqlite", "input_variation_clock_control.json",
    )
    errors.extend(
        f"missing {name}" for name in required
        if not (run_directory / name).is_file()
    )
    if errors:
        return {"valid": False, "errors": errors}
    config_path = run_directory / "config.yaml"
    config = yaml.safe_load(config_path.read_bytes())
    manifest = _load_json(run_directory / "run_manifest.json")
    testbed = _load_json(run_directory / "testbed_result.json")
    clock = _load_json(run_directory / "input_variation_clock_control.json")
    if expected_config is not None and (
            config_path.read_bytes() != Path(expected_config).read_bytes()):
        errors.append("executed config differs from immutable planned config")
    variation = config.get("input_variation", {})
    if manifest.get("state") != "success":
        errors.append(f"run state is {manifest.get('state')}")
    if not manifest.get("gpu_hardware"):
        errors.append("GPU hardware identity is missing")
    if not clock.get("clocks_restored"):
        errors.append("GPU clock controls were not restored")
    if not testbed.get("replay_success") or not testbed.get(
            "all_acknowledged"):
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
        condition_path = variation["corruption"]["dataset_manifest"]
        condition_manifest = _load_json(condition_path)
        errors.extend(_dataset_manifest_errors(condition_manifest))
        if _sha256(condition_path) != variation["corruption"][
                "dataset_manifest_sha256"]:
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
        input_ids[model_id] = {
            str(record["input_id"]) for record in records
        }
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
            topic = (
                "/CAM_FRONT/image_rect_compressed"
                if model["modality"] == "image" else "/LIDAR_TOP"
            )
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
        for name in (
            "mmlab_model", "model_config", "model_config_sha256",
            "checkpoint", "checkpoint_sha256",
        ):
            if status.get(name) != model.get(name):
                errors.append(f"{model_id} observed {name} differs")
    rows, modules, structures = [], [], []
    trace = {}
    try:
        rows, trace = extract_frame_rows(run_directory)
        if len(rows) != sum(len(values) for values in input_ids.values()):
            errors.append("trace/sample row count differs")
        if any(str(row["frame_id"]).startswith("warmup-") for row in rows):
            errors.append("warmup range entered frame metrics")
        valid_boundaries = {
            "post_nvtx_cuda_event_synchronize",
            "nvtx_includes_completion_synchronize",
        }
        if any(row["completion_boundary"] not in valid_boundaries
               for row in rows):
            errors.append("timing completion boundary is not synchronized")
        if set(statuses) == set(configured_models):
            modules, structures = _module_and_structure_rows(
                run_directory, statuses, input_ids)
        warmups = {model["id"]: 0 for model in config["models"]}
        with sqlite3.connect(str(run_directory / "profile.sqlite")) as db:
            for record in nvtx_ranges(db):
                tag = record["tag"]
                if (
                    tag.get("event") == "inference"
                    and str(tag.get("input", "")).startswith("warmup-")
                    and tag.get("model") in warmups
                ):
                    warmups[tag["model"]] += 1
        for model in config["models"]:
            if warmups[model["id"]] != model["warmup_count"]:
                errors.append(
                    f"{model['id']} warmup range count differs: "
                    f"{warmups[model['id']]} != {model['warmup_count']}"
                )
    except (StudyError, OSError, sqlite3.DatabaseError) as exc:
        errors.append(f"trace extraction failed: {exc}")
    return {
        "valid": not errors, "errors": errors, "frames": rows,
        "modules": modules, "structures": structures,
        "trace_evidence": trace, "config": config,
    }
