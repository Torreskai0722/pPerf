"""Validate architecture evidence and module-range timing coverage."""

import argparse
import json
from pathlib import Path
import sqlite3
from statistics import mean
import sys
from typing import Dict, Iterable, List, Tuple

TAG_PREFIX = "closeloop:"

# Schema-v1 manifests did not embed these recorder requirements. New manifests
# are self-describing; this table remains read-only legacy support.
LEGACY_REQUIRED_BINDINGS = {
    "mmdet_single_stage_2d_v1": (
        {"model.predict"},
        {"data_preprocessor", "backbone", "bbox_head"},
    ),
    "mmdet3d_voxel_two_stage_v1": (
        {"model.predict"},
        {
            "data_preprocessor", "pts_voxel_encoder", "pts_middle_encoder",
            "pts_backbone", "pts_bbox_head",
        },
    ),
    "mmdet3d_point_single_stage_v1": (
        {"model.predict", "model.extract_feat"},
        {"data_preprocessor", "backbone", "bbox_head"},
    ),
    "mmseg_encoder_decoder_v1": (
        {"model.predict"},
        {"data_preprocessor", "backbone", "decode_head"},
    ),
    "mmdet_detr_2d_v1": (
        {"model.predict"},
        {
            "data_preprocessor", "backbone", "neck", "positional_encoding",
            "encoder", "decoder", "bbox_head",
        },
    ),
    "mmdet_two_stage_2d_v1": (
        {"model.predict"},
        {"data_preprocessor", "backbone", "rpn_head", "roi_head"},
    ),
}


def _merged_duration(intervals: Iterable[Tuple[int, int]]) -> int:
    """Return the duration of the union of half-open intervals."""
    merged: List[List[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _trace_ranges(database: sqlite3.Connection) -> List[Dict[str, object]]:
    """Decode completed close-loop NVTX ranges from an Nsight export."""
    rows = database.execute(
        "SELECT start, end, text FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND text LIKE ?",
        (TAG_PREFIX + "%",),
    )
    decoded = []
    for start, end, text in rows:
        try:
            tag = json.loads(text[len(TAG_PREFIX):])
        except (TypeError, json.JSONDecodeError):
            continue
        tag.update({"start": start, "end": end})
        decoded.append(tag)
    return decoded


def _coverage_by_model(ranges: Iterable[Dict[str, object]]) -> Dict[str, dict]:
    """Compute non-warmup depth-zero module coverage per model and input."""
    selected = list(ranges)
    modules = {}
    for event in selected:
        if event.get("event") != "module":
            continue
        key = (str(event.get("model")), str(event.get("input")))
        modules.setdefault(key, []).append(event)
    ratios = {}
    for inference in selected:
        input_id = str(inference.get("input"))
        if (inference.get("event") != "inference"
                or input_id.startswith("warmup-")):
            continue
        model_id = str(inference.get("model"))
        start = int(inference["start"])
        end = int(inference["end"])
        if end <= start:
            continue
        intervals = []
        for module in modules.get((model_id, input_id), []):
            module_start = max(start, int(module["start"]))
            module_end = min(end, int(module["end"]))
            if module_end > module_start:
                intervals.append((module_start, module_end))
        ratio = _merged_duration(intervals) / (end - start)
        ratios.setdefault(model_id, []).append(ratio)
    return {
        model_id: {
            "inference_inputs": len(values),
            "mean_module_wall_time_coverage": mean(values),
            "minimum_module_wall_time_coverage": min(values),
            "maximum_module_wall_time_coverage": max(values),
        }
        for model_id, values in ratios.items()
    }


def analyze_run(
        run_directory: Path, minimum: float = 0.75
) -> Dict[str, object]:
    """Validate one completed run and return a serializable report."""
    run_directory = Path(run_directory)
    manifest = json.loads(
        (run_directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    database_path = run_directory / "profile.sqlite"
    failures = []
    if manifest.get("state") != "success":
        failures.append("run manifest is not successful")
    with sqlite3.connect(str(database_path)) as database:
        ranges = _trace_ranges(database)
        kernel_count = database.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
    if kernel_count <= 0:
        failures.append("trace contains no CUDA kernels")
    timing = _coverage_by_model(ranges)
    models = {}
    for model_id, status in manifest.get("models", {}).items():
        profile_name = status["architecture_profile"]
        legacy = LEGACY_REQUIRED_BINDINGS.get(profile_name)
        if legacy is None and (
                "required_method_bindings" not in status
                or "required_module_bindings" not in status):
            raise ValueError(
                f"manifest lacks binding requirements for {profile_name!r}"
            )
        default_methods, default_modules = legacy or (set(), set())
        required_methods = set(
            status.get("required_method_bindings", default_methods)
        )
        required_modules = set(
            status.get("required_module_bindings", default_modules)
        )
        observed_methods = set(status.get("observed_method_bindings", []))
        observed_modules = set(status.get("observed_module_bindings", []))
        missing_methods = sorted(required_methods - observed_methods)
        missing_modules = sorted(required_modules - observed_modules)
        model_failures = []
        if status.get("state") != "acknowledged":
            model_failures.append(
                "model did not acknowledge replay completion"
            )
        if status.get("inputs", 0) <= 0:
            model_failures.append("model processed no replay inputs")
        if missing_methods:
            model_failures.append(
                "missing required methods: " + ", ".join(missing_methods)
            )
        if missing_modules:
            model_failures.append(
                "missing required modules: " + ", ".join(missing_modules)
            )
        metrics = timing.get(model_id)
        if metrics is None:
            model_failures.append(
                "trace contains no non-warmup inference ranges"
            )
            metrics = {
                "inference_inputs": 0,
                "mean_module_wall_time_coverage": 0.0,
                "minimum_module_wall_time_coverage": 0.0,
                "maximum_module_wall_time_coverage": 0.0,
            }
        elif metrics["mean_module_wall_time_coverage"] < minimum:
            model_failures.append(
                "average module wall-time coverage "
                f"{metrics['mean_module_wall_time_coverage']:.3f} is below "
                f"{minimum:.3f}"
            )
        models[model_id] = {
            "architecture_profile": profile_name,
            "observed_method_bindings": sorted(observed_methods),
            "observed_module_bindings": sorted(observed_modules),
            **metrics,
            "failures": model_failures,
        }
        failures.extend(
            f"{model_id}: {failure}" for failure in model_failures
        )
    return {
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "minimum_mean_module_wall_time_coverage": minimum,
        "cuda_kernel_count": kernel_count,
        "models": models,
        "passed": not failures,
        "failures": failures,
    }


def main(argv=None) -> int:
    """Validate one or more completed run directories."""
    parser = argparse.ArgumentParser(
        description="Validate close-loop architecture timing coverage"
    )
    parser.add_argument("run_directories", nargs="+")
    parser.add_argument("--minimum", type=float, default=0.75)
    args = parser.parse_args(argv)
    if not 0.0 <= args.minimum <= 1.0:
        parser.error("--minimum must be between zero and one")
    passed = True
    for value in args.run_directories:
        run_directory = Path(value).expanduser().resolve()
        try:
            report = analyze_run(run_directory, args.minimum)
            output = run_directory / "architecture_coverage.json"
            output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, KeyError, json.JSONDecodeError,
                sqlite3.DatabaseError, ValueError) as exc:
            print(f"{run_directory}: {exc}", file=sys.stderr)
            passed = False
            continue
        print(json.dumps(report, sort_keys=True))
        passed = passed and report["passed"]
    return 0 if passed else 1
