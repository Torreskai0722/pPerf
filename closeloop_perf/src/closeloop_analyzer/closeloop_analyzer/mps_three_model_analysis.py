"""Audit and report the fixed three-model MPS leftover experiment."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sqlite3

import matplotlib
import numpy as np
import yaml

from .input_data.corrected_input_analysis import (
    QUANTILES, _write_csv, distribution_metrics, wasserstein_1,
)
from ._common import load_study as _load_study, nvtx_ranges
from .target_selection import decode_global_id

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCENES = ("scene-1044", "scene-0434")
SCOPES = SCENES + ("pooled",)
SOURCE_MESSAGE_COUNTS = {"scene-1044": 48, "scene-0434": 238}
MPS_CONFIGURATIONS = (
    (40, 40, 40), (60, 60, 60), (80, 80, 80), (90, 90, 90),
    (40, 60, 80), (40, 80, 60), (60, 40, 80), (60, 80, 40),
    (80, 40, 60), (80, 60, 40),
)
MODEL_ORDER = ("faster_rcnn", "deeplabv3plus", "detr")


def load_study(path):
    """Read and validate the fixed three-model study contract."""
    study = _load_study(path)
    data = study["data"]
    triples = tuple(
        tuple(values) for values in data["ordered_mps_configurations"]
    )
    if data.get("schema_version") != 1 or triples != MPS_CONFIGURATIONS:
        raise ValueError("ordered MPS configurations differ")
    if tuple(model["id"] for model in data["models"]) != MODEL_ORDER:
        raise ValueError("three-model order differs")
    return study


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_id(number, triple):
    return f"mps3-c{number:02d}-" + "-".join(map(str, triple))


def _latency_records(run_directory, model_id, ranges):
    inputs = {
        str(record["input_id"]): record
        for record in (
            json.loads(line)
            for line in (
                run_directory / f"model_{model_id}_inputs.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    records = []
    for item in ranges:
        tag = item["tag"]
        input_id = str(tag.get("input", ""))
        if (
            tag.get("event") != "inference"
            or tag.get("model") != model_id
            or input_id.startswith("warmup-")
        ):
            continue
        source = inputs.get(input_id)
        if source is None:
            raise ValueError(f"{model_id}/{input_id} lacks input identity")
        records.append({
            "input_id": input_id,
            "scene_name": source["scene_name"],
            "latency_ms": (item["end"] - item["start"]) / 1_000_000,
        })
    if len(records) != len(inputs):
        raise ValueError(
            f"{model_id} has {len(inputs)} inputs and {len(records)} ranges"
        )
    return records


def collect(study):
    """Audit all ten runs and collect scene-resolved inference latency."""
    output_root = Path(study["data"]["output_root"])
    grouped = {}
    validation = []
    analyses = {}
    for number, triple in enumerate(MPS_CONFIGURATIONS, 1):
        run_id = _run_id(number, triple)
        run_directory = output_root / run_id
        errors = []
        required = (
            "config.yaml", "run_manifest.json", "testbed_result.json",
            "communication_summary.json", "phase3_clock_control.json",
            "profile.sqlite", "mps_leftover_analysis.json",
            "mps_leftover_occurrences.csv",
        )
        for name in required:
            if not (run_directory / name).is_file():
                errors.append(f"missing {name}")
        if errors:
            validation.append({"run_id": run_id, "errors": errors})
            continue
        config = yaml.safe_load(
            (run_directory / "config.yaml").read_text(encoding="utf-8")
        )
        manifest = _read_json(run_directory / "run_manifest.json")
        testbed = _read_json(run_directory / "testbed_result.json")
        communication = _read_json(
            run_directory / "communication_summary.json"
        )
        clock = _read_json(run_directory / "phase3_clock_control.json")
        analysis = _read_json(run_directory / "mps_leftover_analysis.json")
        analyses[number] = analysis
        models = config["models"]
        if manifest.get("state") != "success":
            errors.append(f"manifest state {manifest.get('state')}")
        if tuple(model["id"] for model in models) != MODEL_ORDER:
            errors.append("model order differs")
        if tuple(model.get("mps_percentage") for model in models) != triple:
            errors.append("configured percentage triple differs")
        if not clock.get("clocks_restored"):
            errors.append("GPU clocks were not restored")
        mps = manifest.get("mps", {})
        if (
            len(mps.get("server_ids", [])) != 1
            or mps.get("server_state") != "stopped"
            or not mps.get("quit_succeeded")
            or not mps.get("compute_mode_restored")
        ):
            errors.append("MPS lifecycle evidence differs")
        intervals = testbed.get("playback_intervals", [])
        if (
            testbed.get("scene_tokens") != study["data"]["scene_tokens"]
            or testbed.get("bags_started") != 2
            or testbed.get("bags_completed") != 2
            or len(intervals) != 2
            or any(item.get("completion_status") != "completed"
                   for item in intervals)
            or intervals[0]["end_monotonic_ns"]
            > intervals[1]["process_started_monotonic_ns"]
        ):
            errors.append("ordered two-scene replay differs")

        statuses = {}
        for model, percentage in zip(models, triple):
            model_id = model["id"]
            status = _read_json(run_directory / f"model_{model_id}.json")
            statuses[model_id] = status
            if (
                status.get("state") != "acknowledged"
                or status.get("error") is not None
                or status.get(
                    "configured_cuda_mps_active_thread_percentage"
                ) != percentage
                or status.get(
                    "process_observed_cuda_mps_active_thread_percentage"
                ) != str(percentage)
                or not 40 <= percentage <= 90
            ):
                errors.append(f"{model_id} effective MPS differs")
        server_log = (run_directory / "mps/log/server.log").read_text(
            encoding="utf-8"
        )
        if any(
            f"Status of client {{{status['pid']}, 1}} is ACTIVE"
            not in server_log for status in statuses.values()
        ):
            errors.append("MPS active-client evidence missing")

        with sqlite3.connect(run_directory / "profile.sqlite") as connection:
            ranges = nvtx_ranges(connection)
            context_rows = connection.execute(
                "SELECT globalPid, contextId, COUNT(DISTINCT streamId) "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "GROUP BY globalPid, contextId"
            ).fetchall()
        contexts = {
            decode_global_id(int(global_pid))[0]: (int(context), int(streams))
            for global_pid, context, streams in context_rows
        }
        if any(status["pid"] not in contexts for status in statuses.values()):
            errors.append("model CUDA context missing")

        scene_counts = {}
        for model_id in MODEL_ORDER:
            records = _latency_records(run_directory, model_id, ranges)
            for scope in SCOPES:
                values = [
                    row["latency_ms"] for row in records
                    if scope == "pooled" or row["scene_name"] == scope
                ]
                if not values:
                    errors.append(f"{model_id}/{scope} is empty")
                else:
                    grouped[(number, model_id, scope)] = values
            scene_counts[model_id] = {
                scene: sum(row["scene_name"] == scene for row in records)
                for scene in SCENES
            }

        expected_source_messages = sum(SOURCE_MESSAGE_COUNTS.values())
        source_to_relay_lost = (
            expected_source_messages - communication["relay_messages"]
        )
        expected_deliveries = communication["relay_messages"] * 3
        relay_to_model_lost = expected_deliveries - communication["matched"]
        if relay_to_model_lost != communication.get(
            "dropped_or_overwritten", 0
        ):
            errors.append("communication loss accounting differs")
        if source_to_relay_lost < 0:
            errors.append("relay observed more messages than source bag")
        duplicate_inputs = 0
        out_of_order_inputs = 0
        for model_id in MODEL_ORDER:
            for line in (
                run_directory / f"model_{model_id}_inputs.jsonl"
            ).read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                duplicate_inputs += bool(record.get("duplicate"))
                out_of_order_inputs += bool(record.get("out_of_order"))
        if (
            communication.get("relay_duplicates", 0)
            or communication.get("relay_out_of_order", 0)
            or duplicate_inputs or out_of_order_inputs
        ):
            errors.append("duplicate or out-of-order input observed")

        validation.append({
            "configuration_number": number,
            "run_id": run_id,
            "ordered_percentage_triple": ",".join(map(str, triple)),
            "total_pressure": sum(triple),
            "percentage_asymmetry": max(triple) - min(triple),
            "state": manifest.get("state"),
            "valid": not errors,
            "errors": "; ".join(errors),
            "mps_server_id": ",".join(mps.get("server_ids", [])),
            "mps_server_state_after_run": mps.get("server_state"),
            "model_pids": ",".join(
                str(statuses[model]["pid"]) for model in MODEL_ORDER
            ),
            "cuda_contexts": ",".join(
                f"{contexts[statuses[model]['pid']][0]}"
                for model in MODEL_ORDER
            ),
            "cuda_stream_counts": ",".join(
                f"{contexts[statuses[model]['pid']][1]}"
                for model in MODEL_ORDER
            ),
            "effective_percentage_triple": ",".join(
                statuses[model][
                    "process_observed_cuda_mps_active_thread_percentage"
                ] for model in MODEL_ORDER
            ),
            "scene_a_bag": intervals[0]["bag_path"],
            "scene_b_bag": intervals[1]["bag_path"],
            **{
                f"{model}_{scene}_samples": scene_counts[model][scene]
                for model in MODEL_ORDER for scene in SCENES
            },
            "relay_messages": communication["relay_messages"],
            "matched_model_deliveries": communication["matched"],
            "source_to_relay_dropped": source_to_relay_lost,
            "relay_to_model_dropped_or_overwritten": relay_to_model_lost,
            "duplicate_inputs": duplicate_inputs,
            "out_of_order_inputs": out_of_order_inputs,
            "clock_requested_graphics_mhz": clock["control"]["requested"][
                "graphics_clock_mhz"
            ],
            "clock_effective_graphics_mhz": clock["control"][
                "observed_after_lock"
            ]["graphics_clock_mhz"],
            "clock_requested_memory_mhz": clock["control"]["requested"][
                "memory_clock_mhz"
            ],
            "clock_effective_memory_mhz": clock["control"][
                "observed_after_lock"
            ]["memory_clock_mhz"],
        })
    invalid = [row for row in validation if row.get("errors")]
    if invalid:
        raise ValueError(f"invalid Phase 3 runs: {invalid}")
    if len({row["mps_server_id"] for row in validation}) != 10:
        raise ValueError("MPS server identity was reused")
    return grouped, validation, analyses


def _block_bootstrap(values, repetitions, seed, block_size=10):
    values = np.asarray(values, dtype=float)
    block_size = min(block_size, len(values))
    block_count = int(np.ceil(len(values) / block_size))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(repetitions, block_count))
    indices = (
        starts[:, :, None] + np.arange(block_size)[None, None, :]
    ) % len(values)
    samples = values[indices.reshape(repetitions, -1)[:, :len(values)]]
    p0, p50, p99 = np.percentile(samples, (0, 50, 99), axis=1)
    return {"p50_ms": p50, "primary_range_ms": p99 - p0}


def _holm(rows):
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value"])
    adjusted = [1.0] * len(rows)
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * row["p_value"]))
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
        row["directional_supported"] = False
        row["strict_conclusion"] = "inconclusive_one_run_per_configuration"


def _occurrence_rows(study, analyses):
    summaries = []
    directions = []
    output_root = Path(study["data"]["output_root"])
    for number, triple in enumerate(MPS_CONFIGURATIONS, 1):
        run_id = _run_id(number, triple)
        analysis = analyses[number]
        with (output_root / run_id / "mps_leftover_occurrences.csv").open(
            encoding="utf-8", newline=""
        ) as source:
            occurrences = list(csv.DictReader(source))
        for scope in SCOPES:
            selected = [
                row for row in occurrences
                if scope == "pooled" or row["victim_scene_name"] == scope
            ]
            eligible = 0
            delayed = 0
            for model in MODEL_ORDER:
                if scope == "pooled":
                    eligible += analysis["models"][model][
                        "eligible_kernel_count"
                    ]
                    delayed += analysis["models"][model][
                        "delayed_kernel_count"
                    ]
                else:
                    scene = analysis["models"][model]["per_scene"][scope]
                    eligible += scene["eligible_kernel_count"]
                    delayed += scene["delayed_kernel_count"]
            affected = {
                (row["victim_pid"], row["victim_kernel_id"])
                for row in selected
            }
            summaries.append({
                "configuration_number": number,
                "run_id": run_id,
                "ordered_percentage_triple": ",".join(map(str, triple)),
                "scope": scope,
                "total_pressure": sum(triple),
                "percentage_asymmetry": max(triple) - min(triple),
                "eligible_kernel_count": eligible,
                "eligible_delayed_kernel_count": delayed,
                "affected_leftover_policy_kernel_count": len(affected),
                "occurrence_count": len(selected),
                "occurrence_rate_relative_to_eligible_delayed": (
                    len(affected) / delayed if delayed else 0.0
                ),
            })
            for aggressor in MODEL_ORDER:
                for victim in MODEL_ORDER:
                    if aggressor == victim:
                        continue
                    rows = [
                        row for row in selected
                        if row["aggressor_model"] == aggressor
                        and row["victim_model"] == victim
                    ]
                    if scope == "pooled":
                        denominator = analysis["models"][victim][
                            "delayed_kernel_count"
                        ]
                    else:
                        denominator = analysis["models"][victim][
                            "per_scene"
                        ][scope]["delayed_kernel_count"]
                    affected_direction = len({
                        (row["victim_pid"], row["victim_kernel_id"])
                        for row in rows
                    })
                    directions.append({
                        "configuration_number": number,
                        "ordered_percentage_triple": ",".join(
                            map(str, triple)
                        ),
                        "scope": scope,
                        "direction": f"{aggressor}->{victim}",
                        "eligible_delayed_victim_kernel_count": denominator,
                        "affected_leftover_policy_kernel_count": (
                            affected_direction
                        ),
                        "occurrence_count": len(rows),
                        "affected_rate_relative_to_eligible_delayed": (
                            affected_direction / denominator
                            if denominator else 0.0
                        ),
                    })
    return summaries, directions


def _two_model_rows(study):
    rows = []
    source_root = study["data"].get("two_model_source_root")
    if not source_root:
        return rows
    for run_directory in sorted(Path(source_root).expanduser().glob(
            "mps-leftover-faster*")):
        path = run_directory / "mps_leftover_analysis.json"
        if not path.is_file():
            continue
        result = _read_json(path)
        rows.append({
            "run_id": run_directory.name,
            "ordered_percentage_pair": result["configuration"][
                "ordered_mps_percentage_triple"
            ],
            "occurrence_count": result["occurrence_count"],
            "formal_predicate": (
                "delay>0.9ms; normalized start in [0.80,1.0); "
                "terminal overlap>0"
            ),
        })
    return rows


def analyze(study, analysis_root):
    """Write audited metrics, confidence results, figures, and report."""
    root = Path(analysis_root)
    root.mkdir(parents=True, exist_ok=True)
    grouped, validation, analyses = collect(study)
    _write_csv(root / "run_validation.csv", validation)

    quantiles = []
    for number, triple in enumerate(MPS_CONFIGURATIONS, 1):
        for model in MODEL_ORDER:
            for scope in SCOPES:
                quantiles.append({
                    "configuration_number": number,
                    "ordered_percentage_triple": ",".join(map(str, triple)),
                    "model_id": model,
                    "scope": scope,
                    **distribution_metrics(grouped[(number, model, scope)]),
                })
    _write_csv(root / "quantiles.csv", quantiles)

    wasserstein = []
    for left, right in combinations(range(1, 11), 2):
        for model in MODEL_ORDER:
            for scope in SCOPES:
                wasserstein.append({
                    "reference_configuration": left,
                    "comparison_configuration": right,
                    "model_id": model,
                    "scope": scope,
                    "wasserstein_1_ms": wasserstein_1(
                        grouped[(left, model, scope)],
                        grouped[(right, model, scope)],
                    ),
                })
    _write_csv(root / "wasserstein_comparisons.csv", wasserstein)

    repetitions = study["data"]["bootstrap_repetitions"]
    bootstraps = {}
    for number in range(1, 11):
        for model_index, model in enumerate(MODEL_ORDER):
            for scope_index, scope in enumerate(SCOPES):
                bootstraps[(number, model, scope)] = _block_bootstrap(
                    grouped[(number, model, scope)], repetitions,
                    study["data"]["random_seed"]
                    + number * 100 + model_index * 10 + scope_index,
                )
    strict = []
    for left, right in combinations(range(1, 11), 2):
        for model in MODEL_ORDER:
            for scope in SCOPES:
                left_metrics = distribution_metrics(
                    grouped[(left, model, scope)]
                )
                right_metrics = distribution_metrics(
                    grouped[(right, model, scope)]
                )
                for endpoint in ("p50_ms", "primary_range_ms"):
                    samples = (
                        bootstraps[(right, model, scope)][endpoint]
                        - bootstraps[(left, model, scope)][endpoint]
                    )
                    strict.append({
                        "reference_configuration": left,
                        "comparison_configuration": right,
                        "model_id": model,
                        "scope": scope,
                        "endpoint": endpoint,
                        "independent_runs_per_condition": 1,
                        "effect_ms": (
                            right_metrics[endpoint] - left_metrics[endpoint]
                        ),
                        "block_bootstrap_ci_low_ms": float(
                            np.percentile(samples, 2.5)
                        ),
                        "block_bootstrap_ci_high_ms": float(
                            np.percentile(samples, 97.5)
                        ),
                        "p_value": min(1.0, 2 * min(
                            float(np.mean(samples <= 0)),
                            float(np.mean(samples >= 0)),
                        )),
                    })
    _holm(strict)
    _write_csv(root / "strict_comparisons.csv", strict)

    leftover, directions = _occurrence_rows(study, analyses)
    _write_csv(root / "leftover_summary.csv", leftover)
    _write_csv(root / "leftover_directions.csv", directions)
    two_model = _two_model_rows(study)
    _write_csv(root / "two_model_formal_comparison.csv", two_model)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for model in MODEL_ORDER:
        for scope in SCOPES:
            figure, axis = plt.subplots(figsize=(8, 5))
            for number, color in zip(range(1, 11), colors):
                values = np.sort(grouped[(number, model, scope)])
                axis.step(
                    values, np.arange(1, len(values) + 1) / len(values),
                    where="post", label=f"C{number}", color=color,
                )
            axis.set_xlabel("inference latency (ms)")
            axis.set_ylabel("ECDF")
            axis.set_title(f"{model}: {scope}")
            axis.grid(alpha=0.2)
            axis.legend(ncol=2, fontsize=7)
            figure.tight_layout()
            figure.savefig(root / f"ecdf_{model}_{scope}.png", dpi=160)
            plt.close(figure)

    pooled_leftover = [row for row in leftover if row["scope"] == "pooled"]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        [row["configuration_number"] for row in pooled_leftover],
        [row["occurrence_count"] for row in pooled_leftover],
    )
    axis.set_xlabel("configuration number")
    axis.set_ylabel("formal occurrence count")
    axis.set_xticks(range(1, 11))
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(root / "leftover_occurrences.png", dpi=160)
    plt.close(figure)

    _write_report(
        root / "report.md", study, validation, quantiles, wasserstein,
        strict, leftover, directions, two_model,
    )


def _fmt(value):
    return f"{float(value):.3f}"


def _write_report(path, study, validation, quantiles, wasserstein, strict,
                  leftover, directions, two_model):
    pooled_leftover = [row for row in leftover if row["scope"] == "pooled"]
    nonzero_directions = [
        row for row in directions
        if row["scope"] == "pooled" and row["occurrence_count"]
    ]
    reference_strict = [
        row for row in strict
        if row["reference_configuration"] == 1
        and row["scope"] == "pooled"
    ]
    strict_lookup = {
        (row["comparison_configuration"], row["model_id"], row["endpoint"]):
        row for row in reference_strict
    }
    reference_w1 = [
        row for row in wasserstein
        if row["reference_configuration"] == 1 and row["scope"] == "pooled"
    ]
    lines = [
        "# Three-model explicit non-default MPS leftover-policy experiment",
        "", "## Question and hypothesis", "",
        "Does the formal cross-client leftover-policy pattern require a "
        "default 100% MPS client when Faster R-CNN, DeepLabV3+, and DETR all "
        "have explicit 40–90% caps? The hypothesis was that sufficient "
        "three-client pressure could produce the pattern without a 100% "
        "client.",
        "", "## Executed matrix and completion audit", "",
        "Exactly the ten prespecified configurations ran once each, "
        "sequentially, on real hardware. No adaptive configuration was "
        "added or skipped. Model tuple order was Faster R-CNN, DeepLabV3+, "
        "DETR. Each run kept all processes and settings alive while "
        "`scene-1044` completed before `scene-0434` began.",
        "",
        "| C | requested/effective triple | MPS server | PIDs | contexts | "
        "source→relay / relay→model loss | state |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for row in validation:
        lines.append(
            f"| {row['configuration_number']} | "
            f"{row['ordered_percentage_triple']} / "
            f"{row['effective_percentage_triple']} | "
            f"{row['mps_server_id']} ({row['mps_server_state_after_run']}) | "
            f"{row['model_pids']} | {row['cuda_contexts']} | "
            f"{row['source_to_relay_dropped']} / "
            f"{row['relay_to_model_dropped_or_overwritten']} | "
            f"{row['state']} |"
        )
    lines += [
        "",
        "The MPS daemon log reports its server-level default as 100%, but "
        "that is not a model client cap. Every model process metadata file "
        "records both configured and process-observed "
        "`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`; all 30 values equal their "
        "requested 40–90% values. Each PID also appears as ACTIVE in its "
        "unique server log and owns a traced CUDA context. No model omitted "
        "the variable or fell back to 100%.",
        "", "## Formal definition and occurrence results", "",
        "A kernel is affected only when its ready-to-start delay is strictly "
        "greater than 0.9 ms, a foreign-client aggressor is still executing "
        "at victim start, normalized victim start is in `[0.80, 1.0)`, and "
        "terminal overlap is positive. `mps_leftover_occurrences.csv` in "
        "each raw run records ready delay, normalized position, terminal "
        "overlap in ns and as aggressor fraction, identities, PID, context, "
        "stream, kernel signatures, co-aggressor count, and percentage "
        "triple.",
        "",
        "| C | triple | total | asymmetry | delayed eligible | affected | "
        "occurrences | affected/delayed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_leftover:
        lines.append(
            f"| {row['configuration_number']} | "
            f"{row['ordered_percentage_triple']} | {row['total_pressure']} | "
            f"{row['percentage_asymmetry']} | "
            f"{row['eligible_delayed_kernel_count']} | "
            f"{row['affected_leftover_policy_kernel_count']} | "
            f"{row['occurrence_count']} | "
            f"{row['occurrence_rate_relative_to_eligible_delayed']:.4f} |"
        )
    lines += [
        "", "Observed nonzero directions:", "",
        "| C | direction | delayed eligible | affected/occurrences | rate |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in nonzero_directions:
        lines.append(
            f"| {row['configuration_number']} | {row['direction']} | "
            f"{row['eligible_delayed_victim_kernel_count']} | "
            f"{row['affected_leftover_policy_kernel_count']}/"
            f"{row['occurrence_count']} | "
            f"{row['affected_rate_relative_to_eligible_delayed']:.4f} |"
        )
    lines += [
        "", "## Latency quantiles: each scene and pooled", "",
        "Actual usable samples are retained; counts need not be equal. "
        "Times are inference NVTX range durations and exclude five warm-up "
        "inputs per model.", "",
        "| C | model | scope | n | p0 | p1 | p5 | p25 | p50 | p75 | p95 | "
        "p99 | p99-p0 | normalized |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|",
    ]
    for row in quantiles:
        lines.append(
            f"| {row['configuration_number']} | {row['model_id']} | "
            f"{row['scope']} | {row['sample_count']} | "
            + " | ".join(_fmt(row[f"p{q}_ms"]) for q in QUANTILES)
            + f" | {_fmt(row['primary_range_ms'])} | "
            f"{row['normalized_range']:.4f} |"
        )
    lines += [
        "", "## Strict, confidence-aware results and conclusions", "",
        "There is one independent run per configuration. Therefore the "
        "predefined directional rule requires at least two independent runs "
        "per condition as well as a Holm-adjusted p < 0.05 and a 95% "
        "interval excluding zero. No comparison can satisfy the run-count "
        "rule; every pressure, asymmetry, assignment, latency, and occurrence "
        "directional claim is **inconclusive**.",
        "",
        "For transparency, `strict_comparisons.csv` contains effect sizes "
        "and 10,000-resample circular block-bootstrap 95% intervals using "
        "10-frame blocks, with Holm correction over all 810 endpoint "
        "comparisons. Those intervals quantify within-run temporal sampling "
        "only; frames or blocks are not promoted to independent hardware "
        "repetitions.",
        "",
        "Pooled effects relative to C1:", "",
        "| C | model | median effect [95% CI] ms | range effect [95% CI] ms |",
        "|---:|---|---:|---:|",
    ]
    for number in range(2, 11):
        for model in MODEL_ORDER:
            median = strict_lookup[(number, model, "p50_ms")]
            spread = strict_lookup[(number, model, "primary_range_ms")]
            lines.append(
                f"| {number} | {model} | {_fmt(median['effect_ms'])} "
                f"[{_fmt(median['block_bootstrap_ci_low_ms'])}, "
                f"{_fmt(median['block_bootstrap_ci_high_ms'])}] | "
                f"{_fmt(spread['effect_ms'])} "
                f"[{_fmt(spread['block_bootstrap_ci_low_ms'])}, "
                f"{_fmt(spread['block_bootstrap_ci_high_ms'])}] |"
            )
    lines += [
        "", "## Descriptive, confidence-agnostic results and conclusions", "",
        "Exploratorily, the formal pattern does **not** require a 100% "
        "client: every explicit non-default configuration contained at least "
        "one occurrence. Symmetric C1–C4 occurrence counts increased "
        "2 → 224 → 388 → 754 as total requested pressure increased 120 → "
        "180 → 240 → 270. This is observation, not a confidence-supported "
        "pressure effect.",
        "",
        "At fixed total pressure 180, assignment mattered descriptively: "
        "the six asymmetric permutations ranged from 1 to 786 occurrences, "
        "while symmetric `(60,60,60)` had 224. Because total pressure was "
        "fixed and all asymmetric cells had the same 40-point span, total "
        "pressure or asymmetry alone cannot explain the ranking; the "
        "model-to-percentage assignment is associated with the observed "
        "difference. This remains correlation, not validated causation.",
        "",
        "The prior two-model traces, reanalyzed with this exact formal "
        "predicate, had zero occurrences at matched 20/20, 40/40, 60/60, "
        "and 80/80; 20/100 through 80/100 had 153–238 and 100/100 had 404. "
        "The new three-model evidence therefore changes the earlier "
        "descriptive boundary: a 100% client was associated with the old "
        "two-model occurrences but is not necessary with three clients.",
        "",
        f"Across all 405 distribution comparisons, the largest W1 was "
        f"{max(row['wasserstein_1_ms'] for row in wasserstein):.3f} ms. "
        "`wasserstein_comparisons.csv` contains every configuration pair, "
        "model, and scene/pooled scope. Pooled C1 comparisons are:",
        "",
        "| comparison | model | W1 (ms) |",
        "|---|---|---:|",
    ]
    for row in reference_w1:
        lines.append(
            f"| C1→C{row['comparison_configuration']} | {row['model_id']} | "
            f"{_fmt(row['wasserstein_1_ms'])} |"
        )
    lines += [
        "", "ECDFs are `ecdf_<model>_<scene-or-pooled>.png`; "
        "`leftover_occurrences.png` plots formal counts.",
        "", "## Per-scene completion, loss, and exclusions", "",
        "Each run played one MCAP for scene-1044, then one MCAP for "
        "scene-0434, with no interval overlap. Exact interval times and bag "
        "paths are in each `testbed_result.json` and `run_validation.csv`. "
        "The source MCAPs contain 48 camera messages in scene-1044 and 238 "
        "in scene-0434. C10's raw-topic relay observed 237 of the latter, "
        "so one source-to-relay message was not observed; all other relays "
        "observed all 286 source messages. Depth-1 DDS does not distinguish "
        "overwrite from "
        "transport drop, so the report uses the combined "
        "dropped-or-overwritten count. C1/C7/C9 lost 27/26/31 model "
        "deliveries after relay; the other runs lost zero after relay. "
        "There were zero relay or model duplicates, "
        "zero out-of-order inputs, and zero unmatched model callbacks.",
        "",
        "Five warm-ups per model were excluded by their `warmup-*` identity. "
        "No measured completion was trimmed, winsorized, or equalized. "
        "Scene-specific formal denominators and occurrences are in "
        "`leftover_summary.csv` and `leftover_directions.csv`.",
        "", "## Environment and reproducible commands", "",
        "Hardware was NVIDIA GeForce RTX 4070 SUPER, driver 565.57.01, "
        "compute capability 8.9; ROS 2 Humble used Fast DDS. Model CPU sets "
        "were 0–3, 4–7, and 8–11 with two threads each; relay/replay used "
        "12–15 with four threads. Subscriber depth remained 1. Nsight "
        "Systems 2025.2.1.130 traced CUDA, NVTX, and cuDNN with sampling, "
        "backtraces, CPU context switches, and unsupported MPS GPU context-"
        "switch tracing disabled. Requested clocks were 3105/10501 MHz. "
        "Immediate effective clock observations and successful restoration "
        "are recorded per run in `run_validation.csv` and "
        "`phase3_clock_control.json`.",
        "",
        "Model config/checkpoint SHA-256 pairs are pinned in `study.yaml`: "
        f"Faster R-CNN "
        f"`{study['data']['models'][0]['model_config_sha256']}` / "
        f"`{study['data']['models'][0]['checkpoint_sha256']}`, DeepLabV3+ "
        f"`{study['data']['models'][1]['model_config_sha256']}` / "
        f"`{study['data']['models'][1]['checkpoint_sha256']}`, and DETR "
        f"`{study['data']['models'][2]['model_config_sha256']}` / "
        f"`{study['data']['models'][2]['checkpoint_sha256']}`.",
        "", "```bash",
        "ros2 run closeloop_experiments campaign mps-three-model validate "
        "studies/mps_three_model_leftover/study.yaml "
        "--artifact-root ARTIFACT_ROOT",
        "ros2 run closeloop_experiments campaign mps-three-model run "
        "studies/mps_three_model_leftover/study.yaml "
        "--artifact-root ARTIFACT_ROOT --dry-run",
        "docker exec pPerf-host bash -lc 'source /opt/ros/humble/setup.bash "
        "&& ros2 bag info /mmdetection3d_ros2/data/bag/"
        "NuScenes-v1.0-trainval-scene-1044/"
        "NuScenes-v1.0-trainval-scene-1044_0.mcap && ros2 bag info "
        "/mmdetection3d_ros2/data/bag/"
        "NuScenes-v1.0-trainval-scene-0434/"
        "NuScenes-v1.0-trainval-scene-0434_0.mcap'",
        "ros2 run closeloop_analyzer analyze mps "
        "ARTIFACT_ROOT/runs/RUN_ID --output-root ARTIFACT_ROOT",
        "```", "", "## Raw data and generated artifacts", "",
        "Raw runs: `outputs/clp3/mps3-c*`. Each includes immutable config, "
        "manifest, model status/input identities, replay intervals, "
        "communication evidence, MPS logs, clock evidence, `.nsys-rep`, "
        "SQLite trace, and formal analyzer outputs. Generated tables and "
        "figures are beside this report. `two_model_formal_comparison.csv` "
        "records the reanalyzed prior comparison.",
        "", "## Limitations and evidence boundaries", "",
        "Scene order was fixed and never reversed; the same two source "
        "scenes were reused, sample counts are unequal after depth-1 losses, "
        "and each configuration has only one independent run. Nsight "
        "Systems and the timestamp relay perturb execution; no separate "
        "uninstrumented Phase 3 latency estimate was requested. The formal "
        "predicate establishes observed temporal association, not why the "
        "scheduler admitted a kernel. Pressure and assignment statements "
        "above are descriptive correlations. No Phase 3 intervention was "
        "used to claim causation; GPU-only causal validation belongs to "
        "Phase 4.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    """Generate the Phase 3 evidence package from saved raw traces."""
    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("analysis_root")
    args = parser.parse_args(argv)
    analyze(load_study(args.study), args.analysis_root)
    return 0
