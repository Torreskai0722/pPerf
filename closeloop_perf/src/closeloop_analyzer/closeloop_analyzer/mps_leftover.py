"""Observational cross-client kernels in victim ready-to-start waits."""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple

import yaml

from .target_selection import (
    KernelRecord, TraceSelectionError, TraceTargetSelector,
)


OCCURRENCE_FIELDS = (
    "occurrence_id", "ordered_mps_percentage_triple", "direction",
    "aggressor_model", "victim_model",
    "victim_inference_input_id", "victim_scene_index", "victim_scene_name",
    "victim_scene_token", "victim_kernel_id", "victim_kernel_name",
    "victim_kernel_signature", "victim_inventory_signature",
    "victim_pid", "victim_context_id", "victim_stream_id",
    "aggressor_kernel_id", "aggressor_kernel_name",
    "aggressor_kernel_signature", "aggressor_inventory_signature",
    "aggressor_pid", "aggressor_context_id", "aggressor_stream_id",
    "ready_ns",
    "victim_start_ns", "victim_ready_to_start_delay_ns",
    "aggressor_start_ns", "aggressor_end_ns", "aggressor_duration_ns",
    "normalized_victim_start_position", "terminal_overlap_ns",
    "terminal_overlap_percentage", "ready_interval_overlap_ns",
    "co_aggressor_count",
    "capsule_replayable", "capsule_rejection_reason",
)
KERNEL_INVENTORY_FIELDS = (
    "model", "inventory_signature", "symbol", "launch_api", "grid",
    "block", "registers_per_thread", "static_shared_memory",
    "dynamic_shared_memory", "launch_type", "kernel_count",
    "kernel_count_per_inference", "distinct_inputs", "duration_p50_ns",
    "duration_p95_ns", "duration_max_ns", "summed_kernel_time_ns",
    "summed_kernel_time_per_inference_ns", "model_kernel_time_share",
    "candidate_occurrence_count", "distinct_delayed_victim_inputs",
    "candidate_overlap_p50_ns", "candidate_overlap_p95_ns",
    "candidate_overlap_max_ns", "candidate_overlap_sum_ns",
)
GROUP_FIELDS = (
    "direction", "aggressor_model", "victim_model",
    "aggressor_kernel_signature", "aggressor_kernel_name",
    "victim_kernel_signature", "victim_kernel_name", "occurrence_count",
    "distinct_victim_episode_count", "capsule_replayable_occurrence_count",
    "delay_p50_ns", "delay_p95_ns", "delay_max_ns", "overlap_p50_ns",
    "overlap_p95_ns", "overlap_max_ns", "co_aggressor_p50",
    "co_aggressor_p95", "co_aggressor_max",
)


def _percentile(values: Iterable[int], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def _stats(values: Iterable[int]) -> Dict[str, float]:
    values = list(values)
    return {
        "p50_ns": _percentile(values, 0.50),
        "p95_ns": _percentile(values, 0.95),
        "max_ns": max(values, default=0),
    }


def _variation_stats(values: Iterable[int]) -> Dict[str, float]:
    values = list(values)
    base = _stats(values)
    mean = fmean(values) if values else 0.0
    standard_deviation = pstdev(values) if values else 0.0
    return {
        "count": len(values),
        **base,
        "p99_ns": _percentile(values, 0.99),
        "mean_ns": mean,
        "population_standard_deviation_ns": standard_deviation,
        "coefficient_of_variation": (
            standard_deviation / mean if mean else 0.0
        ),
        "p95_minus_p50_ns": base["p95_ns"] - base["p50_ns"],
        "p99_minus_p50_ns": _percentile(values, 0.99) - base["p50_ns"],
    }


def formal_leftover_metrics(victim_start_ns, aggressor_start_ns,
                            aggressor_end_ns):
    """Return final-20% overlap metrics, or None when the policy fails."""
    duration_ns = aggressor_end_ns - aggressor_start_ns
    if duration_ns <= 0:
        return None
    terminal_overlap_ns = aggressor_end_ns - victim_start_ns
    normalized_start = (
        (victim_start_ns - aggressor_start_ns) / duration_ns
    )
    if terminal_overlap_ns <= 0 or not 0.80 <= normalized_start < 1.0:
        return None
    return normalized_start, terminal_overlap_ns


def compare_kernel_inventories(
    current: Iterable[Dict], baseline: Iterable[Dict]
) -> List[Dict]:
    """Return signature additions, removals, and metric deltas."""
    current = {row["inventory_signature"]: row for row in current}
    baseline = {row["inventory_signature"]: row for row in baseline}
    rows = []
    metrics = (
        "kernel_count_per_inference", "duration_p50_ns", "duration_p95_ns",
        "summed_kernel_time_per_inference_ns", "model_kernel_time_share",
    )
    for signature in sorted(set(current) | set(baseline)):
        now = current.get(signature)
        base = baseline.get(signature)
        source = now or base
        row = {
            "inventory_signature": signature,
            "symbol": source["symbol"],
            "change": (
                "added" if base is None else
                "removed" if now is None else "present"
            ),
        }
        for metric in metrics:
            current_value = float(now[metric]) if now else 0.0
            baseline_value = float(base[metric]) if base else 0.0
            row[f"current_{metric}"] = current_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"delta_{metric}"] = current_value - baseline_value
        rows.append(row)
    return rows


def _write_csv(path: Path, fields: Tuple[str, ...], rows: List[Dict]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class MpsLeftoverAnalyzer:
    """Enumerate delay episodes without treating overlap as causal proof."""

    def __init__(self, source_run: Path, delay_threshold_ms: float = 0.9,
                 output_directory: Path = None):
        if delay_threshold_ms < 0:
            raise ValueError("delay threshold must be non-negative")
        self.source_run = Path(source_run)
        self.output_directory = Path(output_directory or source_run)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.threshold_ns = int(delay_threshold_ms * 1_000_000)
        self.selector = TraceTargetSelector(
            self.source_run / "profile.sqlite", self.source_run
        )

    def _metadata(self) -> Tuple[Dict, Dict, Dict[str, Dict]]:
        manifest_path = self.source_run / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        config_path = self.source_run / "config.yaml"
        if config_path.is_file():
            raw = config_path.read_bytes()
            config = yaml.safe_load(raw) or {}
            config_sha256 = hashlib.sha256(raw).hexdigest()
        else:
            config = {}
            config_sha256 = manifest.get("config_sha256")
        statuses = {}
        for path in sorted(self.source_run.glob("model_*.json")):
            if path.name.endswith("_inputs.json"):
                continue
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
                statuses[str(status["model_id"])] = status
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        provenance = {
            "run_id": manifest.get("run_id", config.get("run", {}).get("id")),
            "run_state": manifest.get("state"),
            "config_sha256": config_sha256,
            "manifest_config_sha256": manifest.get("config_sha256"),
        }
        return config, provenance, statuses

    @staticmethod
    def _inference(kernel: KernelRecord, ranges: List[Dict]) -> Dict:
        record = TraceTargetSelector._enclosing_range(
            kernel, ranges, "inference"
        )
        if record is None:
            return None
        if str(record["tag"].get("input", "")).startswith("warmup-"):
            return None
        return record

    @staticmethod
    def _inference_index(ranges: List[Dict]) -> Dict[int, Tuple]:
        """Index nonoverlapping inference ranges by process and start time."""
        by_pid: DefaultDict[int, List[Dict]] = defaultdict(list)
        for record in ranges:
            if record["tag"].get("event") == "inference":
                by_pid[record["pid"]].append(record)
        result = {}
        for pid, records in by_pid.items():
            records.sort(key=lambda record: record["start_ns"])
            if any(
                current["start_ns"] < previous["end_ns"]
                for previous, current in zip(records, records[1:])
            ):
                raise ValueError(f"overlapping inference ranges for PID {pid}")
            result[pid] = (
                [record["start_ns"] for record in records], records
            )
        return result

    @staticmethod
    def _indexed_inference(kernel: KernelRecord, index: Dict[int, Tuple]):
        starts, records = index.get(kernel.pid, ((), ()))
        position = bisect_right(starts, kernel.launch_start_ns) - 1
        if position < 0:
            return None
        record = records[position]
        if (
            kernel.launch_end_ns <= record["end_ns"]
            and not str(record["tag"].get("input", "")).startswith(
                "warmup-"
            )
        ):
            return record
        return None

    @staticmethod
    def _kernel_signature(kernel: KernelRecord) -> str:
        return TraceTargetSelector._aggressor_signature(kernel)

    @staticmethod
    def _inventory_payload(kernel: KernelRecord) -> Dict:
        return {
            "symbol": kernel.name,
            "launch_api": kernel.launch_api,
            "grid": list(kernel.grid),
            "block": list(kernel.block),
            "registers_per_thread": kernel.registers_per_thread,
            "static_shared_memory": kernel.static_shared_memory,
            "dynamic_shared_memory": kernel.dynamic_shared_memory,
            "launch_type": kernel.launch_type,
        }

    @classmethod
    def _inventory_signature(cls, kernel: KernelRecord) -> str:
        payload = json.dumps(
            cls._inventory_payload(kernel), sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _candidate_index(kernels: List[KernelRecord]) -> Dict[int, Tuple]:
        by_pid: DefaultDict[int, List[KernelRecord]] = defaultdict(list)
        for kernel in kernels:
            by_pid[kernel.pid].append(kernel)
        result = {}
        for pid, records in by_pid.items():
            records.sort(key=lambda item: (
                item.start_ns, item.end_ns, item.kernel_id
            ))
            prefix_max = []
            maximum = 0
            for record in records:
                maximum = max(maximum, record.end_ns)
                prefix_max.append(maximum)
            result[pid] = (
                records, [record.start_ns for record in records], prefix_max
            )
        return result

    @staticmethod
    def _overlapping_candidates(
        index: Dict[int, Tuple], victim: KernelRecord, ready_ns: int
    ) -> List[KernelRecord]:
        matches = []
        for pid, (records, starts, prefix_max) in index.items():
            if pid == victim.pid:
                continue
            position = bisect_left(starts, victim.start_ns) - 1
            while position >= 0 and prefix_max[position] > ready_ns:
                candidate = records[position]
                if candidate.end_ns > ready_ns:
                    matches.append(candidate)
                position -= 1
        return sorted(matches, key=lambda item: (
            item.start_ns, item.end_ns, item.pid, item.kernel_id
        ))

    def _capsule_eligibility(
        self, victim: KernelRecord, aggressor: KernelRecord, trace: Any,
        victim_kernels: List[KernelRecord],
    ) -> Tuple[bool, str]:
        if aggressor.duration_ns < self.selector.minimum_ns:
            return False, "aggressor duration below capsule 1 ms minimum"
        try:
            self.selector._overlap_set(
                victim, aggressor, victim_kernels, trace.streams,
                trace.unresolved_times, trace.ranges,
            )
        except TraceSelectionError as error:
            return False, str(error)
        return True, ""

    def analyze(self) -> Dict[str, Any]:
        """Write all three per-run artifacts and return the JSON record."""
        trace = self.selector.load_trace()
        config, provenance, statuses = self._metadata()
        configured_models = {
            str(model["id"]): model for model in config.get("models", [])
        }
        ordered_mps_percentage_triple = ",".join(
            str(model.get("mps_percentage", "unset"))
            for model in config.get("models", [])
        )
        model_ids = sorted(
            set(configured_models) | set(self.selector.models.values())
        )
        inference_index = self._inference_index(trace.ranges)
        input_records = self.selector._model_input_records()

        eligible: DefaultDict[str, List[KernelRecord]] = defaultdict(list)
        delayed: DefaultDict[str, List[KernelRecord]] = defaultdict(list)
        eligible_by_scene: DefaultDict[str, DefaultDict[str, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        delayed_by_scene: DefaultDict[str, DefaultDict[str, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        inferences: DefaultDict[str, Dict[Tuple, Dict]] = defaultdict(dict)
        inference_kernels: DefaultDict[str, DefaultDict[Tuple, List]] = (
            defaultdict(lambda: defaultdict(list))
        )
        inference_for_kernel = {}
        for kernel in trace.kernels:
            inference = self._indexed_inference(kernel, inference_index)
            if inference is None:
                continue
            model_id = str(
                inference["tag"].get("model")
                or self.selector.models.get(kernel.pid, kernel.pid)
            )
            eligible[model_id].append(kernel)
            input_id = str(inference["tag"].get("input", ""))
            input_record = input_records.get((model_id, input_id), {})
            scene_name = str(
                input_record.get("scene_name")
                or input_record.get("scene_token")
                or "unknown"
            )
            eligible_by_scene[model_id][scene_name] += 1
            inference_key = (
                inference["start_ns"], inference["end_ns"], input_id
            )
            inferences[model_id][inference_key] = inference
            inference_kernels[model_id][inference_key].append(kernel)
            if (
                kernel.start_ns - trace.ready_times[kernel.kernel_id]
                > self.threshold_ns
            ):
                delayed[model_id].append(kernel)
                inference_for_kernel[kernel.kernel_id] = inference
                delayed_by_scene[model_id][scene_name] += 1
        model_ids = sorted(set(model_ids) | set(eligible))

        candidate_index = self._candidate_index([
            kernel for records in eligible.values() for kernel in records
        ])
        kernels_by_pid: DefaultDict[int, List[KernelRecord]] = defaultdict(
            list
        )
        for kernel in trace.kernels:
            kernels_by_pid[kernel.pid].append(kernel)
        occurrences = []
        for victim_model in sorted(delayed):
            for victim in sorted(delayed[victim_model], key=lambda item: (
                item.start_ns, item.kernel_id
            )):
                ready_ns = trace.ready_times[victim.kernel_id]
                aggressors = self._overlapping_candidates(
                    candidate_index, victim, ready_ns
                )
                affected_aggressors = []
                for aggressor in aggressors:
                    metrics = formal_leftover_metrics(
                        victim.start_ns, aggressor.start_ns, aggressor.end_ns
                    )
                    if metrics is not None:
                        normalized_start, terminal_overlap_ns = metrics
                        affected_aggressors.append((
                            aggressor, normalized_start, terminal_overlap_ns
                        ))
                for (
                    aggressor, normalized_start, terminal_overlap_ns
                ) in affected_aggressors:
                    aggressor_model = self.selector.models.get(
                        aggressor.pid, str(aggressor.pid)
                    )
                    victim_input_id = str(
                        inference_for_kernel[victim.kernel_id]["tag"].get(
                            "input", ""
                        )
                    )
                    victim_input = input_records.get(
                        (victim_model, victim_input_id), {}
                    )
                    replayable, reason = self._capsule_eligibility(
                        victim, aggressor, trace, kernels_by_pid[victim.pid]
                    )
                    overlap_ns = min(
                        aggressor.end_ns, victim.start_ns
                    ) - max(aggressor.start_ns, ready_ns)
                    occurrences.append({
                        "ordered_mps_percentage_triple": (
                            ordered_mps_percentage_triple
                        ),
                        "direction": f"{aggressor_model}->{victim_model}",
                        "aggressor_model": aggressor_model,
                        "victim_model": victim_model,
                        "victim_inference_input_id": victim_input_id,
                        "victim_scene_index": victim_input.get(
                            "scene_index"
                        ),
                        "victim_scene_name": victim_input.get("scene_name"),
                        "victim_scene_token": victim_input.get("scene_token"),
                        "victim_kernel_id": victim.kernel_id,
                        "victim_kernel_name": victim.name,
                        "victim_kernel_signature": self._kernel_signature(
                            victim
                        ),
                        "victim_inventory_signature": (
                            self._inventory_signature(victim)
                        ),
                        "victim_pid": victim.pid,
                        "victim_context_id": victim.context_id,
                        "victim_stream_id": victim.stream_id,
                        "aggressor_kernel_id": aggressor.kernel_id,
                        "aggressor_kernel_name": aggressor.name,
                        "aggressor_kernel_signature": self._kernel_signature(
                            aggressor
                        ),
                        "aggressor_inventory_signature": (
                            self._inventory_signature(aggressor)
                        ),
                        "aggressor_pid": aggressor.pid,
                        "aggressor_context_id": aggressor.context_id,
                        "aggressor_stream_id": aggressor.stream_id,
                        "ready_ns": ready_ns,
                        "victim_start_ns": victim.start_ns,
                        "victim_ready_to_start_delay_ns": (
                            victim.start_ns - ready_ns
                        ),
                        "aggressor_start_ns": aggressor.start_ns,
                        "aggressor_end_ns": aggressor.end_ns,
                        "aggressor_duration_ns": aggressor.duration_ns,
                        "normalized_victim_start_position": normalized_start,
                        "terminal_overlap_ns": terminal_overlap_ns,
                        "terminal_overlap_percentage": (
                            terminal_overlap_ns / aggressor.duration_ns
                        ),
                        "ready_interval_overlap_ns": overlap_ns,
                        "co_aggressor_count": len(affected_aggressors),
                        "capsule_replayable": replayable,
                        "capsule_rejection_reason": reason,
                    })
        occurrences.sort(key=lambda row: (
            row["victim_start_ns"], row["victim_kernel_id"],
            row["aggressor_start_ns"], row["aggressor_kernel_id"]
        ))
        for number, row in enumerate(occurrences, 1):
            row["occurrence_id"] = number

        grouped: DefaultDict[Tuple, List[Dict]] = defaultdict(list)
        for row in occurrences:
            grouped[(
                row["direction"], row["aggressor_kernel_signature"],
                row["victim_kernel_signature"],
            )].append(row)
        groups = []
        for key in sorted(grouped):
            rows = grouped[key]
            delay_stats = _stats(
                row["victim_ready_to_start_delay_ns"] for row in rows
            )
            overlap_stats = _stats(
                row["ready_interval_overlap_ns"] for row in rows
            )
            co_aggressors = [row["co_aggressor_count"] for row in rows]
            groups.append({
                "direction": key[0],
                "aggressor_model": rows[0]["aggressor_model"],
                "victim_model": rows[0]["victim_model"],
                "aggressor_kernel_signature": key[1],
                "aggressor_kernel_name": rows[0]["aggressor_kernel_name"],
                "victim_kernel_signature": key[2],
                "victim_kernel_name": rows[0]["victim_kernel_name"],
                "occurrence_count": len(rows),
                "distinct_victim_episode_count": len({
                    row["victim_kernel_id"] for row in rows
                }),
                "capsule_replayable_occurrence_count": sum(
                    bool(row["capsule_replayable"]) for row in rows
                ),
                "delay_p50_ns": delay_stats["p50_ns"],
                "delay_p95_ns": delay_stats["p95_ns"],
                "delay_max_ns": delay_stats["max_ns"],
                "overlap_p50_ns": overlap_stats["p50_ns"],
                "overlap_p95_ns": overlap_stats["p95_ns"],
                "overlap_max_ns": overlap_stats["max_ns"],
                "co_aggressor_p50": _percentile(co_aggressors, 0.50),
                "co_aggressor_p95": _percentile(co_aggressors, 0.95),
                "co_aggressor_max": max(co_aggressors),
            })

        directions = []
        for aggressor_model in model_ids:
            for victim_model in model_ids:
                if aggressor_model == victim_model:
                    continue
                rows = [
                    row for row in occurrences
                    if row["aggressor_model"] == aggressor_model
                    and row["victim_model"] == victim_model
                ]
                victim_delayed = delayed.get(victim_model, [])
                denominator = len(eligible.get(victim_model, []))
                affected_victims = len({
                    row["victim_kernel_id"] for row in rows
                })
                directions.append({
                    "direction": f"{aggressor_model}->{victim_model}",
                    "aggressor_model": aggressor_model,
                    "victim_model": victim_model,
                    "eligible_victim_kernel_count": denominator,
                    "delayed_victim_kernel_count": len(victim_delayed),
                    "delayed_victim_kernel_rate": (
                        len(victim_delayed) / denominator
                        if denominator else 0.0
                    ),
                    "associated_delayed_victim_kernel_count": len({
                        row["victim_kernel_id"] for row in rows
                    }),
                    "eligible_delayed_victim_kernel_count": len(
                        victim_delayed
                    ),
                    "affected_leftover_policy_kernel_count": affected_victims,
                    "affected_kernel_rate_relative_to_eligible_delayed": (
                        affected_victims / len(victim_delayed)
                        if victim_delayed else 0.0
                    ),
                    "occurrences_per_eligible_delayed_kernel": (
                        len(rows) / len(victim_delayed)
                        if victim_delayed else 0.0
                    ),
                    "occurrence_count": len(rows),
                    "evidence_grade": (
                        "observed" if rows else "not_observed_in_trace"
                    ),
                    "distinct_kernel_pair_group_count": sum(
                        group["direction"]
                        == f"{aggressor_model}->{victim_model}"
                        for group in groups
                    ),
                    "victim_delay": _stats(
                        row["victim_ready_to_start_delay_ns"] for row in rows
                    ),
                    "ready_interval_overlap": _stats(
                        row["ready_interval_overlap_ns"] for row in rows
                    ),
                })

        model_results = {}
        mps = {}
        inventory_rows = []
        for model_id in model_ids:
            status = statuses.get(model_id, {})
            model = configured_models.get(model_id, {})
            inference_values = list(inferences.get(model_id, {}).values())
            inference_durations = [
                record["end_ns"] - record["start_ns"]
                for record in inference_values
            ]
            gpu_complete_durations = [
                max(
                    record["end_ns"],
                    max(
                        (kernel.end_ns for kernel in inference_kernels[
                            model_id
                        ][key]),
                        default=record["end_ns"],
                    ),
                ) - record["start_ns"]
                for key, record in inferences.get(model_id, {}).items()
            ]
            model_results[model_id] = {
                "completed_inputs": status.get("inputs"),
                "inference_count": len(inference_values),
                "inference_duration": _stats(inference_durations),
                "gpu_complete_inference_duration": _variation_stats(
                    gpu_complete_durations
                ),
                "gpu_complete_inference_durations_ns": (
                    gpu_complete_durations
                ),
                "eligible_kernel_count": len(eligible.get(model_id, [])),
                "delayed_kernel_count": len(delayed.get(model_id, [])),
                "delayed_kernel_rate": (
                    len(delayed.get(model_id, [])) /
                    len(eligible.get(model_id, []))
                    if eligible.get(model_id) else 0.0
                ),
                "per_scene": {
                    scene_name: {
                        "eligible_kernel_count": count,
                        "delayed_kernel_count": delayed_by_scene[
                            model_id
                        ].get(scene_name, 0),
                    }
                    for scene_name, count in sorted(
                        eligible_by_scene[model_id].items()
                    )
                },
            }
            mps[model_id] = {
                "configured": model.get(
                    "mps_percentage",
                    status.get("configured_cuda_mps_active_thread_percentage"),
                ),
                "process_observed": status.get(
                    "process_observed_cuda_mps_active_thread_percentage"
                ),
            }

            by_signature: DefaultDict[str, List[KernelRecord]] = defaultdict(
                list
            )
            inputs_by_signature: DefaultDict[str, set] = defaultdict(set)
            for key, kernels in inference_kernels.get(model_id, {}).items():
                for kernel in kernels:
                    signature = self._inventory_signature(kernel)
                    by_signature[signature].append(kernel)
                    inputs_by_signature[signature].add(key[2])
            model_kernel_time = sum(
                kernel.duration_ns for kernel in eligible.get(model_id, [])
            )
            inference_count = len(inferences.get(model_id, {}))
            for signature in sorted(by_signature):
                kernels = by_signature[signature]
                payload = self._inventory_payload(kernels[0])
                candidate_rows = [
                    row for row in occurrences
                    if row["aggressor_model"] == model_id
                    and row["aggressor_inventory_signature"] == signature
                ]
                durations = [kernel.duration_ns for kernel in kernels]
                duration_stats = _stats(durations)
                overlap_stats = _stats(
                    row["ready_interval_overlap_ns"]
                    for row in candidate_rows
                )
                total_duration = sum(durations)
                inventory_rows.append({
                    "model": model_id,
                    "inventory_signature": signature,
                    **payload,
                    "grid": "x".join(str(value) for value in payload["grid"]),
                    "block": "x".join(
                        str(value) for value in payload["block"]
                    ),
                    "kernel_count": len(kernels),
                    "kernel_count_per_inference": (
                        len(kernels) / inference_count
                        if inference_count else 0.0
                    ),
                    "distinct_inputs": len(inputs_by_signature[signature]),
                    "duration_p50_ns": duration_stats["p50_ns"],
                    "duration_p95_ns": duration_stats["p95_ns"],
                    "duration_max_ns": duration_stats["max_ns"],
                    "summed_kernel_time_ns": total_duration,
                    "summed_kernel_time_per_inference_ns": (
                        total_duration / inference_count
                        if inference_count else 0.0
                    ),
                    "model_kernel_time_share": (
                        total_duration / model_kernel_time
                        if model_kernel_time else 0.0
                    ),
                    "candidate_occurrence_count": len(candidate_rows),
                    "distinct_delayed_victim_inputs": len({
                        row["victim_inference_input_id"]
                        for row in candidate_rows
                    }),
                    "candidate_overlap_p50_ns": overlap_stats["p50_ns"],
                    "candidate_overlap_p95_ns": overlap_stats["p95_ns"],
                    "candidate_overlap_max_ns": overlap_stats["max_ns"],
                    "candidate_overlap_sum_ns": sum(
                        row["ready_interval_overlap_ns"]
                        for row in candidate_rows
                    ),
                })

        result = {
            "schema": "mps_leftover_analysis_v3",
            "schema_version": 3,
            "provenance": provenance,
            "configuration": {
                "delay_threshold_ms": self.threshold_ns / 1_000_000,
                "delay_threshold_comparison": "strictly_greater_than",
                "victim_readiness": (
                    "max(host_launch_completion,"
                    "previous_same_stream_kernel_completion)"
                ),
                "victim_filter": "non_warmup_inference_kernels",
                "association": (
                    "foreign_process_kernel_executing_at_victim_start_"
                    "and_victim_start_in_aggressor_final_20_percent"
                ),
                "normalized_start_interval": "0.80_inclusive_to_1_exclusive",
                "terminal_overlap_comparison": "strictly_positive",
                "ordered_mps_percentage_triple": (
                    ordered_mps_percentage_triple
                ),
                "causal_attribution": False,
                "capsule_eligibility_is_observational_only": True,
            },
            "mps_percentages": mps,
            "models": model_results,
            "directions": directions,
            "occurrence_count": len(occurrences),
            "group_count": len(groups),
            "groups": groups,
            "kernel_inventory": inventory_rows,
        }
        _write_csv(
            self.output_directory / "mps_leftover_occurrences.csv",
            OCCURRENCE_FIELDS, occurrences,
        )
        _write_csv(
            self.output_directory / "mps_leftover_groups.csv",
            GROUP_FIELDS, groups
        )
        _write_csv(
            self.output_directory / "mps_aggressor_kernels.csv",
            KERNEL_INVENTORY_FIELDS, inventory_rows,
        )
        output = self.output_directory / "mps_leftover_analysis.json"
        temporary = output.with_name("." + output.name + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return result


def main(argv=None) -> int:
    """Analyze one completed source run without capturing or replaying."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run")
    parser.add_argument("--delay-threshold-ms", type=float, default=0.9)
    args = parser.parse_args(argv)
    MpsLeftoverAnalyzer(
        Path(args.source_run), args.delay_threshold_ms
    ).analyze()
    return 0
