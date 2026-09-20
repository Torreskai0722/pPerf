"""Read-only integrity, input coverage, inference, and kernel evidence."""

from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
import yaml

from .._common import nvtx_ranges, runtime_events, tables


def read_json(path):
    """Read one retained JSON record."""
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    """Keep all observations, including failed or duplicate inputs."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha256(path):
    """Hash evidence without materializing large traces in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically write analyzer output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    """Write a flat, portable analysis table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        writer = csv.DictWriter(output, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def profile_connection(run):
    """Restore SQLite outside the retained run; never change historical files."""
    run = Path(run)
    with tempfile.TemporaryDirectory(prefix="input2-sqlite-") as temporary:
        path = run / "profile.sqlite"
        if not path.is_file():
            path = Path(temporary) / "profile.sqlite"
            with gzip.open(run / "profile.sqlite.gz", "rb") as source, path.open("wb") as out:
                shutil.copyfileobj(source, out)
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            yield connection
        finally:
            connection.close()


def archive_index(run):
    """Verify compressed hashes, gzip CRC, restored hashes, and SQLite integrity."""
    run = Path(run)
    retained = read_json(run / "profile_archive.json")
    result = {}
    for name in ("profile.nsys-rep", "profile.sqlite"):
        expected = retained[name]
        # Historical manifests can precede an artifact-root relocation.
        path = run / Path(expected["path"]).name
        observed = sha256(path)
        if observed != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"archive size/hash mismatch: {path}")
        digest = hashlib.sha256()
        with gzip.open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        restored = digest.hexdigest()
        if expected.get("restored_sha256", restored) != restored:
            raise ValueError(f"archive restored hash mismatch: {path}")
        if (run / name).exists() and sha256(run / name) != restored:
            raise ValueError(f"uncompressed trace differs from archive: {path}")
        result[name] = {"path": str(path), "sha256": observed,
                        "restored_sha256": restored, "bytes": path.stat().st_size}
    with profile_connection(run) as connection:
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("SQLite integrity check failed")
        if not {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"} <= tables(connection):
            raise ValueError("trace lacks NVTX or kernel evidence")
    return result


def normalized_protocol(config):
    """Compare settings independently of execution IDs and ROS domain numbers."""
    config = json.loads(json.dumps(config))
    replay = config["replay"]
    for key in ("metadata_path", "bag_directory", "scene_token", "controlled_bag_manifest", "controlled_bag_manifest_sha256"):
        replay.pop(key, None)
    if "recording" not in config and "nsys" in config:
        nsys = dict(config["nsys"])
        nsys.pop("enabled", None)
        config["recording"] = {"level": "level1", "scopes": ["model", "input"], "nsys": nsys}
    ros = config["ros"]
    ros.pop("domain_id", None)
    return {key: config.get(key) for key in ("replay", "models", "recording", "gpu", "ros")}


def inspect_execution(run, expected, hardware):
    """Validate reuse from recorded evidence; collect every concrete rejection."""
    run = Path(run)
    reasons, evidence = [], {}
    exclusions = run.parent.parent / "generated_configs/input2-crossed/execution_exclusions.json"
    if exclusions.is_file():
        excluded = read_json(exclusions).get(run.name)
        if excluded:
            reasons.append("recorded execution exclusion: " + excluded["reason"])
    try:
        manifest = read_json(run / "run_manifest.json")
        config = yaml.safe_load((run / "config.yaml").read_text())
        evidence.update(started_at=manifest.get("started_at"), finished_at=manifest.get("finished_at"))
        if manifest.get("state") != "success":
            reasons.append("run state is not success")
        if sha256(run / "config.yaml") != manifest.get("config_sha256"):
            reasons.append("recorded configuration hash mismatch")
        if normalized_protocol(config) != normalized_protocol(expected):
            reasons.append("replay/model/resource/profiling protocol differs")
        for key in ("gpu_hardware", "cpu_topology"):
            if manifest.get(key) != hardware.get(key):
                reasons.append(f"{key} differs from frozen study hardware")
        testbed = read_json(run / "testbed_result.json")
        intervals = testbed.get("playback_intervals", [])
        if (not testbed.get("replay_success") or not testbed.get("all_acknowledged")
                or testbed.get("error") or testbed.get("repeat_count") != 1
                or len(intervals) != 1 or intervals[0].get("completion_status") != "completed"
                or not intervals[0].get("resume_monotonic_ns")):
            reasons.append("replay/readiness/completion boundary evidence is incomplete")
        controlled_path = config["replay"].get("controlled_bag_manifest")
        target_bag = read_json(expected["replay"]["controlled_bag_manifest"])
        if controlled_path:
            bag = read_json(controlled_path)
            if sha256(controlled_path) != config["replay"].get("controlled_bag_manifest_sha256"):
                reasons.append("controlled manifest hash mismatch")
            if (not bag.get("validation", {}).get("valid")
                    or sha256(bag["frame_index"]) != bag["frame_index_sha256"]):
                reasons.append("bag validation/frame index differs")
            if intervals and (intervals[0].get("bag_path") != bag["bag_path"]
                              or intervals[0].get("input_scenes") != bag["input_scenes"]
                              or intervals[0].get("window_ns") != bag["window_ns"]):
                reasons.append("actual replay bag/scenes/window differ")
            if config["replay"].get("controlled_bag_manifest_sha256") != expected["replay"].get("controlled_bag_manifest_sha256"):
                reasons.append("controlled input payload/timing identity differs")
        else:
            sources = target_bag["sources"]
            durations = sorted({s["duration_ns"] for s in sources.values()})
            evidence["input_window_comparison"] = {"source_durations_ns": durations,
                "required_duration_ns": target_bag["window_ns"], "retrospective_trimming": False}
            if durations != [target_bag["window_ns"]]:
                reasons.append(f"full-scene window {durations} ns differs from required {target_bag['window_ns']} ns; no retrospective trimming")
            if len({s["scene_id"] for s in sources.values()}) != 1:
                reasons.append("single-scene historical bag cannot supply crossed scenes")
            source = sources["lidar"]
            original_bag = Path(intervals[0]["bag_path"]) if intervals else None
            if (original_bag is None or len(source["files"]) != 1
                    or sha256(original_bag) not in source["files"].values()):
                reasons.append("historical replay bag does not match verified source hash")
            if config["replay"]["scene_token"] != source["scene_token"]:
                reasons.append("historical input scene differs")
        clock = read_json(run / "input_variation_clock_control.json")
        for field in ("graphics_clock_mhz", "memory_clock_mhz"):
            if clock.get("requested_" + field) != expected["gpu"][field]:
                reasons.append(f"clock setting differs: {field}")
        commands = clock.get("control", {}).get("lock_commands", [])
        if len(commands) != 2 or any(c.get("returncode") != 0 for c in commands):
            reasons.append("GPU clock lock was not established")
        relay_status = read_json(run / "communication_relay_status.json")
        if relay_status.get("messages") != len(read_jsonl(run / "communication_relay.jsonl")):
            reasons.append("relay count differs between publication log and final status")
        for field in ("cpu_affinity", "cpu_thread_count"):
            if relay_status.get(field) != expected["replay"][field]:
                reasons.append(f"observed relay/replay resource {field} differs")
        for model in expected["models"]:
            status = manifest.get("models", {}).get(model["id"], {})
            if status.get("state") != "acknowledged" or status.get("error"):
                reasons.append(f"model not completed: {model['id']}")
            for field in ("model_config_sha256", "checkpoint_sha256", "cpu_affinity", "cpu_thread_count", "input_queue_depth"):
                if status.get(field) != model.get(field):
                    reasons.append(f"observed {model['id']} {field} differs")
            for library, count in status.get("observed_library_thread_counts", {}).items():
                if count != (1 if library == "pytorch_interop" else model["cpu_thread_count"]):
                    reasons.append(f"observed {model['id']} {library} thread count differs")
            percentage = "100" if expected["gpu"]["mps_enabled"] else None
            if status.get("process_observed_cuda_mps_active_thread_percentage") != percentage:
                reasons.append(f"observed {model['id']} MPS percentage differs")
            inputs = read_jsonl(run / f"model_{model['id']}_inputs.jsonl")
            completed = [record for record in inputs if record.get("completed")]
            if not completed:
                reasons.append(f"no completed model inputs: {model['id']}")
            elif any(not record.get("inference_completion_monotonic_ns") for record in completed):
                reasons.append(f"{model['id']} lacks directly recorded CUDA-completion monotonic boundaries for the common throughput/drain protocol")
            if status.get("inputs") != len(completed):
                reasons.append(f"{model['id']} completed count differs between callback log and final status")
        evidence["archives"] = archive_index(run)
        files = ["run_manifest.json", "config.yaml", "testbed_result.json",
                 "communication_relay.jsonl", "communication_relay_status.json",
                 "input_variation_clock_control.json", "profile_archive.json"]
        files += [f"model_{m['id']}_inputs.jsonl" for m in expected["models"]]
        evidence["file_hashes"] = {name: sha256(run / name) for name in files}
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        reasons.append(f"missing/invalid recorded evidence: {exc}")
    return {"accepted": not reasons, "reasons": reasons, **evidence}


def metrics(values, unique_count, elapsed_seconds):
    """Use linear, untrimmed percentiles; repetitions remain separate."""
    values = np.asarray(values, dtype=float)
    if not len(values) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("completed inference latencies must be positive and finite")
    p50, p99 = map(float, np.percentile(values, [50, 99], method="linear"))
    count = len(values)
    lag1 = (float(np.corrcoef(values[:-1], values[1:])[0, 1])
            if count > 2 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0 else None)
    return {"P50_ms": p50, "P99_ms": p99, "P99_minus_P50_ms": p99 - p50,
            "R": (p99 - p50) / p50, "completed_count": count,
            "unique_source_count": unique_count, "elapsed_seconds": elapsed_seconds,
            "throughput_hz": count / elapsed_seconds if elapsed_seconds else None,
            "sample_warning": "especially fragile P99 (<100)" if count < 100 else
            "sparse-tail P99 (<1000)" if count < 1000 else "",
            "latency_lag1_autocorrelation": lag1,
            "temporal_dependence": "sequential frames; not independent repetitions"}


def coverage_rows(frames, relay, inputs, model):
    """Join expected source frames to observed publications and callbacks."""
    topic = "/LIDAR_TOP" if model["modality"] == "lidar" else "/CAM_FRONT/image_rect_compressed"
    expected = [row for row in frames if row["topic"] == topic]
    source = defaultdict(list)
    for row in expected:
        source[row["output_header_timestamp_ns"]].append(row)
    pubs, callbacks = defaultdict(list), defaultdict(list)
    unmatched = []
    for kind, records, field, output in (
        ("relay", relay, "original_source_timestamp_ns", pubs),
        ("model", inputs, "ros_header_timestamp_ns", callbacks),
    ):
        for record in records:
            if kind == "relay" and record.get("source_topic") != topic:
                continue
            timestamp = record.get(field)
            if len(source.get(timestamp, [])) != 1:
                unmatched.append({"boundary": kind, "reason": "absent or ambiguous source timestamp",
                                  "candidate_source_frame_ids": [r["source_frame_id"] for r in source.get(timestamp, [])],
                                  "record": record})
                continue
            output[timestamp].append(record)
    rows = []
    for timestamp, records in source.items():
        for frame in records:
            received, published = pubs[timestamp], [r for r in pubs[timestamp] if r.get("relay_post_publish_monotonic_ns")]
            model_inputs = callbacks[timestamp]
            completed = [r for r in model_inputs if r.get("completed")]
            rows.append({**frame, "model_id": model["id"],
                         "expected_bag_inputs": 1, "relay_received": len(received),
                         "relay_published": len(published), "model_received": len(model_inputs),
                         "relay_publication_failed": len(received) - len(published),
                         "completed_inferences": len(completed),
                         "upstream_missing": int(not received),
                         "dropped_or_overwritten": int(bool(published) and not model_inputs),
                         "failed": len(model_inputs) - len(completed),
                         "relay_duplicates": max(0, len(received) - 1),
                         "model_duplicates": max(0, len(model_inputs) - 1)})
    return rows, unmatched, {t: r[0] for t, r in source.items() if len(r) == 1}


def export_kernels(connection, destination, ranges, frames, model_pids):
    """Attribute by process and correlated launch inside recorded NVTX ranges."""
    columns = {r[1] for r in connection.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    names = dict(connection.execute("SELECT id,value FROM StringIds"))
    fields = [f for f in ("start", "end", "globalPid", "deviceId", "contextId", "streamId",
                          "correlationId", "demangledName", "shortName", "mangledName") if f in columns]
    launches = defaultdict(list)
    for events in runtime_events(connection).values():
        for event in events:
            if "Launch" in event["name"]:
                launches[(event["pid"], event["correlation"])].append(event)
    by_pid = defaultdict(list)
    for record in ranges:
        by_pid[record["pid"]].append(record)
    range_indexes = {}
    for pid, records in by_pid.items():
        records.sort(key=lambda r: r["start"])
        maximum, ends = 0, []
        for record in records:
            maximum = max(maximum, record["end"])
            ends.append(maximum)
        range_indexes[pid] = (records, [r["start"] for r in records], ends)
    by_frame = {(r["model_id"], str(r["input_id"])): r for r in frames}
    totals = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt") as output:
        writer = csv.DictWriter(output, fieldnames=fields + ["kernel_name", "duration_ns", "pid", "model_id",
            "input_id", "source_frame_id", "module", "launch_start_ns", "attribution", "unresolved_reason"])
        writer.writeheader()
        for values in connection.execute(f"SELECT {','.join(fields)} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
            row = dict(zip(fields, values))
            pid = (int(row.get("globalPid") or 0) >> 24) & 0xFFFFFF
            model_id = model_pids.get(pid, "")
            candidates = (launches[(pid, row["correlationId"])]
                          if row.get("correlationId") is not None else [])
            candidates = [r for r in candidates if r["start"] <= row["start"]]
            launch = max(candidates, key=lambda r: r["start"]) if candidates else None
            scopes = []
            if launch and pid in range_indexes:
                records, starts, ends = range_indexes[pid]
                index = bisect_right(starts, launch["start"]) - 1
                while index >= 0 and ends[index] > launch["start"]:
                    if records[index]["end"] > launch["start"]:
                        scopes.append(records[index])
                    index -= 1
            scopes.sort(key=lambda r: r["end"] - r["start"])
            inference = next((r for r in scopes if r["tag"].get("event") in ("inference", "preprocess")), None)
            input_id = str(inference["tag"].get("input", "")) if inference else ""
            frame = by_frame.get((model_id, input_id), {})
            module = next((r["tag"].get("module", "") for r in scopes if r["tag"].get("event") == "module"), "")
            duration = row["end"] - row["start"]
            row.update(kernel_name=names.get(row.get("demangledName"), names.get(row.get("shortName"), "")),
                       duration_ns=duration, pid=pid, model_id=model_id, input_id=input_id,
                       source_frame_id=frame.get("source_frame_id", ""), module=module,
                       launch_start_ns=launch["start"] if launch else "",
                       attribution="correlated_launch_nvtx" if frame else "process_only" if model_id else "unresolved",
                       unresolved_reason="" if frame else "no completed non-warmup source attribution")
            writer.writerow(row)
            for label, present in (("all", True), ("model", bool(model_id)), ("input", bool(frame)),
                                   ("module", bool(module)), ("unresolved_model", not model_id),
                                   ("unresolved_input", not frame), ("unresolved_module", not module)):
                if present:
                    totals[label + "_kernel_count"] += 1
                    totals[label + "_gpu_duration_ns"] += duration
    for label in ("model", "input", "module", "unresolved_model", "unresolved_input", "unresolved_module"):
        for unit in ("kernel_count", "gpu_duration_ns"):
            totals.setdefault(label + "_" + unit, 0)
            totals[label + "_" + unit + "_fraction"] = totals[label + "_" + unit] / max(1, totals["all_" + unit])
    return dict(totals)


def analyze_execution(entry, output_root):
    """Extract completed inference ranges and all input coverage for one run."""
    run = Path(entry["artifact_directory"])
    config = yaml.safe_load((run / "config.yaml").read_text())
    bag = read_json(config["replay"]["controlled_bag_manifest"])
    source_frames = read_jsonl(bag["frame_index"])
    relay = read_jsonl(run / "communication_relay.jsonl")
    testbed = read_json(run / "testbed_result.json")
    resume = testbed["playback_intervals"][0]["resume_monotonic_ns"]
    end = resume + bag["window_ns"]
    manifest = read_json(run / "run_manifest.json")
    destination = Path(output_root) / "analysis/runs" / run.name / "input-data"
    frame_rows, coverage, unmatched, summaries = [], [], [], []
    with profile_connection(run) as connection:
        ranges = nvtx_ranges(connection)
        runtimes = [r for events in runtime_events(connection).values() for r in events]
        for model in config["models"]:
            model_id = model["id"]
            inputs = read_jsonl(run / f"model_{model_id}_inputs.jsonl")
            cover, missing, source = coverage_rows(source_frames, relay, inputs, model)
            coverage.extend(cover)
            unmatched.extend({"model_id": model_id, **r} for r in missing)
            pid = manifest["models"][model_id]["pid"]
            own_scene = bag["input_scenes"][model["modality"]]
            inference = defaultdict(list)
            preprocessing = defaultdict(list)
            for item in ranges:
                if item["pid"] == pid and item["tag"].get("model") == model_id:
                    if item["tag"].get("event") == "inference":
                        inference[str(item["tag"].get("input"))].append(item)
                    elif item["tag"].get("event") == "preprocess":
                        preprocessing[str(item["tag"].get("input"))].append(item)
            completed = []
            seen = set()
            syncs = sorted([r for r in runtimes if r["pid"] == pid and r["name"] in
                            ("cudaEventSynchronize", "cuEventSynchronize", "cudaDeviceSynchronize")], key=lambda r: r["start"])
            sync_starts = [r["start"] for r in syncs]
            for record in inputs:
                if not record.get("completed"):
                    continue
                key = str(record["input_id"])
                if key in seen or len(inference[key]) != 1:
                    raise ValueError(f"ambiguous completed inference range: {model_id}/{key}")
                seen.add(key)
                item = inference[key][0]
                if item["tag"].get("scene") != own_scene["scene_token"]:
                    raise ValueError(f"NVTX own-input scene differs: {model_id}/{key}")
                stop = bisect_right(sync_starts, item["end"])
                sync = syncs[stop - 1] if stop else None
                if not sync or sync["start"] < item["start"] or sync["end"] > item["end"]:
                    raise ValueError(f"inference lacks recorded CUDA completion: {model_id}/{key}")
                identity = source.get(record["ros_header_timestamp_ns"])
                if identity is None:
                    raise ValueError(f"completed input lacks unique source identity: {model_id}/{key}")
                pre = preprocessing[key]
                row = {"run_id": run.name, "model_id": model_id, "input_id": key,
                       "source_frame_id": identity["source_frame_id"], "scene_id": own_scene["scene_id"],
                       "latency_ms": (item["end"] - item["start"]) / 1e6,
                       "inference_start_ns": item["start"], "inference_end_ns": item["end"],
                       "completion_boundary": "NVTX inference includes CUDA event synchronization",
                       "preprocess_ms": sum(r["end"] - r["start"] for r in pre) / 1e6,
                       "decode_ms": (record["decode_end_monotonic_ns"] - record["decode_start_monotonic_ns"]) / 1e6,
                       "completion_monotonic_ns": record["inference_completion_monotonic_ns"]}
                completed.append(row)
            last = max((r["completion_monotonic_ns"] for r in completed), default=end)
            elapsed = (max(end, last) - resume) / 1e9
            summary = {"run_id": run.name, "model_id": model_id, "scene_id": own_scene["scene_id"],
                       **metrics([r["latency_ms"] for r in completed], len({r["source_frame_id"] for r in completed}), elapsed),
                       "drain_seconds": max(0, last - end) / 1e9,
                       "decode_median_ms": float(np.median([r["decode_ms"] for r in completed])),
                       "preprocess_median_ms": float(np.median([r["preprocess_ms"] for r in completed])),
                       "coverage": {key: sum(r[key] for r in cover) for key in (
                           "expected_bag_inputs", "relay_received", "relay_published", "relay_publication_failed", "model_received",
                           "completed_inferences", "upstream_missing", "dropped_or_overwritten", "failed",
                           "relay_duplicates", "model_duplicates")}, "unmatched_count": len(missing)}
            summaries.append(summary)
            frame_rows.extend(completed)
        kernel_coverage = export_kernels(connection, destination / "kernels.csv.gz", ranges, frame_rows,
                                        {s["pid"]: name for name, s in manifest["models"].items()})
    write_csv(destination / "frames.csv", frame_rows)
    write_json(destination / "frames.json", frame_rows)
    write_csv(destination / "frame_coverage.csv", coverage)
    write_json(destination / "unmatched.json", unmatched)
    result = {"run_id": run.name, "models": summaries, "kernel_attribution": kernel_coverage,
              "layer_limitation": "Depth-zero annotations do not identify individual layers. Layer hooks and correlated launches would be required; no extra runs scheduled.",
              "publication_boundary": "observed relay publication; bag schedule is expected input only"}
    write_json(destination / "summary.json", result)
    return result, frame_rows
