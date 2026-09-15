"""Ingest targeted full-stack CTA records into canonical artifacts."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

SM_COUNT = 56
CROSS_CLIENT_SMID_COMPARABLE = False
SMID_SCOPE = "mps_client_local"
MODEL_ROLES = {
    "faster_rcnn": "victim",
    "deeplabv3plus": "aggressor",
}


CTA_DATABASE_DDL = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE TABLE offline_case (
  offline_case_id TEXT PRIMARY KEY, case_json TEXT NOT NULL
);
CREATE TABLE replay_iteration (
  iteration_id INTEGER PRIMARY KEY AUTOINCREMENT,
  offline_case_id TEXT, execution_mode TEXT, phase TEXT NOT NULL,
  variant TEXT NOT NULL, repetition INTEGER NOT NULL,
  checkpoint_restore_count INTEGER NOT NULL, victim_delay_ns INTEGER,
  aggressor_duration_ns INTEGER, victim_order_json TEXT NOT NULL,
  fingerprint_match INTEGER NOT NULL, foreign_work_count INTEGER NOT NULL,
  profiler_perturbation TEXT NOT NULL, raw_json TEXT NOT NULL
);
CREATE TABLE kernel_instance (
  kernel_instance_id TEXT PRIMARY KEY, iteration_id INTEGER NOT NULL,
  launch_id TEXT NOT NULL, model_id TEXT NOT NULL, client_id TEXT NOT NULL,
  server_pid INTEGER, client_pid INTEGER, source_run_id TEXT,
  source_inference_identity TEXT, context_handle TEXT NOT NULL,
  stream_handle TEXT NOT NULL, function_handle TEXT NOT NULL,
  symbol TEXT NOT NULL, driver_issue_start_ns INTEGER,
  driver_issue_end_ns INTEGER, replay_ready_ns INTEGER,
  cupti_start_ns INTEGER, cupti_end_ns INTEGER, grid_x INTEGER NOT NULL,
  grid_y INTEGER NOT NULL, grid_z INTEGER NOT NULL,
  total_cta_count INTEGER NOT NULL, block_x INTEGER NOT NULL,
  block_y INTEGER NOT NULL, block_z INTEGER NOT NULL,
  registers_per_thread INTEGER, static_shared_memory INTEGER,
  dynamic_shared_memory INTEGER NOT NULL, threads_per_block INTEGER NOT NULL,
  warps_per_block INTEGER NOT NULL, occupancy_blocks_per_sm INTEGER,
  UNIQUE (iteration_id, launch_id)
);
CREATE TABLE cta_interval (
  kernel_instance_id TEXT NOT NULL, iteration_id INTEGER NOT NULL,
  launch_id TEXT NOT NULL, cta_id TEXT NOT NULL, sm_id INTEGER,
  entry_ns INTEGER, exit_ns INTEGER, clock_error_ns INTEGER,
  observation_status TEXT NOT NULL, entry_observed INTEGER NOT NULL,
  exit_observed INTEGER NOT NULL, PRIMARY KEY (kernel_instance_id, cta_id)
);
CREATE TABLE cta_collection_status (
  kernel_instance_id TEXT PRIMARY KEY, iteration_id INTEGER NOT NULL,
  launch_id TEXT NOT NULL, expected_count INTEGER NOT NULL,
  entered_count INTEGER NOT NULL, exited_count INTEGER NOT NULL,
  complete_count INTEGER NOT NULL, missing_count INTEGER NOT NULL,
  dropped_entry_count INTEGER NOT NULL, dropped_exit_count INTEGER NOT NULL,
  coverage_fraction REAL NOT NULL, quality TEXT NOT NULL
);
CREATE TABLE clock_calibration (
  iteration_id INTEGER NOT NULL, client_id TEXT NOT NULL,
  clock_a TEXT NOT NULL, clock_b TEXT NOT NULL, offset_ns INTEGER,
  error_ns INTEGER, precision_class TEXT NOT NULL,
  PRIMARY KEY (iteration_id, client_id, clock_a, clock_b)
);
CREATE VIEW cta_dispatch_timeline AS
SELECT c.kernel_instance_id, c.iteration_id, i.offline_case_id,
       i.execution_mode, c.launch_id, c.cta_id, c.sm_id, c.entry_ns,
       ROW_NUMBER() OVER (
         PARTITION BY c.kernel_instance_id
         ORDER BY c.entry_ns, CAST(c.cta_id AS INTEGER)
       ) AS cumulative_entered_cta_count,
       CAST(ROW_NUMBER() OVER (
         PARTITION BY c.kernel_instance_id
         ORDER BY c.entry_ns, CAST(c.cta_id AS INTEGER)
       ) AS REAL) / s.expected_count AS entered_fraction
FROM cta_interval c
JOIN replay_iteration i USING(iteration_id)
JOIN cta_collection_status s USING(kernel_instance_id)
WHERE c.entry_ns IS NOT NULL;
CREATE VIEW cta_residency AS
SELECT c.kernel_instance_id, c.iteration_id, i.offline_case_id,
       i.execution_mode, c.launch_id, k.model_id, k.client_id, c.cta_id,
       c.sm_id, c.entry_ns, c.exit_ns, c.exit_ns - c.entry_ns AS duration_ns,
       c.clock_error_ns, c.observation_status
FROM cta_interval c
JOIN kernel_instance k USING(kernel_instance_id)
JOIN replay_iteration i USING(iteration_id)
WHERE c.entry_ns IS NOT NULL AND c.exit_ns IS NOT NULL
  AND c.entry_ns <= c.exit_ns;
CREATE VIEW cta_sm_timeline AS
SELECT r.*, ROW_NUMBER() OVER (
  PARTITION BY r.iteration_id, r.sm_id
  ORDER BY r.entry_ns, r.exit_ns, r.kernel_instance_id,
           CAST(r.cta_id AS INTEGER)
) AS sm_sequence
FROM cta_residency r
ORDER BY r.iteration_id, r.sm_id, sm_sequence;
"""


def _csv_path(path):
    compressed = Path(f"{path}.gz")
    return path if path.is_file() or not compressed.is_file() else compressed


def _open_csv(path):
    path = _csv_path(path)
    return (
        gzip.open(path, "rt", newline="", encoding="utf-8")
        if path.suffix == ".gz" else
        path.open(newline="", encoding="utf-8")
    )


def _read_csv(path):
    with _open_csv(path) as source:
        return list(csv.DictReader(source))


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_inputs(run_directory, model):
    return {
        record["nvbit_cta_target_label"]: record
        for record in (
            json.loads(line)
            for line in (
                run_directory / f"model_{model}_inputs.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if record.get("nvbit_cta_target_label")
    }


def _resource(targets, label, model):
    role = MODEL_ROLES[model]
    return targets[f"{label}_{role}_kernel"]


def _activity(model, label, resource, target, status, pid):
    launch_id = f"{label}_{model}"
    return {
        "launch_id": launch_id,
        "client_id": model,
        "client_pid": pid,
        "context_handle": str(target["context_handle"]),
        "stream_handle": str(target["stream_handle"]),
        "function_handle": str(target["function_handle"]),
        "workload_fingerprint": json.dumps({
            "scene_index": status["scene_index"],
            "scene_token": status["scene_token"],
            "message_order": status["message_order"],
            "ros_header_timestamp_ns": status["ros_header_timestamp_ns"],
        }, sort_keys=True),
        "launch_descriptor": {
            "symbol": resource["kernel_name"],
            "grid": resource["grid"],
            "block": resource["block"],
            "registers_per_thread": resource["registers_per_thread"],
            "static_shared_memory": resource["static_shared_memory"],
            "dynamic_shared_memory": resource["dynamic_shared_memory"],
            "occupancy_blocks_per_sm": (
                1 if model == "faster_rcnn" else 2
            ),
        },
    }


def _canonical_record(run_directory, targets, label):
    intervals = []
    collection = []
    activities = []
    calibrations = []
    for model in MODEL_ROLES:
        status = json.loads((
            run_directory / f"model_{model}_cta_status.json"
        ).read_text(encoding="utf-8"))
        target = next(
            item for item in status["targets"] if item["label"] == label
        )
        source = _source_inputs(run_directory, model)[label]
        model_status = json.loads((
            run_directory / f"model_{model}.json"
        ).read_text(encoding="utf-8"))
        resource = _resource(targets, label, model)
        launch_id = f"{label}_{model}"
        activities.append(_activity(
            model, label, resource, target, source, model_status["pid"]
        ))
        collection.append({
            "launch_id": launch_id,
            "expected_count": target["expected_count"],
            "entered_count": target["entered_count"],
            "exited_count": target["exited_count"],
            "complete_count": target["complete_count"],
            "missing_count": target["missing_count"],
            "dropped_entry_count": (
                target["expected_count"] - target["entered_count"]
            ),
            "dropped_exit_count": (
                target["expected_count"] - target["exited_count"]
            ),
            "coverage_fraction": target["coverage_fraction"],
            "quality": (
                "complete" if target["coverage_fraction"] == 1
                else "incomplete"
            ),
        })
        raw = _read_csv(run_directory / f"model_{model}_cta_raw.csv")
        for row in raw:
            if row["target_label"] != label:
                continue
            entry = int(row["entry_globaltimer_ns"])
            exit_ = int(row["exit_globaltimer_ns"])
            intervals.append({
                "launch_id": launch_id,
                "cta_id": row["cta_id"],
                "sm_id": int(row["sm_id"]),
                "entry_ns": entry or None,
                "exit_ns": exit_ or None,
                "clock_error_ns": int(row["clock_error_ns"]),
                "entry_observed": bool(int(row["entry_observed"])),
                "exit_observed": bool(int(row["exit_observed"])),
                "observation_status": row["observation_status"],
            })
        calibration = status["clock_calibration"]
        calibrations.append({
            "client_id": model,
            "clock_a": "gpu_globaltimer",
            "clock_b": "host_monotonic",
            "offset_ns": calibration["offset_ns"],
            "error_ns": calibration["error_ns"],
            "precision_class": "bounded_terminal_calibration",
        })
    return {
        "fingerprint_match": True,
        "profiler_perturbation": "targeted_nvbit_four_kernels",
        "kernel_instances": activities,
        "cta_collection_status": collection,
        "cta_intervals": intervals,
        "clock_calibrations": calibrations,
    }


def _export(connection, output_root, name):
    cursor = connection.execute(f"SELECT * FROM {name}")
    fields = [column[0] for column in cursor.description]
    with (output_root / f"{name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(cursor)


def _placement(connection):
    rows = [dict(zip(
        [column[0] for column in cursor.description], row
    )) for cursor in [connection.execute(
        "SELECT r.*,i.variant,d.cumulative_entered_cta_count "
        "FROM cta_residency r JOIN replay_iteration i USING(iteration_id) "
        "JOIN cta_dispatch_timeline d USING(kernel_instance_id,cta_id) "
        "ORDER BY r.iteration_id,r.entry_ns,CAST(r.cta_id AS INTEGER)"
    )] for row in cursor]
    by_iteration_sm_client = defaultdict(list)
    for row in rows:
        by_iteration_sm_client[
            (row["iteration_id"], row["sm_id"], row["client_id"])
        ].append(row)
    for row in rows:
        other = [
            candidate
            for (iteration, sm, client), candidates
            in by_iteration_sm_client.items()
            if iteration == row["iteration_id"]
            and sm == row["sm_id"]
            and client != row["client_id"]
            for candidate in candidates
            if min(row["exit_ns"], candidate["exit_ns"])
            > max(row["entry_ns"], candidate["entry_ns"])
        ]
        row["dispatch_order"] = row.pop("cumulative_entered_cta_count")
        row["dispatch_wave"] = (row["dispatch_order"] - 1) // SM_COUNT
        row["same_reported_smid_other_client_cta_count"] = len(other)
        row["same_reported_smid_overlap_ns"] = sum(
            min(row["exit_ns"], candidate["exit_ns"])
            - max(row["entry_ns"], candidate["entry_ns"])
            for candidate in other
        )
        row["cross_client_smid_comparable"] = CROSS_CLIENT_SMID_COMPARABLE
        row["physical_co_residency_supported"] = False
    return rows


def _summaries(placement):
    summaries = []
    for label in ("affected", "unaffected"):
        for model in MODEL_ROLES:
            rows = [
                row for row in placement
                if row["variant"] == label and row["client_id"] == model
            ]
            durations = np.asarray(
                [row["duration_ns"] for row in rows], dtype=float
            )
            entries = [row["entry_ns"] for row in rows]
            exits = [row["exit_ns"] for row in rows]
            same_reported_smid = [
                row for row in rows
                if row["same_reported_smid_other_client_cta_count"]
            ]
            summaries.append({
                "episode": label,
                "model_id": model,
                "cta_count": len(rows),
                "sm_count_observed": len({row["sm_id"] for row in rows}),
                "kernel_envelope_ns": max(exits) - min(entries),
                "dispatch_span_ns": max(entries) - min(entries),
                "cta_duration_p50_ns": float(np.percentile(durations, 50)),
                "cta_duration_p95_ns": float(np.percentile(durations, 95)),
                "cta_duration_max_ns": float(np.max(durations)),
                "same_reported_smid_cta_count": len(same_reported_smid),
                "same_reported_smid_cta_fraction": (
                    len(same_reported_smid) / len(rows)
                ),
                "summed_same_reported_smid_overlap_ns": sum(
                    row["same_reported_smid_overlap_ns"] for row in rows
                ),
                "maximum_simultaneous_same_reported_smid_ctas": max(
                    row["same_reported_smid_other_client_cta_count"]
                    for row in rows
                ),
                "cross_client_smid_comparable": CROSS_CLIENT_SMID_COMPARABLE,
                "physical_co_residency_supported": False,
            })
    return summaries


def _write_database(path, run_directory, targets):
    """Normalize the two targeted CTA episodes into the legacy schema-v4 DB."""
    with sqlite3.connect(str(path)) as connection:
        connection.executescript(CTA_DATABASE_DDL)
        metadata = {
            "schema_version": 4,
            "profiler": "gpu_forensics_v1",
            "source_run": run_directory.name,
            "target_episodes": targets,
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            ((key, json.dumps(value, sort_keys=True))
             for key, value in metadata.items()),
        )
        for repetition, label in enumerate(("affected", "unaffected"), 1):
            case_id = f"phase4-{label}"
            connection.execute(
                "INSERT INTO offline_case VALUES (?, ?)",
                (case_id, json.dumps({
                    "source": "full_stack_targeted_nvbit",
                    "episode": label,
                }, sort_keys=True)),
            )
            record = _canonical_record(run_directory, targets, label)
            cursor = connection.execute(
                "INSERT INTO replay_iteration VALUES "
                "(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    case_id, "full_stack_targeted_nvbit", "cta", label,
                    repetition, 0, None, None, "[]",
                    int(record["fingerprint_match"]), 0,
                    record["profiler_perturbation"],
                    json.dumps(record, sort_keys=True),
                ),
            )
            iteration_id = int(cursor.lastrowid)
            for activity in record["kernel_instances"]:
                descriptor = activity["launch_descriptor"]
                grid = [int(value) for value in descriptor["grid"]]
                block = [int(value) for value in descriptor["block"]]
                threads = math.prod(block)
                values = {
                    "kernel_instance_id": (
                        f"{iteration_id}:{activity['launch_id']}"
                    ),
                    "iteration_id": iteration_id,
                    "launch_id": activity["launch_id"],
                    "model_id": activity["client_id"],
                    "client_id": activity["client_id"],
                    "server_pid": None,
                    "client_pid": activity.get("client_pid"),
                    "source_run_id": "offline",
                    "source_inference_identity": activity[
                        "workload_fingerprint"
                    ],
                    "context_handle": activity["context_handle"],
                    "stream_handle": activity["stream_handle"],
                    "function_handle": activity["function_handle"],
                    "symbol": descriptor["symbol"],
                    "driver_issue_start_ns": None,
                    "driver_issue_end_ns": None,
                    "replay_ready_ns": None,
                    "cupti_start_ns": None,
                    "cupti_end_ns": None,
                    "grid_x": grid[0], "grid_y": grid[1],
                    "grid_z": grid[2],
                    "total_cta_count": math.prod(grid),
                    "block_x": block[0], "block_y": block[1],
                    "block_z": block[2],
                    "registers_per_thread": descriptor.get(
                        "registers_per_thread"
                    ),
                    "static_shared_memory": descriptor.get(
                        "static_shared_memory"
                    ),
                    "dynamic_shared_memory": int(
                        descriptor.get("dynamic_shared_memory", 0)
                    ),
                    "threads_per_block": threads,
                    "warps_per_block": (threads + 31) // 32,
                    "occupancy_blocks_per_sm": descriptor.get(
                        "occupancy_blocks_per_sm"
                    ),
                }
                columns = tuple(values)
                connection.execute(
                    "INSERT INTO kernel_instance (" + ",".join(columns)
                    + ") VALUES (" + ",".join(
                        ":" + column for column in columns
                    ) + ")",
                    values,
                )
            for status in record["cta_collection_status"]:
                instance_id = f"{iteration_id}:{status['launch_id']}"
                connection.execute(
                    "INSERT INTO cta_collection_status VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        instance_id, iteration_id, status["launch_id"],
                        status["expected_count"], status["entered_count"],
                        status["exited_count"], status["complete_count"],
                        status["missing_count"],
                        status["dropped_entry_count"],
                        status["dropped_exit_count"],
                        status["coverage_fraction"], status["quality"],
                    ),
                )
            for interval in record["cta_intervals"]:
                instance_id = f"{iteration_id}:{interval['launch_id']}"
                connection.execute(
                    "INSERT INTO cta_interval VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        instance_id, iteration_id, interval["launch_id"],
                        str(interval["cta_id"]), interval["sm_id"],
                        interval["entry_ns"], interval["exit_ns"],
                        interval["clock_error_ns"],
                        interval["observation_status"],
                        int(interval["entry_observed"]),
                        int(interval["exit_observed"]),
                    ),
                )
            for calibration in record["clock_calibrations"]:
                connection.execute(
                    "INSERT INTO clock_calibration VALUES (?,?,?,?,?,?,?)",
                    (
                        iteration_id, calibration["client_id"],
                        calibration["clock_a"], calibration["clock_b"],
                        calibration["offset_ns"], calibration["error_ns"],
                        calibration["precision_class"],
                    ),
                )


def analyze(run_directory, output_root, target_path):
    """Create schema-v4 canonical tables, views, placement, and summaries."""
    output_root.mkdir(parents=True, exist_ok=True)
    targets = json.loads(target_path.read_text(encoding="utf-8"))
    database_path = output_root / "gpu_forensics.sqlite"
    temporary = database_path.with_suffix(".tmp.sqlite")
    temporary.unlink(missing_ok=True)
    _write_database(temporary, run_directory, targets)
    temporary.replace(database_path)
    with sqlite3.connect(str(database_path)) as connection:
        for name in (
            "cta_interval", "cta_collection_status",
            "cta_dispatch_timeline", "cta_residency", "cta_sm_timeline",
        ):
            _export(connection, output_root, name)
        placement = _placement(connection)
    _write_csv(output_root / "capsule_cta_placement.csv", placement)
    summaries = _summaries(placement)
    _write_csv(output_root / "cta_episode_summary.csv", summaries)
    result = {
        "schema": "mps_two_model_cta_analysis_v1",
        "source_run": run_directory.name,
        "gpu_sm_count": SM_COUNT,
        "dispatch_wave_definition": (
            "zero-based floor((dispatch_order-1)/56)"
        ),
        "smid_scope": SMID_SCOPE,
        "cross_client_smid_comparable": CROSS_CLIENT_SMID_COMPARABLE,
        "summaries": summaries,
    }
    (output_root / "cta_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _allocated_registers(launch, warp_size):
    threads = math.prod(launch["block"])
    warps = math.ceil(threads / warp_size)
    per_warp = math.ceil(
        launch["registers_per_thread"] * warp_size / 256
    ) * 256
    return warps * per_warp


def _co_residency_feasibility(first, second, limits):
    threads = math.prod(first["block"]) + math.prod(second["block"])
    registers = _allocated_registers(
        first, limits["warp_size"]
    ) + _allocated_registers(second, limits["warp_size"])
    shared = sum(
        math.ceil((row["static_shared_memory"] +
                   row["dynamic_shared_memory"]) / 256) * 256
        for row in (first, second)
    )
    reasons = []
    if threads > limits["max_threads_per_sm"]:
        reasons.append("threads")
    if registers > limits["registers_per_sm"]:
        reasons.append("registers")
    if shared > limits["shared_memory_per_sm"]:
        reasons.append("shared_memory")
    if limits["max_blocks_per_sm"] < 2:
        reasons.append("block_limit")
    return not reasons, "+".join(reasons) or "none", threads, registers, shared


def _interval_overlap(first, second):
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def _same_sm_overlap(first, second):
    by_sm = defaultdict(lambda: [[], []])
    for interval in first:
        by_sm[interval[2]][0].append(interval)
    for interval in second:
        by_sm[interval[2]][1].append(interval)
    count = total = maximum = 0
    for left, right in by_sm.values():
        left.sort()
        right.sort()
        i = j = 0
        while i < len(left) and j < len(right):
            overlap = _interval_overlap(left[i], right[j])
            if overlap:
                count += 1
                total += overlap
                maximum = max(maximum, overlap)
            if left[i][1] <= right[j][1]:
                i += 1
            else:
                j += 1
    return count, total, maximum


def _target_launch_key(launches, model, label, sequence):
    return next((
        key for key, launch in launches.items()
        if key[0] == model and launch["target_label"] == label
        and int(launch["launch_sequence_index"]) == sequence
    ), None)


def _tracer_kernel_signature(launch):
    payload = {
        "symbol": launch["kernel_name"],
        "driver_launch_api": launch.get("driver_launch_api", "unavailable"),
        "launch_type": launch.get("launch_type", "kernel"),
        "grid": launch["grid"],
        "block": launch["block"],
        "registers_per_thread": launch["registers_per_thread"],
        "static_shared_memory": launch["static_shared_memory"],
        "dynamic_shared_memory": launch["dynamic_shared_memory"],
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def analyze_passive(run_directory, output_root, catalog_path):
    """Reduce complete passive CTA windows without copying raw intervals."""
    output_root.mkdir(parents=True, exist_ok=True)
    catalog = _read_csv(catalog_path)
    selected = {row["target_label"]: row for row in catalog}
    statuses = {}
    launches = {}
    intervals = defaultdict(list)
    collections = []
    for model in MODEL_ROLES:
        status = json.loads((
            run_directory / f"model_{model}_cta_status.json"
        ).read_text(encoding="utf-8"))
        statuses[model] = status
        for target in status["targets"]:
            collections.append({"model_id": model, **target,
                                **status["collection"]})
        for launch in status["launches"]:
            key = (model, int(launch["launch_slot"]))
            launches[key] = {"model_id": model, **launch}
        with _open_csv(
            run_directory / f"model_{model}_cta_raw.csv"
        ) as source:
            for row in csv.DictReader(source):
                if row["observation_status"] != "complete":
                    continue
                intervals[(model, int(row["launch_slot"]))].append((
                    int(row["entry_globaltimer_ns"]),
                    int(row["exit_globaltimer_ns"]),
                    int(row["sm_id"]), int(row["duration_ns"]),
                ))
    launch_rows = []
    for key, launch in launches.items():
        rows = intervals[key]
        if not rows:
            continue
        entries = sorted(row[0] for row in rows)
        durations = np.asarray([row[3] for row in rows], dtype=float)
        sm_count = len({row[2] for row in rows})
        wave_starts = entries[::max(sm_count, 1)]
        wave_gaps = np.diff(wave_starts) if len(wave_starts) > 1 else []
        first_entry = entries[0]
        last_entry = entries[-1]
        last_exit = max(row[1] for row in rows)
        ready = int(launch.get("launch_ready_globaltimer_ns", 0))
        launch_rows.append({
            **{k: launch[k] for k in (
                "model_id", "target_label", "launch_slot",
                "launch_sequence_index", "occurrence_within_signature",
                "kernel_name", "context_handle", "stream_handle",
                "function_handle", "registers_per_thread",
                "static_shared_memory", "dynamic_shared_memory",
            )},
            "driver_launch_api": launch.get(
                "driver_launch_api", "unavailable"
            ),
            "launch_type": launch.get("launch_type", "kernel"),
            "tracer_kernel_signature": _tracer_kernel_signature(launch),
            "grid": "x".join(map(str, launch["grid"])),
            "block": "x".join(map(str, launch["block"])),
            "cta_count": len(rows), "sm_count_observed": sm_count,
            "launch_ready_globaltimer_ns": ready,
            "first_cta_entry_ns": first_entry,
            "ready_to_first_cta_ns": first_entry - ready if ready else "",
            "kernel_envelope_ns": last_exit - first_entry,
            "dispatch_span_ns": last_entry - first_entry,
            "cta_admission_rate_per_ms": (
                len(rows) * 1e6 / max(1, last_entry - first_entry)
            ),
            "dispatch_wave_count": len(wave_starts),
            "interwave_gap_p50_ns": (
                float(np.percentile(wave_gaps, 50)) if len(wave_gaps) else 0
            ),
            "interwave_gap_max_ns": (
                int(max(wave_gaps)) if len(wave_gaps) else 0
            ),
            "cta_service_p50_ns": float(np.percentile(durations, 50)),
            "cta_service_p95_ns": float(np.percentile(durations, 95)),
            "cta_service_max_ns": float(np.max(durations)),
        })
    co_rows = []
    sm_rows = []
    target_coverage = []
    target_launches = []
    for label, target in selected.items():
        model = target["model_id"]
        sequence = int(target["launch_sequence_index"])
        capture_label = target.get("capture_label") or label
        target_key = _target_launch_key(
            launches, model, capture_label, sequence
        )
        collection = next(
            item for item in statuses[model]["targets"]
            if item["label"] == capture_label
        )
        captured = target_key is not None and bool(intervals[target_key])
        target_coverage.append({
            "model_id": model,
            "target_label": label,
            "capture_label": capture_label,
            "launch_sequence_index": sequence,
            "logical_launch_captured": captured,
            "observed_input_launch_count":
                collection["observed_input_launch_count"],
            "captured_window_launch_count":
                collection["captured_launch_count"],
            "complete_cta_count": collection["complete_count"],
            "coverage_fraction": collection["coverage_fraction"],
            "missing_reason": "" if captured else "target input not observed",
        })
        if not captured:
            continue
        launch_summary = next(
            row for row in launch_rows
            if row["model_id"] == target_key[0]
            and int(row["launch_slot"]) == target_key[1]
        )
        target_launches.append({
            **launch_summary,
            "capture_label": launch_summary["target_label"],
            "target_label": label,
        })
        target_launch = launches[target_key]
        target_ready = int(target_launch.get(
            "launch_ready_globaltimer_ns", 0
        ))
        target_span = (
            min(row[0] for row in intervals[target_key]),
            max(row[1] for row in intervals[target_key]),
        )
        target_ready_span = (target_ready or target_span[0], target_span[1])
        target_wait_span = (target_ready or target_span[0], target_span[0])
        other_model = next(item for item in MODEL_ROLES if item != model)
        limits = statuses[model]["device_limits"]
        for key, co_launch in launches.items():
            if key[0] != other_model:
                continue
            if not intervals[key]:
                continue
            co_span = (min(row[0] for row in intervals[key]),
                       max(row[1] for row in intervals[key]))
            overlap = _interval_overlap(target_span, co_span)
            ready_overlap = _interval_overlap(target_ready_span, co_span)
            wait_overlap = _interval_overlap(target_wait_span, co_span)
            feasible, reason, threads, registers, shared = (
                _co_residency_feasibility(target_launch, co_launch, limits)
            )
            same_count = same_total = same_max = 0
            if overlap:
                same_count, same_total, same_max = _same_sm_overlap(
                    intervals[target_key], intervals[key]
                )
            co_rows.append({
                "target_label": label, "target_model": model,
                "target_sequence_index": sequence,
                "co_model": other_model,
                "co_target_label": co_launch["target_label"],
                "co_launch_slot": key[1],
                "co_sequence_index": co_launch["launch_sequence_index"],
                "co_kernel_name": co_launch["kernel_name"],
                "global_overlap_ns": overlap,
                "ready_to_end_overlap_ns": ready_overlap,
                "ready_to_first_cta_overlap_ns": wait_overlap,
                "relative_phase_ns": co_span[0] - target_span[0],
                "resource_feasible_candidate": feasible,
                "infeasible_reason": reason,
                "combined_threads": threads,
                "combined_allocated_registers": registers,
                "combined_allocated_shared_memory": shared,
                "same_reported_smid_overlap_pair_count": same_count,
                "same_reported_smid_overlap_total_ns": same_total,
                "same_reported_smid_overlap_max_ns": same_max,
                "cross_client_smid_comparable":
                    CROSS_CLIENT_SMID_COMPARABLE,
                "physical_co_residency_supported": False,
            })
        by_sm = defaultdict(list)
        for row in intervals[target_key]:
            by_sm[row[2]].append(row)
        for sm, rows in sorted(by_sm.items()):
            sm_rows.append({
                "target_label": label, "model_id": model, "sm_id": sm,
                "cta_count": len(rows),
                "first_entry_ns": min(x[0] for x in rows),
                "last_exit_ns": max(x[1] for x in rows),
                "service_p50_ns": float(np.percentile(
                    [x[3] for x in rows], 50)),
                "service_p95_ns": float(np.percentile(
                    [x[3] for x in rows], 95)),
            })
    _write_csv(output_root / "passive_collection_status.csv", collections)
    _write_csv(output_root / "passive_launch_summary.csv", launch_rows)
    _write_csv(output_root / "passive_co_runner_intervals.csv", co_rows)
    _write_csv(output_root / "passive_target_sm_timeline.csv", sm_rows)
    _write_csv(output_root / "passive_target_coverage.csv", target_coverage)
    result = {
        "schema": "mps_two_model_passive_cta_analysis_v1",
        "source_run": run_directory.name,
        "raw_interval_artifacts": {
            model: str(_csv_path(
                run_directory / f"model_{model}_cta_raw.csv"
            ))
            for model in MODEL_ROLES
        },
        "launch_count": len(launch_rows),
        "cta_count": sum(len(rows) for rows in intervals.values()),
        "all_boundaries_complete": all(
            status["collection"]["incomplete_entry_count"] == 0 and
            status["collection"]["incomplete_exit_count"] == 0
            for status in statuses.values()
        ),
        "drop_count": sum(
            status["collection"]["dropped_capacity_record_count"]
            for status in statuses.values()
        ),
        "smid_scope": SMID_SCOPE,
        "cross_client_smid_comparable": CROSS_CLIENT_SMID_COMPARABLE,
        "smid_validation": (
            "80% MPS clients each report nsmid=44 and smid=0..43; "
            "resource-infeasible 65536-byte CTA pairs have overlapping "
            "same-numbered IDs, so IDs cannot establish cross-client "
            "physical co-residency"
        ),
        "selected_target_count": len(target_coverage),
        "captured_target_count": sum(
            row["logical_launch_captured"] for row in target_coverage
        ),
        "missing_target_labels": [
            row["target_label"] for row in target_coverage
            if not row["logical_launch_captured"]
        ],
        "target_launches": target_launches,
    }
    (output_root / "passive_cta_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("target_episodes", type=Path)
    parser.add_argument("--passive", action="store_true")
    args = parser.parse_args(argv)
    result = (
        analyze_passive(args.run_directory, args.output_root,
                        args.target_episodes)
        if args.passive else
        analyze(args.run_directory, args.output_root, args.target_episodes)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
