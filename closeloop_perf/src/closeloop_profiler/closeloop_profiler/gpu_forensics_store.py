"""Normalized SQLite evidence store for exact MPS kernel capsules."""

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, Optional

from .kernel_capsule import KernelCapsule

SCHEMA_VERSION = 4


DDL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_target (
    capsule_id TEXT PRIMARY KEY,
    source_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS offline_case (
    offline_case_id TEXT PRIMARY KEY,
    case_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capsule_launch (
    launch_id TEXT PRIMARY KEY,
    capsule_id TEXT NOT NULL,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    client_id TEXT NOT NULL,
    context_handle TEXT NOT NULL,
    stream_handle TEXT NOT NULL,
    stream_priority INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    function_handle TEXT NOT NULL,
    code_object_sha256 TEXT NOT NULL,
    fingerprint_sha256 TEXT NOT NULL,
    launch_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kernel_dependency (
    capsule_id TEXT NOT NULL,
    source_launch_id TEXT NOT NULL,
    target_launch_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    event_handle TEXT,
    PRIMARY KEY (capsule_id, source_launch_id, target_launch_id, kind)
);
CREATE TABLE IF NOT EXISTS replay_iteration (
    iteration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    offline_case_id TEXT,
    execution_mode TEXT,
    phase TEXT NOT NULL,
    variant TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    checkpoint_restore_count INTEGER NOT NULL,
    victim_delay_ns INTEGER,
    aggressor_duration_ns INTEGER,
    victim_order_json TEXT NOT NULL,
    fingerprint_match INTEGER NOT NULL,
    foreign_work_count INTEGER NOT NULL,
    profiler_perturbation TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS replay_launch_activity (
    iteration_id INTEGER NOT NULL,
    launch_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    context_handle TEXT NOT NULL,
    stream_handle TEXT NOT NULL,
    submission_ordinal INTEGER NOT NULL,
    driver_command_fingerprint TEXT NOT NULL,
    driver_issue_start_ns INTEGER NOT NULL,
    driver_issue_end_ns INTEGER NOT NULL,
    gpu_start_ns INTEGER NOT NULL,
    gpu_end_ns INTEGER NOT NULL,
    gpu_duration_ns INTEGER NOT NULL,
    issue_to_gpu_start_wait_ns INTEGER NOT NULL,
    replay_ready_ns INTEGER NOT NULL,
    admission_wait_ns INTEGER NOT NULL,
    aggressor_overlap_duration_ns INTEGER NOT NULL,
    overlap_relationship TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (iteration_id, launch_id)
);
CREATE TABLE IF NOT EXISTS kernel_instance (
    kernel_instance_id TEXT PRIMARY KEY,
    iteration_id INTEGER NOT NULL,
    launch_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    server_pid INTEGER,
    client_pid INTEGER,
    source_run_id TEXT,
    source_inference_identity TEXT,
    context_handle TEXT NOT NULL,
    stream_handle TEXT NOT NULL,
    function_handle TEXT NOT NULL,
    symbol TEXT NOT NULL,
    driver_issue_start_ns INTEGER,
    driver_issue_end_ns INTEGER,
    replay_ready_ns INTEGER,
    cupti_start_ns INTEGER,
    cupti_end_ns INTEGER,
    grid_x INTEGER NOT NULL,
    grid_y INTEGER NOT NULL,
    grid_z INTEGER NOT NULL,
    total_cta_count INTEGER NOT NULL,
    block_x INTEGER NOT NULL,
    block_y INTEGER NOT NULL,
    block_z INTEGER NOT NULL,
    registers_per_thread INTEGER,
    static_shared_memory INTEGER,
    dynamic_shared_memory INTEGER NOT NULL,
    threads_per_block INTEGER NOT NULL,
    warps_per_block INTEGER NOT NULL,
    occupancy_blocks_per_sm INTEGER,
    UNIQUE (iteration_id, launch_id)
);
CREATE TABLE IF NOT EXISTS cta_interval (
    kernel_instance_id TEXT NOT NULL,
    iteration_id INTEGER NOT NULL,
    launch_id TEXT NOT NULL,
    cta_id TEXT NOT NULL,
    sm_id INTEGER,
    entry_ns INTEGER,
    exit_ns INTEGER,
    clock_error_ns INTEGER,
    observation_status TEXT NOT NULL,
    entry_observed INTEGER NOT NULL,
    exit_observed INTEGER NOT NULL,
    PRIMARY KEY (kernel_instance_id, cta_id),
    FOREIGN KEY (kernel_instance_id) REFERENCES kernel_instance(
        kernel_instance_id
    )
);
CREATE TABLE IF NOT EXISTS cta_collection_status (
    kernel_instance_id TEXT PRIMARY KEY,
    iteration_id INTEGER NOT NULL,
    launch_id TEXT NOT NULL,
    expected_count INTEGER NOT NULL,
    entered_count INTEGER NOT NULL,
    exited_count INTEGER NOT NULL,
    complete_count INTEGER NOT NULL,
    missing_count INTEGER NOT NULL,
    dropped_entry_count INTEGER NOT NULL,
    dropped_exit_count INTEGER NOT NULL,
    coverage_fraction REAL NOT NULL,
    quality TEXT NOT NULL,
    FOREIGN KEY (kernel_instance_id) REFERENCES kernel_instance(
        kernel_instance_id
    )
);
CREATE TABLE IF NOT EXISTS metric_sample (
    iteration_id INTEGER NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    precision_class TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kernel_resource_profile (
    launch_id TEXT PRIMARY KEY,
    grid_x INTEGER,
    grid_y INTEGER,
    grid_z INTEGER,
    total_cta_count INTEGER,
    block_x INTEGER,
    block_y INTEGER,
    block_z INTEGER,
    registers_per_thread INTEGER,
    static_shared_memory INTEGER,
    dynamic_shared_memory INTEGER,
    threads_per_block INTEGER,
    warps_per_block INTEGER,
    occupancy_blocks_per_sm INTEGER,
    sass_sha256 TEXT,
    ncu_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gpu_topology (
    gpu_id INTEGER PRIMARY KEY,
    gpc_count INTEGER,
    tpc_count INTEGER,
    sm_count INTEGER,
    mapping_json TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capability (
    collector TEXT PRIMARY KEY,
    required INTEGER NOT NULL,
    available INTEGER NOT NULL,
    reason TEXT,
    version TEXT
);
CREATE TABLE IF NOT EXISTS dropped_record (
    collector TEXT NOT NULL,
    record_kind TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (collector, record_kind)
);
CREATE TABLE IF NOT EXISTS overhead (
    phase TEXT PRIMARY KEY,
    estimated_percent REAL,
    class TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS clock_calibration (
    iteration_id INTEGER NOT NULL,
    client_id TEXT NOT NULL,
    clock_a TEXT NOT NULL,
    clock_b TEXT NOT NULL,
    offset_ns INTEGER,
    error_ns INTEGER,
    precision_class TEXT NOT NULL,
    PRIMARY KEY (iteration_id, client_id, clock_a, clock_b)
);
CREATE TABLE IF NOT EXISTS microbenchmark_validation (
    stage INTEGER NOT NULL,
    condition_id TEXT NOT NULL,
    trial INTEGER NOT NULL,
    intended_status TEXT NOT NULL,
    observed_status TEXT NOT NULL,
    direct_wdu_claim INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL,
    PRIMARY KEY (stage, condition_id, trial)
);
CREATE INDEX IF NOT EXISTS idx_kernel_instance_client_time
    ON kernel_instance(client_id, cupti_start_ns, cupti_end_ns);
CREATE INDEX IF NOT EXISTS idx_cta_instance_entry
    ON cta_interval(kernel_instance_id, entry_ns, cta_id);
CREATE INDEX IF NOT EXISTS idx_cta_sm_interval
    ON cta_interval(sm_id, entry_ns, exit_ns);
CREATE VIEW IF NOT EXISTS cta_kernel_summary AS
SELECT
    k.kernel_instance_id,
    k.iteration_id,
    i.offline_case_id,
    i.execution_mode,
    k.launch_id,
    k.model_id,
    k.client_id,
    MIN(c.entry_ns) AS first_observed_entry_ns,
    MAX(c.entry_ns) AS last_observed_entry_ns,
    s.expected_count,
    s.entered_count,
    s.exited_count,
    s.complete_count,
    s.missing_count,
    s.dropped_entry_count,
    s.dropped_exit_count,
    s.coverage_fraction,
    s.quality
FROM kernel_instance AS k
JOIN replay_iteration AS i USING(iteration_id)
JOIN cta_collection_status AS s USING(kernel_instance_id)
LEFT JOIN cta_interval AS c USING(kernel_instance_id)
GROUP BY k.kernel_instance_id;
CREATE VIEW IF NOT EXISTS cta_dispatch_timeline AS
SELECT
    c.kernel_instance_id,
    c.iteration_id,
    i.offline_case_id,
    i.execution_mode,
    c.launch_id,
    c.cta_id,
    c.sm_id,
    c.entry_ns,
    ROW_NUMBER() OVER (
        PARTITION BY c.kernel_instance_id
        ORDER BY c.entry_ns, CAST(c.cta_id AS INTEGER)
    ) AS cumulative_entered_cta_count,
    CAST(ROW_NUMBER() OVER (
        PARTITION BY c.kernel_instance_id
        ORDER BY c.entry_ns, CAST(c.cta_id AS INTEGER)
    ) AS REAL) / s.expected_count AS entered_fraction
FROM cta_interval AS c
JOIN replay_iteration AS i USING(iteration_id)
JOIN cta_collection_status AS s USING(kernel_instance_id)
WHERE c.entry_ns IS NOT NULL;
CREATE VIEW IF NOT EXISTS cta_residency AS
SELECT
    c.kernel_instance_id,
    c.iteration_id,
    i.offline_case_id,
    i.execution_mode,
    c.launch_id,
    k.model_id,
    k.client_id,
    c.cta_id,
    c.sm_id,
    c.entry_ns,
    c.exit_ns,
    c.exit_ns - c.entry_ns AS duration_ns,
    c.clock_error_ns,
    c.observation_status
FROM cta_interval AS c
JOIN kernel_instance AS k USING(kernel_instance_id)
JOIN replay_iteration AS i USING(iteration_id)
WHERE c.entry_ns IS NOT NULL
  AND c.exit_ns IS NOT NULL
  AND c.entry_ns <= c.exit_ns;
CREATE VIEW IF NOT EXISTS cta_sm_timeline AS
SELECT
    r.*,
    ROW_NUMBER() OVER (
        PARTITION BY r.iteration_id, r.sm_id
        ORDER BY r.entry_ns, r.exit_ns, r.kernel_instance_id,
                 CAST(r.cta_id AS INTEGER)
    ) AS sm_sequence
FROM cta_residency AS r
ORDER BY r.iteration_id, r.sm_id, sm_sequence;
CREATE VIEW IF NOT EXISTS kernel_co_residency AS
SELECT
    a.iteration_id,
    a.offline_case_id,
    a.execution_mode,
    a.sm_id,
    a.kernel_instance_id AS kernel_instance_a,
    a.launch_id AS launch_id_a,
    a.cta_id AS cta_id_a,
    b.kernel_instance_id AS kernel_instance_b,
    b.launch_id AS launch_id_b,
    b.cta_id AS cta_id_b,
    MAX(a.entry_ns, b.entry_ns) AS overlap_start_ns,
    MIN(a.exit_ns, b.exit_ns) AS overlap_end_ns,
    MIN(a.exit_ns, b.exit_ns) - MAX(a.entry_ns, b.entry_ns)
        AS overlap_duration_ns
FROM cta_residency AS a
JOIN cta_residency AS b
  ON a.iteration_id = b.iteration_id
 AND a.sm_id = b.sm_id
 AND a.kernel_instance_id < b.kernel_instance_id
 AND MIN(a.exit_ns, b.exit_ns) > MAX(a.entry_ns, b.entry_ns);
"""


class GpuForensicsStore:
    """Create and populate the normalized capsule evidence database."""

    def __init__(self, path: Path):
        """Open a normalized store and create its versioned schema."""
        self.path = Path(path)
        if self.path.is_file() and self.path.stat().st_size:
            with sqlite3.connect(str(self.path)) as existing:
                try:
                    row = existing.execute(
                        "SELECT value_json FROM metadata "
                        "WHERE key='schema_version'"
                    ).fetchone()
                except sqlite3.Error as exc:
                    raise RuntimeError(
                        "refusing to migrate an existing evidence store"
                    ) from exc
            if row is None or json.loads(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(
                    "refusing to migrate an existing evidence store"
                )
        self.connection = sqlite3.connect(str(self.path))
        self.capsule: Optional[KernelCapsule] = None
        self.connection.executescript(DDL)
        self.set_metadata("schema_version", SCHEMA_VERSION)
        self.set_metadata("profiler", "gpu_forensics_v1")

    def close(self) -> None:
        """Commit and close the evidence store."""
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "GpuForensicsStore":
        """Return this open transactional evidence store."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Commit and close the store when leaving its context."""
        self.close()

    def set_metadata(self, key: str, value: Any) -> None:
        """Insert or replace one JSON metadata value."""
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value_json) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )

    def record_target(self, capsule_id: str, target: Dict[str, Any]) -> None:
        """Store the source trace and selected aggregate-delay target."""
        self.connection.execute(
            "INSERT OR REPLACE INTO source_target VALUES (?, ?)",
            (capsule_id, json.dumps(target, sort_keys=True)),
        )

    def record_offline_case(
        self, case_id: str, value: Dict[str, Any]
    ) -> None:
        """Store one normalized offline workload/context identity."""
        self.connection.execute(
            "INSERT OR REPLACE INTO offline_case VALUES (?, ?)",
            (case_id, json.dumps(value, sort_keys=True)),
        )

    def record_capsule(self, capsule: KernelCapsule) -> None:
        """Store exact launch fingerprints and dependency edges."""
        capsule.validate()
        self.capsule = capsule
        self.set_metadata("replay_backend", capsule.schema)
        victim_ordinals = {
            launch_id: index
            for index, launch_id in enumerate(capsule.victim_launch_ids)
        }
        for launch in capsule.launches:
            role = (
                "aggressor"
                if launch.launch_id == capsule.aggressor_launch_id
                else "victim"
            )
            ordinal = (
                -1
                if role == "aggressor"
                else victim_ordinals[launch.launch_id]
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO capsule_launch VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    launch.launch_id,
                    capsule.capsule_id,
                    role,
                    ordinal,
                    launch.client_id,
                    launch.context_handle,
                    launch.stream_handle,
                    launch.stream_priority,
                    launch.symbol,
                    launch.function_handle,
                    launch.code_object_sha256,
                    launch.digest,
                    json.dumps(asdict(launch), sort_keys=True),
                ),
            )
        for edge in capsule.dependencies:
            self.connection.execute(
                "INSERT OR REPLACE INTO kernel_dependency VALUES "
                "(?, ?, ?, ?, ?)",
                (
                    capsule.capsule_id,
                    edge.source_launch_id,
                    edge.target_launch_id,
                    edge.kind,
                    edge.event_handle,
                ),
            )
        self.set_metadata("capsule", capsule.to_dict())

    def record_iteration(
        self, phase: str, variant: str, repetition: int,
        record: Dict[str, Any], offline_case_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
    ) -> int:
        """Store one checkpoint-restored replay iteration."""
        cursor = self.connection.execute(
            "INSERT INTO replay_iteration("
            "offline_case_id, execution_mode, phase, variant, repetition, "
            "checkpoint_restore_count, "
            "victim_delay_ns, aggressor_duration_ns, victim_order_json, "
            "fingerprint_match, foreign_work_count, profiler_perturbation, "
            "raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                offline_case_id,
                execution_mode,
                phase,
                variant,
                repetition,
                int(record.get("checkpoint_restore_count", 0)),
                record.get("victim_delay_ns"),
                record.get("aggressor_duration_ns"),
                json.dumps(record.get("victim_order", [])),
                int(bool(record.get("fingerprint_match", False))),
                int(record.get("foreign_work_count", 0)),
                str(record.get("profiler_perturbation", "none")),
                json.dumps(record, sort_keys=True),
            ),
        )
        iteration_id = int(cursor.lastrowid)
        for sample in record.get("metric_samples", []):
            self.connection.execute(
                "INSERT INTO metric_sample VALUES (?, ?, ?, ?, ?, ?)",
                (
                    iteration_id,
                    int(sample["timestamp_ns"]),
                    sample["metric"],
                    float(sample["value"]),
                    sample.get("unit"),
                    sample.get("precision_class", "sampled"),
                ),
            )
        for activity in record.get("launch_activities", []):
            self.connection.execute(
                "INSERT INTO replay_launch_activity VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    iteration_id,
                    activity["launch_id"],
                    activity["client_id"],
                    activity["context_handle"],
                    activity["stream_handle"],
                    int(activity["submission_ordinal"]),
                    activity["driver_command_fingerprint"],
                    int(activity["driver_issue_start_ns"]),
                    int(activity["driver_issue_end_ns"]),
                    int(activity["gpu_start_ns"]),
                    int(activity["gpu_end_ns"]),
                    int(activity["gpu_duration_ns"]),
                    int(activity["issue_to_gpu_start_wait_ns"]),
                    int(activity["replay_ready_ns"]),
                    int(activity["admission_wait_ns"]),
                    int(activity["aggressor_overlap_duration_ns"]),
                    activity["overlap_relationship"],
                    json.dumps(activity, sort_keys=True),
                ),
            )
            self._record_kernel_instance(
                iteration_id, activity["launch_id"], activity
            )
        for activity in record.get("kernel_instances", []):
            self._record_kernel_instance(
                iteration_id, activity["launch_id"], activity
            )
        for status in record.get("cta_collection_status", []):
            launch_id = str(status["launch_id"])
            instance_id = self._record_kernel_instance(
                iteration_id, launch_id
            )
            self.connection.execute(
                "INSERT INTO cta_collection_status VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    instance_id,
                    iteration_id,
                    launch_id,
                    int(status["expected_count"]),
                    int(status["entered_count"]),
                    int(status["exited_count"]),
                    int(status["complete_count"]),
                    int(status["missing_count"]),
                    int(status["dropped_entry_count"]),
                    int(status["dropped_exit_count"]),
                    float(status["coverage_fraction"]),
                    str(status["quality"]),
                ),
            )
        for cta in record.get("cta_intervals", []):
            launch_id = str(cta["launch_id"])
            instance_id = self._record_kernel_instance(
                iteration_id, launch_id
            )
            entry = cta.get("entry_ns")
            exit_ = cta.get("exit_ns")
            entered = bool(cta.get("entry_observed", entry is not None))
            exited = bool(cta.get("exit_observed", exit_ is not None))
            status = cta.get("observation_status") or (
                "complete" if entered and exited else
                "entry_only" if entered else
                "exit_only" if exited else "missing"
            )
            self.connection.execute(
                "INSERT INTO cta_interval VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    instance_id,
                    iteration_id,
                    launch_id,
                    str(cta["cta_id"]),
                    None if cta.get("sm_id") is None
                    else int(cta["sm_id"]),
                    None if entry is None else int(entry),
                    None if exit_ is None else int(exit_),
                    cta.get("clock_error_ns"),
                    str(status),
                    int(entered),
                    int(exited),
                ),
            )
        for calibration in record.get("clock_calibrations", []):
            self.connection.execute(
                "INSERT INTO clock_calibration VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    iteration_id,
                    str(calibration["client_id"]),
                    str(calibration["clock_a"]),
                    str(calibration["clock_b"]),
                    calibration.get("offset_ns"),
                    calibration.get("error_ns"),
                    str(calibration["precision_class"]),
                ),
            )
        return iteration_id

    def _record_kernel_instance(
        self,
        iteration_id: int,
        launch_id: str,
        activity: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create or enrich the stable iteration:launch instance row."""
        instance_id = f"{iteration_id}:{launch_id}"
        if activity is None and self.connection.execute(
            "SELECT 1 FROM kernel_instance WHERE kernel_instance_id=?",
            (instance_id,),
        ).fetchone() is not None:
            return instance_id
        if self.capsule is None:
            if activity is None:
                raise RuntimeError(
                    "offline kernel activity must precede CTA records"
                )
            return self._record_offline_kernel_instance(
                iteration_id, launch_id, activity
            )
        launch = self.capsule.launch_map()[launch_id]
        clients = {
            client.client_id: client for client in self.capsule.clients
        }
        client = clients.get(launch.client_id)
        resource = self.connection.execute(
            "SELECT registers_per_thread, static_shared_memory, "
            "dynamic_shared_memory, threads_per_block, warps_per_block, "
            "occupancy_blocks_per_sm FROM kernel_resource_profile "
            "WHERE launch_id=?",
            (launch_id,),
        ).fetchone()
        resource = resource or (None,) * 6
        attributes = launch.function_attributes
        registers = resource[0]
        if registers is None:
            registers = attributes.get("4", attributes.get("registers"))
        static_shared = resource[1]
        if static_shared is None:
            static_shared = attributes.get(
                "1", attributes.get("static_shared_memory")
            )
        threads = launch.block[0] * launch.block[1] * launch.block[2]
        identity = {
            "framework_owner": launch.framework_owner,
            "frame_local_launch_occurrence_ordinal": (
                launch.frame_local_launch_occurrence_ordinal
            ),
            "frame_local_launch_sequence_index": (
                launch.frame_local_launch_sequence_index
            ),
        }
        activity = activity or {}
        values = {
            "kernel_instance_id": instance_id,
            "iteration_id": iteration_id,
            "launch_id": launch_id,
            "model_id": launch.client_id,
            "client_id": launch.client_id,
            "server_pid": None if client is None else client.server_pid,
            "client_pid": None if client is None else client.client_pid,
            "source_run_id": self.capsule.source_run_id,
            "source_inference_identity": json.dumps(
                identity, sort_keys=True
            ),
            "context_handle": launch.context_handle,
            "stream_handle": launch.stream_handle,
            "function_handle": launch.function_handle,
            "symbol": launch.symbol,
            "driver_issue_start_ns": activity.get("driver_issue_start_ns"),
            "driver_issue_end_ns": activity.get("driver_issue_end_ns"),
            "replay_ready_ns": activity.get("replay_ready_ns"),
            "cupti_start_ns": activity.get("gpu_start_ns"),
            "cupti_end_ns": activity.get("gpu_end_ns"),
            "grid_x": launch.grid[0],
            "grid_y": launch.grid[1],
            "grid_z": launch.grid[2],
            "total_cta_count": launch.grid[0] * launch.grid[1]
            * launch.grid[2],
            "block_x": launch.block[0],
            "block_y": launch.block[1],
            "block_z": launch.block[2],
            "registers_per_thread": registers,
            "static_shared_memory": static_shared,
            "dynamic_shared_memory": launch.dynamic_shared_memory,
            "threads_per_block": threads,
            "warps_per_block": (threads + 31) // 32,
            "occupancy_blocks_per_sm": resource[5],
        }
        columns = tuple(values)
        self.connection.execute(
            "INSERT OR REPLACE INTO kernel_instance (" +
            ", ".join(columns) + ") VALUES (" +
            ", ".join(":" + column for column in columns) + ")",
            values,
        )
        return instance_id

    def _record_offline_kernel_instance(
        self, iteration_id: int, launch_id: str, activity: Dict[str, Any]
    ) -> str:
        """Create a kernel instance from a normalized adapter descriptor."""
        descriptor = dict(activity["launch_descriptor"])
        grid = tuple(int(value) for value in descriptor["grid"])
        block = tuple(int(value) for value in descriptor["block"])
        threads = block[0] * block[1] * block[2]
        instance_id = f"{iteration_id}:{launch_id}"
        values = {
            "kernel_instance_id": instance_id,
            "iteration_id": iteration_id,
            "launch_id": launch_id,
            "model_id": activity["client_id"],
            "client_id": activity["client_id"],
            "server_pid": None,
            "client_pid": activity.get("client_pid"),
            "source_run_id": "offline",
            "source_inference_identity": activity["workload_fingerprint"],
            "context_handle": str(activity.get("context_handle", "local")),
            "stream_handle": str(activity.get("stream_handle", "local")),
            "function_handle": str(
                activity.get("function_handle", "process-local")
            ),
            "symbol": descriptor["symbol"],
            "driver_issue_start_ns": activity.get("driver_issue_start_ns"),
            "driver_issue_end_ns": activity.get("driver_issue_end_ns"),
            "replay_ready_ns": activity.get("replay_ready_ns"),
            "cupti_start_ns": activity.get("gpu_start_ns"),
            "cupti_end_ns": activity.get("gpu_end_ns"),
            "grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2],
            "total_cta_count": grid[0] * grid[1] * grid[2],
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
        self.connection.execute(
            "INSERT OR REPLACE INTO kernel_instance (" +
            ", ".join(columns) + ") VALUES (" +
            ", ".join(":" + column for column in columns) + ")",
            values,
        )
        return instance_id

    def record_resource(self, launch_id: str, record: Dict[str, Any]) -> None:
        """Store launch/occupancy/SASS facts collected by NCU or CUDA."""
        self.connection.execute(
            "INSERT OR REPLACE INTO kernel_resource_profile VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                launch_id,
                record.get("grid_x"),
                record.get("grid_y"),
                record.get("grid_z"),
                record.get("total_cta_count"),
                record.get("block_x"),
                record.get("block_y"),
                record.get("block_z"),
                record.get("registers_per_thread"),
                record.get("static_shared_memory"),
                record.get("dynamic_shared_memory"),
                record.get("threads_per_block"),
                record.get("warps_per_block"),
                record.get("occupancy_blocks_per_sm"),
                record.get("sass_sha256"),
                json.dumps(record, sort_keys=True),
            ),
        )
        self.connection.execute(
            "UPDATE kernel_instance SET "
            "registers_per_thread=COALESCE(?, registers_per_thread), "
            "static_shared_memory=COALESCE(?, static_shared_memory), "
            "dynamic_shared_memory=COALESCE(?, dynamic_shared_memory), "
            "threads_per_block=COALESCE(?, threads_per_block), "
            "warps_per_block=COALESCE(?, warps_per_block), "
            "occupancy_blocks_per_sm=COALESCE(?, occupancy_blocks_per_sm) "
            "WHERE launch_id=?",
            (
                record.get("registers_per_thread"),
                record.get("static_shared_memory"),
                record.get("dynamic_shared_memory"),
                record.get("threads_per_block"),
                record.get("warps_per_block"),
                record.get("occupancy_blocks_per_sm"),
                launch_id,
            ),
        )

    def record_capability(
        self,
        collector: str,
        required: bool,
        available: bool,
        reason: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """Record collector availability without making optional loss fatal."""
        self.connection.execute(
            "INSERT OR REPLACE INTO capability VALUES (?, ?, ?, ?, ?)",
            (collector, int(required), int(available), reason, version),
        )
