"""Command-line entry points for target selection and live capsule replay."""

import argparse
import json
from pathlib import Path
import sys

from .agent_backend import UnixAgentReplayBackend
from .capture_session import ManagedCaptureSessionBackend
from .kernel_replay import KernelReplayPlannerV3
from closeloop_analyzer.target_selection import TraceTargetSelector


def write_capsule_failure(output_directory: Path, error: Exception) -> None:
    """Atomically retain an inspectable live-campaign failure record."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    message = str(error)
    record = {
        "schema": "kernel_capsule_failure_v3",
        "status": (
            "no_self_contained_overlap_episode"
            if message == "no_self_contained_overlap_episode"
            else (
                "capsule_invalid"
                if "capsule_invalid" in message
                else "failed"
            )
        ),
        "error_type": type(error).__name__,
        "error": message,
    }
    path = output_directory / "capsule_failure.json"
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_target_main(argv=None) -> int:
    """Select the single top pair from an existing source run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run_directory")
    parser.add_argument("--output", default="target_selection.json")
    parser.add_argument("--aggressor-min-duration-ms", type=float, default=1.0)
    args = parser.parse_args(argv)
    source = Path(args.source_run_directory)
    TraceTargetSelector(
        source / "profile.sqlite",
        source,
        args.aggressor_min_duration_ms,
    ).write(Path(args.output))
    return 0


def replay_capsule_main(argv=None) -> int:
    """Run the fixed kernel-only campaign through live replay agents."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run_directory")
    parser.add_argument("output_directory")
    parser.add_argument(
        "--coordinator-socket",
        help="debug override for an already-running coordinator",
    )
    parser.add_argument(
        "--agent-library",
        help="debug override for libpperf_kernel_capsule_agent.so",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--validate-aggressor-priority",
        action="store_true",
        help=(
            "run a separate three-trial per-launch priority "
            "counterfactual"
        ),
    )
    parser.add_argument(
        "--priority-min-effect-ns",
        type=int,
        default=1,
        help="minimum victim-performance effect above sham noise",
    )
    parser.add_argument(
        "--client-release-offset",
        action="append",
        default=[],
        metavar="CLIENT=NS",
        help="signed common-epoch client offset; omitted clients use zero",
    )
    args = parser.parse_args(argv)
    release_offsets = {}
    for value in args.client_release_offset:
        try:
            client_id, raw_offset = value.split("=", 1)
            if not client_id:
                raise ValueError
            release_offsets[client_id] = int(raw_offset)
        except ValueError:
            parser.error(
                "--client-release-offset must have the form CLIENT=NS"
            )
    output_directory = Path(args.output_directory)
    output_existed = output_directory.exists()
    try:
        if args.coordinator_socket:
            backend = UnixAgentReplayBackend(
                Path(args.coordinator_socket), args.timeout_seconds
            )
        else:
            backend = ManagedCaptureSessionBackend(
                Path(args.source_run_directory),
                output_directory,
                Path(args.agent_library) if args.agent_library else None,
                args.timeout_seconds,
            )
        KernelReplayPlannerV3(
            Path(args.source_run_directory),
            output_directory,
            backend,
            profiler_options={
                "mode": "kernel_capsule",
                "checkpoint_required": True,
                "timing_repetitions": 1,
                "deep_repetitions": 1,
                "sequence_match_attempts": 3,
                "priority_counterfactual_enabled": (
                    args.validate_aggressor_priority
                ),
                "priority_counterfactual_trials": 3,
                "priority_min_effect_ns": args.priority_min_effect_ns,
                "required_collectors": [
                    "cupti_activity",
                    "cuda_launch_capture",
                    "cupti_checkpoint",
                ],
                "client_release_offsets_ns": release_offsets,
            },
            analyzer_options={
                "aggressor_min_duration_ms": 1.0,
                "aggressor_mode": "individual",
                "selected_case_count": 1,
                "pair_ranking": "aggregate_excess_wait",
                "representative_episode": "maximum_excess_wait",
                "victim_set": "gpu_execution_overlap_all_streams",
                "conclusion_policy": "conservative",
            },
        ).run()
    except Exception as error:
        if not output_existed:
            write_capsule_failure(output_directory, error)
        print(str(error), file=sys.stderr)
        return 2
    return 0
