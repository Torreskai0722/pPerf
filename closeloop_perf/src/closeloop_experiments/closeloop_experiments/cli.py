"""Public commands for validation, runs, campaigns, and artifact migration."""

import argparse
import importlib
import json
from pathlib import Path
import sys

from .config import ConfigError, load_run_config
from .migration import MigrationError, migrate_artifacts
from .runner import ExperimentRunner, PreflightError, RunFailure


CAMPAIGNS = {
    "communication": "communication_variation",
    "input-data": "corrected_input_variation",
    "input-variation": "input_variation",
    "k1091-cta-mps": "k1091_cta_mps_sweep",
    "mps-three-model": "mps_three_model",
    "mps-two-model": "mps_two_model_cause",
    "non-mps-memcpy": "non_mps_memcpy_compute",
    "non-mps-offset": "non_mps_offset_sweep",
}

CAMPAIGN_ACTIONS = {
    "communication": {"generate", "validate", "run"},
    "input-data": {
        "plan", "generate", "validate", "run", "generate-isolated",
        "validate-isolated", "run-isolated",
    },
    "input-variation": {"plan", "prepare", "generate", "validate", "run"},
    "k1091-cta-mps": {"baselines", "calibrate", "generate", "run"},
    "mps-three-model": {"generate", "validate", "run"},
    "mps-two-model": {
        "generate", "validate", "run", "run-config",
        "ncu-generate", "cta-passive-generate", "postpassive-generate",
    },
    "non-mps-memcpy": {"validate", "calibrate", "generate", "run"},
    "non-mps-offset": {"plan", "validate", "run"},
}


def _config_parser(
        description: str, run: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="schema-v2 YAML run configuration")
    if run:
        parser.add_argument("--artifact-root", required=True)
        parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_main(argv=None) -> int:
    """Validate one authored configuration without creating artifacts."""
    args = _config_parser("Validate a closed-loop run").parse_args(argv)
    try:
        config = load_run_config(args.config)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"valid": True, "sha256": config.sha256}, sort_keys=True))
    return 0


def run_main(argv=None) -> int:
    """Execute or dry-run one configuration under an explicit artifact root."""
    args = _config_parser("Run a closed-loop profile", True).parse_args(argv)
    try:
        config = load_run_config(args.config, artifact_root=args.artifact_root)
        ExperimentRunner(config).run(args.dry_run)
    except (ConfigError, PreflightError, RunFailure, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def campaign_main(argv=None) -> int:
    """Dispatch one named campaign while keeping its artifact root explicit."""
    parser = argparse.ArgumentParser(
        description="Manage a closed-loop campaign")
    parser.add_argument("name", choices=sorted(CAMPAIGNS))
    parser.add_argument("action")
    parser.add_argument("study")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if args.action not in CAMPAIGN_ACTIONS[args.name]:
        print(
            f"error: unsupported {args.name} campaign action "
            f"{args.action!r}",
            file=sys.stderr,
        )
        return 2
    module = importlib.import_module(
        f"closeloop_experiments.{CAMPAIGNS[args.name]}")
    forwarded = [
        "--artifact-root", args.artifact_root, args.action, args.study,
    ]
    if args.dry_run:
        forwarded.append("--dry-run")
    try:
        result = module.main(forwarded + extra)
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return int(result or 0)


def migrate_main(argv=None) -> int:
    """Relocate historical artifacts after copy verification."""
    parser = argparse.ArgumentParser(
        description="Relocate closed-loop artifacts")
    parser.add_argument("source_root")
    parser.add_argument("destination_root")
    args = parser.parse_args(argv)
    try:
        manifest = migrate_artifacts(args.source_root, args.destination_root)
    except (MigrationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(str(Path(manifest)))
    return 0
