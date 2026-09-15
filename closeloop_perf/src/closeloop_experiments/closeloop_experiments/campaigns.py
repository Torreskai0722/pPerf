"""Small shared primitives for immutable campaign execution."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import yaml

from .config import load_run_config
from .runner import ExperimentRunner


def config_hash(data: Dict[str, Any]) -> str:
    """Return the stable SHA-256 for a materialized configuration."""
    payload = yaml.safe_dump(data, sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def materialize_config(path: Path, data: Dict[str, Any]) -> Path:
    """Create one immutable config, refusing a changed collision."""
    payload = yaml.safe_dump(data, sort_keys=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"immutable generated config differs: {path}")
    if not path.exists():
        path.write_bytes(payload)
        path.chmod(0o444)
    return path


def run_complete(artifact_root: Path, run_id: str) -> bool:
    """Return whether a run manifest records successful completion."""
    path = artifact_root / "runs" / run_id / "run_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["state"] == "success"
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def filter_pending(configs: Iterable[Path], artifact_root: Path) -> List[Path]:
    """Drop configurations whose immutable run already completed."""
    return [
        path for path in configs
        if not run_complete(
            artifact_root,
            load_run_config(str(path), check_paths=False).data["run"]["id"],
        )
    ]


def execute(configs: Iterable[Path], artifact_root: Path,
            dry_run: bool = False,
            runner: Callable[..., Any] = ExperimentRunner) -> None:
    """Execute pending configs against one explicit artifact root."""
    for path in filter_pending(configs, artifact_root):
        config = load_run_config(str(path), artifact_root=str(artifact_root))
        runner(config).run(dry_run=dry_run)
