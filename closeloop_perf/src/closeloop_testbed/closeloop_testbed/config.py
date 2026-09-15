"""Read the runner-resolved configuration used by testbed processes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved configuration passed down by the experiment runner."""

    data: Dict[str, Any]


def load_runtime_config(path: str) -> RuntimeConfig:
    """Load a config already validated by ``closeloop_experiments``."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime configuration root must be an object")
    return RuntimeConfig(data)
