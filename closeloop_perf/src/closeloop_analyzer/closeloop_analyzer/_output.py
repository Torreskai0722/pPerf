"""Minimal JSON/CSV output helpers shared by analyzers."""

import json
from pathlib import Path
from typing import Any, Dict


def result_directory(output_root: str, analysis_type: str,
                     manifest: Dict[str, Any]) -> Path:
    """Return the deterministic directory for one analyzer result."""
    return (
        Path(output_root).expanduser().resolve() / "analysis" / "runs" /
        str(manifest.get("run_id", "unknown")) / analysis_type
    )


def write_result(
    output_root: str, analysis_type: str, manifest: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Write one deterministic analyzer result beneath ``analysis/runs/``."""
    destination = result_directory(output_root, analysis_type, manifest)
    destination.mkdir(parents=True, exist_ok=True)
    result["output_directory"] = str(destination)
    (destination / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
