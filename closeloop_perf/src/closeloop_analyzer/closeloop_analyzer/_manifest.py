"""Read and normalize immutable run manifests for offline analyzers."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class EvidenceError(ValueError):
    """Raised when recorded evidence cannot support an analysis."""


def _relocation_manifest(output_root: Path) -> Optional[Path]:
    for root in (output_root, *output_root.parents):
        candidate = root / "legacy" / "relocation_manifest.json"
        if candidate.is_file():
            return candidate
    return None


def resolve_source(source: str, output_root: str) -> Path:
    """Resolve a source, consulting a relocation manifest when necessary."""
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    relocation = _relocation_manifest(Path(output_root).expanduser().resolve())
    if relocation:
        data = json.loads(relocation.read_text(encoding="utf-8"))
        relocated = data.get("relocations", {}).get(str(path.resolve()))
        if relocated and Path(relocated).exists():
            return Path(relocated).resolve()
        old_root = Path(data["source_root"])
        try:
            relative = path.resolve().relative_to(old_root)
        except ValueError:
            pass
        else:
            candidate = Path(data["destination_root"]) / relative
            if candidate.exists():
                return candidate.resolve()
    raise EvidenceError(f"analysis source does not exist: {path}")


def _legacy_recording(manifest: Dict[str, Any], run_directory: Path) -> Dict[str, Any]:
    config_path = run_directory / "config.yaml"
    config = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = loaded if isinstance(loaded, dict) else {}
    instrumentation = manifest.get("instrumentation", config.get("instrumentation", {}))
    profiler = instrumentation.get("profiler", {})
    profiler_id = profiler.get("id", "model_level1_v1")
    nsys = config.get("nsys", {})
    level = "level2b" if config.get("ncu") else (
        "level2a" if nsys.get("gpu_metrics_devices") else "level1")
    scopes = ["model"]
    if config.get("input_variation") or config.get("replay", {}).get(
            "communication_profile"):
        scopes.append("input")
    if profiler_id == "preprocess_contention_v1":
        scopes.append("preprocessing")
    recording = {"level": level, "scopes": scopes}
    if nsys:
        recording["nsys"] = nsys
    if config.get("ncu"):
        recording["ncu"] = config["ncu"]
    if profiler_id == "gpu_forensics_v1":
        recording["capsule"] = profiler.get("options", {})
    recording["_legacy_mps_enabled"] = bool(
        config.get("gpu", {}).get("mps_enabled")
    )
    return recording


def load_manifest(source: str, output_root: str) -> Dict[str, Any]:
    """Return a normalized schema-v1/v2 run manifest and source directory."""
    resolved = resolve_source(source, output_root)
    path = resolved / "run_manifest.json" if resolved.is_dir() else resolved
    if not path.is_file():
        raise EvidenceError(f"run manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid run manifest: {exc}") from exc
    version = manifest.get("schema_version")
    if version == 1:
        recording = _legacy_recording(manifest, path.parent)
        legacy_mps_enabled = recording.pop("_legacy_mps_enabled")
    elif version == 2:
        recording = manifest.get("recording")
        legacy_mps_enabled = False
        if not isinstance(recording, dict):
            raise EvidenceError("schema-v2 manifest lacks recording evidence")
    else:
        raise EvidenceError(f"unsupported run manifest schema: {version!r}")
    if manifest.get("state") not in (None, "success"):
        raise EvidenceError(
            f"run manifest state is {manifest.get('state')!r}, not success")
    return {
        **manifest,
        "recording": recording,
        "run_directory": str(path.parent.resolve()),
        "mps_enabled": bool(
            manifest.get("gpu", {}).get("mps_enabled")
            or manifest.get("mps")
            or legacy_mps_enabled
        ),
    }


def require(manifest: Dict[str, Any], *, mps: Optional[bool] = None,
            scope: Optional[str] = None,
            levels=()) -> None:
    """Reject a run whose recorded evidence cannot support an analyzer."""
    recording = manifest["recording"]
    if mps is not None and manifest["mps_enabled"] is not mps:
        mode = "MPS" if mps else "non-MPS"
        raise EvidenceError(f"analysis requires {mode} evidence")
    if scope and scope not in recording.get("scopes", []):
        raise EvidenceError(f"analysis requires recording scope {scope!r}")
    if levels and recording.get("level") not in levels:
        raise EvidenceError(
            f"analysis requires recording level {' or '.join(levels)}")
