"""Verified relocation of historical runtime artifacts."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Iterable, List


class MigrationError(RuntimeError):
    """Raised before source artifacts are changed."""


TOP_LEVEL = (
    "input1", "input2", "input2_scene-0252-fixed_one_pass_archive", "outputs",
)
GENERATED_NAMES = {
    "generated_configs", "diagnostic_configs", "isolated_configs",
}


def _is_generated_config_directory(name: str) -> bool:
    return (
        name in GENERATED_NAMES
        or name.startswith("generated_") and name.endswith("_configs")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(root: Path) -> List[Path]:
    selected = [root / name for name in TOP_LEVEL if (root / name).exists()]
    covered = tuple(path.resolve() for path in selected)
    for path in root.rglob("*"):
        if not path.is_dir() or not _is_generated_config_directory(path.name):
            continue
        resolved = path.resolve()
        if not any(parent == resolved or parent in resolved.parents
                   for parent in covered):
            selected.append(path)
    return sorted(selected, key=lambda path: str(path.relative_to(root)))


def _inventory(root: Path, sources: Iterable[Path]) -> List[Dict[str, object]]:
    records = []
    for source in sources:
        paths = [source] if source.is_file() else source.rglob("*")
        for path in paths:
            if path.is_file():
                records.append({
                    "path": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                })
    return sorted(records, key=lambda record: str(record["path"]))


def migrate_artifacts(source_root: str, destination_root: str) -> Path:
    """Copy, verify, inventory, then remove explicitly selected sources."""
    source = Path(source_root).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    if not source.is_dir():
        raise MigrationError(f"source root is not a directory: {source}")
    if source == destination:
        raise MigrationError("source and destination roots must differ")
    if source in destination.parents:
        raise MigrationError("destination root cannot be inside source root")
    legacy = destination / "legacy"
    if legacy.exists():
        raise MigrationError(f"destination collision: {legacy}")
    sources = _sources(source)
    if not sources:
        raise MigrationError("no migratable artifacts found")
    inventory = _inventory(source, sources)
    required = sum(int(record["size"]) for record in inventory)
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < required:
        raise MigrationError("insufficient destination space")
    staging = Path(tempfile.mkdtemp(prefix=".legacy-", dir=destination))
    installed = False
    try:
        for item in sources:
            target = staging / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        copied = _inventory(staging, [staging / item.relative_to(source)
                                      for item in sources])
        if copied != inventory:
            raise MigrationError("copied artifact checksum mismatch")
        manifest = {
            "schema_version": 1,
            "source_root": str(source),
            "destination_root": str(legacy),
            "files": inventory,
            "relocations": {
                str((source / record["path"]).resolve()):
                str((legacy / record["path"]).resolve())
                for record in inventory
            },
        }
        (staging / "relocation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(legacy)
        installed = True
        for item in sorted(
                sources, key=lambda path: len(path.parts), reverse=True):
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        return legacy / "relocation_manifest.json"
    except BaseException:
        if installed and legacy.exists():
            for item in sources:
                saved = legacy / item.relative_to(source)
                item.parent.mkdir(parents=True, exist_ok=True)
                if saved.is_dir():
                    shutil.copytree(saved, item, dirs_exist_ok=True)
                elif saved.is_file():
                    shutil.copy2(saved, item)
            shutil.rmtree(legacy)
        elif staging.exists():
            shutil.rmtree(staging)
        raise
