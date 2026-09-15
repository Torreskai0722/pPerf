"""Data-loss guards for explicit artifact relocation."""

from collections import namedtuple
import json

import pytest

import closeloop_experiments.migration as migration
from closeloop_experiments.migration import MigrationError, migrate_artifacts


def _source(tmp_path):
    source = tmp_path / "old"
    (source / "input1").mkdir(parents=True)
    (source / "input1" / "evidence.bin").write_bytes(b"evidence")
    return source


def test_verified_migration_removes_sources_only_after_copy(tmp_path):
    source = _source(tmp_path)
    generated = source / "studies" / "one" / "generated_mixed_configs"
    generated.mkdir(parents=True)
    (generated / "run.yaml").write_text("schema_version: 2\n")
    manifest_path = migrate_artifacts(source, tmp_path / "new")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not (source / "input1").exists()
    assert not generated.exists()
    evidence = next(
        row for row in manifest["files"]
        if row["path"] == "input1/evidence.bin"
    )
    assert evidence["size"] == 8
    assert len(evidence["sha256"]) == 64
    assert (manifest_path.parent / "input1" / "evidence.bin").read_bytes() == (
        b"evidence"
    )
    assert (
        manifest_path.parent / "studies/one/generated_mixed_configs/run.yaml"
    ).is_file()


def test_interrupted_copy_leaves_sources_untouched(tmp_path, monkeypatch):
    source = _source(tmp_path)
    monkeypatch.setattr(
        migration.shutil, "copytree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        migrate_artifacts(source, tmp_path / "new")
    assert (source / "input1" / "evidence.bin").is_file()
    assert not (tmp_path / "new" / "legacy").exists()


def test_insufficient_space_leaves_sources_untouched(tmp_path, monkeypatch):
    source = _source(tmp_path)
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        migration.shutil, "disk_usage", lambda _path: Usage(1, 1, 0)
    )
    with pytest.raises(MigrationError, match="insufficient"):
        migrate_artifacts(source, tmp_path / "new")
    assert (source / "input1" / "evidence.bin").is_file()


def test_destination_collision_leaves_sources_untouched(tmp_path):
    source = _source(tmp_path)
    (tmp_path / "new" / "legacy").mkdir(parents=True)
    with pytest.raises(MigrationError, match="collision"):
        migrate_artifacts(source, tmp_path / "new")
    assert (source / "input1" / "evidence.bin").is_file()


def test_checksum_mismatch_leaves_sources_untouched(tmp_path, monkeypatch):
    source = _source(tmp_path)
    original = migration._inventory
    calls = 0

    def mismatched(root, sources):
        nonlocal calls
        calls += 1
        records = original(root, sources)
        if calls == 2:
            records[0]["sha256"] = "0" * 64
        return records

    monkeypatch.setattr(migration, "_inventory", mismatched)
    with pytest.raises(MigrationError, match="checksum"):
        migrate_artifacts(source, tmp_path / "new")
    assert (source / "input1" / "evidence.bin").is_file()
    assert not (tmp_path / "new" / "legacy").exists()
