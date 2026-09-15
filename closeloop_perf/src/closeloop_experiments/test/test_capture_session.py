"""Tests for the owned live capsule capture session."""

import hashlib
import json

import pytest

from closeloop_experiments.capture_session import (
    CaptureSessionError, nvbit_cupti_incompatibility,
    resolve_source_config,
)


def _write_manifest(run_directory, raw, command, config_source=None):
    manifest = {
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "command": command,
    }
    if config_source is not None:
        manifest["config_source"] = str(config_source)
    (run_directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_source_config_uses_byte_identical_recorded_path(tmp_path):
    """Relative paths resolve against the original config, not its copy."""
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    original = tmp_path / "examples" / "trace.yaml"
    original.parent.mkdir()
    raw = b"run: identity-aware\n"
    original.write_bytes(raw)
    (run_directory / "config.yaml").write_bytes(raw)
    _write_manifest(run_directory, raw, [], original)
    assert resolve_source_config(run_directory) == original.resolve()


def test_source_config_falls_back_to_launch_command(tmp_path):
    """Older manifests recover the exact original launch config path."""
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    original = tmp_path / "trace.yaml"
    raw = b"schema_version: 1\n"
    original.write_bytes(raw)
    (run_directory / "config.yaml").write_bytes(raw)
    _write_manifest(
        run_directory, raw, ["config_file:=" + str(original)]
    )
    assert resolve_source_config(run_directory) == original.resolve()


def test_source_config_rejects_changed_original(tmp_path):
    """Capture never substitutes a modified config for the traced bytes."""
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    original = tmp_path / "trace.yaml"
    original.write_bytes(b"changed\n")
    _write_manifest(run_directory, b"expected\n", [], original)
    with pytest.raises(CaptureSessionError, match="byte-identical"):
        resolve_source_config(run_directory)


def test_nvbit_preload_is_gated_on_legacy_cupti(tmp_path, monkeypatch):
    """Normal replay remains available with legacy single-subscriber CUPTI."""
    include = tmp_path / "include"
    include.mkdir()
    (include / "cuda.h").write_text(
        "#define CUDA_VERSION 12060\n", encoding="utf-8"
    )
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    assert "permits one CUPTI" in nvbit_cupti_incompatibility()
