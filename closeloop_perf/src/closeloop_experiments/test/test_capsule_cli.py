"""Tests for kernel-capsule command-line failure artifacts."""

import json

from closeloop_experiments.capsule_cli import write_capsule_failure


def test_capsule_invalid_failure_is_atomic_and_inspectable(tmp_path):
    """Exactness rejection leaves a stable machine-readable record."""
    output = tmp_path / "campaign"
    write_capsule_failure(
        output, RuntimeError("capsule_invalid: unresolved runtime target")
    )
    record = json.loads(
        (output / "capsule_failure.json").read_text(encoding="utf-8")
    )
    assert record == {
        "schema": "kernel_capsule_failure_v3",
        "status": "capsule_invalid",
        "error_type": "RuntimeError",
        "error": "capsule_invalid: unresolved runtime target",
    }
    assert not (output / ".capsule_failure.json.tmp").exists()
