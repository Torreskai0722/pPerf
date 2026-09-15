"""Tests for architecture binding and module-range coverage validation."""

import json
import sqlite3

from closeloop_analyzer.coverage import analyze_run


def _tag(model, input_id, event, module=None):
    """Return one serialized close-loop NVTX tag."""
    value = {
        "schema_version": 1,
        "run": "run",
        "scene": "scene",
        "model": model,
        "input": input_id,
        "event": event,
    }
    if module is not None:
        value["module"] = module
    return "closeloop:" + json.dumps(value)


def _run_fixture(tmp_path):
    """Create a minimal successful run with overlapping module ranges."""
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    manifest = {
        "state": "success",
        "run_id": "run",
        "models": {
            "model": {
                "state": "acknowledged",
                "inputs": 1,
                "architecture_profile": "mmdet_two_stage_2d_v1",
                "observed_method_bindings": [
                    "model.predict", "model.extract_feat",
                ],
                "observed_module_bindings": [
                    "data_preprocessor", "backbone", "rpn_head", "roi_head",
                ],
            },
        },
    }
    (run_directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    database = sqlite3.connect(str(run_directory / "profile.sqlite"))
    database.execute(
        "CREATE TABLE NVTX_EVENTS "
        "(start INTEGER, end INTEGER, text TEXT)"
    )
    database.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER)"
    )
    database.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
        (0, 100, _tag("model", "0", "inference")),
    )
    database.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
        (10, 60, _tag("model", "0", "module", "backbone")),
    )
    database.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
        (40, 90, _tag("model", "0", "module", "roi_head")),
    )
    database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1)")
    database.commit()
    database.close()
    return run_directory


def test_coverage_uses_interval_union_and_accepts_threshold(tmp_path):
    """Overlapping modules are counted once and meet a 75 percent floor."""
    report = analyze_run(_run_fixture(tmp_path), 0.75)
    assert report["passed"]
    assert report["models"]["model"][
        "mean_module_wall_time_coverage"
    ] == 0.8


def test_coverage_reports_threshold_failure(tmp_path):
    """The checker reports a model failure below the configured floor."""
    report = analyze_run(_run_fixture(tmp_path), 0.85)
    assert not report["passed"]
    assert "below 0.850" in report["failures"][0]
