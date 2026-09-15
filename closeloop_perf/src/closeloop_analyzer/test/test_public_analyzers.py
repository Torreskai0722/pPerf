"""Acceptance checks for the four public offline analyzers."""

import json

import pytest

from closeloop_analyzer._manifest import EvidenceError
from closeloop_analyzer.cli import ANALYZERS
from closeloop_analyzer.input_data import InputDataAnalyzer
from closeloop_analyzer.mps_analyzer import MpsAnalyzer
from closeloop_analyzer.preprocessing_analyzer import PreprocessingAnalyzer
from closeloop_analyzer.time_slice_analyzer import TimeSliceAnalyzer


def _run(tmp_path, *, level="level1", scopes=("model",), mps=False,
         schema=2):
    run = tmp_path / (
        f"run-{level}-{'mps' if mps else 'whole'}-{'-'.join(scopes)}"
    )
    run.mkdir()
    manifest = {
        "schema_version": schema,
        "run_id": run.name,
        "state": "success",
        "gpu": {"mps_enabled": mps},
        "recording": {"level": level, "scopes": list(scopes)},
    }
    if schema == 1:
        manifest.pop("recording")
        (run / "config.yaml").write_text(
            "gpu:\n  mps_enabled: false\nnsys:\n  trace: [cuda, nvtx]\n",
            encoding="utf-8",
        )
    (run / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run


def test_cli_exposes_exactly_four_analyzer_types():
    assert tuple(ANALYZERS) == (
        "time-slice", "mps", "input-data", "preprocessing"
    )


def test_each_analyzer_accepts_compatible_evidence(tmp_path):
    output = tmp_path / "artifacts"
    whole = _run(tmp_path, level="level2a")
    mps = _run(tmp_path, level="level2b", mps=True)
    inputs = _run(tmp_path, scopes=("model", "input"))
    preprocessing = _run(tmp_path, scopes=("model", "preprocessing"))

    assert TimeSliceAnalyzer().analyze(whole, output)["analysis_type"] == (
        "time-slice"
    )
    assert MpsAnalyzer().analyze(
        mps, output, {"analysis": "targeted"}
    )["analysis_type"] == "mps"
    assert InputDataAnalyzer().analyze(inputs, output)["analysis_type"] == (
        "input-data"
    )
    assert PreprocessingAnalyzer().analyze(
        preprocessing, output
    )["analysis_type"] == "preprocessing"

    for run, analysis_type in (
        (whole, "time-slice"), (mps, "mps"),
        (inputs, "input-data"), (preprocessing, "preprocessing"),
    ):
        destination = output / "analysis" / "runs" / run.name / analysis_type
        result = json.loads((destination / "analysis.json").read_text())
        assert result["output_directory"] == str(destination)


def test_analyzers_reject_incompatible_evidence(tmp_path):
    run = _run(tmp_path)
    with pytest.raises(EvidenceError, match="MPS"):
        MpsAnalyzer().analyze(run, tmp_path / "out")
    with pytest.raises(EvidenceError, match="input"):
        InputDataAnalyzer().analyze(run, tmp_path / "out")
    with pytest.raises(EvidenceError, match="preprocessing"):
        PreprocessingAnalyzer().analyze(run, tmp_path / "out")


def test_schema_v1_time_slice_remains_readable(tmp_path):
    run = _run(tmp_path, schema=1)
    result = TimeSliceAnalyzer().analyze(run, tmp_path / "out")
    assert result["recording"]["level"] == "level1"
