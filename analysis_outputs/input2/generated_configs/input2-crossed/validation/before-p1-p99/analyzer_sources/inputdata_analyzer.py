"""Offline input-data and relay communication analysis."""

from pathlib import Path
import json

import yaml

from ..communication import analyze_communication
from .._manifest import load_manifest, require
from .._output import result_directory, write_result


class InputDataAnalyzer:
    """Analyze input geometry, campaigns, and communication evidence."""

    def analyze(self, source, output_root, options=None) -> dict:
        """Analyze one input-scoped run and return the written summary."""
        options = dict(options or {})
        path = Path(source)
        if path.is_file() and path.suffix == ".json":
            if json.loads(path.read_text()).get("schema") == "input2_crossed_experiment_v1":
                from .crossed_analysis import analyze
                return analyze(path, output_root, options)
        manifest = load_manifest(source, output_root)
        require(manifest, scope="input")
        run = Path(manifest["run_directory"])
        destination = result_directory(output_root, "input-data", manifest)
        communication = None
        if (run / "communication_relay.jsonl").is_file():
            config_path = (
                run / "resolved_config.yaml"
                if (run / "resolved_config.yaml").is_file()
                else run / "config.yaml"
            )
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            communication = analyze_communication(
                config, run, destination)
        result = {
            "analysis_type": "input-data",
            "run_id": manifest.get("run_id"),
            "recording": manifest["recording"],
            "communication_evidence": sorted(
                path.name for path in run.glob("*communication*")
                if path.is_file()
            ),
            "input_evidence": sorted(
                path.name for path in run.glob("*input*") if path.is_file()),
            "communication": communication,
            "options": options,
        }
        return write_result(output_root, "input-data", manifest, result)
