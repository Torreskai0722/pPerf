"""Offline non-MPS time-slice evidence analysis."""

from pathlib import Path

from .coverage import analyze_run as analyze_coverage
from ._manifest import load_manifest, require
from ._output import write_result


class TimeSliceAnalyzer:
    """Analyze ownership, overlap, offsets, and Green Context evidence."""

    def analyze(self, source, output_root, options=None) -> dict:
        """Analyze one compatible run and return the written summary."""
        options = dict(options or {})
        manifest = load_manifest(source, output_root)
        require(
            manifest, mps=False, scope="model",
            levels=("level1", "level2a"),
        )
        run = Path(manifest["run_directory"])
        coverage = None
        if (
            (run / "profile.sqlite").is_file()
            and manifest.get("models")
        ):
            coverage = analyze_coverage(
                run, float(options.get("minimum_module_coverage", 0.75)))
        result = {
            "analysis_type": "time-slice",
            "run_id": manifest.get("run_id"),
            "recording": manifest["recording"],
            "evidence": sorted(
                path.name for path in run.iterdir() if path.is_file()
            ),
            "architecture_coverage": coverage,
            "options": options,
        }
        return write_result(output_root, "time-slice", manifest, result)
