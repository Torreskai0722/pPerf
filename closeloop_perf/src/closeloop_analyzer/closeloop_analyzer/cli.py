"""Unified offline analyzer command."""

import argparse
import json
import sys

from ._manifest import EvidenceError
from .input_data import InputDataAnalyzer
from .mps_analyzer import MpsAnalyzer
from .preprocessing_analyzer import PreprocessingAnalyzer
from .time_slice_analyzer import TimeSliceAnalyzer


ANALYZERS = {
    "time-slice": TimeSliceAnalyzer,
    "mps": MpsAnalyzer,
    "input-data": InputDataAnalyzer,
    "preprocessing": PreprocessingAnalyzer,
}


def main(argv=None) -> int:
    """Analyze one source with exactly one of the four analyzer types."""
    parser = argparse.ArgumentParser(description="Analyze recorded evidence")
    parser.add_argument("type", choices=tuple(ANALYZERS))
    parser.add_argument("source")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--options", default="{}", help="JSON options object")
    args = parser.parse_args(argv)
    try:
        options = json.loads(args.options)
        if not isinstance(options, dict):
            raise ValueError("options must be a JSON object")
        result = ANALYZERS[args.type]().analyze(
            args.source, args.output_root, options)
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0
