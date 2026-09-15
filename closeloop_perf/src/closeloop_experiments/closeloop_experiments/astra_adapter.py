"""Run pinned Astra components in isolated subprocesses."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_spec(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_particles(spec):
    sys.path.insert(0, spec["astra_root"])
    from tools.simulation import WeatherSimulation

    simulation = WeatherSimulation(
        0,
        spec["cache"],
        spec["settings"],
        {"weather": "rain", "fallrate": spec["rain_rate"]},
        redo=False,
        deactivate_window_mode=True,
        bin_folder=str(
            Path(spec["astra_root"])
            / "3rdparty/weather-particle-simulator/lin_x64"
        ),
    )
    simulation.run()


def _run_renderer(spec):
    import numpy as np

    # Astra predates NumPy 1.24. Keep compatibility local to this process.
    np.bool = np.bool_
    np.float = float
    np.int = int
    sys.path.insert(0, spec.pop("astra_root"))
    from common.generator import Generator

    Generator(SimpleNamespace(**spec)).run()


def main(argv=None):
    """Run one fail-fast Astra stage from a JSON specification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("particles", "renderer"))
    parser.add_argument("spec")
    args = parser.parse_args(argv)
    spec = _load_spec(args.spec)
    (_run_particles if args.stage == "particles" else _run_renderer)(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
