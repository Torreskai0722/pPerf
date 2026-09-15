"""Enforce the four package responsibility boundaries."""

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2]


def _imports(package):
    for path in (SOURCE_ROOT / package / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    yield path, name.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield path, node.module


def test_profiler_and_testbed_do_not_import_analysis_or_experiments():
    """Evidence/workload packages stay below orchestration and analysis."""
    forbidden = ("closeloop_analyzer", "closeloop_experiments")
    for package in ("closeloop_profiler", "closeloop_testbed"):
        violations = [
            f"{path.name}:{name}" for path, name in _imports(package)
            if name.startswith(forbidden)
        ]
        assert not violations, violations


def test_analyzer_has_no_runtime_package_imports():
    """Offline analysis consumes files without recorder or ROS runtimes."""
    forbidden = (
        "closeloop_experiments", "closeloop_profiler", "rclpy",
        "rosbag2_py", "launch", "launch_ros", "sensor_msgs", "std_msgs",
    )
    violations = [
        f"{path.name}:{name}"
        for path, name in _imports("closeloop_analyzer")
        if name in forbidden or name.startswith(tuple(
            prefix + "." for prefix in forbidden
        ))
    ]
    assert not violations, violations


def test_lower_packages_contain_no_analysis_or_report_modules():
    """Profiler and testbed modules stay focused on evidence and workloads."""
    forbidden = (
        "analysis", "analyzer", "campaign", "plot", "report", "study",
    )
    for package in ("closeloop_profiler", "closeloop_testbed"):
        names = [
            path.stem
            for path in (SOURCE_ROOT / package / package).glob("*.py")
            if any(token in path.stem for token in forbidden)
        ]
        assert not names, names
