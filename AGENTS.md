# Repository Guidelines

## Research Goals & Working Principles

This repository supports a research and engineering project targeting an IPDPS paper. The central research goal is to close the gap in profilers' ability to diagnose the root causes of inference time variation across execution levels. Treat research insight and engineering implementation as joint objectives.

The paper has two major contributions:

1. **Root-cause characterization:** systematically characterize how input data, data preprocessing, memory-copy (`memcpy`) synchronization in time-slicing mode, and leftover policy and resource contention in MPS mode contribute to inference time variation, including interactions across these levels.
2. **Closed-loop profiling tool:** build a tool that automatically diagnoses inference time variation by using collected evidence to identify candidate causes, select follow-up profiling experiments, and refine the diagnosis.

Guide work by the research question as well as the software requirement:

- State the hypothesis or diagnostic gap an experiment or capability addresses and the evidence needed to assess it. Distinguish observed correlations, candidate explanations, and experimentally supported root causes.
- Use controlled comparisons to separate causes and account for confounders. Preserve the configurations, input identities, execution conditions, and artifacts needed to reproduce findings; record uncertainty and limitations when evidence is inconclusive.
- Evaluate characterization by the mechanisms it explains and automated diagnosis by its correctness, ability to distinguish competing causes, and profiling cost. Passing software tests alone does not validate a research claim.
- Prefer the smallest implementation and experiment that answer the current research question. Retain controls, measurements, and analysis needed for scientific validity even when they add engineering work; follow the repetition limits below.

Implement the feedback loop through experiment orchestration while preserving the package boundaries below: the profiler collects evidence, the analyzer diagnoses completed artifacts offline, and experiments coordinate follow-up runs.

## Project Structure & Module Organization

The main ROS 2 colcon workspace is `closeloop_perf/`, with four packages under `closeloop_perf/src/`:

- `closeloop_profiler` records profiling evidence and runtime artifacts; it does not analyze results.
- `closeloop_analyzer` reads completed artifacts offline and produces analysis; it does not collect profiling evidence.
- `closeloop_testbed` provides shared replay/model ROS workloads and native synthetic workloads.
- `closeloop_experiments` validates configurations, prepares data, and orchestrates runs and campaigns.

Keep dependencies directed from experiments to the other packages and from testbed to profiler. Profiler and analyzer must not call each other. Keep collection and offline analysis separate.

Authored study configurations live in `closeloop_perf/studies/`; package configuration and schemas live with their owning packages. Start with `closeloop_perf/README.md` for workspace usage. Analysis and visualization code lives in `closeloop_analyzer`. Container definitions live in `Docker/` and `closeloop_perf/docker/`. `LISA/` and `nuscenes_to_ros2bag/` are integrated upstream projects; keep changes there isolated. Treat `data/`, `analysis_outputs/`, runtime artifact roots, and workspace `build/`, `install/`, and `log/` directories as generated artifacts, not source code.

Per-run analyzer results go under `<output-root>/analysis/runs/<run-id>/<analysis-type>/`. Combined study reports go under `<output-root>/analysis/<study-id>/<analysis-type>/`. Recorded artifacts remain under `<artifact-root>/runs/<run-id>/`; artifact and analyzer output roots are supplied explicitly.

## Development Environment

Develop inside the Docker container `pPerf-host`. Its default workspace is `/mmdetection3d_ros2`, where the current host workspace is mounted. The CUDA 12.6 image keeps its OpenMMLab source trees under `/opt`: MMCV v2.1.0 at `/opt/mmcv`, editable MMDetection v3.3.0 at `/opt/mmdetection`, and editable MMDetection3D v1.4.0 at `/opt/mmdetection3d`. Changes under `/opt` affect only the container; update `closeloop_perf/docker/Dockerfile.cuda126` to persist dependency or installation changes across image rebuilds. Run repository commands in `/mmdetection3d_ros2` unless explicitly changing directories.

## Build, Test, and Development Commands

- `docker build -t perf_ws -f Docker/Dockerfile .` builds the standard CUDA/ROS environment; use `Docker/Dockerfile.blackwell` on Blackwell systems.
- `docker build -t pperf-cuda126 -f closeloop_perf/docker/Dockerfile.cuda126 .` builds the CUDA 12.6 environment used by `pPerf-host`.
- `source /opt/ros/humble/setup.bash` loads ROS 2 before building or using the workspace.
- `cd closeloop_perf && colcon build --symlink-install` builds the workspace; use `--packages-up-to <package>` for a focused build including its dependencies.
- `source closeloop_perf/install/setup.bash` exposes built packages in a new shell from the repository root; use `source install/setup.bash` when already inside the workspace.
- `ros2 run closeloop_experiments validate <run.yaml>` validates a run configuration.
- `ros2 run closeloop_experiments run <run.yaml> --artifact-root <artifact-root> --dry-run` prepares a run without executing the workload; omit `--dry-run` to execute it.
- `ros2 run closeloop_experiments campaign --help` lists campaign commands; use the relevant study configuration under `closeloop_perf/studies/`.
- `ros2 run closeloop_analyzer analyze --help` lists offline analysis commands. Analysis is an explicit step after collection.

These commands are a reference, not a checklist to run for every task. Provide explicit artifact/output roots where required and configure dataset and model paths for the local environment.

## Engineering Principles

- Solve today's immediate research or engineering problem with the simplest implementation that preserves scientific validity. Do not add speculative features, configuration, or abstractions for possible future needs.
- Do not preserve backward compatibility through unnecessary compatibility layers, fallbacks, or historical migrations. Add or retain them only for a concrete current requirement.
- Build in layers: first establish a minimal end-to-end working version, then add capabilities incrementally until the requested scope is complete.
- Keep components modular and enforce strict separation of concerns, including the package boundaries above.
- Audit existing dependencies before installing new ones or writing custom code. Reuse capabilities already provided by the standard library and installed packages.
- When additional functionality is needed, prefer mature, well-maintained, battle-tested external libraries over custom implementations of established functionality.
- Make foundational decisions that support long-term maintainability and scaling. Avoid temporary hacks while keeping the implementation limited to current requirements.
- Follow proven software engineering patterns and established repository conventions instead of inventing new solutions. Use patterns only where they serve the current problem.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions for Python. Name modules, functions, and variables in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Keep ROS launch files named `*.launch.py` and configuration in YAML. Add concise docstrings to public modules and functions, and follow each package's configured lint checks. Avoid committing caches, binaries, profiler reports, or machine-specific absolute paths.

## Testing Guidelines

Choose validation according to the affected code; do not run the full workspace suite for every task.

- For focused Python changes, run the relevant tests with `python -m pytest closeloop_perf/src/<package>/test/<test_file>.py` after sourcing the workspace environment.
- For package-level validation, run `colcon test --packages-select <package> --event-handlers console_direct+` from `closeloop_perf/`; include affected dependent packages when needed.
- Use `colcon test --event-handlers console_direct+` for cross-package integration changes or explicit full-workspace validation. After a colcon test run, inspect results with `colcon test-result --verbose` from the workspace directory.
- For documentation-only edits or read-only reviews, check relevant paths, commands, and the diff; runtime tests are unnecessary.
- Reuse the existing test framework. Name Python test files `test_*.py` and test functions `test_*`. GPU-, NuScenes-, or model-dependent tests should document prerequisites and retain a lightweight unit-test path where possible.
- Once relevant checks pass, do not repeat or broaden them unless new changes, failures, or unresolved concerns justify it. Report unavailable validation and continue work that does not depend on it.

No repository-wide coverage threshold is configured.

## Experiment Repetitions

Default to one execution per condition. Use repetitions only when they are necessary for uncertainty or reproducibility, and never use more than three repetitions per condition unless the user explicitly overrides this limit.

## Commit & Pull Request Guidelines

Use short, lowercase, action-oriented commit summaries such as `updated experiment runner`; keep subjects focused and identify the affected component. Separate generated-data changes from code changes. Pull requests should explain the change and scenario tested, list exact validation commands and results, link related issues when available, and note only relevant GPU, CUDA, ROS, dataset, or model prerequisites. For analysis output or visualization changes, include representative plots or screenshots when the required environment and data are available. Otherwise, state what visual validation could not be performed; missing issue links or unavailable screenshots should not block unrelated work.
