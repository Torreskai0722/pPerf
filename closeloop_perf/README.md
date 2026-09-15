# Closed-loop profiling workspace

`closeloop_perf/` is a standalone colcon workspace with exactly four packages:

- `closeloop_profiler` records Nsight, NVTX, bpftrace, CUPTI, NVBit, manifest,
  and kernel-capsule evidence. It does not analyze results.
- `closeloop_analyzer` reads completed artifacts offline and writes analysis.
- `closeloop_testbed` provides the shared replay/model ROS workload and native
  synthetic workloads.
- `closeloop_experiments` validates configs, prepares data, and manages runs
  and campaigns.

Dependencies point from experiments to the other packages and from testbed to
profiler. Profiler and analyzer do not call each other.

## Build and test

Run inside `pPerf-host`:

```bash
cd /mmdetection3d_ros2/closeloop_perf
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

Architecture profiles are packaged by `closeloop_profiler` under
`src/closeloop_profiler/closeloop_profiler/config/architecture_profiles/`.

## Recording configuration

Run YAML uses schema v2 and has no analyzer selection or output path:

```yaml
schema_version: 2
recording:
  level: level1
  scopes: [model]
  nsys:
    version: 2025.2.1.130
    trace: [cuda, nvtx, cudnn]
    sample: none
    backtrace: none
    gpu_context_switch: true
```

- `level1` records Nsight Systems timing.
- `level2a` additionally requires the three Nsight Systems GPU-metric fields.
- `level2b` requires targeted `recording.ncu` and forbids simultaneous
  `recording.nsys` collection.
- `input` scope adds the relay and input-identity records.
- `preprocessing` scope adds transform/data-preprocessor annotations and the
  configured scheduler evidence collector.

Kernel capsule and CTA capture remain explicit MPS extensions under
`recording.capsule` and per-model `nvbit_cta_profile`; they are not a fourth
recording level. MPS, capsule, and base runs all reuse `testbed.launch.py`.

## Commands

Every operation that creates or locates runtime artifacts requires an explicit
artifact root:

```bash
ros2 run closeloop_experiments validate closeloop_perf/examples/level1.yaml
ros2 run closeloop_experiments run closeloop_perf/examples/level1.yaml \
  --artifact-root /tmp/pperf-artifacts --dry-run
ros2 run closeloop_experiments campaign input-variation plan \
  closeloop_perf/studies/section4_1/study.yaml \
  --artifact-root /tmp/pperf-artifacts
ros2 run closeloop_analyzer analyze time-slice RUN \
  --output-root /tmp/pperf-artifacts
```

Analyzer types are exactly `time-slice`, `mps`, `input-data`, and
`preprocessing`. Analyzers validate `run_manifest.json` and reject evidence
whose MPS mode, recording level, or scopes cannot support the requested work.
They never run automatically after collection.

## Artifact layout and migration

An artifact root contains only these top-level runtime areas:

```text
runs/
generated_configs/
analysis/runs/<run-id>/<analysis-type>/
analysis/<study-id>/<analysis-type>/  # Combined study reports
legacy/
```

Authored examples and studies stay in the repository. To relocate historical
`input1`, `input2`, fixed-input archives, outputs, and generated configuration
trees, use explicit roots:

```bash
ros2 run closeloop_experiments migrate OLD_ROOT NEW_ARTIFACT_ROOT
```

Migration checks free space, inventories every file with size and SHA-256,
copies to staging, verifies the copy, and only then removes the source. Its
`legacy/relocation_manifest.json` lets analyzers resolve stale absolute paths
inside schema-v1 artifacts. `build/`, `install/`, and `log/` are rebuildable
and are never migrated.

See [input-variation.md](docs/input-variation.md),
[input-variation-matrix-v3.md](docs/input-variation-matrix-v3.md),
[kernel-capsule.md](docs/kernel-capsule.md), and
[offline-kernel-capsule.md](docs/offline-kernel-capsule.md) for study-specific
details.
