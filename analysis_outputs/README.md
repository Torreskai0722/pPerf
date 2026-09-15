# Retained experiment outputs

Each collection is an artifact root. Use the same collection directory for
`--artifact-root` when recording and `--output-root` when running an analyzer.
Paths in the table are relative to this directory.

| Collection | Recorded runs | Combined report |
| --- | ---: | --- |
| `input1` | 210 | [Physical-rain report](input1/analysis/input1-two-pass-physical-rain/input-data/report.md) |
| `input2` | 296 retained; 252 in the current report | [Environment-distribution report](input2/analysis/input2-environment-distributions/input-data/report.md) |
| `input2_scene-0252-fixed_one_pass_archive` | 36 | Per-run analysis only; this preserves the earlier one-pass fixed-input batch |

## Directory layout

```text
<collection>/
├── runs/<run-id>/
├── generated_configs/<study-id>/<run-id>.yaml
├── analysis/runs/<run-id>/<analysis-type>/
└── analysis/<study-id>/input-data/  # Combined analysis, where available
```

- `runs/` contains recorded profiles, manifests, immutable executed configs,
  model/input logs, and MPS evidence. Preserved failed attempts live under
  `input2/runs/failed_attempts/`.
- `generated_configs/` preserves the recorded campaign YAML snapshots.
- Per-run `analysis/runs/<run-id>/input-data/` directories contain
  `analysis.json` and the available
  communication latency CSV and summary JSON.
- Combined `input-data/` directories contain the existing inference tables,
  frame metrics, validation, plots, and reports. Their directory key is the
  study ID because they summarize multiple runs.
- Supporting Input2 scene statistics are in
  `input2/analysis/input2-environment-distributions/input-data/scene_metrics/`.

The 44 additional Input2 runs are the earlier `scene-fixed` batch (36 runs)
and the recorded `scene-0539` subset (8 runs). They remain in `input2/runs/`
but are outside the current 252-run report. The current Input2 fixed condition
uses two replay passes; the separate fixed-input archive preserves one-pass
recordings.

Recorded run manifests, executed configs, and profile archive metadata retain
their original bytes, hashes, and historical path strings as provenance.
Current locations are given by this layout and the updated analysis validation
tables; use the current run directory when invoking an analyzer.

## Example: analyze one retained run

Inside `pPerf-host`, from `/mmdetection3d_ros2`, after sourcing ROS and the
workspace environment:

```bash
ros2 run closeloop_analyzer analyze input-data \
  analysis_outputs/input1/runs/0-clear-3dssd-detr \
  --output-root analysis_outputs/input1
```

This writes to
`analysis_outputs/input1/analysis/runs/0-clear-3dssd-detr/input-data/`.
Inference measurements and existing plots were preserved during restructuring;
no experiment was rerun.
